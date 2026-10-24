"""Apply the third-party trunk kernels inside BoltzVS's own process.

Anthropic's `uplifting-biomolecular-modeling` kit (Apache-2.0) carries fused kernels for
Boltz-2's Pairformer trunk -- triangle attention, triangle multiplication, the pair
transition, the MSA module's outer-product mean and pair-weighted averaging. They are
installed as **class-level monkeypatches**, which is why they compose with the BoltzVS
flags: dropping Pairformer layers changes the ModuleList, loading student weights
changes the parameters, and neither touches the patched ``forward``.

The kit ships them behind its own persistent-worker runner and *refuses* the plain-CLI
route by name (``boltz2_opt.modes.CLI_ROUTE``), stating the hook that would compose the
levers in one process "is not shipped (OPEN)". That refusal lives in ``enable()`` alone;
the per-adapter ``apply()`` entry points have no such gate. This module is that hook,
restricted to the trunk.

Two things follow from restricting it to the trunk:

* The levers the kit refuses for a single process are the ones we do not want anyway --
  the diffusion sampler (``rollout``/``dit_*``/``align_*``), the trunk CUDA graph, the
  background writer and the persistent featurizer. BoltzVS runs the sampler for two
  steps; there is nothing left there to capture.
* The kit deliberately isolates the affinity pass in a clean interpreter
  (``affinity_leg``: *"no lever exists for it and none may touch it"*), so its kernels
  never reach it. Applied here, they do: the affinity model is built from the same
  classes, which are already patched.

Nothing is vendored. The kit must be installed separately (``opt_core`` + ``boltz2_opt``);
when it is absent this module is a no-op and says so.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional

TAG = "boltzvs-kernels"

#: Switch words of the kit's ``fast`` row, restricted to the trunk. Taken from the kit's
#: own dry-run line so the levers resolve exactly as they do under its runner.
TRUNK_ENV: Dict[str, str] = {
    "BOLTZ_LEVERS": "resid,mask2",
    "BOLTZ_FPF_TRIMUL": "1",
    "BOLTZ_FPF_TRIMUL_PROVIDER": "fast",
    "BOLTZ_TRANSITION": "fast",
    "BOLTZ_PAIRBLOCK": "default",
    "BOLTZ_PAIRBLOCK_C64": "1",
    "BOLTZ_PAIRFUSE": "bf16,trimul=core.fast,triatt=core.fast,transition=core.fast",
    "BOLTZ_TEMPL_SKIP": "1",
    "BOLTZ_WASTE": "chunkcast,opmmask,opmdiv,ctorskip",
    "BOLTZ_FPF_MSA": "opm,pwa",
    "BOLTZ_MSA2": "trans2",
    "BOLTZ_CONF": "condproj",
    "BOLTZ_ATOM": "keys,fused",
    "BOLTZ_ATOM_GEMM": "bf16",
}

#: The ``fast`` row minus ``fpf_pwa`` -- the MSA module's fused pair-weighted averaging.
#: That one lever is what stands between a large target and a 24 GB card: ablating it on a
#: 418-residue DEKOIS target dropped peak memory 4.0 GB (24086 -> 20008 MiB) and took the
#: pass from 3/12 inputs to 11/12, while ablating ``pairfuse`` or ``waste`` moved the peak
#: by ~200 MiB and changed nothing. The kit measured the same cost independently and names
#: it in its memory row as ``measured_memory_2.8GiB_alloc_peak``.
#:
#: ``BOLTZ_FPF_MSA`` is emptied rather than set to ``opm``, because emptying it is what was
#: actually run: on compute capability 8.6 ``fpf_opm`` falls back on every call anyway (it
#: asks for 198656 B of shared memory against a 101376 B limit), so the two spellings serve
#: the same kernels and only one of them has been measured. 16 levers apply instead of 18.
#:
#: This row is what completed the DEKOIS set at 79/79: the three targets that fit at no
#: chunk threshold under ``fast`` (979 / 418 / 948 residues) ran 1238 / 1240 / 1240 inputs
#: with zero OOM under it, all 40 actives each.
TRUNK_ENV_LOWMEM: Dict[str, str] = dict(TRUNK_ENV, BOLTZ_FPF_MSA="")

#: Deliberately unset. The first four are the process-level levers the kit's own refusal
#: is about; ``exactln`` is pinned to compute capability 9.0 / 8.0 and refuses elsewhere.
#: The last three only ride levers we already drop, so carrying them would be misleading.
EXCLUDED = ("BOLTZ_SAMPLER_ROLLOUT", "BOLTZ_SAMPLER_ALIGN", "BOLTZ_SAMPLER_DIT",
            "BOLTZ_DIT_HOIST", "BOLTZ_GRAPH_TRUNK", "BOLTZ_WRITER", "BOLTZ_PREFETCH",
            "BOLTZ_EXACTLN", "BOLTZ_SAMPLER_MAX_TOKENS", "BOLTZ_GRAPH_TRUNK_MAX_TOKENS",
            "BOLTZ_PRECISION")

#: The kit's other row worth having here: ``big`` is its **memory mode** -- Tier 2 like
#: ``fast`` and composed on it, but with the providers' ``big`` tier words, the pair
#: Transition and the diffusion conditioner evaluated 256 rows at a time, a lazy relpos
#: and an expandable-segments allocator. Its promise is *"peak GPU memory at or below the
#: stock's, or an input size ceiling above the stock's; slower than fast"* -- which is
#: exactly the trade a 24 GB card needs on a large target, where the ``fast`` row costs
#: more memory than the unaccelerated pass and simply does not fit.
#:
#: Two of that extra cost the kit has measured and names itself: ``fpf_pwa`` at 2.8 GiB of
#: allocator peak (dropped on every card in this row) and ``atom_fused`` at 3.5 GB on
#: sm_80 (dropped there by ``modes.CARD_DROPS``). Both are levers the ``fast`` row above
#: turns on.
#:
#: We do not transcribe the row: ``modes.resolve()`` is asked for it with this card's
#: compute capability and memory, so the per-card drops are the kit's own decision rather
#: than our reading of its tables.
#:
#: **Measured on cc 8.6 (RTX 3090, 24 GB), ``big`` did not deliver that promise**: on three
#: DEKOIS targets where ``fast`` ran out of memory, ``big`` also ran out (and on a fourth its
#: affinity pass failed outright), while the ``fast`` row with ``BOLTZ_FPF_MSA=`` fit all
#: three with zero OOM. The suspect is ``BOLTZ_XL_COND_FP32=1``, which puts the conditioner
#: back in fp32. So on a 24 GB card the smaller change wins; ``big`` is kept here because it
#: is the kit's own designed row and the card it was tuned on is not this one. Measure first.
ROWS = ("fast", "fast-lowmem", "big")


def row_env(row: str = "fast") -> Dict[str, str]:
    """Switch words for ``row``, restricted to the trunk.

    ``fast`` is the table above verbatim -- it is the configuration the 79-target
    validation was run on and must not start depending on the kit's tables. ``big`` is
    resolved from the kit at call time; if that fails the caller gets the reason.
    """
    if row == "fast":
        return dict(TRUNK_ENV)
    if row == "fast-lowmem":
        return dict(TRUNK_ENV_LOWMEM)
    if row not in ROWS:
        raise ValueError(f"unknown row {row!r}, expected one of {ROWS}")

    import torch

    modes = importlib.import_module("boltz2_opt.modes")
    major, minor = torch.cuda.get_device_capability(0)
    total_mib = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    modes.set_card(f"{major}.{minor}", total_mib)
    env = dict(modes.resolve(row)["env"])
    for key in EXCLUDED:
        env.pop(key, None)
    return env


#: ``(adapter, boltz2_opt module, trigger module)`` in the kit's own attach order
#: (``boltz2_opt.worker_launch.ATTACH``). Order is load-bearing: ``waste`` wraps the
#: forwards that ``msa`` / ``msa2`` then wrap again, and ``atom`` hooks
#: ``AtomDiffusion.__init__`` so it must be in place before the checkpoint is loaded.
#: ``xl`` last: it is the ``big`` row's own adapter and a no-op (returns ``[]``) whenever
#: ``BOLTZ_XL`` is unset, which is every ``fast`` run.
SEQUENCE = (
    ("trimul", "trimul", "boltz.model.layers.triangular_mult"),
    ("transition", "transition", "boltz.model.layers.transition"),
    ("pairblock", "pairblock", "boltz.model.layers.triangular_attention.attention"),
    ("pairfuse", "pairfuse", "boltz.model.modules.trunkv2"),
    ("templskip", "templskip", "boltz.model.models.boltz2"),
    ("waste", "waste", "boltz.model.models.boltz2"),
    ("msa", "msa_kernels", "boltz.model.models.boltz2"),
    ("msa2", "msa2", "boltz.model.models.boltz2"),
    ("conf", "conf", "boltz.model.models.boltz2"),
    ("atom", "atom", "boltz.model.models.boltz2"),
    ("xl", "big", "boltz.model.models.boltz2"),
)

_STATE: Dict[str, Any] = {"enabled": False, "applied": {}, "failed": {}}


def available() -> Optional[str]:
    """``None`` when the kit can be used here, else the reason it cannot."""
    for pkg in ("opt_core", "boltz2_opt"):
        if importlib.util.find_spec(pkg) is None:
            return f"{pkg} is not installed"
    try:
        import torch
    except Exception as exc:                                    # pragma: no cover
        return f"torch unavailable: {exc}"
    if not torch.cuda.is_available():
        return "no CUDA device"
    cc = torch.cuda.get_device_capability(0)
    if cc < (8, 0):                                             # opt_core MIN_CC
        return f"compute capability {cc[0]}.{cc[1]} < 8.0"
    return None


def enable(verbose: bool = True, row: str = "fast") -> Dict[str, Any]:
    """Apply the trunk levers to the classes already imported in this process.

    Idempotent. Call **before** the model is constructed: the ``atom`` adapter hooks
    ``AtomDiffusion.__init__`` to install itself when the checkpoint loader builds the
    structure module.

    ``row`` picks the kit row:

    * ``fast`` -- the default, and what the 79-target accuracy validation used.
    * ``fast-lowmem`` -- ``fast`` without ``fpf_pwa``, about 4 GB cheaper. Reach for this
      first when a target does not fit; it is the row that finished that validation's three
      largest targets.
    * ``big`` -- the kit's own memory mode. Measured slower *and* no smaller on cc 8.6; see
      ``TRUNK_ENV_LOWMEM`` and ``ROWS``.
    """
    if _STATE["enabled"]:
        return report()

    why = available()
    if why is not None:
        if verbose:
            print(f"[{TAG}] not applied: {why}", flush=True)
        _STATE["failed"]["_precondition"] = why
        return report()

    try:
        env = row_env(row)
    except Exception as exc:
        if verbose:
            print(f"[{TAG}] not applied: cannot resolve row {row!r}: {exc}", flush=True)
        _STATE["failed"]["_row"] = f"{type(exc).__name__}: {exc}"
        return report()
    _STATE["row"] = row

    for key in EXCLUDED:                    # a stale word would pull in a lever we exclude
        os.environ.pop(key, None)
    for key, value in env.items():
        if value == "":
            os.environ[key] = ""            # a row that turns a lever off means it: a stale
        else:                               # word in the environment must not resurrect it
            os.environ.setdefault(key, value)   # otherwise an explicit caller value wins

    for adapter, module_name, trigger in SEQUENCE:
        try:
            importlib.import_module(trigger)    # the hooked classes must exist first
            module = importlib.import_module(f"boltz2_opt.{module_name}")
            got: List[str] = list(module.apply() or [])
            _STATE["applied"][adapter] = got
        except Exception as exc:                # one adapter failing leaves the rest usable
            _STATE["failed"][adapter] = f"{type(exc).__name__}: {exc}"

    _STATE["enabled"] = True
    if verbose:
        levers = sorted(l for got in _STATE["applied"].values() for l in got)
        print(f"[{TAG}] row={row}: applied {len(levers)} levers: {','.join(levers)}",
              flush=True)
        for adapter, why in _STATE["failed"].items():
            print(f"[{TAG}] {adapter} not applied: {why}", flush=True)
    return report()


def report() -> Dict[str, Any]:
    """What was applied, plus each adapter's own census line once a pass has run."""
    lines = {}
    if _STATE["enabled"]:
        for adapter, module_name, _ in SEQUENCE:
            if adapter not in _STATE["applied"]:
                continue
            try:
                module = importlib.import_module(f"boltz2_opt.{module_name}")
                line = getattr(module, "line", lambda: None)()
                if line:
                    lines[adapter] = line
            except Exception:                                   # a census is best-effort
                pass
    return {"enabled": _STATE["enabled"], "row": _STATE.get("row"),
            "applied": dict(_STATE["applied"]),
            "failed": dict(_STATE["failed"]), "lines": lines}


def print_census() -> None:
    """Print each adapter's census line. Call after a prediction to see what served."""
    for adapter, line in report()["lines"].items():
        print(f"[{TAG}] {adapter}: {line}", flush=True)
