"""Structure-first cascade screening for BoltzVS.

One command turns a receptor FASTA and a SMILES library into a ranked hit list:

    1. fetch one receptor MSA per chain, and reuse it for every compound;
    2. run the accelerated structure pathway over the whole library, rank by min-iPAE;
    3. re-score only the best few per cent with the accelerated affinity pathway.

Both passes use the released Boltz-2 weights with the BoltzVS inference flags.

    boltz screen --receptor receptor.fasta --library library.smi --out-dir run/

The library is one SMILES per line, optionally followed by a name:

    CCO                     ethanol
    CC(=O)Oc1ccccc1C(=O)O   aspirin

Each stage is skipped when its output is already present, so an interrupted screen
resumes by re-running the same command; --override forces a recompute.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# The default BoltzVS configuration: training-free, inference settings only.
DEFAULT_STRUCTURE = ["--sampling_steps", "2", "--gamma_0", "0", "--step_scale", "1",
                     "--recycling_steps", "0", "--matmul_precision", "medium"]

# The affinity pathway is a second, complete prediction with its own trunk and
# sampler, so it needs its own flags; without these it dominates the runtime.
DEFAULT_AFFINITY = ["--sampling_steps_affinity", "2", "--recycling_steps_affinity", "1",
                    "--diffusion_samples_affinity", "5"]

# BoltzVS-D17 drops these Pairformer layers. Structure pass only -- the affinity
# pass always runs the full 64-layer affinity checkpoint.
D17_LAYERS = "2,3,4,5,6,7,16,18,19,26,27,28,29,30,31,32,33"

# The paper benchmarks use this less aggressive chunking threshold. It is a
# `boltz screen` default only: bare `boltz predict` defaults to 384.
DEFAULT_SCREEN_CHUNK_SIZE_THRESHOLD = 2048

AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


# ============================================================== min-iPAE
#
# Boltz-2 writes one PAE matrix per complex, sized (n_protein + n_ligand). The
# protein is chain A, tokenised per residue, and comes first; the ligand is chain
# B, tokenised per atom, and follows. A 411-residue protein with a 20-token
# ligand therefore gives a 431 x 431 matrix.
#
# PAE is not symmetric. PAE[i, j] is the expected positional error at token i
# when the predicted and true structures are aligned on token j -- rows are error
# points, columns are alignment frames (Elfmann & Stulke, NAR 2023). The matrix
# thus splits into four blocks, and min-iPAE lives in the top-right one: protein
# error under ligand alignment.
#
#                              j = alignment frame
#                        protein (411)     ligand (20)
#                      +----------------+---------------+
#   i     protein(411) | protein error, | protein error,|  <-- min-iPAE
#   =                  | protein frame  | ligand frame  |      is taken here
#   error              +----------------+---------------+
#   point ligand (20)  | ligand error,  | ligand error, |
#                      | protein frame  | ligand frame  |
#                      +----------------+---------------+
#
# The score is the single smallest value in that block: is there *any* protein
# token confidently placed relative to *any* ligand token? A minimum rather than a
# mean because an interface is local -- most of a 411-residue protein sits far
# from the pocket and necessarily carries large PAE, since its position relative
# to the ligand genuinely is uncertain. Averaging would let that bulk drown the
# signal; a minimum asks only whether a confident anchor exists, which is what
# interface confidence means physically.
#
# This matches min_ipae.py as used for every result in the paper, so the tool and
# the published numbers are the same quantity.


def min_ipae(pae: np.ndarray, n_protein: int) -> float:
    """min-iPAE in Angstroms; lower is more confident."""
    return float(pae[:n_protein, n_protein:].min())


# ============================================================== inputs


def read_fasta(path: Path) -> list[str]:
    """Return the sequences in file order. Wrapping, spacing and '*' are tolerated."""
    seqs, current = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                seqs.append("".join(current))
            current = []
        elif line.strip():
            current.append(re.sub(r"[\s*]+", "", line).upper())
    if current:
        seqs.append("".join(current))

    if not seqs:
        raise SystemExit(f"No sequences found in {path}")
    for i, seq in enumerate(seqs, 1):
        bad = sorted(set(seq) - AA)
        if bad:
            raise SystemExit(
                f"Record {i} in {path} contains {', '.join(map(repr, bad))}, which "
                "are not amino acids. This tool expects a protein FASTA."
            )
        # A/C/G/T are all valid amino-acid letters, so a nucleotide FASTA cannot
        # be rejected outright -- but a long sequence drawn only from them is
        # almost certainly DNA, and would silently produce nonsense.
        if len(seq) >= 30 and set(seq) <= set("ACGTUN"):
            raise SystemExit(
                f"Record {i} in {path} is {len(seq)} letters drawn only from "
                "ACGTUN and looks like a nucleotide sequence. Pass the receptor's "
                "protein sequence instead."
            )
    return seqs


def read_library(path: Path) -> list[tuple[str, str]]:
    """Return [(name, smiles)] with unique, filesystem-safe names."""
    out: list[tuple[str, str]] = []
    used: set[str] = set()
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        smiles = fields[0]
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", fields[1]).strip("_") if len(fields) > 1 else ""
        name = name or f"cmpd{lineno:06d}"
        while name in used:                      # names need not be unique in a .smi
            name += "_x"
        used.add(name)
        out.append((name, smiles))
    if not out:
        raise SystemExit(f"No SMILES found in {path}")
    return out


def split_ids(value: str) -> str | list[str]:
    """Split a comma-separated chain list; a single id stays a plain string."""
    ids = [part.strip() for part in value.split(",") if part.strip()]
    return ids[0] if len(ids) == 1 else ids


def build_templates(args, seqs: list[str]) -> list[dict]:
    """Boltz ``templates:`` entries from the CLI, with absolute paths.

    The paths are resolved here because `boltz predict` runs as a subprocess in
    the same working directory, but a relative path in the YAML would then depend
    on where the screen was launched from.
    """
    receptor_chains = {chr(ord("A") + i) for i in range(len(seqs))}
    entries = []
    for path in args.template or []:
        entry: dict = {
            "pdb" if path.suffix.lower() == ".pdb" else "cif": str(path.resolve())
        }
        if args.template_chain:
            chains = split_ids(args.template_chain)
            wanted = [chains] if isinstance(chains, str) else chains
            unknown = sorted(set(wanted) - receptor_chains)
            if unknown:
                raise SystemExit(
                    f"--template-chain {', '.join(unknown)} is not a receptor chain "
                    f"({', '.join(sorted(receptor_chains))})"
                )
            entry["chain_id"] = chains
        if args.template_id:
            entry["template_id"] = split_ids(args.template_id)
        if args.template_force:
            entry["force"] = True
            entry["threshold"] = args.template_threshold
        entries.append(entry)
    return entries


def parse_residues(spec: str, receptor_chains: set[str]) -> list[list]:
    """``A:45,A:46-48`` -> ``[['A',45], ['A',46], ['A',47], ['A',48]]`` (1-based)."""
    contacts = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        chain, sep, residue = item.partition(":")
        if not sep:                      # no chain given: the first receptor chain
            chain, residue = chr(ord("A")), chain
        chain = chain.strip().upper()
        if chain not in receptor_chains:
            raise SystemExit(
                f"--pocket chain {chain!r} is not a receptor chain "
                f"({', '.join(sorted(receptor_chains))})"
            )
        bounds = residue.split("-")
        try:
            if len(bounds) == 1:
                indices = [int(bounds[0])]
            elif len(bounds) == 2:
                start, end = int(bounds[0]), int(bounds[1])
                indices = list(range(start, end + 1))
            else:
                raise ValueError
        except ValueError:
            raise SystemExit(
                "--pocket expects 1-based residue indices like 'A:45' or 'A:46-48', "
                f"got {item!r}"
            ) from None
        if not indices or any(index < 1 for index in indices):
            raise SystemExit(f"--pocket residue indices start at 1: {item!r}")
        contacts.extend([chain, index] for index in indices)
    if not contacts:
        raise SystemExit("--pocket was given but contains no residues")
    return contacts


def build_constraints(args, seqs: list[str]) -> list[dict]:
    """Boltz ``constraints:`` entries from the CLI (pocket only)."""
    if not args.pocket:
        return []
    receptor_chains = {chr(ord("A") + i) for i in range(len(seqs))}
    pocket: dict = {
        "binder": args.ligand_id,
        "contacts": parse_residues(args.pocket, receptor_chains),
        "max_distance": args.pocket_max_distance,
    }
    if args.pocket_force:
        pocket["force"] = True
    return [{"pocket": pocket}]


def write_yaml(path: Path, seqs: list[str], msas: list[Path], smiles: str,
               ligand_id: str, affinity: bool,
               templates: list[dict], constraints: list[dict]) -> None:
    """One Boltz-2 input: the receptor chains with their MSAs, plus one ligand."""
    doc: dict = {
        "version": 1,
        "sequences": [
            {"protein": {"id": chr(ord("A") + i), "sequence": s, "msa": str(msas[i])}}
            for i, s in enumerate(seqs)
        ] + [{"ligand": {"id": ligand_id, "smiles": smiles}}],
    }
    if templates:
        doc["templates"] = templates
    if constraints:
        doc["constraints"] = constraints
    if affinity:
        doc["properties"] = [{"affinity": {"binder": ligand_id}}]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


# ============================================================== stages


def fetch_msa(seqs: list[str], out: Path, args) -> list[Path]:
    """Return one MSA per receptor chain, reused by every compound.

    Boltz pairs each chain with its own alignment, and the featurizer silently
    falls back to single-sequence mode for a chain whose MSA does not match its
    sequence. So a multi-chain receptor needs one CSV *per chain*, passed as
    repeated --msa flags in FASTA order; giving them all the same file would
    leave every chain but one without evolutionary information.

    The automatically generated (single-chain) file is <out>/msa/receptor_0.csv,
    matching the layout in examples/msa/.
    """
    if args.msa:
        missing = [p for p in args.msa if not p.is_file()]
        if missing:
            raise SystemExit(f"--msa not found: {', '.join(map(str, missing))}")
        if len(args.msa) != len(seqs):
            raise SystemExit(
                f"--msa was given {len(args.msa)} file(s) for {len(seqs)} receptor "
                "chain(s). Pass one --msa per chain, in FASTA order "
                "(e.g. --msa chain_a.csv --msa chain_b.csv)."
            )
        return [p.resolve() for p in args.msa]

    msa_dir = out / "msa"
    cached = msa_dir / "receptor_0.csv"
    if cached.is_file() and not args.override:
        print(f"MSA      reusing {cached}")
        return [cached]
    if len(seqs) > 1:
        raise SystemExit(
            "Automatic MSA generation covers a single-chain receptor only. For a "
            "multi-chain receptor, pass one --msa per chain, in FASTA order "
            "(e.g. --msa chain_a.csv --msa chain_b.csv)."
        )

    print(f"MSA      fetching from {args.msa_server_url} (once, then reused)")
    msa_dir.mkdir(parents=True, exist_ok=True)
    from boltz.main import compute_msa                      # noqa: PLC0415

    compute_msa(
        data={"receptor_0": seqs[0]},
        target_id="receptor",
        msa_dir=msa_dir,
        msa_server_url=args.msa_server_url,
        msa_pairing_strategy=args.msa_pairing_strategy,
        msa_server_username=args.msa_server_username,
        msa_server_password=args.msa_server_password,
        api_key_header=args.api_key_header,
        api_key_value=args.api_key_value,
    )
    if not cached.is_file():
        raise SystemExit(f"MSA server returned but {cached} was not written")
    return [cached]


def share_msas_by_sequence(seqs: list[str], msas: list[Path]) -> list[Path]:
    """Reconcile the per-chain MSA list with Boltz's one-MSA-per-entity rule.

    Boltz groups identical chains into a single entity and rejects an entity
    whose chains name different MSA files. An alignment is keyed by sequence,
    not by chain, so duplicated chains share the first chain's file, and a
    different file passed for a later copy is ignored (with a note). The
    reverse case -- different sequences sharing one file -- cannot work, since
    a CSV's first sequence can match at most one chain, and is an error.
    """
    first_path: dict[Path, int] = {}
    for i, msa in enumerate(msas):
        j = first_path.setdefault(Path(msa), i)
        if j != i and seqs[j] != seqs[i]:
            raise SystemExit(
                f"--msa {Path(msa).name} is given for chains "
                f"{chr(ord('A') + j)} and {chr(ord('A') + i)}, whose sequences "
                "differ. A Boltz MSA file belongs to one sequence; give each "
                "chain its own alignment."
            )

    first_seq: dict[str, int] = {}
    out: list[Path] = []
    for i, (seq, msa) in enumerate(zip(seqs, msas)):
        j = first_seq.setdefault(seq, i)
        kept = msas[j]
        if j != i and Path(msa) != Path(kept):
            print(
                f"MSA      chain {chr(ord('A') + i)} has the same sequence as "
                f"chain {chr(ord('A') + j)}; sharing {kept.name} "
                f"(ignoring {Path(msa).name})"
            )
        out.append(kept)
    return out


def write_inputs(out: Path, tag: str, compounds, seqs, msas, affinity, args,
                 templates: list[dict], constraints: list[dict]) -> Path:
    """Write one YAML per compound. Marked complete so a rerun can skip it.

    The marker alone cannot say *what* went into the YAMLs, so the recipe
    (MSAs, ligand id, templates, constraints) is stored next to it. Changing any
    of those and re-running into the same directory would otherwise keep the old
    inputs -- and the old predictions -- while the command line says otherwise.
    """
    yaml_dir, marker = out / f"{tag}_input", out / f"{tag}_input.done"
    meta_path = out / f"{tag}_input.recipe.json"
    recipe = {
        "ligand_id": args.ligand_id,
        "msas": [str(path) for path in msas],
        "templates": templates,
        "constraints": constraints,
    }
    if args.override and yaml_dir.exists():
        shutil.rmtree(yaml_dir)
        marker.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    if marker.exists():
        if meta_path.is_file():
            try:
                stored = json.loads(meta_path.read_text())
            except ValueError:
                stored = None
            if stored != recipe:
                raise SystemExit(
                    f"{yaml_dir.name} was generated from a different recipe (MSA, "
                    "ligand id, templates or constraints changed). Re-run with "
                    "--override, or use a fresh --out-dir."
                )
        elif templates or constraints:
            raise SystemExit(
                f"{yaml_dir.name} predates the template/constraint options, so its "
                "YAMLs cannot be checked against the current flags. Re-run with "
                "--override, or use a fresh --out-dir."
            )

    if not marker.exists():
        yaml_dir.mkdir(parents=True, exist_ok=True)
        for name, smiles in compounds:
            write_yaml(yaml_dir / f"{name}.yaml", seqs, msas, smiles, args.ligand_id,
                       affinity, templates, constraints)
        meta_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
        marker.touch()
    return yaml_dir


def resolve_boltz_bin(requested: str) -> Path:
    """Locate the `boltz` executable used for the two prediction passes.

    `boltz screen` runs `boltz predict` as a subprocess, so the executable has to
    be findable. PATH is the normal route; when the parent `boltz` was invoked by
    absolute path (common under schedulers, where its bin/ need not be on PATH),
    the sibling in the active interpreter's bin/ directory is the same
    executable and is used as a fallback.
    """
    found = shutil.which(requested)
    if found:
        return Path(found)
    sibling = Path(sys.executable).parent / requested
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return sibling
    raise SystemExit(
        f"cannot find the `boltz` executable ({requested!r}). `boltz screen` runs "
        "`boltz predict` as a subprocess; install the package into this "
        "environment or pass --boltz-bin /path/to/boltz."
    )


def run_boltz(
    inputs: Path,
    out_dir: Path,
    extra: list[str],
    args,
    timing_file: Path | None = None,
) -> Path:
    """Invoke `boltz predict` and return its predictions directory."""
    cmd = [args.boltz_bin, "predict", str(inputs), "--out_dir", str(out_dir),
           "--accelerator", args.accelerator, "--devices", str(args.devices),
           "--chunk_size_threshold", str(args.chunk_size_threshold),
           *DEFAULT_STRUCTURE, *extra]
    if args.variant == "d17":
        cmd += ["--drop_pairformer_layers", D17_LAYERS,
                "--student_pairformer", str(args.student_pairformer)]
    if args.cache:
        cmd += ["--cache", str(args.cache)]
    if args.no_kernels:
        cmd += ["--no_kernels"]
    if getattr(args, "trunk_kernels", False):
        cmd += ["--trunk_kernels"]
        # only meaningful with the kernels on; predict defaults to the same row anyway
        cmd += ["--kernel_row", args.kernel_row or "fast"]
    if args.override:
        cmd += ["--override"]
    if timing_file is not None:
        cmd += ["--library_timing_file", str(timing_file)]
    cmd += args.boltz_arg

    print("\n$ " + " ".join(cmd), flush=True)
    try:
        completed = subprocess.run(cmd, check=False)
    except OSError as exc:
        # Backstop for the gap between resolve_boltz_bin() and this call
        # (removed binary, lost execute permission, ...).
        raise SystemExit(
            f"could not run {args.boltz_bin!r}: {exc}. `boltz screen` calls "
            "`boltz predict` as a subprocess; check --boltz-bin."
        ) from exc
    if completed.returncode != 0:
        raise SystemExit("boltz predict failed; see its output above")

    found = sorted(out_dir.glob("boltz_results_*/predictions"))
    if not found:
        raise SystemExit(f"boltz predict wrote no predictions under {out_dir}")
    return found[0]


def warn_missing(kind: str, missing: list[tuple[str, str]], total: int,
                 out: Path | None) -> None:
    """Say loudly which compounds are absent from a stage, and why.

    A warning, not a failure: RDKit legitimately cannot build 3D coordinates for
    some ligands, boltz swallows a CUDA OOM inside predict_step (one printed line,
    exit 0), and a file left half-written by a killed job is unreadable. None of
    those should take the ranking down with them, so the compound is reported and
    skipped. The list is written as ``name<TAB>reason``.
    """
    if not missing:
        return
    print()
    print(f"  !! WARNING: {len(missing)} of {total} compounds have no {kind}")
    first = ", ".join(f"{n} ({why})" for n, why in missing[:5])
    print(f"     first few: {first}" + (" ..." if len(missing) > 5 else ""))
    print("     Either RDKit could not build the ligand, a prediction hit a CUDA")
    print("     OOM -- boltz catches that inside predict_step and the run still")
    print("     exits 0 -- or a file was left half-written. Check the boltz log for")
    print("     'ran out of memory'; if that is it, lower --chunk-size-threshold and")
    print("     rerun (finished compounds are kept).")
    if out is not None:
        out.write_text("".join(f"{n}\t{why}\n" for n, why in missing))
        print(f"     full list -> {out}")
    print()


def score_structure(pred: Path, compounds, n_protein: int,
                    out: Path | None = None) -> list[dict]:
    """Read each PAE matrix and rank the library by min-iPAE, lowest first."""
    rows, missing, split_checked = [], [], False
    for name, smiles in compounds:
        npz = pred / name / f"pae_{name}_model_0.npz"
        if not npz.is_file():
            missing.append((name, "no PAE matrix"))
            continue
        # npz members are decompressed on access, so a truncated file passes
        # np.load and fails here. One bad compound must not sink the ranking.
        try:
            with np.load(npz) as data:
                pae = data["pae"]
        except Exception as exc:  # noqa: BLE001
            missing.append((name, f"unreadable PAE ({type(exc).__name__}: {exc})"))
            continue

        # A readable matrix can still be unusable: a square matrix with no ligand
        # tokens would make the min-iPAE block empty and .min() raise.
        if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or pae.shape[0] <= n_protein:
            missing.append(
                (name, f"PAE has shape {pae.shape} for {n_protein} receptor residues")
            )
            continue

        # A wrong protein/ligand boundary silently scores the wrong block, so
        # check it once against the ligand's heavy-atom count before trusting any
        # of the numbers.
        if not split_checked:
            n_ligand = pae.shape[0] - n_protein
            expected = heavy_atoms(smiles)
            if n_ligand <= 0 or (expected is not None and n_ligand != expected):
                raise SystemExit(
                    "Protein/ligand token boundary does not check out:\n"
                    f"  receptor residues (FASTA) : {n_protein}\n"
                    f"  PAE matrix                : {pae.shape[0]} tokens\n"
                    f"  implied ligand tokens     : {n_ligand}\n"
                    f"  heavy atoms in {name}     : {expected}\n"
                    "min-iPAE would come from the wrong block. Check that the "
                    "FASTA holds the receptor and nothing else."
                )
            split_checked = True

        rows.append({"name": name, "smiles": smiles,
                     "min_ipae": min_ipae(pae, n_protein)})

    rows.sort(key=lambda r: r["min_ipae"])
    for rank, row in enumerate(rows, 1):
        row["min_ipae_rank"] = rank
    warn_missing("PAE matrix", missing, len(compounds),
                 out / "stage1_missing.txt" if out else None)
    if not rows:
        raise SystemExit("No PAE matrices were produced; nothing to rank.")
    return rows


def heavy_atoms(smiles: str) -> int | None:
    """Heavy-atom count, used only for the boundary check. None if RDKit is absent."""
    try:
        from rdkit import Chem, RDLogger                     # noqa: PLC0415
    except ImportError:
        return None
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else mol.GetNumAtoms()


def score_affinity(pred: Path, survivors: list[dict],
                   out: Path | None = None) -> list[dict]:
    """Attach binding probability to each survivor and rank by it, highest first."""
    hits, missing = [], []
    for row in survivors:
        js = pred / row["name"] / f"affinity_{row['name']}.json"
        if not js.is_file():
            missing.append((row["name"], "no affinity JSON"))
            continue
        try:
            data = json.loads(js.read_text())
            probability = data["affinity_probability_binary"]
        except (OSError, ValueError, KeyError) as exc:
            # Same defensive stance as the PAE reader: a half-written JSON should
            # cost this compound its score, not the whole hit list.
            missing.append(
                (row["name"], f"unreadable affinity JSON ({type(exc).__name__}: {exc})")
            )
            continue
        hits.append({**row,
                     "affinity_probability_binary": probability,
                     "affinity_pred_value": data.get("affinity_pred_value")})
    warn_missing("affinity JSON", missing, len(survivors),
                 out / "stage2_missing.txt" if out else None)
    hits.sort(key=lambda r: r["affinity_probability_binary"], reverse=True)
    return hits


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================== cli


def parse_args(
    argv: list[str] | tuple[str, ...] | None = None,
    *,
    prog: str = "boltz screen",
) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--receptor", required=True, type=Path, help="receptor FASTA")
    p.add_argument("--library", required=True, type=Path, help="SMILES library")
    p.add_argument("--out-dir", required=True, type=Path, help="output directory")

    g = p.add_argument_group("cascade")
    g.add_argument("--keep", type=float, default=0.05,
                   help="user-adjustable fraction retained by the structure pathway (default: 0.05)")
    g.add_argument("--keep-n", type=int, metavar="N",
                   help="retain exactly N compounds (positive), overriding --keep")
    g.add_argument("--stop-after-structure", action="store_true",
                   help="stop once the library is ranked, before the affinity pass")

    g = p.add_argument_group("model")
    g.add_argument(
        "--variant", choices=["default", "d17"], default="default",
        help="structure-stage configuration: 'default' uses the released weights; "
             "'d17' drops 17 Pairformer layers and loads the distilled student "
             "(requires --student-pairformer)")
    g.add_argument(
        "--student-pairformer", type=Path,
        help="distilled student weights for --variant d17; rejected with any other "
             "variant")
    g.add_argument("--ligand-id", default="B", help="chain id for the ligand")

    g = p.add_argument_group("structure guidance (written into every input)")
    g.add_argument(
        "--template", type=Path, action="append", metavar="CIF/PDB",
        help="structural template for the receptor, applied to every compound; "
             "repeatable. The path is stored as an absolute path")
    g.add_argument(
        "--template-chain", metavar="ID[,ID]",
        help="input chain(s) the template applies to (default: Boltz picks the "
             "best-matching protein chains)")
    g.add_argument(
        "--template-id", metavar="ID[,ID]",
        help="chain id(s) inside the template file, paired with --template-chain "
             "(default: Boltz searches the template's protein chains)")
    g.add_argument(
        "--template-force", action="store_true",
        help="restrain the backbone to the template; requires --template-threshold")
    g.add_argument(
        "--template-threshold", type=float, metavar="ANGSTROM",
        help="allowed backbone deviation for --template-force, in Angstroms")
    g.add_argument(
        "--pocket", metavar="RESIDUES",
        help="receptor residues the ligand should contact, e.g. 'A:45,A:46-48' "
             "(chain defaults to A). Adds a Boltz pocket constraint binding the "
             "ligand chain to these residues")
    g.add_argument(
        "--pocket-max-distance", type=float, default=6.0, metavar="ANGSTROM",
        help="pocket contact distance, 4-20 A (default: 6.0)")
    g.add_argument(
        "--pocket-force", action="store_true",
        help="apply the pocket constraint as a potential, not just a feature")

    g = p.add_argument_group("MSA")
    g.add_argument(
        "--msa", type=Path, action="append", metavar="CSV",
        help="precomputed receptor MSA CSV; skips the server. Repeat once per "
             "chain, in FASTA order (single-chain receptors pass it once)")
    g.add_argument(
        "--msa-server-url", default="https://api.colabfold.com",
        help="MSA server used when --msa is not given")
    g.add_argument(
        "--msa-pairing-strategy", default="greedy",
        help="MSA server pairing strategy: greedy or complete")
    g.add_argument("--msa-server-username", help="basic-auth username for the MSA server")
    g.add_argument("--msa-server-password", help="basic-auth password (give it with the username)")
    g.add_argument(
        "--api-key-header",
        help="API-key header name (default: X-API-Key); requires --api-key-value")
    g.add_argument(
        "--api-key-value", help="API-key value; the header defaults to X-API-Key")

    g = p.add_argument_group("runtime")
    g.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"],
                   help="gpu or cpu (default: gpu)")
    g.add_argument("--devices", type=int, default=1,
                   help="devices for boltz predict (default: 1); timing and "
                        "profiling are single-device only")
    g.add_argument("--cache", type=Path,
                   help="Boltz cache directory (default: ~/.boltz or $BOLTZ_CACHE)")
    g.add_argument("--no-kernels", action="store_true",
                   help="disable the cuEquivariance kernels (pure-PyTorch fallback); "
                        "cannot be combined with --trunk-kernels or --affinity-kernels")
    g.add_argument(
        "--trunk-kernels", action="store_true",
        help="apply the third-party fused trunk kernels to the structure pass; with "
             "--affinity-kernels they also serve the affinity pass (needs opt_core + "
             "boltz2_opt installed; a no-op with a message otherwise). Cannot be "
             "combined with --no-kernels")
    g.add_argument(
        "--kernel-row", default=None, choices=["fast", "fast-lowmem", "big"],
        help="which kit row --trunk-kernels applies (requires --trunk-kernels; "
             "'fast' when omitted). 'fast-lowmem' drops the fused pair-weighted "
             "averaging, worth ~4 GB of peak on compute capability 8.6; stage 1 sees "
             "the whole library, so that is where a 24 GB card runs out first")
    g.add_argument(
        "--affinity-kernels", action="store_true",
        help="give the affinity model the cuEquivariance triangle kernels in stage 2; "
             "upstream passes them to the structure model only. Also what lets the fused "
             "trunk kernels serve the affinity pass, which checks the same switch")
    g.add_argument(
        "--chunk-size-threshold",
        type=int,
        default=DEFAULT_SCREEN_CHUNK_SIZE_THRESHOLD,
        help=(
            "Pairformer chunking threshold forwarded to both prediction stages "
            f"(default: {DEFAULT_SCREEN_CHUNK_SIZE_THRESHOLD}; lower values use "
            "less GPU memory)"
        ),
    )
    g.add_argument("--override", action="store_true", help="recompute instead of resuming")
    g.add_argument("--boltz-bin", default="boltz",
                   help="boltz executable used for the two predict passes "
                        "(default: boltz on PATH)")
    g.add_argument("--boltz-arg", action="append", default=[],
                   help="extra argument forwarded to boltz predict; repeatable. "
                        "Values that start with a dash need the --boltz-arg=--flag form")
    g.add_argument(
        "--timing",
        action="store_true",
        help=(
            "write library_scaling_timing.json, excluding model loading and MSA "
            "generation while including per-compound input processing, both "
            "prediction stages, ranking, and output writing. Single-device only; "
            "requires a fresh --out-dir, and --stop-after-structure writes no file"
        ),
    )

    args = p.parse_args(argv)
    if args.variant == "d17" and not args.student_pairformer:
        p.error("--variant d17 requires --student-pairformer")
    if not 0 < args.keep <= 1:
        p.error("--keep must be in (0, 1]")
    if args.keep_n is not None and args.keep_n < 1:
        p.error("--keep-n must be a positive number of compounds")
    if args.chunk_size_threshold <= 0:
        p.error("--chunk-size-threshold must be positive")
    if any(
        value == "--chunk_size_threshold"
        or value.startswith("--chunk_size_threshold=")
        for value in args.boltz_arg
    ):
        p.error(
            "use --chunk-size-threshold instead of forwarding "
            "--chunk_size_threshold through --boltz-arg"
        )
    if args.devices < 1:
        p.error("--devices must be >= 1")
    if args.devices > 1:
        # Every rank would run the script, so a timing file is written N times at
        # once and only rank 0's numbers are meaningful; --profile also
        # synchronizes inside every forward. Both are single-device diagnostics.
        forwarded = {value.split("=", 1)[0] for value in args.boltz_arg}
        if args.timing or forwarded & {"--profile", "--library_timing_file"}:
            p.error(
                "timing and profiling are single-device measurements; they "
                "cannot be used with --devices > 1. Run with --devices 1."
            )

    if args.student_pairformer and args.variant != "d17":
        p.error("--student-pairformer requires --variant d17")
    if args.student_pairformer and not args.student_pairformer.is_file():
        p.error(f"--student-pairformer not found: {args.student_pairformer}")
    if args.kernel_row is not None and not args.trunk_kernels:
        p.error("--kernel-row requires --trunk-kernels")
    if bool(args.msa_server_username) != bool(args.msa_server_password):
        p.error(
            "--msa-server-username and --msa-server-password must be given together"
        )
    if args.api_key_header and not args.api_key_value:
        p.error("--api-key-header requires --api-key-value")

    # Template flags only mean something with a template, and Boltz ignores
    # threshold unless force is set -- fail here rather than silently.
    template_only = {
        "--template-chain": args.template_chain,
        "--template-id": args.template_id,
        "--template-force": args.template_force,
        "--template-threshold": args.template_threshold,
    }
    if not args.template:
        given = [flag for flag, value in template_only.items() if value not in (None, False)]
        if given:
            p.error(f"{', '.join(given)} requires --template")
    else:
        for path in args.template:
            if path.suffix.lower() not in (".cif", ".mmcif", ".pdb"):
                p.error(f"--template expects a .cif/.mmcif/.pdb file, got {path.name}")
            if not path.is_file():
                p.error(f"--template not found: {path}")
        if args.template_id and not args.template_chain:
            p.error("--template-id requires --template-chain (they are paired)")
        if args.template_force and args.template_threshold is None:
            p.error("--template-force requires --template-threshold (Angstroms)")
        if args.template_threshold is not None and not args.template_force:
            p.error("--template-threshold is only used together with --template-force")
        if args.template_threshold is not None and args.template_threshold <= 0:
            p.error("--template-threshold must be positive")

    if not args.pocket:
        if args.pocket_force or args.pocket_max_distance != 6.0:
            p.error("--pocket-max-distance/--pocket-force require --pocket")
    elif not 4.0 <= args.pocket_max_distance <= 20.0:
        p.error("--pocket-max-distance must be between 4 and 20 Angstroms")

    # --no-kernels turns off the cuEquivariance path only; the kit's levers are a
    # replacement for it and mostly ignore the flag. Refuse the mix instead of
    # running a configuration where only some kernels fall back.
    if args.no_kernels and (args.trunk_kernels or args.affinity_kernels):
        kit_flags = [
            flag
            for flag, enabled in (
                ("--trunk-kernels", args.trunk_kernels),
                ("--affinity-kernels", args.affinity_kernels),
            )
            if enabled
        ]
        p.error(
            f"--no-kernels cannot be combined with {' or '.join(kit_flags)}: "
            "--no-kernels disables the cuEquivariance path only, while the kit's "
            "levers keep serving. Use one or the other."
        )
    return args


def main(
    argv: list[str] | tuple[str, ...] | None = None,
    *,
    prog: str = "boltz screen",
) -> None:
    args = parse_args(argv, prog=prog)
    # Resolve the subprocess executable before any MSA work, so a missing or
    # non-executable `boltz` fails now rather than after the receptor MSA and the
    # per-compound YAMLs have been produced.
    args.boltz_bin = str(resolve_boltz_bin(args.boltz_bin))
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.timing and any(
        (out / name).exists()
        for name in ("stage1_input.done", "stage2_input.done", "stage1", "stage2")
    ):
        raise SystemExit(
            "--timing requires a fresh --out-dir; resumed or cached stages would "
            "not provide a complete timing measurement."
        )

    if args.stop_after_structure and args.affinity_kernels:
        print(
            "warning: --affinity-kernels has no effect with --stop-after-structure "
            "(the affinity stage does not run)",
            flush=True,
        )
    if args.stop_after_structure and args.timing:
        print(
            "warning: --timing does not write a timing file with "
            "--stop-after-structure; run the full cascade to measure",
            flush=True,
        )

    timing: dict[str, object] = {
        "schema_version": 1,
        "definition": "Library-scaling wall-clock time after one-time initialization",
        "variant": args.variant,
        "chunk_size_threshold": args.chunk_size_threshold,
        "retention_fraction_requested": args.keep,
        "excluded": [
            "model download",
            "model loading",
            "MSA generation",
            "scheduler queue time",
        ],
        "components_seconds": {},
    }
    components: dict[str, float] = timing["components_seconds"]  # type: ignore[assignment]

    seqs = read_fasta(args.receptor)
    n_protein = sum(map(len, seqs))
    _t0 = time.perf_counter()
    library = read_library(args.library)
    components["library_read"] = time.perf_counter() - _t0
    if args.ligand_id in {chr(ord("A") + i) for i in range(len(seqs))}:
        raise SystemExit(f"--ligand-id {args.ligand_id} collides with a receptor chain")
    print(f"Receptor {len(seqs)} chain(s), {n_protein} residues")
    print(f"Library  {len(library)} compounds")

    msas = share_msas_by_sequence(seqs, fetch_msa(seqs, out, args))

    # Structure guidance is shared by every compound, so it is built once and
    # copied into each generated YAML.
    templates = build_templates(args, seqs)
    constraints = build_constraints(args, seqs)
    if templates:
        print(f"Templates {len(templates)} file(s) applied to every compound")
    if constraints:
        n_contacts = len(constraints[0]["pocket"]["contacts"])
        print(f"Pocket    {n_contacts} receptor residue(s) the ligand should contact")

    # ---- stage 1: structure pathway over the whole library
    print("\n=== stage 1: structure pathway, whole library ===")
    _t0 = time.perf_counter()
    inputs = write_inputs(out, "stage1", library, seqs, msas, False, args,
                          templates, constraints)
    components["stage1_input_generation"] = time.perf_counter() - _t0
    stage1_timing_file = out / "stage1_boltz_timing.json" if args.timing else None
    pred = run_boltz(
        inputs,
        out / "stage1",
        ["--write_full_pae"],
        args,
        stage1_timing_file,
    )
    _t0 = time.perf_counter()
    ranked = score_structure(pred, library, n_protein, out)
    write_csv(out / "stage1_min_ipae.csv", ranked,
              ["min_ipae_rank", "name", "smiles", "min_ipae"])
    components["stage1_scoring_ranking_output"] = time.perf_counter() - _t0
    print(f"         {len(ranked)} ranked -> {out / 'stage1_min_ipae.csv'}")
    if args.stop_after_structure:
        return

    # ---- stage 2: affinity pathway on the survivors only
    # Stage 2 re-runs the structure pass for the survivors, which stage 1 has
    # already computed for them. That is deliberate, not an oversight.
    #
    # It could be avoided: the structure pass skips any record whose predictions
    # directory already exists (filter_inputs_structure), and the affinity pass
    # reads its input structure from pre_affinity_<id>.npz, so pointing stage 2 at
    # stage 1's output would chain them. The one missing piece is that the writer
    # only emits pre_affinity_<id>.npz when the input declares affinity, which
    # stage 1 does not.
    #
    # It is not worth the change. The duplicated work is k * main per compound of
    # the library, and on H100 that is 4.5% of the whole cascade at the default 5%
    # retention (2.8% at 3%, 8.1% at 10%): stage 1 dominates, and the affinity pass
    # is already compressed to roughly the cost of one structure pass. Realising it
    # would also mean writing one structure file for every compound in the library,
    # since the survivors are unknown until stage 1 finishes -- tens of GB on a
    # large library, to save a few per cent.
    _t0 = time.perf_counter()
    if args.keep_n is not None:
        n_keep = min(args.keep_n, len(ranked))
    else:
        n_keep = min(max(1, math.ceil(args.keep * len(ranked))), len(ranked))
    survivors = ranked[:n_keep]
    components["cascade_selection"] = time.perf_counter() - _t0
    print(f"\n=== stage 2: affinity pathway, top {n_keep} "
          f"({100 * n_keep / len(ranked):.1f}%) ===")
    _t0 = time.perf_counter()
    inputs = write_inputs(out, "stage2", [(r["name"], r["smiles"]) for r in survivors],
                          seqs, msas, True, args, templates, constraints)
    components["stage2_input_generation"] = time.perf_counter() - _t0
    stage2_timing_file = out / "stage2_boltz_timing.json" if args.timing else None
    pred = run_boltz(
        inputs,
        out / "stage2",
        # --affinity_kernels only bites here: stage 1 builds no affinity model
        DEFAULT_AFFINITY + (["--affinity_kernels"] if getattr(args, "affinity_kernels", False) else []),
        args,
        stage2_timing_file,
    )
    _t0 = time.perf_counter()
    hits = score_affinity(pred, survivors, out)
    for rank, row in enumerate(hits, 1):
        row["rank"] = rank
    write_csv(out / "hits.csv", hits,
              ["rank", "name", "smiles", "affinity_probability_binary",
               "min_ipae", "min_ipae_rank", "affinity_pred_value"])
    components["stage2_scoring_ranking_output"] = time.perf_counter() - _t0

    if args.timing:
        assert stage1_timing_file is not None and stage2_timing_file is not None
        missing_timing = [
            path for path in (stage1_timing_file, stage2_timing_file)
            if not path.is_file()
        ]
        if missing_timing:
            raise SystemExit(
                "Timing output was not produced by `boltz predict`: "
                + ", ".join(map(str, missing_timing))
                + f". Ensure `boltz` and `{prog}` come from the same "
                "updated checkout."
            )
        stage1_inner = json.loads(stage1_timing_file.read_text())
        stage2_inner = json.loads(stage2_timing_file.read_text())
        components.update({
            "stage1_boltz_input_processing": float(
                stage1_inner["input_processing_seconds"]
            ),
            "stage1_structure_predict": float(
                stage1_inner["structure_predict_seconds"]
            ),
            "stage2_boltz_input_processing": float(
                stage2_inner["input_processing_seconds"]
            ),
            "stage2_structure_predict": float(
                stage2_inner["structure_predict_seconds"]
            ),
            "stage2_affinity_predict": float(
                stage2_inner["affinity_predict_seconds"]
            ),
        })
        timing["library_size"] = len(library)
        timing["retained_count"] = n_keep
        timing["retention_fraction_actual"] = n_keep / len(library)
        timing["total_library_scaling_seconds"] = sum(components.values())
        timing["seconds_per_library_compound"] = (
            float(timing["total_library_scaling_seconds"]) / len(library)
        )
        timing_path = out / "library_scaling_timing.json"
        timing_path.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
        print("\nLibrary-scaling timing (one-time initialization excluded):")
        for name, seconds in components.items():
            print(f"  {name:36s} {seconds:10.4f} s")
        print(f"  {'TOTAL':36s} {timing['total_library_scaling_seconds']:10.4f} s")
        print(f"  JSON                                {timing_path}")

    print(f"\nDone. {len(hits)} hits -> {out / 'hits.csv'}")
    print("Rank on affinity_probability_binary. The regression output "
          "affinity_pred_value degrades under the accelerated settings and is "
          "carried through for reference only.")


if __name__ == "__main__":
    main()
