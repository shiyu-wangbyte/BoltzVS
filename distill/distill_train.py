#!/usr/bin/env python
"""Trunk distillation training (distill_train.py) -- multi-GPU version (manual gradient all-reduce).

teacher = frozen 64-layer Pairformer; student = 17 layers dropped (-> 47 layers), trainable.
loss = MSE(s',s) + lambda_z*MSE(z',z) [+ lambda_cos cosine auxiliary]; no diffusion/confidence,
no labels, recycling=0. The design constraints (R1-R11) are marked inline where they apply.

Paths come from --ckpt/--manifest/--struct_dir/--msa_dir/--mol_dir/--extra_mols_dir/--out_dir;
the defaults assume the Zenodo bundle unpacked as ./data_bundle (see distill/README.md).

Single-GPU debug:  python distill_train.py
4-GPU training:    torchrun --standalone --nproc_per_node=4 distill_train.py

Precision must match deployment: medium matmul + bf16 autocast (R11).
"""
import argparse
import copy
import time
import dataclasses
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from boltz.data import types as _bt
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule  # v2, with Boltz2 prefix
from boltz.data.types import Manifest
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2


# ===== Compatibility shim: the official preprocessed StructureV2 npz has more fields than this fork (e.g. connections) =====
# load via cls(**npz) would crash on the extra keys. Filter to the dataclass's known fields (drop extras); this only affects
# [loading] and does not touch the featurizer -> training/deployment still use the same fork's featurization, so consistency
# is unaffected. If a dtype also changed, this shim won't catch it (it will crash in featurization or surface as loss=NaN).
# With fork start (default on Linux), this patch is inherited by dataloader workers.
def _install_tolerant_structure_load():
    fields = {f.name for f in dataclasses.fields(_bt.StructureV2)}
    warned = set()

    def _load(cls, path):
        data = dict(np.load(path, allow_pickle=True))
        extra = tuple(sorted(set(data) - fields))
        if extra and extra not in warned:
            warned.add(extra)
            print(f"[compat] StructureV2.load dropping extra keys: {list(extra)}", flush=True)
        return cls(**{k: v for k, v in data.items() if k in fields})

    _bt.StructureV2.load = classmethod(_load)


_install_tolerant_structure_load()

# ===== Hyperparameters (fixed by design; only paths go through the CLI) =====
# EPOCHS/LR/ACCUM go through the CLI (see parse_args); the following are fixed hyperparameters
LAMBDA_Z, LAMBDA_COS, VAL_FRAC, SEED = 1.0, 0.1, 0.05, 42
D17 = {2, 3, 4, 5, 6, 7, 16, 18, 19, 26, 27, 28, 29, 30, 31, 32, 33}
DEFAULT_ROOT = Path("data_bundle")


def parse_args():
    p = argparse.ArgumentParser(description="trunk distillation (multi-GPU, manual gradient all-reduce)")
    _PROC = DEFAULT_ROOT / "processed"
    p.add_argument("--ckpt", type=Path, default=DEFAULT_ROOT / "boltz2_conf.ckpt",
                   help="initial weights for teacher/student")
    p.add_argument("--manifest", type=Path, default=_PROC / "manifest_clean.json",
                   help="training manifest (the Zenodo bundle's processed/manifest_clean.json)")
    p.add_argument("--struct_dir", type=Path, default=_PROC / "structures",
                   help="boltz-2 processed StructureV2 .npz files")
    p.add_argument("--msa_dir", type=Path, default=DEFAULT_ROOT / "msa",
                   help="reused MSA .npz files")
    p.add_argument("--mol_dir", type=Path, default=DEFAULT_ROOT / "mols",
                   help="boltz-2 canonical CCD molecule library")
    p.add_argument("--extra_mols_dir", type=Path, default=_PROC / "mols",
                   help="per-record extra molecules (SMILES ligand chemistry)")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_ROOT / "student_pt",
                   help="student weight output directory (ep{N}/best/state/DONE all written here); "
                        "default data_bundle/student_pt")
    p.add_argument("--out", type=Path, default=Path("student_pairformer_ft.pt"),
                   help="student weight filename (relative to --out_dir); if given an absolute path it is used directly and --out_dir is ignored")
    p.add_argument("--num_workers", type=int, default=2, help="number of dataloader workers per rank")
    p.add_argument("--max_tokens", type=int, default=1024,
                   help="filter out oversized complexes by token count (protein residues + ligand atoms); default 1024 = DEKOIS deployment max N~=998 + margin; 0=no limit")
    p.add_argument("--min_tokens", type=int, default=0,
                   help="only keep complexes with token count >= this value; combine with --max_tokens to isolate near-limit complexes for OOM probing (e.g. --min_tokens 950)")
    p.add_argument("--chunk_size_threshold", type=int, default=None,
                   help="chunk triangular attention to save memory once token count exceeds this (with force_chunk); defaults to the value in const.py (384)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N records for a quick smoke test (0=full set)")
    # ===== Training-amount hyperparameters (formerly hardcoded constants, exposed to CLI for tuning) =====
    p.add_argument("--epochs", type=int, default=3, help="number of training epochs (each epoch runs through each rank's own shard)")
    p.add_argument("--accum", type=int, default=16,
                   help="gradient accumulation steps: all-reduce + opt.step once every accum steps. Effective batch = accum x num GPUs; "
                        "smaller -> more frequent updates (learns more per wall-clock time), larger -> steadier gradients")
    p.add_argument("--lr", type=float, default=1e-5, help="AdamW learning rate")
    p.add_argument("--lr_decay_epoch", type=int, default=-1,
                   help="from which epoch (inclusive) to drop lr to --lr_final (step decay); -1=no decay (constant --lr throughout). "
                        "Judged by epoch -> correct for resume/split jobs")
    p.add_argument("--lr_final", type=float, default=None,
                   help="lr used from --lr_decay_epoch onward; defaults to --lr (i.e. no decay)")
    p.add_argument("--epochs_per_run", type=int, default=0,
                   help="max number of epochs to run this invocation before exiting voluntarily (0=run until --epochs). "
                        "Set 1 = each job runs one full epoch then stops, guaranteeing completion within a cluster single-job time limit without being killed mid-run; "
                        "combine with --auto_resume to resubmit the same pbs repeatedly until _DONE is written")
    # ===== Checkpoint resume (single job has a time limit -> resume across split jobs) =====
    p.add_argument("--resume", type=Path, default=None,
                   help="resume from a training-state ckpt (contains student weights + AdamW optimizer + epoch + best). "
                        "Note: --ckpt still points to the original 64-layer conf (used by teacher); --resume only overrides student/opt state")
    p.add_argument("--auto_resume", action="store_true",
                   help="when --resume is not given explicitly, automatically find {stem}_state.pt in the same directory as --out and resume; train from scratch if not found. "
                        "Each segment of a chained job can use the same command")
    return p.parse_args()


def ddp_setup():
    """Initialize DDP if launched via torchrun; otherwise single-GPU. Returns (is_ddp, rank, world, local_rank, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank, torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(0)
    return False, 0, 1, 0, torch.device("cuda:0")


def _atomic_save(obj, path):
    """Write to a temp file then os.replace, to avoid a corrupted ckpt if a job is wall-clock-killed mid-write."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():  # noqa: C901, PLR0915
    args = parse_args()
    is_ddp, rank, world, local_rank, device = ddp_setup()
    is_main = rank == 0
    # Output path resolution: relative filename -> lands under --out_dir; absolute path -> used directly
    # (pathlib: an absolute right operand overrides the left). ep{N}/best/state/DONE are all derived from
    # args.out -> all go into the same directory.
    args.out = args.out_dir / args.out
    if args.lr_final is None:
        args.lr_final = args.lr        # not specified -> no decay
    if is_main:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    def log(*a):
        if is_main:
            print(*a, flush=True)

    # ===== R11 precision alignment: must match deployment (medium + the bf16 autocast below) =====
    torch.set_float32_matmul_precision("medium")
    if args.chunk_size_threshold is not None:  # runtime override of the chunk threshold (with student_pf.force_chunk)
        from boltz.data import const as _cst
        _cst.chunk_size_threshold = args.chunk_size_threshold

    # ===== R1 load the model (must fill in all constructor args, otherwise it crashes) =====
    diffusion_params = Boltz2DiffusionParams()
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False
    predict_args = {"recycling_steps": 0, "sampling_steps": 2, "diffusion_samples": 1,
                    "max_parallel_samples": 1, "write_confidence_summary": False,
                    "write_full_pae": False, "write_full_pde": False}
    model = Boltz2.load_from_checkpoint(
        args.ckpt, strict=True, map_location=f"cuda:{local_rank}",
        predict_args=predict_args, diffusion_process_args=asdict(diffusion_params),
        ema=False, use_kernels=False,  # training uses the pure-PyTorch path (R5, backprop-able)
        pairformer_args=asdict(PairformerArgsV2()), msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(steering_args))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)  # fully frozen
    log("R2 use_templates=", model.use_templates, " bond_type_feature=", model.bond_type_feature)

    # ===== R2 front modules (copied from boltz2.py:426-494, single pass with recycling=0) =====
    def pre_pairformer(feats):
        with torch.no_grad():
            s_inputs = model.input_embedder(feats)
            s_init = model.s_init(s_inputs)
            z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
            rpe = model.rel_pos(feats)
            z_init = z_init + rpe + model.token_bonds(feats["token_bonds"].float())
            if model.bond_type_feature:
                z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
            z_init = z_init + model.contact_conditioning(feats)
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]
            s = s_init + model.s_recycle(model.s_norm(torch.zeros_like(s_init)))
            z = z_init + model.z_recycle(model.z_norm(torch.zeros_like(z_init)))
            if model.use_templates:
                tm = getattr(model.template_module, "_orig_mod", model.template_module)
                z = z + tm(z, feats, pair_mask, use_kernels=model.use_kernels)
            mm = getattr(model.msa_module, "_orig_mod", model.msa_module)
            z = z + mm(z, s_inputs, feats, use_kernels=model.use_kernels)
        return s, z, mask, pair_mask

    # ===== R6 teacher (original 64 layers, frozen) / student (d17 dropped -> 47 layers, trainable); R5 eval() disables dropout =====
    pf = getattr(model.pairformer_module, "_orig_mod", model.pairformer_module)
    teacher_pf = pf
    student_pf = copy.deepcopy(pf)
    student_pf.layers = nn.ModuleList([L for i, L in enumerate(student_pf.layers) if i not in D17])
    assert len(student_pf.layers) == 47
    for p in student_pf.parameters():
        p.requires_grad_(True)
    # Gradient checkpointing saves memory (recompute activations on backward); its checkpoint branch requires
    # self.training=True, so we use train() rather than eval() -- but we set each layer's dropout to 0, so even in
    # train mode there is no randomness -> still a clean match.
    student_pf.activation_checkpointing = True
    student_pf.force_chunk = True  # in train mode also chunk triangular attention per chunk_size_threshold to save memory (see pairformer.py)
    for layer in student_pf.layers:
        layer.dropout = 0.0
    student_pf.train().to(device)

    # [Do NOT use DistributedDataParallel]. DDP's automatic bucketing + autograd hooks conflict with gradient checkpointing:
    #   static_graph=True is needed to build the backward graph, but static_graph is incompatible with no_sync gradient
    #   accumulation (reducer assertion crash); dropping no_sync and all-reducing every step turns into "a barrier every step",
    #   dragged down by the slowest rank's large complex (measured ~30s/step).
    # Switch to [bare module + manual gradient sync]: each rank loads the same ckpt + same D17 -> initial weights are naturally
    # identical (no DDP broadcast needed); accumulate gradients locally for ACCUM steps, then at the boundary manually all-reduce
    # (SUM/world to average) and step -> ranks' weights stay in sync. Barriers go from once per step -> once per ACCUM steps,
    # amortizing the straggler cost; mathematically equivalent to "DDP syncing every step then accumulating".
    def sync_grads():
        """Sum and average each student parameter's .grad across ranks (no-op on single GPU)."""
        if not is_ddp:
            return
        for p in student_pf.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)  # SUM
                p.grad /= world

    def run_teacher(s, z, mask, pair_mask):
        return teacher_pf(s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels)

    def run_student(s, z, mask, pair_mask):
        return student_pf(s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels)

    # R8 gradient-isolation self-check
    n_mdl = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_mdl == 0 and all(p.requires_grad for p in student_pf.parameters()), "gradient isolation broken!"

    # ===== R7 data: inference data module; equal-length shard per rank (DDP requires all ranks have the same batch count, else hang) =====
    recs = Manifest.load(args.manifest).records
    if args.max_tokens or args.min_tokens:  # filter identically across ranks before sharding (else DDP hangs on mismatched batch counts)
        from boltz.data import const as _const
        _PROT = _const.chain_type_ids["PROTEIN"]

        def _n_tokens(rec):
            ch = np.load(args.struct_dir / f"{rec.id}.npz", allow_pickle=True)["chains"]
            return int(sum((c["res_num"] if int(c["mol_type"]) == _PROT else c["atom_num"]) for c in ch))

        before = len(recs)
        hi = args.max_tokens or 10 ** 9
        recs_n = [(r, _n_tokens(r)) for r in recs]
        recs = [r for r, n in recs_n if args.min_tokens <= n <= hi]
        mx = max((n for _, n in recs_n if args.min_tokens <= n <= hi), default=0)
        log(f"[tokens in [{args.min_tokens},{args.max_tokens or 'inf'}]] kept {len(recs)}/{before}"
            f"(dropped {before - len(recs)}); max token in kept set={mx}")
    if args.limit:  # smoke test: take only the first N records
        recs = recs[:args.limit]
    rng0 = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(recs), generator=rng0).tolist()
    n_val = max(world, int(len(recs) * VAL_FRAC))  # ensure at least 1 val per rank
    val_all = [recs[i] for i in perm[:n_val]]
    train_all = [recs[i] for i in perm[n_val:]]

    def shard(lst):
        n = (len(lst) // world) * world  # truncate to a multiple of world -> equal length per rank (drop < world items)
        return lst[rank:n:world]

    train_recs, val_recs = shard(train_all), shard(val_all)

    def make_loader(records):
        dm = Boltz2InferenceDataModule(
            # target_dir must be the directory directly containing {record.id}.npz = finetune_data/structures
            manifest=Manifest(records=records), target_dir=args.struct_dir,
            msa_dir=args.msa_dir, mol_dir=args.mol_dir,  # structures = processed dir; MSA = reuse existing; mol_dir = canonical CCD library
            extra_mols_dir=args.extra_mols_dir,  # SMILES ligand chemistry (produced by process_inputs); missing -> ligand load crashes -> recursion
            num_workers=args.num_workers)  # batch_size defaults to 1, shuffle=False -> same batch count across ranks
        return dm.predict_dataloader()

    train_loader, val_loader = make_loader(train_recs), make_loader(val_recs)
    log(f"world={world}  train/rank={len(train_recs)}  val/rank={len(val_recs)}")
    log(f"[cfg] epochs={args.epochs} accum={args.accum} lr={args.lr} "
        f"effective_batch={args.accum * world}  updates/epoch~={len(train_recs) // args.accum}")

    # ===== R9 masked MSE + cosine auxiliary (loss computed in FP32) =====
    def mse_masked(a, b, m):
        return ((a - b) ** 2 * m).sum() / (m.sum() * a.shape[-1] + 1e-8)

    def cos_aux(a, b, m):
        a2, b2 = nn.functional.normalize(a, dim=-1), nn.functional.normalize(b, dim=-1)
        return (((1 - (a2 * b2).sum(-1, keepdim=True)) * m).sum()) / (m.sum() + 1e-8)

    def to_dev(batch):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def forward_pair(feats):  # returns (s_s,z_s,s_t,z_t,ms,mz)
        with torch.autocast("cuda", dtype=torch.bfloat16):  # R10: teacher/student share the same bf16 context
            s_pre, z_pre, mask, pair_mask = pre_pairformer(feats)
            with torch.no_grad():
                s_t, z_t = run_teacher(s_pre, z_pre, mask, pair_mask)
            s_s, z_s = run_student(s_pre, z_pre, mask, pair_mask)
        return s_s, z_s, s_t, z_t, mask[..., None], pair_mask[..., None]

    def loss_fn(s_s, z_s, s_t, z_t, ms, mz):  # R10: reduce in FP32
        l = mse_masked(s_s.float(), s_t.float(), ms) + LAMBDA_Z * mse_masked(z_s.float(), z_t.float(), mz)
        if LAMBDA_COS:
            l = l + LAMBDA_COS * (cos_aux(s_s.float(), s_t.float(), ms) + cos_aux(z_s.float(), z_t.float(), mz))
        return l

    opt = torch.optim.AdamW(student_pf.parameters(), lr=args.lr, weight_decay=0.0)  # bf16 needs no GradScaler
    torch.cuda.reset_peak_memory_stats(device)  # monitor peak memory (for setting max_tokens)

    def evaluate():
        student_pf.eval()  # eval: enable triangular chunking, low memory; dropout=0 already guarantees determinism; no checkpoint needed under no_grad
        tot = torch.zeros((), device=device)
        n = torch.zeros((), device=device)
        for batch in val_loader:
            with torch.no_grad():
                s_s, z_s, s_t, z_t, ms, mz = forward_pair(to_dev(batch))
                tot += loss_fn(s_s, z_s, s_t, z_t, ms, mz).float()
                n += 1
        if is_ddp:  # reduce across ranks to get the global val_loss (identical on all ranks)
            dist.all_reduce(tot)
            dist.all_reduce(n)
        return (tot / torch.clamp(n, min=1)).item()

    # ===== Checkpoint resume: restore student weights + optimizer + epoch + best =====
    # teacher always comes from --ckpt (original 64 layers); --resume only overrides student/opt training state. Each rank reads
    # the same state ckpt on shared disk -> weights/optimizer are consistent after restore, not breaking manual all-reduce sync.
    # A half-run (unsaved) epoch is simply discarded and restarted from the next full epoch.
    best = float("inf")
    gstep = 0
    start_epoch = 0
    state_path = args.out.with_name(f"{args.out.stem}_state{args.out.suffix}")
    resume_path = args.resume
    if resume_path is None and args.auto_resume and state_path.exists():
        resume_path = state_path
    if resume_path is not None:
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"the state ckpt given to --resume does not exist: {resume_path}")
        ck = torch.load(resume_path, map_location=f"cuda:{local_rank}", weights_only=False)
        student_pf.load_state_dict(ck["student"])   # all ranks load -> stay in sync
        opt.load_state_dict(ck["opt"])
        # lr is not changed here: at the start of each epoch it is set uniformly per --lr/--lr_decay_epoch/--lr_final (see the
        # training loop); both resume and lr-decay are judged by epoch there, naturally correct (AdamW momentum is lr-independent, keep it).
        best = ck.get("best", float("inf"))
        gstep = int(ck.get("gstep", 0))
        start_epoch = int(ck["epoch"]) + 1          # continue after the last fully-completed epoch
        log(f"[resume] resuming from {resume_path}: completed epoch {ck['epoch']} -> starting from epoch {start_epoch}; "
            f"best={best:.5f} gstep={gstep}")
        if start_epoch >= args.epochs:
            log(f"[resume] start_epoch({start_epoch}) >= --epochs({args.epochs}), target epoch count reached, no more training needed.")
    else:
        log("[resume] no resume state, training from scratch.")

    # ===== Training loop =====
    # This run goes at most up to end_epoch: when --epochs_per_run>0, each job runs only K epochs then exits voluntarily
    # (completes within the <=6h limit without being killed mid-run); =0 runs all the way to --epochs.
    end_epoch = args.epochs if args.epochs_per_run <= 0 else min(start_epoch + args.epochs_per_run, args.epochs)
    if start_epoch == 0:  # only print the "untrained baseline" on a fresh start; on resume the previous job already gave the epoch-end val, saving a val forward pass
        log("R-selfcheck val_loss (untrained, should ~= zero-shot drop-17 level):", evaluate())
    log(f"[run] this run does epoch {start_epoch}..{end_epoch - 1} (target --epochs={args.epochs})")
    for ep in range(start_epoch, end_epoch):
        # this epoch's lr (step decay: ep < lr_decay_epoch uses --lr, else --lr_final; lr_decay_epoch=-1 -> --lr throughout)
        cur_lr = args.lr if (args.lr_decay_epoch < 0 or ep < args.lr_decay_epoch) else args.lr_final
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        log(f"[epoch {ep}] lr={cur_lr:g}")
        student_pf.train()  # train mode: activate gradient checkpointing (dropout already set to 0, introduces no randomness)
        opt.zero_grad()
        n_batches = len(train_loader)
        for step, batch in enumerate(train_loader):
            feats = to_dev(batch)
            # Bare module local backward (no DDP; each rank runs its own shard independently without waiting -> no per-step barrier).
            # Gradients accumulate in .grad; every ACCUM steps (or at epoch end) all at once: manual all-reduce to average -> step.
            # Ranks have the same batch count -> sync decisions align -> sync_grads()'s collective comm aligns and won't hang.
            s_s, z_s, s_t, z_t, ms, mz = forward_pair(feats)
            if gstep == 0:
                log("R3 keys:", sorted(feats))
                log("R9 shapes:", tuple(s_s.shape), tuple(z_s.shape))
            loss = loss_fn(s_s, z_s, s_t, z_t, ms, mz)
            (loss / args.accum).backward()
            if ((gstep + 1) % args.accum == 0) or (step == n_batches - 1):
                sync_grads()  # average accumulated gradients across ranks (once per ACCUM steps, the only barrier)
                opt.step()
                opt.zero_grad()
            if gstep % 50 == 0:
                log(f"[{time.strftime('%H:%M:%S')}] ep{ep} step{gstep} loss={loss.item():.5f}")
            gstep += 1
        vl = evaluate()  # consistent across all ranks
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        log(f"[epoch {ep}] val_loss={vl:.5f}  peak_mem={peak:.1f}GB")
        improved = vl < best
        best = min(best, vl)  # updated in sync on all ranks (vl is identical), no broadcast needed
        if is_main:  # save only on rank0; bare-module state_dict (not DDP-wrapped)
            # save one independent ckpt per epoch (no overwrite) -> can later evaluate DEKOIS and plot val/enrichment-vs-epoch convergence curves
            ep_path = args.out.with_name(f"{args.out.stem}_ep{ep}{args.out.suffix}")
            _atomic_save(student_pf.state_dict(), ep_path)
            log(f"  saved epoch ckpt (val {vl:.5f}) -> {ep_path}")
            if improved:  # additionally maintain a best-val pointer (convenient to "just use the best one")
                _atomic_save(student_pf.state_dict(), args.out)
                log(f"  best so far ({best:.5f}) -> {args.out}")
            # rolling training state (with optimizer/epoch/best/gstep) -> used for checkpoint resume; atomic write guards against interruption corruption.
            # written only after a full epoch completes -> guarantees the resume start point is always a complete epoch.
            _atomic_save({"epoch": ep, "student": student_pf.state_dict(),
                          "opt": opt.state_dict(), "best": best, "gstep": gstep,
                          "d17": sorted(D17), "lr": args.lr}, state_path)
            log(f"  saved resume state (epoch {ep}) -> {state_path}")
        if is_ddp:
            dist.barrier()  # while rank0 saves, the other ranks wait for it, keeping lockstep between epochs (prevents stragglers)
    log(f"run done (ran up to epoch {end_epoch - 1}/{args.epochs - 1}). best val_loss = {best}")
    if is_main and end_epoch >= args.epochs:
        # completion marker: written only when --epochs is truly reached (in split-job mode, not written until target reached -> next job continues).
        done_marker = args.out.with_name(f"{args.out.stem}_DONE")
        done_marker.write_text(f"epochs={args.epochs} best_val_loss={best:.6f}\n")
        log(f"[done] reached target {args.epochs} epochs, wrote completion marker -> {done_marker}")
    elif is_main:
        log(f"[run] target not reached (at epoch {end_epoch - 1}/{args.epochs - 1}); resubmit the same pbs to resume from epoch {end_epoch}.")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
