# Trunk distillation

Self-distillation that compresses the Boltz-2 trunk from 64 Pairformer layers to 47 (17 dropped), recovering most of the VS accuracy lost by the zero-shot depth truncation (drop-17). The 47-layer student weights are what BoltzVS-D17 loads at deployment.

- **teacher** = frozen 64-layer Pairformer; **student** = 17 layers dropped (→ 47 layers), trainable.
- **loss** = `MSE(s',s) + λ_z·MSE(z',z)` (+ cosine auxiliary); no diffusion/confidence, no labels, recycling=0.
- **precision** = `medium` matmul + bf16 autocast — **must match deployment**; training in any other precision leaves an irreducible floor in the (s, z) MSE.
- **output** = `student_pairformer_ft.pt`, loaded at inference together with `--drop_pairformer_layers` (see the top-level README).

This folder ships only the training entry point (`distill_train.py`). The data preparation was a one-off; the resulting ready-to-train bundle is published on Zenodo (see below), so the preprocessing scripts are not included.

## 1. Data (Zenodo)

Download `data_bundle.tar.gz` from **Zenodo (DOI: [10.5281/zenodo.21642066](https://doi.org/10.5281/zenodo.21642066))** and unpack it; the resulting `data_bundle/` directory is your `$BUNDLE`. Layout expected by `distill_train.py`:

```
$BUNDLE/
├── boltz2_conf.ckpt          # teacher/student init weights (--ckpt); = the standard Boltz-2 conf checkpoint
├── processed/
│   ├── manifest_clean.json   # training manifest, 4589 records ≤1024 tokens (--manifest)
│   ├── structures/           # boltz-2 StructureV2 .npz (--struct_dir)
│   └── mols/                 # per-record extra molecules; SMILES ligand chemistry (--extra_mols_dir)
├── msa/                      # MSA .npz, named by chain.msa_id, e.g. 1a0l_a.npz (--msa_dir)
└── mols/                     # boltz-2 canonical CCD molecule library (--mol_dir)
```

## 2. Environment

```bash
git clone https://github.com/shiyu-wangbyte/BoltzVS.git boltz
cd boltz
conda create -n boltzd python=3.10 -y && conda activate boltzd
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[cuda]"
```

> Training uses the pure-PyTorch path (`use_kernels=False`) and does not depend on the cuEquivariance ops; `pip install -e .` (without `[cuda]`) also works.

## 3. Running

Multi-GPU (launched with `torchrun`; the script does manual gradient all-reduce every `--accum` steps — **not** a DDP wrapper, the reasoning is in the script's comment block):

```bash
cd distill
torchrun --standalone --nproc_per_node=4 distill_train.py \
    --ckpt           $BUNDLE/boltz2_conf.ckpt \
    --manifest       $BUNDLE/processed/manifest_clean.json \
    --struct_dir     $BUNDLE/processed/structures \
    --extra_mols_dir $BUNDLE/processed/mols \
    --msa_dir        $BUNDLE/msa \
    --mol_dir        $BUNDLE/mols \
    --out_dir        $BUNDLE/student_pt \
    --max_tokens     1024 \
    --accum          8 \
    --epochs         12 \
    --lr             1e-5 \
    --auto_resume
```

Effective batch = `--accum` × number of GPUs. `--max_tokens 1024` covers the DEKOIS deployment size (median N ≈ 976); lower it if memory is tight.

Single-GPU debug (get it working before scaling out):

```bash
python distill_train.py ... --num_workers 2   # no torchrun → single-GPU branch
```

> If P2P collectives hang on a multi-socket node without NVLink (first all-reduce times out at 600s), `export NCCL_P2P_DISABLE=1`.

## 4. Self-check (what a healthy run prints)

- `R2 use_templates=False  bond_type_feature=False`
- gradient-isolation assert passes (teacher 0 trainable params, student all trainable); `student_pf.layers == 47`
- `world=<nproc>  train/rank≈… val/rank≈…`
- `R-selfcheck val_loss (untrained…)`: initial val_loss ≈ the zero-shot drop-17 level, and drops significantly with training.
- each epoch prints `val_loss` and (rank0 only) writes `{out}_ep{N}.pt`, `{out}` (best-val pointer), `{out}_state.pt` (rolling resume state).

## 5. Checkpoint resume

Each fully completed epoch atomically writes `{out}_state.pt` (student weights + AdamW optimizer + epoch + best). A half-run epoch is discarded and resumed from the last full one.

- `--auto_resume`: find `{stem}_state.pt` next to `--out` and resume; train from scratch if absent. Re-run the same command to relay.
- `--resume <path>`: resume from a specific state ckpt. `--ckpt` always stays the original 64-layer conf (teacher); `--resume` only overrides student/optimizer state.
- `--epochs` is the **total** target (resume starts at `completed_epoch+1`; on reaching it, a `{out}_DONE` marker is written).
- `--epochs_per_run 1`: run exactly one epoch then exit voluntarily — pair with `--auto_resume` to fit each job inside a cluster single-job time limit and relay until `_DONE`.
- **Before restarting or changing `--epochs`**: delete `{out}_state.pt` and `{out}_DONE`, else it resumes from old progress.

## 6. Hyperparameters

Fixed in the script: `λ_z=1.0, λ_cos=0.1, VAL_FRAC=0.05, SEED=42`, and the dropped-layer set `D17={2,3,4,5,6,7,16,18,19,26,27,28,29,30,31,32,33}`. Tunable via CLI: `--epochs` / `--accum` / `--lr` (+ optional `--lr_decay_epoch`/`--lr_final`), paths, `--num_workers`, `--max_tokens`, `--resume`/`--auto_resume`.
