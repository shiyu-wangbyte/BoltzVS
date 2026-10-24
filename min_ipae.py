"""Score an existing `boltz predict` run by min-iPAE.

`boltz screen` does this as part of the cascade; this script is for the case
where the structure pass has already been run by hand and only the ranking is
wanted.

    python min_ipae.py --predictions <boltz_results_*/predictions> \
                       --receptor receptor.fasta

The predictions must have been produced with --write_full_pae, which is what
writes the pae_*.npz files this reads. The receptor length can be given directly
with --n-protein instead of a FASTA.

The definition lives in boltz/screen/cascade.py and is imported here, so the two
cannot drift apart: min-iPAE is the smallest value in the protein-error /
ligand-frame block of the PAE matrix. See that module for why.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from boltz.screen.cascade import min_ipae, read_fasta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, type=Path,
                   help="the predictions/ directory of a boltz predict run")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--receptor", type=Path, help="receptor FASTA, to get its length")
    src.add_argument("--n-protein", type=int, help="receptor length in residues")
    p.add_argument("-o", "--output", type=Path, default=Path("min_ipae.csv"))
    args = p.parse_args()

    n_protein = args.n_protein or sum(map(len, read_fasta(args.receptor)))

    rows, skipped = [], []
    for sub in sorted(d for d in args.predictions.iterdir() if d.is_dir()):
        npz = sorted(sub.glob("pae_*_model_0.npz")) or sorted(sub.glob("pae*.npz"))
        if not npz:
            skipped.append((sub.name, "no pae*.npz -- was --write_full_pae set?"))
            continue
        # npz members are decompressed on access, so a file left half-written by a
        # killed job survives np.load and fails at the key lookup. Skip the record
        # rather than losing the whole ranking to it.
        try:
            with np.load(npz[0]) as data:
                pae = data["pae"]
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                (sub.name, f"unreadable {npz[0].name} ({type(exc).__name__}: {exc})")
            )
            continue
        if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
            skipped.append((sub.name, f"PAE is not square: {pae.shape}"))
        elif pae.shape[0] <= n_protein:
            skipped.append((sub.name, f"{pae.shape[0]} tokens <= {n_protein} residues, "
                                      "so no ligand tokens; check --n-protein"))
        else:
            rows.append((sub.name, min_ipae(pae, n_protein)))

    rows.sort(key=lambda r: r[1])
    with args.output.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "name", "min_iPAE"])
        for rank, (name, score) in enumerate(rows, 1):
            w.writerow([rank, name, f"{score:.6f}"])
    print(f"{len(rows)} scored -> {args.output}  (receptor length {n_protein})")
    for name, why in skipped[:10]:
        print(f"  skipped {name}: {why}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
