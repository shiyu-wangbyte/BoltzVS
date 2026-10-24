import multiprocessing
import json
import os
import pickle
import platform
import tarfile
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional

import time
import click
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from rdkit import Chem
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.mol import load_canonicals
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"

BOLTZ1_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
]

BOLTZ2_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
]

BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None


class LibraryPredictionTimer(Callback):
    """Time the prediction loop after Lightning has initialized the model/device."""

    def __init__(self) -> None:
        super().__init__()
        self.started: float | None = None
        self.elapsed_seconds: float | None = None

    @staticmethod
    def _synchronize(trainer: Trainer) -> None:
        device = getattr(trainer.strategy, "root_device", None)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    def on_predict_start(self, trainer: Trainer, pl_module) -> None:  # noqa: ANN001
        self._synchronize(trainer)
        self.started = time.perf_counter()
        self.elapsed_seconds = None

    def on_predict_end(self, trainer: Trainer, pl_module) -> None:  # noqa: ANN001
        self._synchronize(trainer)
        if self.started is not None:
            self.elapsed_seconds = time.perf_counter() - self.started


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = False


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20


@rank_zero_only
def download_boltz1(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ1_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ1_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


@rank_zero_only
def download_boltz2(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    mols = cache / "mols"
    tar_mols = cache / "mols.tar"
    if not tar_mols.exists():
        click.echo(
            f"Downloading the CCD data to {tar_mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        urllib.request.urlretrieve(MOL_URL, str(tar_mols))  # noqa: S310
    if not mols.exists():
        click.echo(
            f"Extracting the CCD data to {mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(cache)  # noqa: S202

    # Download model
    model = cache / "boltz2_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the Boltz-2 weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue

    # Download affinity model
    affinity_model = cache / "boltz2_aff.ckpt"
    if not affinity_model.exists():
        click.echo(
            f"Downloading the Boltz-2 affinity weights to {affinity_model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_AFFINITY_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(affinity_model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_AFFINITY_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


def get_cache_path() -> str:
    """Determine the cache path, prioritising the BOLTZ_CACHE environment variable.

    Returns
    -------
    str: Path
        Path to use for boltz cache location.

    """
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved_cache = Path(env_cache).expanduser().resolve()
        if not resolved_cache.is_absolute():
            msg = f"BOLTZ_CACHE must be an absolute path, got: {env_cache}"
            raise ValueError(msg)
        return str(resolved_cache)

    return str(Path("~/.boltz").expanduser())


def check_inputs(data: Path) -> list[Path]:
    """Check the input data and output directory.

    Parameters
    ----------
    data : Path
        The input data.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        for d in data:
            if d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            if d.suffix.lower() not in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)
    else:
        data = [data]

    return data


def filter_inputs_structure(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Filter the manifest to only include missing predictions.

    Parameters
    ----------
    manifest : Manifest
        The manifest of the input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    # Check if existing predictions are found (only top-level prediction folders)
    pred_dir = outdir / "predictions"
    if pred_dir.exists():
        existing = {d.name for d in pred_dir.iterdir() if d.is_dir()}
    else:
        existing = set()

    # Remove them from the input data
    if existing and not override:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        msg = (
            f"Found some existing predictions ({len(existing)}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = f"Found {len(existing)} existing predictions, will override."
        click.echo(msg)

    return manifest


def filter_inputs_affinity(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Check the input data and output directory for affinity.

    Parameters
    ----------
    manifest : Manifest
        The manifest.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    click.echo("Checking input data for affinity.")

    # Get all affinity targets
    existing = {
        r.id
        for r in manifest.records
        if r.affinity
        and (outdir / "predictions" / r.id / f"affinity_{r.id}.json").exists()
    }

    # Remove them from the input data
    if existing and not override:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        num_skipped = len(existing)
        msg = (
            f"Found some existing affinity predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "affinity predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing affinity predictions, will override."
        click.echo(msg)

    # Drop records whose structure pass produced no pre_affinity_<id>.npz.
    #
    # The affinity pass reads that file for every record it is handed, so a single
    # missing one raises FileNotFoundError inside a DataLoader worker and kills the
    # whole run -- on a 1,240-compound library, a handful of compounds the structure
    # pass skipped (predict_step swallows CUDA OOM and returns, see its except
    # clause) would take the other ~1,230 down with them. Skip them loudly instead:
    # they simply get no affinity score, which is what already happens to them on the
    # structure axis.
    missing = {
        r.id
        for r in manifest.records
        if r.affinity
        and not (
            outdir / "predictions" / r.id / f"pre_affinity_{r.id}.npz"
        ).exists()
    }
    if missing:
        manifest = Manifest([r for r in manifest.records if r.id not in missing])
        listing = outdir / "affinity_missing_inputs.txt"
        listing.write_text("\n".join(sorted(missing)) + "\n")
        click.echo(
            f"\n{'=' * 70}\n"
            f"WARNING: {len(missing)} input(s) have no pre_affinity_<id>.npz from the "
            f"structure pass\nand are skipped by the affinity pass. The usual cause is "
            f"a skipped structure\nprediction (out of memory). Their ids are listed in "
            f"{listing}.\n{'=' * 70}\n"
        )

    return manifest


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    msa_server_username : str, optional
        Username for basic authentication with MSA server.
    msa_server_password : str, optional
        Password for basic authentication with MSA server.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).

    """
    click.echo(f"Calling MSA server for target {target_id} with {len(data)} sequences")
    click.echo(f"MSA server URL: {msa_server_url}")
    click.echo(f"MSA pairing strategy: {msa_pairing_strategy}")
    
    # Construct auth headers if API key header/value is provided
    auth_headers = None
    if api_key_value:
        key = api_key_header if api_key_header else "X-API-Key"
        value = api_key_value
        auth_headers = {
            "Content-Type": "application/json",
            key: value
        }
        click.echo(f"Using API key authentication for MSA server (header: {key})")
    elif msa_server_username and msa_server_password:
        click.echo("Using basic authentication for MSA server")
    else:
        click.echo("No authentication provided for MSA server")
    
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


def process_input(  # noqa: C901, PLR0912, PLR0915, D103
    path: Path,
    ccd: dict,
    msa_dir: Path,
    mol_dir: Path,
    boltz2: bool,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str],
    msa_server_password: Optional[str],
    api_key_header: Optional[str],
    api_key_value: Optional[str],
    max_msa_seqs: int,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
) -> None:
    try:
        # Parse data
        if path.suffix.lower() in (".fa", ".fas", ".fasta"):
            target = parse_fasta(path, ccd, mol_dir, boltz2)
        elif path.suffix.lower() in (".yml", ".yaml"):
            target = parse_yaml(path, ccd, mol_dir, boltz2)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)  # noqa: TRY301
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)  # noqa: TRY301

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)  # noqa: TRY301

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            click.echo(msg)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
                msa_server_username=msa_server_username,
                msa_server_password=msa_server_password,
                api_key_header=api_key_header,
                api_key_value=api_key_value,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)  # noqa: TRY301

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists():
                # Parse A3M
                if msa_path.suffix == ".a3m":
                    msa: MSA = parse_a3m(
                        msa_path,
                        taxonomy=None,
                        max_seqs=max_msa_seqs,
                    )
                elif msa_path.suffix == ".csv":
                    msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                else:
                    msg = f"MSA file {msa_path} not supported, only a3m or csv."
                    raise RuntimeError(msg)  # noqa: TRY301

                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Dump templates
        for template_id, template in target.templates.items():
            name = f"{target.record.id}_{template_id}.npz"
            template_path = processed_templates_dir / name
            template.dump(template_path)

        # Dump constraints
        constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
        target.residue_constraints.dump(constraints_path)

        # Dump extra molecules
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

        # Dump record
        record_path = records_dir / f"{target.record.id}.json"
        target.record.dump(record_path)

    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"Failed to process {path}. Skipping. Error: {e}.")  # noqa: T201


@rank_zero_only
def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 8192,
    use_msa_server: bool = False,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    boltz2: bool = False,
    preprocessing_threads: int = 1,
    timing: Optional[dict[str, float]] = None,
) -> Manifest:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 8192.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.
    msa_server_username : str, optional
        Username for basic authentication with MSA server, by default None.
    msa_server_password : str, optional
        Password for basic authentication with MSA server, by default None.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).
    boltz2: bool, optional
        Whether to use Boltz2, by default False.
    preprocessing_threads: int, optional
        The number of threads to use for preprocessing, by default 1.

    Returns
    -------
    Manifest
        The manifest of the processed input data.

    """
    # Validate mutually exclusive authentication methods
    has_basic_auth = msa_server_username and msa_server_password
    has_api_key = api_key_value is not None
    
    if has_basic_auth and has_api_key:
        raise ValueError(
            "Cannot use both basic authentication (--msa_server_username/--msa_server_password) "
            "and API key authentication (--api_key_header/--api_key_value). Please use only one authentication method."
        )

    # Check if records exist at output path
    records_dir = out_dir / "processed" / "records"
    if records_dir.exists():
        # Load existing records
        existing = [Record.load(p) for p in records_dir.glob("*.json")]
        processed_ids = {record.id for record in existing}

        # Filter to missing only
        data = [d for d in data if d.stem not in processed_ids]

        # Nothing to do, update the manifest and return
        if data:
            click.echo(
                f"Found {len(existing)} existing processed inputs, skipping them."
            )
        else:
            click.echo("All inputs are already processed.")
            updated_manifest = Manifest(existing)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")

    # Create output directories
    msa_dir = out_dir / "msa"
    records_dir = out_dir / "processed" / "records"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    processed_templates_dir = out_dir / "processed" / "templates"
    processed_mols_dir = out_dir / "processed" / "mols"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    processed_constraints_dir.mkdir(parents=True, exist_ok=True)
    processed_templates_dir.mkdir(parents=True, exist_ok=True)
    processed_mols_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    if boltz2:
        ccd = load_canonicals(mol_dir)
    else:
        with ccd_path.open("rb") as file:
            ccd = pickle.load(file)  # noqa: S301

    # Create partial function
    process_input_partial = partial(
        process_input,
        ccd=ccd,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        boltz2=boltz2,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        max_msa_seqs=max_msa_seqs,
        processed_msa_dir=processed_msa_dir,
        processed_constraints_dir=processed_constraints_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        structure_dir=structure_dir,
        records_dir=records_dir,
    )

    # Parse input data. Timing starts after fixed resources (for example the CCD
    # dictionary) are loaded, so it measures only work that grows with the input
    # library.
    _per_input_t0 = time.perf_counter()
    preprocessing_threads = min(preprocessing_threads, len(data))
    click.echo(f"Processing {len(data)} inputs with {preprocessing_threads} threads.")

    if preprocessing_threads > 1 and len(data) > 1:
        with Pool(preprocessing_threads) as pool:
            list(tqdm(pool.imap(process_input_partial, data), total=len(data)))
    else:
        for path in tqdm(data):
            process_input_partial(path)

    # Load all records and write manifest
    records = [Record.load(p) for p in records_dir.glob("*.json")]
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")
    if timing is not None:
        timing["record_processing_seconds"] = time.perf_counter() - _per_input_t0


@click.group()
def cli() -> None:
    """Boltz."""
    return


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,
)
@click.argument("screen_args", nargs=-1, type=click.UNPROCESSED)
def screen(screen_args: tuple[str, ...]) -> None:
    """Run the BoltzVS structure-first cascade screen."""
    from boltz.screen.cascade import main as screen_main

    screen_main(screen_args, prog="boltz screen")


@cli.command(name="install-kernels")
@click.option(
    "--src",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Install from an already-cloned kit tree instead of cloning one.",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    default=None,
    help="Where to clone the kit. Default is <boltz cache>/kernels.",
)
@click.option("--ref", type=str, default=None, help="Git ref to check out.")
@click.option("--check", is_flag=True, help="Report status only; install nothing.")
def install_kernels(
    src: Optional[str], cache: Optional[str], ref: Optional[str], check: bool
) -> None:
    """Install the optional fused trunk kernels (--trunk_kernels/--affinity_kernels).

    BoltzVS never requires these. They come from Anthropic's
    `uplifting-biomolecular-modeling` kit, a separate Apache-2.0 project that is
    deliberately NOT vendored here: it is ~300 MB, carries its own third-party
    notices, and this repository is MIT. This command only clones it and installs
    the two packages the kernels live in.
    """
    import subprocess
    import sys

    from boltz.accel import trunk_kernels

    REPO = "https://github.com/anthropics/uplifting-biomolecular-modeling"
    reason = trunk_kernels.available()
    if reason is None:
        click.echo("The kernels are already installed and usable here.")
        click.echo(trunk_kernels.report())
        return
    if "is not installed" not in reason:
        # Present but unusable on this box (no CUDA, compute capability too low):
        # cloning would not change that.
        click.echo(f"The kernels cannot be used here: {reason}")
        return
    if check:
        click.echo(f"Not installed: {reason}")
        click.echo(f"Run `boltz install-kernels` to fetch them from {REPO}.")
        return

    if src is None:
        root = Path(cache) if cache else Path(get_cache_path()) / "kernels"
        root = root.expanduser()
        tree = root / "uplifting-biomolecular-modeling"
        if not (tree / "common" / "opt_core").is_dir():
            root.mkdir(parents=True, exist_ok=True)
            click.echo(f"Cloning {REPO}\n  into {tree}")
            # The kit carries 36 model packages; a blobless sparse clone fetches
            # only the two directories the trunk kernels need.
            cmds = [
                ["git", "clone", "--filter=blob:none", "--sparse", REPO, str(tree)],
                ["git", "-C", str(tree), "sparse-checkout", "set", "common", "boltz2"],
            ]
            if ref:
                cmds.append(["git", "-C", str(tree), "checkout", ref])
            for cmd in cmds:
                if subprocess.call(cmd) != 0:
                    msg = f"failed: {' '.join(cmd)}"
                    raise click.ClickException(msg)
        else:
            click.echo(f"Using the clone already at {tree}")
    else:
        tree = Path(src).expanduser().resolve()

    pkgs = [tree / "common" / "opt_core", tree / "boltz2" / "opt"]
    for pkg in pkgs:
        if not (pkg / "pyproject.toml").is_file():
            msg = f"not a kit package: {pkg}"
            raise click.ClickException(msg)

    for pkg, label in zip(pkgs, ("opt_core", "boltz2_opt")):
        click.echo(f"\nInstalling {label} from {pkg}")
        # --no-deps is not optional: the kit pins its own tested stack
        # (torch 2.12.0+cu130 and friends). Resolving those here would replace
        # the torch this environment already runs on.
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(pkg)]
        if subprocess.call(cmd) != 0:
            msg = f"pip install failed for {pkg}"
            raise click.ClickException(msg)

    # Verify in a SUBPROCESS, not here: an editable install is wired up by a .pth
    # file that only runs at interpreter start-up, so this already-running process
    # cannot see the new packages however many caches we invalidate. A subprocess
    # tests what the next `boltz predict` will actually see.
    probe = (
        "import importlib.util as u, sys;"
        "missing=[p for p in ('opt_core','boltz2_opt') if u.find_spec(p) is None];"
        "sys.stdout.write(','.join(missing))"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    still_missing = out.stdout.strip()
    if still_missing:
        msg = f"pip reported success but these are still not importable: {still_missing}"
        raise click.ClickException(msg)

    reason = trunk_kernels.available()
    if reason is not None and "is not installed" in reason:
        # Installed fine (the subprocess found them); this process just predates
        # the .pth. Re-probe the part that does not depend on imports.
        reason = None if __import__("torch").cuda.is_available() else "no CUDA device"
    if reason is not None:
        # The packages are in; this box just cannot run them (a login node with no
        # GPU, or a card below compute capability 8.0). That is not an install
        # failure -- say so and exit 0 so a provisioning script does not abort.
        click.echo(f"\nInstalled. They cannot run on this machine ({reason}),")
        click.echo("which is fine if you install here and compute elsewhere.")
    else:
        click.echo("\nDone. Add --trunk_kernels --affinity_kernels to enable them.")
    click.echo(
        "They are third-party Apache-2.0 code; see the kit's LICENSE and NOTICE."
    )


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help=(
        "The directory where to download the data and model. "
        "Default is ~/.boltz, or $BOLTZ_CACHE if set."
    ),
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--max_parallel_samples",
    type=int,
    help="The maximum number of samples to predict in parallel. Default is None.",
    default=5,
)
@click.option(
    "--step_scale",
    type=float,
    help=(
        "The step size is related to the temperature at "
        "which the diffusion process samples the distribution. "
        "The lower the higher the diversity among samples "
        "(recommended between 1 and 2). "
        "Default is 1.638 for Boltz-1 and 1.5 for Boltz-2. "
        "If not provided, the default step size will be used."
    ),
    default=None,
)
@click.option(
    "--gamma_0",
    type=float,
    default=None,
    help=(
        "Diffusion sampler noise coefficient gamma_0. Defaults to the model's built-in value (0.8 for Boltz-2, SDE). "
        "Set to 0 with --step_scale 1.0 for deterministic ODE sampling (recommended for speed: 2 ODE steps match 200 SDE steps)."
    ),
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(  #new
    "--inference_batch_size",  #new
    type=int,  #new
    help="DataLoader batch size for inference. Default is 1. Set >1 to batch multiple targets together (recommended for small systems). Applies to the structure pass only; the affinity pass always runs at batch size 1.",  #new
    default=1,  #new
)  #new
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help=(
        "Pairing strategy to use. Used only if --use_msa_server is set. "
        "Options are 'greedy' and 'complete'"
    ),
    default="greedy",
)
@click.option(
    "--msa_server_username",
    type=str,
    help="MSA server username for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_USERNAME environment variable.",
    default=None,
)
@click.option(
    "--msa_server_password",
    type=str,
    help="MSA server password for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_PASSWORD environment variable.",
    default=None,
)
@click.option(
    "--api_key_header",
    type=str,
    help="Custom header key for API key authentication (default: X-API-Key).",
    default=None,
)
@click.option(
    "--api_key_value",
    type=str,
    help="Custom header value for API key authentication.",
    default=None,
)
@click.option(
    "--use_potentials",
    is_flag=True,
    help="Whether to use potentials for steering. Default is False.",
)
@click.option(
    "--model",
    default="boltz2",
    type=click.Choice(["boltz1", "boltz2"]),
    help="The model to use for prediction. Default is boltz2.",
)
@click.option(
    "--method",
    type=str,
    help="The method to use for prediction. Default is None.",
    default=None,
)
@click.option(
    "--preprocessing-threads",
    type=int,
    help="The number of threads to use for preprocessing. Default is 1.",
    default=multiprocessing.cpu_count(),
)
@click.option(
    "--affinity_mw_correction",
    is_flag=True,
    type=bool,
    help="Whether to add the Molecular Weight correction to the affinity value head.",
)
@click.option(
    "--sampling_steps_affinity",
    type=int,
    help="The number of sampling steps to use for affinity prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples_affinity",
    type=int,
    help="The number of diffusion samples to use for affinity prediction. Default is 5.",
    default=5,
)
@click.option(
    "--recycling_steps_affinity",
    type=int,
    help="The number of recycling steps to use for the affinity pass. Default is 5, "
    "which reproduces the original hard-coded behaviour. Lower values accelerate the "
    "affinity pass, whose trunk accounts for 17-30% of its runtime.",
    default=5,
)
@click.option(
    "--affinity_checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--max_msa_seqs",
    type=int,
    help="The maximum number of MSA sequences to use for prediction. Default is 8192.",
    default=8192,
)
@click.option(
    "--subsample_msa",
    is_flag=True,
    help="Whether to subsample the MSA. Default is True.",
)
@click.option(
    "--num_subsampled_msa",
    type=int,
    help="The number of MSA sequences to subsample. Default is 1024.",
    default=1024,
)
@click.option(
    "--affinity_kernels",
    is_flag=True,
    help="Give the affinity model the cuEquivariance triangle kernels. Upstream passes "
         "use_kernels to the structure model only, so the affinity trunk runs without them; "
         "this turns them on. Off by default because it shifts affinity scores.",
)
@click.option(
    "--trunk_kernels",
    is_flag=True,
    help="Apply third-party fused trunk kernels (Anthropic's uplifting-biomolecular-modeling "
         "kit) in this process. Requires opt_core + boltz2_opt to be installed; a no-op with a "
         "message when they are not. Composes with the BoltzVS flags and, unlike the kit's own "
         "runner, also reaches the affinity pass.",
)
@click.option(
    "--kernel_row",
    type=click.Choice(["fast", "fast-lowmem", "big"]),
    default=None,
    help="Which row of the kit --trunk_kernels applies; requires --trunk_kernels. "
         "'fast' is used when the flag is omitted. "
         "'fast-lowmem' is 'fast' without the fused pair-weighted averaging (fpf_pwa), "
         "which measured 4.0 GB of peak on cc 8.6 -- use it when a target does not fit, it "
         "is what ran the three largest DEKOIS targets on a 24 GB card. 'big' is the kit's "
         "own memory mode; on cc 8.6 it did not fit where 'fast' did not, so measure before "
         "trusting it on a card other than the one it was tuned for.",
)
@click.option(
    "--no_kernels",
    is_flag=True,
    help="Whether to disable the kernels. Default False",
)
@click.option(
    "--write_embeddings",
    is_flag=True,
    help=" to dump the s and z embeddings into a npz file. Default is False.",
)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help="Print timing for the model-prediction stage (trainer.predict) and the forward breakdown (embed/trunk/diffusion/confidence). Off by default.",
)
@click.option(
    "--library_timing_file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Write library-scaling wall-clock components as JSON. Model download, "
        "model loading, and MSA generation are excluded; input processing, "
        "trainer.predict (including featurization and output writing), and the "
        "affinity pass are timed."
    ),
)
@click.option(
    "--matmul_precision",
    type=click.Choice(["highest", "high", "medium"]),
    default="highest",
    help="float32 matmul precision: highest (true FP32, = original Boltz-2 default) / high (TF32) / medium (bf16, fastest). "
    "Default highest preserves original behaviour; use --matmul_precision medium for accelerated VS deployment (verified lossless on DEKOIS).",
)
@click.option(
    "--chunk_size_threshold",
    type=int,
    default=const.chunk_size_threshold,
    help="Chunk the triangle ops (trunk/pairformer) when the token count exceeds this threshold, to save memory. Default in const.py. Lower = less memory (slightly slower).",
)
@click.option(
    "--drop_pairformer_layers",
    type=str,
    default=None,
    help="Comma-separated Pairformer layer indices (0-63) to drop from the trunk at inference, e.g. '2,3,5,6'. For zero-shot depth-truncation speed-up (checkpoint unchanged).",
)
@click.option(
    "--student_pairformer",
    type=str,
    default=None,
    help="Distilled student weights (a .pt produced by distill_train.py; a pairformer state_dict). "
    "Must be used with --drop_pairformer_layers using the same D17 layer set: the trunk is first reduced to 47 layers, then these weights are loaded.",
)
def predict(  # noqa: C901, PLR0915, PLR0912
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    affinity_checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 5,
    recycling_steps_affinity: int = 5,
    max_parallel_samples: Optional[int] = None,
    step_scale: Optional[float] = None,
    gamma_0: Optional[float] = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    inference_batch_size: int = 1,  #new
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    use_potentials: bool = False,
    model: Literal["boltz1", "boltz2"] = "boltz2",
    method: Optional[str] = None,
    affinity_mw_correction: Optional[bool] = False,
    preprocessing_threads: int = 1,
    max_msa_seqs: int = 8192,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    no_kernels: bool = False,
    affinity_kernels: bool = False,
    trunk_kernels: bool = False,
    kernel_row: Optional[str] = None,
    write_embeddings: bool = False,
    profile: bool = False,
    library_timing_file: Optional[Path] = None,
    matmul_precision: str = "highest",
    chunk_size_threshold: int = const.chunk_size_threshold,
    drop_pairformer_layers: Optional[str] = None,
    student_pairformer: Optional[str] = None,
) -> None:
    """Run predictions with Boltz."""
    library_timing: dict[str, object] = {
        "schema_version": 1,
        "definition": "Library-scaling wall-clock time after one-time initialization",
        "runtime": {
            "accelerator": accelerator,
            "devices": devices,
            "inference_batch_size": inference_batch_size,
            # the affinity pass is always single-record, whatever the structure pass uses
            "affinity_batch_size": 1,
            "no_kernels": no_kernels,
        },
        "structure_settings": {
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "recycling_steps": recycling_steps,
            "matmul_precision": matmul_precision,
        },
        "affinity_settings": {
            "sampling_steps": sampling_steps_affinity,
            "diffusion_samples": diffusion_samples_affinity,
            "recycling_steps": recycling_steps_affinity,
        },
        "excluded": [
            "model download",
            "model loading",
            "MSA generation",
            "scheduler queue time",
        ],
    }

    def write_library_timing() -> None:
        if library_timing_file is None:
            return
        library_timing["total_library_scaling_seconds"] = sum(
            float(library_timing.get(key, 0.0))
            for key in (
                "input_processing_seconds",
                "structure_predict_seconds",
                "affinity_predict_seconds",
            )
        )
        library_timing_file.parent.mkdir(parents=True, exist_ok=True)
        library_timing_file.write_text(
            json.dumps(library_timing, indent=2, sort_keys=True) + "\n"
        )

    def synchronize_for_timing() -> None:
        if accelerator == "gpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timing and profiling are single-device measurements. With --devices > 1 the
    # script runs once per rank, so --library_timing_file would be written by every
    # rank at the same time (racing on one path, and non-zero ranks skip the
    # preprocessing that only rank 0 does, so their input numbers are near zero),
    # and --profile synchronizes inside every forward. Refuse up front instead of
    # producing numbers that cannot be trusted. Checked against the requested
    # count, before Lightning may reduce it to the number of records.
    n_devices = len(devices) if isinstance(devices, list) else int(devices)
    if n_devices > 1:
        refused = [
            flag
            for flag, enabled in (
                ("--profile", profile),
                ("--library_timing_file", library_timing_file is not None),
            )
            if enabled
        ]
        if refused:
            msg = (
                f"{' and '.join(refused)} cannot be used with --devices "
                f"{n_devices}: timing and profiling are single-device "
                "measurements. Run with --devices 1."
            )
            raise click.ClickException(msg)

    # --no_kernels turns off the cuEquivariance path only (use_kernels=False). The
    # kit's fused levers are a replacement for that path, and most of its adapters
    # do not consult the flag -- they would keep serving while the model believes
    # the kernels are off. The mix has no defined semantics (and no memory saving),
    # so refuse it rather than run it.
    if no_kernels and (trunk_kernels or affinity_kernels):
        kit_flags = [
            flag
            for flag, enabled in (
                ("--trunk_kernels", trunk_kernels),
                ("--affinity_kernels", affinity_kernels),
            )
            if enabled
        ]
        msg = (
            f"--no_kernels cannot be combined with {' or '.join(kit_flags)}: "
            "--no_kernels disables the cuEquivariance path only, while the kit's "
            "levers keep serving. Use one or the other."
        )
        raise click.ClickException(msg)

    # D17 is a Boltz-2 structure-pass variant: the student was distilled from the
    # 64-layer Boltz-2 trunk, and the layer set assumes that depth. Boltz-1 has a
    # 48-layer trunk and no matching student.
    if model == "boltz1" and (drop_pairformer_layers or student_pairformer):
        msg = (
            "--drop_pairformer_layers/--student_pairformer are Boltz-2 only; "
            "run with --model boltz2 (the default)."
        )
        raise click.ClickException(msg)
    if student_pairformer and not drop_pairformer_layers:
        msg = (
            "--student_pairformer requires --drop_pairformer_layers with the D17 "
            "layer set (2,3,4,5,6,7,16,18,19,26,27,28,29,30,31,32,33)."
        )
        raise click.ClickException(msg)
    if student_pairformer and not Path(student_pairformer).is_file():
        msg = f"--student_pairformer not found: {student_pairformer}"
        raise click.ClickException(msg)
    if drop_pairformer_layers:
        try:
            [int(part) for part in drop_pairformer_layers.split(",") if part.strip()]
        except ValueError:
            msg = (
                "--drop_pairformer_layers must be a comma-separated list of layer "
                "indices, e.g. '2,3,5'."
            )
            raise click.ClickException(msg) from None

    # The kit row only exists for the kit; silently ignoring it hides a typo.
    if kernel_row is not None and not trunk_kernels:
        raise click.ClickException("--kernel_row requires --trunk_kernels.")
    if trunk_kernels and kernel_row is None:
        kernel_row = "fast"

    # Numeric ranges: 0 or negative values either crash deep in the sampler
    # (max_parallel_samples=0 divides by zero) or silently produce noise
    # (sampling_steps=0 skips every denoising step).
    limits = (
        ("--sampling_steps", sampling_steps, 1),
        ("--recycling_steps", recycling_steps, 0),
        ("--diffusion_samples", diffusion_samples, 1),
        ("--sampling_steps_affinity", sampling_steps_affinity, 1),
        ("--recycling_steps_affinity", recycling_steps_affinity, 0),
        ("--diffusion_samples_affinity", diffusion_samples_affinity, 1),
        ("--inference_batch_size", inference_batch_size, 1),
        ("--chunk_size_threshold", chunk_size_threshold, 1),
        ("--num_workers", num_workers, 0),
        ("--max_msa_seqs", max_msa_seqs, 1),
        ("--num_subsampled_msa", num_subsampled_msa, 1),
    )
    for flag, value, minimum in limits:
        if value < minimum:
            msg = f"{flag} must be >= {minimum} (got {value})."
            raise click.ClickException(msg)
    if max_parallel_samples is not None and max_parallel_samples < 1:
        raise click.ClickException("--max_parallel_samples must be >= 1.")

    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Supress some lightning warnings
    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision(matmul_precision)
    const.chunk_size_threshold = chunk_size_threshold   # runtime override (trunk/pairformer both read this const attribute)

    # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        # Disable kernel tuning by default,
        # but do not modify envvar if already set by caller
        os.environ[key] = os.environ.get(key, "1")

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Get MSA server credentials from environment variables if not provided
    if use_msa_server:
        if msa_server_username is None:
            msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
        if msa_server_password is None:
            msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
        if api_key_value is None:
            api_key_value = os.environ.get("MSA_API_KEY_VALUE")

        # Half an auth pair is not auth: process_inputs would silently fall back to
        # an anonymous request.
        if bool(msa_server_username) != bool(msa_server_password):
            msg = (
                "--msa_server_username and --msa_server_password must be provided "
                "together."
            )
            raise click.ClickException(msg)
        if api_key_header and not api_key_value:
            raise click.ClickException("--api_key_header requires --api_key_value.")

        click.echo(f"MSA server enabled: {msa_server_url}")
        if api_key_value:
            click.echo("MSA server authentication: using API key header")
        elif msa_server_username and msa_server_password:
            click.echo("MSA server authentication: using basic auth")
        else:
            click.echo("MSA server authentication: no credentials provided")

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (
        library_timing_file is not None
        and (out_dir / "processed").exists()
    ):
        raise click.ClickException(
            "--library_timing_file requires a fresh output directory; cached "
            "preprocessing or predictions would make the measurement incomplete."
        )

    # Download necessary data and model
    if model == "boltz1":
        download_boltz1(cache)
    elif model == "boltz2":
        download_boltz2(cache)
    else:
        msg = f"Model {model} not supported. Supported: boltz1, boltz2."
        raise ValueError(f"Model {model} not supported.")

    # Validate inputs. Directory discovery scales with the number of input files.
    _input_discovery_t0 = time.perf_counter()
    data = check_inputs(data)
    _input_discovery_seconds = time.perf_counter() - _input_discovery_t0

    # Check method
    if method is not None:
        if model == "boltz1":
            msg = "Method conditioning is not supported for Boltz-1."
            raise ValueError(msg)
        if method.lower() not in const.method_types_ids:
            method_names = list(const.method_types_ids.keys())
            msg = f"Method {method} not supported. Supported: {method_names}"
            raise ValueError(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    _process_timing: dict[str, float] = {}
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        boltz2=model == "boltz2",
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
        timing=_process_timing,
    )

    # Load manifest
    _manifest_t0 = time.perf_counter()
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Filter out existing predictions
    filtered_manifest = filter_inputs_structure(
        manifest=manifest,
        outdir=out_dir,
        override=override,
    )
    _manifest_seconds = time.perf_counter() - _manifest_t0
    _input_components = {
        "input_discovery": _input_discovery_seconds,
        "record_processing": _process_timing.get("record_processing_seconds", 0.0),
        "manifest_and_structure_filtering": _manifest_seconds,
    }
    library_timing["input_processing_components_seconds"] = _input_components
    library_timing["input_processing_seconds"] = sum(_input_components.values())

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        start_method = "fork" if platform.system() != "win32" and platform.system() != "Windows" else "spawn"
        strategy = DDPStrategy(start_method=start_method)
        if len(filtered_manifest.records) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions, taking the minimum."
            )
            click.echo(msg)
            if isinstance(devices, list):
                devices = devices[: max(1, len(filtered_manifest.records))]
            else:
                devices = max(1, min(len(filtered_manifest.records), devices))

    # Set up model parameters
    if model == "boltz2":
        diffusion_params = Boltz2DiffusionParams()
        step_scale = 1.5 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgsV2()
    else:
        diffusion_params = BoltzDiffusionParams()
        step_scale = 1.638 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgs()

    # gamma_0: override only when explicitly provided (otherwise each model's built-in value); gamma_0=0 + step_scale=1.0 -> pure ODE
    if gamma_0 is not None:
        diffusion_params.gamma_0 = gamma_0

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
    )

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
        boltz2=model == "boltz2",
        write_embeddings=write_embeddings,
    )

    # Set up trainer. The callback starts after Lightning has initialized and
    # transferred the model, keeping fixed model/device setup out of the
    # library-scaling measurement.
    prediction_timer = (
        LibraryPredictionTimer() if library_timing_file is not None else None
    )
    prediction_callbacks = [pred_writer]
    if prediction_timer is not None:
        prediction_callbacks.append(prediction_timer)
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=prediction_callbacks,
        accelerator=accelerator,
        devices=devices,
        precision=32 if model == "boltz1" else "bf16-mixed",
    )

    # The fused trunk kernels patch classes, so they must be in place before the
    # checkpoint loader builds the modules: the atom adapter installs itself from
    # AtomDiffusion.__init__. This sits OUTSIDE the structure block on purpose: on a
    # resumed run whose structure predictions are all cached that block is skipped,
    # yet the affinity pass still runs -- and it loads a fresh model from the same
    # classes. Applying here covers both passes.
    if trunk_kernels:
        from boltz.accel import trunk_kernels as _trunk_kernels
        _trunk_kernels.enable(row=kernel_row)

    if filtered_manifest.records:
        msg = f"Running structure prediction for {len(filtered_manifest.records)} input"
        msg += "s." if len(filtered_manifest.records) > 1 else "."
        click.echo(msg)

        # Create data module
        if model == "boltz2":
            data_module = Boltz2InferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                mol_dir=mol_dir,
                num_workers=num_workers,
                batch_size=inference_batch_size,  #new
                constraints_dir=processed.constraints_dir,
                template_dir=processed.template_dir,
                extra_mols_dir=processed.extra_mols_dir,
                override_method=method,
            )
        else:
            data_module = BoltzInferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
            )

        # Load model
        if checkpoint is None:
            if model == "boltz2":
                checkpoint = cache / "boltz2_conf.ckpt"
            else:
                checkpoint = cache / "boltz1_conf.ckpt"

        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
        }

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = use_potentials
        steering_args.physical_guidance_update = use_potentials

        model_cls = Boltz2 if model == "boltz2" else Boltz1
        model_module = model_cls.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not no_kernels,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        model_module.eval()

        # zero-shot depth truncation: drop the given layers from the Pairformer ModuleList at inference (checkpoint / num_blocks unchanged)
        if drop_pairformer_layers:
            drop = {int(x) for x in drop_pairformer_layers.split(",") if x.strip() != ""}
            pf = model_module.pairformer_module
            pf = getattr(pf, "_orig_mod", pf)   # handle torch.compile; inference uses _orig_mod (see boltz2.py:491)
            n_before = len(pf.layers)
            pf.layers = torch.nn.ModuleList(
                [layer for i, layer in enumerate(pf.layers) if i not in drop]
            )
            print(
                f"[trunc] Pairformer dropped layers {sorted(drop)}: {n_before} -> {len(pf.layers)} layers",
                flush=True,
            )

        # load distilled student weights (requires --drop_pairformer_layers first to reduce the trunk to the student's layer count)
        if student_pairformer:
            pf = model_module.pairformer_module
            pf = getattr(pf, "_orig_mod", pf)   # same unwrap as the layer drop (see boltz2.py:491)
            sd = torch.load(student_pairformer, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:  # handle a wrapped checkpoint
                sd = sd["state_dict"]
            try:
                pf.load_state_dict(sd, strict=True)  # strict: mismatched layer count/keys raise immediately, preventing silent mis-loading
            except RuntimeError as e:
                raise RuntimeError(
                    f"[student] failed to load {student_pairformer}: current pairformer has {len(pf.layers)} layers, which does not match the student weights. "
                    f"Ensure --drop_pairformer_layers uses the same D17 layer set as training "
                    f"(2,3,4,5,6,7,16,18,19,26,27,28,29,30,31,32,33).\nOriginal error: {e}"
                ) from e
            print(f"[student] loaded distilled weights {student_pairformer} (pairformer {len(pf.layers)} layers)", flush=True)

        model_module.profile_timing = profile   # passed to Boltz2.forward to control per-stage timing output

        # Compute structure predictions
        # profiling (when --profile is set): time only the model-prediction stage wall-clock (data prep / trainer setup / weight loading excluded)
        if profile:
            synchronize_for_timing()
            _pred_t0 = time.perf_counter()
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )
        if profile:
            synchronize_for_timing()
            _profile_structure_seconds = time.perf_counter() - _pred_t0
            print(
                f"##model prediction (trainer.predict): "
                f"{_profile_structure_seconds:.4f}s",
                flush=True,
            )
        if prediction_timer is not None:
            if prediction_timer.elapsed_seconds is None:
                raise RuntimeError("Structure prediction timing callback did not run")
            _structure_predict_seconds = prediction_timer.elapsed_seconds
        else:
            _structure_predict_seconds = 0.0
        library_timing["structure_input_count"] = len(filtered_manifest.records)
        library_timing["structure_predict_seconds"] = _structure_predict_seconds
    else:
        library_timing["structure_input_count"] = 0
        library_timing["structure_predict_seconds"] = 0.0

    # Check if affinity predictions are needed
    if any(r.affinity for r in manifest.records):
        # Print header
        click.echo("\nPredicting property: affinity\n")

        # Validate inputs
        _affinity_filter_t0 = time.perf_counter()
        manifest_filtered = filter_inputs_affinity(
            manifest=manifest,
            outdir=out_dir,
            override=override,
        )
        _affinity_filter_seconds = time.perf_counter() - _affinity_filter_t0
        input_components = library_timing["input_processing_components_seconds"]
        assert isinstance(input_components, dict)
        input_components["affinity_filtering"] = _affinity_filter_seconds
        library_timing["input_processing_seconds"] = sum(input_components.values())
        if not manifest_filtered.records:
            click.echo("Found existing affinity predictions for all inputs, skipping.")
            library_timing["affinity_input_count"] = 0
            library_timing["affinity_predict_seconds"] = 0.0
            write_library_timing()
            return

        msg = f"Running affinity prediction for {len(manifest_filtered.records)} input"
        msg += "s." if len(manifest_filtered.records) > 1 else "."
        click.echo(msg)

        # The affinity path is written for one record at a time: Boltz2.forward reads
        # the masks of batch item 0 and keeps only the best-scoring diffusion sample,
        # and BoltzAffinityWriter writes one JSON via .item(). Running it batched would
        # either raise or silently score the first compound for the whole batch, so the
        # structure-pass batch size is NOT forwarded here.
        if inference_batch_size > 1:
            click.echo(
                "Note: the affinity pass runs at batch size 1; "
                f"--inference_batch_size {inference_batch_size} applies to the "
                "structure pass only."
            )

        pred_writer = BoltzAffinityWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
        )

        data_module = Boltz2InferenceDataModule(
            manifest=manifest_filtered,
            target_dir=out_dir / "predictions",
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            batch_size=1,  # affinity is single-record by construction (see the note above)
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method="other",
            affinity=True,
        )

        predict_affinity_args = {
            "recycling_steps": recycling_steps_affinity,
            "sampling_steps": sampling_steps_affinity,
            "diffusion_samples": diffusion_samples_affinity,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        # Load affinity model
        if affinity_checkpoint is None:
            affinity_checkpoint = cache / "boltz2_aff.ckpt"

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        steering_args.contact_guidance_update = False
        
        model_module = Boltz2.load_from_checkpoint(
            affinity_checkpoint,
            strict=True,
            predict_args=predict_affinity_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
            affinity_mw_correction=affinity_mw_correction,
            # Upstream never passes use_kernels here, so the affinity model takes the class
            # default (False) and its trunk runs without the cuEquivariance triangle kernels
            # the structure pass is given (main.py, the structure load_from_checkpoint).
            # Opt-in rather than on by default: it moves the affinity scores by about as much
            # as any other kernel-path change, so it needs the 79-target check before it can
            # be the default.
            **({"use_kernels": True} if affinity_kernels and not no_kernels else {}),
        )
        model_module.eval()
        model_module.profile_timing = profile   # added: pass 2 builds a fresh module, so --profile must be re-applied here

        trainer.callbacks[0] = pred_writer
        if profile:
            synchronize_for_timing()
            _affinity_t0 = time.perf_counter()
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )
        if profile:
            synchronize_for_timing()
            _profile_affinity_seconds = time.perf_counter() - _affinity_t0
            print(
                f"##affinity prediction (trainer.predict): "
                f"{_profile_affinity_seconds:.4f}s",
                flush=True,
            )
        if prediction_timer is not None:
            if prediction_timer.elapsed_seconds is None:
                raise RuntimeError("Affinity prediction timing callback did not run")
            _affinity_predict_seconds = prediction_timer.elapsed_seconds
        else:
            _affinity_predict_seconds = 0.0
        library_timing["affinity_input_count"] = len(manifest_filtered.records)
        library_timing["affinity_predict_seconds"] = _affinity_predict_seconds
    else:
        library_timing["affinity_input_count"] = 0
        library_timing["affinity_predict_seconds"] = 0.0

    write_library_timing()


if __name__ == "__main__":
    cli()
