import argparse
import hashlib
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from Bio.PDB import PDBParser, MMCIFParser
from Bio.Data.PDBData import protein_letters_3to1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import get_default_embeddings_dir
from training.structure_loading import find_structure_files
from training.esm_feature_loading import (
    build_embedding_payload,
    embedding_metadata_from_payload,
    residue_keys_for_structure_chain,
    write_embedding_metadata_sidecar,
)


DEFAULT_ESMC_MODEL_NAME = "esmc_300m"


def parse_structure(structure_file):
    structure_file = Path(structure_file)
    if structure_file.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif structure_file.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {structure_file.suffix}")
    return parser.get_structure("model", str(structure_file))


def extract_chain_sequences(structure):
    sequences = {}

    first_model = next(structure.get_models())

    for chain in first_model:
        seq = []

        for residue in chain:
            if residue.id[0] != " ":
                continue

            resname = residue.get_resname().upper()
            aa = protein_letters_3to1.get(resname, "X")
            seq.append(aa)

        chain_seq = "".join(seq)
        if chain_seq:
            chain_id = chain.id.strip() if chain.id.strip() else "_"
            sequences[chain_id] = chain_seq

    if not sequences:
        raise ValueError("No protein sequences found in the parsed structure.")

    return sequences


def clean_embedding_length(emb, sequence_length):
    if emb.dim() == 3:
        emb = emb[0]

    if emb.shape[0] == sequence_length + 2:
        emb = emb[1:-1]
    elif emb.shape[0] != sequence_length:
        raise ValueError(
            f"Unexpected embedding length: got {emb.shape[0]}, "
            f"expected {sequence_length} or {sequence_length + 2}"
        )

    return emb


def resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_esmc_model(
    device: str | None = None,
    *,
    model_name: str = DEFAULT_ESMC_MODEL_NAME,
) -> tuple[ESMC, str]:
    resolved_device = resolve_device(device)
    model = ESMC.from_pretrained(model_name).to(resolved_device)
    model.eval()
    return model, resolved_device


def expected_embedding_path(structure_file: Path, chain_id: str, out_dir: Path) -> Path:
    return out_dir / f"{structure_file.stem}_chain_{chain_id}_esmc.pt"


def create_resi_embed_pt(
    structure_file,
    out_dir: Path | None = None,
    *,
    model: ESMC | None = None,
    model_name: str = DEFAULT_ESMC_MODEL_NAME,
    device: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    structure_file = Path(structure_file)
    structure = parse_structure(structure_file)
    chain_sequences = extract_chain_sequences(structure)

    if out_dir is None:
        out_dir = get_default_embeddings_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"embeddings dir: {out_dir}")

    owns_model = model is None
    if model is None:
        model, device = load_esmc_model(device, model_name=model_name)
    else:
        device = resolve_device(device)
    print(f"device: {device}")

    saved_files: list[Path] = []
    with torch.no_grad():
        for chain_id, sequence in chain_sequences.items():
            out_file = expected_embedding_path(structure_file, chain_id, out_dir)
            if out_file.exists() and not overwrite:
                print(f"skipping existing: {out_file}")
                saved_files.append(out_file)
                continue

            print(f"\nProcessing chain {chain_id} | sequence length = {len(sequence)}")

            protein = ESMProtein(sequence=sequence)
            protein_tensor = model.encode(protein)

            output = model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )

            emb = clean_embedding_length(output.embeddings, len(sequence))
            residue_ids, _residue_ids_with_ca = residue_keys_for_structure_chain(structure, chain_id)
            payload = build_embedding_payload(
                emb,
                residue_ids,
                structure_id=structure_file.stem,
                chain_id=chain_id,
                source_path=str(structure_file),
                metadata={
                    "esm_model_name": model_name,
                    "esm_checkpoint_name": model_name,
                    "embedding_dim": int(emb.size(1)),
                    "n_residues": int(emb.size(0)),
                    "generated_at": utc_timestamp(),
                    "code_version": git_commit_hash(),
                    "source_structure_id": structure_file.stem,
                    "source_chain_id": chain_id,
                    "source_sequence_identifier": f"{structure_file.stem}:chain:{chain_id}",
                    "source_sequence_length": len(sequence),
                    "source_sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                    "source_sequence": sequence,
                },
            )
            print("embedding shape:", emb.shape)

            torch.save(payload, out_file)
            sidecar_path = write_embedding_metadata_sidecar(
                out_file,
                embedding_metadata_from_payload(payload),
            )
            print(f"saved: {out_file}")
            print(f"saved metadata: {sidecar_path}")
            saved_files.append(out_file)

    if owns_model:
        del model
    return saved_files


def create_resi_embed_batch(
    structure_files: Sequence[str | Path],
    out_dir: Path | None = None,
    *,
    device: str | None = None,
    model_name: str = DEFAULT_ESMC_MODEL_NAME,
    overwrite: bool = False,
) -> dict[str, object]:
    if out_dir is None:
        out_dir = get_default_embeddings_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, resolved_device = load_esmc_model(device, model_name=model_name)
    processed = 0
    failed: list[dict[str, str]] = []
    saved_files: list[str] = []

    for structure_file in structure_files:
        structure_path = Path(structure_file)
        print(f"\n=== {structure_path} ===")
        try:
            result_files = create_resi_embed_pt(
                structure_path,
                out_dir=out_dir,
                model=model,
                model_name=model_name,
                device=resolved_device,
                overwrite=overwrite,
            )
            processed += 1
            saved_files.extend(str(path) for path in result_files)
        except Exception as exc:
            failed.append(
                {
                    "structure_file": str(structure_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"failed: {type(exc).__name__}: {exc}")

    return {
        "processed_structures": processed,
        "failed_structures": failed,
        "saved_files": saved_files,
        "device": resolved_device,
        "esm_model_name": model_name,
        "out_dir": str(out_dir),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate canonical ESMC residue embeddings.")
    parser.add_argument("structure_files", nargs="*", help="Specific structure files to embed.")
    parser.add_argument("--structure-dir", type=Path, default=None, help="Recursively scan for structure files.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output embeddings directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on scanned structure files.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even if they already exist.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu or cuda.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_ESMC_MODEL_NAME,
        help="ESMC checkpoint/model name to load and record in embedding metadata.",
    )
    return parser


def resolve_cli_structure_files(args: argparse.Namespace) -> list[Path]:
    if args.structure_files:
        return [Path(path) for path in args.structure_files]
    if args.structure_dir is None:
        raise ValueError("Provide structure files or --structure-dir.")

    structure_files = find_structure_files(args.structure_dir)
    if args.limit is not None:
        structure_files = structure_files[: args.limit]
    return structure_files


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    structure_files = resolve_cli_structure_files(args)
    summary = create_resi_embed_batch(
        structure_files,
        out_dir=args.out_dir,
        device=args.device,
        model_name=args.model_name,
        overwrite=args.overwrite,
    )
    print("\nSummary:")
    print(f"processed_structures: {summary['processed_structures']}")
    print(f"failed_structures: {len(summary['failed_structures'])}")
    print(f"saved_files: {len(summary['saved_files'])}")
    print(f"device: {summary['device']}")
    print(f"esm_model_name: {summary['esm_model_name']}")
    print(f"out_dir: {summary['out_dir']}")
    if summary["failed_structures"]:
        print(f"failure_sample: {summary['failed_structures'][:5]}")

if __name__ == "__main__":
    main()
