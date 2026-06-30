#!/usr/bin/env python3
"""Build a CLEAN-predictor source-data bundle.

This bundle is intentionally separate from the DeepMzyme graph-training bundle:
it contains only sequence/split CSVs and extracted metalloenzyme summary CSVs
needed by CLEAN/train_clean_predictor_baselines.ipynb. It does not include
structures, ESMC embeddings, RING files, or DeepMzyme external graph features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


SUMMARY_CSV = "final_data_summarazing_table_transition_metals_only_catalytic.csv"


@dataclass(frozen=True)
class BundleFile:
    source: str
    archive_path: str
    bytes: int
    role: str


def parse_csv_ints(value: str) -> list[int]:
    folds: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        folds.append(int(item))
    if not folds:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return folds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(
    *,
    project_root: Path,
    staging_root: Path,
    source: Path,
    archive_path: Path | None = None,
    role: str,
    files: list[BundleFile],
) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if archive_path is None:
        archive_path = source.relative_to(project_root)
    destination = staging_root / archive_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    files.append(
        BundleFile(
            source=str(source),
            archive_path=str(archive_path),
            bytes=source.stat().st_size,
            role=role,
        )
    )


def add_clean_identity_files(
    *,
    project_root: Path,
    staging_root: Path,
    identity: int,
    folds: list[int],
    files: list[BundleFile],
) -> None:
    data_root = project_root / "DeepMzyme_Data"
    full_split_root = data_root / "CLEAN_all_train_valid_splits" / f"split{identity}"
    shared_root = data_root / f"CLEAN_{identity}_shared"
    if not full_split_root.exists():
        raise FileNotFoundError(full_split_root)
    if not shared_root.exists():
        raise FileNotFoundError(shared_root)

    for fold in folds:
        add_file(
            project_root=project_root,
            staging_root=staging_root,
            source=full_split_root / f"split{identity}_train_split_{fold}.csv",
            role=f"clean{identity}_full_train_fold{fold}",
            files=files,
        )
        add_file(
            project_root=project_root,
            staging_root=staging_root,
            source=full_split_root / f"split{identity}_test_split_{fold}_curate.csv",
            role=f"clean{identity}_full_test_sequence_source_fold{fold}",
            files=files,
        )
        add_file(
            project_root=project_root,
            staging_root=staging_root,
            source=shared_root / "folds" / f"CLEAN_{identity}_train_test_split_{fold}_train.csv",
            role=f"clean{identity}_metallo_train_fold{fold}",
            files=files,
        )
        add_file(
            project_root=project_root,
            staging_root=staging_root,
            source=shared_root / "folds" / f"CLEAN_{identity}_train_test_split_{fold}_test.csv",
            role=f"clean{identity}_metallo_test_fold{fold}",
            files=files,
        )

    for optional in ["README.md", "split_metadata.json"]:
        path = shared_root / optional
        if path.exists():
            add_file(
                project_root=project_root,
                staging_root=staging_root,
                source=path,
                role=f"clean{identity}_metadata",
                files=files,
            )
    metadata = shared_root / "metadata" / "structure_sources.csv"
    if metadata.exists():
        add_file(
            project_root=project_root,
            staging_root=staging_root,
            source=metadata,
            role=f"clean{identity}_metadata",
            files=files,
        )


def add_care30_files(*, project_root: Path, staging_root: Path, files: list[BundleFile]) -> None:
    data_root = project_root / "DeepMzyme_Data"
    care_split_root = data_root / "CARE_dataset" / "CARE_datasets" / "splits" / "task1"
    care_metallo_root = data_root / "CARE_task1_30_clusterRes30_train_test_metallo"
    add_file(
        project_root=project_root,
        staging_root=staging_root,
        source=care_split_root / "protein_train.csv",
        role="care30_full_train",
        files=files,
    )
    add_file(
        project_root=project_root,
        staging_root=staging_root,
        source=care_split_root / "30_protein_test.csv",
        role="care30_full_test_sequence_source",
        files=files,
    )
    add_file(
        project_root=project_root,
        staging_root=staging_root,
        source=care_metallo_root / "train" / SUMMARY_CSV,
        role="care30_clusterRes30_metallo_train",
        files=files,
    )
    add_file(
        project_root=project_root,
        staging_root=staging_root,
        source=care_metallo_root / "test" / SUMMARY_CSV,
        role="care30_clusterRes30_metallo_test",
        files=files,
    )
    for optional in ["README.md", "split_metadata.json"]:
        path = care_metallo_root / optional
        if path.exists():
            add_file(
                project_root=project_root,
                staging_root=staging_root,
                source=path,
                role="care30_clusterRes30_metadata",
                files=files,
            )
    metadata_root = care_metallo_root / "metadata"
    for name in [
        "care_task1_30_audit.csv",
        "care_task1_30_audit.json",
        "care_task1_30_train_proteins.csv",
        "care_task1_30_test_proteins.csv",
    ]:
        path = metadata_root / name
        if path.exists():
            add_file(
                project_root=project_root,
                staging_root=staging_root,
                source=path,
                role="care30_clusterRes30_metadata",
                files=files,
            )


def write_manifest(
    *,
    staging_root: Path,
    files: list[BundleFile],
    clean_identities: list[int],
    folds: list[int],
    include_care30: bool,
) -> None:
    manifest_dir = staging_root / "DeepMzyme_Data" / "CLEAN_predictor_bundle"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle_kind": "clean_predictor_baseline_sources",
        "description": (
            "Sequence/split CSVs and extracted metalloenzyme summary CSVs for "
            "CLEAN/train_clean_predictor_baselines.ipynb. Excludes DeepMzyme "
            "structures, ESMC embeddings, RING, and graph external features."
        ),
        "clean_identities": clean_identities,
        "folds": folds,
        "include_care30": include_care30,
        "files": [asdict(item) for item in files],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (manifest_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "archive_path", "bytes", "source"])
        writer.writeheader()
        for item in files:
            writer.writerow(asdict(item))


def build_bundle(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    output_bundle = args.output_bundle.resolve()
    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    clean_identities = [int(item) for item in args.clean_identities.split(",") if item.strip()]
    if not clean_identities:
        raise ValueError("At least one CLEAN identity is required.")

    with tempfile.TemporaryDirectory(prefix="clean_predictor_bundle_") as tmp:
        staging_root = Path(tmp)
        files: list[BundleFile] = []
        for identity in clean_identities:
            add_clean_identity_files(
                project_root=project_root,
                staging_root=staging_root,
                identity=identity,
                folds=args.folds,
                files=files,
            )
        if args.include_care30:
            add_care30_files(project_root=project_root, staging_root=staging_root, files=files)
        write_manifest(
            staging_root=staging_root,
            files=files,
            clean_identities=clean_identities,
            folds=args.folds,
            include_care30=args.include_care30,
        )
        compression = args.compression
        if shutil.which("zstd") is None:
            raise RuntimeError("zstd is required to build .tar.zst bundles.")
        subprocess.run(
            [
                "tar",
                f"--use-compress-program={compression}",
                "-cf",
                str(output_bundle),
                "-C",
                str(staging_root),
                "DeepMzyme_Data",
            ],
            check=True,
        )
    sha = sha256_file(output_bundle)
    output_bundle.with_suffix(output_bundle.suffix + ".sha256").write_text(
        f"{sha}  {output_bundle.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "bundle": str(output_bundle),
        "sha256": sha,
        "size_bytes": output_bundle.stat().st_size,
    }, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--output-bundle",
        type=Path,
        default=Path("/media/Data/clean_predictor_bundles/CLEAN_predictor_baselines_v1_clean30x5_care30_sources.tar.zst"),
    )
    parser.add_argument("--clean-identities", default="30", help="Comma-separated CLEAN identity thresholds to include.")
    parser.add_argument("--folds", type=parse_csv_ints, default=parse_csv_ints("0,1,2,3,4"))
    parser.add_argument("--include-care30", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compression", default="zstd -T0 -19")
    return parser.parse_args()


if __name__ == "__main__":
    build_bundle(parse_args())
