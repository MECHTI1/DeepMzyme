#!/usr/bin/env python3
"""Audit and migrate DeepMzyme structure files into one content-addressed store."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from project_paths import DATA_DIR, STRUCTURE_STORE_DIR
from structure_store import (
    STRUCTURE_MANIFEST_FILENAME,
    StructureReference,
    is_structure_file,
    read_structure_manifest,
    sha256_file,
    store_structure,
    write_structure_manifest,
)


CATALOG_FILENAME = "catalog.csv"
AUDIT_REPORT_FILENAME = "audit_report.json"
PRE_MIGRATION_REPORT_FILENAME = "migration_audit_before.json"


@dataclass(frozen=True)
class LogicalStructure:
    membership_dir: Path
    structure_name: str
    source_path: Path
    sha256: str
    size_bytes: int
    source_kind: str


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _legacy_structure_paths(
    directory: Path,
    *,
    allow_broken_symlinks: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not is_structure_file(path):
            continue
        if path.is_symlink() and not path.exists():
            if allow_broken_symlinks:
                paths.append(path)
                continue
            raise FileNotFoundError(f"Broken structure symlink: {path}")
        if path.is_file():
            paths.append(path)
    return paths


def discover_membership_dirs(data_root: Path, store_root: Path) -> list[Path]:
    data_root = data_root.resolve()
    store_root = store_root.resolve()
    discovered: list[Path] = []
    for current_root, directory_names, filenames in os.walk(data_root, followlinks=False):
        current = Path(current_root)
        directory_names[:] = [
            name
            for name in directory_names
            if (current / name).resolve() != store_root
        ]
        if STRUCTURE_MANIFEST_FILENAME in filenames or any(
            is_structure_file(Path(filename)) for filename in filenames
        ):
            discovered.append(current)
    return sorted(set(discovered))


def inventory_structures(data_root: Path, store_root: Path) -> list[LogicalStructure]:
    inode_hash_cache: dict[tuple[int, int, int], str] = {}
    inventory: list[LogicalStructure] = []
    for membership_dir in discover_membership_dirs(data_root, store_root):
        manifest_path = membership_dir / STRUCTURE_MANIFEST_FILENAME
        if manifest_path.is_file():
            for reference in read_structure_manifest(membership_dir):
                inventory.append(
                    LogicalStructure(
                        membership_dir=membership_dir,
                        structure_name=reference.structure_name,
                        source_path=reference.path,
                        sha256=reference.sha256,
                        size_bytes=reference.size_bytes,
                        source_kind="manifest",
                    )
                )
            continue

        seen_names: set[str] = set()
        for source_path in _legacy_structure_paths(membership_dir):
            if source_path.name in seen_names:
                raise ValueError(f"Duplicate structure filename in {membership_dir}: {source_path.name}")
            stat_result = source_path.stat()
            inode_key = (stat_result.st_dev, stat_result.st_ino, stat_result.st_size)
            digest = inode_hash_cache.get(inode_key)
            if digest is None:
                digest = sha256_file(source_path)
                inode_hash_cache[inode_key] = digest
            inventory.append(
                LogicalStructure(
                    membership_dir=membership_dir,
                    structure_name=source_path.name,
                    source_path=source_path,
                    sha256=digest,
                    size_bytes=stat_result.st_size,
                    source_kind="symlink" if source_path.is_symlink() else "regular",
                )
            )
            seen_names.add(source_path.name)
    return inventory


def _pairwise_overlap_report(
    inventory: Sequence[LogicalStructure],
    *,
    data_root: Path,
) -> list[dict[str, object]]:
    by_directory: dict[Path, dict[str, str]] = defaultdict(dict)
    for item in inventory:
        previous = by_directory[item.membership_dir].get(item.structure_name)
        if previous is not None and previous != item.sha256:
            raise ValueError(
                f"Conflicting content for {item.structure_name} within {item.membership_dir}"
            )
        by_directory[item.membership_dir][item.structure_name] = item.sha256

    results: list[dict[str, object]] = []
    directories = sorted(by_directory)
    for index, left_dir in enumerate(directories):
        for right_dir in directories[index + 1 :]:
            left = by_directory[left_dir]
            right = by_directory[right_dir]
            common_names = set(left).intersection(right)
            same = sum(left[name] == right[name] for name in common_names)
            different = len(common_names) - same
            if common_names:
                results.append(
                    {
                        "left": _relative(left_dir, data_root),
                        "right": _relative(right_dir, data_root),
                        "same_name_same_sha256": same,
                        "same_name_different_sha256": different,
                    }
                )
    return results


def _pdb_coordinate_record_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith((b"ATOM  ", b"HETATM", b"HET   ", b"CONECT")):
                digest.update(line)
    return digest.hexdigest()


def build_audit_report(
    inventory: Sequence[LogicalStructure],
    *,
    data_root: Path,
    store_root: Path,
) -> dict[str, object]:
    by_digest: dict[str, list[LogicalStructure]] = defaultdict(list)
    by_name: dict[str, set[str]] = defaultdict(set)
    for item in inventory:
        by_digest[item.sha256].append(item)
        by_name[item.structure_name].add(item.sha256)

    inode_sizes: dict[tuple[int, int], int] = {}
    for item in inventory:
        stat_result = item.source_path.stat()
        inode_sizes[(stat_result.st_dev, stat_result.st_ino)] = stat_result.st_size

    digest_sizes = {digest: items[0].size_bytes for digest, items in by_digest.items()}
    source_kind_counts = Counter(item.source_kind for item in inventory)
    basename_conflicts = {
        name: sorted(digests)
        for name, digests in sorted(by_name.items())
        if len(digests) > 1
    }
    items_by_name_and_digest: dict[tuple[str, str], LogicalStructure] = {}
    for item in inventory:
        items_by_name_and_digest.setdefault((item.structure_name, item.sha256), item)
    coordinate_equal_conflicts: list[str] = []
    coordinate_different_conflicts: list[str] = []
    for name, digests in basename_conflicts.items():
        if Path(name).suffix.lower() != ".pdb":
            coordinate_different_conflicts.append(name)
            continue
        coordinate_hashes = {
            _pdb_coordinate_record_sha256(items_by_name_and_digest[(name, digest)].source_path)
            for digest in digests
        }
        target = coordinate_equal_conflicts if len(coordinate_hashes) == 1 else coordinate_different_conflicts
        target.append(name)
    same_digest_different_names = {
        digest: sorted({item.structure_name for item in items})
        for digest, items in sorted(by_digest.items())
        if len({item.structure_name for item in items}) > 1
    }
    membership_counts = Counter(_relative(item.membership_dir, data_root) for item in inventory)

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "structure_store": str(store_root.resolve()),
        "membership_directory_count": len(membership_counts),
        "logical_structure_references": len(inventory),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "unique_structure_names": len(by_name),
        "unique_sha256_contents": len(by_digest),
        "duplicate_content_groups": sum(len(items) > 1 for items in by_digest.values()),
        "redundant_logical_references": len(inventory) - len(by_digest),
        "unique_referenced_inodes": len(inode_sizes),
        "unique_referenced_inode_bytes": sum(inode_sizes.values()),
        "unique_content_bytes": sum(digest_sizes.values()),
        "same_name_different_content_count": len(basename_conflicts),
        "same_name_different_bytes_coordinate_records_equal_count": len(coordinate_equal_conflicts),
        "same_name_different_bytes_coordinate_records_different_count": len(coordinate_different_conflicts),
        "same_name_different_bytes_coordinate_records_equal": coordinate_equal_conflicts,
        "same_name_different_bytes_coordinate_records_different": coordinate_different_conflicts,
        "same_content_different_name_count": len(same_digest_different_names),
        "same_name_different_content": basename_conflicts,
        "same_content_different_name": same_digest_different_names,
        "membership_counts": dict(sorted(membership_counts.items())),
        "pairwise_filename_overlap": _pairwise_overlap_report(inventory, data_root=data_root),
    }


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def print_report(report: Mapping[str, object]) -> None:
    kinds = report["source_kind_counts"]
    print(f"Membership directories: {report['membership_directory_count']}")
    print(f"Logical structure references: {report['logical_structure_references']}")
    print(f"Reference kinds: {json.dumps(kinds, sort_keys=True)}")
    print(f"Unique structure names: {report['unique_structure_names']}")
    print(f"Unique byte contents: {report['unique_sha256_contents']}")
    print(f"Redundant logical references: {report['redundant_logical_references']}")
    print(f"Same filename with different bytes: {report['same_name_different_content_count']}")
    print(
        "Same-name byte variants with identical PDB coordinate records: "
        f"{report['same_name_different_bytes_coordinate_records_equal_count']}"
    )
    print(f"Same bytes with different filenames: {report['same_content_different_name_count']}")
    print(f"Unique referenced inode bytes: {report['unique_referenced_inode_bytes']}")
    print(f"Unique content bytes: {report['unique_content_bytes']}")


def write_catalog(
    store_root: Path,
    stored_by_digest: Mapping[str, StructureReference],
    inventory: Sequence[LogicalStructure],
) -> Path:
    counts = Counter(item.sha256 for item in inventory)
    membership_dirs_by_digest: dict[str, set[Path]] = defaultdict(set)
    for item in inventory:
        membership_dirs_by_digest[item.sha256].add(item.membership_dir.resolve())
    catalog_path = store_root / CATALOG_FILENAME
    temporary_path = store_root / f".{CATALOG_FILENAME}.tmp"
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "sha256",
            "structure_name",
            "relative_path",
            "size_bytes",
            "logical_reference_count",
            "membership_directory_count",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for digest, reference in sorted(stored_by_digest.items()):
            writer.writerow(
                {
                    "sha256": digest,
                    "structure_name": reference.structure_name,
                    "relative_path": reference.path.relative_to(store_root.resolve()).as_posix(),
                    "size_bytes": reference.size_bytes,
                    "logical_reference_count": counts[digest],
                    "membership_directory_count": len(membership_dirs_by_digest[digest]),
                }
            )
    os.replace(temporary_path, catalog_path)
    return catalog_path


def verify_store(data_root: Path, store_root: Path) -> dict[str, object]:
    inventory = inventory_structures(data_root, store_root)
    if not inventory:
        raise ValueError(f"No manifest-backed structures found under {data_root}")
    legacy_paths = [
        path
        for membership_dir in discover_membership_dirs(data_root, store_root)
        if (membership_dir / STRUCTURE_MANIFEST_FILENAME).is_file()
        for path in _legacy_structure_paths(membership_dir, allow_broken_symlinks=True)
    ]
    non_manifest = [item for item in inventory if item.source_kind != "manifest"]
    if non_manifest or legacy_paths:
        raise ValueError(
            f"Found {len(non_manifest) + len(legacy_paths)} legacy structure entries after migration; "
            "rerun migrate --apply."
        )

    hashed: dict[Path, str] = {}
    for item in inventory:
        actual = hashed.get(item.source_path)
        if actual is None:
            actual = sha256_file(item.source_path)
            hashed[item.source_path] = actual
        if actual != item.sha256:
            raise ValueError(f"SHA-256 verification failed for {item.source_path}")
        try:
            item.source_path.relative_to((store_root / "objects").resolve())
        except ValueError as exc:
            raise ValueError(f"Manifest target is outside the structure store: {item.source_path}") from exc

    report = build_audit_report(inventory, data_root=data_root, store_root=store_root)
    write_json(store_root / AUDIT_REPORT_FILENAME, report)
    return report


def _verify_legacy_identity(path: Path, expected: LogicalStructure) -> None:
    if not path.is_file():
        raise ValueError(f"Refusing to remove unreadable legacy structure: {path}")
    actual_digest = sha256_file(path)
    if path.stat().st_size != expected.size_bytes or actual_digest != expected.sha256:
        raise ValueError(
            f"Conflicting legacy structure {path}: expected SHA-256 {expected.sha256} "
            f"and {expected.size_bytes} bytes, found SHA-256 {actual_digest} "
            f"and {path.stat().st_size} bytes; refusing migration cleanup."
        )


def migrate(data_root: Path, store_root: Path, *, apply: bool) -> dict[str, object]:
    inventory = inventory_structures(data_root, store_root)
    if not inventory:
        raise ValueError(f"No structures or structure manifests found under {data_root}")
    pre_report = build_audit_report(inventory, data_root=data_root, store_root=store_root)
    print_report(pre_report)
    if not apply:
        print("Preview only. Re-run with --apply to create the store and replace structure entries with manifests.")
        return pre_report

    # Validate every cleanup candidate before writing any objects or manifests.
    # A manifest is authoritative for membership, not proof that a leftover
    # file with the same name contains the same bytes.
    expected_by_directory: dict[Path, dict[str, LogicalStructure]] = defaultdict(dict)
    for item in inventory:
        expected_by_directory[item.membership_dir][item.structure_name] = item
    cleanup: list[tuple[Path, LogicalStructure]] = []
    for membership_dir in discover_membership_dirs(data_root, store_root):
        expected_names = expected_by_directory.get(membership_dir, {})
        for path in _legacy_structure_paths(membership_dir, allow_broken_symlinks=True):
            if path.name not in expected_names:
                raise ValueError(f"Refusing to remove structure absent from manifest: {path}")
            expected = expected_names[path.name]
            _verify_legacy_identity(path, expected)
            cleanup.append((path, expected))

    store_root.mkdir(parents=True, exist_ok=True)
    pre_migration_report_path = store_root / PRE_MIGRATION_REPORT_FILENAME
    if not pre_migration_report_path.exists():
        write_json(pre_migration_report_path, pre_report)

    by_digest: dict[str, list[LogicalStructure]] = defaultdict(list)
    for item in inventory:
        by_digest[item.sha256].append(item)
    differently_named = {
        digest: sorted({item.structure_name for item in items})
        for digest, items in by_digest.items()
        if len({item.structure_name for item in items}) != 1
    }
    if differently_named:
        raise ValueError(
            "Cannot preserve logical filename semantics while storing these identical contents once: "
            + json.dumps(differently_named, sort_keys=True)
        )

    stored_by_digest: dict[str, StructureReference] = {}
    for digest, items in sorted(by_digest.items()):
        source_item = min(
            items,
            key=lambda item: (item.source_path.is_symlink(), str(item.source_path)),
        )
        stored_by_digest[digest] = store_structure(
            source_item.source_path,
            store_root,
            sha256=digest,
        )

    by_directory: dict[Path, list[StructureReference]] = defaultdict(list)
    for item in inventory:
        stored = stored_by_digest[item.sha256]
        by_directory[item.membership_dir].append(
            StructureReference(
                structure_name=item.structure_name,
                path=stored.path,
                sha256=item.sha256,
                size_bytes=item.size_bytes,
            )
        )
    for membership_dir, references in sorted(by_directory.items()):
        write_structure_manifest(membership_dir, references)

    # Verify each unique stored object and every newly written manifest before unlinking sources.
    verified_digests: set[str] = set()
    for membership_dir, expected_references in sorted(by_directory.items()):
        actual_references = read_structure_manifest(membership_dir)
        expected = {(item.structure_name, item.sha256, item.size_bytes) for item in expected_references}
        actual = {(item.structure_name, item.sha256, item.size_bytes) for item in actual_references}
        if actual != expected:
            raise ValueError(f"Written manifest does not match inventory: {membership_dir}")
        for reference in actual_references:
            if reference.sha256 not in verified_digests:
                if sha256_file(reference.path) != reference.sha256:
                    raise ValueError(f"Stored object failed SHA-256 verification: {reference.path}")
                verified_digests.add(reference.sha256)

    write_catalog(store_root.resolve(), stored_by_digest, inventory)

    # Recheck all candidates before unlinking any: deleting a symlink's legacy
    # target first can make a previously verified symlink unreadable.
    for legacy_path, expected in cleanup:
        _verify_legacy_identity(legacy_path, expected)
    for legacy_path, _ in cleanup:
        legacy_path.unlink()

    report = verify_store(data_root, store_root)
    print("Migration completed and verified.")
    print_report(report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("audit", "migrate", "verify"),
        help="Audit current references, migrate them, or verify an existing manifest-backed store.",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--store-root", type=Path, default=STRUCTURE_STORE_DIR)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON output path for the audit command.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Required with migrate to create manifests and remove redundant structure entries.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    store_root = args.store_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    try:
        store_root.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Structure store must be inside the data root: {store_root}") from exc

    if args.command == "audit":
        report = build_audit_report(
            inventory_structures(data_root, store_root),
            data_root=data_root,
            store_root=store_root,
        )
        if args.report is not None:
            write_json(args.report.expanduser().resolve(), report)
        print_report(report)
    elif args.command == "migrate":
        migrate(data_root, store_root, apply=args.apply)
    elif args.command == "verify":
        print_report(verify_store(data_root, store_root))
        print("Structure store verified.")
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
