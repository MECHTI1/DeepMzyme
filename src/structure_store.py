from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


STRUCTURE_MANIFEST_FILENAME = "structure_manifest.csv"
STRUCTURE_MANIFEST_FIELDS = ("structure_name", "relative_path", "sha256", "size_bytes")
STRUCTURE_SUFFIXES = frozenset({".pdb", ".cif", ".mmcif"})


@dataclass(frozen=True)
class StructureReference:
    """One logical structure name and its content-addressed physical file."""

    structure_name: str
    path: Path
    sha256: str
    size_bytes: int


def is_structure_file(path: Path) -> bool:
    return path.suffix.lower() in STRUCTURE_SUFFIXES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_object_path(store_root: Path, *, sha256: str, structure_name: str) -> Path:
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError(f"Invalid SHA-256 digest: {sha256!r}")
    if Path(structure_name).name != structure_name or not is_structure_file(Path(structure_name)):
        raise ValueError(f"Invalid structure filename: {structure_name!r}")
    return Path(store_root) / "objects" / sha256[:2] / sha256 / structure_name


def store_structure(
    source_path: Path,
    store_root: Path,
    *,
    sha256: str | None = None,
    prefer_hardlink: bool = True,
) -> StructureReference:
    """Add a structure to the global store without replacing an existing object."""

    source_path = Path(source_path)
    if not source_path.is_file() or not is_structure_file(source_path):
        raise FileNotFoundError(f"Structure file not found or unsupported: {source_path}")
    digest = sha256 or sha256_file(source_path)
    size_bytes = source_path.stat().st_size
    destination = structure_object_path(
        Path(store_root), sha256=digest, structure_name=source_path.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    other_names = sorted(
        path.name
        for path in destination.parent.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.name != source_path.name
    )
    if other_names:
        raise ValueError(
            "Identical structure bytes are already stored under a different filename; "
            "one physical object cannot preserve two path-derived structure IDs: "
            f"{other_names[0]!r} versus {source_path.name!r}"
        )

    if destination.exists():
        if destination.stat().st_size != size_bytes or sha256_file(destination) != digest:
            raise ValueError(f"Existing structure-store object failed verification: {destination}")
    else:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{source_path.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        try:
            if prefer_hardlink:
                try:
                    os.link(source_path.resolve(), temporary_path)
                except OSError:
                    shutil.copy2(source_path, temporary_path)
            else:
                shutil.copy2(source_path, temporary_path)
            if temporary_path.stat().st_size != size_bytes or sha256_file(temporary_path) != digest:
                raise ValueError(f"Staged structure-store object failed verification: {source_path}")
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    return StructureReference(
        structure_name=source_path.name,
        path=destination.resolve(),
        sha256=digest,
        size_bytes=size_bytes,
    )


def _validate_reference(reference: StructureReference, *, verify_hash: bool) -> None:
    if Path(reference.structure_name).name != reference.structure_name:
        raise ValueError(f"Manifest structure_name must be a filename: {reference.structure_name!r}")
    if not is_structure_file(Path(reference.structure_name)):
        raise ValueError(f"Unsupported structure manifest entry: {reference.structure_name!r}")
    if len(reference.sha256) != 64 or any(
        character not in "0123456789abcdef" for character in reference.sha256
    ):
        raise ValueError(f"Invalid SHA-256 for {reference.structure_name}: {reference.sha256!r}")
    if reference.size_bytes < 0:
        raise ValueError(f"Invalid size for {reference.structure_name}: {reference.size_bytes}")
    if reference.path.name != reference.structure_name:
        raise ValueError(
            "Manifest logical and physical filenames must match so downstream IDs remain stable: "
            f"{reference.structure_name!r} != {reference.path.name!r}"
        )
    if not reference.path.is_file():
        raise FileNotFoundError(f"Manifest target does not exist: {reference.path}")
    actual_size = reference.path.stat().st_size
    if actual_size != reference.size_bytes:
        raise ValueError(
            f"Manifest size mismatch for {reference.structure_name}: "
            f"expected {reference.size_bytes}, found {actual_size}"
        )
    if verify_hash and sha256_file(reference.path) != reference.sha256:
        raise ValueError(f"Manifest SHA-256 mismatch for {reference.structure_name}: {reference.path}")


def read_structure_manifest(
    structure_dir: Path,
    *,
    verify_hashes: bool = True,
) -> list[StructureReference]:
    """Resolve targets only after checking their declared content identities.

    ``verify_hashes`` remains accepted for caller compatibility, but cannot
    disable integrity verification on reads, including runtime resolution.
    """
    del verify_hashes
    structure_dir = Path(structure_dir)
    manifest_path = structure_dir / STRUCTURE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Structure manifest not found: {manifest_path}")

    references: list[StructureReference] = []
    seen_names: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = [field for field in STRUCTURE_MANIFEST_FIELDS if field not in (reader.fieldnames or [])]
        if missing_fields:
            raise ValueError(
                f"Structure manifest {manifest_path} is missing columns: {', '.join(missing_fields)}"
            )
        for row_number, row in enumerate(reader, start=2):
            structure_name = (row.get("structure_name") or "").strip()
            relative_path = (row.get("relative_path") or "").strip()
            digest = (row.get("sha256") or "").strip().lower()
            size_text = (row.get("size_bytes") or "").strip()
            if not structure_name or not relative_path or not digest or not size_text:
                raise ValueError(f"Incomplete structure manifest row at {manifest_path}:{row_number}")
            if Path(relative_path).is_absolute():
                raise ValueError(f"Absolute paths are not portable: {manifest_path}:{row_number}")
            if structure_name in seen_names:
                raise ValueError(f"Duplicate structure_name {structure_name!r} in {manifest_path}")
            try:
                size_bytes = int(size_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid size_bytes at {manifest_path}:{row_number}: {size_text!r}"
                ) from exc
            reference = StructureReference(
                structure_name=structure_name,
                path=(structure_dir / Path(relative_path)).resolve(),
                sha256=digest,
                size_bytes=size_bytes,
            )
            _validate_reference(reference, verify_hash=True)
            references.append(reference)
            seen_names.add(structure_name)
    return sorted(references, key=lambda reference: reference.structure_name)


def resolve_structure_files(
    structure_dir: Path,
    *,
    verify_hashes: bool = True,
    recursive_legacy_scan: bool = True,
) -> list[Path]:
    """Resolve manifest-backed structures, with a legacy file-directory fallback."""

    structure_dir = Path(structure_dir)
    manifest_path = structure_dir / STRUCTURE_MANIFEST_FILENAME
    if manifest_path.is_file():
        return [reference.path for reference in read_structure_manifest(structure_dir, verify_hashes=verify_hashes)]

    iterator = structure_dir.rglob("*") if recursive_legacy_scan else structure_dir.iterdir()
    return sorted(path for path in iterator if path.is_file() and is_structure_file(path))


def references_for_paths(
    paths: Iterable[Path],
    *,
    known_hashes: dict[Path, str] | None = None,
) -> list[StructureReference]:
    references: list[StructureReference] = []
    seen_names: set[str] = set()
    normalized_hashes = {Path(path).resolve(): digest for path, digest in (known_hashes or {}).items()}
    for path in sorted((Path(path).resolve() for path in paths), key=lambda item: item.name):
        if path.name in seen_names:
            raise ValueError(f"A structure manifest cannot contain duplicate filename {path.name!r}")
        digest = normalized_hashes.get(path) or sha256_file(path)
        reference = StructureReference(path.name, path, digest, path.stat().st_size)
        _validate_reference(reference, verify_hash=False)
        references.append(reference)
        seen_names.add(path.name)
    return references


def write_structure_manifest(
    structure_dir: Path,
    references: Sequence[StructureReference],
    *,
    verify_hashes: bool = False,
) -> Path:
    """Atomically write a portable structure manifest after validating every target."""

    structure_dir = Path(structure_dir)
    structure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = structure_dir / STRUCTURE_MANIFEST_FILENAME
    ordered = sorted(references, key=lambda reference: reference.structure_name)
    seen_names: set[str] = set()
    for reference in ordered:
        if reference.structure_name in seen_names:
            raise ValueError(f"Duplicate structure_name {reference.structure_name!r} for {manifest_path}")
        _validate_reference(reference, verify_hash=verify_hashes)
        seen_names.add(reference.structure_name)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=structure_dir
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STRUCTURE_MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            for reference in ordered:
                writer.writerow(
                    {
                        "structure_name": reference.structure_name,
                        "relative_path": os.path.relpath(reference.path, structure_dir),
                        "sha256": reference.sha256,
                        "size_bytes": reference.size_bytes,
                    }
                )
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return manifest_path


def write_structure_manifest_for_paths(
    structure_dir: Path,
    paths: Iterable[Path],
    *,
    verify_hashes: bool = False,
) -> Path:
    return write_structure_manifest(
        structure_dir,
        references_for_paths(paths),
        verify_hashes=verify_hashes,
    )


def index_structure_files_by_name(structure_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in resolve_structure_files(structure_dir, recursive_legacy_scan=False):
        if path.name in index and index[path.name].resolve() != path.resolve():
            raise ValueError(f"Duplicate structure filename in {structure_dir}: {path.name}")
        index[path.name] = path
    return index
