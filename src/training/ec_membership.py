"""Frozen EC fold membership: one row per protein structure, keyed by structure ID.

The schema part (reading, writing, validation) is torch-free so the identity tools can
write memberships without loading the trainer; the split imports ``training.splits``
lazily. A frozen membership is never re-split: loaded proteins must match it exactly.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training.ec_classes import EC_CLASS_TOKENS, accession_from_structure_id


EC_FOLD_MEMBERSHIP_COLUMNS: tuple[str, ...] = ("structure_id", "accession", "group_id", "fold")
EC_FOLD_MEMBERSHIP_OPTIONAL_COLUMNS: tuple[str, ...] = ("ec_class",)


class EcMembershipError(ValueError):
    """A frozen EC membership is malformed or does not match the loaded proteins."""


@dataclass(frozen=True)
class EcFoldMembership:
    rows: tuple[dict[str, str], ...]
    file_sha256: str
    identity_sha256: str

    @property
    def fold_by_structure(self) -> dict[str, int]:
        return {row["structure_id"]: int(row["fold"]) for row in self.rows}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalised_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    normalised = []
    for row in rows:
        item = {column: str(row.get(column, "")).strip() for column in EC_FOLD_MEMBERSHIP_COLUMNS}
        ec_class = row.get("ec_class")
        if ec_class not in (None, ""):
            item["ec_class"] = str(ec_class).strip()
        normalised.append(item)
    return sorted(normalised, key=lambda r: (_fold_value(r), r["group_id"], r["structure_id"]))


def _fold_value(row: Mapping[str, str]) -> int:
    try:
        fold = int(row["fold"])
    except (TypeError, ValueError):
        raise EcMembershipError(f"Fold {row.get('fold')!r} of {row.get('structure_id')!r} is not an integer") from None
    if fold < 0 or str(fold) != str(row["fold"]).strip():
        raise EcMembershipError(f"Fold {row.get('fold')!r} of {row.get('structure_id')!r} is not a non-negative integer")
    return fold


def validate_ec_fold_membership_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise EcMembershipError("An EC fold membership needs at least one row")
    structures: set[str] = set()
    accessions: set[str] = set()
    group_folds: dict[str, int] = {}
    with_class = {"ec_class" in row for row in rows}
    if len(with_class) > 1:
        raise EcMembershipError("Either every row or no row carries ec_class")
    for row in rows:
        structure_id, accession, group_id = row["structure_id"], row["accession"], row["group_id"]
        if not structure_id or structure_id in structures:
            raise EcMembershipError(f"Empty or repeated structure_id {structure_id!r}")
        structures.add(structure_id)
        if not accession or accession in accessions:
            raise EcMembershipError(f"Empty or repeated accession {accession!r}")
        accessions.add(accession)
        try:
            stem_accession = accession_from_structure_id(structure_id)
        except ValueError as exc:
            raise EcMembershipError(str(exc)) from None
        if stem_accession != accession:
            raise EcMembershipError(f"Accession {accession!r} does not match structure {structure_id!r}")
        if not group_id:
            raise EcMembershipError(f"Structure {structure_id!r} has an empty group_id")
        fold = _fold_value(row)
        if group_folds.setdefault(group_id, fold) != fold:
            raise EcMembershipError(f"Group {group_id!r} is spread over more than one fold")
        if "ec_class" in row and row["ec_class"] not in EC_CLASS_TOKENS:
            raise EcMembershipError(f"Structure {structure_id!r} has EC class {row['ec_class']!r} outside 1-7")


def membership_identity_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    """Content identity independent of CSV formatting (sorted canonical JSON)."""
    canonical = json.dumps([dict(sorted(row.items())) for row in rows], sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def write_ec_fold_membership(rows: Iterable[Mapping[str, Any]], path: Path) -> str:
    """Write a validated membership with deterministic order and LF line endings; return its SHA-256."""
    normalised = _normalised_rows(rows)
    validate_ec_fold_membership_rows(normalised)
    columns = list(EC_FOLD_MEMBERSHIP_COLUMNS) + (["ec_class"] if "ec_class" in normalised[0] else [])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalised)
    payload = buffer.getvalue().encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(payload)
    return _sha256_bytes(payload)


def read_ec_fold_membership(path: Path, expected_sha256: str) -> EcFoldMembership:
    payload = Path(path).read_bytes()
    actual = _sha256_bytes(payload)
    if actual != str(expected_sha256).lower():
        raise EcMembershipError(f"EC fold membership {path} has SHA-256 {actual}, expected {expected_sha256}")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    fieldnames = list(reader.fieldnames or [])
    missing = [column for column in EC_FOLD_MEMBERSHIP_COLUMNS if column not in fieldnames]
    unknown = [column for column in fieldnames
               if column not in EC_FOLD_MEMBERSHIP_COLUMNS + EC_FOLD_MEMBERSHIP_OPTIONAL_COLUMNS]
    if missing or unknown:
        raise EcMembershipError(f"EC fold membership {path}: missing columns {missing}, unknown columns {unknown}")
    rows = _normalised_rows(reader)
    validate_ec_fold_membership_rows(rows)
    return EcFoldMembership(rows=tuple(rows), file_sha256=actual, identity_sha256=membership_identity_sha256(rows))


def ec_membership_split(
    pockets: Sequence[Any],
    membership: EcFoldMembership,
    *,
    fold_index: int,
    n_folds: int,
):
    """Train/validation split taken directly from a frozen EC membership (validation = ``fold_index``).

    Keeps the loaded order inside each side and fails when any loaded structure is
    unknown, any membership structure was not loaded, or a fold is outside ``range(n_folds)``.
    """
    from training.splits import PocketSplit

    fold_by_structure = membership.fold_by_structure
    folds = set(fold_by_structure.values())
    if not folds <= set(range(int(n_folds))):
        raise EcMembershipError(f"EC membership uses folds {sorted(folds)}, outside range({n_folds})")
    if int(fold_index) not in folds:
        raise EcMembershipError(f"Fold {fold_index} has no members in the EC membership")
    train, val = [], []
    loaded: set[str] = set()
    for pocket in pockets:
        structure_id = pocket.structure_id
        if structure_id not in fold_by_structure:
            raise EcMembershipError(f"Loaded structure {structure_id!r} is not in the EC membership")
        loaded.add(structure_id)
        (val if fold_by_structure[structure_id] == int(fold_index) else train).append(pocket)
    missing = sorted(set(fold_by_structure) - loaded)
    if missing:
        raise EcMembershipError(
            f"{len(missing)} membership structure(s) were not loaded, e.g. {missing[:3]}; no fallback split is made"
        )
    if not train or not val:
        raise EcMembershipError(f"Fold {fold_index} leaves an empty training or validation side")
    receipt = {
        "split_type": "frozen_ec_membership",
        "fold_index": int(fold_index),
        "n_folds": int(n_folds),
        "membership_file_sha256": membership.file_sha256,
        "membership_identity_sha256": membership.identity_sha256,
        "n_train_structures": len({p.structure_id for p in train}),
        "n_val_structures": len({p.structure_id for p in val}),
        "n_train_pockets": len(train),
        "n_val_pockets": len(val),
    }
    return PocketSplit(train_pockets=train, val_pockets=val), receipt
