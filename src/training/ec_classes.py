"""Fixed seven-class EC-class (first EC digit) vocabulary and strict label checks.

Torch-free on purpose: the campaign scorer and identity tools import this module
without loading the trainer. ``training.labels`` stays the legacy rule for every
existing run; this module is used only when a run opts into ``fixed_seven``.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Optional, Sequence


EC_CLASS_TOKENS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7")
EC_CLASS_TOKEN_TO_INDEX: dict[str, int] = {token: index for index, token in enumerate(EC_CLASS_TOKENS)}
EC_PRIMARY_CLASS_TOKENS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")
EC_CLASS_NAMES: dict[str, str] = {
    "1": "oxidoreductases",
    "2": "transferases",
    "3": "hydrolases",
    "4": "lyases",
    "5": "isomerases",
    "6": "ligases",
    "7": "translocases",
}

STATUS_OK = "ok"
STATUS_AMBIGUOUS = "ambiguous_ec_class"
STATUS_MISSING = "missing"
STATUS_OUT_OF_RANGE = "out_of_range"
STATUS_MALFORMED = "malformed"
STATUS_MIXED_UNKNOWN = "mixed_unknown"
FATAL_STATUSES = frozenset({STATUS_MISSING, STATUS_OUT_OF_RANGE, STATUS_MALFORMED, STATUS_MIXED_UNKNOWN})

FLAG_NONCANONICAL_DEEPER = "noncanonical_deeper"
FLAG_INCOMPLETE_DEEPER = "incomplete_deeper"

# Copies of training.labels.STRUCTURE_ID_RE and normalize_ec_number_list, kept here so
# this module stays importable without torch; tests assert parity with the originals.
STRUCTURE_ID_RE = re.compile(r"^(?P<pdbid>[^_]+)__chain_(?P<chain>[^_]+)__EC_(?P<ec>.+)$")
_EC_CLASS_RE = re.compile(r"^[1-7]$")
_DIGITS_RE = re.compile(r"^\d+$")


def normalize_ec_number_list(value: str) -> str:
    values = []
    seen = set()
    for ec in re.split(r"[;,]", value):
        ec = ec.strip()
        if not ec or ec in seen:
            continue
        seen.add(ec)
        values.append(ec)
    return ";".join(values)


def ec_numbers_from_structure_id(structure_id: str) -> tuple[str, ...]:
    match = STRUCTURE_ID_RE.match(structure_id.strip())
    if match is None:
        raise ValueError(f"Could not parse structure identity from {structure_id!r}")
    return tuple(ec for ec in normalize_ec_number_list(match.group("ec")).split(";") if ec)


def accession_from_structure_id(structure_id: str) -> str:
    match = STRUCTURE_ID_RE.match(structure_id.strip())
    if match is None:
        raise ValueError(f"Could not parse structure identity from {structure_id!r}")
    return match.group("pdbid").strip()


@dataclass(frozen=True)
class EcClassResult:
    token: Optional[str]
    status: str
    flags: tuple[str, ...] = ()

    @property
    def is_fatal(self) -> bool:
        return self.status in FATAL_STATUSES


def _entry_kind(entry: str) -> str:
    first = entry.split(".")[0].strip()
    if first in {"", "-"}:
        return STATUS_MISSING
    if _EC_CLASS_RE.match(first):
        return STATUS_OK
    if _DIGITS_RE.match(first):
        return STATUS_OUT_OF_RANGE
    return STATUS_MALFORMED


def _deeper_flags(entry: str) -> set[str]:
    flags = set()
    deeper = [part.strip() for part in entry.split(".")[1:]]
    if len(deeper) < 3 or any(part in {"", "-"} for part in deeper):
        flags.add(FLAG_INCOMPLETE_DEEPER)
    if any(part not in {"", "-"} and not _DIGITS_RE.match(part) for part in deeper):
        flags.add(FLAG_NONCANONICAL_DEEPER)
    return flags


def classify_ec_class(ec_numbers: Iterable[str]) -> EcClassResult:
    """Strict EC-class (first digit) label for one protein's EC annotations.

    Fatal statuses mean the annotation cannot be trusted and the run must stop;
    ``ambiguous_ec_class`` (several valid classes) is a counted exclusion.
    """
    entries = [entry for entry in normalize_ec_number_list(";".join(str(e) for e in ec_numbers)).split(";") if entry]
    if not entries:
        return EcClassResult(None, STATUS_MISSING)
    kinds = [_entry_kind(entry) for entry in entries]
    valid_entries = [entry for entry, kind in zip(entries, kinds) if kind == STATUS_OK]
    invalid_kinds = {kind for kind in kinds if kind != STATUS_OK}
    if invalid_kinds and valid_entries:
        return EcClassResult(None, STATUS_MIXED_UNKNOWN)
    if STATUS_MALFORMED in invalid_kinds:
        return EcClassResult(None, STATUS_MALFORMED)
    if STATUS_OUT_OF_RANGE in invalid_kinds:
        return EcClassResult(None, STATUS_OUT_OF_RANGE)
    if invalid_kinds:
        return EcClassResult(None, STATUS_MISSING)
    flags = sorted(set().union(*(_deeper_flags(entry) for entry in valid_entries)))
    classes = sorted({entry.split(".")[0].strip() for entry in valid_entries})
    if len(classes) > 1:
        return EcClassResult(None, STATUS_AMBIGUOUS, tuple(flags))
    return EcClassResult(classes[0], STATUS_OK, tuple(flags))


def complete_ec_prefix(ec_numbers: Iterable[str], depth: int) -> Optional[str]:
    """The single complete EC prefix at ``depth`` (all components numeric), else None."""
    if depth < 1:
        raise ValueError(f"EC prefix depth must be at least 1, got {depth}.")
    prefixes = set()
    for entry in normalize_ec_number_list(";".join(str(e) for e in ec_numbers)).split(";"):
        parts = [part.strip() for part in entry.split(".")]
        if len(parts) < depth:
            continue
        head = parts[:depth]
        if all(_DIGITS_RE.match(part) for part in head) and _EC_CLASS_RE.match(head[0]):
            prefixes.add(".".join(head))
    return next(iter(prefixes)) if len(prefixes) == 1 else None


def _pocket_ec_numbers(pocket: Any) -> tuple[str, ...]:
    values = pocket.metadata.get("ec_numbers")
    if values is None:
        return ec_numbers_from_structure_id(pocket.structure_id)
    if isinstance(values, str):
        values = values.split(";")
    return tuple(str(value) for value in values)


def apply_fixed_ec_class_vocabulary(pockets: Sequence[Any]) -> dict[str, Any]:
    """Set ``metadata['ec_label_token']`` from the strict rule for every pocket.

    Raises ValueError listing structure IDs when any annotation has a fatal status.
    Ambiguous proteins lose their token and get ``ec_label_status``; they are
    excluded by the loader and counted in the returned report.
    """
    status_by_structure: dict[str, EcClassResult] = {}
    for pocket in pockets:
        result = classify_ec_class(_pocket_ec_numbers(pocket))
        previous = status_by_structure.setdefault(pocket.structure_id, result)
        if previous != result:
            raise ValueError(f"Pockets of {pocket.structure_id!r} carry different EC annotations.")
        legacy = pocket.metadata.get("ec_label_token")
        if result.status == STATUS_OK and legacy is not None and str(legacy) != result.token:
            raise ValueError(
                f"Strict EC class {result.token!r} disagrees with legacy token {legacy!r} for {pocket.structure_id!r}."
            )
        pocket.metadata["ec_label_status"] = result.status
        if result.status == STATUS_OK:
            pocket.metadata["ec_label_token"] = result.token
        else:
            pocket.metadata.pop("ec_label_token", None)

    fatal = {sid: result.status for sid, result in status_by_structure.items() if result.is_fatal}
    if fatal:
        listed = ", ".join(f"{sid} ({status})" for sid, status in sorted(fatal.items())[:20])
        raise ValueError(
            f"{len(fatal)} structure(s) have EC annotations that cannot give an EC class: {listed}"
            + (" ..." if len(fatal) > 20 else "")
        )

    counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    class_counts = {token: 0 for token in EC_CLASS_TOKENS}
    for result in status_by_structure.values():
        counts[result.status] = counts.get(result.status, 0) + 1
        for flag in result.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        if result.status == STATUS_OK:
            class_counts[result.token] += 1
    return {
        "vocabulary": "fixed_seven",
        "ec_class_tokens": list(EC_CLASS_TOKENS),
        "n_structures": len(status_by_structure),
        "structures_by_status": dict(sorted(counts.items())),
        "structures_by_flag": dict(sorted(flag_counts.items())),
        "structures_by_ec_class": class_counts,
        "excluded_ambiguous_structure_ids": sorted(
            sid for sid, result in status_by_structure.items() if result.status == STATUS_AMBIGUOUS
        ),
    }


def assert_consistent_ec_protein_labels(
    pockets: Sequence[Any],
    *,
    membership_ec_class: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Fail when one accession carries more than one EC class, EC string or structure.

    ``membership_ec_class`` optionally maps accession -> expected EC class token.
    """
    tokens: dict[str, set[str]] = {}
    ec_strings: dict[str, set[str]] = {}
    structures: dict[str, set[str]] = {}
    for pocket in pockets:
        accession = accession_from_structure_id(pocket.structure_id)
        token = pocket.metadata.get("ec_label_token")
        if token is not None:
            tokens.setdefault(accession, set()).add(str(token))
        ec_strings.setdefault(accession, set()).add(";".join(_pocket_ec_numbers(pocket)))
        structures.setdefault(accession, set()).add(pocket.structure_id)

    problems = []
    for accession in sorted(structures):
        if len(tokens.get(accession, set())) > 1:
            problems.append(f"{accession}: EC classes {sorted(tokens[accession])}")
        if len(ec_strings[accession]) > 1:
            problems.append(f"{accession}: EC annotations {sorted(ec_strings[accession])}")
        if len(structures[accession]) > 1:
            problems.append(f"{accession}: structures {sorted(structures[accession])}")
        if membership_ec_class is not None and accession in membership_ec_class:
            observed = tokens.get(accession, set())
            if observed and observed != {str(membership_ec_class[accession])}:
                problems.append(
                    f"{accession}: membership EC class {membership_ec_class[accession]!r} vs {sorted(observed)}"
                )
    if problems:
        raise ValueError("Conflicting EC protein labels: " + "; ".join(problems[:20]))
    return {"n_accessions": len(structures), "conflicts": 0}


def ec_class_split_support(train_pockets: Sequence[Any], val_pockets: Sequence[Any]) -> dict[str, Any]:
    """Protein (structure) counts per EC class in each side of a split, over all seven classes.

    Classes absent from training are counted, never dropped, so their validation
    proteins stay visible as guaranteed errors.
    """
    def per_class(pockets: Sequence[Any]) -> dict[str, int]:
        structures: dict[str, set[str]] = {token: set() for token in EC_CLASS_TOKENS}
        for pocket in pockets:
            token = pocket.metadata.get("ec_label_token")
            if token is not None and str(token) in structures:
                structures[str(token)].add(pocket.structure_id)
        return {token: len(ids) for token, ids in structures.items()}

    train = per_class(train_pockets)
    val = per_class(val_pockets)
    absent = [token for token in EC_CLASS_TOKENS if train[token] == 0]
    return {
        "train_proteins_by_ec_class": train,
        "val_proteins_by_ec_class": val,
        "ec_classes_absent_from_training": absent,
        "n_val_proteins_in_classes_absent_from_training": sum(val[token] for token in absent),
    }
