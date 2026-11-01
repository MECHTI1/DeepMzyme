"""EC-class transfer list from the published enzyme nomenclature record (WP-12, decision A8).

Input: the ENZYME nomenclature flat file (``enzyme.dat``, Expasy), downloaded once with its URL,
retrieval date and SHA-256 recorded (P2-9c). A ``CC`` comment header (with its ``Release of``
line) may precede the records. Records end with ``//``; ``ID`` gives the EC number
and the ``DE`` lines of a transferred entry read ``Transferred entry: 1.1.1.198, 1.1.1.227 and
1.1.1.228.`` (wrapped over several ``DE`` lines when long). A deleted entry reads ``Deleted
entry.``.

Output, write-once in a new folder:
  - ``ec_transfer_list.csv`` in the scorer's format (``scoring.TRANSFER_COLUMNS``: exactly the
    columns ``ec_number, transferred_to``, targets ``;``-separated, rows in EC-number order). Every
    transferred entry is listed; the scorer itself ignores those whose targets keep the source's
    EC class and flags a protein only for a transfer that crosses an EC class.
  - ``transfer_list_receipt.json``: source (file name, SHA-256, URL, retrieval date, release line),
    counts (records, active, deleted, transferred, crossing and within an EC class), the output's
    SHA-256 and rows, and the provenance block.

The list is keyed by EC number only: no protein, label or test file is read. The parser is
strict: an ID that is not one EC number with an EC class, a transferred entry without parseable
targets, or a record without an ID stops the run with the entry named, so no transfer is dropped
silently.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

from training.ec_classes import STATUS_OK, classify_ec_class


SCHEMA_VERSION = 1
KIND = "ec_clean30_transfer_list"
# The plan's name with the scorer's comma-separated format (the scorer reads only commas).
OUTPUT_NAME = "ec_transfer_list.csv"
RECEIPT_NAME = "transfer_list_receipt.json"
RULES = (
    "Every transferred entry of the nomenclature file gets one row: its current entries, found by following "
    "later transfers (a branch that ends in a deleted entry is dropped; an entry whose every branch ends in a "
    "deletion gets no row). A target missing from the file, a cycle or text outside the transfer grammar stops "
    "the run. The scorer flags a protein only when one of its native EC numbers is a listed source whose current "
    "entries leave its EC class; it matches exact EC numbers, so a partial native label (for example 3.6.3.-) is "
    "never matched. Keyed by EC number: no protein, label or test file is read."
)
# The scorer's columns (scoring.TRANSFER_COLUMNS); a test keeps the two equal.
COLUMNS = ("ec_number", "transferred_to")
SOURCE_FORMAT = "expasy_enzyme_dat"

TRANSFER_LIST_HELP = ("EC-class transfer list (P2-9d): parse the downloaded nomenclature file into the scorer's "
                      "ec_transfer_list.csv and a receipt (keyed by EC number; no label is read).")

_EC_NUMBER = r"\d+\.\d+\.\d+\.n?\d+"
_EC_NUMBER_RE = re.compile(rf"^{_EC_NUMBER}$")
_EC_IN_TEXT_RE = re.compile(rf"(?<![\d.]){_EC_NUMBER}(?![\d])")
_TRANSFERRED_RE = re.compile(r"^Transferred entry:\s*(?P<targets>.+?)\.?\s*$")
_TARGETS_GRAMMAR_RE = re.compile(rf"^{_EC_NUMBER}(\s*(,|,?\s+and)\s+{_EC_NUMBER})*$")
_DELETED_RE = re.compile(r"^Deleted entry\.?\s*$")
_RELEASE_RE = re.compile(r"^CC\s+Release of\s+(?P<release>.+?)\s*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TransferListError(ValueError):
    """The nomenclature file or the arguments cannot produce a transfer list."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransferListError(message)


@dataclass(frozen=True)
class NomenclatureRecord:
    ec_number: str
    status: str  # "active", "deleted" or "transferred"
    targets: tuple[str, ...] = ()


def _ec_sort_key(ec_number: str) -> tuple:
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in ec_number.split("."))


def _check_ec_number(value: str, what: str) -> str:
    _require(bool(_EC_NUMBER_RE.match(value)) and classify_ec_class([value]).status == STATUS_OK,
             f"{what}: {value!r} is not one EC number with an EC class")
    return value


def parse_enzyme_dat(text: str) -> tuple[list[NomenclatureRecord], Optional[str]]:
    """Every record of an ENZYME flat file, and its release line (None when absent)."""
    records: list[NomenclatureRecord] = []
    release: Optional[str] = None
    ec_number: Optional[str] = None
    description: list[str] = []
    seen: set[str] = set()
    in_record = False

    def close(line_number: int) -> None:
        nonlocal ec_number, description, in_record
        _require(ec_number is not None, f"line {line_number}: a record ends without an ID line")
        _require(ec_number not in seen, f"EC number {ec_number} has two records")
        seen.add(ec_number)
        joined = " ".join(part.strip() for part in description).strip()
        transferred = _TRANSFERRED_RE.match(joined)
        if transferred:
            targets_text = transferred.group("targets").strip()
            _require(bool(_TARGETS_GRAMMAR_RE.match(targets_text)),
                     f"{ec_number}: the transferred entry's targets cannot be parsed: {joined!r}")
            targets = tuple(_check_ec_number(value, f"{ec_number} target")
                            for value in _EC_IN_TEXT_RE.findall(targets_text))
            _require(bool(targets) and ec_number not in targets,
                     f"{ec_number}: the transferred entry names no other EC number: {joined!r}")
            records.append(NomenclatureRecord(ec_number, "transferred", tuple(dict.fromkeys(targets))))
        elif _DELETED_RE.match(joined):
            records.append(NomenclatureRecord(ec_number, "deleted"))
        else:
            _require(not joined.startswith("Transferred entry"),
                     f"{ec_number}: the transferred entry cannot be parsed: {joined!r}")
            records.append(NomenclatureRecord(ec_number, "active"))
        ec_number, description, in_record = None, [], False

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n")
        if line.startswith("//"):
            if not in_record:  # the comment header ends with // before the first record
                _require(not records and not seen, f"line {line_number}: a record ends without an ID line")
                continue
            close(line_number)
            continue
        code, body = line[:2], line[5:] if len(line) > 5 else ""
        if code == "ID":
            _require(not in_record, f"line {line_number}: an ID line inside a record")
            ec_number = _check_ec_number(body.strip(), f"line {line_number}")
            in_record = True
        elif code == "DE" and in_record:
            description.append(body)
        elif code == "CC" and not in_record and release is None:
            match = _RELEASE_RE.match(line)
            if match:
                release = match.group("release")
    _require(not in_record, "the file ends inside a record (no closing //)")
    _require(bool(records), "the file holds no nomenclature record")
    return records, release


def resolve_transfers(records: Sequence[NomenclatureRecord]) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    """Each transferred entry's current entries, following later transfers (A -> B, B -> C gives A -> C).

    A branch that ends in a deleted entry is dropped; an entry whose every branch ends in a deletion
    gets no row (the scorer needs at least one target). A target missing from the file or a cycle
    stops the run. Counts: chains resolved (an entry with a transferred target) and both deletion
    endings.
    """
    by_ec = {record.ec_number: record for record in records}
    resolved: dict[str, tuple[str, ...]] = {}
    counts = {"chains_resolved": 0, "branches_ending_in_deletion": 0, "entries_ending_in_deletion": 0}

    def current(ec_number: str, path: tuple[str, ...]) -> tuple[tuple[str, ...], int]:
        record = by_ec.get(ec_number)
        _require(record is not None, f"{path[0]}: target {ec_number} has no record in the file")
        if record.status == "active":
            return (ec_number,), 0
        if record.status == "deleted":
            return (), 1
        _require(ec_number not in path, f"{path[0]}: transfer cycle through {ec_number}")
        found, dropped = [], 0
        for target in record.targets:
            values, ends = current(target, path + (ec_number,))
            found.extend(values)
            dropped += ends
        return tuple(dict.fromkeys(found)), dropped

    for record in records:
        if record.status != "transferred":
            continue
        found, dropped = [], 0
        for target in record.targets:
            values, ends = current(target, (record.ec_number,))
            found.extend(values)
            dropped += ends
        if any(by_ec[target].status == "transferred" for target in record.targets):
            counts["chains_resolved"] += 1
        counts["branches_ending_in_deletion"] += dropped
        if found:
            resolved[record.ec_number] = tuple(sorted(dict.fromkeys(found), key=_ec_sort_key))
        else:
            counts["entries_ending_in_deletion"] += 1
    return resolved, counts


def transfer_rows(records: Sequence[NomenclatureRecord]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """The scorer's rows (every transferred entry with its current entries, in EC-number order)."""
    resolved, counts = resolve_transfers(records)
    rows = [{"ec_number": ec_number, "transferred_to": ";".join(targets)} for ec_number, targets in resolved.items()]
    return sorted(rows, key=lambda row: _ec_sort_key(row["ec_number"])), counts


def crosses_ec_class(row: Mapping[str, str]) -> bool:
    """The scorer's rule: a transfer crosses an EC class unless every target keeps the source's class."""
    targets = [value for value in row["transferred_to"].split(";") if value]
    return {target.split(".")[0] for target in targets} != {row["ec_number"].split(".")[0]}


def csv_text(rows: Sequence[Mapping[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _check_date(value: str) -> str:
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError:
        raise TransferListError(f"--retrieved must be a date YYYY-MM-DD, got {value!r}") from None
    _require(parsed.isoformat() == value, f"--retrieved must be a date YYYY-MM-DD, got {value!r}")
    _require(parsed <= dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1),
             f"--retrieved {value} lies in the future")
    return value


def _scorer_parity(body: bytes, rows: Sequence[Mapping[str, str]]) -> None:
    """The scorer's own reader must accept the list and find the same crossing and within-class rows."""
    from ec_clean30 import scoring

    crossing, ignored = scoring.parse_transfer_list(body, OUTPUT_NAME)
    expected = {row["ec_number"] for row in rows if crosses_ec_class(row)}
    _require(crossing == frozenset(expected) and ignored == len(rows) - len(expected),
             "the scorer reads the list differently from the producer")


def build_transfer_list(source: Path, *, source_sha256: str, source_url: str, retrieved: str, out: Path,
                        provenance: Mapping[str, Any], check_scorer_parity: bool = True) -> dict[str, Any]:
    """Parse ``source`` and write the list and its receipt into the new folder ``out``."""
    from ec_clean30.provenance import sha256_bytes, write_new_json

    source, out = Path(source), Path(out)
    _require(source.is_file(), f"--source is not a file: {source}")
    expected = source_sha256.strip().lower()
    _require(bool(_SHA256_RE.match(expected)), "--source-sha256 must be 64 hexadecimal characters")
    data = source.read_bytes()
    actual = sha256_bytes(data)
    _require(actual == expected, f"--source does not hash to the --source-sha256 given (it hashes to {actual})")
    _require(source_url.startswith("https://"), "--source-url must be an https URL")
    _check_date(retrieved)
    _require(not out.exists(), f"--out must be a new folder; {out} exists")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    records, release = parse_enzyme_dat(text)
    rows, chain_counts = transfer_rows(records)
    crossing = sum(crosses_ec_class(row) for row in rows)
    body = csv_text(rows).encode("utf-8")
    if check_scorer_parity:
        _scorer_parity(body, rows)
    out.mkdir(parents=True)
    with (out / OUTPUT_NAME).open("xb") as handle:
        handle.write(body)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "source": {"name": source.name, "sha256": actual, "url": source_url, "retrieved": retrieved,
                   "format": SOURCE_FORMAT, "release": release},
        "counts": {"records": len(records),
                   "active": sum(record.status == "active" for record in records),
                   "deleted": sum(record.status == "deleted" for record in records),
                   "transferred": sum(record.status == "transferred" for record in records),
                   "rows": len(rows),
                   "transferred_crossing_ec_class": crossing,
                   "transferred_within_ec_class": len(rows) - crossing,
                   **chain_counts},
        "output": {"name": OUTPUT_NAME, "sha256": sha256_bytes(body), "rows": len(rows), "columns": list(COLUMNS)},
        "rules": RULES,
        "labels_read": False,
        "provenance": dict(provenance),
    }
    receipt_sha256 = write_new_json(out / RECEIPT_NAME, receipt)
    return {"out": str(out), "receipt_sha256": receipt_sha256, **receipt["counts"],
            "output_sha256": receipt["output"]["sha256"]}


def add_transfer_list_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--source", required=True, help="The downloaded nomenclature file (enzyme.dat).")
    parser.add_argument("--source-sha256", required=True, help="Its SHA-256, as recorded at download.")
    parser.add_argument("--source-url", required=True, help="The URL approved in A8.")
    parser.add_argument("--retrieved", required=True, help="Retrieval date, YYYY-MM-DD.")
    parser.add_argument("--out", required=True, help="New output folder (for example <campaign root>/labels/...).")
    return parser


def action_transfer_list(args: argparse.Namespace) -> int:
    import json

    from ec_clean30.provenance import provenance_record, real_action_git_state, refuse_test_paths

    refuse_test_paths([args.source, args.out], what="--source/--out")
    git = real_action_git_state(smoke=False)
    summary = build_transfer_list(Path(args.source), source_sha256=args.source_sha256, source_url=args.source_url,
                                  retrieved=args.retrieved, out=Path(args.out),
                                  provenance=provenance_record(smoke=False, git=git))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
