"""EC-class transfer list producer (WP-12; synthetic nomenclature files only, no dataset access)."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ec_clean30 import scoring, transfers
from ec_clean30.transfers import (
    TransferListError, build_transfer_list, parse_enzyme_dat, resolve_transfers, transfer_rows,
)


HEADER = """CC   -----------------------------------------------------------------------
CC
CC   ENZYME nomenclature database
CC
CC   Release of 01-Oct-2026
CC   -----------------------------------------------------------------------
//
"""


def _record(ec_number: str, *description: str) -> str:
    return f"ID   {ec_number}\n" + "".join(f"DE   {line}\n" for line in description) + "//\n"


WRAPPED_TARGETS = ("3.6.1.15", "3.6.4.1", "5.6.1.1", "5.6.1.2", "5.6.1.3", "5.6.1.4", "5.6.1.5", "5.6.1.6",
                   "5.6.1.7", "7.1.2.2")
ACTIVE = ("1.1.1.1", "1.1.1.9", "1.1.1.303", "1.1.1.304", "4.2.1.3", "2.1.1.1") + WRAPPED_TARGETS
RECORDS = "".join(_record(ec, "An active enzyme.") for ec in ACTIVE) + "".join([
    _record("1.1.1.10", "Transferred entry: 1.1.1.9."),
    _record("1.1.1.5", "Transferred entry: 1.1.1.303 and 1.1.1.304."),
    _record("3.6.1.3", "Transferred entry: 3.6.1.15, 3.6.4.1, 5.6.1.1, 5.6.1.2, 5.6.1.3, 5.6.1.4, 5.6.1.5,",
            "5.6.1.6, 5.6.1.7 and 7.1.2.2."),
    _record("2.7.1.69", "Deleted entry."),
    _record("4.2.1.n1", "Transferred entry: 4.2.1.3."),
    # A chain that leaves the EC class only at its second step: 1.1.1.20 -> 1.1.1.21 -> 2.1.1.1.
    _record("1.1.1.20", "Transferred entry: 1.1.1.21."),
    _record("1.1.1.21", "Transferred entry: 2.1.1.1."),
    # One branch ends in a deleted entry; an entry whose only target was deleted gets no row.
    _record("1.1.1.30", "Transferred entry: 1.1.1.1 and 2.7.1.69."),
    _record("1.1.1.31", "Transferred entry: 2.7.1.69."),
])


def _write(tmp_path: Path, text: str, name: str = "enzyme.dat") -> tuple[Path, str]:
    path = tmp_path / name
    path.write_text(text)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_columns_are_the_scorers():
    assert transfers.COLUMNS == scoring.TRANSFER_COLUMNS
    assert transfers.OUTPUT_NAME.endswith(".csv")


def test_parse_records_targets_and_release():
    records, release = parse_enzyme_dat(HEADER + RECORDS)
    assert release == "01-Oct-2026"
    by_ec = {record.ec_number: record for record in records}
    assert [record.status for record in records].count("active") == len(ACTIVE)
    assert by_ec["2.7.1.69"].status == "deleted"
    assert by_ec["1.1.1.5"].targets == ("1.1.1.303", "1.1.1.304")
    assert by_ec["3.6.1.3"].targets == WRAPPED_TARGETS  # wrapped DE lines are joined
    assert by_ec["4.2.1.n1"].targets == ("4.2.1.3",)


def test_chains_resolve_to_current_entries():
    records, _ = parse_enzyme_dat(HEADER + RECORDS)
    resolved, counts = resolve_transfers(records)
    assert resolved["1.1.1.20"] == ("2.1.1.1",) and resolved["1.1.1.21"] == ("2.1.1.1",)
    assert resolved["1.1.1.30"] == ("1.1.1.1",) and "1.1.1.31" not in resolved
    assert counts == {"chains_resolved": 1, "branches_ending_in_deletion": 2, "entries_ending_in_deletion": 1}


def test_rows_are_in_ec_number_order_and_the_scorer_reads_them():
    records, _ = parse_enzyme_dat(HEADER + RECORDS)
    rows, _counts = transfer_rows(records)
    assert [row["ec_number"] for row in rows] == ["1.1.1.5", "1.1.1.10", "1.1.1.20", "1.1.1.21", "1.1.1.30",
                                                  "3.6.1.3", "4.2.1.n1"]
    text = transfers.csv_text(rows)
    assert text.splitlines()[0] == "ec_number,transferred_to"
    assert ";".join(WRAPPED_TARGETS) in text
    crossing, ignored = scoring.parse_transfer_list(text.encode(), transfers.OUTPUT_NAME)
    assert crossing == frozenset({"1.1.1.20", "1.1.1.21", "3.6.1.3"}) and ignored == 4
    assert [transfers.crosses_ec_class(row) for row in rows] == [False, False, True, True, False, True, False]


def test_build_writes_the_list_and_receipt_once(tmp_path):
    source, sha = _write(tmp_path, HEADER + RECORDS)
    out = tmp_path / "labels" / "ec_transfer_list"
    url = "https://ftp.expasy.org/databases/enzyme/enzyme.dat"
    summary = build_transfer_list(source, source_sha256=sha, source_url=url, retrieved="2026-11-01", out=out,
                                  provenance={"git": {"head": "x"}})
    assert (summary["records"], summary["active"], summary["deleted"]) == (len(ACTIVE) + 9, len(ACTIVE), 1)
    assert (summary["transferred"], summary["rows"]) == (8, 7)
    assert (summary["transferred_crossing_ec_class"], summary["transferred_within_ec_class"]) == (3, 4)
    receipt = json.loads((out / transfers.RECEIPT_NAME).read_text())
    body = (out / transfers.OUTPUT_NAME).read_bytes()
    assert receipt["output"]["sha256"] == hashlib.sha256(body).hexdigest() == summary["output_sha256"]
    assert receipt["source"] == {"name": "enzyme.dat", "sha256": sha, "retrieved": "2026-11-01", "url": url,
                                 "format": transfers.SOURCE_FORMAT, "release": "01-Oct-2026"}
    assert receipt["output"]["rows"] == 7 and receipt["provenance"] == {"git": {"head": "x"}}
    assert receipt["labels_read"] is False and "partial native label" in receipt["rules"]
    assert receipt["counts"]["entries_ending_in_deletion"] == 1
    with pytest.raises(TransferListError, match="new folder"):
        build_transfer_list(source, source_sha256=sha, source_url=url, retrieved="2026-11-01", out=out,
                            provenance={})


@pytest.mark.parametrize("kwargs, message", [
    ({"source_sha256": "0" * 64}, "does not hash"),
    ({"source_sha256": "abc"}, "64 hexadecimal"),
    ({"source_url": "http://ftp.expasy.org/enzyme.dat"}, "https URL"),
    ({"retrieved": "01-11-2026"}, "YYYY-MM-DD"),
    ({"retrieved": "2999-01-01"}, "future"),
])
def test_build_refuses_bad_arguments(tmp_path, kwargs, message):
    source, sha = _write(tmp_path, HEADER + RECORDS)
    arguments = {"source_sha256": sha, "source_url": "https://ftp.expasy.org/databases/enzyme/enzyme.dat",
                 "retrieved": "2026-11-01", "out": tmp_path / "out", "provenance": {}, **kwargs}
    with pytest.raises(TransferListError, match=message):
        build_transfer_list(source, **arguments)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("text, message", [
    ("ID   1.1.1\nDE   Short.\n//\n", "not one EC number"),
    ("ID   9.1.1.1\nDE   Out of range.\n//\n", "not one EC number"),
    ("ID   1.1.1.2\nDE   Transferred entry: see the 1.1.1.1 entry.\n//\n", "cannot be parsed"),
    ("ID   1.1.1.2\nDE   Transferred entry: 1.1.1.2.\n//\n", "names no other"),
    ("ID   1.1.1.2\nDE   Transferred entry: 8.1.1.1.\n//\n", "target"),
    ("ID   1.1.1.2\nDE   Transferred entry: 3.6.3.-.\n//\n", "cannot be parsed"),
    ("ID   1.1.1.2\nDE   Active.\n//\nID   1.1.1.2\nDE   Again.\n//\n", "two records"),
    ("ID   1.1.1.2\nDE   Active.\n", "no closing"),
    ("ID   1.1.1.2\nDE   Active.\n//\n//\n", "without an ID"),
    (HEADER, "no nomenclature record"),
])
def test_parser_is_strict(text, message):
    with pytest.raises(TransferListError, match=message):
        parse_enzyme_dat(text)


@pytest.mark.parametrize("text, message", [
    (_record("1.1.1.2", "Transferred entry: 1.1.1.3."), "has no record"),
    (_record("1.1.1.2", "Transferred entry: 1.1.1.3.") + _record("1.1.1.3", "Transferred entry: 1.1.1.2."),
     "cycle"),
])
def test_resolution_is_strict(text, message):
    records, _ = parse_enzyme_dat(text)
    with pytest.raises(TransferListError, match=message):
        resolve_transfers(records)
