"""Synthetic-only tests of offline evaluation-query preparation."""
from __future__ import annotations

import copy
import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metal_completion_evaluation as evaluation


@pytest.fixture
def provenance():
    value = evaluation.provenance_template()
    value["verification"] = {"status": "verified", "review_reference": "synthetic-only review",
                             "evidence_sha256": "a" * 64,
                             **{flag: True for flag in evaluation.COVERAGE_FLAGS}}
    for source in value["sources"]:
        source.update(status="verified", use_status="used", evidence_sha256="b" * 64,
                      pdb_ids=["1abc", "2def"])
    value["entries"] = [
        {"pdb_id": "1abc", "initial_release_date": "2020-01-01", "evidence_sha256": "c" * 64},
        {"pdb_id": "2def", "initial_release_date": "2021-06-02", "evidence_sha256": "d" * 64},
    ]
    return value


def test_query_uses_maximum_initial_date_all_elements_and_exclusions(provenance):
    cutoff, excluded = evaluation.validate_provenance(provenance, "2022-01-01")
    assert cutoff == "2021-06-02" and excluded == ["1ABC", "2DEF"]
    query = evaluation.query_payload(cutoff, "2022-01-01", excluded)
    nodes = {node["parameters"]["attribute"] + ":" + node["parameters"]["operator"]: node["parameters"]
             for node in query["query"]["nodes"]}
    assert nodes["rcsb_accession_info.initial_release_date:greater"]["value"] == cutoff
    assert nodes["rcsb_accession_info.initial_release_date:less_or_equal"]["value"] == "2022-01-01"
    assert nodes["rcsb_entry_info.structure_determination_methodology:exact_match"]["value"] == "experimental"
    assert nodes["rcsb_entry_info.polymer_entity_count_protein:greater"]["value"] == 0
    assert nodes["rcsb_entry_container_identifiers.entry_id:in"]["negation"] is True
    assert set(evaluation.ION_COMPONENTS.values()) == {"Mn", "Cu", "Zn", "Fe", "Co", "Ni"}
    assert "FES" not in nodes["pdbx_entity_nonpoly.comp_id:in"]["value"]


@pytest.mark.parametrize("change, message", [
    (lambda x: x["verification"].update(status="unverified"), "reviewed and verified"),
    (lambda x: x["verification"].update(all_previously_opened_test_ids_included=False), "coverage"),
    (lambda x: x["sources"].pop(), "families are missing"),
    (lambda x: x["sources"][0].update(status="unverified"), "Unverified source"),
    (lambda x: x["sources"][0].update(evidence_sha256=None), "SHA-256"),
    (lambda x: x["entries"].pop(), "every known/excluded"),
    (lambda x: x["entries"][0].update(initial_release_date=None), "release date"),
    (lambda x: x["entries"][0].update(initial_release_date="2020-02-30"), "invalid date"),
    (lambda x: x["entries"][0].update(metal="Zn"), "no labels or scores"),
    (lambda x: x["entries"].append(copy.deepcopy(x["entries"][0])), "Duplicate"),
])
def test_refuses_incomplete_or_nonmetadata_provenance(provenance, change, message):
    change(provenance)
    with pytest.raises(evaluation.PreparationError, match=message):
        evaluation.validate_provenance(provenance, "2022-01-01")


def test_no_invented_temporal_window_or_unreviewed_template(provenance):
    with pytest.raises(evaluation.PreparationError, match="temporal window"):
        evaluation.validate_provenance(provenance, "2021-06-02")
    with pytest.raises(evaluation.PreparationError, match="not be in the future"):
        evaluation.validate_provenance(provenance, "2999-01-01")
    assert {"clean30-original", "clean30-conservative", "care-task1-clusterres30",
            "care-task1-legacy30"} <= set(evaluation.REQUIRED_SOURCES)
    with pytest.raises(evaluation.PreparationError, match="intentionally blocked"):
        evaluation.validate_provenance(evaluation.provenance_template(), "2022-01-01")


def test_prepare_is_offline_hash_bound_and_does_not_overwrite(provenance, tmp_path, monkeypatch):
    def deny_network(*args, **kwargs):
        pytest.fail("offline preparation attempted network access")

    monkeypatch.setattr(socket, "socket", deny_network)
    source = tmp_path / "metadata.json"
    raw = evaluation.encoded(provenance)
    source.write_bytes(raw)
    output = tmp_path / "prepared"
    allowed_reads = {source, Path(evaluation.__file__), evaluation.ROOT /
                     "docs/campaigns/metal_completion_2026-10/evaluation.md"}
    original_read = Path.read_bytes

    def metadata_and_tool_only(path):
        assert path in allowed_reads, f"Unexpected input read: {path}"
        return original_read(path)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "read_bytes", metadata_and_tool_only)
        receipt = evaluation.prepare(source, evaluation.digest(raw), "2022-01-01", output)
    assert receipt["query_sha256"] == evaluation.digest((output / "query.json").read_bytes())
    assert receipt["prepared_on"] == evaluation.date.today().isoformat()
    assert receipt["candidate_query_executed"] is False
    assert receipt["heldout_access_authorized"] is False
    assert receipt["evaluation_authorized"] is False
    assert source.read_bytes() == raw == (output / "provenance.json").read_bytes()
    assert json.loads((output / "preparation.json").read_text()) == receipt
    with pytest.raises(evaluation.PreparationError, match="already exists"):
        evaluation.prepare(source, evaluation.digest(raw), "2022-01-01", output)


def test_bad_pin_does_not_create_outputs(provenance, tmp_path):
    source = tmp_path / "metadata.json"
    source.write_bytes(evaluation.encoded(provenance))
    with pytest.raises(evaluation.PreparationError, match="mismatch"):
        evaluation.prepare(source, "0" * 64, "2022-01-01", tmp_path / "output")
    assert not (tmp_path / "output").exists()
