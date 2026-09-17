"""Scientific boundary tests for development sequence-remoteness annotation."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_sequence_remoteness import (
    DEFAULT_PROTOCOL, alignment_metrics, annotate_task, collect_sequences, digest, identity_bin,
    saved_membership, sha256, support_counts, verify_search_receipt, write_frozen,
)


def record(identifier, group, sequence="A" * 100):
    import hashlib
    return dict(sequence_id=identifier, group_id=group, structure_id=identifier,
                sequence=sequence, sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                status="resolved", scope="represented_coordinate_chain", task="metal")


def example(identifier, group, target=0):
    return dict(example_id=identifier + "_pocket", structure_id=identifier,
                group_id=group, target=target)


def hit(q, t, identity, coverage=1.0, pairs=100, evalue=1e-20):
    return dict(query=q, target=t, identity=identity, shorter_coverage=coverage,
                residue_pairs=pairs, evalue=evalue)


def test_threshold_boundaries_are_exact_and_no_hit_is_not_zero():
    assert [identity_bin(x) for x in (.31, .30, .20, .15, 0, None)] == [
        ">30", "(20,30]", "(15,20]", "<=15", "<=15", None]
    for value in (float("nan"), -0.1, 1.1):
        with pytest.raises(ValueError):
            identity_bin(value)


def test_identity_uses_all_columns_and_shorter_sequence_coverage():
    # One 50-aa sequence aligns against 55 positions in a longer sequence:
    # five insertions count in the identity denominator, not residue pairs.
    q, t = record("q", "Q", "A" * 50), record("t", "T", "A" * 100)
    row = dict(query="q", target="t", nident="50", alnlen="55", qstart="1", qend="50",
               qlen="50", tstart="1", tend="55", tlen="100", evalue="1e-10",
               bits="100", qaln="A" * 25 + "-" * 5 + "A" * 25, taln="A" * 55)
    metrics = alignment_metrics(row, {"q": q, "t": t})
    assert metrics["identity"] == 50 / 55
    assert metrics["residue_pairs"] == 50
    assert metrics["shorter_coverage"] == 1
    broken = {**row, "nident": "51"}
    with pytest.raises(ValueError, match="nident"):
        alignment_metrics(broken, {"q": q, "t": t})


def test_alignment_is_bound_to_frozen_sequences():
    q, t = record("q", "Q"), record("t", "T")
    row = dict(query="q", target="t", nident="99", alnlen="100", qstart="1", qend="100",
               qlen="100", tstart="1", tend="100", tlen="100", evalue="1e-10",
               bits="100", qaln="A" * 100, taln="C" + "A" * 99)
    with pytest.raises(ValueError, match="frozen input"):
        alignment_metrics(row, {"q": q, "t": t})


def test_serialized_mmseqs_nident_rounding_cannot_shift_remote_boundary():
    q, t = record("q", "Q"), record("t", "T", "A" * 16 + "C" * 84)
    row = dict(query="q", target="t", nident="15", alnlen="100", qstart="1", qend="100",
               qlen="100", tstart="1", tend="100", tlen="100", evalue="1e-10",
               bits="100", qaln=q["sequence"], taln=t["sequence"])
    result = alignment_metrics(row, {"q": q, "t": t})
    assert result["identity"] == .16
    assert identity_bin(result["identity"]) == "(15,20]"
    assert result["nident_recount_minus_export"] == 1
    row["nident"] = "14"
    with pytest.raises(ValueError, match="verified rounding"):
        alignment_metrics(row, {"q": q, "t": t})


def test_ambiguous_identical_symbols_fail_closed():
    q, t = record("q", "Q", "X" * 100), record("t", "T", "X" * 100)
    row = dict(query="q", target="t", nident="100", alnlen="100", qstart="1", qend="100",
               qlen="100", tstart="1", tend="100", tlen="100", evalue="1e-10",
               bits="100", qaln=q["sequence"], taln=t["sequence"])
    with pytest.raises(ValueError, match="Ambiguous identical"):
        alignment_metrics(row, {"q": q, "t": t})


def test_max_uses_exact_run_training_and_every_validation_group_chain():
    records = [record("train", "T"), record("other", "O", "C" * 100),
               record("va", "V", "D" * 100), record("vb", "V", "E" * 100)]
    memberships = [dict(task="metal", run_id="fold_a", train=[example("train", "T")],
                        val=[example("va", "V"), example("vb", "V")])]
    rows = annotate_task(records, memberships, [hit("va", "other", .99),
                         hit("va", "train", .14), hit("train", "vb", .25)], DEFAULT_PROTOCOL)
    assert {r["max_identity"] for r in rows} == {.25}
    assert {r["bin"] for r in rows} == {"(20,30]"}
    assert len({r["component_id"] for r in rows}) == 1


def test_ineligible_hits_are_excluded_and_domain_audit_is_separate():
    records = [record("t", "T"), record("v", "V", "C" * 100)]
    membership = dict(task="metal", run_id="r", train=[example("t", "T")], val=[example("v", "V")])
    row = annotate_task(records, [membership], [hit("v", "t", .99, pairs=49),
                        hit("v", "t", .98, evalue=.01), hit("v", "t", .5, coverage=.6)],
                        DEFAULT_PROTOCOL)[0]
    assert row["status"] == "no_qualifying_hit" and row["bin"] is None
    assert row["max_identity"] is None
    assert row["audit50_max_identity"] == .5


def test_unresolved_training_sequence_cannot_certify_remote_bin():
    records = [record("t", "T"), record("v", "V", "C" * 100),
               dict(structure_id="missing", group_id="M", status="missing_sequence", scope="represented_coordinate_chain")]
    membership = dict(task="metal", run_id="r", train=[example("t", "T"), example("missing", "M")],
                      val=[example("v", "V")])
    row = annotate_task(records, [membership], [hit("v", "t", .14)], DEFAULT_PROTOCOL)[0]
    assert row["status"] == "incomplete_training_sequences"
    assert row["max_identity"] == .14 and row["bin"] is None


def test_components_use_transitive_domain_links_and_same_group():
    records = [record("a", "A"), record("b", "B", "C" * 100), record("c", "C", "D" * 100)]
    membership = dict(task="metal", run_id="r", train=[], val=[example("a", "A"), example("c", "C")])
    rows = annotate_task(records, [membership], [hit("a", "b", .3, coverage=.5),
                        hit("b", "c", .3, coverage=.5)], DEFAULT_PROTOCOL)
    assert rows[0]["component_id"] == rows[1]["component_id"]


def test_support_preserves_missing_classes_and_counts_components_once():
    rows = [dict(task="metal", run_id="r", target=0, bin="<=15", group_id=str(i), component_id="same")
            for i in range(15)]
    support = support_counts(rows, DEFAULT_PROTOCOL)
    low = next(r for r in support if r["bin"] == "<=20")
    assert low["classes"][0] == dict(target=0, pockets=15, groups=15, components=1)
    assert len(low["classes"]) == 4 and low["classes"][1]["components"] == 0
    assert not low["support_gate_passed"]


def saved(examples):
    return {"examples": examples, "n_examples": len(examples), "ordered_examples_sha256": digest(examples)}


def test_saved_membership_rejects_digest_changes_and_group_leakage():
    tr = dict(structure_id="tr", pocket_id="tr_p", group="T", y_metal=0)
    va = dict(structure_id="va", pocket_id="va_p", group="V", y_metal=1)
    value = {"retained_split_identity": {"train": saved([tr]), "validation": saved([va])}}
    assert saved_membership(value, "metal", "r")["val"][0]["target"] == 1
    broken = copy.deepcopy(value)
    broken["retained_split_identity"]["validation"]["examples"][0]["y_metal"] = 2
    with pytest.raises(ValueError, match="digest"):
        saved_membership(broken, "metal", "r")
    value["retained_split_identity"]["validation"] = saved([{**va, "group": "T"}])
    with pytest.raises(ValueError, match="overlap"):
        saved_membership(value, "metal", "r")


def test_frozen_artifacts_are_idempotent_and_refuse_changes(tmp_path):
    path = tmp_path / "counts.json"
    write_frozen(path, {"a": 1})
    write_frozen(path, {"a": 1})
    with pytest.raises(ValueError, match="replace"):
        write_frozen(path, {"a": 2})


def sequence_fixture(tmp_path):
    import hashlib
    structure = "Q12345__chain_A__EC_1.1.1.1"
    esm = tmp_path / "esm"
    esm.mkdir()
    sidecar = dict(structure_id=structure, chain_id="A", source_sequence="ACDEF",
                   source_sequence_sha256=hashlib.sha256(b"ACDEF").hexdigest())
    (esm / (structure + "_chain_A_esmc.pt.json")).write_text(json.dumps(sidecar))
    care = tmp_path / "care.csv"
    care.write_text("split,source_split,protein_id,sequence\ntrain,train,Q12345,ACDEFG\n")
    ledger = dict(sequence_sources=dict(esm_embeddings_dir=str(esm), care_source_csv=str(care)))
    members = [dict(task="ec", run_id="r", train=[], val=[example(structure, "q12345")])]
    return ledger, members


def test_lowercase_accession_maps_to_care_without_hiding_discrepancy(tmp_path):
    ledger, members = sequence_fixture(tmp_path)
    row = collect_sequences(ledger, tmp_path, members, "represented_coordinate_chain")[0]
    assert row["care_protein_id"] == "Q12345"
    assert row["group_id"] == "q12345"
    assert row["care_relationship"] == "coordinate_subsequence"
    assert row["sequence"] == "ACDEF" and row["status"] == "resolved"
    full = collect_sequences(ledger, tmp_path, members, "full_protein")[0]
    assert full["sequence"] == "ACDEFG" and full["status"] == "unresolved_mapping"


def test_reviewed_override_cannot_change_saved_group(tmp_path):
    ledger, members = sequence_fixture(tmp_path)
    source = tmp_path / "review.json"
    source.write_text("{}")
    override = dict(task="ec", structure_id=members[0]["val"][0]["structure_id"], chain_id="A",
                    group_id="different", source_path=str(source), source_sha256=sha256(source),
                    sequence="ACDEFG", scope="full_protein", site_chain_coverage_certified=True,
                    mapping_evidence="Reviewed accession mapping")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"sequences": [override]}))
    with pytest.raises(ValueError, match="membership identifiers"):
        collect_sequences(ledger, tmp_path, members, "full_protein", overrides)


def test_search_receipt_requires_uncapped_pinned_search(tmp_path):
    root = tmp_path / "metal"
    root.mkdir()
    fasta = root / "development.fasta"
    fasta.write_text(">a\nAAAA\n>b\nCCCC\n")
    align = root / "alignments.tsv"
    align.write_text("\t".join("query target nident alnlen qstart qend qlen tstart tend tlen evalue bits qaln taln".split()) + "\n")
    command = ["mmseqs", "easy-search", str(fasta), str(fasta), str(align), "tmp"]
    flags = {"-s": "7.5", "-e": ".001", "--alignment-mode": "3", "--seq-id-mode": "0",
             "-a": "1", "--min-seq-id": "0", "--min-aln-len": "50", "-c": "0",
             "--cov-mode": "0", "--mask": "1", "--comp-bias-corr": "1", "--search-type": "1",
             "--max-seqs": "2", "--max-accept": "2", "--max-rejected": "2"}
    for key, value in flags.items():
        command.extend([key, value])
    receipt = dict(program="MMseqs2", release="18-8cc5c", version=DEFAULT_PROTOCOL["version"],
                   binary_sha256="a" * 64, fasta_sha256=sha256(fasta), alignments_sha256=sha256(align),
                   command=command, database_convention=DEFAULT_PROTOCOL["database_convention"])
    path = root / "search_receipt.json"
    path.write_text(json.dumps(receipt))
    manifest = {"fasta_files": {"metal": dict(path="metal/development.fasta", sha256=sha256(fasta), n_sequences=2)}}
    assert verify_search_receipt(tmp_path, "metal", manifest, DEFAULT_PROTOCOL)[0] == align
    command[command.index("--max-seqs") + 1] = "1"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="cap"):
        verify_search_receipt(tmp_path, "metal", manifest, DEFAULT_PROTOCOL)
