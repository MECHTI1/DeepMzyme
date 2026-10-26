"""R2 and R2-inner lists of the EC class vs CLEAN30 campaign (synthetic identity folders only; no dataset access)."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]

import run_clean_split30_comparator as comparator  # noqa: E402
from ec_clean30 import cohort as cohort_audit  # noqa: E402
from ec_clean30 import identity, r2_lists  # noqa: E402
from ec_clean30.identity import (  # noqa: E402
    assign_inner, build_edges, build_inner_components, build_inputs, build_r1_inner_exclusions, certify_inner,
    read_tsv, run_search, sha256_file,
)
from ec_clean30.provenance import DirtyTreeError, HeldOutInputError  # noqa: E402
from ec_clean30.r2_lists import (  # noqa: E402
    DIFFERENCE_COLUMNS, R2ListError, build_r2_lists, fold_work_dirs, parse_fold_work, read_allowlist, verify_r2_lists,
)
from test_ec_clean30_identity import FakeMMseqs, fake_binary, mutate, random_sequence  # noqa: E402
from training.ec_membership import read_ec_fold_membership, write_ec_fold_membership  # noqa: E402

EC = {"1": "1.1.1.1", "2": "2.7.1.1", "3": "3.4.24.17"}
ELIGIBLE = {f"E{index:04d}": str(1 + index % 3) for index in range(1, 21)}
BRIDGE = "S0001"  # several EC classes, D3-linked to eligible proteins: follows their component
LONE = "S0002"  # several EC classes, no link: its component holds no eligible protein
ISOLATED = "E0008"  # an eligible protein without links (in folds 0, 1, 3 and 4)
LINKED = "E0002"  # an eligible protein D3-linked to E0005 (both in folds 0, 2 and 3)
SEVERAL_EC = {BRIDGE: "1.1.1.1;2.7.1.1", LONE: "3.4.24.17;5.3.1.1"}
NON_METALLO = ("N0001", "N0002")
# Outer test fold of every cohort protein: each protein is a training protein of the other four folds.
TEST_FOLD = {accession: index % 5 for index, accession in enumerate(sorted(ELIGIBLE))} | {BRIDGE: 4, LONE: 3}
D3_PAIRS = [("E0001", BRIDGE), (BRIDGE, "E0004"), ("E0002", "E0005"), ("E0003", "N0001"), ("E0007", "N0002"),
            ("E0009", "E0010")]
HITS = {frozenset(pair): "1.000E-20" for pair in D3_PAIRS}
FAMILIES = [("E0001", BRIDGE, "E0004"), ("E0002", "E0005"), ("E0003", "N0001"), ("E0007", "N0002"),
            ("E0009", "E0010")]
PROVENANCE = {"source": "synthetic test"}
SMOKE_PROVENANCE = {"source": "synthetic test", "smoke": True}


def ec_text(accession):
    return SEVERAL_EC.get(accession) or EC[ELIGIBLE[accession]]


def training_proteins(fold):
    return sorted(accession for accession, test_fold in TEST_FOLD.items() if test_fold != fold)


def world_sequences():
    rng = random.Random(5)
    sequences = {accession: random_sequence(rng, 120) for accession in [*sorted(TEST_FOLD), *NON_METALLO]}
    for family in FAMILIES:
        base = random_sequence(rng, 120)
        sequences.update({accession: mutate(base, 24, rng) for accession in family})
    return sequences


def write_world_data(root):
    """Five official training files (metallo plus non-metallo proteins) and five metallo training files."""
    sequences = world_sequences()
    official_dir, metallo_dir = root / "split30", root / "metallo"
    official_dir.mkdir(parents=True)
    metallo_dir.mkdir(parents=True)
    for fold in range(5):
        proteins = training_proteins(fold)
        with (official_dir / f"split30_train_split_{fold}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["Entry", "EC number", "Sequence"])
            writer.writerows((accession, ec_text(accession) if accession in TEST_FOLD else "4.2.1.1",
                              sequences[accession]) for accession in [*proteins, *NON_METALLO])
        with (metallo_dir / f"CLEAN_30_train_test_split_{fold}_train.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["structure", "chain_resi", "metaltype", "ecnumber", "uniprot_id", identity.DONOR_COLUMN])
            writer.writerows((f"{accession}__chain_A__EC_{ec_text(accession)}", "A_1", "ZN", ec_text(accession),
                              accession, "") for accession in proteins)
    return official_dir, metallo_dir


def write_cohort(cohort, exclusions=None, *, status="passed", failed_checks=(), smoke=False):
    """A cohort folder in the audit's own format: a receipt, and an exclusions table for each fold given.

    ``exclusions`` maps a fold to {accession: load status} or to a list of (accession, load status).
    """
    cohort.mkdir(parents=True, exist_ok=True)
    for name in [r2_lists.COHORT_EXCLUSIONS_NAME.format(fold=fold) for fold in range(5)] + ["quarantine.tsv"]:
        (cohort / name).unlink(missing_ok=True)
    outputs, per_fold = {}, {}
    for fold in range(5):
        given = (exclusions or {}).get(fold) or {}
        rows = list(given.items()) if isinstance(given, dict) else list(given)
        if rows:
            name = cohort_audit.FOLD_EXCLUSIONS_NAME.format(fold=fold)
            (cohort / name).write_bytes(cohort_audit._tsv_bytes(cohort_audit.EXCLUSION_COLUMNS, [
                {"accession": accession, "chain": "A", "load_status": load_status,
                 "load_reason": "synthetic\treason, \"quoted\""} for accession, load_status in rows]))
            outputs[name] = {"sha256": sha256_file(cohort / name), "n_rows": len(rows)}
        per_fold[str(fold)] = {"file": f"cohort_counts_fold{fold}.json", "no_load_time_exclusions": not rows}
    receipt = {"format": cohort_audit.COHORT_AUDIT_FORMAT, "schema_version": 1, "kind": "cohort_audit",
               "smoke": smoke, "status": status, "failed_checks": list(failed_checks), "per_fold": per_fold,
               "outputs": outputs}
    (cohort / r2_lists.COHORT_RECEIPT_NAME).write_text(json.dumps(receipt, indent=2))
    return cohort


def edit_cohort_receipt(world, edit):
    path = world.cohort / r2_lists.COHORT_RECEIPT_NAME
    receipt = json.loads(path.read_text())
    edit(receipt)
    path.write_text(json.dumps(receipt))


@dataclass
class World:
    root: Path
    identity_root: Path
    cohort: Path
    metallo_dir: Path

    def work(self, fold):
        return self.identity_root / f"fold{fold}"


def build_world(root):
    official_dir, metallo_dir = write_world_data(root / "data")
    binary = fake_binary(root / "fakebin")
    fake = FakeMMseqs(hits=HITS)
    identity_root = root / "identity"
    for fold in range(5):
        work = identity_root / f"fold{fold}"
        build_inputs(work, fold, official_split_dir=official_dir, metallo_folds_dir=metallo_dir)
        run_search(work, fold, binary=binary, expected_sha256=sha256_file(binary), runner=fake)
        build_edges(work, fold)
        build_inner_components(work, fold)
        assign_inner(work, fold, metallo_folds_dir=metallo_dir)
        certify_inner(work, fold, metallo_folds_dir=metallo_dir)
    for inner in range(4):
        build_r1_inner_exclusions(identity_root / "fold0", 0, inner)
    return World(root, identity_root, write_cohort(root / "cohort"), metallo_dir)


@pytest.fixture(scope="module")
def world_template(tmp_path_factory):
    # Neutral directory names: the comparator refuses any path part such as "test_...".
    root = tmp_path_factory.mktemp("r2world")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(identity, "MMSEQS_BINARY_SHA256", sha256_file(fake_binary(root / "fakebin")))
        return build_world(root)


@pytest.fixture
def world(world_template, tmp_path_factory, monkeypatch):
    """A private copy of the synthetic world; the fake binary counts as the pinned build."""
    root = tmp_path_factory.mktemp("r2copy") / "world"
    shutil.copytree(world_template.root, root)
    monkeypatch.setattr(identity, "MMSEQS_BINARY_SHA256", sha256_file(root / "fakebin" / "mmseqs"))
    return World(root, root / "identity", root / "cohort", root / "data" / "metallo")


def build(world, out_name="r2_lists", **kwargs):
    kwargs.setdefault("provenance", SMOKE_PROVENANCE if kwargs.get("smoke") else PROVENANCE)
    return build_r2_lists(world.identity_root, world.cohort, world.root / out_name, **kwargs)


def membership_of(work, fold):
    receipt = json.loads((work / identity.INNER_ASSIGNMENT_RECEIPT).read_text())
    output = receipt["outputs"]["membership"]
    return read_ec_fold_membership(work / output["file"], output["sha256"]), receipt


def components_of(work):
    return {row["accession"]: row["component_id"] for row in read_tsv(work / "components_inner.tsv",
                                                                       identity.COMPONENT_COLUMNS)}


def r1_list(work, inner):
    return {row["accession"]: row["reason"]
            for row in read_tsv(work / f"r1_inner_exclusions_fold0_inner{inner}.tsv", identity.R1_EXCLUSION_COLUMNS)}


def read_lines(path):
    return path.read_text().splitlines()


def rebind_assignment(work, receipt):
    """Write an edited assignment receipt and rebind the certificate to it."""
    (work / identity.INNER_ASSIGNMENT_RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    certificate_path = work / identity.INNER_CERTIFICATE
    certificate = json.loads(certificate_path.read_text())
    certificate["assignment_receipt_sha256"] = sha256_file(work / identity.INNER_ASSIGNMENT_RECEIPT)
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")


def rewrite_membership(world, fold, transform):
    """Rewrite one fold's membership rows with ``transform`` and rebind its receipts (a rerun with other rows)."""
    work = world.work(fold)
    membership, receipt = membership_of(work, fold)
    output = receipt["outputs"]["membership"]
    rows = transform([dict(row) for row in membership.rows])
    (work / output["file"]).unlink()
    output["sha256"] = write_ec_fold_membership(rows, work / output["file"])
    rewritten = read_ec_fold_membership(work / output["file"], output["sha256"])
    certificate_path = work / identity.INNER_CERTIFICATE
    certificate = json.loads(certificate_path.read_text())
    certificate["membership"].update(sha256=rewritten.file_sha256, identity_sha256=rewritten.identity_sha256)
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    rebind_assignment(work, receipt)
    if fold == 0:
        for inner in range(4):
            (work / f"r1_inner_exclusions_fold0_inner{inner}.tsv").unlink()
            (work / f"r1_inner_exclusions_receipt_fold0_inner{inner}.json").unlink()
            build_r1_inner_exclusions(work, 0, inner)


def drop_from_membership(world, fold, accession):
    """Rewrite one fold's membership without ``accession`` (a rerun that excluded it)."""
    def drop(rows):
        kept = [row for row in rows if row["accession"] != accession]
        assert len(kept) == len(rows) - 1
        return kept
    rewrite_membership(world, fold, drop)


# ---------------------------------------------------------------- the lists

def test_r2_lists_follow_the_identity_chain(world):
    build(world)
    out = world.root / "r2_lists"
    for fold in range(5):
        work = world.work(fold)
        r2 = read_lines(out / f"r2_allowlist_fold{fold}.txt")
        assert r2 == training_proteins(fold)
        accessions, record = comparator.read_accession_allowlist(out / f"r2_allowlist_fold{fold}.txt", [])
        assert accessions == r2 and record["n_accessions"] == len(r2)
        membership, _ = membership_of(work, fold)
        deepmzyme = {row["accession"] for row in membership.rows}
        assert deepmzyme < set(r2)
        table = read_tsv(out / f"r2_vs_deepmzyme_fold{fold}.tsv", DIFFERENCE_COLUMNS)
        several = sorted(set(SEVERAL_EC) & set(r2))
        assert [row["accession"] for row in table] == several == sorted(set(r2) - deepmzyme)
        assert all(row["in_deepmzyme"] == "0" and row["reason"] == "several_ec_classes" for row in table)
    assert BRIDGE in read_lines(out / "r2_allowlist_fold0.txt") and LONE in read_lines(out / "r2_allowlist_fold0.txt")
    for path in out.iterdir():
        text = path.read_text()
        assert "__EC_" not in text and not any(number in text for number in (*EC.values(), "5.3.1.1")), path.name


def test_r2_inner_lists_follow_components_and_avoid_the_r1_inner_exclusions(world):
    summary = build(world)
    out, work = world.root / "r2_lists", world.work(0)
    membership, _ = membership_of(work, 0)
    fold_of_group = {row["group_id"]: int(row["fold"]) for row in membership.rows}
    components = components_of(work)
    bridge_fold = fold_of_group[components[BRIDGE]]
    assert components[BRIDGE] == components["E0004"] and components[LONE] not in fold_of_group
    for inner in range(4):
        path = out / f"r2_inner_allowlist_fold0_inner{inner}.txt"
        listed = read_lines(path)
        expected = sorted(accession for accession in training_proteins(0)
                          if fold_of_group.get(components[accession]) != inner)
        assert listed == expected
        assert LONE in listed  # a component without eligible proteins is in every list
        assert (BRIDGE in listed) == (inner != bridge_fold)  # the several-class bridge follows its component
        assert not set(listed) & set(r1_list(work, inner))
        assert comparator.read_accession_allowlist(path, [])[0] == listed
        assert summary["r2_inner_allowlists"]["0"][str(inner)]["n_accessions"] == len(listed)
    assert set(summary["r2_inner_allowlists"]) == {"0"}
    assert not list(out.glob("r2_inner_allowlist_fold1_*"))


def test_receipt_binds_inputs_and_outputs_and_verifies(world):
    summary = build(world)
    out = world.root / "r2_lists"
    receipt = json.loads((out / r2_lists.RECEIPT_NAME).read_text())
    assert summary["receipt_sha256"] == sha256_file(out / r2_lists.RECEIPT_NAME)
    for fold in range(5):
        work = world.work(fold)
        inputs = receipt["folds"][str(fold)]["inputs"]
        fasta = json.loads((work / "fasta_receipt_metallo_train.json").read_text())
        assert inputs["nodes_sha256"] == fasta["outputs"]["nodes"]["sha256"]
        assert inputs["certificate_sha256"] == sha256_file(work / identity.INNER_CERTIFICATE)
        assert inputs["membership_sha256"] == sha256_file(work / f"inner_membership_fold{fold}.csv")
        assert inputs["components_sha256"] == sha256_file(work / "components_inner.tsv")
        assert inputs["pinned_binary"] is True
        assert receipt["folds"][str(fold)]["work_override"] is False
        assert receipt["folds"][str(fold)]["counts"]["r2_proteins"] == len(training_proteins(fold))
    for inner in range(4):
        record = receipt["r2_inner"]["0"][str(inner)]["r1_inner_exclusions"]
        assert record["sha256"] == sha256_file(world.work(0) / record["file"])
    for name, output in receipt["outputs"].items():
        assert output["sha256"] == sha256_file(out / name)
    assert receipt["provenance"] == PROVENANCE and receipt["ec_column_written"] is False and receipt["smoke"] is False
    assert receipt["cohort"]["receipt_sha256"] == sha256_file(world.cohort / r2_lists.COHORT_RECEIPT_NAME)
    assert (receipt["cohort"]["status"], receipt["cohort"]["failed_checks"], receipt["cohort"]["smoke"]) == (
        "passed", [], False)
    assert verify_r2_lists(out)["status"] == "verified"
    (out / "r2_allowlist_fold3.txt").write_text((out / "r2_allowlist_fold3.txt").read_text() + "Z9999\n")
    with pytest.raises(R2ListError, match="r2_allowlist_fold3.txt changed"):
        verify_r2_lists(out)


def _numbers(value):
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, list):
        return [number for item in value for number in _numbers(item)]
    return []


def test_printed_summary_shows_counts_for_fold_0_only(world):
    """R2 size minus a fold's eligible proteins is its several-class count, so only one fold shows sizes."""
    summary = build(world)
    text = json.dumps(summary)
    for accession in [*TEST_FOLD, *NON_METALLO]:
        assert accession not in text, accession
    membership, assignment = membership_of(world.work(0), 0)
    fold0 = summary["r2_allowlists"]["0"]
    assert fold0 == {"file": "r2_allowlist_fold0.txt", "n_accessions": len(training_proteins(0)),
                     "deepmzyme_proteins": len(membership.rows), "not_in_deepmzyme": 2,
                     "not_in_deepmzyme_by_reason": {"several_ec_classes": 2}}
    assert fold0["n_accessions"] - assignment["counts"]["eligible_proteins"] == \
        assignment["counts"]["excluded_ambiguous_ec_class"]
    for fold in range(1, 5):
        assert summary["r2_allowlists"][str(fold)] == {"file": f"r2_allowlist_fold{fold}.txt",
                                                       "size_in_receipt": True}
    others = {key: value for key, value in summary.items() if key not in ("r2_allowlists", "r2_inner_allowlists")}
    assert _numbers(others) == [0]  # the overlap check only
    assert set(summary["r2_inner_allowlists"]) == set(map(str, r2_lists.R2_INNER_OUTER_FOLDS)) == {"0"}
    assert summary["smoke"] is False and summary["cohort_audit_status"] == "passed"


def test_builds_are_deterministic(world):
    build(world, "first")
    build(world, "second")
    for path in sorted((world.root / "first").iterdir()):
        assert path.read_bytes() == (world.root / "second" / path.name).read_bytes(), path.name


def test_receipt_records_absolute_paths_and_the_folder_is_readable(world, monkeypatch):
    monkeypatch.chdir(world.root)
    build_r2_lists(Path("identity"), Path("cohort"), Path("rel_out"), provenance=PROVENANCE)
    out = world.root / "rel_out"
    receipt = json.loads((out / r2_lists.RECEIPT_NAME).read_text())
    assert receipt["identity_root"] == str(world.identity_root) and "out" not in receipt
    assert receipt["cohort"]["path"] == str(world.cohort)
    assert all(receipt["folds"][str(fold)]["work"] == str(world.work(fold)) for fold in range(5))
    assert (out.stat().st_mode & 0o777) == 0o755


# ---------------------------------------------------------------- refusals

def _rebind_r1_output(work, inner, text):
    name = f"r1_inner_exclusions_fold0_inner{inner}.tsv"
    (work / name).write_text(text)
    receipt_path = work / f"r1_inner_exclusions_receipt_fold0_inner{inner}.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"][name]["sha256"] = sha256_file(work / name)
    receipt_path.write_text(json.dumps(receipt))


def test_an_overlap_with_the_r1_inner_exclusions_is_refused_not_subtracted(world):
    work = world.work(0)
    name = "r1_inner_exclusions_fold0_inner2.tsv"
    _rebind_r1_output(work, 2, (work / name).read_text() + f"{LONE}\td3_neighbour\n")
    with pytest.raises(R2ListError, match=r"inner fold 2: the R2-inner list would share 1 accession\(s\)") as refused:
        build(world)
    assert LONE not in str(refused.value)
    assert sorted(path.name for path in world.root.iterdir()) == ["cohort", "data", "fakebin", "identity"]


def test_r1_inner_lists_are_read_through_their_receipts(world):
    work = world.work(0)
    name = "r1_inner_exclusions_fold0_inner1.tsv"
    original = (work / name).read_text()
    (work / name).write_text(original + "N0002\td3_neighbour\n")
    with pytest.raises(R2ListError, match="inner1.tsv is missing or changed"):
        build(world)
    lines = original.splitlines()
    validation = next(line for line in lines[1:] if line.endswith("inner_validation"))
    _rebind_r1_output(work, 1, "\n".join(line for line in lines if line != validation) + "\n")
    with pytest.raises(R2ListError, match="other inner-validation proteins than the certified membership"):
        build(world)
    _rebind_r1_output(work, 1, original + "N0002\tother_reason\n")
    with pytest.raises(R2ListError, match="Unknown reason or repeated accession"):
        build(world)
    _rebind_r1_output(work, 1, original)
    (work / "r1_inner_exclusions_receipt_fold0_inner3.json").unlink()
    with pytest.raises(R2ListError, match="Run r1-inner-exclusions for fold 0, inner fold 3 first"):
        build(world)


@pytest.mark.parametrize("edit, message", [
    (lambda receipt: receipt.update(step="components"), "belongs to another fold or inner fold"),
    (lambda receipt: receipt.update(fold=1), "belongs to another fold or inner fold"),
    (lambda receipt: receipt.update(inner_fold=0), "belongs to another fold or inner fold"),
    (lambda receipt: receipt.update(freeze1_sha256="a" * 64), "another Freeze 1 value"),
    (lambda receipt: receipt.update(pinned_binary=False), "comes from an unpinned search"),
    (lambda receipt: receipt["inputs"].update(edges_sha256="0" * 64), "built from other identity tables"),
    (lambda receipt: receipt["inputs"].update(membership_sha256="0" * 64), "another certified membership"),
    (lambda receipt: receipt["inputs"].update(certificate_sha256="0" * 64), "another certified membership"),
    (lambda receipt: receipt["outputs"].clear(), "does not name"),
], ids=["step", "fold", "inner_fold", "freeze1", "pinned", "chain", "membership", "certificate", "outputs"])
def test_r1_inner_receipts_are_bound_to_the_fold_chain(world, edit, message):
    path = world.work(0) / "r1_inner_exclusions_receipt_fold0_inner1.json"
    receipt = json.loads(path.read_text())
    edit(receipt)
    path.write_text(json.dumps(receipt))
    with pytest.raises(R2ListError, match=message):
        build(world)


@pytest.mark.parametrize("name", ["components_inner.tsv", "inner_membership_fold2.csv", "nodes_metallo_train.tsv",
                                  identity.INNER_ASSIGNMENT_RECEIPT])
def test_changed_identity_tables_are_refused(world, name):
    path = world.work(2) / name
    path.write_text(path.read_text() + "\n")
    with pytest.raises(R2ListError, match="Fold 2"):
        build(world)


@pytest.mark.parametrize("edit, message", [
    (lambda receipt: receipt["inputs"].update(nodes_sha256="0" * 64), "was built from other identity tables"),
    (lambda receipt: receipt["inputs"].update(components_receipt_sha256="0" * 64), "another components step"),
    (lambda receipt: receipt["excluded_ambiguous_ec_class_accessions"].append(BRIDGE), "no valid several-class"),
    (lambda receipt: receipt.update(excluded_ambiguous_ec_class_accessions=None), "no valid several-class"),
    (lambda receipt: receipt["excluded_ambiguous_ec_class_accessions"].append("Z9999"),
     "1 several-class accession\\(s\\) of the assignment receipt are not metallo training"),
    (lambda receipt: receipt["excluded_ambiguous_ec_class_accessions"].append(ISOLATED),
     "1 several-class protein\\(s\\) are in the membership"),
], ids=["chain", "components", "repeated", "missing", "outside", "in_membership"])
def test_the_assignment_receipt_is_checked_against_the_chain(world, edit, message):
    work = world.work(1)
    receipt = json.loads((work / identity.INNER_ASSIGNMENT_RECEIPT).read_text())
    assert BRIDGE in receipt["excluded_ambiguous_ec_class_accessions"]
    edit(receipt)
    rebind_assignment(work, receipt)
    with pytest.raises(R2ListError, match=f"Fold 1: .*{message}"):
        build(world)


@pytest.mark.parametrize("transform, message", [
    (lambda rows: [{**row, "fold": "2" if row["fold"] == "3" else row["fold"]} for row in rows],
     "an inner fold has no protein"),
    (lambda rows: [{**row, "group_id": "other_group"} if row["accession"] == ISOLATED else row for row in rows],
     "a membership group differs from the protein's inner component"),
], ids=["empty_inner_fold", "group"])
def test_membership_rows_must_match_the_components(world, transform, message):
    rewrite_membership(world, 1, transform)
    with pytest.raises(R2ListError, match=f"Fold 1: {message}"):
        build(world)


def test_identity_inputs_need_the_pinned_search(world, monkeypatch):
    monkeypatch.setattr(identity, "MMSEQS_BINARY_SHA256", "0" * 64)
    with pytest.raises(R2ListError, match="unpinned MMseqs2 binary"):
        build(world)


def test_an_unexplained_difference_is_refused(world):
    drop_from_membership(world, 1, ISOLATED)
    with pytest.raises(R2ListError, match="Fold 1: 1 metallo training protein\\(s\\) are neither") as refused:
        build(world)
    assert ISOLATED not in str(refused.value)


@pytest.mark.parametrize("name", r2_lists.complement_output_names(3))
def test_complement_outputs_in_an_identity_folder_are_refused(world, name):
    path = world.work(3) / name
    if name.startswith("search_"):
        path.mkdir()
    else:
        path.write_text("{}")
    with pytest.raises(R2ListError, match="holds complement \\(test-side\\) outputs"):
        build(world)


def test_a_dangling_complement_link_is_refused(world):
    (world.work(3) / "strata_fold3.tsv").symlink_to(world.root / "absent")
    with pytest.raises(R2ListError, match="holds complement \\(test-side\\) outputs"):
        build(world)


@pytest.mark.parametrize("which", ["identity_root", "cohort", "fold_work", "out"])
def test_test_named_inputs_and_outputs_are_refused(world, which):
    if which == "identity_root":
        target = world.root / "identity_test"
        world.identity_root.rename(target)
        call = lambda: build_r2_lists(target, world.cohort, world.root / "o", provenance=PROVENANCE)  # noqa: E731
    elif which == "cohort":
        target = world.root / "test"
        world.cohort.rename(target)
        call = lambda: build_r2_lists(world.identity_root, target, world.root / "o",  # noqa: E731
                                      provenance=PROVENANCE)
    elif which == "fold_work":
        target = world.identity_root / "fold2_test"
        world.work(2).rename(target)
        call = lambda: build(world, fold_work=[f"2={target}"])  # noqa: E731
    else:
        call = lambda: build(world, "r2.test")  # noqa: E731
    with pytest.raises(HeldOutInputError):
        call()
    assert not (world.root / "o").exists() and not (world.root / "r2.test").exists()


# ---------------------------------------------------------------- cohort exclusions

def test_a_load_time_exclusion_stays_in_r2_with_its_reason_and_follows_its_component(world):
    drop_from_membership(world, 0, ISOLATED)
    drop_from_membership(world, 0, LINKED)
    write_cohort(world.cohort, {0: {ISOLATED: "invalid_structure", LINKED: "no_selected_sites"}})
    summary = build(world)
    out, work = world.root / "r2_lists", world.work(0)
    assert {ISOLATED, LINKED} <= set(read_lines(out / "r2_allowlist_fold0.txt"))
    table = {row["accession"]: row for row in read_tsv(out / "r2_vs_deepmzyme_fold0.tsv", DIFFERENCE_COLUMNS)}
    assert table[ISOLATED] == {"accession": ISOLATED, "in_deepmzyme": "0", "reason": "invalid_structure"}
    assert table[LINKED]["reason"] == "no_selected_sites"
    assert table[BRIDGE]["reason"] == table[LONE]["reason"] == "several_ec_classes"
    membership, _ = membership_of(work, 0)
    fold_of_group = {row["group_id"]: int(row["fold"]) for row in membership.rows}
    components = components_of(work)
    assert components[LINKED] == components["E0005"] and components[ISOLATED] not in fold_of_group
    for inner in range(4):
        listed = read_lines(out / f"r2_inner_allowlist_fold0_inner{inner}.txt")
        assert ISOLATED in listed  # its component holds no eligible protein now
        assert (LINKED in listed) == (inner != fold_of_group[components[LINKED]])
    receipt = json.loads((out / r2_lists.RECEIPT_NAME).read_text())
    assert receipt["folds"]["0"]["cohort_exclusion_table"]["n_rows"] == 2
    by_reason = {"invalid_structure": 1, "no_selected_sites": 1, "several_ec_classes": 2}
    assert receipt["folds"]["0"]["counts"]["not_in_deepmzyme_by_reason"] == by_reason
    assert summary["r2_allowlists"]["0"]["not_in_deepmzyme_by_reason"] == by_reason
    assert summary["checks"]["every_difference_explained"] is True


@pytest.mark.parametrize("exclusions, message", [
    ({1: {ISOLATED: "invalid_structure"}}, "Fold 1: 1 load-time exclusion\\(s\\) .* still in the certified"),
    ({0: {BRIDGE: "no_selected_sites"}}, "Fold 0: 1 cohort exclusion\\(s\\) are several-class proteins"),
    ({2: {"Z0001": "invalid_structure"}}, "Fold 2: 1 cohort exclusion\\(s\\) are not fold-2 metallo training"),
], ids=["still_in_membership", "several_class", "not_a_node"])
def test_cohort_exclusions_must_fit_the_identity_chain(world, exclusions, message):
    write_cohort(world.cohort, exclusions)
    with pytest.raises(R2ListError, match=message) as refused:
        build(world)
    assert not any(accession in str(refused.value) for accession in (ISOLATED, BRIDGE, "Z0001"))


@pytest.mark.parametrize("rows, message", [
    ([(ISOLATED, "invalid_structure"), (ISOLATED, "no_selected_sites")], "Repeated accession in row 3"),
    ([(ISOLATED, "ok")], "load status that is not a load-time exclusion"),
    ([(ISOLATED, "ambiguous_ec_class")], "load status that is not a load-time exclusion"),
    ([(ISOLATED, "fatal_load_error")], "load status that is not a load-time exclusion"),
    ([(ISOLATED, "several_ec_classes")], "load status that is not a load-time exclusion"),
    ([("bad id", "invalid_structure")], "Invalid accession in row 2"),
], ids=["repeated", "ok", "ambiguous", "fatal", "reserved", "accession"])
def test_cohort_exclusion_rows_are_checked(world, rows, message):
    write_cohort(world.cohort, {3: rows})
    with pytest.raises(R2ListError, match=message) as refused:
        build(world)
    assert ISOLATED not in str(refused.value)


def test_cohort_tables_are_read_through_the_cohort_receipt(world):
    name = r2_lists.COHORT_EXCLUSIONS_NAME.format(fold=4)
    write_cohort(world.cohort)
    (world.cohort / name).write_bytes(cohort_audit._tsv_bytes(cohort_audit.EXCLUSION_COLUMNS, []))
    with pytest.raises(R2ListError, match="exists but cohort_audit_receipt.json does not record it"):
        build(world)

    write_cohort(world.cohort, {4: {LINKED: "invalid_structure"}})
    (world.cohort / name).write_text((world.cohort / name).read_text() + "E0003\tA\tinvalid_structure\tx\n")
    with pytest.raises(R2ListError, match="changed after the cohort audit"):
        build(world)

    def rebind(text, n_rows=1):
        (world.cohort / name).write_text(text)
        edit_cohort_receipt(world, lambda receipt: receipt["outputs"][name].update(
            sha256=sha256_file(world.cohort / name), n_rows=n_rows))

    write_cohort(world.cohort, {4: {LINKED: "invalid_structure"}})
    rebind(f"accession\treason\n{LINKED}\tinvalid_structure\n")
    with pytest.raises(R2ListError, match="does not have the cohort audit's tab-separated columns"):
        build(world)
    rebind("accession\tchain\tload_status\tload_reason\n", n_rows=0)
    with pytest.raises(R2ListError, match="is empty or holds another number of rows"):
        build(world)
    write_cohort(world.cohort, {4: {LINKED: "invalid_structure"}})
    edit_cohort_receipt(world, lambda receipt: receipt["outputs"][name].update(n_rows=2))
    with pytest.raises(R2ListError, match="is empty or holds another number of rows"):
        build(world)
    write_cohort(world.cohort, {4: {LINKED: "invalid_structure"}})
    edit_cohort_receipt(world, lambda receipt: receipt["per_fold"]["4"].update(no_load_time_exclusions=True))
    with pytest.raises(R2ListError, match="records cohort_exclusions_fold4.tsv but flags fold 4 as having no"):
        build(world)
    write_cohort(world.cohort, {4: {LINKED: "invalid_structure"}})
    edit_cohort_receipt(world, lambda receipt: receipt["outputs"][name].update(sha256="xyz"))
    with pytest.raises(R2ListError, match="records no valid SHA-256 for cohort_exclusions_fold4.tsv"):
        build(world)
    (world.cohort / name).unlink()
    write_cohort(world.cohort)
    edit_cohort_receipt(world, lambda receipt: receipt["per_fold"]["2"].update(no_load_time_exclusions=False))
    with pytest.raises(R2ListError, match="flags load-time exclusions for fold 2 but records no"):
        build(world)
    write_cohort(world.cohort)
    edit_cohort_receipt(world, lambda receipt: receipt["per_fold"].pop("3"))
    with pytest.raises(R2ListError, match="no load-time exclusion flag for fold 3"):
        build(world)

    write_cohort(world.cohort)
    (world.cohort / r2_lists.COHORT_QUARANTINE_NAME).write_text("accession\ta4_outcome\n")
    with pytest.raises(R2ListError, match="quarantined"):
        build(world)
    write_cohort(world.cohort)
    edit_cohort_receipt(world, lambda receipt: receipt["outputs"].update(
        {r2_lists.COHORT_QUARANTINE_NAME: {"sha256": "0" * 64, "n_rows": 1}}))
    with pytest.raises(R2ListError, match="quarantined"):
        build(world)
    shutil.rmtree(world.cohort)
    world.cohort.mkdir()
    with pytest.raises(R2ListError, match="Run the cohort audit first"):
        build(world)


@pytest.mark.parametrize("edit, message", [
    (lambda receipt: receipt.update(status="stopped", failed_checks=["selected_sites_verified",
                                                                     "sequence_map_differences_as_expected"]),
     "did not pass \\(status stopped; 2 failed check\\(s\\) \\['selected_sites_verified', "
     "'sequence_map_differences_as_expected'\\]\\)"),
    (lambda receipt: receipt.update(failed_checks=["all_structures_load"]), "did not pass"),
    (lambda receipt: receipt.update(status="stopped", failed_checks=["Q96KN2"]),
     "did not pass \\(status stopped; 1 failed check\\(s\\) \\['<not shown>'\\]\\)"),
    (lambda receipt: receipt.update(format="other"), "is not a cohort audit receipt"),
    (lambda receipt: receipt.update(kind="cohort_counts_fold"), "is not a cohort audit receipt"),
    (lambda receipt: receipt.update(outputs=None), "is not a cohort audit receipt"),
    (lambda receipt: receipt.pop("status"), "no valid status, failed checks or smoke flag"),
    (lambda receipt: receipt.update(failed_checks="none"), "no valid status, failed checks or smoke flag"),
    (lambda receipt: receipt.update(smoke=None), "no valid status, failed checks or smoke flag"),
    (lambda receipt: receipt.update(per_fold=None), "no per-fold record"),
    (lambda receipt: receipt.update(smoke=True), "is an engineering smoke"),
    (lambda receipt: receipt.clear() or receipt.update(outputs={}), "is not a cohort audit receipt"),
], ids=["stopped", "failed", "unsafe_name", "format", "kind", "outputs", "status", "failed_type", "smoke_type",
        "per_fold", "smoke_audit", "bare"])
def test_the_cohort_receipt_must_record_a_passed_real_audit(world, edit, message):
    edit_cohort_receipt(world, edit)
    with pytest.raises(R2ListError, match=message) as refused:
        build(world)
    assert "Q96KN2" not in str(refused.value)
    assert not (world.root / "r2_lists").exists()


def test_a_smoke_cohort_audit_feeds_a_smoke_run_only(world):
    write_cohort(world.cohort, smoke=True)
    with pytest.raises(R2ListError, match="is an engineering smoke"):
        build(world)
    summary = build(world, smoke=True)
    receipt = json.loads((world.root / "r2_lists" / r2_lists.RECEIPT_NAME).read_text())
    assert summary["smoke"] is True and receipt["smoke"] is True and receipt["cohort"]["smoke"] is True
    assert receipt["provenance"] == SMOKE_PROVENANCE


@pytest.mark.parametrize("smoke, provenance, message", [
    (False, {"smoke": True}, "smoke flag differs"),
    (True, {"smoke": False}, "smoke flag differs"),
    (False, {"smoke": False, "git": {"clean": False}}, "needs a clean tree"),
    (False, None, "needs the action's provenance record"),
])
def test_the_provenance_record_must_fit_the_run(world, smoke, provenance, message):
    with pytest.raises(R2ListError, match=message):
        build(world, smoke=smoke, provenance=provenance)
    build(world, smoke=True, provenance={"smoke": True, "git": {"clean": False}})


def test_cohort_names_match_the_cohort_audit():
    assert r2_lists.COHORT_AUDIT_FORMAT == cohort_audit.COHORT_AUDIT_FORMAT
    assert r2_lists.COHORT_RECEIPT_NAME == cohort_audit.RECEIPT_NAME
    assert r2_lists.COHORT_EXCLUSIONS_NAME == cohort_audit.FOLD_EXCLUSIONS_NAME
    assert r2_lists.COHORT_QUARANTINE_NAME == cohort_audit.QUARANTINE_NAME
    assert r2_lists.COHORT_EXCLUSION_COLUMNS == cohort_audit.EXCLUSION_COLUMNS
    assert r2_lists.COHORT_EXCLUDING_LOAD_STATUSES == cohort_audit.EXCLUDING_LOAD_STATUSES


# ---------------------------------------------------------------- folders, arguments and the action

def test_fold_work_mapping(world):
    assert parse_fold_work(["2=/a/b", " 4 = rel/c "]) == {2: Path("/a/b"), 4: Path("rel/c")}
    for bad, message in (("5=/a", "outside 0..4"), ("x=/a", "K=PATH"), ("2", "K=PATH"), ("2=", "K=PATH")):
        with pytest.raises(R2ListError, match=message):
            parse_fold_work([bad])
    with pytest.raises(R2ListError, match="names fold 1 twice"):
        parse_fold_work(["1=/a", "1=/b"])
    with pytest.raises(R2ListError, match="same identity folder"):
        fold_work_dirs(world.identity_root, [f"1={world.work(0)}"])
    for bad, message in (({7: "/a"}, "must be in 0..4"), ({-1: "/a"}, "must be in 0..4"),
                         ({"x": "/a"}, "must be integers"), ({1: "/a", "1": "/b"}, "names a fold twice")):
        with pytest.raises(R2ListError, match=message):
            fold_work_dirs(world.identity_root, bad)
    rerun = world.identity_root / "fold3_cap1-3"
    world.work(3).rename(rerun)
    with pytest.raises(R2ListError, match="fold-3 identity folder not found"):
        build(world)
    with pytest.raises(R2ListError, match="Fold 3 .*belongs to fold 4"):
        build(world, fold_work=[f"4={rerun}", f"3={world.work(4)}"])
    build(world, fold_work=[f"3={rerun}"])
    receipt = json.loads((world.root / "r2_lists" / r2_lists.RECEIPT_NAME).read_text())
    assert receipt["folds"]["3"]["work"] == str(rerun) and receipt["folds"]["3"]["work_override"] is True
    assert receipt["folds"]["2"]["work_override"] is False
    assert read_lines(world.root / "r2_lists" / "r2_allowlist_fold3.txt") == training_proteins(3)
    build(world, "mapped", fold_work={3: rerun})
    mapped = json.loads((world.root / "mapped" / r2_lists.RECEIPT_NAME).read_text())
    assert mapped["folds"] == receipt["folds"]


def test_output_folder_rules(world):
    build(world)
    with pytest.raises(R2ListError, match="output folder exists"):
        build(world)
    for name in ("held_test_lists", "folds/r2_lists", "metadata/r2_lists"):
        with pytest.raises(HeldOutInputError):
            build(world, name)
    assert not (world.root / "folds").exists() and not (world.root / "metadata").exists()
    assert not [path for path in world.root.iterdir() if path.name.startswith(".")]


def test_verify_checks_the_output_set_and_sizes(world):
    build(world)
    out = world.root / "r2_lists"
    receipt_path = out / r2_lists.RECEIPT_NAME
    original = receipt_path.read_text()
    edits = [
        (lambda receipt: receipt["outputs"]["r2_allowlist_fold1.txt"].update(
            n_records=receipt["outputs"]["r2_allowlist_fold1.txt"]["n_records"] + 1), "has another size"),
        (lambda receipt: receipt["outputs"]["r2_vs_deepmzyme_fold0.tsv"].update(n_records=0), "has another size"),
        (lambda receipt: receipt["outputs"].update({"extra.txt": {"sha256": "0" * 64, "n_records": 1}}),
         "names other outputs"),
        (lambda receipt: receipt["outputs"].pop("r2_inner_allowlist_fold0_inner2.txt"), "names other outputs"),
        (lambda receipt: receipt.update(step="cohort_audit"), "is not an R2-list receipt"),
    ]
    for edit, message in edits:
        receipt = json.loads(original)
        edit(receipt)
        receipt_path.write_text(json.dumps(receipt))
        with pytest.raises(R2ListError, match=message):
            verify_r2_lists(out)
    receipt_path.write_text(original)
    assert verify_r2_lists(out)["n_files"] == 5 + 5 + 4


def test_allowlists_hold_valid_unique_accessions(tmp_path):
    with pytest.raises(R2ListError, match="would hold an invalid accession"):
        r2_lists._allowlist_text(["P12345", "bad id"], "x.txt")
    with pytest.raises(R2ListError, match="would be empty"):
        r2_lists._allowlist_text([], "x.txt")
    for index, text in enumerate(("", "P12345\nP12345\n", "P12345\nbad id\n")):
        path = tmp_path / f"list{index}.txt"
        path.write_text(text)
        with pytest.raises(R2ListError, match="is not a valid accession allowlist"):
            read_allowlist(path)
    path = tmp_path / "good.txt"
    path.write_text("P12345\nQ67890\n")
    assert read_allowlist(path) == ["P12345", "Q67890"]


def _action(world, monkeypatch, argv, *, clean, calls=None):
    """Run the r2-lists action with a fake git state (``clean``) and a light provenance record."""
    calls = [] if calls is None else calls

    def git_state(*, smoke):
        calls.append(smoke)
        if not smoke and not clean:
            raise DirtyTreeError("synthetic dirty tree")
        return {"head": "0" * 40, "clean": clean}

    monkeypatch.setattr(r2_lists, "real_action_git_state", git_state)
    monkeypatch.setattr(r2_lists, "provenance_record", lambda *, smoke, git: {"smoke": smoke, "git": dict(git)})
    parser = r2_lists.add_r2_lists_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--identity-root", str(world.identity_root), "--cohort", str(world.cohort), *argv])
    return r2_lists.action_r2_lists(args), calls


def test_the_action_checks_the_tree_and_prints_the_summary(world, monkeypatch, capsys):
    out = world.root / "wired"
    with pytest.raises(DirtyTreeError):
        _action(world, monkeypatch, ["--out", str(out)], clean=False)
    assert not out.exists()
    code, calls = _action(world, monkeypatch, ["--out", str(out), "--fold-work", f"0={world.work(0)}"], clean=True)
    printed = json.loads(capsys.readouterr().out)
    assert code == 0 and calls == [False] and printed["out"] == str(out) and printed["smoke"] is False
    assert printed["r2_allowlists"]["4"] == {"file": "r2_allowlist_fold4.txt", "size_in_receipt": True}
    receipt = json.loads((out / r2_lists.RECEIPT_NAME).read_text())
    assert receipt["provenance"]["git"]["clean"] is True and receipt["folds"]["0"]["work_override"] is False
    code, calls = _action(world, monkeypatch, ["--out", str(world.root / "smoked"), "--smoke"], clean=False)
    assert code == 0 and calls == [True] and json.loads(capsys.readouterr().out)["smoke"] is True


def test_the_action_reports_refusals_without_a_traceback(world, monkeypatch, capsys):
    build(world, "taken")
    code, _calls = _action(world, monkeypatch, ["--out", str(world.root / "taken")], clean=True)
    assert code == 2 and capsys.readouterr().err.startswith("error: The output folder exists")
    code, calls = _action(world, monkeypatch, ["--out", str(world.root / "o"), "--fold-work", "9=/a"], clean=True)
    assert code == 2 and calls == [] and "outside 0..4" in capsys.readouterr().err
    world.work(2).rename(world.identity_root / "fold2_test")
    for argv in (["--out", str(world.root / "lists_test")],
                 ["--out", str(world.root / "o"), "--fold-work", f"2={world.identity_root / 'fold2_test'}"]):
        calls = []
        with pytest.raises(HeldOutInputError):
            _action(world, monkeypatch, argv, clean=True, calls=calls)
        assert calls == []  # refused before the tree check and the build
    (world.identity_root / "fold2_test").rename(world.work(2))
    _code, calls = _action(world, monkeypatch, ["--out", str(world.root / "o")], clean=True)
    assert calls == [False]
    assert not (world.root / "lists_test").exists()


def test_arguments_wire_to_the_builder(world):
    parser = r2_lists.add_r2_lists_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--identity-root", str(world.identity_root), "--cohort", str(world.cohort),
                              "--out", str(world.root / "wired"), "--fold-work", f"0={world.work(0)}"])
    assert (args.fold_work, args.smoke) == ([f"0={world.work(0)}"], False)
    summary = build_r2_lists(args.identity_root, args.cohort, args.out, fold_work=args.fold_work,
                             provenance=PROVENANCE, smoke=args.smoke)
    assert summary["out"] == str(world.root / "wired") and len(summary["r2_allowlists"]) == 5


def test_constants_match_the_identity_tool_and_the_comparator():
    assert r2_lists.COMPARATOR_HELD_OUT_NAME_RE.pattern == comparator.HELD_OUT_NAME_RE.pattern
    assert r2_lists.COMPARATOR_HELD_OUT_NAME_RE.flags == comparator.HELD_OUT_NAME_RE.flags
    assert r2_lists.COMPARATOR_HELD_OUT_DIR_NAMES == comparator.HELD_OUT_DIR_NAMES
    assert r2_lists.R1_INNER_EXCLUSIONS_NAME == comparator.EXCLUSIONS_NAME
    assert r2_lists.R1_REASONS == comparator.EXCLUSION_REASONS
    assert (r2_lists.N_FOLDS, r2_lists.N_INNER_FOLDS) == (comparator.N_FOLDS, comparator.N_INNER_FOLDS)
    assert set(r2_lists.SUMMARY_COUNT_FOLDS) == {0} and set(r2_lists.R2_INNER_OUTER_FOLDS) <= {0}


def test_module_imports_without_torch():
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import ec_clean30.r2_lists; "
            "print([name for name in ('torch', 'numpy', 'ec_clean30.views', 'ec_clean30.cohort') "
            "if name in sys.modules])")
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert output.strip() == "[]"
