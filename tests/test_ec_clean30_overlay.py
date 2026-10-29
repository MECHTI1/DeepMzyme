"""Data overlay of the EC class vs CLEAN30 campaign: builder and verifier (synthetic data only; no dataset access).

The fixture writes a five-fold synthetic data root with the cohort audit's test builder, runs the
cohort audit, the training-side identity chain (fasta, a fake pinned MMseqs2 search, edges, inner
components, assignment, certification and fold 0's R1-inner lists) and the R2 lists on it, and
builds overlays from those outputs. Every folder name is neutral: the tools refuse test-named
path parts. No real campaign file is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]

from ec_clean30 import identity, overlay, r2_lists  # noqa: E402
from ec_clean30.campaign import CAMPAIGN_ID  # noqa: E402
from ec_clean30.cohort import RECEIPT_NAME as COHORT_RECEIPT, run_cohort_audit  # noqa: E402
from ec_clean30.identity import (  # noqa: E402
    DONOR_COLUMN, assign_inner, build_edges, build_inner_components, build_inputs, build_r1_inner_exclusions,
    certify_inner, run_search, sha256_file,
)
from ec_clean30.overlay import (  # noqa: E402
    DEFAULT_OVERLAY_ID, MANIFEST_NAME, OverlayError, OverlayVerificationError, build_overlay, verify_overlay,
)
from ec_clean30.provenance import DirtyTreeError, HeldOutInputError  # noqa: E402
from ec_clean30.views import SourceLayout, ViewError, build_view, verify_view  # noqa: E402
from ec_clean30_fixtures import make_writable  # noqa: E402
from test_ec_clean30_cohort import (  # noqa: E402
    AMBIGUOUS, DEFAULT_RESIDUES, TRAIN_COLUMNS, build_cohort, write_structure,
)
from test_ec_clean30_identity import FakeMMseqs, fake_binary, random_sequence  # noqa: E402

N_FOLDS = 5
REAL_PROVENANCE = {"smoke": False, "note": "synthetic test provenance"}
SMOKE_PROVENANCE = {"smoke": True, "note": "synthetic test provenance"}
# One backbone-only residue far from the metal differs per protein, so no two sequences are equal (the
# identity tool links equal sequences into one component).
VARIABLE_RESIDUES = ("ALA", "GLY", "LEU", "ILE", "VAL", "MET", "PHE", "TRP", "PRO", "THR", "ASN", "GLN", "TYR")
A13_ACCESSION = "P3002"
NON_METALLO = ("N0001", "N0002")


D2_ACCESSION = "P2001"
# P2001: two selected sites in separate pockets (ZN 900, MN 902), a copy of its ZN site from another donor 1.0 Å
# away (no site row) and a CU whose site row is not catalytic.
D2_METALS = [(900, "ZN", (0.0, 0.0, 0.0)), (901, "ZN", (1.0, 0.0, 0.0)), (902, "MN", (0.0, 0.0, 10.0)),
             (903, "CU", (-8.0, 0.0, 0.0))]
D2_SITES = [(900, "ZN", (0.0, 0.0, 0.0), "1"), (902, "MN", (0.0, 0.0, 10.0), "1"), (903, "CU", (-8.0, 0.0, 0.0), "0")]
D2_SELECTED_RESSEQ = (900, 902)


def overlay_proteins() -> dict[str, dict]:
    """Twelve single-class proteins (EC classes 1-3) and one with two EC classes; P1001 holds an unselected
    FE, P2001 the metals of ``D2_METALS``, and P3002's payload names a test folder in its source path (an A13
    string)."""
    proteins: dict[str, dict] = {}
    for class_index, ec_class in enumerate(("1", "2", "3")):
        for k in range(4):
            proteins[f"P{class_index + 1}00{k}"] = {"ec": f"{ec_class}.1.1.{k + 1}"}
    proteins["P1001"]["metals"] = [(900, "ZN", (0.0, 0.0, 0.0)), (901, "FE", (6.0, 0.0, 0.0))]
    proteins[D2_ACCESSION].update(metals=list(D2_METALS), sites=list(D2_SITES))
    proteins[A13_ACCESSION]["esm_source_path"] = f"/synthetic/CLEAN_30_train_test_split_0/test/{A13_ACCESSION}.pdb"
    proteins[AMBIGUOUS] = {"ec": "2.5.1.3,4.1.99.17"}
    for index, accession in enumerate(sorted(proteins)):
        proteins[accession]["residues"] = (*DEFAULT_RESIDUES[:7], (VARIABLE_RESIDUES[index], (25.0, 0.0, 0.0)))
    return proteins


@dataclass(frozen=True)
class World:
    base: Path
    data_root: Path
    campaign: Path
    binary_sha: str
    test_fold: dict
    stems: dict

    @property
    def layout(self) -> SourceLayout:
        return SourceLayout.from_data_root(self.data_root)

    @property
    def cohort(self) -> Path:
        return self.campaign / "cohort"

    @property
    def identity_root(self) -> Path:
        return self.campaign / "identity"

    @property
    def r2(self) -> Path:
        return self.campaign / "r2_lists"

    def work(self, fold: int) -> Path:
        return self.identity_root / f"fold{fold}"

    def training(self, fold: int) -> list[str]:
        return sorted(accession for accession, test in self.test_fold.items() if test != fold)

    def moved(self, base: Path) -> "World":
        relative = {name: getattr(self, name).relative_to(self.base) for name in ("data_root", "campaign")}
        return World(base, base / relative["data_root"], base / relative["campaign"], self.binary_sha,
                     self.test_fold, self.stems)


def _rewrite_donors(folds_dir: Path, accessions) -> None:
    """Give each protein its own donor (the cohort builder gives all one donor, which links every protein)."""
    donors = {accession: f"9{index:03d}" for index, accession in enumerate(sorted(accessions))}
    for fold in range(N_FOLDS):
        path = folds_dir / f"CLEAN_30_train_test_split_{fold}_train.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row[DONOR_COLUMN] = donors[row["structure"]]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TRAIN_COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def build_world(base: Path) -> World:
    source = build_cohort(base / "source", overlay_proteins())
    data = source["data_root"]
    folds_dir = data / "CLEAN_30_main" / "folds"
    _rewrite_donors(folds_dir, source["stems"])
    split30 = base / "source" / "split30"
    split30.mkdir(parents=True)
    rng = random.Random(11)
    others = {accession: random_sequence(rng, 40) for accession in NON_METALLO}
    for fold in range(N_FOLDS):
        training = sorted(accession for accession, test in source["test_fold"].items() if test != fold)
        with (split30 / f"split30_train_split_{fold}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["Entry", "EC number", "Sequence"])
            writer.writerows([(accession, "1.1.1.1", source["sequences"][accession]) for accession in training]
                             + [(accession, "4.2.1.1", sequence) for accession, sequence in others.items()])
    campaign = base / CAMPAIGN_ID
    summary = run_cohort_audit(layout=SourceLayout.from_data_root(data), out_dir=campaign / "cohort",
                               label_table=source["label_table"], label_table_sha256=source["label_sha"],
                               sequence_map=source["sequence_map"], sequence_map_sha256=source["map_sha"],
                               provenance=REAL_PROVENANCE)
    assert summary["status"] == "passed", summary["failed_checks"]
    binary = fake_binary(base / "fakebin")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(identity, "MMSEQS_BINARY_SHA256", sha256_file(binary))
        for fold in range(N_FOLDS):
            work = campaign / "identity" / f"fold{fold}"
            build_inputs(work, fold, official_split_dir=split30, metallo_folds_dir=folds_dir,
                         metallo_manifest_root=data)
            run_search(work, fold, binary=binary, expected_sha256=sha256_file(binary), runner=FakeMMseqs(hits={}))
            build_edges(work, fold)
            build_inner_components(work, fold)
            assign_inner(work, fold, metallo_folds_dir=folds_dir, metallo_manifest_root=data)
            certify_inner(work, fold, metallo_folds_dir=folds_dir, metallo_manifest_root=data)
        for inner in range(identity.N_INNER_FOLDS):
            build_r1_inner_exclusions(campaign / "identity" / "fold0", 0, inner)
        r2_lists.build_r2_lists(campaign / "identity", campaign / "cohort", campaign / "r2_lists",
                                provenance=REAL_PROVENANCE)
    return World(base, data, campaign, sha256_file(binary), dict(source["test_fold"]), dict(source["stems"]))


@pytest.fixture(scope="module")
def world_template(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("overlayworld"))


@pytest.fixture(autouse=True)
def pinned_binary(world_template, monkeypatch):
    """The fake binary counts as the pinned MMseqs2 build (the identity loaders require it)."""
    monkeypatch.setattr(identity, "MMSEQS_BINARY_SHA256", world_template.binary_sha)


@pytest.fixture
def world(world_template, tmp_path_factory):
    """A private copy of the synthetic world (for tests that change inputs)."""
    base = tmp_path_factory.mktemp("worldcopy") / "world"
    shutil.copytree(world_template.base, base)
    return world_template.moved(base)


def campaign_root(tmp_path_factory, name: str = "overlaybuild") -> Path:
    return tmp_path_factory.mktemp(name) / CAMPAIGN_ID


# The synthetic cohort audit recommends option d for decision A1 (P1001's and P2001's other metal atoms reach their
# pockets in every fold's training side), so a real overlay of this world takes option d; option b is built as a
# smoke overlay.
REAL_D2_OPTION = "d"
SMOKE_B = {"d2_option": "b", "provenance": SMOKE_PROVENANCE, "smoke": True}


def build(world: World, campaign: Path, *, overlay_id: str = DEFAULT_OVERLAY_ID, **kwargs) -> dict:
    options = {"cohort_dir": world.cohort, "identity_root": world.identity_root, "r2_lists_dir": world.r2,
               "d2_option": REAL_D2_OPTION, "provenance": REAL_PROVENANCE, "layout": world.layout}
    options.update(kwargs)
    return build_overlay(out_dir=campaign / "overlay" / overlay_id, **options)


def _built(world_template, tmp_path_factory, name: str, **kwargs) -> dict:
    campaign = campaign_root(tmp_path_factory, name)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(identity, "MMSEQS_BINARY_SHA256", world_template.binary_sha)
        summary = build(world_template, campaign, layout=None, **kwargs)  # the layout the cohort audit recorded
    root = campaign / "overlay" / DEFAULT_OVERLAY_ID
    return {"summary": summary, "root": root, "campaign": campaign,
            "manifest": json.loads((root / MANIFEST_NAME).read_text())}


@pytest.fixture(scope="module")
def built(world_template, tmp_path_factory):
    """The real overlay of the synthetic world (option d, as its cohort audit recommends)."""
    return _built(world_template, tmp_path_factory, "builtoverlay")


@pytest.fixture(scope="module")
def built_b(world_template, tmp_path_factory):
    """A smoke overlay under option b (a real one is refused while the audit recommends d)."""
    return _built(world_template, tmp_path_factory, "builtoverlayb", **SMOKE_B)


def copy_overlay(built: dict, tmp_path_factory) -> Path:
    """A writable copy of the built overlay folder with its tar and side files; returns the copied folder."""
    target = tmp_path_factory.mktemp("overlaycopy") / CAMPAIGN_ID / "overlay"
    shutil.copytree(built["root"].parent, target)
    make_writable(target)
    for path in target.iterdir():
        if path.is_file():
            os.chmod(path, 0o644)
    return target / DEFAULT_OVERLAY_ID


def assert_nothing_written(campaign: Path) -> None:
    folder = campaign / "overlay"
    assert not folder.exists() or not any(folder.iterdir()), sorted(path.name for path in folder.iterdir())


def accessions_named(text: str, accessions) -> set[str]:
    return {accession for accession in accessions if accession in text}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cohort_proteins(world: World) -> dict[str, dict[str, str]]:
    with (world.cohort / "cohort_proteins.tsv").open(newline="") as handle:
        return {row["accession"]: row for row in csv.DictReader(handle, delimiter="\t")}


# ---------------------------------------------------------------- the built overlay

def test_build_writes_every_input_by_role(built, world_template):
    root, manifest, summary = built["root"], built["manifest"], built["summary"]
    assert summary["status"] == "built" and summary["overlay_id"] == DEFAULT_OVERLAY_ID
    assert manifest["format"] == overlay.OVERLAY_FORMAT and manifest["overlay_id"] == DEFAULT_OVERLAY_ID
    assert manifest["smoke"] is False and "layout_from" not in manifest["sources"]
    build_receipt = json.loads(Path(summary["build_receipt"]).read_text())
    assert build_receipt["layout_from"] == "cohort_receipt" and build_receipt["tar"]["sha256"] == summary["tar_sha256"]
    assert build_receipt["manifest_sha256"] == summary["manifest_sha256"] == _sha((root / MANIFEST_NAME).read_bytes())
    union = sorted(world_template.stems)
    assert manifest["counts"]["n_proteins"] == len(union) == 13
    by_role = manifest["counts"]["files_by_role"]
    for role in (overlay.ROLE_STRUCTURE, overlay.ROLE_ESM, overlay.ROLE_ESM_SIDECAR, overlay.ROLE_EXTERNAL):
        assert by_role[role] == len(union), role
    assert by_role[overlay.ROLE_TRAINING_MANIFEST] == by_role[overlay.ROLE_TRAINING_SITE_TABLE] == N_FOLDS
    for fold in range(N_FOLDS):
        assert (root / f"CLEAN_30_train_test_split_{fold}" / "train" / "structure_manifest.csv").is_file()
        assert (root / "CLEAN_30_main" / "folds" / f"CLEAN_30_train_test_split_{fold}_train.csv").is_file()
        assert (root / f"allowlist_fold{fold}.txt").read_text().splitlines() == sorted(
            world_template.stems[accession] for accession in world_template.training(fold))
        assert (root / f"eligible_fold{fold}.txt").read_text().splitlines() == sorted(
            world_template.stems[accession] for accession in world_template.training(fold) if accession != AMBIGUOUS)
        assert manifest["counts"]["per_fold"][str(fold)]["n_eligible"] == len(
            [accession for accession in world_template.training(fold) if accession != AMBIGUOUS])
    for stem in world_template.stems.values():
        assert (root / "esm" / f"{stem}_chain_A_esmc.pt").is_file()
        assert (root / "esm" / f"{stem}_chain_A_esmc.pt.json").is_file()
        assert (root / "external" / stem / "residue_features.json").is_file()
        assert len(list((root / "structure_store").rglob(f"{stem}.pdb"))) == 1
    with (root / overlay.LABEL_FREE_INDEX_NAME).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["accession"] for row in rows] == union and list(rows[0]) == list(overlay.LABEL_FREE_INDEX_COLUMNS)
    for path in (root / "cohort" / COHORT_RECEIPT, root / "r2_lists" / r2_lists.RECEIPT_NAME):
        assert path.is_file()
    for entry in manifest["files"]:
        assert _sha((root / entry["path"]).read_bytes()) == entry["sha256"]
    assert manifest["base"]["archive_sha256"].startswith("90c08998")
    cohort_receipt = json.loads((world_template.cohort / COHORT_RECEIPT).read_text())
    assert cohort_receipt["a1_recommended_option"] == REAL_D2_OPTION
    assert {key: manifest["d2"][key] for key in ("option", "cohort_a1_recommended_option",
                                                 "follows_cohort_recommendation")} == {
        "option": "d", "cohort_a1_recommended_option": "d", "follows_cohort_recommendation": True}
    assert "RING not used" in manifest["ring"] and not (root / "ring").exists()
    assert not list(root.rglob("split_metadata.json")) and not list(root.rglob("*_test.csv"))


def test_option_b_copies_external_features_as_before(built_b, world_template):
    """Regression for option b (a smoke overlay here): byte copies with the copy origin, the cohort audit's SHA-256
    values in the label-free index, the unchanged rules and D2 block, and no derivation record anywhere."""
    root, manifest, summary = built_b["root"], built_b["manifest"], built_b["summary"]
    assert manifest["smoke"] is True and manifest["d2"]["option"] == "b"
    assert manifest["d2"]["cohort_a1_recommended_option"] == "d"
    assert manifest["d2"]["follows_cohort_recommendation"] is False
    cohort_rows = _cohort_proteins(world_template)
    externals = [entry for entry in manifest["files"] if entry["role"] == overlay.ROLE_EXTERNAL]
    assert len(externals) == len(world_template.stems)
    for accession, stem in world_template.stems.items():
        relative = f"{stem}/residue_features.json"
        entry = next(entry for entry in externals if entry["path"] == f"external/{relative}")
        assert entry["origin"] == {"kind": "copy", "root": "external_root", "path": relative}
        assert entry["sha256"] == cohort_rows[accession]["external_sha256"]
        assert (root / entry["path"]).read_bytes() == (world_template.layout.external_root / relative).read_bytes()
    with (root / overlay.LABEL_FREE_INDEX_NAME).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            assert row["external_sha256"] == cohort_rows[row["accession"]]["external_sha256"]
    index = next(entry for entry in manifest["files"] if entry["path"] == overlay.LABEL_FREE_INDEX_NAME)
    assert index["origin"] == {"kind": "derived", "rule": overlay.RULES["label_free_index"],
                               "from": {"cohort_proteins_sha256": _sha((world_template.cohort
                                                                        / "cohort_proteins.tsv").read_bytes())}}
    assert set(manifest["d2"]) == {"option", "policy", "cohort_a1_recommended_option", "follows_cohort_recommendation"}
    assert manifest["d2"]["policy"] == overlay.D2_POLICY["b"] and manifest["rules"] == dict(overlay.RULES)
    build_receipt = json.loads(Path(summary["build_receipt"]).read_text())
    assert "d2_derivation" not in summary and "d2_derivation" not in build_receipt
    assert "d2_option" not in verify_overlay(root)


def test_identity_copies_hold_what_the_loaders_read(built, world_template):
    root, manifest = built["root"], built["manifest"]
    names = {entry["path"] for entry in manifest["files"] if entry["role"] == overlay.ROLE_IDENTITY}
    for fold in range(N_FOLDS):
        folder = f"identity/fold{fold}"
        for name in ("fasta_receipt_metallo_train.json", f"target_train_fold{fold}.fasta", "query_metallo_train.fasta",
                     "nodes_metallo_train.tsv", "search_metallo_train/search_receipt.json",
                     "search_metallo_train/alignments.sorted.tsv", "edges_receipt_metallo_train.json",
                     "edges_metallo_train.tsv", "inner_pairs.tsv", "nonmetallo_d3_hits.tsv", "components_inner.tsv",
                     "component_links_inner.tsv", "components_receipt_inner.json", "inner_assignment_receipt.json",
                     f"inner_membership_fold{fold}.csv", "inner_certification_receipt.json"):
            assert f"{folder}/{name}" in names, (fold, name)
        assert f"{folder}/search_metallo_train/alignments.raw.tsv" not in names
        r1 = {name for name in names if name.startswith(f"{folder}/r1_inner_exclusions")}
        assert len(r1) == (8 if fold == 0 else 0)
        copy = root / folder
        chain = identity.load_inner_chain(copy, fold)
        components, _receipt_sha = identity._load_inner_components(copy, fold, chain)
        membership, certificate_sha = identity._load_certified_membership(copy, fold, chain)
        source_chain = identity.load_inner_chain(world_template.work(fold), fold)
        assert chain.record() == source_chain.record()
        assert set(components) == set(world_template.training(fold))
        assert certificate_sha == sha256_file(world_template.work(fold) / identity.INNER_CERTIFICATE)
        assert manifest["inputs"]["identity"][str(fold)]["record"] == r2_lists.load_fold_inputs(copy, fold).record
    inputs = r2_lists.load_fold_inputs(root / "identity" / "fold0", 0)
    for inner in range(identity.N_INNER_FOLDS):
        assert r2_lists.load_r1_inner_exclusions(inputs, inner)[1]["n_rows"] > 0


def test_simulation_calibration_reads_the_overlay_copy(built):
    from ec_clean30.simulation import load_fold_calibration

    for fold in (0, 3):
        calibration = load_fold_calibration(fold, built["root"] / "identity" / f"fold{fold}", smoke=False)
        assert calibration.inputs["overlay_manifest_sha256"] == built["summary"]["manifest_sha256"]


def test_string_scan_matches_the_cohort_count(built, world_template):
    scan = built["manifest"]["string_scan"]
    cohort_receipt = json.loads((world_template.cohort / COHORT_RECEIPT).read_text())
    assert scan["count"] == scan["cohort_a13_total"] == cohort_receipt["totals"]["a13_test_named_string_count"]["total"]
    assert scan["count"] >= 1 and scan["by_role"][overlay.ROLE_ESM] >= 1


def test_printed_summary_holds_union_totals_only(built, world_template):
    summary = built["summary"]
    text = json.dumps(summary)
    assert not accessions_named(text, world_template.stems) and "per_fold" not in text and "n_eligible" not in text
    assert not any(isinstance(value, dict) and set(value) & {str(fold) for fold in range(N_FOLDS)}
                   for value in summary.values())


def test_no_complement_list_is_ever_written(built, world_template):
    union = set(world_template.stems)
    complements = [union - set(world_template.training(fold)) for fold in range(N_FOLDS)]
    for path in built["root"].rglob("*"):
        if not path.is_file() or path.suffix == ".pt":
            continue
        named = accessions_named(path.read_text(encoding="utf-8"), union)
        assert all(named != complement for complement in complements), path.relative_to(built["root"])
        assert "complement" not in path.name


# ---------------------------------------------------------------- verification and the tar

def test_overlay_verifies_and_writes_a_receipt_once(built, tmp_path_factory):
    receipt = tmp_path_factory.mktemp("verifyreceipt") / "overlay_verify.json"
    result = verify_overlay(built["root"], receipt_path=receipt, provenance=REAL_PROVENANCE)
    assert result["status"] == "verified" and result["tar_sha256"] == built["summary"]["tar_sha256"]
    assert result["manifest_sha256"] == built["summary"]["manifest_sha256"] and result["n_proteins"] == 13
    assert result["build_receipt_sha256"] == built["summary"]["build_receipt_sha256"]
    assert json.loads(receipt.read_text())["provenance"] == REAL_PROVENANCE
    with pytest.raises(OverlayError, match="new file outside the overlay"):
        verify_overlay(built["root"], receipt_path=receipt)
    with pytest.raises(OverlayError, match="new file outside the overlay"):
        verify_overlay(built["root"], receipt_path=built["root"] / "receipt.json")


def test_tar_is_reproducible_and_checks_in_another_folder(built, tmp_path_factory):
    root, summary = built["root"], built["summary"]
    tar_path = Path(summary["tar"])
    assert tar_path.name == f"{DEFAULT_OVERLAY_ID}.tar"
    assert Path(summary["tar_sha256_file"]).read_text() == f"{summary['tar_sha256']}  {tar_path.name}\n"
    with tarfile.open(tar_path) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        assert names == sorted(names) and names[0] == DEFAULT_OVERLAY_ID
        for member in members:
            assert (member.mtime, member.uid, member.gid, member.uname, member.gname) == (0, 0, 0, "", "")
            assert member.mode == (0o555 if member.isdir() else 0o444)
            if member.isfile():
                data = archive.extractfile(member).read()
                assert data == (root / member.name[len(DEFAULT_OVERLAY_ID) + 1:]).read_bytes()
    files = {member.name for member in members if member.isfile()}
    assert files == {f"{DEFAULT_OVERLAY_ID}/{entry['path']}" for entry in built["manifest"]["files"]} | {
        f"{DEFAULT_OVERLAY_ID}/{MANIFEST_NAME}"}
    mirror = tmp_path_factory.mktemp("mirror")
    for path in (tar_path, Path(summary["tar_sha256_file"])):
        shutil.copyfile(path, mirror / path.name)
    checked = subprocess.run(["sha256sum", "-c", f"{DEFAULT_OVERLAY_ID}.tar.sha256"], cwd=mirror,
                             capture_output=True, text=True)
    assert checked.returncode == 0 and checked.stdout.strip().endswith("OK")


def test_two_builds_give_the_same_tar(built, world_template, tmp_path_factory):
    # The first build read the layout from the cohort receipt; this one is given the same layout.
    second = build(world_template, campaign_root(tmp_path_factory, "secondbuild"))
    assert second["tar_sha256"] == built["summary"]["tar_sha256"]
    assert second["manifest_sha256"] == built["summary"]["manifest_sha256"]
    assert Path(second["build_receipt"]).read_bytes() != Path(built["summary"]["build_receipt"]).read_bytes()
    assert json.loads(Path(second["build_receipt"]).read_text())["layout_from"] == "given"


def test_verification_reports_changes_without_naming_proteins(built, world_template, tmp_path_factory):
    stems = world_template.stems
    copy = copy_overlay(built, tmp_path_factory)
    assert verify_overlay(copy)["status"] == "verified"

    payload = copy / "esm" / f"{stems['P2001']}_chain_A_esmc.pt"
    original = payload.read_bytes()
    payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    text = "\n".join(failed.value.problems)
    assert "differ from the overlay manifest" in text and "under esm/" in text
    assert not accessions_named(text, stems)
    payload.write_bytes(original)

    extra = copy / "external" / stems["P1002"] / "notes.txt"
    extra.write_text("x")
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    text = "\n".join(failed.value.problems)
    assert "not in the overlay manifest: 1 under external/" in text and not accessions_named(text, stems)
    extra.unlink()

    components = copy / "identity" / "fold2" / "components_inner.tsv"
    original = components.read_bytes()
    components.write_bytes(original + b"\n")
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    text = "\n".join(failed.value.problems)
    assert "identity/fold2/components_inner.tsv" in text and "do not load as inputs" in text
    assert not accessions_named(text, stems)
    components.write_bytes(original)
    assert verify_overlay(copy)["status"] == "verified"


def test_entries_outside_the_fixed_folders_are_counted_not_named(built, world_template, tmp_path_factory):
    stem = world_template.stems["P1003"]
    copy = copy_overlay(built, tmp_path_factory)
    (copy / f"{stem}.pdb").write_text("x")
    (copy / stem).mkdir()
    (copy / f"{stem}_dir").mkdir()
    (copy / f"{stem}_dir" / "notes.txt").write_text("x")
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    problems = failed.value.problems
    assert "2 files are not in the overlay manifest: 1 under other top-level folders, 1 under the overlay root" \
        in problems
    assert "2 folders hold no manifest file: 2 under the overlay root" in problems
    assert not accessions_named("\n".join(problems), world_template.stems)


def test_verification_checks_the_tar(built, tmp_path_factory):
    copy = copy_overlay(built, tmp_path_factory)
    tar_path = copy.parent / f"{DEFAULT_OVERLAY_ID}.tar"
    data = bytearray(tar_path.read_bytes())
    data[600] ^= 1
    tar_path.write_bytes(bytes(data))
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    assert any(".tar.sha256 does not hold the tar's SHA-256" in problem for problem in failed.value.problems)
    assert any("differs from a re-creation" in problem for problem in failed.value.problems)
    (copy.parent / f"{DEFAULT_OVERLAY_ID}.tar.sha256").unlink()
    with pytest.raises(OverlayVerificationError, match="cannot be read"):
        verify_overlay(copy)


def _rewrite_tar(copy: Path, change) -> None:
    """Rewrite the tar next to ``copy`` member by member through ``change(info)`` (it may return extra members),
    and its .sha256 file to match."""
    tar_path = copy.parent / f"{DEFAULT_OVERLAY_ID}.tar"
    buffer = io.BytesIO()
    with tarfile.open(tar_path) as source, tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as out:
        for info in source.getmembers():
            data = source.extractfile(info).read() if info.isfile() else None
            for member, payload in change(info, data):
                out.addfile(member, io.BytesIO(payload) if payload is not None else None)
    tar_path.write_bytes(buffer.getvalue())
    (copy.parent / f"{DEFAULT_OVERLAY_ID}.tar.sha256").write_text(f"{_sha(buffer.getvalue())}  {tar_path.name}\n")


def test_verification_checks_the_tar_members_and_headers(built, tmp_path_factory):
    def extra_member(info, data):
        yield info, data
        if info.name == DEFAULT_OVERLAY_ID:
            added = tarfile.TarInfo(f"{DEFAULT_OVERLAY_ID}/added.txt")
            added.size, added.mode = 1, 0o444
            yield added, b"x"

    copy = copy_overlay(built, tmp_path_factory)
    _rewrite_tar(copy, extra_member)
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    problems = failed.value.problems
    assert any("the tar's member list differs from the overlay manifest" in problem for problem in problems)

    def later_mtime(info, data):
        if info.name == f"{DEFAULT_OVERLAY_ID}/{MANIFEST_NAME}":
            info.mtime = 1
        yield info, data

    copy = copy_overlay(built, tmp_path_factory)
    _rewrite_tar(copy, later_mtime)
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    assert "1 tar members differ from the reproducible header rule" in failed.value.problems


def _write_manifest(copy: Path, change) -> None:
    path = copy / MANIFEST_NAME
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def reseal(copy: Path, *, receipt: bool = True) -> None:
    """After a deliberate change to an overlay copy: rewrite its tar, the .sha256 file and (by default) the build
    receipt, so only the content checks can see the change."""
    manifest_bytes = (copy / MANIFEST_NAME).read_bytes()
    files = [entry["path"] for entry in json.loads(manifest_bytes)["files"]]
    buffer = io.BytesIO()
    overlay.write_tar(copy, DEFAULT_OVERLAY_ID, [*files, MANIFEST_NAME], buffer)
    side = overlay.overlay_side_paths(copy)
    side["tar"].write_bytes(buffer.getvalue())
    side["tar_sha256"].write_text(f"{_sha(buffer.getvalue())}  {side['tar'].name}\n")
    if receipt:
        payload = json.loads(side["build_receipt"].read_text())
        payload["manifest_sha256"], payload["tar"]["sha256"] = _sha(manifest_bytes), _sha(buffer.getvalue())
        side["build_receipt"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _replace_file(copy: Path, relative: str, data: bytes) -> None:
    """Change one overlay file and its manifest entry consistently."""
    (copy / relative).write_bytes(data)

    def change(payload):
        entry = next(entry for entry in payload["files"] if entry["path"] == relative)
        entry["sha256"], entry["size_bytes"] = _sha(data), len(data)

    _write_manifest(copy, change)


def _case_derived_list(copy, world):
    lines = (copy / "allowlist_fold1.txt").read_text().splitlines()[1:]
    _replace_file(copy, "allowlist_fold1.txt", "".join(f"{line}\n" for line in lines).encode())
    return ("1 files differ from the SHA-256 their receipts or the cohort audit record, or from their derivation: "
            "allowlist_fold1.txt")


def _case_copied_input(copy, world):
    relative = f"external/{world.stems['P1002']}/residue_features.json"
    payload = json.loads((copy / relative).read_text())
    payload["note"] = "changed after the audit"
    _replace_file(copy, relative, (json.dumps(payload) + "\n").encode())
    return ("1 files differ from the SHA-256 their receipts or the cohort audit record, or from their derivation: "
            "1 under external/")


def _case_role(copy, world):
    def change(payload):
        entry = next(entry for entry in payload["files"] if entry["role"] == overlay.ROLE_STRUCTURE)
        entry["role"] = overlay.ROLE_COHORT

    _write_manifest(copy, change)
    return "1 manifest entries carry another role than their input: 1 under structure_store/"


def _case_per_fold(copy, world):
    _write_manifest(copy, lambda payload: payload["counts"]["per_fold"]["3"].update(
        n_eligible=payload["counts"]["per_fold"]["3"]["n_eligible"] + 1))
    return "the manifest's per-fold eligible counts differ from the overlay's memberships"


def _case_counts(copy, world):
    _write_manifest(copy, lambda payload: payload["counts"].update(n_proteins=payload["counts"]["n_proteins"] + 1))
    return "the manifest's protein or file count differs from the overlay"


def _case_cohort_record(copy, world):
    _write_manifest(copy, lambda payload: payload["inputs"]["cohort"].update(receipt_sha256="0" * 64))
    return "the overlay's cohort receipt differs from the one the manifest records"


def _case_r2_record(copy, world):
    _write_manifest(copy, lambda payload: payload["inputs"]["r2_lists"].update(receipt_sha256="0" * 64))
    return "the overlay's R2-list receipt differs from the one the manifest records"


def _case_identity_record(copy, world):
    _write_manifest(copy, lambda payload: payload["inputs"]["identity"]["1"]["record"].update(nodes_sha256="0" * 64))
    return "the fold-1 identity copy differs from the chain the manifest records"


def _case_unneeded(copy, world):
    data = b"a note\n"
    (copy / "cohort" / "extra_note.txt").write_bytes(data)
    _write_manifest(copy, lambda payload: payload["files"].append(
        {"path": "cohort/extra_note.txt", "sha256": _sha(data), "size_bytes": len(data), "role": overlay.ROLE_COHORT,
         "origin": {"kind": "copy", "root": "cohort", "path": "extra_note.txt"}}))
    return "1 manifest entries are not needed by the overlay's inputs: cohort/extra_note.txt"


def _case_needed(copy, world):
    (copy / "eligible_fold2.txt").unlink()
    _write_manifest(copy, lambda payload: payload.update(
        files=[entry for entry in payload["files"] if entry["path"] != "eligible_fold2.txt"]))
    return "1 files the overlay's inputs need are not in the manifest: eligible_fold2.txt"


VERIFY_CASES = {"derived_list": _case_derived_list, "copied_input": _case_copied_input, "role": _case_role,
                "per_fold": _case_per_fold, "counts": _case_counts, "cohort_record": _case_cohort_record,
                "r2_record": _case_r2_record, "identity_record": _case_identity_record, "unneeded": _case_unneeded,
                "needed": _case_needed}


@pytest.mark.parametrize("case", sorted(VERIFY_CASES))
def test_verification_re_derives_files_counts_and_records(built, world_template, tmp_path_factory, case):
    copy = copy_overlay(built, tmp_path_factory)
    expected = VERIFY_CASES[case](copy, world_template)
    reseal(copy)
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    assert expected in failed.value.problems, failed.value.problems
    assert not accessions_named("\n".join(failed.value.problems), world_template.stems)


@pytest.mark.parametrize("change", ["count", "unreadable"])
def test_verification_repeats_the_string_scan(built, world_template, tmp_path_factory, monkeypatch, change):
    copy = copy_overlay(built, tmp_path_factory)
    original = overlay.string_scan

    def changed(root, items):
        scan = original(root, items)
        if change == "count":
            return {**scan, "count": scan["count"] + 1}
        return {**scan, "n_unreadable": {overlay.ROLE_ESM: 1}}

    monkeypatch.setattr(overlay, "string_scan", changed)
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    assert failed.value.problems == ["the string scan of the per-protein inputs differs from the manifest's count"]


def test_verification_checks_the_build_receipt(built, tmp_path_factory):
    copy = copy_overlay(built, tmp_path_factory)
    _write_manifest(copy, lambda payload: payload.update(union_note="edited after the build"))
    reseal(copy, receipt=False)  # a consistent tar and .sha256 file, but the receipt still names the built ones
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    assert failed.value.problems == [f"{DEFAULT_OVERLAY_ID}.build_receipt.json does not bind this overlay "
                                     "(manifest_sha256, tar.sha256 differ)"]

    copy = copy_overlay(built, tmp_path_factory)
    side = overlay.overlay_side_paths(copy)
    payload = json.loads(side["build_receipt"].read_text())
    payload["provenance"] = {**payload["provenance"], "smoke": True}
    side["build_receipt"].write_text(json.dumps(payload))
    with pytest.raises(OverlayVerificationError, match=r"\(provenance differ\)"):
        verify_overlay(copy)
    side["build_receipt"].write_text("[]")
    with pytest.raises(OverlayVerificationError, match="is not a build receipt"):
        verify_overlay(copy)
    side["build_receipt"].unlink()
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    assert failed.value.problems == [f"the build receipt {DEFAULT_OVERLAY_ID}.build_receipt.json next to the overlay "
                                     "cannot be read (missing); a real overlay keeps it next to the tar"]


def test_a_symlink_in_the_overlay_is_a_problem(built, world_template, tmp_path_factory):
    copy = copy_overlay(built, tmp_path_factory)
    target = copy / "external" / world_template.stems["P1000"] / "residue_features.json"
    moved = copy.parent / "moved.json"
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(OverlayVerificationError, match="symlinks in the overlay"):
        verify_overlay(copy)


# ---------------------------------------------------------------- views and other readers of the overlay

def test_a_fold_view_builds_from_the_overlay_and_verifies_its_sources(built):
    root = built["root"]
    layout = SourceLayout.from_data_root(root, esm_dir=root / "esm", external_root=root / "external")
    for fold in (0, 2):
        work = root / "identity" / f"fold{fold}"
        assignment = json.loads((work / identity.INNER_ASSIGNMENT_RECEIPT).read_text())
        membership = work / assignment["outputs"]["membership"]["file"]
        result = build_view(fold=fold, layout=layout, out_dir=built["campaign"] / "views" / f"fold{fold}",
                            inner_membership=(membership, assignment["outputs"]["membership"]["sha256"]),
                            provenance=REAL_PROVENANCE)
        assert result["manifest"]["overlay"]["id"] == DEFAULT_OVERLAY_ID
        assert result["manifest"]["overlay"]["manifest_sha256"] == built["summary"]["manifest_sha256"]
        receipt = verify_view(built["campaign"] / "views" / f"fold{fold}", check_sources=True,
                              provenance=REAL_PROVENANCE)
        assert receipt["status"] == "verified" and receipt["checked_sources"] is True
        assert receipt["overlay"]["id"] == DEFAULT_OVERLAY_ID
        assert receipt["n_eligible"] == built["manifest"]["counts"]["per_fold"][str(fold)]["n_eligible"]


def test_overlay_r2_lists_load_in_the_comparator(built, world_template):
    import run_clean_split30_comparator as comparator

    for fold in range(N_FOLDS):
        accessions, _record = comparator.read_accession_allowlist(
            built["root"] / "r2_lists" / f"r2_allowlist_fold{fold}.txt", [])
        assert accessions == world_template.training(fold)


# ---------------------------------------------------------------- decision A1 option d: derived external features

PROXY = overlay.D2_DERIVED_FEATURE
# Standalone structures for the derivation: chain A as in the cohort fixture plus four residues of chain B.
UNIT_RESIDUES = (*DEFAULT_RESIDUES, ("ASP", (12.0, 2.0, 0.0), "B"), ("LYS", (2.0, 12.0, 3.0), "B"),
                 ("HIS", (-6.0, -9.0, 4.0), "B"), ("GLU", (9.0, -6.0, -3.0), "B"))
# Selected ZN 900 and MN 902; a copy of the ZN site from another donor 1.0 Å away; unselected CU and FE.
UNIT_METALS = ((900, "ZN", (0.0, 0.0, 0.0)), (901, "ZN", (1.0, 0.0, 0.0)), (902, "MN", (0.0, 0.0, 10.0)),
               (903, "CU", (-8.0, 0.0, 0.0)), (904, "FE", (6.0, 6.0, 6.0)))
UNIT_SELECTED = (("A", 900), ("A", 902))
UNIT_STEM = "P7001__chain_A__EC_1.1.1.1"


def _unit_structure(folder: Path, metals=UNIT_METALS, *, edit=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{UNIT_STEM}.pdb"
    write_structure(path, UNIT_RESIDUES, metals)
    if edit is not None:
        path.write_text("".join(edit(line) for line in path.read_text().splitlines(keepends=True)))
    return path


def _stored_features(structure: Path) -> bytes:
    """The external file the feature generator writes for ``structure`` (every metal present, no PROPKA), through
    its own writer (``generate_features.write_structure_payload``) in a sibling folder."""
    from feature_extraction.core import build_structure_feature_payload
    from feature_extraction.generate_features import write_structure_payload

    payload = build_structure_feature_payload(structure, include_propka=False)
    return write_structure_payload(structure.parent / "written_features", structure, payload).read_bytes()


def _module_run(structure: Path, keep_resseq, folder: Path) -> dict:
    """A full run of the feature module on a copy of ``structure`` holding only the metal atoms (HETATM records)
    of the residue numbers ``keep_resseq``."""
    from feature_extraction.core import generate_feature_map_for_structure

    folder.mkdir(parents=True, exist_ok=True)
    kept = folder / structure.name
    kept.write_text("".join(line for line in structure.read_text().splitlines(keepends=True)
                            if not line.startswith("HETATM") or int(line[22:26]) in keep_resseq))
    feature_map, _metadata = generate_feature_map_for_structure(kept, include_propka=False)
    return feature_map


def _residue_key(entry: dict) -> tuple:
    return entry["chain_id"], entry["resseq"], entry["icode"]


def _without_proxy(features: dict) -> dict:
    return {name: value for name, value in features.items() if name != PROXY}


def test_derived_charge_proxy_equals_a_full_run_without_the_other_metal_atoms(tmp_path):
    """The WP-8 check: selected metals, unselected metals and a same-site copy 1 Å from a selected metal."""
    from feature_extraction.constants import RESIDUE_CHARGE_PROXIES

    full = _unit_structure(tmp_path / "full")
    source = _stored_features(full)
    derived, stats = overlay.derive_external_features(source, full, structure_id=UNIT_STEM,
                                                      selected_sites=UNIT_SELECTED)
    reference = _module_run(full, (900, 902), tmp_path / "selected_only")
    with_copy = _module_run(full, (900, 901, 902), tmp_path / "with_copy")
    payload, original = json.loads(derived), json.loads(source)
    assert {key: value for key, value in payload.items() if key != "residues"} == {
        key: value for key, value in original.items() if key != "residues"}
    assert len(payload["residues"]) == len(original["residues"]) == len(reference) == len(UNIT_RESIDUES)
    changed = copy_differs = 0
    for new, old, residue in zip(payload["residues"], original["residues"], UNIT_RESIDUES):
        key = _residue_key(new)
        assert key == _residue_key(old) and _without_proxy(new["features"]) == _without_proxy(old["features"])
        assert new["features"][PROXY] == pytest.approx(reference[key][PROXY], rel=1e-12, abs=1e-12)
        charged = residue[0] in RESIDUE_CHARGE_PROXIES
        assert (new["features"][PROXY] != old["features"][PROXY]) == charged, key
        changed += charged
        copy_differs += abs(with_copy[key][PROXY] - reference[key][PROXY]) > 1e-6
    assert changed == 9 and copy_differs == 9  # the copy from another donor would have changed every charged residue
    assert stats == {"n_residues": 12, "n_residues_with_other_metal_terms": 9, "n_selected_metal_atoms": 2,
                     "n_other_metal_atoms": 3, "n_residues_changed": 9, "source_in_writer_form": True,
                     "bytes_unchanged": False}
    checked = overlay.check_derived_external_features(derived, full, structure_id=UNIT_STEM,
                                                      selected_sites=["A_900", "A_902"])
    assert checked == {key: stats[key] for key in ("n_residues", "n_residues_with_other_metal_terms",
                                                   "n_selected_metal_atoms", "n_other_metal_atoms")}


def test_a_file_without_other_metal_atoms_keeps_its_bytes(tmp_path):
    only = _unit_structure(tmp_path, [metal for metal in UNIT_METALS if metal[0] in (900, 902)])
    source = _stored_features(only)
    derived, stats = overlay.derive_external_features(source, only, structure_id=UNIT_STEM,
                                                      selected_sites=UNIT_SELECTED)
    assert derived == source and stats["bytes_unchanged"] and stats["n_residues_changed"] == 0
    assert stats["n_other_metal_atoms"] == 0 and stats["n_selected_metal_atoms"] == 2
    assert overlay.D2_DERIVED_FEATURE == __import__("data_structures").EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY


def _edit_payload(source: bytes, change) -> bytes:
    payload = json.loads(source)
    change(payload)
    return overlay.external_writer_bytes(payload)


def _shift_first_proxy(payload, delta):
    payload["residues"][0]["features"][PROXY] += delta


def _element_line(resseq: int, element: str):
    def edit(line: str) -> str:
        if line.startswith("HETATM") and int(line[22:26]) == resseq:
            return line.rstrip("\n")[:-2] + f"{element:>2}\n"
        return line
    return edit


def _alternate_locations(resseq: int, shift: float):
    """Two alternate locations of the metal atom ``resseq``: A (occupancy 0.40) where it was, B (0.60) ``shift`` Å
    along x. The trainer's parser (Bio.PDB) keeps the higher occupancy (B), the feature module's (biotite) the first
    location (A); element, residue and count agree, so only the coordinate check can tell them apart."""
    def edit(line: str) -> str:
        if not (line.startswith("HETATM") and int(line[22:26]) == resseq):
            return line
        x = float(line[30:38])
        first = line[:16] + "A" + line[17:54] + f"{0.40:>6.2f}" + line[60:]
        second = line[:16] + "B" + line[17:30] + f"{x + shift:>8.3f}" + line[38:54] + f"{0.60:>6.2f}" + line[60:]
        return first + second
    return edit


@pytest.mark.parametrize("case, reason", [
    ("parity", "charge_proxy_parity"),
    ("missing_residue", "residue_keys_differ"),
    ("duplicate_residue", "duplicate_residue_keys"),
    ("no_proxy", "malformed_residues"),
    ("not_json", "unreadable_json"),
    ("absent_site", "selected_site_without_one_metal_record"),
    ("two_atoms", "selected_site_without_one_metal_record"),
    ("element", "selected_metal_differs_from_trainer_record"),
    ("alternate_location", "selected_metal_differs_from_trainer_record"),
    ("no_sites", "no_selected_site"),
])
def test_derivation_refusals_carry_a_reason_token(tmp_path, monkeypatch, case, reason):
    metals, edit, sites = UNIT_METALS, None, UNIT_SELECTED
    if case == "two_atoms":  # a second atom in the selected MN residue (the records stay contiguous)
        metals = [*UNIT_METALS[:3], (902, "MN", (0.0, 0.0, 13.0), "MN1"), *UNIT_METALS[3:]]
    elif case == "element":
        edit = _element_line(902, "FE")  # residue MN (the trainer's symbol), element FE (the feature module's)
    elif case == "alternate_location":
        edit = _alternate_locations(902, 0.5)
    structure = _unit_structure(tmp_path, metals, edit=edit)
    source = _stored_features(structure)
    if case == "parity":
        source = _edit_payload(source, lambda payload: _shift_first_proxy(payload, 1e-3))
    elif case == "missing_residue":
        source = _edit_payload(source, lambda payload: payload["residues"].pop())
    elif case == "duplicate_residue":
        source = _edit_payload(source, lambda payload: payload["residues"].append(payload["residues"][0]))
    elif case == "no_proxy":
        source = _edit_payload(source, lambda payload: payload["residues"][3]["features"].pop(PROXY))
    elif case == "not_json":
        source = source[:-1]
    elif case == "absent_site":
        sites = (*UNIT_SELECTED, ("A", 950))
    elif case == "no_sites":
        sites = ()
    with pytest.raises(overlay.ExternalDerivationError) as refused:
        overlay.derive_external_features(source, structure, structure_id=UNIT_STEM, selected_sites=sites)
    assert refused.value.reason == reason and str(refused.value) == reason
    if case == "alternate_location":
        # The two parsers' atoms lie 0.5 Å apart: a tolerance above that lets the same file derive, so the
        # coordinate check (not the count or the element) is what refused it.
        monkeypatch.setattr(overlay, "D2_SELECTED_COORDINATE_TOLERANCE", 0.6)
        _derived, stats = overlay.derive_external_features(source, structure, structure_id=UNIT_STEM,
                                                           selected_sites=sites)
        assert stats["n_selected_metal_atoms"] == 2 and stats["n_other_metal_atoms"] == 3


def test_the_check_refuses_a_changed_derived_value(tmp_path):
    full = _unit_structure(tmp_path)
    derived, _stats = overlay.derive_external_features(_stored_features(full), full, structure_id=UNIT_STEM,
                                                       selected_sites=UNIT_SELECTED)
    changed = _edit_payload(derived, lambda payload: _shift_first_proxy(payload, 1e-3))
    with pytest.raises(overlay.ExternalDerivationError, match="^derived_value_differs$"):
        overlay.check_derived_external_features(changed, full, structure_id=UNIT_STEM, selected_sites=UNIT_SELECTED)
    # The stored (every-metal) file is no derived file either.
    with pytest.raises(overlay.ExternalDerivationError, match="^derived_value_differs$"):
        overlay.check_derived_external_features(_stored_features(full), full, structure_id=UNIT_STEM,
                                                selected_sites=UNIT_SELECTED)


@pytest.fixture(scope="module")
def built_d(built):
    """The real overlay, built under option d (the synthetic audit's recommendation)."""
    assert built["manifest"]["d2"]["option"] == "d"
    return built


def _selected_resseq(accession: str) -> tuple[int, ...]:
    return D2_SELECTED_RESSEQ if accession == D2_ACCESSION else (900,)


def test_option_d_counts_only_the_selected_metal_atoms(built_d, world_template, tmp_path):
    """End to end: selected sites from the training site tables, every residue of every protein against a full
    feature-module run on the overlay's structure without the other metal atoms."""
    root, manifest = built_d["root"], built_d["manifest"]
    entries = {entry["path"]: entry for entry in manifest["files"]}
    cohort_rows = _cohort_proteins(world_template)
    changed = Counter()
    for accession, stem in world_template.stems.items():
        relative = f"external/{stem}/residue_features.json"
        source = json.loads((world_template.layout.external_root / stem / "residue_features.json").read_text())
        derived = json.loads((root / relative).read_text())
        structure = next((root / "structure_store").rglob(f"{stem}.pdb"))
        reference = _module_run(structure, _selected_resseq(accession), tmp_path / accession)
        assert {key: value for key, value in derived.items() if key != "residues"} == {
            key: value for key, value in source.items() if key != "residues"}
        assert len(derived["residues"]) == len(source["residues"]) == len(reference)
        for new, old in zip(derived["residues"], source["residues"]):
            key = _residue_key(new)
            assert key == _residue_key(old) and _without_proxy(new["features"]) == _without_proxy(old["features"])
            assert new["features"][PROXY] == pytest.approx(reference[key][PROXY], rel=1e-12, abs=1e-12)
            changed[accession] += new["features"][PROXY] != old["features"][PROXY]
        entry = entries[relative]
        assert entry["sha256"] == _sha((root / relative).read_bytes())
        assert entry["origin"] == {
            "kind": "derived", "root": "external_root", "path": f"{stem}/residue_features.json",
            "rule": overlay.D2_DERIVATION_RULE, "source_sha256": cohort_rows[accession]["external_sha256"],
            "structure": structure.relative_to(root).as_posix(),
            "structure_sha256": cohort_rows[accession]["structure_sha256"],
            "selected_sites": [f"A_{resseq}" for resseq in _selected_resseq(accession)],
            "derived_sha256": entry["sha256"]}
    assert dict(+changed) == {"P1001": 5, D2_ACCESSION: 5}  # the charged residues of the two proteins
    assert manifest["d2"]["option"] == "d" and manifest["d2"]["policy"] == overlay.D2_POLICY["d"]
    assert manifest["d2"]["derivation"] == overlay.derivation_record({
        "n_files": 13, "n_files_with_other_metal_atoms": 2, "n_selected_metal_atoms": 14, "n_other_metal_atoms": 3,
        "n_residues": 104, "n_residues_with_other_metal_terms": 10})
    assert manifest["rules"] == overlay.overlay_rules("d") == {**overlay.RULES, "copies": overlay.D2_COPIES_RULE,
                                                               "hashes": overlay.D2_HASHES_RULE}
    # The cohort fixture writes compact JSON, so no source is in the writer's form and every file is rewritten.
    assert built_d["summary"]["d2_derivation"] == {
        **manifest["d2"]["derivation"]["totals"], "n_residues_changed": 10, "n_files_bytes_unchanged": 0,
        "n_sources_in_writer_form": 0}
    receipt = json.loads(Path(built_d["summary"]["build_receipt"]).read_text())
    assert receipt["d2_derivation"] == built_d["summary"]["d2_derivation"]
    assert not accessions_named(json.dumps(built_d["summary"]), world_template.stems)


def test_option_d_index_scan_and_counts(built_d, world_template):
    root, manifest = built_d["root"], built_d["manifest"]
    entries = {entry["path"]: entry for entry in manifest["files"]}
    cohort_rows = _cohort_proteins(world_template)
    with (root / overlay.LABEL_FREE_INDEX_NAME).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["accession"] for row in rows] == sorted(world_template.stems)
    for row in rows:
        stem = world_template.stems[row["accession"]]
        assert row["external_sha256"] == entries[f"external/{stem}/residue_features.json"]["sha256"]
        assert row["external_sha256"] != cohort_rows[row["accession"]]["external_sha256"]  # compact source, rewritten
        assert row["structure_sha256"] == cohort_rows[row["accession"]]["structure_sha256"]
    index = entries[overlay.LABEL_FREE_INDEX_NAME]
    assert index["origin"]["from"]["external_sha256"] == "the derived external feature files (decision A1 option d)"
    scan = manifest["string_scan"]
    assert scan["count"] == scan["cohort_a13_total"] >= 1
    assert manifest["counts"]["n_files"] == len(manifest["files"]) and manifest["counts"]["n_proteins"] == 13


def test_option_d_overlay_verifies(built_d):
    # Two option-d builds agree: test_two_builds_give_the_same_tar.
    result = verify_overlay(built_d["root"])
    assert result["status"] == "verified" and result["d2_option"] == "d"
    assert any("charge proxy recomputed" in check for check in result["checks"])


def _rewrite_derived(copy: Path, world: World, change) -> str:
    """Change P2001's derived file through ``change(payload)`` and record it consistently: manifest entry, origin,
    label-free index and its entry. Returns the file's overlay path."""
    relative = f"external/{world.stems[D2_ACCESSION]}/residue_features.json"
    data = _edit_payload((copy / relative).read_bytes(), change)
    (copy / relative).write_bytes(data)
    index_path = copy / overlay.LABEL_FREE_INDEX_NAME
    with index_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        if row["accession"] == D2_ACCESSION:
            row["external_sha256"] = _sha(data)
    index = overlay._tsv_bytes(overlay.LABEL_FREE_INDEX_COLUMNS, rows)
    index_path.write_bytes(index)

    def record(payload):
        for entry in payload["files"]:
            if entry["path"] == relative:
                entry.update(sha256=_sha(data), size_bytes=len(data))
                entry["origin"]["derived_sha256"] = _sha(data)
            elif entry["path"] == overlay.LABEL_FREE_INDEX_NAME:
                entry.update(sha256=_sha(index), size_bytes=len(index))

    _write_manifest(copy, record)
    return relative


def _case_d_value(copy, world):
    def change(payload):
        entry = next(entry for entry in payload["residues"] if entry["features"][PROXY] != 0.0)
        entry["features"][PROXY] += 0.5

    _rewrite_derived(copy, world, change)
    return ["1 derived external feature files do not re-derive from the overlay's structures and site tables "
            "(1 derived_value_differs)"]


def _case_d_same_values(copy, world):
    # Any consistently recorded byte change that keeps the values is only bound by the recorded SHA-256 values.
    _rewrite_derived(copy, world, lambda payload: payload.update(note="kept values"))
    return []


def _case_d_origin(copy, world):
    relative = f"external/{world.stems['P1000']}/residue_features.json"
    _write_manifest(copy, lambda payload: next(entry for entry in payload["files"] if entry["path"] == relative)[
        "origin"].update(source_sha256="0" * 64))
    return ["1 derived files do not carry their derivation record (rule, source and derived SHA-256, structure, "
            "selected sites): 1 under external/"]


def _case_d_unrecorded_bytes(copy, world):
    relative = f"external/{world.stems['P1000']}/residue_features.json"
    data = json.dumps(json.loads((copy / relative).read_bytes())).encode()  # same values, other bytes
    _replace_file(copy, relative, data)
    return ["1 files differ from the SHA-256 their receipts or the cohort audit record, or from their derivation: "
            "1 under external/",
            "1 derived files do not carry their derivation record (rule, source and derived SHA-256, structure, "
            "selected sites): 1 under external/"]


def _case_d_totals(copy, world):
    _write_manifest(copy, lambda payload: payload["d2"]["derivation"]["totals"].update(n_other_metal_atoms=4))
    return ["the manifest's derivation record differs from a re-derivation from the overlay's copies"]


def _case_d_option(copy, world):
    _write_manifest(copy, lambda payload: payload["d2"].update(option="a"))
    return ["the overlay manifest records D2 option 'a'; only ['b', 'd'] are built and verified"]


D2_BLOCK_PROBLEM = ("the manifest's D2 block differs from this code's policy text for option {option} or from the "
                    "overlay's cohort receipt")


def _case_d_policy(copy, world):
    _write_manifest(copy, lambda payload: payload["d2"].update(policy="edited after the build"))
    return [D2_BLOCK_PROBLEM.format(option="d")]


def _case_d_recommendation(copy, world):
    _write_manifest(copy, lambda payload: payload["d2"].update(cohort_a1_recommended_option="b",
                                                               follows_cohort_recommendation=False))
    return [D2_BLOCK_PROBLEM.format(option="d")]


def _case_d_no_derivation(copy, world):
    _write_manifest(copy, lambda payload: payload["d2"].pop("derivation"))
    return [D2_BLOCK_PROBLEM.format(option="d"),
            "the manifest's derivation record differs from a re-derivation from the overlay's copies"]


def _case_d_rules(copy, world):
    _write_manifest(copy, lambda payload: payload["rules"].update(copies=overlay.RULES["copies"]))
    return ["the manifest's rules differ from this code's rules for D2 option d"]


def _case_d_real_b(copy, world):
    # A real overlay that states option b (consistently) over files derived under d: refused while its inputs load.
    def change(payload):
        payload["d2"], payload["rules"] = overlay.d2_block("b", "d"), overlay.overlay_rules("b")

    _write_manifest(copy, change)
    return ["the overlay's own copies do not load as inputs: A real overlay takes the D2 option the cohort audit "
            "recommends for decision A1 ('d'); option 'b' is refused (a smoke overlay may take either built option)."]


D_VERIFY_CASES = {"value": _case_d_value, "same_values": _case_d_same_values, "origin": _case_d_origin,
                  "unrecorded_bytes": _case_d_unrecorded_bytes, "totals": _case_d_totals, "option": _case_d_option,
                  "policy": _case_d_policy, "recommendation": _case_d_recommendation,
                  "no_derivation": _case_d_no_derivation, "rules": _case_d_rules, "real_b": _case_d_real_b}


@pytest.mark.parametrize("case", sorted(D_VERIFY_CASES))
def test_option_d_verification_checks_records_and_values(built_d, world_template, tmp_path_factory, case):
    copy = copy_overlay(built_d, tmp_path_factory)
    expected = D_VERIFY_CASES[case](copy, world_template)
    reseal(copy)
    if not expected:
        assert verify_overlay(copy)["status"] == "verified"
        return
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    assert failed.value.problems == expected
    assert not accessions_named("\n".join(failed.value.problems), world_template.stems)


@pytest.mark.parametrize("change, expected", [
    ("policy", D2_BLOCK_PROBLEM.format(option="b")),
    ("derivation", D2_BLOCK_PROBLEM.format(option="b")),
    ("rules", "the manifest's rules differ from this code's rules for D2 option b"),
])
def test_option_b_verification_checks_its_d2_block_and_rules(built_b, world_template, tmp_path_factory, change,
                                                             expected):
    copy = copy_overlay(built_b, tmp_path_factory)
    assert verify_overlay(copy)["status"] == "verified"

    def edit(payload):
        if change == "policy":
            payload["d2"]["policy"] = overlay.D2_POLICY["d"]
        elif change == "derivation":
            payload["d2"]["derivation"] = overlay.derivation_record(dict.fromkeys(overlay.DERIVATION_TOTAL_KEYS, 0))
        else:
            payload["rules"] = overlay.overlay_rules("d")

    _write_manifest(copy, edit)
    reseal(copy)
    with pytest.raises(OverlayVerificationError) as failed:
        verify_overlay(copy)
    assert failed.value.problems == [expected]


def test_a_real_overlay_takes_the_option_the_cohort_audit_recommends(world_template, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "d2recommended")
    with pytest.raises(OverlayError) as refused:
        build(world_template, campaign, d2_option="b")
    assert str(refused.value) == ("A real overlay takes the D2 option the cohort audit recommends for decision A1 "
                                  "('d'); option 'b' is refused (a smoke overlay may take either built option).")
    assert_nothing_written(campaign)


def test_the_view_states_every_built_d2_option_with_the_builders_text():
    from ec_clean30 import views

    assert views.OVERLAY_D2_OPTIONS == overlay.D2_OPTIONS_BUILT == tuple(overlay.D2_POLICY) == ("b", "d")
    assert overlay.D2_POLICY is views.OVERLAY_D2_POLICY and set(overlay.D2_OPTIONS_BUILT) < set(overlay.D2_OPTIONS)


def test_a_fold_view_from_an_option_d_overlay_states_its_policy(built_d):
    from ec_clean30 import views

    root = built_d["root"]
    layout = SourceLayout.from_data_root(root, esm_dir=root / "esm", external_root=root / "external")
    work = root / "identity" / "fold1"
    assignment = json.loads((work / identity.INNER_ASSIGNMENT_RECEIPT).read_text())
    out = built_d["campaign"] / "views" / "fold1"
    result = build_view(fold=1, layout=layout, out_dir=out,
                        inner_membership=(work / assignment["outputs"]["membership"]["file"],
                                          assignment["outputs"]["membership"]["sha256"]),
                        provenance=REAL_PROVENANCE)
    manifest = result["manifest"]
    assert manifest["overlay"]["d2_option"] == "d" and manifest["overlay"]["d2_policy"] == overlay.D2_POLICY["d"]
    assert manifest["external_features_policy"] == views.OVERLAY_EXTERNAL_FEATURES_POLICY.format(
        overlay_id=DEFAULT_OVERLAY_ID, option="d", policy=overlay.D2_POLICY["d"])
    assert "including metals outside the selected sites (decision D2 open" not in manifest["external_features_policy"]
    for item in manifest["files"]:
        if item["role"] == "external_features":
            assert (out / item["path"]).read_bytes() == (root / item["path"]).read_bytes()
    receipt = verify_view(out, check_sources=True, provenance=REAL_PROVENANCE)
    assert receipt["status"] == "verified" and receipt["overlay"]["d2_option"] == "d"


def test_option_d_refusals_name_no_protein(world, tmp_path_factory, monkeypatch):
    stems = world.stems
    campaign = campaign_root(tmp_path_factory, "derivationrefusals")
    source = world.layout.external_root / stems[D2_ACCESSION] / "residue_features.json"
    original = source.read_bytes()
    source.write_bytes(original + b" ")
    with pytest.raises(OverlayError, match="^1 inputs differ from the SHA-256 the cohort audit or their receipts "
                                           "record: 1 under external/$"):
        build(world, campaign, d2_option="d")
    source.write_bytes(original)

    derive = overlay.derive_external_features

    def failing(data, structure, *, structure_id, selected_sites):
        if structure_id == stems[D2_ACCESSION]:
            raise overlay.ExternalDerivationError("charge_proxy_parity")
        return derive(data, structure, structure_id=structure_id, selected_sites=selected_sites)

    monkeypatch.setattr(overlay, "derive_external_features", failing)
    with pytest.raises(OverlayError) as refused:
        build(world, campaign, d2_option="d")
    assert str(refused.value) == ("1 external feature files cannot be derived under option d (1 charge_proxy_parity); "
                                  "no file is named because the names carry structure stems.")
    monkeypatch.setattr(overlay, "derive_external_features", derive)

    sites = overlay._selected_sites
    monkeypatch.setattr(overlay, "_selected_sites", lambda folds, union: {
        accession: (*value, ("A", 950)) if accession == "P1000" else value
        for accession, value in sites(folds, union).items()})
    with pytest.raises(OverlayError, match=r"\(1 selected_site_without_one_metal_record\)") as refused:
        build(world, campaign, d2_option="d")
    assert not accessions_named(str(refused.value), stems)
    assert_nothing_written(campaign)


def test_selected_sites_come_from_the_site_tables_as_the_trainer_reads_them(world_template):
    from dataclasses import replace

    folds, union = overlay._union(world_template.layout, overlay._load_cohort(world_template.cohort, smoke=False))
    selected = overlay._selected_sites(folds, union)
    assert selected == {accession: tuple(("A", resseq) for resseq in _selected_resseq(accession))
                        for accession in world_template.stems}
    # Another EC in the structure name selects nothing (key mode accession_ec), which option d refuses.
    other = replace(union["P1000"], structure_name="P1000__chain_A__EC_9.9.9.9.pdb")
    with pytest.raises(OverlayError, match="^1 proteins have no selected site in the training site tables"):
        overlay._selected_sites(folds, {**union, "P1000": other})


# ---------------------------------------------------------------- refusals (nothing is left behind)

def test_an_existing_output_is_refused(built, world_template):
    with pytest.raises(OverlayError, match="already exist"):
        build(world_template, built["campaign"])


def test_a_real_overlay_needs_its_place_and_name(world_template, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "placement")
    with pytest.raises(OverlayError, match="lives at <campaign root>/overlay"):
        build_overlay(out_dir=campaign / "views" / DEFAULT_OVERLAY_ID, cohort_dir=world_template.cohort,
                      identity_root=world_template.identity_root, r2_lists_dir=world_template.r2, d2_option="b",
                      provenance=REAL_PROVENANCE)
    with pytest.raises(OverlayError, match="is named ecc30-overlay-v<N>"):
        build(world_template, campaign, overlay_id="my-overlay")
    other = tmp_path_factory.mktemp("othername") / "campaign_x"
    with pytest.raises(OverlayError, match="lives at <campaign root>/overlay"):
        build(world_template, other)
    assert_nothing_written(campaign)
    assert not (campaign / "views").exists() and not other.exists()


def test_test_named_and_comparator_refused_paths(world, tmp_path_factory):
    base = tmp_path_factory.mktemp("namedpaths")
    with pytest.raises(HeldOutInputError):
        build_overlay(out_dir=base / "held_test_overlay", cohort_dir=world.cohort, identity_root=world.identity_root,
                      r2_lists_dir=world.r2, d2_option="b", provenance=SMOKE_PROVENANCE, layout=world.layout,
                      smoke=True)
    with pytest.raises(HeldOutInputError, match="comparator refuses"):
        build_overlay(out_dir=base / "folds" / "smoke-overlay", cohort_dir=world.cohort,
                      identity_root=world.identity_root, r2_lists_dir=world.r2, d2_option="b",
                      provenance=SMOKE_PROVENANCE, layout=world.layout, smoke=True)
    renamed = world.campaign / "cohort_test"
    world.cohort.rename(renamed)
    with pytest.raises(HeldOutInputError):
        build(world, campaign_root(tmp_path_factory, "testnamedinput"), cohort_dir=renamed)
    assert not list(base.iterdir())


def test_an_overlay_inside_a_shared_root_is_refused(world):
    with pytest.raises(OverlayError, match="shared or test-bearing data root"):
        build_overlay(out_dir=world.data_root / "structure_store" / "smoke-overlay", cohort_dir=world.cohort,
                      identity_root=world.identity_root, r2_lists_dir=world.r2, d2_option="b",
                      provenance=SMOKE_PROVENANCE, layout=world.layout, smoke=True)


def test_unbuilt_d2_options_are_refused(world_template, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "d2options")
    for option, message in (("a", "is not built"), ("c", "never used")):
        with pytest.raises(OverlayError, match=message):
            build(world_template, campaign, d2_option=option)
    assert_nothing_written(campaign)


def test_an_identity_folder_with_complement_outputs_is_refused(world, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "complement")
    for name in ("fasta_receipt_complement.json", "strata_fold3.tsv"):
        (world.work(3) / name).write_text("{}")
        # The overlay's own refusal, before any identity loader runs (the R2-list loader has the same guard).
        with pytest.raises(OverlayError, match=r"^The fold-3 identity folder .* holds complement \(test-side\) outputs "
                                               r".*; the overlay copies training-side identity folders only"):
            build(world, campaign)
        (world.work(3) / name).unlink()
    assert_nothing_written(campaign)


def _edit_json(path: Path, change) -> None:
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_the_cohort_audit_must_have_passed_and_be_real(world, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "cohortstate")
    receipt = world.cohort / COHORT_RECEIPT
    original = receipt.read_bytes()
    _edit_json(receipt, lambda payload: payload.update(status="stopped", failed_checks=["all_structures_load"]))
    with pytest.raises(OverlayError, match="did not pass"):
        build(world, campaign)
    receipt.write_bytes(original)
    _edit_json(receipt, lambda payload: payload.update(smoke=True))
    with pytest.raises(OverlayError, match="engineering smoke"):
        build(world, campaign)
    receipt.write_bytes(original)
    (world.cohort / "quarantine.tsv").write_text("accession\ta4_outcome\n")
    with pytest.raises(OverlayError, match="quarantined"):
        build(world, campaign)
    (world.cohort / "quarantine.tsv").unlink()
    proteins = world.cohort / "cohort_proteins.tsv"
    proteins.write_bytes(proteins.read_bytes() + b"\n")
    with pytest.raises(OverlayError, match="changed after the cohort audit"):
        build(world, campaign)
    assert_nothing_written(campaign)


def test_changed_or_linked_inputs_are_refused_without_naming_proteins(world, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "changedinputs")
    payload = world.layout.esm_dir / f"{world.stems['P2002']}_chain_A_esmc.pt"
    original = payload.read_bytes()
    payload.write_bytes(original + b"\0")
    with pytest.raises(OverlayError, match="1 inputs differ from the SHA-256") as refused:
        build(world, campaign)
    assert "under esm/" in str(refused.value) and not accessions_named(str(refused.value), world.stems)
    payload.write_bytes(original)

    external = world.layout.external_root / world.stems["P1003"] / "residue_features.json"
    moved = world.base / "moved_features.json"
    external.rename(moved)
    external.symlink_to(moved)
    with pytest.raises(OverlayError, match="cannot be copied") as refused:
        build(world, campaign)
    assert "1 external_features" in str(refused.value) and not accessions_named(str(refused.value), world.stems)
    external.unlink()
    moved.rename(external)

    manifest = world.data_root / "CLEAN_30_train_test_split_1" / "train" / "structure_manifest.csv"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(OverlayError, match="differs from the one the cohort audit read"):
        build(world, campaign)
    assert_nothing_written(campaign)


def test_identity_and_r2_inputs_must_match(world, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "identityr2")
    table = world.work(2) / "components_inner.tsv"
    original = table.read_bytes()
    table.write_bytes(original + b"\n")
    with pytest.raises(OverlayError, match="fold-2 identity chain does not load"):
        build(world, campaign)
    table.write_bytes(original)

    receipt = world.r2 / r2_lists.RECEIPT_NAME
    original = receipt.read_bytes()
    _edit_json(receipt, lambda payload: payload["folds"]["1"]["inputs"].update(nodes_sha256="0" * 64))
    with pytest.raises(OverlayError, match="Fold 1: the R2 lists were built from other identity tables"):
        build(world, campaign)
    receipt.write_bytes(original)
    _edit_json(receipt, lambda payload: payload["provenance"].update(smoke=True))
    with pytest.raises(OverlayError, match="R2 lists are an engineering smoke"):
        build(world, campaign)
    receipt.write_bytes(original)
    (world.r2 / "r2_allowlist_fold4.txt").write_text("P1000\n")
    with pytest.raises(OverlayError, match="R2 lists do not verify"):
        build(world, campaign)
    assert_nothing_written(campaign)


def test_identity_of_other_training_files_is_refused(world, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "swappedfolds")
    with pytest.raises(OverlayError, match="Fold 3 .*belongs to fold 4|Fold 3: the identity chain"):
        build(world, campaign, fold_work=[f"3={world.work(4)}", f"4={world.work(3)}"])
    assert_nothing_written(campaign)


def test_identity_built_from_other_site_table_bytes_is_refused(world, tmp_path_factory):
    """Same proteins, other donors: the chain links P1000 and P1001, the copied site tables do not."""
    folds_dir = world.data_root / "CLEAN_30_main" / "folds"
    other = world.base / "other_folds"
    shutil.copytree(folds_dir, other)
    for fold in range(N_FOLDS):
        path = other / f"CLEAN_30_train_test_split_{fold}_train.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["structure"] in ("P1000", "P1001"):
                row[DONOR_COLUMN] = "9ZZZ"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TRAIN_COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    work = world.base / "other_identity" / "fold0"
    binary = fake_binary(world.base / "fakebin_other")
    assert sha256_file(binary) == world.binary_sha
    build_inputs(work, 0, official_split_dir=world.base / "source" / "split30", metallo_folds_dir=other,
                 metallo_manifest_root=world.data_root)
    run_search(work, 0, binary=binary, expected_sha256=world.binary_sha, runner=FakeMMseqs(hits={}))
    build_edges(work, 0)
    build_inner_components(work, 0)
    assign_inner(work, 0, metallo_folds_dir=other, metallo_manifest_root=world.data_root)
    certify_inner(work, 0, metallo_folds_dir=other, metallo_manifest_root=world.data_root)
    campaign = campaign_root(tmp_path_factory, "othertables")
    with pytest.raises(OverlayError, match=r"Fold 0: the identity chain read other training files than the cohort "
                                           r"audit \(SHA-256 differs for the site tables of folds \[") as refused:
        build(world, campaign, fold_work=[f"0={work}"])
    assert not accessions_named(str(refused.value), world.stems)
    assert_nothing_written(campaign)


def _edit_fasta_tables(world):
    _edit_json(world.work(2) / "fasta_receipt_metallo_train.json",
               lambda payload: payload["sources"]["metallo_training_files"]["3"].update(sha256="0" * 64))
    return r"Fold 2: the identity chain read .*site tables of folds \[3\] and the structure manifests of folds \[\]"


def _edit_fasta_manifests(world):
    _edit_json(world.work(2) / "fasta_receipt_metallo_train.json",
               lambda payload: payload["sources"]["metallo_structure_manifests"]["4"].update(sha256="0" * 64))
    return r"Fold 2: the identity chain read .*site tables of folds \[\] and the structure manifests of folds \[4\]"


def _edit_certificate(world):
    _edit_json(world.work(2) / identity.INNER_CERTIFICATE,
               lambda payload: payload["inputs"]["metallo_training_file"].update(sha256="0" * 64))
    return "Fold 2: the inner certificate was made from another fold-2 training site table"


def _edit_assignment(world):
    work = world.work(2)
    _edit_json(work / identity.INNER_ASSIGNMENT_RECEIPT,
               lambda payload: payload["inputs"]["metallo_training_file"].update(sha256="0" * 64))
    _edit_json(work / identity.INNER_CERTIFICATE, lambda payload: payload.update(
        assignment_receipt_sha256=sha256_file(work / identity.INNER_ASSIGNMENT_RECEIPT)))
    return "Fold 2: the assignment receipt was made from another fold-2 training site table"


IDENTITY_RECEIPT_EDITS = {"fasta_tables": _edit_fasta_tables, "fasta_manifests": _edit_fasta_manifests,
                          "certificate": _edit_certificate, "assignment": _edit_assignment}


@pytest.mark.parametrize("edit", sorted(IDENTITY_RECEIPT_EDITS))
def test_identity_receipts_must_name_the_audited_training_files(world, tmp_path_factory, edit):
    message = IDENTITY_RECEIPT_EDITS[edit](world)
    campaign = campaign_root(tmp_path_factory, f"binding{edit}")
    with pytest.raises(OverlayError, match=message) as refused:
        build(world, campaign)
    assert not accessions_named(str(refused.value), world.stems)
    assert_nothing_written(campaign)


@pytest.mark.parametrize("field, message", [
    ("accessions", "Fold 2: the identity chain's node table and the fold's training manifest name different proteins"),
    ("eligible", "Fold 2: the certified inner membership does not cover exactly the fold's training structures with "
                 "one EC class"),
    ("several", "Fold 2: the assignment receipt's several-class list differs from the fold's training structures "
                "with several EC classes"),
])
def test_identity_must_cover_the_folds_training_side(world_template, tmp_path_factory, monkeypatch, field, message):
    original = overlay._union

    def changed(layout, cohort_info):
        folds, union = original(layout, cohort_info)
        side = dict(folds[2])
        if field == "accessions":
            side["accessions"] = frozenset(sorted(side["accessions"])[1:])
        elif field == "eligible":
            side["eligible"] = side["eligible"][1:]
        else:
            side["several"] = [*side["several"], side["eligible"][0]]
        return {**folds, 2: side}, union

    monkeypatch.setattr(overlay, "_union", changed)
    campaign = campaign_root(tmp_path_factory, f"coverage{field}")
    with pytest.raises(OverlayError) as refused:
        build(world_template, campaign)
    assert str(refused.value).startswith(message)
    assert_nothing_written(campaign)


def test_a_refused_inner_assignment_is_refused(world, tmp_path_factory):
    (world.work(1) / identity.INNER_ASSIGNMENT_REFUSAL).write_text("{}")
    campaign = campaign_root(tmp_path_factory, "refusedassignment")
    with pytest.raises(OverlayError, match="The fold-1 identity folder .* holds a refused inner assignment"):
        build(world, campaign)
    assert_nothing_written(campaign)


def _set_cohort_output(world: World, name: str, data: bytes) -> None:
    """Change a cohort output and the audit receipt's record of it, as if the audit had written it."""
    (world.cohort / name).write_bytes(data)
    _edit_json(world.cohort / COHORT_RECEIPT, lambda payload: payload["outputs"][name].update(sha256=_sha(data)))


def test_cohort_proteins_must_match_the_training_sides(world, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "cohortproteins")
    path = world.cohort / "cohort_proteins.tsv"
    original = path.read_bytes()
    lines = original.decode().splitlines(keepends=True)
    _set_cohort_output(world, "cohort_proteins.tsv", "".join(lines[:-1]).encode())
    with pytest.raises(OverlayError, match="13 proteins and cohort_proteins.tsv 12, with 1 not shared"):
        build(world, campaign)
    header = lines[0].rstrip("\n").split("\t")
    row = lines[1].rstrip("\n").split("\t")
    row[header.index("structure_sha256")] = "0" * 64
    _set_cohort_output(world, "cohort_proteins.tsv", "".join([lines[0], "\t".join(row) + "\n", *lines[2:]]).encode())
    with pytest.raises(OverlayError, match="^1 structures differ from the SHA-256 the cohort audit recorded"):
        build(world, campaign)
    assert_nothing_written(campaign)


def test_load_time_exclusions_are_refused(world, tmp_path_factory):
    (world.cohort / "cohort_exclusions_fold2.tsv").write_text("accession\tchain\tload_status\tload_reason\n")
    campaign = campaign_root(tmp_path_factory, "exclusions")
    with pytest.raises(OverlayError, match="found load-time exclusions"):
        build(world, campaign)
    assert_nothing_written(campaign)


@pytest.mark.parametrize("place, message", [
    ("outside", "1 structure paths lie outside the data root and 0 inside a folder name the overlay reserves"),
    ("reserved", "0 structure paths lie outside the data root and 1 inside a folder name the overlay reserves"),
])
def test_structures_must_lie_inside_the_data_root(world_template, tmp_path_factory, monkeypatch, place, message):
    original = overlay._data_relative
    stem = world_template.stems["P2001"]

    def relative(path, data_root):
        if Path(path).name.startswith(stem):
            return None if place == "outside" else f"esm/{Path(path).name}"
        return original(path, data_root)

    monkeypatch.setattr(overlay, "_data_relative", relative)
    campaign = campaign_root(tmp_path_factory, f"structures{place}")
    with pytest.raises(OverlayError, match=message) as refused:
        build(world_template, campaign)
    assert not accessions_named(str(refused.value), world_template.stems)
    assert_nothing_written(campaign)


def test_two_inputs_for_one_overlay_path_are_refused(world_template, tmp_path_factory, monkeypatch):
    monkeypatch.setattr(overlay, "esm_file_name", lambda stem: "one_chain_A_esmc.pt")
    campaign = campaign_root(tmp_path_factory, "samepath")
    with pytest.raises(OverlayError, match="^1 overlay paths would receive two inputs: 1 under esm/$"):
        build(world_template, campaign)
    assert_nothing_written(campaign)


def _r2_inner_record(payload):
    return payload["r2_inner"]["0"]["1"]["r1_inner_exclusions"]


def test_r2_lists_must_bind_this_cohort_and_these_identity_tables(world, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "r2binding")
    receipt = world.r2 / r2_lists.RECEIPT_NAME
    original = receipt.read_bytes()
    _edit_json(receipt, lambda payload: payload["cohort"].update(receipt_sha256="0" * 64))
    with pytest.raises(OverlayError, match="The R2 lists were built from another cohort audit receipt"):
        build(world, campaign)
    receipt.write_bytes(original)
    _edit_json(receipt, lambda payload: _r2_inner_record(payload).update(sha256="0" * 64))
    with pytest.raises(OverlayError, match="Fold 0, inner fold 1: the R2-inner list was built from other R1-inner"):
        build(world, campaign)
    receipt.write_bytes(original)
    allowlist = world.r2 / "r2_allowlist_fold4.txt"
    data = "".join(f"{line}\n" for line in allowlist.read_text().splitlines()[1:]).encode()
    allowlist.write_bytes(data)
    _edit_json(receipt, lambda payload: payload["outputs"]["r2_allowlist_fold4.txt"].update(
        sha256=_sha(data), n_records=payload["outputs"]["r2_allowlist_fold4.txt"]["n_records"] - 1))
    assert r2_lists.verify_r2_lists(world.r2)["status"] == "verified"
    with pytest.raises(OverlayError, match="Fold 4: the R2 allowlist differs from the identity chain's node table"):
        build(world, campaign)
    assert_nothing_written(campaign)


@pytest.mark.parametrize("d2_option", ["b", "d"])
@pytest.mark.parametrize("change, message", [
    ("count", r"The string scan counts \d+ test-named strings and the cohort audit \d+; both read the same bytes"),
    ("unreadable", "The string scan cannot read"),
])
def test_the_string_scan_must_match_the_cohort_audit(world_template, tmp_path_factory, monkeypatch, change, message,
                                                     d2_option):
    original = overlay.string_scan

    def changed(root, items):
        scan = original(root, items)
        if change == "count":
            return {**scan, "count": scan["count"] + 1}
        return {**scan, "n_unreadable": {overlay.ROLE_ESM: 1}}

    monkeypatch.setattr(overlay, "string_scan", changed)
    campaign = campaign_root(tmp_path_factory, f"scan{change}{d2_option}")
    with pytest.raises(OverlayError, match=message):
        build(world_template, campaign, **(SMOKE_B if d2_option == "b" else {}))
    assert_nothing_written(campaign)


def test_an_overlay_inside_the_repository_is_refused(world_template, tmp_path_factory, monkeypatch):
    """Against a stand-in checkout; the inputs are never read, so a broken check cannot write into a real one."""
    checkout = tmp_path_factory.mktemp("checkout")
    monkeypatch.setattr(overlay, "REPO_ROOT", checkout)

    def no_read(**_kwargs):
        raise AssertionError("the location check let the build read its inputs")

    monkeypatch.setattr(overlay, "_collect", no_read)
    options = {"cohort_dir": world_template.cohort, "identity_root": world_template.identity_root,
               "r2_lists_dir": world_template.r2, "d2_option": "b", "provenance": SMOKE_PROVENANCE, "smoke": True}
    with pytest.raises(OverlayError, match="inside the repository's tracked tree"):
        build_overlay(out_dir=checkout / "work" / "smoke-overlay", **options)
    with pytest.raises(AssertionError, match="let the build read"):  # the private data folder is allowed
        build_overlay(out_dir=checkout / "DeepMzyme_Data" / "smoke-overlay", **options)
    assert not list(checkout.iterdir())


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads files without permission")
def test_unreadable_inputs_are_counted_by_errno_without_paths(world, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "unreadable")
    payload = world.layout.esm_dir / f"{world.stems['P2002']}_chain_A_esmc.pt"
    os.chmod(payload, 0)
    try:
        with pytest.raises(OverlayError) as refused:
            build(world, campaign)
    finally:
        os.chmod(payload, 0o644)
    assert str(refused.value) == "Inputs cannot be copied: 1 esm_residue (PermissionError EACCES)"
    folder = world.layout.external_root / world.stems["P1003"]
    os.chmod(folder, 0)
    try:
        with pytest.raises(OverlayError) as refused:
            build(world, campaign)
    finally:
        os.chmod(folder, 0o755)
    assert str(refused.value) == "Inputs cannot be copied: 1 external_features (not readable (PermissionError EACCES))"
    assert_nothing_written(campaign)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads files without permission")
def test_unreadable_overlay_files_are_counted_by_errno_without_paths(built, world_template, tmp_path_factory):
    copy = copy_overlay(built, tmp_path_factory)
    target = copy / "external" / world_template.stems["P1002"] / "residue_features.json"
    os.chmod(target, 0)
    try:
        with pytest.raises(OverlayVerificationError) as failed:
            verify_overlay(copy)
    finally:
        os.chmod(target, 0o644)
    problems = failed.value.problems
    assert "1 files cannot be read (1 PermissionError EACCES): 1 under external/" in problems
    assert ("the tar cannot be re-created from the overlay folder (An overlay file under external/ cannot be archived "
            "(PermissionError EACCES).)") in problems
    assert not accessions_named("\n".join(problems), world_template.stems)


# ---------------------------------------------------------------- fold-work mapping, smoke builds and the runner

def test_fold_work_maps_a_rerun_folder_into_identity_fold_k(world, tmp_path_factory):
    rerun = world.identity_root / "fold3_cap1-3"
    world.work(3).rename(rerun)
    campaign = campaign_root(tmp_path_factory, "foldwork")
    with pytest.raises(OverlayError, match="fold-3 identity folder is not a folder|not found"):
        build(world, campaign)
    summary = build(world, campaign, fold_work=[f"3={rerun}"])
    root = Path(summary["overlay"])
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    assert manifest["sources"]["identity_folds"]["3"] == {"path": str(rerun), "override": True}
    assert manifest["sources"]["identity_folds"]["2"]["override"] is False
    assert (root / "identity" / "fold3" / "inner_membership_fold3.csv").is_file()
    assert not (root / "identity" / "fold3_cap1-3").exists()
    assert verify_overlay(root)["status"] == "verified"


def test_a_smoke_overlay_can_live_anywhere_and_verifies(world_template, tmp_path_factory):
    out = tmp_path_factory.mktemp("smokeplace") / "smoke-overlay"
    summary = build_overlay(out_dir=out, cohort_dir=world_template.cohort, identity_root=world_template.identity_root,
                            r2_lists_dir=world_template.r2, d2_option="b", provenance=SMOKE_PROVENANCE, smoke=True)
    assert summary["smoke"] is True
    result = verify_overlay(out)
    assert result["status"] == "verified" and result["smoke"] is True and result["build_receipt_sha256"]
    Path(summary["build_receipt"]).unlink()  # a smoke overlay may lack its build receipt; a real one may not
    result = verify_overlay(out)
    assert result["status"] == "verified" and result["build_receipt"] is None


def _runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    overlay.add_overlay_arguments(sub.add_parser("overlay", help=overlay.OVERLAY_HELP))
    return parser


def test_runner_actions_wire_to_build_and_verify(world_template, tmp_path_factory, monkeypatch, capsys):
    assert issubclass(OverlayError, ViewError)
    calls = []
    monkeypatch.setattr(overlay, "real_action_git_state",
                        lambda *, smoke: calls.append(smoke) or {"head": "f" * 40, "clean": True})
    campaign = campaign_root(tmp_path_factory, "runner")
    out = campaign / "overlay" / DEFAULT_OVERLAY_ID
    parser = _runner_parser()
    build_argv = ["overlay", "build", "--out", str(out), "--cohort", str(world_template.cohort), "--identity-root",
                  str(world_template.identity_root), "--r2-lists", str(world_template.r2), "--d2-option"]
    args = parser.parse_args([*build_argv, "b"])
    with pytest.raises(OverlayError, match=r"takes the D2 option the cohort audit recommends for decision A1 \('d'\)"):
        args.handler(args)
    assert calls == [False] and not out.exists()
    calls.clear()
    args = parser.parse_args([*build_argv, "d", "--fold-work", f"1={world_template.work(1)}"])
    assert (args.overlay_action, args.fold_work, args.smoke) == ("build", [f"1={world_template.work(1)}"], False)
    assert args.handler(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "built" and printed["overlay"] == str(out) and calls == [False]
    assert not accessions_named(json.dumps(printed), world_template.stems)
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["provenance"]["git"]["head"] == "f" * 40 and "created_utc" not in json.dumps(manifest)

    args = parser.parse_args(["overlay", "verify", "--overlay", str(out)])
    assert args.handler(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified" and calls == [False, False]

    make_writable(out)
    (out / overlay.LABEL_FREE_INDEX_NAME).write_text("accession\n")
    assert args.handler(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"

    def dirty(*, smoke):
        raise DirtyTreeError("dirty")

    monkeypatch.setattr(overlay, "real_action_git_state", dirty)
    second = campaign_root(tmp_path_factory, "runnerdirty")
    args = parser.parse_args(["overlay", "build", "--out", str(second / "overlay" / DEFAULT_OVERLAY_ID),
                              "--cohort", str(world_template.cohort), "--identity-root",
                              str(world_template.identity_root), "--r2-lists", str(world_template.r2),
                              "--d2-option", "b"])
    with pytest.raises(DirtyTreeError):
        args.handler(args)
    assert not second.exists()
    args = parser.parse_args(["overlay", "build", "--out", str(second / "overlay" / DEFAULT_OVERLAY_ID),
                              "--cohort", str(world_template.cohort), "--identity-root",
                              str(world_template.identity_root), "--r2-lists", str(world_template.r2),
                              "--d2-option", "b", "--fold-work", "2=/somewhere/held_test_fold2"])
    with pytest.raises(HeldOutInputError):
        args.handler(args)
    with pytest.raises(SystemExit):
        parser.parse_args(["overlay", "build", "--out", "x", "--cohort", "c", "--identity-root", "i",
                           "--r2-lists", "r", "--d2-option", "c"])


def test_module_imports_without_torch():
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
            "import ec_clean30.overlay; print('torch' in sys.modules)")
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"
