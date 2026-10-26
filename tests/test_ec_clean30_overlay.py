"""Data overlay of the EC class vs CLEAN30 campaign: builder and verifier (synthetic data only; no dataset access).

The fixture writes a five-fold synthetic data root with the cohort audit's test builder, runs the
cohort audit, the training-side identity chain (fasta, a fake pinned MMseqs2 search, edges, inner
components, assignment, certification and fold 0's R1-inner lists) and the R2 lists on it, and
builds overlays from those outputs. Every folder name is neutral: the tools refuse test-named
path parts. No real campaign file is read.
"""
from __future__ import annotations

import argparse
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
from test_ec_clean30_cohort import AMBIGUOUS, DEFAULT_RESIDUES, TRAIN_COLUMNS, build_cohort  # noqa: E402
from test_ec_clean30_identity import FakeMMseqs, fake_binary, random_sequence  # noqa: E402

N_FOLDS = 5
REAL_PROVENANCE = {"smoke": False, "note": "synthetic test provenance"}
SMOKE_PROVENANCE = {"smoke": True, "note": "synthetic test provenance"}
# One backbone-only residue far from the metal differs per protein, so no two sequences are equal (the
# identity tool links equal sequences into one component).
VARIABLE_RESIDUES = ("ALA", "GLY", "LEU", "ILE", "VAL", "MET", "PHE", "TRP", "PRO", "THR", "ASN", "GLN", "TYR")
A13_ACCESSION = "P3002"
NON_METALLO = ("N0001", "N0002")


def overlay_proteins() -> dict[str, dict]:
    """Twelve single-class proteins (EC classes 1-3) and one with two EC classes; P1001 holds an unselected
    FE, and P3002's payload names a test folder in its source path (an A13 string)."""
    proteins: dict[str, dict] = {}
    for class_index, ec_class in enumerate(("1", "2", "3")):
        for k in range(4):
            proteins[f"P{class_index + 1}00{k}"] = {"ec": f"{ec_class}.1.1.{k + 1}"}
    proteins["P1001"]["metals"] = [(900, "ZN", (0.0, 0.0, 0.0)), (901, "FE", (6.0, 0.0, 0.0))]
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


def build(world: World, campaign: Path, *, overlay_id: str = DEFAULT_OVERLAY_ID, **kwargs) -> dict:
    options = {"cohort_dir": world.cohort, "identity_root": world.identity_root, "r2_lists_dir": world.r2,
               "d2_option": "b", "provenance": REAL_PROVENANCE, "layout": world.layout}
    options.update(kwargs)
    return build_overlay(out_dir=campaign / "overlay" / overlay_id, **options)


@pytest.fixture(scope="module")
def built(world_template, tmp_path_factory):
    campaign = campaign_root(tmp_path_factory, "builtoverlay")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(identity, "MMSEQS_BINARY_SHA256", world_template.binary_sha)
        summary = build(world_template, campaign, layout=None)  # the layout the cohort audit recorded
    root = campaign / "overlay" / DEFAULT_OVERLAY_ID
    return {"summary": summary, "root": root, "campaign": campaign,
            "manifest": json.loads((root / MANIFEST_NAME).read_text())}


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
    assert manifest["d2"]["option"] == "b" and manifest["d2"]["cohort_a1_recommended_option"] in ("b", "d")
    assert "RING not used" in manifest["ring"] and not (root / "ring").exists()
    assert not list(root.rglob("split_metadata.json")) and not list(root.rglob("*_test.csv"))


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
    _write_manifest(copy, lambda payload: payload["d2"].update(policy="edited after the build"))
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
    for option, message in (("d", "not built"), ("a", "not built"), ("c", "never used")):
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


@pytest.mark.parametrize("change, message", [
    ("count", r"The string scan counts \d+ test-named strings and the cohort audit \d+; both read the same bytes"),
    ("unreadable", "The string scan cannot read"),
])
def test_the_string_scan_must_match_the_cohort_audit(world_template, tmp_path_factory, monkeypatch, change, message):
    original = overlay.string_scan

    def changed(root, items):
        scan = original(root, items)
        if change == "count":
            return {**scan, "count": scan["count"] + 1}
        return {**scan, "n_unreadable": {overlay.ROLE_ESM: 1}}

    monkeypatch.setattr(overlay, "string_scan", changed)
    campaign = campaign_root(tmp_path_factory, f"scan{change}")
    with pytest.raises(OverlayError, match=message):
        build(world_template, campaign)
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
    args = parser.parse_args(["overlay", "build", "--out", str(out), "--cohort", str(world_template.cohort),
                              "--identity-root", str(world_template.identity_root), "--r2-lists",
                              str(world_template.r2), "--d2-option", "b", "--fold-work",
                              f"1={world_template.work(1)}"])
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
