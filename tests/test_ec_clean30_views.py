"""Training-only views and the dry-run probe of the EC class vs CLEAN30 runner (synthetic data only)."""
from __future__ import annotations

import csv
from dataclasses import replace
import errno
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_clean30_fixtures import (  # noqa: E402
    AMBIGUOUS_ACCESSION, ESM_DIM, build_chain_means, build_smoke_view, make_writable, unlock, write_clean30_data_root,
    write_inner_membership,
)
from ec_clean30 import views  # noqa: E402
from ec_clean30.campaign import (  # noqa: E402
    CAMPAIGN_ID, KIND_INNER, KIND_REFIT, canonical_json, default_smoke_recipe, smoke_unit_id,
)
from ec_clean30.provenance import REPO_ROOT, SRC_ROOT, HeldOutInputError, names_test_split, sha256_file  # noqa: E402
from ec_clean30.views import (  # noqa: E402
    SourceLayout, View, ViewError, ViewVerificationError, apply_view_modes, build_view, classify_probe_records,
    forbidden_read_roots, verify_view,
)
from training.ec_membership import read_ec_fold_membership, write_ec_fold_membership  # noqa: E402

PROVENANCE = {"note": "synthetic test provenance"}


def _layout(paths: dict) -> SourceLayout:
    return SourceLayout.from_data_root(paths["data_root"])


def _fold_dir(base: Path, fold: int = 0) -> Path:
    """Where the fold-k view of a campaign rooted below ``base`` lives (the location rule)."""
    return Path(base) / CAMPAIGN_ID / views.VIEWS_DIR / f"fold{fold}"


def _smoke_dir(base: Path, name: str = "view") -> Path:
    return Path(base) / CAMPAIGN_ID / views.SMOKE_VIEWS_DIR / name


def _smoke(paths: dict, out: Path, *, seed: int = 42, max_proteins: int = 8, chain_means: bool = True) -> View:
    return build_smoke_view(paths, out, seed=seed, max_proteins=max_proteins, chain_means=chain_means,
                            provenance=PROVENANCE)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    paths = write_clean30_data_root(tmp_path_factory.mktemp("clean30_source"))
    yield paths
    unlock(paths)


@pytest.fixture()
def fresh_source(tmp_path):
    paths = write_clean30_data_root(tmp_path / "source", lock_decoys=False)
    yield paths
    unlock(paths)


@pytest.fixture(scope="module")
def fold_view(source, tmp_path_factory):
    out = _fold_dir(tmp_path_factory.mktemp("campaign"))
    return build_view(fold=0, layout=_layout(source), out_dir=out, provenance=PROVENANCE)


@pytest.fixture(scope="module")
def smoke_view(source, tmp_path_factory):
    return _smoke(source, _smoke_dir(tmp_path_factory.mktemp("campaign_smoke")))


# ---------------------------------------------------------------------------
# Build


def test_fold_view_holds_exactly_the_training_inputs(fold_view, source):
    root = Path(fold_view["view"])
    view = View.load(root)
    stems = sorted(source["stems"].values())
    assert view.structure_ids() == stems
    assert view.manifest["kind"] == views.KIND_FOLD_VIEW
    assert view.manifest["schema_version"] == views.VIEW_SCHEMA_VERSION == 2
    assert view.manifest["eligible"]["n"] == len(stems) - 1
    assert [stem.split("__")[0] for stem in view.manifest["eligible"]["excluded_structure_ids"]] == [AMBIGUOUS_ACCESSION]
    assert view.manifest_sha256 == fold_view["manifest_sha256"] == sha256_file(root / views.VIEW_MANIFEST_NAME)
    assert view.manifest["source"]["training_manifest"]["sha256"] == sha256_file(source["train_manifest"])
    assert view.manifest["source"]["training_csv"]["sha256"] == sha256_file(source["training_csv"])
    assert sha256_file(view.summary) == sha256_file(source["training_csv"])
    assert view.overlay is None and "forbidden_read_roots" not in view.manifest
    names = {item["path"] for item in view.manifest["files"]}
    assert f"structures/{source['decoys']['test_stem']}.pdb" not in names
    assert not any(source["decoys"]["test_stem"] in name for name in names)
    assert f"esm/{source['decoys']['colliding_esm'].name}" not in names
    with (view.structures / views.STRUCTURE_MANIFEST_NAME).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert sorted(row["structure_name"] for row in rows) == [f"{stem}.pdb" for stem in stems]
    assert all(row["relative_path"] == row["structure_name"] for row in rows)
    for current, dirs, files in os.walk(root):
        for name in dirs + files:
            assert not os.path.islink(Path(current) / name)
    for item in view.manifest["files"]:
        path = root / item["path"]
        assert not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        assert path.stat().st_nlink == 1  # a byte copy, not a hardlink into the store
    assert not os.access(root, os.W_OK) or os.geteuid() == 0
    assert os.access(view.caches / "parse", os.W_OK) and os.access(view.chain_means_dir("final"), os.W_OK)
    assert not any(view.ring.iterdir())
    modes = json.loads((view.receipts / views.FILE_MODES_RECEIPT_NAME).read_text())
    assert modes["outcome"] == views.FILE_MODES_ENFORCED == fold_view["file_modes"]["outcome"]
    assert modes["view_manifest_sha256"] == view.manifest_sha256 and modes["n_refused"] == modes["n_ignored"] == 0


def test_view_build_never_opens_test_side_files(fold_view, source):
    if os.geteuid() == 0:
        pytest.skip("permission checks do not bind root")
    for path in source["decoys"]["locked"]:
        assert not os.access(path, os.R_OK)
    assert Path(fold_view["view"]).is_dir()


def test_fold_view_verifies_and_records_a_receipt(fold_view):
    receipt = verify_view(fold_view["view"], provenance=PROVENANCE)
    assert receipt["status"] == "verified" and receipt["n_structures"] == 13 and receipt["n_eligible"] == 12
    assert receipt["file_modes_enforced"] is True and receipt["file_modes"]["outcome"] == views.FILE_MODES_ENFORCED
    stored = json.loads((Path(fold_view["view"]) / "receipts" / views.VERIFY_RECEIPT_NAME).read_text())
    assert stored["view_manifest_sha256"] == fold_view["manifest_sha256"]
    assert not any(path.name.startswith(".file_mode_probe") for path in (Path(fold_view["view"]) / "receipts").iterdir())
    with pytest.raises(ViewVerificationError, match="chain means for layer final are required"):
        verify_view(fold_view["view"], require_chain_means=("final",), write_receipt=False, provenance=PROVENANCE)


def _tamper(root: Path, case: str, source: dict) -> str:
    make_writable(root)
    view = View.load(root)
    esm_file = sorted(view.esm.glob("*_esmc.pt"))[0]
    if case == "extra_file":
        (view.esm / "unlisted_chain_A_esmc.pt").write_bytes(b"x")
        expected = "extra file not in the view manifest"
    elif case == "symlink":
        esm_file.unlink()
        os.symlink(source["esm_dir"] / esm_file.name, esm_file)
        expected = "symlink in view"
    elif case == "content":
        with view.summary.open("a", encoding="utf-8") as handle:
            handle.write("P0000,A_901,ZN,1.1.1.1,1,P0000,30,0,1abc\n")
        expected = "content differs from the view manifest: site_summary_train.csv"
    elif case == "test_named":
        shutil.copyfile(view.structures / f"{view.structure_ids()[0]}.pdb", view.structures / "decoy_test.pdb")
        expected = "test-named entry in view: structures/decoy_test.pdb"
    else:
        raise AssertionError(case)
    views._set_permissions(root)
    return expected


@pytest.mark.parametrize("case", ["extra_file", "symlink", "content", "test_named"])
def test_verify_refuses_a_tampered_view(source, tmp_path, case):
    root = _smoke_dir(tmp_path)
    _smoke(source, root, chain_means=False)
    verify_view(root, write_receipt=False, provenance=PROVENANCE)
    expected = _tamper(root, case, source)
    with pytest.raises(ViewVerificationError) as error:
        verify_view(root, write_receipt=False, provenance=PROVENANCE)
    assert any(expected in problem for problem in error.value.problems), error.value.problems
    make_writable(root)


def test_build_refuses_locations_inside_shared_roots_or_reused(source, tmp_path):
    layout = _layout(source)
    with pytest.raises(ViewError, match="never under CLEAN_30_main"):
        build_view(fold=0, layout=layout, out_dir=_fold_dir(source["data_root"] / "CLEAN_30_main"),
                   provenance=PROVENANCE)
    occupied = _fold_dir(tmp_path / "occupied")
    occupied.mkdir(parents=True)
    (occupied / "file.txt").write_text("x")
    with pytest.raises(ViewError, match="not empty"):
        build_view(fold=0, layout=layout, out_dir=occupied, provenance=PROVENANCE)
    with pytest.raises(HeldOutInputError):
        build_view(fold=0, layout=layout, out_dir=tmp_path / CAMPAIGN_ID / "views" / "fold0_test",
                   provenance=PROVENANCE)
    with pytest.raises(ViewError, match="smoke subset"):
        build_view(fold=0, layout=layout, out_dir=_fold_dir(tmp_path / "v1"), seed=1, provenance=PROVENANCE)
    with pytest.raises(ViewError, match="max-proteins"):
        build_view(fold=0, layout=layout, out_dir=_smoke_dir(tmp_path, "v2"), smoke=True, max_proteins=1, seed=1,
                   provenance=PROVENANCE)
    elsewhere = SourceLayout.from_data_root(source["data_root"], esm_dir=tmp_path / "esm_elsewhere")
    with pytest.raises(ViewError, match="outside the data root"):
        build_view(fold=0, layout=elsewhere, out_dir=_fold_dir(tmp_path / "v3"), provenance=PROVENANCE)


def test_views_live_only_at_their_campaign_location(source, fold_view, tmp_path):
    layout = _layout(source)
    campaign = tmp_path / CAMPAIGN_ID
    refused = [
        (campaign / views.SMOKE_VIEWS_DIR / "x", False, 0),   # a fold view in the smoke folder
        (campaign / views.VIEWS_DIR / "fold1", False, 0),     # another fold's place
        (campaign / views.VIEWS_DIR / "fold0", True, 0),      # a smoke view in the real views folder
        (tmp_path / "other_campaign" / views.VIEWS_DIR / "fold0", False, 0),
        (tmp_path / "loose_view", False, 0),
    ]
    for out, smoke, fold in refused:
        kwargs = {"smoke": True, "max_proteins": 4, "seed": 1} if smoke else {}
        with pytest.raises(ViewError, match="View location refused"):
            build_view(fold=fold, layout=layout, out_dir=out, provenance=PROVENANCE, **kwargs)
        assert not out.exists()
    moved = tmp_path / "moved" / "fold0"
    moved.parent.mkdir()
    shutil.copytree(fold_view["view"], moved, symlinks=True)
    with pytest.raises(ViewError, match="View location refused"):
        View.load(moved)
    with pytest.raises(ViewError, match="View location refused"):
        verify_view(moved, write_receipt=False, provenance=PROVENANCE)
    make_writable(moved)


def test_schema_one_views_no_longer_load(fold_view, tmp_path):
    copy = _fold_dir(tmp_path)
    copy.parent.mkdir(parents=True)
    shutil.copytree(fold_view["view"], copy, symlinks=True)
    make_writable(copy)
    path = copy / views.VIEW_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 1
    manifest.pop("read_guard")
    manifest["forbidden_read_roots"] = ["/somewhere"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ViewError, match="retired as a view"):
        View.load(copy)


def _rewrite_manifest(paths: dict, edit) -> None:
    manifest = paths["train_manifest"]
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = edit(rows)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["structure_name", "relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)


def _source_object(paths: dict, row_index: int = 0) -> Path:
    with paths["train_manifest"].open(encoding="utf-8", newline="") as handle:
        row = list(csv.DictReader(handle))[row_index]
    return paths["train_dir"] / row["relative_path"]


def test_build_refuses_a_manifest_row_through_a_test_folder(fresh_source, tmp_path):
    def through_test(rows):
        rows[0]["relative_path"] = "../test/structures/" + rows[0]["structure_name"]
        return rows

    _rewrite_manifest(fresh_source, through_test)
    out = _fold_dir(tmp_path)
    with pytest.raises(HeldOutInputError, match="test or mixed metadata folder"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=out, provenance=PROVENANCE)
    assert not out.exists()


def test_build_refuses_symlinked_sources(fresh_source, tmp_path):
    target = _source_object(fresh_source)
    copy = tmp_path / "elsewhere.pdb"
    shutil.copyfile(target, copy)
    target.unlink()
    os.symlink(copy, target)
    out = _fold_dir(tmp_path)
    with pytest.raises(ViewError, match="symlink"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=out, provenance=PROVENANCE)
    assert not out.exists()
    assert not any(path.name.startswith(".fold0.partial") for path in out.parent.iterdir())


def test_build_refuses_hash_mismatch_and_cohort_mismatch(fresh_source, tmp_path):
    target = _source_object(fresh_source)
    original = target.read_bytes()
    target.write_bytes(original.replace(b"ATOM", b"ATOM", 1) + b"REMARK changed\n")
    with pytest.raises(ViewError, match="size .* differs|SHA-256 differs"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=_fold_dir(tmp_path / "v1"), provenance=PROVENANCE)
    target.write_bytes(original)
    _rewrite_manifest(fresh_source, lambda rows: rows[1:])
    with pytest.raises(ViewError, match="must describe the same proteins"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=_fold_dir(tmp_path / "v2"), provenance=PROVENANCE)


def test_build_needs_every_exact_input_and_leaves_nothing_on_failure(fresh_source, tmp_path):
    stem = sorted(fresh_source["stems"].values())[0]
    (fresh_source["external"] / stem / "residue_features.json").unlink()
    out = _fold_dir(tmp_path)
    with pytest.raises(ViewError, match="required inputs are missing"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=out, provenance=PROVENANCE)
    assert not out.exists() and list(out.parent.iterdir()) == []


def test_smoke_view_is_a_deterministic_subset_with_a_two_fold_membership(smoke_view, source, tmp_path):
    view = smoke_view
    assert view.smoke and view.bundle_id == "g1-smoke-view"
    ids = view.structure_ids()
    assert len(ids) == 8 and set(ids) <= set(source["stems"].values())
    assert view.eligible_ids() == ids
    membership = read_ec_fold_membership(view.membership_file(), view.membership["sha256"])
    folds = [int(row["fold"]) for row in membership.rows]
    assert sorted({row["structure_id"] for row in membership.rows}) == ids
    assert folds.count(0) == folds.count(1) == 4
    assert set(view.probe_ids()) <= set(ids) and len(view.probe_ids()) == views.SMOKE_PROBE_COUNT
    again = _smoke(source, _smoke_dir(tmp_path, "again"), chain_means=False)
    assert again.structure_ids() == ids and again.membership["sha256"] == view.membership["sha256"]
    other = _smoke(source, _smoke_dir(tmp_path, "other"), seed=7, chain_means=False)
    assert other.structure_ids() != ids
    receipt = verify_view(view.root, require_chain_means=("final",), write_receipt=False, provenance=PROVENANCE)
    assert receipt["chain_means"]["final"]["n_keys"] == 8


def test_fold_view_copies_a_certified_inner_membership(source, tmp_path):
    from ec_clean30.identity import INNER_CERTIFICATE

    assert views.INNER_CERTIFICATE_NAME == INNER_CERTIFICATE
    membership, digest, certificate = write_inner_membership(source, tmp_path / "identity")
    root = _fold_dir(tmp_path / "view")
    result = build_view(fold=0, layout=_layout(source), out_dir=root, inner_membership=(membership, digest),
                        provenance=PROVENANCE)
    record = result["manifest"]["membership"]
    assert record["role"] == "inner" and record["n_folds"] == 4
    assert record["certificate_sha256"] == sha256_file(certificate)
    assert sha256_file(root / views.INNER_CERTIFICATE_NAME) == sha256_file(certificate)
    receipt = verify_view(root, write_receipt=False, provenance=PROVENANCE)
    assert receipt["membership"]["certificate_sha256"] == record["certificate_sha256"]
    eligible = [stem for stem in sorted(source["stems"].values()) if not stem.startswith(AMBIGUOUS_ACCESSION)]
    rows = [{"structure_id": stem, "accession": stem.split("__")[0], "group_id": stem.split("__")[0],
             "fold": str(index % 4)} for index, stem in enumerate(eligible)]
    short = tmp_path / "short" / "inner_membership_fold0.csv"
    short.parent.mkdir()
    short_digest = write_ec_fold_membership(rows[1:], short)
    refused = {name: _fold_dir(tmp_path / name, 1 if name == "view5" else 0)
               for name in ("view2", "view3", "view4", "view5")}
    with pytest.raises(ViewError, match="certify-inner receipt"):
        build_view(fold=0, layout=_layout(source), out_dir=refused["view2"], inner_membership=(short, short_digest),
                   provenance=PROVENANCE)
    with pytest.raises(ViewError, match="another membership file"):
        build_view(fold=0, layout=_layout(source), out_dir=refused["view3"], inner_membership=(short, short_digest),
                   inner_certificate=certificate, provenance=PROVENANCE)
    short_certificate = json.loads(certificate.read_text())
    short_certificate["membership"] = {"file": short.name, "sha256": short_digest}
    (short.parent / views.INNER_CERTIFICATE_NAME).write_text(json.dumps(short_certificate))
    with pytest.raises(ViewError, match="must be identical"):
        build_view(fold=0, layout=_layout(source), out_dir=refused["view3"], inner_membership=(short, short_digest),
                   provenance=PROVENANCE)
    _uncertified, uncertified_digest, uncertified = write_inner_membership(source, tmp_path / "refused",
                                                                         certified=False)
    with pytest.raises(ViewError, match="not a passed certify-inner receipt"):
        build_view(fold=0, layout=_layout(source), out_dir=refused["view4"],
                   inner_membership=(_uncertified, uncertified_digest), provenance=PROVENANCE)
    with pytest.raises(ViewError, match="not fold 1"):
        build_view(fold=1, layout=_layout(source), out_dir=refused["view5"], inner_membership=(membership, digest),
                   provenance=PROVENANCE)
    assert not any(path.exists() for path in refused.values())
    assert uncertified.is_file()


def test_verify_refuses_a_changed_inner_certificate(source, tmp_path):
    membership, digest, _certificate = write_inner_membership(source, tmp_path / "identity")
    root = _fold_dir(tmp_path)
    build_view(fold=0, layout=_layout(source), out_dir=root, inner_membership=(membership, digest),
               provenance=PROVENANCE)
    make_writable(root)
    path = root / views.INNER_CERTIFICATE_NAME
    path.write_text(path.read_text().replace('"certified": true', '"certified": false'))
    views._set_permissions(root)
    with pytest.raises(ViewVerificationError) as error:
        verify_view(root, write_receipt=False, provenance=PROVENANCE)
    assert any("inner certificate is missing or differs" in problem for problem in error.value.problems)
    make_writable(root)


# ---------------------------------------------------------------------------
# Read guard: rules recorded, expanded where the view runs


def _files_of(root: Path) -> list[str]:
    return [views._real(Path(current) / name) for current, _dirs, names in os.walk(root) for name in names]


def test_forbidden_roots_cover_shared_and_test_bearing_roots(source, smoke_view):
    from project_paths import DATA_DIR

    roots = forbidden_read_roots(_layout(source), view_root=smoke_view.root, fold=0, smoke=True)
    data = str(source["data_root"])
    for name in ("CLEAN_30_main", "CLEAN_30_train_test_split_0", "CLEAN_30_train_test_split_4", "esm_embeddings",
                 "updated_feature_extraction", "RING_features", "structure_store", "CLEAN_all_train_valid_splits",
                 "CARE_dataset", "backups", views.CAMPAIGN_LOCAL_DIR_NAME):
        assert os.path.join(data, name) in roots
    # CLEAN embedding farms and fit folders on the internal disk are barred below the project data root.
    assert str(Path(DATA_DIR) / views.CAMPAIGN_LOCAL_DIR_NAME) in roots
    for relative in ("prepare_training_and_test_set", "CLEAN_prepare_training_and_test_set",
                     "CARE_prepare_training_and_test_set", "docs/notebook_outputs", "runs"):
        assert str(REPO_ROOT / relative) in roots
    assert not any(views._under(path, [views._real(root)]) for path in _files_of(smoke_view.root) for root in roots)
    assert smoke_view.forbidden_read_roots == roots
    rules = smoke_view.read_guard
    assert rules["source_layout_dirs"] == ["esm_embeddings", "updated_feature_extraction"]
    assert not any(os.path.isabs(name) for key in ("data_root_children", "source_layout_dirs", "campaign_children",
                                                   "repository_paths") for name in rules[key])


def test_expanded_roots_cover_the_campaign_folders_and_the_other_folds(fold_view, smoke_view):
    real = View.load(fold_view["view"])
    campaign = real.campaign_root
    roots = set(real.forbidden_read_roots)
    for child in (*views.CAMPAIGN_FORBIDDEN_CHILDREN, views.SMOKE_VIEWS_DIR, views.SMOKE_RUNS_DIR):
        assert str(campaign / child) in roots
    for other in (1, 2, 3, 4):
        assert str(campaign / views.VIEWS_DIR / f"fold{other}") in roots
        assert str(campaign / views.RUNS_DIR / f"fold{other}") in roots
    for allowed in (views.VIEWS_DIR, views.RUNS_DIR, f"{views.VIEWS_DIR}/fold0", f"{views.RUNS_DIR}/fold0"):
        assert str(campaign / allowed) not in roots
    assert not any(views._under(path, [views._real(root)]) for path in _files_of(real.root) for root in roots)
    assert real.default_runs_dir == campaign / views.RUNS_DIR / "fold0"

    smoke_campaign = smoke_view.campaign_root
    smoke_roots = set(smoke_view.forbidden_read_roots)
    assert {str(smoke_campaign / views.VIEWS_DIR), str(smoke_campaign / views.RUNS_DIR),
            str(smoke_campaign / "identity"), str(smoke_campaign / "overlay")} <= smoke_roots
    assert str(smoke_campaign / views.SMOKE_VIEWS_DIR) not in smoke_roots
    assert not any(views._under(path, [views._real(root)]) for path in _files_of(smoke_view.root)
                   for root in smoke_roots)
    assert smoke_view.default_runs_dir == smoke_campaign / views.SMOKE_RUNS_DIR / smoke_view.root.name


def test_forbidden_roots_follow_the_data_root_that_holds_the_view(source, tmp_path):
    other_disk = tmp_path / "other_disk" / "DeepMzyme_Data"
    (other_disk / "campaigns").mkdir(parents=True)
    project_data = tmp_path / "project" / "DeepMzyme_Data"
    project_data.mkdir(parents=True)
    os.symlink(other_disk / "campaigns", project_data / "campaigns")
    view_root = _fold_dir(project_data / "campaigns")
    roots = forbidden_read_roots(_layout(source), view_root=view_root, fold=0)
    for base in (project_data, other_disk):
        for name in ("esm_embeddings", "RING_features", "updated_feature_extraction", "CLEAN_30_main"):
            assert str(base / name) in roots
        # Campaign children by the given and the real path of the campaign root.
        assert str(base / "campaigns" / CAMPAIGN_ID / "identity") in roots
        assert str(base / "campaigns" / CAMPAIGN_ID / views.VIEWS_DIR / "fold3") in roots
    result = build_view(fold=0, layout=_layout(source), out_dir=view_root, provenance=PROVENANCE)
    assert str(other_disk / "esm_embeddings") in View.load(result["view"]).forbidden_read_roots
    make_writable(Path(result["view"]))


def test_a_copied_view_expands_its_guard_where_it_sits(fold_view, tmp_path):
    original = View.load(fold_view["view"])
    target = _fold_dir(tmp_path / "vm_root")
    target.parent.mkdir(parents=True)
    shutil.copytree(original.root, target, symlinks=True)
    copy = View.load(target)
    assert copy.manifest_sha256 == original.manifest_sha256
    roots = copy.forbidden_read_roots
    for child in ("identity", "overlay", "plans", f"{views.VIEWS_DIR}/fold2", f"{views.RUNS_DIR}/fold4"):
        assert str(copy.campaign_root / child) in roots
        assert str(original.campaign_root / child) not in roots
    assert verify_view(target, write_receipt=False, provenance=PROVENANCE)["status"] == "verified"
    make_writable(target)


def test_a_view_inside_an_expanded_root_is_refused(smoke_view, tmp_path):
    inside = _smoke_dir(tmp_path / "elsewhere" / "DeepMzyme_Data" / "backups")
    inside.parent.mkdir(parents=True)
    shutil.copytree(smoke_view.root, inside, symlinks=True)
    with pytest.raises(ViewError, match="lies inside the forbidden read root .*backups"):
        View.load(inside)
    make_writable(inside)


def test_weakened_guard_rules_are_refused(fold_view, tmp_path):
    copy = _fold_dir(tmp_path)
    copy.parent.mkdir(parents=True)
    shutil.copytree(fold_view["view"], copy, symlinks=True)
    make_writable(copy)
    path = copy / views.VIEW_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["read_guard"]["campaign_children"].remove("identity")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ViewError, match="lacks \\['identity'\\]"):
        View.load(copy)
    manifest["read_guard"]["campaign_children"].append("/absolute/identity")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ViewError, match="plain relative names"):
        View.load(copy)
    # The recorded source layout's ESMC and external folders must stay in the rules.
    manifest = json.loads(path.read_text())
    manifest["read_guard"]["campaign_children"].remove("/absolute/identity")
    manifest["read_guard"]["campaign_children"].append("identity")
    path.write_text(json.dumps(manifest))
    assert View.load(copy).fold == 0
    manifest["read_guard"]["source_layout_dirs"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ViewError,
                       match="source_layout_dirs lacks \\['esm_embeddings', 'updated_feature_extraction'\\]"):
        View.load(copy)
    manifest["read_guard"]["source_layout_dirs"] = ["esm_embeddings", "updated_feature_extraction"]
    manifest["source"]["esm_dir"] = str(Path(manifest["source"]["data_root"]) / "esm_custom")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ViewError, match="source_layout_dirs lacks \\['esm_custom'\\]"):
        View.load(copy)
    # A smoke view's rules must keep the sibling rule for the other smoke views and their runs.
    manifest = json.loads(path.read_text())
    assert manifest["read_guard"]["campaign_sibling_dirs"] == []
    problems = views.read_guard_rule_problems({**manifest["read_guard"], "campaign_children": views._campaign_children(
        fold=0, smoke=True)}, fold=0, smoke=True)
    assert problems == ["read-guard rule campaign_sibling_dirs lacks ['smoke_runs', 'smoke_views'] (weaker than this "
                        "code's rules)"]


def test_smoke_views_forbid_each_other_and_each_others_runs(source, tmp_path):
    first = _smoke(source, _smoke_dir(tmp_path, "fold0_a"), chain_means=False)
    second = _smoke(source, _smoke_dir(tmp_path, "fold0_b"), chain_means=False)
    campaign = first.campaign_root
    assert first.read_guard["campaign_sibling_dirs"] == [views.SMOKE_VIEWS_DIR, views.SMOKE_RUNS_DIR]
    (campaign / views.SMOKE_RUNS_DIR / "fold0_a").mkdir(parents=True)
    (campaign / views.SMOKE_RUNS_DIR / "fold0_b").mkdir()
    (campaign / views.SMOKE_VIEWS_DIR / ".fold0_c.partial-1").mkdir()
    roots = set(View.load(first.root).forbidden_read_roots)
    assert {str(campaign / views.SMOKE_VIEWS_DIR / "fold0_b"), str(campaign / views.SMOKE_RUNS_DIR / "fold0_b"),
            str(campaign / views.SMOKE_VIEWS_DIR / ".fold0_c.partial-1")} <= roots
    assert not {str(campaign / views.SMOKE_VIEWS_DIR / "fold0_a"),
                str(campaign / views.SMOKE_RUNS_DIR / "fold0_a")} & roots
    other_roots = set(View.load(second.root).forbidden_read_roots)
    assert str(campaign / views.SMOKE_VIEWS_DIR / "fold0_a") in other_roots
    assert str(campaign / views.SMOKE_VIEWS_DIR / "fold0_b") not in other_roots
    loaded = View.load(first.root)
    probe_dir = loaded.caches / "dry_run" / "sibling_case"
    report = _synthetic_report(loaded, probe_dir, reads=[second.summary])
    assert classify_probe_records(report, view=loaded, probe_dir=probe_dir)["outside_reads"] == [
        views._real(second.summary)]
    assert "forbidden read root" in views.runs_dir_problem(loaded, campaign / views.SMOKE_RUNS_DIR / "fold0_b")
    assert views.runs_dir_problem(loaded, campaign / views.SMOKE_RUNS_DIR / "fold0_a" / "units") is None
    for path in (first.root, second.root):
        make_writable(path)


def test_dry_run_refuses_a_runs_folder_inside_the_view_or_an_expanded_root(smoke_view):
    from ec_clean30.commands import CommandError, dry_run

    with pytest.raises(CommandError, match="forbidden read root"):
        dry_run(smoke_view, runs_dir=smoke_view.campaign_root / views.RUNS_DIR / "fold0", probe_families=(),
                cap_prefix=(), provenance=PROVENANCE, write=False)
    with pytest.raises(CommandError, match="inside the view"):
        dry_run(smoke_view, runs_dir=smoke_view.root / "runs", probe_families=(), cap_prefix=(),
                provenance=PROVENANCE, write=False)
    with pytest.raises(CommandError, match="must be <campaign root>/smoke_runs/view or a folder inside it"):
        dry_run(smoke_view, runs_dir=smoke_view.campaign_root / views.SMOKE_RUNS_DIR / "other_view",
                probe_families=(), cap_prefix=(), provenance=PROVENANCE, write=False)


def test_a_fold_view_runs_only_in_its_own_runs_folder(fold_view, tmp_path):
    view = View.load(fold_view["view"])
    campaign = view.campaign_root
    refused = [campaign / views.RUNS_DIR, campaign / views.RUNS_DIR / "fold0_extra", campaign / views.VIEWS_DIR /
               "fold0_runs", campaign, campaign / "runs_fold0", tmp_path / "outside" / "runs" / "fold0"]
    for runs in refused:
        assert "must be <campaign root>/runs/fold0 or a folder inside it" in views.runs_dir_problem(view, runs), runs
    assert "forbidden read root" in views.runs_dir_problem(view, campaign / views.RUNS_DIR / "fold1")
    for runs in (campaign / views.RUNS_DIR / "fold0", campaign / views.RUNS_DIR / "fold0" / "s1"):
        assert views.runs_dir_problem(view, runs) is None


def test_a_runs_folder_symlinked_into_a_forbidden_root_is_refused(fold_view, tmp_path):
    from ec_clean30.commands import CommandError, dry_run

    copy = _fold_dir(tmp_path)
    copy.parent.mkdir(parents=True)
    shutil.copytree(fold_view["view"], copy, symlinks=True)
    view = View.load(copy)
    campaign = view.campaign_root
    (campaign / "identity").mkdir()
    (campaign / views.RUNS_DIR).mkdir()
    os.symlink(campaign / "identity", campaign / views.RUNS_DIR / "fold0")
    with pytest.raises(CommandError, match="lies inside the forbidden read root .*identity"):
        dry_run(view, runs_dir=campaign / views.RUNS_DIR / "fold0", probe_families=(), cap_prefix=("env",),
                provenance=PROVENANCE, write=False)
    make_writable(copy)


# ---------------------------------------------------------------------------
# File modes


def _refusing(error: OSError):
    def chmod(_path, _mode):
        raise error
    return chmod


@pytest.mark.parametrize("chmod, outcome", [
    (lambda _path, _mode: None, views.FILE_MODES_IGNORED),
    (_refusing(PermissionError("chmod refused")), views.FILE_MODES_REFUSED),
    (_refusing(OSError(errno.EPERM, "Operation not permitted")), views.FILE_MODES_REFUSED),
], ids=["no_op", "permission_error", "eperm"])
def test_views_build_and_verify_where_chmod_is_ignored_or_refused(source, tmp_path, monkeypatch, chmod, outcome):
    monkeypatch.setattr(views, "_chmod", chmod)
    root = _smoke_dir(tmp_path)
    result = build_view(fold=0, layout=_layout(source), out_dir=root, smoke=True, max_proteins=4, seed=42,
                        provenance=PROVENANCE)
    recorded = json.loads((root / views.RECEIPTS_DIR / views.FILE_MODES_RECEIPT_NAME).read_text())
    assert recorded["outcome"] == outcome == result["file_modes"]["outcome"]
    assert recorded["n_ignored" if outcome == views.FILE_MODES_IGNORED else "n_refused"] > 0
    view = View.load(root)
    data_file = view.root / view.manifest["files"][0]["path"]
    assert data_file.stat().st_mode & stat.S_IWUSR  # the modes really were not set
    receipt = verify_view(root, write_receipt=False, provenance=PROVENANCE)
    assert receipt["status"] == "verified" and receipt["file_modes_enforced"] is False
    assert receipt["file_modes"]["outcome"] == outcome
    # Content hashes are still checked.
    with view.summary.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ViewVerificationError) as error:
        verify_view(root, write_receipt=False, provenance=PROVENANCE)
    assert any("content differs from the view manifest: site_summary_train.csv" in problem
               for problem in error.value.problems)


def test_a_view_built_where_chmod_was_ignored_needs_apply_view_modes(source, tmp_path, monkeypatch):
    from ec_clean30.commands import CommandError, build_unit_command, clear_view_for_unit, dry_run

    monkeypatch.setattr(views, "_chmod", lambda _path, _mode: None)
    root = _smoke_dir(tmp_path)
    view = _smoke(source, root)
    dry = dry_run(view, probe_families=(), cap_prefix=(), provenance=PROVENANCE)
    assert dry["status"] == "passed" and dry["verification"]["file_modes_enforced"] is False
    dry_sha = sha256_file(view.receipts / views.DRY_RUN_RECEIPT_NAME)
    monkeypatch.undo()  # the driver changed: chmod works now

    with pytest.raises(ViewVerificationError) as error:
        verify_view(root, write_receipt=False, provenance=PROVENANCE)
    assert any("not applied since the build" in problem and "run apply-view-modes" in problem
               for problem in error.value.problems), error.value.problems
    assert not any(problem.startswith("data file is writable") for problem in error.value.problems)
    recipe = default_smoke_recipe("F2")
    command = build_unit_command(view, recipe, unit_id=smoke_unit_id(0, recipe, refit=False, seed=42), kind=KIND_INNER,
                                 inner_fold=0, seed=42, runs_dir=view.default_runs_dir, step="smoke")
    with pytest.raises(CommandError, match="apply-view-modes"):
        clear_view_for_unit(view, command)

    receipt = apply_view_modes(root, provenance=PROVENANCE)
    assert receipt["outcome"] == views.FILE_MODES_ENFORCED and receipt["n_ignored"] == receipt["n_refused"] == 0
    assert receipt["view_manifest_sha256"] == view.manifest_sha256 == sha256_file(root / views.VIEW_MANIFEST_NAME)
    applied = json.loads((view.receipts / views.FILE_MODES_APPLIED_RECEIPT_NAME).read_text())
    assert applied["stage"] == "apply_view_modes" and "filesystem" in applied and "checked_utc" in applied
    verification = verify_view(root, write_receipt=False, provenance=PROVENANCE)
    assert verification["file_modes_enforced"] is True
    reloaded = View.load(root)
    assert reloaded.manifest_sha256 == view.manifest_sha256
    assert sha256_file(view.receipts / views.DRY_RUN_RECEIPT_NAME) == dry_sha
    assert views.dry_run_problems(reloaded, views.read_dry_run_receipt(reloaded),
                                  chain_means=verification["chain_means"]) == []
    assert clear_view_for_unit(reloaded, command)["dry_run_receipt_sha256"] == dry_sha
    # A view whose modes are enforced reports a writable data file by name.
    path = root / view.manifest["files"][0]["path"]
    os.chmod(path, 0o644)
    with pytest.raises(ViewVerificationError) as error:
        verify_view(root, write_receipt=False, provenance=PROVENANCE)
    assert f"data file is writable: {view.manifest['files'][0]['path']}" in error.value.problems
    make_writable(root)


def test_apply_view_modes_refuses_a_manifest_changed_while_modes_are_set(source, tmp_path, monkeypatch):
    root = _smoke_dir(tmp_path)
    _smoke(source, root, chain_means=False)
    original = views._set_permissions

    def set_and_edit(path):
        result = original(path)
        manifest = Path(path) / views.VIEW_MANIFEST_NAME
        os.chmod(manifest, 0o644)
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(views, "_set_permissions", set_and_edit)
    with pytest.raises(ViewError, match="the view manifest changed while modes were applied"):
        apply_view_modes(root, provenance=PROVENANCE)
    assert not (root / views.RECEIPTS_DIR / views.FILE_MODES_APPLIED_RECEIPT_NAME).exists()
    make_writable(root)


def test_verify_needs_a_writable_receipts_folder_to_test_file_modes(source, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("permission checks do not bind root")
    root = _smoke_dir(tmp_path)
    _smoke(source, root, chain_means=False)
    receipts = root / views.RECEIPTS_DIR
    os.chmod(receipts, 0o555)
    try:
        with pytest.raises(ViewVerificationError) as error:
            verify_view(root, write_receipt=False, provenance=PROVENANCE)
    finally:
        os.chmod(receipts, 0o755)
    assert any(problem.startswith("file modes cannot be tested in receipts/") for problem in error.value.problems), \
        error.value.problems
    make_writable(root)


def test_a_dry_run_without_write_leaves_no_receipt(source, tmp_path):
    from ec_clean30.commands import dry_run

    root = _smoke_dir(tmp_path)
    view = _smoke(source, root)
    receipt = dry_run(view, probe_families=(), cap_prefix=(), provenance=PROVENANCE, write=False)
    assert receipt["status"] == "passed" and receipt["verified_view"] is True and receipt["capped"] is False
    assert not (view.receipts / views.VERIFY_RECEIPT_NAME).exists()
    assert not (view.receipts / views.DRY_RUN_RECEIPT_NAME).exists()
    make_writable(root)


# ---------------------------------------------------------------------------
# Chain means of the L8 layer


class _FakeEsmc:
    """Stands in for ESMC in ``from-model``: 30 hidden layers of fixed rows (no weights, no network)."""

    n_layers = 30

    def parameters(self):
        import torch

        return iter([SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)])

    def encode(self, protein):
        return protein.sequence

    def logits(self, sequence, _config):
        import torch

        n_positions = len(sequence) + 2
        base = torch.arange(n_positions * ESM_DIM, dtype=torch.float32).reshape(n_positions, ESM_DIM)
        base = torch.sin(base / 97.0)
        hidden = torch.stack([base + 0.01 * layer for layer in range(self.n_layers)])
        return SimpleNamespace(embeddings=(0.5 * base)[None], hidden_states=hidden[:, None])


_FAKE_SDK = (_FakeEsmc, lambda sequence: SimpleNamespace(sequence=sequence), lambda **kwargs: dict(kwargs))


def test_a_two_layer_from_model_cache_verifies_as_block26(source, tmp_path):
    pytest.importorskip("torch")
    from embed_helpers.esmc_chain_means import build_from_model

    view = _smoke(source, _smoke_dir(tmp_path))
    summary = build_from_model(view.root / views.ALLOWLIST_NAME, view.structures, view.chain_means_dir("block26"),
                               layer_index=26, device="cpu", embedding_dim=ESM_DIM, model=_FakeEsmc(), sdk=_FAKE_SDK)
    assert summary["layers"] == ["final", "block26"]
    receipt = verify_view(view.root, require_chain_means=("final", "block26"), write_receipt=False,
                          provenance=PROVENANCE)
    assert receipt["chain_means"]["block26"]["manifest_sha256"] == summary["manifest_sha256"]
    assert receipt["chain_means"]["final"]["manifest_sha256"] == view.chain_means_manifest_sha256("final")


# ---------------------------------------------------------------------------
# Overlay binding


def _write_overlay_manifest(data_root: Path, *, overlay_id: str = "ecc30-overlay-synthetic",
                            override: dict | None = None) -> Path:
    """An overlay manifest over the synthetic training-side files (test-named and mixed folders left out)."""
    files = []
    for current, dirs, names in os.walk(data_root):
        dirs.sort()
        relative_dir = Path(current).relative_to(data_root)
        if any(names_test_split(part) for part in relative_dir.parts) or relative_dir.parts[:2] == (
                "CLEAN_30_main", "structures"):
            continue
        for name in sorted(names):
            relative = (relative_dir / name).as_posix()
            if name == views.OVERLAY_MANIFEST_NAME or names_test_split(name):
                continue
            path = Path(current) / name
            files.append({"path": relative, "sha256": (override or {}).get(relative, sha256_file(path)),
                          "size_bytes": path.stat().st_size, "role": "copy", "origin": "shared_root_copy"})
    path = data_root / views.OVERLAY_MANIFEST_NAME
    path.write_text(json.dumps({"overlay_id": overlay_id, "files": files}, indent=2, sort_keys=True) + "\n")
    return path


def test_the_overlay_record_survives_a_round_trip(fresh_source, tmp_path):
    from ec_clean30.commands import CommandError, build_unit_command, check_command

    data = fresh_source["data_root"]
    membership, digest, _certificate = write_inner_membership(fresh_source, data / "identity" / "fold0")
    overlay_path = _write_overlay_manifest(data)
    layout = _layout(fresh_source)
    result = build_view(fold=0, layout=layout, out_dir=_fold_dir(tmp_path / "a"), inner_membership=(membership, digest),
                        provenance=PROVENANCE)
    record = result["manifest"]["overlay"]
    assert record["id"] == "ecc30-overlay-synthetic" and record["manifest_sha256"] == sha256_file(overlay_path)
    # Training manifest and site table; per protein the structure, ESMC payload, sidecar and external
    # features; the inner membership and its certificate.
    assert record["n_files_checked"] == 2 + 4 * 13 + 2
    view = View.load(result["view"])
    assert dict(view.overlay) == record
    receipt = verify_view(view.root, check_sources=True, write_receipt=False, provenance=PROVENANCE)
    assert receipt["overlay"] == record
    make_writable(view.root)
    build_chain_means(view.root)
    views._set_permissions(view.root)
    assert views.freeze1_view_record(view.root, require_dry_run=False, provenance=PROVENANCE)["overlay"] == {
        "id": record["id"], "manifest_sha256": record["manifest_sha256"]}

    recipe = default_smoke_recipe("F2")
    kwargs = dict(unit_id="ecc30__fold0__overlay_case__inner0__seed42", kind=KIND_INNER, inner_fold=0, seed=42,
                  runs_dir=view.default_runs_dir, step="S1")
    command = build_unit_command(view, recipe, freeze1b_sha256="b" * 64, **kwargs)
    assert (command.identity["overlay_id"], command.identity["overlay_sha256"]) == (record["id"],
                                                                                    record["manifest_sha256"])
    assert command.identity["freeze1b_sha256"] == "b" * 64 and check_command(command, view) == []
    with pytest.raises(CommandError, match="differs from the view's overlay record"):
        build_unit_command(view, recipe, overlay_id="another-overlay", **kwargs)
    with pytest.raises(CommandError, match="freeze1b_sha256 must be"):
        build_unit_command(view, recipe, freeze1b_sha256="not-a-hash", **kwargs)
    # An identity that names no overlay (argv and identity edited together) is refused for this view.
    unbound = {**command.identity, "overlay_id": None}
    argv = list(command.argv)
    argv[argv.index("--campaign-run-identity") + 1] = canonical_json(unbound)
    problems = check_command(replace(command, argv=tuple(argv), identity=unbound), view)
    assert "the campaign run identity's overlay_id does not match this unit and view" in problems

    # The membership and its certificate must come from the overlay as well.
    outside, outside_digest, _ = write_inner_membership(fresh_source, tmp_path / "identity_elsewhere")
    with pytest.raises(ViewError, match="outside the overlay root"):
        build_view(fold=0, layout=layout, out_dir=_fold_dir(tmp_path / "b"), inner_membership=(outside, outside_digest),
                   provenance=PROVENANCE)
    # A used file whose SHA-256 differs from the overlay manifest refuses the build.
    stem = sorted(fresh_source["stems"].values())[0]
    _write_overlay_manifest(data, override={f"esm_embeddings/{views.esm_file_name(stem)}": "0" * 64})
    with pytest.raises(ViewError, match="missing from the overlay manifest or differ"):
        build_view(fold=0, layout=layout, out_dir=_fold_dir(tmp_path / "c"), inner_membership=(membership, digest),
                   provenance=PROVENANCE)
    # A path listed twice is refused (counted, not named).
    duplicated = json.loads(_write_overlay_manifest(data).read_text())
    duplicated["files"].append(dict(duplicated["files"][0]))
    overlay_path.write_text(json.dumps(duplicated))
    with pytest.raises(ViewError, match="lists 1 paths more than once") as error:
        build_view(fold=0, layout=layout, out_dir=_fold_dir(tmp_path / "d"), inner_membership=(membership, digest),
                   provenance=PROVENANCE)
    assert duplicated["files"][0]["path"] not in str(error.value)
    assert not any(_fold_dir(tmp_path / name).exists() for name in ("b", "c", "d"))
    # The rewritten overlay manifest no longer matches the one the view recorded.
    with pytest.raises(ViewVerificationError) as error:
        verify_view(view.root, check_sources=True, write_receipt=False, provenance=PROVENANCE)
    assert any("overlay manifest differs" in problem for problem in error.value.problems)
    make_writable(view.root)


# ---------------------------------------------------------------------------
# Dry run


def test_dry_run_passes_on_a_clean_smoke_view(smoke_view):
    from ec_clean30.commands import dry_run

    receipt = dry_run(smoke_view, cap_prefix=(), provenance=PROVENANCE, timeout=600)
    assert receipt["status"] == "passed", receipt["probes"]
    assert receipt["n_commands_checked"] == 12 and not receipt["static_failures"]
    assert [(probe["family"], probe["kind"]) for probe in receipt["probes"]] == [("F4", KIND_INNER), ("F4", KIND_REFIT)]
    inner, refit = receipt["probes"]
    assert inner["n_train_structures"] == 4 and inner["n_val_structures"] == 4
    assert refit["n_train_structures"] == 8 and refit["n_val_structures"] == 0
    assert inner["records"]["n_view_reads"] > 0 and not any(inner["outside"].values())
    assert receipt["runs_dir_checked"] == str(smoke_view.default_runs_dir)
    assert receipt["read_guard"]["expanded_roots"] == smoke_view.forbidden_read_roots
    assert receipt["read_guard"]["rules"] == dict(smoke_view.read_guard)
    assert receipt["verification"]["file_modes_enforced"] is True and receipt["verification"]["overlay"] is None
    stored = json.loads((smoke_view.receipts / views.DRY_RUN_RECEIPT_NAME).read_text())
    assert stored["status"] == "passed" and stored["view_manifest_sha256"] == smoke_view.manifest_sha256


def test_dry_run_catches_a_read_outside_the_view(source, tmp_path):
    from ec_clean30.commands import dry_run

    root = _smoke_dir(tmp_path)
    view = _smoke(source, root)
    make_writable(root)
    esm_file = sorted(view.esm.glob("*_esmc.pt"))[0]
    outside = tmp_path / "outside_copy" / esm_file.name
    outside.parent.mkdir()
    shutil.copyfile(esm_file, outside)
    esm_file.unlink()
    os.symlink(outside, esm_file)
    views._set_permissions(root)
    with pytest.raises(ViewVerificationError, match="symlink in view"):
        verify_view(root, write_receipt=False, provenance=PROVENANCE)
    receipt = dry_run(View.load(root), cap_prefix=(), verify=False, provenance=PROVENANCE, timeout=600)
    assert receipt["status"] == "failed"
    for probe in receipt["probes"]:
        assert str(outside) in probe["outside"]["outside_reads"]
        assert any("outside reads" in problem for problem in probe["problems"])
    make_writable(root)


def test_dry_run_catches_a_generation_attempt(source, tmp_path):
    from ec_clean30.commands import build_unit_command, check_command, evaluate_probe

    root = _smoke_dir(tmp_path)
    view = _smoke(source, root, chain_means=False)
    recipe = default_smoke_recipe("F2")
    command = build_unit_command(view, recipe, unit_id=smoke_unit_id(0, recipe, refit=True, seed=42), kind=KIND_REFIT,
                                 seed=42, runs_dir=tmp_path / "runs", step="smoke")
    make_writable(root)
    stem = view.structure_ids()[0]
    (view.external / stem / "residue_features.json").unlink()
    views._set_permissions(root)
    loose = replace(command, argv=tuple(token for token in command.argv
                                        if token != "--no-prepare-missing-external-features"))
    assert any("--no-prepare-missing-external-features is missing" in problem for problem in check_command(loose, view))
    probe_root = view.caches / "dry_run" / "generation_case"
    report = views.run_probe(view=view, unit_id=loose.unit_id, argv=loose.argv, env=loose.env, probe_root=probe_root,
                             cap_prefix=(), timeout=600)
    assert report["generation_attempts"] == ["_generate_missing_updated_external_features"]
    assert report["error"]["type"] == "GenerationAttempted"
    evaluated = evaluate_probe(report, view=view, command=loose, probe_dir=probe_root / loose.unit_id)
    assert evaluated["status"] == "failed"
    assert any("generation was attempted" in problem for problem in evaluated["problems"])
    make_writable(root)


def test_probe_child_guard_blocks_a_shared_root(smoke_view, source):
    from ec_clean30.commands import build_unit_command

    recipe = default_smoke_recipe("F2")
    command = build_unit_command(smoke_view, recipe, unit_id="ecc30smoke__fold0__guard_case__refit__seed42",
                                 kind=KIND_REFIT, seed=42, runs_dir=smoke_view.root.parent / "runs", step="smoke")
    argv = list(command.argv)
    argv[argv.index("--esm-embeddings-dir") + 1] = str(source["esm_dir"])
    report = views.run_probe(view=smoke_view, unit_id=command.unit_id, argv=argv, env=command.env,
                             probe_root=smoke_view.caches / "dry_run" / "guard_case", cap_prefix=(), timeout=600)
    assert report["status"] == "failed" and report["error"]["type"] == "PermissionError"
    assert "held-out path" in report["error"]["message"]
    # The recorder runs before the guard, so the denied access is recorded as well.
    denied = [path for table in ("reads", "listings") for path in report["records"][table]
              if path.startswith(str(source["esm_dir"]))]
    assert denied, report["records"]["listings"]
    classified = classify_probe_records(report, view=smoke_view, probe_dir=Path(report["probe_dir"]))
    assert any(path.startswith(str(source["esm_dir"])) for path in
               classified["outside_reads"] + classified["outside_listings"])


def _synthetic_report(view: View, probe_dir: Path, *, reads=(), listings=(), subprocesses=(), python_roots=None):
    return {"records": {"reads": {str(path): 1 for path in reads}, "writes": {},
                        "listings": {str(path): 1 for path in listings}, "existing_dirs": {},
                        "subprocesses": list(subprocesses)},
            "python_roots": [sys.prefix] if python_roots is None else list(python_roots),
            "sys_path": [str(REPO_ROOT), str(SRC_ROOT), str(probe_dir)]}


def test_probe_classification_never_trusts_the_repository_root(smoke_view, tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    allowed_reads = [SRC_ROOT / "train.py", SRC_ROOT / "ec_clean30" / "__pycache__" / "views.cpython-311.pyc",
                     smoke_view.summary, probe_dir / "probe_request.json"]
    outside_reads = [REPO_ROOT / "docs" / "DATASETS.md", REPO_ROOT / "README.md",
                     REPO_ROOT / "CARE_prepare_training_and_test_set" / "README.md", SRC_ROOT / "requirements.txt",
                     REPO_ROOT / "scripts" / "run_ec_clean30_campaign.py"]
    report = _synthetic_report(smoke_view, probe_dir, reads=allowed_reads + outside_reads,
                               listings=[REPO_ROOT, SRC_ROOT / "training", REPO_ROOT / "runs", REPO_ROOT / "docs"])
    classified = classify_probe_records(report, view=smoke_view, probe_dir=probe_dir)
    assert classified["outside_reads"] == sorted(views._real(path) for path in outside_reads)
    assert classified["outside_listings"] == sorted(views._real(path) for path in (REPO_ROOT / "runs",
                                                                                    REPO_ROOT / "docs"))
    # A Python root that contains the repository does not make the repository a code root.
    wide = _synthetic_report(smoke_view, probe_dir, reads=[REPO_ROOT / "docs" / "DATASETS.md"],
                             python_roots=[str(REPO_ROOT.parent), sys.prefix])
    assert views._real(REPO_ROOT / "docs" / "DATASETS.md") in classify_probe_records(
        wide, view=smoke_view, probe_dir=probe_dir)["outside_reads"]


def test_probe_classification_treats_campaign_folders_as_outside(smoke_view, tmp_path):
    campaign = smoke_view.campaign_root
    reads = [campaign / "identity" / "fold0" / "nodes_metallo_train.tsv", campaign / views.VIEWS_DIR / "fold0" / "x"]
    report = _synthetic_report(smoke_view, tmp_path, reads=reads)
    classified = classify_probe_records(report, view=smoke_view, probe_dir=tmp_path)
    assert classified["outside_reads"] == sorted(views._real(path) for path in reads)


def test_probe_classification_checks_subprocess_programs(smoke_view, tmp_path):
    recorder = views._Recorder()
    recorder.hook("subprocess.Popen", ("/usr/bin/git", ["git", "rev-parse", "HEAD"], None, None))
    recorder.hook("subprocess.Popen", (None, ["uname", "-p"], None, None))
    recorder.hook("subprocess.Popen", ("/bin/sh", ["/bin/sh", "-c", "cat secret"], None, None))
    recorder.hook("os.system", ("ls /data",))
    recorder.hook("os.posix_spawn", (b"/usr/bin/curl", [b"curl", b"http://example.invalid"], {}))
    records = recorder.snapshot()
    assert [entry["program"] for entry in records["subprocesses"]] == ["git", "uname", "sh", "sh", "curl"]
    report = _synthetic_report(smoke_view, tmp_path, subprocesses=records["subprocesses"])
    classified = classify_probe_records(report, view=smoke_view, probe_dir=tmp_path)
    assert classified["outside_subprocesses"] == ["/bin/sh -c cat secret", "/bin/sh -c ls /data",
                                                  "curl http://example.invalid"]


def test_static_check_refuses_paths_and_caches_outside_the_view(smoke_view, source, tmp_path):
    from ec_clean30.commands import build_unit_command, check_command

    recipe = default_smoke_recipe("F2")
    command = build_unit_command(smoke_view, recipe, unit_id=smoke_unit_id(0, recipe, refit=False, seed=42),
                                 kind=KIND_INNER, inner_fold=0, seed=42, runs_dir=tmp_path / "runs", step="smoke")
    assert check_command(command, smoke_view) == []
    argv = list(command.argv)
    argv[argv.index("--esm-embeddings-dir") + 1] = str(source["esm_dir"])
    moved = replace(command, argv=tuple(argv))
    assert any("--esm-embeddings-dir" in problem and "outside the view" in problem
               for problem in check_command(moved, smoke_view))
    env = {**command.env, "DEEPMZYME_GRAPH_CACHE_DIR": str(tmp_path / "shared_graph_cache")}
    recached = replace(command, env=env)
    assert any("DEEPMZYME_GRAPH_CACHE_DIR" in problem for problem in check_command(recached, smoke_view))
    argv = list(command.argv)
    argv[argv.index("--runs-dir") + 1] = str(smoke_view.root / "runs")
    inside = replace(command, argv=tuple(argv))
    assert any("--runs-dir" in problem for problem in check_command(inside, smoke_view))
    argv[argv.index("--runs-dir") + 1] = str(smoke_view.campaign_root / views.RUNS_DIR / "fold0")
    real_runs = replace(command, argv=tuple(argv))
    assert any("--runs-dir" in problem and "forbidden read root" in problem
               for problem in check_command(real_runs, smoke_view))
