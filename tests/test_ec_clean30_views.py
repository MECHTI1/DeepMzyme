"""Training-only views and the dry-run probe of the EC class vs CLEAN30 runner (synthetic data only)."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_clean30_fixtures import (  # noqa: E402
    AMBIGUOUS_ACCESSION, build_smoke_view, make_writable, unlock, write_clean30_data_root, write_inner_membership,
)
from ec_clean30 import views  # noqa: E402
from ec_clean30.campaign import KIND_INNER, KIND_REFIT, default_smoke_recipe, smoke_unit_id  # noqa: E402
from ec_clean30.provenance import REPO_ROOT, SRC_ROOT, HeldOutInputError, sha256_file  # noqa: E402
from ec_clean30.views import (  # noqa: E402
    SourceLayout, View, ViewError, ViewVerificationError, build_view, classify_probe_records, forbidden_read_roots,
    verify_view,
)
from training.ec_membership import read_ec_fold_membership, write_ec_fold_membership  # noqa: E402

PROVENANCE = {"note": "synthetic test provenance"}


def _layout(paths: dict) -> SourceLayout:
    return SourceLayout.from_data_root(paths["data_root"])


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
    out = tmp_path_factory.mktemp("campaign") / "fold0_view"
    return build_view(fold=0, layout=_layout(source), out_dir=out, provenance=PROVENANCE)


@pytest.fixture(scope="module")
def smoke_view(source, tmp_path_factory):
    return _smoke(source, tmp_path_factory.mktemp("campaign_smoke") / "view")


# ---------------------------------------------------------------------------
# Build


def test_fold_view_holds_exactly_the_training_inputs(fold_view, source):
    root = Path(fold_view["view"])
    view = View.load(root)
    stems = sorted(source["stems"].values())
    assert view.structure_ids() == stems
    assert view.manifest["kind"] == views.KIND_FOLD_VIEW
    assert view.manifest["eligible"]["n"] == len(stems) - 1
    assert [stem.split("__")[0] for stem in view.manifest["eligible"]["excluded_structure_ids"]] == [AMBIGUOUS_ACCESSION]
    assert view.manifest_sha256 == fold_view["manifest_sha256"] == sha256_file(root / views.VIEW_MANIFEST_NAME)
    assert view.manifest["source"]["training_manifest"]["sha256"] == sha256_file(source["train_manifest"])
    assert view.manifest["source"]["training_csv"]["sha256"] == sha256_file(source["training_csv"])
    assert sha256_file(view.summary) == sha256_file(source["training_csv"])
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


def test_view_build_never_opens_test_side_files(fold_view, source):
    if os.geteuid() == 0:
        pytest.skip("permission checks do not bind root")
    for path in source["decoys"]["locked"]:
        assert not os.access(path, os.R_OK)
    assert Path(fold_view["view"]).is_dir()


def test_fold_view_verifies_and_records_a_receipt(fold_view):
    receipt = verify_view(fold_view["view"], provenance=PROVENANCE)
    assert receipt["status"] == "verified" and receipt["n_structures"] == 13 and receipt["n_eligible"] == 12
    stored = json.loads((Path(fold_view["view"]) / "receipts" / views.VERIFY_RECEIPT_NAME).read_text())
    assert stored["view_manifest_sha256"] == fold_view["manifest_sha256"]
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
    root = tmp_path / "view"
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
        build_view(fold=0, layout=layout, out_dir=source["data_root"] / "CLEAN_30_main" / "view", provenance=PROVENANCE)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "file.txt").write_text("x")
    with pytest.raises(ViewError, match="not empty"):
        build_view(fold=0, layout=layout, out_dir=occupied, provenance=PROVENANCE)
    with pytest.raises(HeldOutInputError):
        build_view(fold=0, layout=layout, out_dir=tmp_path / "fold0_test", provenance=PROVENANCE)
    with pytest.raises(ViewError, match="smoke subset"):
        build_view(fold=0, layout=layout, out_dir=tmp_path / "v1", seed=1, provenance=PROVENANCE)
    with pytest.raises(ViewError, match="max-proteins"):
        build_view(fold=0, layout=layout, out_dir=tmp_path / "v2", smoke=True, max_proteins=1, seed=1,
                   provenance=PROVENANCE)


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
    with pytest.raises(HeldOutInputError, match="test or mixed metadata folder"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=tmp_path / "view", provenance=PROVENANCE)
    assert not (tmp_path / "view").exists()


def test_build_refuses_symlinked_sources(fresh_source, tmp_path):
    target = _source_object(fresh_source)
    copy = tmp_path / "elsewhere.pdb"
    shutil.copyfile(target, copy)
    target.unlink()
    os.symlink(copy, target)
    with pytest.raises(ViewError, match="symlink"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=tmp_path / "view", provenance=PROVENANCE)
    assert not (tmp_path / "view").exists()
    assert not any(path.name.startswith(".view.partial") for path in tmp_path.iterdir())


def test_build_refuses_hash_mismatch_and_cohort_mismatch(fresh_source, tmp_path):
    target = _source_object(fresh_source)
    original = target.read_bytes()
    target.write_bytes(original.replace(b"ATOM", b"ATOM", 1) + b"REMARK changed\n")
    with pytest.raises(ViewError, match="size .* differs|SHA-256 differs"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=tmp_path / "v1", provenance=PROVENANCE)
    target.write_bytes(original)
    _rewrite_manifest(fresh_source, lambda rows: rows[1:])
    with pytest.raises(ViewError, match="must describe the same proteins"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=tmp_path / "v2", provenance=PROVENANCE)


def test_build_needs_every_exact_input_and_leaves_nothing_on_failure(fresh_source, tmp_path):
    stem = sorted(fresh_source["stems"].values())[0]
    (fresh_source["external"] / stem / "residue_features.json").unlink()
    with pytest.raises(ViewError, match="required inputs are missing"):
        build_view(fold=0, layout=_layout(fresh_source), out_dir=tmp_path / "view", provenance=PROVENANCE)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source"]


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
    again = _smoke(source, tmp_path / "again", chain_means=False)
    assert again.structure_ids() == ids and again.membership["sha256"] == view.membership["sha256"]
    other = _smoke(source, tmp_path / "other", seed=7, chain_means=False)
    assert other.structure_ids() != ids
    receipt = verify_view(view.root, require_chain_means=("final",), write_receipt=False, provenance=PROVENANCE)
    assert receipt["chain_means"]["final"]["n_keys"] == 8


def test_fold_view_copies_a_certified_inner_membership(source, tmp_path):
    from ec_clean30.identity import INNER_CERTIFICATE

    assert views.INNER_CERTIFICATE_NAME == INNER_CERTIFICATE
    membership, digest, certificate = write_inner_membership(source, tmp_path / "identity")
    result = build_view(fold=0, layout=_layout(source), out_dir=tmp_path / "view", inner_membership=(membership, digest),
                        provenance=PROVENANCE)
    record = result["manifest"]["membership"]
    assert record["role"] == "inner" and record["n_folds"] == 4
    assert record["certificate_sha256"] == sha256_file(certificate)
    assert sha256_file(tmp_path / "view" / views.INNER_CERTIFICATE_NAME) == sha256_file(certificate)
    receipt = verify_view(tmp_path / "view", write_receipt=False, provenance=PROVENANCE)
    assert receipt["membership"]["certificate_sha256"] == record["certificate_sha256"]
    eligible = [stem for stem in sorted(source["stems"].values()) if not stem.startswith(AMBIGUOUS_ACCESSION)]
    rows = [{"structure_id": stem, "accession": stem.split("__")[0], "group_id": stem.split("__")[0],
             "fold": str(index % 4)} for index, stem in enumerate(eligible)]
    short = tmp_path / "short" / "inner_membership_fold0.csv"
    short.parent.mkdir()
    short_digest = write_ec_fold_membership(rows[1:], short)
    with pytest.raises(ViewError, match="certify-inner receipt"):
        build_view(fold=0, layout=_layout(source), out_dir=tmp_path / "view2", inner_membership=(short, short_digest),
                   provenance=PROVENANCE)
    with pytest.raises(ViewError, match="another membership file"):
        build_view(fold=0, layout=_layout(source), out_dir=tmp_path / "view3", inner_membership=(short, short_digest),
                   inner_certificate=certificate, provenance=PROVENANCE)
    short_certificate = json.loads(certificate.read_text())
    short_certificate["membership"] = {"file": short.name, "sha256": short_digest}
    (short.parent / views.INNER_CERTIFICATE_NAME).write_text(json.dumps(short_certificate))
    with pytest.raises(ViewError, match="must be identical"):
        build_view(fold=0, layout=_layout(source), out_dir=tmp_path / "view3", inner_membership=(short, short_digest),
                   provenance=PROVENANCE)
    _uncertified, uncertified_digest, uncertified = write_inner_membership(source, tmp_path / "refused",
                                                                         certified=False)
    with pytest.raises(ViewError, match="not a passed certify-inner receipt"):
        build_view(fold=0, layout=_layout(source), out_dir=tmp_path / "view4",
                   inner_membership=(_uncertified, uncertified_digest), provenance=PROVENANCE)
    with pytest.raises(ViewError, match="not fold 1"):
        build_view(fold=1, layout=_layout(source), out_dir=tmp_path / "view5", inner_membership=(membership, digest),
                   provenance=PROVENANCE)
    assert not any((tmp_path / name).exists() for name in ("view2", "view3", "view4", "view5"))
    assert uncertified.is_file()


def test_verify_refuses_a_changed_inner_certificate(source, tmp_path):
    membership, digest, _certificate = write_inner_membership(source, tmp_path / "identity")
    root = tmp_path / "view"
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


def test_forbidden_roots_cover_shared_and_test_bearing_roots(source, smoke_view):
    roots = forbidden_read_roots(_layout(source))
    data = str(source["data_root"])
    for name in ("CLEAN_30_main", "CLEAN_30_train_test_split_0", "CLEAN_30_train_test_split_4", "esm_embeddings",
                 "updated_feature_extraction", "RING_features", "structure_store", "CLEAN_all_train_valid_splits",
                 "CARE_dataset"):
        assert os.path.join(data, name) in roots
    for relative in ("prepare_training_and_test_set", "CLEAN_prepare_training_and_test_set",
                     "CARE_prepare_training_and_test_set", "docs/notebook_outputs", "runs"):
        assert str(REPO_ROOT / relative) in roots
    assert not any(str(smoke_view.root).startswith(root + os.sep) for root in roots)
    assert smoke_view.forbidden_read_roots == forbidden_read_roots(_layout(source), view_root=smoke_view.root) == roots


def test_forbidden_roots_follow_the_data_root_that_holds_the_view(source, tmp_path):
    other_disk = tmp_path / "other_disk" / "DeepMzyme_Data"
    (other_disk / "campaigns").mkdir(parents=True)
    project_data = tmp_path / "project" / "DeepMzyme_Data"
    project_data.mkdir(parents=True)
    os.symlink(other_disk / "campaigns", project_data / "campaigns")
    view_root = project_data / "campaigns" / "ecc30" / "fold0_view"
    roots = forbidden_read_roots(_layout(source), view_root=view_root)
    for base in (project_data, other_disk):
        for name in ("esm_embeddings", "RING_features", "updated_feature_extraction", "CLEAN_30_main"):
            assert str(base / name) in roots
    result = build_view(fold=0, layout=_layout(source), out_dir=view_root, provenance=PROVENANCE)
    assert str(other_disk / "esm_embeddings") in result["manifest"]["forbidden_read_roots"]
    make_writable(Path(result["view"]))


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
    stored = json.loads((smoke_view.receipts / views.DRY_RUN_RECEIPT_NAME).read_text())
    assert stored["status"] == "passed" and stored["view_manifest_sha256"] == smoke_view.manifest_sha256


def test_dry_run_catches_a_read_outside_the_view(source, tmp_path):
    from ec_clean30.commands import dry_run

    root = tmp_path / "view"
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

    root = tmp_path / "view"
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
