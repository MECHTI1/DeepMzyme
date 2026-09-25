"""Synthetic regressions for transfer exclusion and matched PMM result integrity."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from benchmarking import pmm_campaign_bundle as bundle
from benchmarking import pmm_comparator as comparator
from benchmarking import pmm_ion_analysis as analysis
from benchmarking import pmm_ion_campaign as campaign
from benchmarking import pmm_ion_cohort as cohort_module
from pmm_campaign_fixtures import build_train_dir, standard_entries
from training.source_cohort import COHORT_COLUMNS


class _SyntheticEstimator:
    def fit(self, features, labels):
        import numpy as np
        self.classes_ = np.array(sorted(set(labels)))
        self.n_fitted = len(features)
        return self

    def predict_proba(self, features):
        import numpy as np
        return np.full((len(features), len(self.classes_)), 1.0 / len(self.classes_))


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    source = tmp_path / "classmodel_train_set"
    source_rows, cohort, membership = [], [], []
    for fold in range(5):
        for element in comparator.ELEMENT_TO_COMMON4:
            index = len(cohort) + 1
            uid = f"u{index}"
            source_rows.append({"pdbid": f"p{index}", "residueid_ion": index, "metalid": index,
                                "label_metal": (1, 6, 7, 2)[comparator.ELEMENT_TO_COMMON4[element]], "f1": index})
            row = {key: "" for key in COHORT_COLUMNS}
            row.update(source_uid=uid, source_row=index, pdbid=f"p{index}", group_id=f"g{index}",
                       physical_ion_id=f"ion{index}", native_element=element, structure_name=f"p{index}.pdb",
                       structure_sha256="a" * 64, model_index=0, chain="A", resseq=201, resname=element,
                       atom_name=element, coord_x="0", coord_y="0", coord_z="0")
            cohort.append(row)
            membership.append({key: row[key] for key in ("source_uid", "physical_ion_id", "group_id", "native_element")}
                              | {"fold": fold})
    _write_csv(source, source_rows)
    monkeypatch.setattr(comparator, "PMM_SOURCE_SHA256", comparator.sha256_file(source))
    for index, (row, member) in enumerate(zip(cohort, membership), start=1):
        row["source_uid"] = member["source_uid"] = f"sha256:{comparator.PMM_SOURCE_SHA256}:row:{index}"
    _write_csv(root / "train_cohort.csv", cohort)
    _write_csv(root / "fold_membership.csv", membership)
    _json(root / "campaign_manifest.json", {"campaign_id": "pmm_ion_metal_v1", "cohort": {
        "sha256": comparator.sha256_file(root / "train_cohort.csv")}})
    _json(root / "fold_class_weights.json", {"fold_membership_sha256": comparator.sha256_file(root / "fold_membership.csv")})
    out = root / "pmm_comparator"
    out.mkdir()
    reports = {}
    for fold in range(5):
        rows = _predictions(membership, fold)
        path = out / f"fold{fold}_predictions.csv"
        _write_csv(path, rows)
        reports[str(fold)] = {"n_val": len(rows), "n_train": len(cohort) - len(rows),
                              "predictions_sha256": comparator.sha256_file(path)}
    receipt = {"source_file_sha256": comparator.PMM_SOURCE_SHA256,
               "cohort_sha256": comparator.sha256_file(root / "train_cohort.csv"),
               "fold_membership_sha256": comparator.sha256_file(root / "fold_membership.csv"),
               "recipe": comparator.PMM_SOURCE_RELEASE, "versions": comparator.CAMPAIGN_PMM_VERSIONS,
               "feature_columns": ["f1"], "folds": reports}
    _json(out / "pmm_comparator_manifest.json", receipt)
    return root, source, membership


def _predictions(membership, fold, native_labels=None):
    rows = []
    for member in membership:
        if int(member["fold"]) != fold:
            continue
        y4 = comparator.ELEMENT_TO_COMMON4[member["native_element"]]
        row = dict(member, y_common4=y4, pred_common4=y4)
        row.update({f"p_common4_{label.replace(' ', '_')}": float(i == y4)
                    for i, label in enumerate(comparator.COMMON4_LABELS)})
        if native_labels:
            yn = [label.upper() for label in native_labels].index(member["native_element"]) if len(native_labels) == 6 else y4
            row.update(y_native=yn, pred_native=yn, model_seed=42, checkpoint_sha256="a" * 64)
            row.update({f"p_native_{label.replace(' ', '_')}": float(i == yn) for i, label in enumerate(native_labels)})
        rows.append(row)
    return rows


def test_existing_comparator_outputs_are_reusable_without_estimator_imports(tmp_path, monkeypatch):
    root, source, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(comparator, "build_released_ensemble", lambda: pytest.fail("must not refit"))
    assert len(comparator.run_comparator(root, source)["folds"]) == 5


@pytest.mark.parametrize("mutation", ["hash", "duplicate", "wrong_fold", "wrong_ion", "wrong_label", "nan", "wrong_argmax"])
def test_comparator_rejects_invalid_prediction_artifacts(tmp_path, monkeypatch, mutation):
    root, source, _ = _fixture(tmp_path, monkeypatch)
    path = root / "pmm_comparator" / "fold0_predictions.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if mutation in {"hash", "wrong_ion"}:
        rows[0]["physical_ion_id"] = "wrong"
    elif mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "wrong_fold":
        rows[0]["fold"] = "1"
    elif mutation == "wrong_label":
        rows[0]["y_common4"] = "2"
    elif mutation == "nan":
        rows[0]["p_common4_Mn"] = "nan"
    else:
        rows[0]["pred_common4"] = "2"
    _write_csv(path, rows)
    if mutation != "hash":
        receipt_path = path.parent / "pmm_comparator_manifest.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["folds"]["0"]["predictions_sha256"] = comparator.sha256_file(path)
        _json(receipt_path, receipt)
    with pytest.raises(ValueError):
        comparator.verify_comparator_outputs(root, source)


def test_neural_probabilities_must_collapse_before_argmax(tmp_path, monkeypatch):
    _, _, membership = _fixture(tmp_path, monkeypatch)
    rows = _predictions(membership, 0, comparator.NATIVE_LABELS)
    indexed = {row["source_uid"]: row for row in membership}
    comparator.validate_prediction_rows(rows, indexed, 0, native_labels=comparator.NATIVE_LABELS,
                                         seed=42, checkpoint_sha256="a" * 64)
    rows[0]["p_common4_Mn"], rows[0]["p_common4_Class_VIII"] = 0.5, 0.5
    with pytest.raises(ValueError, match="collapse"):
        comparator.validate_prediction_rows(rows, indexed, 0, native_labels=comparator.NATIVE_LABELS)


def test_completion_requires_pmm_and_reconciled_replay(tmp_path, monkeypatch):
    root, _, membership = _fixture(tmp_path, monkeypatch)
    entries = {}
    for config in campaign.grid_configs():
        folds = {}
        labels = analysis.native_labels_for(config)
        for fold in range(5):
            rows = _predictions(membership, fold, labels)
            folds[fold] = {"run_name": f"{config.config_id}-{fold}", "rows": rows,
                           "metrics": analysis.prediction_metrics(rows, native_labels=labels),
                           "receipt": {"campaign_run_identity": {"source_tree_sha256": "a" * 64},
                                       "reconciliation_status": "match", "selected_epoch": 1,
                                       "selected_checkpoint_sha256": "a" * 64}}
        entries[config.config_id] = {"config": config, "folds": folds, "complete": True, "missing": []}
    monkeypatch.setattr(analysis, "collect_units", lambda *args: entries)
    monkeypatch.setattr(analysis, "pmm_summary", lambda *args: None)
    monkeypatch.setattr(analysis, "group_bootstrap_pooled", lambda *args: {})
    result = analysis.assess_campaign(campaign.CampaignPaths(root), tmp_path / "train")
    assert result["neural_grid_complete"] and result["status"] == "incomplete"
    next(iter(entries.values()))["folds"][0]["receipt"]["reconciliation_status"] = "mismatch"
    with pytest.raises(ValueError, match="unreconciled"):
        analysis.assess_campaign(campaign.CampaignPaths(root), tmp_path / "train")


def test_code_snapshot_excludes_heldout_source_before_opening_it(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    names = ["src/example.py", bundle.TRAIN_SOURCE,
             "prepare_training_and_test_set/pinmymetal_files/classmodel_test_set", "dataset/test/labels.csv"]
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture\n")
    monkeypatch.setattr(bundle, "REPO_ROOT", repo)

    def fake_git(command, **kwargs):
        if command[1] == "ls-files":
            return subprocess.CompletedProcess(command, 0, "\0".join(names).encode(), b"")
        return subprocess.CompletedProcess(command, 0, "fixture\n", "")

    monkeypatch.setattr(bundle.subprocess, "run", fake_git)
    archive = tmp_path / "code.tar.gz"
    result = bundle.build_code_snapshot(archive)
    with tarfile.open(archive) as handle:
        members = handle.getnames()
    assert members == sorted(["DeepMzyme/src/example.py", f"DeepMzyme/{bundle.TRAIN_SOURCE}"])
    assert result["members"] == 2


@pytest.mark.parametrize("name,kind,link", [
    ("DeepMzyme/prepare_training_and_test_set/pinmymetal_files/classmodel_test_set", "code", None),
    ("dataset/train/structures/x.pdb", "train_side", "../../test/structure.pdb"),
    ("dataset/train/structures/x.pdb", "train_side", "/outside.pdb"),
])
def test_archive_verification_rejects_excluded_data_and_links(tmp_path, name, kind, link):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo(name)
        if link:
            info.type, info.linkname = tarfile.LNKTYPE, link
            handle.addfile(info)
        else:
            info.size = 1
            handle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(bundle.BundleError):
        bundle.verify_archive(archive, kind=kind, root="DeepMzyme" if kind == "code" else "dataset/train")


def test_reference_rejects_unfrozen_route_before_opening_reference_inputs(tmp_path, monkeypatch):
    route = tmp_path / "route.json"
    _json(route, {"route": "zenodo_pmm_secondary_reference", "selection_frozen": False})
    monkeypatch.setattr(comparator, "verify_refit", lambda *args: {"cohort_sha256": "a" * 64, "model_sha256": "b" * 64})
    with pytest.raises(ValueError, match="Reference route"):
        comparator.run_reference_prediction(tmp_path, tmp_path, route, tmp_path / "absent_cohort.csv",
                                             tmp_path / "absent_source.csv", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_pmm_refit_reuse_and_one_shot_reference_prediction(tmp_path, monkeypatch):
    root, source, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(comparator, "environment_versions", lambda: comparator.CAMPAIGN_PMM_VERSIONS.copy())
    monkeypatch.setattr(comparator, "build_released_ensemble", _SyntheticEstimator)
    monkeypatch.setattr(comparator, "estimator_parameters", lambda: {"synthetic": {}})
    refit_dir = tmp_path / "refit"
    receipt = comparator.run_refit(root, refit_dir, source)
    assert receipt["status"] == "completed" and receipt["n_train"] == 30 and not receipt["test_accessed"]
    assert comparator.run_refit(root, refit_dir, source) == receipt
    route = tmp_path / "route.json"
    _json(route, {"route": "zenodo_pmm_secondary_reference", "selection_frozen": True,
                  "campaign_cohort_sha256": receipt["cohort_sha256"], "pmm_refit_model_sha256": receipt["model_sha256"],
                  "reference_source_sha256": comparator.sha256_file(source),
                  "reference_cohort_sha256": comparator.sha256_file(root / "train_cohort.csv"),
                  "deepmzyme_refit_checkpoint_sha256": "c" * 64})
    output = tmp_path / "reference_output"
    report = comparator.run_reference_prediction(root, refit_dir, route, root / "train_cohort.csv", source, output)
    assert report["status"] == "completed" and report["n_rows"] == 30
    with pytest.raises(FileExistsError):
        comparator.run_reference_prediction(root, refit_dir, route, root / "train_cohort.csv", source, output)
    (refit_dir / "pmm_model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model hash"):
        comparator.verify_refit(root, refit_dir)


def test_context_audit_distinguishes_declared_inputs_from_source_parity(tmp_path, monkeypatch):
    train, source, source_sha = build_train_dir(tmp_path / "dataset", standard_entries(2))
    monkeypatch.setattr(cohort_module, "PMM_SOURCE_SHA256", source_sha)
    root = tmp_path / "campaign"
    cohort_module.run_audit(train, source, root, workers=1)
    before = comparator.sha256_file(root / "train_cohort.csv")
    report = cohort_module.audit_training_context(train, root)
    assert report["input_contract_certified"] and not report["source_context_parity_certified"]
    assert report["n_ions_with_target_symmetry_links"] == 0
    assert comparator.sha256_file(root / "train_cohort.csv") == before


def test_source_audit_excludes_explicit_protein_symmetry_but_not_water(tmp_path, monkeypatch):
    entries = {"1sym": {"ions": [("A", 201, "", "ZN", (0, 0, 0)), ("A", 202, "", "FE", (4, 0, 0))],
                         "chains": ("A",), "rows": [("A", 201, "ZN", "7"), ("A", 202, "FE", "2")]}}
    train, source, source_sha = build_train_dir(tmp_path / "dataset", entries)
    structure = next((train / "structures").glob("*.pdb"))

    def link(element, resseq, partner):
        line = [" "] * 80
        for start, value in ((0, "LINK  "), (12, f"{element:>4}"), (17, f"{element:>3}"), (21, "A"),
                             (22, f"{resseq:>4}"), (42, " NE2"), (47, partner), (51, "A"), (52, "   1"),
                             (59, "  1555"), (66, "  2555")):
            line[start:start + len(value)] = value
        return "".join(line) + "\n"

    structure.write_text(link("ZN", 201, "HIS") + link("FE", 202, "HOH") + structure.read_text())
    manifest_path = train / "structure_manifest.csv"
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["sha256"] = comparator.sha256_file(structure)
    _write_csv(manifest_path, rows)
    monkeypatch.setattr(cohort_module, "PMM_SOURCE_SHA256", source_sha)
    root = tmp_path / "campaign"
    audit = cohort_module.run_audit(train, source, root, workers=1)
    assert audit["n_retained_ions"] == 1
    assert audit["reason_counts"]["unresolved:missing_protein_symmetry_context"] == 1
    context = cohort_module.audit_training_context(train, root)
    assert context["input_contract_certified"]
    assert context["n_ions_with_target_symmetry_links"] == 1
    assert context["n_ions_with_target_protein_symmetry_links"] == 0


def test_reference_source_audit_checks_authorization_before_reading_inputs(tmp_path, monkeypatch):
    from benchmarking import pmm_final_report

    def denied(*args):
        raise ValueError("frozen refits absent")

    monkeypatch.setattr(pmm_final_report, "require_reference_authorization", denied)
    with pytest.raises(ValueError, match="frozen refits absent"):
        cohort_module.run_reference_audit(tmp_path / "unopened", tmp_path / "unopened.csv", tmp_path / "output",
                                          route_path=tmp_path / "unopened_route.json")
    assert not (tmp_path / "output").exists()
