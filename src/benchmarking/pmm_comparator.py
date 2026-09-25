"""Matched PinMyMetal comparator: refit the released recipe on each training fold.

Recipe: ``data_model/train_chedhclassmodel.py`` of hhz-lab/PinMyMetal at the pinned
commit — ``dropna()``, drop identifier/label columns, soft-voting ensemble of
LogisticRegression, BalancedRandomForest, MLP, SVC(probability) and EasyEnsemble
with the released hyperparameters and ``random_state=1``; no feature scaling.
Resampling happens inside the two imbalanced-learn estimators, i.e. inside each
training fold only. A released full-training fitted model is never used.

Run it with the campaign pins (Python 3.11.5, scikit-learn 1.3.0,
numpy 1.23.5, pandas 2.1.4, joblib 1.2.0, imbalanced-learn 0.11.0): newer scikit-learn rejects the released
``liblinear`` multiclass logistic regression.

Adaptations (recorded in the manifest):
- the released drop list names a ``source`` column absent from the released
  ``classmodel_train_set``; only columns present are dropped;
- rows are restricted to the frozen DeepMzyme cohort (retained UIDs), so both
  systems are evaluated on identical ions; ``dropna`` is recorded but removes no row
  of the pinned source.

Outputs per fold: ``fold{k}_predictions.csv`` with the same identity and common-four
columns as DeepMzyme validation predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Deliberately dependency-free imports: this module runs in the pinned PinMyMetal
# environment, which has no torch.
from pmm_source_release import (  # noqa: E402
    PMM_NON_FEATURE_COLUMNS,
    PMM_RELEASED_ENVIRONMENT,
    PMM_SOURCE_DEFAULT,
    PMM_SOURCE_RELEASE,
    PMM_SOURCE_SHA256,
    sha256_file,
)

# PinMyMetal class codes in the released source -> common-four index.
PMM_CODE_TO_COMMON4 = {1: 0, 6: 1, 7: 2, 2: 3}
COMMON4_LABELS = ("Mn", "Cu", "Zn", "Class VIII")
N_FOLDS = 5
NATIVE_LABELS = ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")
ELEMENT_TO_COMMON4 = {"MN": 0, "CU": 1, "ZN": 2, "FE": 3, "CO": 3, "NI": 3}
CAMPAIGN_PMM_VERSIONS = {"python": "3.11.5", "scikit_learn": "1.3.0", "numpy": "1.23.5",
                         "pandas": "2.1.4", "joblib": "1.2.0", "imbalanced_learn": "0.11.0"}
PROBABILITY_TOLERANCE = 2.0e-6  # CSV exports round each probability to eight decimal places.


def _read_unique_rows(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row[key]: row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError(f"{path}: empty table or repeated {key}")
    return indexed


def read_campaign_contract(campaign_dir: Path) -> tuple[dict, dict, dict]:
    """Read only frozen training metadata; reject duplicates and inconsistent identities."""
    manifest = json.loads((campaign_dir / "campaign_manifest.json").read_text(encoding="utf-8"))
    cohort_path, folds_path = campaign_dir / "train_cohort.csv", campaign_dir / "fold_membership.csv"
    if sha256_file(cohort_path) != manifest["cohort"]["sha256"]:
        raise ValueError("Campaign cohort changed after freezing")
    cohort = _read_unique_rows(cohort_path, "source_uid")
    membership = _read_unique_rows(folds_path, "source_uid")
    weights = json.loads((campaign_dir / "fold_class_weights.json").read_text(encoding="utf-8"))
    if sha256_file(folds_path) != weights["fold_membership_sha256"]:
        raise ValueError("Campaign fold membership changed after freezing")
    if set(cohort) != set(membership):
        raise ValueError("Cohort and frozen folds have different source UIDs")
    if len({row["physical_ion_id"] for row in cohort.values()}) != len(cohort):
        raise ValueError("Cohort repeats a physical ion")
    groups: dict[str, set[int]] = {}
    for uid, row in membership.items():
        if any(row[key] != cohort[uid][key] for key in ("physical_ion_id", "group_id", "native_element")):
            raise ValueError(f"{uid}: cohort/fold identity mismatch")
        fold = int(row["fold"])
        if fold not in range(N_FOLDS):
            raise ValueError(f"{uid}: invalid fold {fold}")
        groups.setdefault(row["group_id"], set()).add(fold)
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("An identity group crosses frozen folds")
    return manifest, membership, cohort


def verify_source_row_bindings(source_rows: list[dict], cohort: dict[str, dict], source_sha256: str) -> None:
    seen = set()
    for uid, row in cohort.items():
        index = int(row["source_row"])
        if index in seen or index < 1 or index > len(source_rows) or uid != f"sha256:{source_sha256}:row:{index}":
            raise ValueError(f"{uid}: source-row provenance is not reversible")
        source = source_rows[index - 1]
        if (str(source["pdbid"]).lower() != row["pdbid"].lower()
                or PMM_CODE_TO_COMMON4[int(source["label_metal"])] != ELEMENT_TO_COMMON4[row["native_element"].upper()]):
            raise ValueError(f"{uid}: pinned source PDB or label disagrees with the cohort")
        seen.add(index)


def validate_prediction_rows(rows: list[dict[str, str]], membership: dict[str, dict[str, str]], fold: int,
                             *, native_labels: tuple[str, ...] | None = None,
                             seed: int | None = None, checkpoint_sha256: str | None = None) -> None:
    """Validate UID alignment, labels, finite probabilities and probability-first collapse."""
    expected = {uid: row for uid, row in membership.items() if int(row["fold"]) == fold}
    actual = {row["source_uid"]: row for row in rows}
    if not rows or len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError(f"Fold {fold}: predictions do not cover its frozen UIDs exactly once")
    for uid, row in actual.items():
        member = expected[uid]
        if int(row["fold"]) != fold or any(row[key] != member[key] for key in ("physical_ion_id", "group_id", "native_element")):
            raise ValueError(f"{uid}: prediction fold or ion identity differs from the cohort")
        element = member["native_element"].upper()
        y4 = ELEMENT_TO_COMMON4[element]
        if int(row["y_common4"]) != y4:
            raise ValueError(f"{uid}: common-four label differs from the frozen native element")

        def probabilities(prefix: str, labels: tuple[str, ...]) -> list[float]:
            values = [float(row[f"{prefix}_{label.replace(' ', '_')}"]) for label in labels]
            if (any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values)
                    or abs(sum(values) - 1.0) > PROBABILITY_TOLERANCE):
                raise ValueError(f"{uid}: invalid {prefix} probability vector")
            return values

        def validate_argmax(values: list[float], prediction: int) -> None:
            if prediction not in range(len(values)) or max(values) - values[prediction] > PROBABILITY_TOLERANCE:
                raise ValueError(f"{uid}: prediction disagrees with its probability vector")

        p4 = probabilities("p_common4", COMMON4_LABELS)
        validate_argmax(p4, int(row["pred_common4"]))
        if native_labels is not None:
            pn = probabilities("p_native", native_labels)
            yn = [label.upper() for label in native_labels].index(element) if len(native_labels) == 6 else y4
            if int(row["y_native"]) != yn:
                raise ValueError(f"{uid}: native label differs from the frozen element")
            validate_argmax(pn, int(row["pred_native"]))
            collapsed = pn[:3] + [sum(pn[3:])] if len(native_labels) == 6 else pn
            if any(abs(a - b) > PROBABILITY_TOLERANCE for a, b in zip(p4, collapsed)):
                raise ValueError(f"{uid}: common-four probabilities are not the native probability collapse")
        if seed is not None and int(row["model_seed"]) != seed:
            raise ValueError(f"{uid}: model seed differs from run identity")
        if checkpoint_sha256 is not None and row["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError(f"{uid}: checkpoint hash differs from the selected checkpoint")


def verify_comparator_outputs(campaign_dir: Path, pmm_source_csv: Path = PMM_SOURCE_DEFAULT) -> dict[str, Any]:
    """Verify existing fivefold PMM outputs without fitting or loading estimator weights."""
    campaign, membership, cohort = read_campaign_contract(campaign_dir)
    manifest = json.loads((campaign_dir / "pmm_comparator" / "pmm_comparator_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(pmm_source_csv) != PMM_SOURCE_SHA256 or manifest["source_file_sha256"] != PMM_SOURCE_SHA256:
        raise ValueError("PMM source is not the pinned training release")
    if (manifest["cohort_sha256"] != campaign["cohort"]["sha256"]
            or manifest["fold_membership_sha256"] != sha256_file(campaign_dir / "fold_membership.csv")):
        raise ValueError("PMM outputs belong to a different cohort or fold membership")
    if any(manifest["recipe"].get(key) != value for key, value in PMM_SOURCE_RELEASE.items()):
        raise ValueError("PMM outputs use a different source recipe")
    if manifest["versions"] != CAMPAIGN_PMM_VERSIONS:
        raise ValueError("PMM output environment differs from the frozen campaign environment")
    with pmm_source_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        source_rows = list(reader)
    verify_source_row_bindings(source_rows, cohort, PMM_SOURCE_SHA256)
    expected_columns = [column for column in columns if column not in PMM_NON_FEATURE_COLUMNS]
    if manifest["feature_columns"] != expected_columns:
        raise ValueError("PMM outputs use a different feature recipe")
    for fold in range(N_FOLDS):
        path = campaign_dir / "pmm_comparator" / f"fold{fold}_predictions.csv"
        receipt = manifest["folds"][str(fold)]
        if sha256_file(path) != receipt["predictions_sha256"]:
            raise ValueError(f"PMM fold {fold}: prediction hash mismatch")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        validate_prediction_rows(rows, membership, fold)
        if receipt["n_val"] != len(rows) or receipt["n_train"] != len(cohort) - len(rows):
            raise ValueError(f"PMM fold {fold}: sample counts mismatch")
    return manifest


def build_released_ensemble():
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    return VotingClassifier(estimators=[
        ("LR", LogisticRegression(solver="liblinear", C=1, penalty="l2", random_state=1)),
        ("RF", BalancedRandomForestClassifier(n_estimators=50, max_depth=10, random_state=1)),
        ("MLP", MLPClassifier(solver="adam", hidden_layer_sizes=(100,), alpha=0.0001, activation="relu", random_state=1)),
        ("SVM", SVC(kernel="rbf", gamma="scale", C=100, probability=True, random_state=1)),
        ("Easy", EasyEnsembleClassifier(n_estimators=100, random_state=1)),
    ], voting="soft")


def environment_versions() -> dict[str, str]:
    return {"python": platform.python_version(),
            **{key: __import__(module).__version__ for key, module in (
                ("scikit_learn", "sklearn"), ("imbalanced_learn", "imblearn"),
                ("numpy", "numpy"), ("pandas", "pandas"), ("joblib", "joblib"))}}


def estimator_parameters() -> dict[str, dict[str, str]]:
    return {name: {key: repr(value) for key, value in estimator.get_params().items()}
            for name, estimator in build_released_ensemble().estimators}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_refit(campaign_dir: Path, refit_dir: Path) -> dict[str, Any]:
    campaign, _membership, cohort = read_campaign_contract(campaign_dir)
    receipt = json.loads((refit_dir / "pmm_refit_receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("fit_status") != "completed" or receipt["cohort_sha256"] != campaign["cohort"]["sha256"]
            or receipt["source_file_sha256"] != PMM_SOURCE_SHA256 or receipt["n_train"] != len(cohort)
            or receipt["versions"] != CAMPAIGN_PMM_VERSIONS
            or receipt["source_release"] != PMM_SOURCE_RELEASE):
        raise ValueError("PMM refit receipt is incompatible with the frozen campaign")
    if sha256_file(refit_dir / "pmm_model.joblib") != receipt["model_sha256"]:
        raise ValueError("PMM refit model hash changed")
    return receipt


def run_refit(campaign_dir: Path, refit_dir: Path, pmm_source_csv: Path = PMM_SOURCE_DEFAULT) -> dict[str, Any]:
    """Fit the frozen released recipe once on all eligible non-test training rows."""
    if (refit_dir / "pmm_refit_receipt.json").is_file():
        return verify_refit(campaign_dir, refit_dir)
    import joblib

    versions = environment_versions()
    if versions != CAMPAIGN_PMM_VERSIONS:
        raise ValueError(f"PMM refitting requires {CAMPAIGN_PMM_VERSIONS}")
    source, _membership, cohort = load_inputs(campaign_dir, pmm_source_csv)
    feature_columns = [column for column in source.columns if column not in PMM_NON_FEATURE_COLUMNS and column != "source_row"]
    source = source.dropna()
    source = source[source["source_row"].isin({int(row["source_row"]) for row in cohort.values()})]
    if len(source) != len(cohort):
        raise ValueError("PMM refit source does not cover the eligible cohort")
    refit_dir.mkdir(parents=True, exist_ok=True)
    with (refit_dir / "attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": time.time(), "cohort_sha256": sha256_file(campaign_dir / "train_cohort.csv")}, handle)
    started = time.time()
    model = build_released_ensemble()
    model.fit(source[feature_columns], source["label_metal"])
    model_path = refit_dir / "pmm_model.joblib"
    joblib.dump(model, model_path)
    receipt = {"fit_status": "completed", "cohort_sha256": sha256_file(campaign_dir / "train_cohort.csv"),
               "status": "completed",
               "source_file_sha256": PMM_SOURCE_SHA256, "source_release": PMM_SOURCE_RELEASE,
               "n_train": len(source), "feature_columns": feature_columns, "class_order": [int(v) for v in model.classes_],
               "resolved_estimator_params": estimator_parameters(), "versions": versions,
               "model_sha256": sha256_file(model_path), "model_path": model_path.name,
               "fit_seconds": time.time() - started, "test_accessed": False}
    _write_json(refit_dir / "pmm_refit_receipt.json", receipt)
    return receipt


def run_reference_prediction(campaign_dir: Path, refit_dir: Path, route_path: Path,
                             reference_cohort_csv: Path, reference_source_csv: Path,
                             output_dir: Path) -> dict[str, Any]:
    """Predict one declared secondary reference report after validating frozen training state.

    The caller owns the global reporting lock and authorizes reference access. This
    additional exclusive reservation prevents accidental repeated comparator calls.
    No reference data are opened until route and completed refit checks pass.
    """
    route = json.loads(route_path.read_text(encoding="utf-8"))
    receipt = verify_refit(campaign_dir, refit_dir)
    if (route.get("route") != "zenodo_pmm_secondary_reference" or route.get("selection_frozen") is not True
            or route.get("campaign_cohort_sha256") != receipt["cohort_sha256"]
            or route.get("pmm_refit_model_sha256") != receipt["model_sha256"]
            or not all(isinstance(route.get(key), str) and len(route[key]) == 64 for key in
                       ("reference_source_sha256", "reference_cohort_sha256", "deepmzyme_refit_checkpoint_sha256"))):
        raise ValueError("Reference route does not bind both frozen refits and the secondary reference inputs")
    if environment_versions() != CAMPAIGN_PMM_VERSIONS:
        raise ValueError("Reference prediction requires the frozen PMM environment")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": time.time(), "route_sha256": sha256_file(route_path),
                   "pmm_model_sha256": receipt["model_sha256"]}, handle)
    # Reference access starts here, after training identity and one-shot admission.
    if (sha256_file(reference_source_csv) != route["reference_source_sha256"]
            or sha256_file(reference_cohort_csv) != route["reference_cohort_sha256"]):
        raise ValueError("Reference inputs differ from the frozen reporting route")
    import joblib
    import numpy as np
    import pandas as pd

    cohort = _read_unique_rows(reference_cohort_csv, "source_uid")
    source = pd.read_csv(reference_source_csv)
    verify_source_row_bindings(source.to_dict("records"), cohort, route["reference_source_sha256"])
    source["source_row"] = range(1, len(source) + 1)
    uid_by_row = {int(row["source_row"]): uid for uid, row in cohort.items()}
    if len(uid_by_row) != len(cohort):
        raise ValueError("Reference cohort repeats source rows")
    source = source.dropna()
    source = source[source["source_row"].isin(uid_by_row)].copy()
    if len(source) != len(cohort):
        raise ValueError("Reference source does not cover the certified cohort")
    model = joblib.load(refit_dir / "pmm_model.joblib")
    probabilities = model.predict_proba(source[receipt["feature_columns"]])
    common4 = np.zeros((len(source), 4))
    for column, code in enumerate(model.classes_):
        common4[:, PMM_CODE_TO_COMMON4[int(code)]] += probabilities[:, column]
    rows = []
    for (_, row), probabilities in zip(source.iterrows(), common4):
        uid = uid_by_row[int(row["source_row"])]
        member = cohort[uid]
        rows.append({"source_uid": uid, "physical_ion_id": member["physical_ion_id"],
                     "group_id": member["group_id"], "native_element": member["native_element"], "fold": 0,
                     "y_common4": PMM_CODE_TO_COMMON4[int(row["label_metal"])], "pred_common4": int(probabilities.argmax()),
                     **{f"p_common4_{label.replace(' ', '_')}": f"{value:.8f}"
                        for label, value in zip(COMMON4_LABELS, probabilities)}})
    validate_prediction_rows(rows, {uid: {**member, "fold": 0} for uid, member in cohort.items()}, 0)
    prediction_path = output_dir / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "completed", "route_sha256": sha256_file(route_path), "n_rows": len(rows),
              "pmm_refit_model_sha256": receipt["model_sha256"], "predictions_sha256": sha256_file(prediction_path),
              "reference_source_sha256": route["reference_source_sha256"],
              "reference_cohort_sha256": route["reference_cohort_sha256"],
              "scope": "secondary possibly overlapping PMM reference; not the primary held-out route"}
    _write_json(output_dir / "receipt.json", result)
    return result


def load_inputs(campaign_dir: Path, pmm_source_csv: Path):
    import pandas as pd

    if sha256_file(pmm_source_csv) != PMM_SOURCE_SHA256:
        raise ValueError("PinMyMetal source differs from the pinned training source")
    source = pd.read_csv(pmm_source_csv)
    source["source_row"] = range(1, len(source) + 1)
    _manifest, membership, cohort = read_campaign_contract(campaign_dir)
    verify_source_row_bindings(source.to_dict("records"), cohort, PMM_SOURCE_SHA256)
    return source, membership, cohort


def run_comparator(campaign_dir: Path, pmm_source_csv: Path = PMM_SOURCE_DEFAULT) -> dict[str, Any]:
    if (campaign_dir / "pmm_comparator" / "pmm_comparator_manifest.json").is_file():
        return verify_comparator_outputs(campaign_dir, pmm_source_csv)
    import numpy as np
    import sklearn

    import imblearn

    versions = environment_versions()
    if versions != CAMPAIGN_PMM_VERSIONS:
        raise ValueError(f"PMM requires the frozen environment {CAMPAIGN_PMM_VERSIONS}, got {versions}")

    source, membership, cohort = load_inputs(campaign_dir, pmm_source_csv)
    n_before = len(source)
    source = source.dropna()
    dropped_by_dropna = n_before - len(source)
    present_drop = [column for column in PMM_NON_FEATURE_COLUMNS if column in source.columns]
    absent_drop = [column for column in PMM_NON_FEATURE_COLUMNS if column not in source.columns]
    feature_columns = [column for column in source.columns if column not in present_drop and column != "source_row"]
    uid_by_row = {int(row["source_row"]): uid for uid, row in cohort.items()}
    source = source[source["source_row"].isin(uid_by_row)].copy()
    source["source_uid"] = source["source_row"].map(uid_by_row)
    if set(source["source_uid"]) != set(membership):
        raise ValueError("PinMyMetal feature rows do not cover the frozen cohort exactly")
    source["fold"] = source["source_uid"].map(lambda uid: int(membership[uid]["fold"]))
    source["y_common4"] = source["label_metal"].map(lambda code: PMM_CODE_TO_COMMON4[int(code)])

    out_dir = campaign_dir / "pmm_comparator"
    out_dir.mkdir(exist_ok=True)
    fold_reports = {}
    for fold in range(N_FOLDS):
        train = source[source["fold"] != fold]
        val = source[source["fold"] == fold]
        started = time.time()
        model = build_released_ensemble()
        model.fit(train[feature_columns], train["label_metal"])
        probabilities = model.predict_proba(val[feature_columns])
        classes = [int(code) for code in model.classes_]
        common4 = np.zeros((len(val), 4))
        for column, code in enumerate(classes):
            common4[:, PMM_CODE_TO_COMMON4[code]] += probabilities[:, column]
        rows = []
        for (_, row), probs in zip(val.iterrows(), common4):
            member = membership[row["source_uid"]]
            rows.append({
                "source_uid": row["source_uid"], "physical_ion_id": member["physical_ion_id"],
                "group_id": member["group_id"], "native_element": member["native_element"], "fold": fold,
                "y_common4": int(row["y_common4"]), "pred_common4": int(probs.argmax()),
                **{f"p_common4_{label.replace(' ', '_')}": f"{value:.8f}" for label, value in zip(COMMON4_LABELS, probs)},
            })
        validate_prediction_rows(rows, membership, fold)
        path = out_dir / f"fold{fold}_predictions.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        fold_reports[str(fold)] = {"n_train": len(train), "n_val": len(val), "fit_seconds": time.time() - started,
                                   "predictions_sha256": sha256_file(path), "class_order": classes}
        print(f"[PMM] fold {fold}: {len(train)} train / {len(val)} val in {time.time() - started:.0f}s", flush=True)
    manifest = {
        "recipe": {**PMM_SOURCE_RELEASE, "estimators": "LR + BalancedRF + MLP + SVC + EasyEnsemble, soft voting",
                   "scaling": "none (as released)", "random_state": 1},
        "source_file_sha256": PMM_SOURCE_SHA256,
        "class_code_order": {"1": "Mn", "6": "Cu", "7": "Zn", "2": "Class VIII (Fe+Co+Ni)"},
        "feature_columns": feature_columns,
        "n_feature_columns": len(feature_columns),
        "adaptations": {
            "released_drop_columns_absent_from_source": absent_drop,
            "rows_restricted_to_frozen_cohort": True,
            "dropna_rows_removed": int(dropped_by_dropna),
            "resampling": "internal to imbalanced-learn estimators, within each training fold",
        },
        "fold_membership_sha256": sha256_file(campaign_dir / "fold_membership.csv"),
        "cohort_sha256": sha256_file(campaign_dir / "train_cohort.csv"),
        "versions": versions,
        "released_environment_pins": PMM_RELEASED_ENVIRONMENT,
        "resolved_estimator_params": {name: {key: repr(value) for key, value in estimator.get_params().items()}
                                      for name, estimator in build_released_ensemble().estimators},
        "implementation_sha256": sha256_file(Path(__file__)),
        "folds": fold_reports,
        "comparison_scope": "classification at known sites; PinMyMetal's published features vs DeepMzyme ESM/geometry inputs",
    }
    (out_dir / "pmm_comparator_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                          encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--pmm-source-csv", type=Path, default=PMM_SOURCE_DEFAULT)
    parser.add_argument("--action", choices=("cv", "verify", "refit", "reference-predict"), default="cv")
    parser.add_argument("--refit-dir", type=Path)
    parser.add_argument("--reference-route-json", type=Path)
    parser.add_argument("--reference-cohort-csv", type=Path)
    parser.add_argument("--reference-source-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.action == "verify":
        result = verify_comparator_outputs(args.campaign_dir, args.pmm_source_csv)
        print(json.dumps({"status": "verified", "n_folds": len(result["folds"])}))
        return
    if args.action == "refit":
        if args.refit_dir is None:
            parser.error("--refit-dir is required for refit")
        print(json.dumps(run_refit(args.campaign_dir, args.refit_dir, args.pmm_source_csv), indent=2))
        return
    if args.action == "reference-predict":
        if any(getattr(args, key) is None for key in ("refit_dir", "reference_route_json", "reference_cohort_csv",
                                                     "reference_source_csv", "output_dir")):
            parser.error("reference-predict requires refit, route, reference cohort/source and output paths")
        print(json.dumps(run_reference_prediction(args.campaign_dir, args.refit_dir, args.reference_route_json,
                                                 args.reference_cohort_csv, args.reference_source_csv, args.output_dir), indent=2))
        return
    manifest = run_comparator(args.campaign_dir, args.pmm_source_csv)
    print(json.dumps({"folds": manifest["folds"], "adaptations": manifest["adaptations"]}, indent=2))


if __name__ == "__main__":
    main()
