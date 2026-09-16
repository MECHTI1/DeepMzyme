"""Pure selection and conservative reporting for the serial metal campaign.

Inputs are verified selected-checkpoint records, never epoch histories. Native
metrics select recipes; the same checkpoints supply the common-four contrasts.
This profile cannot promote a model or authorize held-out evaluation.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


BASE_ARMS = (
    "gvp_four", "gvp_six", "esm_four", "esm_six", "late_four", "late_six",
    "early_four", "hybrid_four",
)
FOLDS = tuple(range(5))
SEEDS = (42, 43)
COMMON_CLASSES = ("Mn", "Cu", "Zn", "Class VIII")
NATIVE_CLASSES = (*COMMON_CLASSES[:3], "Fe", "Co", "Ni")
FIVE_CLASSES = (*COMMON_CLASSES[:3], "Fe", "Class VIII")
IDENTITY_FIELDS = (
    "stage", "block", "arm", "family", "scheme", "seed", "epochs",
    "parameters", "recipe_id", "fold_index",
)
CONTRASTS = (
    ("target_gvp", "gvp_four", "gvp_six"),
    ("target_esm", "esm_four", "esm_six"),
    ("target_late", "late_four", "late_six"),
    ("modality_gvp_vs_esm", "gvp_four", "esm_four"),
    ("modality_late_vs_esm", "late_four", "esm_four"),
    ("fusion_early_vs_late", "early_four", "late_four"),
    ("fusion_hybrid_vs_late", "hybrid_four", "late_four"),
)
FIVE_CONTRASTS = (
    ("fixed_five_vs_four", "late_five", "late_four"),
    ("fixed_five_vs_six", "late_five", "late_six"),
)


def _number(value, name, probability=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (probability and not 0 <= result <= 1):
        raise ValueError(f"Invalid {name}: {value!r}")
    return result


def _recalls(row, collapsed=False):
    key = "collapsed4_per_class_recall" if collapsed else "per_class_recall"
    values = row.get(key)
    expected = COMMON_CLASSES if collapsed else {
        "four_class": COMMON_CLASSES, "five_class": FIVE_CLASSES, "six_class": NATIVE_CLASSES,
    }[row["scheme"]]
    if not isinstance(values, dict) or set(values) != set(expected):
        raise ValueError(f"{row['id']}: {key} must contain exactly {expected}")
    return {name: _number(values[name], f"{key}.{name}", True) for name in expected}


def _validate_metrics(row):
    if row.get("scheme") not in ("four_class", "five_class", "six_class"):
        raise ValueError(f"{row['id']}: unsupported scheme")
    for prefix, collapsed in (("", False), ("collapsed4_", True)):
        score = _number(row.get(prefix + "balanced_accuracy"), prefix + "balanced_accuracy", True)
        minimum = _number(row.get(prefix + "minimum_recall"), prefix + "minimum_recall", True)
        recalls = _recalls(row, collapsed)
        if not math.isclose(score, statistics.mean(recalls.values()), abs_tol=1e-6):
            raise ValueError(f"{row['id']}: {prefix}balanced accuracy disagrees with recalls")
        if not math.isclose(minimum, min(recalls.values()), abs_tol=1e-6):
            raise ValueError(f"{row['id']}: {prefix}minimum recall disagrees with recalls")
    if row["scheme"] == "four_class":
        native, collapsed = _recalls(row), _recalls(row, True)
        if any(not math.isclose(native[name], collapsed[name], abs_tol=1e-6) for name in COMMON_CLASSES):
            raise ValueError(f"{row['id']}: direct-four native and collapsed metrics disagree")
    epoch = row.get("selected_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= row["epochs"]:
        raise ValueError(f"{row['id']}: invalid selected_epoch")
    if row.get("training_balanced_accuracy") is not None:
        _number(row["training_balanced_accuracy"], "training_balanced_accuracy", True)
    if row.get("parameter_count") is not None:
        count = _number(row["parameter_count"], "parameter_count")
        if count <= 0 or not count.is_integer():
            raise ValueError("parameter_count must be a positive integer")


def _indexed(runs, results):
    """Reject duplicate and mismatched records without silently taking the last."""
    expected, indexed, errors, duplicate_ids = {}, {}, [], set()
    for run in runs:
        key = run["id"]
        if key in expected:
            errors.append(f"Duplicate planned run id: {key}")
            duplicate_ids.add(key)
        expected[key] = run
    seen = set()
    for result in results:
        key = result.get("id")
        if key in seen:
            errors.append(f"Duplicate result id: {key}")
            duplicate_ids.add(key)
            continue
        seen.add(key)
        if key not in expected:
            errors.append(f"Unplanned result id: {key}")
            continue
        run = expected[key]
        try:
            if any(result.get(field) for field in ("held_out_evaluation", "run_test_eval", "test_report")):
                raise ValueError(f"{key}: held-out evaluation is forbidden")
            for field in IDENTITY_FIELDS:
                if result.get(field) != run.get(field):
                    raise ValueError(f"{key}: result differs from planned {field}")
            for field in ("lr", "ring", "folds_sha256", "cohort_sha256"):
                if field in run and result.get(field) != run[field]:
                    raise ValueError(f"{key}: result differs from planned {field}")
            _validate_metrics(result)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
            continue
        indexed[key] = {**run, **result}
    for key in duplicate_ids:
        indexed.pop(key, None)
    return indexed, errors


def _complexity(row):
    # This predeclared proxy remains comparable even when measured counts are absent.
    value = row.get("complexity_proxy")
    if value is not None:
        result = _number(value, "complexity_proxy")
        if result < 0:
            raise ValueError("complexity_proxy must be nonnegative")
        return result
    return {"compact": 0, "reference": 1}.get(row.get("parameters", {}).get("capacity"), 2)


def _rank_rows(rows, score="balanced_accuracy", recall="minimum_recall"):
    """Use measured parameter counts within metric ties when all are available."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row[score], row[recall])].append(row)
    ranked = []
    for key in sorted(groups, reverse=True):
        group = groups[key]
        measured = all(row.get("parameter_count") is not None for row in group)
        ranked.extend(sorted(group, key=lambda row: (
            row["parameter_count"] if measured else _complexity(row),
            _complexity(row), row["recipe_id"], row["id"],
        )))
    return ranked


def _selection_index(runs, results):
    discovery = [run for run in runs if run.get("stage") == "discovery"]
    allowed = {run["id"] for run in discovery}
    selected = [row for row in results if row.get("id") in allowed]
    indexed, errors = _indexed(discovery, selected)
    cohort_ids = {row.get("cohort_sha256") for row in indexed.values()}
    if indexed and (None in cohort_ids or "" in cohort_ids or len(cohort_ids) != 1):
        errors.append("Discovery runs must retain the same certified train/validation cohort")
    if errors:
        raise ValueError("; ".join(errors))
    return discovery, indexed


def choose_screen_candidates(runs, results):
    """Choose two native-selected recipes per arm after the six-cell screen."""
    discovery, indexed = _selection_index(runs, results)
    screen = [run for run in discovery if run.get("block") == "screen"]
    chosen = {}
    if {run["arm"] for run in screen} != set(BASE_ARMS):
        raise ValueError("Screen must declare all eight required arms")
    for arm in BASE_ARMS:
        grid = [run for run in screen if run["arm"] == arm]
        cells = {(run["parameters"].get("capacity"), run["parameters"].get("learning_rate"))
                 for run in grid}
        expected_cells = {(capacity, lr) for capacity in ("compact", "reference")
                          for lr in (1e-5, 3e-5, 1e-4)}
        if (len(grid) != 6 or cells != expected_cells
                or any(run["seed"] != 42 or run["epochs"] != 50 for run in grid)):
            raise ValueError(f"{arm}: screen requires the six predeclared LR/capacity cells at seed 42")
        if any(run["id"] not in indexed for run in grid):
            raise ValueError(f"{arm}: incomplete screen")
        ranked = _rank_rows([indexed[run["id"]] for run in grid])
        chosen[arm] = [dict(next(run for run in grid if run["id"] == row["id"])) for row in ranked[:2]]
    return chosen


def freeze_candidates(runs, results):
    """Select only fully repeated recipes; no collapsed score selects a recipe."""
    top_two = choose_screen_candidates(runs, results)
    discovery, indexed = _selection_index(runs, results)
    recipes = defaultdict(dict)
    for run in discovery:
        if run["id"] not in indexed or run["arm"] not in BASE_ARMS:
            continue
        if run["seed"] not in SEEDS:
            raise ValueError(f"{run['id']}: unexpected discovery seed")
        key = (run["arm"], run["recipe_id"])
        if run["seed"] in recipes[key]:
            raise ValueError(f"Duplicate discovery recipe/seed unit: {key}, {run['seed']}")
        recipes[key][run["seed"]] = indexed[run["id"]]
    chosen = {}
    for arm in BASE_ARMS:
        mandatory = {row["recipe_id"] for row in top_two.get(arm, [])}
        mandatory.update(run["recipe_id"] for run in discovery if run["arm"] == arm and run["block"] == "large")
        for recipe_id in mandatory:
            if set(recipes.get((arm, recipe_id), {})) != set(SEEDS):
                raise ValueError(f"{arm}: no complete seed-42/43 recipe for mandatory {recipe_id}")
        completed = []
        for (recipe_arm, _), rows in recipes.items():
            if recipe_arm != arm or set(rows) != set(SEEDS):
                continue
            first, second = rows[42], rows[43]
            for field in ("parameters", "family", "scheme", "epochs"):
                if first[field] != second[field]:
                    raise ValueError(f"{arm}: repeated recipe differs in {field}")
            counts = {row["parameter_count"] for row in rows.values() if row.get("parameter_count") is not None}
            if len(counts) > 1:
                raise ValueError(f"{arm}: repeated recipe has different parameter counts")
            completed.append({**first,
                "mean_native_ba": statistics.mean(row["balanced_accuracy"] for row in rows.values()),
                "mean_native_min_recall": statistics.mean(row["minimum_recall"] for row in rows.values()),
                "parameter_count": next(iter(counts)) if counts else None,
            })
        if not completed:
            raise ValueError(f"{arm}: no complete seed-42/43 recipe")
        winner = _rank_rows(completed, "mean_native_ba", "mean_native_min_recall")[0]["id"]
        chosen[arm] = dict(next(run for run in discovery if run["id"] == winner))
    return chosen


def _quantile(values, probability):
    position = (len(values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return values[lower] + fraction * (values[min(lower + 1, len(values) - 1)] - values[lower])


def paired_fold_bootstrap(left, right, *, resamples=10000, seed=42):
    """Pair complete fold/seed units, averaging seeds before resampling folds."""
    expected = {(fold, seed_value) for fold in FOLDS for seed_value in SEEDS}
    if set(left) != expected or set(right) != expected:
        raise ValueError("Paired bootstrap requires all five folds and both shared seeds")
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    differences = [statistics.mean(
        _number(left[(fold, model_seed)], "left score", True)
        - _number(right[(fold, model_seed)], "right score", True)
        for model_seed in SEEDS
    ) for fold in FOLDS]
    rng = random.Random(seed)
    draws = sorted(sum(differences[rng.randrange(len(FOLDS))] for _ in FOLDS) / len(FOLDS)
                   for _ in range(resamples))
    return {
        "mean_difference": statistics.mean(differences),
        "ci_lower": _quantile(draws, .025), "ci_upper": _quantile(draws, .975),
        "fold_differences": differences, "resamples": resamples, "bootstrap_seed": seed,
        "bootstrap_unit": "fold after averaging the two common model seeds",
    }


def _arm_summary(arm, rows, planned_count, complete):
    record = {"arm": arm, "planned_runs": planned_count, "completed_runs": len(rows), "complete": complete}
    if not rows:
        return record
    record.update(family=rows[0]["family"], scheme=rows[0]["scheme"])
    for field in ("balanced_accuracy", "collapsed4_balanced_accuracy", "minimum_recall",
                  "collapsed4_minimum_recall"):
        values = [row[field] for row in rows]
        record["mean_" + field] = statistics.mean(values)
        record["sd_" + field] = statistics.stdev(values) if len(values) > 1 else None
    record["mean_per_class_recall"] = {
        name: statistics.mean(_recalls(row)[name] for row in rows) for name in _recalls(rows[0])
    }
    record["mean_collapsed4_per_class_recall"] = {
        name: statistics.mean(_recalls(row, True)[name] for row in rows) for name in COMMON_CLASSES
    }
    for name in ("Fe", "Co", "Ni"):
        record["native_" + name + "_recall"] = record["mean_per_class_recall"].get(name)
    record["native_CoNi_recall"] = record["mean_per_class_recall"].get("Class VIII") if record["scheme"] == "five_class" else None
    record["native_class_viii_meaning"] = "Co+Ni" if record["scheme"] == "five_class" else "Fe+Co+Ni"
    return record


def _confirmation_report(confirmation, indexed, errors):
    runs = confirmation.get("runs", [])
    expected_units = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    by_arm, units, complete_arms = defaultdict(list), {}, {}
    for run in runs:
        by_arm[run["arm"]].append(run)
    for fold in FOLDS:
        cohorts = {indexed[run["id"]].get("cohort_sha256") for run in runs
                   if run.get("fold_index") == fold and run["id"] in indexed}
        if len(cohorts) > 1:
            errors.append(f"Fold {fold}: cohort differs across candidates or model seeds")
    for arm, planned in by_arm.items():
        rows = [indexed[run["id"]] for run in planned if run["id"] in indexed]
        keys = [(run.get("fold_index"), run["seed"]) for run in planned]
        valid = len(keys) == len(set(keys)) == 10 and set(keys) == expected_units
        if not valid:
            errors.append(f"{arm}: confirmation manifest lacks unique five-fold/two-seed units")
        if any(run["epochs"] != 50 for run in planned):
            errors.append(f"{arm}: confirmation must keep the frozen 50-epoch budget")
            valid = False
        if len({run["recipe_id"] for run in planned}) != 1:
            errors.append(f"{arm}: confirmation mixes frozen recipes")
            valid = False
        if any(run["parameters"] != planned[0]["parameters"] for run in planned):
            errors.append(f"{arm}: parameters changed within confirmation")
            valid = False
        candidate = confirmation.get("candidates", {}).get(arm)
        if arm in BASE_ARMS and candidate is None:
            errors.append(f"{arm}: missing frozen candidate")
            valid = False
        if candidate is not None and any(
            run["parameters"] != candidate["parameters"] or run["recipe_id"] != candidate["recipe_id"]
            for run in planned
        ):
            errors.append(f"{arm}: confirmation differs from frozen candidate")
            valid = False
        for row in rows:
            if not row.get("cohort_sha256") or not row.get("folds_sha256"):
                errors.append(f"{row['id']}: missing cohort/fold identity")
                valid = False
            if row.get("folds_sha256") != confirmation.get("folds_sha256"):
                errors.append(f"{row['id']}: changed confirmation fold definitions")
                valid = False
        complete_arms[arm] = valid and len(rows) == 10
        units[arm] = {(row["fold_index"], row["seed"]): row for row in rows}
    summaries = [_arm_summary(arm, list(units[arm].values()), len(planned), complete_arms[arm])
                 for arm, planned in sorted(by_arm.items())]
    comparisons = []
    contrasts = list(CONTRASTS)
    if "late_five" in by_arm or "five" in confirmation.get("planned_blocks", []):
        contrasts.extend(FIVE_CONTRASTS)
    ring_off = "gvp_ring_off" if "gvp_ring_off" in by_arm else "gvp_four"
    if "ring" in confirmation.get("planned_blocks", []) or "gvp_ring_on" in by_arm:
        contrasts.append(("ring_on_vs_off", "gvp_ring_on", ring_off))
    for name, left_arm, right_arm in contrasts:
        item = {"contrast": name, "left_arm": left_arm, "right_arm": right_arm,
                "status": "incomplete", "paired_improvement": False, "promoted": False}
        if not complete_arms.get(left_arm) or not complete_arms.get(right_arm):
            comparisons.append(item)
            continue
        left, right = units[left_arm], units[right_arm]
        mismatch = any(left[unit]["cohort_sha256"] != right[unit]["cohort_sha256"] for unit in expected_units)
        if name == "ring_on_vs_off":
            certificate = confirmation.get("ring_input_audit", {})
            fold_receipts = certificate.get("folds", {})
            if certificate.get("verified") is not True or any(str(fold) not in fold_receipts for fold in FOLDS):
                item["reason"] = "Missing verified input certificates for all five RING folds"
                comparisons.append(item)
                continue
            if any(fold_receipts[str(fold)].get("status") != "passed" or
                   fold_receipts[str(fold)].get("cohort_sha256") != left[(fold, seed)]["cohort_sha256"]
                   for fold in FOLDS for seed in SEEDS):
                item.update(status="invalid", reason="RING input certificate disagrees with paired cohort")
                errors.append("ring_on_vs_off: invalid RING input certificate")
                comparisons.append(item)
                continue
            mismatch = mismatch or any(
                not left[unit].get("normalization_sha256")
                or not right[unit].get("normalization_sha256")
                for unit in expected_units
            )
            for unit in expected_units:
                for field in ("family", "scheme", "epochs"):
                    mismatch = mismatch or left[unit][field] != right[unit][field]
                def non_ring_parameters(row):
                    return {key: value for key, value in row["parameters"].items()
                            if key not in ("ring", "ring_edge_mode", "use_ring")}
                mismatch = mismatch or non_ring_parameters(left[unit]) != non_ring_parameters(right[unit])
        if mismatch:
            item.update(status="invalid", reason="Mismatched paired inputs or missing RING normalization binding")
            errors.append(f"{name}: mismatched paired inputs")
            comparisons.append(item)
            continue
        bootstrap = paired_fold_bootstrap(
            {key: row["collapsed4_balanced_accuracy"] for key, row in left.items()},
            {key: row["collapsed4_balanced_accuracy"] for key, row in right.items()},
        )
        drops = {label: statistics.mean(
            _recalls(left[unit], True)[label] - _recalls(right[unit], True)[label]
            for unit in sorted(expected_units)
        ) for label in COMMON_CLASSES}
        protected = all(value >= -.03 - 1e-12 for value in drops.values())
        nonzero = all(statistics.mean(_recalls(row, True)[label] for row in left.values()) > 0
                      for label in COMMON_CLASSES)
        positive = bootstrap["mean_difference"] > 0 and bootstrap["ci_lower"] > 0
        item.update(bootstrap, status="complete", per_class_recall_differences=drops,
                    rare_recall_protected=protected and nonzero,
                    paired_improvement=positive and protected and nonzero,
                    practical_tie=abs(bootstrap["mean_difference"]) <= .002)
        if item["practical_tie"]:
            use_counts = all(row.get("parameter_count") is not None for row in (*left.values(), *right.values()))
            def tie_key(arm_rows):
                fold_means = [statistics.mean(arm_rows[(fold, seed)]["collapsed4_balanced_accuracy"]
                                             for seed in SEEDS) for fold in FOLDS]
                return (-statistics.mean(row["collapsed4_minimum_recall"] for row in arm_rows.values()),
                        statistics.stdev(fold_means),
                        next(iter(arm_rows.values()))["parameter_count"] if use_counts else _complexity(next(iter(arm_rows.values()))),
                        _complexity(next(iter(arm_rows.values()))))
            left_key, right_key = tie_key(left), tie_key(right)
            item["descriptive_tie_preference"] = (left_arm if left_key < right_key else
                                                  right_arm if right_key < left_key else "unresolved")
        comparisons.append(item)
    required_arms = set(BASE_ARMS)
    if "late_five" in confirmation.get("candidates", {}) or "five" in confirmation.get("planned_blocks", []):
        required_arms.add("late_five")
    if "ring" in confirmation.get("planned_blocks", []):
        required_arms.add("gvp_ring_on")
    complete = (required_arms <= set(by_arm) and all(complete_arms.values())
                and all(row["status"] == "complete" for row in comparisons) and not errors)
    # Any integrity failure invalidates the report as a whole, including an
    # otherwise attractive pair. Incompleteness alone does not erase valid data.
    if errors:
        for item in comparisons:
            item["paired_improvement"] = False
            if item["status"] == "complete":
                item["status"] = "invalid_report"
    return {"complete": complete, "arms": summaries, "pairwise": comparisons,
            "planned_blocks": confirmation.get("planned_blocks", []),
            "deferred_blocks": confirmation.get("deferred_blocks", [])}


def _five_challenger(confirmation, diagnostics):
    """Historical promise never substitutes for the current shared-fold gates."""
    arm = next((row for row in confirmation["arms"] if row["arm"] == "late_five"), None)
    pairs = {row["contrast"]: row for row in confirmation["pairwise"]}
    complete = bool(arm and arm["complete"] and not diagnostics and all(
        pairs.get(name, {}).get("status") == "complete" for name, _, _ in FIVE_CONTRASTS))
    if complete:
        status = "shared_fold_complete"
    elif arm is not None or "five" in confirmation["planned_blocks"]:
        status = "incomplete"
    elif "five" in confirmation["deferred_blocks"]:
        status = "deferred"
    else:
        status = "unresolved"
    limitation = (
        "The fixed late-five contender has complete shared-fold comparisons with late-four and late-six; "
        "superiority still requires the directional paired-CI and per-class recall gates and remains exploratory."
        if complete else
        "A selected four- or six-class model cannot be claimed to beat all target formulations: "
        "the fixed late-five contender lacks complete shared-fold confirmation against both."
    )
    return dict(arm="late_five", role="fixed_recipe_contender_no_discovery_search", status=status,
                shared_fold_complete=complete, required_runs=10, completed_runs=arm["completed_runs"] if arm else 0,
                predeclared_contrasts=[name for name, _, _ in FIVE_CONTRASTS],
                native_class_viii="Co+Ni", common_four_class_viii="Fe+Co+Ni",
                historical_reference=dict(common_four_balanced_accuracy_percent=74.718, sample_sd_percent=2.486,
                    seeds=[42, 43], evidence_status="historical_fixed_split_validation_only",
                    current_shared_fold_evidence=False,
                    source="docs/notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md"),
                claim_limitation=limitation)


def build_report(manifest, results, confirmation_manifest=None):
    """Build a JSON-compatible report without reading checkpoints or test files."""
    discovery_runs = manifest.get("runs", [])
    confirmation_runs = (confirmation_manifest or {}).get("runs", [])
    all_runs = [*discovery_runs, *confirmation_runs]
    indexed, errors = _indexed(all_runs, results)
    report = {
        "profile": manifest.get("profile"), "evidence_status": "exploratory_validation_only",
        "held_out_evaluation": False, "promoted": False,
        "stage6b_allowed": False, "stage7_allowed": False,
        "final_test_route": "unresolved", "completed_runs": len(indexed),
        "planned_runs": len(all_runs), "diagnostics": errors,
        "checkpoint_rule": "native validation balanced accuracy; same checkpoint for common-four reporting",
        "limitations": [
            "No held-out reporting or model promotion is authorized by this profile.",
            "Fold bootstrap is conditional on frozen, development-selected recipes; it does not remove selection bias.",
            "The predeclared pairwise intervals are exploratory and not multiplicity-adjusted.",
            "Missing runs and deferred blocks remain incomplete coverage.",
        ],
        "runs": list(indexed.values()),
    }
    try:
        report["screen_candidates"] = choose_screen_candidates(discovery_runs, results)
    except ValueError as exc:
        report["screen_candidates"] = {}
        report["screen_status"] = str(exc)
    try:
        report["frozen_candidate_preview"] = freeze_candidates(discovery_runs, results)
    except ValueError as exc:
        report["frozen_candidate_preview"] = {}
        report["freeze_status"] = str(exc)
    if confirmation_manifest is not None:
        report["confirmation"] = _confirmation_report(confirmation_manifest, indexed, errors)
    else:
        report["confirmation"] = {"complete": False, "arms": [], "pairwise": [],
                                  "planned_blocks": [], "deferred_blocks": []}
    report["fixed_five_challenger"] = _five_challenger(report["confirmation"], errors)
    report["limitations"].append(report["fixed_five_challenger"]["claim_limitation"])
    cap_rows = [row for row in indexed.values() if row["epochs"] == 50
                and "potentially_training_budget_limited" in row]
    flagged_ids = sorted(row["id"] for row in cap_rows if row["potentially_training_budget_limited"] is True)
    report["epoch50_diagnostics"] = dict(evaluated_runs=len(cap_rows),
        potentially_training_budget_limited_count=len(flagged_ids), run_ids=flagged_ids,
        rule="selected epoch >= 46 OR mean native BA(46:50) minus mean native BA(41:45) > 0.002",
        interpretation="advisory_only_no_extension_or_checkpoint_reselection")
    return report


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(value), sort_keys=True) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def write_report(output, manifest, results, confirmation_manifest=None):
    """Write portable results, confirmation tables and a human-readable decision."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = build_report(manifest, results, confirmation_manifest)
    paths = {"json": output / "campaign_report.json", "runs": output / "campaign_results.csv",
             "confirmation": output / "confirmation_summary.csv", "pairwise": output / "confirmation_pairwise.csv",
             "markdown": output / "campaign_report.md"}
    report["artifacts"] = {key: str(path) for key, path in paths.items()}
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
    _write_csv(paths["runs"], report["runs"])
    _write_csv(paths["confirmation"], report["confirmation"]["arms"])
    _write_csv(paths["pairwise"], report["confirmation"]["pairwise"])
    five = report["fixed_five_challenger"]
    cap = report["epoch50_diagnostics"]
    lines = ["# Serial metal campaign report", "", "Status: exploratory validation only. No model is promoted.", "",
             "The final-test route is unresolved. Stage 6B and Stage 7 remain blocked; held-out evaluation is disabled.", "",
             f"Verified result records: {report['completed_runs']} / {report['planned_runs']} planned.",
             f"Required confirmation coverage complete: {report['confirmation']['complete']}.", "",
             "Planned confirmation blocks: " + (", ".join(report["confirmation"]["planned_blocks"]) or "not frozen") + ".",
             "Deferred confirmation blocks: " + (", ".join(report["confirmation"]["deferred_blocks"]) or "none recorded") + ".", "",
             f"Fixed late-five contender: {five['status']} ({five['completed_runs']} / 10 confirmation runs).",
             "Historical common-four validation reference: **74.718 ± 2.486%** across seeds 42/43 on one fixed split; "
             "this is historical evidence, not current shared-fold confirmation.",
             five["claim_limitation"], "",
             f"Epoch-50 potentially training-budget-limited runs: {cap['potentially_training_budget_limited_count']} "
             f"of {cap['evaluated_runs']} with recorded diagnostics.",
             "Flagged run IDs: " + (", ".join("`" + ident + "`" for ident in cap["run_ids"]) or "none") + ".",
             "These flags are advisory and do not extend training or reselect checkpoints.", "",
             "Native validation metrics select checkpoints and discovery recipes. Every common-four score comes from the same checkpoint.", "",
             "Five-class native Class VIII means Co+Ni; common-four Class VIII always means Fe+Co+Ni.", "",
             "| Arm | Complete | Mean native BA | Mean common-four BA | Native Fe / Co / Ni recall | Native Co+Ni recall (five-class) |",
             "|---|---|---|---|---|---|"]
    for row in report["confirmation"]["arms"]:
        def fmt(value):
            return "—" if value is None else f"{value:.6f}"
        lines.append(f"| {row['arm']} | {row['complete']} | {fmt(row.get('mean_balanced_accuracy'))} | "
                     f"{fmt(row.get('mean_collapsed4_balanced_accuracy'))} | "
                     + " / ".join(fmt(row.get(f"native_{name}_recall")) for name in ("Fe", "Co", "Ni"))
                     + " | " + fmt(row.get("native_CoNi_recall")) + " |")
    lines += ["", "| Predeclared contrast (left minus right) | Status | Mean difference | 95% fold CI | Recall gate |",
              "|---|---|---|---|---|"]
    for row in report["confirmation"]["pairwise"]:
        interval = (f"[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]" if "ci_lower" in row else "—")
        delta = f"{row['mean_difference']:.6f}" if "mean_difference" in row else "—"
        lines.append(f"| {row['contrast']} | {row['status']} | {delta} | {interval} | "
                     f"{row.get('rare_recall_protected', 'not evaluated')} |")
    lines += ["", "CIs use 10,000 seeded bootstrap draws over five fold differences after averaging the two shared seeds within each fold.",
              "A positive interval and protected class recalls describe an exploratory paired improvement, not promotion. "
              "Practical ties use mean minimum recall, fold stability, then complexity descriptively.", ""]
    lines += ["- " + note for note in report["limitations"]]
    if report["diagnostics"]:
        lines += ["", "Integrity diagnostics:", ""] + ["- " + item for item in report["diagnostics"]]
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
