"""Synthetic scientific-gate tests; no data loading, training, or GPU access."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import reporting


def run(arm, *, capacity="compact", lr=1e-5, seed=42, fold=None, stage="discovery", block="screen"):
    scheme = "six_class" if arm.endswith("six") else "five_class" if arm.endswith("five") else "four_class"
    recipe = f"{arm}_{capacity}_{lr}"
    row = dict(id=f"{recipe}_{seed}_{fold}", stage=stage, block=block, arm=arm,
               family=arm.split("_")[0], scheme=scheme, seed=seed, epochs=50,
               parameters={"capacity": capacity, "learning_rate": lr},
               recipe_id=recipe, complexity_proxy=1 if capacity == "compact" else 2,
               run_dir=f"/unused/{recipe}")
    if fold is not None:
        row["fold_index"] = fold
    return row


def result(row, native=.6, common=None, native_recalls=None, common_recalls=None):
    common = native if common is None else common
    native_names = {"six_class": reporting.NATIVE_CLASSES, "five_class": reporting.FIVE_CLASSES,
                    "four_class": reporting.COMMON_CLASSES}[row["scheme"]]
    native_recalls = native_recalls or dict.fromkeys(native_names, native)
    common_recalls = common_recalls or dict.fromkeys(reporting.COMMON_CLASSES, common)
    return {**row, "balanced_accuracy": sum(native_recalls.values()) / len(native_recalls),
            "collapsed4_balanced_accuracy": sum(common_recalls.values()) / len(common_recalls),
            "minimum_recall": min(native_recalls.values()), "collapsed4_minimum_recall": min(common_recalls.values()),
            "per_class_recall": native_recalls, "collapsed4_per_class_recall": common_recalls,
            "selected_epoch": 50, "cohort_sha256": f"cohort-{row.get('fold_index', 'discovery')}",
            "folds_sha256": "frozen-folds", "normalization_sha256": "fixed-normalization"}


def discovery():
    runs = [run(arm, capacity=capacity, lr=lr) for arm in reporting.BASE_ARMS
            for capacity in ("compact", "reference") for lr in (1e-5, 3e-5, 1e-4)]
    return runs, [result(row) for row in runs]


def confirmation():
    candidates = {arm: run(arm) for arm in reporting.BASE_ARMS}
    runs = [run(arm, seed=seed, fold=fold, stage="confirmation",
                block="fusion" if arm.startswith(("early", "hybrid")) else "core")
            for arm in reporting.BASE_ARMS for fold in range(5) for seed in (42, 43)]
    manifest = dict(runs=runs, candidates=candidates, planned_blocks=["core", "fusion"],
                    deferred_blocks=["ring"], cohort_sha256="all-development", folds_sha256="frozen-folds")
    return manifest, [result(row) for row in runs]


def test_screen_requires_complete_grid_and_uses_native_checkpoint_score():
    runs, results = discovery()
    native_winner = next(row for row in runs if row["arm"] == "gvp_six" and
                         row["parameters"] == {"capacity": "reference", "learning_rate": 1e-4})
    results = [result(row, native=.8, common=.3) if row["id"] == native_winner["id"] else
               result(row, native=.6, common=.9 if row["scheme"] == "six_class" else .6)
               for row in runs]
    chosen = reporting.choose_screen_candidates(runs, results)
    assert chosen["gvp_six"][0]["id"] == native_winner["id"]
    assert len(chosen["gvp_six"]) == 2
    assert chosen["gvp_four"][0]["parameters"]["capacity"] == "compact"
    with pytest.raises(ValueError, match="incomplete screen"):
        reporting.choose_screen_candidates(runs, results[:-1])
    with pytest.raises(ValueError, match="Duplicate result"):
        reporting.choose_screen_candidates(runs, results + [results[0]])


def add_repeats(runs, results):
    for candidates in reporting.choose_screen_candidates(runs, results).values():
        for selected in candidates:
            repeat = {**selected, "id": selected["id"] + "_repeat", "seed": 43, "block": "repeat"}
            runs.append(repeat)
            results.append(result(repeat, native=.65))


def test_freeze_requires_both_top_candidates_and_uses_repeated_recipe_mean():
    runs, results = discovery()
    add_repeats(runs, results)
    chosen = reporting.freeze_candidates(runs, results)
    assert set(chosen) == set(reporting.BASE_ARMS)
    assert chosen["gvp_four"]["parameters"]["capacity"] == "compact"
    with pytest.raises(ValueError, match="no complete"):
        reporting.freeze_candidates(runs, results[:-1])


def test_freeze_detects_mismatched_recipe_metadata():
    runs, results = discovery()
    add_repeats(runs, results)
    runs[-1]["parameters"] = {**runs[-1]["parameters"], "learning_rate": .0002}
    results[-1] = result(runs[-1])
    with pytest.raises(ValueError, match="differs in parameters"):
        reporting.freeze_candidates(runs, results)


def test_screen_rejects_changed_cohort():
    runs, results = discovery()
    results[0]["cohort_sha256"] = "other-cohort"
    with pytest.raises(ValueError, match="same certified"):
        reporting.choose_screen_candidates(runs, results)


def test_bootstrap_averages_seeds_within_fold_before_resampling():
    left = {(fold, seed): .5 + (.2 if seed == 42 else -.2) + fold * .01
            for fold in range(5) for seed in (42, 43)}
    right = {key: .5 for key in left}
    report = reporting.paired_fold_bootstrap(left, right)
    assert report["fold_differences"] == pytest.approx([0, .01, .02, .03, .04])
    assert report["mean_difference"] == pytest.approx(.02)
    assert report["ci_upper"] < .05  # Seed scatter was not misused as independent fold scatter.
    assert report == reporting.paired_fold_bootstrap(left, right)
    with pytest.raises(ValueError, match="five folds"):
        reporting.paired_fold_bootstrap(dict(list(left.items())[:-1]), right)


def test_complete_confirmation_stays_exploratory_and_preserves_native_six_recalls(tmp_path):
    manifest, results = confirmation()
    report = reporting.write_report(tmp_path, {"profile": "test", "runs": []}, results, manifest)
    assert report["confirmation"]["complete"]
    assert not report["promoted"] and not report["stage6b_allowed"] and not report["stage7_allowed"]
    assert all(row["practical_tie"] for row in report["confirmation"]["pairwise"])
    assert all(not row["paired_improvement"] for row in report["confirmation"]["pairwise"])
    six = next(row for row in report["confirmation"]["arms"] if row["arm"] == "gvp_six")
    assert six["native_Fe_recall"] == .6 and six["native_Ni_recall"] == .6
    persisted = json.loads((tmp_path / "campaign_report.json").read_text())
    assert persisted["held_out_evaluation"] is False
    assert "Native Fe / Co / Ni" in (tmp_path / "campaign_report.md").read_text()


def test_positive_ba_interval_is_blocked_by_one_class_recall_drop():
    manifest, results = confirmation()
    harmed = dict(Mn=.99, Cu=.99, Zn=.99, **{"Class VIII": .5})
    results = [result(row, native_recalls=harmed, common_recalls=harmed)
               if row["arm"] == "gvp_four" else row for row in results]
    report = reporting.build_report({"runs": []}, results, manifest)
    pair = next(row for row in report["confirmation"]["pairwise"] if row["contrast"] == "target_gvp")
    assert pair["ci_lower"] > 0
    assert not pair["rare_recall_protected"] and not pair["paired_improvement"]


def test_positive_paired_interval_is_descriptive_and_never_promotes():
    manifest, results = confirmation()
    results = [result(row, native=.7) if row["arm"] == "gvp_four" else row for row in results]
    report = reporting.build_report({"runs": []}, results, manifest)
    pair = next(row for row in report["confirmation"]["pairwise"] if row["contrast"] == "target_gvp")
    assert pair["paired_improvement"] and pair["rare_recall_protected"]
    assert pair["mean_difference"] == pytest.approx(.1)
    assert not pair["promoted"] and not report["promoted"]


@pytest.mark.parametrize("fault", ["duplicate", "nonfinite", "metadata", "cohort", "folds", "unit", "recipe"])
def test_corrupt_confirmation_cannot_claim_completion_or_improvement(fault):
    manifest, results = confirmation()
    results = [result(row, native=.8) if row["arm"] == "gvp_four" else row for row in results]
    if fault == "duplicate":
        results.append(copy.deepcopy(results[0]))
    elif fault == "nonfinite":
        results[0]["balanced_accuracy"] = float("nan")
    elif fault == "metadata":
        results[0]["seed"] = 99
    elif fault == "cohort":
        results[0]["cohort_sha256"] = "other-cohort"
    elif fault == "folds":
        results[0]["folds_sha256"] = "changed-folds"
    elif fault == "unit":
        manifest["runs"][0]["fold_index"] = 1
        results[0]["fold_index"] = 1
    elif fault == "recipe":
        manifest["runs"][0]["recipe_id"] = "changed-recipe"
        results[0]["recipe_id"] = "changed-recipe"
    report = reporting.build_report({"runs": []}, results, manifest)
    assert not report["confirmation"]["complete"]
    assert report["diagnostics"]
    assert all(not row["paired_improvement"] for row in report["confirmation"]["pairwise"])


def test_missing_run_and_ring_block_remain_incomplete():
    manifest, results = confirmation()
    manifest["planned_blocks"].append("ring")
    report = reporting.build_report({"runs": []}, results[:-1], manifest)
    assert not report["confirmation"]["complete"]
    pairs = {row["contrast"]: row for row in report["confirmation"]["pairwise"]}
    assert pairs["fusion_hybrid_vs_late"]["status"] == "incomplete"
    assert pairs["ring_on_vs_off"]["status"] == "incomplete"


def test_ring_requires_bound_normalization():
    manifest, results = confirmation()
    manifest["planned_blocks"].append("ring")
    manifest["ring_input_audit"] = {"verified": True, "folds": {
        str(fold): {"status": "passed", "cohort_sha256": f"cohort-{fold}"} for fold in range(5)}}
    for fold in range(5):
        for seed in (42, 43):
            row = run("gvp_ring_on", fold=fold, seed=seed, stage="confirmation", block="ring")
            manifest["runs"].append(row)
            measured = result(row, native=.7)
            measured["normalization_sha256"] = None
            results.append(measured)
    report = reporting.build_report({"runs": []}, results, manifest)
    pair = report["confirmation"]["pairwise"][-1]
    assert pair["status"] == "invalid" and not pair["paired_improvement"]
    assert not report["confirmation"]["complete"]


def test_confirmation_does_not_allow_seed_dependent_fold_membership():
    manifest, results = confirmation()
    for row in results:
        if row["seed"] == 43:
            row["cohort_sha256"] += "-different-for-seed-43"
    report = reporting.build_report({"runs": []}, results, manifest)
    assert not report["confirmation"]["complete"]
    assert any("model seeds" in item for item in report["diagnostics"])


def test_held_out_record_is_rejected_without_opening_files():
    manifest, results = confirmation()
    results[0]["test_report"] = {"accuracy": .9}
    report = reporting.build_report({"runs": []}, results, manifest)
    assert not report["confirmation"]["complete"]
    assert any("held-out evaluation is forbidden" in item for item in report["diagnostics"])


def test_measured_parameter_count_precedes_bundled_capacity_proxy():
    runs, results = discovery()
    for row in results:
        row["parameter_count"] = 100 if row["parameters"]["capacity"] == "reference" else 200
    chosen = reporting.choose_screen_candidates(runs, results)
    assert all(rows[0]["parameters"]["capacity"] == "reference" for rows in chosen.values())
    # A partially measured tie uses a common proxy, not incompatible units.
    results[0].pop("parameter_count")
    assert reporting.choose_screen_candidates(runs, results)["gvp_four"][0]["parameters"]["capacity"] == "compact"


def test_mandatory_large_late_profiles_must_both_repeat_before_freeze():
    runs, results = discovery()
    add_repeats(runs, results)
    for arm in ("late_four", "late_six"):
        for seed in (42, 43):
            row = run(arm, capacity="large", lr=3e-5, seed=seed, block="large")
            runs.append(row)
            results.append(result(row, native=.8))
    with pytest.raises(ValueError, match="mandatory"):
        reporting.freeze_candidates(runs, results[:-1])
    chosen = reporting.freeze_candidates(runs, results)
    assert chosen["late_four"]["parameters"]["capacity"] == "large"
    assert chosen["late_six"]["parameters"]["capacity"] == "large"


def test_fixed_five_reference_keeps_both_class_viii_meanings(tmp_path):
    manifest, results = confirmation()
    manifest["candidates"]["late_five"] = run("late_five")
    for fold in range(5):
        for seed in (42, 43):
            row = run("late_five", fold=fold, seed=seed, stage="confirmation", block="five")
            manifest["runs"].append(row)
            results.append(result(row, native=.5, common=.7))
    report = reporting.write_report(tmp_path, {"runs": []}, results, manifest)
    five = next(row for row in report["confirmation"]["arms"] if row["arm"] == "late_five")
    assert five["native_CoNi_recall"] == .5
    assert five["mean_collapsed4_per_class_recall"]["Class VIII"] == .7
    assert report["fixed_five_challenger"]["status"] == "shared_fold_complete"
    pairs = {row["contrast"]: row for row in report["confirmation"]["pairwise"]}
    assert pairs["fixed_five_vs_six"]["status"] == "complete"
    assert pairs["fixed_five_vs_six"]["mean_difference"] == pytest.approx(.1)
    assert "Co+Ni; common-four Class VIII always means Fe+Co+Ni" in (tmp_path / "campaign_report.md").read_text()


@pytest.mark.parametrize("certificate", [None, {"verified": True, "folds": {}},
    {"verified": True, "folds": {str(fold): {"status": "passed", "cohort_sha256": "wrong"} for fold in range(5)}}])
def test_ring_cannot_claim_improvement_without_matching_five_fold_certificate(certificate):
    manifest, results = confirmation()
    manifest["planned_blocks"].append("ring")
    if certificate is not None:
        manifest["ring_input_audit"] = certificate
    for fold in range(5):
        for seed in (42, 43):
            row = run("gvp_ring_on", fold=fold, seed=seed, stage="confirmation", block="ring")
            manifest["runs"].append(row)
            results.append(result(row, native=.8))
    report = reporting.build_report({"runs": []}, results, manifest)
    pair = report["confirmation"]["pairwise"][-1]
    assert pair["status"] in ("incomplete", "invalid")
    assert not pair["paired_improvement"] and not report["confirmation"]["complete"]


def test_verified_ring_comparison_allows_intended_edge_normalization_changes():
    manifest, results = confirmation()
    manifest["planned_blocks"].append("ring")
    manifest["ring_input_audit"] = {"verified": True, "folds": {
        str(fold): {"status": "passed", "cohort_sha256": f"cohort-{fold}"} for fold in range(5)}}
    for fold in range(5):
        for seed in (42, 43):
            row = run("gvp_ring_on", fold=fold, seed=seed, stage="confirmation", block="ring")
            manifest["runs"].append(row)
            results.append({**result(row, native=.8), "normalization_sha256": "different-intended-edge-statistics"})
    report = reporting.build_report({"runs": []}, results, manifest)
    pair = report["confirmation"]["pairwise"][-1]
    assert pair["status"] == "complete" and pair["paired_improvement"]
    assert not pair["promoted"]


@pytest.mark.parametrize("state", ["unresolved", "deferred", "incomplete"])
def test_five_contender_and_historical_reference_never_disappear(tmp_path, state):
    if state == "unresolved":
        manifest, results = None, []
    else:
        manifest, results = confirmation()
        manifest["deferred_blocks"] = ["five", "ring"] if state == "deferred" else ["ring"]
        if state == "incomplete":
            manifest["planned_blocks"].append("five")
    report = reporting.write_report(tmp_path, {"runs": []}, results, manifest)
    contender = report["fixed_five_challenger"]
    assert contender["status"] == state and not contender["shared_fold_complete"]
    assert contender["historical_reference"]["common_four_balanced_accuracy_percent"] == 74.718
    assert not contender["historical_reference"]["current_shared_fold_evidence"]
    assert "cannot be claimed to beat all target formulations" in contender["claim_limitation"]
    markdown = (tmp_path / "campaign_report.md").read_text()
    assert f"Fixed late-five contender: {state}" in markdown
    assert "74.718 ± 2.486%" in markdown and "historical evidence" in markdown
    if state == "deferred":
        assert "Deferred confirmation blocks: five, ring." in markdown


def test_complete_five_arm_without_complete_six_pair_remains_incomplete():
    manifest, results = confirmation()
    manifest["planned_blocks"].append("five")
    for fold in range(5):
        for seed in (42, 43):
            row = run("late_five", fold=fold, seed=seed, stage="confirmation", block="five")
            manifest["runs"].append(row)
            results.append(result(row))
    results = [row for row in results if not (row["arm"] == "late_six" and row["fold_index"] == 0 and row["seed"] == 42)]
    report = reporting.build_report({"runs": []}, results, manifest)
    assert report["fixed_five_challenger"]["completed_runs"] == 10
    assert report["fixed_five_challenger"]["status"] == "incomplete"
    pairs = {row["contrast"]: row for row in report["confirmation"]["pairwise"]}
    assert pairs["fixed_five_vs_four"]["status"] == "complete"
    assert pairs["fixed_five_vs_six"]["status"] == "incomplete"


def test_markdown_lists_epoch50_flags_without_claiming_missing_diagnostics_are_negative(tmp_path):
    manifest, results = confirmation()
    results[0]["potentially_training_budget_limited"] = True
    results[0]["late_validation_mean_gain"] = .003
    results[1]["potentially_training_budget_limited"] = False
    report = reporting.write_report(tmp_path, {"runs": []}, results, manifest)
    diagnostic = report["epoch50_diagnostics"]
    assert diagnostic["evaluated_runs"] == 2 and diagnostic["potentially_training_budget_limited_count"] == 1
    assert diagnostic["run_ids"] == [results[0]["id"]]
    markdown = (tmp_path / "campaign_report.md").read_text()
    assert "runs: 1 of 2 with recorded diagnostics" in markdown
    assert results[0]["id"] in markdown
    assert "do not extend training or reselect checkpoints" in markdown
