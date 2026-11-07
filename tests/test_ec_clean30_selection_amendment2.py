"""Amendment-2 selection rules: primary score, harm guard, best-inner S1, L9 v2 and the S4 v2 specification.

Synthetic pooled predictions only; no dataset access.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ec_clean30 import selection, stats  # noqa: E402
from ec_clean30.logit_adjustment import protein_class_priors  # noqa: E402
from ec_clean30.selection import (  # noqa: E402
    BEST_INNER_S1_RULE, LEVER_FAILED, LEVER_PASSED, PooledOof, PromotionConfig, harm_guard_v2, promote_primary,
    s1_best_inner, s1_choose_learning_rate_primary, s1_learning_rate_labels, s4_refit_specification_v2,
    screen_logit_adjustment_primary, seed_mean_ba_points, seed_mean_primary_points, selection_policy_record_v2,
    selection_policy_sha256_v2, sha256_canonical,
)
from ec_clean30.stats import BootstrapAbort  # noqa: E402

# EC classes 1-6 hold 200 proteins, so one protein right in both seeds is exactly 0.5 points.
SIZES = (50, 40, 40, 30, 20, 20, 5)
IDS = tuple(f"c{cls}_p{k:03d}" for cls, n in enumerate(SIZES) for k in range(n))
TRUE = np.repeat(np.arange(7), SIZES)
SINGLETONS = {pid: pid for pid in IDS}
FRACTIONS = (0.7, 0.7, 0.7, 0.7, 0.3, 0.7, 0.4)
FAST_L9 = selection.promotion_config_for("L9", PromotionConfig(n_draws=2000))
SHA = "ab" * 32


def _wrong(true):
    return (np.asarray(true) + 1) % 6


def _base():
    predicted = _wrong(TRUE)
    for cls, n in enumerate(SIZES):
        rows = np.flatnonzero(TRUE == cls)
        predicted[rows[: int(round(FRACTIONS[cls] * n))]] = cls
    return predicted


BASE = _base()


def _changed(predicted, gains=None, zero=()):
    out = predicted.copy()
    for cls, count in (gains or {}).items():
        rows = np.flatnonzero((TRUE == cls) & (out != TRUE))[:count]
        assert rows.size == count
        out[rows] = cls
    for cls in zero:
        rows = np.flatnonzero(TRUE == cls)
        out[rows] = _wrong(TRUE)[rows]
    return out


def _oof(name, predicted, other=None, *, ids=IDS, true=TRUE):
    rows = [predicted, predicted if other is None else other]
    return PooledOof(name, ids, true, (42, 43), np.eye(7)[np.stack(rows)])


def _family_set(**gains):
    """F2 at the base; ``gains`` maps a family to extra class-1 proteins right in both seeds (0.5 points each)."""
    return {family: _oof(family, _changed(BASE, {0: gains.get(family, 0)})) for family in selection.S1_V2_CANDIDATES}


# ---------------------------------------------------------------------------
# Primary score (A2-S2)


def test_primary_score_is_accuracy_over_ec_classes_1_6():
    pooled = _oof("base", BASE)
    assert seed_mean_primary_points(pooled) == pytest.approx(100 * 132 / 200)
    # Class-7 predictions and proteins: adding class-7 proteins leaves it unchanged.
    extra_ids = IDS + tuple(f"c6_extra{k}" for k in range(30))
    extra_true = np.concatenate([TRUE, np.full(30, 6)])
    extra = _oof("extra", np.concatenate([BASE, np.zeros(30, dtype=np.int64)]), ids=extra_ids, true=extra_true)
    assert seed_mean_primary_points(extra) == pytest.approx(seed_mean_primary_points(pooled))
    # A class-7 prediction for a right EC class 1 protein is wrong.
    seventh = BASE.copy()
    seventh[np.flatnonzero((TRUE == 0) & (BASE == 0))[0]] = 6
    assert seed_mean_primary_points(_oof("seventh", seventh, BASE)) == pytest.approx(100 * 132 / 200 - 0.25)


def test_primary_score_is_the_seed_mean_and_matches_the_claim_metric():
    other = _changed(BASE, {1: 4})
    pooled = _oof("two", BASE, other)
    by_seed = [100 * np.mean(predicted[TRUE < 6] == TRUE[TRUE < 6]) for predicted in (BASE, other)]
    assert seed_mean_primary_points(pooled) == pytest.approx(np.mean(by_seed))
    summary = selection._summarise_primary(pooled)
    assert summary.points_by_seed == pytest.approx(tuple(by_seed))
    assert summary.ba_points == pytest.approx(seed_mean_ba_points(pooled))
    # The claim engine's accuracy on the class-7-free rows is the same number.
    keep = TRUE < 6
    table = stats.build_cluster_count_table([pid for pid, k in zip(IDS, keep) if k], TRUE[keep],
                                            {"a": BASE[keep]}, SINGLETONS)
    point = stats.confusion_metric(table.counts.sum(axis=0)[0, 0], stats.METRIC_ACCURACY, range(6))
    assert 100 * point == pytest.approx(by_seed[0])
    if hasattr(stats, "METRIC_EC_CLASSES_1_6_ACCURACY"):  # the amended claim engine, once present
        full = stats.build_cluster_count_table(IDS, TRUE, {"a": BASE}, SINGLETONS)
        value = stats.confusion_metric(full.counts.sum(axis=0)[0, 0], stats.METRIC_EC_CLASSES_1_6_ACCURACY, range(6))
        assert 100 * value == pytest.approx(by_seed[0])


def test_primary_score_refuses_a_population_without_ec_classes_1_6():
    ids = ("a", "b")
    with pytest.raises(ValueError, match="no protein of EC classes 1-6"):
        seed_mean_primary_points(_oof("seven", np.array([6, 6]), ids=ids, true=np.array([6, 6])))


# ---------------------------------------------------------------------------
# Learning rate (A2-S3)


def _scored_settings(monkeypatch, family, first, second):
    labels = s1_learning_rate_labels(family, selection.S1_LEARNING_RATE_GRID_V2)
    settings = {labels[0]: _oof("first", BASE), labels[1]: _oof("second", _changed(BASE, {0: 1}))}
    scores = {"first": first, "second": second}
    monkeypatch.setattr(selection, "seed_mean_primary_points", lambda pooled: scores[pooled.name])
    return labels, settings


def test_learning_rate_band_is_inclusive_at_half_a_point(monkeypatch):
    labels, settings = _scored_settings(monkeypatch, "F6", 70.0, 70.5)
    choice = s1_choose_learning_rate_primary("F6", settings)
    assert choice.chosen == labels[0] and choice.best == labels[1]
    record = choice.to_record()
    assert record["metric"] == selection.PRIMARY_METRIC_V2 and record["chosen_setting"] == {"learning_rate": 1e-4}
    labels, settings = _scored_settings(monkeypatch, "F3", 70.0, 70.5 + 1e-6)
    assert s1_choose_learning_rate_primary("F3", settings).chosen == labels[1]


def test_learning_rate_grid_v2_adds_f6_and_keeps_freeze1():
    assert selection.S1_LEARNING_RATE_GRID_V2["F6"] == selection.S1_LEARNING_RATE_GRID["F5"]
    assert "F6" not in selection.S1_LEARNING_RATE_GRID and "C-GF" not in selection.S1_LEARNING_RATE_GRID_V2
    with pytest.raises(ValueError, match="no learning-rate grid"):
        s1_choose_learning_rate_primary("C-GF", {})
    with pytest.raises(ValueError, match="Unknown family"):
        s1_choose_learning_rate_primary("F9", {})
    with pytest.raises(ValueError, match="Unknown family"):
        s1_learning_rate_labels("F6")  # the Freeze-1 grid has no F6


def test_learning_rate_v1_record_has_no_metric_key():
    labels = s1_learning_rate_labels("F2")
    choice = selection.s1_choose_learning_rate("F2", {labels[0]: _oof("a", BASE), labels[1]: _oof("b", BASE)})
    assert "metric" not in choice.to_record()


# ---------------------------------------------------------------------------
# Harm guard (A2-S4)


def _class_hits(cls, hits_42, hits_43, base=BASE):
    """``base`` with exactly ``hits`` right proteins of class ``cls`` in each seed."""
    out = []
    for hits in (hits_42, hits_43):
        predicted = base.copy()
        rows = np.flatnonzero(TRUE == cls)
        predicted[rows] = _wrong(TRUE)[rows]
        predicted[rows[:hits]] = cls
        out.append(predicted)
    return out


def test_harm_guard_vetoes_total_collapse_against_one_anchor_hit():
    # Rule A2-SEL-4, choice C1: X is ineligible when k_X(c) = 0 and k_F2(c) >= 1 (summed over both seeds).
    candidate = _oof("X", *_class_hits(4, 0, 0))
    rows, eligible = harm_guard_v2(_oof("F2", *_class_hits(4, 1, 1)), candidate)
    assert not eligible
    row = next(item for item in rows if item["ec_class"] == 5)
    assert row == {"ec_class": 5, "pooled_proteins": 20, "anchor_correct_both_seeds": 2,
                   "candidate_correct_both_seeds": 0, "vetoed": True}
    assert not harm_guard_v2(_oof("F2", *_class_hits(4, 1, 0)), candidate)[1]  # a single anchor hit guards
    assert harm_guard_v2(_oof("F2", *_class_hits(4, 0, 0)), candidate)[1]  # no anchor hit: no veto
    assert harm_guard_v2(_oof("F2", *_class_hits(4, 6, 6)), _oof("X", *_class_hits(4, 0, 1)))[1]


def test_harm_guard_exempts_class_7_and_skips_absent_classes():
    no_seventh = BASE.copy()
    no_seventh[TRUE == 6] = 0
    assert harm_guard_v2(_oof("F2", BASE), _oof("X", no_seventh))[1]
    keep = TRUE != 5
    ids = tuple(pid for pid, k in zip(IDS, keep) if k)
    rows, eligible = harm_guard_v2(_oof("F2", BASE[keep], ids=ids, true=TRUE[keep]),
                                   _oof("X", BASE[keep], ids=ids, true=TRUE[keep]))
    assert eligible and [row["ec_class"] for row in rows] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Family choice (A2-S5, A2-S6)


def test_highest_eligible_score_wins_and_a_vetoed_maximum_is_skipped():
    winners = _family_set(F3=4, F4=6, F5=2, F6=3)
    screen = s1_best_inner(winners)
    assert screen.winner_family == "F4" and screen.maximum_points == pytest.approx(66 + 3.0)
    # F4 gets EC class 5 wrong in both seeds (F2: 12 right) and is the highest score: it is skipped.
    winners["F4"] = _oof("F4", _changed(BASE, {0: 14}, zero=(4,)))
    screen = s1_best_inner(winners)
    assert screen.candidate_scores_points["F4"] == pytest.approx(70.0)
    assert not screen.candidates["F4"]["eligible"] and "F4" not in screen.eligible_families
    assert screen.winner_family == "F3" and screen.maximum_points == pytest.approx(66 + 2.0)
    assert screen.tied_families == ("F3", "F6")


@pytest.mark.parametrize("gains, winner, tied", [
    ({"F5": 1}, "F2", ("F2", "F5", "F3", "F4", "F6")),  # F2 exactly 0.5 below the maximum: inclusive
    ({"F5": 2}, "F5", ("F5",)),                 # F2 1.0 below: out of the band
    ({"F3": 3, "F5": 2, "F4": 3}, "F5", ("F5", "F3", "F4")),
    ({"F3": 3, "F5": 3, "F6": 3}, "F5", ("F5", "F3", "F6")),   # equal scores resolve by the order
    ({"F6": 4, "F4": 3}, "F4", ("F4", "F6")),
])
def test_simplicity_order_decides_within_the_band(gains, winner, tied):
    screen = s1_best_inner(_family_set(**gains))
    assert screen.winner_family == winner and screen.tied_families == tied


def test_controls_are_never_selected_and_every_candidate_is_required():
    winners = _family_set(F3=1)
    strong = _oof("control", _changed(BASE, {0: 15}))
    screen = s1_best_inner(winners, controls={"F1": strong, "C-G": strong, "C-GF": strong})
    assert screen.winner_family == "F2"
    assert set(screen.controls) == {"F1", "C-G", "C-GF"}
    assert screen.control_scores_points["C-GF"] == pytest.approx(66 + 7.5)
    assert all(item["selectable"] is False for item in screen.controls.values())
    with pytest.raises(ValueError, match="cannot be selected"):
        s1_best_inner({**winners, "C-G": strong})
    with pytest.raises(ValueError, match="missing \\['F6'\\]"):
        s1_best_inner({family: pooled for family, pooled in winners.items() if family != "F6"})
    with pytest.raises(ValueError, match="Unknown S1 controls"):
        s1_best_inner(winners, controls={"F5": strong})
    unpaired = PooledOof("other", IDS[:-1], TRUE[:-1], (42, 43), np.eye(7)[np.stack([BASE[:-1]] * 2)])
    with pytest.raises(ValueError, match="cover different proteins"):
        s1_best_inner(winners, controls={"C-GF": unpaired})


def test_anchor_is_never_vetoed_and_recall_guard_rows_never_decide():
    winners = _family_set(F5=2)
    winners["F2"] = _oof("F2", _changed(BASE, zero=(4,)))  # F2 itself gets EC class 5 wrong everywhere
    big_drop = _changed(BASE, {0: 15})
    big_drop[np.flatnonzero((TRUE == 1) & (BASE == 1))[:6]] = 2  # 15 recall points and 6 proteins of class 2
    winners["F3"] = _oof("F3", big_drop)
    screen = s1_best_inner(winners)
    assert screen.candidates["F2"]["eligible"] is True and "F2" in screen.eligible_families
    f3 = screen.candidates["F3"]
    assert f3["eligible"] is True and f3["descriptive_recall_guard"]["would_veto"] is True
    assert f3["descriptive_recall_guard"]["decides"] is False
    assert any(row["vetoed"] for row in f3["descriptive_recall_guard"]["rows"])
    assert screen.candidate_scores_points["F3"] == pytest.approx(66 + 7.5 - 3.0)
    assert screen.winner_family == "F3"


def test_best_inner_record_has_no_bootstrap_and_binds_the_policy():
    record = s1_best_inner(_family_set(F5=3)).to_record()
    assert record["rule_id"] == BEST_INNER_S1_RULE == "best_inner_harm_guarded_v1"
    assert record["policy_sha256"] == selection_policy_sha256_v2() == sha256_canonical(selection_policy_record_v2())
    assert record["bootstrap"] is None and record["significance_test"] is None
    assert record["metric"] == "ec_classes_1_6_accuracy" and record["simplicity_order"] == ["F2", "F5", "F3", "F4", "F6"]
    assert "bootstrap" not in json.dumps(record["candidates"])
    assert set(record["inputs"]["predictions_sha256"]) == set(selection.S1_V2_CANDIDATES)
    assert not any(pid in json.dumps(record) for pid in IDS)


def test_policy_record_is_fresh_and_names_the_rules():
    first = selection_policy_record_v2()
    first["s1"]["simplicity_order"].append("F9")
    record = selection_policy_record_v2()
    assert record["s1"]["simplicity_order"] == ["F2", "F5", "F3", "F4", "F6"]
    assert record["selection_pipeline"] == "amendment2_selection_v1"
    assert record["s1"]["rule_id"] == "best_inner_harm_guarded_v1"
    assert record["s1"]["harm_guard"]["min_anchor_correct"] == 1
    assert record["s1"]["significance_test"] is None and record["s3"] == {"run": False}
    assert record["s2"]["rule_id"] == "l9_tau_promote_primary_v1" and record["s2"]["adoption"]["alpha"] == 0.01
    assert record["s4"]["rule_id"] == "refit_spec_v2" and record["s4"]["epochs"] == 30
    futility = record["futility"]
    assert futility["rule_id"] == "futility_predictive_cluster_bootstrap_v1"
    assert (futility["replicates"], futility["continue_if_successes_at_least"], futility["seed"]) == (2000, 400, 20261104)
    assert futility["target_sizes"] == {"P1": 526, "P2": 226} and futility["nested_populations"] is True
    assert "step_c" in futility and "none toward continuing" in futility["override"]
    assert record["s4"]["fold0_requires_futility"] and record["s4"]["rbf_distances"] == {
        "F1": "raw", "F3": "raw", "F4": "raw"}
    assert json.loads(json.dumps(record)) == record  # plain JSON


# ---------------------------------------------------------------------------
# L9 v2 (A2-S8)


TRAINING_PRIORS = protein_class_priors(np.repeat(np.arange(7), (200, 200, 200, 100, 5, 100, 3)))


def _adjustable(easy, hard=0):
    """Class-5 proteins the S1 winner misses: ``easy`` flip at tau 0.5 and 1, ``hard`` only at 1."""
    logits = np.full((len(IDS), 7), -12.0)
    logits[np.arange(len(IDS)), BASE] = 12.0
    missed = np.flatnonzero((TRUE == 4) & (BASE != 4))
    assert missed.size >= easy + hard
    for position, row in enumerate(missed[: easy + hard]):
        logits[row] = -12.0
        logits[row, 2], logits[row, 4] = (1.0 if position < easy else 3.0), 0.5
    priors = np.broadcast_to(TRAINING_PRIORS, (2, len(IDS), 7))
    return PooledOof("s1", IDS, TRUE, (42, 43), np.stack([logits, logits]), row_priors=priors)


def _l9(pooled, config=FAST_L9):
    return screen_logit_adjustment_primary("F2", pooled, component_of=SINGLETONS, training_class_counts=SIZES,
                                           config=config)


def test_l9_v2_chooses_tau_on_the_primary_score_and_adopts_at_alpha_001():
    outcome = _l9(_adjustable(easy=4, hard=4))
    assert outcome.details["chosen_tau"] == 1.0 and outcome.status == LEVER_PASSED
    assert outcome.details["adopted_tau"] == 1.0 and outcome.details["rule_id"] == "l9_tau_promote_primary_v1"
    assert outcome.decision.gain_points == pytest.approx(8 * 0.5)
    assert outcome.decision.config["alpha"] == 0.01 and outcome.decision.metric == "ec_classes_1_6_accuracy"
    assert outcome.decision.bootstrap["n_rows"] == 200
    assert outcome.decision.bootstrap["metric"] == "ec_classes_1_6_accuracy"
    assert outcome.decision.bootstrap["engine"] == "stratified_rao_wu_v2"
    scores = outcome.details["tau_scores_points"]
    assert scores["+0.0"] == pytest.approx(66.0) and scores["+0.5"] == pytest.approx(68.0)
    assert outcome.to_record()["decision"]["bootstrap_abort"] is None


def test_l9_v2_ties_go_to_the_smaller_tau_and_small_gains_are_not_adopted():
    tied = _l9(_adjustable(easy=6))
    assert tied.details["chosen_tau"] == 0.5 and tied.status == LEVER_PASSED
    small = _l9(_adjustable(easy=1))
    assert small.details["chosen_tau"] == 0.5 and small.status == LEVER_FAILED
    assert small.details["adopted_tau"] == 0.0 and "gain" in small.decision.vetoes
    with pytest.raises(ValueError, match="training priors"):
        _l9(_oof("no_priors", BASE))
    with pytest.raises(ValueError, match="L9 v2 applies"):
        screen_logit_adjustment_primary("C-G", _adjustable(easy=4), component_of=SINGLETONS,
                                        training_class_counts=SIZES)


def test_l9_v2_bootstrap_abort_means_tau_zero(monkeypatch):
    def abort(*args, **kwargs):
        raise BootstrapAbort("forced")

    monkeypatch.setattr(selection, "_primary_bootstrap", abort)
    outcome = _l9(_adjustable(easy=4, hard=4))
    assert outcome.status == LEVER_FAILED and outcome.details["bootstrap_abort"] is True
    assert outcome.details["adopted_tau"] == 0.0 and outcome.decision.bootstrap_abort == "forced"
    assert outcome.decision.checks["lower_bound"] is None and "lower_bound" in outcome.decision.vetoes


def _rare_class_6_pair():
    """EC class 6 in three inner components only, so many unstratified draws hold none of them."""
    component_of = {pid: pid for pid in IDS}
    class_6 = [pid for pid in IDS if pid.startswith("c5_")]
    for index, pid in enumerate(class_6):
        component_of[pid] = f"class6_component{index % 3}"
    return component_of, _oof("inc", BASE), _oof("ch", _changed(BASE, {0: 10, 1: 6}))


def test_l9_adoption_bootstrap_is_engine_v2_on_the_presence_free_primary_metric():
    component_of, incumbent, challenger = _rare_class_6_pair()
    config = selection.promotion_config_for("L9", PromotionConfig(n_draws=2000))
    result = selection._primary_bootstrap(incumbent, challenger, component_of, config)
    record = result.to_record()
    assert result.metric == stats.METRIC_EC_CLASSES_1_6_ACCURACY and result.engine == stats.ENGINE_AMENDMENT2
    assert result.n_skipped == 0 and result.n_valid == 2000 and result.stratified is False
    assert record["engine"] == "stratified_rao_wu_v2" and record["resampling"] == "rao_wu_k_minus_1_of_k"
    assert "class_absence_probability" in record and "multiplicities_dyadic" in record
    # The same table under the historical v1 accuracy skips draws without an EC class 1-6 present in the pool.
    keep = np.isin(TRUE, range(6))
    arms = {f"{role}/seed_{seed}": pooled.predictions[position][keep]
            for role, pooled in (("incumbent", incumbent), ("challenger", challenger))
            for position, seed in enumerate(pooled.seeds)}
    table = stats.build_cluster_count_table([pid for pid, k in zip(IDS, keep) if k], TRUE[keep], arms, component_of)
    with pytest.raises(BootstrapAbort, match="lacked an EC class 1-6"):
        stats.paired_cluster_bootstrap(
            table, [stats.Contrast.difference("gain", "challenger", "incumbent")],
            systems={"incumbent": [k for k in arms if k.startswith("inc")],
                     "challenger": [k for k in arms if k.startswith("ch")]},
            metric=stats.METRIC_ACCURACY, n_draws=2000, stratified=False)
    decision = promote_primary(incumbent, challenger, component_of=component_of, training_class_counts=SIZES,
                               config=config)
    assert decision.bootstrap_abort is None and decision.bootstrap["engine"] == "stratified_rao_wu_v2"
    assert decision.bootstrap_population == selection.PRIMARY_BOOTSTRAP_POPULATION
    assert "presence-free" in decision.bootstrap_population and "skipped" in decision.bootstrap_population
    adoption = selection_policy_record_v2()["s2"]["adoption"]
    assert adoption["engine"] == "stratified_rao_wu_v2" and adoption["stratified"] is False
    assert adoption["metric"] == "ec_classes_1_6_accuracy"


def test_the_historical_promote_keeps_engine_v1():
    component_of, incumbent, challenger = _rare_class_6_pair()
    config = PromotionConfig(n_draws=500)
    decision = selection.promote(incumbent, challenger, component_of={pid: pid for pid in IDS},
                                 training_class_counts=SIZES, config=config)
    assert "engine" not in (decision.bootstrap or {})
    assert (decision.bootstrap or {}).get("metric") == stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY


def test_promote_primary_keeps_the_historical_guards():
    incumbent = _oof("inc", BASE)
    loses_class_5 = _oof("ch", _changed(BASE, {0: 15}, zero=(4,)))
    decision = promote_primary(incumbent, loses_class_5, component_of=SINGLETONS, training_class_counts=SIZES,
                               config=FAST_L9)
    assert decision.gain_points > 1.0 and not decision.promoted
    assert 5 in decision.zero_recall_ec_classes and decision.checks["zero_recall"] is False


# ---------------------------------------------------------------------------
# S4 v2 (A2-S9)


def _spec(**changes):
    options = {"family": "F3", "learning_rate_setting": {"gvp_learning_rate": 3e-4, "learning_rate": 1e-4},
               "adopted_levers": [], "s1_record_sha256": SHA, "s2_record_sha256": "cd" * 32}
    options.update(changes)
    return s4_refit_specification_v2(**options)


def test_s4_v2_specification_binds_the_records_and_the_defaults():
    spec = _spec()
    assert spec["rule_id"] == "refit_spec_v2" and spec["epochs"] == 30 and spec["seeds"] == [42, 43, 44, 45, 46]
    assert spec["rbf_distances"] == "raw"
    assert spec["whole_protein_pretraining"] is None and spec["logit_adjustment_tau"] == 0.0
    assert (spec["s1_record_sha256"], spec["s2_record_sha256"]) == (SHA, "cd" * 32)
    assert spec["policy_sha256"] == selection_policy_sha256_v2()
    f6 = _spec(family="F6", learning_rate_setting={"learning_rate": 3e-4}, adopted_levers=["L9"], tau=-0.5)
    assert f6["rbf_distances"] == "normalized" and f6["logit_adjustment_tau"] == -0.5 and f6["adopted_levers"] == ["L9"]
    assert f6["whole_protein_pretraining"]["pretraining_seeds"] == [42, 43, 44, 45, 46]
    assert "inside each refit" in f6["whole_protein_pretraining"]["location"]
    assert _spec(family="F4", learning_rate_setting={"gvp_learning_rate": 1e-4, "learning_rate": 1e-4}
                 )["rbf_distances"] == "raw"
    assert _spec(family="F2", learning_rate_setting={"learning_rate": 1e-4})["rbf_distances"] == "normalized"
    # Rule A2-SEL-9: the fold-0 record needs a verified FUT = continue; other folds bind no futility record.
    with pytest.raises(ValueError, match="verified FUT = continue"):
        _spec(fold=0)
    with pytest.raises(ValueError, match="verified FUT = continue"):
        _spec(fold=0, futility_decision="stop_futility", futility_record_sha256="ef" * 32)
    fold0 = _spec(fold=0, futility_decision="continue", futility_record_sha256="ef" * 32)
    assert (fold0["fold"], fold0["futility_record_sha256"]) == (0, "ef" * 32)
    with pytest.raises(ValueError, match="Only the fold-0"):
        _spec(fold=1, futility_decision="continue", futility_record_sha256="ef" * 32)
    assert _spec(fold=2)["futility_record_sha256"] is None


@pytest.mark.parametrize("changes, message", [
    ({"epochs": 20}, "fixes 30 epochs"),
    ({"epochs": True}, "fixes 30 epochs"),
    ({"adopted_levers": ["L4"]}, "adopts only"),
    ({"adopted_levers": ["L9", "L9"], "tau": 0.5}, "adopts only"),
    ({"family": "C-G", "learning_rate_setting": {"learning_rate": 1e-4}}, "control"),
    ({"family": "F1", "learning_rate_setting": {"learning_rate": 1e-4}}, "control"),
    ({"family": "C-GF"}, "control"),
    ({"adopted_levers": ["L9"]}, "does not match"),
    ({"tau": 0.5}, "does not match"),
    ({"learning_rate_setting": {"learning_rate": 1e-4}}, "not in the S1 grid"),
    ({"seeds": (42, 43, 44)}, "seeds"),
    ({"s1_record_sha256": "x"}, "SHA-256"),
])
def test_s4_v2_specification_refusals(changes, message):
    with pytest.raises(ValueError, match=message):
        _spec(**changes)
