"""Inner-fold promotion rule, selection steps, step records and null simulator (synthetic; no dataset access)."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ec_clean30.logit_adjustment import protein_class_priors
from ec_clean30.selection import (
    GENESIS_SHA256, GUARD_MIN_TRAINING_PROTEINS, GUARD_NET_PROTEINS, GUARD_RECALL_DROP_POINTS, LEVER_FAILED,
    LEVER_NOT_APPLICABLE, LEVER_NOT_TESTED, LEVER_PASSED, LR_TIE_BAND_POINTS, MIN_GAIN_POINTS, N_INNER_FOLDS,
    L9_PROMOTION_ALPHA, PROMOTION_ALPHA, REFIT_SEEDS, SCREENING_SEEDS, SUPCON_MIN_POSITIVE_STEP_FRACTION,
    InnerFoldPredictions, LeverOutcome, NullNoiseModel, PooledOof, PromotionConfig, StepChain, append_step_record,
    apply_tau, canonical_json, lever_applies, lever_not_applicable, lever_not_tested, membership_noise_statistics,
    noise_model_from_membership, pool_inner_oof, promote, promotion_config_for,
    read_step_chain, s1_choose_learning_rate, s1_family_escalation, s1_learning_rate_labels, s2_lever_screen,
    s3_combination, s4_refit_specification, screen_epoch_lever, screen_lever, screen_logit_adjustment,
    seed_mean_ba_points, simulate_null_adoption, supcon_lever_testable, supcon_positive_step_fraction,
    verify_step_chain,
)
from ec_clean30.stats import DEFAULT_BOOTSTRAP_SEED, DEFAULT_MAX_SKIP_FRACTION, DEFAULT_N_DRAWS, BootstrapAbort


# Pooled inner out-of-fold fixture: proteins per EC class 1..7. Class 4 (index 3) has 40 proteins
# so 2 and 3 lost proteins are 5.0 and 7.5 recall points; class 1 (index 0) has 101.
SIZES = (101, 100, 100, 40, 100, 100, 3)
IDS = tuple(f"c{cls}_p{k:03d}" for cls, n in enumerate(SIZES) for k in range(n))
TRUE = np.repeat(np.arange(7), SIZES)
SINGLETONS = {pid: pid for pid in IDS}
FAST = PromotionConfig(n_draws=2000)


def _wrong(true):
    return (np.asarray(true) + 1) % 6


def _base():
    predicted = _wrong(TRUE)
    for cls, n in enumerate(SIZES):
        rows = np.flatnonzero(TRUE == cls)
        predicted[rows[: int(0.7 * n)]] = cls
    return predicted


BASE = _base()


def _changed(predicted, gains=None, losses=None):
    out = predicted.copy()
    for cls, count in (gains or {}).items():
        rows = np.flatnonzero((TRUE == cls) & (out != TRUE))[:count]
        assert rows.size == count
        out[rows] = cls
    for cls, count in (losses or {}).items():
        rows = np.flatnonzero((TRUE == cls) & (out == TRUE))[:count]
        assert rows.size == count
        out[rows] = _wrong(TRUE)[rows]
    return out


def _oof(name, predicted, seeds=(42, 43), priors=None, per_seed=None):
    rows = per_seed if per_seed is not None else [predicted] * len(seeds)
    logits = np.eye(7)[np.stack(rows)]
    return PooledOof(name, IDS, TRUE, seeds, logits, row_priors=priors)


def _promote(incumbent, challenger, counts=SIZES, config=FAST, components=SINGLETONS, **kwargs):
    return promote(incumbent, challenger, component_of=components, training_class_counts=counts, config=config, **kwargs)


INCUMBENT = _oof("incumbent", BASE)

FROZEN_PROMOTION_CONFIG = {
    "min_gain_points": 1.0, "alpha": 0.04, "n_draws": 10_000, "bootstrap_seed": 20261023, "stratified": False,
    "max_skip_fraction": 0.01, "guard_min_training_proteins": 30, "guard_recall_drop_points": 5.0,
    "guard_net_proteins": 3.0, "zero_recall_unit": "seed_mean",
}
FROZEN_L9_PROMOTION_CONFIG = {**FROZEN_PROMOTION_CONFIG, "alpha": 0.01}


def test_frozen_promotion_settings():
    # D1 (P2-9a, confirmed on fresh draws): alpha 0.04 for every promotion including L7; L9 alone 0.01.
    assert (MIN_GAIN_POINTS, PROMOTION_ALPHA, L9_PROMOTION_ALPHA, LR_TIE_BAND_POINTS) == (1.0, 0.04, 0.01, 0.5)
    assert (GUARD_MIN_TRAINING_PROTEINS, GUARD_RECALL_DROP_POINTS, GUARD_NET_PROTEINS) == (30, 5.0, 3.0)
    assert (DEFAULT_N_DRAWS, DEFAULT_MAX_SKIP_FRACTION, DEFAULT_BOOTSTRAP_SEED) == (10_000, 0.01, 20261023)
    assert (SCREENING_SEEDS, REFIT_SEEDS, N_INNER_FOLDS) == ((42, 43), (42, 43, 44, 45, 46), 4)
    assert SUPCON_MIN_POSITIVE_STEP_FRACTION == 0.90
    assert PromotionConfig().to_record() == FROZEN_PROMOTION_CONFIG
    for function in (promote, s1_family_escalation, screen_lever, screen_epoch_lever, s3_combination,
                     simulate_null_adoption):
        assert inspect.signature(function).parameters["config"].default.to_record() == FROZEN_PROMOTION_CONFIG
    assert inspect.signature(screen_logit_adjustment).parameters["config"].default.to_record() == \
        FROZEN_L9_PROMOTION_CONFIG


def test_promotion_level_per_lever():
    assert promotion_config_for(None).to_record() == FROZEN_PROMOTION_CONFIG
    for lever in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"):
        assert promotion_config_for(lever).alpha == 0.04
    assert promotion_config_for("L9").to_record() == FROZEN_L9_PROMOTION_CONFIG
    # Only alpha changes: the other Freeze-1 choices of a base config carry over.
    base = PromotionConfig(n_draws=2000, stratified=True, alpha=0.2)
    assert promotion_config_for("L9", base) == PromotionConfig(n_draws=2000, stratified=True, alpha=0.01)
    assert promotion_config_for("L7", base).alpha == 0.04 and promotion_config_for(None, base).alpha == 0.04
    with pytest.raises(ValueError, match="Unknown lever"):
        promotion_config_for("L10")
    assert s1_learning_rate_labels("F2") == ("learning_rate=0.0001", "learning_rate=0.0003")
    assert s1_learning_rate_labels("F4") == (
        "gvp_learning_rate=0.0001,learning_rate=0.0001", "gvp_learning_rate=0.0003,learning_rate=0.0001",
    )


def _churned(net_gain, churn):
    moved = {0: churn, 2: churn}
    return _oof("c", _changed(BASE, gains={**moved, 1: net_gain}, losses=moved))


@pytest.mark.parametrize(
    ("net_gain", "churn", "p_band", "promoted"),
    [
        (9, 4, (0.02, 0.04), True),    # passes at the frozen one-sided 96% level (alpha 0.04), would fail at 98%
        (8, 4, (0.04, 0.05), False),   # fails at 96%, would pass at 95%
    ],
)
def test_default_promotion_uses_a_one_sided_ninety_six_percent_bound(net_gain, churn, p_band, promoted):
    decision = promote(INCUMBENT, _churned(net_gain, churn), component_of=SINGLETONS, training_class_counts=SIZES)
    low, high = p_band
    assert low < decision.p_value <= high
    assert decision.gain_points >= 1.0
    assert decision.promoted is promoted
    assert decision.vetoes == (() if promoted else ("lower_bound",))
    assert decision.config == FROZEN_PROMOTION_CONFIG
    assert decision.bootstrap["n_draws"] == 10_000 and decision.bootstrap["seed"] == 20261023


def test_gain_below_one_point_is_rejected():
    decision = _promote(INCUMBENT, _oof("c", _changed(BASE, gains={0: 6})))
    assert decision.gain_points == pytest.approx(100 * 6 / 101 / 6)
    assert not decision.promoted
    assert decision.vetoes == ("gain",)
    assert decision.checks["lower_bound"] is True


def test_gain_of_one_point_with_positive_bound_is_promoted():
    decision = _promote(INCUMBENT, _oof("c", _changed(BASE, gains={1: 6})))
    assert decision.gain_points == pytest.approx(1.0)
    assert decision.lower_bound_points > 0
    assert decision.promoted and decision.vetoes == ()


def test_noisy_gain_with_nonpositive_bound_is_rejected():
    churn = {0: 20, 2: 20, 4: 20, 5: 20}
    challenger = _changed(BASE, gains={**churn, 1: 7}, losses=churn)
    decision = _promote(INCUMBENT, _oof("c", challenger))
    assert decision.gain_points >= 1.0
    assert decision.lower_bound_points <= 0
    assert decision.vetoes == ("lower_bound",)


@pytest.mark.parametrize(
    ("lost", "counts", "promoted"),
    [
        (2, SIZES, True),                          # 5.0 recall points but only 2 net proteins
        (3, SIZES, False),                         # 7.5 recall points and 3 net proteins
        (3, (101, 100, 100, 29, 100, 100, 3), True),  # class with 29 training proteins is exempt
    ],
)
def test_recall_guard(lost, counts, promoted):
    decision = _promote(INCUMBENT, _oof("c", _changed(BASE, gains={1: 30}, losses={3: lost})), counts=counts)
    assert decision.promoted is promoted
    guarded = {row["ec_class"]: row for row in decision.recall_guard}
    if counts[3] >= 30:
        assert guarded[4]["recall_drop_points"] == pytest.approx(100 * lost / 40)
        assert guarded[4]["net_proteins_lost"] == pytest.approx(lost)
        assert decision.vetoes == (() if promoted else ("recall_guard",))
    else:
        assert 4 not in guarded
    assert 7 not in guarded


def test_zero_recall_veto():
    challenger = _changed(BASE, gains={0: 31, 1: 30, 2: 30, 3: 12, 5: 30}, losses={4: 70})
    counts = (101, 100, 100, 40, 29, 100, 3)
    decision = _promote(INCUMBENT, _oof("c", challenger), counts=counts)
    assert decision.gain_points > 10 and decision.lower_bound_points > 0
    assert decision.zero_recall_ec_classes == (5,)
    assert decision.vetoes == ("zero_recall",)
    # A class already at zero recall for the incumbent does not "fall" to zero.
    weak = _oof("weak", _changed(BASE, losses={4: 70}))
    assert _promote(weak, _oof("c", challenger), counts=counts).zero_recall_ec_classes == ()


def test_zero_recall_unit_seed_mean_or_per_seed():
    lost_in_one_seed = _changed(BASE, gains={1: 30}, losses={4: 70})
    challenger = _oof("c", None, per_seed=[lost_in_one_seed, _changed(BASE, gains={1: 30})])
    counts = (101, 100, 100, 40, 29, 100, 3)
    assert _promote(INCUMBENT, challenger, counts=counts).zero_recall_ec_classes == ()
    per_seed = PromotionConfig(n_draws=2000, zero_recall_unit="per_seed")
    assert _promote(INCUMBENT, challenger, counts=counts, config=per_seed).zero_recall_ec_classes == (5,)


def test_promotion_is_deterministic_and_recorded():
    challenger = _oof("c", _changed(BASE, gains={1: 12}))
    first, second = _promote(INCUMBENT, challenger), _promote(INCUMBENT, challenger)
    assert canonical_json(first.to_record()) == canonical_json(second.to_record())
    record = first.to_record()
    assert record["bootstrap"]["seed"] == FAST.bootstrap_seed
    assert record["bootstrap"]["n_draws"] == 2000 and record["bootstrap"]["n_skipped"] == 0
    assert all(len(value) == 64 for value in record["inputs"].values())


def test_cheap_checks_first_gives_the_same_decision_without_a_bootstrap():
    challenger = _oof("c", _changed(BASE, gains={0: 6}))
    lazy = _promote(INCUMBENT, challenger, evaluate_all_checks=False)
    assert lazy.promoted is False and lazy.bootstrap is None and lazy.lower_bound_points is None
    assert lazy.vetoes == ("gain",)


def test_unstratified_component_bootstrap_aborts_on_frequent_skips():
    components = {pid: ("big" if pid.startswith("c5_") else pid) for pid in IDS}
    challenger = _oof("c", _changed(BASE, gains={1: 30}))
    with pytest.raises(BootstrapAbort):
        _promote(INCUMBENT, challenger, components=components)
    stratified = PromotionConfig(n_draws=2000, stratified=True)
    assert _promote(INCUMBENT, challenger, components=components, config=stratified).promoted


def test_unpaired_inputs_are_refused():
    with pytest.raises(ValueError, match="different seeds"):
        _promote(INCUMBENT, _oof("c", BASE, seeds=(42, 44)))
    with pytest.raises(ValueError, match="training_class_counts"):
        _promote(INCUMBENT, _oof("c", BASE), counts=(1, 2, 3))


@pytest.mark.parametrize("evaluate_all_checks", [True, False])
def test_missing_components_are_refused_before_any_check(evaluate_all_checks):
    partial = {pid: pid for pid in IDS[1:]}
    with pytest.raises(ValueError, match="1 pooled proteins have no inner identity component"):
        _promote(INCUMBENT, _oof("c", _changed(BASE, gains={0: 6})), components=partial,
                 evaluate_all_checks=evaluate_all_checks)


# Pooling of inner folds


def _parts(drop=None, extra=None, duplicate=None, move=None, priors=True):
    parts = []
    for seed in (42, 43):
        for fold in range(4):
            ids = [pid for k, pid in enumerate(IDS) if k % 4 == fold and pid != drop]
            if extra and fold == 0:
                ids.append(extra)
            if duplicate and fold == 1:
                ids.append(duplicate)
            if move and seed == 43:
                ids = [pid for pid in ids if pid != move] + ([move] if fold == 3 else [])
            logits = np.eye(7)[[BASE[IDS.index(pid)] if pid in IDS else 0 for pid in ids]]
            fit_priors = protein_class_priors([fold % 7, 0, 1]) if priors else None
            parts.append(InnerFoldPredictions(seed, fold, tuple(ids), logits, fit_priors))
    return parts[::-1]


TRUE_CLASS_OF = dict(zip(IDS, TRUE.tolist()))


def test_pooling_covers_every_protein_once_per_seed():
    pooled = pool_inner_oof("F2", _parts(), TRUE_CLASS_OF)
    assert pooled.protein_ids == tuple(sorted(IDS))
    assert pooled.seeds == (42, 43)
    order = [IDS.index(pid) for pid in pooled.protein_ids]
    assert np.array_equal(pooled.predictions[0], BASE[order])
    assert np.array_equal(pooled.inner_fold, np.array(order) % 4)
    assert np.allclose(pooled.row_priors[1, 0], protein_class_priors([order[0] % 4 % 7, 0, 1]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"drop": IDS[5]}, "lack a prediction"),
        ({"extra": "unknown_protein"}, "not eligible"),
        ({"duplicate": IDS[0]}, "more than once"),
        ({"move": IDS[0]}, "different inner folds"),
    ],
)
def test_pooling_refuses_incomplete_or_duplicated_coverage(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pool_inner_oof("F2", _parts(**kwargs), TRUE_CLASS_OF)


def test_pooling_refuses_wrong_seeds_folds_and_mixed_priors():
    parts = _parts()
    with pytest.raises(ValueError, match="differ from the expected"):
        pool_inner_oof("F2", [p for p in parts if p.seed == 42], TRUE_CLASS_OF)
    with pytest.raises(ValueError, match="expected 5 inner folds"):
        pool_inner_oof("F2", parts, TRUE_CLASS_OF, expected_inner_folds=5)
    with pytest.raises(ValueError, match="more than once"):
        pool_inner_oof("F2", parts + parts[:1], TRUE_CLASS_OF)
    mixed = _parts(priors=False)[:1] + parts[1:]
    with pytest.raises(ValueError, match="every inner fit or for none"):
        pool_inner_oof("F2", mixed, TRUE_CLASS_OF)


def test_pooled_predictions_validate_their_shape():
    with pytest.raises(ValueError, match="unique"):
        PooledOof("x", ("a", "a"), np.array([0, 1]), (42,), np.zeros((1, 2, 7)))
    with pytest.raises(ValueError, match="logits must be finite"):
        PooledOof("x", ("a", "b"), np.array([0, 1]), (42, 43), np.zeros((1, 2, 7)))


# S1


F2_LOW_LR, F2_HIGH_LR = s1_learning_rate_labels("F2")


def test_learning_rate_tie_band_prefers_the_first_listed_setting():
    first = _oof("run_a", BASE)
    within = _oof("run_b", _changed(BASE, gains={1: 3}))
    beyond = _oof("run_c", _changed(BASE, gains={1: 4}))
    # Mapping order is ignored; the grid's listed order decides the tie band.
    choice = s1_choose_learning_rate("F2", {F2_HIGH_LR: within, F2_LOW_LR: first})
    assert (choice.chosen, choice.best) == (F2_LOW_LR, F2_HIGH_LR)
    assert choice.chosen_setting == {"learning_rate": 1e-4}
    assert choice.winner is first
    record = json.loads(canonical_json(choice.to_record()))
    assert record["chosen_setting"] == {"learning_rate": 1e-4}
    assert [label for label, _ in record["scores_points"]] == [F2_LOW_LR, F2_HIGH_LR]
    assert [label for label, _ in record["setting_predictions_sha256"]] == [F2_LOW_LR, F2_HIGH_LR]
    beyond_choice = s1_choose_learning_rate("F2", {F2_LOW_LR: first, F2_HIGH_LR: beyond})
    assert beyond_choice.chosen == F2_HIGH_LR and beyond_choice.chosen_setting == {"learning_rate": 3e-4}


def test_learning_rate_settings_must_be_keyed_by_the_family_grid():
    first, second = _oof("run_a", BASE), _oof("run_b", BASE)
    with pytest.raises(ValueError, match="grid labels"):
        s1_choose_learning_rate("F2", {"lr=1e-4": first, "lr=3e-4": second})
    with pytest.raises(ValueError, match="grid labels"):
        s1_choose_learning_rate("F4", {F2_LOW_LR: first, F2_HIGH_LR: second})
    with pytest.raises(ValueError, match="grid labels"):
        s1_choose_learning_rate("F3", {s1_learning_rate_labels("F3")[0]: first})
    with pytest.raises(ValueError, match="Unknown family"):
        s1_choose_learning_rate("F6", {F2_LOW_LR: first, F2_HIGH_LR: second})


def test_family_escalation_takes_the_global_branch_then_f4_vs_f5():
    winners = {
        "F2": _oof("F2", BASE),
        "F5": _oof("F5", _changed(BASE, gains={1: 20})),
        "F4": _oof("F4", _changed(BASE, gains={1: 20, 2: 20})),
        "F3": _oof("F3", _changed(BASE, gains={0: 30, 1: 30, 2: 30})),
        "F1": _oof("F1", _changed(BASE, gains={0: 31, 1: 30, 2: 30, 4: 30, 5: 30})),
    }
    result = s1_family_escalation(winners, component_of=SINGLETONS, training_class_counts=SIZES, config=FAST)
    assert result.global_branch_adopted
    assert result.pocket_graph_comparison == ("F5", "F4")
    assert (result.pocket_graph_decision.incumbent, result.pocket_graph_decision.challenger) == ("F5", "F4")
    assert result.winner_family == "F4" and result.winner is winners["F4"]
    assert set(result.control_scores_points) == {"F1"}
    json.loads(canonical_json(result.to_record()))


def test_family_escalation_without_the_global_branch_compares_f3_with_f2():
    winners = {
        "F2": _oof("F2", BASE),
        "F5": _oof("F5", _changed(BASE, gains={1: 3})),
        "F4": _oof("F4", _changed(BASE, gains={1: 30, 2: 30})),
        "F3": _oof("F3", _changed(BASE, gains={2: 20})),
        "C-G": _oof("C-G", _changed(BASE, gains={0: 31, 1: 30, 2: 30})),
    }
    result = s1_family_escalation(winners, component_of=SINGLETONS, training_class_counts=SIZES, config=FAST)
    assert not result.global_branch_adopted
    assert result.pocket_graph_comparison == ("F2", "F3")
    assert result.winner_family == "F3"
    assert "C-G" in result.control_scores_points
    with pytest.raises(ValueError, match="missing"):
        s1_family_escalation({"F2": winners["F2"]}, component_of=SINGLETONS, training_class_counts=SIZES)


# S2


def _screen(**overrides):
    kwargs = {"component_of": SINGLETONS, "training_class_counts": SIZES, "config": FAST}
    kwargs.update(overrides)
    return kwargs


def test_epoch_lever_uses_the_better_look_against_thirty_epochs():
    looks = {12: _oof("e12", _changed(BASE, gains={1: 30})), 20: _oof("e20", _changed(BASE, gains={1: 12}))}
    outcome = screen_epoch_lever("F2", INCUMBENT, looks, **_screen())
    assert outcome.status == LEVER_PASSED
    assert outcome.details["challenger_epochs"] == 12
    assert (outcome.decision.incumbent, outcome.decision.challenger) == ("incumbent", "e12")
    swapped = {12: looks[20], 20: looks[12]}
    assert screen_epoch_lever("F2", INCUMBENT, swapped, **_screen()).details["challenger_epochs"] == 20
    tied = {12: _oof("e12", BASE), 20: _oof("e20", BASE)}
    tie = screen_epoch_lever("F2", INCUMBENT, tied, **_screen())
    assert tie.details["challenger_epochs"] == 12 and tie.status == LEVER_FAILED
    with pytest.raises(ValueError, match="exactly the looks"):
        screen_epoch_lever("F2", INCUMBENT, {12: looks[12]}, **_screen())
    # The look that is not chosen must still cover the same proteins.
    short = PooledOof("e20", IDS[:-1], TRUE[:-1], (42, 43), np.eye(7)[np.stack([_wrong(TRUE[:-1])] * 2)])
    with pytest.raises(ValueError, match="cover different proteins"):
        screen_epoch_lever("F2", INCUMBENT, {12: looks[12], 20: short}, **_screen())


TRAINING_PRIORS = protein_class_priors(np.repeat(np.arange(7), (200, 200, 200, 100, 5, 100, 3)))


def _adjustable(hard_rows):
    """Class 5 (index 4) proteins the incumbent misses: 10 flip at tau 0.5 and 1, ``hard_rows`` only at 1."""
    logits = np.full((len(IDS), 7), -12.0)
    logits[np.arange(len(IDS)), BASE] = 12.0
    missed = np.flatnonzero((TRUE == 4) & (BASE != 4))
    for position, row in enumerate(missed[: 10 + hard_rows]):
        logits[row] = -12.0
        logits[row, 2], logits[row, 4] = (1.0 if position < 10 else 3.0), 0.5
    priors = np.broadcast_to(TRAINING_PRIORS, (2, len(IDS), 7))
    return PooledOof("s1", IDS, TRUE, (42, 43), np.stack([logits, logits]), row_priors=priors)


def test_logit_adjustment_grid_picks_the_best_tau_and_promotes_against_zero():
    s1 = _adjustable(hard_rows=10)
    outcome = screen_logit_adjustment("F2", s1, **_screen())
    assert outcome.details["chosen_tau"] == 1.0
    assert outcome.status == LEVER_PASSED
    assert outcome.decision.gain_points == pytest.approx(100 * 20 / 100 / 6)
    scores = outcome.details["tau_scores_points"]
    assert scores["-1.0"] == scores["-0.5"] == scores["+0.0"] == pytest.approx(seed_mean_ba_points(s1))
    assert np.array_equal(apply_tau(s1, 0.0).logits, s1.logits)


def test_logit_adjustment_ties_go_to_the_smaller_tau_and_priors_are_required():
    outcome = screen_logit_adjustment("F2", _adjustable(hard_rows=0), **_screen())
    assert outcome.details["chosen_tau"] == 0.5
    with pytest.raises(ValueError, match="training priors"):
        screen_logit_adjustment("F2", INCUMBENT, **_screen())


def test_lever_applicability_and_untested_levers():
    assert not lever_applies("L1", "F2") and not lever_applies("L8", "F3")
    assert all(lever_applies(lever, "F4") for lever in ("L1", "L2", "L8"))
    assert lever_not_applicable("L8", "F2").status == LEVER_NOT_APPLICABLE
    with pytest.raises(ValueError, match="applies to F4"):
        lever_not_applicable("L8", "F4")
    with pytest.raises(ValueError, match="does not apply"):
        screen_lever("L1", "F2", INCUMBENT, INCUMBENT, **_screen())
    with pytest.raises(ValueError, match="own screening rule"):
        screen_lever("L7", "F2", INCUMBENT, INCUMBENT, **_screen())
    assert lever_not_tested("L3", "F2", "cost ceiling reached").status == LEVER_NOT_TESTED
    assert supcon_lever_testable(0.9) and not supcon_lever_testable(0.89)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        supcon_lever_testable(1.5)


def test_supcon_lever_is_compared_only_with_enough_positive_steps():
    challenger = _oof("l5", _changed(BASE, gains={1: 30}))
    with pytest.raises(ValueError, match="L5 needs positive_step_fraction"):
        screen_lever("L5", "F2", INCUMBENT, challenger, **_screen())
    sparse = screen_lever("L5", "F2", INCUMBENT, challenger, positive_step_fraction=0.4, **_screen())
    assert sparse.status == LEVER_NOT_TESTED and sparse.decision is None and "40.0%" in sparse.reason
    assert sparse.details == {"positive_step_fraction": 0.4}
    tested = screen_lever("L5", "F2", INCUMBENT, challenger, positive_step_fraction=0.95, **_screen())
    assert tested.status == LEVER_PASSED and tested.details == {"positive_step_fraction": 0.95}
    with pytest.raises(ValueError, match="L5 only"):
        screen_lever("L4", "F2", INCUMBENT, challenger, positive_step_fraction=0.95, **_screen())


def test_supcon_coverage_holds_for_each_required_fit_and_missing_counters_never_pass():
    good = {"train_batches": 100, "batches_with_positive": 95}
    fraction, details = supcon_positive_step_fraction({"f0s42": good, "f1s42": {"train_batches": 50,
                                                                                "batches_with_positive": 46}})
    assert fraction == pytest.approx(0.92) and supcon_lever_testable(fraction) and not details["uncertified_fits"]
    # Pooled the two fits reach 92.7%, but one fit is at 88%: the minimum decides.
    low, details = supcon_positive_step_fraction({"f0s42": good, "f1s42": {"train_batches": 50,
                                                                           "batches_with_positive": 44}})
    assert low == pytest.approx(0.88) and not supcon_lever_testable(low)
    assert details["aggregation"] == "minimum over the required inner fits"
    for missing in (None, {}, {"train_batches": None, "batches_with_positive": None},
                    {"train_batches": 0, "batches_with_positive": 0}, {"train_batches": 10, "batches_with_positive": 11},
                    {"train_batches": True, "batches_with_positive": True}, {"train_batches": 10.0,
                                                                             "batches_with_positive": 9}):
        fraction, details = supcon_positive_step_fraction({"f0s42": good, "f1s42": missing})
        assert fraction is None and details["uncertified_fits"] == ["f1s42"]
    with pytest.raises(ValueError, match="at least one required fit"):
        supcon_positive_step_fraction({})


def _f2_s1_winner():
    return _oof("s1", BASE, priors=np.full((2, len(IDS), 7), 1 / 7))


def _f2_outcomes(l4_gain=20, l6_gain=25):
    s1 = _f2_s1_winner()
    flat = {12: _oof("e12", BASE), 20: _oof("e20", BASE)}
    return s1, [
        lever_not_applicable("L1", "F2"),
        lever_not_applicable("L2", "F2"),
        lever_not_tested("L3", "F2", "cost ceiling reached"),
        screen_lever("L4", "F2", s1, _oof("l4", _changed(BASE, gains={1: l4_gain})), **_screen()),
        screen_lever("L5", "F2", s1, _oof("l5", _changed(BASE, gains={1: 30})), positive_step_fraction=0.4,
                     **_screen()),
        screen_lever("L6", "F2", s1, _oof("l6", _changed(BASE, gains={2: l6_gain})), **_screen()),
        screen_epoch_lever("F2", s1, flat, **_screen()),
        lever_not_applicable("L8", "F2"),
        screen_logit_adjustment("F2", s1, **_screen()),
    ]


def _replace_lever(outcomes, replacement):
    return [replacement if outcome.lever == replacement.lever else outcome for outcome in outcomes]


def test_lever_screen_bookkeeping():
    s1, outcomes = _f2_outcomes()
    screen = s2_lever_screen("F2", s1, outcomes[::-1])
    assert [o.lever for o in screen.outcomes] == ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
    assert [o.lever for o in screen.passing] == ["L4", "L6"]
    assert screen.outcomes[4].status == LEVER_NOT_TESTED
    record = json.loads(canonical_json(screen.to_record()))
    assert len(record["s1_winner_predictions_sha256"]) == 64
    with pytest.raises(ValueError, match="without an outcome"):
        s2_lever_screen("F2", s1, outcomes[:-1])
    with pytest.raises(ValueError, match="more than one outcome"):
        s2_lever_screen("F2", s1, outcomes + outcomes[:1])
    with pytest.raises(ValueError, match="is applicable"):
        s2_lever_screen("F2", s1, _replace_lever(outcomes, LeverOutcome("L4", LEVER_NOT_APPLICABLE)))
    with pytest.raises(ValueError, match="without a promotion decision"):
        s2_lever_screen("F2", s1, _replace_lever(outcomes, LeverOutcome("L4", LEVER_PASSED)))


def test_lever_screen_checks_each_decision_against_the_s1_winner():
    s1, outcomes = _f2_outcomes()
    weak = _oof("weak", _changed(BASE, gains={0: 6}))
    failed = _promote(s1, weak)
    assert not failed.promoted
    mislabelled = LeverOutcome("L4", LEVER_PASSED, decision=failed, challenger=weak)
    with pytest.raises(ValueError, match="promoted=False"):
        s2_lever_screen("F2", s1, _replace_lever(outcomes, mislabelled))
    other_incumbent = _oof("other", _changed(BASE, losses={1: 20}))
    elsewhere = screen_lever("L4", "F2", other_incumbent, _oof("l4", _changed(BASE, gains={1: 5})), **_screen())
    assert elsewhere.status == LEVER_PASSED
    with pytest.raises(ValueError, match="not the S1 winner"):
        s2_lever_screen("F2", s1, _replace_lever(outcomes, elsewhere))
    swapped = LeverOutcome("L4", LEVER_FAILED, decision=failed, challenger=_oof("weak", BASE))
    with pytest.raises(ValueError, match="differ from the ones its decision scored"):
        s2_lever_screen("F2", s1, _replace_lever(outcomes, swapped))
    carried = LeverOutcome("L3", LEVER_NOT_TESTED, reason="cost", decision=failed, challenger=weak)
    with pytest.raises(ValueError, match="carries a comparison"):
        s2_lever_screen("F2", s1, _replace_lever(outcomes, carried))
    supcon = _oof("l5", _changed(BASE, gains={1: 30}))
    unchecked = LeverOutcome("L5", LEVER_PASSED, decision=_promote(s1, supcon), challenger=supcon)
    with pytest.raises(ValueError, match="L5 was screened without"):
        s2_lever_screen("F2", s1, _replace_lever(outcomes, unchecked))


def test_combination_is_adopted_only_if_it_beats_the_best_single_lever():
    s1, outcomes = _f2_outcomes()
    screen = s2_lever_screen("F2", s1, outcomes)
    combined = _oof("combined", _changed(BASE, gains={1: 20, 2: 25}))
    result = s3_combination(s1, screen, combined, **_screen())
    assert (result.selected, result.adopted_levers, result.best_single_lever) == ("combination", ("L4", "L6"), "L6")
    assert (result.decision.incumbent, result.decision.challenger) == ("l6", "combined")
    fallback = s3_combination(s1, screen, _oof("combined", _changed(BASE, gains={2: 25})), **_screen())
    assert (fallback.selected, fallback.adopted_levers) == ("L6", ("L6",))
    with pytest.raises(ValueError, match="needs the combined fit"):
        s3_combination(s1, screen, None, **_screen())
    with pytest.raises(ValueError, match="another S1 winner"):
        s3_combination(INCUMBENT, screen, combined, **_screen())


def test_combination_falls_back_to_one_lever_or_the_s1_winner():
    s1, outcomes = _f2_outcomes(l4_gain=0)
    single = s3_combination(s1, s2_lever_screen("F2", s1, outcomes), **_screen())
    assert (single.selected, single.adopted_levers, single.decision) == ("L6", ("L6",), None)
    with pytest.raises(ValueError, match="at least two levers"):
        s3_combination(s1, s2_lever_screen("F2", s1, outcomes), s1, **_screen())
    s1, outcomes = _f2_outcomes(l4_gain=0, l6_gain=0)
    none = s3_combination(s1, s2_lever_screen("F2", s1, outcomes), **_screen())
    assert (none.selected, none.adopted_levers, none.selected_oof) == ("s1_winner", (), s1)


def test_refit_specification_is_consistent_with_the_adopted_levers():
    spec = s4_refit_specification(
        family="F4", learning_rate_setting={"learning_rate": 1e-4, "gvp_learning_rate": 3e-4},
        adopted_levers=["L7", "L1"], epochs=12,
    )
    assert spec["adopted_levers"] == ["L1", "L7"] and spec["seeds"] == [42, 43, 44, 45, 46]
    assert spec["learning_rate_setting"] == {"gvp_learning_rate": 3e-4, "learning_rate": 1e-4}
    assert spec["learning_rate_label"] == s1_learning_rate_labels("F4")[1]
    assert "gvp_weight_decay_for_target_shrink" in spec["recompute_on_refit_training_set"]
    bad = [
        {"family": "F2", "adopted_levers": ["L7"], "epochs": 30},
        {"family": "F2", "adopted_levers": [], "epochs": 12},
        {"family": "F2", "adopted_levers": ["L9"], "tau": 0.0},
        {"family": "F2", "adopted_levers": [], "tau": 0.5},
        {"family": "F2", "adopted_levers": ["L8"]},
        {"family": "F1", "adopted_levers": []},
        {"family": "F2", "adopted_levers": [], "seeds": (42, 43)},
        {"family": "F2", "adopted_levers": [], "seeds": (1, 2, 3, 4, 5)},
        {"family": "F2", "adopted_levers": [], "seeds": (42, 43, 44, 45, 47)},
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            s4_refit_specification(learning_rate_setting={"learning_rate": 1e-4}, **kwargs)


@pytest.mark.parametrize(
    ("family", "setting"),
    [
        ("F2", {"learning_rate": 5e-3}),
        ("F4", {"learning_rate": 5e-3}),
        ("F4", {"learning_rate": 1e-4}),
        ("F5", {"gvp_learning_rate": 1e-4, "learning_rate": 1e-4}),
    ],
)
def test_refit_refuses_a_learning_rate_outside_the_s1_grid(family, setting):
    with pytest.raises(ValueError, match="not in the S1 grid"):
        s4_refit_specification(family=family, learning_rate_setting=setting, adopted_levers=[])


# Hash-chained step records


def test_step_chain_links_each_record_to_the_previous_one():
    decision = _promote(INCUMBENT, _oof("c", _changed(BASE, gains={1: 12})))
    chain = StepChain()
    chain.append("s1_learning_rate", {"family": "F2", "chosen": "lr=1e-4"})
    chain.append("s1_family_escalation", decision.to_record())
    chain.append("s2_lever_screen", {"passing": ["L4"]})
    records = [json.loads(canonical_json(record)) for record in chain.records]
    assert records[0]["previous_sha256"] == GENESIS_SHA256
    assert verify_step_chain(records) == chain.head_sha256
    tampered = json.loads(json.dumps(records))
    tampered[0]["payload"]["chosen"] = "lr=3e-4"
    with pytest.raises(ValueError, match="record 1 does not link"):
        verify_step_chain(tampered)
    reindexed = json.loads(json.dumps(records))
    reindexed[2]["index"] = 5
    with pytest.raises(ValueError, match="carries index"):
        verify_step_chain(reindexed)


def test_step_chain_file_round_trip_with_anchor(tmp_path):
    anchor = "ab" * 32
    path = tmp_path / "selection_steps.jsonl"
    for step in ("s1", "s2", "s3"):
        append_step_record(path, step, {"step": step, "value": 1.5}, anchor_sha256=anchor)
    records = read_step_chain(path, anchor_sha256=anchor)
    assert [record["step"] for record in records] == ["s1", "s2", "s3"]
    assert records[0]["previous_sha256"] == anchor
    with pytest.raises(ValueError, match="does not link"):
        read_step_chain(path)
    lines = path.read_text(encoding="ascii").splitlines()
    lines[1] = lines[1].replace("1.5", "2.5")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="record 2 does not link"):
        read_step_chain(path, anchor_sha256=anchor)


# Null operating characteristics


SMALL_MODEL = NullNoiseModel(n_proteins=240)
SMALL_CONFIG = PromotionConfig(n_draws=300, stratified=True)


def test_null_simulation_is_reproducible():
    first = simulate_null_adoption(SMALL_MODEL, n_replicates=8, config=SMALL_CONFIG, seed=5)
    second = simulate_null_adoption(SMALL_MODEL, n_replicates=8, config=SMALL_CONFIG, seed=5)
    assert canonical_json(first.to_record()) == canonical_json(second.to_record())
    assert 0.0 <= first.adoption_rate <= 1.0 and first.target_rate == 0.10
    assert first.n_bootstrap_evaluated >= first.n_lower_bound_passed >= first.n_adopted
    assert first.noise_model["n_proteins"] == 240


def test_simulated_true_effect_is_adopted_and_two_looks_run():
    strong = NullNoiseModel(n_proteins=240, challenger_shift=3.0)
    assert simulate_null_adoption(strong, n_replicates=6, config=SMALL_CONFIG, seed=6).adoption_rate >= 0.8
    two_looks = simulate_null_adoption(SMALL_MODEL, n_replicates=4, looks=2, config=SMALL_CONFIG, seed=7)
    assert two_looks.looks == 2 and two_looks.n_replicates == 4


def test_noise_model_and_simulation_inputs_are_validated():
    with pytest.raises(ValueError, match="summing to 1"):
        NullNoiseModel(class_proportions=(0.5,) * 7)
    with pytest.raises(ValueError, match="strictly between"):
        NullNoiseModel(class_recall=(1.0,) * 7)
    with pytest.raises(ValueError, match="positive"):
        simulate_null_adoption(SMALL_MODEL, n_replicates=0)
    with pytest.raises(ValueError, match="zero_recall_unit"):
        PromotionConfig(zero_recall_unit="per_fold")


def test_module_does_not_import_torch():
    src = Path(__file__).resolve().parents[1] / "src"
    code = f"import sys; sys.path.insert(0, {str(src)!r}); import ec_clean30.selection; print('torch' in sys.modules)"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"


SIMULATOR_SOURCE_SHA256 = "673d16453eead70266eb493b7ad5c654275b79cafe41dd2f2e9c3d56763c589a"  # as at 697427e


def test_null_simulator_source_is_unchanged():
    # simulate_null_adoption is frozen for the campaign (Phase 2 plan, WP-6); calibration lives beside it.
    source = inspect.getsource(simulate_null_adoption).encode("utf-8")
    assert hashlib.sha256(source).hexdigest() == SIMULATOR_SOURCE_SHA256


# ---------------------------------------------------------------------------
# Noise-model calibration from a certified inner membership


def _membership_rows(groups):
    """``groups`` maps a group ID to its members' EC class tokens."""
    rows, components = [], {}
    for group, tokens in groups.items():
        for position, token in enumerate(tokens):
            accession = f"{group.upper()}{position:03d}"
            rows.append({"accession": accession, "group_id": group, "ec_class": token, "fold": "0",
                         "structure_id": f"{accession}__chain_A__EC_{token}.1.1.1"})
            components[accession] = group
    return rows, components


CALIBRATION_GROUPS = {"g1": ["1", "1", "1", "2"], "g2": ["3", "3"], "s1": ["3"], "s2": ["4"], "s3": ["5"],
                      "s4": ["6"], "s5": ["7"], "s6": ["2"]}


def test_membership_statistics_by_hand():
    rows, components = _membership_rows(CALIBRATION_GROUPS)
    components["X0001"] = "g1"  # an excluded node of a component: not counted
    components["Y0001"] = components["Y0002"] = "x1"  # a component of excluded nodes only: not a component here
    stats = membership_noise_statistics(rows, components)
    counts = np.array([3, 2, 3, 1, 1, 1, 1])
    concentration = float(np.sum((counts / 12) ** 2))
    agreement = 4 / 7  # g1: 3 of 6 pairs agree; g2: 1 of 1
    assert stats["n_proteins"] == 12 and stats["n_components"] == 8
    assert np.allclose(stats["class_proportions"], counts / 12)
    assert stats["mean_component_size"] == 1.5 and stats["largest_component_size"] == 4
    assert stats["singleton_components"] == 6 and stats["within_component_pairs"] == 7
    assert stats["component_size_variance"] == pytest.approx(1.0)
    assert stats["geometric_size_variance"] == pytest.approx(0.75)
    assert stats["within_component_pair_agreement"] == pytest.approx(agreement)
    assert stats["class_concentration"] == pytest.approx(concentration)
    expected = np.sqrt((agreement - concentration) / (1 - concentration))
    assert stats["component_class_purity"] == pytest.approx(expected) and stats["purity_identified"]
    model = noise_model_from_membership(rows, components, base=NullNoiseModel(fit_shift_sd=0.2, protein_sd=0.7))
    assert (model.n_proteins, model.mean_component_size) == (12, 1.5)
    assert np.allclose(model.class_proportions, counts / 12)
    assert model.component_class_purity == pytest.approx(expected)
    assert (model.fit_shift_sd, model.protein_sd, model.class_recall) == (0.2, 0.7, NullNoiseModel().class_recall)


def test_membership_statistics_without_pairs_and_refusals():
    rows, components = _membership_rows({f"s{k}": [token] for k, token in enumerate("1223")})
    stats = membership_noise_statistics(rows, components)
    assert (stats["component_class_purity"], stats["purity_identified"]) == (1.0, False)
    assert stats["mean_component_size"] == 1.0 and stats["within_component_pair_agreement"] is None
    refusals = [
        ((rows, dict(components, S0000="s9", S2000="s9")),
         r"2 row\(s\) name a group other than their inner component \(first at data row 1\)"),
        ((rows, {key: value for key, value in components.items() if key != "S1000"}),
         r"1 row\(s\) have no inner identity component \(first at data row 2\)"),
        ((rows + rows[:1], components), r"1 row\(s\) repeat the accession of an earlier row \(first at data row 5\)"),
        (([dict(rows[0], ec_class="")] + rows[1:], components), r"1 row\(s\) have no EC class 1-7"),
        (([dict(rows[0], accession="")] + rows[1:], components), r"1 row\(s\) lack an accession or a group"),
        (([dict(rows[0], ec_class="x"), dict(rows[1], group_id="s9")] + rows[2:], components),
         r"have no EC class 1-7 \(first at data row 1\); 1 row\(s\) name a group other"),
    ]
    for arguments, message in refusals:
        with pytest.raises(ValueError, match=message) as caught:
            membership_noise_statistics(*arguments)
        # No accession, group or class token is printed (a step may read every fold's membership).
        assert not any(text in str(caught.value) for text in ("S0", "S1", "S2", "S3", "s9", "'x'", "'1'"))
    with pytest.raises(ValueError, match="fewer than two EC classes"):
        membership_noise_statistics(*_membership_rows({"g": ["3", "3"], "s": ["3"]}))
    with pytest.raises(ValueError, match="no rows"):
        membership_noise_statistics([], components)


def test_purity_estimate_recovers_the_simulated_purity():
    from ec_clean30.selection import _simulated_population

    model = NullNoiseModel(n_proteins=6000, mean_component_size=3.0, component_class_purity=0.6)
    true, component, _difficulty = _simulated_population(np.random.default_rng(11), model)
    rows = [{"accession": f"P{k:05d}", "group_id": f"c{group}", "ec_class": str(cls + 1)}
            for k, (cls, group) in enumerate(zip(true.tolist(), component.tolist()))]
    components = {row["accession"]: row["group_id"] for row in rows}
    stats = membership_noise_statistics(rows, components)
    assert stats["component_class_purity"] == pytest.approx(0.6, abs=0.05)
    assert stats["mean_component_size"] == pytest.approx(3.0, abs=0.15)
