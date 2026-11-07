"""Paired cluster bootstrap, one-sided bounds, intersection-union and Holm (synthetic; no dataset access)."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ec_clean30 import stats
from ec_clean30.stats import (
    CLASS_7_ONLY_STRATUM, CO_PRIMARY_FAMILY_ALPHA, DEFAULT_BOOTSTRAP_SEED, DEFAULT_MAX_SKIP_FRACTION, DEFAULT_N_DRAWS,
    DRAW_BATCH_SIZE, UNSTRATIFIED_STRATUM, BootstrapAbort, Contrast, assign_cluster_strata,
    build_cluster_count_table, co_primary_holm, confusion_metric, holm_intersection_union, holm_step_down,
    intersection_union_test, one_sided_lower_bound, one_sided_p_value, paired_cluster_bootstrap, point_contrasts,
    rarest_class_stratum, single_cluster_primary_strata, single_cluster_strata, stratified_multiplicities,
    stratum_members, stratum_sizes,
)
from training.ec_class_metrics import ec_class_metrics


def test_frozen_bootstrap_defaults():
    assert (DEFAULT_N_DRAWS, DEFAULT_MAX_SKIP_FRACTION, DEFAULT_BOOTSTRAP_SEED) == (10_000, 0.01, 20261023)
    assert (DRAW_BATCH_SIZE, CO_PRIMARY_FAMILY_ALPHA) == (1_000, 0.05)
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(paired_cluster_bootstrap).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    assert defaults == {
        "systems": None, "metric": "ec_classes_1_6_balanced_accuracy", "n_draws": 10_000, "seed": 20261023,
        "stratified": True, "max_skip_fraction": 0.01, "engine": "stratified_k_of_k_v1",
    }
    # The Freeze-1 list is unchanged (freeze.live_constants reads it); engine v2 adds the primary metric.
    assert stats.BOOTSTRAP_METRICS == ("ec_classes_1_6_balanced_accuracy", "ec_classes_1_6_macro_f1", "accuracy")
    assert stats.BOOTSTRAP_METRICS_V2 == (*stats.BOOTSTRAP_METRICS, "ec_classes_1_6_accuracy")
    assert stats.ENGINE_METRICS == {"stratified_k_of_k_v1": stats.BOOTSTRAP_METRICS,
                                    "stratified_rao_wu_v2": stats.BOOTSTRAP_METRICS_V2}
    assert stats.BOOTSTRAP_ENGINES == ("stratified_k_of_k_v1", "stratified_rao_wu_v2")
    assert stats.PRESENCE_FREE_METRICS == ("ec_classes_1_6_accuracy",)


def _wrong(true):
    return (np.asarray(true) + 1) % 6


def _population(n_per_class=30, n_classes=6, cluster_size=1):
    ids, true = [], []
    for cls in range(n_classes):
        for k in range(n_per_class):
            ids.append(f"c{cls}_p{k}")
            true.append(cls)
    clusters = {pid: f"g{index // cluster_size}" for index, pid in enumerate(ids)}
    return ids, np.array(true), clusters


def _predictions(true, accuracy, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(true.size) < accuracy, true, _wrong(true))


def _difference(table, **kwargs):
    return paired_cluster_bootstrap(table, [Contrast.difference("d", "a", "b")], **kwargs)


def test_identical_predictions_give_zero_bound_and_unit_p_value():
    ids, true, clusters = _population()
    pred = _predictions(true, 0.7, seed=1)
    table = build_cluster_count_table(ids, true, {"a": pred, "b": pred.copy()}, clusters)
    result = _difference(table, n_draws=500)
    assert result.estimate("d") == 0.0
    assert result.lower_bound("d", 0.025) == 0.0
    assert result.p_value("d") == 1.0
    assert not result.test("d", 0.025).passed


def test_exchangeable_systems_give_p_near_half_and_negative_bound():
    ids, true, a, b = [], [], [], []
    for cls in range(6):
        for k in range(20):
            for role in ("a_only", "b_only", "both"):
                ids.append(f"{cls}_{k}_{role}")
                true.append(cls)
                a.append(cls if role != "b_only" else (cls + 1) % 6)
                b.append(cls if role != "a_only" else (cls + 1) % 6)
    clusters = {pid: pid for pid in ids}
    table = build_cluster_count_table(ids, np.array(true), {"a": np.array(a), "b": np.array(b)}, clusters)
    result = _difference(table, n_draws=2000)
    assert result.estimate("d") == pytest.approx(0.0)
    assert 0.4 < result.p_value("d") < 0.65
    assert result.lower_bound("d", 0.05) < 0.0


def test_large_true_difference_gives_positive_bound():
    ids, true, clusters = _population(cluster_size=2)
    table = build_cluster_count_table(
        ids, true, {"a": _predictions(true, 0.95, 2), "b": _predictions(true, 0.5, 3)}, clusters
    )
    result = _difference(table, n_draws=2000)
    assert result.lower_bound("d", 0.025) > 0.0
    assert result.p_value("d") == 0.0
    assert result.test("d", 0.025).passed


@pytest.mark.parametrize(
    ("metric", "key", "engine"),
    [
        ("ec_classes_1_6_balanced_accuracy", "ec_classes_1_6_balanced_accuracy", stats.ENGINE_FREEZE1),
        ("ec_classes_1_6_macro_f1", "ec_classes_1_6_macro_f1", stats.ENGINE_FREEZE1),
        ("accuracy", "accuracy", stats.ENGINE_FREEZE1),
        ("ec_classes_1_6_balanced_accuracy", "ec_classes_1_6_balanced_accuracy", stats.ENGINE_AMENDMENT2),
        ("ec_classes_1_6_accuracy", "ec_classes_1_6_accuracy", stats.ENGINE_AMENDMENT2),
    ],
)
def test_point_estimate_matches_ec_class_metrics(metric, key, engine):
    rng = np.random.default_rng(5)
    true = rng.choice(7, size=120, p=[0.2, 0.2, 0.3, 0.1, 0.05, 0.1, 0.05])
    pred = np.where(rng.random(true.size) < 0.6, true, rng.integers(0, 7, true.size))
    weights = rng.choice([0.5, 1.0], size=true.size)
    ids = [f"p{k}" for k in range(true.size)]
    clusters = {pid: f"g{k // 3}" for k, pid in enumerate(ids)}
    table = build_cluster_count_table(ids, true, {"a": pred}, clusters, weights=weights)
    result = paired_cluster_bootstrap(
        table, [Contrast("a_value", ((1.0, "a", "all"),))], metric=metric, n_draws=50, max_skip_fraction=1.0,
        engine=engine,
    )
    assert result.system_value("a") == pytest.approx(ec_class_metrics(true, pred, weights=weights)[key])
    assert confusion_metric(table.counts.sum(axis=0)[0, 0], metric, list(range(6))) == pytest.approx(
        ec_class_metrics(true, pred, weights=weights)[key])


def test_resampled_confusion_metric_matches_expanded_rows():
    rng = np.random.default_rng(6)
    ids, true, clusters = _population(n_per_class=8, cluster_size=3)
    pred = _predictions(true, 0.6, 7)
    table = build_cluster_count_table(ids, true, {"a": pred}, clusters)
    multiplicity = rng.integers(0, 3, size=len(table.cluster_ids)).astype(float)
    multiplicity[:6] = 1.0
    confusion = np.tensordot(multiplicity, table.counts, axes=(0, 0))[0, 0]
    cluster_position = {cluster: k for k, cluster in enumerate(table.cluster_ids)}
    row_weight = np.array([multiplicity[cluster_position[clusters[pid]]] for pid in ids])
    keep = row_weight > 0
    reference = ec_class_metrics(true[keep], pred[keep], weights=row_weight[keep])
    classes = [c for c in range(6) if reference["per_class_support"][c] > 0]
    assert confusion_metric(confusion, "ec_classes_1_6_balanced_accuracy", classes) == pytest.approx(
        reference["ec_classes_1_6_balanced_accuracy"]
    )
    assert confusion_metric(confusion, "ec_classes_1_6_macro_f1", classes) == pytest.approx(
        reference["ec_classes_1_6_macro_f1"]
    )


def test_strata_follow_the_rarest_class_with_documented_ties():
    # Population support: class index 0: 10, 1: 10, 2: 3, 3: 3, 4: 1, 6: 2 (class 7).
    support = np.zeros((6, 7))
    support[0, [0, 6]] = (6, 1)    # classes 1 and 7: class 7 is ignored -> class 1
    support[1, [0, 1]] = (4, 4)    # classes 1 and 2 tie on population support -> lower class (1)
    support[2, [1, 2]] = (6, 1)    # class 3 rarer than class 2
    support[3, [2, 3]] = (2, 3)    # classes 3 and 4 tie -> class 3
    support[4, [4, 0]] = (1, 0)    # class 5, the rarest
    support[5, 6] = 1              # class 7 only
    strata = assign_cluster_strata(support)
    assert strata == (
        rarest_class_stratum(0), rarest_class_stratum(0), rarest_class_stratum(2),
        rarest_class_stratum(2), rarest_class_stratum(4), CLASS_7_ONLY_STRATUM,
    )


def test_stratified_draws_keep_each_stratum_cluster_count():
    strata = ("a",) * 4 + ("b",) * 7 + ("c",)
    members = stratum_members(strata)
    multiplicity = stratified_multiplicities(np.random.default_rng(0), list(members.values()), len(strata), 300)
    for indices in members.values():
        assert np.all(multiplicity[:, indices].sum(axis=1) == indices.size)
    assert np.all(multiplicity[:, members["c"]] == 1)


def test_same_seed_reproduces_draws_and_other_seed_differs():
    ids, true, clusters = _population(cluster_size=2)
    table = build_cluster_count_table(
        ids, true, {"a": _predictions(true, 0.8, 8), "b": _predictions(true, 0.7, 9)}, clusters
    )
    first = _difference(table, n_draws=1500, seed=11)
    second = _difference(table, n_draws=1500, seed=11)
    other = _difference(table, n_draws=1500, seed=12)
    assert np.array_equal(first.draws("d"), second.draws("d"))
    assert first.draws_sha256() == second.draws_sha256()
    assert not np.array_equal(first.draws("d"), other.draws("d"))


def _rare_class_table():
    ids, true, clusters = _population(n_per_class=20, n_classes=5)
    ids.append("rare")
    true = np.append(true, 5)
    clusters["rare"] = "g_rare"
    table = build_cluster_count_table(ids, true, {"a": true.copy(), "b": _wrong(true)}, clusters)
    return table


def test_unstratified_draws_missing_a_present_class_are_skipped_counted_and_abort():
    table = _rare_class_table()
    with pytest.raises(BootstrapAbort, match="lacked an EC class"):
        _difference(table, n_draws=2000, stratified=False)
    result = _difference(table, n_draws=2000, stratified=False, max_skip_fraction=1.0)
    rng = np.random.default_rng(result.seed)
    rare = table.cluster_ids.index("g_rare")
    expected = 0
    for start in range(0, 2000, DRAW_BATCH_SIZE):
        batch = min(DRAW_BATCH_SIZE, 2000 - start)
        multiplicity = stratified_multiplicities(rng, [np.arange(len(table.cluster_ids))], len(table.cluster_ids), batch)
        expected += int((multiplicity[:, rare] == 0).sum())
    assert result.n_skipped == expected
    assert result.n_skipped + result.n_valid == 2000
    assert 0.3 < result.n_skipped / 2000 < 0.45
    assert result.strata_sizes == {UNSTRATIFIED_STRATUM: len(table.cluster_ids)}


def test_stratification_protects_the_rare_class():
    result = _difference(_rare_class_table(), n_draws=2000, stratified=True)
    assert result.n_skipped == 0
    assert result.strata_sizes[rarest_class_stratum(5)] == 1


@pytest.mark.parametrize(("n_missing", "aborts"), [(100, False), (101, True)])
def test_default_skip_limit_is_one_percent_of_ten_thousand_draws(monkeypatch, n_missing, aborts):
    table = _rare_class_table()
    rare = table.cluster_ids.index("g_rare")
    donor = (rare + 1) % len(table.cluster_ids)
    drawn = {"count": 0}

    def scripted(rng, members, n_clusters, n_draws):
        # Every cluster once, except that the first ``n_missing`` draws swap the rare cluster for a second copy of another.
        multiplicity = np.ones((n_draws, n_clusters))
        rows = np.arange(drawn["count"], drawn["count"] + n_draws) < n_missing
        multiplicity[rows, rare], multiplicity[rows, donor] = 0.0, 2.0
        drawn["count"] += n_draws
        return multiplicity

    monkeypatch.setattr(stats, "stratified_multiplicities", scripted)
    contrast = [Contrast.difference("d", "a", "b")]
    if aborts:
        with pytest.raises(BootstrapAbort, match="101 of 10000"):
            paired_cluster_bootstrap(table, contrast)
    else:
        result = paired_cluster_bootstrap(table, contrast)
        assert (result.n_draws, result.n_skipped, result.max_skip_fraction) == (10_000, 100, 0.01)


def test_skip_limit_is_more_than_one_percent():
    table = _rare_class_table()
    probe = _difference(table, n_draws=200, stratified=False, max_skip_fraction=1.0)
    exact = probe.n_skipped / 200
    assert _difference(table, n_draws=200, stratified=False, max_skip_fraction=exact).n_skipped == probe.n_skipped
    with pytest.raises(BootstrapAbort):
        _difference(table, n_draws=200, stratified=False, max_skip_fraction=exact - 0.005)


def test_class_7_never_triggers_a_skip():
    ids, true, clusters = _population(n_per_class=30)
    ids.append("translocase")
    true = np.append(true, 6)
    clusters["translocase"] = "g_translocase"
    table = build_cluster_count_table(ids, true, {"a": true.copy(), "b": _wrong(true)}, clusters)
    result = _difference(table, n_draws=2000, stratified=False)
    assert result.n_skipped == 0
    rng = np.random.default_rng(result.seed)
    multiplicity = stratified_multiplicities(rng, [np.arange(len(table.cluster_ids))], len(table.cluster_ids), DRAW_BATCH_SIZE)
    assert (multiplicity[:, table.cluster_ids.index("g_translocase")] == 0).any()


def test_lower_bound_is_positive_exactly_when_p_is_at_most_alpha():
    rng = np.random.default_rng(13)
    for _ in range(200):
        n = int(rng.integers(50, 3000))
        draws = rng.normal(rng.uniform(-0.05, 0.1), 0.05, size=n)
        draws[rng.random(n) < 0.02] = 0.0
        for alpha in (0.025, 0.05, 0.10):
            assert (one_sided_lower_bound(draws, alpha) > 0) == (one_sided_p_value(draws) <= alpha)


@pytest.mark.parametrize(("n_nonpositive", "passes"), [(250, True), (251, False)])
def test_lower_bound_boundary_at_exact_alpha_share(n_nonpositive, passes):
    draws = np.ones(10_000)
    draws[:n_nonpositive] = 0.0
    alpha = 0.05 / 2
    assert one_sided_p_value(draws) == pytest.approx(n_nonpositive / 10_000)
    assert (one_sided_lower_bound(draws, alpha) > 0) is passes


def test_systems_average_their_arms_and_share_draws():
    ids, true, clusters = _population(cluster_size=2)
    arms = {f"r1_seed{k}": _predictions(true, 0.6 + 0.05 * k, 20 + k) for k in range(3)}
    arms["ens"] = _predictions(true, 0.75, 30)
    arms["ours"] = _predictions(true, 0.9, 31)
    table = build_cluster_count_table(ids, true, arms, clusters)
    systems = {"seed_mean": ["r1_seed0", "r1_seed1", "r1_seed2"], "ens": ["ens"], "ours": ["ours"]}
    result = paired_cluster_bootstrap(
        table,
        [Contrast.difference("vs_seed_mean", "ours", "seed_mean"), Contrast.difference("vs_ens", "ours", "ens")],
        systems=systems,
        n_draws=1000,
    )
    seed_values = [ec_class_metrics(true, arms[f"r1_seed{k}"])["ec_classes_1_6_balanced_accuracy"] for k in range(3)]
    assert result.system_value("seed_mean") == pytest.approx(np.mean(seed_values))
    paired = result.system_draws[:, 0, 2] - result.system_draws[:, 0, 0]
    assert np.allclose(result.draws("vs_seed_mean"), paired)


def test_intersection_union_needs_every_component_and_takes_the_largest_p():
    ids, true, clusters = _population()
    ours = _predictions(true, 0.9, 40)
    close = ours.copy()
    close[np.flatnonzero(ours == true)[[0, 40, 80]]] = _wrong(true)[np.flatnonzero(ours == true)[[0, 40, 80]]]
    close[np.flatnonzero(ours != true)[:2]] = true[np.flatnonzero(ours != true)[:2]]
    arms = {"ours": ours, "weak": _predictions(true, 0.5, 41), "close": close}
    table = build_cluster_count_table(ids, true, arms, clusters)
    result = paired_cluster_bootstrap(
        table,
        [Contrast.difference("vs_weak", "ours", "weak"), Contrast.difference("vs_close", "ours", "close")],
        n_draws=1000,
    )
    test = intersection_union_test(result, ["vs_weak", "vs_close"], 0.025, name="P1")
    assert test.components[0].passed and not test.components[1].passed
    assert not test.passed
    assert test.p_value == max(result.p_value("vs_weak"), result.p_value("vs_close"))
    assert intersection_union_test(result, ["vs_weak"], 0.025, name="P1").passed


def test_holm_step_down_levels_and_adjusted_p_values():
    steps = holm_step_down({"a": 0.04, "b": 0.01}, family_alpha=0.05)
    assert [step.name for step in steps] == ["b", "a"]
    assert [step.alpha for step in steps] == pytest.approx([0.025, 0.05])
    assert all(step.rejected for step in steps)
    assert [step.adjusted_p_value for step in steps] == pytest.approx([0.02, 0.04])
    blocked = holm_step_down({"a": 0.03, "b": 0.04}, family_alpha=0.05)
    assert not blocked[0].rejected and blocked[0].tested
    assert not blocked[1].tested and not blocked[1].rejected
    three = holm_step_down({"x": 0.01, "y": 0.02, "z": 0.03}, family_alpha=0.05)
    assert [step.alpha for step in three] == pytest.approx([0.05 / 3, 0.025, 0.05])
    assert [step.rejected for step in three] == [True, True, True]
    tied = holm_step_down({"first": 0.01, "second": 0.01})
    assert [step.name for step in tied] == ["first", "second"]


def _contrast_run(accuracy_ours, accuracy_other, seed, identical=False):
    ids, true, clusters = _population(cluster_size=2)
    ours = _predictions(true, accuracy_ours, seed)
    if identical:
        arms = {"ours": ours, "seed_mean": ours.copy(), "ens": ours.copy()}
    else:
        arms = {"ours": ours, "seed_mean": _predictions(true, accuracy_other, seed + 1),
                "ens": _predictions(true, accuracy_other, seed + 2)}
    table = build_cluster_count_table(ids, true, arms, clusters)
    result = paired_cluster_bootstrap(
        table, [Contrast.difference("vs_seed_mean", "ours", "seed_mean"), Contrast.difference("vs_ens", "ours", "ens")],
        n_draws=1000,
    )
    return result, ["vs_seed_mean", "vs_ens"]


def test_co_primary_holm_tests_the_smaller_p_first_and_stops_on_failure():
    strong, strong_components = _contrast_run(0.95, 0.5, 50)
    weak, weak_components = _contrast_run(0.7, 0.7, 60, identical=True)
    result = co_primary_holm(("P1", weak, weak_components), ("P2", strong, strong_components))
    first, second = result.steps
    assert (first.name, first.alpha, first.passed) == ("P2", 0.025, True)
    assert second.name == "P1" and second.alpha == 0.05 and second.tested and not second.passed
    # Steps are in test order, not argument order: claims are read by name.
    assert result.step("P1") is second and result.passed("P2") and not result.passed("P1")
    with pytest.raises(TypeError):
        _, _ = result
    with pytest.raises(KeyError, match="P3"):
        result.step("P3")
    json.dumps(result.to_record())
    first, second = co_primary_holm(("P1", weak, weak_components), ("P2", weak, weak_components)).steps
    assert first.tested and not first.passed
    assert not second.tested and not second.passed and second.test is None
    both = co_primary_holm(("P1", strong, strong_components), ("P2", strong, strong_components))
    assert all(step.passed for step in both.steps)


def test_an_unavailable_co_primary_keeps_the_family_size():
    # Analysis amendment 2026-10-30: the available claim is still tested at 0.025, never at 0.05.
    strong, components = _contrast_run(0.95, 0.5, 50)
    for arguments in ((("P1", None, components), ("P2", strong, components)),
                      (("P2", strong, components), ("P1", None, components))):
        result = co_primary_holm(*arguments)
        first, second = result.steps
        assert (first.name, first.alpha, first.tested, first.passed, first.available) == ("P2", 0.025, True, True, True)
        assert (second.name, second.alpha, second.p_value) == ("P1", 0.05, 1.0)
        assert not second.tested and not second.passed and not second.available and second.test is None
        assert result.to_record()["steps"][1]["available"] is False
    neither = co_primary_holm(("P1", None, components), ("P2", None, components))
    assert not any(step.tested or step.passed or step.available for step in neither.steps)


def test_single_cluster_strata_of_classes_1_6_are_named_and_class_7_is_exempt():
    support = np.zeros((5, 7))
    support[0, 0] = support[1, 0] = 3     # class 1: two clusters
    support[2, 4] = 5                     # class 5: one cluster
    support[3, 1] = support[4, 6] = 1     # class 2: one cluster; class 7 only: one cluster
    assert stratum_sizes(support) == {rarest_class_stratum(0): 2, rarest_class_stratum(1): 1,
                                      rarest_class_stratum(4): 1, CLASS_7_ONLY_STRATUM: 1}
    assert single_cluster_primary_strata(support) == (rarest_class_stratum(1), rarest_class_stratum(4))
    lone_class_7 = np.zeros((3, 7))
    lone_class_7[0, 0] = lone_class_7[1, 0] = lone_class_7[2, 6] = 1
    assert stratum_sizes(lone_class_7)[CLASS_7_ONLY_STRATUM] == 1
    assert single_cluster_primary_strata(lone_class_7) == ()


def test_the_class_7_exemption_covers_balanced_accuracy_only():
    # Two clusters for each of EC classes 1 and 2, and one class-7-only family of two proteins.
    ids = [f"p{i}" for i in range(6)]
    true = np.array([0, 0, 1, 1, 6, 6])
    clusters = {pid: ("family7" if true[i] == 6 else pid) for i, pid in enumerate(ids)}
    as_class_7 = true.copy()                     # the class-7 proteins predicted as class 7
    as_class_2 = np.where(true == 6, 1, true)    # one arm calls them EC class 2: false positives
    as_class_3 = np.where(true == 6, 2, true)    # EC class 3 is absent, so it is not averaged
    plain = build_cluster_count_table(ids, true, {"a": true, "b": as_class_7}, clusters)
    leaking = build_cluster_count_table(ids, true, {"a": true, "b": as_class_2}, clusters)
    absent = build_cluster_count_table(ids, true, {"a": true, "b": as_class_3}, clusters)
    assert stratum_sizes(plain.class_support)[CLASS_7_ONLY_STRATUM] == 1
    ba, accuracy, macro_f1 = stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY, stats.METRIC_ACCURACY, \
        stats.METRIC_EC_CLASSES_1_6_MACRO_F1
    for table in (plain, leaking, absent):
        assert single_cluster_strata(table, ba) == ()
        assert single_cluster_strata(table, accuracy) == (CLASS_7_ONLY_STRATUM,)
    # Macro-F1 over classes 1-6 reads the class-7 family only through false positives of an averaged class.
    contrast = [Contrast.difference("gain", "a", "b")]
    assert single_cluster_strata(leaking, macro_f1) == (CLASS_7_ONLY_STRATUM,)
    assert point_contrasts(leaking, contrast, metric=macro_f1)["gain"] > 0
    for table in (plain, absent):
        assert single_cluster_strata(table, macro_f1) == ()
        assert point_contrasts(table, contrast, metric=macro_f1)["gain"] == 0
    assert point_contrasts(absent, contrast, metric=accuracy)["gain"] == pytest.approx(2 / 6)
    # A single-cluster stratum of an EC class 1-6 counts for every metric, before the class-7 one.
    lone = {**clusters, "p1": "p0"}
    table = build_cluster_count_table(ids, true, {"a": true, "b": as_class_2}, lone)
    for metric, expected in ((ba, (rarest_class_stratum(0),)),
                             (accuracy, (rarest_class_stratum(0), CLASS_7_ONLY_STRATUM)),
                             (macro_f1, (rarest_class_stratum(0), CLASS_7_ONLY_STRATUM))):
        assert single_cluster_strata(table, metric) == expected
    # Two class-7-only families avoid the degeneracy for every metric.
    two = {**clusters, "p5": "family7b"}
    assert all(single_cluster_strata(build_cluster_count_table(ids, true, {"a": true, "b": as_class_2}, two), metric)
               == () for metric in (ba, accuracy, macro_f1))
    with pytest.raises(ValueError, match="Unknown bootstrap metric"):
        single_cluster_strata(plain, "top_1")


def test_a_single_cluster_stratum_gives_a_degenerate_bootstrap():
    # Why the gate exists: one family of class 5, right for one system and wrong for the other.
    ids = [f"p{i}" for i in range(12)]
    true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    clusters = {pid: ("family5" if true[i] == 4 else pid) for i, pid in enumerate(ids)}
    other = np.where(true == 4, 0, true)
    table = build_cluster_count_table(ids, true, {"ours": true, "other": other}, clusters)
    assert single_cluster_primary_strata(table.class_support) == (rarest_class_stratum(4),)
    result = paired_cluster_bootstrap(table, [Contrast.difference("gain", "ours", "other")], n_draws=500)
    assert result.lower_bound("gain", 0.025) == pytest.approx(1 / 6) and result.p_value("gain") == 0.0
    assert point_contrasts(table, [Contrast.difference("gain", "ours", "other")]) == {
        "gain": pytest.approx(result.estimate("gain"))}


def test_bootstrap_holm_matches_p_value_holm():
    runs = {
        "P1": _contrast_run(0.95, 0.5, 70),
        "P2": _contrast_run(0.72, 0.7, 80),
        "P3": _contrast_run(0.8, 0.7, 90),
    }
    steps = holm_intersection_union(runs, family_alpha=0.05)
    p_values = {name: max(result.p_value(c) for c in components) for name, (result, components) in runs.items()}
    reference = {step.name: step.rejected for step in holm_step_down(p_values, family_alpha=0.05)}
    assert {step.name: step.passed for step in steps} == reference


def test_repeated_rows_are_resampled_as_one_unit():
    ids, true, clusters = _population(n_per_class=10)
    pred = _predictions(true, 0.7, 100)
    single = build_cluster_count_table(ids, true, {"a": pred, "b": _wrong(true)}, clusters)
    doubled_ids = ids + [ids[0]]
    doubled = build_cluster_count_table(
        doubled_ids, np.append(true, true[0]), {"a": np.append(pred, pred[0]), "b": np.append(_wrong(true), _wrong(true)[0])},
        clusters, weights=np.r_[0.5, np.ones(len(ids) - 1), 0.5],
    )
    assert np.allclose(single.counts, doubled.counts)
    assert _difference(single, n_draws=300).draws_sha256() == _difference(doubled, n_draws=300).draws_sha256()


def test_difference_in_differences_over_subsets():
    ids, true, clusters = _population(n_per_class=40, cluster_size=2)
    high = np.array([int(pid.split("_p")[1]) % 2 == 0 for pid in ids])
    near = np.where(high, true, _wrong(true))
    remote = np.where(high, true, _predictions(true, 0.6, 110))
    other = _predictions(true, 0.7, 111)
    table = build_cluster_count_table(
        ids, true, {"comparator": near, "ours": np.where(high, other, remote)}, clusters,
        subsets={"high": high, "remote": ~high},
    )
    did = Contrast("did", ((1.0, "comparator", "high"), (-1.0, "comparator", "remote"),
                           (-1.0, "ours", "high"), (1.0, "ours", "remote")))
    result = paired_cluster_bootstrap(table, [did], n_draws=300)

    def ba(pred, mask):
        return ec_class_metrics(true[mask], pred[mask])["ec_classes_1_6_balanced_accuracy"]

    expected = (ba(near, high) - ba(near, ~high)) - (ba(np.where(high, other, remote), high)
                                                     - ba(np.where(high, other, remote), ~high))
    assert result.estimate("did") == pytest.approx(expected)


def test_subsets_must_cover_the_population_unless_allowed():
    ids, true, clusters = _population(n_per_class=20)
    nh30 = np.arange(len(ids)) % 3 == 0
    arms = {"a": true.copy(), "b": _wrong(true)}
    with pytest.raises(ValueError, match="one table per population"):
        build_cluster_count_table(ids, true, arms, clusters, subsets={"nh30": nh30})
    allowed = build_cluster_count_table(ids, true, arms, clusters, subsets={"nh30": nh30}, allow_rows_outside_subsets=True)
    record = paired_cluster_bootstrap(allowed, [Contrast.difference("d", "a", "b", "nh30")], n_draws=200).to_record()
    assert record["n_rows_outside_subsets"] == int((~nh30).sum()) and record["n_rows"] == len(ids)
    own = [pid for pid, keep in zip(ids, nh30) if keep]
    restricted = build_cluster_count_table(
        own, true[nh30], {name: values[nh30] for name, values in arms.items()}, clusters,
        subsets={"nh30": np.ones(len(own), dtype=bool)},
    )
    assert restricted.n_rows_outside_subsets == 0 and len(restricted.cluster_ids) == len(own)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"cluster_of": {}}, "no cluster"),
        ({"true_idx": [0, 1]}, "true_idx must have shape"),
        ({"arm_predictions": {}}, "At least one prediction arm"),
        ({"arm_predictions": {"a": [0, 1, 9]}}, "outside"),
        ({"weights": [1.0, 0.0, 1.0]}, "finite and positive"),
        ({"subsets": {"all": np.zeros(3, dtype=bool)}}, "is empty"),
    ],
)
def test_table_inputs_are_validated(change, message):
    kwargs = {
        "protein_ids": ["x", "y", "z"],
        "true_idx": [0, 1, 2],
        "arm_predictions": {"a": [0, 1, 1]},
        "cluster_of": {"x": "g1", "y": "g1", "z": "g2"},
    }
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        build_cluster_count_table(**kwargs)


def test_unsupported_metric_and_unknown_contrast_terms_are_refused():
    ids, true, clusters = _population(n_per_class=5)
    table = build_cluster_count_table(ids, true, {"a": true.copy(), "b": _wrong(true)}, clusters)
    with pytest.raises(ValueError, match="Unknown bootstrap metric"):
        _difference(table, metric="seven_class_balanced_accuracy")
    with pytest.raises(ValueError, match="unknown system"):
        paired_cluster_bootstrap(table, [Contrast.difference("d", "a", "missing")])
    with pytest.raises(ValueError, match="alpha"):
        one_sided_lower_bound([0.1, 0.2], 0.0)


def test_record_is_json_ready():
    ids, true, clusters = _population(n_per_class=5)
    table = build_cluster_count_table(ids, true, {"a": true.copy(), "b": _wrong(true)}, clusters)
    record = _difference(table, n_draws=100).to_record()
    assert record["n_draws"] == 100 and record["n_skipped"] == 0
    assert record["averaged_ec_classes"] == {"all": [1, 2, 3, 4, 5, 6]}
    assert len(record["contrast_draws_sha256"]) == 64
    json.dumps(record)
    # Engine v1 records keep their Freeze-1 format; engine v2 records add the amendment-2 fields.
    assert not {"engine", "strata_merges", "class_absence_probability", "multiplicities_dyadic"} & set(record)
    v2 = _difference(table, n_draws=100, metric=stats.METRIC_EC_CLASSES_1_6_ACCURACY,
                     engine=stats.ENGINE_AMENDMENT2).to_record()
    assert v2["engine"] == stats.ENGINE_AMENDMENT2 and v2["resampling"] == stats.RESAMPLING_RAO_WU
    assert v2["strata_merges"] == [] and v2["multiplicities_dyadic"] is True  # k = 5 strata: 5 / 4 is dyadic
    # Every class fills its own stratum, so it is in every draw.
    assert v2["class_absence_probability"] == {"all": {str(k): 0.0 for k in range(1, 7)}}
    json.dumps(v2)


def test_record_names_the_generator_and_whether_sums_are_exact():
    ids, true, clusters = _population(n_per_class=5)
    arms = {"a": true.copy(), "b": _wrong(true)}
    record = _difference(build_cluster_count_table(ids, true, arms, clusters), n_draws=100).to_record()
    assert record["numpy_version"] == np.__version__ and record["bit_generator"] == "PCG64"
    assert record["dyadic_counts"] is True and record["n_rows_outside_subsets"] == 0
    halves = build_cluster_count_table(ids, true, arms, clusters, weights=np.full(len(ids), 0.5))
    thirds = build_cluster_count_table(ids, true, arms, clusters, weights=np.full(len(ids), 1 / 3))
    assert halves.dyadic_counts and not thirds.dyadic_counts
    # Rao-Wu multipliers k / (k - 1) are dyadic only when k - 1 is a power of two.
    sizes = {1: True, 2: True, 3: True, 4: False, 5: True, 6: False, 9: True, 10: False}
    for size, dyadic in sizes.items():
        members = [np.arange(size)]
        assert stats.multiplicities_are_dyadic(members, stats.RESAMPLING_RAO_WU) is dyadic
        assert stats.multiplicities_are_dyadic(members, stats.RESAMPLING_K_OF_K) is True
    v2 = _difference(build_cluster_count_table(ids, true, arms, {pid: pid[:4] for pid in ids}), n_draws=50,
                     metric=stats.METRIC_EC_CLASSES_1_6_ACCURACY, engine=stats.ENGINE_AMENDMENT2).to_record()
    assert v2["numpy_version"] == np.__version__ and v2["bit_generator"] == "PCG64"


# ---------------------------------------------------------------- engine v2 (amendment 2)

# contrast_draws_sha256 of engine v1 runs recorded at commit c163995, before any amendment-2 edit.
GOLDEN_V1_DRAWS_C163995 = {
    "ba_cluster2_seed11_1500": "21ba74d1d6ba7c800308a37c3932f6b9a9ae13d92628d1072d632a2c28cae859",
    "accuracy_cluster2_seed11_1500": "94a9fc3048741024405da5064ced665b3693845bbcf51616aeb047784abd7aa4",
    "macro_f1_weighted_seed11_1500": "5750cd4f587b5df8d4a1e44fafb1d166e5c74642c5a5e1c779e4ebe3a4b8aa0b",
    "ba_rare_class_default_2000": "9c7d87294de0610ce1ba95775bb9e8ae510ec6e3b29772fa9d31f0a01fa99de3",
}
GOLDEN_V1_RECORDS_C163995 = {
    "ba_cluster2_seed11_1500": "65a6d8e596341fc8700a0fb86943cc4532b68d42118ed6d637b0a96f3ddd3ca4",
    "accuracy_cluster2_seed11_1500": "1d5f2a42ae5f77d03a6826ea7cf60a71e5b4d09a0f62e81a526e817a6c494f8a",
    "macro_f1_weighted_seed11_1500": "5ab8099748766b2a26791d4c43081ca25d548bc7db3c199068374dede311df09",
    "ba_rare_class_default_2000": "940b1040a6e289bc6096a741a69fa78fce1ac0f5f57a754fc79b8a604adffdf8",
}


def _golden_runs(**kwargs):
    ids, true, clusters = _population(cluster_size=2)
    a, b = _predictions(true, 0.8, 8), _predictions(true, 0.7, 9)
    plain = build_cluster_count_table(ids, true, {"a": a, "b": b}, clusters)
    weighted = build_cluster_count_table(ids, true, {"a": a, "b": b}, clusters,
                                         weights=np.where(np.arange(len(ids)) % 2 == 0, 0.5, 1.0))
    contrast = [Contrast.difference("d", "a", "b")]
    return {
        "ba_cluster2_seed11_1500": paired_cluster_bootstrap(plain, contrast, n_draws=1500, seed=11, **kwargs),
        "accuracy_cluster2_seed11_1500": paired_cluster_bootstrap(plain, contrast, n_draws=1500, seed=11,
                                                                  metric="accuracy", **kwargs),
        "macro_f1_weighted_seed11_1500": paired_cluster_bootstrap(weighted, contrast, n_draws=1500, seed=11,
                                                                  metric="ec_classes_1_6_macro_f1", **kwargs),
        "ba_rare_class_default_2000": paired_cluster_bootstrap(_rare_class_table(), contrast, n_draws=2000, **kwargs),
    }


def test_engine_v1_replays_freeze1_draws():
    for kwargs in ({}, {"engine": stats.ENGINE_FREEZE1}):
        for name, result in _golden_runs(**kwargs).items():
            assert result.draws_sha256() == GOLDEN_V1_DRAWS_C163995[name], name
            record = hashlib.sha256(json.dumps(result.to_record(), sort_keys=True).encode()).hexdigest()
            assert record == GOLDEN_V1_RECORDS_C163995[name], name
    # Engine v2 draws differently (the small-stratum rule), from the same seed.
    v2 = _golden_runs(engine=stats.ENGINE_AMENDMENT2)
    assert v2["ba_cluster2_seed11_1500"].draws_sha256() != GOLDEN_V1_DRAWS_C163995["ba_cluster2_seed11_1500"]


def test_rao_wu_draws_k_minus_1_and_keep_stratum_totals():
    strata = ("a",) * 4 + ("b",) * 7 + ("c",) + ("d",) * 2
    members = stratum_members(strata)
    multiplicity = stratified_multiplicities(np.random.default_rng(0), list(members.values()), len(strata), 500,
                                             stats.RESAMPLING_RAO_WU)
    for name, indices in members.items():
        k = indices.size
        assert np.allclose(multiplicity[:, indices].sum(axis=1), k)
        if k > 1:
            picks = multiplicity[:, indices] * (k - 1) / k
            assert np.allclose(picks, np.round(picks)) and np.allclose(picks.sum(axis=1), k - 1)
            assert (multiplicity[:, indices] == 0).any()  # some cluster is left out of some draw
    assert np.all(multiplicity[:, members["c"]] == 1)
    # A one-cluster stratum uses no random numbers: adding one leaves the other strata's draws unchanged.
    without = stratified_multiplicities(np.random.default_rng(0), [members["a"], members["b"]], len(strata), 50,
                                        stats.RESAMPLING_RAO_WU)
    with_single = stratified_multiplicities(np.random.default_rng(0), [members["a"], members["c"], members["b"]],
                                            len(strata), 50, stats.RESAMPLING_RAO_WU)
    assert np.array_equal(without[:, members["b"]], with_single[:, members["b"]])
    with pytest.raises(ValueError, match="Unknown resampling"):
        stratified_multiplicities(np.random.default_rng(0), list(members.values()), len(strata), 5, "half")


@pytest.mark.parametrize("k", [2, 3])
def test_rao_wu_variance_matches_unbiased_stratified_variance(k):
    rng = np.random.default_rng(100 + k)
    values = rng.normal(size=k)
    members = [np.arange(k)]
    unbiased = k * values.var(ddof=1)  # unbiased variance estimate of the stratum total
    n = 400_000
    rao_wu = stratified_multiplicities(np.random.default_rng(1), members, k, n, stats.RESAMPLING_RAO_WU) @ values
    k_of_k = stratified_multiplicities(np.random.default_rng(1), members, k, n) @ values
    assert rao_wu.var() == pytest.approx(unbiased, rel=0.02)
    assert k_of_k.var() == pytest.approx((k - 1) / k * unbiased, rel=0.02)
    assert rao_wu.mean() == pytest.approx(values.sum(), abs=0.02) and k_of_k.mean() == pytest.approx(values.sum(),
                                                                                                      abs=0.02)


def _no_own_stratum_table():
    """EC class 6 (rarest) shares clusters with every EC class 5 protein; class 5 has no stratum of its own."""
    ids, true, clusters = [], [], {}
    for cls in range(4):
        for k in range(10):
            ids.append(f"c{cls}_{k}")
            true.append(cls)
            clusters[ids[-1]] = ids[-1]
    for k in range(6):  # class 5: six proteins in two mixed clusters
        ids.append(f"c4_{k}")
        true.append(4)
        clusters[ids[-1]] = f"mixed{k // 3}"
    for k in range(5):  # class 6: two in the mixed clusters, three alone
        ids.append(f"c5_{k}")
        true.append(5)
        clusters[ids[-1]] = f"mixed{k}" if k < 2 else ids[-1]
    true = np.array(true)
    return build_cluster_count_table(ids, true, {"a": true.copy(), "b": _wrong(true)}, clusters)


def test_primary_metric_needs_engine_v2_never_skips_and_never_aborts():
    table = _no_own_stratum_table()
    contrast = [Contrast.difference("d", "a", "b")]
    with pytest.raises(BootstrapAbort, match="lacked an EC class"):
        paired_cluster_bootstrap(table, contrast, n_draws=2000)  # engine v1, balanced accuracy
    with pytest.raises(BootstrapAbort):
        paired_cluster_bootstrap(table, contrast, n_draws=2000, engine=stats.ENGINE_AMENDMENT2)
    result = paired_cluster_bootstrap(table, contrast, n_draws=2000, metric=stats.METRIC_EC_CLASSES_1_6_ACCURACY,
                                      engine=stats.ENGINE_AMENDMENT2)
    assert result.n_skipped == 0 and result.n_valid == 2000
    assert result.estimate("d") == pytest.approx(1.0)
    for engine in (stats.ENGINE_FREEZE1,):
        with pytest.raises(ValueError, match="defined for bootstrap engine"):
            paired_cluster_bootstrap(table, contrast, metric=stats.METRIC_EC_CLASSES_1_6_ACCURACY, engine=engine)
        with pytest.raises(ValueError, match="defined for bootstrap engine"):
            stats.inference_gate(table, stats.METRIC_EC_CLASSES_1_6_ACCURACY, engine)
    with pytest.raises(ValueError, match="Unknown bootstrap engine"):
        paired_cluster_bootstrap(table, contrast, engine="stratified_k_of_k_v3")


def test_rarity_order_matches_the_strata():
    support = np.zeros((6, 7))
    support[0, [0, 6]] = (6, 1)
    support[1, [0, 1]] = (4, 4)
    support[2, [1, 2]] = (6, 1)
    support[3, [2, 3]] = (2, 3)
    support[4, [4, 0]] = (1, 0)
    support[5, 6] = 1
    assert stats.rarity_order(support) == (4, 2, 3, 0, 1)
    strata = assign_cluster_strata(support)
    for row, name in zip(support, strata):
        contained = [index for index in stats.rarity_order(support) if row[index] > 0]
        assert name == (rarest_class_stratum(contained[0]) if contained else CLASS_7_ONLY_STRATUM)


def test_single_cluster_strata_merge_to_next_more_common_class():
    def support_of(classes_per_cluster):
        support = np.zeros((len(classes_per_cluster), 7))
        for row, classes in enumerate(classes_per_cluster):
            for cls, weight in classes.items():
                support[row, cls] = weight
        return support

    # Rarity: class 5 (index 4, weight 1) < class 6 (2) < class 2 (3) < class 1 (10); class 7 alone.
    support = support_of([{4: 1}, {5: 1}, {5: 1}, {1: 3}, {0: 5}, {0: 5}, {6: 1}])
    strata, merges = stats.merge_single_cluster_strata(assign_cluster_strata(support), support)
    assert merges == ({"from_stratum": "rarest_ec_class_5", "to_stratum": "rarest_ec_class_6", "n_clusters": 1},
                      {"from_stratum": "rarest_ec_class_2", "to_stratum": "rarest_ec_class_1", "n_clusters": 1})
    assert strata == ("rarest_ec_class_6",) * 3 + ("rarest_ec_class_1",) * 3 + (CLASS_7_ONLY_STRATUM,)
    # Cascading: a one-cluster stratum that receives a cluster is no longer single.
    support = support_of([{4: 1}, {5: 2}, {0: 5}, {0: 5}])
    strata, merges = stats.merge_single_cluster_strata(assign_cluster_strata(support), support)
    assert [item["from_stratum"] for item in merges] == ["rarest_ec_class_5"]
    assert strata == ("rarest_ec_class_6", "rarest_ec_class_6", "rarest_ec_class_1", "rarest_ec_class_1")
    # The most common non-empty stratum gives its single cluster to the next rarer one.
    support = support_of([{4: 1}, {4: 1}, {0: 5}])
    strata, merges = stats.merge_single_cluster_strata(assign_cluster_strata(support), support)
    assert merges == ({"from_stratum": "rarest_ec_class_1", "to_stratum": "rarest_ec_class_5", "n_clusters": 1},)
    assert set(strata) == {"rarest_ec_class_5"}
    # A single non-empty stratum with one cluster stays (the gate asks for two clusters); class 7 is untouched.
    support = support_of([{2: 1}, {6: 1}])
    strata, merges = stats.merge_single_cluster_strata(assign_cluster_strata(support), support)
    assert merges == () and strata == ("rarest_ec_class_3", CLASS_7_ONLY_STRATUM)
    # bootstrap_strata merges only for the presence-free metric under engine v2.
    support = support_of([{4: 1}, {5: 1}, {5: 1}, {0: 5}, {0: 5}])
    for metric, engine, merged in ((stats.METRIC_EC_CLASSES_1_6_ACCURACY, stats.ENGINE_AMENDMENT2, True),
                                   (stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY, stats.ENGINE_AMENDMENT2, False),
                                   (stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY, stats.ENGINE_FREEZE1, False)):
        _, merges = stats.bootstrap_strata(support, metric, engine)
        assert bool(merges) is merged


@pytest.mark.parametrize("resampling", [stats.RESAMPLING_K_OF_K, stats.RESAMPLING_RAO_WU])
def test_class_absence_probability_is_exact(resampling):
    table = _no_own_stratum_table()
    strata = assign_cluster_strata(table.class_support)
    probability = stats.class_absence_probability(table.class_support, strata, resampling)
    # Class 5 (index 4) is in 2 of the 5 clusters of the class-6 stratum.
    exponent = 4 if resampling == stats.RESAMPLING_RAO_WU else 5
    assert probability[4] == pytest.approx((3 / 5) ** exponent)
    assert all(probability[index] == 0.0 for index in (0, 1, 2, 3, 5))
    members = stratum_members(strata)
    rng = np.random.default_rng(7)
    n = 200_000
    multiplicity = stratified_multiplicities(rng, list(members.values()), len(strata), n, resampling)
    holds = table.class_support[:, 4] > 0
    empirical = float(np.mean((multiplicity[:, holds] == 0).all(axis=1)))
    assert empirical == pytest.approx(probability[4], abs=4 * np.sqrt(probability[4] / n) + 1e-4)
    single = stats.class_absence_probability(np.array([[1.0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0]]),
                                             ("x", "y"), resampling)
    assert single == {0: 0.0, 1: 0.0}


def test_inference_gate_v2():
    ba, macro_f1, primary = (stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY, stats.METRIC_EC_CLASSES_1_6_MACRO_F1,
                             stats.METRIC_EC_CLASSES_1_6_ACCURACY)
    v1, v2 = stats.ENGINE_FREEZE1, stats.ENGINE_AMENDMENT2
    table = _no_own_stratum_table()
    assert stats.inference_gate(table, ba, v1) == ()  # the Freeze-1 gate only sees single-cluster strata
    assert stats.inference_gate(table, ba, v2) == ("ec_class_5_not_guaranteed",)
    assert stats.inference_gate(table, macro_f1, v2) == ("ec_class_5_not_guaranteed",)
    assert stats.inference_gate(table, primary, v2) == ()
    absence = stats.table_absence_probabilities(table, ba, v2)
    assert absence["all"][4] == pytest.approx((3 / 5) ** 4)
    # A single-cluster stratum of an EC class 1-6 gates balanced accuracy (SA1) but not the primary metric.
    ids = [f"p{i}" for i in range(12)]
    true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    clusters = {pid: ("family5" if true[i] == 4 else pid) for i, pid in enumerate(ids)}
    lone = build_cluster_count_table(ids, true, {"ours": true, "other": np.where(true == 4, 0, true)}, clusters)
    assert stats.inference_gate(lone, ba, v1) == stats.inference_gate(lone, ba, v2) == (rarest_class_stratum(4),)
    assert stats.inference_gate(lone, primary, v2) == ()
    # Macro-F1 follows SA1-c: a one-cluster class-7-only family predicted as an averaged class.
    ids7 = [f"q{i}" for i in range(6)]
    true7 = np.array([0, 0, 1, 1, 6, 6])
    family = {pid: ("family7" if true7[i] == 6 else pid) for i, pid in enumerate(ids7)}
    leaking = build_cluster_count_table(ids7, true7, {"a": true7, "b": np.where(true7 == 6, 1, true7)}, family)
    plain = build_cluster_count_table(ids7, true7, {"a": true7, "b": true7}, family)
    assert stats.inference_gate(leaking, macro_f1, v2) == (CLASS_7_ONLY_STRATUM,)
    assert stats.inference_gate(plain, macro_f1, v2) == () and stats.inference_gate(leaking, ba, v2) == ()
    # The primary metric is gated only below two clusters with EC class 1-6 support.
    two = build_cluster_count_table(["x", "y"], np.array([0, 1]), {"a": np.array([0, 1])}, {"x": "g", "y": "g"})
    assert stats.inference_gate(two, primary, v2) == (stats.GATE_TOO_FEW_CLUSTERS,)
    # Per subset: a class present in a subset must be guaranteed within that subset.
    subset = np.array([i % 2 == 0 for i in range(len(ids))])
    halves = build_cluster_count_table(ids, true, {"ours": true}, {pid: pid for pid in ids},
                                       subsets={"even": subset, "odd": ~subset})
    reasons = stats.inference_gate(halves, ba, v2)
    assert len(reasons) == 12 and "ec_class_1_not_guaranteed in even" in reasons
    assert stats.table_absence_probabilities(halves, ba, v2)["odd"][5] == pytest.approx(0.5)


def test_campaign_statistics_modules_do_not_import_torch():
    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(src)!r}); "
        "import ec_clean30.stats, ec_clean30.selection, ec_clean30.logit_adjustment; print('torch' in sys.modules)"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"


def test_co_primary_population_test_turns_a_gate_or_abort_into_unavailable(monkeypatch):
    from ec_clean30 import stats

    ids, true, clusters = _population(cluster_size=2)
    table = build_cluster_count_table(ids, true, {"a": _predictions(true, 0.9, 1), "b": _predictions(true, 0.6, 2)},
                                      clusters)
    result, reason = stats.co_primary_population_test(table, [Contrast.difference("d", "a", "b")], n_draws=300)
    assert reason is None and result.engine == stats.ENGINE_AMENDMENT2 and result.lower_bound("d", 0.025) > 0
    one = build_cluster_count_table(ids[:2], true[:2], {"a": true[:2], "b": true[:2]}, {ids[0]: "g", ids[1]: "g"})
    result, reason = stats.co_primary_population_test(one, [Contrast.difference("d", "a", "b")], n_draws=50)
    assert result is None and reason.startswith("stratum gate")
    original = stats.paired_cluster_bootstrap

    def aborting(*args, **kwargs):
        raise stats.BootstrapAbort("synthetic abort")

    monkeypatch.setattr(stats, "paired_cluster_bootstrap", aborting)
    result, reason = stats.co_primary_population_test(table, [Contrast.difference("d", "a", "b")], n_draws=50)
    assert result is None and reason == "bootstrap aborted: synthetic abort"
    monkeypatch.setattr(stats, "paired_cluster_bootstrap", original)
    policy = stats.engine_policy_record_v2()
    assert policy["rule_id"] == "stratified_rao_wu_v2" and policy["historical_metrics"] == list(stats.BOOTSTRAP_METRICS)
    with pytest.raises(ValueError, match="engine 'stratified_rao_wu_v2' only"):
        paired_cluster_bootstrap(table, [Contrast.difference("d", "a", "b")], metric=stats.METRIC_EC_CLASSES_1_6_ACCURACY,
                                 n_draws=10)
