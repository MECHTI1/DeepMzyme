"""Amendment-2 fold-0 futility rule (rule A2-F): projection, decision and the write-once record (synthetic only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec_clean30 import futility, selection, selector, stats  # noqa: E402
from ec_clean30.selection import PooledOof  # noqa: E402
from ec_clean30_amendment2_fixtures import (  # noqa: E402
    FREEZE1, IDS, INNER, TABLES, TRUE, amendment2, install_family_table, sanity_stub, select, world,
    write_clean_exports,
)

FAST = {"n_replicates": 6, "n_descriptive_replicates": 3, "n_draws": 200, "metric": stats.METRIC_ACCURACY}
REAL_VERIFY_CLEAN_SANITY = futility._verify_clean_sanity


@pytest.fixture(autouse=True)
def family_table(monkeypatch):
    install_family_table(monkeypatch)


def _scorer(n_draws=200):
    return futility.claim_engine_scorer(metric=stats.METRIC_ACCURACY, n_draws=n_draws)


def _inputs(dm_correct, clean_correct, *, per_class=8, size=2):
    """Six EC classes, ``per_class`` components of ``size`` proteins each; arms right where the masks say."""
    true = np.repeat(np.arange(6), per_class * size)
    labels = [f"g{index // size}" for index in range(true.size)]
    wrong = (true + 1) % 6
    dm = np.where(dm_correct(np.arange(true.size)), true, wrong)
    clean = np.where(clean_correct(np.arange(true.size)), true, wrong)
    return futility.projection_inputs(true, dm, clean, labels)


def test_policy_record_is_the_bound_futility_block():
    assert futility.policy_record() == selection.selection_policy_record_v2()["futility"]
    record = futility.policy_record()
    assert record["rule_id"] == "futility_predictive_cluster_bootstrap_v1"
    assert record["claim_engine"]["n_draws"] == 10_000 and record["claim_engine"]["bootstrap_seed"] == 20261023
    assert record["projected_probability_threshold"] == 0.2 and record["binding"] is True


def test_decision_threshold_is_an_integer_comparison():
    assert futility.futility_decision(399, 2000) == "stop_futility"
    assert futility.futility_decision(400, 2000) == "continue"
    assert futility.futility_decision(2, 10) == "continue" and futility.futility_decision(1, 10) == "stop_futility"
    with pytest.raises(futility.FutilityInputError):
        futility.futility_decision(2001, 2000)
    with pytest.raises(futility.FutilityInputError):
        futility.futility_decision(0.3, 2000)


def test_identical_arms_never_succeed():
    inputs = _inputs(lambda i: i % 3 != 0, lambda i: i % 3 != 0)
    summary = futility.project_success(inputs, scorer=_scorer(), n_replicates=8)
    assert summary["n_successes"] == 0 and summary["projected_probability"] == 0.0


def test_a_large_effect_always_succeeds_and_a_modest_one_sometimes():
    strong = _inputs(lambda i: np.ones_like(i, dtype=bool), lambda i: i % 2 == 0)
    assert futility.project_success(strong, scorer=_scorer(), n_replicates=8)["projected_probability"] == 1.0
    modest = _inputs(lambda i: i % 10 >= 2, lambda i: (i % 10 < 2) | (i % 10 >= 5))  # 80% against 70%, discordant
    small = futility.project_success(modest, scorer=_scorer(), n_replicates=40, targets={"P1": 60, "P2": 30})
    assert 0.0 < small["projected_probability"] < 1.0


def test_components_move_as_units_and_copies_are_distinct_clusters():
    inputs = _inputs(lambda i: i % 2 == 0, lambda i: i % 3 == 0)
    population = futility.synthetic_population(inputs, [0, 0, 5])
    assert population.n_proteins == 6 and len(set(population.cluster_of.values())) == 3
    rows = inputs.rows_of_component[0]
    assert list(population.true[:2]) == list(population.true[2:4]) == list(inputs.true[rows])
    clusters = [population.cluster_of[pid] for pid in population.protein_ids]
    assert clusters[0] == clusters[1] != clusters[2] == clusters[3]
    summary = futility.project_success(inputs, scorer=_scorer(n_draws=50), n_replicates=4)
    largest = inputs.totals()["largest_component_size"]
    for name, target in (("P1", 530), ("P2", 230)):
        item = summary["per_population"][name]
        assert abs(item["realized_size_min"] - target) <= largest and abs(item["realized_size_max"] - target) <= largest
    assert futility.stratum_draw_counts(inputs, 530) == {name: 44 for name in inputs.stratum_members()}
    assert futility.stratum_draw_counts(inputs, 230) == {name: 19 for name in inputs.stratum_members()}


def test_the_claim_engine_receives_the_frozen_settings_and_unavailable_keeps_its_slot(monkeypatch):
    calls, holm = [], []
    real_bootstrap, real_holm = stats.paired_cluster_bootstrap, stats.co_primary_holm

    def recording_bootstrap(table, contrasts, **kwargs):
        calls.append(kwargs)
        return real_bootstrap(table, contrasts, **{**kwargs, "n_draws": 50})

    def recording_holm(a, b):
        holm.append((a[0], a[1] is None, b[0], b[1] is None))
        return real_holm(a, b)

    monkeypatch.setattr(stats, "METRIC_EC_CLASSES_1_6_ACCURACY", stats.METRIC_ACCURACY, raising=False)
    monkeypatch.setattr(stats, "paired_cluster_bootstrap", recording_bootstrap)
    monkeypatch.setattr(stats, "co_primary_holm", recording_holm)
    scorer = futility.claim_engine_scorer()
    inputs = _inputs(lambda i: np.ones_like(i, dtype=bool), lambda i: i % 2 == 0)
    futility.project_success(inputs, scorer=scorer, n_replicates=2)
    assert calls and all(call["n_draws"] == 10_000 and call["seed"] == 20261023 and call["stratified"] is True
                         and call["max_skip_fraction"] == 0.01 and call["metric"] == stats.METRIC_ACCURACY
                         for call in calls)
    assert holm == [("P1", False, "P2", False)] * 2

    def p2_unavailable(population):
        if population.n_proteins < 400:
            return None, "stratum gate: synthetic"
        return scorer(population)

    holm.clear()
    summary = futility.project_success(inputs, scorer=p2_unavailable, n_replicates=2)
    assert holm == [("P1", False, "P2", True)] * 2
    assert summary["per_population"]["P2"]["n_unavailable"] == 2 and summary["per_population"]["P2"]["n_outperforms"] == 0
    assert summary["n_successes"] == 2  # P1 alone, tested at 0.025


def test_the_default_scorer_refuses_without_the_amended_claim_engine(monkeypatch):
    monkeypatch.delattr(stats, "METRIC_EC_CLASSES_1_6_ACCURACY", raising=False)
    with pytest.raises(futility.FutilityInputError, match="amended claim engine"):
        futility.claim_engine_scorer()


def test_population_excludes_class_7_and_outside_closure_proteins():
    ids = ("A", "B", "C", "D", "E", "F")
    true = np.array([0, 1, 2, 6, 3, 4])
    logits = np.eye(7)[np.stack([true, true])]
    pooled = PooledOof("sel", ids, true, (42, 43), logits, inner_fold=np.array([0, 0, 1, 1, 2, 3]))
    numbers = {"A": ("1.1.1.1",), "B": ("2.1.1.1",), "C": ("3.1.1.1",), "D": ("7.1.1.1",), "E": ("4.9.9.9",),
               "F": ("5.1.1.1",)}
    configuration = selector.PooledConfiguration(pooled, (), numbers)
    labels = ("1.1.1.1", "2.1.1.1", "3.1.1.1", "7.1.1.1", "5.1.1.1", "4.1.1.1")
    exports = {}
    for inner in range(4):
        queries = tuple(pid for pid, fold in zip(ids, pooled.inner_fold) if fold == inner)
        distances = np.full((len(queries), len(labels)), 5.0)
        for row, pid in enumerate(queries):
            target = numbers[pid][0] if numbers[pid][0] in labels else "4.1.1.1"
            distances[row, labels.index(target)] = 1.0
        exports[inner] = SimpleNamespace(distances=distances, query_ids=queries, ec_labels=labels)
    inputs, totals = futility.build_projection_inputs(configuration, 0.0, exports, {pid: pid for pid in ids})
    assert totals["n_inner_validation_proteins"] == 6 and totals["n_in_r1_inner_closure"] == 5
    assert totals["n_outside_closure"] == 1 and totals["n_class_7_in_closure"] == 1
    assert inputs.n_proteins == totals["n_population"] == 4 and inputs.n_components == 4
    with pytest.raises(selector.SelectionInputError, match="query IDs must exactly equal"):
        futility.build_projection_inputs(configuration, 0.0, {**exports, 0: SimpleNamespace(
            distances=exports[0].distances[:1], query_ids=exports[0].query_ids[:1], ec_labels=labels)},
            {pid: pid for pid in ids})


def test_deepmzyme_arm_adjusts_the_seed_mean_with_the_inner_priors():
    ids = ("A", "B")
    true = np.array([0, 4])
    logits = np.zeros((2, 2, 7))
    logits[:, 0, 0] = 6.0
    logits[0, 1, 0], logits[0, 1, 4] = 3.0, 2.0
    logits[1, 1, 0], logits[1, 1, 4] = 3.0, 2.4
    priors = np.broadcast_to(np.array([0.4, 0.2, 0.2, 0.1, 0.05, 0.04, 0.01]), (2, 2, 7)).copy()
    pooled = PooledOof("sel", ids, true, (42, 43), logits, priors)
    assert list(futility.deepmzyme_predictions(pooled, 0.0)) == [0, 0]
    assert list(futility.deepmzyme_predictions(pooled, 1.0)) == [0, 4]
    priors[1, 1, 0] = 0.3
    with pytest.raises(futility.FutilityInputError, match="share their training priors"):
        futility.deepmzyme_predictions(PooledOof("sel", ids, true, (42, 43), logits, priors), 1.0)


# ---------------------------------------------------------------------------
# The record from fold-0 step records


CHECKPOINTS = {arm: {str(inner): "a" * 64 for inner in range(4)} for arm in ("r1_inner", "r2_inner")}


def _sanity(monkeypatch, tmp_path, status="passed", disposition=None, arm_disposition=None, checkpoints=None):
    """The fold-0 held-out sanity record that the S1 record binds (``sanity_stub``), verified with ``status``; the
    verifier names the checked fits (``checkpoints``) and the per-arm dispositions with their C6b consequence."""
    from ec_clean30 import clean_sanity

    path = tmp_path / "sanity" / "held_out_sanity.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"synthetic": path.name}) + "\n")
    arms = {"r1_inner": None, "r2_inner": None, **(arm_disposition or {})}
    if disposition is not None and arm_disposition is None:
        arms["r2_inner"] = disposition
    result = {"status": status, "disposition": disposition, "arm_disposition": arms,
              "consequence": clean_sanity.held_out_consequence(arms),
              "checkpoint_sha256_by_fit": CHECKPOINTS if checkpoints is None else checkpoints}
    monkeypatch.setattr(futility, "_verify_clean_sanity", lambda given, dispositions=(): dict(result))
    return path


@pytest.fixture
def fold0(monkeypatch, tmp_path):
    amendment, _ = amendment2(monkeypatch, tmp_path)
    world_ = world(tmp_path, accuracies={"F5": 0.9})
    world_["sanity"] = sanity_stub(monkeypatch, tmp_path)
    out = tmp_path / "records" / "fold0_steps.jsonl"
    out.parent.mkdir()
    select(world_, "S1", out, amendment)
    select(world_, "S2", out, amendment)
    r1 = write_clean_exports(tmp_path / "clean", "r1_inner", wrong=IDS[:12], tie=IDS[13])
    r2 = write_clean_exports(tmp_path / "clean", "r2_inner", wrong=IDS[:20])
    monkeypatch.setattr(futility, "_TEST_OVERRIDES", dict(FAST))
    monkeypatch.setattr(futility, "_TEST_ALLOW_NON_BINDING", True)
    sanity = _sanity(monkeypatch, tmp_path)
    return {"world": world_, "chain": out, "amendment": amendment, "r1": r1, "r2": r2, "sanity": sanity,
            "tmp": tmp_path}


def _run(fold0, name, **changes):
    """One futility run on its own copy of the fold-0 chain and its own campaign root (the record, its pointer file
    and the FUT step record are each written once)."""
    chain = fold0["tmp"] / "chains" / f"{name}.jsonl"
    chain.parent.mkdir(exist_ok=True)
    source = changes.pop("step_chain", fold0["chain"])
    if not chain.exists():
        chain.write_text(Path(source).read_text())
    changes.setdefault("campaign_root", fold0["tmp"] / "roots" / name)
    options = {"fold": 0, "step_chain": chain, "freeze1_sha256": FREEZE1,
               "plan_path": fold0["world"]["plan_path"], "runs_dirs": [fold0["world"]["runs"]],
               "clean_r1_inner": fold0["r1"], "clean_r2_inner": fold0["r2"], "clean_sanity_record": fold0["sanity"],
               "tables": TABLES, "out": fold0["tmp"] / "futility" / name, "provenance": {"note": "synthetic"},
               "freeze1_amendment_path": fold0["amendment"]}
    options.update(changes)
    return futility.run_futility(**options)


def test_record_is_deterministic_write_once_and_names_no_protein(fold0):
    first = _run(fold0, "a.json")
    second = _run(fold0, "b.json")
    assert first["futility_record_sha256"] == second["futility_record_sha256"]
    text = (fold0["tmp"] / "futility" / "a.json").read_text()
    assert not any(pid in text for pid in IDS)
    assert first["format"] == "deepmzyme_ec_clean30_futility_record_v1" and first["fold"] == 0
    assert first["selection_pipeline"] == "amendment2_selection_v1"
    assert first["population"]["n_population"] == len(IDS) and first["population"]["n_clean_ties_in_population"] == 1
    assert first["decision"]["n_replicates"] == 6 and first["decision"]["binding"] is False
    assert first["test_overrides"] == FAST and not futility.futility_allows_continuation(first)
    assert first["descriptive"]["conditional_power"]["step_a_resampling"] is False
    assert set(first["descriptive"]["sensitivity"]) == {"targets_divided_by_1.25", "targets_divided_by_1.5"}
    with pytest.raises(futility.FutilityInputError, match="written once|stop_futility record exists"):
        _run(fold0, "a.json")


def test_embedded_g4_block_equals_the_historical_gap_computation(fold0):
    record = _run(fold0, "g4.json")
    chain = selector._fold_records(read_chain(fold0), 0, selector.STEP_RECORD_FORMAT_V2)
    plan = selector.load_frozen_plan(fold0["world"]["plan_path"], 0)
    records = selector.read_runner_records([fold0["world"]["runs"]], fold=0)
    configuration = selector.pool_selected_v2(chain["S1"][0]["selected"], plan, records, tables=TABLES,
                                              freeze1_sha256=FREEZE1)
    r1 = selector._read_clean_arm(fold0["r1"], "r1_inner")
    r2 = selector._read_clean_arm(fold0["r2"], "r2_inner")
    expected = selector.score_gap_gate(configuration, r1, r2, after="S1")
    block = dict(record["descriptive"]["historical_rule_outcome"])
    assert block.pop("after") == "S1"
    assert selection.to_jsonable(block) == selection.to_jsonable(expected)
    assert block["gate"]["threshold_points"] == -3.0


def read_chain(fold0):
    return selection.read_step_chain(fold0["chain"], anchor_sha256=FREEZE1)


@pytest.mark.parametrize("successes, decision", [(399, "stop_futility"), (400, "continue")])
def test_record_decision_follows_the_success_count(fold0, monkeypatch, successes, decision):
    real = futility.project_success

    def counted(inputs, **kwargs):
        summary = real(inputs, **{**kwargs, "n_replicates": 1})
        if kwargs.get("stream") is None:
            summary.update(n_replicates=2000, n_successes=successes, projected_probability=successes / 2000)
        return summary

    monkeypatch.setattr(futility, "project_success", counted)
    record = _run(fold0, f"decision_{successes}.json")
    assert record["decision"]["decision"] == decision
    assert (record["decision"]["wording"] is not None) == (decision == "stop_futility")
    if decision == "stop_futility":
        wording = record["decision"]["wording"]
        assert wording.startswith("On a conservative metalloenzyme subset of the official CLEAN split30 benchmark")
        assert "stopped for futility after fold 0 under its pre-declared rule; no test protein was scored" in wording


def test_record_refusals_reuse_the_gap_gate_checks(fold0, monkeypatch):
    tmp = fold0["tmp"]
    in_sample = write_clean_exports(tmp / "in_sample", "r1_inner", metadata_changes={"n_in_sample_queries": 3})
    with pytest.raises(futility.FutilityInputError, match="in its own training table"):
        _run(fold0, "in_sample.json", clean_r1_inner=in_sample)
    other_fold = write_clean_exports(tmp / "fold1", "r1_inner", metadata_changes={"fold": 1})
    with pytest.raises(futility.FutilityInputError, match="fold-0 r1_inner validation export"):
        _run(fold0, "fold1.json", clean_r1_inner=other_fold)
    missing = write_clean_exports(tmp / "missing", "r1_inner", drop_query=IDS[0])
    with pytest.raises(futility.FutilityInputError, match="query IDs must exactly equal"):
        _run(fold0, "missing.json", clean_r1_inner=missing)
    with pytest.raises(futility.FutilityInputError, match="fold 0 only"):
        _run(fold0, "fold.json", fold=1)
    _sanity(monkeypatch, tmp, status="comparator_diagnosis_required")
    with pytest.raises(futility.FutilityInputError, match="comparator diagnosis is required"):
        _run(fold0, "sanity.json")
    _sanity(monkeypatch, tmp, status="failed", disposition="proceed_disclosed")
    assert _run(fold0, "disclosed.json")["clean_sanity"]["disposition"] == "proceed_disclosed"
    assert not (tmp / "futility" / "in_sample.json").exists()
    with pytest.raises(futility.FutilityInputError, match="belongs to amendment2_selection_v1"):
        _run(fold0, "historical.json", freeze1_amendment_path=None)


def test_the_sanity_verifier_is_required(fold0, monkeypatch):
    import ec_clean30

    monkeypatch.setattr(futility, "_verify_clean_sanity", REAL_VERIFY_CLEAN_SANITY)
    monkeypatch.setitem(sys.modules, "ec_clean30.clean_sanity", None)
    monkeypatch.delattr(ec_clean30, "clean_sanity", raising=False)
    with pytest.raises(futility.FutilityInputError, match="CLEAN sanity verifier"):
        _run(fold0, "no_verifier.json")


def _parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action")
    futility.add_futility_arguments(sub.add_parser("futility")).set_defaults(handler=futility.action_futility)
    return parser


def _argv(fold0, out):
    return ["futility", "--fold", "0", "--step-chain", str(fold0["chain"]), "--plan", str(fold0["world"]["plan_path"]),
            "--runs", str(fold0["world"]["runs"]), "--components", "o/identity/fold0", "--freeze1-sha256", FREEZE1,
            "--freeze1-amendment", str(fold0["amendment"]),
            *sum((["--clean-r1-inner", str(path)] for path in fold0["r1"]), []),
            *sum((["--clean-r2-inner", str(path)] for path in fold0["r2"]), []),
            "--clean-sanity-record", str(fold0["sanity"]), "--out", str(out)]


def test_action_prints_totals_only_and_refuses_with_exit_2(fold0, monkeypatch, capsys):
    monkeypatch.setattr(selector, "load_inner_tables", lambda *args, **kwargs: TABLES)
    monkeypatch.setattr(futility, "real_action_git_state", lambda smoke: {"note": "synthetic"})
    monkeypatch.setattr(futility, "provenance_record", lambda **kwargs: {"note": "synthetic"})
    args = _parser().parse_args(_argv(fold0, fold0["tmp"] / "action" / "futility.json"))
    assert args.handler(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"decision", "binding", "n_successes", "n_replicates", "projected_probability", "totals",
                            "futility_record_sha256"}
    assert set(printed["totals"]) == set(futility.PRINTED_TOTALS)
    assert not any(pid in json.dumps(printed) for pid in IDS)
    assert args.handler(args) == 2
    error = capsys.readouterr().err
    assert "written once" in error or "stop_futility record exists" in error


@pytest.mark.parametrize("flag", ["--out", "--step-chain", "--clean-sanity-record"])
def test_action_refuses_test_named_paths_before_reading(fold0, monkeypatch, flag):
    from ec_clean30.provenance import HeldOutInputError

    def refuse(*args, **kwargs):
        raise AssertionError("read before the path refusal")

    monkeypatch.setattr(selector, "load_inner_tables", refuse)
    monkeypatch.setattr(futility, "run_futility", refuse)
    argv = _argv(fold0, fold0["tmp"] / "action" / "futility.json")
    argv[argv.index(flag) + 1] = str(fold0["tmp"] / "held_test")
    args = _parser().parse_args(argv)
    with pytest.raises(HeldOutInputError):
        args.handler(args)


def test_selected_configuration_must_match_the_s2_record(fold0, monkeypatch):
    chain = read_chain(fold0)
    payload = dict(chain[1]["payload"], s1_winner_predictions_sha256="0" * 64)
    other = fold0["tmp"] / "records" / "other_steps.jsonl"
    selection.append_step_record(other, "S1", chain[0]["payload"], anchor_sha256=FREEZE1)
    record = selection.read_step_chain(other, anchor_sha256=FREEZE1)[0]
    payload["s1_record_sha256"] = selection.record_sha256(record)
    selection.append_step_record(other, "S2", payload, anchor_sha256=FREEZE1)
    with pytest.raises(futility.FutilityInputError, match="differ from the ones the S2 record screened"):
        _run(fold0, "mismatch.json", step_chain=other)


def test_true_classes_of_the_fixture_are_ec_classes_1_6():
    assert set(TRUE.values()) == set(range(6))


# ---------------------------------------------------------------------------
# Amendment-2 review fixes: nested populations, targets, the identical-sequence exclusion and precedence


def test_nested_b1_nh30_copies_are_a_subset_of_the_b1_all_copies(monkeypatch):
    inputs = _inputs(lambda rows: rows % 3 != 0, lambda rows: rows % 2 == 0, per_class=10)
    drawn = []
    real = futility.synthetic_population

    def record(inputs_, components):
        drawn.append(list(int(value) for value in components))
        return real(inputs_, components)

    monkeypatch.setattr(futility, "synthetic_population", record)
    futility.project_success(inputs, scorer=lambda population: (None, "skipped"), n_replicates=3)
    assert len(drawn) == 6
    for p1, p2 in zip(drawn[0::2], drawn[1::2]):
        assert len(p2) < len(p1) and all(p2.count(value) <= p1.count(value) for value in set(p2))
    counts = futility.stratum_draw_counts(inputs, 526)
    nested = futility.nested_draw_counts(counts, 526, 226)
    assert all(1 <= nested[name] <= counts[name] for name in counts)
    assert nested == {name: max(1, int(np.floor(count * 226 / 526 + 0.5))) for name, count in counts.items()}
    drawn.clear()
    summary = futility.project_success(inputs, scorer=lambda population: (None, "skipped"), n_replicates=2,
                                       nested=False)
    assert summary["nested_populations"] is False


def test_targets_are_proteins_of_ec_classes_1_6():
    assert selection.FUTILITY_TARGET_SIZES == {"P1": 526, "P2": 226}
    policy = futility.policy_record()
    assert policy["nested_populations"] is True and "step_c" in policy and "independent" in " ".join(
        policy["descriptive"]["items"])


def test_proteins_identical_to_a_kept_r1_inner_training_protein_leave_u():
    inner_fold_of = {"V1": 0, "V2": 0, "V3": 1}
    sequences = {"V1": "s1", "V2": "s2", "V3": "s3", "K1": "s1", "X2": "s2", "K3": "s9"}
    lists = {0: {"V1", "V2", "X2"}, 1: {"V3"}}
    # V1's sequence is shared with K1, which R1-inner keeps; V2's copy X2 is excluded by R1-inner.
    assert futility.kept_identical_sequence_ids(inner_fold_of, sequences, lists) == {"V1"}
    with pytest.raises(futility.FutilityInputError, match="R1-inner list per inner fold"):
        futility.kept_identical_sequence_ids(inner_fold_of, sequences, {0: lists[0]})


def test_a_record_without_the_identical_sequence_exclusion_is_never_binding(fold0):
    record = _run(fold0, "no_exclusion.json")
    assert record["population"]["identical_sequence_exclusion_applied"] is False
    assert record["decision"]["binding"] is False


def test_the_identical_sequence_exclusion_is_applied_and_counted(fold0):
    root = fold0["tmp"] / "identity_inputs"
    root.mkdir()
    corpus = root / "corpus.tsv"
    keep = IDS[0]
    rows = ["accession\tsequence_sha256\tec_class\texclude_inner0\texclude_inner1\texclude_inner2\texclude_inner3"]
    rows += [f"{pid}\t{index:064x}\t1\t0\t0\t0\t0" for index, pid in enumerate(IDS)]
    rows.append(f"KEPT1\t{0:064x}\t1\t0\t0\t0\t0")  # same sequence as IDS[0], kept by R1-inner
    corpus.write_text("\n".join(rows) + "\n")
    lists = []
    for inner in range(4):
        path = root / f"r1_inner_exclusions_fold0_inner{inner}.tsv"
        members = [pid for pid in IDS if INNER[pid] == inner]
        path.write_text("accession\treason\n" + "".join(f"{pid}\tinner_validation\n" for pid in members))
        lists.append(path)
    record = _run(fold0, "with_exclusion.json", f6_corpus=corpus, r1_inner_lists=lists)
    totals = record["population"]
    assert totals["identical_sequence_exclusion_applied"] is True
    assert totals["n_excluded_identical_to_kept_r1_inner_training"] == 1 and totals["n_population"] == len(IDS) - 1
    assert keep not in json.dumps(record)


def _exclusion_inputs(fold0):
    """The fold-0 F6 corpus table and the four R1-inner lists (rule A2-FUT-2; nothing shared with U)."""
    root = fold0["tmp"] / "identity_inputs_binding"
    if not root.exists():
        root.mkdir()
        rows = ["accession\tsequence_sha256\tec_class\texclude_inner0\texclude_inner1\texclude_inner2\texclude_inner3"]
        rows += [f"{pid}\t{index:064x}\t1\t0\t0\t0\t0" for index, pid in enumerate(IDS)]
        (root / "corpus.tsv").write_text("\n".join(rows) + "\n")
        for inner in range(4):
            members = [pid for pid in IDS if INNER[pid] == inner]
            (root / f"r1_inner_exclusions_fold0_inner{inner}.tsv").write_text(
                "accession\treason\n" + "".join(f"{pid}\tinner_validation\n" for pid in members))
    return {"f6_corpus": root / "corpus.tsv",
            "r1_inner_lists": [root / f"r1_inner_exclusions_fold0_inner{inner}.tsv" for inner in range(4)]}


def _binding(fold0, monkeypatch, successes):
    """A binding run: no override, the exclusion inputs, the frozen claim engine (one counted replicate)."""
    monkeypatch.setattr(futility, "_TEST_OVERRIDES", {})
    monkeypatch.setattr(futility, "_TEST_ALLOW_NON_BINDING", False)
    real = futility.project_success

    def counted(inputs, **kwargs):
        summary = real(inputs, **{**kwargs, "n_replicates": 1})
        if kwargs.get("stream") is None:
            summary.update(n_replicates=2000, n_successes=successes, projected_probability=successes / 2000)
        return summary

    monkeypatch.setattr(futility, "project_success", counted)
    return _exclusion_inputs(fold0)


def test_the_record_writes_its_pointer_and_fut_step_and_a_stop_is_never_rerun(fold0, monkeypatch):
    inputs = _binding(fold0, monkeypatch, 10)
    record = _run(fold0, "stop.json", campaign_root=fold0["tmp"] / "shared_root", **inputs)
    assert record["decision"]["decision"] == "stop_futility" and record["decision"]["binding"] is True
    pointer = json.loads(Path(record["pointer"]["path"]).read_text())
    assert pointer["decision"] == "stop_futility" and Path(record["pointer"]["path"]).name == "fut_seq2.json"
    chain = selection.read_step_chain(fold0["tmp"] / "chains" / "stop.json.jsonl", anchor_sha256=FREEZE1)
    assert [item["step"] for item in chain] == ["S1", "S2", "FUT"]
    assert chain[-1]["payload"]["futility_record_sha256"] == record["futility_record_sha256"]
    # A rerun on a fresh copy of S1 and S2 in the same campaign is refused: the stop takes precedence.
    with pytest.raises(futility.FutilityInputError, match="stop_futility record exists"):
        _run(fold0, "rerun.json", campaign_root=fold0["tmp"] / "shared_root", **inputs)


# ---------------------------------------------------------------------------
# Review fixes: binding gate, smoke isolation, bound CLEAN inputs, once per fit chain, C6b, isolation, C-GF, OC


def test_a_run_that_cannot_bind_is_refused_and_writes_nothing(fold0, monkeypatch):
    monkeypatch.setattr(futility, "_TEST_ALLOW_NON_BINDING", False)
    root = fold0["tmp"] / "roots" / "refused"
    before = fold0["chain"].read_bytes()
    with pytest.raises(futility.FutilityInputError, match="must be able to bind.*--f6-corpus"):
        _run(fold0, "refused.json", campaign_root=root)
    assert not (fold0["tmp"] / "futility" / "refused.json").exists() and not root.exists()
    chain = fold0["tmp"] / "chains" / "refused.json.jsonl"
    assert chain.read_bytes() == before
    monkeypatch.setattr(futility, "_TEST_OVERRIDES", {})
    with pytest.raises(futility.FutilityInputError, match="frozen settings"):
        _run(fold0, "scorer.json", campaign_root=root, scorer=_scorer(), **_exclusion_inputs(fold0))
    assert not root.exists()


def test_a_smoke_run_needs_its_own_root_and_never_writes_a_pointer_or_fut_step(fold0, monkeypatch):
    monkeypatch.setattr(futility, "_TEST_ALLOW_NON_BINDING", False)
    plan_root = Path(fold0["world"]["plan_path"]).resolve().parent.parent
    with pytest.raises(futility.FutilityInputError, match="own --campaign-root"):
        _run(fold0, "smoke_plan_root.json", smoke=True, campaign_root=plan_root)
    with pytest.raises(futility.FutilityInputError, match="own --campaign-root"):
        futility.run_futility(fold=0, step_chain=fold0["chain"], freeze1_sha256=FREEZE1,
                              plan_path=fold0["world"]["plan_path"], runs_dirs=[fold0["world"]["runs"]],
                              clean_r1_inner=fold0["r1"], clean_r2_inner=fold0["r2"],
                              clean_sanity_record=fold0["sanity"], tables=TABLES,
                              out=fold0["tmp"] / "futility" / "smoke_default.json", provenance={"note": "synthetic"},
                              freeze1_amendment_path=fold0["amendment"], smoke=True)
    monkeypatch.setattr(selector, "_identity_problems", lambda *args, **kwargs: [])  # smoke reads the real fits
    real_root = fold0["tmp"] / "roots" / "real"
    existing = futility.pointer_path(real_root, 2)
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps({"format": futility.FUTILITY_POINTER_FORMAT, "decision": "continue",
                                    "futility_record_sha256": "f" * 64, "binding": True, "fit_chain": ["x"]}))
    with pytest.raises(futility.FutilityInputError, match="holds futility pointer files"):
        _run(fold0, "smoke_real_root.json", smoke=True, campaign_root=real_root)
    smoke_root = fold0["tmp"] / "roots" / "smoke"
    record = _run(fold0, "smoke.json", smoke=True, campaign_root=smoke_root)
    assert record["smoke"] is True and record["decision"]["binding"] is False and record["pointer"] is None
    assert not smoke_root.exists()
    chain = selection.read_step_chain(fold0["tmp"] / "chains" / "smoke.json.jsonl", anchor_sha256=FREEZE1)
    assert [item["step"] for item in chain] == ["S1", "S2"]


def test_the_clean_inputs_must_be_the_fits_the_held_out_record_checked(fold0, monkeypatch):
    other = {arm: dict(values) for arm, values in CHECKPOINTS.items()}
    other["r2_inner"]["3"] = "e" * 64
    _sanity(monkeypatch, fold0["tmp"], checkpoints=other)
    with pytest.raises(futility.FutilityInputError, match="not the fits the held-out sanity record checked"):
        _run(fold0, "other_fit.json")
    _sanity(monkeypatch, fold0["tmp"], checkpoints={"r1_inner": CHECKPOINTS["r1_inner"]})
    with pytest.raises(futility.FutilityInputError, match="not the fits the held-out sanity record checked"):
        _run(fold0, "missing_arm.json")
    named = fold0["tmp"] / "sanity" / "another.json"
    named.write_text(json.dumps({"synthetic": "another"}) + "\n")
    _sanity(monkeypatch, fold0["tmp"])
    with pytest.raises(futility.FutilityInputError, match="differs from the one the fold-0 S1 record binds"):
        _run(fold0, "another.json", clean_sanity_record=named)


def test_futility_runs_once_per_fit_chain(fold0, monkeypatch):
    inputs = _binding(fold0, monkeypatch, 1999)
    root = fold0["tmp"] / "roots" / "once"
    first = _run(fold0, "first.json", campaign_root=root, **inputs)
    assert first["decision"]["decision"] == "continue" and first["pointer"] is not None
    assert futility.verify_futility_decision(fold0["tmp"] / "chains" / "first.json.jsonl", freeze1_sha256=FREEZE1,
                                             amendment_tip=fold0["amendment"], campaign_root=root)["decision"] == \
        "continue"
    with pytest.raises(futility.FutilityInputError, match="once per fit chain"):
        _run(fold0, "second.json", campaign_root=root, **inputs)


def test_a_disclosed_r1_inner_held_out_arm_stops_with_zero_successes(fold0, monkeypatch):
    _sanity(monkeypatch, fold0["tmp"], status="comparator_diagnosis_required", disposition="proceed_disclosed",
            arm_disposition={"r1_inner": "proceed_disclosed"})

    def never(*args, **kwargs):
        if kwargs.get("stream") is None and kwargs.get("resample_components", True) and kwargs.get("divisor", 1) == 1 \
                and kwargs.get("nested", True):
            raise AssertionError("the decision projection must not run")
        return REAL_PROJECT(*args, **kwargs)

    monkeypatch.setattr(futility, "project_success", never)
    record = _run(fold0, "r1_disclosed.json")
    assert record["decision"]["decision"] == "stop_futility" and record["decision"]["n_successes"] == 0
    assert record["projection"]["projection_run"] is False
    assert record["clean_sanity"]["consequence"]["co_primaries_available"] is False
    # An R2-inner disposition leaves the co-primaries and the projection in place.
    monkeypatch.setattr(futility, "project_success", REAL_PROJECT)
    _sanity(monkeypatch, fold0["tmp"], status="comparator_diagnosis_required", disposition="proceed_disclosed",
            arm_disposition={"r2_inner": "proceed_disclosed"})
    assert _run(fold0, "r2_disclosed.json")["projection"]["n_replicates"] == FAST["n_replicates"]


REAL_PROJECT = futility.project_success


def test_the_futility_rule_loads_no_post_fit_analysis_file_at_run_time(fold0, monkeypatch):
    import importlib.abc
    import ec_clean30

    blocked = ("ec_clean30.scoring", "ec_clean30.baseline", "score_ec_clean30_comparison", "run_ec_clean30_baseline")
    attempts = []

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name in blocked:
                attempts.append(name)
                raise ImportError(f"{name} is a post-fit analysis file")
            return None

    for name in blocked:
        monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.delattr(ec_clean30, name.rsplit(".", 1)[-1], raising=False)
    monkeypatch.setattr(sys, "meta_path", [Block(), *sys.meta_path])
    record = _run(fold0, "isolated.json")
    assert record["projection"]["n_replicates"] == FAST["n_replicates"] and not attempts
    assert not any(name in sys.modules for name in blocked)


def test_project_success_in_a_fresh_interpreter_loads_no_post_fit_analysis_file():
    import subprocess

    code = (f"import sys; sys.path[:0] = [{str(ROOT / 'src')!r}]; import numpy as np; "
            "from ec_clean30 import futility, stats; "
            "true = np.repeat(np.arange(6), 8); labels = [f'g{i // 2}' for i in range(true.size)]; "
            "u = futility.projection_inputs(true, true, (true + 1) % 6, labels); "
            "futility.project_success(u, scorer=futility.claim_engine_scorer(n_draws=50), n_replicates=2); "
            "print([name for name in ('ec_clean30.scoring', 'ec_clean30.baseline', 'score_ec_clean30_comparison') "
            "if name in sys.modules])")
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "[]"


def test_cgf_is_reported_beside_clean_on_u(fold0):
    record = _run(fold0, "cgf.json")
    cgf = record["descriptive"]["cgf_beside_clean"]
    assert cgf["control"] == "C-GF" and cgf["matches_s1_record"] is True
    assert cgf["n_population"] == record["population"]["n_population"]
    assert cgf["cgf_minus_clean_points"] == pytest.approx(cgf["cgf_points"] - cgf["clean_r1_inner_points"])
    assert not any(pid in json.dumps(cgf) for pid in IDS)


def test_operating_characteristics_are_deterministic_and_monotone_in_the_margin(tmp_path):
    scorer = futility.claim_engine_scorer(n_draws=200)
    first = futility.operating_characteristics(n_replicates=4, margins=(0, 8), scorer=scorer)
    second = futility.operating_characteristics(n_replicates=4, margins=(0, 8), scorer=scorer)
    assert first == second and first["synthetic"] is True and first["decides"] is False
    zero, eight = first["rows"]
    assert zero["n_population"] == sum(futility.OC_DESIGN["proteins_per_ec_class_1_6"])
    assert eight["projected_probability"] >= zero["projected_probability"]
    assert futility.OC_MARGINS_POINTS == (0, 2, 4, 6, 8) and futility.OC_REPLICATES == 200
    u0, u8 = futility.synthetic_oc_inputs(0), futility.synthetic_oc_inputs(8)
    gain = np.mean(u8.dm_predicted == u8.true) - np.mean(u0.dm_predicted == u0.true)
    assert gain * 100 == pytest.approx(8, abs=0.25)
    out = tmp_path / "oc" / "futility_oc.json"
    out.parent.mkdir()
    written = futility.operating_characteristics(out=out, n_replicates=2, margins=(0,), scorer=scorer,
                                                 provenance={"note": "synthetic"})
    assert json.loads(out.read_text())["rows"] == selection.to_jsonable(written["rows"])
    with pytest.raises(FileExistsError):
        futility.operating_characteristics(out=out, n_replicates=2, margins=(0,), scorer=scorer,
                                           provenance={"note": "synthetic"})
