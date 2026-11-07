"""Amendment 2 (rule A2-R): the Freeze-1 and amendment-1 rules replay byte for byte (synthetic; no dataset access).

The golden hashes were computed from the historical functions before the amendment-2 code was added. A change
to any of them means a historical record would no longer replay.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ec_clean30 import campaign, selection, selector  # noqa: E402
from ec_clean30.selection import (  # noqa: E402
    InnerFoldPredictions, PooledOof, PromotionConfig, pool_inner_oof, promote, promotion_config_for,
    s1_choose_learning_rate, s1_family_escalation, s1_family_screen, s1_learning_rate_labels, s1_policy_record,
    s4_refit_specification, screen_logit_adjustment, sha256_canonical,
)

CAMPAIGN_DOCS = ROOT / "docs" / "campaigns" / "ec_class_vs_clean30_2026-10"

SIZES = (30, 28, 26, 20, 16, 16, 4)
IDS = tuple(f"r{cls}_{k:02d}" for cls, n in enumerate(SIZES) for k in range(n))
TRUE = np.repeat(np.arange(7), SIZES)
COMPONENTS = {pid: f"g{index // 2}" for index, pid in enumerate(IDS)}
INNER = np.array([index % 4 for index in range(len(IDS))])
FAST = PromotionConfig(n_draws=400)


def _logits(seed: int, strength: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(IDS), 7))
    values[np.arange(len(IDS)), TRUE] += strength
    return values


def _pooled(name: str, strength: float, offset: int, priors: bool = False) -> PooledOof:
    logits = np.stack([_logits(1000 * offset + seed, strength) for seed in (42, 43)])
    row_priors = None
    if priors:
        counts = np.array([30, 28, 26, 20, 12, 10, 4], dtype=np.float64)
        row_priors = np.broadcast_to((counts + 1) / (counts.sum() + 7), logits.shape).copy()
    return PooledOof(name, IDS, TRUE, (42, 43), logits, row_priors, INNER)


def _winners() -> dict[str, PooledOof]:
    strengths = {"F1": 1.2, "F2": 1.6, "F3": 1.9, "F4": 2.3, "F5": 2.0, "C-G": 1.0}
    return {family: _pooled(family, strength, index) for index, (family, strength) in enumerate(strengths.items())}


# Computed at c163995 with the historical functions on the fixtures above.
GOLDEN = {
    "s1_choose_learning_rate": "7818f7f13c3b66207c41dc2c6dc1ce73184167ec52f48f6f0acdc5545ddaa84b",
    "s1_family_escalation": "aed1d7d9df490a38636c9bd7eb2d44a318d2810c04d41ad8e39be48aee7acb75",
    "s1_family_screen": "48c62c00d3664d597775242186d75a653b1e0b1be97b8f746cf09462ebd6ddf3",
    "promote": "f11f8792e30ad2dacffc87313c7843ae47b14c413b8eb3a2652d1b14842732cf",
    "screen_logit_adjustment": "abb912bd87125d1bb68e46507453c5d5e2340a486c1c4ff9c90b2ec82e3036b2",
    "s4_refit_specification": "4b2fa3c12b6ce6a0f54f4d049d3bc5498fca18e2520aea35d5cba3448c8d75f3",
    "gap_gate_decision_s1": "606e79087317cc20dbc951720e24049392f053c673b6e750c2ad4c189bae1595",
    "gap_gate_decision_s3": "3610a79d312a54ecd2d7156349389b8c2a36ecd3b98f6ceb65b0bd11c67494ee",
    "score_gap_gate": "8b8c8f5691ad3d9fe1eb6ef4b740a045cf134f04d2dfe7ecb80ef436b2b3d500",
}


def _records() -> dict[str, str]:
    winners = _winners()
    labels = s1_learning_rate_labels("F2")
    choice = s1_choose_learning_rate("F2", {labels[0]: _pooled("low", 1.5, 11), labels[1]: _pooled("high", 1.8, 12)})
    kwargs = {"component_of": COMPONENTS, "training_class_counts": SIZES, "config": FAST}
    escalation = s1_family_escalation(winners, **kwargs)
    screen = s1_family_screen(winners, **kwargs)
    decision = promote(winners["F2"], winners["F5"], **kwargs)
    l9 = screen_logit_adjustment("F2", _pooled("F2p", 1.6, 1, priors=True), component_of=COMPONENTS,
                                 training_class_counts=SIZES, config=promotion_config_for("L9", FAST))
    spec = s4_refit_specification(family="F3", learning_rate_setting={"gvp_learning_rate": 3e-4, "learning_rate": 1e-4},
                                  adopted_levers=["L4", "L9"], tau=0.5)
    return {
        "s1_choose_learning_rate": sha256_canonical(choice.to_record()),
        "s1_family_escalation": sha256_canonical(escalation.to_record()),
        "s1_family_screen": sha256_canonical(screen.to_record()),
        "promote": sha256_canonical(decision.to_record()),
        "screen_logit_adjustment": sha256_canonical(l9.to_record()),
        "s4_refit_specification": sha256_canonical(spec),
        "gap_gate_decision_s1": sha256_canonical(selector.gap_gate_decision(70.0, 72.5, after="S1")),
        "gap_gate_decision_s3": sha256_canonical(selector.gap_gate_decision(70.0, 72.5, after="S3")),
        "score_gap_gate": sha256_canonical(_score_gap_gate()),
    }


EC_NUMBERS = {pid: (f"{cls + 1}.1.1.{index % 3 + 1}",) for index, (pid, cls) in enumerate(zip(IDS, TRUE.tolist()))}


def _clean_exports(wrong_every: int) -> dict[int, SimpleNamespace]:
    exports = {}
    labels = sorted({number for numbers in EC_NUMBERS.values() for number in numbers})
    for inner in range(4):
        queries = [pid for pid, fold in zip(IDS, INNER) if fold == inner]
        distances = np.full((len(queries), len(labels)), 5.0)
        for row, pid in enumerate(queries):
            target = EC_NUMBERS[pid][0]
            if IDS.index(pid) % wrong_every == 0:
                target = next(label for label in labels if label.split(".")[0] != target.split(".")[0])
            distances[row, labels.index(target)] = 1.0
        exports[inner] = SimpleNamespace(distances=distances, query_ids=tuple(queries), ec_labels=tuple(labels))
    return exports


def _score_gap_gate() -> dict:
    configuration = selector.PooledConfiguration(_pooled("F2", 1.6, 1), (), EC_NUMBERS)
    return selector.score_gap_gate(configuration, _clean_exports(4), _clean_exports(3), after="S1")


def test_historical_selection_records_replay_byte_for_byte():
    records = _records()
    assert records == GOLDEN


def test_freeze1_plans_and_constants_replay():
    freeze1 = json.loads((CAMPAIGN_DOCS / "freeze1.json").read_text(encoding="utf-8"))
    for fold in range(campaign.N_OUTER_FOLDS):
        assert campaign.build_fold_plan(fold)["plan_sha256"] == freeze1["plans"][str(fold)]["plan_sha256"]
    assert campaign.campaign_constants() == freeze1["constants"]["campaign"]["campaign_constants"]
    from ec_clean30 import freeze

    live = freeze.live_constants()
    assert live["selection"] == freeze1["constants"]["selection"]
    assert live["campaign"] == freeze1["constants"]["campaign"]


def test_amendment1_policy_record_replays():
    amendment = json.loads((CAMPAIGN_DOCS / "s1_amendment" / "amendment.json").read_text(encoding="utf-8"))
    assert s1_policy_record() == amendment["policy"]
    assert amendment["s1_rule"] == selection.AMENDED_S1_RULE == "fixed_anchor_holm_v1"
    assert selection.HISTORICAL_S1_RULE == "family_escalation_v1"


def test_pooling_fixture_is_well_formed():
    parts = [InnerFoldPredictions(seed, inner, tuple(pid for pid, fold in zip(IDS, INNER) if fold == inner),
                                  _logits(seed, 1.0)[INNER == inner])
             for seed in (42, 43) for inner in range(4)]
    pooled = pool_inner_oof("fixture", parts, dict(zip(IDS, TRUE.tolist())))
    assert pooled.protein_ids == tuple(sorted(IDS))
    with pytest.raises(ValueError):
        PooledOof("bad", IDS, TRUE, (42,), np.zeros((2, len(IDS), 7)))


# ---------------------------------------------------------------------------
# WP0 (reviews S-B1/F-B1, S-M1/F-M5, F-M6, F-M8): path-normalised goldens recorded at c163995 before any edit


WP0_GOLDEN = json.loads((ROOT / "tests" / "fixtures" / "ec_clean30_amendment2_golden.json").read_text())


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normaliser(roots):
    pairs = sorted(((str(key), value) for key, value in roots.items()), key=lambda item: -len(item[0]))

    def norm(obj):
        if isinstance(obj, str):
            for prefix, token in pairs:
                obj = obj.replace(prefix, token)
            return obj
        if isinstance(obj, dict):
            return {norm(key): norm(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [norm(value) for value in obj]
        return obj
    return norm


def _view_roots(tag, view):
    import hashlib

    roots = {view.manifest_sha256: f"<{tag}_VIEW_SHA>"}
    if view.allowlist_sha256:
        roots[view.allowlist_sha256] = f"<{tag}_ALLOWLIST_SHA>"
    if view.probe_sha256:
        roots[view.probe_sha256] = f"<{tag}_PROBE_SHA>"
    if view.membership:
        roots[view.membership["sha256"]] = f"<{tag}_MEMBERSHIP_SHA>"
    manifest = view.chain_means_dir("final") / "manifest.json"
    if manifest.exists():
        roots[hashlib.sha256(manifest.read_bytes()).hexdigest()] = f"<{tag}_CHAIN_SHA>"
    return roots


def _unit_records(view, runs, norm):
    from ec_clean30.commands import view_unit_commands
    from training.config import config_to_payload, parse_args

    out = {}
    for item in view_unit_commands(view, runs_dir=runs):
        if item.command is None:
            out[item.unit_id] = {"problem": norm(item.problem)}
            continue
        argv = list(item.command.argv)
        assert argv[-2] == "--campaign-run-identity"
        base = argv[:-2]
        payload = json.loads(json.dumps(config_to_payload(parse_args(base)), sort_keys=True, default=str))
        identity = dict(item.command.identity)
        identity.pop("resolved_config_sha256")
        out[item.unit_id] = {"argv_sha256": _sha(json.dumps(norm(base))),
                             "payload_sha256": _sha(json.dumps(norm(payload), sort_keys=True)),
                             "identity_sha256": _sha(json.dumps(norm(identity), sort_keys=True)),
                             "recipe": item.command.recipe.to_dict()}
    return out


@pytest.fixture(scope="module")
def replay_views(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "tests"))
    from ec_clean30_fixtures import (
        build_chain_means, build_smoke_view, unlock, write_clean30_data_root, write_inner_membership,
    )
    from ec_clean30 import views
    from ec_clean30.campaign import CAMPAIGN_ID
    from ec_clean30.views import SourceLayout, View, build_view

    base = tmp_path_factory.mktemp("wp0")
    source = write_clean30_data_root(base / "clean30_source")
    provenance = {"note": "synthetic test provenance"}
    membership, digest, _ = write_inner_membership(source, base / "identity")
    fold_root = base / "campaign" / CAMPAIGN_ID / views.VIEWS_DIR / "fold0"
    build_view(fold=0, layout=SourceLayout.from_data_root(source["data_root"]), out_dir=fold_root,
               inner_membership=(membership, digest), provenance=provenance)
    build_chain_means(fold_root)
    smoke_root = base / "smoke" / CAMPAIGN_ID / views.SMOKE_VIEWS_DIR / "view"
    smoke = build_smoke_view(source, smoke_root, provenance=provenance)
    yield {"base": base, "fold": View.load(fold_root), "smoke": smoke}
    unlock(source)


def test_v1_unit_commands_and_resolved_configurations_replay_on_the_fold_view(replay_views):
    base, fold, smoke = replay_views["base"], replay_views["fold"], replay_views["smoke"]
    runs = base / "runs"
    roots = {fold.root: "<FOLD_VIEW>", smoke.root: "<SMOKE_VIEW>", runs: "<RUNS>", base: "<TMP>",
             ROOT / "src": "<SRC>", ROOT: "<REPO>", **_view_roots("FOLD", fold), **_view_roots("SMOKE", smoke)}
    norm = _normaliser(roots)
    for name, view in (("fold_view", fold), ("smoke_view", smoke)):
        records = _unit_records(view, runs, norm)
        live = {unit_id: _sha(json.dumps(record, sort_keys=True)) for unit_id, record in sorted(records.items())}
        golden = WP0_GOLDEN[f"{name}_unit_sha256"]
        assert sorted(live) == sorted(golden), name
        differing = sorted(unit_id for unit_id in golden if live[unit_id] != golden[unit_id])
        assert not differing, (name, differing[:5])
        assert _sha(json.dumps(records, sort_keys=True)) == WP0_GOLDEN[f"{name}_commands_digest"]


def test_view_manifests_keep_their_c163995_shape(replay_views):
    from ec_clean30 import views

    fold, smoke = replay_views["fold"], replay_views["smoke"]
    assert sorted(fold.manifest) == WP0_GOLDEN["fold_view_manifest_keys"]
    assert fold.manifest["writable_dirs"] == WP0_GOLDEN["fold_view_writable_dirs"] == list(views.WRITABLE_DIRS)
    assert smoke.manifest["writable_dirs"] == WP0_GOLDEN["smoke_view_writable_dirs"]
    assert "whole_protein_pretrain" not in views.WRITABLE_DIRS and views.ATTACHED_DIRS == ("whole_protein_pretrain",)
    assert not (fold.root / "whole_protein_pretrain").exists()  # never created at build


def test_live_constants_plans_and_campaign_constants_replay_in_full():
    from ec_clean30 import freeze
    from ec_clean30.campaign import build_fold_plan, campaign_constants, canonical_json

    live = freeze.live_constants()
    groups = {key: _sha(canonical_json(value)) for key, value in sorted(live.items())}
    assert groups == WP0_GOLDEN["live_constants_groups_sha256"], sorted(
        key for key in groups if groups[key] != WP0_GOLDEN["live_constants_groups_sha256"].get(key))
    assert _sha(canonical_json(live)) == WP0_GOLDEN["live_constants_sha256"]
    assert _sha(canonical_json(campaign_constants())) == WP0_GOLDEN["campaign_constants_sha256"]
    assert {str(fold): build_fold_plan(fold)["plan_sha256"] for fold in range(5)} == WP0_GOLDEN["plan_sha256"]
    from ec_clean30 import amendment_rules

    assert amendment_rules.live_constants_sha256() == WP0_GOLDEN["live_constants_sha256"]


def test_v2_plan_commands_on_the_fold_view(replay_views):
    # Plan version 2 (rule A2-GOV-4): raw-distance units render --rbf-use-raw-distances, F6 needs the attached corpus.
    from ec_clean30.commands import view_unit_commands

    fold = replay_views["fold"]
    planned = view_unit_commands(fold, runs_dir=replay_views["base"] / "runs_v2", plan_version=2)
    s1 = [item for item in planned if item.unit_id.split("__")[2] == "S1"]
    s4 = [item for item in planned if item.template_of]
    assert len(s1) == 112 and len(s4) == 25 and len(planned) == 137
    problems = {item.unit_id: item.problem for item in planned if item.problem}
    assert {item.family for item in planned if item.problem} == {"F6"} and len(problems) == 16 + 5
    assert all("whole_protein_pretrain" in problem for problem in problems.values())
    for item in planned:
        if item.command is None:
            continue
        raw = "--rbf-use-raw-distances" in item.command.argv
        assert raw == (item.family in ("F1", "F3", "F4")), item.unit_id
        assert raw == ("_rbfraw" in item.unit_id or bool(item.template_of and item.family in ("F3", "F4")))
    with pytest.raises(Exception, match="plan_version"):
        view_unit_commands(fold, runs_dir=replay_views["base"] / "runs_v2", plan_version=3)


def test_v2_dry_run_commands_build_against_the_declared_pretraining_input_stub(replay_views):
    # Rule A2-GOV-4: the repeated dry runs build every F6 command against the declared stub, so no static failure
    # remains before the corpus exists; a stub command never runs and never verifies.
    from ec_clean30 import commands
    from ec_clean30.pretrain_corpus import PRETRAIN_STUB_SHA256, PRETRAIN_STUB_SPEC, stub_pretrain_inputs

    fold = replay_views["fold"]
    planned = commands.view_unit_commands(fold, runs_dir=replay_views["base"] / "runs_v2", plan_version=2,
                                          pretrain_stub=True)
    assert len(planned) == 137 and not [item.problem for item in planned if item.problem]
    f6 = [item.command for item in planned if item.family == "F6"]
    assert len(f6) == 21
    stub = stub_pretrain_inputs(fold)
    for command in f6:
        assert command.identity["wp_pretrain_input_stub_sha256"] == PRETRAIN_STUB_SHA256
        assert command.identity["wp_pretrain_corpus_sha256"] == stub.corpus_sha256
        assert "--global-esm-pretrain-corpus-sha256" in command.argv
    others = [item.command for item in planned if item.family != "F6"]
    assert not any("wp_pretrain_input_stub_sha256" in command.identity for command in others)
    assert PRETRAIN_STUB_SPEC["use"].startswith("amendment-2 dry runs only")
    with pytest.raises(commands.CommandError, match="Refusing to run"):
        commands.run_unit(f6[0], view=fold, cap_prefix=())
    with pytest.raises(commands.CommandError, match="plan-v2 dry runs only"):
        commands.view_unit_commands(fold, runs_dir=replay_views["base"] / "runs_v1", pretrain_stub=True)
    with pytest.raises(commands.CommandError, match="write=False"):
        commands.dry_run(fold, plan_version=2, pretrain_stub=True, write=True)
    from ec_clean30 import amendment2

    with pytest.raises(amendment2.AmendmentRecordError, match="outside the view"):
        amendment2.dry_run_v2(fold.root, Path(fold.root) / "receipts" / "dry_run_v2.json")
    # The real builder keeps refusing F6 without the attached corpus.
    real = commands.view_unit_commands(fold, runs_dir=replay_views["base"] / "runs_v2", plan_version=2)
    assert {item.family for item in real if item.problem} == {"F6"}
