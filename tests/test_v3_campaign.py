"""Tests for the v3 campaign runner (pmm_v3_campaign.py, run_pmm_v3_campaign.py)."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import pmm_v3_campaign as v3  # noqa: E402
from benchmarking import pmm_ion_features as features  # noqa: E402
from training.source_cohort import sha256_file  # noqa: E402


# ---------------------------------------------------------------------------
# Recipes and step plans
# ---------------------------------------------------------------------------

def test_step_unit_counts_match_the_plan():
    assert len(v3.step_units("C")) == 12  # plus the separate v2-fold regression run
    assert len(v3.step_units("D-A")) == 16
    assert len(v3.step_units("D-B")) == 18
    assert len(v3.step_units("E-neutral")) == 36
    improvement = v3.step_units("E-improvement", final_recipes={"only_gvp": "meanagg",
                                                                 "gvp_late_fusion": "combo-meanagg+structlr"})
    assert len(improvement) == 8 and {u.fold for u in improvement} == {1, 2, 3, 4}


def test_unit_names_round_trip_and_reject_invalid_cells():
    unit = v3.Unit("gvp_late_fusion", "four_class", "structlr", 0, 43)
    assert v3.Unit.parse(unit.name) == unit
    for bad in [("only_esm", "four_class", "meanagg", 0, 42),      # GVP-only recipe
                ("only_gvp", "four_class", "structlr", 0, 42),     # fusion-only recipe
                ("only_esm", "six_class", "v2recipe", 0, 42),      # v2 recipe is four-class only
                ("only_esm", "four_class", "baseline", 5, 42),     # fold out of range
                ("only_esm", "four_class", "baseline", 0, 44)]:    # undeclared seed
        with pytest.raises(ValueError):
            v3.Unit(*bad)


def test_combination_recipes_follow_the_plan_rule():
    combo = v3.resolve_recipe("combo-meanagg+resdrop01+structlr")
    assert combo["families"] == ("gvp_late_fusion",)
    assert combo["flags"] == ["--gvp-normalize-message-aggregation", "--gvp-residual-dropout", "0.1",
                              "--gvp-lr-scope", "structural"]
    assert combo["checkpoint_rule"] == "terminal" and combo["lr_schedule"] == "cosine"
    for bad in ("combo-gvpaux03+esmdrop02", "combo-posnoise01+outerdrop01", "combo-structlr+meanagg",
                "combo-meanagg", "combo-baseline+meanagg", "combo-meanagg+unknown"):
        with pytest.raises(ValueError):
            v3.resolve_recipe(bad)


# ---------------------------------------------------------------------------
# Synthetic campaign (training side only)
# ---------------------------------------------------------------------------

def certify_fake_inventory(v2_paths, esm_dir: Path) -> None:
    with (v2_paths.root / "esm_generation_plan.csv").open(encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    records = []
    for row in plan:
        name = f"{Path(row['structure_name']).stem}_chain_{row['chain']}_esmc.pt"
        path = esm_dir / name
        records.append({"path": name, "sequence_sha256": row["sequence_sha256"], "sha256": sha256_file(path),
                        "sidecar_sha256": sha256_file(path.with_name(path.name + ".json"))})
    v2_paths.feature_inventory.write_text(json.dumps({
        "schema_version": 2, "certified": True, "certified_scope": "structure_and_esmc600m",
        "cohort_sha256": sha256_file(v2_paths.cohort), "certified_at": "t0", "load_seconds": 1.0,
        "esm": {"embeddings_dir": str(esm_dir), "model_name": "esmc_600m", "embedding_dim": 8,
                "files": records, "n_files": len(records),
                "plan_csv_sha256": sha256_file(v2_paths.root / "esm_generation_plan.csv")}}))


def write_fold_set(v2_paths, fold_dir: Path) -> None:
    fold_dir.mkdir(parents=True)
    with v2_paths.fold_membership.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = fold_dir / "fold_membership.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_uid", "physical_ion_id", "group_id",
                                                    "native_element", "fold", "pdbid"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "group_id": f"v3g_{row['group_id']}", "pdbid": row["group_id"]})
    (fold_dir / "fold_receipt.json").write_text(json.dumps({
        "accepted": True, "builder_version": "test", "outputs_sha256": {"fold_membership.csv": sha256_file(out)},
        "inputs": {"cohort_sha256": sha256_file(v2_paths.cohort)}}))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    from pmm_campaign_fixtures import write_fake_embeddings
    from test_pmm_training_integrity import make_campaign
    from benchmarking import pmm_ion_campaign as v2

    train, v2_paths = make_campaign(tmp_path, monkeypatch, n_pdb=30)
    v2.freeze_folds(v2_paths)
    monkeypatch.setattr(features, "ESM_DIM", 8)
    features.plan_embeddings(v2_paths, train)
    esm_dir = tmp_path / "esm"
    write_fake_embeddings(train, esm_dir, dim=8)
    certify_fake_inventory(v2_paths, esm_dir)
    if not (v2_paths.root / "train_context_audit.json").exists():
        (v2_paths.root / "train_context_audit.json").write_text("{}")
    monkeypatch.setitem(v3.PROFILE, "esm_dim", 8)
    fold_dir = tmp_path / "folds" / "v3-test"
    write_fold_set(v2_paths, fold_dir)
    paths = v3.V3Paths(tmp_path / "v3")
    v3.prepare_campaign(paths.root, v2_root=v2_paths.root, fold_dir=fold_dir)
    freeze_test_spec(paths, monkeypatch, tmp_path)
    return train, esm_dir, paths, fold_dir, v2_paths


def freeze_test_spec(paths, monkeypatch, tmp_path, **identity_overrides):
    """A frozen A4 specification that describes this synthetic campaign (pinned by monkeypatch)."""
    import pmm_v3_assessment as assess

    manifest = json.loads(paths.manifest.read_text())
    identities = {"campaign_id": manifest["campaign_id"], "fold_set_id": manifest["fold_set"]["id"],
                  "fold_membership_sha256": manifest["fold_set"]["fold_membership_sha256"],
                  "fold_builder": manifest["fold_set"]["builder_version"],
                  "cohort_sha256": manifest["cohort"]["sha256"], "cohort_rows": manifest["cohort"]["n_rows"],
                  "baseline_profile_sha256": manifest["profile_sha256"],
                  "baseline_recipe": manifest["recipes"]["baseline"], **identity_overrides}
    spec_path = tmp_path / "assessment_spec.json"
    spec_path.write_text(json.dumps({**assess.SPEC_DEFAULTS, "identities": identities}, sort_keys=True))
    monkeypatch.setattr(assess, "SPEC_PATH", spec_path)
    monkeypatch.setattr(assess, "FROZEN_SPEC_SHA256", sha256_file(spec_path))
    return spec_path


def test_prepare_freezes_inputs_and_never_reprepares(campaign):
    _, _, paths, fold_dir, v2_paths = campaign
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["campaign_id"] == v3.CAMPAIGN_ID and manifest["fold_set"]["id"] == "v3-test"
    assert manifest["frozen_source_tree_sha256"] and manifest["held_out_access"] is False
    weights = json.loads(paths.fold_class_weights.read_text())
    assert weights["fold_membership_sha256"] == sha256_file(paths.fold_membership)
    with pytest.raises(ValueError, match="never re-prepared"):
        v3.prepare_campaign(paths.root, v2_root=v2_paths.root, fold_dir=fold_dir)


def test_verify_campaign_detects_any_frozen_change(campaign):
    _, esm_dir, paths, _, _ = campaign
    v3.verify_campaign(paths, esm_dir=esm_dir)
    payload = sorted(esm_dir.glob("*_esmc.pt"))[0]
    payload.write_bytes(payload.read_bytes() + b"x")
    with pytest.raises(ValueError, match="ESMC payload changed"):
        v3.verify_campaign(paths, esm_dir=esm_dir)


def test_commands_bind_identity_recipe_and_class_weights(campaign):
    train, esm_dir, paths, _, _ = campaign
    common = dict(python_bin=sys.executable, train_dir=train, esm_dir=esm_dir, device="cpu", lane=1)
    command, env, identity = v3.build_command(paths, v3.Unit("gvp_late_fusion", "five_class", "baseline", 2, 42),
                                              **common)
    joined = " ".join(command)
    assert "--fold-split-source membership" in joined and "--checkpoint-rule terminal" in joined
    assert "--lr-schedule cosine" in joined and "--metal-label-scheme five_class" in joined
    assert "--fe-loss-multiplier" in joined and "--class-viii-loss-multiplier" in joined
    assert "--run-test-eval" not in command and str(paths.lane(1) / "runs") in command
    assert identity["fold_set_id"] == "v3-test" and identity["recipe"] == "baseline"
    assert "classmodel_test_set" in env["DEEPMZYME_FORBIDDEN_READ_ROOTS"]
    v2r, _, id2 = v3.build_command(paths, v3.Unit("only_esm", "four_class", "v2recipe", 0, 42), **common)
    assert "--lr-schedule fixed" in " ".join(v2r) and "--checkpoint-rule best_validation" in " ".join(v2r)
    w, _, _ = v3.build_command(paths, v3.Unit("only_gvp", "four_class", "invsqrtw", 0, 42), **common)
    assert "inverse_sqrt_frequency" in w and "--mn-loss-multiplier" not in w
    assert id2["resolved_config_sha256"] != identity["resolved_config_sha256"]


def test_every_planned_unit_builds_a_valid_training_command(campaign):
    train, esm_dir, paths, _, _ = campaign
    units = [u for step in ("C", "D-A", "D-B", "E-neutral") for u in v3.step_units(step)]
    units += v3.step_units("D-combo", combo="combo-meanagg+resdrop01+sitecountsangles+structlr",
                           family="gvp_late_fusion")
    units += v3.step_units("D-combo", combo="combo-invsqrtw+outerdrop01+vecnorm", family="only_gvp")
    hashes = set()
    for unit in units:
        _, _, identity = v3.build_command(paths, unit, python_bin=sys.executable, train_dir=train, esm_dir=esm_dir,
                                          device="cpu", lane=0)
        assert (identity["family"], identity["recipe"], identity["fold"], identity["model_seed"]) == (
            unit.family, unit.recipe, unit.fold, unit.seed)
        hashes.add(identity["resolved_config_sha256"])
    assert len(hashes) == len(set(units))  # every unit resolves to its own configuration


def settle(paths):
    """Record a step-B execution setting (FP32, two lanes) so campaign units may run."""
    return v3.set_execution_settings(paths, amp=False, lanes=2, evidence="test")


def policy(tmp_path):
    now = time.time()
    return {"session_id": "test-session", "deadline_unix": now + 3600, "allocation_started_unix": now - 1,
            "max_total_seconds": 3600, "estimated_fit_seconds": 60, "durable_root": str(tmp_path / "durable"),
            "persistence_mode": "mounted"}


def test_one_unit_runs_replays_persists_and_is_never_retried(campaign, tmp_path):
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    unit = v3.Unit("only_esm", "six_class", "baseline", 1, 42)
    result = v3.run_unit(paths, unit, lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable,
                         device="cpu", execution_policy=policy(tmp_path), load_workers=1, epochs=2)
    assert result["status"] == "completed", result
    assert result["selected_epoch"] == 2 and result["selected_checkpoint"] == "terminal_model_checkpoint.pt"
    run_dir = paths.lane(0) / "runs" / unit.name
    # The trainer applies the v3 fold weights: Fe, Co and Ni each carry the Class VIII weight.
    w = json.loads(paths.fold_class_weights.read_text())["folds"]["1"]["common_four_weights"]
    applied = json.loads((run_dir / "run_metadata.json").read_text())["metal_class_weights"]
    assert applied == pytest.approx({"Mn": w["Mn"], "Cu": w["Cu"], "Zn": w["Zn"], "Fe": w["Class VIII"],
                                     "Co": w["Class VIII"], "Ni": w["Class VIII"]})
    assert (run_dir / "independent_validation_replay" / "replay_receipt.json").is_file()
    assert (tmp_path / "durable" / "lane0" / "runs" / unit.name / "terminal_model_checkpoint.pt").is_file()
    assert v3.completed_units(paths)[unit.name]["status"] == "completed"
    from training.campaign_runtime import replay_epoch_checkpoint

    last = replay_epoch_checkpoint(run_dir, "last_model_checkpoint.pt", output_dir=tmp_path / "last_replay")
    assert last["selected_epoch"] == 2 and last["reconciliation_status"] == "match"
    with pytest.raises(ValueError, match="never retried"):
        v3.run_unit(paths, unit, lane=1, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable,
                    device="cpu", execution_policy=policy(tmp_path), load_workers=1, epochs=2)


def test_run_refuses_a_changed_source_tree(campaign, tmp_path, monkeypatch):
    train, esm_dir, paths, _, _ = campaign
    from benchmarking import pmm_ion_campaign as v2

    monkeypatch.setattr(v2, "source_tree_sha256", lambda: "0" * 64)
    settle(paths)
    with pytest.raises(ValueError, match="frozen source tree"):
        v3.run_unit(paths, v3.Unit("only_esm", "four_class", "baseline", 1, 42), lane=0, train_dir=train,
                    esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                    execution_policy=policy(tmp_path), load_workers=1, epochs=2)


def test_regression_run_replays_and_applies_the_predeclared_gate(campaign, tmp_path, monkeypatch):
    from benchmarking import pmm_ion_campaign as v2

    train, esm_dir, paths, _, v2_paths = campaign
    monkeypatch.setattr(v2, "ESM_DIM", 8)
    result = v3.run_regression(paths, v2_root=v2_paths.root, lane=2, train_dir=train, esm_dir=esm_dir,
                               python_bin=sys.executable, device="cpu", execution_policy=policy(tmp_path),
                               load_workers=1, epochs=2)
    gate = result["gate"]
    assert gate["checks"]["all_epochs_completed"] and gate["checks"]["replay_confirmed"]
    # A two-epoch toy fit cannot match the real v2 values, so the score bands must fail.
    assert result["status"] == "failed_regression_gate" and not gate["passed"]
    run_dir = paths.lane(2) / "runs" / v3.REGRESSION_NAME
    assert (run_dir / "best_model_checkpoint.pt").is_file()
    with pytest.raises(ValueError, match="never repeated"):
        v3.run_regression(paths, v2_root=v2_paths.root, lane=0, train_dir=train, esm_dir=esm_dir,
                          python_bin=sys.executable, device="cpu", execution_policy=policy(tmp_path),
                          load_workers=1, epochs=2)


def dead_worker():
    """A process identity of a worker that has exited."""
    import subprocess as sp

    from benchmarking.pmm_execution import _process_identity

    child = sp.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
    identity = _process_identity(child.pid)
    child.wait()
    return identity


def fake_attempt(paths, name, status, *, lane=1, worker=None, identity=None):
    root = paths.lane(lane)
    (root / "runs" / name).mkdir(parents=True)
    (root / "runs" / name / "epoch_metrics.csv").write_text("epoch\n1\n")
    (root / "runs" / f"{name}.log").write_text("crashed\n")
    if status is not None:
        (root / f"run_status_{name}.json").write_text(json.dumps({"run_name": name, "status": status}))
    paths.claims.mkdir(parents=True, exist_ok=True)
    (paths.claims / f"{name}.json").write_text(json.dumps({
        "run_name": name, "lane": lane, "worker": worker or dead_worker(), "identity": identity or {"id": 1}}))


def test_failed_unit_is_archived_once_and_completed_units_never_rerun(campaign):
    _, _, paths, _, _ = campaign
    name = v3.Unit("only_gvp", "four_class", "baseline", 3, 42).name
    fake_attempt(paths, name, None)  # interrupted: artifacts but no status record
    target = v3.archive_failed_attempt(paths, name)
    assert not v3.unit_artifacts(paths, name)
    assert (target / "lanes" / "lane1" / "runs" / name / "epoch_metrics.csv").is_file()
    assert (target / "claims" / f"{name}.json").is_file()
    receipt = json.loads((target / "archive_receipt.json").read_text())
    assert receipt["status_meaning"].startswith("interrupted") and receipt["identity"] == {"id": 1}
    assert receipt["worker_stopped"]["lane_lock_free"]
    fake_attempt(paths, name, "failed")  # the single rerun fails as well
    with pytest.raises(ValueError, match="already rerun"):
        v3.archive_failed_attempt(paths, name)
    other = v3.Unit("only_gvp", "four_class", "baseline", 4, 42).name
    fake_attempt(paths, other, "completed", lane=0)
    with pytest.raises(ValueError, match="never rerun"):
        v3.archive_failed_attempt(paths, other)


def test_archiving_needs_a_stopped_worker_not_just_a_missing_status(campaign):
    import fcntl
    import os

    from benchmarking.pmm_execution import _process_identity

    _, _, paths, _, _ = campaign
    name = v3.Unit("only_gvp", "four_class", "baseline", 2, 42).name
    fake_attempt(paths, name, None, worker=_process_identity(os.getpid()))  # this process is alive
    with pytest.raises(ValueError, match="still running"):
        v3.archive_failed_attempt(paths, name)
    (paths.claims / f"{name}.json").write_text(json.dumps({"run_name": name, "lane": 1, "worker": dead_worker(),
                                                           "identity": {"id": 1}}))
    with (paths.lane(1) / "execution.lock").open("a+") as held:  # a worker still owns the lane
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="still owned"):
            v3.archive_failed_attempt(paths, name)
        fcntl.flock(held, fcntl.LOCK_UN)
    (paths.claims / f"{name}.json").unlink()  # no claim: the worker cannot be shown to have stopped
    with pytest.raises(ValueError, match="no claim"):
        v3.archive_failed_attempt(paths, name)


def test_claims_are_atomic_across_lanes(campaign, tmp_path):
    train, esm_dir, paths, _, _ = campaign
    unit = v3.Unit("only_gvp", "four_class", "baseline", 3, 43)
    first = v3.claim_run(paths, unit.name, lane=0, identity={"a": 1}, session_id="s")
    with pytest.raises(ValueError, match="already claimed"):
        v3.claim_run(paths, unit.name, lane=2, identity={"a": 1}, session_id="s")
    assert json.loads(first.read_text())["lane"] == 0
    settle(paths)
    with pytest.raises(ValueError, match="artifacts exist"):
        v3.run_unit(paths, unit, lane=1, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable,
                    device="cpu", execution_policy=policy(tmp_path), load_workers=1, epochs=2)


def test_admission_refusal_releases_the_claim(campaign, tmp_path):
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    unit = v3.Unit("only_gvp", "four_class", "baseline", 3, 42)
    tight = {**policy(tmp_path), "deadline_unix": time.time() + 60}  # less than 1.25 x 60 s + 900 s
    from benchmarking.pmm_execution import ExecutionBlocked

    with pytest.raises(ExecutionBlocked):
        v3.run_unit(paths, unit, lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable,
                    device="cpu", execution_policy=tight, load_workers=1, epochs=2)
    assert not v3.unit_artifacts(paths, unit.name)


def test_a_rerun_must_match_the_archived_identity(campaign, tmp_path):
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    unit = v3.Unit("only_esm", "four_class", "baseline", 2, 42)
    _, _, identity = v3.build_command(paths, unit, python_bin=sys.executable, train_dir=train, esm_dir=esm_dir,
                                      device="cpu", lane=0, epochs=2)
    fake_attempt(paths, unit.name, "failed", identity={**identity, "epochs": 50})
    v3.archive_failed_attempt(paths, unit.name)
    with pytest.raises(ValueError, match="differs from the archived attempt"):
        v3.run_unit(paths, unit, lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable,
                    device="cpu", execution_policy=policy(tmp_path), load_workers=1, epochs=2)
    assert not v3.unit_artifacts(paths, unit.name)  # refused before claiming


def test_runs_refuse_changed_runner_recipes_or_a_foreign_spec(campaign, tmp_path, monkeypatch):
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    unit = v3.Unit("only_gvp", "four_class", "baseline", 1, 42)
    common = dict(lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                  execution_policy=policy(tmp_path), load_workers=1, epochs=2)
    with monkeypatch.context() as patch:
        patch.setattr(v3, "runner_sha256", lambda: {"pmm_v3_campaign.py": "0" * 64})
        with pytest.raises(ValueError, match="changed since preparation"):
            v3.run_unit(paths, unit, **common)
    with monkeypatch.context() as patch:
        patch.setitem(v3.RECIPES, "meanagg", {"families": v3.GVP_FAMILIES, "flags": ["--gvp-residual-dropout", "0.2"]})
        with pytest.raises(ValueError, match="Recipe definitions changed"):
            v3.run_unit(paths, unit, **common)
    freeze_test_spec(paths, monkeypatch, tmp_path, fold_set_id="v3-other")
    with pytest.raises(ValueError, match="another campaign: \\['fold_set_id'\\]"):
        v3.run_unit(paths, unit, **common)
    import pmm_v3_assessment as assess

    monkeypatch.setattr(assess, "FROZEN_SPEC_SHA256", None)
    with pytest.raises(ValueError, match="not frozen"):
        v3.run_unit(paths, unit, **common)
    assert not v3.unit_artifacts(paths, unit.name)


def test_step_b_probes_setting_and_reuse_of_the_matching_full_run(campaign, tmp_path):
    train, esm_dir, paths, _, _ = campaign
    unit = v3.Unit("gvp_late_fusion", "four_class", "baseline", 0, 42)
    common = dict(lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                  execution_policy=policy(tmp_path), load_workers=1, epochs=2)
    with pytest.raises(ValueError, match="before campaign units"):
        v3.run_unit(paths, unit, **common)
    with pytest.raises(ValueError, match="must state AMP"):
        v3.run_unit(paths, unit, probe="ampfull", **common)
    result = v3.run_unit(paths, unit, probe="ampfull", amp=True, **common)
    probe_name = "probe-ampfull__" + unit.name
    assert result["status"] == "completed" and result["run_name"] == probe_name
    assert "--amp" in json.loads((paths.lane(0) / "commands" / f"{probe_name}.json").read_text())["argv"]
    # The probe ran with AMP, so it cannot stand for the step C unit under an FP32 setting.
    with pytest.raises(ValueError, match="does not match"):
        v3.set_execution_settings(paths, amp=False, lanes=2, evidence="b.json", reuse={unit.name: probe_name},
                                  epochs=2)
    assert not paths.execution_settings.exists()
    settings = v3.set_execution_settings(paths, amp=True, lanes=2, evidence="b.json",
                                         reuse={unit.name: probe_name}, epochs=2)
    assert settings["reuse"] == {unit.name: probe_name}
    with pytest.raises(ValueError, match="recorded once"):
        v3.set_execution_settings(paths, amp=True, lanes=2, evidence="b.json")
    with pytest.raises(ValueError, match="only before"):
        v3.run_unit(paths, unit, probe="late", amp=True, **common)
    with pytest.raises(ValueError, match="AMP differs"):
        v3.run_unit(paths, v3.Unit("only_gvp", "four_class", "baseline", 0, 42), amp=False, **common)

    import pmm_v3_assessment as assess

    record = assess.collect(paths, [unit], epochs=2)[unit.name]
    assert record["status"] == "completed" and record["run_name"] == probe_name
