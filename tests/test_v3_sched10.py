"""Tests for the v3 10-epoch schedule comparison (run_pmm_v3_sched10.py, pmm_v3_sched10.py,
pmm_v3_sched10_assessment.py; specification docs/campaigns/pmm_ion_metal_v3/sched10_assessment_spec.json).

Every test here runs on CPU without fitting a model, except the one marked synthetic fit, which runs a tiny fit of
the comparison on the synthetic campaign and is skipped unless DEEPMZYME_SYNTHETIC_FIT_TESTS=1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import pmm_v3_assessment as pa  # noqa: E402
import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_sched10 as s10  # noqa: E402
import pmm_v3_sched10_assessment as s10a  # noqa: E402
import pmm_v3_step_c as c  # noqa: E402
import pmm_v3_step_d as d  # noqa: E402
import pmm_v3_step_e2 as e2  # noqa: E402
import run_pmm_v3_sched10 as vm  # noqa: E402
from test_v3_campaign import campaign, policy, settle  # noqa: E402,F401  (fixture reuse)
from test_v3_step_c_launcher import FakeVM, session  # noqa: E402
from test_v3_step_c_launcher import vm as vm_snapshot  # noqa: E402

FUSION = "gvp_late_fusion__four_class__baseline"
RUNNER = d.local_runner_sha256()
ISTATS = {"confidence": 0.95, "bootstrap_resamples": 10000, "bootstrap_seed": 42, "recall_drop_limit": 0.03}
UNITS = [f"gvp_late_fusion__four_class__sched10__fold{k}__seed42" for k in (1, 2, 3, 4)]
CONTROLS = [f"{FUSION}__fold{k}__seed42" for k in (1, 2, 3, 4)]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Specification, names and import hygiene
# ---------------------------------------------------------------------------

def test_the_specification_is_frozen_and_names_the_selected_cells_four_folds():
    spec = s10.load_spec()
    assert spec["units"] == UNITS and spec["controls"] == CONTROLS and spec["selection_cell"] == FUSION
    assert (spec["candidate_epochs"], spec["control_epochs"], spec["seed"]) == (10, 50, 42)
    assert spec["adoption_rule"]["c_min_mean_gain"] == 0.002 and spec["adoption_rule"]["d_recall_drop_limit"] == 0.03
    assert spec["sensitivity"]["n_contrasts"] == 18 and "never a gate" in spec["sensitivity"]["role"]
    assert s10a.load_spec() == spec
    with pytest.raises(s10.LauncherError, match="differs from the frozen"):
        s10.load_spec(expected_sha256="0" * 64)


def test_only_the_four_comparison_runs_are_accepted_and_none_collides_with_a_campaign_unit():
    assert [vm.run_name(k) for k in vm.FOLDS] == UNITS
    assert [vm.fold_of(name) for name in UNITS] == [1, 2, 3, 4]
    for other in (CONTROLS[0], "gvp_late_fusion__four_class__sched10__fold0__seed42",
                  "gvp_late_fusion__four_class__sched10__fold1__seed43",
                  "probe-w1-fp32-r1__gvp_late_fusion__four_class__baseline__fold0__seed42"):
        with pytest.raises(ValueError, match="not a run of the 10-epoch comparison"):
            vm.fold_of(other)
    with pytest.raises(ValueError, match="Unknown recipe"):
        v3.Unit.parse(UNITS[0])  # never a frozen-runner unit: the frozen runner refuses these names
    assert vm.control_unit(3).name == CONTROLS[2]


def test_the_launcher_never_imports_torch():
    code = "import sys; sys.path.insert(0, %r); import pmm_v3_sched10; print('torch' in sys.modules)" % str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_no_frozen_runner_file_is_touched():
    extension = json.loads((s10.CAMPAIGN / e2.EXTENSION_COPY).read_text()) if (s10.CAMPAIGN / e2.EXTENSION_COPY).is_file() \
        else None
    if extension is None:
        pytest.skip("the Data1 extension record copy is not mounted")
    assert d.local_runner_sha256() == extension["runner_sha256"]
    assert s10.VM_RUNNER not in v3.RUNNER_FILES


# ---------------------------------------------------------------------------
# The comparison command against its 50-epoch control (synthetic campaign; no fit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fold", [1, 2, 3, 4])
def test_each_fit_differs_from_its_50_epoch_control_only_in_the_epoch_count(campaign, fold):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    common = dict(python_bin=sys.executable, train_dir=train, esm_dir=esm_dir, device="cpu", lane=1, amp=False)
    diff = vm.differences_from_control(paths, fold, **common)
    assert diff["unexpected"] == [] and diff["epochs"] == [50, 10]
    assert set(diff["config_differences"]) == {"epochs", "run_name", "campaign_run_identity"}
    assert diff["config_differences"]["run_name"] == [CONTROLS[fold - 1], UNITS[fold - 1]]
    assert set(diff["identity_differences"]) == {"epochs", "resolved_config_sha256", "schedule_extension"}
    command, _, identity = vm.build(paths, fold, **common)
    assert command[command.index("--epochs") + 1] == "10" and command[command.index("--run-name") + 1] == UNITS[fold - 1]
    assert json.loads(command[-1]) == identity and identity["epochs"] == 10 and identity["recipe"] == "baseline"
    assert identity["schedule_extension"] == {"id": "v3-ext2-sched10", "runner_file": "run_pmm_v3_sched10.py",
                                              "runner_sha256": sha(ROOT / "run_pmm_v3_sched10.py"), "epochs": 10,
                                              "control_epochs": 50}
    from training.config import config_to_payload, parse_args

    config = config_to_payload(parse_args(command[3:]))  # the trainer's own parser accepts it
    assert (config["lr_schedule"], config["checkpoint_rule"], config["seed"], config["fold_index"]) == (
        "cosine", "terminal", 42, fold)
    control, _, _ = v3.build_command(paths, vm.control_unit(fold), **common)
    reference = config_to_payload(parse_args(control[3:]))
    optimizer_keys = ("learning_rate", "gvp_learning_rate", "weight_decay", "gvp_weight_decay", "gvp_lr_scope",
                      "batch_size", "amp")
    assert {k: config.get(k) for k in optimizer_keys} == {k: reference.get(k) for k in optimizer_keys}


def test_the_ten_epoch_cosine_schedule_is_the_recorded_probe_schedule(campaign):  # noqa: F811
    import torch

    from training.config import parse_args
    from training.run import build_scheduler

    train, esm_dir, paths, _, _ = campaign
    command, _, _ = vm.build(paths, 1, python_bin=sys.executable, train_dir=train, esm_dir=esm_dir, device="cpu",
                             lane=0, amp=False)
    config = parse_args(command[3:])
    trunk, base = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([{"params": [trunk], "lr": config.gvp_learning_rate},
                                   {"params": [base], "lr": config.learning_rate}])
    scheduler = build_scheduler(optimizer, config)
    rates = []
    for _ in range(config.epochs):  # the trainer logs the epoch's rate, then steps once per epoch
        rates.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    assert len(rates) == 10 and rates[0] == [3e-4, 3e-5]
    # The seven complete step-B 10-epoch probes logged 7.341522555726971e-06 at epoch 10 (log v3-032, r5 review).
    assert math.isclose(rates[-1][0], 7.341522555726971e-06, rel_tol=1e-12)
    assert math.isclose(rates[-1][1], 7.341522555726971e-07, rel_tol=1e-12)


def test_a_run_is_refused_before_any_fit_without_extension_1_or_with_existing_artifacts(campaign, tmp_path):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    kwargs = dict(lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                  execution_policy=policy(tmp_path), load_workers=1)
    with pytest.raises(ValueError, match="extension-1 campaign root only"):
        vm.run(paths, 1, **kwargs)
    v3.extend_campaign(paths, epochs=2)
    stray = paths.lane(2) / "runs" / UNITS[0]
    stray.mkdir(parents=True)
    with pytest.raises(ValueError, match="never retried"):
        vm.run(paths, 1, **kwargs)
    assert not (paths.claims / f"{UNITS[0]}.json").exists()


@pytest.mark.skipif(os.environ.get("DEEPMZYME_SYNTHETIC_FIT_TESTS") != "1",
                    reason="runs a tiny synthetic CPU fit; opt in with DEEPMZYME_SYNTHETIC_FIT_TESTS=1")
def test_a_comparison_fit_runs_ten_epochs_replays_and_persists_synthetic_fit(campaign, tmp_path):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    v3.extend_campaign(paths, epochs=2)
    result = vm.run(paths, 2, lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                    execution_policy=policy(tmp_path), load_workers=1)
    assert result["status"] == "completed", result
    assert result["selected_epoch"] == 10 and result["selected_checkpoint"] == "terminal_model_checkpoint.pt"
    assert result["replay_selected_epoch"] == 10 and result["replay_policy_id"] == v3.REPLAY_POLICY["policy_id"]
    status = v3.completed_units(paths)[UNITS[1]]
    assert status["unit"] == UNITS[1] and status["control_unit"] == CONTROLS[1]
    identity = status["identity"]
    assert identity["epochs"] == 10 and identity["recipe"] == "baseline" and identity["fold"] == 2
    assert identity["schedule_extension"]["runner_sha256"] == sha(ROOT / "run_pmm_v3_sched10.py")
    # Ten-epoch execution with the terminal checkpoint: ten history rows, epoch 10 selected, the cosine schedule.
    run_dir = paths.lane(0) / "runs" / UNITS[1]
    saved = json.loads((run_dir / "run_config.json").read_text())
    assert len(saved["history"]) == 10 and saved["config"]["epochs"] == 10
    assert saved["config"]["lr_schedule"] == "cosine" and saved["config"]["checkpoint_rule"] == "terminal"
    receipt = v3.verify_completed_unit(run_dir, identity, v3.resolve_recipe("baseline"))
    assert receipt["selected_epoch"] == receipt["planned_epochs"] == receipt["completed_epochs"] == 10
    assert receipt["selected_checkpoint"] == "terminal_model_checkpoint.pt"
    # Independent replay of the terminal checkpoint under the v3 replay policy.
    replay = v3.verify_independent_replay(run_dir, receipt)
    assert replay["selected_epoch"] == 10 and replay["reconciliation_status"] == "match"
    # Persistence: every artifact of the fit is in the durable copy, byte for byte.
    durable = tmp_path / "durable" / "lane0"
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        copy = durable / "runs" / UNITS[1] / path.relative_to(run_dir)
        assert copy.is_file() and copy.read_bytes() == path.read_bytes(), copy
    assert (durable / f"run_status_{UNITS[1]}.json").read_bytes() == (
        paths.lane(0) / f"run_status_{UNITS[1]}.json").read_bytes()
    # The fit is never rerun automatically.
    with pytest.raises(ValueError, match="never retried"):
        vm.run(paths, 2, lane=1, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
               execution_policy=policy(tmp_path), load_workers=1)


# ---------------------------------------------------------------------------
# Manifest and launch gates (workstation records of a campaign whose E2 is complete)
# ---------------------------------------------------------------------------

def evidence_of(name: str) -> dict[str, str]:
    return {"checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "predictions_sha256": hashlib.sha256(name[::-1].encode()).hexdigest()}


def durable_run(campaign: Path, run: str, epochs: int, evidence: dict[str, str]) -> None:
    lane = campaign / "durable" / "lane0"
    (lane / "runs" / run).mkdir(parents=True)
    (lane / "runs" / run / "selected_checkpoint.json").write_text(json.dumps({
        "fit_status": "completed", "checkpoint_rule": "terminal", "reconciliation_status": "match",
        "selected_checkpoint_sha256": evidence["checkpoint_sha256"], "selected_epoch": epochs,
        "validation_predictions": {"path": "val_predictions.csv", "sha256": evidence["predictions_sha256"]}}))
    (lane / f"run_status_{run}.json").write_text(json.dumps({
        "status": "completed", "run_name": run, "selected_checkpoint_sha256": evidence["checkpoint_sha256"]}))


def fake_campaign(tmp_path: Path, monkeypatch, *, selection=FUSION, mismatch=None) -> Path:
    campaign = tmp_path / "campaign"
    ext_path = campaign / e2.EXTENSION_COPY
    ext_path.parent.mkdir(parents=True)
    ext_path.write_text(json.dumps({"extension_id": d.EXTENSION_ID, "runner_sha256": RUNNER,
                                    "parent": {"manifest_sha256": "m" * 64}}))
    spec = s10.load_spec()
    dev = spec["development_fold"]
    fold0 = f"{FUSION}__fold0__seed42"
    names = CONTROLS + [fold0]
    reused = {name: evidence_of(name) for name in names}
    if mismatch:
        reused[mismatch] = evidence_of("x" + mismatch)
    e2_manifest = campaign / s10.E2_MANIFEST
    e2_manifest.parent.mkdir(parents=True)
    e2_manifest.write_text(json.dumps({"extension": {"sha256": sha(ext_path)}, "runner_sha256": RUNNER,
                                       "reused_units": reused}))
    assessment = {"kind": "E2", "held_out_test_accessed": False, "a4_parity": {"equal": True},
                  "a4_spec_sha256": {name: sha(s10.SPEC_DIR / name) for name in s10.A4_SPEC_FILES},
                  "manifest": {"sha256": sha(e2_manifest)},
                  "unit_evidence": {name: evidence_of(name) for name in names},
                  "result": {"status": "complete", "held_out_test_accessed": False,
                             "final_selection": {"selected": selection}}}
    a_path = campaign / spec["selection_source"]["e2_assessment"]
    a_path.parent.mkdir(parents=True)
    a_path.write_text(json.dumps(assessment))
    durable_run(campaign, dev["candidate_run"], 10, evidence_of(dev["candidate_run"]))
    durable_run(campaign, dev["control_run"], 50, evidence_of(fold0))
    test_spec = {**spec, "selection_source": {**spec["selection_source"], "e2_assessment_sha256": sha(a_path)}}
    spec_path = tmp_path / "sched10_assessment_spec.json"
    spec_path.write_text(json.dumps(test_spec, indent=2, sort_keys=True))
    monkeypatch.setattr(s10, "SPEC_JSON", spec_path)
    monkeypatch.setattr(s10, "FROZEN_SPEC_SHA256", sha(spec_path))
    return campaign


def test_the_manifest_binds_the_e2_selection_controls_development_runs_and_tooling(tmp_path, monkeypatch):
    campaign = fake_campaign(tmp_path, monkeypatch)
    manifest = s10.build_manifest(campaign, log_entry="v3-099")
    assert manifest["units"] == UNITS and sorted(manifest["controls"]) == CONTROLS
    assert manifest["controls"][CONTROLS[0]]["checkpoint_sha256"] == evidence_of(CONTROLS[0])["checkpoint_sha256"]
    assert manifest["development_fold"]["candidate"]["epochs"] == 10
    assert manifest["development_fold"]["control"]["epochs"] == 50
    assert manifest["vm_runner"] == {"file": "run_pmm_v3_sched10.py", "sha256": sha(ROOT / "run_pmm_v3_sched10.py"),
                                     "remote_path": f"{d.REMOTE_CODE}/run_pmm_v3_sched10.py"}
    assert set(manifest["tooling_sha256"]) == set(s10.TOOL_FILES) and manifest["e2_assessment"]["selection"] == FUSION
    assert s10.verify_manifest(manifest) == []
    assert any("VM copy" in p for p in s10.verify_manifest(manifest, vm_runner_sha256="0" * 64))


@pytest.mark.parametrize("kwargs,message", [({"selection": "only_esm__four_class__baseline"}, "final selection"),
                                            ({"mismatch": CONTROLS[2]}, "do not bind the same verified evidence")])
def test_the_manifest_refuses_another_selection_or_mismatched_control_evidence(tmp_path, monkeypatch, kwargs, message):
    campaign = fake_campaign(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(s10.LauncherError, match=message):
        s10.build_manifest(campaign, log_entry="v3-099")


def gated(tmp_path, monkeypatch, *, authorize=True, ceiling=True):
    campaign = fake_campaign(tmp_path, monkeypatch)
    path = campaign / "audits" / "d_readiness_20261006T200000Z" / "readiness_report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"ready": True, "runner_sha256": RUNNER, "checks": {}}))
    records = s10.record_paths(campaign)
    manifest = s10.write_json_once(records["manifest"], s10.build_manifest(campaign, log_entry="v3-099"))
    if ceiling:
        s10.write_json_once(records["ceiling"], s10.ceiling_record(
            usd=55.0, spent_usd=40.06, spent_at="2026-10-12T09:02:00Z", spent_source="log v3-038", log_entry="v3-099"))
    if authorize:
        s10.write_json_once(records["authorization"], s10.authorization_record(
            manifest_path=records["manifest"], ceiling_path=records["ceiling"], log_entry="v3-100"))
    return campaign, manifest


def ready_state(manifest):
    return vm_snapshot(extension={"id": d.EXTENSION_ID, "runner_sha256": RUNNER, "sha256": manifest["extension"]["sha256"]})


def step_for(campaign, manifest, *, vm_runner=None, spend=lambda _ceiling: {"within_ceiling": True}):
    bound = manifest["vm_runner"]["sha256"]
    return s10.build_step(campaign, vm_runner=vm_runner or (lambda: bound), runner=RUNNER, spend=spend)


def test_an_authorized_verified_manifest_admits_the_four_fits_with_their_forecast(tmp_path, monkeypatch):
    campaign, manifest = gated(tmp_path, monkeypatch)
    step = step_for(campaign, manifest)
    planned = c.plan_units([(UNITS[0], 0), (UNITS[1], 1), (UNITS[2], 2)], ready_state(manifest), step=step)
    assert [item["unit"] for item in planned] == UNITS[:3] and {item["fit_seconds"] for item in planned} == {1100.0}
    assert c.launch_dir(step).endswith("/step_s10/launch") and step.code == d.REMOTE_CODE
    assert step.evidence == campaign / "step_s10_evidence" and "step_s10/**/*" in step.evidence_patterns
    fake = FakeVM(ready_state(manifest))
    out = s10.cmd_launch(fake, session(), [(UNITS[3], 0)], wait=False, step=step)
    script = fake.launched()[0]
    assert out["batch"].startswith("s10-") and f"cd {d.REMOTE_CODE}" in script
    assert f"run_pmm_v3_sched10.py --action run --campaign-dir {s10.b.REMOTE_V3} --unit {UNITS[3]}" in script
    assert "run_pmm_v3_campaign.py" not in script


@pytest.mark.parametrize("change,message", [
    ({"authorize": False}, "no recorded GPU-start authorization"),
    ({"ceiling": False, "authorize": False}, "no ceiling record"),
])
def test_no_launch_without_the_recorded_authorization_or_ceiling(tmp_path, monkeypatch, change, message):
    campaign, manifest = gated(tmp_path, monkeypatch, **change)
    with pytest.raises(s10.LauncherError, match=message):
        c.plan_units([(UNITS[0], 0)], ready_state(manifest), step=step_for(campaign, manifest))


def test_no_launch_with_a_missing_or_different_vm_runner_above_the_ceiling_or_for_another_unit(tmp_path, monkeypatch):
    campaign, manifest = gated(tmp_path, monkeypatch)
    with pytest.raises(s10.LauncherError, match="VM copy of run_pmm_v3_sched10.py"):
        c.plan_units([(UNITS[0], 0)], ready_state(manifest), step=step_for(campaign, manifest, vm_runner=lambda: None))
    above = lambda _c: {"within_ceiling": False, "committed_usd": 54.0, "reserve_usd": 6.0, "ceiling_usd": 55.0}  # noqa: E731
    with pytest.raises(s10.LauncherError, match="exceed the recorded 55.00 USD ceiling"):
        c.plan_units([(UNITS[0], 0)], ready_state(manifest), step=step_for(campaign, manifest, spend=above))
    with pytest.raises(s10.LauncherError, match="not a 10-epoch schedule comparison unit"):
        c.plan_units([(CONTROLS[0], 0)], ready_state(manifest), step=step_for(campaign, manifest))
    done = ready_state(manifest)
    done["statuses"][UNITS[0]] = {"status": "completed", "lane": 0, "family": "gvp_late_fusion",
                                  "target": "four_class", "recipe": "baseline"}
    with pytest.raises(s10.LauncherError, match="completed; a completed unit is never rerun"):
        c.plan_units([(UNITS[0], 0)], done, step=step_for(campaign, manifest))


def test_the_launch_script_is_the_step_c_script_with_only_the_runner_command_changed(tmp_path, monkeypatch):
    campaign, manifest = gated(tmp_path, monkeypatch)
    step = step_for(campaign, manifest)
    planned = [{"unit": UNITS[0], "lane": 0, "fit_seconds": 1100.0}, {"unit": UNITS[1], "lane": 1, "fit_seconds": 1100.0}]
    monkeypatch.setattr(time, "strftime", lambda fmt, t=None: "2026-10-12T00:00:00Z")
    mine = s10.launch_script("s10-x", planned, session(), step)
    monkeypatch.setattr(s10, "unit_command", c.unit_command)
    assert s10.launch_script("s10-x", planned, session(), step) == c.launch_script("s10-x", planned, session(), step)
    for item in planned:
        assert shlex_join(vm_command(item)) in mine


def vm_command(item):
    return [s10.b.REMOTE_PY, "-u", "run_pmm_v3_sched10.py", "--action", "run", "--campaign-dir", s10.b.REMOTE_V3,
            "--unit", item["unit"], "--train-dir", s10.b.REMOTE_TRAIN, "--esm-dir", s10.b.REMOTE_ESM,
            "--lane", str(item["lane"]), *s10.b.allocation_args(session(), fit_seconds=item["fit_seconds"])]


def shlex_join(command):
    import shlex

    return shlex.join(command)


def test_ceiling_and_authorization_records_name_this_comparison(tmp_path, monkeypatch):
    campaign, _ = gated(tmp_path, monkeypatch)
    records = s10.record_paths(campaign)
    ceiling = json.loads(records["ceiling"].read_text())
    assert ceiling["ceiling_usd"] == 55.0 and "10-epoch comparison" in ceiling["scope"]
    assert "authorization for this comparison" in ceiling["grants"]
    authorization = json.loads(records["authorization"].read_text())
    assert authorization["manifest_sha256"] == sha(records["manifest"]) and "10-epoch" in authorization["decision"]
    with pytest.raises(s10.LauncherError, match="recorded once"):
        s10.write_json_once(records["authorization"], authorization)


# ---------------------------------------------------------------------------
# The adoption rule on synthetic predictions
# ---------------------------------------------------------------------------

SUPPORT = (250, 60, 200, 250)


def fold_rows(fold: int, recalls) -> list[dict[str, str]]:
    rows = []
    for label, (n, recall) in enumerate(zip(SUPPORT, recalls)):
        right = round(recall * n)
        for i in range(n):
            pred = label if i < right else (label + 1) % 4
            probabilities = [0.1] * 4
            probabilities[pred] = 0.7
            rows.append({"source_uid": f"f{fold}_c{label}_{i}", "y_common4": str(label), "pred_common4": str(pred),
                         "y_native": str(label), "pred_native": str(pred),
                         **{f"p_common4_{name.replace(' ', '_')}": repr(p) for name, p in zip(pa.COMMON4, probabilities)}})
    return rows


def unit(name: str, fold: int, recalls) -> dict:
    rows = fold_rows(fold, recalls)
    return {"name": name, "run_name": name, "status": "completed", "rows": rows,
            "metrics": pa.prediction_metrics(rows, "four_class")}


def collected_for(candidate_recalls, *, control=(0.70, 0.60, 0.80, 0.65), missing=(), failed=()):
    units = {}
    for fold in (1, 2, 3, 4):
        units[CONTROLS[fold - 1]] = unit(CONTROLS[fold - 1], fold, control)
        name = UNITS[fold - 1]
        if fold in missing:
            units[name] = {"name": name, "status": "missing", "final_failure": False}
        elif fold in failed:
            units[name] = {"name": name, "status": "failed", "final_failure": True}
        else:
            units[name] = unit(name, fold, candidate_recalls[fold - 1])
    return {"units": units, "development": {"candidate": {"status": "missing", "note": "not loaded"},
                                            "control": {"status": "missing", "note": "not loaded"}}}


MEMBERSHIP = {f"f{fold}_c{label}_{i}": {"group_id": f"g{label}_{i // 25}"}
              for fold in (1, 2, 3, 4) for label, n in enumerate(SUPPORT) for i in range(n)}


def decide(candidate_recalls, **kwargs):
    return s10a.assess(collected_for(candidate_recalls, **kwargs), {}, s10.load_spec(), ISTATS, MEMBERSHIP)


def test_a_consistent_gain_without_recall_losses_adopts_the_ten_epoch_schedule():
    result = decide([(0.74, 0.65, 0.82, 0.68), (0.72, 0.65, 0.84, 0.70), (0.76, 0.6667, 0.82, 0.70),
                     (0.74, 0.65, 0.83, 0.69)])
    assert result["status"] == "complete" and result["reasons"] == []
    decision = result["schedule_decision"]
    assert decision["adopted"] is True and decision["epochs"] == 10 and decision["outcome"] == "adopted"
    assert result["comparison"]["unadjusted"]["outcome"] == "better"
    assert result["comparison"]["sensitivity_bonferroni"]["n_contrasts"] == 18
    assert set(result["comparison"]["breakdown"]["per_fold"]) == {1, 2, 3, 4}


def test_a_gain_without_interval_support_is_not_adopted():
    result = decide([(0.80, 0.70, 0.86, 0.72), (0.66, 0.55, 0.76, 0.60), (0.80, 0.70, 0.86, 0.72),
                     (0.66, 0.56, 0.77, 0.61)])
    assert result["schedule_decision"]["adopted"] is False and result["schedule_decision"]["epochs"] == 50
    assert any(reason.startswith("(b) not interval-supported") for reason in result["reasons"])


def test_a_supported_gain_below_the_tie_band_is_not_adopted():
    # +1 Mn example (0.1 point) on three folds, +2 on one: interval-supported, mean 0.125 point.
    result = decide([(0.704, 0.60, 0.80, 0.65), (0.704, 0.60, 0.80, 0.65), (0.708, 0.60, 0.80, 0.65),
                     (0.704, 0.60, 0.80, 0.65)])
    assert result["comparison"]["unadjusted"]["outcome"] == "better"
    assert [r[:3] for r in result["reasons"]] == ["(c)"] and result["schedule_decision"]["adopted"] is False


def test_a_class_recall_loss_or_a_zero_fold_recall_blocks_adoption():
    cu_loss = decide([(0.80, 0.55, 0.88, 0.75)] * 4)
    assert cu_loss["comparison"]["unadjusted"]["outcome"] == "better"
    assert any("Cu mean recall -5.00 points" in r for r in cu_loss["reasons"])
    assert cu_loss["schedule_decision"]["adopted"] is False
    zero = decide([(0.85, 0.70, 0.90, 0.80), (0.85, 0.0, 0.90, 0.80), (0.85, 0.70, 0.90, 0.80), (0.85, 0.70, 0.90, 0.80)])
    assert "(d) fold 2: Cu recall is zero" in zero["reasons"] and zero["schedule_decision"]["epochs"] == 50


def test_missing_fits_give_no_decision_and_a_second_failure_blocks_it():
    good = [(0.74, 0.65, 0.82, 0.68)] * 4
    incomplete = decide(good, missing=(3,))
    assert incomplete["status"].startswith("incomplete") and incomplete["schedule_decision"] is None
    blocked = decide(good, failed=(2,))
    assert blocked["status"].startswith("blocked") and blocked["schedule_decision"]["outcome"] == "blocked"
    assert blocked["schedule_decision"]["epochs"] == 50 and blocked["schedule_decision"]["adopted"] is False
