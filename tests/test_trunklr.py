"""Tests for the trunk-learning-rate comparison (run_pmm_trunklr.py, pmm_trunklr.py, pmm_trunklr_assessment.py;
specification docs/campaigns/pmm_trunk_lr_2026-10/trunklr_assessment_spec.json).

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

import pmm_trunklr as tl  # noqa: E402
import pmm_trunklr_assessment as tla  # noqa: E402
import pmm_v3_assessment as pa  # noqa: E402
import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_step_c as c  # noqa: E402
import pmm_v3_step_d as d  # noqa: E402
import pmm_v3_step_e2 as e2  # noqa: E402
import run_pmm_trunklr as vm  # noqa: E402
from test_v3_campaign import campaign, policy, settle  # noqa: E402,F401  (fixture reuse)
from test_v3_step_c_launcher import FakeVM, session  # noqa: E402
from test_v3_step_c_launcher import vm as vm_snapshot  # noqa: E402

FUSION = "gvp_late_fusion__four_class__baseline"
RUNNER = d.local_runner_sha256()
ISTATS = {"confidence": 0.95, "bootstrap_resamples": 10000, "bootstrap_seed": 42, "recall_drop_limit": 0.03}
FOLDS = (0, 1, 2, 3, 4)
UNITS = [f"gvp_late_fusion__four_class__trunklr1e4__fold{k}__seed42" for k in FOLDS]
CONTROLS = [f"{FUSION}__fold{k}__seed42" for k in FOLDS]
CONTROL_GROUPS = [{"n_params": 84, "initial_lr": 3e-4, "weight_decay": 1e-4},
                  {"n_params": 31, "initial_lr": 3e-5, "weight_decay": 1e-4}]
CANDIDATE_GROUPS = [{**CONTROL_GROUPS[0], "initial_lr": 1e-4}, dict(CONTROL_GROUPS[1])]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Specification, names and import hygiene
# ---------------------------------------------------------------------------

def test_the_specification_is_frozen_and_names_the_selected_cells_five_folds():
    spec = tl.load_spec()
    assert spec["units"] == UNITS and spec["controls"] == CONTROLS and spec["selection_cell"] == FUSION
    assert (spec["candidate_gvp_learning_rate"], spec["control_gvp_learning_rate"], spec["base_learning_rate"]) == (
        "1e-4", "3e-4", "3e-5")
    assert spec["t_degrees_of_freedom"] == 4 and spec["sensitivity"]["folds"] == [1, 2, 3, 4]
    rule = spec["adoption_rule"]
    assert rule["c_min_mean_gain"] == 0.002 and rule["d_recall_drop_limit"] == 0.03
    assert rule["e_folds_1_4_mean_gain_positive"] is True
    assert tla.load_spec() == spec
    with pytest.raises(tl.LauncherError, match="differs from the frozen"):
        tl.load_spec(expected_sha256="0" * 64)


def test_only_the_five_comparison_runs_are_accepted_and_none_collides_with_a_campaign_unit():
    assert [vm.run_name(k) for k in vm.FOLDS] == UNITS
    assert [vm.fold_of(name) for name in UNITS] == list(FOLDS)
    for other in (CONTROLS[0], "gvp_late_fusion__four_class__trunklr1e4__fold5__seed42",
                  "gvp_late_fusion__four_class__trunklr1e4__fold1__seed43",
                  "gvp_late_fusion__four_class__sched10__fold1__seed42"):
        with pytest.raises(ValueError, match="not a run of the trunk-LR comparison"):
            vm.fold_of(other)
    with pytest.raises(ValueError, match="Unknown recipe"):
        v3.Unit.parse(UNITS[1])  # never a frozen-runner unit: the frozen runner refuses these names
    assert vm.control_unit(3).name == CONTROLS[3] and vm.control_gvp_lr() == "3e-4"


def test_the_launcher_never_imports_torch():
    code = "import sys; sys.path.insert(0, %r); import pmm_trunklr; print('torch' in sys.modules)" % str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_no_frozen_runner_file_is_touched():
    copy = tl.CAMPAIGN / e2.EXTENSION_COPY
    if not copy.is_file():
        pytest.skip("the Data1 extension record copy is not mounted")
    assert d.local_runner_sha256() == json.loads(copy.read_text())["runner_sha256"]
    assert tl.VM_RUNNER not in v3.RUNNER_FILES


# ---------------------------------------------------------------------------
# The comparison command against its control (synthetic campaign; no fit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fold", FOLDS)
def test_each_fit_differs_from_its_control_only_in_the_gvp_learning_rate(campaign, fold):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    common = dict(python_bin=sys.executable, train_dir=train, esm_dir=esm_dir, device="cpu", lane=1, amp=False)
    diff = vm.differences_from_control(paths, fold, **common)
    assert diff["unexpected"] == [] and vm.expected_lr_difference(diff)
    assert diff["gvp_learning_rate"] == [3e-4, 1e-4] and diff["learning_rate"] == [3e-5, 3e-5]
    assert set(diff["config_differences"]) == {"gvp_learning_rate", "run_name", "campaign_run_identity"}
    assert diff["config_differences"]["run_name"] == [CONTROLS[fold], UNITS[fold]]
    assert set(diff["identity_differences"]) == {"resolved_config_sha256", "lr_extension"}
    command, _, identity = vm.build(paths, fold, **common)
    assert command.count("--gvp-learning-rate") == 1 and command[command.index("--gvp-learning-rate") + 1] == "1e-4"
    assert command[command.index("--learning-rate") + 1] == "3e-5" and command[command.index("--epochs") + 1] == "50"
    assert json.loads(command[-1]) == identity and identity["epochs"] == 50 and identity["recipe"] == "baseline"
    assert identity["resolved_config_sha256"] == vm.resolved_config_sha256(command[:-2])
    assert identity["lr_extension"] == {"campaign_id": "pmm_trunk_lr_2026-10", "runner_file": "run_pmm_trunklr.py",
                                        "runner_sha256": sha(ROOT / "run_pmm_trunklr.py"), "gvp_learning_rate": "1e-4",
                                        "control_gvp_learning_rate": "3e-4"}
    from training.config import config_to_payload, parse_args

    config = config_to_payload(parse_args(command[3:]))  # the trainer's own parser accepts it
    control, _, _ = v3.build_command(paths, vm.control_unit(fold), **common)
    reference = config_to_payload(parse_args(control[3:]))
    unchanged = ("learning_rate", "weight_decay", "gvp_weight_decay", "gvp_lr_scope", "batch_size", "amp", "epochs",
                 "lr_schedule", "checkpoint_rule", "seed", "fold_index")
    assert {k: config.get(k) for k in unchanged} == {k: reference.get(k) for k in unchanged}
    assert config["gvp_lr_scope"] == "trunk" and config["gvp_learning_rate"] == 1e-4


def test_the_cosine_schedule_starts_the_trunk_group_at_1e4_and_keeps_the_base_group(campaign):  # noqa: F811
    import torch

    from training.config import parse_args
    from training.run import build_scheduler

    train, esm_dir, paths, _, _ = campaign
    command, _, _ = vm.build(paths, 1, python_bin=sys.executable, train_dir=train, esm_dir=esm_dir, device="cpu",
                             lane=0, amp=False)
    control, _, _ = v3.build_command(paths, vm.control_unit(1), python_bin=sys.executable, train_dir=train,
                                     esm_dir=esm_dir, device="cpu", lane=0, amp=False)

    def rates(argv):
        config = parse_args(argv[3:])
        trunk, base = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([{"params": [trunk], "lr": config.gvp_learning_rate},
                                       {"params": [base], "lr": config.learning_rate}])
        scheduler = build_scheduler(optimizer, config)
        out = []
        for _ in range(config.epochs):  # the trainer logs the epoch's rate, then steps once per epoch
            out.append([group["lr"] for group in optimizer.param_groups])
            optimizer.step()
            scheduler.step()
        return out

    mine, theirs = rates(command), rates(control)
    assert len(mine) == len(theirs) == 50 and mine[0] == [1e-4, 3e-5] and theirs[0] == [3e-4, 3e-5]
    for (trunk, base), (trunk_c, base_c) in zip(mine, theirs):  # the same cosine shape, the trunk scaled by 1/3
        assert math.isclose(base, base_c, rel_tol=1e-12) and math.isclose(trunk, trunk_c / 3, rel_tol=1e-9)


def test_a_run_is_refused_before_any_fit_without_extension_1_or_with_existing_artifacts(campaign, tmp_path):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    kwargs = dict(lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                  execution_policy=policy(tmp_path), load_workers=1)
    with pytest.raises(ValueError, match="extension-1 campaign root only"):
        vm.run(paths, 1, **kwargs)
    v3.extend_campaign(paths, epochs=2)
    stray = paths.lane(2) / "runs" / UNITS[1]
    stray.mkdir(parents=True)
    with pytest.raises(ValueError, match="never retried"):
        vm.run(paths, 1, **kwargs)
    assert not (paths.claims / f"{UNITS[1]}.json").exists()


@pytest.mark.skipif(os.environ.get("DEEPMZYME_SYNTHETIC_FIT_TESTS") != "1",
                    reason="runs a tiny synthetic CPU fit; opt in with DEEPMZYME_SYNTHETIC_FIT_TESTS=1")
def test_a_comparison_fit_runs_with_the_trunk_group_at_1e4_replays_and_persists_synthetic_fit(campaign, tmp_path):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    v3.extend_campaign(paths, epochs=2)
    result = vm.run(paths, 2, lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                    execution_policy=policy(tmp_path), load_workers=1)
    assert result["status"] == "completed", result
    epochs = int(v3.PROFILE["epochs"])
    assert result["selected_epoch"] == epochs and result["selected_checkpoint"] == "terminal_model_checkpoint.pt"
    assert result["replay_selected_epoch"] == epochs and result["replay_policy_id"] == v3.REPLAY_POLICY["policy_id"]
    status = v3.completed_units(paths)[UNITS[2]]
    assert status["unit"] == UNITS[2] and status["control_unit"] == CONTROLS[2]
    identity = status["identity"]
    assert identity["epochs"] == epochs and identity["recipe"] == "baseline" and identity["fold"] == 2
    assert identity["lr_extension"]["runner_sha256"] == sha(ROOT / "run_pmm_trunklr.py")
    run_dir = paths.lane(0) / "runs" / UNITS[2]
    saved = json.loads((run_dir / "run_config.json").read_text())
    assert len(saved["history"]) == epochs and saved["config"]["gvp_learning_rate"] == 1e-4
    assert saved["config"]["learning_rate"] == 3e-5 and saved["config"]["gvp_lr_scope"] == "trunk"
    receipt = v3.verify_completed_unit(run_dir, identity, v3.resolve_recipe("baseline"))
    assert receipt["selected_epoch"] == receipt["planned_epochs"] == receipt["completed_epochs"] == epochs
    # The trainer's real optimizer: the frozen trunk/base groups, the trunk group starting at 1e-4.
    assert tla.same_groups(tla.optimizer_groups(run_dir, receipt["selected_checkpoint"]), CANDIDATE_GROUPS)
    replay = v3.verify_independent_replay(run_dir, receipt)
    assert replay["selected_epoch"] == epochs and replay["reconciliation_status"] == "match"
    durable = tmp_path / "durable" / "lane0"
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        copy = durable / "runs" / UNITS[2] / path.relative_to(run_dir)
        assert copy.is_file() and copy.read_bytes() == path.read_bytes(), copy
    assert (durable / f"run_status_{UNITS[2]}.json").read_bytes() == (
        paths.lane(0) / f"run_status_{UNITS[2]}.json").read_bytes()
    with pytest.raises(ValueError, match="never retried"):
        vm.run(paths, 2, lane=1, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
               execution_policy=policy(tmp_path), load_workers=1)


# ---------------------------------------------------------------------------
# Manifest and launch gates (workstation records of a campaign whose A4 and E2 are complete)
# ---------------------------------------------------------------------------

def evidence_of(name: str) -> dict[str, str]:
    return {"checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "predictions_sha256": hashlib.sha256(name[::-1].encode()).hexdigest()}


A4_VALUES = [0.71, 0.73, 0.72, 0.75, 0.78]


def fake_campaign(tmp_path: Path, monkeypatch, *, selection=FUSION, mismatch=None) -> Path:
    campaign = tmp_path / "campaign"
    ext_path = campaign / e2.EXTENSION_COPY
    ext_path.parent.mkdir(parents=True)
    ext_path.write_text(json.dumps({"extension_id": d.EXTENSION_ID, "runner_sha256": RUNNER,
                                    "parent": {"manifest_sha256": "m" * 64}}))
    spec = tl.load_spec()
    reused = {name: evidence_of(name) for name in CONTROLS}
    if mismatch:
        reused[mismatch] = evidence_of("x" + mismatch)
    e2_manifest = campaign / tl.E2_MANIFEST
    e2_manifest.parent.mkdir(parents=True)
    e2_manifest.write_text(json.dumps({"extension": {"sha256": sha(ext_path)}, "runner_sha256": RUNNER,
                                       "reused_units": reused}))
    e2_assessment = {"kind": "E2", "held_out_test_accessed": False, "a4_parity": {"equal": True},
                     "a4_spec_sha256": {name: sha(tl.V3_SPEC_DIR / name) for name in tl.A4_SPEC_FILES},
                     "manifest": {"sha256": sha(e2_manifest)},
                     "unit_evidence": {name: evidence_of(name) for name in CONTROLS},
                     "result": {"status": "complete", "held_out_test_accessed": False,
                                "final_selection": {"selected": selection}}}
    a4 = {"unit_evidence": {name: evidence_of(name) for name in CONTROLS},
          "result": {"held_out_test_accessed": False,
                     "summaries": {FUSION: {"folds": list(FOLDS), "fold_common4_ba": A4_VALUES,
                                            "mean_common4_ba": sum(A4_VALUES) / 5}}}}
    paths = {}
    for key, payload in (("a4_assessment", a4), ("e2_assessment", e2_assessment)):
        path = campaign / spec["control_source"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        paths[key] = path
    source = {**spec["control_source"], "a4_assessment_sha256": sha(paths["a4_assessment"]),
              "e2_assessment_sha256": sha(paths["e2_assessment"]), "reference_mean_common4_ba": sum(A4_VALUES) / 5}
    spec_path = tmp_path / "trunklr_assessment_spec.json"
    spec_path.write_text(json.dumps({**spec, "control_source": source}, indent=2, sort_keys=True))
    monkeypatch.setattr(tl, "SPEC_JSON", spec_path)
    monkeypatch.setattr(tl, "FROZEN_SPEC_SHA256", sha(spec_path))
    return campaign


def test_the_manifest_binds_the_a4_and_e2_controls_and_tooling(tmp_path, monkeypatch):
    campaign = fake_campaign(tmp_path, monkeypatch)
    manifest = tl.build_manifest(campaign, log_entry="tlr-099")
    assert manifest["units"] == UNITS and sorted(manifest["controls"]) == CONTROLS
    assert manifest["controls"][CONTROLS[0]]["checkpoint_sha256"] == evidence_of(CONTROLS[0])["checkpoint_sha256"]
    assert [manifest["controls"][name]["a4_common4_ba"] for name in CONTROLS] == A4_VALUES
    assert manifest["vm_runner"] == {"file": "run_pmm_trunklr.py", "sha256": sha(ROOT / "run_pmm_trunklr.py"),
                                     "remote_path": f"{d.REMOTE_CODE}/run_pmm_trunklr.py"}
    assert set(manifest["tooling_sha256"]) == set(tl.TOOL_FILES) and manifest["e2_assessment"]["selection"] == FUSION
    assert manifest["campaign_id"] == "pmm_trunk_lr_2026-10" and tl.verify_manifest(manifest) == []
    assert any("VM copy" in p for p in tl.verify_manifest(manifest, vm_runner_sha256="0" * 64))
    with pytest.raises(tl.LauncherError, match="tlr-NNN"):
        tl.build_manifest(campaign, log_entry="v3-099")


@pytest.mark.parametrize("kwargs,message", [({"selection": "only_esm__four_class__baseline"}, "final selection"),
                                            ({"mismatch": CONTROLS[0]}, "do not bind the same evidence")])
def test_the_manifest_refuses_another_selection_or_mismatched_control_evidence(tmp_path, monkeypatch, kwargs, message):
    campaign = fake_campaign(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(tl.LauncherError, match=message):
        tl.build_manifest(campaign, log_entry="tlr-099")


def gated(tmp_path, monkeypatch, *, authorize=True, ceiling=True):
    campaign = fake_campaign(tmp_path, monkeypatch)
    records = tmp_path / "records"
    path = campaign / "audits" / "d_readiness_20261006T200000Z" / "readiness_report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"ready": True, "runner_sha256": RUNNER, "checks": {}}))
    paths = tl.record_paths(records)
    manifest = tl.write_json_once(paths["manifest"], tl.build_manifest(campaign, log_entry="tlr-099"))
    if ceiling:
        tl.write_json_once(paths["ceiling"], tl.ceiling_record(usd=2.5, baseline_at="2026-10-13T16:00:00Z",
                                                                log_entry="tlr-099"))
    if authorize:
        tl.write_json_once(paths["authorization"], tl.authorization_record(
            manifest_path=paths["manifest"], ceiling_path=paths["ceiling"], log_entry="tlr-100"))
    return campaign, records, manifest


def ready_state(manifest):
    return vm_snapshot(extension={"id": d.EXTENSION_ID, "runner_sha256": RUNNER, "sha256": manifest["extension"]["sha256"]})


def step_for(campaign, records, manifest, *, vm_runner=None, spend=lambda _ceiling: {"within_ceiling": True}):
    bound = manifest["vm_runner"]["sha256"]
    return tl.build_step(campaign, records=records, vm_runner=vm_runner or (lambda: bound), runner=RUNNER, spend=spend)


def test_an_authorized_verified_manifest_admits_the_five_fits_with_their_forecast(tmp_path, monkeypatch):
    campaign, records, manifest = gated(tmp_path, monkeypatch)
    step = step_for(campaign, records, manifest)
    planned = c.plan_units([(UNITS[0], 0), (UNITS[1], 1), (UNITS[2], 2)], ready_state(manifest), step=step)
    assert [item["unit"] for item in planned] == UNITS[:3] and {item["fit_seconds"] for item in planned} == {2400.0}
    assert c.launch_dir(step).endswith("/step_tlr/launch") and step.code == d.REMOTE_CODE
    assert step.evidence == records and "step_tlr/**/*" in step.evidence_patterns
    fake = FakeVM(ready_state(manifest))
    out = tl.cmd_launch(fake, session(), [(UNITS[3], 0)], wait=False, step=step)
    script = fake.launched()[0]
    assert out["batch"].startswith("tlr-") and f"cd {d.REMOTE_CODE}" in script
    assert f"run_pmm_trunklr.py --action run --campaign-dir {tl.b.REMOTE_V3} --unit {UNITS[3]}" in script
    assert "run_pmm_v3_campaign.py" not in script


@pytest.mark.parametrize("change,message", [
    ({"authorize": False}, "no recorded GPU-start authorization"),
    ({"ceiling": False, "authorize": False}, "no ceiling record"),
])
def test_no_launch_without_the_recorded_authorization_or_ceiling(tmp_path, monkeypatch, change, message):
    campaign, records, manifest = gated(tmp_path, monkeypatch, **change)
    with pytest.raises(tl.LauncherError, match=message):
        c.plan_units([(UNITS[0], 0)], ready_state(manifest), step=step_for(campaign, records, manifest))


def test_no_launch_with_a_missing_vm_runner_above_the_ceiling_or_for_another_unit(tmp_path, monkeypatch):
    campaign, records, manifest = gated(tmp_path, monkeypatch)
    with pytest.raises(tl.LauncherError, match="VM copy of run_pmm_trunklr.py"):
        c.plan_units([(UNITS[0], 0)], ready_state(manifest),
                     step=step_for(campaign, records, manifest, vm_runner=lambda: None))
    above = lambda _c: {"within_ceiling": False, "committed_usd": 2.6, "reserve_usd": 0.0, "ceiling_usd": 2.5}  # noqa: E731
    with pytest.raises(tl.LauncherError, match="exceed the recorded 2.50 USD ceiling"):
        c.plan_units([(UNITS[0], 0)], ready_state(manifest), step=step_for(campaign, records, manifest, spend=above))
    with pytest.raises(tl.LauncherError, match="not a trunk-LR comparison unit"):
        c.plan_units([(CONTROLS[1], 0)], ready_state(manifest), step=step_for(campaign, records, manifest))


def test_the_launch_script_is_the_step_c_script_with_only_the_runner_command_changed(tmp_path, monkeypatch):
    campaign, records, manifest = gated(tmp_path, monkeypatch)
    step = step_for(campaign, records, manifest)
    planned = [{"unit": UNITS[0], "lane": 0, "fit_seconds": 2400.0}, {"unit": UNITS[1], "lane": 1, "fit_seconds": 2400.0}]
    monkeypatch.setattr(time, "strftime", lambda fmt, t=None: "2026-10-13T00:00:00Z")
    mine = tl.launch_script("tlr-x", planned, session(), step)
    monkeypatch.setattr(tl, "unit_command", c.unit_command)
    assert tl.launch_script("tlr-x", planned, session(), step) == c.launch_script("tlr-x", planned, session(), step)
    import shlex

    for item in planned:
        command = [tl.b.REMOTE_PY, "-u", "run_pmm_trunklr.py", "--action", "run", "--campaign-dir", tl.b.REMOTE_V3,
                   "--unit", item["unit"], "--train-dir", tl.b.REMOTE_TRAIN, "--esm-dir", tl.b.REMOTE_ESM,
                   "--lane", str(item["lane"]), *tl.b.allocation_args(session(), fit_seconds=item["fit_seconds"])]
        assert shlex.join(command) in mine


def test_the_compute_ceiling_counts_running_time_after_the_baseline_and_never_storage(tmp_path):
    ceiling = tl.ceiling_record(usd=2.5, baseline_at="2026-10-13T16:00:00Z", log_entry="tlr-099")
    assert ceiling["reserve_usd"] == 0.0 and "storage excluded" in ceiling["scope"]
    ledger = tmp_path / "usage_ledger.jsonl"
    events = [("s-old", "2026-10-12T10:00:00Z", "2026-10-12T11:00:00Z", 3600.0),
              ("s-new", "2026-10-13T17:00:00Z", "2026-10-13T18:00:00Z", 3600.0)]
    ledger.write_text("".join(json.dumps({"event": "STOPPED", "session_id": sid, "session_start": start,
                                          "session_stop": stop, "elapsed_running_s": run}) + "\n"
                              for sid, start, stop, run in events))
    now = e2.stamp_seconds("2026-10-20T00:00:00Z")
    spent = tl.estimated_spend(ceiling, ledger=ledger, session_file=tmp_path / "none.json", now=now)
    assert spent["closed_running_hours_since"] == 1.0 and spent["committed_usd"] == round(e2.RUNNING_USD_PER_HOUR, 2)
    assert spent["within_ceiling"] is True  # a week later: storage never enters the compute ceiling
    active = tmp_path / "current_session.json"
    active.write_text(json.dumps({"active": True, "session_id": "s-live", "session_start": "2026-10-20T00:00:00Z",
                                  "termination_ts": "2026-10-20T02:30:00Z"}))
    live = tl.estimated_spend(ceiling, ledger=ledger, session_file=active, now=now)
    assert math.isclose(live["committed_usd"], round(3.5 * e2.RUNNING_USD_PER_HOUR, 2), abs_tol=0.011)
    assert live["within_ceiling"] is False  # one closed hour plus a 2.5-hour hard stop exceed 2.50 USD
    with pytest.raises(tl.LauncherError, match="straddle"):
        tl.estimated_spend(tl.ceiling_record(usd=2.5, baseline_at="2026-10-13T17:30:00Z", log_entry="tlr-099"),
                           ledger=ledger, session_file=tmp_path / "none.json", now=now)


# ---------------------------------------------------------------------------
# The adoption rule on synthetic predictions
# ---------------------------------------------------------------------------

SUPPORT = (250, 60, 200, 250)
CONTROL_RECALLS = (0.70, 0.60, 0.80, 0.65)


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


def unit(name: str, fold: int, recalls, groups) -> dict:
    rows = fold_rows(fold, recalls)
    return {"name": name, "run_name": name, "status": "completed", "rows": rows,
            "metrics": pa.prediction_metrics(rows, "four_class"), "optimizer_groups": groups}


def collected_for(candidate_recalls, *, missing=(), failed=(), groups=CANDIDATE_GROUPS):
    units = {}
    for fold in FOLDS:
        units[CONTROLS[fold]] = unit(CONTROLS[fold], fold, CONTROL_RECALLS, CONTROL_GROUPS)
        name = UNITS[fold]
        if fold in missing:
            units[name] = {"name": name, "status": "missing", "final_failure": False}
        elif fold in failed:
            units[name] = {"name": name, "status": "failed", "final_failure": True}
        else:
            units[name] = unit(name, fold, candidate_recalls[fold], groups)
    return {"units": units}


MEMBERSHIP = {f"f{fold}_c{label}_{i}": {"group_id": f"g{label}_{i // 25}"}
              for fold in FOLDS for label, n in enumerate(SUPPORT) for i in range(n)}
GOOD = [(0.73, 0.65, 0.82, 0.68), (0.74, 0.65, 0.82, 0.68), (0.72, 0.65, 0.84, 0.70), (0.76, 0.6667, 0.82, 0.70),
        (0.74, 0.65, 0.83, 0.69)]


def decide(candidate_recalls, **kwargs):
    return tla.assess(collected_for(candidate_recalls, **kwargs), {}, tl.load_spec(), ISTATS, MEMBERSHIP)


def test_a_consistent_gain_without_recall_losses_is_a_promising_candidate():
    result = decide(GOOD)
    assert result["status"] == "complete" and result["reasons"] == []
    decision = result["decision"]
    assert decision["promising_candidate"] is True and decision["outcome"] == "promising_candidate"
    assert decision["gvp_learning_rate"] == "1e-4"
    assert result["comparison"]["unadjusted"]["outcome"] == "better" and len(result["comparison"]["fold_differences"]) == 5
    assert result["comparison"]["sensitivity_folds_1_4"]["mean_difference"] > 0
    assert set(result["comparison"]["breakdown"]["per_fold"]) == set(FOLDS)


def test_a_gain_without_interval_support_retains_the_reference():
    result = decide([(0.80, 0.70, 0.86, 0.72), (0.66, 0.55, 0.76, 0.60), (0.80, 0.70, 0.86, 0.72),
                     (0.66, 0.56, 0.77, 0.61), (0.80, 0.70, 0.86, 0.72)])
    assert result["decision"]["outcome"] == "reference_retained" and result["decision"]["gvp_learning_rate"] == "3e-4"
    assert any(reason.startswith("(b) not interval-supported") for reason in result["reasons"])


def test_a_supported_gain_below_the_tie_band_retains_the_reference():
    result = decide([(0.704, 0.60, 0.80, 0.65)] * 4 + [(0.708, 0.60, 0.80, 0.65)])
    assert result["comparison"]["unadjusted"]["outcome"] == "better"
    assert [r[:3] for r in result["reasons"]] == ["(c)"] and result["decision"]["promising_candidate"] is False


def test_a_class_recall_loss_or_a_zero_fold_recall_retains_the_reference():
    cu_loss = decide([(0.80, 0.55, 0.88, 0.75)] * 5)
    assert cu_loss["comparison"]["unadjusted"]["outcome"] == "better"
    assert any("Cu mean recall -5.00 points" in r for r in cu_loss["reasons"])
    zero = decide([(0.85, 0.70, 0.90, 0.80)] * 2 + [(0.85, 0.0, 0.90, 0.80)] + [(0.85, 0.70, 0.90, 0.80)] * 2)
    assert "(d) fold 2: Cu recall is zero" in zero["reasons"] and zero["decision"]["promising_candidate"] is False


def test_a_gain_carried_by_fold_0_alone_fails_condition_e():
    result = decide([(0.95, 0.95, 0.95, 0.95)] + [(0.70, 0.60, 0.80, 0.64)] * 4)
    assert result["comparison"]["sensitivity_folds_1_4"]["mean_difference"] < 0
    assert any(reason.startswith("(e) folds 1-4 mean gain") for reason in result["reasons"])
    assert result["decision"]["promising_candidate"] is False


def test_wrong_optimizer_groups_fail_condition_a():
    wrong = [{**CONTROL_GROUPS[0]}, dict(CONTROL_GROUPS[1])]  # the trunk group still at 3e-4
    result = decide(GOOD, groups=wrong)
    assert any(reason.startswith("(a) fold 0: optimizer groups") for reason in result["reasons"])
    assert result["decision"]["promising_candidate"] is False


def test_missing_fits_give_no_decision_and_a_second_failure_blocks_it():
    incomplete = decide(GOOD, missing=(3,))
    assert incomplete["status"].startswith("incomplete") and incomplete["decision"] is None
    blocked = decide(GOOD, failed=(0,))
    assert blocked["status"].startswith("blocked") and blocked["decision"]["outcome"] == "blocked"
    assert blocked["decision"]["promising_candidate"] is False and blocked["decision"]["gvp_learning_rate"] == "3e-4"
