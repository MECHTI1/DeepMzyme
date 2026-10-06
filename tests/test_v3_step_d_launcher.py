"""Tests for the step D/E workstation launcher (pmm_v3_step_d.py): unit lists and every gate before a launch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import pmm_v3_step_b as b  # noqa: E402
import pmm_v3_step_c as c  # noqa: E402
import pmm_v3_step_d as d  # noqa: E402
from test_v3_step_c_launcher import FakeVM, launch_record, session, vm  # noqa: E402

RUNNER = {name: "a" * 64 for name in d.RUNNER_FILES}
EXTENSION = {"id": d.EXTENSION_ID, "runner_sha256": RUNNER, "sha256": "e" * 64}
A_UNITS, R_UNITS, B_UNITS = (d.round_units(name) for name in d.ROUND_ORDER)
MEANAGG = "gvp_late_fusion__four_class__meanagg__fold0__seed42"
WD01 = "only_gvp__four_class__wd01__fold0__seed42"
VECNORM = "only_gvp__four_class__vecnorm__fold0__seed42"
POSNOISE = "only_gvp__four_class__posnoise01__fold0__seed42"
COMBO = "only_gvp__four_class__combo-vecnorm+wd01__fold0__seed42"


def write_readiness(campaign: Path, *, ready=True, runner=None, stamp="20261006T200000Z") -> Path:
    path = campaign / "audits" / f"d_readiness_{stamp}" / "readiness_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ready": ready, "runner_sha256": RUNNER if runner is None else runner,
                                "checks": {"unit_inventory": {"passed": True}, "decay_audit": {"passed": ready}}}))
    return path


def write_assessment(campaign: Path, kind: str, result: dict, *, stamp: str, extension: str = "e" * 64) -> None:
    path = campaign / "step_d_evidence" / "assessments" / f"{kind}_{stamp}" / "assessment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": kind, "result": result, "campaign_extension": {"sha256": extension}}))


def rounds(a_passed=(), r_passed=(), *, r_complete=True, b_complete=None):
    out = {"D-A": {"complete": True, "passed": list(a_passed)},
           "D-R": {"complete": r_complete, "passed": list(r_passed)}}
    if b_complete is not None:
        out["D-B"] = {"complete": b_complete, "passed": []}
    return out


def ready_vm(done=(), **overrides):
    """The step C snapshot plus the extension record and completed units."""
    state = vm(extension=EXTENSION, **overrides)
    for name in done:
        state["statuses"][name] = {"status": "completed", "lane": 0, "gate_passed": None,
                                   "family": name.split("__")[0], "target": "four_class", "recipe": d.recipe_of(name)}
    state["claims"] = sorted(state["statuses"])
    return state


def plan(campaign, requests, state, *, key="D", fit_seconds=None):
    step = d.build_step(key, campaign, fit_seconds=fit_seconds, runner=RUNNER)
    return c.plan_units(requests, state, fit_seconds=fit_seconds, step=step)


# ---------------------------------------------------------------------------
# Unit lists and import hygiene
# ---------------------------------------------------------------------------

def test_unit_lists_match_the_frozen_campaign_module():
    sys.path.insert(0, str(ROOT / "src"))
    import pmm_v3_campaign as v3

    for name in d.ROUND_ORDER:
        assert d.round_units(name) == tuple(unit.name for unit in v3.step_units(name)), name
    assert (len(A_UNITS), len(R_UNITS), len(B_UNITS)) == (16, 26, 18)
    assert d.ROUND_ORDER == v3.ROUND_ORDER and d.COST_GATED == v3.COST_GATED and d.EXTENSION_ID == v3.EXTENSION_ID
    assert d.RUNNER_FILES == v3.RUNNER_FILES and d.GVP_FAMILIES == v3.GVP_FAMILIES
    assert d.e_neutral_units() == tuple(unit.name for unit in v3.step_units("E-neutral"))
    finals = {"only_gvp": "combo-vecnorm+wd01", "gvp_late_fusion": "baseline"}
    assert d.e_improvement_units(finals) == tuple(
        unit.name for unit in v3.step_units("E-improvement", final_recipes={"only_gvp": "combo-vecnorm+wd01"}))
    assert [d.round_of(name) for name in (A_UNITS[0], MEANAGG, WD01, VECNORM, COMBO)] == \
        ["D-A", "D-A", "D-R", "D-B", "D-combo"]
    assert d.round_of("only_esm__four_class__baseline__fold1__seed42") is None
    assert d.round_of("only_esm__four_class__combo-a+b__fold0__seed42") is None  # Only-ESMC keeps its baseline


def test_launcher_never_imports_torch():
    code = "import sys; sys.path.insert(0, %r); import pmm_v3_step_d; print('torch' in sys.modules)" % str(ROOT)
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


def test_site_geometry_recipes_are_charged_their_own_cold_caches():
    statuses = ready_vm(done=[MEANAGG, "only_gvp__four_class__meanagg__fold0__seed42"])["statuses"]
    warm = c.FIT_SECONDS["only_gvp"]
    assert c.forecast_seconds(WD01, statuses) == warm  # shares the baseline graphs
    site = "only_gvp__four_class__sitenone__fold0__seed42"
    assert c.forecast_seconds(site, statuses) == warm + c.COLD_CACHE_SECONDS
    statuses[site] = {"status": "completed", "family": "only_gvp", "target": "four_class", "recipe": "sitenone"}
    assert c.forecast_seconds(site.replace("seed42", "seed43"), statuses) == warm
    assert c.forecast_seconds("only_gvp__four_class__sitecountsangles__fold0__seed42", statuses) == warm + c.COLD_CACHE_SECONDS
    assert c.graph_variant("combo-sitecountsangles+wd01") == "sitecountsangles" and c.graph_variant(None) == "legacy"


# ---------------------------------------------------------------------------
# Readiness: nothing starts before the CPU report and the VM extension agree with the worktree
# ---------------------------------------------------------------------------

def test_no_step_d_launch_before_readiness_and_the_extension_record(tmp_path):
    request = [(MEANAGG, 0)]
    with pytest.raises(d.LauncherError, match="no CPU readiness report"):
        plan(tmp_path, request, ready_vm())
    write_readiness(tmp_path, ready=False)
    with pytest.raises(d.LauncherError, match="not ready \\(failed checks: \\['decay_audit'\\]\\)"):
        plan(tmp_path, request, ready_vm())
    write_readiness(tmp_path, runner={**RUNNER, "pmm_v3_campaign.py": "b" * 64}, stamp="20261006T210000Z")
    with pytest.raises(d.LauncherError, match="differ from the ones the readiness report checked"):
        plan(tmp_path, request, ready_vm())
    write_readiness(tmp_path, stamp="20261006T220000Z")  # the newest report decides
    with pytest.raises(d.LauncherError, match="no extension record"):
        plan(tmp_path, request, vm())
    other = {**EXTENSION, "runner_sha256": {**RUNNER, "run_pmm_v3_campaign.py": "c" * 64}}
    with pytest.raises(d.LauncherError, match="froze other runner files"):
        plan(tmp_path, request, vm(extension=other))
    planned = plan(tmp_path, request, ready_vm())
    assert planned == [{"unit": MEANAGG, "lane": 0, "fit_seconds": float(c.FIT_SECONDS["gvp_late_fusion"])}]
    with pytest.raises(d.LauncherError, match="not a step D unit"):
        plan(tmp_path, [("only_esm__six_class__baseline__fold0__seed42", 0)], ready_vm())
    with pytest.raises(d.LauncherError, match="not a step D unit"):
        plan(tmp_path, [("only_esm__four_class__baseline__fold1__seed42", 0)], ready_vm())


# ---------------------------------------------------------------------------
# Round order, Round B, the cost gate and the one combination
# ---------------------------------------------------------------------------

def test_round_r_follows_round_a(tmp_path):
    write_readiness(tmp_path)
    with pytest.raises(d.LauncherError, match="Round R follows Round A; 16 Round A units have not started"):
        plan(tmp_path, [(WD01, 0)], ready_vm())
    almost = ready_vm(done=A_UNITS[:-1])
    with pytest.raises(d.LauncherError, match="1 Round A units have not started"):
        plan(tmp_path, [(WD01, 0)], almost)
    # The last Round A unit and Round R units may fill the lanes together; R does not wait for A's results.
    assert len(plan(tmp_path, [(A_UNITS[-1], 0), (WD01, 1)], almost)) == 2
    running = ready_vm(done=A_UNITS[:-1], launches={A_UNITS[-1]: launch_record(A_UNITS[-1], 2, "running")})
    assert len(plan(tmp_path, [(WD01, 0)], running)) == 1
    with pytest.raises(d.LauncherError, match="completed; a completed unit is never rerun"):
        plan(tmp_path, [(A_UNITS[0], 0)], almost)


def test_round_b_needs_an_assessment_of_a_and_r_with_a_pass(tmp_path):
    write_readiness(tmp_path)
    state = ready_vm(done=A_UNITS + R_UNITS)
    with pytest.raises(d.LauncherError, match="Round B: no checked assessment copy"):
        plan(tmp_path, [(VECNORM, 0)], state)
    write_assessment(tmp_path, "D-A", {"status": "provisional: run D-R next", "rounds": {
        "D-A": {"complete": True, "passed": ["only_gvp:meanagg"]}}}, stamp="20261007T010000Z")
    with pytest.raises(d.LauncherError, match="rounds A and R are not both complete"):
        plan(tmp_path, [(VECNORM, 0)], state)
    write_assessment(tmp_path, "D-R", {"status": "final (D stopped after D-R: ...)", "stopped_after": "D-R",
                                       "final": True, "rounds": rounds(),
                                       "families": {f: {"recipe": "baseline"} for f in d.GVP_FAMILIES}},
                     stamp="20261007T020000Z")
    with pytest.raises(d.LauncherError, match="step D ends after Round R and the baselines are kept"):
        plan(tmp_path, [(VECNORM, 0)], state)
    write_assessment(tmp_path, "D-R", {"status": "provisional: run D-B next", "rounds": rounds(r_passed=["only_gvp:wd01"])},
                     stamp="20261007T030000Z", extension="f" * 64)
    with pytest.raises(d.LauncherError, match="belongs to another extension record"):
        plan(tmp_path, [(VECNORM, 0)], state)
    write_assessment(tmp_path, "D-R", {"status": "provisional: run D-B next", "rounds": rounds(r_passed=["only_gvp:wd01"])},
                     stamp="20261007T040000Z")
    planned = plan(tmp_path, [(VECNORM, 0), ("only_gvp__four_class__sitenone__fold0__seed42", 1)], state)
    assert planned[1]["fit_seconds"] == c.FIT_SECONDS["only_gvp"] + c.COLD_CACHE_SECONDS
    # Cost-gated augmentations: the user's recorded OK and a measured forecast, or they stay "not tested (cost)".
    with pytest.raises(d.LauncherError, match="cost-gated candidate.*\\n.*measured --fit-seconds"):
        plan(tmp_path, [(POSNOISE, 0)], state)
    d.write_record_once(tmp_path / "step_d_evidence" / "cost_gate_approval.json",
                        {"recipes": ["posnoise01"], "log_entry": "v3-099"})
    with pytest.raises(d.LauncherError, match="measured --fit-seconds"):
        plan(tmp_path, [(POSNOISE, 0)], state)
    assert plan(tmp_path, [(POSNOISE, 0)], state, fit_seconds=9000)[0]["fit_seconds"] == 9000.0
    with pytest.raises(d.LauncherError, match="cost-gated candidate"):
        plan(tmp_path, [(POSNOISE.replace("posnoise01", "outerdrop01"), 0)], state, fit_seconds=9000)
    with pytest.raises(d.LauncherError, match="recorded once"):
        d.write_record_once(tmp_path / "step_d_evidence" / "cost_gate_approval.json",
                            {"recipes": ["outerdrop01"], "log_entry": "v3-100"})
    with pytest.raises(d.LauncherError, match="log entry"):
        d.write_record_once(tmp_path / "other.json", {"recipes": ["outerdrop01"], "log_entry": None})


def test_only_the_required_combination_is_admitted(tmp_path):
    write_readiness(tmp_path)
    state = ready_vm(done=A_UNITS + R_UNITS + B_UNITS)
    provisional = {"status": "provisional: run D-B next", "rounds": rounds(r_passed=["only_gvp:wd01"]),
                   "families": {"only_gvp": {"decision": "run combination", "recipe": "combo-meanagg+wd01"}}}
    write_assessment(tmp_path, "D-R", provisional, stamp="20261007T040000Z")
    with pytest.raises(d.LauncherError, match="not a step D unit"):  # a provisional combination is never run
        plan(tmp_path, [(COMBO.replace("vecnorm+wd01", "meanagg+wd01"), 0)], state)
    blocked = {"status": "blocked: a combination run is still missing", "final": False,
               "rounds": rounds(r_passed=["only_gvp:wd01"], b_complete=True),
               "families": {"only_gvp": {"decision": "run combination", "recipe": "combo-vecnorm+wd01"},
                            "gvp_late_fusion": {"decision": "baseline", "recipe": "baseline"}}}
    write_assessment(tmp_path, "D-B", blocked, stamp="20261008T010000Z")
    assert d.required_combinations(d.latest_assessment(tmp_path / "step_d_evidence", ("D-B",))) == {
        "only_gvp": "combo-vecnorm+wd01"}
    assert len(plan(tmp_path, [(COMBO, 0), (COMBO.replace("seed42", "seed43"), 1)], state)) == 2
    for other in (COMBO.replace("vecnorm+wd01", "vecnorm+wd10"),
                  COMBO.replace("only_gvp", "gvp_late_fusion").replace("vecnorm+wd01", "meanagg+wd01")):
        with pytest.raises(d.LauncherError, match="not a step D unit"):
            plan(tmp_path, [(other, 0)], state)
    # A combination that already ran stays listed, but a different one is still refused by its gate.
    ran = COMBO.replace("vecnorm+wd01", "vecnorm+wd10")
    seen = ready_vm(done=A_UNITS + R_UNITS + B_UNITS, launches={ran: launch_record(ran, 0, "lost")})
    with pytest.raises(d.LauncherError, match="not the combination the assessment of every completed round requires"):
        plan(tmp_path, [(ran, 0)], seen)
    final = {**blocked, "status": "final", "final": True,
             "families": {"only_gvp": {"decision": "combination", "recipe": "combo-vecnorm+wd01"},
                          "gvp_late_fusion": {"decision": "baseline", "recipe": "baseline"}}}
    write_assessment(tmp_path, "D-B", final, stamp="20261008T020000Z")
    gate = d.latest_assessment(tmp_path / "step_d_evidence", ("D-A", "D-R", "D-B"))
    assert d.required_combinations(gate) == {} and d.final_recipes(gate) == {
        "only_gvp": "combo-vecnorm+wd01", "gvp_late_fusion": "baseline"}


# ---------------------------------------------------------------------------
# Step E: the final-test label decision and a final step D
# ---------------------------------------------------------------------------

def test_step_e_waits_for_the_label_decision_and_a_final_step_d(tmp_path):
    write_readiness(tmp_path)
    state = ready_vm()
    neutral = "only_esm__five_class__baseline__fold1__seed42"
    improved = "only_gvp__four_class__combo-vecnorm+wd01__fold2__seed42"
    with pytest.raises(d.LauncherError, match="final-test label decision.*\\n.*no checked assessment copy"):
        plan(tmp_path, [(neutral, 0)], state, key="E")
    d.write_record_once(tmp_path / "step_e_evidence" / "step_e_gate.json",
                        {"final_test_label": "both_results_secondary", "log_entry": "v3-030"})
    write_assessment(tmp_path, "D-R", {"status": "provisional: run D-B next", "final": False,
                                       "rounds": rounds(r_passed=["only_gvp:wd01"])}, stamp="20261007T040000Z")
    with pytest.raises(d.LauncherError, match="step D is not final"):
        plan(tmp_path, [(neutral, 0)], state, key="E")
    with pytest.raises(d.LauncherError, match="not a step E unit"):
        plan(tmp_path, [(improved, 0)], state, key="E")
    write_assessment(tmp_path, "D-B", {"status": "final", "final": True, "rounds": rounds(b_complete=True), "families": {
        "only_gvp": {"recipe": "combo-vecnorm+wd01"}, "gvp_late_fusion": {"recipe": "baseline"}}},
        stamp="20261008T020000Z")
    planned = plan(tmp_path, [(neutral, 0), (improved, 1)], state, key="E")
    assert [item["unit"] for item in planned] == [neutral, improved]
    with pytest.raises(d.LauncherError, match="not a step E unit"):
        plan(tmp_path, [(improved.replace("only_gvp", "gvp_late_fusion"), 0)], state, key="E")
    with pytest.raises(d.LauncherError, match="not a step E unit"):
        plan(tmp_path, [(MEANAGG, 0)], state, key="E")
    assert d.assess_args("E", round_name=None, not_tested=[], campaign=tmp_path) == (
        "--step", "E", "--final-recipe", "only_gvp=combo-vecnorm+wd01")


# ---------------------------------------------------------------------------
# Commands built for the VM
# ---------------------------------------------------------------------------

def test_step_d_runs_from_the_extension_code_directory_and_its_own_folders(tmp_path):
    write_readiness(tmp_path)
    step = d.build_step("D", tmp_path, runner=RUNNER)
    script = c.launch_script("d-test", [{"unit": MEANAGG, "lane": 1, "fit_seconds": 2400.0}], session(), step)
    assert f"cd {d.REMOTE_CODE}" in script and f"{b.REMOTE_V3}/step_d/launch/{MEANAGG}.rc" in script
    assert d.REMOTE_CODE != b.REMOTE_CODE and "step_c" not in script
    assert step.evidence == tmp_path / "step_d_evidence"
    assert "step_d/**/*" in step.evidence_patterns and "campaign_extension.json" in step.evidence_patterns
    assert "step_c/**/*" not in step.evidence_patterns
    remote = FakeVM(ready_vm())
    out = c.cmd_launch(remote, session(), [(MEANAGG, 1)], wait=False, step=step)
    assert out["batch"].startswith("d-") and len(remote.launched()) == 1
    assert d.assess_args("D", round_name="D-B", not_tested=["posnoise01", "outerdrop01"]) == (
        "--step", "D-B", "--not-tested-cost", "posnoise01", "outerdrop01")
    with pytest.raises(d.LauncherError, match="Round B only"):
        d.assess_args("D", round_name="D-R", not_tested=["posnoise01"])
    with pytest.raises(d.LauncherError, match="needs --round"):
        d.assess_args("D", round_name=None, not_tested=[])


def test_a_second_failure_in_step_d_is_not_rerun(tmp_path):
    write_readiness(tmp_path)
    state = ready_vm()
    state["statuses"][MEANAGG] = {"status": "failed", "lane": 0, "family": "gvp_late_fusion", "target": "four_class",
                                  "recipe": "meanagg"}
    state["archived"] = {MEANAGG: 1}
    with pytest.raises(d.LauncherError, match="did not pass the one-fold screen"):
        plan(tmp_path, [(MEANAGG, 0)], state)


def test_the_extension_is_recorded_only_after_readiness_and_copied_with_its_hash(tmp_path, monkeypatch):
    class Transport:
        def read_bytes(self, path):
            return b'{"extension_id": "v3-ext1-round-r"}'

    class Remote:
        transport = Transport()

        def __init__(self, printed):
            self.printed, self.scripts = printed, []

        def run(self, script, **kwargs):
            self.scripts.append(script)
            return subprocess.CompletedProcess([], 0, stdout=json.dumps(self.printed), stderr="")

    import hashlib

    monkeypatch.setattr(d, "local_runner_sha256", lambda root=ROOT: RUNNER)
    digest = hashlib.sha256(Transport().read_bytes("")).hexdigest()
    with pytest.raises(d.LauncherError, match="no CPU readiness report"):
        d.cmd_extend(Remote({}), tmp_path)
    write_readiness(tmp_path)
    with pytest.raises(d.LauncherError, match="other runner files"):
        d.cmd_extend(Remote({"runner_sha256": {}, "extension_sha256": digest}), tmp_path)
    with pytest.raises(d.LauncherError, match="differs from the VM file"):
        d.cmd_extend(Remote({"runner_sha256": RUNNER, "extension_sha256": "0" * 64}), tmp_path)
    remote = Remote({"runner_sha256": RUNNER, "extension_sha256": digest})
    out = d.cmd_extend(remote, tmp_path)
    assert Path(out["copy"]).read_bytes() == Transport().read_bytes("")
    assert f"cd {d.REMOTE_CODE}" in remote.scripts[0] and "--action extend" in remote.scripts[0]


def test_push_bundle_refuses_a_bundle_that_is_not_this_worktrees_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "local_runner_sha256", lambda root=ROOT: RUNNER)
    (tmp_path / "v3_bundle_manifest.json").write_text(json.dumps({"parent_bundle": None, "runner_sha256": RUNNER}))
    with pytest.raises(d.LauncherError, match="Not an extension bundle"):
        d.cmd_push_bundle(None, tmp_path)
    (tmp_path / "v3_bundle_manifest.json").write_text(json.dumps({"parent_bundle": {"git_commit": "x"},
                                                                 "runner_sha256": {}}))
    with pytest.raises(d.LauncherError, match="differ from this worktree's"):
        d.cmd_push_bundle(None, tmp_path)


def test_offline_commands_list_the_rounds_and_gates(tmp_path, capsys):
    assert d.main(["units"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["counts"] == {"D-A": 16, "D-R": 26, "D-B": 18} and listed["cost_gated"] == list(d.COST_GATED)
    assert d.main(["units", "--step", "E"]) == 0
    assert len(json.loads(capsys.readouterr().out)["neutral"]) == 36
    gates = d.cmd_gates("D", tmp_path)
    assert gates["readiness_report"] is None and gates["required_combinations"] == {}
    write_readiness(tmp_path)
    assert d.cmd_gates("E", tmp_path)["readiness_ready"] is True
