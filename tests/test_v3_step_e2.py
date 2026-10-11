"""Tests for step E2: the workstation launcher (pmm_v3_step_e2.py) and the E2 assessor (pmm_v3_e2_assessment.py).

The launcher part checks the unit list against the frozen E2 specification, the one-setting recipes, the manifest
bindings and refusals, the ceiling and spending control and every launch gate (no VM contact). The assessor part
checks the declared comparisons, both conclusions, the replacement rule with its reasons, ties, incomplete and
failed evidence, a five- or six-class original selection, and the durable-copy and view checks on synthetic data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import pmm_v3_assessment as pa  # noqa: E402
import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_e2_assessment as e2a  # noqa: E402
import pmm_v3_step_c as c  # noqa: E402
import pmm_v3_step_d as d  # noqa: E402
import pmm_v3_step_e2 as e2  # noqa: E402
from test_v3_assessment import NOISE, metrics  # noqa: E402
from test_v3_campaign import campaign  # noqa: E402,F401  (fixture reuse)
from test_v3_step_c_launcher import FakeVM, session, vm  # noqa: E402

SPEC = e2.load_e2_spec()
FUSION_SELECTION = "gvp_late_fusion__four_class__baseline"
ISTATS = {"confidence": 0.95, "bootstrap_resamples": 10000, "bootstrap_seed": 42, "recall_drop_limit": 0.03}
RUNNER = d.local_runner_sha256()
A4_ASSESSOR = hashlib.sha256((ROOT / "pmm_v3_assessment.py").read_bytes()).hexdigest()
E2_UNIT = "gvp_late_fusion__four_class__gvpaux03__fold1__seed42"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Unit list, recipes and comparisons
# ---------------------------------------------------------------------------

def test_the_unit_list_is_the_frozen_specifications_36_fits():
    units = e2.e2_units(SPEC)
    assert len(units) == 36 and len(set(units)) == 36
    parsed = [v3.Unit.parse(name) for name in units]
    assert {(u.target, u.seed) for u in parsed} == {("four_class", 42)} and {u.fold for u in parsed} == {1, 2, 3, 4}
    assert not any(u.recipe.startswith("combo-") for u in parsed)
    by_recipe = {}
    for u in parsed:
        by_recipe.setdefault((u.family, u.recipe), []).append(u.fold)
    assert by_recipe == {("only_gvp", "wd10"): [1, 2, 3, 4], ("only_gvp", "sitecountsangles"): [1, 2, 3, 4],
                         **{("gvp_late_fusion", r): [1, 2, 3, 4] for r in
                            ("invsqrtw", "meanagg", "gvpaux03", "resdrop01", "esmdrop02", "wd10")},
                         ("only_gvp", "sitenone"): [1, 2, 3, 4]}
    # Not a step E unit, and the step E launcher admits none of them.
    assert not set(units) & set(d.e_neutral_units() + d.e_improvement_units({"only_gvp": "meanagg",
                                                                              "gvp_late_fusion": "headdrop03"}))


def test_launcher_never_imports_torch():
    code = "import sys; sys.path.insert(0, %r); import pmm_v3_step_e2; print('torch' in sys.modules)" % str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


EXPECTED_CHANGE = {"wd10": {"weight_decay"}, "meanagg": {"normalize_message_aggregation"},
                   "gvpaux03": {"gvp_auxiliary_loss_weight"}, "resdrop01": {"gvp_residual_dropout"},
                   "esmdrop02": {"esm_modality_dropout"}, "sitecountsangles": {"site_geometry_features"}}
WEIGHT_KEYS = {"metal_class_weight_mode", "mn_loss_multiplier", "cu_loss_multiplier", "zn_loss_multiplier",
               "fe_loss_multiplier", "co_loss_multiplier", "ni_loss_multiplier", "class_viii_loss_multiplier"}


def test_each_candidate_changes_exactly_its_own_setting_from_its_matched_control(campaign):  # noqa: F811
    from training.config import config_to_payload, parse_args

    train, esm_dir, paths, _, _ = campaign

    def resolved(cell: str) -> dict:
        family, target, recipe = cell.split("__")
        command = v3.build_command(paths, v3.Unit(family, target, recipe, 1, 42), python_bin="python",
                                   train_dir=train, esm_dir=esm_dir, device="cpu", lane=0)[0]
        drop = set(e2a.v3.v2.NON_IDENTITY_CONFIG_KEYS) | {"campaign_run_identity"}
        return {k: v for k, v in config_to_payload(parse_args(command[3:])).items() if k not in drop}

    for item in e2.candidate_cells(SPEC):
        control, candidate = resolved(item["matched_control"]), resolved(item["cell"])
        changed = {key for key in set(control) | set(candidate) if control.get(key) != candidate.get(key)}
        if item["recipe"] == "invsqrtw":
            assert "metal_class_weight_mode" in changed and changed <= WEIGHT_KEYS, changed
            assert candidate["metal_class_weight_mode"] == "inverse_sqrt_frequency"
        else:
            assert changed == EXPECTED_CHANGE[item["recipe"]], (item["cell"], changed)
    # The geometry control itself differs from the Only-GVP baseline only in its site-geometry setting.
    base, none = resolved("only_gvp__four_class__baseline"), resolved("only_gvp__four_class__sitenone")
    assert {k for k in set(base) | set(none) if base.get(k) != none.get(k)} == {"site_geometry_features"}


@pytest.mark.parametrize("selection, coincident", [
    (FUSION_SELECTION, {"C12": "C3", "C13": "C4", "C14": "C5", "C15": "C6", "C16": "C7", "C17": "C8"}),
    ("only_gvp__four_class__baseline", {"C10": "C1", "C11": "C9"}),
    ("only_esm__five_class__baseline", {}),
])
def test_the_declared_family_has_17_comparisons_and_flags_coincident_ones(selection, coincident):
    comparisons = e2.declared_comparisons(SPEC, selection)
    assert [item["id"] for item in comparisons] == [f"C{k}" for k in range(1, 18)]
    assert {item["id"]: item["coincident_with"] for item in comparisons if item["coincident_with"]} == coincident
    assert comparisons[8]["candidate"] == "only_gvp__four_class__sitecountsangles"
    assert comparisons[8]["comparator"] == "only_gvp__four_class__baseline"
    assert comparisons[1]["comparator"] == "only_gvp__four_class__sitenone"
    assert all(item["comparator"] == selection for item in comparisons[9:])


# ---------------------------------------------------------------------------
# Manifest: bindings and refusals
# ---------------------------------------------------------------------------

def unit_evidence(names):
    return {name: {"checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest(),
                   "predictions_sha256": hashlib.sha256(name[::-1].encode()).hexdigest()} for name in names}


def fake_campaign(tmp_path: Path, *, selection=FUSION_SELECTION, improvements=None, finals=None,
                  drop_evidence=(), runner=None) -> Path:
    """Workstation records of a campaign whose step E is complete."""
    campaign = tmp_path / "campaign"
    extension = {"extension_id": d.EXTENSION_ID, "runner_sha256": runner or RUNNER, "assessor_sha256": A4_ASSESSOR,
                 "parent": {"manifest_sha256": "m" * 64}}
    ext_path = campaign / e2.EXTENSION_COPY
    ext_path.parent.mkdir(parents=True)
    ext_path.write_text(json.dumps(extension))
    ext_sha = sha(ext_path)
    improvements = improvements or {"gvp_late_fusion": "headdrop03", "only_gvp": "meanagg"}
    finals = finals or improvements
    reused = e2.reused_unit_names(SPEC, selection)
    a4_units = e2.a4_expected(improvements)[1]
    d_units = [n for n in reused["development_fold0"] if n.split("__")[2] != "baseline"]
    contrasts = {f"{f}: {t} vs four_class": {"status": "complete"} for f in c.FAMILIES for t in ("five_class", "six_class")}
    assessment = {"kind": "E", "spec_sha256": sha(e2.SPEC_DIR / "assessment_spec.json"), "assessor_sha256": A4_ASSESSOR,
                  "manifest_sha256": "m" * 64, "campaign_extension": {"sha256": ext_sha},
                  "unit_evidence": unit_evidence([n for n in a4_units if n not in drop_evidence]),
                  "result": {"neutral_target_test": {"contrasts": contrasts},
                             "improvement_check": {f: {"recipe": r, "status": "no positive mean gain on folds 1-4"}
                                                   for f, r in improvements.items()},
                             "summaries": {cell: {} for cell in e2.a4_expected(improvements)[0]},
                             "stage6_selection": {"selected": selection, "candidates": [
                                 {} for _ in e2.a4_expected(improvements)[0]]},
                             "held_out_test_accessed": False}}
    path = campaign / e2.STEP_E_ASSESSMENTS / "E_20261011T130128Z" / "assessment.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(assessment))
    d_assessment = {"kind": "D-B", "campaign_extension": {"sha256": ext_sha},
                    "unit_evidence": unit_evidence([n for n in d_units if n not in drop_evidence]),
                    "result": {"final": True, "families": {f: {"recipe": r} for f, r in finals.items()}}}
    path = campaign / e2.STEP_D_ASSESSMENTS / "D-B_20261007T201334Z" / "assessment.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(d_assessment))
    return campaign


def original_path(campaign: Path) -> Path:
    return campaign / e2.STEP_E_ASSESSMENTS / "E_20261011T130128Z" / "assessment.json"


def rewrite(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_the_manifest_binds_the_original_selection_specifications_tooling_and_reused_evidence(tmp_path):
    campaign = fake_campaign(tmp_path)
    manifest = e2.build_manifest(campaign, log_entry="v3-099")
    assert manifest["original_assessment"]["selection"] == FUSION_SELECTION
    assert manifest["original_assessment"]["sha256"] == sha(original_path(campaign))
    assert manifest["e2_spec_sha256"] == e2.FROZEN_E2_SPEC_SHA256
    assert manifest["tooling_sha256"] == {name: sha(ROOT / name) for name in e2.TOOL_FILES}
    assert manifest["runner_sha256"] == RUNNER and len(manifest["units"]) == 36
    assert manifest["new_control_units"] == [f"only_gvp__four_class__sitenone__fold{k}__seed42" for k in (1, 2, 3, 4)]
    assert len(manifest["reused_units"]) == 19 and manifest["held_out_access"] is False
    assert manifest["reused_units"]["only_gvp__four_class__sitenone__fold0__seed42"]["source"] == "final step D assessment"
    assert e2.verify_manifest(manifest) == []
    record = campaign / e2.EVIDENCE_DIR / e2.MANIFEST_NAME
    e2.write_json_once(record, manifest)
    with pytest.raises(e2.LauncherError, match="recorded once"):
        e2.write_json_once(record, manifest)


@pytest.mark.parametrize("change, message", [
    (lambda a: a["result"].update(stage6_selection=None), "no Stage 6 selection"),
    (lambda a: a["result"]["neutral_target_test"]["contrasts"]["only_gvp: six_class vs four_class"].update(
        status="incomplete"), "neutral contrasts"),
    (lambda a: a["result"]["improvement_check"]["only_gvp"].update(status="incomplete"), "improvement checks"),
    (lambda a: a["result"].update(held_out_test_accessed=True), "held_out_test_accessed"),
    (lambda a: a.update(kind="D-B"), "not a step E assessment"),
    (lambda a: a.update(spec_sha256="0" * 64), "another A4 specification"),
    (lambda a: a.update(campaign_extension={"sha256": "0" * 64}), "another extension record"),
    (lambda a: a.update(assessor_sha256="0" * 64), "not the assessor of the original step E assessment"),
    (lambda a: a["unit_evidence"].pop("only_esm__six_class__baseline__fold3__seed42"), "not exactly the 55 units"),
    (lambda a: a["result"]["summaries"].pop("gvp_late_fusion__four_class__headdrop03"), "not exactly the 11"),
])
def test_the_manifest_refuses_an_incomplete_or_foreign_original_assessment(tmp_path, change, message):
    campaign = fake_campaign(tmp_path)
    rewrite(original_path(campaign), change)
    with pytest.raises(e2.LauncherError, match=message):
        e2.build_manifest(campaign, log_entry="v3-099")


def test_the_manifest_refuses_mismatched_recipes_runner_or_missing_reused_evidence(tmp_path):
    with pytest.raises(e2.LauncherError, match="not the final step D recipes"):
        e2.build_manifest(fake_campaign(tmp_path / "a", finals={"gvp_late_fusion": "headdrop03",
                                                                "only_gvp": "wd10"}), log_entry="v3-099")
    with pytest.raises(e2.LauncherError, match="runner files differ"):
        e2.build_manifest(fake_campaign(tmp_path / "b", runner={**RUNNER, "pmm_v3_campaign.py": "0" * 64}),
                          log_entry="v3-099")
    missing = "only_gvp__four_class__sitenone__fold0__seed42"
    with pytest.raises(e2.LauncherError, match=f"{missing}: no verified evidence"):
        e2.build_manifest(fake_campaign(tmp_path / "c", drop_evidence=(missing,)), log_entry="v3-099")
    with pytest.raises(e2.LauncherError, match="log entry"):
        e2.build_manifest(fake_campaign(tmp_path / "d"), log_entry="decided")


def test_the_expected_step_e_coverage_equals_the_frozen_assessor_units():
    finals = {"gvp_late_fusion": "headdrop03", "only_gvp": "meanagg"}
    cells, units = e2.a4_expected(finals)
    assert len(cells) == 11 and sorted(units) == sorted(e2a.a4_unit_names(finals))


def test_a_five_class_original_selection_is_bound_with_its_target(tmp_path):
    manifest = e2.build_manifest(fake_campaign(tmp_path, selection="only_esm__five_class__baseline"),
                                 log_entry="v3-099")
    assert manifest["original_assessment"]["selection_target"] == "five_class"
    assert "only_esm__five_class__baseline__fold3__seed42" in manifest["reused_units"]
    assert all(item["coincident_with"] is None for item in manifest["comparisons"])


def test_verify_manifest_detects_every_changed_binding(tmp_path):
    campaign = fake_campaign(tmp_path)
    manifest = e2.build_manifest(campaign, log_entry="v3-099")
    assert e2.verify_manifest(manifest, vm_extension={"sha256": manifest["extension"]["sha256"],
                                                      "runner_sha256": RUNNER}) == []
    assert any("VM extension" in p for p in e2.verify_manifest(manifest, vm_extension={"sha256": "0" * 64,
                                                                                       "runner_sha256": RUNNER}))
    tooled = copy.deepcopy(manifest)
    tooled["tooling_sha256"]["pmm_v3_e2_assessment.py"] = "0" * 64
    assert any("E2 tooling or a reused launcher/assessor module changed" in p for p in e2.verify_manifest(tooled))
    rewrite(original_path(campaign), lambda a: a.update(assessor_sha256="b" * 64))
    assert any("original_assessment" in p for p in e2.verify_manifest(manifest))


# ---------------------------------------------------------------------------
# Ceiling and spending
# ---------------------------------------------------------------------------

def ledger(path: Path, sessions) -> Path:
    lines = [json.dumps({"event": "START_REQUESTED", "ts": "2026-10-12T00:00:00Z"})]
    for sid, start, stop, seconds in sessions:
        lines.append(json.dumps({"event": "STOPPED", "session_id": sid, "session_start": start,
                                 "session_stop": stop, "elapsed_running_s": seconds}))
    path.write_text("\n".join(lines) + "\n")
    return path


CEILING = e2.ceiling_record(usd=55.0, spent_usd=31.98, spent_at="2026-10-11T13:10:00Z", spent_source="log v3-031",
                            log_entry="v3-099")


def test_spending_counts_sessions_after_the_baseline_storage_and_the_active_session_to_its_hard_stop(tmp_path):
    path = ledger(tmp_path / "ledger.jsonl", [
        ("old", "2026-10-11T11:57:03Z", "2026-10-11T13:03:24Z", 3981),
        ("s1", "2026-10-12T08:00:00Z", "2026-10-12T12:00:00Z", 14400)])
    active = tmp_path / "current.json"
    active.write_text(json.dumps({"active": False}))
    now = e2.stamp_seconds("2026-10-13T13:10:00Z")  # two days after the baseline
    out = e2.estimated_spend(CEILING, ledger=path, session_file=active, now=now)
    assert out["closed_running_hours_since"] == 4.0
    assert out["spent_usd"] == pytest.approx(31.98 + 4 * 0.858624 + 2 * 0.55, abs=0.01)
    assert out["within_ceiling"] and out["active_session"] is None
    active.write_text(json.dumps({"active": True, "session_id": "s2", "session_start": "2026-10-13T12:00:00Z",
                                  "termination_ts": "2026-10-13T16:00:00Z"}))
    out = e2.estimated_spend(CEILING, ledger=path, session_file=active, now=now)
    assert out["active_session_to_hard_stop_usd"] == pytest.approx(4 * 0.858624, abs=0.01)
    hours_to_stop = 16.0 - 13 - 10 / 60  # storage is committed over the same span as the running session
    assert out["committed_usd"] == pytest.approx(out["spent_usd"] + 4 * 0.858624 + hours_to_stop / 24 * 0.55, abs=0.011)
    assert out["projected_usd"] == pytest.approx(out["committed_usd"] + 6.0, abs=0.011)
    tight = {**CEILING, "ceiling_usd": out["committed_usd"] + 1.0}
    assert not e2.estimated_spend(tight, ledger=path, session_file=active, now=now)["within_ceiling"]
    ledger(path, [("x", "2026-10-11T13:00:00Z", "2026-10-11T14:00:00Z", 3600)])
    with pytest.raises(e2.LauncherError, match="straddle"):
        e2.estimated_spend(CEILING, ledger=path, session_file=active, now=now)


def test_ceiling_and_authorization_records_are_explicit(tmp_path):
    with pytest.raises(e2.LauncherError, match="Implausible"):
        e2.ceiling_record(usd=30.0, spent_usd=31.98, spent_at="2026-10-11T13:10:00Z", spent_source="x",
                          log_entry="v3-099")
    assert CEILING["grants"].startswith("nothing") and CEILING["reserve_usd"] == 6.0
    assert "storage" in CEILING["reserve_basis"] and "step F" in CEILING["reserve_basis"]
    with pytest.raises(e2.LauncherError, match="manifest and the ceiling first"):
        e2.authorization_record(manifest_path=tmp_path / "m.json", ceiling_path=tmp_path / "c.json",
                                log_entry="v3-099")


# ---------------------------------------------------------------------------
# Launch gates
# ---------------------------------------------------------------------------

def write_readiness(campaign: Path) -> None:
    path = campaign / "audits" / "d_readiness_20261006T200000Z" / "readiness_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ready": True, "runner_sha256": RUNNER, "checks": {}}))


def gated_campaign(tmp_path, *, authorize=True, ceiling=True) -> tuple[Path, dict]:
    campaign = fake_campaign(tmp_path)
    write_readiness(campaign)
    records = e2.record_paths(campaign)
    manifest = e2.write_json_once(records["manifest"], e2.build_manifest(campaign, log_entry="v3-099"))
    if ceiling:
        e2.write_json_once(records["ceiling"], CEILING)
    if authorize:
        e2.write_json_once(records["authorization"], e2.authorization_record(
            manifest_path=records["manifest"], ceiling_path=records["ceiling"], log_entry="v3-100"))
    return campaign, manifest


def ready_state(manifest, done=()):
    state = vm(extension={"id": d.EXTENSION_ID, "runner_sha256": RUNNER, "sha256": manifest["extension"]["sha256"]})
    for name in done:
        family, target, recipe = name.split("__")[:3]
        state["statuses"][name] = {"status": "completed", "lane": 0, "gate_passed": None, "family": family,
                                   "target": target, "recipe": recipe}
    state["claims"] = sorted(state["statuses"])
    return state


def within(_ceiling):
    return {"within_ceiling": True}


def plan(campaign, requests, state, *, spend=within):
    return c.plan_units(requests, state, step=e2.build_step(campaign, runner=RUNNER, spend=spend))


def test_an_authorized_manifest_within_the_ceiling_admits_e2_units_from_the_extension_code(tmp_path):
    campaign, manifest = gated_campaign(tmp_path)
    planned = plan(campaign, [(E2_UNIT, 0), ("only_gvp__four_class__sitenone__fold2__seed42", 1)],
                   ready_state(manifest))
    assert [item["unit"] for item in planned] == [E2_UNIT, "only_gvp__four_class__sitenone__fold2__seed42"]
    step = e2.build_step(campaign, runner=RUNNER, spend=within)
    assert c.launch_dir(step).endswith("/step_e2/launch") and step.code == d.REMOTE_CODE
    assert step.evidence == campaign / "step_e2_evidence" and "step_e2/**/*" in step.evidence_patterns
    fake = FakeVM(ready_state(manifest))
    out = c.cmd_launch(fake, session(), [(E2_UNIT, 2)], wait=False, step=step)
    script = fake.launched()[0]
    assert out["batch"].startswith("e2-") and f"cd {d.REMOTE_CODE}" in script
    assert f"--action run --campaign-dir {e2.b.REMOTE_V3} --unit {E2_UNIT}" in script


def test_no_e2_launch_without_the_users_recorded_authorization(tmp_path):
    campaign, manifest = gated_campaign(tmp_path, authorize=False)
    with pytest.raises(e2.LauncherError, match="separate authorization"):
        plan(campaign, [(E2_UNIT, 0)], ready_state(manifest))


def test_no_e2_launch_without_a_ceiling_or_above_it(tmp_path):
    campaign, manifest = gated_campaign(tmp_path / "a", authorize=False, ceiling=False)
    with pytest.raises(e2.LauncherError, match="no E2 ceiling record"):
        plan(campaign, [(E2_UNIT, 0)], ready_state(manifest))
    campaign, manifest = gated_campaign(tmp_path / "b")
    over = lambda _c: {"within_ceiling": False, "committed_usd": 54.0, "reserve_usd": 2.0, "ceiling_usd": 55.0}
    with pytest.raises(e2.LauncherError, match="exceed the recorded 55.00 USD ceiling"):
        plan(campaign, [(E2_UNIT, 0)], ready_state(manifest), spend=over)


def test_an_authorization_for_another_manifest_or_ceiling_is_refused(tmp_path):
    campaign, manifest = gated_campaign(tmp_path)
    records = e2.record_paths(campaign)
    rewrite(records["authorization"], lambda a: a.update(manifest_sha256="0" * 64, ceiling_sha256="0" * 64))
    with pytest.raises(e2.LauncherError, match="another E2 manifest(.|\n)*another ceiling record"):
        plan(campaign, [(E2_UNIT, 0)], ready_state(manifest))


def test_a_changed_binding_or_the_wrong_vm_extension_blocks_every_launch(tmp_path):
    campaign, manifest = gated_campaign(tmp_path)
    state = ready_state(manifest)
    state["extension"] = {**state["extension"], "sha256": "0" * 64}
    with pytest.raises(e2.LauncherError, match="VM extension record is not the one|froze other runner"):
        plan(campaign, [(E2_UNIT, 0)], state)
    rewrite(original_path(campaign), lambda a: a.update(assessor_sha256="c" * 64))
    with pytest.raises(e2.LauncherError, match="original_assessment: .* differs from its bound SHA-256"):
        plan(campaign, [(E2_UNIT, 0)], ready_state(manifest))


def test_no_e2_launch_before_readiness(tmp_path):
    campaign, manifest = gated_campaign(tmp_path)
    shutil.rmtree(campaign / "audits")
    with pytest.raises(e2.LauncherError, match="no CPU readiness report"):
        plan(campaign, [(E2_UNIT, 0)], ready_state(manifest))


def test_only_e2_units_are_admitted_and_e2_units_never_through_step_e(tmp_path):
    campaign, manifest = gated_campaign(tmp_path)
    state = ready_state(manifest)
    for other in ("gvp_late_fusion__four_class__baseline__fold1__seed42",  # a step E control
                  "gvp_late_fusion__four_class__gvpaux03__fold0__seed42",  # a step D unit
                  "gvp_late_fusion__four_class__structlr__fold1__seed42",  # not an E2 candidate
                  "gvp_late_fusion__four_class__gvpaux03__fold1__seed43"):  # another seed
        with pytest.raises(e2.LauncherError, match="not a step E2 unit"):
            plan(campaign, [(other, 0)], state)
    label = campaign / "step_e_evidence" / "step_e_gate.json"
    label.write_text(json.dumps({"final_test_label": "both_results_secondary", "log_entry": "v3-022"}))
    step_e = d.build_step("E", campaign, runner=RUNNER)
    with pytest.raises(d.LauncherError, match="not a step E unit"):
        c.plan_units([(E2_UNIT, 0)], state, step=step_e)


def test_completed_e2_units_are_never_rerun_and_a_second_failure_stops(tmp_path):
    campaign, manifest = gated_campaign(tmp_path)
    with pytest.raises(e2.LauncherError, match="never rerun"):
        plan(campaign, [(E2_UNIT, 0)], ready_state(manifest, done=[E2_UNIT]))
    state = ready_state(manifest)
    state["statuses"][E2_UNIT] = {"status": "failed", "lane": 0, "family": "gvp_late_fusion",
                                  "target": "four_class", "recipe": "gvpaux03"}
    state["archived"][E2_UNIT] = 1
    with pytest.raises(e2.LauncherError, match="second failure blocks the final E2 assessment"):
        plan(campaign, [(E2_UNIT, 0)], state)
    fake = FakeVM(ready_state(manifest))
    with pytest.raises(e2.LauncherError, match="not a step E2 unit"):
        c.cmd_archive_failed(fake, "gvp_late_fusion__four_class__baseline__fold1__seed42",
                             e2.build_step(campaign, runner=RUNNER, spend=within))


# ---------------------------------------------------------------------------
# Assessor: specification
# ---------------------------------------------------------------------------

def test_the_assessor_pins_the_e2_specification_and_its_tested_values(tmp_path):
    spec = e2a.load_spec()
    assert spec["comparison_family_size"] == 17
    changed = tmp_path / "spec.json"
    changed.write_text(json.dumps({**SPEC, "tie_band": 0.005}))
    with pytest.raises(ValueError, match="frozen E2 specification"):
        e2a.load_spec(changed)
    with pytest.raises(ValueError, match="tested rules in \\['tie_band'\\]"):
        e2a.load_spec(changed, expected_sha256=sha(changed))
    with pytest.raises(ValueError, match="interval settings differ"):
        e2a.interval_spec(spec, {**pa.SPEC_DEFAULTS, "bootstrap_resamples": 2000})
    assert e2a.interval_spec(spec, pa.SPEC_DEFAULTS) == ISTATS


def test_the_interpretation_is_section_0_of_the_frozen_page_verbatim():
    text = e2a.interpretation()
    page = e2.E2_SPEC_MD.read_text(encoding="utf-8")
    assert text in page and text.startswith("Step E2 is an exploratory extension")
    assert text.endswith("not estimates of this campaign's false-replacement rate or power.")


# ---------------------------------------------------------------------------
# Assessor: comparisons, conclusions and the replacement rule (synthetic metrics)
# ---------------------------------------------------------------------------

JITTER = {0: 0.0, 1: 0.001, 2: -0.002, 3: 0.0015, 4: -0.0005}


def put_cell(collected, cell, base, *, jitter=0.0, recall_shift=None, folds=range(5), native=None):
    family, target, recipe = cell.split("__")
    for fold in folds:
        recalls = [base + NOISE[fold] + jitter * JITTER[fold]] * 4
        for index, delta in (recall_shift or {}).items():
            recalls[index] += delta
        unit = v3.Unit(family, target, recipe, fold, 42)
        native_recall = native(recalls) if native else None
        if native_recall is None and target != "four_class":
            labels = pa.NATIVE[target]
            native_recall = dict(zip(labels, recalls[:3] + [recalls[3]] * (len(labels) - 3)))
        collected[unit.name] = {"name": unit.name, "status": "completed", "metrics": metrics(recalls, native_recall)}


def grid(deltas=None, *, selection=FUSION_SELECTION, selection_base=None, shifts=None):
    """Controls at 0.70 (Only-GVP, sitenone) and 0.75 (late fusion); candidates at control + delta."""
    collected = {}
    put_cell(collected, "only_gvp__four_class__baseline", 0.70)
    put_cell(collected, "only_gvp__four_class__sitenone", 0.70)
    put_cell(collected, FUSION_SELECTION, 0.75)
    if selection not in (FUSION_SELECTION, "only_gvp__four_class__baseline"):
        put_cell(collected, selection, selection_base or 0.74)
    for item in e2.candidate_cells(SPEC):
        base = 0.70 if item["family"] == "only_gvp" else 0.75
        put_cell(collected, item["cell"], base + (deltas or {}).get(item["cell"], -0.01), jitter=1.0,
                 recall_shift=(shifts or {}).get(item["cell"]))
    return collected


def fake_manifest(selection=FUSION_SELECTION):
    return {"comparisons": e2.declared_comparisons(SPEC, selection), "units": list(e2.e2_units(SPEC)),
            "original_assessment": {"selection": selection, "selection_target": selection.split("__")[1],
                                    "sha256": "s" * 64}}


def assess(collected, selection=FUSION_SELECTION):
    return e2a.assess_e2(collected, fake_manifest(selection), SPEC, ISTATS)


def test_no_eligible_candidate_keeps_the_original_selection():
    result = assess(grid())
    assert result["status"] == "complete"
    assert result["final_selection"]["selected"] == FUSION_SELECTION
    assert result["final_selection"]["replaced_original"] is False and result["final_selection"]["eligible"] == []
    item = result["candidates"]["gvp_late_fusion__four_class__gvpaux03"]
    assert item["conclusions"]["own_model"]["status"] == "no positive mean gain on folds 1-4"
    assert item["replacement"]["label"].startswith("not eligible: (b) not interval-supported against "
                                                   "gvp_late_fusion__four_class__baseline (C5 = C14)")
    assert "(c) mean gain over the original selection" in item["replacement"]["label"]


def test_a_supported_candidate_replaces_the_selection_with_the_adjustment_over_17():
    cell = "gvp_late_fusion__four_class__gvpaux03"
    result = assess(grid({cell: 0.03}))
    final = result["final_selection"]
    assert final["selected"] == cell and final["replaced_original"] and final["eligible"] == [cell]
    assert result["candidates"][cell]["replacement"]["label"] == "selected"
    assert result["candidates"][cell]["conclusions"]["own_model"]["status"] == \
        "improvement interval-supported on folds 1-4"
    comparison = result["comparisons"]["C5"]
    assert comparison["coincident_with"] is None and result["comparisons"]["C14"]["coincident_with"] == "C5"
    assert comparison["adjusted"]["confidence"] == pytest.approx(1 - 0.05 / 17)
    assert comparison["n_folds"] == 4 and comparison["t_degrees_of_freedom"] == 3
    assert len(comparison["fold_differences"]) == 4  # fold 0 never enters the interval
    assert result["candidates"][cell]["development_fold0_difference"]["values"][FUSION_SELECTION] == \
        pytest.approx(0.03)
    assert final["refit"] == {"seed": 42, "checkpoint_rule": "terminal", "calibration": None, "ensemble": False}


def test_a_gain_of_02_points_or_less_over_the_selection_does_not_replace():
    cell = "gvp_late_fusion__four_class__resdrop01"
    collected = grid()
    put_cell(collected, cell, 0.7515, jitter=0.1)  # +0.15 points with little spread: interval-supported
    result = assess(collected)
    item = result["candidates"][cell]
    assert result["comparisons"]["C6"]["verdict"] == "better"  # interval-supported but small
    assert not item["replacement"]["eligible"]
    assert any(reason.startswith("(c) mean gain over the original selection +0.15 points, not above 0.20")
               for reason in item["replacement"]["reasons"])
    assert result["final_selection"]["selected"] == FUSION_SELECTION


def test_a_recall_drop_is_named_by_comparator_class_and_magnitude_and_overrides_promising():
    cell = "gvp_late_fusion__four_class__invsqrtw"
    collected = grid({cell: 0.03}, shifts={cell: {1: -0.07}})  # BA +1.25 points, Cu -4 points vs the control
    item = assess(collected)["candidates"][cell]
    own = item["conclusions"]["own_model"]
    assert own["status"] == "blocked by a recall gate against the matched control"
    assert own["failed_gates"] == ["recall gate against gvp_late_fusion__four_class__baseline (C3 = C12): "
                                   "Cu mean recall -4.00 points (limit -3.00)"]
    assert any(r.startswith("(d) recall gate against gvp_late_fusion__four_class__baseline (C3 = C12): Cu")
               for r in item["replacement"]["reasons"])


def test_ties_prefer_higher_minimum_recall_then_the_simpler_family():
    gvp, fusion = "only_gvp__four_class__wd10", "gvp_late_fusion__four_class__meanagg"
    # Selection Only-GVP baseline at 0.70; both candidates reach 0.76 (within the band).
    collected = grid({gvp: 0.06, fusion: 0.01}, selection="only_gvp__four_class__baseline")
    result = assess(collected, "only_gvp__four_class__baseline")
    assert set(result["final_selection"]["eligible"]) == {gvp, fusion}
    assert result["final_selection"]["selected"] == gvp  # same minimum recall, worst fold and SD: simpler family
    assert result["final_selection"]["reason"] == "tie-breakers within the tie band"
    collected = grid({gvp: 0.06, fusion: 0.01}, selection="only_gvp__four_class__baseline",
                     shifts={gvp: {0: 0.01, 1: 0.01, 2: 0.01, 3: -0.03}})  # same BA, lower minimum recall
    assert assess(collected, "only_gvp__four_class__baseline")["final_selection"]["selected"] == fusion


def test_the_geometry_candidate_must_also_beat_the_only_gvp_baseline():
    geometry = "only_gvp__four_class__sitecountsangles"
    collected = grid({geometry: 0.03}, selection="only_esm__four_class__baseline", selection_base=0.60)
    put_cell(collected, "only_gvp__four_class__baseline", 0.74)  # baseline above the geometry candidate
    item = assess(collected, "only_esm__four_class__baseline")["candidates"][geometry]
    assert item["conclusions"]["own_model"]["status"] == "improvement interval-supported on folds 1-4"
    assert item["conclusions"]["against_additional_comparator"]["status"] == "no positive mean gain on folds 1-4"
    assert any(r.startswith("(b) not interval-supported against only_gvp__four_class__baseline (C9)")
               for r in item["replacement"]["reasons"])


@pytest.mark.parametrize("target", ["five_class", "six_class"])
def test_a_five_or_six_class_original_selection_is_compared_on_common_four(target):
    selection = f"only_esm__{target}__baseline"
    cell = "gvp_late_fusion__four_class__esmdrop02"
    collected = grid({cell: 0.03}, selection=selection, selection_base=0.74)
    result = assess(collected, selection)
    against = result["comparisons"]["C16"]
    assert against["comparator"] == selection and against["status"] == "complete"
    assert against["fold_differences"] == pytest.approx([0.04 + JITTER[k] for k in (1, 2, 3, 4)])
    assert result["final_selection"]["selected"] == cell
    native = result["original_selection"]["summary_folds_1_4"]["mean_native_recall"]
    assert set(native) == set(pa.NATIVE[target])  # the selection's own scheme is kept


def test_the_general_comparison_equals_the_frozen_compare_for_a_four_class_control():
    collected = grid({"gvp_late_fusion__four_class__wd10": 0.02})
    control = e2a.cell_folds(collected, FUSION_SELECTION, e2.FOLDS)
    challenger = e2a.cell_folds(collected, "gvp_late_fusion__four_class__wd10", e2.FOLDS)
    frozen = pa.compare(control, challenger, ISTATS, challenger_target="four_class", n_contrasts=17)
    general = e2a.compare_any_target(control, "four_class", challenger, "four_class", ISTATS, n_contrasts=17)
    assert json.dumps(pa._jsonable(frozen), sort_keys=True) == json.dumps(pa._jsonable(general), sort_keys=True)


def test_missing_or_failed_e2_evidence_leaves_no_final_selection():
    collected = grid()
    missing = "only_gvp__four_class__sitenone__fold3__seed42"
    collected[missing] = {"name": missing, "status": "invalid evidence", "note": "differs", "final_failure": False}
    result = assess(collected)
    assert result["status"].startswith("incomplete: 1 of 36") and result["final_selection"] is None
    assert result["comparisons"]["C2"]["status"] == "incomplete"
    collected[missing] = {"name": missing, "status": "failed", "final_failure": True}
    result = assess(collected)
    assert result["status"].startswith("blocked:") and result["final_selection"] is None


def test_the_parity_comparison_tolerates_rounding_only():
    assert e2a.differences({"a": [1.0, {"b": 0.5}]}, {"a": [1.0 + 1e-12, {"b": 0.5}]}) == []
    assert e2a.differences({"a": 1.0}, {"a": 1.0 + 1e-6}) == ["/a: 1.0 != 1.000001"]
    assert e2a.differences({"a": 1}, {"a": 1, "b": 2}) == ["/b: key differs"]


# ---------------------------------------------------------------------------
# Assessor: durable copies and the read-only view
# ---------------------------------------------------------------------------

RUN = "gvp_late_fusion__four_class__gvpaux03__fold1__seed42"


def durable_lane(data: Path, *, lane=0, run=RUN, files=None, acknowledge=True, upload=True) -> Path:
    lane_root = data / "durable" / f"lane{lane}"
    files = files or {f"runs/{run}/val_predictions.csv": b"uid,p\n", f"runs/{run}/selected_checkpoint.json": b"{}",
                      f"run_status_{run}.json": json.dumps({"run_name": run}).encode()}
    entries = []
    for relative, content in files.items():
        path = lane_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)})
    receipts = lane_root / "persistence_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    receipt = receipts / "r1.json"
    receipt.write_text(json.dumps({"ack_path": "persistence_receipts/r1.ack.json", "files": entries}))
    if acknowledge:
        (receipts / "r1.ack.json").write_text(json.dumps({"manifest_sha256": sha(receipt),
                                                          "file_count": len(entries), "verified_unix": 1.0}))
        if upload:  # the VM lane's copy of the acknowledgment, as the evidence copy holds it
            vm_ack = evidence_of(data) / "campaign" / "lanes" / f"lane{lane}" / "persistence_receipts" / "r1.ack.json"
            vm_ack.parent.mkdir(parents=True, exist_ok=True)
            vm_ack.write_bytes((receipts / "r1.ack.json").read_bytes())
    return lane_root


def evidence_of(data: Path) -> Path:
    return data / "step_e2_evidence" / "evidence_20261012T000000Z"


def test_a_durable_copy_must_equal_an_acknowledged_receipt_file_by_file(tmp_path):
    lane_root = durable_lane(tmp_path)
    check = lambda data: e2a.verify_durable_unit(data, 0, RUN, evidence_dir=evidence_of(data))  # noqa: E731
    out = check(tmp_path)
    assert out["receipts"][0]["run_files"] == 2
    (lane_root / "runs" / RUN / "val_predictions.csv").write_bytes(b"uid,p\nx,1\n")
    with pytest.raises(ValueError, match="differs from receipt"):
        check(tmp_path)
    (lane_root / "runs" / RUN / "val_predictions.csv").write_bytes(b"uid,p\n")
    (lane_root / "runs" / RUN / "extra.txt").write_text("x")
    with pytest.raises(ValueError, match="outside receipt"):
        check(tmp_path)
    durable_lane(tmp_path / "unacked", acknowledge=False)
    with pytest.raises(ValueError, match="never acknowledged"):
        check(tmp_path / "unacked")
    durable_lane(tmp_path / "not_uploaded", upload=False)
    with pytest.raises(ValueError, match="never uploaded"):
        check(tmp_path / "not_uploaded")
    lane = durable_lane(tmp_path / "other_ack")
    (evidence_of(tmp_path / "other_ack") / "campaign" / "lanes" / "lane0" / "persistence_receipts"
     / "r1.ack.json").write_text("{}")
    with pytest.raises(ValueError, match="never uploaded"):
        check(tmp_path / "other_ack")
    assert lane.is_dir()
    (lane_root / f"run_status_{RUN}.json").write_text("{}")
    with pytest.raises(ValueError, match="no host-pull receipt lists its current status"):
        check(tmp_path)


def evidence_copy(tmp_path: Path, *, status_bytes: bytes) -> tuple[Path, Path, dict]:
    data, evidence = tmp_path / "data", tmp_path / "data" / "step_e2_evidence" / "evidence_20261012T000000Z"
    campaign = evidence / "campaign"
    (campaign / "lanes" / "lane0").mkdir(parents=True)
    folds = data / "folds" / "fold-set" / "fold_membership.csv"
    folds.parent.mkdir(parents=True)
    folds.write_text("source_uid,fold\n")
    cohort = tmp_path / "cohort.csv"
    cohort.write_text("source_uid\n")
    (campaign / "fold_class_weights.json").write_text("{}")
    (campaign / "execution_settings.json").write_text("{}")
    (campaign / "campaign_manifest.json").write_text(json.dumps({
        "fold_class_weights_sha256": sha(campaign / "fold_class_weights.json"),
        "fold_set": {"id": "fold-set", "fold_membership_sha256": sha(folds)}, "cohort": {"sha256": sha(cohort)}}))
    (campaign / "campaign_extension.json").write_text(json.dumps({"parent": {
        "manifest_sha256": sha(campaign / "campaign_manifest.json"),
        "execution_settings_sha256": sha(campaign / "execution_settings.json")}}))
    (campaign / "lanes" / "lane0" / f"run_status_{RUN}.json").write_bytes(status_bytes)
    durable_lane(data)
    manifest = {"extension": {"sha256": sha(campaign / "campaign_extension.json")}}
    return data, evidence, {"manifest": manifest, "cohort": cohort}


def test_the_view_links_only_statuses_identical_to_their_durable_copies(tmp_path):
    status = json.dumps({"run_name": RUN}).encode()
    data, evidence, extra = evidence_copy(tmp_path, status_bytes=status)
    view = e2a.build_view(tmp_path / "view", data=data, evidence_dir=evidence, cohort=extra["cohort"],
                          manifest=extra["manifest"])
    assert view["status_records"] == 1
    linked = tmp_path / "view" / "lanes" / "lane0" / f"run_status_{RUN}.json"
    assert linked.resolve() == (data / "durable" / "lane0" / f"run_status_{RUN}.json").resolve()
    assert (tmp_path / "view" / "lanes" / "lane0" / "runs").resolve() == (data / "durable" / "lane0" / "runs").resolve()
    data, evidence, extra = evidence_copy(tmp_path / "stale", status_bytes=b'{"run_name": "other"}')
    with pytest.raises(ValueError, match="durable copy differs from the evidence copy"):
        e2a.build_view(tmp_path / "stale" / "view", data=data, evidence_dir=evidence, cohort=extra["cohort"],
                       manifest=extra["manifest"])
    data, evidence, extra = evidence_copy(tmp_path / "foreign", status_bytes=status)
    with pytest.raises(ValueError, match="not the one the E2 manifest binds"):
        e2a.build_view(tmp_path / "foreign" / "view", data=data, evidence_dir=evidence, cohort=extra["cohort"],
                       manifest={"extension": {"sha256": "0" * 64}})


def test_collect_records_evidence_that_differs_from_the_manifest_instead_of_using_it(tmp_path, monkeypatch):
    record = {"name": RUN, "status": "completed", "run_name": RUN,
              "evidence": {"checkpoint_sha256": "a" * 64, "predictions_sha256": "b" * 64}}
    monkeypatch.setattr(e2a.pa, "read_contract", lambda paths: ({}, {}))
    monkeypatch.setattr(e2a.v3, "completed_units",
                        lambda paths: {RUN: {"status_path": str(tmp_path / "lanes" / "lane0" / "x.json")}})
    monkeypatch.setattr(e2a.pa, "load_unit", lambda *args, **kwargs: dict(record))
    monkeypatch.setattr(e2a, "verify_durable_unit", lambda data, lane, run, **kwargs: {"lane": lane})
    bound = {RUN: {"checkpoint_sha256": "a" * 64, "predictions_sha256": "0" * 64}}
    out = e2a.collect_verified(None, [RUN], data=tmp_path, bound=bound, evidence_dir=tmp_path)
    assert out[RUN]["status"] == "invalid evidence" and "differs from the evidence the E2 manifest binds" in out[RUN]["note"]
    out = e2a.collect_verified(None, [RUN], data=tmp_path, bound={}, evidence_dir=tmp_path)
    assert out[RUN]["status"] == "completed" and out[RUN]["durable"] == {"lane": 0}

    def unreplayed(*args, **kwargs):
        raise ValueError("Independent replay does not confirm the selected checkpoint")

    monkeypatch.setattr(e2a.pa, "load_unit", unreplayed)
    out = e2a.collect_verified(None, [RUN], data=tmp_path, bound={}, evidence_dir=tmp_path)
    assert out[RUN]["status"] == "invalid evidence" and "replay" in out[RUN]["note"]


def test_a_unit_recorded_in_two_lanes_is_refused(tmp_path, monkeypatch):
    for lane in (0, 1):
        path = tmp_path / "lanes" / f"lane{lane}" / f"run_status_{RUN}.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"run_name": RUN, "status": "completed"}))
    monkeypatch.setattr(e2a.pa, "read_contract", lambda paths: ({}, {}))
    with pytest.raises(ValueError, match="ran in two lanes"):
        e2a.collect_verified(v3.V3Paths(tmp_path), [RUN], data=tmp_path, bound={}, evidence_dir=tmp_path)


# ---------------------------------------------------------------------------
# Assessor: further cases from the tooling review (2026-10-11)
# ---------------------------------------------------------------------------

def put_folds(collected, cell, values):
    """fold -> balanced value, every class recall equal to it (plus the fold noise)."""
    family, target, recipe = cell.split("__")
    for fold, value in values.items():
        unit = v3.Unit(family, target, recipe, fold, 42)
        collected[unit.name] = {"name": unit.name, "status": "completed",
                                "metrics": metrics([value + NOISE[fold]] * 4, None)}


def test_a_recall_gate_against_the_selection_alone_blocks_replacement_but_not_the_own_model_finding():
    cell = "only_gvp__four_class__wd10"
    collected = grid({cell: 0.08})
    put_cell(collected, FUSION_SELECTION, 0.70, recall_shift={0: 0.15})  # selection Mn 0.85
    result = assess(collected)
    item = result["candidates"][cell]
    assert item["conclusions"]["own_model"]["status"] == "improvement interval-supported on folds 1-4"
    assert item["conclusions"]["own_model"]["failed_gates"] == []
    assert item["replacement"]["reasons"] == [
        "(d) recall gate against gvp_late_fusion__four_class__baseline (C10): Mn mean recall -7.00 points (limit -3.00)"]
    assert result["final_selection"]["selected"] == FUSION_SELECTION


def test_a_recall_gate_against_the_only_gvp_baseline_alone_blocks_the_geometry_candidate():
    geometry, selection = "only_gvp__four_class__sitecountsangles", "only_esm__four_class__baseline"
    collected = grid({geometry: 0.05}, selection=selection, selection_base=0.60)
    put_cell(collected, "only_gvp__four_class__baseline", 0.70, recall_shift={0: 0.10})
    item = assess(collected, selection)["candidates"][geometry]
    assert item["conclusions"]["own_model"]["status"] == "improvement interval-supported on folds 1-4"
    assert item["replacement"]["reasons"] == [
        "(d) recall gate against only_gvp__four_class__baseline (C9): Mn mean recall -5.00 points (limit -3.00)"]


def test_a_zero_class_recall_is_named():
    cell = "gvp_late_fusion__four_class__wd10"
    collected = grid()
    for fold in range(5):
        unit = v3.Unit("gvp_late_fusion", "four_class", "wd10", fold, 42)
        collected[unit.name] = {"name": unit.name, "status": "completed", "metrics": metrics([0.0, 0.95, 0.95, 0.95])}
    own = assess(collected)["candidates"][cell]["conclusions"]["own_model"]
    assert own["status"] == "no positive mean gain on folds 1-4"
    assert any("Mn mean recall is zero" in failure for failure in own["failed_gates"])


def test_invalid_reused_or_selection_evidence_leaves_the_result_incomplete_with_reason_a():
    collected = grid({"gvp_late_fusion__four_class__gvpaux03": 0.03})
    name = f"{FUSION_SELECTION}__fold2__seed42"
    collected[name] = {"name": name, "status": "invalid evidence", "note": "x", "final_failure": False}
    result = assess(collected)
    assert result["status"].startswith("incomplete: 0 of 36") and result["final_selection"] is None
    assert result["comparisons"]["C5"]["status"] == "incomplete"
    reasons = result["candidates"]["gvp_late_fusion__four_class__gvpaux03"]["replacement"]["reasons"]
    assert reasons[0].startswith("(a) incomplete evidence against gvp_late_fusion__four_class__baseline (C5 = C14)")


def test_an_eligible_candidate_is_only_provisional_while_the_assessment_is_incomplete():
    cell = "gvp_late_fusion__four_class__gvpaux03"
    collected = grid({cell: 0.03})
    missing = "only_gvp__four_class__sitenone__fold3__seed42"
    collected[missing] = {"name": missing, "status": "missing", "final_failure": False}
    result = assess(collected)
    label = result["candidates"][cell]["replacement"]["label"]
    assert result["final_selection"] is None and label == "eligible so far; no final selection (assessment incomplete)"


def test_unadjusted_intervals_decide_and_bonferroni_is_sensitivity_only():
    cell = "gvp_late_fusion__four_class__meanagg"
    collected = grid()
    put_folds(collected, cell, {0: 0.765, 1: 0.755, 2: 0.775, 3: 0.760, 4: 0.770})
    result = assess(collected)
    comparison = result["comparisons"]["C4"]
    assert comparison["unadjusted"]["outcome"] == "better" and comparison["verdict"] == "better"
    assert comparison["adjusted"]["outcome"] == "no clear difference"
    assert comparison["adjusted_claim_supported"] is False
    assert result["final_selection"]["selected"] == cell


def test_the_tie_band_is_02_points_from_the_top():
    gvp, fusion, selection = "only_gvp__four_class__wd10", "gvp_late_fusion__four_class__meanagg", \
        "only_gvp__four_class__baseline"
    low_minimum = {fusion: {0: 0.01, 1: 0.01, 2: 0.01, 3: -0.03}}  # same BA, lower minimum recall
    inside = assess(grid({gvp: 0.06, fusion: 0.0115}, selection=selection, shifts=low_minimum),
                    selection)["final_selection"]
    assert set(inside["eligible"]) == {gvp, fusion}
    assert inside["selected"] == gvp and inside["reason"] == "tie-breakers within the tie band"
    outside = assess(grid({gvp: 0.06, fusion: 0.0125}, selection=selection, shifts=low_minimum),
                     selection)["final_selection"]
    assert outside["selected"] == fusion
    assert outside["reason"] == "highest mean common-four BA on folds 1-4 among eligible candidates"


def test_the_tie_order_is_minimum_recall_worst_fold_sd_family_recipe():
    def row(family, recipe, *, minimum=0.70, worst=0.75, sd=0.01):
        return {"family": family, "recipe": recipe, "mean_min_recall": minimum, "worst_fold_common4_ba": worst,
                "sd_common4_ba": sd}

    pick = lambda *rows: min(rows, key=e2a.tie_key)["recipe"]  # noqa: E731
    assert pick(row("only_gvp", "a", minimum=0.69), row("gvp_late_fusion", "b")) == "b"
    assert pick(row("only_gvp", "a", worst=0.748, sd=0.008), row("gvp_late_fusion", "b", worst=0.75, sd=0.0115)) == "b"
    assert pick(row("only_gvp", "a", sd=0.012), row("gvp_late_fusion", "b", sd=0.011)) == "b"
    assert pick(row("gvp_late_fusion", "a"), row("only_gvp", "b")) == "b"
    assert pick(row("gvp_late_fusion", "wd10"), row("gvp_late_fusion", "meanagg")) == "meanagg"


def test_a4_parity_needs_identical_results_and_evidence(monkeypatch):
    finals = {"gvp_late_fusion": "headdrop03", "only_gvp": "meanagg"}
    names = e2a.a4_unit_names(finals)
    assert len(names) == 55
    collected = {name: {"status": "completed", "evidence": {"c": name}} for name in names}
    original = {"result": {"x": 1.0}, "unit_evidence": {name: {"c": name} for name in names}}
    monkeypatch.setattr(e2a.pa, "assess_step_e", lambda subset, a4, final_recipes: {"x": 1.0})
    assert e2a.a4_parity(collected, original, {}, finals)["equal"] is True
    monkeypatch.setattr(e2a.pa, "assess_step_e", lambda subset, a4, final_recipes: {"x": 1.000001})
    assert e2a.a4_parity(collected, original, {}, finals)["problems"] == ["/x: 1.0 != 1.000001"]
    monkeypatch.setattr(e2a.pa, "assess_step_e", lambda subset, a4, final_recipes: {"x": 1.0})
    changed = {**original, "unit_evidence": {**original["unit_evidence"], names[0]: {"c": "other"}}}
    assert e2a.a4_parity(collected, changed, {}, finals)["equal"] is False
    collected[names[3]] = {"status": "invalid evidence"}
    assert e2a.a4_parity(collected, original, {}, finals)["problems"][0].startswith("not verified")


def test_a_failed_a4_parity_check_withholds_the_final_selection(tmp_path, monkeypatch):
    cell = "gvp_late_fusion__four_class__gvpaux03"
    collected = grid({cell: 0.03})
    reused = e2.reused_unit_names(SPEC, FUSION_SELECTION)
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"result": {}, "unit_evidence": {}}))
    manifest = {**fake_manifest(), "e2_spec_sha256": e2.FROZEN_E2_SPEC_SHA256, "a4_spec_sha256": {},
                "reused_units": {name: {} for name in reused["folds_1_4"] + reused["development_fold0"]}}
    manifest["original_assessment"].update(path=str(original), improvement_recipes={
        "gvp_late_fusion": "headdrop03", "only_gvp": "meanagg"})
    manifest_path = tmp_path / "e2_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def view(directory, **kwargs):
        directory.mkdir(parents=True)
        (directory / "campaign_manifest.json").write_text("{}")
        return {}

    monkeypatch.setattr(e2a.e2, "verify_manifest", lambda m: [])
    monkeypatch.setattr(e2a, "check_evidence_copy", lambda path: {"path": str(path)})
    monkeypatch.setattr(e2a, "build_view", view)
    monkeypatch.setattr(e2a.pa, "check_spec_against_campaign", lambda spec, campaign_manifest: None)
    monkeypatch.setattr(e2a, "collect_verified", lambda *args, **kwargs: copy.deepcopy(collected))
    for equal in (False, True):
        monkeypatch.setattr(e2a, "a4_parity", lambda *args, equal=equal: {"equal": equal, "problems": []})
        written = json.loads(e2a.run(data=tmp_path, manifest_path=manifest_path, evidence_dir=tmp_path,
                                     out_dir=tmp_path / f"out_{equal}").read_text())
        result, labels = written["result"], [c["replacement"]["label"] for c in written["result"]["candidates"].values()]
        assert written["held_out_test_accessed"] is False and written["interpretation"] == e2a.interpretation()
        if equal:
            assert result["status"] == "complete" and result["final_selection"]["selected"] == cell
            assert result["final_selection"]["manifest_sha256"] == sha(manifest_path) and "selected" in labels
        else:
            assert result["status"].startswith("blocked: the original A4 result") and result["final_selection"] is None
            assert "selected" not in labels
            assert "eligible so far; no final selection (assessment blocked)" in labels


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m["units"].pop(), "units are not the specification's 36"),
    (lambda m: m["comparisons"].pop(), "comparisons are not the declared family"),
    (lambda m: m["a4_spec_sha256"].update({"assessment_spec.md": "0" * 64}), "A4 specification differs"),
    (lambda m: m.update(runner_sha256={**RUNNER, "pmm_v3_campaign.py": "0" * 64}), "runner files differ"),
    (lambda m: m["e2_spec_sha256"].update({"e2_assessment_spec.md": "0" * 64}), "E2 specification files differ"),
    (lambda m: m["tooling_sha256"].update({"pmm_v3_step_c.py": "0" * 64}), "reused launcher/assessor module changed"),
    (lambda m: m["tooling_sha256"].update({"pmm_v3_assessment.py": "0" * 64}), "reused launcher/assessor module"),
    (lambda m: rewrite(Path(m["final_step_d_assessment"]["path"]), lambda a: a.update(note="x")),
     "final_step_d_assessment"),
    (lambda m: rewrite(Path(m["extension"]["copy"]), lambda a: a.update(note="x")),
     "workstation copy of the extension record differs"),
])
def test_verify_manifest_reports_each_binding(tmp_path, mutate, message):
    manifest = e2.build_manifest(fake_campaign(tmp_path), log_entry="v3-099")
    assert e2.verify_manifest(manifest) == []
    mutate(manifest)
    assert any(message in problem for problem in e2.verify_manifest(manifest)), e2.verify_manifest(manifest)


def test_the_manifest_pins_the_reused_modules():
    assert set(e2.REUSED_FILES) == {"pmm_v3_assessment.py", "pmm_v3_step_b.py", "pmm_v3_step_c.py",
                                    "pmm_v3_step_d.py", "pmm_v3_host_pull.py"}
    assert set(e2.TOOL_FILES) == set(e2.REUSED_FILES) | {"pmm_v3_step_e2.py", "pmm_v3_e2_assessment.py"}
