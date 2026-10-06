"""Tests for extension 1 of the v3 campaign: step D Round R (regularization amendment, log v3-017)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import pmm_v3_assessment as assess  # noqa: E402
import pmm_v3_campaign as v3  # noqa: E402
from test_v3_campaign import campaign, fake_attempt, policy, settle  # noqa: E402,F401  (fixture reuse)

# The one trainer setting each Round R recipe changes from the baseline (playbook table).
ROUND_R_SETTINGS = {"wd001": ("weight_decay", 0.01), "wd01": ("weight_decay", 0.1), "wd10": ("weight_decay", 1.0),
                    "headdrop01": ("head_mlp_dropout", 0.1), "headdrop03": ("head_mlp_dropout", 0.3),
                    "resdrop02": ("gvp_residual_dropout", 0.2), "esmdrop04": ("esm_modality_dropout", 0.4)}


def resolved(paths, unit, train, esm_dir):
    from benchmarking import pmm_ion_campaign as v2
    from training.config import config_to_payload, parse_args

    command, _, identity = v3.build_command(paths, unit, python_bin=sys.executable, train_dir=train, esm_dir=esm_dir,
                                            device="cpu", lane=0)
    payload = config_to_payload(parse_args(command[3:]))
    return {k: v for k, v in payload.items() if k not in v2.NON_IDENTITY_CONFIG_KEYS}, identity


def test_round_r_inventory_matches_the_playbook_table():
    units = v3.step_units("D-R")
    assert len(units) == 26 and len(set(units)) == 26
    assert len(v3.step_units("D-A")) + len(units) == 42
    assert len(v3.step_units("D-A")) + len(units) + len(v3.step_units("D-B")) == 60
    per_recipe = {recipe: sum(u.recipe == recipe for u in units) for recipe in v3.ROUND_CANDIDATES["D-R"]}
    assert per_recipe == {"wd001": 4, "wd01": 4, "wd10": 4, "headdrop01": 4, "headdrop03": 4, "resdrop02": 4,
                          "esmdrop04": 2}
    assert all(u.target == "four_class" and u.fold == 0 and u.recipe != "baseline" for u in units)  # no new controls
    assert {u.seed for u in units} == set(v3.SEEDS) and {u.family for u in units} == set(v3.GVP_FAMILIES)
    assert v3.ROUND_ORDER == ("D-A", "D-R", "D-B") and set(v3.ROUND_CANDIDATES) == set(v3.ROUND_ORDER)
    assert set(ROUND_R_SETTINGS) == set(v3.EXTENSION_RECIPES) == set(v3.ROUND_CANDIDATES["D-R"])
    for step in ("D-A", "D-B"):  # the prepared rounds keep their candidates
        screened = {u.recipe for u in v3.step_units(step)} - {"baseline", *v3.SCREEN_CONTROLS.values()}
        assert screened == set(v3.ROUND_CANDIDATES[step])
    with pytest.raises(ValueError):
        v3.Unit("only_esm", "four_class", "wd01", 0, 42)  # Only-ESMC keeps its baseline recipe
    with pytest.raises(ValueError):
        v3.Unit("only_gvp", "four_class", "esmdrop04", 0, 42)


def test_each_round_r_recipe_changes_exactly_one_resolved_setting(campaign):
    train, esm_dir, paths, _, _ = campaign
    for unit in v3.step_units("D-R"):
        control, _ = resolved(paths, v3.Unit(unit.family, "four_class", "baseline", 0, unit.seed), train, esm_dir)
        candidate, identity = resolved(paths, unit, train, esm_dir)
        key, value = ROUND_R_SETTINGS[unit.recipe]
        assert {k for k in control if control[k] != candidate[k]} == {key}, unit.name
        assert candidate[key] == pytest.approx(value) and candidate["gvp_weight_decay"] is None
        assert identity["recipe_definition"]["flags"] == v3.EXTENSION_RECIPES[unit.recipe]["flags"]
    # The prepared baseline still resolves the prepared values.
    baseline, _ = resolved(paths, v3.Unit("gvp_late_fusion", "four_class", "baseline", 0, 43), train, esm_dir)
    assert (baseline["weight_decay"], baseline["head_mlp_dropout"], baseline["gvp_residual_dropout"],
            baseline["esm_modality_dropout"]) == (1e-4, 0.2, 0.0, 0.0)


def test_the_extension_is_recorded_once_and_binds_the_prepared_campaign(campaign, tmp_path, monkeypatch):
    train, esm_dir, paths, _, _ = campaign
    common = dict(lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                  execution_policy=policy(tmp_path), load_workers=1, epochs=2)
    with pytest.raises(ValueError, match="execution setting"):
        v3.extend_campaign(paths, epochs=2)
    settle(paths)
    control = v3.Unit("only_gvp", "four_class", "baseline", 0, 42)
    assert v3.run_unit(paths, control, **common)["status"] == "completed"
    before = json.loads((paths.lane(0) / f"run_status_{control.name}.json").read_text())["identity"]
    assert "campaign_extension" not in before
    candidate = v3.Unit("only_gvp", "four_class", "wd01", 0, 42)
    with pytest.raises(ValueError, match="record the extension first"):
        v3.run_unit(paths, candidate, **common)
    assert not v3.unit_artifacts(paths, candidate.name)
    manifest_before = paths.manifest.read_bytes()

    record = v3.extend_campaign(paths, epochs=2)
    assert paths.manifest.read_bytes() == manifest_before  # nothing prepared is rewritten
    manifest = json.loads(manifest_before)
    assert record["parent"]["manifest_sha256"] == v3.file_sha(paths.manifest)
    assert record["parent"]["runner_sha256"] == manifest["runner_sha256"]
    assert record["parent"]["unchanged_unit_identities"] == [control.name]
    assert record["parent"]["completed_runs"][control.name]["identity_sha256"] == v3.stable_hash(before)
    assert {step: len(names) for step, names in record["units"].items()} == {"D-A": 16, "D-R": 26, "D-B": 18}
    assert record["definitions"]["round_order"] == ["D-A", "D-R", "D-B"] and record["held_out_access"] is False
    with pytest.raises(ValueError, match="recorded once"):
        v3.extend_campaign(paths, epochs=2)

    # Every later run binds the frozen record; the reused control keeps its own identity.
    _, _, identity = v3.build_command(paths, candidate, python_bin=sys.executable, train_dir=train, esm_dir=esm_dir,
                                      device="cpu", lane=0, epochs=2)
    assert identity["campaign_extension"] == {"id": v3.EXTENSION_ID, "sha256": v3.file_sha(paths.extension)}
    assert v3.run_unit(paths, candidate, **dict(common, lane=1))["status"] == "completed"
    collected = assess.collect(paths, [control, candidate], epochs=2)
    assert collected[control.name]["status"] == collected[candidate.name]["status"] == "completed"
    assert collected[control.name]["runner_sha256"] == manifest["runner_sha256"]
    screen_inputs = {u.name: collected[u.name]["metrics"]["common4"]["balanced_accuracy"] for u in (control, candidate)}
    assert all(0.0 <= value <= 1.0 for value in screen_inputs.values())

    # Checked reuse: a control whose recorded identity changed after the freeze is refused.
    status_path = paths.lane(0) / f"run_status_{control.name}.json"
    original = status_path.read_text()
    tampered = json.loads(original)
    tampered["identity"]["model_seed"] = 43
    status_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="changed since the extension froze it"):
        assess.collect(paths, [control], epochs=2)
    status_path.write_text(original)

    # The frozen record refuses changed runner files, definitions, manifest and step-B setting.
    third = v3.Unit("only_gvp", "four_class", "headdrop03", 0, 42)
    with monkeypatch.context() as patch:
        patch.setattr(v3, "runner_sha256", lambda: {"pmm_v3_campaign.py": "0" * 64})
        with pytest.raises(ValueError, match="changed since the extension was recorded"):
            v3.run_unit(paths, third, **dict(common, lane=1))
    with monkeypatch.context() as patch:
        patch.setitem(v3.EXTENSION_RECIPES, "wd10", {"families": v3.GVP_FAMILIES, "flags": ["--weight-decay", "10.0"]})
        with pytest.raises(ValueError, match="Extension definitions changed"):
            v3.run_unit(paths, third, **dict(common, lane=1))
    with monkeypatch.context() as patch:
        patch.setattr(v3, "EXTENSION_EXCLUSIVE_GROUPS", v3.EXTENSION_EXCLUSIVE_GROUPS[1:])
        with pytest.raises(ValueError, match="Extension definitions changed"):
            v3.run_unit(paths, third, **dict(common, lane=1))
    with monkeypatch.context() as patch:
        patch.setitem(v3.RECIPES, "meanagg", {"families": v3.GVP_FAMILIES, "flags": ["--gvp-residual-dropout", "0.2"]})
        with pytest.raises(ValueError, match="Recipe definitions changed"):
            v3.run_unit(paths, third, **dict(common, lane=1))
    settings = paths.execution_settings.read_text()
    paths.execution_settings.write_text(settings.replace('"concurrent_lanes": 2', '"concurrent_lanes": 3'))
    with pytest.raises(ValueError, match="execution setting changed"):
        v3.run_unit(paths, third, **dict(common, lane=1))
    paths.execution_settings.write_text(settings)
    monkeypatch.setattr(assess, "FROZEN_SPEC_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        v3.run_unit(paths, third, **dict(common, lane=1))
    assert not v3.unit_artifacts(paths, third.name)  # every refusal came before the claim


def test_the_extension_is_refused_once_step_d_started_or_a_completed_unit_resolves_differently(campaign, tmp_path,
                                                                                              monkeypatch):
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    started = v3.Unit("gvp_late_fusion", "four_class", "meanagg", 0, 42).name
    fake_attempt(paths, started, "failed")
    with pytest.raises(ValueError, match="frozen before its first fit"):
        v3.extend_campaign(paths, epochs=2)
    v3.archive_failed_attempt(paths, started)
    with pytest.raises(ValueError, match="frozen before its first fit"):  # an archived attempt still counts
        v3.extend_campaign(paths, epochs=2)
    assert not paths.extension.exists()


def test_a_changed_baseline_command_blocks_the_extension(campaign, tmp_path, monkeypatch):
    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    control = v3.Unit("only_gvp", "four_class", "baseline", 0, 42)
    assert v3.run_unit(paths, control, lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable,
                       device="cpu", execution_policy=policy(tmp_path), load_workers=1, epochs=2)["status"] == "completed"
    monkeypatch.setitem(v3.PROFILE, "batch_size", 8)  # the code would now build another baseline command
    with pytest.raises(ValueError, match="Profile changed since preparation"):
        v3.extend_campaign(paths, epochs=2)
    monkeypatch.setitem(v3.PROFILE, "batch_size", 16)
    with pytest.raises(ValueError, match="resolves the completed unit differently in \\['epochs', 'resolved_config_sha256'\\]"):
        v3.extend_campaign(paths)  # 50 planned epochs against the recorded two-epoch toy fit
    assert not paths.extension.exists()
    assert v3.extend_campaign(paths, epochs=2)["parent"]["unchanged_unit_identities"] == [control.name]


def test_the_command_line_exposes_the_extension_and_round_r(campaign, capsys):
    import run_pmm_v3_campaign as cli

    _, _, paths, _, _ = campaign
    settle(paths)
    assert cli.main(["--action", "plan", "--campaign-dir", str(paths.root), "--step", "D-R"]) == 0
    assert json.loads(capsys.readouterr().out)["n_units"] == 26
    assert cli.main(["--action", "extend", "--campaign-dir", str(paths.root)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["extension_id"] == v3.EXTENSION_ID and printed["units"] == {"D-A": 16, "D-R": 26, "D-B": 18}
    assert printed["extension_sha256"] == v3.file_sha(paths.extension)
    with pytest.raises(ValueError, match="recorded once"):
        cli.main(["--action", "extend", "--campaign-dir", str(paths.root)])
