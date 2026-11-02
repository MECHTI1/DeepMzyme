"""WP-11 selection steps, L5 certification and the gap gate (synthetic run folders in tmp_path only)."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ec_clean30 import selector  # noqa: E402
from ec_clean30.campaign import CAMPAIGN_ID, Recipe, build_fold_plan, canonical_json, recipe_with_lever  # noqa: E402
from ec_clean30.commands import VERIFY_UNIT_RECEIPT_NAME  # noqa: E402
from ec_clean30.provenance import sha256_file  # noqa: E402
from ec_clean30.selection import (  # noqa: E402
    L9_PROMOTION_ALPHA, LEVER_NOT_APPLICABLE, LEVER_NOT_TESTED, LEVER_PASSED, PROMOTION_ALPHA, PooledOof,
    PromotionConfig, promotion_config_for, read_step_chain, screen_lever, verify_step_chain,
)
from ec_clean30.selector import SelectionInputError  # noqa: E402

PYTHON = sys.executable
FREEZE1 = "f1" * 32
FAST = PromotionConfig(n_draws=2000)
S1_WINNER = Recipe("F2", (("learning_rate", 1e-4),), 30)
N_PROTEINS = 48
IDS = tuple(f"Q{index:03d}" for index in range(N_PROTEINS))
TRUE = {pid: index % 6 for index, pid in enumerate(IDS)}
INNER = {pid: (index // 6) % 4 for index, pid in enumerate(IDS)}
EC_NUMBER = {pid: f"{TRUE[pid] + 1}.1.1.{(index // 6) % 3 + 1}" for index, pid in enumerate(IDS)}
COMPONENTS = {pid: f"comp{index // 2}" for index, pid in enumerate(IDS)}
# One uniform draw per (seed, protein), shared by every configuration: a configuration with accuracy a is
# right where u < a, so a better configuration is right on a superset of proteins.
_DRAWS = {seed: dict(zip(IDS, np.random.default_rng(seed).random(N_PROTEINS))) for seed in range(40, 47)}
TRAIN_COUNTS = {str(token): (6 if token <= 6 else 0) for token in range(1, 8)}
MEMBERSHIP = "ee" * 32  # the certified inner membership's file SHA-256 every synthetic fit was trained on


def _logits(pid, seed, accuracy):
    values = [0.0] * 7
    true = TRUE[pid]
    if _DRAWS[seed][pid] < accuracy:
        values[true], values[(true + 1) % 6] = 3.0, 1.0
    else:
        values[(true + 1) % 6], values[true] = 3.0, 1.0
    return values


def exporter_coverage(history):
    """The exporter's ec_supcon_coverage rule (training.ec_prediction_export.supcon_coverage)."""
    keys = (selector.SUPCON_BATCHES_KEY, selector.SUPCON_BATCHES_WITH_POSITIVE_KEY)
    complete = bool(history) and all(all(key in record for key in keys) for record in history)
    return {"train_batches": sum(int(r[keys[0]]) for r in history) if complete else None,
            "batches_with_positive": sum(int(r[keys[1]]) for r in history) if complete else None,
            "epochs_counted": sum(keys[0] in record for record in history), "definition": "synthetic"}


def write_fit(runs, *, unit_id, step, inner, seed, recipe, fold=0, accuracy=0.7, counters=None,
              checkpoint_history=None, checkpoint_config=None, receipt_changes=None, identity_changes=None):
    """One finished synthetic inner fit: runner record, run folder, receipts, checkpoint and protein rows."""
    import torch

    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / unit_id
    run_dir.mkdir()
    identity = {"campaign_id": CAMPAIGN_ID, "unit_id": unit_id, "step": step, "kind": "inner", "fold": fold,
                "inner_fold": inner, "seed": seed, "smoke": False, "recipe": recipe.to_dict(),
                "membership_sha256": MEMBERSHIP, "freeze1_sha256": FREEZE1, **(identity_changes or {})}
    runner = {"schema_version": 1, "unit_id": unit_id, "status": "completed", "returncode": 0,
              "unit": {"unit_id": unit_id, "kind": "inner", "step": identity["step"], "seed": identity["seed"],
                       "inner_fold": identity["inner_fold"], "recipe": identity["recipe"], "identity": identity}}
    (runs / f"{unit_id}.runner.json").write_text(json.dumps(runner, indent=2, sort_keys=True))
    config = {"seed": seed, "campaign_run_identity": canonical_json(identity)}
    supcon = "L5" in recipe.lever_configs
    if supcon:
        config.update({"ec_supcon_weight": 0.1, "ec_supcon_temperature": 0.1, "ec_supcon_memory": 256,
                       "ec_supcon_projection_dim": 128})
    epochs = recipe.epochs
    history = []
    for epoch in range(1, epochs + 1):
        record = {"epoch": epoch, "train_loss": 0.5}
        if supcon:
            batches, positive = (10, 10) if counters is None else counters(epoch)
            if batches is not None:
                record[selector.SUPCON_BATCHES_KEY] = batches
            if positive is not None:
                record[selector.SUPCON_BATCHES_WITH_POSITIVE_KEY] = positive
        history.append(record)
    (run_dir / "run_config.json").write_text(json.dumps({"config": config, "history": history}, sort_keys=True))
    checkpoint = run_dir / "terminal_model_checkpoint.pt"
    if supcon:
        saved = history if checkpoint_history is None else checkpoint_history(history)
        saved_config = config if checkpoint_config is None else checkpoint_config(dict(config))
        torch.save({"epoch": epochs, "history": saved, "config": saved_config}, checkpoint)
    else:
        checkpoint.write_bytes(f"checkpoint {unit_id}".encode())
    rows = ["run_id,mode,fold_index,n_folds,seed,accession,ec_group_key,structure_ids,n_pockets,y,pred,"
            + ",".join(selector.LOGIT_COLUMNS)]
    members = [pid for pid in IDS if INNER[pid] == inner]
    for pid in members:
        logits = _logits(pid, seed, accuracy)
        sid = f"{pid}__chain_A__EC_{EC_NUMBER[pid]}"
        if IDS.index(pid) % 5 == 0:  # two structures, one with a ";" EC list, joined with ";" as the exporter does
            sids = f"{pid}__chain_A__EC_{EC_NUMBER[pid]};{EC_NUMBER[pid]}", f"{pid}__chain_B__EC_{EC_NUMBER[pid]}"
            sid = ";".join(sids)
        rows.append(",".join([unit_id, "validation", str(inner), "4", str(seed), pid, pid, sid,
                              "2" if ";" in sid else "1", str(TRUE[pid]), str(int(np.argmax(logits))),
                              *[repr(value) for value in logits]]))
    csv_path = run_dir / "ec_val_protein_predictions.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    receipt = {
        "mode": "validation", "run_id": unit_id, "campaign_run_identity": identity, "fold_index": inner,
        "n_folds": 4, "seed": seed,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "epoch": epochs,
                       "checkpoint_rule": "terminal"},
        "run_config_sha256": sha256_file(run_dir / "run_config.json"),
        "training_protein_class_counts": dict(TRAIN_COUNTS),
        "epochs": {"planned": epochs, "completed": epochs},
        "ec_supcon_coverage": exporter_coverage(history) if supcon else None,
        "protein_predictions": {"path": csv_path.name, "sha256": sha256_file(csv_path), "n_rows": len(members)},
        "reconciliation_status": "match",
    }
    for key, value in (receipt_changes or {}).items():
        receipt[key] = value(receipt[key]) if callable(value) else value
    (run_dir / "ec_prediction_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return run_dir


def write_verify_receipt(runs, status="passed"):
    """The runs folder's verify_unit_receipt.json over every runner record in it."""
    units = []
    for path in sorted(runs.glob("*.runner.json")):
        unit_id = path.name[: -len(".runner.json")]
        checkpoint = runs / unit_id / "terminal_model_checkpoint.pt"
        units.append({"unit_id": unit_id, "passed": True, "terminal_checkpoint_sha256": sha256_file(checkpoint)})
    (runs / VERIFY_UNIT_RECEIPT_NAME).write_text(json.dumps({"status": status, "units": units}, indent=2))


def write_plan(root, fold=0):
    path = root / "plans" / f"fold{fold}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_fold_plan(fold), indent=2, sort_keys=True) + "\n")
    return path


def l5_id(inner, seed, fold=0, config="L5"):
    return f"ecc30__fold{fold}__S2__{config}__inner{inner}__seed{seed}"


L5_RECIPE = recipe_with_lever(S1_WINNER, "L5")
GRID = [(inner, seed) for inner in range(4) for seed in (42, 43)]


@pytest.fixture
def l5_world(tmp_path):
    """A frozen plan and a runs folder writer for the eight L5 fits; ``build(**per_fit)`` writes them."""
    root = tmp_path / CAMPAIGN_ID
    plan_path = write_plan(root)
    runs = root / "runs" / "fold0"

    def build(special=None, finalize=True, **defaults):
        for inner, seed in GRID:
            options = dict(defaults)
            options.update((special or {}).get((inner, seed), {}))
            write_fit(runs, unit_id=l5_id(inner, seed), step="S2", inner=inner, seed=seed, recipe=L5_RECIPE,
                      **options)
        if finalize:
            write_verify_receipt(runs)
        return runs

    plan = selector.load_frozen_plan(plan_path, 0)
    return {"root": root, "plan_path": plan_path, "plan": plan, "runs": runs, "build": build}


def certify(world, runs_dirs=None):
    return selector.certify_l5(world["plan"], runs_dirs or [world["runs"]], s1_winner=S1_WINNER,
                               freeze1_sha256=FREEZE1)


# ---------------------------------------------------------------------------
# Plan grid and constants


def test_required_l5_grid_comes_from_the_frozen_plan(l5_world, tmp_path):
    units = selector.required_lever_units(l5_world["plan"], "L5")
    assert [unit.unit_id for unit in units] == [l5_id(inner, seed) for inner, seed in GRID]
    assert all(unit.step == "S2" and unit.lever_configs == ("L5",) for unit in units)
    changed = json.loads(l5_world["plan_path"].read_text())
    changed["units"] = changed["units"][:-1]
    (tmp_path / "plans").mkdir()
    (tmp_path / "plans" / "fold0.json").write_text(json.dumps(changed))
    with pytest.raises(SelectionInputError, match="plan_sha256|does not verify"):
        selector.load_frozen_plan(tmp_path / "plans" / "fold0.json", 0)
    resealed = dict(changed)
    from ec_clean30.campaign import canonical_sha256

    resealed["plan_sha256"] = canonical_sha256({k: v for k, v in resealed.items() if k != "plan_sha256"})
    (tmp_path / "plans" / "fold0.json").write_text(json.dumps(resealed))
    with pytest.raises(SelectionInputError, match="differs from the live plan"):
        selector.load_frozen_plan(tmp_path / "plans" / "fold0.json", 0)
    with pytest.raises(SelectionInputError, match="plans/fold1.json"):
        selector.load_frozen_plan(l5_world["plan_path"], 1)


def test_file_names_and_counter_keys_match_the_trainer():
    from training import ec_prediction_export, loop

    assert selector.SUPCON_BATCHES_KEY == loop.SUPCON_BATCHES_KEY
    assert selector.SUPCON_BATCHES_WITH_POSITIVE_KEY == loop.SUPCON_BATCHES_WITH_POSITIVE_KEY
    assert selector.TERMINAL_CHECKPOINT_NAME == ec_prediction_export.TERMINAL_CHECKPOINT_NAME
    assert selector.EC_PREDICTION_RECEIPT_NAME == ec_prediction_export.EC_PREDICTION_RECEIPT_NAME
    assert selector.PROTEIN_PREDICTIONS_NAME == ec_prediction_export.PREDICTION_FILE_NAMES["validation"][1]
    assert selector.PROTEIN_PREDICTION_COLUMNS == ec_prediction_export.EC_PROTEIN_PREDICTION_COLUMNS
    from ec_clean30 import commands, logit_adjustment

    assert commands.runner_receipt_path(Path("runs"), "unit").name == "unit" + selector.RUNNER_SUFFIX
    assert selector.PRIOR_SMOOTHING == logit_adjustment.PRIOR_SMOOTHING
    counts = {str(token): token for token in range(1, 8)}
    expected = logit_adjustment.protein_class_priors(np.repeat(np.arange(7), np.arange(1, 8)))
    assert np.allclose(selector.training_priors(counts, "u"), expected)
    history = [{"epoch": 1, loop.SUPCON_BATCHES_KEY: 4, loop.SUPCON_BATCHES_WITH_POSITIVE_KEY: 3},
               {"epoch": 2, loop.SUPCON_BATCHES_KEY: True, loop.SUPCON_BATCHES_WITH_POSITIVE_KEY: 0}]
    reference = ec_prediction_export.supcon_coverage({"ec_supcon_weight": 0.1}, history)
    assert {key: reference[key] for key in ("train_batches", "batches_with_positive", "epochs_counted")} \
        == selector._exporter_totals(history)
    partial = [history[0], {"epoch": 2}]
    reference = ec_prediction_export.supcon_coverage({"ec_supcon_weight": 0.1}, partial)
    assert reference["train_batches"] is None and selector._exporter_totals(partial)["train_batches"] is None


def test_lever_promotion_levels_follow_d1():
    assert promotion_config_for("L9").alpha == L9_PROMOTION_ALPHA == 0.01
    assert promotion_config_for("L7").alpha == PROMOTION_ALPHA == 0.04
    assert promotion_config_for(None).alpha == promotion_config_for("L5").alpha == 0.04
    assert promotion_config_for("L9", FAST).n_draws == FAST.n_draws


# ---------------------------------------------------------------------------
# L5 certification


def test_eight_valid_fits_certify_with_the_minimum_share(l5_world):
    low = {(2, 43): {"counters": lambda epoch: (10, 10 if epoch > 3 else 7)}}
    l5_world["build"](special=low)
    result = certify(l5_world)
    assert result.fraction == pytest.approx(291 / 300) and result.testable
    assert len(result.fits) == 8 and "certified" in result.reason
    record = result.record()
    by_unit = {fit["unit_id"]: fit for fit in record["fits"]}
    assert set(by_unit) == {l5_id(inner, seed) for inner, seed in GRID}
    low_fit = by_unit[l5_id(2, 43)]
    assert low_fit["recomputed_totals"] == {"train_batches": 300, "batches_with_positive": 291, "epochs_counted": 30}
    assert low_fit["share"] == pytest.approx(0.97)
    run_dir = l5_world["runs"] / l5_id(2, 43)
    assert low_fit["checkpoint_sha256"] == sha256_file(run_dir / "terminal_model_checkpoint.pt")
    assert low_fit["run_config_sha256"] == sha256_file(run_dir / "run_config.json")
    assert low_fit["prediction_receipt_sha256"] == sha256_file(run_dir / "ec_prediction_receipt.json")
    assert all(fit["share"] == 1.0 for unit, fit in by_unit.items() if unit != l5_id(2, 43))


def _drop_fit(world):
    runs = world["runs"]
    shutil.rmtree(runs / l5_id(3, 43))
    (runs / f"{l5_id(3, 43)}.runner.json").unlink()


def _copy_to_second_folder(world):
    second = world["root"] / "runs" / "fold0_more"
    second.mkdir()
    for name in (l5_id(0, 42), f"{l5_id(0, 42)}.runner.json"):
        source = world["runs"] / name
        (shutil.copytree if source.is_dir() else shutil.copy)(source, second / name)
    write_verify_receipt(second)
    return [world["runs"], second]


def _extra(world, unit_id, *, inner, seed, fold=0, recipe=L5_RECIPE, step="S2"):
    write_fit(world["runs"], unit_id=unit_id, step=step, inner=inner, seed=seed, recipe=recipe, fold=fold)
    write_verify_receipt(world["runs"])


def _rewrite(world, inner, seed, **options):
    unit = l5_id(inner, seed)
    shutil.rmtree(world["runs"] / unit)
    (world["runs"] / f"{unit}.runner.json").unlink()
    write_fit(world["runs"], unit_id=unit, step="S2", inner=inner, seed=seed, **{"recipe": L5_RECIPE, **options})
    write_verify_receipt(world["runs"])


def _off_by_one(key):
    return lambda value: {**value, key: value[key] - 1}


def _tamper(world, name, change):
    path = world["runs"] / l5_id(1, 42) / name
    path.write_bytes(change(path.read_bytes()))


REFUSALS = {
    "seven_fits": (_drop_fit, "1 required fits are missing"),
    "same_unit_in_two_runs_folders": (_copy_to_second_folder, "two runs folders"),
    "two_units_same_inner_fold_and_seed": (
        lambda w: _extra(w, "ecc30__fold0__S2__L5b__inner1__seed43", inner=1, seed=43),
        "duplicate fits of one recipe at the same inner fold and seed"),
    "foreign_fold": (lambda w: _extra(w, l5_id(0, 42, fold=1), inner=0, seed=42, fold=1), "Foreign fit"),
    "seed_44": (lambda w: _extra(w, l5_id(0, 44), inner=0, seed=44), "foreign fits not in the frozen plan's grid"),
    "inner_4": (lambda w: _extra(w, l5_id(4, 42), inner=4, seed=42), "foreign fits not in the frozen plan's grid"),
    "l5_fit_of_another_step": (
        lambda w: _extra(w, "ecc30__fold0__S3__combination__inner0__seed40", inner=0, seed=40, step="S3"),
        "foreign fits"),
    "different_recipe": (
        lambda w: _rewrite(w, 2, 42, recipe=recipe_with_lever(Recipe("F2", (("learning_rate", 3e-4),), 30), "L5")),
        "identity disagrees with the plan unit in \\['recipe'\\]"),
    "identity_seed_disagrees": (lambda w: _rewrite(w, 2, 42, identity_changes={"seed": 43}),
                                "duplicate fits|identity disagrees"),
    "tampered_run_config": (lambda w: _tamper(w, "run_config.json", lambda data: data.replace(b"0.5", b"0.6")),
                            "run_config_sha256"),
    "tampered_checkpoint": (lambda w: _tamper(w, "terminal_model_checkpoint.pt", lambda data: data + b"\0"),
                            "terminal checkpoint SHA-256"),
    "receipt_identity_mismatch": (
        lambda w: _rewrite(w, 1, 42, receipt_changes={"campaign_run_identity": lambda value: {**value, "seed": 99}}),
        "campaign_run_identity"),
    "checkpoint_counters_disagree": (
        lambda w: _rewrite(w, 3, 42, checkpoint_history=lambda history: [
            {**record, selector.SUPCON_BATCHES_WITH_POSITIVE_KEY: 9} if record["epoch"] == 5 else record
            for record in history]),
        "terminal checkpoint's per-epoch counters disagree"),
    "receipt_totals_off_by_one": (
        lambda w: _rewrite(w, 0, 43, receipt_changes={"ec_supcon_coverage": _off_by_one("batches_with_positive")}),
        "ec_supcon_coverage disagrees"),
    "receipt_batches_off_by_one": (
        lambda w: _rewrite(w, 2, 43, receipt_changes={"ec_supcon_coverage": _off_by_one("train_batches")}),
        "ec_supcon_coverage disagrees"),
    "identity_step_disagrees": (lambda w: _rewrite(w, 2, 42, identity_changes={"step": "S3"}),
                                "identity disagrees with the plan unit in \\['step'\\]"),
    "identity_fold_disagrees": (lambda w: _rewrite(w, 2, 42, identity_changes={"fold": 1}), "Foreign fit"),
    "identity_freeze1_disagrees": (lambda w: _rewrite(w, 2, 42, identity_changes={"freeze1_sha256": "ab" * 32}),
                                   "identity disagrees with the plan unit in \\['freeze1_sha256'\\]"),
    "tampered_checkpoint_after_a_fresh_verify_receipt": (
        lambda w: (_tamper(w, "terminal_model_checkpoint.pt", lambda data: data + b"\0"),
                   write_verify_receipt(w["runs"]))[0],
        "prediction receipt's checkpoint SHA-256 differs"),
    "checkpoint_configuration_disagrees": (
        lambda w: _rewrite(w, 3, 43, checkpoint_config=lambda config: {**config, "ec_supcon_memory": 128}),
        "terminal checkpoint's configuration differs"),
    "checkpoint_epoch_records_disagree": (
        lambda w: _rewrite(w, 3, 43, checkpoint_history=lambda history: history[:-1]),
        "terminal checkpoint's per-epoch counters disagree"),
    "receipt_epochs_counted_off": (
        lambda w: _rewrite(w, 0, 43, receipt_changes={
            "ec_supcon_coverage": lambda value: {**value, "epochs_counted": 29}}),
        "ec_supcon_coverage disagrees"),
    "receipt_without_coverage": (lambda w: _rewrite(w, 0, 43, receipt_changes={"ec_supcon_coverage": None}),
                                 "no ec_supcon_coverage"),
    "verify_receipt_failed": (lambda w: write_verify_receipt(w["runs"], status="failed"), "not 'passed'"),
    "verify_receipt_omits_a_fit": (lambda w: _verify_receipt_entry(w, l5_id(2, 42), None),
                                   "is not covered as passed by"),
    "verify_receipt_marks_a_fit_failed": (lambda w: _verify_receipt_entry(w, l5_id(2, 42), False),
                                          "is not covered as passed by"),
    "tampered_protein_predictions": (
        lambda w: _tamper(w, "ec_val_protein_predictions.csv", lambda data: data + b"\n"),
        "differs from its prediction receipt"),
    "history_shorter_than_the_completed_epochs": (
        lambda w: _reseal(w, 2, 43, lambda history: history[:-1]), "holds 29 epoch records"),
    "wrong_supcon_settings": (
        lambda w: _tamper_config(w), "does not carry the plan's L5 settings"),
}


def _verify_receipt_entry(world, unit_id, passed):
    """A passed verify receipt that drops ``unit_id`` (``passed`` None) or marks it with ``passed``."""
    write_verify_receipt(world["runs"])
    path = world["runs"] / VERIFY_UNIT_RECEIPT_NAME
    receipt = json.loads(path.read_text())
    units = [entry for entry in receipt["units"] if passed is not None or entry["unit_id"] != unit_id]
    receipt["units"] = [{**entry, "passed": passed} if entry["unit_id"] == unit_id else entry for entry in units]
    path.write_text(json.dumps(receipt, indent=2))


def _reseal(world, inner, seed, change_history):
    """Change one L5 fit's history in run_config.json and the checkpoint alike, then reseal every hash and the
    coverage totals (the receipt's completed epochs stay as written)."""
    import torch

    run_dir = world["runs"] / l5_id(inner, seed)
    payload = json.loads((run_dir / "run_config.json").read_text())
    payload["history"] = change_history(payload["history"])
    (run_dir / "run_config.json").write_text(json.dumps(payload, sort_keys=True))
    saved = torch.load(run_dir / "terminal_model_checkpoint.pt", weights_only=False)
    saved["history"] = change_history(saved["history"])
    torch.save(saved, run_dir / "terminal_model_checkpoint.pt")
    receipt = json.loads((run_dir / "ec_prediction_receipt.json").read_text())
    receipt["run_config_sha256"] = sha256_file(run_dir / "run_config.json")
    receipt["checkpoint"]["sha256"] = sha256_file(run_dir / "terminal_model_checkpoint.pt")
    receipt["ec_supcon_coverage"] = exporter_coverage(payload["history"])
    (run_dir / "ec_prediction_receipt.json").write_text(json.dumps(receipt))
    write_verify_receipt(world["runs"])


def _tamper_config(world):
    """A fit whose run configuration (and receipt hash) carry another SupCon weight: a consistent, wrong fit."""
    run_dir = world["runs"] / l5_id(1, 43)
    payload = json.loads((run_dir / "run_config.json").read_text())
    payload["config"]["ec_supcon_weight"] = 0.2
    (run_dir / "run_config.json").write_text(json.dumps(payload, sort_keys=True))
    receipt = json.loads((run_dir / "ec_prediction_receipt.json").read_text())
    receipt["run_config_sha256"] = sha256_file(run_dir / "run_config.json")
    (run_dir / "ec_prediction_receipt.json").write_text(json.dumps(receipt))


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_integrity_problems_refuse_the_l5_certification(l5_world, case):
    mutate, message = REFUSALS[case]
    l5_world["build"]()
    runs_dirs = mutate(l5_world)
    with pytest.raises(SelectionInputError, match=message):
        certify(l5_world, runs_dirs if isinstance(runs_dirs, list) else None)


UNCERTIFIED = {
    "no_counter_keys": ({"counters": lambda epoch: (None, None)}, "lacks the SupCon counters"),
    "one_epoch_without_positive_key": ({"counters": lambda epoch: (10, None if epoch == 7 else 10)},
                                       "lacks the SupCon counters"),
    "boolean_counter": ({"counters": lambda epoch: (True if epoch == 2 else 10, 1 if epoch == 2 else 10)},
                        "booleans refused"),
    "more_positives_than_batches": ({"counters": lambda epoch: (10, 11 if epoch == 4 else 10)},
                                    "11 minibatches with a positive of 10"),
    "zero_batches": ({"counters": lambda epoch: (0 if epoch == 9 else 10, 0 if epoch == 9 else 10)},
                     "counts 0 training minibatches"),
}


@pytest.mark.parametrize("case", sorted(UNCERTIFIED))
def test_invalid_counters_leave_l5_uncertified(l5_world, case):
    options, problem = UNCERTIFIED[case]
    l5_world["build"](special={(1, 42): options})
    result = certify(l5_world)
    assert result.fraction is None and not result.testable
    assert result.reason.startswith("L5 uncertified: 1 of 8") and problem in result.reason
    record = result.record()
    assert record["aggregation"]["uncertified_fits"] == [l5_id(1, 42)]
    failing = next(fit for fit in record["fits"] if fit["unit_id"] == l5_id(1, 42))
    assert failing["certified_coverage"] is None and failing["share"] is None and failing["problems"]


def test_non_contiguous_epochs_leave_l5_uncertified(l5_world):
    def renumber(history):
        return [{**record, "epoch": record["epoch"] + (1 if record["epoch"] >= 10 else 0)} for record in history]

    l5_world["build"]()
    _reseal(l5_world, 0, 42, renumber)  # same length as the completed epochs: uncertified, not refused
    result = certify(l5_world)
    assert result.fraction is None and "epochs are not 1..E in order" in result.reason


def _tiny_pooled(name):
    ids = tuple(f"t{k}" for k in range(6))
    return PooledOof(name, ids, np.arange(6), (42, 43), np.zeros((2, 6, 7)))


@pytest.mark.parametrize("positives, testable", [(267, False), (270, True), (273, True)])
def test_ninety_percent_boundary_over_the_eight_fits(l5_world, positives, testable):
    # 30 epochs x 10 minibatches = 300; the low fit has `positives` of them (the shortfall taken from the first
    # epochs), every other fit all 300.
    shortfall = 300 - positives
    per_epoch = [10 - min(10, max(0, shortfall - 10 * (epoch - 1))) for epoch in range(1, 31)]
    assert sum(per_epoch) == positives
    low = {(3, 43): {"counters": lambda epoch: (10, per_epoch[epoch - 1])}}
    l5_world["build"](special=low)
    result = certify(l5_world)
    assert result.fraction == pytest.approx(positives / 300) and result.testable is testable
    if not testable:
        pooled = _tiny_pooled("s1")
        outcome = screen_lever("L5", "F2", pooled, pooled, component_of={pid: pid for pid in pooled.protein_ids},
                               training_class_counts=[1] * 7, positive_step_fraction=result.fraction)
        assert outcome.status == LEVER_NOT_TESTED and "89.0%" in outcome.reason


# ---------------------------------------------------------------------------
# select S1 -> S2 -> S3 and the gap gate on one synthetic fold


def _s1_world(root, runs, *, family_accuracies=None, identity_changes=None):
    plan = build_fold_plan(0)
    for unit in plan["units"]:
        if unit["step"] != "S1":
            continue
        recipe = Recipe(unit["family"], tuple(unit["learning_rate"].items()), unit["epochs"])
        write_fit(runs, unit_id=unit["unit_id"], step="S1", inner=unit["inner_fold"], seed=unit["seed"], recipe=recipe,
                  accuracy=(family_accuracies or {}).get(unit["family"], 0.7), identity_changes=identity_changes)


def _s2_world(runs, family_recipe, accuracies, special=None, *, identity_changes=None):
    """The S2 fits of the applicable levers; ``special`` maps (config, inner, seed) to write_fit options."""
    for config_id in ("L3", "L4", "L5", "L6", "L7e12", "L7e20"):
        recipe = recipe_with_lever(family_recipe, config_id)
        for inner, seed in GRID:
            options = {"identity_changes": identity_changes, **(special or {}).get((config_id, inner, seed), {})}
            write_fit(runs, unit_id=l5_id(inner, seed, config=config_id), step="S2", inner=inner, seed=seed,
                      recipe=recipe, accuracy=accuracies.get(config_id, 0.7),
                      **options)


TABLES = selector.InnerTables(COMPONENTS, INNER, {"source": "synthetic"}, MEMBERSHIP)


def _select(world, step, out, tables=TABLES, **options):
    return selector.run_select(fold=0, step=step, plan_path=world["plan_path"], runs_dirs=[world["runs"]],
                               tables=tables, freeze1_sha256=FREEZE1, out=out, provenance={"note": "synthetic"},
                               base_config=FAST, **options)


def _s1_selected(tmp_path):
    root = tmp_path / CAMPAIGN_ID
    world = {"root": root, "plan_path": write_plan(root), "runs": root / "runs" / "fold0"}
    _s1_world(root, world["runs"])
    write_verify_receipt(world["runs"])
    out = tmp_path / "records" / "fold0_steps.jsonl"
    out.parent.mkdir()
    _select(world, "S1", out)
    return world, out


@pytest.fixture
def selected_fold(tmp_path):
    root = tmp_path / CAMPAIGN_ID
    world = {"root": root, "plan_path": write_plan(root), "runs": root / "runs" / "fold0"}
    _s1_world(root, world["runs"])
    write_verify_receipt(world["runs"])
    out = tmp_path / "records" / "fold0_steps.jsonl"
    out.parent.mkdir()
    s1 = _select(world, "S1", out)
    _s2_world(world["runs"], S1_WINNER, {"L4": 0.95})
    write_verify_receipt(world["runs"])
    s2 = _select(world, "S2", out)
    return world, out, s1, s2


def test_select_appends_hash_chained_s1_s2_s3_records(selected_fold):
    world, out, s1, s2 = selected_fold
    payload = s1["payload"]
    assert s1["index"] == 0 and s1["previous_sha256"] == FREEZE1 and s1["step"] == "S1"
    assert payload["s1_winner"] == S1_WINNER.to_dict() and payload["selected"]["config"] == "F2_lr1e-4_e30"
    assert payload["promotion_config"]["alpha"] == 0.04 and payload["plan_sha256"] == build_fold_plan(0)["plan_sha256"]
    assert len(payload["configurations"]) == 12
    assert s2["index"] == 1 and s2["previous_sha256"] != FREEZE1
    outcomes = {item["lever"]: item for item in s2["payload"]["screen"]["outcomes"]}
    assert {lever: outcomes[lever]["status"] for lever in ("L1", "L2", "L8")} == dict.fromkeys(
        ("L1", "L2", "L8"), LEVER_NOT_APPLICABLE)
    assert outcomes["L9"]["decision"]["config"]["alpha"] == 0.01
    assert outcomes["L7"]["decision"]["config"]["alpha"] == 0.04
    assert outcomes["L4"]["status"] == LEVER_PASSED and outcomes["L4"]["decision"]["config"]["alpha"] == 0.04
    assert outcomes["L5"]["details"]["positive_step_fraction"] == 1.0
    assert outcomes["L5"]["details"]["certification"]["testable"] is True
    assert len(outcomes["L5"]["details"]["certification"]["fits"]) == 8
    assert s2["payload"]["promotion_alpha_by_lever"]["L9"] == 0.01
    s3 = _select(world, "S3", out)
    assert s3["payload"]["passing_levers"] == ["L4"] and s3["payload"]["selected"]["config"] == "L4"
    assert s3["payload"]["combination"]["selected"] == "L4" and s3["payload"]["promotion_config"]["alpha"] == 0.04
    records = read_step_chain(out, anchor_sha256=FREEZE1)
    assert [record["step"] for record in records] == ["S1", "S2", "S3"]
    assert verify_step_chain(records, anchor_sha256=FREEZE1)
    with pytest.raises(ValueError, match="does not link"):
        read_step_chain(out, anchor_sha256="0" * 64)
    with pytest.raises(SelectionInputError, match="already holds the S2 record"):
        _select(world, "S2", out)
    text = out.read_text()
    assert not any(pid in text for pid in IDS)  # step records name no protein


def test_select_s2_refuses_a_stray_fit_and_an_unrecorded_s1(selected_fold, tmp_path):
    world, out, _s1, _s2 = selected_fold
    with pytest.raises(SelectionInputError, match="needs the S1 record"):
        _select(world, "S2", tmp_path / "other_chain.jsonl")
    lines = out.read_text().splitlines()
    partial = tmp_path / "partial.jsonl"
    partial.write_text(lines[0] + "\n")
    _extra({"runs": world["runs"]}, l5_id(0, 42, config="L1"), inner=0, seed=42,
           recipe=Recipe("F2", (("learning_rate", 1e-4),), 30, ("L6",)))
    with pytest.raises(SelectionInputError, match="foreign fits not in the applicable levers' grid"):
        _select(world, "S2", partial)


def test_select_s2_logs_an_uncertified_l5_as_not_tested(tmp_path):
    world, out = _s1_selected(tmp_path)
    # L5 would pass on its scores; one fit without counters keeps it untested.
    _s2_world(world["runs"], S1_WINNER, {"L5": 0.95},
              special={("L5", 1, 42): {"counters": lambda epoch: (None, None)}})
    write_verify_receipt(world["runs"])
    s2 = _select(world, "S2", out)
    outcome = next(item for item in s2["payload"]["screen"]["outcomes"] if item["lever"] == "L5")
    assert outcome["status"] == LEVER_NOT_TESTED and outcome["decision"] is None
    assert outcome["reason"].startswith("L5 uncertified: 1 of 8 required fits")
    certification = outcome["details"]["certification"]
    assert certification["positive_step_fraction"] is None and certification["testable"] is False
    assert [fit["unit_id"] for fit in certification["fits"] if fit["share"] is None] == [l5_id(1, 42)]
    assert "L5" not in s2["payload"]["screen"]["passing"]
    assert s2["payload"]["l5_certification"] == certification
    assert verify_step_chain(read_step_chain(out, anchor_sha256=FREEZE1), anchor_sha256=FREEZE1)


def test_select_s2_refuses_an_l5_receipt_whose_totals_disagree_without_a_record(tmp_path):
    world, out = _s1_selected(tmp_path)
    _s2_world(world["runs"], S1_WINNER, {},
              special={("L5", 0, 43): {"receipt_changes": {"ec_supcon_coverage": _off_by_one("train_batches")}}})
    write_verify_receipt(world["runs"])
    with pytest.raises(SelectionInputError, match="ec_supcon_coverage disagrees"):
        _select(world, "S2", out)
    assert [record["step"] for record in read_step_chain(out, anchor_sha256=FREEZE1)] == ["S1"]


@pytest.mark.parametrize("s3_positive", [8, None])
def test_select_s3_combines_two_levers_and_recertifies_l5(tmp_path, s3_positive):
    world, out = _s1_selected(tmp_path)
    _s2_world(world["runs"], S1_WINNER, {"L4": 0.95, "L5": 0.95})
    write_verify_receipt(world["runs"])
    s2 = _select(world, "S2", out)
    assert s2["payload"]["screen"]["passing"] == ["L4", "L5"]
    combined = Recipe("F2", (("learning_rate", 1e-4),), 30, ("L4", "L5"))
    for inner, seed in GRID:
        write_fit(world["runs"], unit_id=f"ecc30__fold0__S3__combination__inner{inner}__seed{seed}", step="S3",
                  inner=inner, seed=seed, recipe=combined, accuracy=1.0,
                  counters=lambda epoch: (10, s3_positive))
    write_verify_receipt(world["runs"])
    backup = tmp_path / "backup"
    shutil.copytree(world["runs"], backup)

    def restore():
        shutil.rmtree(world["runs"])
        shutil.copytree(backup, world["runs"])

    # An L5 fit rewritten without counters (same predictions): S3 refuses to build on the S2 certification.
    _rewrite({"runs": world["runs"]}, 1, 42, accuracy=0.95, counters=lambda epoch: (None, None))
    with pytest.raises(SelectionInputError, match="no longer certify"):
        _select(world, "S3", out)
    # Replaced with other counters at the same minimum share and the same predictions: refused by its hashes.
    _rewrite({"runs": world["runs"]}, 1, 42, accuracy=0.95, counters=lambda epoch: (20, 20))
    with pytest.raises(SelectionInputError, match="differ from the ones the S2 record certified"):
        _select(world, "S3", out)
    restore()
    s3 = _select(world, "S3", out)
    payload = s3["payload"]
    assert payload["passing_levers"] == ["L4", "L5"] and payload["combined_recipe"] == combined.to_dict()
    assert payload["combination"]["decision"]["config"]["alpha"] == 0.04
    assert payload["combination"]["best_single_lever"] == "L4"
    descriptive = payload["combination_supcon_coverage"]
    assert descriptive["role"] == "descriptive_only" and len(descriptive["fits"]) == 8
    assert descriptive["positive_step_fraction"] == (None if s3_positive is None else 0.8)
    assert payload["combination_source"]["step"] == "S3"
    expected = "combination" if payload["combination"]["decision"]["promoted"] else "L4"
    assert payload["selected"]["config"] == expected
    assert [record["step"] for record in read_step_chain(out, anchor_sha256=FREEZE1)] == ["S1", "S2", "S3"]


def test_select_refuses_fits_of_another_inner_membership(tmp_path):
    root = tmp_path / CAMPAIGN_ID
    world = {"root": root, "plan_path": write_plan(root), "runs": root / "runs" / "fold0"}
    _s1_world(root, world["runs"])
    write_verify_receipt(world["runs"])
    out = tmp_path / "steps.jsonl"
    other = selector.InnerTables(COMPONENTS, INNER, {"source": "synthetic"}, "ab" * 32)
    with pytest.raises(SelectionInputError, match="another inner membership"):
        _select(world, "S1", out, tables=other)
    moved = {**INNER, IDS[0]: (INNER[IDS[0]] + 1) % 4}
    with pytest.raises(SelectionInputError, match="come from another inner fold"):
        _select(world, "S1", out, tables=selector.InnerTables(COMPONENTS, moved, {}, MEMBERSHIP))
    partial = {pid: component for pid, component in COMPONENTS.items() if pid != IDS[5]}
    with pytest.raises(SelectionInputError, match="1 pooled proteins have no inner identity component") as caught:
        _select(world, "S1", out, tables=selector.InnerTables(partial, INNER, {}, MEMBERSHIP))
    assert IDS[5] not in str(caught.value)
    assert not out.exists()


def test_select_refuses_a_chain_that_does_not_verify(tmp_path):
    world, out = _s1_selected(tmp_path)
    other_anchor = "ab" * 32
    with pytest.raises(SelectionInputError, match="step chain does not verify"):
        selector.run_select(fold=0, step="S2", plan_path=world["plan_path"], runs_dirs=[world["runs"]], tables=TABLES,
                            freeze1_sha256=other_anchor, out=out, provenance={"note": "synthetic"}, base_config=FAST)
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text(out.read_text() + "{not json\n")
    with pytest.raises(SelectionInputError, match="step chain does not verify"):
        _select(world, "S2", corrupt)
    record = json.loads(out.read_text().splitlines()[0])
    unlinked = tmp_path / "unlinked.jsonl"
    unlinked.write_text(json.dumps({**record, "previous_sha256": "cd" * 32}) + "\n")
    with pytest.raises(SelectionInputError, match="does not link"):
        _select(world, "S2", unlinked)
    assert [item["step"] for item in read_step_chain(out, anchor_sha256=FREEZE1)] == ["S1"]


def test_select_s2_refuses_other_inner_tables_than_the_s1_record(tmp_path):
    world, out = _s1_selected(tmp_path)
    other = selector.InnerTables(COMPONENTS, INNER, {"source": "synthetic", "components_receipt_sha256": "ab" * 32},
                                 MEMBERSHIP)
    with pytest.raises(SelectionInputError, match="--components differs from the inner tables the S1 record"):
        _select(world, "S2", out, tables=other)
    moved = selector.InnerTables(COMPONENTS, INNER, {"source": "synthetic", "folder": "elsewhere"}, MEMBERSHIP)
    assert selector._components_binding(moved.record) == selector._components_binding(TABLES.record)
    assert len(read_step_chain(out, anchor_sha256=FREEZE1)) == 1


def test_record_outputs_inside_an_overlay_or_the_components_folder_are_refused(tmp_path):
    overlay = tmp_path / "overlay" / "ecc30-overlay-v1"
    (overlay / "identity" / "fold0").mkdir(parents=True)
    (overlay / "overlay_manifest.json").write_text("{}")
    inside = overlay / "identity" / "fold0" / "steps.jsonl"
    with pytest.raises(SelectionInputError, match="inside a data overlay"):
        selector.run_select(fold=0, step="S1", plan_path=tmp_path / "plans" / "fold0.json", runs_dirs=[],
                            tables=TABLES, freeze1_sha256=FREEZE1, out=inside)
    components = tmp_path / "work" / "fold0"
    components.mkdir(parents=True)
    tables = selector.InnerTables(COMPONENTS, INNER, {"folder": str(components)}, MEMBERSHIP)
    with pytest.raises(SelectionInputError, match="inside the --components folder of fold 0"):
        selector.run_select(fold=0, step="S1", plan_path=tmp_path / "plans" / "fold0.json", runs_dirs=[],
                            tables=tables, freeze1_sha256=FREEZE1, out=components / "steps.jsonl")
    with pytest.raises(SelectionInputError, match="inside a data overlay"):
        selector.run_gap_gate(tables=TABLES, fold=0, after="S1", step_chain=tmp_path / "steps.jsonl", freeze1_sha256=FREEZE1,
                              plan_path=tmp_path / "plans" / "fold0.json", runs_dirs=[], clean_r1_inner=[],
                              clean_r2_inner=[], out=overlay / "gate.json")
    assert not inside.exists() and not (overlay / "gate.json").exists()


def test_structure_ids_split_keeps_ec_lists_with_semicolons():
    joined = "P1__chain_A__EC_1.1.1.1;2.7.1.1;P1__chain_B__EC_1.1.1.1"
    assert selector.split_structure_ids(joined) == ("P1__chain_A__EC_1.1.1.1;2.7.1.1", "P1__chain_B__EC_1.1.1.1")
    assert selector.split_structure_ids("P1__chain_A__EC_3.4.21.4") == ("P1__chain_A__EC_3.4.21.4",)
    with pytest.raises(ValueError):
        selector.split_structure_ids("2.7.1.1;P1__chain_A__EC_1.1.1.1")


def test_l3_without_fits_needs_verified_cost_evidence(tmp_path):
    root = tmp_path / CAMPAIGN_ID
    world = {"root": root, "plan_path": write_plan(root), "runs": root / "runs" / "fold0"}
    _s1_world(root, world["runs"])
    write_verify_receipt(world["runs"])
    out = tmp_path / "steps.jsonl"
    _select(world, "S1", out)
    _s2_world(world["runs"], S1_WINNER, {})
    for inner, seed in GRID:
        shutil.rmtree(world["runs"] / l5_id(inner, seed, config="L3"))
        (world["runs"] / f"{l5_id(inner, seed, config='L3')}.runner.json").unlink()
    write_verify_receipt(world["runs"])
    with pytest.raises(SelectionInputError, match="L3 has no fits: a verified Freeze-1c"):
        _select(world, "S2", out)
    assert len(read_step_chain(out, anchor_sha256=FREEZE1)) == 1


def _write_clean_exports(folder, arm, *, tie=None, drop_ec=None, wrong=()):
    from ec_clean30 import scoring

    comparator = scoring.comparator
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for inner in range(4):
        queries = [pid for pid in IDS if INNER[pid] == inner]
        labels = sorted({EC_NUMBER[pid] for pid in IDS} - ({drop_ec} if drop_ec else set()))
        distances = np.full((len(queries), len(labels)), 5.0, dtype=np.float32)
        for row, pid in enumerate(queries):
            target = EC_NUMBER[pid]
            if pid in wrong or target not in labels:
                target = next(label for label in labels if not label.startswith(f"{TRUE[pid] + 1}."))
            distances[row, labels.index(target)] = 1.0
            if pid == tie:
                other = next(label for label in labels if not label.startswith(f"{TRUE[pid] + 1}."))
                distances[row, labels.index(other)] = 1.0
        metadata = {"fold": 0, "regime": arm, "seed": 7, "input_dim": 4, "embedding": "synthetic", "mode": "validation",
                    "checkpoint_sha256": "a" * 64, "training_table_sha256": "b" * 64,
                    "embedding_farm_sha256": "c" * 64, "query_embeddings_sha256": "d" * 64, "inner_fold": inner,
                    "n_in_sample_queries": 0}
        path = folder / f"{arm}_fold0_inner{inner}.npz"
        comparator.write_distance_export(path, distances, queries, labels, metadata)
        paths.append(path)
    return paths


def test_gap_gate_after_s1_and_s3_on_the_r1_inner_closure(selected_fold, tmp_path):
    world, out, s1, _s2 = selected_fold
    _select(world, "S3", out)
    dropped = EC_NUMBER[IDS[0]]  # proteins with this EC number leave the closure
    r1 = _write_clean_exports(tmp_path / "clean", "r1_inner", tie=IDS[1], drop_ec=dropped)
    r2 = _write_clean_exports(tmp_path / "clean", "r2_inner", wrong=IDS[:12])

    def gate(after, name):
        return selector.run_gap_gate(tables=TABLES, fold=0, after=after, step_chain=out, freeze1_sha256=FREEZE1,
                                     plan_path=world["plan_path"], runs_dirs=[world["runs"]], clean_r1_inner=r1,
                                     clean_r2_inner=r2, out=tmp_path / name, provenance={"note": "synthetic"})

    after_s1 = gate("S1", "gate_s1.json")
    n_out = sum(1 for pid in IDS if EC_NUMBER[pid] == dropped)
    assert after_s1["denominators"]["n_inner_validation_proteins"] == N_PROTEINS
    assert after_s1["denominators"]["n_in_r1_inner_closure"] == N_PROTEINS - n_out
    assert after_s1["systems"]["clean_r1_inner"]["n_ties_counted_wrong"] == 1
    kept = [pid for pid in IDS if EC_NUMBER[pid] != dropped]
    true = np.array([TRUE[pid] for pid in kept])
    clean_correct = np.array([pid != IDS[1] for pid in kept])
    expected_clean = selector.correctness_balanced_accuracy(true, clean_correct)[0]
    assert after_s1["systems"]["clean_r1_inner"]["ec_classes_1_6_ba_points"] == pytest.approx(expected_clean)
    dm = np.mean([selector.correctness_balanced_accuracy(true, np.array([_DRAWS[seed][pid] < 0.7 for pid in kept]))[0]
                  for seed in (42, 43)])
    assert after_s1["systems"]["deepmzyme"]["ec_classes_1_6_ba_points"] == pytest.approx(dm)
    assert after_s1["gate"]["optimism_discount_points"] == 0.0
    assert after_s1["gate"]["gap_points"] == pytest.approx(dm - expected_clean)
    assert after_s1["gate"]["decision"] == selector.DECISION_MAINTAINER
    assert after_s1["selected"]["config"] == "F2_lr1e-4_e30"
    after_s3 = gate("S3", "gate_s3.json")
    assert after_s3["selected"]["config"] == "L4" and after_s3["gate"]["optimism_discount_points"] == 1.0
    assert after_s3["systems"]["deepmzyme"]["ec_classes_1_6_ba_points"] > dm
    systems = after_s3["systems"]
    assert after_s3["seven_class_gap_points"] == pytest.approx(
        systems["deepmzyme"]["seven_class_ba_points"] - systems["clean_r1_inner"]["seven_class_ba_points"])
    assert after_s3["r2_inner_gap_points"] == pytest.approx(
        systems["deepmzyme"]["ec_classes_1_6_ba_points"] - systems["clean_r2_inner"]["ec_classes_1_6_ba_points"])
    assert after_s3["components"] == TABLES.record
    text = (tmp_path / "gate_s3.json").read_text()
    assert not any(pid in text for pid in IDS)
    with pytest.raises(SelectionInputError, match="written once"):
        gate("S3", "gate_s3.json")
    with pytest.raises(SelectionInputError, match="one export per inner fold"):
        selector.run_gap_gate(tables=TABLES, fold=0, after="S1", step_chain=out, freeze1_sha256=FREEZE1, plan_path=world["plan_path"],
                              runs_dirs=[world["runs"]], clean_r1_inner=r1[:3], clean_r2_inner=r2,
                              out=tmp_path / "gate_three.json")
    with pytest.raises(SelectionInputError, match="fold 0 only"):
        selector.run_gap_gate(tables=TABLES, fold=1, after="S1", step_chain=out, freeze1_sha256=FREEZE1, plan_path=world["plan_path"],
                              runs_dirs=[world["runs"]], clean_r1_inner=r1, clean_r2_inner=r2,
                              out=tmp_path / "gate_fold1.json")


def _parser():
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action")
    selector.add_select_arguments(sub.add_parser("select")).set_defaults(handler=selector.action_select)
    selector.add_gap_gate_arguments(sub.add_parser("gap-gate")).set_defaults(handler=selector.action_gap_gate)
    return parser


def _gate_argv(world, chain, r1, r2, out, anchor=FREEZE1):
    return ["gap-gate", "--fold", "0", "--after", "S1", "--step-chain", str(chain), "--freeze1-sha256", anchor,
            "--plan", str(world["plan_path"]), "--runs", str(world["runs"]), "--components", "o/identity/fold0",
            *sum((["--clean-r1-inner", str(path)] for path in r1), []),
            *sum((["--clean-r2-inner", str(path)] for path in r2), []), "--out", str(out)]


def test_gap_gate_action_prints_totals_only_and_refuses_with_exit_2(selected_fold, tmp_path, monkeypatch, capsys):
    world, out, _s1, _s2 = selected_fold
    monkeypatch.setattr(selector, "load_inner_tables", lambda *args, **kwargs: TABLES)
    monkeypatch.setattr(selector, "real_action_git_state", lambda smoke: {"note": "synthetic"})
    monkeypatch.setattr(selector, "provenance_record", lambda **kwargs: {"note": "synthetic"})
    r1 = _write_clean_exports(tmp_path / "clean", "r1_inner")
    r2 = _write_clean_exports(tmp_path / "clean", "r2_inner")
    args = _parser().parse_args(_gate_argv(world, out, r1, r2, tmp_path / "gate.json"))
    assert args.handler(args) == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed)
    assert set(summary) == {"after", "decision", "gap_points", "totals", "gate_record_sha256"}
    assert set(summary["totals"]) == set(selector.GAP_GATE_PRINTED_TOTALS)
    assert not any(word in printed for word in ("support_by_ec_class", "ec7", "n_by_inner_fold", "systems"))
    assert "support_by_ec_class" in json.loads((tmp_path / "gate.json").read_text())["denominators"]
    args = _parser().parse_args(_gate_argv(world, out, r1, r2, tmp_path / "gate_other.json", anchor="ab" * 32))
    assert args.handler(args) == 2
    assert "error: --step-chain" in capsys.readouterr().err and not (tmp_path / "gate_other.json").exists()
    args = _parser().parse_args(_gate_argv(world, tmp_path / "missing.jsonl", r1, r2, tmp_path / "gate_missing.json"))
    assert args.handler(args) == 2
    assert "step chain does not verify" in capsys.readouterr().err


SELECT_PATHS = ("plan", "components", "out", "runs")
GATE_PATHS = ("step-chain", "plan", "out", "runs", "components")


def _refuse_any_reading(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("an input was read before the test-named path was refused")

    for name in ("real_action_git_state", "load_inner_tables", "run_select", "run_gap_gate"):
        monkeypatch.setattr(selector, name, unexpected)


@pytest.mark.parametrize("name", SELECT_PATHS)
def test_select_action_refuses_test_named_paths_before_reading(tmp_path, monkeypatch, name):
    from ec_clean30.provenance import HeldOutInputError

    _refuse_any_reading(monkeypatch)
    values = {"plan": "p/plans/fold0.json", "components": "o/identity/fold0", "out": "steps.jsonl", "runs": "r"}
    values[name] = str(tmp_path / "held_test")
    argv = ["select", "--fold", "0", "--step", "S1", "--freeze1-sha256", FREEZE1,
            *sum(([f"--{key}", value] for key, value in values.items()), [])]
    args = _parser().parse_args(argv)
    with pytest.raises(HeldOutInputError, match=f"--{name}"):
        args.handler(args)


@pytest.mark.parametrize("name", GATE_PATHS)
def test_gap_gate_action_refuses_test_named_paths_before_reading(tmp_path, monkeypatch, name):
    from ec_clean30.provenance import HeldOutInputError

    _refuse_any_reading(monkeypatch)
    values = {"step-chain": "steps.jsonl", "plan": "p/plans/fold0.json", "out": "gate.json", "runs": "r",
              "components": "o/identity/fold0"}
    values[name] = str(tmp_path / "held_test")
    argv = ["gap-gate", "--fold", "0", "--after", "S1", "--freeze1-sha256", FREEZE1,
            *sum(([f"--{key}", value] for key, value in values.items()), []),
            *sum((["--clean-r1-inner", f"r1_{k}.npz", "--clean-r2-inner", f"r2_{k}.npz"] for k in range(4)), [])]
    args = _parser().parse_args(argv)
    with pytest.raises(HeldOutInputError, match=f"--{name}"):
        args.handler(args)


@pytest.mark.parametrize("arm", ["r1_inner", "r2_inner"])
def test_test_named_clean_exports_are_refused_before_reading(tmp_path, arm):
    from ec_clean30.provenance import HeldOutInputError

    paths = [tmp_path / f"{arm}_inner{k}.npz" for k in range(3)] + [tmp_path / f"{arm}_test.npz"]
    with pytest.raises(HeldOutInputError, match=f"--clean-{arm.replace('_', '-')}"):
        selector._read_clean_arm(paths[::-1], arm)


@pytest.mark.parametrize("deepmzyme, clean, after, decision, gap", [
    (70.0, 72.9, "S1", selector.DECISION_CONTINUE, -2.9),
    (70.0, 73.1, "S1", selector.DECISION_MAINTAINER, -3.1),
    (70.0, 73.0, "S1", selector.DECISION_CONTINUE, -3.0),
    (70.0, 71.9, "S3", selector.DECISION_CONTINUE, -2.9),
    (70.0, 72.1, "S3", selector.DECISION_MAINTAINER, -3.1),
    (70.0, 72.5, "S3", selector.DECISION_MAINTAINER, -3.5),
])
def test_gap_gate_rule_and_optimism_discount(deepmzyme, clean, after, decision, gap):
    result = selector.gap_gate_decision(deepmzyme, clean, after=after)
    assert result["decision"] == decision and result["gap_points"] == pytest.approx(gap)
    assert result["optimism_discount_points"] == (1.0 if after == "S3" else 0.0)


def test_correctness_balanced_accuracy_counts_ties_as_wrong():
    from training.ec_class_metrics import ec_class_metrics

    true = np.array([0, 0, 1, 1, 2, 6])
    predicted = np.array([0, 1, 1, 1, 0, 6])
    reference = 100 * ec_class_metrics(true, predicted)["ec_classes_1_6_balanced_accuracy"]
    assert selector.correctness_balanced_accuracy(true, predicted == true)[0] == pytest.approx(reference)
    tie_wrong = (predicted == true) & np.array([True, True, False, True, True, True])
    assert selector.correctness_balanced_accuracy(true, tie_wrong) == (pytest.approx(100 * (0.5 + 0.5 + 0) / 3), 3)


# ---------------------------------------------------------------------------
# Inputs, imports and runner arguments


def test_components_under_the_campaign_identity_folder_are_refused(tmp_path):
    root = tmp_path / CAMPAIGN_ID
    (root / "identity" / "fold0").mkdir(parents=True)
    (root / "identity" / "fold0" / "components_inner.tsv").write_text("accession\tcomponent_id\n")
    for path in (root / "identity" / "fold0", root / "identity" / "fold0" / "components_inner.tsv"):
        with pytest.raises(SelectionInputError, match="campaign's identity folder"):
            selector.check_components_path(0, path, smoke=True)
        with pytest.raises(SelectionInputError, match="campaign's identity folder"):
            selector.load_inner_tables(0, path, smoke=True)
    overlay = root / "overlay" / "ecc30-overlay-v1"
    (overlay / "identity").mkdir(parents=True)
    (overlay / "identity" / "fold0").symlink_to(root / "identity" / "fold0", target_is_directory=True)
    with pytest.raises(SelectionInputError, match="campaign's identity folder"):
        selector.check_components_path(0, overlay / "identity" / "fold0", smoke=True)
    plain = tmp_path / "plain" / "fold0"
    plain.mkdir(parents=True)
    with pytest.raises(SelectionInputError, match="data overlay's copy"):
        selector.check_components_path(0, plain, smoke=False)
    with pytest.raises(SelectionInputError, match="identity folder"):
        selector.run_select(fold=0, step="S1", plan_path=tmp_path / "plans" / "fold0.json", runs_dirs=[],
                            tables=TABLES, freeze1_sha256=FREEZE1, out=root / "identity" / "steps.jsonl")


def test_protected_modules_do_not_load_the_selector_and_the_selector_loads_no_torch():
    code = "\n".join([
        "import sys",
        "sys.path.insert(0, 'src')",
        "import ec_clean30.campaign, ec_clean30.commands, ec_clean30.views, ec_clean30.provenance, ec_clean30.identity",
        "print('ec_clean30.selector' in sys.modules)",
        "import ec_clean30.selector",
        "print('torch' in sys.modules, 'ec_clean30.simulation' in sys.modules, 'ec_clean30.scoring' in sys.modules)",
    ])
    env = {**os.environ, "MKL_THREADING_LAYER": "GNU"}
    completed = subprocess.run([PYTHON, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env, timeout=120)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.splitlines() == ["False", "False False False"]


def test_runner_arguments_parse():
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action")
    selector.add_select_arguments(sub.add_parser("select")).set_defaults(handler=selector.action_select)
    selector.add_gap_gate_arguments(sub.add_parser("gap-gate")).set_defaults(handler=selector.action_gap_gate)
    args = parser.parse_args(["select", "--fold", "0", "--step", "S2", "--plan", "p/plans/fold0.json", "--runs", "a",
                              "--runs", "b", "--components", "o/identity/fold0", "--freeze1-sha256", FREEZE1,
                              "--out", "steps.jsonl"])
    assert args.runs == ["a", "b"] and args.step == "S2" and args.handler is selector.action_select
    args = parser.parse_args(["gap-gate", "--fold", "0", "--after", "S3", "--step-chain", "s", "--freeze1-sha256",
                              FREEZE1, "--plan", "p", "--runs", "r", "--components", "c", *sum((["--clean-r1-inner", f"r1_{k}.npz"]
                                                                          for k in range(4)), []),
                              *sum((["--clean-r2-inner", f"r2_{k}.npz"] for k in range(4)), []), "--out", "g.json"])
    assert len(args.clean_r1_inner) == len(args.clean_r2_inner) == 4 and args.handler is selector.action_gap_gate
    with pytest.raises(SystemExit):
        parser.parse_args(["select", "--fold", "0", "--step", "S4", "--plan", "p", "--runs", "r", "--components", "c",
                           "--freeze1-sha256", FREEZE1, "--out", "o"])
    assert issubclass(SelectionInputError, ValueError)
    assert hashlib.sha256(b"").hexdigest() != FREEZE1


# ---------------------------------------------------------------------------
# Versioned S1 amendment: source authentication is tested in test_ec_clean30_freeze;
# these tests exercise its verified result through real synthetic fit/step/gap chains.

AMENDMENT = "a1" * 32


def _verified_amendment(monkeypatch, tmp_path):
    from ec_clean30 import freeze

    path = tmp_path / "s1_amendment.json"
    path.write_text("{}\n")
    calls = []

    def verify(given, *, expected_freeze1_sha256, require_published):
        assert given == path and expected_freeze1_sha256 == FREEZE1
        calls.append(require_published)
        return {"document_sha256": AMENDMENT, "amendment_chain": [AMENDMENT],
                "s1_rule": selector.AMENDED_S1_RULE, "published": True}

    monkeypatch.setattr(freeze, "verify_s1_amendment", verify)
    return path, calls


def _amended_world(tmp_path, *, family_accuracies=None):
    root = tmp_path / CAMPAIGN_ID
    world = {"root": root, "plan_path": write_plan(root), "runs": root / "runs" / "fold0"}
    _s1_world(root, world["runs"], family_accuracies=family_accuracies,
              identity_changes={"freeze1_amendment_chain": [AMENDMENT]})
    write_verify_receipt(world["runs"])
    return world


def test_amended_s1_selects_f4_even_when_both_single_additions_fail(tmp_path, monkeypatch):
    amendment, calls = _verified_amendment(monkeypatch, tmp_path)
    world = _amended_world(tmp_path, family_accuracies={"F4": 0.98})
    record = _select(world, "S1", tmp_path / "steps.jsonl", freeze1_amendment_path=amendment)
    payload = record["payload"]
    assert payload["s1_winner"]["family"] == "F4"
    assert payload["s1_rule"] == selector.AMENDED_S1_RULE
    assert payload["freeze1_amendment_chain"] == [AMENDMENT]
    assert payload["family_screen"]["winner_family"] == "F4" and "family_escalation" not in payload
    assert record["previous_sha256"] == FREEZE1 and calls == [True]


def test_amendment_identity_persists_through_s1_s2_s3_and_gap(tmp_path, monkeypatch):
    amendment, calls = _verified_amendment(monkeypatch, tmp_path)
    world = _amended_world(tmp_path)
    out = tmp_path / "steps.jsonl"
    s1 = _select(world, "S1", out, freeze1_amendment_path=amendment)
    assert s1["payload"]["s1_winner"] == S1_WINNER.to_dict()
    _s2_world(world["runs"], S1_WINNER, {"L4": 0.95},
              identity_changes={"freeze1_amendment_chain": [AMENDMENT]})
    write_verify_receipt(world["runs"])
    s2 = _select(world, "S2", out, freeze1_amendment_path=amendment)
    s3 = _select(world, "S3", out, freeze1_amendment_path=amendment)
    r1 = _write_clean_exports(tmp_path / "clean", "r1_inner")
    r2 = _write_clean_exports(tmp_path / "clean", "r2_inner")
    gate = selector.run_gap_gate(tables=TABLES, fold=0, after="S3", step_chain=out, freeze1_sha256=FREEZE1,
                                 plan_path=world["plan_path"], runs_dirs=[world["runs"]], clean_r1_inner=r1,
                                 clean_r2_inner=r2, out=tmp_path / "gap.json", provenance={"note": "synthetic"},
                                 freeze1_amendment_path=amendment)
    for payload in (s1["payload"], s2["payload"], s3["payload"], gate):
        assert payload["freeze1_sha256"] == FREEZE1
        assert payload["freeze1_amendment_chain"] == [AMENDMENT]
        assert payload["s1_rule"] == selector.AMENDED_S1_RULE
    assert gate["selected"]["config"] == "L4"
    assert calls == [True] * 4
    assert verify_step_chain(read_step_chain(out, anchor_sha256=FREEZE1), anchor_sha256=FREEZE1)


@pytest.mark.parametrize("chain", [[], ["b2" * 32], [AMENDMENT, "b2" * 32], "not-a-list"])
def test_amended_s1_refuses_missing_foreign_or_malformed_fit_chain(tmp_path, monkeypatch, chain):
    amendment, _ = _verified_amendment(monkeypatch, tmp_path)
    world = _amended_world(tmp_path)
    path = next(world["runs"].glob("*.runner.json"))
    value = json.loads(path.read_text())
    value["unit"]["identity"]["freeze1_amendment_chain"] = chain
    path.write_text(json.dumps(value))
    out = tmp_path / "steps.jsonl"
    with pytest.raises(SelectionInputError, match="amendment chain"):
        _select(world, "S1", out, freeze1_amendment_path=amendment)
    assert not out.exists()


def test_historical_selector_refuses_amended_fits_without_amendment(tmp_path):
    world = _amended_world(tmp_path)
    out = tmp_path / "steps.jsonl"
    with pytest.raises(SelectionInputError, match="matching --freeze1-amendment"):
        _select(world, "S1", out)
    assert not out.exists()


def test_amendment_cannot_be_added_or_removed_mid_chain(tmp_path, monkeypatch):
    amendment, _ = _verified_amendment(monkeypatch, tmp_path)
    world, out = _s1_selected(tmp_path)
    with pytest.raises(SelectionInputError, match="amendment chain"):
        _select(world, "S2", out, freeze1_amendment_path=amendment)
    # Check the converse before any selected fit or CLEAN export is opened.
    records = read_step_chain(out, anchor_sha256=FREEZE1)
    payload = dict(records[0]["payload"], freeze1_amendment_chain=[AMENDMENT], s1_rule=selector.AMENDED_S1_RULE)
    amended = tmp_path / "amended_steps.jsonl"
    selector.append_step_record(amended, "S1", payload, anchor_sha256=FREEZE1)
    with pytest.raises(SelectionInputError, match="amendment chain"):
        selector.run_gap_gate(tables=TABLES, fold=0, after="S1", step_chain=amended, freeze1_sha256=FREEZE1,
                             plan_path=world["plan_path"], runs_dirs=[], clean_r1_inner=[], clean_r2_inner=[],
                             out=tmp_path / "gap.json")
    assert not (tmp_path / "gap.json").exists()


def test_step_record_rule_must_match_authenticated_amendment(tmp_path, monkeypatch):
    amendment, _ = _verified_amendment(monkeypatch, tmp_path)
    world = _amended_world(tmp_path)
    original = _select(world, "S1", tmp_path / "original.jsonl", freeze1_amendment_path=amendment)
    payload = dict(original["payload"], s1_rule=selector.HISTORICAL_S1_RULE)
    out = tmp_path / "wrong_rule.jsonl"
    selector.append_step_record(out, "S1", payload, anchor_sha256=FREEZE1)
    with pytest.raises(SelectionInputError, match="S1 rule differs"):
        _select(world, "S2", out, freeze1_amendment_path=amendment)


def test_historical_s1_record_cannot_be_relabelled_as_amended(tmp_path, monkeypatch):
    amendment, _ = _verified_amendment(monkeypatch, tmp_path)
    world, historical = _s1_selected(tmp_path)
    payload = dict(read_step_chain(historical, anchor_sha256=FREEZE1)[0]["payload"],
                   s1_rule=selector.AMENDED_S1_RULE, freeze1_amendment_chain=[AMENDMENT])
    out = tmp_path / "relabelled.jsonl"
    selector.append_step_record(out, "S1", payload, anchor_sha256=FREEZE1)
    with pytest.raises(SelectionInputError, match="family-screen rule"):
        _select(world, "S2", out, freeze1_amendment_path=amendment)


def test_real_amended_selection_requires_publication_and_smoke_does_not(tmp_path, monkeypatch):
    from ec_clean30 import freeze

    calls = []

    def unpublished(path, *, expected_freeze1_sha256, require_published):
        calls.append(require_published)
        return {"document_sha256": AMENDMENT, "amendment_chain": [AMENDMENT],
                "s1_rule": selector.AMENDED_S1_RULE, "published": False}

    monkeypatch.setattr(freeze, "verify_s1_amendment", unpublished)
    with pytest.raises(SelectionInputError, match="published amendment"):
        selector._selection_policy(tmp_path / "amendment.json", FREEZE1, smoke=False)
    policy = selector._selection_policy(tmp_path / "amendment.json", FREEZE1, smoke=True)
    assert policy["freeze1_amendment_chain"] == [AMENDMENT] and calls == [True, False]


@pytest.mark.parametrize("action", ["select", "gap-gate"])
def test_amendment_cli_rejects_test_named_path_before_any_read(tmp_path, monkeypatch, action):
    from ec_clean30.provenance import HeldOutInputError

    _refuse_any_reading(monkeypatch)
    if action == "select":
        argv = ["select", "--fold", "0", "--step", "S1", "--freeze1-sha256", FREEZE1,
                "--plan", "p/plans/fold0.json", "--components", "o/identity/fold0", "--out", "s", "--runs", "r"]
    else:
        argv = _gate_argv({"plan_path": Path("p"), "runs": Path("r")}, Path("s"),
                          [Path(f"r1_{k}.npz") for k in range(4)], [Path(f"r2_{k}.npz") for k in range(4)], Path("g"))
    args = _parser().parse_args([*argv, "--freeze1-amendment", str(tmp_path / "held_test.json")])
    with pytest.raises(HeldOutInputError):
        args.handler(args)


# ---------------------------------------------------------------------------
# Narrow round-2 selection contracts


def _json_write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": path.name, "sha256": sha256_file(path)}


def _cost_inputs(folder, plan, records):
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder / "freeze1c.json"
    _json_write(frozen, {"campaign_id": CAMPAIGN_ID, "freeze1_sha256": FREEZE1,
                         "l3_vm_hours_per_fold": {"0": 1.0}})
    binding = {"campaign_id": CAMPAIGN_ID, "fold": 0, "plan_sha256": plan["plan_sha256"],
               "freeze1_sha256": FREEZE1, "freeze1c_sha256": sha256_file(frozen)}
    attempts = [{"unit_id": uid, "status": record.runner["status"], "runner_sha256": record.runner_sha256}
                for uid, record in records.items() if record.step == "S2" and "L3" in record.lever_configs]
    forecast = _json_write(folder / "forecast.json", {"timing": "synthetic", "remaining_vm_hours_estimate": 2.0})
    accounting = {**binding, "vm_intervals": [[1000, 1360]] if attempts else [],
                  "used_vm_hours": 0.1 if attempts else 0.0, "remaining_vm_hours_estimate": 2.0,
                  "forecast_evidence": forecast, "attempted_fits": attempts}
    account_ref = _json_write(folder / "accounting.json", accounting)
    authority = {**binding, "accounting_sha256": account_ref["sha256"], "decision": "stop_and_close_out",
                 "decision_reference": "synthetic maintainer decision"}
    authority_ref = _json_write(folder / "authority.json", authority)
    disposition = {**binding, "disposition": "not_tested_cost", "reason": "projected_completion_exceeds_ceiling",
                   "accounting": account_ref, "authority": authority_ref}
    path = folder / "l3_cost_disposition.json"
    _json_write(path, disposition)
    return frozen, path


def _cost_world(tmp_path, n_fits):
    plan = build_fold_plan(0)
    runs = tmp_path / "runs"
    runs.mkdir()
    recipe = recipe_with_lever(S1_WINNER, "L3")
    for inner, seed in GRID[:n_fits]:
        write_fit(runs, unit_id=l5_id(inner, seed, config="L3"), step="S2", inner=inner, seed=seed, recipe=recipe)
    write_verify_receipt(runs)
    records = selector.read_runner_records([runs], fold=0)
    frozen, disposition = _cost_inputs(tmp_path / "cost", plan, records)
    return plan, runs, records, frozen, disposition


def _verify_cost(world):
    plan, _runs, records, frozen, disposition = world
    return selector.verify_l3_cost_disposition(plan, records, S1_WINNER, freeze1c_path=frozen,
                                               disposition_path=disposition, freeze1_sha256=FREEZE1)


def _reseal_cost(folder):
    """Refresh the links after an intentional invalid-field mutation, to reach semantic checks."""
    authority_path = folder / "authority.json"
    authority = json.loads(authority_path.read_text())
    authority["accounting_sha256"] = sha256_file(folder / "accounting.json")
    _json_write(authority_path, authority)
    path = folder / "l3_cost_disposition.json"
    disposition = json.loads(path.read_text())
    disposition["accounting"]["sha256"] = sha256_file(folder / "accounting.json")
    disposition["authority"]["sha256"] = sha256_file(authority_path)
    _json_write(path, disposition)


@pytest.mark.parametrize("n_fits", [0, 3, 8])
def test_l3_cost_skip_requires_evidence_and_never_suppresses_a_complete_grid(tmp_path, n_fits):
    world = _cost_world(tmp_path, n_fits)
    evidence = _verify_cost(world)
    assert evidence["skip"] is (n_fits < 8)
    assert len(evidence["attempted_fits"]) == n_fits
    assert evidence["disposition_sha256"] == sha256_file(world[-1])
    assert evidence["ceiling_vm_hours"] == 1.0


@pytest.mark.parametrize("record_name,key,value,message", [
    ("authority", "decision", "continue", "recorded stop decision"),
    ("authority", "decision_reference", "", "recorded stop decision"),
    ("accounting", "fold", 1, "hashes disagree"),
    ("accounting", "used_vm_hours", -1, "finite nonnegative"),
    ("accounting", "used_vm_hours", float("nan"), "finite nonnegative"),
    ("accounting", "used_vm_hours", 0.2, "disagrees with charged"),
    ("accounting", "vm_intervals", [[1000, 1360], [1200, 1400]], "non-overlapping"),
    ("accounting", "remaining_vm_hours_estimate", 0.0, "not supported"),
    ("accounting", "attempted_fits", [], "exactly the attempted fits"),
    ("l3_cost_disposition", "reason", "training_failed", "not supported"),
])
def test_l3_cost_record_refuses_unbound_or_unsupported_skips(tmp_path, record_name, key, value, message):
    world = _cost_world(tmp_path, 1)
    path = world[-1].parent / f"{record_name}.json"
    record = json.loads(path.read_text())
    record[key] = value
    _json_write(path, record)
    _reseal_cost(path.parent)
    with pytest.raises(SelectionInputError, match=message):
        _verify_cost(world)


@pytest.mark.parametrize("status", ["running", "failed", "stopped_cost"])
def test_l3_cost_record_needs_an_explicit_terminal_cost_stop(tmp_path, status):
    plan, runs, _records, _frozen, _disposition = _cost_world(tmp_path, 1)
    runner_path = next(runs.glob("*.runner.json"))
    runner = json.loads(runner_path.read_text())
    runner.update(status=status, returncode=143, stop_reason="cost" if status == "stopped_cost" else "failure")
    _json_write(runner_path, runner)
    records = selector.read_runner_records([runs], fold=0)
    frozen, disposition = _cost_inputs(tmp_path / "cost", plan, records)
    world = plan, runs, records, frozen, disposition
    if status == "stopped_cost":
        assert _verify_cost(world)["skip"]
    else:
        with pytest.raises(SelectionInputError, match="unresolved/failed"):
            _verify_cost(world)


def test_l3_cost_record_refuses_changed_forecast_and_freeze1c(tmp_path):
    world = _cost_world(tmp_path, 0)
    forecast = world[-1].parent / "forecast.json"
    original = forecast.read_bytes()
    forecast.write_text("changed")
    with pytest.raises(SelectionInputError, match="L3 forecast: evidence SHA-256 disagrees"):
        _verify_cost(world)
    forecast.write_bytes(original)
    frozen = world[-2]
    data = json.loads(frozen.read_text())
    data["l3_vm_hours_per_fold"]["0"] = 1.5
    _json_write(frozen, data)
    with pytest.raises(SelectionInputError, match="hashes disagree"):
        _verify_cost(world)


def test_select_s2_records_partial_l3_as_cost_untested(tmp_path):
    world, chain = _s1_selected(tmp_path)
    _s2_world(world["runs"], S1_WINNER, {"L3": 1.0})  # excellent partial scores must never enter screening
    for inner, seed in GRID[2:]:
        uid = l5_id(inner, seed, config="L3")
        shutil.rmtree(world["runs"] / uid)
        (world["runs"] / f"{uid}.runner.json").unlink()
    write_verify_receipt(world["runs"])
    plan = selector.load_frozen_plan(world["plan_path"], 0)
    records = selector.read_runner_records([world["runs"]], fold=0)
    frozen, disposition = _cost_inputs(tmp_path / "cost", plan, records)
    result = _select(world, "S2", chain, freeze1c_path=frozen, l3_cost_disposition_path=disposition)
    l3 = next(item for item in result["payload"]["screen"]["outcomes"] if item["lever"] == "L3")
    assert l3["status"] == LEVER_NOT_TESTED and l3["decision"] is None
    assert "L3" not in result["payload"]["configurations"]
    assert result["payload"]["l3_cost_disposition"]["skip"]


@pytest.mark.parametrize("missing", ["freeze1c", "disposition"])
def test_select_refuses_unpaired_cost_inputs_before_any_file_read(tmp_path, missing):
    options = {"freeze1c_path": tmp_path / "f.json", "l3_cost_disposition_path": tmp_path / "d.json"}
    options["freeze1c_path" if missing == "freeze1c" else "l3_cost_disposition_path"] = None
    with pytest.raises(SelectionInputError, match="supplied together"):
        selector.run_select(fold=0, step="S2", plan_path=tmp_path / "plans" / "fold0.json", runs_dirs=[],
                            tables=TABLES, freeze1_sha256=FREEZE1, out=tmp_path / "steps.jsonl", **options)


@pytest.mark.parametrize("promoted", [False, True])
def test_s3_l9_plus_one_fitted_lever_reuses_s2_and_fixed_tau(tmp_path, monkeypatch, promoted):
    world, chain = _s1_selected(tmp_path)
    _s2_world(world["runs"], S1_WINNER, {"L4": 0.95})
    write_verify_receipt(world["runs"])
    original_screen = selector.screen_logit_adjustment

    def passing_l9(*args, **kwargs):
        # Force the branch while retaining the actual computed tau and prediction hash. Numeric promotion is
        # tested separately; this test checks source identity and no additional fits/search in the coordinator.
        result = original_screen(*args, **kwargs)
        return replace(result, status=LEVER_PASSED, decision=replace(result.decision, promoted=True))

    monkeypatch.setattr(selector, "screen_logit_adjustment", passing_l9)
    s2 = _select(world, "S2", chain)
    assert s2["payload"]["screen"]["passing"] == ["L4", "L9"]
    tau = next(item["details"]["chosen_tau"] for item in s2["payload"]["screen"]["outcomes"] if item["lever"] == "L9")
    original_combination = selector.s3_combination
    observed = {}

    def decision(s1, screen, combination, **kwargs):
        observed["hash"] = selector.predictions_sha256(combination)
        result = original_combination(s1, screen, combination, **kwargs)
        return replace(result, selected="combination", selected_oof=combination) if promoted else result

    monkeypatch.setattr(selector, "s3_combination", decision)
    monkeypatch.setattr(selector, "screen_logit_adjustment", lambda *a, **k: pytest.fail("S3 retuned tau"))
    result = _select(world, "S3", chain)["payload"]
    assert result["combination_source"]["step"] == "S2"
    expected_ids = [l5_id(inner, seed, config="L4") for inner, seed in GRID]
    assert [fit["unit_id"] for fit in result["combination_source"]["fits"]] == expected_ids
    assert result["selected"]["unit_ids"] == expected_ids
    assert result["selected"]["config"] == ("combination" if promoted else "L4")
    if promoted:
        assert result["selected"]["tau"] == tau
        assert result["selected"]["predictions_sha256"] == observed["hash"]
    assert not list(world["runs"].glob("*__S3__*.runner.json"))
    # Replaying the selected configuration uses the same real S2 unit IDs and yields the recorded predictions.
    replay = selector.pool_selected(result["selected"], selector.load_frozen_plan(world["plan_path"], 0),
                                    selector.read_runner_records([world["runs"]], fold=0), tables=TABLES,
                                    freeze1_sha256=FREEZE1)
    assert selector.predictions_sha256(replay.pooled) == result["selected"]["predictions_sha256"]


@pytest.mark.parametrize("case", ["extra", "missing", "duplicate", "wrong_fold"])
def test_gap_queries_must_match_all_validation_ids_before_closure_filtering(case):
    ids = tuple(IDS)
    pooled = PooledOof("synthetic", ids, np.array([TRUE[pid] for pid in ids]), (42, 43),
                       np.zeros((2, len(ids), 7)), inner_fold=np.array([INNER[pid] for pid in ids]))
    configuration = selector.PooledConfiguration(pooled, (), {pid: (EC_NUMBER[pid],) for pid in ids})
    # Empty centres ensure a premature closure filter would lose the membership violation.
    exports = {inner: SimpleNamespace(query_ids=[pid for pid in ids if INNER[pid] == inner], ec_labels=[])
               for inner in range(4)}
    queries = exports[0].query_ids
    if case == "extra":
        queries.append("foreign")
    elif case == "missing":
        queries.pop()
    elif case == "duplicate":
        queries.append(queries[0])
    else:
        queries[0] = next(pid for pid in ids if INNER[pid] == 1)
    with pytest.raises(SelectionInputError, match="exactly equal the certified validation membership before closure"):
        selector.score_gap_gate(configuration, exports, exports, after="S3")


def test_gap_gate_rejects_changed_components_binding_before_export_read(selected_fold, tmp_path):
    world, chain, _s1, _s2 = selected_fold
    tables = selector.InnerTables(COMPONENTS, INNER, {"changed": True}, MEMBERSHIP)
    with pytest.raises(SelectionInputError, match="--components differs"):
        selector.run_gap_gate(tables=tables, fold=0, after="S1", step_chain=chain, freeze1_sha256=FREEZE1,
                              plan_path=world["plan_path"], runs_dirs=[world["runs"]], clean_r1_inner=[],
                              clean_r2_inner=[], out=tmp_path / "gate.json")
    assert not (tmp_path / "gate.json").exists()
