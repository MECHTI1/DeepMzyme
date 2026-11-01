"""EC prediction export (contract C3): in-run export, reconciliation, replay, probe mode and the test pass (synthetic)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_fixtures import default_ec_proteins, tiny_ec_argv, write_ec_dataset  # noqa: E402
from training.ec_membership import write_ec_fold_membership  # noqa: E402
import training.ec_prediction_export as export  # noqa: E402


PYTHON = sys.executable


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _chain_mean_cache(structures: Path, esm: Path, stems, out: Path) -> tuple[Path, str]:
    from embed_helpers.esmc_chain_means import build_from_residue_cache

    allowlist = out.parent / f"{out.name}_allowlist.txt"
    allowlist.write_text("\n".join(sorted(stems)) + "\n")
    summary = build_from_residue_cache(allowlist, structures, esm, out, embedding_dim=8)
    return out, summary["manifest_sha256"]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


VALIDATION_FAMILIES = ("plain", "rich", "component")
TEST_PASS_FAMILIES = ("plain", "rich")
SITE_IDS = {"A_900", "A_901"}


def _two_site_proteins(proteins: dict[str, dict]) -> dict[str, dict]:
    """Each protein gets a second zinc 40 Å from the first, so it has two pockets in either extraction mode."""
    two = {}
    for accession, spec in proteins.items():
        chain, resseq, icode, element, xyz = spec["ions"][0]
        second = (chain, resseq + 1, icode, element, [xyz[0], xyz[1], xyz[2] + 40.0])
        two[accession] = {**spec, "ions": [spec["ions"][0], second]}
    return two


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """One synthetic dataset, a two-fold EC membership and four exported fits (no real data)."""
    from training.config import parse_args
    from training.run import run_training

    base = tmp_path_factory.mktemp("ec_export")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(base / "parse_cache"))
        patch.delenv("DEEPMZYME_GRAPH_CACHE_DIR", raising=False)
        patch.delenv("DEEPMZYME_FORBIDDEN_READ_ROOTS", raising=False)
        paths = write_ec_dataset(base / "data", _two_site_proteins(default_ec_proteins()))
        membership_rows = [{"structure_id": stem, "accession": accession, "group_id": accession, "fold": str(index % 2)}
                           for index, (accession, stem) in enumerate(sorted(paths["stems"].items()))]
        membership = base / "membership.csv"
        membership_sha = write_ec_fold_membership(membership_rows, membership)
        cache, cache_sha = _chain_mean_cache(paths["structures"], paths["esm"], paths["stems"].values(),
                                             base / "chain_means")
        common = ["--ec-class-vocabulary", "fixed_seven", "--ec-conflicting-labels", "fail", "--checkpoint-rule",
                  "terminal", "--lr-schedule", "cosine", "--no-prepare-missing-external-features"]
        membership_args = ["--fold-split-source", "membership", "--fold-membership-csv", str(membership),
                           "--fold-membership-sha256", membership_sha, "--n-folds", "2", "--fold-index", "1"]
        global_args = ["--use-global-esm-branch", "--global-esm-cache-dir", str(cache), "--global-esm-manifest-sha256",
                       cache_sha, "--global-esm-dim", "8", "--global-esm-standardize", "train_proteins"]
        aux_args = ["--ec-aux-levels", "2,3", "--ec-supcon-weight", "0.1", "--ec-supcon-memory", "8"]

        def argv(name, *extra, architecture="only_esm"):
            values = tiny_ec_argv(paths, base / "runs", *common, "--export-ec-predictions", "--run-name", name, *extra)
            values[values.index("only_esm")] = architecture
            return values

        probe_ids = sorted(paths["stems"].values())[::3]
        probe = base / "probe_ids.txt"
        probe.write_text("\n".join(probe_ids) + "\n")
        selected = ["--site-extraction", "selected_sites"]
        argvs = {
            "plain": argv("plain", *membership_args, *selected),
            "rich": argv("rich", *membership_args, *global_args, *aux_args, *selected, architecture="gvp"),
            "component": argv("component", *membership_args),
            "probe": argv("probe", *global_args, "--ec-aux-levels", "2", "--ec-probe-structure-ids", str(probe),
                          "--ec-probe-structure-ids-sha256", _sha256(probe)),
        }
        runs = {name: run_training(parse_args(values)) for name, values in argvs.items()}
    return {"base": base, "paths": paths, "membership": membership, "membership_rows": membership_rows,
            "membership_sha": membership_sha, "argvs": argvs, "runs": runs, "probe_ids": probe_ids,
            "cache": cache, "cache_sha": cache_sha}


def _replay_env(**extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if key not in {"DEEPMZYME_FORBIDDEN_READ_ROOTS", "DEEPMZYME_PARSE_CACHE_DIR", "DEEPMZYME_GRAPH_CACHE_DIR"}}
    env.update(PYTHONPATH=str(ROOT / "src"), MKL_THREADING_LAYER="GNU", PYTHONDONTWRITEBYTECODE="1", **extra)
    return env


def _replay_subprocess(run_dir: Path, output_dir: Path, cwd: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "training.ec_prediction_export", "replay", "--run-dir", str(run_dir), "--output-dir",
         str(output_dir)], cwd=str(cwd), env=_replay_env(**env), capture_output=True, text=True, timeout=600,
    )


# ---------------------------------------------------------------------------
# In-run export
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", VALIDATION_FAMILIES)
def test_export_writes_the_three_files_and_reconciles(campaign, family):
    run_dir = campaign["runs"][family]
    receipt = json.loads((run_dir / export.EC_PREDICTION_RECEIPT_NAME).read_text())
    assert receipt["mode"] == "validation" and receipt["reconciliation_status"] == "match"
    reconciliation = receipt["reconciliation"]
    for key in ("val_ec_group_ec_classes_1_6_balanced_acc", "val_ec_group_ec_classes_1_6_n_present",
                "val_ec_group_ec_classes_1_6_macro_f1", "val_ec_group_seven_class_balanced_acc",
                "val_ec_group_seven_class_n_present", "val_ec_group_seven_class_macro_f1",
                "val_ec_group_balanced_acc", "val_ec_group_confusion_matrix", "val_ec_acc", "val_ec_group_acc"):
        assert reconciliation[key]["match"], key
    history = json.loads((run_dir / "run_config.json").read_text())["history"]
    assert {key for key in history[-1] if key.startswith("val_ec_")} == set(reconciliation)
    assert reconciliation["val_ec_group_confusion_matrix"]["recorded"] == history[-1]["val_ec_group_confusion_matrix"]
    assert receipt["checkpoint"]["sha256"] == _sha256(run_dir / "terminal_model_checkpoint.pt")
    assert receipt["vocabulary_sha256"] == export.ec_class_vocabulary_sha256()
    assert receipt["ec_class_tokens"] == ["1", "2", "3", "4", "5", "6", "7"]
    assert receipt["membership"]["file_sha256"] == campaign["membership_sha"]
    assert receipt["training_protein_class_counts"] == {"1": 2, "2": 2, "3": 2, "4": 0, "5": 0, "6": 0, "7": 0}
    steps = receipt["optimizer_steps"]
    assert steps["realised_steps_match"] and steps["realised_total_steps"][0] == steps["planned_total_steps"]
    assert receipt["model"]["campaign_family"] == {"plain": "F2", "rich": "F4", "component": "F2"}[family]
    pockets = _rows(run_dir / "ec_val_pocket_predictions.csv")
    proteins = _rows(run_dir / "ec_val_protein_predictions.csv")
    assert list(pockets[0]) == list(export.EC_POCKET_PREDICTION_COLUMNS)
    assert list(proteins[0]) == list(export.EC_PROTEIN_PREDICTION_COLUMNS)
    validation = {row["structure_id"] for row in campaign["membership_rows"] if row["fold"] == "1"}
    assert {row["structure_id"] for row in pockets} == validation and len(proteins) == len(validation)
    assert len(pockets) == 2 * len(validation) and {row["n_pockets"] for row in proteins} == {"2"}
    assert {row["fold_index"] for row in pockets} == {"1"} and {row["seed"] for row in proteins} == {"42"}
    for row in pockets:
        token = row["structure_id"].split("__EC_")[1].split(".")[0]
        assert int(row["y"]) == int(token) - 1
        float_text = row["logit_1"]
        assert float_text == repr(float(float_text))
    # One pocket per site in either mode (selected sites key them by the summary site).
    for structure_id in validation:
        assert {row["site_ids"] for row in pockets if row["structure_id"] == structure_id} == SITE_IDS


def test_supcon_coverage_is_bound_to_the_fit(campaign):
    # L5: the receipt carries the fit's training minibatches and those with a SupCon positive,
    # summed from the history the checkpoint (hashed in the receipt) also holds.
    from ec_clean30.selection import supcon_positive_step_fraction
    from training.loop import SUPCON_BATCHES_KEY, SUPCON_BATCHES_WITH_POSITIVE_KEY

    rich_dir = campaign["runs"]["rich"]
    rich = json.loads((rich_dir / export.EC_PREDICTION_RECEIPT_NAME).read_text())
    plain = json.loads((campaign["runs"]["plain"] / export.EC_PREDICTION_RECEIPT_NAME).read_text())
    assert plain["ec_supcon_coverage"] is None
    coverage = rich["ec_supcon_coverage"]
    epochs = rich["epochs"]["completed"]
    assert coverage["train_batches"] == rich["optimizer_steps"]["optimizer_steps_per_epoch"] * epochs
    assert coverage["epochs_counted"] == epochs and 0 <= coverage["batches_with_positive"] <= coverage["train_batches"]
    checkpoint = torch.load(rich_dir / "terminal_model_checkpoint.pt", map_location="cpu", weights_only=False)
    assert sum(record[SUPCON_BATCHES_KEY] for record in checkpoint["history"]) == coverage["train_batches"]
    assert sum(record[SUPCON_BATCHES_WITH_POSITIVE_KEY] for record in checkpoint["history"]) == \
        coverage["batches_with_positive"]
    fraction, details = supcon_positive_step_fraction({"rich": coverage})
    assert fraction == coverage["batches_with_positive"] / coverage["train_batches"] and not details["uncertified_fits"]
    # A history without counters (an older run) leaves the fit uncertified.
    older = export.supcon_coverage({"ec_supcon_weight": 0.1}, [{"epoch": 1}, {"epoch": 2}])
    assert older["train_batches"] is None and supcon_positive_step_fraction({"older": older})[0] is None


@pytest.mark.parametrize("family", VALIDATION_FAMILIES)
def test_protein_rows_are_the_mean_of_their_pocket_logits(campaign, family):
    run_dir = campaign["runs"][family]
    pockets = _rows(run_dir / "ec_val_pocket_predictions.csv")
    proteins = _rows(run_dir / "ec_val_protein_predictions.csv")
    columns = export.LOGIT_COLUMNS
    for protein in proteins:
        members = [row for row in pockets if row["ec_group_key"] == protein["ec_group_key"]]
        assert len(members) == int(protein["n_pockets"]) == 2 and {row["y"] for row in members} == {protein["y"]}
        stacked = torch.tensor([[float(row[c]) for c in columns] for row in members], dtype=torch.float32)
        expected = stacked.mean(dim=0)
        assert [float(protein[c]) for c in columns] == expected.tolist()
        assert int(protein["pred"]) == int(expected.argmax())


def test_reconciliation_recomputes_the_terminal_record_and_detects_changes(campaign):
    run_dir = campaign["runs"]["plain"]
    record = json.loads((run_dir / "run_config.json").read_text())["history"][-1]
    expected = export.recompute_ec_validation_metrics(
        export.read_prediction_csv(run_dir / "ec_val_pocket_predictions.csv", export.EC_POCKET_PREDICTION_COLUMNS),
        export.read_prediction_csv(run_dir / "ec_val_protein_predictions.csv", export.EC_PROTEIN_PREDICTION_COLUMNS),
    )
    assert export.reconcile_ec_validation_metrics(record, expected)[1] == "match"
    for key, change in (
        ("val_ec_group_ec_classes_1_6_balanced_acc", lambda value: value + 1e-6),
        ("val_ec_group_confusion_matrix", lambda value: [row[:] for row in value[:-1]] + [value[-1][:-1] + [1]]),
        ("val_ec_group_n_groups", lambda value: value + 1),
        ("val_ec_per_class_recall", lambda value: {**value, "7": 0.0}),
    ):
        details, status = export.reconcile_ec_validation_metrics({**record, key: change(record[key])}, expected)
        assert status == "mismatch" and not details[key]["match"], key
    details, status = export.reconcile_ec_validation_metrics({**record, "val_ec_new_metric": 1.0}, expected)
    assert status == "mismatch" and "not recomputed" in details["val_ec_new_metric"]["reason"]
    details, status = export.reconcile_ec_validation_metrics(
        {key: value for key, value in record.items() if key != "val_ec_group_seven_class_macro_f1"}, expected)
    assert status == "mismatch" and "absent" in details["val_ec_group_seven_class_macro_f1"]["reason"]


def test_a_tampered_metric_raises_after_the_files_are_written(campaign, monkeypatch, tmp_path):
    import training.run as run_module
    from training.config import parse_args

    original = run_module.train_and_select_checkpoint

    def tampered(prepared, config):
        history, checkpoint = original(prepared, config)
        history[-1]["val_ec_group_ec_classes_1_6_balanced_acc"] += 0.01
        return history, checkpoint

    monkeypatch.setattr(run_module, "train_and_select_checkpoint", tampered)
    monkeypatch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(campaign["base"] / "parse_cache"))
    argv = list(campaign["argvs"]["plain"])
    argv[argv.index("plain")] = "tampered"
    with pytest.raises(export.EcPredictionExportError, match="do not reconcile"):
        run_module.run_training(parse_args(argv))
    run_dir = campaign["base"] / "runs" / "tampered"
    receipt = json.loads((run_dir / export.EC_PREDICTION_RECEIPT_NAME).read_text())
    assert receipt["reconciliation_status"] == "mismatch"
    assert not receipt["reconciliation"]["val_ec_group_ec_classes_1_6_balanced_acc"]["match"]
    assert (run_dir / "ec_val_pocket_predictions.csv").is_file() and (run_dir / "ec_val_protein_predictions.csv").is_file()
    # A run that did not reconcile cannot be certified by a replay.
    with pytest.raises(export.EcPredictionExportError, match="reconciliation status is 'mismatch'"):
        export.replay_ec_predictions(run_dir, output_dir=tmp_path / "replay", install_guards=False)
    assert not (tmp_path / "replay").exists()


def test_probe_mode_exports_the_listed_training_structures(campaign):
    run_dir = campaign["runs"]["probe"]
    receipt = json.loads((run_dir / export.EC_PREDICTION_RECEIPT_NAME).read_text())
    assert receipt["mode"] == "probe" and receipt["reconciliation_status"] == "not_applicable"
    assert receipt["probe"]["n_structures"] == len(campaign["probe_ids"]) and receipt["membership"] is None
    assert receipt["model"]["campaign_family"] == "F5"
    pockets = _rows(run_dir / "ec_probe_pocket_predictions.csv")
    proteins = _rows(run_dir / "ec_probe_protein_predictions.csv")
    assert sorted({row["structure_id"] for row in pockets}) == campaign["probe_ids"]
    assert len(proteins) == len(campaign["probe_ids"]) and {row["mode"] for row in proteins} == {"probe"}
    assert {row["fold_index"] for row in proteins} == {""}
    assert not (run_dir / "ec_val_pocket_predictions.csv").exists()
    assert receipt["training_protein_class_counts"] == {"1": 4, "2": 4, "3": 4, "4": 0, "5": 0, "6": 0, "7": 0}


# ---------------------------------------------------------------------------
# Rebuild and replay
# ---------------------------------------------------------------------------

def test_rebuild_replays_the_recorded_attachments(campaign):
    from model_variants import build_pocket_classifier
    from export_validation_predictions import factory_kwargs

    model, checkpoint = export.rebuild_model_from_checkpoint(campaign["runs"]["rich"])
    assert hasattr(model, "ec_aux_heads") and hasattr(model, "ec_supcon") and hasattr(model, "global_esm_encoder")
    assert not model.training
    labels = {int(k): v for k, v in checkpoint["ec_labels"].items()}
    metal = {int(k): v for k, v in checkpoint["metal_labels"].items()}
    bare = build_pocket_classifier(**factory_kwargs(ROOT / "src/training/run.py", checkpoint["config"], checkpoint,
                                                    metal, labels))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        bare.load_state_dict(checkpoint["model_state_dict"], strict=True)


@pytest.mark.parametrize("family", (*VALIDATION_FAMILIES, "probe"))
def test_replay_in_a_fresh_process_matches(campaign, family, tmp_path):
    run_dir = campaign["runs"][family]
    proc = _replay_subprocess(run_dir, tmp_path / "replay", tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    receipt = json.loads((tmp_path / "replay" / export.EC_REPLAY_RECEIPT_NAME).read_text())
    assert receipt["replay_status"] == "match" and receipt["guards_installed"]
    assert receipt["max_pocket_logit_abs_difference"] <= 1e-6 and receipt["max_protein_logit_abs_difference"] <= 1e-6
    assert receipt["source_reconciliation_status"] == ("not_applicable" if family == "probe" else "match")
    read = [Path(item["path"]) for item in receipt["input_file_hashes"]]
    structures = {path.stem for path in read if path.suffix == ".pdb"}
    exported = {row["structure_id"] for row in _rows(run_dir / export.PREDICTION_FILE_NAMES[receipt["mode"]][0])}
    assert structures == exported, "the replay parsed only the exported structures"


def test_replay_refuses_changed_checkpoint_predictions_or_config(campaign, tmp_path):
    source = campaign["runs"]["plain"]
    for target in ("terminal_model_checkpoint.pt", "ec_val_protein_predictions.csv", "run_config.json"):
        copy = tmp_path / target.replace(".", "_")
        shutil.copytree(source, copy, ignore=shutil.ignore_patterns("ec_prediction_replay"))
        with (copy / target).open("ab") as handle:
            handle.write(b"\n")
        with pytest.raises(export.EcPredictionExportError, match="changed since the export"):
            export.replay_ec_predictions(copy, output_dir=tmp_path / f"out_{target}", install_guards=False)
        assert not (tmp_path / f"out_{target}").exists()


def test_replay_under_a_forbidden_root_raises_on_reading_it(campaign, tmp_path):
    run_dir = campaign["runs"]["plain"]
    forbidden = str(campaign["paths"]["summary"])
    proc = _replay_subprocess(run_dir, tmp_path / "replay", tmp_path, DEEPMZYME_FORBIDDEN_READ_ROOTS=forbidden)
    assert proc.returncode != 0 and "PermissionError" in proc.stderr and "held-out path" in proc.stderr
    # A forbidden folder is not even listed (pathlib turns the refusal into an empty listing).
    proc = _replay_subprocess(run_dir, tmp_path / "replay_dir", tmp_path,
                              DEEPMZYME_FORBIDDEN_READ_ROOTS=str(campaign["paths"]["structures"]))
    assert proc.returncode != 0 and "were not found in the recorded structure directory" in proc.stderr


def test_replay_comparison_allows_a_class_change_only_at_a_near_tie():
    zeros = {column: "0.0" for column in export.LOGIT_COLUMNS}
    keys = ("structure_id", "pocket_id")
    tie = [{"structure_id": "s", "pocket_id": "p", "pred": "0", **zeros, "logit_1": "1.0", "logit_2": "0.9999995"}]
    flipped = [{**tie[0], "pred": "1", "logit_1": "0.9999995", "logit_2": "1.0"}]
    largest, problems, near_ties = export.compare_prediction_rows(tie, flipped, key_columns=keys, logit_atol=1e-6)
    assert not problems and near_ties == 1 and largest == pytest.approx(5e-7)
    wide = [{**tie[0], "logit_2": "0.5"}]
    _largest, problems, near_ties = export.compare_prediction_rows(wide, [{**wide[0], "pred": "1"}], key_columns=keys,
                                                                   logit_atol=1e-6)
    assert near_ties == 0 and problems == ["('s', 'p')/pred: '0' != '1'"]
    _largest, problems, _near = export.compare_prediction_rows(tie, [{**tie[0], "logit_3": "1e-5"}], key_columns=keys,
                                                               logit_atol=1e-6)
    assert problems and "logit_3" in problems[0]


def test_recorded_input_guard_denies_reads_and_writes_elsewhere(tmp_path):
    allowed, output, other = tmp_path / "inputs", tmp_path / "out", tmp_path / "other"
    for directory in (allowed, output, other):
        directory.mkdir()
    (allowed / "a.txt").write_text("a")
    (other / "b.txt").write_text("b")
    code = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'src')!r})
from training.ec_prediction_export import install_recorded_input_guard
install_recorded_input_guard([{str(allowed)!r}], output_dir=Path({str(output)!r}))
assert Path({str(allowed / 'a.txt')!r}).read_text() == "a"
Path({str(output / 'c.txt')!r}).write_text("c")
for action in (lambda: Path({str(other / 'b.txt')!r}).read_text(), lambda: list(Path({str(other)!r}).iterdir()),
               lambda: Path({str(other / 'd.txt')!r}).write_text("d")):
    try:
        action()
    except PermissionError:
        continue
    raise SystemExit("not refused")
print("guarded")
"""
    proc = subprocess.run([PYTHON, "-c", code], env=_replay_env(), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and proc.stdout.strip() == "guarded", proc.stderr[-2000:]


# ---------------------------------------------------------------------------
# Configuration refusals
# ---------------------------------------------------------------------------

def test_export_is_off_by_default_and_needs_its_contract(tmp_path):
    from dataclasses import replace

    from training.config import parse_args

    default = parse_args(["--task", "ec"])
    assert default.export_ec_predictions is False and default.ec_probe_structure_ids is None
    assert export.validate_ec_export_configuration(default) is None
    membership = ["--fold-split-source", "membership", "--fold-membership-csv", "m.csv", "--fold-membership-sha256",
                  "0" * 64, "--n-folds", "2", "--fold-index", "0", "--no-prepare-missing-esm-embeddings"]
    good = ["--task", "ec", "--ec-class-vocabulary", "fixed_seven", "--checkpoint-rule", "terminal",
            "--export-ec-predictions", *membership]
    config = parse_args(good)
    assert config.export_ec_predictions and export.validate_ec_export_configuration(config) == "validation"
    probe = tmp_path / "probe.txt"
    probe.write_text("P0000__chain_A__EC_1.1.1.1\n")
    probe_args = ["--ec-probe-structure-ids", str(probe), "--ec-probe-structure-ids-sha256", _sha256(probe)]
    refit = parse_args(["--task", "ec", "--ec-class-vocabulary", "fixed_seven", "--checkpoint-rule", "terminal",
                        "--export-ec-predictions", *probe_args])
    assert export.validate_ec_export_configuration(refit) == "probe"
    for argv in (
        [value for value in good if value not in {"--ec-class-vocabulary", "fixed_seven"}],
        [value for value in good if value not in {"--checkpoint-rule", "terminal"}],
        ["--task", "metal", "--checkpoint-rule", "terminal", "--export-ec-predictions"],
        ["--task", "ec", "--ec-class-vocabulary", "fixed_seven", "--checkpoint-rule", "terminal",
         "--export-ec-predictions"],
        good + probe_args,
        ["--task", "ec", "--ec-class-vocabulary", "fixed_seven", "--checkpoint-rule", "terminal",
         "--export-ec-predictions", "--val-fraction", "0.3", *probe_args],
        ["--task", "ec", "--ec-class-vocabulary", "fixed_seven", "--checkpoint-rule", "terminal",
         "--export-ec-predictions", "--ec-probe-structure-ids", str(probe)],
        ["--task", "ec", *probe_args],
        good + ["--test-summary-csv", "test.csv"],
    ):
        with pytest.raises(SystemExit):
            parse_args(argv)
    for changes, message in (
        ({"ec_class_vocabulary": "observed"}, "fixed_seven"),
        ({"checkpoint_rule": "best_validation"}, "terminal"),
        ({"task": "metal"}, "--task ec"),
        ({"export_validation_predictions": True}, "--export-validation-predictions"),
        ({"run_test_eval": True}, "held-out"),
        ({"fold_split_source": "computed"}, "exactly one"),
    ):
        with pytest.raises(ValueError, match=message):
            export.validate_ec_export_configuration(replace(config, **changes))
    with pytest.raises(ValueError, match="requires --export-ec-predictions"):
        export.validate_ec_export_configuration(replace(default, ec_probe_structure_ids=str(probe)))
    with pytest.raises(export.EcPredictionExportError, match="Probe structure list"):
        export.validate_ec_export_configuration(replace(refit, ec_probe_structure_ids_sha256="0" * 64))


def test_campaign_family_labels():
    assert export.campaign_family({"model_architecture": "only_gvp"}) == "F1"
    assert export.campaign_family({"model_architecture": "only_esm"}) == "F2"
    assert export.campaign_family({"model_architecture": "gvp", "fusion_mode": "late_fusion"}) == "F3"
    assert export.campaign_family({"model_architecture": "gvp", "fusion_mode": "late_fusion",
                                   "use_global_esm_branch": True}) == "F4"
    assert export.campaign_family({"model_architecture": "only_esm", "use_global_esm_branch": True}) == "F5"
    assert export.campaign_family({"model_architecture": "only_global", "use_global_esm_branch": True}) == "C-G"
    assert export.campaign_family({"model_architecture": "gvp", "fusion_mode": "hybrid"}) is None


# ---------------------------------------------------------------------------
# predict_logits
# ---------------------------------------------------------------------------

def test_predict_logits_needs_no_targets():
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool

    from training.loop import evaluate_epoch_with_predictions, predict_logits

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 7)

        def forward(self, batch):
            return {"logits_ec": self.linear(global_mean_pool(batch.x, batch.batch))}

    graphs = [Data(x=torch.randn(4, 3), y_ec=torch.tensor([-1]), ec_group_id=torch.tensor([-1])) for _ in range(5)]
    model = Probe()
    result = predict_logits(model, DataLoader(graphs, batch_size=2, shuffle=False))
    assert tuple(result["ec_logits"].shape) == (5, 7) and result["graph_index"].tolist() == [0, 1, 2, 3, 4]
    assert result["ec_group_id"].tolist() == [-1] * 5 and "loss" not in result
    with pytest.raises(KeyError):
        evaluate_epoch_with_predictions(model, DataLoader(graphs, batch_size=2))


# ---------------------------------------------------------------------------
# Label-free test pass
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blind(campaign):
    """Label-blind structures named with the placeholder, an accession-keyed site table and a chain-mean cache."""
    base = campaign["base"] / "blind"
    proteins = _two_site_proteins({f"T{index:04d}": {"ec": export.TEST_PLACEHOLDER_EC,
                                                     "ions": [("A", 900, "", "ZN", [10.0 * index, 7.0, 0.0])]}
                                   for index in range(5)})
    paths = write_ec_dataset(base, proteins)
    site_table = base / "test_sites.csv"
    site_table.write_text("accession,chain_resi,metaltype\n"
                          + "".join(f"{acc},{site},ZN\n" for acc in proteins for site in sorted(SITE_IDS)))
    cache, cache_sha = _chain_mean_cache(paths["structures"], paths["esm"], paths["stems"].values(), base / "chain_means")
    return {"paths": paths, "site_table": site_table, "cache": cache, "cache_sha": cache_sha,
            "accessions": sorted(proteins)}


def _freeze2(tmp_path: Path, checkpoint: Path, *, code: dict[str, str] | None = None, drop: tuple[str, ...] = (),
             output_root: Path | None = None) -> tuple[Path, str]:
    """A Freeze-2 manifest pinning the live required code (``code`` overrides or adds entries, ``drop`` removes)."""
    pinned = {path: _sha256(export.REPO_ROOT / path) for path in export.freeze2_required_code_files()}
    pinned.update(code or {})
    for path in drop:
        pinned.pop(path)
    manifest = {"schema_version": 1, "checkpoint_sha256s": [_sha256(checkpoint)], "code_sha256": pinned,
                "test_pass_output_root": str(output_root or tmp_path / "test_passes")}
    path = tmp_path / "freeze2_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return path, _sha256(path)


def _test_pass(campaign, blind, tmp_path, family, pass_id, *, manifest=None, structure_dir=None, site_table=None,
               output_root=None):
    checkpoint = campaign["runs"][family] / "terminal_model_checkpoint.pt"
    manifest_path, manifest_sha = manifest or _freeze2(tmp_path, checkpoint, output_root=output_root)
    rich = family == "rich"
    return export.export_ec_test_predictions(
        manifest_path, manifest_sha, pass_id, checkpoint_path=checkpoint,
        structure_dir=structure_dir or blind["paths"]["structures"], site_table_csv=site_table or blind["site_table"],
        output_root=output_root or tmp_path / "test_passes", esm_embeddings_dir=blind["paths"]["esm"],
        external_features_root_dir=blind["paths"]["external"],
        global_esm_cache_dir=blind["cache"] if rich else None,
        global_esm_manifest_sha256=blind["cache_sha"] if rich else None,
    )


def test_freeze2_required_code_covers_the_prediction_path():
    required = export.freeze2_required_code_files()
    for path in ("src/training/ec_prediction_export.py", "src/training/run.py", "src/export_validation_predictions.py",
                 "src/graph/construction.py", "src/graph/structure_parsing.py", "src/model_variants/factory.py",
                 "src/model_variants/models.py", "src/training/structure_loading.py", "src/training/site_filter.py",
                 "src/training/graph_dataset.py", "src/training/global_esm_features.py", "src/ec_auxiliary_objectives.py",
                 "src/training/loop.py", "src/model.py"):
        assert path in required, path
    assert all(path.startswith("src/") and path.endswith(".py") for path in required)


@pytest.mark.parametrize("family", TEST_PASS_FAMILIES)
def test_test_pass_is_label_free_and_write_once(campaign, blind, tmp_path, monkeypatch, family):
    import training.data
    import training.ec_classes
    import training.labels
    import training.structure_loading

    seen: list[tuple[str, ...]] = []
    original = training.structure_loading.parse_structure_ec_numbers

    def spy(structure_id):
        values = original(structure_id)
        seen.append(values)
        return values

    def refuse(*args, **kwargs):
        raise AssertionError("the test pass touched a label path")

    monkeypatch.setattr(training.structure_loading, "parse_structure_ec_numbers", spy)
    for module, name in ((training.ec_classes, "classify_ec_class"), (training.data, "apply_fixed_ec_class_vocabulary"),
                         (training.data, "assign_ec_targets"), (training.labels, "assign_ec_targets")):
        monkeypatch.setattr(module, name, refuse)
    receipt = _test_pass(campaign, blind, tmp_path, family, "pass-1")
    assert set(seen) == {(export.TEST_PLACEHOLDER_EC,)}
    assert receipt["labels_read"] is False and receipt["metrics_computed"] is False and not receipt["failures"]
    assert receipt["n_proteins_predicted"] == len(blind["accessions"]) and receipt["site_extraction"] == "selected_sites"
    assert receipt["checkpoint_site_extraction"] == "selected_sites"
    assert receipt["freeze2_manifest"]["n_code_files_verified"] == len(export.freeze2_required_code_files())
    assert receipt["freeze2_manifest"]["test_pass_output_root"] == str((tmp_path / "test_passes").resolve())
    pass_dir = tmp_path / "test_passes" / "pass-1"
    rows = _rows(pass_dir / export.TEST_PROTEIN_PREDICTIONS_NAME)
    assert list(rows[0]) == list(export.TEST_PROTEIN_PREDICTION_COLUMNS) and "y" not in rows[0]
    assert [row["accession"] for row in rows] == blind["accessions"]
    assert {row["n_pockets"] for row in rows} == {"2"}
    for path in pass_dir.iterdir():
        assert not re.search(r"__EC_(?!0\.0\.0\.0)", path.read_text()), path.name
    ledger_path = tmp_path / "test_passes" / export.TEST_PASS_LEDGER_NAME
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [entry["status"] for entry in ledger] == ["claimed", "completed"]
    shutil.rmtree(pass_dir)
    with pytest.raises(export.EcPredictionExportError, match="already used"):
        _test_pass(campaign, blind, tmp_path, family, "pass-1")
    (tmp_path / "test_passes" / "pass-2").mkdir()
    with pytest.raises(export.EcPredictionExportError, match="already exists"):
        _test_pass(campaign, blind, tmp_path, family, "pass-2")


def test_test_pass_refusals(campaign, blind, tmp_path):
    checkpoint = campaign["runs"]["plain"] / "terminal_model_checkpoint.pt"
    manifest_path, manifest_sha = _freeze2(tmp_path, checkpoint)
    output_root = tmp_path / "test_passes"
    with pytest.raises(export.EcPredictionExportError, match="not found"):
        _test_pass(campaign, blind, tmp_path, "plain", "p", manifest=(tmp_path / "absent.json", manifest_sha))
    with pytest.raises(export.EcPredictionExportError, match="SHA-256"):
        _test_pass(campaign, blind, tmp_path, "plain", "p", manifest=(manifest_path, "0" * 64))
    with pytest.raises(export.EcPredictionExportError, match="does not list this checkpoint"):
        _test_pass(campaign, blind, tmp_path, "rich", "p", manifest=(manifest_path, manifest_sha))
    with pytest.raises(export.EcPredictionExportError, match="Invalid pass ID"):
        _test_pass(campaign, blind, tmp_path, "plain", "../p")

    # The manifest pins the live code of the prediction path and the one output root.
    for kwargs, message in (
        ({"code": {export.EXPORT_SOURCE_KEY: "0" * 64}}, r"Live code differs .*ec_prediction_export\.py"),
        ({"code": {"src/training/run.py": "0" * 64}}, r"Live code differs .*training/run\.py"),
        ({"drop": ("src/graph/construction.py",)}, "does not pin 1 required code file"),
        ({"code": {"src/not_a_module.py": "0" * 64}}, "missing from this tree"),
        ({"code": {"DeepMzyme_Data/table.csv": "0" * 64}}, "not a src/...py path"),
        ({"code": {"src/../setup.py": "0" * 64}}, "not a src/...py path"),
    ):
        with pytest.raises(export.EcPredictionExportError, match=message):
            _test_pass(campaign, blind, tmp_path, "plain", "p", manifest=_freeze2(tmp_path, checkpoint, **kwargs))
    with pytest.raises(export.EcPredictionExportError, match="test_pass_output_root"):
        _test_pass(campaign, blind, tmp_path, "plain", "p", manifest=_freeze2(tmp_path, checkpoint),
                   output_root=tmp_path / "elsewhere")

    # A checkpoint trained on component pockets is not run on selected-site pockets.
    with pytest.raises(export.EcPredictionExportError, match="--site-extraction selected_sites"):
        _test_pass(campaign, blind, tmp_path, "component", "p")

    labelled = tmp_path / "labelled"
    shutil.copytree(blind["paths"]["structures"], labelled)
    shutil.copy(next(campaign["paths"]["structures"].glob("*.pdb")), labelled)
    with pytest.raises(export.EcPredictionExportError, match="placeholder") as refused:
        _test_pass(campaign, blind, tmp_path, "plain", "p", structure_dir=labelled)
    assert not re.search(r"\d\.\d+\.\d+\.\d+", str(refused.value).replace(export.TEST_PLACEHOLDER_SUFFIX, ""))

    with_ec = tmp_path / "sites_with_ec.csv"
    with_ec.write_text("accession,chain_resi,metaltype,ecnumber\nT0000,A_900,ZN,0.0.0.0\n")
    with pytest.raises(export.EcPredictionExportError, match="exactly the columns"):
        _test_pass(campaign, blind, tmp_path, "plain", "p", site_table=with_ec)
    with pytest.raises(export.EcPredictionExportError, match="whole-protein branch"):
        export.export_ec_test_predictions(
            *_freeze2(tmp_path, campaign["runs"]["rich"] / "terminal_model_checkpoint.pt"), "p",
            checkpoint_path=campaign["runs"]["rich"] / "terminal_model_checkpoint.pt",
            structure_dir=blind["paths"]["structures"], site_table_csv=blind["site_table"], output_root=output_root,
            esm_embeddings_dir=blind["paths"]["esm"], external_features_root_dir=blind["paths"]["external"])
    assert not output_root.exists(), "a refused pass claims nothing and writes nothing"
    assert not (tmp_path / "elsewhere").exists()


def test_test_pass_refuses_an_unreadable_ledger(campaign, blind, tmp_path):
    output_root = tmp_path / "test_passes"
    output_root.mkdir()
    (output_root / export.TEST_PASS_LEDGER_NAME).write_text('{"pass_id": "earlier", "status": "completed"}\nnot json\n')
    with pytest.raises(export.EcPredictionExportError, match="line 2 is unreadable"):
        _test_pass(campaign, blind, tmp_path, "plain", "next")
    assert sorted(path.name for path in output_root.iterdir()) == [export.TEST_PASS_LEDGER_NAME]
