"""Tests for v3 training options (plan step A2), all off by default."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training import campaign_runtime as runtime  # noqa: E402
from training.config import parse_args  # noqa: E402


def pocket(uid: str, element: str = "ZN"):
    return SimpleNamespace(pocket_id=f"p_{uid}", metal_element=element,
                           metadata={"source_uid": uid, "physical_ion_id": f"ion_{uid}"})


def membership(rows):
    return {uid: {"source_uid": uid, "physical_ion_id": f"ion_{uid}", "group_id": group,
                  "native_element": "ZN", "fold": str(fold)} for uid, group, fold in rows}


ROWS = [("a", "g1", 0), ("b", "g1", 0), ("c", "g2", 1), ("d", "g3", 2), ("e", "g4", 3), ("f", "g5", 4)]


def test_membership_split_takes_folds_from_the_file_in_cohort_order():
    pockets = [pocket(uid) for uid, _, _ in ROWS]
    split = runtime.membership_split(pockets, membership(ROWS), fold_index=0, n_folds=5)
    assert [p.metadata["source_uid"] for p in split.val_pockets] == ["a", "b"]
    assert [p.metadata["source_uid"] for p in split.train_pockets] == ["c", "d", "e", "f"]
    receipt = runtime.verify_fold_membership(train_pockets=split.train_pockets, val_pockets=split.val_pockets,
                                             fold_index=0, membership=membership(ROWS))
    assert receipt["verified"] and receipt["n_val"] == 2


@pytest.mark.parametrize("case", ["unknown", "missing", "twice", "range", "crossing", "empty_fold"])
def test_membership_split_refuses_every_mismatch(case):
    rows, pockets, fold, n_folds = list(ROWS), [pocket(uid) for uid, _, _ in ROWS], 0, 5
    if case == "unknown":
        pockets.append(pocket("zzz"))
    elif case == "missing":
        pockets = pockets[:-1]
    elif case == "twice":
        pockets.append(pocket("a"))
    elif case == "range":
        n_folds = 4
    elif case == "crossing":
        rows[2] = ("c", "g1", 1)
    elif case == "empty_fold":
        fold = 7
        n_folds = 8
    with pytest.raises(runtime.CampaignContractError):
        runtime.membership_split(pockets, membership(rows), fold_index=fold, n_folds=n_folds)


BASE = ["--task", "metal", "--metal-example-unit", "ion", "--source-cohort-csv", "c.csv",
        "--source-cohort-sha256", "0" * 64, "--fold-membership-csv", "f.csv", "--fold-membership-sha256", "0" * 64,
        "--n-folds", "5", "--fold-index", "1"]


def test_fold_split_source_defaults_to_computed_and_parses_membership():
    assert parse_args(BASE).fold_split_source == "computed"
    assert parse_args(BASE + ["--fold-split-source", "membership"]).fold_split_source == "membership"


def test_fold_split_source_membership_requires_cohort_and_fold_file():
    with pytest.raises(SystemExit):
        parse_args(["--task", "metal", "--fold-split-source", "membership"])


# ---------------------------------------------------------------------------
# End-to-end tiny fits (synthetic training-side campaign fixture)
# ---------------------------------------------------------------------------

def tiny_argv(tmp_path, monkeypatch, run_name, extra=()):
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pmm_campaign_fixtures import write_fake_embeddings
    from test_pmm_training_integrity import make_campaign
    from benchmarking import pmm_ion_campaign as campaign
    from training.source_cohort import sha256_file

    train, paths = make_campaign(tmp_path, monkeypatch, n_pdb=30)
    campaign.freeze_folds(paths)
    esm = tmp_path / "esm"
    write_fake_embeddings(train, esm, dim=8)
    return [
        "--task", "metal", "--metal-example-unit", "ion", "--metal-label-scheme", "six_class",
        "--metal-eligibility-scheme", "six_class", "--structure-dir", str(train),
        "--source-cohort-csv", str(paths.cohort), "--source-cohort-sha256", sha256_file(paths.cohort),
        "--fold-membership-csv", str(paths.fold_membership),
        "--fold-membership-sha256", sha256_file(paths.fold_membership),
        "--n-folds", "5", "--fold-index", "1", "--train-val-split-by", "pdbid", "--split-seed", "42",
        "--split-stratify-by", "metal_site", "--epochs", "3", "--batch-size", "8", "--seed", "42",
        "--model-architecture", "only_esm", "--learning-rate", "3e-4", "--hidden-s", "16",
        "--hidden-v", "4", "--edge-hidden", "8", "--gvp-layers", "2", "--esm-fusion-dim", "8",
        "--esm-dim", "8", "--esm-embeddings-dir", str(esm), "--no-prepare-missing-esm-embeddings",
        "--external-features-root-dir", str(tmp_path / "empty_ext"), "--allow-missing-external-features",
        "--omit-node-features", ",".join(campaign.OMITTED_EXTERNAL_FEATURES), "--shell-role-source", "geometry",
        "--no-prepare-missing-ring-edges", "--metal-class-weight-mode", "manual",
        "--selection-metric", "val_metal_balanced_acc", "--export-validation-predictions",
        "--campaign-run-identity", json.dumps({"fixture": True}),
        "--runs-dir", str(tmp_path / "runs"), "--load-workers", "1", "--device", "cpu",
        "--run-name", run_name, *extra,
    ]


def test_terminal_rule_selects_the_last_epoch_everywhere(tmp_path, monkeypatch):
    import csv
    import json

    import torch
    from training import run as run_module
    from training.campaign_runtime import replay_campaign_run

    # Make epoch 1 the "best" so the terminal rule must override it.
    monkeypatch.setattr(run_module, "metric_sort_value", lambda record, metric: (-float(record["epoch"]), True))
    argv = tiny_argv(tmp_path, monkeypatch, "terminal",
                     ["--checkpoint-rule", "terminal", "--lr-schedule", "cosine", "--fold-split-source", "membership"])
    run_dir = run_module.run_training(parse_args(argv))
    assert (run_dir / "terminal_model_checkpoint.pt").is_file()
    assert not (run_dir / "best_model_checkpoint.pt").exists()
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["selected_checkpoint_epoch"] == 3 and config["checkpoint_rule"] == "terminal"
    assert config["descriptive_best_epoch"] == 1
    receipt = json.loads((run_dir / "selected_checkpoint.json").read_text())
    assert receipt["selected_epoch"] == 3 and receipt["selected_checkpoint"] == "terminal_model_checkpoint.pt"
    assert receipt["checkpoint_rule"] == "terminal" and receipt["descriptive_best_epoch"] == 1
    rows = list(csv.DictReader((run_dir / "val_predictions.csv").open()))
    assert rows and {row["selected_epoch"] for row in rows} == {"3"}
    terminal = torch.load(run_dir / "terminal_model_checkpoint.pt", weights_only=False)
    last = torch.load(run_dir / "last_model_checkpoint.pt", weights_only=False)
    for name, tensor in last["model_state_dict"].items():
        assert torch.equal(tensor, terminal["model_state_dict"][name]), name
    replay = replay_campaign_run(run_dir, output_dir=tmp_path / "replay")
    assert replay["selected_epoch"] == 3 and replay["prediction_rows_verified"]


def test_default_rule_outputs_carry_no_terminal_fields(tmp_path, monkeypatch):
    import json

    from training import run as run_module

    run_dir = run_module.run_training(parse_args(tiny_argv(tmp_path, monkeypatch, "default")))
    assert (run_dir / "best_model_checkpoint.pt").is_file()
    assert not (run_dir / "terminal_model_checkpoint.pt").exists()
    config = json.loads((run_dir / "run_config.json").read_text())
    assert "descriptive_best_epoch" not in config and "checkpoint_rule" not in config
    receipt = json.loads((run_dir / "selected_checkpoint.json").read_text())
    assert "checkpoint_rule" not in receipt and receipt["tie_rule"].startswith("earliest epoch")
