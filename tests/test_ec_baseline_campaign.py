"""Campaign orchestration checks; never run training or access Colab/Drive."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_ec_baselines as campaign


def test_exact_notebook_grid_and_protected_test(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    train = data / campaign.DATASET / "train"
    train.mkdir(parents=True)
    (train / "final_data_summarazing_table_transition_metals_only_catalytic.csv").write_text(
        "structure,chain_resi,metaltype,ecnumber\n")
    (data / "esm_embeddings").mkdir()
    (data / "esm_embeddings/fixture_esmc.pt").touch()
    output = tmp_path / "output"
    monkeypatch.setattr(campaign, "storage_check", lambda p: p.mkdir(parents=True, exist_ok=True))
    campaign.plan(root, data, output, "fixture_commit")
    plan = json.loads((output / "campaign_plan.json").read_text())
    assert {(r["family"], r["lr"], r["seed"]) for r in plan["runs"]} == {
        (family, lr, seed) for family in campaign.FAMILIES for lr in (3e-5, 1e-4) for seed in (42, 43)}
    for row in plan["runs"]:
        config = campaign.parse_args(row["command"][2:])
        campaign.validate(config)
        assert config.structure_dir == train
        assert not any(flag in row["command"] for flag in
                       ("--run-test-eval", "--test-structure-dir", "--test-summary-csv"))


def test_summary_uses_selected_epoch_and_keeps_lr_groups_separate(tmp_path):
    reference = dict(n_train_pockets=993, n_val_pockets=175, n_train_ec_groups=751, n_val_ec_groups=42,
                     ec_labels={str(i): str(i + 1) for i in range(7)},
                     retained_split_identity={p: {"examples": [p]} for p in ("train", "validation")},
                     ec_group_weighting="structure_id", ec_class_weight_unit="group",
                     ec_group_metric_mode="structure_id", eligibility="ec_label_required")
    campaign.save(tmp_path / "expected_split.json", reference)
    live = dict(reference, ec_labels={int(k): v for k, v in reference["ec_labels"].items()})
    campaign.assert_identity(live, reference)
    runs = []
    for lr in (3e-5, 1e-4):
        for seed, score in ((42, .2), (43, .4)):
            directory = tmp_path / f"{lr}_{seed}"
            directory.mkdir()
            runs.append(dict(family="Only-GVP", lr=lr, seed=seed, run_dir=str(directory)))
            campaign.save(directory / "dataset_summary.json", reference)
            campaign.save(directory / "run_metadata.json", dict(selected_checkpoint_epoch=2, test_report=None))
            history = [{"epoch": i, campaign.METRIC: score if i == 2 else .99,
                        "val_ec_group_per_class_recall": {str(k): score if i == 2 else .99 for k in range(1, 8)}}
                       for i in range(1, 31)]
            campaign.save(directory / "run_config.json", dict(history=history))
            (directory / "epoch_metrics.csv").write_text("epoch\n" + "\n".join(str(i) for i in range(1, 31)))
            for name in ("best_model_checkpoint.pt", "last_model_checkpoint.pt"):
                (directory / name).touch()
    campaign.save(tmp_path / "campaign_plan.json", dict(runs=runs))
    campaign.summarize(tmp_path)
    result = json.loads((tmp_path / "validation_results.json").read_text())
    assert result["completed_runs"] == 4
    assert len(result["family_lr_summary"]) == 2
    for row in result["family_lr_summary"]:
        assert abs(row["mean"] - .3) < 1e-9
        assert all(abs(value - .3) < 1e-9 for value in row["per_class_recall_mean"].values())
    assert not result["promoted"]


def test_identity_matches_successful_smoke_schema():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/notebook_outputs/raw/colab_care_cache_smoke_20260914/care_smoke_stock/only_gvp"
    reference = json.loads(next(evidence.rglob("dataset_summary.json")).read_text())
    live = dict(reference, ec_labels={int(k): v for k, v in reference["ec_labels"].items()})
    campaign.assert_identity(live, reference)


def test_explicit_notebook_output_does_not_attempt_drive_mount():
    import ast
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/DeepMzyme_training_colab.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "75e1f97ec96a047f")
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "default_runs_root")
    ns = {"Path": Path, "CONFIG": {"output": {"runs_dir": "/content/explicit_campaign"}}, "resolve_path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_output", "exec"), ns)
    assert ns["default_runs_root"]() == Path("/content/explicit_campaign")
