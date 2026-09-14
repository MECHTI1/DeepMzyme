"""Standalone comparison safeguards; synthetic data only."""
from dataclasses import replace
from pathlib import Path
import json
import re
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_structures import PocketRecord
from training.config import parse_args
from training.loop import balanced_class_weights_from_pockets
from training.splits import assign_ec_group_metadata, split_pockets, split_pockets_k_fold
from training.run import validate_training_configuration, metrics_from_predictions
from training.preflight import run_preflight_checks
from training.config import TrainConfig
from training.splits import PocketSplit
from training.structure_loading import pocket_has_required_supervision
from label_schemes import active_metal_label_scheme_name, configure_active_metal_label_scheme


@pytest.fixture(autouse=True)
def restore_scheme(monkeypatch):
    original = active_metal_label_scheme_name()
    monkeypatch.setenv("DEEPGM_METAL_LABEL_SCHEME", original)
    yield
    configure_active_metal_label_scheme(original)


def examples():
    return [PocketRecord(
        structure_id=f"p{i:03d}__chain_A__EC_{i % 3 + 1}.1.1.1",
        pocket_id=f"p{i:03d}_site{j}",
        metal_element=["MN", "CU", "ZN", "FE", "CO", "NI"][i % 6],
        metal_coords=[torch.zeros(3)], residues=[], y_metal=i % 6, y_ec=i % 3,
    ) for i in range(90) for j in range(i % 4 + 1)]


@pytest.mark.parametrize("fold", [None, 0, 1, 2, 3, 4])
def test_paired_metal_split_does_not_depend_on_training_targets(fold):
    six = examples()
    four = [replace(p, y_metal=min(p.y_metal, 3)) for p in six]
    kwargs = dict(split_by="pdbid", seed=42, task="metal", stratify_by="metal_site")
    if fold is None:
        split = lambda ps: split_pockets(ps, val_fraction=0.15, **kwargs)
    else:
        split = lambda ps: split_pockets_k_fold(ps, n_folds=5, fold_index=fold, **kwargs)
    a, b = split(six), split(four)
    for part in ("train_pockets", "val_pockets"):
        assert [p.pocket_id for p in getattr(a, part)] == [p.pocket_id for p in getattr(b, part)]
    assert not ({p.structure_id for p in a.train_pockets} & {p.structure_id for p in a.val_pockets})
    assert a.train_pockets and a.val_pockets
    assert {p.y_metal for p in a.train_pockets + a.val_pockets} == set(range(6))


def test_ec_class_weights_are_invariant_to_repeated_pockets():
    ps = examples()[:1] + [replace(examples()[1], y_ec=1, structure_id="other__chain_A__EC_2.1.1.1")]
    repeated = ps[:1] * 20 + ps[1:]
    for pockets in (ps, repeated):
        assign_ec_group_metadata(pockets, weighting_mode="structure_id")
        _, weights = balanced_class_weights_from_pockets(pockets, 6, 2, ec_class_weight_unit="group")
        torch.testing.assert_close(weights, torch.ones(2))


def test_group_class_weights_reject_conflicting_labels():
    ps = examples()[:2]
    for p in ps:
        p.metadata["ec_group_key"] = "same"
    ps[1].y_ec = 2
    with pytest.raises(ValueError, match="Conflicting"):
        balanced_class_weights_from_pockets(ps, 6, 3, ec_class_weight_unit="group")


def test_new_controls_parse_without_enabling_test():
    config = parse_args(["--task", "metal", "--metal-label-scheme", "six_class",
                         "--split-stratify-by", "metal_site", "--split-seed", "42",
                         "--val-fraction", "0.15", "--selection-metric",
                         "val_metal_collapsed4_balanced_acc", "--ec-class-weight-unit", "group"])
    assert config.split_stratify_by == "metal_site"
    assert config.ec_class_weight_unit == "group"
    assert not config.run_test_eval


def test_metal_site_stratification_cannot_silently_change_ec_split():
    with pytest.raises(ValueError, match="requires task"):
        split_pockets(examples(), 0.15, "pdbid", 42, task="ec", stratify_by="metal_site")


def test_required_validation_classes_fail_closed():
    ps = examples()
    for p in ps:
        p.residues = [object()]  # Graphs are supplied; feature access must not be reached.
    split = PocketSplit(ps[:12], [replace(ps[-1], y_ec=0)])
    config = TrainConfig(task="ec", val_fraction=0.15, require_all_task_classes=True)
    with pytest.raises(ValueError, match="Validation split is missing EC classes"):
        run_preflight_checks(split, config, ec_label_map={0: "1", 1: "2", 2: "3"},
                             train_graphs=[None] * len(split.train_pockets), val_graphs=[None])


def test_paired_eligibility_excludes_mixed_sites_that_merge_only_in_four_class():
    p = replace(examples()[0], y_metal=3)
    p.metadata["matched_summary_site_metal_types"] = ["FE", "CO"]
    assert pocket_has_required_supervision(p, ("metal",))
    assert not pocket_has_required_supervision(p, ("metal",), "six_class")
    assert pocket_has_required_supervision(p, ("ec",), "six_class")
    p.metadata["matched_summary_site_metal_types"] = ["FE"]
    assert pocket_has_required_supervision(p, ("metal",), "six_class")


def test_ec_group_diagnostics_count_groups_not_pockets():
    metrics = metrics_from_predictions(
        {"loss": 0.0, "ec_logits": torch.tensor([[5., 0.], [5., 0.], [5., 0.]]),
         "ec_y": torch.tensor([0, 0, 1]), "ec_group_id": torch.tensor([0, 0, 1])},
        "val", task="ec", ec_label_map={0: "1", 1: "2"}, ec_label_depth=1,
    )
    assert metrics["val_ec_group_per_class_support"] == {"1": 1, "2": 1}
    assert metrics["val_ec_group_per_class_recall"] == {"1": 1.0, "2": 0.0}
    assert metrics["val_ec_group_min_recall"] == 0.0
    assert metrics["val_ec_group_level_1_balanced_acc"] == 0.5


@pytest.mark.parametrize("task,scheme", [("metal", "four_class"), ("metal", "six_class"), ("ec", "four_class")])
@pytest.mark.parametrize("family", ["Only-GVP", "Only-ESM", "GVP + late fusion"])
@pytest.mark.parametrize("phase", ["smoke", "baseline"])
def test_playbook_expands_through_current_notebook(tmp_path, task, scheme, family, phase):
    """Exercise real command expansion with synthetic paths, never run training."""
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks/DeepMzyme_training_colab.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    doc = (root / f"docs/{task.upper()}_TRAINING_PIPELINE_PLAYBOOK.md").read_text()
    block = re.findall(r"```python\n(.*?)```", doc, flags=re.S)[0]
    block = block.replace('STANDALONE_PHASE = "smoke"', f'STANDALONE_PHASE = "{phase}"')
    block = block.replace('MODEL_PRESET = "Only-GVP"', f'MODEL_PRESET = "{family}"')
    block = block.replace('METAL_LABEL_SCHEME = "four_class"', f'METAL_LABEL_SCHEME = "{scheme}"')
    ns = {"IN_COLAB": False, "NOTEBOOK_START_CWD": root, "DEVICE": "cpu", "Path": Path}
    exec(cells["eb4db512"], ns)
    exec(block, ns)
    ns.update(DRIVE_ROOT=str(tmp_path / "drive"), RUNS_DIR=str(tmp_path / "runs"),
              ESM_EMBEDDINGS_DIR=str(tmp_path / "esm"), COPY_OUTPUTS_TO_DRIVE=False, DEVICE="cpu")
    (tmp_path / "esm").mkdir()
    (tmp_path / "esm" / "fixture_esmc.pt").touch()  # Discovery only; never loaded.
    train = tmp_path / "train"
    train.mkdir()
    csv = train / "summary.csv"
    csv.write_text("structure,chain_resi,metaltype,ecnumber\n")
    exec(cells["ba89d9f5"], ns)
    ns.update(REPO_DIR=root, SRC_DIR=root / "src", TRAIN_DIR=train,
              TRAIN_SITE_SUMMARY_CSV=csv, TRAIN_CSV=csv,
              TEST_DIR=tmp_path / "protected_test", TEST_SITE_SUMMARY_CSV=tmp_path / "protected_test.csv",
              TEST_CSV=tmp_path / "protected_test.csv", TRAIN_STRUCTURES=[], TEST_STRUCTURES=[],
              DATA_ROOT=tmp_path, DATASET_ROOT=tmp_path / "dataset", DRIVE_DATA_DIR=tmp_path / "drive_data")
    exec(cells["75e1f97ec96a047f"], ns)
    runs = ns["planned_runs"]
    assert len(runs) == (1 if phase == "smoke" else 4)
    expected_metric = ("val_ec_group_level_1_balanced_acc" if task == "ec" else
                       "val_metal_balanced_acc" if scheme == "four_class" else
                       "val_metal_collapsed4_balanced_acc")
    for row in runs:
        command = list(map(str, row["command"]))
        assert not any(flag in command for flag in ("--run-test-eval", "--test-structure-dir", "--test-summary-csv"))
        config = parse_args(command[2:])
        validate_training_configuration(config)
        assert config.task == task
        assert config.selection_metric == expected_metric
        assert config.epochs == (1 if phase == "smoke" else 50 if task == "metal" else 30)
        assert config.split_seed == 42 and config.val_fraction == 0.15
        assert config.train_val_split_by == "pdbid"
        assert config.split_stratify_by == ("metal_site" if task == "metal" else "active_targets")
        assert config.metal_eligibility_scheme == ("six_class" if task == "metal" else "active")
        assert config.ec_label_depth == 1
        if task == "ec":
            assert config.ec_class_weight_unit == "group"
            assert config.ec_group_weighting == "structure_id"
        assert not config.use_ring_edges
        assert not config.use_early_esm
        assert not config.run_test_eval
        assert config.ec_contrastive_weight == 0
    assert not ns["LAUNCH_PLANNED_MAIN_TRAINING_RUNS"]
    assert not ns["LAUNCH_FINAL_HELD_OUT_TEST_EVAL"]


def test_chat4_snapshot_setup_preserves_main_bundle_selection(tmp_path, monkeypatch):
    """Exercise the upload override without a GPU, network, Drive or training."""
    import hashlib
    import io
    import tarfile
    import tempfile
    import types

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/DeepMzyme_training_colab.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    name = "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"
    data = (root / name).read_bytes()
    manifest = json.dumps({"base_commit": "fixture", "files": {
        name: hashlib.sha256(data).hexdigest()}}).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for filename, payload in [(name, data), ("code_snapshot_manifest.json", manifest)]:
            info = tarfile.TarInfo(filename)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    snapshot = buffer.getvalue()
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"snapshot.tar.gz": snapshot})
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(extracted))
    ns = {"IN_COLAB": False, "NOTEBOOK_START_CWD": root, "Path": Path}
    exec(cells["eb4db512"], ns)
    selected = {key: ns[key] for key in ("BUNDLE_FILENAME", "BUNDLE_URL", "BUNDLE_SHA256")}
    assert "_v12_" in selected["BUNDLE_FILENAME"]
    setup = cells["b16d9206c7e17d9e"].replace(
        "PREPARE_CHAT4_METAL_SMOKE = False", "PREPARE_CHAT4_METAL_SMOKE = True"
    ).replace('CHAT4_SNAPSHOT_SHA256 = ""',
              f'CHAT4_SNAPSHOT_SHA256 = "{hashlib.sha256(snapshot).hexdigest()}"')
    exec(setup, ns)
    assert {key: ns[key] for key in selected} == selected
    exec(cells["ba89d9f5"], ns)
    assert ns["CONFIG"]["data"]["bundle_sha256"] == selected["BUNDLE_SHA256"]
    assert ns["REPO_ROOT"] == str(extracted)
    assert ns["DEVICE"] == "cuda" and ns["EPOCHS"] == 1
    assert ns["MODEL_PRESET"] == "Only-GVP" and ns["TASK"] == "metal"
    assert not ns["INCLUDE_HELD_OUT_TEST_DURING_TRAINING"]
    assert not ns["LAUNCH_PLANNED_MAIN_TRAINING_RUNS"]
