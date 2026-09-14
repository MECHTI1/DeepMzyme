"""Synthetic prerequisite checks; no training or source dataset access."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_structures import PocketRecord, ResidueRecord
from label_schemes import (
    active_metal_label_scheme_name, configure_active_metal_label_scheme,
    map_site_metal_symbols_for_scheme, metal_labels_for_scheme,
)
from model import GVPPocketClassifier, TaskLossWeighter
from model_variants.models import OnlyESMPocketClassifier
from training.config import TrainConfig, parse_args, required_targets_for_task
from training.labels import ec_label_token_from_numbers
from training.run import resolve_selection_metric, validate_training_configuration
from training.splits import (
    PocketSplit, assign_ec_group_metadata, build_dataset_summary,
    pocket_split_key, retained_split_identity, split_pockets,
)
from training.structure_loading import pocket_has_required_supervision


@pytest.fixture(autouse=True)
def restore_scheme():
    original = active_metal_label_scheme_name()
    yield
    configure_active_metal_label_scheme(original)


def pocket(metal=0, ec=0, pdbid="1abc", site="site0"):
    return PocketRecord(
        structure_id=f"{pdbid}__chain_A__EC_1.1.1.1", pocket_id=site,
        metal_element="MN", metal_coords=[torch.zeros(3)], residues=[],
        y_metal=metal, y_ec=ec,
    )


def controlled_config(**changes):
    return replace(TrainConfig(
        task="joint", controlled_ec_auxiliary=True,
        metal_label_scheme="merge_fe_class_viii", joint_loss_weighting="fixed",
        val_fraction=0.15, train_val_split_by="pdbid", ec_contrastive_weight=0.0,
        selection_metric="val_ec_group_balanced_acc",
    ), **changes)


def test_exact_four_class_mapping():
    assert metal_labels_for_scheme("four_class") == {0: "Mn", 1: "Cu", 2: "Zn", 3: "Class VIII"}
    assert [map_site_metal_symbols_for_scheme(m, scheme_name="four_class")
            for m in ["Mn", "Cu", "Zn", "Fe", "Co", "Ni"]] == [0, 1, 2, 3, 3, 3]
    assert map_site_metal_symbols_for_scheme(["Fe", "Ni"], scheme_name="four_class") == 3


@pytest.mark.parametrize("metal,ec,expected", [
    (0, 0, (True, True, True)), (0, None, (True, False, False)),
    (None, 0, (False, True, False)), (None, None, (False, False, False)),
])
def test_task_eligibility(metal, ec, expected):
    assert tuple(pocket_has_required_supervision(pocket(metal, ec), required_targets_for_task(task))
                 for task in ("metal", "ec", "joint")) == expected


def test_ec_ambiguity_is_depth_specific():
    numbers = ("1.1.1.1", "1.2.3.4")
    assert ec_label_token_from_numbers(numbers, depth=1) == "1"
    assert ec_label_token_from_numbers(numbers, depth=2) is None
    assert ec_label_token_from_numbers(("1.1.1.1", "2.1.1.1"), depth=1) is None
    assert ec_label_token_from_numbers((), depth=1) is None


@pytest.mark.parametrize("mode", ["fixed", "uncertainty"])
def test_zero_metal_weight_removes_gradient(mode):
    metal = torch.tensor(2.0, requires_grad=True)
    ec = torch.tensor(3.0, requires_grad=True)
    weighter = TaskLossWeighter(mode=mode, metal_loss_weight=0.0)
    loss, _ = weighter({"metal": metal, "ec": ec})
    loss.backward()
    assert metal.grad is None
    assert ec.grad.item() == 1.0
    if mode == "uncertainty":
        assert weighter.metal_log_variance.grad is None


@pytest.mark.parametrize("model_type", [GVPPocketClassifier, OnlyESMPocketClassifier])
def test_zero_ec_weight_disables_contrastive_supervision(model_type):
    model = model_type(esm_dim=4, hidden_s=8, n_metal=4, n_ec=2,
                       ec_loss_weight=0.0, ec_contrastive_weight=1.0)
    embedding = torch.randn(4, 8, requires_grad=True)
    metal_logits = torch.randn(4, 4, requires_grad=True)
    ec_logits = torch.randn(4, 2, requires_grad=True)
    data = Data(y_metal=torch.tensor([0, 1, 2, 3]), y_ec=torch.tensor([0, 0, 1, 1]))
    loss, diagnostics = model._compute_supervised_loss(embedding, metal_logits, ec_logits, data)
    loss.backward()
    assert "ec_contrastive_loss_raw" not in diagnostics
    assert embedding.grad is None
    assert ec_logits.grad is None
    assert metal_logits.grad is not None


def test_controlled_pair_differs_only_in_metal_weight():
    a = controlled_config(metal_loss_weight=0.0)
    b = replace(a, metal_loss_weight=0.5)
    for config in (a, b):
        validate_training_configuration(config)
        assert required_targets_for_task(config.task) == ("metal", "ec")
    assert {k for k, v in asdict(a).items() if v != asdict(b)[k]} == {"metal_loss_weight"}
    assert resolve_selection_metric(replace(a, selection_metric=None)).selection_metric == "val_ec_group_balanced_acc"
    parsed = parse_args(["--controlled-ec-auxiliary", "--val-fraction", "0.15"])
    assert parsed.selection_metric == "val_ec_group_balanced_acc"


@pytest.mark.parametrize("changes", [
    {"task": "ec"}, {"model_architecture": "only_gvp"}, {"fusion_mode": "hybrid"},
    {"joint_loss_weighting": "uncertainty"}, {"metal_label_scheme": "six_class"},
    {"selection_metric": "val_joint_balanced_acc"}, {"selection_metric": "val_metal_balanced_acc"},
    {"selection_metric": "val_loss"}, {"selection_metric": "train_loss"},
    {"val_fraction": 0.0}, {"train_val_split_by": "pocket_id"},
    {"ec_group_weighting": "none"}, {"ec_loss_weight": 0.0}, {"ec_contrastive_weight": 0.1},
    {"run_test_eval": True}, {"allow_train_loss_test_eval_debug": True},
    {"allow_final_refit_test_eval": True},
])
def test_controlled_protocol_rejects_unmatched_or_test_controls(changes):
    with pytest.raises(ValueError, match="Controlled EC auxiliary comparison requires"):
        validate_training_configuration(controlled_config(**changes))


def test_retained_identity_captures_membership_order_targets_and_counts():
    pockets = [pocket(), pocket(site="site1"), pocket(pdbid="2abc")]
    identity = retained_split_identity(pockets, "pdbid")
    assert (identity["n_examples"], identity["n_structures"], identity["n_groups"]) == (3, 2, 2)
    for changed in (pockets[:2], pockets[::-1], [replace(pockets[0], y_ec=1), *pockets[1:]]):
        assert retained_split_identity(changed, "pdbid")["ordered_examples_sha256"] != identity["ordered_examples_sha256"]
    summary = build_dataset_summary(PocketSplit(pockets[:2], pockets[2:]), controlled_config(), {}, {0: "1"})
    assert summary["eligibility"] == "fully_labelled_intersection"
    assert summary["retained_split_identity"]["train"]["n_examples"] == 2


def test_late_fusion_predictions_do_not_consume_targets():
    from graph.construction import pocket_to_pyg_data
    from torch_geometric.data import Batch

    sample = pocket()
    sample.residues = [ResidueRecord(
        chain_id="A", resseq=i + 1, icode="", resname="CYS",
        atoms={"CA": torch.tensor([float(i + 1), 0., 0.]),
               "SG": torch.tensor([float(i + 1), 1., 0.])},
    ) for i in range(3)]
    graph = pocket_to_pyg_data(sample, esm_dim=4)
    batch = Batch.from_data_list([graph])
    model = GVPPocketClassifier(esm_dim=4, hidden_s=8, hidden_v=2, edge_hidden=4,
                               n_layers=1, n_metal=4, n_ec=2, fusion_mode="late_fusion")
    model.eval()
    with torch.no_grad():
        original = model(batch)
        batch.y_metal = torch.tensor([3])
        batch.y_ec = torch.tensor([1])
        changed = model(batch)
    for field in ("embed", "logits_ec", "logits_metal"):
        torch.testing.assert_close(original[field], changed[field], rtol=0, atol=0)


def test_single_task_selection_and_test_guard_remain_active():
    assert resolve_selection_metric(TrainConfig(task="ec", val_fraction=0.15)).selection_metric == "val_ec_group_balanced_acc"
    assert resolve_selection_metric(TrainConfig(task="metal", val_fraction=0.15)).selection_metric == "val_metal_balanced_acc"
    with pytest.raises(ValueError, match="Reportable --run-test-eval requires"):
        validate_training_configuration(TrainConfig(task="metal", joint_loss_weighting="fixed",
                                                    val_fraction=0.15, run_test_eval=True))


def test_controlled_arms_share_grouped_split_and_ec_weights():
    samples = [pocket(metal=i % 4, ec=i % 2, pdbid=f"{i:04d}", site=f"{i}_{j}")
               for i in range(20) for j in range(2)]
    identities = []
    for config in (controlled_config(metal_loss_weight=0.0), controlled_config(metal_loss_weight=0.5)):
        split = split_pockets(samples, val_fraction=config.val_fraction,
                              split_by=config.train_val_split_by, seed=config.seed, task=config.task)
        train_groups = {pocket_split_key(p, "pdbid") for p in split.train_pockets}
        val_groups = {pocket_split_key(p, "pdbid") for p in split.val_pockets}
        assert train_groups and val_groups and train_groups.isdisjoint(val_groups)
        assign_ec_group_metadata(split.train_pockets, weighting_mode="structure_id")
        assert all(p.metadata["ec_sample_weight"] == 0.5 for p in split.train_pockets)
        identities.append([retained_split_identity(part, "pdbid")
                           for part in (split.train_pockets, split.val_pockets)])
    assert identities[0] == identities[1]


def invoke_entrypoint(run, entrypoint, config, checkpoint):
    if entrypoint == "training":
        return run.run_training(config)
    return run.evaluate_saved_checkpoint(config, checkpoint)


@pytest.mark.parametrize("entrypoint", ["training", "saved_checkpoint"])
@pytest.mark.parametrize("changes,message", [
    ({"run_test_eval": True}, "Controlled EC auxiliary comparison requires"),
    ({"selection_metric": "val_joint_balanced_acc"}, "Controlled EC auxiliary comparison requires"),
    ({"controlled_ec_auxiliary": False, "run_test_eval": True}, "Reportable --run-test-eval requires"),
])
def test_entrypoints_reject_configuration_before_any_inspection(monkeypatch, entrypoint, changes, message):
    import training.run as run

    config = controlled_config(
        test_structure_dir=Path("synthetic-never-read/test"),
        test_summary_csv=Path("synthetic-never-read/summary.csv"), **changes,
    )
    blocked = {}
    for name in ("validate_held_out_structure_disjointness", "find_structure_files",
                 "prepare_run", "train_and_select_checkpoint", "evaluate_held_out_test_split"):
        blocked[name] = Mock(side_effect=AssertionError(f"Unexpected access: {name}"))
        monkeypatch.setattr(run, name, blocked[name])
    checkpoint = Mock()
    checkpoint.is_file.side_effect = AssertionError("Checkpoint path inspected before configuration rejection")
    with pytest.raises(ValueError, match=message):
        invoke_entrypoint(run, entrypoint, config, checkpoint)
    checkpoint.is_file.assert_not_called()
    for operation in blocked.values():
        operation.assert_not_called()


def test_valid_validation_only_run_reaches_preparation_without_membership_inspection(monkeypatch):
    import training.run as run

    class PreparationReached(Exception):
        pass

    config = controlled_config(selection_metric=None)
    inspection = Mock(side_effect=AssertionError("Validation-only run inspected membership"))
    preparation = Mock(side_effect=PreparationReached)
    guard = Mock(wraps=run.validate_held_out_structure_disjointness)
    monkeypatch.setattr(run, "held_out_structure_overlap_report", inspection)
    monkeypatch.setattr(run, "validate_held_out_structure_disjointness", guard)
    monkeypatch.setattr(run, "prepare_run", preparation)
    monkeypatch.setattr(run, "train_and_select_checkpoint", Mock(side_effect=AssertionError("Training forbidden")))
    with pytest.raises(PreparationReached):
        run.run_training(config)
    guard.assert_called_once()
    preparation.assert_called_once()
    assert guard.call_args.args[0].selection_metric == "val_ec_group_balanced_acc"
    inspection.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["training", "saved_checkpoint"])
def test_permitted_configuration_still_reaches_fail_closed_overlap_guard(monkeypatch, entrypoint):
    import training.run as run

    config = controlled_config(
        controlled_ec_auxiliary=False, run_test_eval=True,
        allow_final_refit_test_eval=True, final_test_selected_config_id="synthetic-selected-config",
        test_structure_dir=Path("synthetic-never-read/test"),
        test_summary_csv=Path("synthetic-never-read/summary.csv"),
    )
    events = []
    original_validator = run.validate_training_configuration
    def validate(candidate):
        original_validator(candidate)
        events.append("configuration_validated")
    def overlap_report(*args):
        events.append("overlap_guard")
        return {"train_test_overlap_detected": True, "overlap_counts": {"pdb_id": 1},
                "overlap_examples": {"pdb_id": ["synthetic"]}}
    preparation = Mock(side_effect=AssertionError("Preparation must not follow rejected overlap"))
    checkpoint_load = Mock(side_effect=AssertionError("Checkpoint must not load before overlap guard"))
    monkeypatch.setattr(run, "validate_training_configuration", validate)
    monkeypatch.setattr(run, "held_out_structure_overlap_report", overlap_report)
    monkeypatch.setattr(run, "prepare_run", preparation)
    monkeypatch.setattr(run.torch, "load", checkpoint_load)
    checkpoint = Mock()
    checkpoint.is_file.return_value = True
    with pytest.raises(RuntimeError, match="active held-out overlap policy is 'forbid'"):
        invoke_entrypoint(run, entrypoint, config, checkpoint)
    assert events == ["configuration_validated", "overlap_guard"]
    preparation.assert_not_called()
    checkpoint_load.assert_not_called()
