"""Targeted regressions for the PMM ion campaign contracts (plan Step 5.4).

All fixtures are synthetic training-side structures; no held-out example is used.
"""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from pmm_campaign_fixtures import LABEL_CODE, build_train_dir, standard_entries, write_fake_embeddings  # noqa: E402

from benchmarking import pmm_ion_cohort  # noqa: E402
from benchmarking import pmm_ion_campaign as campaign  # noqa: E402
from benchmarking.pmm_ion_analysis import classification_summary, fold_bootstrap, prediction_metrics  # noqa: E402
from label_schemes import configure_active_metal_label_scheme  # noqa: E402
from training.campaign_runtime import CampaignContractError, read_fold_membership, verify_fold_membership  # noqa: E402
from training.data import load_training_pockets_with_report_from_dir  # noqa: E402
from training.final_test_reporting import collapse_metal_probabilities  # noqa: E402
from training.source_cohort import CohortBindingError, read_cohort_csv, sha256_file  # noqa: E402

ESM_DIM = 8


@pytest.fixture(autouse=True)
def _restore_scheme():
    yield
    configure_active_metal_label_scheme("split_all_metals")


def _audit(tmp_path: Path, monkeypatch, entries: dict) -> tuple[Path, Path, dict]:
    train, source, source_sha = build_train_dir(tmp_path / "dataset", entries)
    monkeypatch.setattr(pmm_ion_cohort, "PMM_SOURCE_SHA256", source_sha)
    campaign_dir = tmp_path / "campaign"
    audit = pmm_ion_cohort.run_audit(train, source, campaign_dir, workers=1)
    return train, campaign_dir, audit


def _dispositions(campaign_dir: Path) -> dict[int, dict[str, str]]:
    with (campaign_dir / "train_row_dispositions.csv").open(encoding="utf-8", newline="") as handle:
        return {int(row["source_row"]): row for row in csv.DictReader(handle)}


# ---------------------------------------------------------------------------
# Step 1: provenance and example contract
# ---------------------------------------------------------------------------


def test_mixed_metal_siblings_remain_distinct_examples_in_one_group(tmp_path, monkeypatch):
    entries = standard_entries(6)
    train, campaign_dir, audit = _audit(tmp_path, monkeypatch, entries)
    assert audit["disposition_counts"]["retained"] == 12
    bindings = read_cohort_csv(campaign_dir / "train_cohort.csv")
    assert len({b.physical_ion_id for b in bindings}) == 12
    configure_active_metal_label_scheme("split_all_metals")
    result = load_training_pockets_with_report_from_dir(
        train, required_targets=("metal",), metal_example_unit="ion", esm_dim=ESM_DIM,
        esm_embeddings_dir=tmp_path / "no_esm", require_esm_embeddings=False,
        external_features_root_dir=tmp_path / "no_ext", require_external_features=False,
        source_cohort_csv=campaign_dir / "train_cohort.csv",
        source_cohort_sha256=sha256_file(campaign_dir / "train_cohort.csv"), load_workers=1,
    )
    pockets = result.pockets
    assert [p.metadata["source_uid"] for p in pockets] == [b.source_uid for b in bindings]
    first, second = pockets[0], pockets[1]
    assert first.structure_id == second.structure_id and first.metal_element != second.metal_element
    assert first.metal_count() == second.metal_count() == 1
    assert not torch.equal(first.metal_coords[0], second.metal_coords[0])
    rows = campaign.compute_fold_membership(bindings)
    folds_by_group: dict[str, set[int]] = {}
    for row in rows:
        folds_by_group.setdefault(row["group_id"], set()).add(row["fold"])
    assert all(len(folds) == 1 for folds in folds_by_group.values())


def test_aliases_conflicts_insertion_codes_altlocs_and_models(tmp_path, monkeypatch):
    zn, fe = (0.0, 0.0, 0.0), (3.2, 0.0, 0.0)
    entries = {
        "1dup": {"ions": [("A", 201, "", "ZN", zn)], "chains": ("A",),
                 "rows": [("A", 201, "ZN", "7"), ("A", 201, "ZN", "7")]},
        "1cfl": {"ions": [("A", 201, "", "FE", zn)], "chains": ("A",),
                 "rows": [("A", 201, "FE", "2"), ("A", 201, "FE", "7")]},
        "1ins": {"ions": [("A", 205, "", "ZN", zn), ("A", 205, "A", "ZN", fe)], "chains": ("A",),
                 "rows": [("A", 205, "ZN", "7")]},
        "1alt": {"ions": [("A", 201, "", "ZN", zn)], "chains": ("A",), "altloc_ion": True,
                 "rows": [("A", 201, "ZN", "7")]},
        "1nmr": {"ions": [("A", 201, "", "CU", zn)], "chains": ("A",), "n_models": 2,
                 "rows": [("A", 201, "CU", "6")]},
        "1bad": {"ions": [("A", 201, "", "ZN", zn)], "chains": ("A",), "rows": [("A", 201, "ZN", "1")]},
        "1mis": {"ions": [("A", 201, "", "ZN", zn)], "chains": ("A",), "rows": [("A", 999, "ZN", "7")]},
        "1wat": {"ions": [("A", 201, "", "HOH", fe), ("A", 201, "N", "NI", zn)], "chains": ("A",),
                 "rows": [("A", 201, "NI", "2")]},
    }
    _train, campaign_dir, audit = _audit(tmp_path, monkeypatch, entries)
    rows = _dispositions(campaign_dir)
    assert rows[1]["disposition"] == "retained" and rows[2]["disposition"] == "duplicate_alias"
    assert rows[2]["retained_source_uid"] == rows[1]["source_uid"]
    assert rows[3]["disposition"] == "conflicting_label" and rows[4]["disposition"] == "conflicting_label"
    assert rows[5]["reason"] == "ambiguous_insertion_code_or_residue"
    assert rows[6]["reason"] == "target_atom_alternate_locations"
    assert rows[7]["disposition"] == "retained" and "multi_model" in rows[7]["detail"]
    assert rows[8]["reason"] == "resolved_element_incompatible_with_source_code"
    assert rows[9]["reason"] == "target_residue_absent_in_model_0"
    cohort = read_cohort_csv(campaign_dir / "train_cohort.csv")
    assert rows[10]["disposition"] == "retained" and "non_metal_residue_same_number=HOH" in rows[10]["detail"]
    assert [b.source_row for b in cohort] == [1, 7, 10]
    assert cohort[0].alias_source_uids == (rows[2]["source_uid"],)
    assert cohort[2].icode == "N" and cohort[2].native_element == "NI"
    assert sum(audit["disposition_counts"].values()) == 10
    manifest = pmm_ion_cohort.verify_campaign_identity(_train, campaign_dir)
    assert manifest["cohort"]["n_rows"] == 3


def test_cohort_reader_rejects_repeated_uid_and_hash_drift(tmp_path, monkeypatch):
    _train, campaign_dir, _audit_payload = _audit(tmp_path, monkeypatch, standard_entries(1))
    path = campaign_dir / "train_cohort.csv"
    with pytest.raises(CohortBindingError):
        read_cohort_csv(path, expected_sha256="f" * 64)
    lines = path.read_text(encoding="utf-8").splitlines()
    duplicate = tmp_path / "dup.csv"
    duplicate.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(CohortBindingError):
        read_cohort_csv(duplicate)


def test_multi_model_file_uses_first_model_residues_only(tmp_path, monkeypatch):
    entries = {"1nmr": {"ions": [("A", 201, "", "CU", (0.0, 0.0, 0.0))], "chains": ("A",), "n_models": 3,
                        "rows": [("A", 201, "CU", "6")]}}
    train, campaign_dir, _ = _audit(tmp_path, monkeypatch, entries)
    result = load_training_pockets_with_report_from_dir(
        train, required_targets=("metal",), metal_example_unit="ion", esm_dim=ESM_DIM,
        esm_embeddings_dir=tmp_path / "no_esm", require_esm_embeddings=False,
        external_features_root_dir=tmp_path / "no_ext", require_external_features=False,
        source_cohort_csv=campaign_dir / "train_cohort.csv",
        source_cohort_sha256=sha256_file(campaign_dir / "train_cohort.csv"), load_workers=1,
    )
    residues = result.pockets[0].residues
    assert len({r.residue_id() for r in residues}) == len(residues) == 6


def test_complete_embeddings_are_required_for_every_context_residue(tmp_path, monkeypatch):
    train, campaign_dir, _ = _audit(tmp_path, monkeypatch, standard_entries(2))
    esm_dir = tmp_path / "esm"
    write_fake_embeddings(train, esm_dir, dim=ESM_DIM, skip_chain="B")
    options = dict(required_targets=("metal",), metal_example_unit="ion", esm_dim=ESM_DIM,
                   esm_embeddings_dir=esm_dir, require_esm_embeddings=True,
                   external_features_root_dir=tmp_path / "no_ext", require_external_features=False,
                   source_cohort_csv=campaign_dir / "train_cohort.csv",
                   source_cohort_sha256=sha256_file(campaign_dir / "train_cohort.csv"), load_workers=1)
    with pytest.raises(CohortBindingError):
        load_training_pockets_with_report_from_dir(train, **options)
    write_fake_embeddings(train, esm_dir, dim=ESM_DIM)
    pockets = load_training_pockets_with_report_from_dir(train, **options).pockets
    assert all(r.has_esm_embedding for p in pockets for r in p.residues)
    assert any(r.chain_id == "B" for r in pockets[0].residues)


# ---------------------------------------------------------------------------
# Step 3: folds, weights and identity
# ---------------------------------------------------------------------------


def _campaign_with_folds(tmp_path, monkeypatch, n_pdb=30):
    train, campaign_dir, _ = _audit(tmp_path, monkeypatch, standard_entries(n_pdb))
    paths = campaign.CampaignPaths(campaign_dir)
    weights = campaign.freeze_folds(paths)
    return train, paths, weights


def test_four_and_six_class_runs_share_identical_folds(tmp_path, monkeypatch):
    train, paths, _ = _campaign_with_folds(tmp_path, monkeypatch)
    membership = read_fold_membership(paths.fold_membership, sha256_file(paths.fold_membership))
    cohort_sha = sha256_file(paths.cohort)
    from training.splits import split_pockets_k_fold

    val_sets = {}
    for scheme in ("merge_fe_class_viii", "split_all_metals"):
        configure_active_metal_label_scheme(scheme)
        pockets = load_training_pockets_with_report_from_dir(
            train, required_targets=("metal",), metal_example_unit="ion", esm_dim=ESM_DIM,
            esm_embeddings_dir=tmp_path / "no_esm", require_esm_embeddings=False,
            external_features_root_dir=tmp_path / "no_ext", require_external_features=False,
            source_cohort_csv=paths.cohort, source_cohort_sha256=cohort_sha, load_workers=1,
        ).pockets
        for fold in range(campaign.N_FOLDS):
            split = split_pockets_k_fold(pockets, n_folds=5, fold_index=fold, split_by="pdbid", seed=42,
                                         task="metal", stratify_by="metal_site")
            receipt = verify_fold_membership(train_pockets=split.train_pockets, val_pockets=split.val_pockets,
                                             fold_index=fold, membership=membership)
            val_sets.setdefault(fold, []).append({p.metadata["source_uid"] for p in split.val_pockets})
            assert receipt["verified"]
    assert all(a == b for a, b in val_sets.values())
    with pytest.raises(CampaignContractError):
        split = split_pockets_k_fold(pockets, n_folds=5, fold_index=0, split_by="pdbid", seed=7,
                                     task="metal", stratify_by="metal_site")
        verify_fold_membership(train_pockets=split.train_pockets, val_pockets=split.val_pockets,
                               fold_index=0, membership=membership)


def test_class_weights_are_common_four_equalized_for_both_schemes(tmp_path, monkeypatch):
    _train, _paths, weights = _campaign_with_folds(tmp_path, monkeypatch)
    for fold in weights["folds"].values():
        four, six = fold["four_class_multipliers"], fold["six_class_multipliers"]
        assert six["fe"] == six["co"] == six["ni"] == four["class_viii"]
        assert {key: four[key] for key in ("mn", "cu", "zn")} == {key: six[key] for key in ("mn", "cu", "zn")}
        counts = fold["train_common_four_counts"]
        total = sum(counts.values())
        assert four["mn"] == pytest.approx(total / (4 * counts["Mn"]))


def _full_inventory(paths, tmp_path):
    esm_dir = tmp_path / "esm_certified"
    esm_dir.mkdir(exist_ok=True)
    campaign.write_json(paths.feature_inventory, {
        "certified": True, "certified_scope": "structure_and_esmc600m", "cohort_sha256": sha256_file(paths.cohort),
        "esm": {"embeddings_dir": str(esm_dir), "model_name": "esmc_600m"}, "certified_at": "t0", "load_seconds": 1.0})


def test_commands_bind_identity_and_forbid_test_paths(tmp_path, monkeypatch):
    train, paths, _ = _campaign_with_folds(tmp_path, monkeypatch)
    campaign.write_json(paths.feature_inventory, {"certified": True, "certified_scope": "structure_only",
                                                  "esm": None, "cohort_sha256": sha256_file(paths.cohort)})
    config = campaign.GridConfig("only_gvp", "six_class", "none")
    options = dict(python_bin=sys.executable, train_dir=train, fold=2, seed=42, device="cpu", runs_dir=paths.runs)
    # Grid fits wait for the complete inventory; a structure-only inventory admits smoke units only.
    with pytest.raises(ValueError, match="complete ESMC-600M"):
        campaign.build_train_command(paths, config=config, **options)
    campaign.build_train_command(paths, config=config, smoke=True, **options)
    _full_inventory(paths, tmp_path)
    command, env, identity = campaign.build_train_command(paths, config=config, **options)
    assert "--run-test-eval" not in command and "--test-structure-dir" not in command
    assert command[command.index("--metal-label-scheme") + 1] == "split_all_metals"
    assert "--fe-loss-multiplier" in command and "--class-viii-loss-multiplier" not in command
    assert identity["target_scheme"] == "six_class" and identity["fold"] == 2
    assert identity["source_tree_sha256"] == campaign.source_tree_sha256()
    assert len(identity["resolved_config_sha256"]) == 64
    forbidden = env[campaign.FORBIDDEN_READ_ROOTS_ENV]
    assert str((train.parent / "test").resolve()) in forbidden and "classmodel_test_set" in forbidden
    # Placement does not change identity; a scientific flag does.
    other = campaign.build_train_command(paths, config=config, **{**options, "device": "cuda", "load_workers": 4})[2]
    assert other == identity
    esm_config = campaign.GridConfig("only_esm", "four_class", "first_shell_bias")
    esm_identity = campaign.build_train_command(paths, config=esm_config, **options)[2]
    assert esm_identity["resolved_config_sha256"] != identity["resolved_config_sha256"]
    # Re-certification timestamps do not change the inventory identity.
    before = campaign.inventory_identity_sha256(paths)
    inventory = json.loads(paths.feature_inventory.read_text())
    inventory.update(certified_at="t1", load_seconds=99.0)
    campaign.write_json(paths.feature_inventory, inventory)
    assert campaign.inventory_identity_sha256(paths) == before
    assert len(campaign.grid_configs()) == 9
    assert len(campaign.selected_units(None, None, None, None)) == 45


def test_smoke_campaign_marks_folds_lacking_classes_not_runnable(tmp_path, monkeypatch):
    train, paths, _ = _campaign_with_folds(tmp_path, monkeypatch)
    _full_inventory(paths, tmp_path)
    smoke = campaign.prepare_smoke_campaign(paths, per_element_groups=5)
    weights = json.loads(smoke.fold_class_weights.read_text())["folds"]
    runnable = campaign.runnable_folds(smoke)
    assert runnable and all(weights[str(fold)]["runnable"] for fold in runnable)
    for fold, entry in weights.items():
        if not entry["runnable"]:
            assert entry["absent_native_classes"]
            with pytest.raises(ValueError, match="not runnable"):
                campaign.build_train_command(smoke, python_bin=sys.executable, train_dir=train,
                                             config=campaign.GridConfig("only_gvp", "four_class", "none"),
                                             fold=int(fold), seed=42, device="cpu", runs_dir=smoke.runs, smoke=True)
    # The full campaign stays strict; the smoke records non-runnable folds instead of failing.
    rows = [{"source_uid": "u", "fold": 0, "native_element": "ZN"}]
    with pytest.raises(ValueError, match="absent"):
        campaign.fold_class_weights(rows)
    lenient = campaign.fold_class_weights(rows, strict=False)["folds"]
    assert not lenient["0"]["runnable"] and "MN" in lenient["0"]["absent_native_classes"]


# ---------------------------------------------------------------------------
# Step 4: readout intervention
# ---------------------------------------------------------------------------


def _graph_batch(tmp_path, monkeypatch, *, empty_shell: bool = False):
    from torch_geometric.data import Batch

    from training.graph_dataset import build_graph_data_list

    train, campaign_dir, _ = _audit(tmp_path, monkeypatch, standard_entries(2))
    esm_dir = tmp_path / "esm"
    write_fake_embeddings(train, esm_dir, dim=ESM_DIM)
    configure_active_metal_label_scheme("merge_fe_class_viii")
    pockets = load_training_pockets_with_report_from_dir(
        train, required_targets=("metal",), metal_example_unit="ion", esm_dim=ESM_DIM,
        esm_embeddings_dir=esm_dir, require_esm_embeddings=True,
        external_features_root_dir=tmp_path / "no_ext", require_external_features=False,
        source_cohort_csv=campaign_dir / "train_cohort.csv",
        source_cohort_sha256=sha256_file(campaign_dir / "train_cohort.csv"), load_workers=1,
    ).pockets
    graphs = build_graph_data_list(pockets, esm_dim=ESM_DIM, edge_radius=8.0, use_ring_edges=False,
                                   shell_role_source="geometry", omit_node_features=campaign.OMITTED_EXTERNAL_FEATURES)
    if empty_shell:
        for graph in graphs:
            graph.x_role = torch.zeros_like(graph.x_role)
    assert all(bool((g.x_role[:, 0] > 0.5).any()) != empty_shell for g in graphs)
    return Batch.from_data_list(graphs), pockets, graphs


def _model(architecture: str, readout: str, seed: int = 0):
    from model_variants import build_pocket_classifier

    torch.manual_seed(seed)
    kwargs = dict(model_architecture=architecture, esm_dim=ESM_DIM, hidden_s=16, hidden_v=4, edge_hidden=8,
                  n_layers=2, n_metal=4, n_ec=1, esm_fusion_dim=8, predict_metal=True, predict_ec=False,
                  binding_residue_pooling=readout, node_rbf_use_raw_distances=architecture != "only_esm",
                  edge_rbf_use_raw_distances=architecture != "only_esm")
    if architecture == "gvp":
        kwargs["fusion_mode"] = "late_fusion"
    return build_pocket_classifier(**kwargs).eval()


@pytest.mark.parametrize("architecture", ["only_esm", "only_gvp", "gvp"])
def test_zero_bias_reproduces_baseline_and_bias_receives_gradients(tmp_path, monkeypatch, architecture):
    batch, _, _ = _graph_batch(tmp_path, monkeypatch)
    baseline, aware = _model(architecture, "none"), _model(architecture, "first_shell_bias")
    missing, unexpected = aware.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected and missing and all("binding_bias" in key for key in missing)
    for name, parameter in baseline.state_dict().items():
        assert torch.equal(parameter, aware.state_dict()[name]), name
    with torch.no_grad():
        torch.testing.assert_close(aware(batch)["logits_metal"], baseline(batch)["logits_metal"], atol=1e-6, rtol=1e-5)
    aware.train()
    torch.manual_seed(1)
    aware(batch)["loss"].backward()
    bias_grads = {name: p.grad for name, p in aware.named_parameters() if "binding_bias" in name}
    expected = 2 if architecture in {"only_esm", "only_gvp"} else 4
    assert len(bias_grads) == expected
    assert all(grad is not None and float(grad.abs()) > 0 for grad in bias_grads.values())


def test_empty_first_shell_keeps_the_ordinary_readout(tmp_path, monkeypatch):
    batch, _, _ = _graph_batch(tmp_path, monkeypatch, empty_shell=True)
    baseline, aware = _model("gvp", "none"), _model("gvp", "first_shell_bias")
    aware.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad():
        for name, parameter in aware.named_parameters():
            if "binding_bias" in name:
                parameter.fill_(3.0)
        torch.testing.assert_close(aware(batch)["logits_metal"], baseline(batch)["logits_metal"], atol=1e-6, rtol=1e-5)


def test_logits_are_invariant_to_labels_and_element_metadata(tmp_path, monkeypatch):
    from torch_geometric.data import Batch

    from training.graph_dataset import build_graph_data_list

    _batch, pockets, graphs = _graph_batch(tmp_path, monkeypatch)
    altered = copy.deepcopy(pockets)
    for pocket in altered:
        pocket.metal_element = "MN"
        pocket.y_metal = 0
        pocket.metadata["matched_summary_site_metal_types"] = ["MN"]
        pocket.metadata["metal_site_symbols"] = ["MN"]
    graphs_b = build_graph_data_list(altered, esm_dim=ESM_DIM, edge_radius=8.0, use_ring_edges=False,
                                     shell_role_source="geometry",
                                     omit_node_features=campaign.OMITTED_EXTERNAL_FEATURES)
    model = _model("gvp", "first_shell_bias")
    with torch.no_grad():
        torch.testing.assert_close(model(Batch.from_data_list(graphs))["logits_metal"],
                                   model(Batch.from_data_list(graphs_b))["logits_metal"])
    for graph in graphs:
        assert float(graph.x_env_burial.abs().sum()) == 0.0 and float(graph.x_env_electrostatics.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# Step 5: reporting, replay and test-access guard
# ---------------------------------------------------------------------------


def test_common_four_collapse_sums_probabilities_before_argmax():
    configure_active_metal_label_scheme("split_all_metals")
    probabilities = torch.tensor([[0.40, 0.0, 0.0, 0.25, 0.20, 0.15]])
    collapsed = collapse_metal_probabilities(probabilities)
    torch.testing.assert_close(collapsed, torch.tensor([[0.40, 0.0, 0.0, 0.60]]))
    assert int(probabilities.argmax()) == 0 and int(collapsed.argmax()) == 3


def test_fold_bootstrap_is_deterministic_and_bonferroni_is_wider():
    import numpy as np

    differences = np.array([0.01, 0.02, -0.005, 0.015, 0.03])
    first, second = fold_bootstrap(differences), fold_bootstrap(differences)
    assert first == second
    assert first["bonferroni_low"] <= first["ci95_low"] <= first["mean_difference"] <= first["ci95_high"]
    summary = classification_summary(np.array([0, 1, 2, 3, 3]), np.array([0, 1, 1, 3, 2]),
                                     ("Mn", "Cu", "Zn", "Class VIII"))
    assert summary["balanced_accuracy"] == pytest.approx((1 + 1 + 0 + 0.5) / 4)


def test_development_process_cannot_read_test_side(tmp_path):
    test_dir = tmp_path / "dataset" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "secret.csv").write_text("x\n", encoding="utf-8")
    code = (
        "import sys, os; sys.path.insert(0, %r)\n"
        "from training.access_guard import install_guard_from_environment\n"
        "install_guard_from_environment()\n"
        "results = []\n"
        "for action in (lambda: open(%r).read(), lambda: os.listdir(%r)):\n"
        "    try:\n        action(); results.append('allowed')\n"
        "    except PermissionError:\n        results.append('denied')\n"
        "print(','.join(results))\n"
    ) % (str(REPO_ROOT / "src"), str(test_dir / "secret.csv"), str(test_dir))
    env = {**__import__("os").environ, "DEEPMZYME_FORBIDDEN_READ_ROOTS": str(test_dir)}
    output = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert output.stdout.strip() == "denied,denied"


def test_training_run_exports_replayable_selected_checkpoint(tmp_path, monkeypatch):
    """End-to-end tiny CPU fit through the real training entry point (training data only)."""
    from model_variants import build_pocket_classifier
    from training.config import parse_args
    from training.run import run_training

    train, paths, _ = _campaign_with_folds(tmp_path, monkeypatch)
    esm_dir = tmp_path / "esm"
    write_fake_embeddings(train, esm_dir, dim=ESM_DIM)
    identity = {"campaign_id": "test", "fold": 1}
    argv = ["--task", "metal", "--metal-example-unit", "ion", "--metal-label-scheme", "six_class",
            "--metal-eligibility-scheme", "six_class", "--structure-dir", str(train),
            "--source-cohort-csv", str(paths.cohort), "--source-cohort-sha256", sha256_file(paths.cohort),
            "--fold-membership-csv", str(paths.fold_membership),
            "--fold-membership-sha256", sha256_file(paths.fold_membership),
            "--n-folds", "5", "--fold-index", "1", "--train-val-split-by", "pdbid", "--split-seed", "42",
            "--split-stratify-by", "metal_site", "--epochs", "2", "--batch-size", "8",
            "--model-architecture", "gvp", "--fusion-mode", "late_fusion", "--learning-rate", "3e-4",
            "--gvp-learning-rate", "3e-4", "--hidden-s", "16", "--hidden-v", "4", "--edge-hidden", "8",
            "--gvp-layers", "2", "--esm-fusion-dim", "8", "--esm-dim", str(ESM_DIM),
            "--esm-embeddings-dir", str(esm_dir), "--no-prepare-missing-esm-embeddings",
            "--external-features-root-dir", str(tmp_path / "no_ext"), "--allow-missing-external-features",
            "--omit-node-features", ",".join(campaign.OMITTED_EXTERNAL_FEATURES),
            "--shell-role-source", "geometry", "--no-prepare-missing-ring-edges",
            "--metal-class-weight-mode", "manual", "--binding-residue-pooling", "first_shell_bias",
            "--selection-metric", "val_metal_balanced_acc", "--export-validation-predictions",
            "--campaign-run-identity", json.dumps(identity), "--runs-dir", str(tmp_path / "runs"),
            "--run-name", "unit", "--load-workers", "1", "--rbf-use-raw-distances",
            "--train-metrics-every-n-epochs", "2"]
    run_dir = run_training(parse_args(argv))
    receipt = json.loads((run_dir / "selected_checkpoint.json").read_text(encoding="utf-8"))
    assert receipt["fit_status"] == "completed" and receipt["campaign_run_identity"] == identity
    assert campaign.completed_run_receipt(run_dir, identity) is not None
    assert campaign.completed_run_receipt(run_dir, {**identity, "fold": 2}) is None
    with (run_dir / "val_predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    membership = read_fold_membership(paths.fold_membership, sha256_file(paths.fold_membership))
    assert {row["source_uid"] for row in rows} == {uid for uid, row in membership.items() if row["fold"] == "1"}
    metrics = prediction_metrics(rows, native_labels=("Mn", "Cu", "Zn", "Fe", "Co", "Ni"))
    assert metrics["common4"]["balanced_accuracy"] == pytest.approx(
        receipt["metrics"]["val_metal_collapsed4_balanced_acc"], abs=1e-9)
    for row in rows:
        p6 = [float(row[f"p_native_{label}"]) for label in ("Fe", "Co", "Ni")]
        assert float(row["p_common4_Class_VIII"]) == pytest.approx(sum(p6), abs=1e-6)
    history = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["history"]
    assert any(key.startswith("readout_bias.") for key in history[-1])
    # Training-set metrics are skipped on epoch 1 and computed on the last epoch.
    assert history[0].get("train_metrics_skipped") is True and history[1].get("train_metal_acc") is not None
    assert receipt["reconciliation_status"] == "match"
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["test_report"] is None and metadata["fit_status"] == "completed"
    checkpoint = torch.load(run_dir / "best_model_checkpoint.pt", map_location="cpu", weights_only=False)
    saved = checkpoint["config"]
    replay = build_pocket_classifier(
        model_architecture=saved["model_architecture"], esm_dim=saved["esm_dim"], hidden_s=saved["hidden_s"],
        hidden_v=saved["hidden_v"], edge_hidden=saved["edge_hidden"], n_layers=saved["gvp_layers"],
        n_metal=len(checkpoint["metal_labels"]), n_ec=1, esm_fusion_dim=saved["esm_fusion_dim"],
        predict_metal=True, predict_ec=False, fusion_mode=saved["fusion_mode"],
        binding_residue_pooling=saved["binding_residue_pooling"],
        node_rbf_use_raw_distances=saved["node_rbf_use_raw_distances"],
        edge_rbf_use_raw_distances=saved["edge_rbf_use_raw_distances"],
        metal_class_weights=checkpoint["model_state_dict"]["metal_class_weights"],
    )
    replay.load_state_dict(checkpoint["model_state_dict"], strict=True)


def _synthetic_entry(config_id: str, per_fold_correct: list[list[int]]):
    """Five folds; each fold has 4 classes x 10 ions; ``per_fold_correct[k][c]`` ions of class c correct."""
    family, target, readout = config_id.split("__")
    folds = {}
    for fold, correct in enumerate(per_fold_correct):
        rows = []
        for label in range(4):
            for index in range(10):
                predicted = label if index < correct[label] else (label + 1) % 4
                rows.append({"source_uid": f"u{fold}_{label}_{index}", "group_id": f"g{fold}_{index}",
                             "fold": str(fold), "y_common4": str(label), "pred_common4": str(predicted),
                             "y_native": str(label), "pred_native": str(predicted)})
        folds[fold] = {"rows": rows, "metrics": prediction_metrics(rows, native_labels=("Mn", "Cu", "Zn", "Class VIII"))}
    return {"config": campaign.GridConfig(family, target, readout), "folds": folds, "missing": [], "complete": True}


def test_promotion_requires_ci_and_rare_recall_protection():
    from benchmarking.pmm_ion_analysis import evaluate_contrast

    control = _synthetic_entry("only_gvp__four_class__none", [[8, 6, 7, 6]] * 5)
    better = _synthetic_entry("only_gvp__four_class__first_shell_bias", [[9, 7, 8, 7]] * 5)
    promoted = evaluate_contrast("aware", control, better)
    assert promoted["promoted"] and promoted["ci95_low"] > 0
    # Higher mean BA but Cu recall falls by 0.10 (> 0.03): not promoted.
    trade = _synthetic_entry("only_gvp__four_class__first_shell_bias", [[10, 5, 10, 8]] * 5)
    blocked = evaluate_contrast("aware", control, trade)
    assert blocked["mean_difference"] > 0 and not blocked["promoted"]
    assert blocked["gates"]["no_class_recall_drop_over_0.03"] is False
    incomplete = dict(better, complete=False, missing=["x"])
    assert evaluate_contrast("aware", control, incomplete)["status"] == "incomplete"


def test_plan_records_resolved_config_without_fitting(tmp_path, monkeypatch):
    train, paths, _ = _campaign_with_folds(tmp_path, monkeypatch)
    _full_inventory(paths, tmp_path)
    campaign.campaign_manifest_guard(paths)
    config = campaign.GridConfig("only_gvp", "four_class", "first_shell_bias")
    command, env, identity = campaign.build_train_command(
        paths, python_bin=sys.executable, train_dir=train, config=config, fold=0, seed=42, device="cpu",
        runs_dir=paths.runs)
    name = campaign.run_name(config, 0, 42)
    result = campaign.execute_unit(paths, command, env, identity, paths.runs, name, dry_run=True)
    assert result["status"] == "planned" and not (paths.runs / name).exists()
    record = json.loads((paths.commands / f"{name}.json").read_text(encoding="utf-8"))
    resolved = record["resolved_config"]
    assert resolved["binding_residue_pooling"] == "first_shell_bias"
    assert resolved["metal_label_scheme"] == "merge_fe_class_viii" and resolved["run_test_eval"] is False
    assert resolved["require_esm_embeddings"] is False and resolved["use_ring_edges"] is False
    assert resolved["omit_node_features"] == list(campaign.OMITTED_EXTERNAL_FEATURES)


def test_transfer_bundle_keeps_hard_links_and_refuses_test_paths(tmp_path, monkeypatch):
    import os
    import tarfile

    from benchmarking import pmm_campaign_bundle as bundle

    train, paths, _ = _campaign_with_folds(tmp_path, monkeypatch, n_pdb=30)
    structures = sorted((train / "structures").glob("*.pdb"))
    alias = train / "structures" / structures[0].name.replace("__chain_A__", "__chain_Z__")
    os.link(structures[0], alias)
    (paths.root / "runs").mkdir(exist_ok=True)
    (paths.root / "runs" / "big.pt").write_text("x", encoding="utf-8")
    side = bundle.build_train_side(train, tmp_path / "train_side.tar.gz")
    frozen = bundle.build_campaign(paths.root, tmp_path / "campaign.tar.gz")
    with tarfile.open(tmp_path / "train_side.tar.gz") as tar:
        members = {member.name: member for member in tar.getmembers()}
    assert any(member.islnk() for member in members.values())
    assert all("test" not in Path(name).parts for name in members)
    with tarfile.open(tmp_path / "campaign.tar.gz") as tar:
        names = tar.getnames()
    assert any(name.endswith("train_cohort.csv") for name in names) and not any("/runs/" in name for name in names)
    assert side["members"] >= len(structures) and frozen["members"] > 0
    with pytest.raises(bundle.BundleError):
        bundle._check_member("dataset/test/structures/x.pdb")
    with pytest.raises(bundle.BundleError):
        bundle._check_member("dataset/site_crosswalk.csv")
