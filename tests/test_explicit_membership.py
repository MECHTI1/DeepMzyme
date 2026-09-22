"""Diagnostic membership and seal safeguards, using synthetic data only."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_structures import PocketRecord
from diagnostic_metal_pilot import rebalance, verify_outer_lock, write_rows, write_json
from training.config import TrainConfig, parse_args
from training.explicit_membership import (
    PARTITIONS, ENDPOINT, counts, fixed_split, load_membership, select_exact,
    read_rows, sha256, validate_mode, validate_rows,
)
from training.splits import PocketSplit, split_pockets


@pytest.fixture
def cohort():
    partitions = {}
    index = 0
    for part, n in zip(PARTITIONS, (5, 3, 5)):
        rows = []
        for target in range(4):
            for _ in range(n):
                group = f"{index:04d}"
                sid = f"{group}__chain_A__EC_1.1.1.1"
                rows.append(dict(example_id=f"{sid}_METAL_0", structure_id=sid, group_id=group,
                                 protein_id=group, component_id=f"c{index}", target=target))
                index += 1
        partitions[part] = rows
    return partitions


def development(parts):
    return [r for p in PARTITIONS for r in parts[p]]


def pockets(rows):
    return [PocketRecord(structure_id=r["structure_id"], pocket_id=r["example_id"],
                         metal_element=("MN", "CU", "ZN", "FE", "CO")[r["target"]],
                         metal_coords=[torch.zeros(3)], residues=[], y_metal=r["target"], y_ec=0) for r in rows]


def test_exact_membership_and_shared_systems(cohort):
    membership = {"partitions": cohort}
    ps = pockets(cohort["inner_validation"] + cohort["train"])
    results = [fixed_split(ps, membership) for _ in ("only_esm", "only_gvp")]
    for split in results:
        assert [p.pocket_id for p in split.train_pockets] == [r["example_id"] for r in cohort["train"]]
        assert [p.pocket_id for p in split.val_pockets] == [r["example_id"] for r in cohort["inner_validation"]]
    assert results[0] == results[1]
    with pytest.raises(ValueError, match="membership changed"):
        fixed_split(ps + pockets(cohort["outer_evaluation"][:1]), membership)


@pytest.mark.parametrize("kind", ["duplicate", "overlap", "group_id", "protein_id", "component_id", "unknown", "target", "protected"])
def test_bad_membership_rejected(cohort, kind):
    allowed = copy.deepcopy(development(cohort))
    train, val = cohort["train"], cohort["inner_validation"]
    if kind == "duplicate":
        train.append(train[0])
    elif kind == "overlap":
        val.append(train[0])
    elif kind in ("group_id", "protein_id", "component_id"):
        val[0][kind] = train[0][kind]
        next(r for r in allowed if r["example_id"] == val[0]["example_id"])[kind] = val[0][kind]
    elif kind == "unknown":
        train[0]["example_id"] = "unknown"
    elif kind == "target":
        train[0]["target"] = 3
    else:
        train[0]["protected_test"] = True
    with pytest.raises(ValueError):
        validate_rows(cohort, allowed, [])


def test_detected_edge_rejected(cohort):
    with pytest.raises(ValueError, match="edge crosses"):
        validate_rows(cohort, development(cohort), [{"query_group": cohort["train"][0]["group_id"],
                                                    "target_group": cohort["outer_evaluation"][0]["group_id"]}])


def test_protected_development_allowlist_rejected(cohort):
    allowed = copy.deepcopy(development(cohort))
    allowed[0]["protected_test"] = True
    with pytest.raises(ValueError, match="Protected-test membership"):
        validate_rows(cohort, allowed, [])


def test_edge_sequence_target_is_not_class_label(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text("query,target,query_group,target_group\ns_1,s_2,p_1,p_2\n")
    assert read_rows(path)[0]["target"] == "s_2"


@pytest.mark.parametrize("part", PARTITIONS)
def test_missing_class_rejected(cohort, part):
    cohort[part] = [r for r in cohort[part] if r["target"] != 1]
    with pytest.raises(ValueError, match="Missing active class"):
        validate_rows(cohort, development(cohort), [])


@pytest.mark.parametrize("part", ["inner_validation", "outer_evaluation"])
def test_insufficient_component_support(cohort, part):
    cohort[part].pop()
    with pytest.raises(ValueError, match="class-component support"):
        validate_rows(cohort, development(cohort), [])


def test_loaded_target_unknown_and_group_rejected(cohort):
    rows = cohort["train"]
    ps = pockets(rows)
    ps[0].y_metal = 3
    with pytest.raises(ValueError, match="target"):
        select_exact(ps, rows)
    with pytest.raises(ValueError, match="Unknown"):
        select_exact([], rows)


@pytest.fixture
def bound_config(tmp_path, cohort):
    manifests = {}
    for part, rows in cohort.items():
        p = tmp_path / f"{part}.csv"
        write_rows(p, rows)
        manifests[part] = {"path": p.name, "sha256": sha256(p)}
    write_rows(tmp_path / "allow.csv", development(cohort))
    (tmp_path / "edges.csv").write_text("query_group,target_group\n")
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "summary.csv").write_text("development only")
    (tmp_path / "structure_manifest.csv").write_text("development only")
    ref = lambda name: {"path": name, "sha256": sha256(tmp_path / name)}
    spec = dict(endpoint=ENDPOINT, split_seed=20260917, label_scheme="merge_fe_class_viii",
                manifests=manifests, development_allowlist=ref("allow.csv"), detected_edges=ref("edges.csv"),
                search_protocol=ref("protocol.json"), counts={p: counts(r) for p, r in cohort.items()}, source_control={},
                dataset=dict(scope="external_training_only", bundle_id="synthetic", bundle_sha256="a" * 64,
                             summary_sha256=sha256(tmp_path / "summary.csv"),
                             structure_manifest_sha256=sha256(tmp_path / "structure_manifest.csv")))
    write_json(tmp_path / "split.json", spec)
    return TrainConfig(structure_dir=tmp_path, summary_csv=tmp_path / "summary.csv", task="metal",
                       model_architecture="only_gvp", metal_label_scheme="four_class", split_seed=20260917,
                       selection_metric="val_metal_balanced_acc", dataset_bundle_id="synthetic", dataset_bundle_sha256="a" * 64,
                       prepare_missing_esm_embeddings=False, prepare_missing_ring_edges=False,
                       explicit_membership_manifest=str(tmp_path / "split.json"), explicit_membership_sha256=sha256(tmp_path / "split.json"))


def test_descriptor_and_partition_checksums(bound_config):
    assert load_membership(bound_config, verify_features=False)["receipt"]["enabled"]
    with pytest.raises(ValueError, match="checksum"):
        load_membership(replace(bound_config, explicit_membership_sha256="0" * 64), verify_features=False)
    path = bound_config.structure_dir / "train.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_membership(bound_config, verify_features=False)


def test_manifest_label_scheme_mismatch(bound_config):
    path = Path(bound_config.explicit_membership_manifest)
    spec = json.loads(path.read_text())
    spec["label_scheme"] = "split_all_metals"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="label-scheme"):
        load_membership(replace(bound_config, explicit_membership_sha256=sha256(path)), verify_features=False)


def test_dataset_identity_mismatch(bound_config):
    with pytest.raises(ValueError, match="Dataset identity"):
        load_membership(replace(bound_config, dataset_bundle_id="protected_test"), verify_features=False)


def test_missing_cache_and_changed_feature_fail_before_loading(bound_config, cohort, monkeypatch):
    from training.explicit_membership import verify_feature_files
    base = bound_config.structure_dir
    rows = development(cohort)
    paths = {r["structure_id"]: base / f"{r['structure_id']}.pdb" for r in rows}
    for p in paths.values():
        p.write_text("structure")
    monkeypatch.setattr("training.structure_loading.find_structure_files", lambda p: list(paths.values()))
    inventory = [{"structure_id": r["structure_id"], "structure_sha256": sha256(paths[r["structure_id"]]),
                  "external_sha256": "0" * 64, "esm_sha256": "0" * 64, "esm_file": "missing.pt"} for r in rows]
    write_rows(base / "features.csv", inventory)
    spec = {"feature_inventory": {"path": "features.csv", "sha256": sha256(base / "features.csv")}}
    config = replace(bound_config, external_features_root_dir=str(base), require_esm_embeddings=False)
    with pytest.raises(ValueError, match="Missing required"):
        verify_feature_files(config, spec, base, rows)
    paths[rows[0]["structure_id"]].write_text("changed structure")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_feature_files(config, spec, base, rows)


@pytest.mark.parametrize("change", [dict(run_test_eval=True), dict(test_structure_dir=Path("test")),
    dict(metal_label_scheme="six_class"), dict(prepare_missing_esm_embeddings=True),
    dict(selection_metric="train_loss"), dict(n_folds=5), dict(explicit_membership_sha256=None)])
def test_unsafe_mode_rejected(bound_config, change):
    with pytest.raises(ValueError):
        validate_mode(replace(bound_config, **change))


def prepare_stubs(monkeypatch, tmp_path, cohort, explicit):
    import training.run as run
    from training.data import PocketLoadResult
    cfg = TrainConfig(task="metal", metal_label_scheme="four_class", val_fraction=.15, split_seed=20260917,
                      joint_loss_weighting="fixed",
                      model_architecture="only_gvp", selection_metric="val_metal_balanced_acc",
                      prepare_missing_esm_embeddings=False, prepare_missing_ring_edges=False,
                      explicit_membership_manifest="fake" if explicit else None,
                      explicit_membership_sha256="a" * 64 if explicit else None)
    membership = {"partitions": cohort, "receipt": {"endpoint": ENDPOINT}}
    monkeypatch.setattr("training.explicit_membership.load_membership", lambda c: membership if explicit else None)
    rows = cohort["train"] + cohort["inner_validation"]
    all_pockets = pockets(rows)
    seen = {}
    def load(**kwargs):
        seen["allowed"] = kwargs.get("allowed_structure_ids")
        return PocketLoadResult(all_pockets, {}, {"1": 0}, {0: "1"})
    monkeypatch.setattr(run, "load_training_pockets_with_report_from_dir", load)
    monkeypatch.setattr(run, "build_run_dir", lambda c: tmp_path)
    monkeypatch.setattr(run, "prepare_runtime_inputs", lambda **kw: {})
    monkeypatch.setattr(run, "build_split_diagnostics", lambda *a, **kw: {})
    monkeypatch.setattr(run, "format_split_diagnostics", lambda x: "synthetic")
    monkeypatch.setattr(run, "build_dataset_summary", lambda *a, **kw: {})
    monkeypatch.setattr(run, "build_graph_data_list", lambda ps, **kw: ps)
    monkeypatch.setattr(run, "run_preflight_checks", lambda *a, **kw: {})
    def stats(ps, **kw):
        seen["statistics_ids"] = [p.pocket_id for p in ps]
        return SimpleNamespace(means={}, stds={}, clamp_value=5.)
    monkeypatch.setattr(run, "compute_feature_normalization_stats", stats)
    monkeypatch.setattr(run, "PocketGraphDataset", lambda ps, **kw: ps)
    monkeypatch.setattr(run, "DataLoader", lambda ds, **kw: ds)
    weights = run.balanced_class_weights_from_pockets
    def record_weights(ps, **kw):
        seen["class_weight_ids"] = [p.pocket_id for p in ps]
        return weights(ps, **kw)
    monkeypatch.setattr(run, "balanced_class_weights_from_pockets", record_weights)
    monkeypatch.setattr(run, "build_pocket_classifier", lambda **kw: torch.nn.Linear(1, 1))
    return run, cfg, all_pockets, seen


def test_default_auto_split_unchanged(monkeypatch, tmp_path, cohort):
    run, cfg, ps, seen = prepare_stubs(monkeypatch, tmp_path, cohort, False)
    expected = split_pockets(ps, val_fraction=.15, split_by="pdbid", seed=20260917, task="metal", stratify_by="active_targets")
    prepared = run.prepare_run(cfg)
    assert seen["allowed"] is None
    assert prepared.split == expected
    assert "explicit_membership" not in prepared.config_payload


def test_training_statistics_and_loaders_exclude_outer(monkeypatch, tmp_path, cohort):
    run, cfg, _, seen = prepare_stubs(monkeypatch, tmp_path, cohort, True)
    def forbidden(*args, **kwargs):
        pytest.fail("Automatic splitter or general runtime feature preparation reached in explicit mode")
    monkeypatch.setattr(run, "split_pockets", forbidden)
    monkeypatch.setattr(run, "split_pockets_k_fold", forbidden)
    monkeypatch.setattr(run, "prepare_runtime_inputs", forbidden)
    prepared = run.prepare_run(cfg)
    train_ids = [r["example_id"] for r in cohort["train"]]
    assert seen["statistics_ids"] == seen["class_weight_ids"] == train_ids
    assert seen["allowed"] == {r["structure_id"] for p in ("train", "inner_validation") for r in cohort[p]}
    assert [p.pocket_id for p in prepared.val_loader] == [r["example_id"] for r in cohort["inner_validation"]]
    assert not hasattr(prepared, "outer_loader")


def test_checkpoint_selection_uses_inner_validation(monkeypatch, tmp_path, cohort):
    run, cfg, _, _ = prepare_stubs(monkeypatch, tmp_path, cohort, True)
    prepared = run.prepare_run(cfg)
    cfg = replace(cfg, epochs=3)
    calls = []
    values = iter([.4, .8, .5])
    monkeypatch.setattr(run, "train_epoch", lambda *a, **kw: .1)
    def evaluate(model, loader, device, *, prefix, **kw):
        calls.append((prefix, loader))
        return {"val_metal_balanced_acc": next(values)} if prefix == "val" else {}
    monkeypatch.setattr(run, "evaluate_split_metrics", evaluate)
    monkeypatch.setattr(run, "task_loss_weighting_state", lambda m: {})
    monkeypatch.setattr(run, "checkpoint_payload", lambda **kw: kw)
    monkeypatch.setattr(run, "format_epoch_log", lambda *a, **kw: "epoch")
    history, best = run.train_and_select_checkpoint(prepared, cfg)
    assert len(history) == 3 and best["epoch"] == 2
    assert all(loader is prepared.val_loader for prefix, loader in calls if prefix == "val")


def test_outer_loader_not_deserialized(monkeypatch, tmp_path):
    import training.data as data
    paths = [tmp_path / f"{n}.pdb" for n in ("train", "inner", "outer")]
    monkeypatch.setattr(data, "find_structure_files", lambda p: paths)
    monkeypatch.setattr(data, "resolve_allowed_site_metal_labels", lambda p: {})
    monkeypatch.setattr(data, "resolve_runtime_feature_paths", lambda **kw: (tmp_path, tmp_path))
    seen = []
    def load(**kwargs):
        name = kwargs["structure_path"].stem
        assert name != "outer"
        seen.append(name)
        return [], [], []
    monkeypatch.setattr(data, "load_structure_pockets", load)
    with pytest.raises(ValueError, match="No metal-centered"):
        data.load_training_pockets_with_report_from_dir(tmp_path, allowed_structure_ids={"train", "inner"})
    assert seen == ["train", "inner"]


def test_outer_command_locked_until_both_checkpoints(tmp_path):
    with pytest.raises(ValueError, match="locked"):
        verify_outer_lock(tmp_path, "0" * 64)
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "pilot_plan.json").write_text("{}")
    payload = dict(status="both_required_fits_completed", outer_observed_during_training=False,
                   pilot_plan_sha256=sha256(tmp_path / "pilot_plan.json"),
                   split={"path": "plan.json", "sha256": sha256(tmp_path / "plan.json")},
                   models={f: {"files": {"best_model_checkpoint.pt": "0" * 64}} for f in ("only_esm", "only_gvp")})
    write_json(tmp_path / "frozen_pilot.json", payload)
    with pytest.raises(ValueError, match="locked"):
        verify_outer_lock(tmp_path, sha256(tmp_path / "frozen_pilot.json"))


def test_cli_and_absent_manifest_leave_defaults_disabled():
    cfg = parse_args(["--task", "metal", "--metal-label-scheme", "four_class"])
    assert cfg.explicit_membership_manifest is None and not cfg.run_test_eval
    assert load_membership(cfg) is None


def test_rebalance_is_deterministic_and_preserves_training(cohort):
    # Move two Cu components from inner to outer to recreate the weak old split.
    moved = [r for r in cohort["inner_validation"] if r["target"] == 1][:2]
    cohort["inner_validation"] = [r for r in cohort["inner_validation"] if r not in moved]
    cohort["outer_evaluation"] += moved
    original = [dict(r, partition=p) for p in PARTITIONS for r in cohort[p]]
    first, receipt = rebalance(original, [])
    assert first == rebalance(original, [])[0]
    assert first["train"] == cohort["train"]
    assert receipt["moved_pockets"] == 2 and receipt["outcomes_used"] is False
    assert counts(first["inner_validation"])["classes"]["1"]["components"] == 3


def make_completed_pilot(tmp_path, cohort):
    """Minimal actual serialized run/checkpoint contract, with synthetic history."""
    from diagnostic_metal_pilot import FAMILIES, reference
    manifests = {}
    for part, rows in cohort.items():
        write_rows(tmp_path / f"{part}.csv", rows)
        manifests[part] = reference(tmp_path / f"{part}.csv", tmp_path)
    write_json(tmp_path / "split.json", {"manifests": manifests})
    split_hash = sha256(tmp_path / "split.json")
    models = {}
    for family in FAMILIES:
        directory = tmp_path / "runs" / family
        directory.mkdir(parents=True)
        cfg = dict(model_architecture=family, seed=42, epochs=50, explicit_membership_sha256=split_hash,
                   run_test_eval=False)
        write_json(tmp_path / f"{family}.json", cfg)
        models[family] = {"configuration": reference(tmp_path / f"{family}.json", tmp_path)}
        dataset = {"retained_split_identity": {
            p: {"examples": [dict(structure_id=r["structure_id"], pocket_id=r["example_id"],
                                  group=r["group_id"], y_metal=r["target"])
                                for r in cohort["inner_validation" if p == "validation" else p]]}
            for p in ("train", "validation")}}
        payload = dict(config=cfg, normalization_stats={}, dataset_summary=dataset,
                       selected_checkpoint_epoch=7, test_report=None,
                       history=[dict(epoch=i, val_metal_balanced_acc=.8 if i == 7 else .4) for i in range(1, 51)])
        metadata = dict(payload, fit_status="completed", outer_observed_during_training=False,
                        selected_checkpoint="best_model_checkpoint.pt")
        write_json(directory / "run_config.json", payload)
        write_json(directory / "run_metadata.json", metadata)
        write_json(directory / "dataset_summary.json", dataset)
        torch.save(dict(config=cfg, normalization_stats={}, dataset_summary=dataset, epoch=7,
                        selection_metric="val_metal_balanced_acc", selection_metric_value=.8),
                   directory / "best_model_checkpoint.pt")
    write_json(tmp_path / "pilot_plan.json", {"split": reference(tmp_path / "split.json", tmp_path), "models": models})


def test_two_completed_checkpoints_unlock_once_and_detect_tampering(tmp_path, cohort):
    from diagnostic_metal_pilot import freeze_checkpoints
    make_completed_pilot(tmp_path, cohort)
    checkpoint = tmp_path / "runs/only_gvp/best_model_checkpoint.pt"
    saved = checkpoint.read_bytes()
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        freeze_checkpoints(tmp_path)
    assert not (tmp_path / "frozen_pilot.json").exists()
    checkpoint.write_bytes(saved)
    frozen = freeze_checkpoints(tmp_path)
    assert frozen["models"]["only_esm"]["selected_epoch"] == 7
    assert verify_outer_lock(tmp_path, sha256(tmp_path / "frozen_pilot.json")) == frozen
    with pytest.raises(FileExistsError):
        freeze_checkpoints(tmp_path)
    checkpoint.write_bytes(saved + b"changed")
    with pytest.raises(ValueError, match="locked"):
        verify_outer_lock(tmp_path, sha256(tmp_path / "frozen_pilot.json"))


def test_agreeing_models_cannot_hide_changed_manifest_membership(tmp_path, cohort):
    from diagnostic_metal_pilot import freeze_checkpoints
    make_completed_pilot(tmp_path, cohort)
    for family in ("only_esm", "only_gvp"):
        directory = tmp_path / "runs" / family
        for name in ("run_config.json", "run_metadata.json", "dataset_summary.json"):
            path = directory / name
            payload = json.loads(path.read_text())
            dataset = payload if name == "dataset_summary.json" else payload["dataset_summary"]
            dataset["retained_split_identity"]["train"]["examples"].pop()
            path.write_text(json.dumps(payload))
        path = directory / "best_model_checkpoint.pt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["dataset_summary"]["retained_split_identity"]["train"]["examples"].pop()
        torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="differs from frozen manifests"):
        freeze_checkpoints(tmp_path)


@pytest.mark.parametrize("n_classes", [4, 5])
def test_outer_evaluator_uses_saved_normalization_and_writes_identifiers(tmp_path, cohort, monkeypatch, n_classes):
    import diagnostic_metal_pilot as pilot
    from label_schemes import configure_active_metal_label_scheme
    scheme = "four_class" if n_classes == 4 else "five_class"
    configure_active_metal_label_scheme(scheme)
    rows = [cohort["outer_evaluation"][i] for i in (0, 5, 10, 15)]
    if n_classes == 5:
        sid = "9000__chain_A__EC_1.1.1.1"
        rows.append(dict(rows[-1], example_id=f"{sid}_METAL_0", structure_id=sid, group_id="9000", target=4))
    cfg = TrainConfig(task="metal", metal_label_scheme=scheme, require_esm_embeddings=False)
    monkeypatch.setattr(pilot, "verify_outer_lock", lambda *a: {})
    monkeypatch.setattr(pilot, "load_config", lambda *a: cfg)
    monkeypatch.setattr(pilot, "load_membership", lambda c: {"partitions": {"outer_evaluation": rows}})
    monkeypatch.setattr("training.data.load_training_pockets_with_report_from_dir", lambda **kw: SimpleNamespace(pockets=pockets(rows)))
    monkeypatch.setattr("training.graph_dataset.build_graph_data_list", lambda ps, **kw: ps)
    norms = []
    def dataset(ps, **kw):
        norms.append(kw["normalization_stats"])
        return ps
    monkeypatch.setattr("training.graph_dataset.PocketGraphDataset", dataset)
    monkeypatch.setattr("torch_geometric.loader.DataLoader", lambda ds, **kw: ds)
    model = SimpleNamespace(load_state_dict=lambda *a, **kw: None)
    model.to = lambda d: model
    monkeypatch.setattr("model_variants.build_pocket_classifier", lambda **kw: model)
    monkeypatch.setattr("export_validation_predictions.factory_kwargs", lambda *a: {})
    predictions = dict(loss=0., metal_y=torch.arange(n_classes), metal_logits=torch.eye(n_classes))
    monkeypatch.setattr("training.loop.evaluate_epoch_with_predictions", lambda *a, **kw: predictions)
    for family in pilot.FAMILIES:
        directory = tmp_path / "runs" / family
        directory.mkdir(parents=True)
        torch.save(dict(normalization_stats={"means": {"x": [2.]}, "stds": {"x": [3.]}},
                        ec_labels={}, config={}, model_state_dict={}), directory / "best_model_checkpoint.pt")
    result = pilot.evaluate_outer(tmp_path, Path.cwd(), "synthetic", "cpu")
    assert len(norms) == 2 and all(n.means["x"].item() == 2. for n in norms)
    assert all(m["outer_metal_balanced_acc"] == 1. for m in result.values())
    exported = read_rows(tmp_path / "outer_evaluation/only_esm_predictions.csv")
    assert [r["component_id"] for r in exported] == [r["component_id"] for r in rows]
    with pytest.raises(FileExistsError):
        pilot.evaluate_outer(tmp_path, Path.cwd(), "synthetic", "cpu")
    configure_active_metal_label_scheme("four_class")


@pytest.fixture
def five_cohort(cohort):
    for pi, part in enumerate(PARTITIONS):
        for i, original in enumerate([r for r in cohort[part] if r["target"] == 3]):
            group = str(9000 + pi * 100 + i)
            sid = f"{group}__chain_A__EC_1.1.1.1"
            cohort[part].append(dict(original, target=4, example_id=f"{sid}_METAL_0", structure_id=sid,
                                     group_id=group, protein_id=group, component_id=f"c{group}"))
    return cohort


def five_sources(parts):
    examples = [dict(pocket_id=r["example_id"], structure_id=r["structure_id"], group=r["group_id"],
                     y_metal=r["target"]) for r in development(parts)]
    return {family: dict(config=dict(task="metal", metal_label_scheme="five_class", model_architecture=family,
                                    seed=42, run_test_eval=False), test_report=None,
                         dataset_summary={"retained_split_identity": {
                             "train": {"examples": copy.deepcopy(examples)}, "validation": {"examples": []}}})
            for family in ("only_esm", "only_gvp")}


def test_five_class_support_and_exact_membership(five_cohort):
    support = validate_rows(five_cohort, development(five_cohort), [], label_scheme="five_class")
    assert support["inner_validation"]["classes"]["4"]["components"] == 3
    assert support["outer_evaluation"]["classes"]["4"]["components"] == 5
    with pytest.raises(ValueError, match="Target outside"):
        validate_rows(five_cohort, development(five_cohort), [])
    fitting = pockets(five_cohort["train"] + five_cohort["inner_validation"])
    assert len(fixed_split(fitting, {"partitions": five_cohort}).train_pockets) == 25


@pytest.mark.parametrize("part", PARTITIONS)
def test_five_class_requires_support_for_co_ni(five_cohort, part):
    five_cohort[part].pop()
    with pytest.raises(ValueError, match="class-component support"):
        validate_rows(five_cohort, development(five_cohort), [], label_scheme="five_class")


@pytest.mark.parametrize("change", [None, "disagreement", "wrong_collapse", "missing_id", "duplicate", "protected", "wrong_group"])
def test_five_class_relabel_provenance(five_cohort, change):
    from training.explicit_membership import relabel_five_class_memberships
    parent = {p: [dict(r, target=min(r["target"], 3)) for r in rows] for p, rows in five_cohort.items()}
    sources = five_sources(five_cohort)
    examples = sources["only_gvp"]["dataset_summary"]["retained_split_identity"]["train"]["examples"]
    if change == "disagreement":
        examples[-1]["y_metal"] = 3
    elif change == "wrong_collapse":
        examples[0]["y_metal"] = 4
    elif change == "missing_id":
        examples.pop()
    elif change == "duplicate":
        examples.append(examples[0])
    elif change == "protected":
        sources["only_gvp"]["config"]["run_test_eval"] = True
    elif change == "wrong_group":
        examples[0]["group"] = "unknown"
    if change is None:
        assert relabel_five_class_memberships(parent, sources) == five_cohort
    else:
        with pytest.raises(ValueError):
            relabel_five_class_memberships(parent, sources)


def test_five_class_config_cannot_reuse_four_class_manifest(bound_config):
    validate_mode(replace(bound_config, metal_label_scheme="five_class"))
    with pytest.raises(ValueError, match="label-scheme mismatch"):
        load_membership(replace(bound_config, metal_label_scheme="five_class"), verify_features=False)


def test_five_class_bound_provenance_detects_relabeling_and_tampering(tmp_path, five_cohort):
    from diagnostic_metal_pilot import reference
    from training.explicit_membership import verify_five_class_provenance
    parent_parts = {p: [dict(r, target=min(r["target"], 3)) for r in rows] for p, rows in five_cohort.items()}
    for part, rows in parent_parts.items():
        write_rows(tmp_path / f"parent_{part}.csv", rows)
    parent = {"label_scheme": "merge_fe_class_viii", "dataset": {},
              "manifests": {p: reference(tmp_path / f"parent_{p}.csv", tmp_path) for p in PARTITIONS}}
    for key in ("detected_edges", "search_protocol", "feature_inventory"):
        parent[key] = {"sha256": "unchanged"}
    write_json(tmp_path / "parent.json", parent)
    sources = five_sources(five_cohort)
    for family, payload in sources.items():
        write_json(tmp_path / f"{family}.json", payload)
    spec = dict(parent, label_scheme="five_class", parent_split=reference(tmp_path / "parent.json", tmp_path),
                label_sources={f: reference(tmp_path / f"{f}.json", tmp_path) for f in sources})
    verify_five_class_provenance(tmp_path, spec, five_cohort)
    wrong = copy.deepcopy(five_cohort)
    wrong["train"][-1]["target"] = 3
    with pytest.raises(ValueError, match="changed parent memberships or native labels"):
        verify_five_class_provenance(tmp_path, spec, wrong)
    path = tmp_path / "only_esm.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_five_class_provenance(tmp_path, spec, five_cohort)


def test_five_class_statistics_use_exact_training_membership(monkeypatch, tmp_path, five_cohort):
    run, config, _, seen = prepare_stubs(monkeypatch, tmp_path, five_cohort, explicit=True)
    prepared = run.prepare_run(replace(config, metal_label_scheme="five_class"))
    wanted = [r["example_id"] for r in five_cohort["train"]]
    assert seen["statistics_ids"] == wanted
    assert seen["class_weight_ids"] == wanted
    assert not hasattr(prepared, "outer_loader")


def test_five_class_collapse_sums_probabilities_before_argmax():
    from label_schemes import configure_active_metal_label_scheme
    from training.run import metrics_from_predictions
    configure_active_metal_label_scheme("five_class")
    try:
        prediction = dict(loss=0., metal_y=torch.tensor([4]),
                          metal_logits=torch.log(torch.tensor([[.4, .03, .02, .3, .25]])))
        result = metrics_from_predictions(prediction, "val", task="metal", ec_label_map={}, ec_label_depth=1)
        assert result["val_metal_acc"] == 0.
        assert result["val_metal_collapsed4_acc"] == 1.
    finally:
        configure_active_metal_label_scheme("four_class")


def test_local_only_device_and_admission_are_enforced(tmp_path):
    from diagnostic_metal_pilot import validate_execution_device, verify_local_admission
    plan = {"execution": {"local_only": True}}
    validate_execution_device(plan, "cpu", fitting=True)
    with pytest.raises(ValueError, match="local CPU"):
        validate_execution_device(plan, "cuda", fitting=True)
    with pytest.raises(ValueError, match="GPU"):
        validate_execution_device({}, "cpu", fitting=True)
    write_json(tmp_path / "pilot_plan.json", plan)
    with pytest.raises(ValueError, match="CPU admission"):
        verify_local_admission(tmp_path, tmp_path)
    (tmp_path / "source.py").write_text("tested source")
    gate = {"pilot_plan_sha256": sha256(tmp_path / "pilot_plan.json"),
            "gates": [{"status": "PASS"}], "source_files": {"source.py": sha256(tmp_path / "source.py")}}
    write_json(tmp_path / "cpu_admission_gate.json", gate)
    assert verify_local_admission(tmp_path, tmp_path) == gate
    (tmp_path / "source.py").write_text("changed source")
    with pytest.raises(ValueError, match="Tested source changed"):
        verify_local_admission(tmp_path, tmp_path)
