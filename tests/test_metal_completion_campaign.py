"""CPU contracts only: no real data, model fitting, inference or held-out reads."""

from __future__ import annotations

import json
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import metal_completion_campaign as mc
import pmm_v3_campaign as v3
import run_metal_completion as worker
from test_v3_campaign import campaign  # noqa: F401; synthetic train-only fixture
from training.config import config_to_payload, parse_args


def test_exact_grids_and_unambiguous_names():
    discovery = mc.discovery_units()
    assert len(discovery) == len({unit.name for unit in discovery}) == 45
    for recipe, count in (("B", 30), ("P", 40), ("S", 40)):
        confirmation = mc.confirmation_units(recipe)
        assert len(confirmation) == len({unit.name for unit in confirmation}) == count
        assert {unit.seed for unit in confirmation} == {43, 44}
        assert not set(unit.name for unit in discovery) & {unit.name for unit in confirmation}
        for unit in confirmation:
            assert mc.Unit.parse(unit.name) == unit
            assert unit.recipe == recipe or (unit.recipe, unit.target) == ("B", "four_class")
    for unit in discovery:
        assert mc.Unit.parse(unit.name) == unit
        with pytest.raises(ValueError):
            v3.Unit.parse(unit.name)
    for args in (("discovery", "B", "four_class", 0, 43),
                 ("confirmation", "P", "six_class", 0, 42),
                 ("discovery", "B", "four_class", True, 42)):
        with pytest.raises(ValueError):
            mc.Unit(*args)


def test_all_targets_recipes_and_seed44_use_the_real_parser_without_changing_parent(campaign, tmp_path):
    train, esm, parent, _, _ = campaign
    snapshot = {path: mc.sha256_file(path) for path in parent.root.rglob("*") if path.is_file()}
    common = dict(successor_root=tmp_path / "successor", manifest_sha256="a" * 64,
                  code_sha256={"run_metal_completion.py": "b" * 64}, python_bin=sys.executable,
                  train_dir=train, esm_dir=esm, device="cpu", lane=0, amp=False)
    units = [mc.Unit(phase, recipe, target, 2, seed)
             for phase, seed in (("discovery", 42), ("confirmation", 44))
             for recipe in mc.RECIPES for target in mc.TARGETS]
    for unit in units:
        command, env, identity = mc.build_command(parent, unit, **common)
        config = config_to_payload(parse_args(command[3:]))
        mc.validate_recipe_config(unit.recipe, config)
        assert config["seed"] == identity["model_seed"] == unit.seed
        assert identity["campaign_id"] == mc.CAMPAIGN_ID and identity["recipe"] == unit.recipe
        assert identity["target_scheme"] == unit.target
        assert identity["resolved_config_sha256"] == mc.resolved_config_sha256(command[:-2])
        assert json.loads(command[-1]) == identity
        assert config["checkpoint_rule"] == "terminal" and config["epochs"] == 50
        assert config["learning_rate"] == 3e-5 and config["gvp_learning_rate"] == 3e-4
        assert Path(config["runs_dir"]).is_relative_to(tmp_path / "successor")
        assert Path(env["DEEPMZYME_PARSE_CACHE_DIR"]).is_relative_to(tmp_path / "successor")
        assert Path(env["DEEPMZYME_GRAPH_CACHE_DIR"]).is_relative_to(tmp_path / "successor")
        assert "--run-test-eval" not in command and config["run_test_eval"] is False
        weights = json.loads(parent.fold_class_weights.read_text())["folds"]["2"]["common_four_weights"]
        assert config["mn_loss_multiplier"] == weights["Mn"]
        assert config["cu_loss_multiplier"] == weights["Cu"]
        if unit.target == "five_class":
            assert config["fe_loss_multiplier"] == config["class_viii_loss_multiplier"] == weights["Class VIII"]
        if unit.target == "six_class":
            assert {config[f"{name}_loss_multiplier"] for name in ("fe", "co", "ni")} == {weights["Class VIII"]}
    assert {path: mc.sha256_file(path) for path in parent.root.rglob("*") if path.is_file()} == snapshot
    assert v3.SEEDS == (42, 43)
    assert not (tmp_path / "successor").exists()


def test_refuses_recipe_mislabeling_and_combined_focal_smoothing():
    config = {**mc.recipe_config("P"), "metal_class_weight_mode": "manual",
              "metal_collapsed_loss_weight": 0.0, "gvp_auxiliary_loss_weight": 0.0}
    mc.validate_recipe_config("P", config)
    for key, value in (("metal_label_smoothing", 0.1), ("metal_focal_gamma", 2.0),
                       ("metal_class_weight_mode", "inverse_frequency"),
                       ("gvp_auxiliary_loss_weight", 0.3)):
        with pytest.raises(ValueError):
            mc.validate_recipe_config("P", {**config, key: value})


@pytest.mark.parametrize("target,counts", [("four_class", [40, 5, 35, 20]),
                                           ("five_class", [40, 5, 35, 15, 5]),
                                           ("six_class", [40, 5, 35, 15, 3, 2])])
def test_population_weights_and_actual_model_loss_gradients(target, counts):
    from model import GVPPocketClassifier

    weights = torch.tensor([100 / (4 * count) for count in (40, 5, 35)] + [1.25] * (len(counts) - 3))
    masses = torch.tensor(counts) * weights / 100
    assert float(masses.sum()) == pytest.approx(1)
    assert masses[:3].tolist() == pytest.approx([0.25] * 3)
    assert float(masses[3:].sum()) == pytest.approx(0.25)
    for recipe in mc.RECIPES:
        torch.manual_seed(7)
        model = GVPPocketClassifier(esm_dim=4, hidden_s=8, hidden_v=2, edge_hidden=8, n_layers=1,
                                   esm_fusion_dim=8, n_metal=len(counts), n_ec=1, predict_ec=False,
                                   metal_class_weights=weights, **mc.recipe_config(recipe))
        # Includes a partial batch, a batch without Cu, and a missing-label row.
        for labels in ([0, 2, 3], [1], [0, 1, -1, len(counts) - 1]):
            y = torch.tensor(labels)
            logits = torch.arange(len(y) * len(counts), dtype=torch.float32).reshape(len(y), len(counts)) / 7
            logits.requires_grad_()
            actual, _ = model._compute_supervised_loss(torch.zeros(len(y), 8), logits, None,
                                                       SimpleNamespace(y_metal=y))
            valid = y != -1
            reference_logits = logits.detach().clone().requires_grad_()
            if recipe == "P":
                logp = F.log_softmax(reference_logits[valid], dim=-1)
                expected = -(logp.gather(1, y[valid, None]).squeeze(1) * weights[y[valid]]).mean()
            else:
                expected = F.cross_entropy(reference_logits[valid], y[valid], weight=weights,
                                           label_smoothing=0.1 if recipe == "S" else 0.0)
            assert float(actual.detach()) == pytest.approx(float(expected.detach()), rel=1e-6)
            assert torch.allclose(torch.autograd.grad(actual, logits)[0],
                                  torch.autograd.grad(expected, reference_logits)[0], atol=1e-7)


def test_recipe_changes_do_not_change_initial_weights():
    from model import GVPPocketClassifier

    states = []
    for recipe in mc.RECIPES:
        torch.manual_seed(42)
        model = GVPPocketClassifier(esm_dim=4, hidden_s=8, hidden_v=2, edge_hidden=8, n_layers=1,
                                   esm_fusion_dim=8, n_metal=4, predict_ec=False, **mc.recipe_config(recipe))
        states.append(model.state_dict())
    assert all(torch.equal(states[0][key], state[key]) for state in states[1:] for key in states[0])


def test_outputs_cannot_write_into_parent_or_through_cache_symlink(tmp_path):
    parent, successor = tmp_path / "parent", tmp_path / "successor"
    parent.mkdir()
    with pytest.raises(ValueError, match="separate"):
        mc.isolated_paths(parent, parent / "new")
    successor.mkdir()
    (successor / "raw_graph_cache").symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        mc.isolated_paths(parent, successor)


def test_run_cli_has_no_authorization_defaults_or_runtime_override():
    argv = ["--action", "run", "--manifest", "manifest.json", "--manifest-sha256", "a" * 64,
            "--unit", mc.discovery_units()[0].name, "--train-dir", "/synthetic/train", "--esm-dir", "/synthetic/esm"]
    with pytest.raises(SystemExit):
        worker.parse_args(argv)
    authorized = argv + ["--policy", "policy.json", "--session", "session.json", "--spending", "spending.json"]
    assert worker.parse_args(authorized).policy == Path("policy.json")
    with pytest.raises(SystemExit):
        worker.parse_args(authorized + ["--deadline-unix", "9999999999"])


def test_worker_refuses_before_command_or_fit_if_authorization_fails(tmp_path, monkeypatch):
    import training.access_guard as guard

    unit = mc.discovery_units()[0]
    manifest = {"parent_campaign_dir": str(tmp_path / "parent"), "successor_root": str(tmp_path / "successor"),
                "authorized_grid_units": [unit.name],
                "train_dir": str(tmp_path / "train"), "esm_dir": str(tmp_path / "esm")}
    monkeypatch.setattr(mc, "verify_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(guard, "install_forbidden_read_guard", lambda roots: None)
    calls = []

    def refused(**kwargs):
        calls.append(kwargs)
        raise ValueError("No approved execution policy")

    monkeypatch.setitem(sys.modules, "metal_completion_safety", SimpleNamespace(validate_worker_authorization=refused))
    monkeypatch.setattr(mc, "build_command", lambda *args, **kwargs: pytest.fail("built command before gate"))
    with pytest.raises(ValueError, match="No approved"):
        worker.run(manifest_path=tmp_path / "manifest.json", manifest_sha256="a" * 64,
                   policy_path=tmp_path / "policy.json", session_path=tmp_path / "session.json",
                   spending_path=tmp_path / "spending.json", unit=unit, lane=0,
                   train_dir=tmp_path / "train", esm_dir=tmp_path / "esm", python_bin=sys.executable, device="cpu")
    assert calls[0]["unit_name"] == unit.name
    assert not (tmp_path / "successor").exists()


@pytest.mark.parametrize("action", ["plan", "run"])
@pytest.mark.parametrize("key,missing", [("train_dir", False), ("esm_dir", False),
                                          ("train_dir", True), ("esm_dir", True)])
def test_worker_refuses_unbound_input_paths_before_command_or_authorization(
        tmp_path, monkeypatch, action, key, missing):
    import training.access_guard as guard

    unit = mc.discovery_units()[0]
    train, esm = tmp_path / "train", tmp_path / "esm"
    manifest = {"train_dir": str(train), "esm_dir": str(esm)}
    if missing:
        manifest.pop(key)
    else:
        manifest[key] = str(tmp_path / "other" / Path(manifest[key]).name)
    monkeypatch.setattr(mc, "verify_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(guard, "install_forbidden_read_guard", lambda roots: None)

    def unexpected(*args, **kwargs):
        pytest.fail("Input mismatch reached command construction or authorization")

    monkeypatch.setattr(mc, "build_command", unexpected)
    monkeypatch.setitem(sys.modules, "metal_completion_safety", SimpleNamespace(
        validate_worker_authorization=unexpected))
    argv = ["--action", action, "--manifest", str(tmp_path / "manifest.json"),
            "--manifest-sha256", "a" * 64, "--unit", unit.name, "--train-dir", str(train),
            "--esm-dir", str(esm)]
    if action == "run":
        argv += ["--policy", "policy.json", "--session", "session.json", "--spending", "spending.json"]
    with pytest.raises(ValueError, match=key):
        worker.main(argv)
    assert list(tmp_path.iterdir()) == []


def test_worker_input_binding_compares_resolved_paths(tmp_path):
    train, esm = tmp_path / "train", tmp_path / "esm"
    train.mkdir()
    esm.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    worker.require_manifest_input_paths({"train_dir": str(train), "esm_dir": str(esm)},
                                        train_dir=alias / "train", esm_dir=alias / "esm")


@pytest.fixture
def manifest_fixture(tmp_path, monkeypatch):
    package, parent, output = tmp_path / "code", tmp_path / "parent", tmp_path / "output"
    package.mkdir()
    parent.mkdir()
    names = ("metal_completion_campaign.py", "run_metal_completion.py", "metal_completion_safety.py")
    for name in names:
        (package / name).write_text("# synthetic package identity\n")
    (parent / "campaign_manifest.json").write_text("{}")
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    manifest = {"schema_version": 1, "campaign_id": mc.CAMPAIGN_ID, "code_root": str(package),
                "parent_campaign_dir": str(parent), "successor_root": str(output),
                "train_dir": str(tmp_path / "train"), "esm_dir": str(tmp_path / "esm"),
                "training_input_verification": {"synthetic_verified": True},
                "code_sha256": {name: mc.sha256_file(package / name) for name in names},
                "parent_manifest_sha256": mc.sha256_file(parent / "campaign_manifest.json"),
                "parent_extension_sha256": None, "source_tree_sha256": "c" * 64,
                "authorized_grid_units": [unit.name for unit in mc.discovery_units()],
                "assessment_spec_path": str(spec), "assessment_spec_sha256": mc.sha256_file(spec)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(mc, "ROOT", package)
    monkeypatch.setattr(v3, "ROOT", package)
    monkeypatch.setattr(v3, "verify_campaign", lambda *args, **kwargs: {
        "frozen_source_tree_sha256": "c" * 64, "cohort": {"sha256": "d" * 64},
        "fold_set": {"fold_membership_sha256": "e" * 64}})
    monkeypatch.setattr(v3, "require_frozen_spec", lambda paths: {})
    monkeypatch.setattr(mc, "verify_training_inputs", lambda *args: {"synthetic_verified": True})
    return path, manifest, package, parent


@pytest.mark.parametrize("changed", ["worker", "parent", "source", "spec", "grid", "training", "esm"])
def test_changed_package_or_scientific_inputs_block_manifest(manifest_fixture, changed):
    path, manifest, package, parent = manifest_fixture
    assert mc.verify_manifest(path, mc.sha256_file(path), esm_dir=path.parent / "esm") == manifest
    if changed == "worker":
        (package / "run_metal_completion.py").write_text("# changed")
    elif changed == "parent":
        (parent / "campaign_manifest.json").write_text('{"changed":true}')
    elif changed == "source":
        manifest["source_tree_sha256"] = "a" * 64
    elif changed == "spec":
        Path(manifest["assessment_spec_path"]).write_text('{"changed":true}')
    elif changed == "grid":
        manifest["authorized_grid_units"].pop()
    elif changed == "training":
        manifest["training_input_verification"] = {"synthetic_verified": False}
    else:
        manifest["esm_dir"] = str(path.parent / "different_esm")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        mc.verify_manifest(path, mc.sha256_file(path), esm_dir=path.parent / "esm")


def test_confirmation_requires_frozen_complete_matching_selection(manifest_fixture):
    path, manifest, _, _ = manifest_fixture
    selection = path.parent / "discovery_selection.json"
    selection.write_text(json.dumps({"phase": "discovery", "complete": True, "shared_recipe": "P"}))
    manifest.update(shared_recipe="P", selection_receipt_path=str(selection),
                    selection_receipt_sha256=mc.sha256_file(selection),
                    authorized_grid_units=[unit.name for unit in mc.confirmation_units("P")])
    path.write_text(json.dumps(manifest))
    mc.verify_manifest(path, mc.sha256_file(path), esm_dir=path.parent / "esm")
    selection.write_text(json.dumps({"phase": "discovery", "complete": True, "shared_recipe": "S"}))
    manifest["selection_receipt_sha256"] = mc.sha256_file(selection)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        mc.verify_manifest(path, mc.sha256_file(path), esm_dir=path.parent / "esm")


@pytest.mark.parametrize("retry", ["none", "matching", "changed", "excessive"])
def test_mocked_worker_keeps_seed44_identity_and_persists_only_successor_lane(campaign, tmp_path, monkeypatch, retry):
    import benchmarking.pmm_execution as execution_module
    import training.access_guard as guard

    train, esm, parent, _, _ = campaign
    unit = mc.Unit("confirmation", "P", "six_class", 1, 44)
    output = tmp_path / "successor"
    manifest = {"parent_campaign_dir": str(parent.root), "successor_root": str(output),
                "authorized_grid_units": [unit.name], "code_sha256": {},
                "train_dir": str(train), "esm_dir": str(esm)}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(mc, "verify_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(guard, "install_forbidden_read_guard", lambda roots: None)
    monkeypatch.setattr(v3, "resolve_amp", lambda *args, **kwargs: False)
    runtime = {"session_id": "synthetic-controller-session", "deadline_unix": 10000,
               "allocation_started_unix": 100, "max_total_seconds": 9900,
               "estimated_fit_seconds": 100, "durable_root": str(tmp_path / "durable"),
               "persistence_mode": "host_pull", "lanes": 1}
    monkeypatch.setitem(sys.modules, "metal_completion_safety", SimpleNamespace(
        validate_worker_authorization=lambda **kwargs: runtime))
    events = []

    class FakeExecution:
        def __init__(self, root, **kwargs):
            self.root = root

        def __enter__(self):
            self.root.mkdir(parents=True, exist_ok=True)
            return self

        def __exit__(self, *args):
            return False

        def admit(self, name, forecast):
            events.append(("admit", name))

        def record_result(self, *args):
            events.append(("result", *args))

        def persist(self, paths):
            assert paths and all(path.is_relative_to(self.root) for path in paths)
            events.append(("persist",))

    def fake_execute(paths, lane, template, command, env, identity, **kwargs):
        assert paths.root == output
        assert template.seed == 42  # only the legacy checkpoint-rule template
        assert identity["model_seed"] == parse_args(command[3:]).seed == 44
        assert identity["recipe"] == "P"
        events.append(("execute", kwargs["run_name"]))
        return {"status": "completed", "elapsed_seconds": 0.01}

    monkeypatch.setattr(execution_module, "CampaignExecution", FakeExecution)
    monkeypatch.setattr(v3, "execute_unit", fake_execute)
    if retry != "none":
        _, _, identity = mc.build_command(parent, unit, successor_root=output,
                                          manifest_sha256=mc.sha256_file(manifest_path), code_sha256={},
                                          python_bin=sys.executable, train_dir=train, esm_dir=esm,
                                          device="cpu", lane=0, amp=False)
        if retry == "changed":
            identity["model_seed"] = 43
        for attempt in range(1, 3 if retry == "excessive" else 2):
            archived = output / "failed_attempts" / unit.name / f"attempt{attempt}"
            archived.mkdir(parents=True)
            (archived / "archive_receipt.json").write_text(json.dumps({"identity": identity}))
    kwargs = dict(manifest_path=manifest_path, manifest_sha256=mc.sha256_file(manifest_path),
                  policy_path=policy, session_path=tmp_path / "session.json",
                  spending_path=tmp_path / "spend.json", unit=unit, lane=0, train_dir=train,
                  esm_dir=esm, python_bin=sys.executable, device="cpu")
    if retry in ("changed", "excessive"):
        with pytest.raises(ValueError, match="archived attempt|one unchanged rerun"):
            worker.run(**kwargs)
        assert events == [] and not (output / "claims").exists()
        return
    result = worker.run(**kwargs)
    assert result["status"] == "completed"
    assert [event[0] for event in events] == ["admit", "execute", "result", "persist"]
    claim = json.loads((output / "claims" / f"{unit.name}.json").read_text())
    assert claim["identity"]["model_seed"] == 44 and claim["identity"]["recipe"] == "P"
    assert not (parent.lane(0) / "runs" / unit.name).exists()


@pytest.fixture
def training_input_fixture(tmp_path, monkeypatch):
    from benchmarking import pmm_ion_cohort as cohort
    import training.access_guard as guard

    # Explicit per-child checks remain active; avoid a permanent audit hook in pytest.
    monkeypatch.setattr(guard, "install_forbidden_read_guard", lambda roots: None)
    train = tmp_path / "dataset" / "train"
    (train / "structures").mkdir(parents=True)
    structure = train / "structures" / "synthetic.pdb"
    structure.write_bytes(b"synthetic training structure\n")
    hashes = {structure.name: mc.sha256_file(structure)}
    for name in cohort.TRAIN_FILES:
        (train / name).write_text("synthetic_metadata\n")
    with (train / "structure_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["structure_name", "relative_path", "sha256"])
        writer.writeheader()
        writer.writerow({"structure_name": structure.name, "relative_path": f"structures/{structure.name}",
                         "sha256": hashes[structure.name]})
    parent = v3.V3Paths(tmp_path / "parent")
    parent.frozen.mkdir(parents=True)
    parent.cohort.write_text(f"structure_name,structure_sha256\n{structure.name},{hashes[structure.name]}\n")
    frozen = {"train_metadata_sha256": {name: mc.sha256_file(train / name) for name in cohort.TRAIN_FILES},
              "n_structure_files": 1, "structure_content_sha256": cohort._aggregate_structure_hash(hashes),
              "cohort": {"sha256": mc.sha256_file(parent.cohort)}}
    (parent.frozen / "campaign_manifest.json").write_text(json.dumps(frozen))
    return train, parent, structure, frozen


def test_training_input_receipt_checks_actual_bytes_and_allows_shared_training_store(training_input_fixture, tmp_path):
    train, parent, structure, frozen = training_input_fixture
    receipt = mc.verify_training_inputs(parent, train)
    assert receipt["train_metadata_sha256"] == frozen["train_metadata_sha256"]
    assert receipt["structure_content_sha256"] == frozen["structure_content_sha256"]
    assert receipt["n_structure_files"] == 1 and receipt["train_dir"] == str(train.resolve())
    shared = tmp_path / "shared_training_store.pdb"
    structure.rename(shared)
    structure.symlink_to(shared)
    assert mc.verify_training_inputs(parent, train) == receipt


@pytest.mark.parametrize("changed", ["metadata", "structure", "metadata_symlink", "structure_symlink"])
def test_training_input_mutations_and_held_out_child_symlinks_refused(training_input_fixture, monkeypatch, changed):
    train, parent, structure, _ = training_input_fixture
    target = train / "site_manifest.csv" if changed.startswith("metadata") else structure
    if changed.endswith("symlink"):
        held_out = train.parent / "test" / target.name
        held_out.parent.mkdir()
        held_out.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(held_out)
        original = mc.sha256_file

        def no_held_out_read(path):
            assert not Path(path).resolve().is_relative_to(held_out.parent)
            return original(path)

        monkeypatch.setattr(mc, "sha256_file", no_held_out_read)
        expected = "held-out path"
    else:
        target.write_bytes(b"changed bytes\n")
        expected = "metadata changed|structure bytes changed"
    with pytest.raises(ValueError, match=expected):
        mc.verify_training_inputs(parent, train)


@pytest.mark.parametrize("changed", ["basename", "relative_path", "duplicate", "aggregate", "count", "cohort"])
def test_training_input_manifest_shape_and_cohort_bindings_refused(training_input_fixture, changed):
    train, parent, structure, frozen = training_input_fixture
    path = train / "structure_manifest.csv"
    rows = list(csv.DictReader(path.read_text().splitlines()))
    if changed == "basename":
        rows[0]["structure_name"] = "../synthetic.pdb"
    elif changed == "relative_path":
        rows[0]["relative_path"] = "elsewhere/synthetic.pdb"
    elif changed == "duplicate":
        rows.append({**rows[0], "sha256": "0" * 64})
    elif changed == "aggregate":
        frozen["structure_content_sha256"] = "0" * 64
    elif changed == "count":
        frozen["n_structure_files"] = 2
    else:
        parent.cohort.write_text(f"structure_name,structure_sha256\n{structure.name},{'0' * 64}\n")
        frozen["cohort"]["sha256"] = mc.sha256_file(parent.cohort)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frozen["train_metadata_sha256"][path.name] = mc.sha256_file(path)
    (parent.frozen / "campaign_manifest.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError):
        mc.verify_training_inputs(parent, train)
