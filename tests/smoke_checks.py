from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_structures import PocketRecord
from training.config import default_selection_metric_for_task, parse_args, required_targets_for_task
from training.run import ec_group_metrics_from_logits, validate_training_configuration
from training.splits import assign_ec_group_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class SkipCheck(RuntimeError):
    """Raised when an optional smoke check needs local data that is absent."""


def run_help(script_path: Path) -> str:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [PYTHON, str(script_path), "--help"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def check_training_cli_help() -> None:
    help_text = run_help(REPO_ROOT / "src" / "train.py")
    expected_options = (
        "--deterministic",
        "--metal-loss-weight",
        "--ec-loss-weight",
        "--ec-group-weighting",
        "--fusion-mode",
        "--use-ring-edges",
        "--cross-attention-layers",
        "--cross-attention-heads",
        "--cross-attention-dropout",
        "--cross-attention-neighborhood",
        "--cross-attention-bidirectional",
        "--allow-train-loss-test-eval-debug",
    )
    missing = [option for option in expected_options if option not in help_text]
    if missing:
        raise AssertionError(f"Training CLI help is missing expected options: {missing}")


def check_test_eval_safety() -> None:
    unsafe_config = parse_args(
        [
            "--task",
            "metal",
            "--test-structure-dir",
            "/tmp/deepmzyme_missing_test_structures",
            "--test-summary-csv",
            "/tmp/deepmzyme_missing_test_summary.csv",
            "--run-test-eval",
        ]
    )
    try:
        validate_training_configuration(unsafe_config)
    except ValueError as exc:
        message = str(exc)
        if "--run-test-eval is for held-out reporting" not in message:
            raise AssertionError(f"Unsafe test-eval config failed with an unexpected error: {message}") from exc
    else:
        raise AssertionError("Unsafe test-eval config without validation was not rejected.")

    debug_config = parse_args(
        [
            "--task",
            "metal",
            "--test-structure-dir",
            "/tmp/deepmzyme_missing_test_structures",
            "--test-summary-csv",
            "/tmp/deepmzyme_missing_test_summary.csv",
            "--run-test-eval",
            "--allow-train-loss-test-eval-debug",
        ]
    )
    validate_training_configuration(debug_config)


def check_loss_weight_validation() -> None:
    default_config = parse_args([])
    if default_config.metal_loss_weight != 1.0:
        raise AssertionError(f"Expected default metal_loss_weight=1.0, got {default_config.metal_loss_weight}")
    if default_config.ec_loss_weight != 1.0:
        raise AssertionError(f"Expected default ec_loss_weight=1.0, got {default_config.ec_loss_weight}")

    for option in ("--metal-loss-weight", "--ec-loss-weight"):
        config = parse_args([option, "-0.1"])
        try:
            validate_training_configuration(config)
        except ValueError as exc:
            if option not in str(exc):
                raise AssertionError(f"{option} failed with an unexpected error: {exc}") from exc
        else:
            raise AssertionError(f"{option} accepted a negative value.")


def check_ec_group_weighting_config() -> None:
    default_config = parse_args([])
    if default_config.ec_group_weighting != "structure_id":
        raise AssertionError(
            f"Expected default ec_group_weighting='structure_id', got {default_config.ec_group_weighting!r}"
        )
    ec_config = parse_args(["--task", "ec", "--val-fraction", "0.2"])
    if ec_config.selection_metric != "val_ec_group_balanced_acc":
        raise AssertionError(f"Expected EC default selection metric to use group metrics, got {ec_config.selection_metric!r}")
    if default_selection_metric_for_task("joint", has_validation=True) != "val_joint_balanced_acc":
        raise AssertionError("Joint default selection metric changed unexpectedly.")

    metal_config = parse_args(["--task", "metal", "--ec-group-weighting", "pdbid"])
    validate_training_configuration(metal_config)
    if required_targets_for_task("metal") != ("metal",):
        raise AssertionError("Metal-only task unexpectedly requires EC supervision.")


def check_cross_attention_config() -> None:
    config = parse_args(
        [
            "--model-architecture",
            "gvp",
            "--fusion-mode",
            "cross_modal_attention",
            "--cross-attention-layers",
            "2",
            "--cross-attention-heads",
            "8",
            "--cross-attention-dropout",
            "0.2",
            "--cross-attention-neighborhood",
            "first_second_shell",
            "--cross-attention-bidirectional",
        ]
    )
    validate_training_configuration(config)
    expected = {
        "model_architecture": "gvp",
        "fusion_mode": "cross_modal_attention",
        "cross_attention_layers": 2,
        "cross_attention_heads": 8,
        "cross_attention_dropout": 0.2,
        "cross_attention_neighborhood": "first_second_shell",
        "cross_attention_bidirectional": True,
    }
    for key, expected_value in expected.items():
        observed_value = getattr(config, key)
        if observed_value != expected_value:
            raise AssertionError(f"Expected {key}={expected_value!r}, got {observed_value!r}")


def check_ring_edge_cli_config() -> None:
    default_config = parse_args([])
    if default_config.use_ring_edges:
        raise AssertionError("Default training config should use radius-only edges.")
    if not default_config.prepare_missing_ring_edges:
        raise AssertionError("Default training config should prepare missing RING edges when RING is enabled.")

    optional_config = parse_args(["--use-ring-edges"])
    if (
        not optional_config.use_ring_edges
        or optional_config.require_ring_edges
        or not optional_config.prepare_missing_ring_edges
    ):
        raise AssertionError("Expected --use-ring-edges to enable optional RING edges without requiring them.")

    required_config = parse_args(["--require-ring-edges"])
    if (
        not required_config.use_ring_edges
        or not required_config.require_ring_edges
        or not required_config.prepare_missing_ring_edges
    ):
        raise AssertionError("Expected --require-ring-edges to imply use_ring_edges.")

    prepared_config = parse_args(["--prepare-missing-ring-edges"])
    if not prepared_config.use_ring_edges or not prepared_config.prepare_missing_ring_edges:
        raise AssertionError("Expected --prepare-missing-ring-edges to imply use_ring_edges.")

    disabled_prepare_config = parse_args(["--use-ring-edges", "--no-prepare-missing-ring-edges"])
    if not disabled_prepare_config.use_ring_edges or disabled_prepare_config.prepare_missing_ring_edges:
        raise AssertionError("Expected --no-prepare-missing-ring-edges to disable automatic RING generation.")


def check_only_gvp_does_not_require_esm() -> None:
    only_gvp_config = parse_args(["--model-architecture", "only_gvp"])
    if only_gvp_config.require_esm_embeddings or only_gvp_config.use_esm_branch:
        raise AssertionError("Only-GVP runs should not require or generate ESM embeddings.")

    esm_config = parse_args(["--model-architecture", "only_esm"])
    if not esm_config.require_esm_embeddings or not esm_config.use_esm_branch:
        raise AssertionError("Only-ESM runs should require ESM embeddings by default.")


def check_graph_ring_edges_are_opt_in() -> None:
    from data_structures import EDGE_SOURCE_TO_INDEX, ResidueRecord
    from graph.construction import pocket_to_pyg_data

    with tempfile.TemporaryDirectory(prefix="deepmzyme_ring_opt_in_") as tmp:
        ring_path = Path(tmp) / "example_ringEdges"
        ring_path.write_text(
            "NodeId1\tNodeId2\tInteraction\tAtom1\tAtom2\n"
            "A:1:_:ALA\tA:2:_:GLU\tHBOND:SC_SC\tCA\tCA\n",
            encoding="utf-8",
        )
        pocket = PocketRecord(
            structure_id="example",
            pocket_id="example_A_1",
            metal_element="ZN",
            metal_coords=[torch.tensor([0.0, 0.0, 0.0])],
            residues=[
                ResidueRecord(
                    chain_id="A",
                    resseq=1,
                    icode="",
                    resname="ALA",
                    atoms={"CA": torch.tensor([1.0, 0.0, 0.0]), "CB": torch.tensor([1.5, 0.0, 0.0])},
                ),
                ResidueRecord(
                    chain_id="A",
                    resseq=2,
                    icode="",
                    resname="GLU",
                    atoms={
                        "CA": torch.tensor([3.0, 0.0, 0.0]),
                        "CB": torch.tensor([3.5, 0.0, 0.0]),
                        "OE1": torch.tensor([3.5, 0.5, 0.0]),
                        "OE2": torch.tensor([3.5, -0.5, 0.0]),
                    },
                ),
            ],
            metadata={"ring_edges_path": str(ring_path)},
        )

        default_graph = pocket_to_pyg_data(pocket, esm_dim=2)
        ring_idx = EDGE_SOURCE_TO_INDEX["ring"]
        if int((default_graph.edge_source_type[:, ring_idx] > 0.5).sum().item()) != 0:
            raise AssertionError("Default graph construction used RING edges; expected radius-only.")

        ring_graph = pocket_to_pyg_data(pocket, esm_dim=2, use_ring_edges=True)
        if int((ring_graph.edge_source_type[:, ring_idx] > 0.5).sum().item()) == 0:
            raise AssertionError("--use-ring-edges path did not include available RING edges.")


def check_colab_notebook_sweep_source() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "DeepMzyme_training_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
    required_tokens = (
        'RUN_TRAINING = False',
        'RUN_HELD_OUT_TEST_EVAL = False',
        'MODEL_PRESETS_CSV = "Only-GVP"',
        'RING_EDGE_MODES_CSV = "radius_only"',
        'OMIT_NODE_FEATURE_SETS = ""',
        'MAX_CONFIGURATION_RUNS',
        "CONFIG = {",
        "COLAB_DATA_SOURCE",
        "huggingface_link",
        "DeepMzyme_Data_runtime_local_2026-05-03.tar.zst",
        "c86faa40ff69c021de02b72b5fef9ebd1712f5ef8e6cb3da27b3a9e8261816c1",
        "site-level MAHOMES summary CSV",
        "structure-level inspection CSV",
        "MODEL_PRESET_MAP",
        "Only-GVP",
        "GVP + cross-modal attention",
        "SimpleGNN + ESM",
        "def parse_omit_node_feature_sets",
        "validate_node_feature_omissions",
        "def build_train_command",
        "ring_mode",
        "omit_node_features",
        "--omit-node-features",
        "--use-ring-edges",
        "--require-ring-edges",
        "--ring-features-dir",
        "--prepare-missing-ring-edges",
        "--no-prepare-missing-ring-edges",
        "--no-prepare-missing-esm-embeddings",
        "RING_EXE_PATH",
        '"src" / "report_runs.py"',
        "Recommended First Runs",
    )
    missing = [token for token in required_tokens if token not in source]
    if missing:
        raise AssertionError(f"Colab notebook source is missing expected tokens: {missing}")
    forbidden_tokens = ("ipywidgets", "widgets.", "MAX_SWEEP_RUNS", "RUN_SWEEP_MODE")
    present = [token for token in forbidden_tokens if token in source]
    if present:
        raise AssertionError(f"Colab notebook still contains retired widget/old-runner tokens: {present}")
    if "sweep" in source.lower():
        raise AssertionError("Colab notebook should not contain user-facing sweep terminology.")
    return
    required_tokens = (
        "RUN_SINGLE_MODE",
        "RUN_SWEEP_MODE",
        "ALLOW_SINGLE_AND_SWEEP",
        "def _running_in_colab",
        'RUN_MODE = "auto"',
        "RUN_MODE must be 'auto', 'local', or 'colab'.",
        'COLAB_DATA_SOURCE = "huggingface_link"',
        "COLAB_DATA_SOURCE must be 'upload_file', 'huggingface_link', or 'drive'.",
        "https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/DeepMzyme_Data_runtime_local_2026-05-03.tar.zst",
        "COLAB_HUGGINGFACE_BUNDLE_SHA256",
        'BUNDLE_FILENAME = "DeepMzyme_Data_runtime_local_2026-05-03.tar.zst"',
        'MOUNT_DRIVE = RUN_MODE == "colab" and COLAB_DATA_SOURCE == "drive"',
        "DRIVE_MOUNT_TIMEOUT_MS",
        "use_drive_paths = drive_requested and drive_available",
        "COLAB_LOCAL_DATA_DIR",
        "COLAB_BUNDLE_PATH",
        "UPLOAD_LOCAL_BUNDLE_IN_COLAB",
        "DOWNLOAD_HUGGINGFACE_BUNDLE_IN_COLAB",
        "def download_file",
        "def verify_sha256",
        "def adopt_unpacked_runtime_data_dir",
        "DeepMzyme Experiment Console",
        "def _has_ec_task",
        "def _has_metal_task",
        "def _has_cross_attention_model",
        "def _refresh_contextual_sections",
        "_info('head MLP layers'",
        "_info('EC label depth'",
        "_info('cross-modal attention'",
        "widgets.Tab",
        "Runtime-local paths under /content will be used",
        "def build_train_command",
        "def stream_command",
        "def write_sweep_status",
        "def summarize_completed_run",
        "def make_safe_run_name",
        "def validate_run_before_training",
        "USE_ESM_EMBEDDINGS = True",
        "ESM_EMBEDDINGS_DIR = '/content/drive/MyDrive/DeepMzyme/DeepMzyme_Data/esm_embeddings'",
        "ALLOW_MISSING_ESM_EMBEDDINGS = False",
        "RING_EDGES_MODE = 'radius_only'",
        "RING_EXE_PATH = 'DeepMzyme_Data/ring-4.0/out/bin/ring'",
        "def resolve_ring_exe_control",
        "drive_candidate = drive_data_dir / ring_path",
        "repo_candidate = repo_dir / drive_data_dir.name / ring_path",
        "drive_mount_available = str(drive_data_dir).startswith('/content/drive/') and Path('/content/drive').exists()",
        "RING_EXE_PATH user setting",
        "RING_EXE_PATH resolved",
        "os.environ['RING_EXE_PATH'] = str(resolve_ring_exe_control(RING_EXE_PATH))",
        "run_env['RING_EXE_PATH'] = config['ring_exe_path'] or str(resolve_ring_exe_control(RING_EXE_PATH))",
        "RING_EDGE_OUTPUT_DIR",
        "PRECOMPUTED_RING_EDGES_DIR",
        "RING_EDGES_MODES",
        "'use_ring_edges'",
        "itertools.product",
        "MAX_SWEEP_RUNS",
        "STOP_ON_FIRST_FAILURE",
        "sweep_status.csv",
        "val_ec_group_balanced_acc",
        "src' / 'train.py'",
        "src' / 'report_runs.py'",
        "--cross-attention-layers",
        "--cross-attention-heads",
        "--lr-schedule",
        "--use-early-esm",
        "--esm-embeddings-dir",
        "--allow-missing-esm-embeddings",
        "--no-prepare-missing-esm-embeddings",
        "--use-ring-edges",
        "--require-ring-edges",
        "--prepare-missing-ring-edges",
        "'selected_checkpoint'",
        "'selected_metric_value'",
        "'test_summary'",
        "'command'",
        "'uses_esm'",
        "'esm_embeddings_dir'",
        "'ring_edges_mode'",
        "'ring_edge_output_dir'",
    )
    missing = [token for token in required_tokens if token not in source]
    if missing:
        raise AssertionError(f"Colab notebook sweep source is missing expected tokens: {missing}")
    if "subprocess.run(train_cmd" in source:
        raise AssertionError("Colab notebook should stream training logs instead of subprocess.run(train_cmd).")
    if "os.environ['RING_EXE_PATH'] = str(path_from_control(RING_EXE_PATH))" in source:
        raise AssertionError("Colab notebook should resolve RING_EXE_PATH before exporting it.")
    if "MODEL_PRESET = 'Only-GVP'" not in source:
        raise AssertionError("Colab notebook default MODEL_PRESET no longer preserves Only-GVP.")
    if "NODE_FEATURE_SET = 'conservative'  #@param ['conservative', 'expanded']" in source:
        raise AssertionError("Colab notebook exposes node feature sets not accepted by the current CLI.")


def check_colab_generated_training_commands_parse() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "DeepMzyme_training_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    command_builder_source = next(
        (
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code" and "def build_train_command" in "".join(cell.get("source", []))
        ),
        None,
    )
    if command_builder_source is None:
        raise AssertionError("Could not find the Colab command-builder cell.")

    import contextlib
    import copy
    import io

    def base_config(tmp_root: Path) -> dict[str, object]:
        return {
            "basic": {
                "task": "metal",
                "epochs": 1,
                "run_training": False,
                "device": "cpu",
                "run_held_out_test_eval": False,
            },
            "configuration_comparison": {
                "model_presets_csv": "Only-GVP",
                "ring_edge_modes_csv": "radius_only",
                "batch_sizes_csv": "4",
                "learning_rates_csv": "1e-4",
                "weight_decays_csv": "1e-4",
                "seeds_csv": "42",
                "max_configuration_runs": 24,
                "stop_on_first_failure": True,
                "skip_existing_runs": True,
            },
            "data": {"colab_data_source": "huggingface_link"},
            "esm": {
                "esm_embeddings_dir": "",
                "prepare_missing_esm_embeddings": False,
                "allow_missing_esm_embeddings": False,
                "disable_esm_branch": False,
                "esm_dim": 960,
            },
            "ring": {
                "ring_features_dir": str(tmp_root / "ring_features"),
                "ring_exe_path": str(tmp_root / "ring"),
            },
            "node_features": {
                "node_feature_set": "conservative",
                "omit_node_feature_sets": "",
            },
            "advanced": {
                "val_fraction": 0.15,
                "split_by": "pdbid",
                "selection_metric": "val_metal_balanced_acc",
                "hidden_s_values_csv": "128",
                "hidden_v_values_csv": "16",
                "edge_hidden_values_csv": "64",
                "gvp_layers_values_csv": "4",
                "head_mlp_layers_values_csv": "2",
                "edge_radius_values_csv": "8.0",
                "esm_fusion_dim_values_csv": "128",
                "lr_schedules_csv": "fixed",
                "lr_step_size": 10,
                "lr_decay_gamma": 0.5,
                "node_rbf_sigma": 0.75,
                "edge_rbf_sigma": 0.75,
                "node_rbf_use_raw_distances": False,
                "early_esm_dim": 32,
                "early_esm_dropout": 0.2,
                "early_esm_raw": False,
                "early_esm_scope": "all",
                "cross_attention_layers_csv": "1",
                "cross_attention_heads_csv": "4",
                "cross_attention_dropout": 0.1,
                "cross_attention_neighborhood": "all",
                "cross_attention_bidirectional": False,
                "ec_label_depths_csv": "1",
                "ec_group_weighting": "structure_id",
                "ec_contrastive_weights_csv": "0.0",
                "ec_contrastive_temperature": 0.1,
                "metal_loss_function": "cross_entropy",
                "metal_focal_gamma": 2.0,
                "metal_label_smoothing": 0.0,
                "metal_loss_weight": 1.0,
                "ec_loss_weight": 1.0,
                "mn_loss_multiplier": 1.0,
                "cu_loss_multiplier": 1.0,
                "zn_loss_multiplier": 1.0,
                "fe_loss_multiplier": 1.0,
                "co_loss_multiplier": 1.0,
                "ni_loss_multiplier": 1.0,
                "class_viii_loss_multiplier": 1.0,
                "balance_metal_site_symbols": False,
                "require_all_task_classes": False,
                "allow_missing_external_features": True,
                "external_features_root_dir": "",
                "external_feature_source": "auto",
                "n_folds": "",
                "fold_index": "",
                "deterministic": False,
                "save_epoch_checkpoints": False,
                "allow_train_loss_test_eval_debug": False,
                "unsupported_metal_policy": "error",
                "invalid_structure_policy": "skip",
            },
            "output": {
                "runs_dir": str(tmp_root / "runs"),
                "run_name_prefix": "",
                "copy_outputs_to_drive": False,
                "summary_basename": "summary",
            },
        }

    def run_builder(config_updates: dict[str, dict[str, object]]) -> list[dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="deepmzyme_colab_command_smoke_") as tmp:
            tmp_root = Path(tmp)
            config = base_config(tmp_root)
            for section, updates in config_updates.items():
                nested = config[section]
                if not isinstance(nested, dict):
                    raise AssertionError(f"Unexpected non-dict config section: {section}")
                nested.update(updates)
            ring_dir = Path(str(config["ring"]["ring_features_dir"]))
            ring_dir.mkdir(parents=True)
            (ring_dir / "example_ringEdges").write_text("NodeId1\tNodeId2\tInteraction\n", encoding="utf-8")
            ring_exe = Path(str(config["ring"]["ring_exe_path"]))
            ring_exe.write_text("#!/bin/sh\n", encoding="utf-8")
            ring_exe.chmod(0o755)
            train_dir = tmp_root / "train"
            test_dir = tmp_root / "test"
            train_dir.mkdir()
            test_dir.mkdir()
            train_csv = train_dir / "summary.csv"
            test_csv = test_dir / "summary.csv"
            train_csv.write_text("pdbid,metal residue number,EC number,metal residue type\n", encoding="utf-8")
            test_csv.write_text("pdbid,metal residue number,EC number,metal residue type\n", encoding="utf-8")
            namespace = {
                "CONFIG": config,
                "REPO_DIR": REPO_ROOT,
                "SRC_DIR": REPO_ROOT / "src",
                "TRAIN_DIR": train_dir,
                "TEST_DIR": test_dir,
                "TRAIN_CSV": train_csv,
                "TEST_CSV": test_csv,
                "DATA_ROOT": tmp_root,
                "DRIVE_DATA_DIR": tmp_root / "drive" / "DeepMzyme_Data",
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(command_builder_source, namespace)
            return copy.deepcopy(namespace["planned_runs"])

    def assert_training_command_parses(cmd: list[object]) -> None:
        parts = [str(part) for part in cmd]
        if parts[1] != str(REPO_ROOT / "src" / "train.py"):
            raise AssertionError(f"Notebook command used an unexpected training entry point: {parts[:2]}")
        config = parse_args(parts[2:])
        validate_training_configuration(config)

    default_runs = run_builder({})
    if len(default_runs) != 1:
        raise AssertionError(f"Expected one default planned command, got {len(default_runs)}")
    default_cmd = [str(part) for part in default_runs[0]["command"]]
    assert_training_command_parses(default_cmd)
    if "--use-ring-edges" in default_cmd or "--require-ring-edges" in default_cmd:
        raise AssertionError("Default radius-only command unexpectedly enables RING edges.")
    if "--esm-embeddings-dir" in default_cmd:
        raise AssertionError("Only-GVP default command should not require an ESM embeddings directory.")
    if "--omit-node-features" in default_cmd:
        raise AssertionError("Full-feature default command unexpectedly omits node features.")

    ring_runs = run_builder(
        {"configuration_comparison": {"ring_edge_modes_csv": "radius_only,radius_plus_precomputed_ring"}}
    )
    if len(ring_runs) != 2:
        raise AssertionError(f"Expected two RING planned commands, got {len(ring_runs)}")
    radius_cmd = [str(part) for part in ring_runs[0]["command"]]
    precomputed_cmd = [str(part) for part in ring_runs[1]["command"]]
    if "--use-ring-edges" in radius_cmd or "--require-ring-edges" in radius_cmd:
        raise AssertionError("radius_only command unexpectedly includes RING enable/require flags.")
    for expected_flag in ("--use-ring-edges", "--require-ring-edges", "--ring-features-dir"):
        if expected_flag not in precomputed_cmd:
            raise AssertionError(f"Precomputed-RING command is missing {expected_flag}.")

    omit_runs = run_builder(
        {"node_features": {"omit_node_feature_sets": ";v_cb_to_fg;v_cb_to_fg,v_res_to_metal"}}
    )
    if len(omit_runs) != 3:
        raise AssertionError(f"Expected three omission planned commands, got {len(omit_runs)}")
    omit_cmds = [[str(part) for part in run["command"]] for run in omit_runs]
    if "--omit-node-features" in omit_cmds[0]:
        raise AssertionError("Full-feature omission entry unexpectedly passed --omit-node-features.")
    if omit_cmds[1][omit_cmds[1].index("--omit-node-features") + 1] != "v_cb_to_fg":
        raise AssertionError("Single-feature omission command passed the wrong value.")
    if omit_cmds[2][omit_cmds[2].index("--omit-node-features") + 1] != "v_cb_to_fg,v_res_to_metal":
        raise AssertionError("Combined-feature omission command passed the wrong value.")
    return

    with tempfile.TemporaryDirectory(prefix="deepmzyme_colab_command_smoke_") as tmp:
        tmp_root = Path(tmp)
        runs_dir = tmp_root / "runs"
        drive_data_dir = tmp_root / "drive" / "DeepMzyme_Data"
        namespace = {
            "repo_dir": REPO_ROOT,
            "drive_data_dir": drive_data_dir,
            "local_runs_dir": runs_dir,
            "train_dir": tmp_root / "dataset" / "train",
            "test_dir": tmp_root / "dataset" / "test",
            "train_csv": tmp_root / "dataset" / "train" / "site_summary.csv",
            "test_csv": tmp_root / "dataset" / "test" / "site_summary.csv",
            "display": lambda *_args, **_kwargs: None,
            "TASK_MODE": "metal_6_class",
            "MODEL_PRESET": "Only-GVP",
            "EPOCHS": 1,
            "BATCH_SIZE": 2,
            "LEARNING_RATE": 3e-4,
            "WEIGHT_DECAY": 1e-4,
            "SEED": 42,
            "VAL_FRACTION": 0.2,
            "DEVICE": "cpu",
            "RUN_NAME": "",
            "EC_LABEL_DEPTH": 1,
            "EC_CONTRASTIVE_WEIGHT": 0.0,
            "NODE_FEATURE_SET": "conservative",
            "SPLIT_BY": "pdbid",
            "LR_SCHEDULE": "fixed",
            "LR_STEP_SIZE": 10,
            "LR_DECAY_GAMMA": 0.5,
            "CROSS_ATTENTION_LAYERS": [1],
            "CROSS_ATTENTION_HEADS": [4],
            "CROSS_ATTENTION_DROPOUT": 0.1,
            "CROSS_ATTENTION_NEIGHBORHOOD": "all",
            "CROSS_ATTENTION_BIDIRECTIONAL": False,
            "SINGLE_CROSS_ATTENTION_LAYERS": 1,
            "SINGLE_CROSS_ATTENTION_HEADS": 4,
            "USE_ESM_EMBEDDINGS": True,
            "ESM_EMBEDDINGS_DIR": str(drive_data_dir / "esm_embeddings"),
            "PREPARE_MISSING_ESM_EMBEDDINGS": False,
            "ALLOW_MISSING_ESM_EMBEDDINGS": False,
            "ALLOW_MISSING_EXTERNAL_FEATURES": True,
            "RING_EDGES_MODE": "radius_only",
            "RING_EXE_PATH": "DeepMzyme_Data/ring-4.0/out/bin/ring",
            "RING_EDGE_OUTPUT_DIR": str(drive_data_dir / "RING_features"),
            "PRECOMPUTED_RING_EDGES_DIR": str(drive_data_dir / "precomputed_ring_edges"),
            "REQUIRE_RING_EDGES": False,
            "PREPARE_MISSING_RING_EDGES": False,
            "RUN_HELD_OUT_TEST_EVAL": True,
            "TASK_MODES": ["metal_6_class"],
            "MODEL_PRESETS": ["Only-GVP"],
            "RING_EDGES_MODES": ["radius_only"],
            "LEARNING_RATES": [3e-4],
            "WEIGHT_DECAYS": [1e-4],
            "BATCH_SIZES": [2],
            "SEEDS": [42],
            "NODE_FEATURE_SETS": ["conservative"],
            "EC_LABEL_DEPTHS": [1],
            "EC_CONTRASTIVE_WEIGHTS": [0.0],
            "LR_SCHEDULES": ["fixed"],
            "MAX_SWEEP_RUNS": 24,
        }
        namespace["task_map"] = {
            "metal_6_class": ("metal", "val_metal_balanced_acc"),
            "metal_collapsed4_metric": ("metal", "val_metal_collapsed4_balanced_acc"),
            "ec_prediction": ("ec", "val_ec_group_balanced_acc"),
        }
        namespace["preset_map"] = {
            "Only-GVP": {"model_architecture": "only_gvp", "uses_esm": False},
            "Only-ESM": {"model_architecture": "only_esm", "uses_esm": True},
            "GVP + late fusion": {"model_architecture": "gvp", "fusion_mode": "late_fusion", "uses_esm": True},
            "GVP + early fusion": {
                "model_architecture": "gvp",
                "fusion_mode": "early_fusion",
                "early_esm_dim": 32,
                "early_esm_dropout": 0.2,
                "uses_esm": True,
            },
            "GVP + cross-modal attention": {
                "model_architecture": "gvp",
                "fusion_mode": "cross_modal_attention",
                "uses_esm": True,
            },
        }

        exec(command_builder_source, namespace)

        def assert_training_command_parses(cmd: list[object]) -> None:
            parts = [str(part) for part in cmd]
            if parts[1] != str(REPO_ROOT / "src" / "train.py"):
                raise AssertionError(f"Notebook command used an unexpected training entry point: {parts[:2]}")
            config = parse_args(parts[2:])
            validate_training_configuration(config)

        default_cmd = namespace["train_cmd"]
        assert_training_command_parses(default_cmd)
        if "--use-ring-edges" in default_cmd:
            raise AssertionError("Notebook radius_only command unexpectedly enables RING edges.")

        config = namespace["build_run_config"](
            task_mode="metal_6_class",
            model_preset="GVP + cross-modal attention",
            ring_edges_mode="radius_plus_precomputed_ring",
            learning_rate=3e-4,
            weight_decay=1e-4,
            batch_size=2,
            seed=42,
            node_feature_set="conservative",
            ec_label_depth=1,
            ec_contrastive_weight=0.0,
            cross_attention_layers=2,
            cross_attention_heads=4,
            lr_schedule="fixed",
        )
        ring_cmd = namespace["build_train_command"](config)
        assert_training_command_parses(ring_cmd)
        if "--use-ring-edges" not in ring_cmd:
            raise AssertionError("Notebook precomputed-RING command did not pass --use-ring-edges.")
        if "--cross-attention-layers" not in ring_cmd:
            raise AssertionError("Notebook cross-modal command did not pass cross-attention flags.")

        namespace["PREPARE_MISSING_RING_EDGES"] = True
        config = namespace["build_run_config"](
            task_mode="metal_6_class",
            model_preset="Only-GVP",
            ring_edges_mode="generate_missing_ring",
            learning_rate=3e-4,
            weight_decay=1e-4,
            batch_size=2,
            seed=42,
            node_feature_set="conservative",
            ec_label_depth=1,
            ec_contrastive_weight=0.0,
            cross_attention_layers=1,
            cross_attention_heads=4,
            lr_schedule="fixed",
        )
        generate_cmd = namespace["build_train_command"](config)
        assert_training_command_parses(generate_cmd)
        for expected_flag in ("--use-ring-edges", "--prepare-missing-ring-edges"):
            if expected_flag not in generate_cmd:
                raise AssertionError(f"Notebook generated-RING command is missing {expected_flag}.")


def check_ring_environment_overrides() -> None:
    from graph.ring_edges import canonical_ring_edges_output_path
    from embed_helpers.Interaction_edge import DEFAULT_RING_EXE, create_ring_edges_batch
    from training.runtime_preparation import prepare_runtime_inputs

    personal_ring_path = str(Path("/home") / "mechti" / "ring-4.0" / "out" / "bin" / "ring")
    if DEFAULT_RING_EXE.is_absolute():
        raise AssertionError(f"RING default fallback should be repo-relative, got: {DEFAULT_RING_EXE}")
    expected_default = Path("DeepMzyme_Data") / "ring-4.0" / "out" / "bin" / "ring"
    if DEFAULT_RING_EXE != expected_default:
        raise AssertionError(f"Unexpected RING default fallback: {DEFAULT_RING_EXE}")
    if str(DEFAULT_RING_EXE) == personal_ring_path:
        raise AssertionError(f"RING default fallback still uses a personal path: {DEFAULT_RING_EXE}")
    if personal_ring_path in str(DEFAULT_RING_EXE):
        raise AssertionError(f"RING default fallback unexpectedly contains a personal path: {DEFAULT_RING_EXE}")

    old_ring_features_dir = os.environ.get("RING_FEATURES_DIR")
    old_ring_exe_path = os.environ.get("RING_EXE_PATH")
    try:
        with tempfile.TemporaryDirectory(prefix="deepmzyme_ring_edges_smoke_") as tmp:
            tmp_root = Path(tmp)
            structure_dir = tmp_root / "structures"
            structure_dir.mkdir()
            ring_dir = tmp_root / "ring_edges"
            os.environ["RING_FEATURES_DIR"] = str(ring_dir)

            expected_path = canonical_ring_edges_output_path("/tmp/example_structure.pdb")
            if not str(expected_path).startswith(f"{ring_dir}/"):
                raise AssertionError(f"RING edge lookup did not honor RING_FEATURES_DIR: {expected_path}")

            report = prepare_runtime_inputs(
                structure_dir=structure_dir,
                esm_embeddings_dir=None,
                require_esm_embeddings=False,
                prepare_missing_esm_embeddings=False,
                use_ring_edges=False,
                require_ring_edges=False,
                prepare_missing_ring_edges=False,
            )
            if report["ring_edges_output_dir"] != str(ring_dir):
                raise AssertionError(f"RING edge output did not honor RING_FEATURES_DIR: {report}")

        os.environ["RING_EXE_PATH"] = "/tmp/deepmzyme_missing_ring_executable"
        try:
            create_ring_edges_batch([], dir_results="/tmp/deepmzyme_ring_edges_smoke")
        except FileNotFoundError as exc:
            if "/tmp/deepmzyme_missing_ring_executable" not in str(exc):
                raise AssertionError(f"RING executable error did not mention RING_EXE_PATH: {exc}") from exc
        else:
            raise AssertionError("Missing RING_EXE_PATH executable was not rejected.")
    finally:
        if old_ring_features_dir is None:
            os.environ.pop("RING_FEATURES_DIR", None)
        else:
            os.environ["RING_FEATURES_DIR"] = old_ring_features_dir
        if old_ring_exe_path is None:
            os.environ.pop("RING_EXE_PATH", None)
        else:
            os.environ["RING_EXE_PATH"] = old_ring_exe_path


def synthetic_pocket(structure_id: str, pocket_id: str, y_ec: int | None) -> PocketRecord:
    return PocketRecord(
        structure_id=structure_id,
        pocket_id=pocket_id,
        metal_element="ZN",
        metal_coords=[torch.zeros(3)],
        residues=[],
        y_ec=y_ec,
    )


def check_ec_group_weights_sum_per_group() -> None:
    pockets = [
        synthetic_pocket("1abc__chain_A__EC_1.1.1.1", "p0", 0),
        synthetic_pocket("1abc__chain_A__EC_1.1.1.1", "p1", 0),
        synthetic_pocket("2def__chain_B__EC_2.2.2.2", "p2", 1),
        synthetic_pocket("3ghi__chain_C__EC_3.3.3.3", "p3", None),
    ]
    assign_ec_group_metadata(pockets, weighting_mode="structure_id")
    sums: dict[str, float] = {}
    for pocket in pockets:
        if pocket.y_ec is None:
            continue
        group_key = str(pocket.metadata["ec_group_key"])
        sums[group_key] = sums.get(group_key, 0.0) + float(pocket.metadata["ec_sample_weight"])
    for group_key, total in sums.items():
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(f"EC weights for group {group_key} sum to {total}, expected 1.0")


def check_ec_group_metric_aggregation() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1], dtype=torch.long)
    group_indices = torch.tensor([0, 0, 1], dtype=torch.long)
    metrics = ec_group_metrics_from_logits(
        logits,
        targets,
        group_indices,
        ec_label_map={0: "1", 1: "2"},
        ec_label_depth=1,
    )
    if metrics["n_groups"] != 2 or metrics["n_conflicting_groups"] != 0:
        raise AssertionError(f"Unexpected EC group counts: {metrics}")
    if metrics["accuracy"] != 1.0 or metrics["balanced_accuracy"] != 1.0 or metrics["macro_f1"] != 1.0:
        raise AssertionError(f"Expected perfect EC group metrics, got {metrics}")
    if metrics["level_1_accuracy"] != 1.0:
        raise AssertionError(f"Expected perfect EC level-1 group metrics, got {metrics}")


def check_ec_group_id_batches_without_increment() -> None:
    loader = DataLoader(
        [
            Data(x=torch.zeros(2, 1), y_ec=torch.tensor([0]), ec_group_id=torch.tensor([0])),
            Data(x=torch.zeros(3, 1), y_ec=torch.tensor([0]), ec_group_id=torch.tensor([0])),
        ],
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))
    if batch.ec_group_id.tolist() != [0, 0]:
        raise AssertionError(f"EC group IDs were shifted during PyG batching: {batch.ec_group_id.tolist()}")


def check_conflicting_ec_group_metrics_are_skipped() -> None:
    metrics = ec_group_metrics_from_logits(
        torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
        ec_label_map={0: "1", 1: "2"},
        ec_label_depth=1,
    )
    if metrics["n_groups"] != 0 or metrics["n_conflicting_groups"] != 1:
        raise AssertionError(f"Expected one skipped conflicting EC group, got {metrics}")


def check_bundle_cli_help() -> None:
    help_text = run_help(REPO_ROOT / "src" / "build_colab_bundle.py")
    expected_options = (
        "--allow-multi-metal-structures",
        "--strict-single-metal-structures",
    )
    missing = [option for option in expected_options if option not in help_text]
    if missing:
        raise AssertionError(f"Bundle CLI help is missing expected options: {missing}")


def check_docs_do_not_use_broken_training_command() -> None:
    broken_module = ".".join(("src", "training", "run"))
    broken_patterns = (f"python -m {broken_module}", broken_module)
    for relative_path in ("README.md", "list_train_commands.md"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        matches = [pattern for pattern in broken_patterns if pattern in text]
        if matches:
            raise AssertionError(f"{relative_path} still contains broken command patterns: {matches}")


def check_multi_metal_site_level_granularity() -> None:
    structure_id = "1cob__chain_A__EC_1.15.1.1"
    dataset_root = REPO_ROOT / "DeepMzyme_Data" / "train_and_test_sets_structures_non_overlapped_pinmymetal"
    train_dir = dataset_root / "train"
    structure_path = train_dir / f"{structure_id}.pdb"
    site_summary_csv = train_dir / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
    inspection_csv = (
        REPO_ROOT
        / "DeepMzyme_Data"
        / "DeepMzyme_Colab_Bundles"
        / dataset_root.name
        / f"{dataset_root.name}_train.csv"
    )

    required_paths = (structure_path, site_summary_csv, inspection_csv)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise SkipCheck(f"local multi-metal fixture files are absent: {missing}")

    from graph.construction import pocket_to_pyg_data
    from label_schemes import METAL_TARGET_LABELS
    from training.site_filter import load_allowed_site_metal_labels, resolve_allowed_site_metal_labels
    from training.structure_loading import load_structure_pockets

    pockets, _feature_fallbacks, skipped_pockets = load_structure_pockets(
        structure_path=structure_path,
        structure_root=train_dir,
        allowed_site_metal_labels=resolve_allowed_site_metal_labels(site_summary_csv),
        esm_dim=960,
        embeddings_dir=train_dir / "esm_embeddings",
        require_esm_embeddings=False,
        feature_root_dir=train_dir,
        external_feature_source="auto",
        require_external_features=False,
        ec_label_depth=1,
    )
    if skipped_pockets:
        raise AssertionError(f"Expected no skipped pockets for {structure_id}, got: {skipped_pockets}")
    if len(pockets) != 2:
        raise AssertionError(f"Expected {structure_id} to load as 2 pocket samples, got {len(pockets)}")

    observed_labels: dict[str, str] = {}
    for pocket in pockets:
        if pocket.y_metal is None:
            raise AssertionError(f"Pocket {pocket.pocket_id} is missing y_metal.")
        metal_label = METAL_TARGET_LABELS[int(pocket.y_metal)]
        data = pocket_to_pyg_data(pocket, esm_dim=960)
        if tuple(data.y_metal.shape) != (1,):
            raise AssertionError(f"Pocket {pocket.pocket_id} has non-scalar y_metal shape {tuple(data.y_metal.shape)}")
        if str(data.y_metal.dtype) != "torch.int64":
            raise AssertionError(f"Pocket {pocket.pocket_id} has non-integer y_metal dtype {data.y_metal.dtype}")
        if ";" in metal_label:
            raise AssertionError(f"Pocket {pocket.pocket_id} received joined metal label {metal_label!r}")
        observed_labels[pocket.pocket_id] = metal_label

    if sorted(observed_labels.values()) != ["Co", "Cu"]:
        raise AssertionError(f"Expected separate Co and Cu pocket labels, got {observed_labels}")

    import csv

    with inspection_csv.open("r", encoding="utf-8", newline="") as handle:
        row = next(
            (
                csv_row
                for csv_row in csv.DictReader(handle)
                if csv_row.get("structure_name") == structure_id
            ),
            None,
        )
    if row is None:
        raise AssertionError(f"Inspection CSV {inspection_csv} is missing row for {structure_id}")
    if row.get("metal_type") != "Co;Cu":
        raise AssertionError(f"Expected inspection CSV to contain Co;Cu metadata, got {row.get('metal_type')!r}")

    try:
        load_allowed_site_metal_labels(inspection_csv)
    except ValueError as exc:
        if "Missing required columns" not in str(exc):
            raise AssertionError(f"Inspection CSV was rejected for an unexpected reason: {exc}") from exc
    else:
        raise AssertionError("Structure-level inspection CSV was accepted as a site-level training summary CSV.")


def main() -> int:
    checks = (
        check_training_cli_help,
        check_test_eval_safety,
        check_loss_weight_validation,
        check_ec_group_weighting_config,
        check_cross_attention_config,
        check_ring_edge_cli_config,
        check_only_gvp_does_not_require_esm,
        check_graph_ring_edges_are_opt_in,
        check_colab_notebook_sweep_source,
        check_colab_generated_training_commands_parse,
        check_ring_environment_overrides,
        check_ec_group_weights_sum_per_group,
        check_ec_group_metric_aggregation,
        check_ec_group_id_batches_without_increment,
        check_conflicting_ec_group_metrics_are_skipped,
        check_bundle_cli_help,
        check_docs_do_not_use_broken_training_command,
        check_multi_metal_site_level_granularity,
    )
    for check in checks:
        try:
            check()
        except SkipCheck as exc:
            print(f"SKIP {check.__name__}: {exc}")
        else:
            print(f"PASS {check.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
