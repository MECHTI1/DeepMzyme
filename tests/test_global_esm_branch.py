"""Whole-protein ESMC branch for late fusion (F4), pocket ESM (F5) and the whole-protein-only control (C-G)."""
from pathlib import Path
import sys

import pytest
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_fixtures import default_ec_proteins, tiny_ec_argv, write_ec_dataset
from model import GlobalESMEncoder
from model_variants.factory import build_pocket_classifier


def _cache(paths, tmp_path):
    from embed_helpers.esmc_chain_means import build_from_residue_cache

    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("\n".join(sorted(paths["stems"].values())) + "\n")
    summary = build_from_residue_cache(allowlist, paths["structures"], paths["esm"], tmp_path / "chain_means",
                                       embedding_dim=8)
    return tmp_path / "chain_means", summary["manifest_sha256"]


def test_encoder_shapes_and_missing_attribute():
    encoder = GlobalESMEncoder(8, 16, 0.2)
    batch = Batch.from_data_list([Data(x_global_esm=torch.randn(1, 8)) for _ in range(3)])
    assert encoder(batch).shape == (3, 16)
    with pytest.raises(ValueError, match="x_global_esm"):
        encoder(Data())


@pytest.mark.parametrize(("architecture", "extra"), [("only_esm", {}), ("gvp", {"fusion_mode": "late_fusion"})])
def test_branch_widens_the_heads_and_keeps_shared_initialization(architecture, extra):
    common = dict(model_architecture=architecture, esm_dim=8, hidden_s=16, predict_metal=False, predict_ec=True,
                  n_ec=7, **extra)
    torch.manual_seed(0)
    plain = build_pocket_classifier(**common)
    torch.manual_seed(0)
    branched = build_pocket_classifier(**common, use_global_esm_branch=True, global_esm_dim=8)
    first_plain = next(m for m in plain.head_ec if isinstance(m, torch.nn.Linear))
    first_branched = next(m for m in branched.head_ec if isinstance(m, torch.nn.Linear))
    assert first_branched.in_features == first_plain.in_features + 16
    plain_state, branched_state = plain.state_dict(), branched.state_dict()
    shared = [key for key in plain_state if not key.startswith(("head_", "global_esm_encoder"))]
    assert shared and all(torch.equal(plain_state[key], branched_state[key]) for key in shared)
    assert any(key.startswith("global_esm_encoder.") for key in branched_state)


def test_factory_refusals_and_only_global():
    with pytest.raises(ValueError, match="not supported for only_gvp"):
        build_pocket_classifier(model_architecture="only_gvp", esm_dim=8, hidden_s=16, use_global_esm_branch=True)
    with pytest.raises(ValueError, match="late fusion"):
        build_pocket_classifier(model_architecture="gvp", esm_dim=8, hidden_s=16, fusion_mode="early_fusion",
                                use_global_esm_branch=True)
    with pytest.raises(ValueError, match="needs the whole-protein branch"):
        build_pocket_classifier(model_architecture="only_global", esm_dim=8, hidden_s=16)
    model = build_pocket_classifier(model_architecture="only_global", esm_dim=8, hidden_s=16, predict_metal=False,
                                    n_ec=7, use_global_esm_branch=True, global_esm_dim=8)
    graphs = [Data(x_global_esm=torch.randn(1, 8), y_ec=torch.tensor([2]), y_metal=torch.tensor([-1]),
                   ec_sample_weight=torch.ones(1), ec_group_id=torch.tensor([0]))]
    model.eval()
    outputs = model(Batch.from_data_list(graphs))
    assert outputs["logits_ec"].shape == (1, 7)


def test_config_refusals():
    from training.config import parse_args

    for argv in (
        ["--task", "ec", "--model-architecture", "only_global"],
        ["--task", "ec", "--use-global-esm-branch"],
        ["--task", "ec", "--model-architecture", "only_gvp", "--use-global-esm-branch",
         "--global-esm-cache-dir", "x", "--global-esm-manifest-sha256", "0" * 64],
    ):
        with pytest.raises(SystemExit):
            parse_args(argv)


@pytest.mark.parametrize("architecture", ["only_esm", "gvp", "only_global"])
def test_tiny_fits_attach_vectors_and_train(tmp_path, monkeypatch, architecture):
    import json

    from training.config import parse_args
    from training.run import run_training

    monkeypatch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(tmp_path / "cache"))
    paths = write_ec_dataset(tmp_path / "data", default_ec_proteins())
    cache, sha = _cache(paths, tmp_path)
    argv = tiny_ec_argv(paths, tmp_path / "runs", "--val-fraction", "0.3", "--ec-class-vocabulary", "fixed_seven",
                        "--checkpoint-rule", "terminal", "--use-global-esm-branch", "--global-esm-cache-dir",
                        str(cache), "--global-esm-manifest-sha256", sha, "--global-esm-dim", "8",
                        "--global-esm-standardize", "train_proteins", "--run-name", architecture)
    argv[argv.index("only_esm")] = architecture
    run_dir = run_training(parse_args(argv))
    summary = json.loads((run_dir / "run_config.json").read_text())["dataset_summary"]["global_esm"]
    assert summary["manifest_sha256"] == sha and summary["n_structures"] == 12
    assert summary["standardization"]["n_train_proteins"] > 0
    state = torch.load(run_dir / "terminal_model_checkpoint.pt", map_location="cpu", weights_only=False)
    assert any(key.startswith("global_esm_encoder.") for key in state["model_state_dict"])


def test_tiny_fit_refuses_a_wrong_manifest_hash(tmp_path, monkeypatch):
    from training.config import parse_args
    from training.global_esm_features import GlobalEsmCacheError
    from training.run import run_training

    monkeypatch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(tmp_path / "cache"))
    paths = write_ec_dataset(tmp_path / "data", default_ec_proteins())
    cache, _sha = _cache(paths, tmp_path)
    argv = tiny_ec_argv(paths, tmp_path / "runs", "--val-fraction", "0.3", "--use-global-esm-branch",
                        "--global-esm-cache-dir", str(cache), "--global-esm-manifest-sha256", "0" * 64,
                        "--global-esm-dim", "8", "--run-name", "bad_hash")
    with pytest.raises(GlobalEsmCacheError):
        run_training(parse_args(argv))


def test_old_configs_rebuild_without_the_branch():
    from export_validation_predictions import V3_OPTIONAL_CONFIG_DEFAULTS

    assert V3_OPTIONAL_CONFIG_DEFAULTS["use_global_esm_branch"] is False
