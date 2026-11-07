"""Amendment-2 trainer and embedding integration: the fp32 ESMC forward, TF32 policy and raw-RBF inputs of campaign
units, the F6 family label of exports (synthetic tensors only; no weights, no GPU, no data)."""
from __future__ import annotations

from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import embed_helpers.esmc_chain_means as chain_means  # noqa: E402
from training.global_esm_features import GlobalEsmCacheError  # noqa: E402

DIM = 8


class TinyESMC(torch.nn.Module):
    """Stands in for ESMC: bfloat16 parameters as ``ESMC.from_pretrained`` leaves them off CPU, a direct
    ``forward`` and a ``logits`` that must never be called on the fp32 path."""

    def __init__(self, *, flash: bool = False, dtype=torch.bfloat16, allow_logits: bool = False):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(64, DIM)
        self.transformer = torch.nn.Linear(DIM, DIM)
        self._use_flash_attn = flash
        self.allow_logits = allow_logits
        self.to(dtype)

    def encode(self, protein):
        return SimpleNamespace(sequence=torch.tensor([0, *[3 + ord(char) % 20 for char in protein.sequence], 2]))

    def forward(self, sequence_tokens):
        x = self.embed(sequence_tokens)
        h = self.transformer(x)
        return SimpleNamespace(embeddings=h, hidden_states=torch.stack([x, h], dim=0))

    def logits(self, encoded, config):
        if not self.allow_logits:
            raise AssertionError("ESMC.logits is bypassed on the fp32 path")
        with torch.no_grad():
            output = self.forward(sequence_tokens=encoded.sequence.unsqueeze(0))
        return SimpleNamespace(embeddings=output.embeddings, hidden_states=output.hidden_states)


SDK = (TinyESMC, lambda sequence: SimpleNamespace(sequence=sequence), lambda **config: config)


def _means(model, precision):
    return chain_means.forward_layer_means(model, SDK, "MKTAYIAK", n_tokens=8, layer_index=0, embedding_dim=DIM,
                                           forward_precision=precision)


def test_the_fp32_path_casts_bypasses_logits_and_emits_float32():
    model = TinyESMC()
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    record = chain_means.configure_fp32_precision(model, "cpu")
    assert record["parameter_dtypes"] == ["float32"]
    assert record["precision_policy"]["cuda_matmul_allow_tf32"] is False
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
    means = _means(model, chain_means.FORWARD_PRECISION_FP32)
    assert means.emitted_dtypes == {"final": "float32", "hidden": "float32"} and means.n_layers == 2
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):  # an outer autocast is switched off inside
        assert _means(model, chain_means.FORWARD_PRECISION_FP32).emitted_dtypes["final"] == "float32"


def test_the_fp32_path_refuses_flash_attention_active_autocast_and_other_dtypes(monkeypatch):
    with pytest.raises(GlobalEsmCacheError, match="refuses flash attention"):
        chain_means.configure_fp32_precision(TinyESMC(flash=True), "cpu")
    model = TinyESMC()
    chain_means.configure_fp32_precision(model, "cpu")
    monkeypatch.setattr(chain_means, "_autocast_active", lambda: True)
    with pytest.raises(GlobalEsmCacheError, match="Autocast was active"):
        _means(model, chain_means.FORWARD_PRECISION_FP32)
    monkeypatch.undo()
    assert not torch.is_autocast_enabled()
    half = TinyESMC()  # never cast: the fp32 path refuses its bfloat16 outputs
    with pytest.raises(GlobalEsmCacheError, match="not float32"):
        _means(half, chain_means.FORWARD_PRECISION_FP32)


def test_the_fp32_path_on_cuda_refuses_without_the_runtime_precision_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "ec_clean30.runtime_precision", None)
    model = TinyESMC()
    with pytest.raises(GlobalEsmCacheError, match="needs ec_clean30.runtime_precision"):
        chain_means.configure_fp32_precision(model, "cuda")
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}  # refused before any change
    # On CPU the local record stays, marked; attach refuses it because it does not validate.
    record = chain_means.configure_fp32_precision(model, "cpu")
    assert "not available" in record["precision_policy"]["recorded_by"]


def test_on_cpu_the_fp32_and_sdk_paths_give_bit_identical_means():
    sdk_model = TinyESMC(dtype=torch.float32, allow_logits=True)
    fp32_model = TinyESMC(dtype=torch.float32)
    chain_means.configure_fp32_precision(fp32_model, "cpu")
    first = _means(sdk_model, chain_means.FORWARD_PRECISION_SDK)
    second = _means(fp32_model, chain_means.FORWARD_PRECISION_FP32)
    assert torch.equal(first.final, second.final) and torch.equal(first.block, second.block)


def test_runtime_records_change_only_on_the_fp32_path():
    model = TinyESMC(dtype=torch.float32)
    sdk = chain_means.describe_runtime(model, model_name="esmc_300m", requested_device="cpu")
    assert "forward_precision" not in sdk and "precision" not in sdk
    fp32 = chain_means.describe_runtime(model, model_name="esmc_300m", requested_device="cpu",
                                        forward_precision=chain_means.FORWARD_PRECISION_FP32,
                                        precision={"parameter_dtypes": ["float32"]})
    assert fp32["forward_precision"] == "fp32" and fp32["autocast_dtype"] is None
    assert {key: value for key, value in fp32.items() if key not in ("forward_precision", "forward_call", "precision",
                                                                     "autocast_dtype")} == {
        key: value for key, value in sdk.items() if key != "autocast_dtype"}
    parser = chain_means.build_arg_parser()
    args = parser.parse_args(["sequences", "--fasta", "a.fa", "--sequence-map", "m.tsv", "--out", "o", "--layer-index",
                              "0", "--device", "cpu", "--forward-precision", "fp32"])
    assert args.forward_precision == "fp32"
    assert parser.parse_args(["compare", "--cache-a", "a", "--manifest-a-sha256", "x", "--cache-b", "b",
                              "--manifest-b-sha256", "y", "--ids", "i", "--kind", "sequence_mean"]).kind == \
        "sequence_mean"


# ---------------------------------------------------------------------------
# run.py: TF32 policy and raw-RBF inputs of campaign units


def test_campaign_units_record_the_tf32_policy(monkeypatch):
    from training import run

    import ec_clean30

    monkeypatch.setitem(sys.modules, "ec_clean30.runtime_precision", None)
    # An earlier import of the real module binds it on the package; ``from ec_clean30 import ...`` would find it.
    monkeypatch.delattr(ec_clean30, "runtime_precision", raising=False)
    assert run.campaign_precision_policy("cpu") == {"device": "cpu", "status": "runtime_precision_unavailable"}
    with pytest.raises(ValueError, match="needs ec_clean30.runtime_precision"):
        run.campaign_precision_policy("cuda")
    calls = []
    stub = types.ModuleType("ec_clean30.runtime_precision")

    def configure_and_record(device):
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("no GPU")
        return {"device": device, "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}

    stub.configure_and_record = configure_and_record
    monkeypatch.setitem(sys.modules, "ec_clean30.runtime_precision", stub)
    import ec_clean30

    monkeypatch.setattr(ec_clean30, "runtime_precision", stub, raising=False)
    assert run.campaign_precision_policy("cpu")["cudnn_allow_tf32"] is False
    with pytest.raises(ValueError, match="does not validate"):
        run.campaign_precision_policy("cuda")
    assert calls == ["cpu", "cuda"]


def _graph(with_distances: bool):
    from torch_geometric.data import Data

    data = Data(x=torch.zeros(3, 2))
    if with_distances:
        data.x_dist_raw = torch.tensor([[1.0], [999.0], [4.0]])
        data.edge_dist_raw = torch.tensor([[2.0], [3.0]])
    return data


def test_raw_rbf_units_require_the_raw_distance_tensors():
    from training import run
    from training.graph_dataset import compute_feature_normalization_stats

    config = SimpleNamespace(node_rbf_use_raw_distances=True, edge_rbf_use_raw_distances=True,
                             campaign_run_identity='{"unit_id": "u"}')
    graphs = [_graph(True)]
    record = run.rbf_distance_inputs(config, graphs, compute_feature_normalization_stats(graphs, clamp_value=5.0))
    assert record["node"] == "x_dist_raw_raw" and record["edge"] == "edge_dist_raw_raw" and record["missing"] == []
    assert not hasattr(graphs[0], "x_dist_raw_raw")  # checked on a copy; no training graph is changed
    bare = [_graph(False)]
    with pytest.raises(ValueError, match="lack \\['x_dist_raw_raw', 'edge_dist_raw_raw'\\]"):
        run.rbf_distance_inputs(config, bare, compute_feature_normalization_stats(bare, clamp_value=5.0))
    plain = SimpleNamespace(**{**vars(config), "campaign_run_identity": None})
    assert run.rbf_distance_inputs(plain, bare, None)["missing"] == ["x_dist_raw_raw", "edge_dist_raw_raw"]
    assert run.rbf_distance_inputs(SimpleNamespace(node_rbf_use_raw_distances=False, edge_rbf_use_raw_distances=False,
                                                   campaign_run_identity="x"), graphs, None) is None


def test_exports_label_f6_and_keep_historical_identities():
    from training.ec_prediction_export import _model_identity, campaign_family

    f5 = {"model_architecture": "only_esm", "use_global_esm_branch": True, "global_esm_layer": "final"}
    f6 = {**f5, "global_esm_branch_init": "pretrained_frozen", "global_esm_pretrain_corpus_sha256": "c" * 64}
    assert campaign_family(f5) == "F5" and campaign_family(f6) == "F6"
    assert campaign_family({**f5, "global_esm_branch_init": "random"}) == "F5"
    assert "global_esm_branch_init" not in _model_identity(f5)
    assert _model_identity(f6)["global_esm_pretrain_corpus_sha256"] == "c" * 64


def test_the_runner_registers_the_amendment2_actions_and_keeps_its_pinned_functions():
    import importlib.util
    import json

    from ec_clean30 import freeze

    runner_path = ROOT / "scripts" / "run_ec_clean30_campaign.py"
    spec = importlib.util.spec_from_file_location("amendment2_runner_under_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    parser = runner.build_parser()
    assert parser.parse_args(["plan-v2", "--fold", "0"]).handler is runner.action_plan_v2
    assert parser.parse_args(["pretrain-corpus", "verify", "--view", "v"]).handler.__name__ == "action_pretrain_corpus"
    args = parser.parse_args(["futility", "--fold", "0", "--step-chain", "c", "--plan", "p", "--runs", "r",
                              "--components", "o", "--freeze1-sha256", "f" * 64, "--freeze1-amendment", "a",
                              "--clean-r1-inner", "x", "--clean-r2-inner", "y", "--clean-sanity-record", "s",
                              "--out", "o.json"])
    assert args.handler.__name__ == "action_futility"
    recorded = json.loads((ROOT / "docs" / "campaigns" / "ec_class_vs_clean30_2026-10" / "freeze1.json").read_text())
    pins = None
    stack = [recorded]
    while stack and pins is None:
        item = stack.pop()
        if isinstance(item, dict):
            pins = item.get("pinned_runner_function_sha256")
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    live, problems = freeze.runner_function_sha256(runner_path.read_text(encoding="utf-8"))
    assert not problems and pins and live == pins
