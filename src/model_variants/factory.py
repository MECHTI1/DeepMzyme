from __future__ import annotations

from typing import Any

MODEL_ARCHITECTURE_CHOICES = (
    "gvp",
    "only_esm",
    "only_gvp",
    "simple_gnn_esm",
)

MODEL_ARCHITECTURE_ALIASES = {
    "gvp": "gvp",
    "gvp_esm": "gvp",
    "gvp+esm": "gvp",
    "only_esm": "only_esm",
    "esm_only": "only_esm",
    "only-gvp": "only_gvp",
    "only_gvp": "only_gvp",
    "gvp_only": "only_gvp",
    "simple_gnn_esm": "simple_gnn_esm",
    "simplegnn+esm": "simple_gnn_esm",
    "simplegnn_esm": "simple_gnn_esm",
}

FUSION_MODE_CHOICES = (
    "late_fusion",
    "early_fusion",
    "node_level_late_fusion",
    "hybrid",
    "cross_modal_attention",
)

FUSION_MODE_ALIASES = {
    "gated": "late_fusion",
    "late": "late_fusion",
    "late_fusion": "late_fusion",
    "early": "early_fusion",
    "early_fusion": "early_fusion",
    "node_level_late": "node_level_late_fusion",
    "node_level_late_fusion": "node_level_late_fusion",
    "hybrid": "hybrid",
    "cross_attention": "cross_modal_attention",
    "cross_modal_attention": "cross_modal_attention",
}


def normalize_model_architecture(raw_name: str) -> str:
    normalized = raw_name.strip().lower().replace("-", "_")
    try:
        return MODEL_ARCHITECTURE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model architecture {raw_name!r}. "
            f"Expected one of {sorted(MODEL_ARCHITECTURE_ALIASES)}."
        ) from exc


def normalize_fusion_mode(raw_name: str) -> str:
    normalized = raw_name.strip().lower().replace("-", "_")
    try:
        return FUSION_MODE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported fusion mode {raw_name!r}. "
            f"Expected one of {sorted(FUSION_MODE_ALIASES)}."
        ) from exc


def _apply_fusion_defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(kwargs)
    fusion_mode = normalize_fusion_mode(str(resolved.get("fusion_mode", "late_fusion")))
    resolved["fusion_mode"] = fusion_mode
    resolved_use_esm_branch = bool(resolved.get("use_esm_branch", True))
    resolved_use_early_esm = bool(resolved.get("use_early_esm", False))

    if fusion_mode == "early_fusion":
        resolved_use_esm_branch = False
        resolved_use_early_esm = True
    elif fusion_mode == "hybrid":
        resolved_use_esm_branch = True
        resolved_use_early_esm = True
    elif fusion_mode in {"late_fusion", "node_level_late_fusion", "cross_modal_attention"}:
        resolved_use_esm_branch = True

    resolved["use_esm_branch"] = resolved_use_esm_branch
    resolved["use_early_esm"] = resolved_use_early_esm
    return resolved


def build_pocket_classifier(
    *,
    model_architecture: str,
    **kwargs: Any,
):
    architecture = normalize_model_architecture(model_architecture)
    resolved_kwargs = _apply_fusion_defaults(kwargs)

    if architecture == "gvp":
        from model import GVPPocketClassifier

        return GVPPocketClassifier(**resolved_kwargs)
    if architecture == "only_gvp":
        from model import GVPPocketClassifier

        resolved_kwargs["use_esm_branch"] = False
        resolved_kwargs["use_early_esm"] = False
        resolved_kwargs["fusion_mode"] = "late_fusion"
        return GVPPocketClassifier(**resolved_kwargs)
    if architecture == "only_esm":
        from model_variants.models import OnlyESMPocketClassifier

        resolved_kwargs.pop("fusion_mode", None)
        resolved_kwargs.pop("use_esm_branch", None)
        resolved_kwargs.pop("use_early_esm", None)
        resolved_kwargs.pop("cross_attention_layers", None)
        resolved_kwargs.pop("cross_attention_heads", None)
        resolved_kwargs.pop("cross_attention_dropout", None)
        resolved_kwargs.pop("cross_attention_neighborhood", None)
        resolved_kwargs.pop("cross_attention_bidirectional", None)
        resolved_kwargs.pop("early_esm_dim", None)
        resolved_kwargs.pop("early_esm_dropout", None)
        resolved_kwargs.pop("early_esm_raw", None)
        resolved_kwargs.pop("early_esm_scope", None)
        resolved_kwargs.pop("node_rbf_sigma", None)
        resolved_kwargs.pop("edge_rbf_sigma", None)
        resolved_kwargs.pop("node_rbf_use_raw_distances", None)
        resolved_kwargs.pop("hidden_v", None)
        resolved_kwargs.pop("edge_hidden", None)
        resolved_kwargs.pop("n_layers", None)
        return OnlyESMPocketClassifier(**resolved_kwargs)
    if architecture == "simple_gnn_esm":
        from model_variants.models import SimpleGNNPocketClassifier

        resolved_kwargs.pop("hidden_v", None)
        resolved_kwargs.pop("node_rbf_use_raw_distances", None)
        return SimpleGNNPocketClassifier(**resolved_kwargs)
    raise AssertionError(f"Unhandled model architecture: {architecture!r}")
