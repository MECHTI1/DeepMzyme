from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

from data_structures import (
    AA_ORDER,
    DEFAULT_SITE_LIGAND_ANGLE_FEATURE_DIM,
    EDGE_SOURCE_TYPES,
    INTERACTION_SUMMARIES_OPTIONAL_WITH_RING,
    STRUCTURAL_READOUT_SCOPE_CHOICES,
)
from data_structures import MISSING_CLASS_LABEL
from label_schemes import N_EC_CLASSES, N_METAL_CLASSES
from metal_objectives import metal_loss_with_optional_collapsed4

VALID_FUSION_MODES = {
    "late_fusion",
    "early_fusion",
    "node_level_late_fusion",
    "hybrid",
    "cross_modal_attention",
}
VALID_TASK_LOSS_WEIGHTING_MODES = {"fixed", "uncertainty"}
DEFAULT_SITE_FEATURE_INPUT_DIM = 4
DEFAULT_SITE_FEATURE_DIM = 32
DEFAULT_NODE_RESCHEM_DIM = len(AA_ORDER) + 5
DEFAULT_NODE_HYDROPHOBICITY_DIM = 1
DEFAULT_NODE_ROLE_DIM = 2
DEFAULT_NODE_MISC_DIM = 1
DEFAULT_NODE_BURIAL_INPUT_DIM = 1
DEFAULT_NODE_BURIAL_LATENT_DIM = 4
DEFAULT_NODE_ELECTROSTATICS_DIM = 2
DEFAULT_NODE_DISTANCE_FEATURE_COUNT = 3
VALID_STRUCTURAL_READOUT_SCOPES = set(STRUCTURAL_READOUT_SCOPE_CHOICES) - {"auto"}


class TaskLossWeighter(nn.Module):
    def __init__(
        self,
        *,
        mode: str = "fixed",
        metal_loss_weight: float = 1.0,
        ec_loss_weight: float = 1.0,
        predict_metal: bool = True,
        predict_ec: bool = True,
    ):
        super().__init__()
        if mode not in VALID_TASK_LOSS_WEIGHTING_MODES:
            raise ValueError(f"Unsupported task loss weighting mode {mode!r}.")
        self.mode = str(mode)
        self.metal_loss_weight = float(metal_loss_weight)
        self.ec_loss_weight = float(ec_loss_weight)
        if self.metal_loss_weight < 0.0:
            raise ValueError(f"metal_loss_weight must be non-negative, got {self.metal_loss_weight}.")
        if self.ec_loss_weight < 0.0:
            raise ValueError(f"ec_loss_weight must be non-negative, got {self.ec_loss_weight}.")
        self.use_uncertainty_weighting = self.mode == "uncertainty" and bool(predict_metal) and bool(predict_ec)
        self.register_parameter(
            "metal_log_variance",
            nn.Parameter(torch.zeros(())) if self.use_uncertainty_weighting else None,
        )
        self.register_parameter(
            "ec_log_variance",
            nn.Parameter(torch.zeros(())) if self.use_uncertainty_weighting else None,
        )

    def _base_weight(self, task_name: str) -> float:
        if task_name == "metal":
            return self.metal_loss_weight
        if task_name == "ec":
            return self.ec_loss_weight
        raise ValueError(f"Unsupported task loss name {task_name!r}.")

    def _log_variance(self, task_name: str) -> Tensor:
        if task_name == "metal" and self.metal_log_variance is not None:
            return self.metal_log_variance
        if task_name == "ec" and self.ec_log_variance is not None:
            return self.ec_log_variance
        raise ValueError(f"Missing uncertainty parameter for task {task_name!r}.")

    def forward(self, task_losses: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        if not task_losses:
            raise ValueError("TaskLossWeighter received no task losses.")
        weighted_losses = []
        diagnostics: dict[str, Tensor] = {}
        for task_name, task_loss in task_losses.items():
            base_weight = self._base_weight(task_name)
            diagnostics[f"{task_name}_loss_raw"] = task_loss.detach()
            if base_weight == 0.0:
                diagnostics[f"{task_name}_loss_scale"] = task_loss.new_zeros(())
                continue
            if self.use_uncertainty_weighting:
                log_variance = self._log_variance(task_name)
                precision = torch.exp(-log_variance)
                weighted_losses.append(base_weight * (precision * task_loss + log_variance))
                diagnostics[f"{task_name}_loss_scale"] = (base_weight * precision).detach()
                diagnostics[f"{task_name}_loss_log_variance"] = log_variance.detach()
            else:
                weighted_losses.append(task_loss * base_weight)
                diagnostics[f"{task_name}_loss_scale"] = task_loss.new_tensor(base_weight)
        if not weighted_losses:
            raise ValueError("All task losses were disabled by zero task weights.")
        diagnostics["loss"] = torch.stack(weighted_losses).sum()
        return diagnostics["loss"], diagnostics


class RBFExpansion(nn.Module):
    def __init__(self, n_rbf: int = 16, d_min: float = 0.0, d_max: float = 12.0, sigma: float | None = None):
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        if sigma is None:
            sigma = (d_max - d_min) / n_rbf
        self.sigma = float(sigma)
        self.gamma = 1.0 / (self.sigma * self.sigma + 1e-8)

    def forward(self, d: Tensor) -> Tensor:
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class TinyFeatureGroupMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class NodeScalarEncoder(nn.Module):
    def __init__(
        self,
        n_rbf: int = 16,
        out_dim: int = 128,
        distance_sigma: float = 0.75,
        extra_scalar_dim: int = 0,
        reschem_dim: int = DEFAULT_NODE_RESCHEM_DIM,
        hydrophobicity_dim: int = DEFAULT_NODE_HYDROPHOBICITY_DIM,
        role_dim: int = DEFAULT_NODE_ROLE_DIM,
        misc_dim: int = DEFAULT_NODE_MISC_DIM,
        burial_input_dim: int = DEFAULT_NODE_BURIAL_INPUT_DIM,
        burial_latent_dim: int = DEFAULT_NODE_BURIAL_LATENT_DIM,
        electrostatics_dim: int = DEFAULT_NODE_ELECTROSTATICS_DIM,
        distance_feature_count: int = DEFAULT_NODE_DISTANCE_FEATURE_COUNT,
    ):
        super().__init__()
        self.reschem_dim = int(reschem_dim)
        self.hydrophobicity_dim = int(hydrophobicity_dim)
        self.role_dim = int(role_dim)
        self.misc_dim = int(misc_dim)
        self.burial_input_dim = int(burial_input_dim)
        self.burial_latent_dim = int(burial_latent_dim)
        self.electrostatics_dim = int(electrostatics_dim)
        self.distance_feature_count = int(distance_feature_count)
        self.dist_rbf = RBFExpansion(n_rbf=n_rbf, d_min=0.0, d_max=12.0, sigma=distance_sigma)
        self.burial_encoder = TinyFeatureGroupMLP(
            in_dim=self.burial_input_dim,
            hidden_dim=max(1, self.burial_latent_dim),
            out_dim=self.burial_latent_dim,
        )

        # Keep the heuristic q1*q2/r-style proxy and the PROPKA-derived
        # dpka_titr contribution as separate scalars rather than summing them.
        self.base_in_dim = (
            self.reschem_dim
            + self.hydrophobicity_dim
            + self.role_dim
            + self.misc_dim
            + self.burial_latent_dim
            + self.electrostatics_dim
            + self.distance_feature_count * int(n_rbf)
        )
        self.extra_scalar_dim = int(extra_scalar_dim)
        self.in_dim = self.base_in_dim + self.extra_scalar_dim
        self.out_proj = nn.Sequential(
            nn.Linear(self.in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        x_reschem: Tensor,
        hydrophobicity_kd: Tensor,
        x_role: Tensor,
        x_dist_raw: Tensor,
        x_misc: Tensor,
        x_env_burial: Tensor,
        x_env_electrostatics: Tensor,
        extra_scalar_features: Tensor | None = None,
    ) -> Tensor:
        d_rbf = self.dist_rbf(x_dist_raw).flatten(start_dim=1)
        burial_latent = self.burial_encoder(x_env_burial)
        feature_groups = [
            x_reschem,
            hydrophobicity_kd,
            x_role,
            x_misc,
            burial_latent,
            x_env_electrostatics,
            d_rbf,
        ]
        if extra_scalar_features is not None:
            feature_groups.append(extra_scalar_features)
        x = torch.cat(
            feature_groups,
            dim=-1,
        )
        return self.out_proj(x)


class EarlyESMEncoder(nn.Module):
    def __init__(self, esm_input_dim: int, early_esm_dim: int, early_esm_dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            # Recommended early ESMC injection uses a bottleneck projection such as
            # ESMC_dim -> 32/64 before residue-level message passing.
            nn.Linear(esm_input_dim, early_esm_dim),
            nn.ReLU(),
            nn.Dropout(early_esm_dropout),
            nn.Linear(early_esm_dim, early_esm_dim),
            nn.ReLU(),
        )

    def forward(self, x_esm: Tensor) -> Tensor:
        return self.net(x_esm)


class AttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, in_dim // 2)
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, batch: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=x.device)
            x = x[mask]
            batch = batch[mask]
        logits = self.score(x).squeeze(-1)
        weights = softmax(logits, batch, num_nodes=batch_size)
        return global_add_pool(x * weights.unsqueeze(-1), batch, size=batch_size)


class ESMGraphEncoder(nn.Module):
    def __init__(self, esm_dim: int, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.attn_pool = AttentionPool(proj_dim)

    def forward(self, x_esm: Tensor, batch: Tensor, mask: Tensor | None = None) -> Tensor:
        z = self.esm_proj(x_esm)
        z_mean = masked_global_mean_pool(z, batch, mask)
        z_attn = self.attn_pool(z, batch, mask)
        return torch.cat([z_mean, z_attn], dim=-1)


def shell_mask_from_roles(x_role: Tensor, scope: str) -> Tensor:
    if scope == "all":
        return torch.ones(x_role.size(0), dtype=torch.bool, device=x_role.device)
    if scope == "first_shell":
        return x_role[:, 0] > 0.5
    if scope == "first_second_shell":
        return (x_role[:, 0] > 0.5) | (x_role[:, 1] > 0.5)
    raise ValueError(f"Unsupported shell scope {scope!r}.")


def ensure_nonempty_pool_mask(mask: Tensor, batch: Tensor) -> Tensor:
    mask = mask.to(dtype=torch.bool, device=batch.device).clone()
    if batch.numel() == 0:
        return mask
    batch_size = int(batch.max().item()) + 1
    counts = global_add_pool(mask.float().unsqueeze(-1), batch, size=batch_size).view(-1)
    for graph_idx in torch.nonzero(counts <= 0.0, as_tuple=False).view(-1).tolist():
        node_indices = torch.nonzero(batch == int(graph_idx), as_tuple=False).view(-1)
        if node_indices.numel() > 0:
            mask[node_indices[0]] = True
    return mask


def residue_node_mask(data: Data) -> Tensor:
    if hasattr(data, "residue_node_mask"):
        return data.residue_node_mask.to(dtype=torch.bool, device=data.batch.device)
    return torch.ones(data.batch.size(0), dtype=torch.bool, device=data.batch.device)


def metal_node_mask(data: Data) -> Tensor:
    if hasattr(data, "metal_node_mask"):
        return data.metal_node_mask.to(dtype=torch.bool, device=data.batch.device)
    return torch.zeros(data.batch.size(0), dtype=torch.bool, device=data.batch.device)


def metal_distance_pool_mask(data: Data, cutoff: float, base_mask: Tensor | None = None) -> Tensor | None:
    cutoff = float(cutoff)
    if cutoff <= 0.0:
        return ensure_nonempty_pool_mask(base_mask, data.batch) if base_mask is not None else None
    distance_features = (
        data.x_dist_raw_raw.float()
        if hasattr(data, "x_dist_raw_raw")
        else data.x_dist_raw.float()
    )
    ca_to_metal = distance_features if distance_features.ndim == 1 else distance_features[:, 0]
    mask = ca_to_metal <= cutoff
    if base_mask is not None:
        mask = mask & base_mask.to(dtype=torch.bool, device=mask.device)
    return ensure_nonempty_pool_mask(mask, data.batch)


def masked_global_mean_pool(x: Tensor, batch: Tensor, mask: Tensor | None = None) -> Tensor:
    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=x.device)
        x = x[mask]
        batch = batch[mask]
    return global_mean_pool(x, batch, size=batch_size)


def pool_graph_states(x: Tensor, batch: Tensor, attn_pool: AttentionPool, mask: Tensor | None = None) -> Tensor:
    return torch.cat([masked_global_mean_pool(x, batch, mask), attn_pool(x, batch, mask)], dim=-1)


class LocalizedCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float, *, bidirectional: bool):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.struct_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.struct_norm = nn.LayerNorm(hidden_dim)
        self.struct_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.struct_ff_norm = nn.LayerNorm(hidden_dim)
        if self.bidirectional:
            self.esm_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.esm_norm = nn.LayerNorm(hidden_dim)
            self.esm_ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.esm_ff_norm = nn.LayerNorm(hidden_dim)
        else:
            self.esm_attn = None
            self.esm_norm = None
            self.esm_ff = None
            self.esm_ff_norm = None

    def _residual_update(self, x: Tensor, attn_out: Tensor, norm: nn.LayerNorm, ff: nn.Sequential, ff_norm: nn.LayerNorm) -> Tensor:
        x = norm(x + attn_out)
        return ff_norm(x + ff(x))

    def forward(
        self,
        struct_states: Tensor,
        esm_states: Tensor,
        batch: Tensor,
        active_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        updated_struct = struct_states.clone()
        updated_esm = esm_states.clone()
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        for graph_idx in range(n_graphs):
            graph_mask = batch == graph_idx
            graph_indices = torch.nonzero(graph_mask, as_tuple=False).flatten()
            if graph_indices.numel() == 0:
                continue
            graph_active = active_mask[graph_mask]
            if not bool(graph_active.any().item()):
                continue
            active_indices = graph_indices[graph_active]
            struct_local = updated_struct[active_indices].unsqueeze(0)
            esm_local = updated_esm[active_indices].unsqueeze(0)
            struct_attn_out, _ = self.struct_attn(struct_local, esm_local, esm_local, need_weights=False)
            updated_struct[active_indices] = self._residual_update(
                struct_local,
                struct_attn_out,
                self.struct_norm,
                self.struct_ff,
                self.struct_ff_norm,
            ).squeeze(0)
            if self.bidirectional and self.esm_attn is not None and self.esm_norm is not None and self.esm_ff is not None and self.esm_ff_norm is not None:
                struct_local_updated = updated_struct[active_indices].unsqueeze(0)
                esm_local_updated = updated_esm[active_indices].unsqueeze(0)
                esm_attn_out, _ = self.esm_attn(esm_local_updated, struct_local_updated, struct_local_updated, need_weights=False)
                updated_esm[active_indices] = self._residual_update(
                    esm_local_updated,
                    esm_attn_out,
                    self.esm_norm,
                    self.esm_ff,
                    self.esm_ff_norm,
                ).squeeze(0)
        return updated_struct, updated_esm


class EdgeScalarEncoder(nn.Module):
    def __init__(self, n_rbf: int = 16, out_dim: int = 64, distance_sigma: float = 0.75):
        super().__init__()
        self.dist_rbf = RBFExpansion(n_rbf=n_rbf, d_min=0.0, d_max=12.0, sigma=distance_sigma)
        in_dim = 2 * n_rbf + 2 + len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING) + len(EDGE_SOURCE_TYPES)
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        edge_dist_raw: Tensor,
        edge_seqsep: Tensor,
        edge_same_chain: Tensor,
        edge_interaction_type: Tensor,
        edge_source_type: Tensor,
    ) -> Tensor:
        d_rbf = self.dist_rbf(edge_dist_raw).flatten(start_dim=1)
        x = torch.cat(
            [d_rbf, edge_seqsep, edge_same_chain, edge_interaction_type, edge_source_type],
            dim=-1,
        )
        return self.out_proj(x)


def vector_norm(v: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))


class SimpleGVP(nn.Module):
    def __init__(self, s_in: int, v_in: int, s_out: int, v_out: int):
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            nn.Linear(s_in + v_in, s_out),
            nn.SiLU(),
            nn.Linear(s_out, s_out),
        )
        self.vector_linear = nn.Linear(v_in, v_out, bias=False)
        self.vector_gate = nn.Linear(s_out, v_out)

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        v_norm = vector_norm(v)
        s_cat = torch.cat([s, v_norm], dim=-1)
        s_out = self.scalar_mlp(s_cat)

        v_t = v.transpose(1, 2)
        v_proj = self.vector_linear(v_t).transpose(1, 2)
        gate = torch.sigmoid(self.vector_gate(s_out)).unsqueeze(-1)
        v_out = v_proj * gate
        return s_out, v_out


class SimpleGVPLayer(nn.Module):
    def __init__(self, s_dim: int, v_dim: int, e_dim: int, *, normalize_message_aggregation: bool = False):
        super().__init__()
        self.normalize_message_aggregation = bool(normalize_message_aggregation)

        self.message_gvp = SimpleGVP(
            s_in=2 * s_dim + e_dim + 1,
            v_in=2 * v_dim + 1,
            s_out=s_dim,
            v_out=v_dim,
        )
        self.update_gvp = SimpleGVP(
            s_in=2 * s_dim,
            v_in=2 * v_dim,
            s_out=s_dim,
            v_out=v_dim,
        )
        self.norm_s = nn.LayerNorm(s_dim)

    def forward(self, s: Tensor, v: Tensor, edge_index: Tensor, edge_s: Tensor, edge_v: Tensor) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index

        s_src = s[src]
        s_dst = s[dst]
        v_src = v[src]
        v_dst = v[dst]

        edge_len = vector_norm(edge_v)
        m_s_in = torch.cat([s_src, s_dst, edge_s, edge_len], dim=-1)
        m_v_in = torch.cat([v_src, v_dst, edge_v], dim=1)

        m_s, m_v = self.message_gvp(m_s_in, m_v_in)

        agg_s = torch.zeros_like(s)
        agg_s.index_add_(0, dst, m_s)

        agg_v = torch.zeros_like(v)
        agg_v.index_add_(0, dst, m_v)

        if self.normalize_message_aggregation and dst.numel() > 0:
            degree = torch.bincount(dst, minlength=s.size(0)).clamp_min(1).to(device=s.device)
            agg_s = agg_s / degree.to(dtype=agg_s.dtype).unsqueeze(-1)
            agg_v = agg_v / degree.to(dtype=agg_v.dtype).view(-1, 1, 1)

        u_s_in = torch.cat([s, agg_s], dim=-1)
        u_v_in = torch.cat([v, agg_v], dim=1)
        ds, dv = self.update_gvp(u_s_in, u_v_in)

        s_out = self.norm_s(s + ds)
        v_out = v + dv
        return s_out, v_out


def build_classifier_head(
    *,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_linear_layers: int,
    dropout: float = 0.2,
) -> nn.Sequential:
    if n_linear_layers < 1:
        raise ValueError(f"Classifier head requires at least 1 linear layer, got {n_linear_layers}.")
    if not 0.0 <= float(dropout) <= 1.0:
        raise ValueError(f"Classifier head dropout must be in [0, 1], got {dropout}.")
    if n_linear_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(n_linear_layers - 1):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    if embeddings.ndim != 2:
        raise ValueError(
            f"Contrastive loss expects 2D embeddings, got shape {tuple(embeddings.shape)}."
        )
    if labels.ndim != 1 or labels.size(0) != embeddings.size(0):
        raise ValueError(
            "Contrastive loss expects one label per embedding. "
            f"Got embeddings={tuple(embeddings.shape)} labels={tuple(labels.shape)}."
        )
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())

    normalized = F.normalize(embeddings, dim=-1)
    logits = torch.matmul(normalized, normalized.transpose(0, 1)) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positive_mask = same_label & ~self_mask
    valid_anchors = positive_mask.any(dim=1)
    if not bool(valid_anchors.any().item()):
        return embeddings.new_zeros(())

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return (-mean_log_prob_pos[valid_anchors]).mean()


class GVPPocketClassifier(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        hidden_s: int = 128,
        hidden_v: int = 16,
        edge_hidden: int = 64,
        n_layers: int = 4,
        n_metal: int = N_METAL_CLASSES,
        n_ec: int = N_EC_CLASSES,
        esm_fusion_dim: int = 128,
        head_mlp_layers: int = 2,
        head_mlp_dropout: float = 0.2,
        esm_graph_encoder_dropout: float = 0.1,
        node_rbf_sigma: float = 0.75,
        edge_rbf_sigma: float = 0.75,
        node_rbf_use_raw_distances: bool = False,
        joint_loss_weighting: str = "fixed",
        metal_loss_weight: float = 1.0,
        ec_loss_weight: float = 1.0,
        metal_class_weights: Optional[Tensor] = None,
        metal_collapsed4_class_weights: Optional[Tensor] = None,
        ec_class_weights: Optional[Tensor] = None,
        metal_loss_function: str = "cross_entropy",
        metal_focal_gamma: float = 2.0,
        metal_label_smoothing: float = 0.0,
        metal_collapsed_loss_weight: float = 0.0,
        predict_metal: bool = True,
        predict_ec: bool = True,
        use_esm_branch: bool = True,
        fusion_mode: str = "late_fusion",
        cross_attention_layers: int = 1,
        cross_attention_heads: int = 4,
        cross_attention_dropout: float = 0.1,
        cross_attention_neighborhood: str = "all",
        cross_attention_bidirectional: bool = False,
        use_early_esm: bool = False,
        early_esm_dim: int = 32,
        early_esm_dropout: float = 0.2,
        early_esm_raw: bool = False,
        early_esm_scope: str = "all",
        ec_contrastive_weight: float = 0.0,
        ec_contrastive_temperature: float = 0.1,
        normalize_message_aggregation: bool = False,
        site_feature_dim: int = DEFAULT_SITE_FEATURE_DIM,
        classifier_pool_distance_cutoff: float = 0.0,
        structural_readout_scope: str = "residue_only",
        use_node_type_embedding: bool = False,
        use_site_angle_features: bool = False,
    ):
        super().__init__()
        # Current supervised targets:
        # - EC head: first EC digit only, mapped from EC 1..7 to class ids 0..6.
        # - Metal head: class count follows the active metal label scheme.

        self.use_early_esm = bool(use_early_esm)
        self.early_esm_raw = bool(early_esm_raw)
        self.early_esm_dim = int(early_esm_dim)
        self.early_esm_dropout = float(early_esm_dropout)
        self.early_esm_scope = str(early_esm_scope)
        self.fusion_mode = str(fusion_mode)
        self.cross_attention_neighborhood = str(cross_attention_neighborhood)
        self.cross_attention_bidirectional = bool(cross_attention_bidirectional)
        self.normalize_message_aggregation = bool(normalize_message_aggregation)
        self.site_feature_dim = int(site_feature_dim)
        self.classifier_pool_distance_cutoff = float(classifier_pool_distance_cutoff)
        self.structural_readout_scope = str(structural_readout_scope)
        self.use_node_type_embedding = bool(use_node_type_embedding)
        self.use_site_angle_features = bool(use_site_angle_features)
        if self.site_feature_dim < 1:
            raise ValueError(f"site_feature_dim must be positive, got {self.site_feature_dim}.")
        if self.classifier_pool_distance_cutoff < 0.0:
            raise ValueError(
                "classifier_pool_distance_cutoff must be non-negative, "
                f"got {self.classifier_pool_distance_cutoff}."
            )
        if self.structural_readout_scope not in VALID_STRUCTURAL_READOUT_SCOPES:
            raise ValueError(f"Unsupported structural_readout_scope {self.structural_readout_scope!r}.")
        early_scalar_dim = 0
        if self.use_early_esm:
            early_scalar_dim = esm_dim if self.early_esm_raw else self.early_esm_dim
        # Early ESMC injection adds residue-level ESMC information before GVP
        # message passing. The default is disabled so the current architecture is unchanged.
        self.node_scalar_encoder = NodeScalarEncoder(
            n_rbf=16,
            out_dim=hidden_s,
            distance_sigma=node_rbf_sigma,
            extra_scalar_dim=early_scalar_dim,
        )
        # Raw early ESMC is an ablation/control path only; the recommended path is
        # compressed residue-level ESMC via a small bottleneck projection.
        self.early_esm_proj = (
            None
            if not self.use_early_esm or self.early_esm_raw
            else EarlyESMEncoder(
                esm_input_dim=esm_dim,
                early_esm_dim=self.early_esm_dim,
                early_esm_dropout=self.early_esm_dropout,
            )
        )
        self.esm_graph_encoder = ESMGraphEncoder(
            esm_dim=esm_dim,
            proj_dim=esm_fusion_dim,
            dropout=esm_graph_encoder_dropout,
        )
        self.edge_scalar_encoder = EdgeScalarEncoder(n_rbf=16, out_dim=edge_hidden, distance_sigma=edge_rbf_sigma)
        self.gvp_attn_pool = AttentionPool(hidden_s)
        self.init_vec_proj = nn.Linear(2, hidden_v, bias=False)
        self.node_type_embedding = nn.Embedding(2, hidden_s) if self.use_node_type_embedding else None

        self.layers = nn.ModuleList(
            [
                SimpleGVPLayer(
                    s_dim=hidden_s,
                    v_dim=hidden_v,
                    e_dim=edge_hidden,
                    normalize_message_aggregation=self.normalize_message_aggregation,
                )
                for _ in range(n_layers)
            ]
        )

        gvp_graph_dim = 2 * hidden_s
        esm_graph_dim = 2 * esm_fusion_dim
        self.gvp_fusion_proj = nn.Sequential(
            nn.Linear(gvp_graph_dim, hidden_s),
            nn.LayerNorm(hidden_s),
            nn.SiLU(),
        )
        self.esm_fusion_proj = nn.Sequential(
            nn.Linear(esm_graph_dim, hidden_s),
            nn.LayerNorm(hidden_s),
            nn.SiLU(),
        )
        if self.fusion_mode == "cross_modal_attention":
            self.esm_residue_proj = nn.Sequential(
                nn.Linear(esm_dim, hidden_s),
                nn.LayerNorm(hidden_s),
                nn.SiLU(),
                nn.Dropout(cross_attention_dropout),
            )
            self.cross_attn_esm_pool = AttentionPool(hidden_s)
            self.cross_attn_esm_fusion_proj = nn.Sequential(
                nn.Linear(2 * hidden_s, hidden_s),
                nn.LayerNorm(hidden_s),
                nn.SiLU(),
            )
            self.cross_attention_blocks = nn.ModuleList(
                [
                    LocalizedCrossAttentionBlock(
                        hidden_dim=hidden_s,
                        n_heads=cross_attention_heads,
                        dropout=cross_attention_dropout,
                        bidirectional=self.cross_attention_bidirectional,
                    )
                    for _ in range(cross_attention_layers)
                ]
            )
        else:
            self.esm_residue_proj = None
            self.cross_attn_esm_pool = None
            self.cross_attn_esm_fusion_proj = None
            self.cross_attention_blocks = nn.ModuleList()
        if self.fusion_mode == "node_level_late_fusion":
            self.node_level_esm_proj = nn.Sequential(
                nn.Linear(esm_dim, hidden_s),
                nn.LayerNorm(hidden_s),
                nn.SiLU(),
            )
            self.node_level_gate = nn.Sequential(
                nn.Linear(2 * hidden_s, hidden_s),
                nn.Sigmoid(),
            )
        else:
            self.node_level_esm_proj = None
            self.node_level_gate = None
        site_feature_input_dim = DEFAULT_SITE_FEATURE_INPUT_DIM + (
            DEFAULT_SITE_LIGAND_ANGLE_FEATURE_DIM if self.use_site_angle_features else 0
        )
        self.site_feature_encoder = nn.Sequential(
            nn.Linear(site_feature_input_dim, self.site_feature_dim),
            nn.LayerNorm(self.site_feature_dim),
            nn.SiLU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_s, hidden_s),
            nn.Sigmoid(),
        )
        fused_dim = 2 * hidden_s + self.site_feature_dim
        self.predict_metal = bool(predict_metal)
        self.predict_ec = bool(predict_ec)
        self.use_esm_branch = bool(use_esm_branch)
        if not self.predict_metal and not self.predict_ec:
            raise ValueError("GVPPocketClassifier requires at least one enabled prediction head.")
        if self.fusion_mode not in VALID_FUSION_MODES:
            raise ValueError(f"Unsupported fusion_mode {self.fusion_mode!r}.")
        if self.early_esm_scope not in {"all", "first_shell", "first_second_shell"}:
            raise ValueError(f"Unsupported early_esm_scope {self.early_esm_scope!r}.")
        if self.cross_attention_neighborhood not in {"all", "first_shell", "first_second_shell"}:
            raise ValueError(
                f"Unsupported cross_attention_neighborhood {self.cross_attention_neighborhood!r}."
            )
        if self.fusion_mode == "cross_modal_attention" and not self.use_esm_branch:
            raise ValueError("fusion_mode='cross_modal_attention' requires the ESM branch to remain enabled.")
        if self.fusion_mode == "node_level_late_fusion" and not self.use_esm_branch:
            raise ValueError("fusion_mode='node_level_late_fusion' requires the ESM branch to remain enabled.")

        self.head_metal = (
            build_classifier_head(
                in_dim=fused_dim,
                hidden_dim=hidden_s,
                out_dim=n_metal,
                n_linear_layers=head_mlp_layers,
                dropout=head_mlp_dropout,
            )
            if self.predict_metal
            else None
        )
        self.head_ec = (
            build_classifier_head(
                in_dim=fused_dim,
                hidden_dim=hidden_s,
                out_dim=n_ec,
                n_linear_layers=head_mlp_layers,
                dropout=head_mlp_dropout,
            )
            if self.predict_ec
            else None
        )

        self.metal_loss_weight = float(metal_loss_weight)
        self.ec_loss_weight = float(ec_loss_weight)
        self.task_loss_weighter = TaskLossWeighter(
            mode=joint_loss_weighting,
            metal_loss_weight=metal_loss_weight,
            ec_loss_weight=ec_loss_weight,
            predict_metal=self.predict_metal,
            predict_ec=self.predict_ec,
        )
        self.node_rbf_use_raw_distances = bool(node_rbf_use_raw_distances)
        self.metal_loss_function = str(metal_loss_function)
        self.metal_focal_gamma = float(metal_focal_gamma)
        self.metal_label_smoothing = float(metal_label_smoothing)
        self.metal_collapsed_loss_weight = float(metal_collapsed_loss_weight)
        self.ec_contrastive_weight = float(ec_contrastive_weight)
        self.ec_contrastive_temperature = float(ec_contrastive_temperature)
        self.register_buffer(
            "metal_class_weights",
            metal_class_weights.float() if metal_class_weights is not None else torch.empty(0),
        )
        self.register_buffer(
            "metal_collapsed4_class_weights",
            (
                metal_collapsed4_class_weights.float()
                if metal_collapsed4_class_weights is not None
                else torch.empty(0)
            ),
        )
        self.register_buffer(
            "ec_class_weights",
            ec_class_weights.float() if ec_class_weights is not None else torch.empty(0),
        )

    def _early_esm_scalar_features(self, x_esm: Tensor) -> Tensor | None:
        if not self.use_early_esm:
            return None
        if self.early_esm_raw:
            return x_esm
        if self.early_esm_proj is None:
            raise ValueError("Early ESM projection is missing while compressed early ESM mode is enabled.")
        return self.early_esm_proj(x_esm)

    def _masked_early_esm_scalar_features(self, data: Data) -> Tensor | None:
        early_esm = self._early_esm_scalar_features(data.x_esm)
        if early_esm is None:
            return None
        scope_mask = shell_mask_from_roles(data.x_role, self.early_esm_scope)
        scope_mask = scope_mask & residue_node_mask(data)
        scope_mask = scope_mask.unsqueeze(-1).to(dtype=early_esm.dtype)
        return early_esm * scope_mask

    def _add_node_type_embedding(self, s: Tensor, data: Data) -> Tensor:
        if self.node_type_embedding is None:
            return s
        if hasattr(data, "node_type_id"):
            node_type_id = data.node_type_id.to(dtype=torch.long, device=s.device).clamp(0, 1)
        else:
            node_type_id = torch.zeros(s.size(0), dtype=torch.long, device=s.device)
        return s + self.node_type_embedding(node_type_id)

    def _structural_pool_mask(self, data: Data) -> Tensor | None:
        residue_mask = residue_node_mask(data)
        metal_mask = metal_node_mask(data)
        if self.structural_readout_scope == "residue_only":
            base_mask = residue_mask
        elif self.structural_readout_scope == "metal_only":
            base_mask = metal_mask
        elif self.structural_readout_scope == "residue_and_metal":
            base_mask = residue_mask | metal_mask
        else:
            raise ValueError(f"Unsupported structural_readout_scope {self.structural_readout_scope!r}.")

        if self.classifier_pool_distance_cutoff <= 0.0:
            if bool(base_mask.all().item()):
                return None
            return ensure_nonempty_pool_mask(base_mask, data.batch)

        if self.structural_readout_scope == "residue_and_metal":
            residue_cutoff_mask = metal_distance_pool_mask(
                data,
                self.classifier_pool_distance_cutoff,
                base_mask=residue_mask,
            )
            return ensure_nonempty_pool_mask(residue_cutoff_mask | metal_mask, data.batch)
        return metal_distance_pool_mask(data, self.classifier_pool_distance_cutoff, base_mask=base_mask)

    def _esm_pool_mask(self, data: Data) -> Tensor | None:
        residue_mask = residue_node_mask(data)
        return metal_distance_pool_mask(data, self.classifier_pool_distance_cutoff, base_mask=residue_mask)

    def _site_feature_tensor(self, data: Data, batch_size: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        if hasattr(data, "site_metal_stats"):
            site_features = [data.site_metal_stats.float().to(device=device)]
        else:
            site_features = [torch.zeros(batch_size, DEFAULT_SITE_FEATURE_INPUT_DIM, dtype=dtype, device=device)]
        if self.use_site_angle_features:
            if hasattr(data, "site_ligand_angle_stats"):
                angle_stats = data.site_ligand_angle_stats.float().to(device=device)
            else:
                angle_stats = torch.zeros(
                    batch_size,
                    DEFAULT_SITE_LIGAND_ANGLE_FEATURE_DIM,
                    dtype=dtype,
                    device=device,
                )
            site_features.append(angle_stats)
        return torch.cat(site_features, dim=-1)

    @staticmethod
    def _supervised_mask(target: Tensor) -> Tensor:
        return target != MISSING_CLASS_LABEL

    def _init_vector_channels(self, x_vec: Tensor) -> Tensor:
        # x_vec stores two explicit geometric vector channels per residue,
        # each represented in xyz coordinates.
        if x_vec.ndim != 3:
            raise ValueError(f"x_vec must be a 3D tensor, got shape {tuple(x_vec.shape)}.")
        if x_vec.size(1) == self.init_vec_proj.in_features and x_vec.size(2) == 3:
            x_t = x_vec.transpose(1, 2)
        elif x_vec.size(1) == 3 and x_vec.size(2) == self.init_vec_proj.in_features:
            x_t = x_vec
        else:
            raise ValueError(
                "x_vec must have two vector channels and three xyz coordinates per residue. "
                f"Got shape {tuple(x_vec.shape)}."
            )
        x_proj = self.init_vec_proj(x_t)
        return x_proj.transpose(1, 2)

    def _prepare_edge_vectors(self, data: Data) -> Tensor:
        if hasattr(data, "edge_vector_raw"):
            rel = data.edge_vector_raw.float()
        else:
            src, dst = data.edge_index
            rel = (data.pos[dst] - data.pos[src]).float()
        return rel.unsqueeze(1)

    def _compute_supervised_loss(
        self,
        pocket_embed: Tensor,
        logits_metal: Optional[Tensor],
        logits_ec: Optional[Tensor],
        data: Data,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        task_losses: dict[str, Tensor] = {}
        auxiliary_losses: dict[str, Tensor] = {}
        if self.predict_metal and logits_metal is not None and hasattr(data, "y_metal"):
            metal_mask = self._supervised_mask(data.y_metal)
            if bool(metal_mask.any().item()):
                metal_weights = self.metal_class_weights if self.metal_class_weights.numel() > 0 else None
                metal_logits = logits_metal[metal_mask]
                metal_targets = data.y_metal[metal_mask]
                if self.metal_loss_function == "cross_entropy":
                    metal_loss = F.cross_entropy(
                        metal_logits,
                        metal_targets,
                        weight=metal_weights,
                        label_smoothing=self.metal_label_smoothing,
                    )
                elif self.metal_loss_function == "focal":
                    ce_per_sample = F.cross_entropy(
                        metal_logits,
                        metal_targets,
                        weight=metal_weights,
                        reduction="none",
                    )
                    pt = torch.exp(-ce_per_sample)
                    metal_loss = (((1.0 - pt) ** self.metal_focal_gamma) * ce_per_sample).mean()
                else:
                    raise ValueError(f"Unsupported metal loss function {self.metal_loss_function!r}.")
                if self.metal_collapsed_loss_weight > 0.0:
                    collapsed_weights = (
                        self.metal_collapsed4_class_weights
                        if self.metal_collapsed4_class_weights.numel() > 0
                        else None
                    )
                    metal_loss, _collapsed_loss = metal_loss_with_optional_collapsed4(
                        metal_loss,
                        metal_logits,
                        metal_targets,
                        alpha=self.metal_collapsed_loss_weight,
                        collapsed4_weight=collapsed_weights,
                        label_smoothing=self.metal_label_smoothing,
                    )
                task_losses["metal"] = metal_loss
        if self.predict_ec and logits_ec is not None and hasattr(data, "y_ec"):
            ec_mask = self._supervised_mask(data.y_ec)
            if bool(ec_mask.any().item()):
                ec_weights = self.ec_class_weights if self.ec_class_weights.numel() > 0 else None
                ec_ce = F.cross_entropy(
                    logits_ec[ec_mask],
                    data.y_ec[ec_mask],
                    weight=ec_weights,
                    reduction="none",
                )
                if hasattr(data, "ec_sample_weight"):
                    ec_sample_weight = data.ec_sample_weight.view(-1).to(
                        dtype=ec_ce.dtype,
                        device=ec_ce.device,
                    )[ec_mask]
                else:
                    ec_sample_weight = torch.ones_like(ec_ce)
                ec_loss = (ec_ce * ec_sample_weight).sum() / ec_sample_weight.sum().clamp_min(1e-8)
                task_losses["ec"] = ec_loss
                if self.ec_contrastive_weight > 0.0:
                    ec_contrastive = supervised_contrastive_loss(
                        pocket_embed[ec_mask],
                        data.y_ec[ec_mask],
                        temperature=self.ec_contrastive_temperature,
                    )
                    auxiliary_losses["ec_contrastive"] = self.ec_contrastive_weight * ec_contrastive
        if not task_losses and not auxiliary_losses:
            raise ValueError("No supervised targets were available for the enabled prediction heads.")
        if task_losses:
            supervised_loss, diagnostics = self.task_loss_weighter(task_losses)
        else:
            supervised_loss = pocket_embed.new_zeros(())
            diagnostics = {}
        if auxiliary_losses:
            supervised_loss = supervised_loss + torch.stack(list(auxiliary_losses.values())).sum()
            diagnostics.update({f"{name}_loss_raw": loss.detach() for name, loss in auxiliary_losses.items()})
        diagnostics["loss"] = supervised_loss
        return supervised_loss, diagnostics

    def forward(self, data: Data) -> Dict[str, Tensor]:
        node_distances = (
            data.x_dist_raw_raw
            if self.node_rbf_use_raw_distances and hasattr(data, "x_dist_raw_raw")
            else data.x_dist_raw
        )
        # Early ESMC injection = residue-level ESMC features added before GVP
        # message passing. Late ESM fusion below remains unchanged.
        early_esm = self._masked_early_esm_scalar_features(data)
        s = self.node_scalar_encoder(
            data.x_reschem,
            data.hydrophobicity_kd,
            data.x_role,
            node_distances,
            data.x_misc,
            data.x_env_burial,
            data.x_env_electrostatics,
            extra_scalar_features=early_esm,
        )
        s = self._add_node_type_embedding(s, data)
        v = self._init_vector_channels(data.x_vec)

        edge_s = self.edge_scalar_encoder(
            data.edge_dist_raw,
            data.edge_seqsep,
            data.edge_same_chain,
            data.edge_interaction_type,
            data.edge_source_type,
        )
        edge_v = self._prepare_edge_vectors(data)

        for layer in self.layers:
            s, v = layer(s, v, data.edge_index, edge_s, edge_v)

        structural_pool_mask = self._structural_pool_mask(data)
        esm_pool_mask = self._esm_pool_mask(data)

        # Structural branch: pool the GVP states into one graph embedding.
        # Metal-node experiments may include generic metal anchor nodes in this
        # readout; ESM pooling below remains residue-only.
        if self.node_level_esm_proj is not None and self.node_level_gate is not None and self.use_esm_branch:
            node_level_esm = self.node_level_esm_proj(data.x_esm)
            node_level_gate = self.node_level_gate(torch.cat([s, node_level_esm], dim=-1))
            residue_gate_mask = residue_node_mask(data).unsqueeze(-1).to(dtype=s.dtype)
            s = s + (node_level_gate * node_level_esm * residue_gate_mask)

        if self.fusion_mode == "cross_modal_attention" and self.use_esm_branch:
            esm_residue_states = self.esm_residue_proj(data.x_esm)
            active_mask = shell_mask_from_roles(data.x_role, self.cross_attention_neighborhood)
            active_mask = active_mask & residue_node_mask(data)
            for block in self.cross_attention_blocks:
                s, esm_residue_states = block(s, esm_residue_states, data.batch, active_mask)
            gvp_graph_embed = pool_graph_states(s, data.batch, self.gvp_attn_pool, structural_pool_mask)
            esm_graph_embed = pool_graph_states(
                esm_residue_states,
                data.batch,
                self.cross_attn_esm_pool,
                esm_pool_mask,
            )
            gvp_fused = self.gvp_fusion_proj(gvp_graph_embed)
            esm_fused = self.cross_attn_esm_fusion_proj(esm_graph_embed)
        else:
            gvp_graph_embed = pool_graph_states(s, data.batch, self.gvp_attn_pool, structural_pool_mask)
            gvp_fused = self.gvp_fusion_proj(gvp_graph_embed)
            if self.use_esm_branch:
                # Late ESM fusion: pool residue ESM embeddings separately, then inject the
                # graph-level sequence signal near the classifier head.
                esm_graph_embed = self.esm_graph_encoder(data.x_esm, data.batch, esm_pool_mask)
                esm_fused = self.esm_fusion_proj(esm_graph_embed)
            else:
                batch_size = int(data.batch.max().item()) + 1
                esm_graph_embed = torch.zeros(
                    batch_size,
                    2 * self.esm_graph_encoder.attn_pool.score[0].in_features,
                    dtype=gvp_fused.dtype,
                    device=gvp_fused.device,
                )
                esm_fused = torch.zeros_like(gvp_fused)
        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 0
        site_stats = self._site_feature_tensor(
            data,
            batch_size,
            dtype=gvp_fused.dtype,
            device=gvp_fused.device,
        )
        site_fused = self.site_feature_encoder(site_stats)
        # The gate lets the model decide how much ESM information to inject per pocket.
        fusion_gate = self.fusion_gate(torch.cat([gvp_fused, esm_fused], dim=-1))
        pocket_embed = torch.cat([gvp_fused, fusion_gate * esm_fused, site_fused], dim=-1)

        outputs = {
            "embed": pocket_embed,
            "gvp_embed": gvp_graph_embed,
            "esm_embed": esm_graph_embed,
            "fusion_gate": fusion_gate,
        }
        logits_metal = self.head_metal(pocket_embed) if self.head_metal is not None else None
        logits_ec = self.head_ec(pocket_embed) if self.head_ec is not None else None
        if logits_metal is not None:
            outputs["logits_metal"] = logits_metal
        if logits_ec is not None:
            outputs["logits_ec"] = logits_ec

        has_supervised_targets = bool(
            self.predict_metal
            and hasattr(data, "y_metal")
            and self._supervised_mask(data.y_metal).any().item()
        ) or bool(
            self.predict_ec
            and hasattr(data, "y_ec")
            and self._supervised_mask(data.y_ec).any().item()
        )
        if has_supervised_targets:
            loss, loss_diagnostics = self._compute_supervised_loss(pocket_embed, logits_metal, logits_ec, data)
            outputs.update(loss_diagnostics)
            outputs["loss"] = loss

        return outputs
