from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

from data_structures import EDGE_SOURCE_TYPES, INTERACTION_SUMMARIES_OPTIONAL_WITH_RING
from data_structures import MISSING_CLASS_LABEL
from label_schemes import N_EC_CLASSES, N_METAL_CLASSES

VALID_FUSION_MODES = {
    "late_fusion",
    "early_fusion",
    "node_level_late_fusion",
    "hybrid",
    "cross_modal_attention",
}


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
    ):
        super().__init__()
        self.dist_rbf = RBFExpansion(n_rbf=n_rbf, d_min=0.0, d_max=12.0, sigma=distance_sigma)
        self.burial_encoder = TinyFeatureGroupMLP(in_dim=1, hidden_dim=4, out_dim=4)

        # Keep the heuristic q1*q2/r-style proxy and the PROPKA-derived
        # dpka_titr contribution as separate scalars rather than summing them.
        self.base_in_dim = 25 + 1 + 2 + 1 + 4 + 2 + 3 * n_rbf
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

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        logits = self.score(x).squeeze(-1)
        weights = softmax(logits, batch)
        return global_add_pool(x * weights.unsqueeze(-1), batch)


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

    def forward(self, x_esm: Tensor, batch: Tensor) -> Tensor:
        z = self.esm_proj(x_esm)
        z_mean = global_mean_pool(z, batch)
        z_attn = self.attn_pool(z, batch)
        return torch.cat([z_mean, z_attn], dim=-1)


def shell_mask_from_roles(x_role: Tensor, scope: str) -> Tensor:
    if scope == "all":
        return torch.ones(x_role.size(0), dtype=torch.bool, device=x_role.device)
    if scope == "first_shell":
        return x_role[:, 0] > 0.5
    if scope == "first_second_shell":
        return (x_role[:, 0] > 0.5) | (x_role[:, 1] > 0.5)
    raise ValueError(f"Unsupported shell scope {scope!r}.")


def pool_graph_states(x: Tensor, batch: Tensor, attn_pool: AttentionPool) -> Tensor:
    return torch.cat([global_mean_pool(x, batch), attn_pool(x, batch)], dim=-1)


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
    def __init__(self, s_dim: int, v_dim: int, e_dim: int):
        super().__init__()

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
) -> nn.Sequential:
    if n_linear_layers < 1:
        raise ValueError(f"Classifier head requires at least 1 linear layer, got {n_linear_layers}.")
    if n_linear_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(n_linear_layers - 1):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.2),
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
        node_rbf_sigma: float = 0.75,
        edge_rbf_sigma: float = 0.75,
        node_rbf_use_raw_distances: bool = False,
        metal_loss_weight: float = 1.0,
        ec_loss_weight: float = 1.0,
        metal_class_weights: Optional[Tensor] = None,
        ec_class_weights: Optional[Tensor] = None,
        metal_loss_function: str = "cross_entropy",
        metal_focal_gamma: float = 2.0,
        metal_label_smoothing: float = 0.0,
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
        self.esm_graph_encoder = ESMGraphEncoder(esm_dim=esm_dim, proj_dim=esm_fusion_dim, dropout=0.1)
        self.edge_scalar_encoder = EdgeScalarEncoder(n_rbf=16, out_dim=edge_hidden, distance_sigma=edge_rbf_sigma)
        self.gvp_attn_pool = AttentionPool(hidden_s)
        self.init_vec_proj = nn.Linear(2, hidden_v, bias=False)

        self.layers = nn.ModuleList(
            [SimpleGVPLayer(s_dim=hidden_s, v_dim=hidden_v, e_dim=edge_hidden) for _ in range(n_layers)]
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
        self.site_feature_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_s, hidden_s),
            nn.Sigmoid(),
        )
        fused_dim = 2 * hidden_s + 32
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
            )
            if self.predict_ec
            else None
        )

        self.metal_loss_weight = float(metal_loss_weight)
        self.ec_loss_weight = float(ec_loss_weight)
        self.node_rbf_use_raw_distances = bool(node_rbf_use_raw_distances)
        self.metal_loss_function = str(metal_loss_function)
        self.metal_focal_gamma = float(metal_focal_gamma)
        self.metal_label_smoothing = float(metal_label_smoothing)
        self.ec_contrastive_weight = float(ec_contrastive_weight)
        self.ec_contrastive_temperature = float(ec_contrastive_temperature)
        self.register_buffer(
            "metal_class_weights",
            metal_class_weights.float() if metal_class_weights is not None else torch.empty(0),
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
        scope_mask = shell_mask_from_roles(data.x_role, self.early_esm_scope).unsqueeze(-1).to(dtype=early_esm.dtype)
        return early_esm * scope_mask

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
    ) -> Tensor:
        # Final baseline policy:
        # - keep class-balanced CE on both heads because both targets are imbalanced
        # - keep equal task weights so the two supervised objectives remain symmetric
        # - choose checkpoints with balanced metrics rather than introducing a more complex loss first
        losses: list[Tensor] = []
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
                losses.append(self.metal_loss_weight * metal_loss)
        if self.predict_ec and logits_ec is not None and hasattr(data, "y_ec"):
            ec_mask = self._supervised_mask(data.y_ec)
            if bool(ec_mask.any().item()):
                ec_weights = self.ec_class_weights if self.ec_class_weights.numel() > 0 else None
                ec_loss = F.cross_entropy(logits_ec[ec_mask], data.y_ec[ec_mask], weight=ec_weights)
                losses.append(self.ec_loss_weight * ec_loss)
                if self.ec_contrastive_weight > 0.0:
                    ec_contrastive = supervised_contrastive_loss(
                        pocket_embed[ec_mask],
                        data.y_ec[ec_mask],
                        temperature=self.ec_contrastive_temperature,
                    )
                    losses.append(self.ec_contrastive_weight * ec_contrastive)
        if not losses:
            raise ValueError("No supervised targets were available for the enabled prediction heads.")
        return torch.stack(losses).sum()

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

        # Structural branch: pool the residue-level GVP states into one graph embedding.
        if self.node_level_esm_proj is not None and self.node_level_gate is not None and self.use_esm_branch:
            node_level_esm = self.node_level_esm_proj(data.x_esm)
            node_level_gate = self.node_level_gate(torch.cat([s, node_level_esm], dim=-1))
            s = s + (node_level_gate * node_level_esm)

        if self.fusion_mode == "cross_modal_attention" and self.use_esm_branch:
            esm_residue_states = self.esm_residue_proj(data.x_esm)
            active_mask = shell_mask_from_roles(data.x_role, self.cross_attention_neighborhood)
            for block in self.cross_attention_blocks:
                s, esm_residue_states = block(s, esm_residue_states, data.batch, active_mask)
            gvp_graph_embed = pool_graph_states(s, data.batch, self.gvp_attn_pool)
            esm_graph_embed = pool_graph_states(esm_residue_states, data.batch, self.cross_attn_esm_pool)
            gvp_fused = self.gvp_fusion_proj(gvp_graph_embed)
            esm_fused = self.cross_attn_esm_fusion_proj(esm_graph_embed)
        else:
            gvp_graph_embed = pool_graph_states(s, data.batch, self.gvp_attn_pool)
            gvp_fused = self.gvp_fusion_proj(gvp_graph_embed)
            if self.use_esm_branch:
                # Late ESM fusion: pool residue ESM embeddings separately, then inject the
                # graph-level sequence signal near the classifier head.
                esm_graph_embed = self.esm_graph_encoder(data.x_esm, data.batch)
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
        if hasattr(data, "site_metal_stats"):
            site_stats = data.site_metal_stats.float()
        else:
            batch_size = int(data.batch.max().item()) + 1
            site_stats = torch.zeros(batch_size, 4, dtype=torch.float32, device=gvp_fused.device)
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
            outputs["loss"] = self._compute_supervised_loss(pocket_embed, logits_metal, logits_ec, data)

        return outputs
