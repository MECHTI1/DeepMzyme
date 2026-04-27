from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool

from data_structures import MISSING_CLASS_LABEL
from label_schemes import N_EC_CLASSES, N_METAL_CLASSES
from model import (
    AttentionPool,
    ESMGraphEncoder,
    EarlyESMEncoder,
    EdgeScalarEncoder,
    LocalizedCrossAttentionBlock,
    NodeScalarEncoder,
    build_classifier_head,
    pool_graph_states,
    shell_mask_from_roles,
    supervised_contrastive_loss,
)


class PocketClassifierBase(nn.Module):
    def __init__(self):
        super().__init__()

    def _initialize_prediction_heads(
        self,
        *,
        fused_dim: int,
        hidden_dim: int,
        n_metal: int,
        n_ec: int,
        head_mlp_layers: int,
        predict_metal: bool,
        predict_ec: bool,
        metal_loss_weight: float,
        ec_loss_weight: float,
        metal_class_weights: Optional[Tensor],
        ec_class_weights: Optional[Tensor],
        metal_loss_function: str,
        metal_focal_gamma: float,
        metal_label_smoothing: float,
        ec_contrastive_weight: float,
        ec_contrastive_temperature: float,
    ) -> None:
        self.predict_metal = bool(predict_metal)
        self.predict_ec = bool(predict_ec)
        if not self.predict_metal and not self.predict_ec:
            raise ValueError("Pocket classifier requires at least one enabled prediction head.")
        self.head_metal = (
            build_classifier_head(
                in_dim=fused_dim,
                hidden_dim=hidden_dim,
                out_dim=n_metal,
                n_linear_layers=head_mlp_layers,
            )
            if self.predict_metal
            else None
        )
        self.head_ec = (
            build_classifier_head(
                in_dim=fused_dim,
                hidden_dim=hidden_dim,
                out_dim=n_ec,
                n_linear_layers=head_mlp_layers,
            )
            if self.predict_ec
            else None
        )
        self.metal_loss_weight = float(metal_loss_weight)
        self.ec_loss_weight = float(ec_loss_weight)
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

    @staticmethod
    def _supervised_mask(target: Tensor) -> Tensor:
        return target != MISSING_CLASS_LABEL

    def _compute_supervised_loss(
        self,
        pocket_embed: Tensor,
        logits_metal: Optional[Tensor],
        logits_ec: Optional[Tensor],
        data: Data,
    ) -> Tensor:
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

    def _attach_outputs(
        self,
        *,
        pocket_embed: Tensor,
        logits_metal: Optional[Tensor],
        logits_ec: Optional[Tensor],
        extra_outputs: dict[str, Tensor],
        data: Data,
    ) -> dict[str, Tensor]:
        outputs = {
            "embed": pocket_embed,
            **extra_outputs,
        }
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


class OnlyESMPocketClassifier(PocketClassifierBase):
    def __init__(
        self,
        *,
        esm_dim: int,
        hidden_s: int = 128,
        n_metal: int = N_METAL_CLASSES,
        n_ec: int = N_EC_CLASSES,
        esm_fusion_dim: int = 128,
        head_mlp_layers: int = 2,
        metal_loss_weight: float = 1.0,
        ec_loss_weight: float = 1.0,
        metal_class_weights: Optional[Tensor] = None,
        ec_class_weights: Optional[Tensor] = None,
        metal_loss_function: str = "cross_entropy",
        metal_focal_gamma: float = 2.0,
        metal_label_smoothing: float = 0.0,
        ec_contrastive_weight: float = 0.0,
        ec_contrastive_temperature: float = 0.1,
        predict_metal: bool = True,
        predict_ec: bool = True,
    ):
        super().__init__()
        self.esm_graph_encoder = ESMGraphEncoder(esm_dim=esm_dim, proj_dim=esm_fusion_dim, dropout=0.1)
        self.esm_fusion_proj = nn.Sequential(
            nn.Linear(2 * esm_fusion_dim, hidden_s),
            nn.LayerNorm(hidden_s),
            nn.SiLU(),
        )
        self.site_feature_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self._initialize_prediction_heads(
            fused_dim=hidden_s + 32,
            hidden_dim=hidden_s,
            n_metal=n_metal,
            n_ec=n_ec,
            head_mlp_layers=head_mlp_layers,
            predict_metal=predict_metal,
            predict_ec=predict_ec,
            metal_loss_weight=metal_loss_weight,
            ec_loss_weight=ec_loss_weight,
            metal_class_weights=metal_class_weights,
            ec_class_weights=ec_class_weights,
            metal_loss_function=metal_loss_function,
            metal_focal_gamma=metal_focal_gamma,
            metal_label_smoothing=metal_label_smoothing,
            ec_contrastive_weight=ec_contrastive_weight,
            ec_contrastive_temperature=ec_contrastive_temperature,
        )

    def forward(self, data: Data) -> dict[str, Tensor]:
        esm_graph_embed = self.esm_graph_encoder(data.x_esm, data.batch)
        esm_fused = self.esm_fusion_proj(esm_graph_embed)
        site_stats = (
            data.site_metal_stats.float()
            if hasattr(data, "site_metal_stats")
            else torch.zeros(esm_fused.size(0), 4, dtype=esm_fused.dtype, device=esm_fused.device)
        )
        site_fused = self.site_feature_encoder(site_stats)
        pocket_embed = torch.cat([esm_fused, site_fused], dim=-1)
        logits_metal = self.head_metal(pocket_embed) if self.head_metal is not None else None
        logits_ec = self.head_ec(pocket_embed) if self.head_ec is not None else None
        return self._attach_outputs(
            pocket_embed=pocket_embed,
            logits_metal=logits_metal,
            logits_ec=logits_ec,
            extra_outputs={
                "esm_embed": esm_graph_embed,
                "gvp_embed": torch.zeros_like(esm_graph_embed),
            },
            data=data,
        )


class SimpleGNNPocketClassifier(PocketClassifierBase):
    def __init__(
        self,
        *,
        esm_dim: int,
        hidden_s: int = 128,
        edge_hidden: int = 64,
        n_layers: int = 4,
        n_metal: int = N_METAL_CLASSES,
        n_ec: int = N_EC_CLASSES,
        esm_fusion_dim: int = 128,
        head_mlp_layers: int = 2,
        node_rbf_sigma: float = 0.75,
        edge_rbf_sigma: float = 0.75,
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
        self.use_esm_branch = bool(use_esm_branch)
        self.use_early_esm = bool(use_early_esm)
        self.early_esm_raw = bool(early_esm_raw)
        self.early_esm_scope = str(early_esm_scope)
        self.fusion_mode = str(fusion_mode)
        self.cross_attention_neighborhood = str(cross_attention_neighborhood)
        early_scalar_dim = 0
        if self.use_early_esm:
            early_scalar_dim = esm_dim if self.early_esm_raw else int(early_esm_dim)
        self.node_scalar_encoder = NodeScalarEncoder(
            n_rbf=16,
            out_dim=hidden_s,
            distance_sigma=node_rbf_sigma,
            extra_scalar_dim=early_scalar_dim,
        )
        self.early_esm_proj = (
            None
            if not self.use_early_esm or self.early_esm_raw
            else EarlyESMEncoder(
                esm_input_dim=esm_dim,
                early_esm_dim=early_esm_dim,
                early_esm_dropout=early_esm_dropout,
            )
        )
        self.edge_scalar_encoder = EdgeScalarEncoder(n_rbf=16, out_dim=edge_hidden, distance_sigma=edge_rbf_sigma)
        self.gnn_layers = nn.ModuleList(
            [
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_s, hidden_s),
                        nn.SiLU(),
                        nn.Linear(hidden_s, hidden_s),
                    ),
                    edge_dim=edge_hidden,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_s) for _ in range(n_layers)])
        self.struct_attn_pool = AttentionPool(hidden_s)
        self.esm_graph_encoder = ESMGraphEncoder(esm_dim=esm_dim, proj_dim=esm_fusion_dim, dropout=0.1)
        self.gnn_fusion_proj = nn.Sequential(
            nn.Linear(2 * hidden_s, hidden_s),
            nn.LayerNorm(hidden_s),
            nn.SiLU(),
        )
        self.esm_fusion_proj = nn.Sequential(
            nn.Linear(2 * esm_fusion_dim, hidden_s),
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
                        bidirectional=cross_attention_bidirectional,
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
        self._initialize_prediction_heads(
            fused_dim=(2 * hidden_s) + 32,
            hidden_dim=hidden_s,
            n_metal=n_metal,
            n_ec=n_ec,
            head_mlp_layers=head_mlp_layers,
            predict_metal=predict_metal,
            predict_ec=predict_ec,
            metal_loss_weight=metal_loss_weight,
            ec_loss_weight=ec_loss_weight,
            metal_class_weights=metal_class_weights,
            ec_class_weights=ec_class_weights,
            metal_loss_function=metal_loss_function,
            metal_focal_gamma=metal_focal_gamma,
            metal_label_smoothing=metal_label_smoothing,
            ec_contrastive_weight=ec_contrastive_weight,
            ec_contrastive_temperature=ec_contrastive_temperature,
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

    def forward(self, data: Data) -> dict[str, Tensor]:
        early_esm = self._masked_early_esm_scalar_features(data)
        x = self.node_scalar_encoder(
            data.x_reschem,
            data.hydrophobicity_kd,
            data.x_role,
            data.x_dist_raw,
            data.x_misc,
            data.x_env_burial,
            data.x_env_electrostatics,
            extra_scalar_features=early_esm,
        )
        edge_attr = self.edge_scalar_encoder(
            data.edge_dist_raw,
            data.edge_seqsep,
            data.edge_same_chain,
            data.edge_interaction_type,
            data.edge_source_type,
        )
        for conv, norm in zip(self.gnn_layers, self.layer_norms):
            x = norm(x + conv(x, data.edge_index, edge_attr=edge_attr))

        if self.node_level_esm_proj is not None and self.node_level_gate is not None and self.use_esm_branch:
            node_level_esm = self.node_level_esm_proj(data.x_esm)
            node_level_gate = self.node_level_gate(torch.cat([x, node_level_esm], dim=-1))
            x = x + (node_level_gate * node_level_esm)

        if self.fusion_mode == "cross_modal_attention" and self.use_esm_branch:
            esm_residue_states = self.esm_residue_proj(data.x_esm)
            active_mask = shell_mask_from_roles(data.x_role, self.cross_attention_neighborhood)
            for block in self.cross_attention_blocks:
                x, esm_residue_states = block(x, esm_residue_states, data.batch, active_mask)
            struct_embed = pool_graph_states(x, data.batch, self.struct_attn_pool)
            esm_graph_embed = pool_graph_states(esm_residue_states, data.batch, self.cross_attn_esm_pool)
            struct_fused = self.gnn_fusion_proj(struct_embed)
            esm_fused = self.cross_attn_esm_fusion_proj(esm_graph_embed)
        else:
            struct_embed = torch.cat([global_mean_pool(x, data.batch), self.struct_attn_pool(x, data.batch)], dim=-1)
            struct_fused = self.gnn_fusion_proj(struct_embed)
            if self.use_esm_branch:
                esm_graph_embed = self.esm_graph_encoder(data.x_esm, data.batch)
                esm_fused = self.esm_fusion_proj(esm_graph_embed)
            else:
                batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 0
                esm_graph_embed = torch.zeros(
                    batch_size,
                    2 * self.esm_graph_encoder.attn_pool.score[0].in_features,
                    dtype=struct_fused.dtype,
                    device=struct_fused.device,
                )
                esm_fused = torch.zeros_like(struct_fused)

        site_stats = (
            data.site_metal_stats.float()
            if hasattr(data, "site_metal_stats")
            else torch.zeros(struct_fused.size(0), 4, dtype=struct_fused.dtype, device=struct_fused.device)
        )
        site_fused = self.site_feature_encoder(site_stats)
        fusion_gate = self.fusion_gate(torch.cat([struct_fused, esm_fused], dim=-1))
        pocket_embed = torch.cat([struct_fused, fusion_gate * esm_fused, site_fused], dim=-1)
        logits_metal = self.head_metal(pocket_embed) if self.head_metal is not None else None
        logits_ec = self.head_ec(pocket_embed) if self.head_ec is not None else None
        return self._attach_outputs(
            pocket_embed=pocket_embed,
            logits_metal=logits_metal,
            logits_ec=logits_ec,
            extra_outputs={
                "gvp_embed": struct_embed,
                "esm_embed": esm_graph_embed,
                "fusion_gate": fusion_gate,
            },
            data=data,
        )
