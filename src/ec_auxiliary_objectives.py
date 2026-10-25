"""Optional training-only EC objectives on the pocket embedding (default off).

L4: auxiliary heads for the second and third EC levels (masked cross-entropy).
L5: supervised contrastive loss on the third EC level with a projection head and a
cross-batch memory of detached embeddings. Same-protein entries are neither
positives nor part of the denominator.

Both are attached after the model is built, so models without them keep their
parameters, initialization and loss unchanged; predictions never use them.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model import build_classifier_head


EC_AUX_TARGET_FIELD = "y_ec_l{level}"
SUPCON_LEVEL = 3


def _ec_head_layout(model: nn.Module) -> tuple[int, int, int, float]:
    """(in_dim, hidden_dim, n_linear_layers, dropout) of the model's EC head."""
    head = getattr(model, "head_ec", None)
    if head is None:
        raise ValueError("EC auxiliary objectives need a model with an EC head")
    linears = [module for module in head if isinstance(module, nn.Linear)]
    dropouts = [module.p for module in head if isinstance(module, nn.Dropout)]
    hidden = linears[0].out_features if len(linears) > 1 else linears[0].in_features
    return linears[0].in_features, hidden, len(linears), float(dropouts[0]) if dropouts else 0.0


class EcSupConMemory(nn.Module):
    """Projection head plus a FIFO memory of detached, normalized projections."""

    def __init__(self, in_dim: int, projection_dim: int, memory_size: int, temperature: float, weight: float):
        super().__init__()
        if memory_size < 1 or projection_dim < 1 or temperature <= 0.0 or weight <= 0.0:
            raise ValueError("SupCon memory needs positive size, projection dimension, temperature and weight")
        self.projection = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, projection_dim))
        self.temperature = float(temperature)
        self.weight = float(weight)
        self.register_buffer("mem_z", torch.zeros(memory_size, projection_dim), persistent=False)
        self.register_buffer("mem_label", torch.full((memory_size,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("mem_protein", torch.full((memory_size,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("mem_valid", torch.zeros(memory_size, dtype=torch.bool), persistent=False)
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long), persistent=False)

    def project(self, pocket_embed: Tensor) -> Tensor:
        return F.normalize(self.projection(pocket_embed), dim=-1)

    @torch.no_grad()
    def enqueue(self, z: Tensor, labels: Tensor, proteins: Tensor) -> None:
        keep = labels >= 0
        z, labels, proteins = z[keep].detach().float(), labels[keep], proteins[keep]
        size = self.mem_z.shape[0]
        for row in range(z.shape[0]):
            slot = int(self.ptr.item())
            self.mem_z[slot] = z[row]
            self.mem_label[slot] = labels[row]
            self.mem_protein[slot] = proteins[row]
            self.mem_valid[slot] = True
            self.ptr.fill_((slot + 1) % size)


def memory_supcon_loss(
    z: Tensor,
    labels: Tensor,
    proteins: Tensor,
    mem_z: Tensor,
    mem_labels: Tensor,
    mem_proteins: Tensor,
    temperature: float,
) -> tuple[Tensor, dict[str, int]]:
    """Supervised contrastive loss of labelled batch anchors against batch and memory candidates.

    Candidates are the other labelled batch items plus the memory; any candidate of the
    anchor's own protein is removed from positives and denominator. Anchors without a
    positive are left out of the mean; with none at all the loss is zero.
    """
    anchors = labels >= 0
    cand_z = torch.cat([z, mem_z.to(z.dtype)], dim=0)
    cand_labels = torch.cat([labels, mem_labels])
    cand_proteins = torch.cat([proteins, mem_proteins])
    n_batch = z.shape[0]
    similarity = (z @ cand_z.T) / float(temperature)
    allowed = (cand_labels.unsqueeze(0) >= 0) & (proteins.unsqueeze(1) != cand_proteins.unsqueeze(0))
    allowed[:, :n_batch] &= ~torch.eye(n_batch, dtype=torch.bool, device=z.device)
    allowed &= anchors.unsqueeze(1)
    positive = allowed & (labels.unsqueeze(1) == cand_labels.unsqueeze(0))
    n_positive = positive.sum(dim=1)
    has_positive = n_positive > 0
    stats = {
        "anchors": int(anchors.sum().item()),
        "anchors_with_positive": int(has_positive.sum().item()),
        "positive_pairs": int(n_positive.sum().item()),
    }
    if not bool(has_positive.any().item()):
        return z.sum() * 0.0, stats
    # Only rows with a positive are computed, so no row's denominator is empty.
    rows_similarity = similarity[has_positive]
    rows_allowed = allowed[has_positive]
    rows_positive = positive[has_positive]
    denominator = torch.logsumexp(rows_similarity.masked_fill(~rows_allowed, float("-inf")), dim=1, keepdim=True)
    log_prob = rows_similarity - denominator
    mean_positive = (log_prob * rows_positive).sum(dim=1) / n_positive[has_positive]
    return -mean_positive.mean(), stats


def attach_ec_auxiliary_objectives(
    model: nn.Module,
    *,
    aux_level_sizes: dict[int, int],
    aux_weight: float,
    supcon_weight: float = 0.0,
    supcon_temperature: float = 0.1,
    supcon_memory: int = 256,
    supcon_projection_dim: int = 128,
) -> dict[str, Any]:
    """Attach L4 heads (levels -> class counts) and/or the L5 SupCon module; returns a replayable record."""
    in_dim, hidden, n_layers, dropout = _ec_head_layout(model)
    record: dict[str, Any] = {"ec_aux_levels": {}, "ec_supcon": None}
    if aux_level_sizes:
        if aux_weight <= 0.0:
            raise ValueError("EC auxiliary heads need a positive weight")
        model.ec_aux_heads = nn.ModuleDict({
            str(level): build_classifier_head(in_dim=in_dim, hidden_dim=hidden, out_dim=int(size),
                                              n_linear_layers=n_layers, dropout=dropout)
            for level, size in sorted(aux_level_sizes.items())
        })
        model.ec_aux_weight = float(aux_weight)
        record["ec_aux_levels"] = {str(level): int(size) for level, size in sorted(aux_level_sizes.items())}
        record["ec_aux_weight"] = float(aux_weight)
    if supcon_weight > 0.0:
        model.ec_supcon = EcSupConMemory(in_dim, supcon_projection_dim, supcon_memory, supcon_temperature, supcon_weight)
        record["ec_supcon"] = {"weight": float(supcon_weight), "temperature": float(supcon_temperature),
                               "memory": int(supcon_memory), "projection_dim": int(supcon_projection_dim)}
    return record


def _ec_sample_weight(data: Any, reference: Tensor) -> Tensor:
    if hasattr(data, "ec_sample_weight"):
        return data.ec_sample_weight.view(-1).to(dtype=reference.dtype, device=reference.device)
    return torch.ones(reference.shape[0], dtype=reference.dtype, device=reference.device)


def ec_auxiliary_terms(model: nn.Module, pocket_embed: Tensor, data: Any) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Weighted training-only loss terms and detached diagnostics; empty when nothing is attached."""
    heads: Optional[nn.ModuleDict] = getattr(model, "ec_aux_heads", None)
    supcon: Optional[EcSupConMemory] = getattr(model, "ec_supcon", None)
    if not model.training or (heads is None and supcon is None):
        return {}, {}
    losses: dict[str, Tensor] = {}
    diagnostics: dict[str, Tensor] = {}
    weights = _ec_sample_weight(data, pocket_embed)
    if heads is not None:
        for level, head in heads.items():
            targets = getattr(data, EC_AUX_TARGET_FIELD.format(level=level)).view(-1)
            mask = targets >= 0
            diagnostics[f"ec_aux_l{level}_labelled"] = mask.sum().detach()
            if not bool(mask.any().item()):
                continue
            ce = F.cross_entropy(head(pocket_embed[mask]), targets[mask], reduction="none")
            level_weights = weights[mask]
            loss = (ce * level_weights).sum() / level_weights.sum().clamp_min(1e-8)
            losses[f"ec_aux_l{level}"] = model.ec_aux_weight * loss
    if supcon is not None:
        labels = getattr(data, EC_AUX_TARGET_FIELD.format(level=SUPCON_LEVEL)).view(-1)
        proteins = data.ec_group_id.view(-1).to(device=labels.device)
        z = supcon.project(pocket_embed)
        valid = supcon.mem_valid
        loss, stats = memory_supcon_loss(z, labels, proteins, supcon.mem_z[valid], supcon.mem_label[valid],
                                         supcon.mem_protein[valid], supcon.temperature)
        losses["ec_supcon"] = supcon.weight * loss
        for key, value in stats.items():
            diagnostics[f"ec_supcon_{key}"] = torch.tensor(value)
        supcon.enqueue(z, labels, proteins)
    return losses, diagnostics
