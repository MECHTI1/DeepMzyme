"""Training-only EC auxiliary heads (L4) and memory supervised contrastive loss (L5); synthetic."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_auxiliary_objectives import (
    EcSupConMemory, attach_ec_auxiliary_objectives, ec_auxiliary_terms, memory_supcon_loss,
)


def _reference(z, labels, proteins, mem_z, mem_labels, mem_proteins, temperature):
    cand = list(zip(torch.cat([z, mem_z]).tolist(), torch.cat([labels, mem_labels]).tolist(),
                    torch.cat([proteins, mem_proteins]).tolist(), range(z.shape[0] + mem_z.shape[0])))
    losses = []
    for i in range(z.shape[0]):
        if labels[i] < 0:
            continue
        allowed = [(vec, lab) for vec, lab, prot, idx in cand
                   if idx != i and lab >= 0 and prot != int(proteins[i])]
        sims = [sum(a * b for a, b in zip(z[i].tolist(), vec)) / temperature for vec, _ in allowed]
        positives = [s for (vec, lab), s in zip(allowed, sims) if lab == int(labels[i])]
        if not positives:
            continue
        denom = math.log(sum(math.exp(s) for s in sims))
        losses.append(-sum(s - denom for s in positives) / len(positives))
    return sum(losses) / len(losses) if losses else 0.0


def _normalized(n, d, seed):
    return torch.nn.functional.normalize(torch.randn(n, d, generator=torch.Generator().manual_seed(seed)), dim=1)


def test_memory_supcon_matches_brute_force_and_excludes_same_protein():
    z = _normalized(6, 4, 0)
    labels = torch.tensor([0, 0, 1, 1, -1, 0])
    proteins = torch.tensor([10, 10, 11, 12, 13, 14])
    mem_z = _normalized(3, 4, 1)
    mem_labels = torch.tensor([1, 0, 0])
    mem_proteins = torch.tensor([11, 10, 20])
    loss, stats = memory_supcon_loss(z, labels, proteins, mem_z, mem_labels, mem_proteins, 0.5)
    assert float(loss) == pytest.approx(_reference(z, labels, proteins, mem_z, mem_labels, mem_proteins, 0.5), rel=1e-5)
    assert stats["anchors"] == 5
    # Protein 10's two pockets are not each other's positives; the memory copy of protein 10 is excluded too.
    same_protein_only = memory_supcon_loss(z[:2], labels[:2], proteins[:2], mem_z[1:2], mem_labels[1:2],
                                           mem_proteins[1:2], 0.5)
    assert same_protein_only[1]["anchors_with_positive"] == 0 and float(same_protein_only[0]) == 0.0


def test_empty_memory_is_in_batch_supcon_and_memory_gets_no_gradient():
    z = _normalized(4, 3, 2).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1])
    proteins = torch.tensor([1, 2, 3, 4])
    empty = torch.zeros(0, 3)
    loss, _ = memory_supcon_loss(z, labels, proteins, empty, torch.zeros(0, dtype=torch.long),
                                 torch.zeros(0, dtype=torch.long), 0.2)
    assert float(loss.detach()) == pytest.approx(_reference(z.detach(), labels, proteins, empty, torch.zeros(0, dtype=torch.long),
                                                   torch.zeros(0, dtype=torch.long), 0.2), rel=1e-5)
    memory = EcSupConMemory(3, 3, 4, 0.2, 1.0)
    memory.enqueue(z, labels, proteins)
    loss2, _ = memory_supcon_loss(z, labels, proteins, memory.mem_z, memory.mem_label, memory.mem_protein, 0.2)
    loss2.backward()
    assert z.grad is not None and not memory.mem_z.requires_grad and torch.isfinite(z.grad).all()


def test_fifo_wraps_and_skips_unlabelled():
    memory = EcSupConMemory(2, 2, 3, 0.1, 1.0)
    z = torch.eye(2).repeat(3, 1)[:5]
    memory.enqueue(z, torch.tensor([0, -1, 1, 2, 3]), torch.arange(5))
    assert memory.mem_valid.all() and int(memory.ptr) == 1
    assert memory.mem_label.tolist() == [3, 1, 2]
    assert set(dict(memory.state_dict())) == {"projection.0.weight", "projection.0.bias",
                                              "projection.2.weight", "projection.2.bias"}


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_ec = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Dropout(0.0),
                                           torch.nn.Linear(8, 7))


def _batch(n=4):
    return SimpleNamespace(
        y_ec_l2=torch.tensor([0, 1, -1, 0][:n]), y_ec_l3=torch.tensor([0, -1, -1, 0][:n]),
        ec_sample_weight=torch.tensor([1.0, 2.0, 1.0, 0.5][:n]), ec_group_id=torch.tensor([0, 1, 2, 3][:n]),
    )


def test_auxiliary_heads_masked_weighted_ce_and_eval_mode():
    torch.manual_seed(0)
    model = _TinyModel()
    record = attach_ec_auxiliary_objectives(model, aux_level_sizes={2: 2, 3: 1}, aux_weight=0.3)
    assert record["ec_aux_levels"] == {"2": 2, "3": 1}
    assert [m.in_features for m in model.ec_aux_heads["2"] if isinstance(m, torch.nn.Linear)] == [4, 8]
    embed = torch.randn(4, 4)
    data = _batch()
    model.train()
    losses, diagnostics = ec_auxiliary_terms(model, embed, data)
    logits = model.ec_aux_heads["2"](embed[[0, 1, 3]])
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0]), reduction="none")
    expected = 0.3 * (ce * torch.tensor([1.0, 2.0, 0.5])).sum() / 3.5
    assert float(losses["ec_aux_l2"].detach()) == pytest.approx(float(expected.detach()), rel=1e-6)
    assert int(diagnostics["ec_aux_l3_labelled"]) == 2
    all_masked = SimpleNamespace(**{**vars(data), "y_ec_l2": torch.full((4,), -1), "y_ec_l3": torch.full((4,), -1)})
    assert set(ec_auxiliary_terms(model, embed, all_masked)[0]) == set()
    model.eval()
    assert ec_auxiliary_terms(model, embed, data) == ({}, {})


def test_supcon_enqueues_only_in_training_and_reports_stats():
    model = _TinyModel()
    attach_ec_auxiliary_objectives(model, aux_level_sizes={}, aux_weight=0.3, supcon_weight=0.1,
                                   supcon_memory=8, supcon_projection_dim=3)
    data = _batch()
    model.train()
    losses, diagnostics = ec_auxiliary_terms(model, torch.randn(4, 4), data)
    assert "ec_supcon" in losses and int(diagnostics["ec_supcon_anchors"]) == 2
    assert int(model.ec_supcon.mem_valid.sum()) == 2
    model.eval()
    ec_auxiliary_terms(model, torch.randn(4, 4), data)
    assert int(model.ec_supcon.mem_valid.sum()) == 2


def test_attach_refusals():
    with pytest.raises(ValueError, match="EC head"):
        attach_ec_auxiliary_objectives(torch.nn.Linear(2, 2), aux_level_sizes={2: 3}, aux_weight=0.3)
    with pytest.raises(ValueError, match="positive weight"):
        attach_ec_auxiliary_objectives(_TinyModel(), aux_level_sizes={2: 3}, aux_weight=0.0)


def test_training_only_vocabularies():
    from training.config import TrainConfig
    from training.run import assign_ec_auxiliary_targets
    from training.splits import PocketSplit

    def pocket(ec):
        return SimpleNamespace(metadata={"ec_numbers": [ec]})

    train = [pocket("3.1.1.1"), pocket("3.1.2.1"), pocket("3.-.-.-")]
    val = [pocket("3.9.9.9")]
    vocab = assign_ec_auxiliary_targets(PocketSplit(train, val), TrainConfig(task="ec", ec_aux_levels="2,3"))
    assert vocab["2"]["tokens"] == ["3.1"] and vocab["3"]["tokens"] == ["3.1.1", "3.1.2"]
    assert [p.metadata["ec_aux_l3"] for p in train] == [0, 1, -1]
    assert val[0].metadata["ec_aux_l2"] == -1 and vocab["3"]["n_train_pockets_masked"] == 1
    assert assign_ec_auxiliary_targets(PocketSplit(train, val), TrainConfig(task="ec")) == {}


@pytest.mark.parametrize(
    "argv",
    [
        ["--task", "metal", "--ec-aux-levels", "2"],
        ["--task", "ec", "--ec-aux-levels", "4"],
        ["--task", "ec", "--ec-aux-levels", "2,2"],
        ["--task", "ec", "--ec-aux-levels", "2", "--ec-aux-weight", "0"],
        ["--task", "ec", "--ec-supcon-weight", "0.1", "--ec-contrastive-weight", "0.1"],
        ["--task", "ec", "--ec-supcon-weight", "0.1", "--ec-supcon-memory", "0"],
    ],
)
def test_config_refusals(argv):
    from training.config import parse_args

    with pytest.raises(SystemExit):
        parse_args(argv)


def test_default_model_is_unchanged_and_tiny_fit_runs(tmp_path, monkeypatch):
    import json

    from ec_fixtures import default_ec_proteins, tiny_ec_argv, write_ec_dataset
    from training.config import parse_args
    from training.run import run_training

    monkeypatch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(tmp_path / "cache"))
    paths = write_ec_dataset(tmp_path / "data", default_ec_proteins())
    common = ["--val-fraction", "0.3", "--ec-class-vocabulary", "fixed_seven", "--checkpoint-rule", "terminal"]
    plain = run_training(parse_args(tiny_ec_argv(paths, tmp_path / "runs", *common, "--run-name", "plain")))
    plain_state = torch.load(plain / "terminal_model_checkpoint.pt", map_location="cpu", weights_only=False)
    assert not any(key.startswith(("ec_aux_heads", "ec_supcon")) for key in plain_state["model_state_dict"])
    assert "ec_auxiliary_objectives" not in plain_state["dataset_summary"]
    with_objectives = run_training(parse_args(tiny_ec_argv(
        paths, tmp_path / "runs", *common, "--ec-aux-levels", "2,3", "--ec-supcon-weight", "0.1",
        "--ec-supcon-memory", "8", "--run-name", "objectives")))
    state = torch.load(with_objectives / "terminal_model_checkpoint.pt", map_location="cpu", weights_only=False)
    assert any(key.startswith("ec_aux_heads.2") for key in state["model_state_dict"])
    assert not any("mem_" in key for key in state["model_state_dict"])
    objectives = state["dataset_summary"]["ec_auxiliary_objectives"]
    assert objectives["attachments"]["ec_supcon"]["memory"] == 8 and objectives["vocabularies"]["2"]["tokens"]
    base_keys = set(plain_state["model_state_dict"])
    assert base_keys <= set(state["model_state_dict"])
    history = json.loads((with_objectives / "run_config.json").read_text())["history"]
    assert all(math.isfinite(epoch["train_loss"]) for epoch in history)
