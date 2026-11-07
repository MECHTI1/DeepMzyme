"""Amendment 2, family F6: whole-protein branch pretraining, freezing, the C-GF control and the trainer flags.

Synthetic fixtures only: a small corpus table, an 8-dimensional sequence-mean cache and stand-in pockets.
"""
from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from model_variants.factory import build_pocket_classifier  # noqa: E402
from training import global_esm_pretraining as gp  # noqa: E402
from training.config import TrainConfig, config_to_payload, parse_args  # noqa: E402
from training.ec_classes import EC_CLASS_TOKEN_TO_INDEX  # noqa: E402
from training.global_esm_features import (  # noqa: E402
    GLOBAL_ESM_METADATA_KEY, KIND_SEQUENCE_MEAN, build_manifest, encode_cache_file, sequence_mean_payload,
    write_bytes_atomic, write_manifest,
)
from training.loop import class_weights_from_labels  # noqa: E402

DIM = 8
HIDDEN = 16
N_INNER = 4


def _key(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_sequence_cache(directory: Path, vectors: dict[str, torch.Tensor], *, dim: int = DIM) -> str:
    """A sequence-mean cache (final layer) holding ``vectors`` keyed by sequence SHA-256; returns the manifest SHA."""
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for key, vector in sorted(vectors.items()):
        payload = sequence_mean_payload(vector.to(torch.float32), sequence_sha256=key, layer="final", n_tokens=10,
                                        n_mask_tokens=0, n_tokens_input=10, source={"kind": "synthetic"},
                                        embedding_dim=dim)
        item = encode_cache_file(payload)
        write_bytes_atomic(directory / item.file_name, item.data)
        files.append(item)
    manifest = build_manifest(files, kind=KIND_SEQUENCE_MEAN, allowlist={"file_sha256": "0" * 64}, build={"action": "t"},
                              embedding_dim=dim)
    return write_manifest(directory, manifest)[1]


class World:
    """A synthetic fold: 16 metallo proteins in 4 inner folds plus 24 other enzymes of the official file."""

    def __init__(self, root: Path, *, seed: int = 3):
        rng = random.Random(seed)
        generator = torch.Generator().manual_seed(seed)
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.metallo = {f"P{index:04d}": index % N_INNER for index in range(16)}
        self.classes = {accession: str(1 + index % 7) for index, accession in enumerate(sorted(self.metallo))}
        others = {f"Q{index:04d}": str(1 + rng.randrange(7)) for index in range(24)}
        self.classes.update(others)
        self.sequences = {accession: "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(30))
                          for accession in self.classes}
        rows = []
        for accession in sorted(self.classes):
            flags = tuple(self.metallo.get(accession) == inner for inner in range(N_INNER))
            rows.append(gp.CorpusRow(accession, _key(self.sequences[accession]), self.classes[accession], flags))
        self.rows = rows
        self.corpus = root / "corpus.tsv"
        self.corpus.write_bytes(gp.render_corpus(rows))
        self.corpus_sha = gp._sha256_file(self.corpus)
        self.vectors = {row.sequence_sha256: torch.randn(DIM, generator=generator) for row in rows}
        self.cache = root / "seq_means_final"
        self.manifest_sha = write_sequence_cache(self.cache, self.vectors)
        self.chain_means = {accession: torch.randn(DIM, generator=generator) for accession in self.metallo}

    def pockets(self, inner: int):
        def pocket(accession):
            sid = f"{accession}__chain_A__EC_{self.classes[accession]}.1.1.1"
            return SimpleNamespace(structure_id=sid, pocket_id=f"{sid}__p0", y_ec=EC_CLASS_TOKEN_TO_INDEX[
                self.classes[accession]], metadata={GLOBAL_ESM_METADATA_KEY: self.chain_means[accession].clone(),
                                                    "ec_group_key": sid, "selected_site_keys": [("A", 900)]})
        train = [pocket(accession) for accession, fold in sorted(self.metallo.items()) if fold != inner]
        val = [pocket(accession) for accession, fold in sorted(self.metallo.items()) if fold == inner]
        return train, val

    def config(self, split: str = "inner0", **changes) -> TrainConfig:
        inner = None if split == "refit" else int(split.removeprefix("inner"))
        base = replace(TrainConfig(), task="ec", ec_class_vocabulary="fixed_seven", model_architecture="only_esm",
                       use_global_esm_branch=True, global_esm_dim=DIM, hidden_s=HIDDEN,
                       global_esm_branch_init="pretrained_frozen", global_esm_pretrain_cache_dir=self.cache,
                       global_esm_pretrain_manifest_sha256=self.manifest_sha, global_esm_pretrain_corpus=self.corpus,
                       global_esm_pretrain_corpus_sha256=self.corpus_sha, global_esm_pretrain_split=split,
                       global_esm_pretrain_batch_size=8, fold_index=inner, n_folds=None if inner is None else N_INNER,
                       seed=42)
        return replace(base, **changes)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path / "world")


def _run(world: World, tmp_path: Path, name: str, split: str = "inner0", **changes):
    config = world.config(split, **changes)
    inner = None if split == "refit" else int(split.removeprefix("inner"))
    train, val = world.pockets(inner if inner is not None else -1)
    run_dir = tmp_path / name
    run_dir.mkdir()
    return gp.run_global_esm_pretraining(config, train_pockets=train, val_pockets=val, run_dir=run_dir), run_dir


def _small_vectors(n: int = 40, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    vectors = torch.randn(n, DIM, generator=generator)
    labels = torch.tensor([index % 7 for index in range(n)])
    return vectors, labels


def _pretrain(seed: int = 42, epochs: int = 3):
    vectors, labels = _small_vectors()
    return gp.pretrain_whole_protein_classifier(vectors, labels, seed=seed, embedding_dim=DIM, hidden_dim=HIDDEN,
                                                branch_dropout=0.2, epochs=epochs, batch_size=8)


def _f6_model(seed: int = 42):
    torch.manual_seed(seed)
    return build_pocket_classifier(model_architecture="only_esm", esm_dim=DIM, hidden_s=HIDDEN, predict_metal=False,
                                   predict_ec=True, n_ec=7, use_global_esm_branch=True, global_esm_dim=DIM)


# ---------------------------------------------------------------------------
# Corpus table and split selection


def test_corpus_round_trip_and_strict_reader(world):
    corpus = gp.read_pretrain_corpus(world.corpus, world.corpus_sha)
    assert corpus.rows == tuple(world.rows) and gp.render_corpus(reversed(world.rows)) == world.corpus.read_bytes()
    assert world.corpus.read_text().splitlines()[0].split("\t") == list(gp.CORPUS_COLUMNS)
    with pytest.raises(gp.GlobalEsmPretrainError, match="SHA-256"):
        gp.read_pretrain_corpus(world.corpus, "0" * 64)
    text = world.corpus.read_text()
    lines = text.splitlines()
    for broken, message in (
        (text.replace("exclude_inner3", "exclude_inner9"), "columns"),
        ("\n".join([lines[0], lines[2], lines[1], *lines[3:]]) + "\n", "sorted order"),
        (text.replace("\t0\n", "\t2\n", 1), "0 or 1"),
        (text.rstrip("\n"), "newline"),
    ):
        with pytest.raises(gp.GlobalEsmPretrainError, match=message):
            gp.parse_corpus(broken.encode("utf-8"), origin="corpus.tsv")


def test_inner_split_excludes_exactly_the_flagged_rows_and_refit_keeps_all(world):
    corpus = gp.read_pretrain_corpus(world.corpus, world.corpus_sha)
    train, val = world.pockets(1)
    accession = lambda pockets: {pocket.structure_id.split("__")[0] for pocket in pockets}  # noqa: E731
    selection = gp.select_pretrain_rows(corpus, "inner1", training_accessions=accession(train),
                                        validation_accessions=accession(val))
    assert {row.accession for row in selection.rows} == {row.accession for row in world.rows if not row.exclude[1]}
    assert selection.n_excluded == sum(row.exclude[1] for row in world.rows) == len(val)
    assert selection.checks["validation_in_corpus_flagged"] == len(val) and selection.checks["training_flagged"] == 0
    refit = gp.select_pretrain_rows(corpus, "refit", training_accessions=set(world.metallo))
    assert refit.rows == corpus.rows and refit.n_excluded == 0


def test_split_selection_refuses_leaks_and_foreign_corpora(world):
    corpus = gp.read_pretrain_corpus(world.corpus, world.corpus_sha)
    training = {accession for accession, fold in world.metallo.items() if fold != 0}
    validation = {accession for accession, fold in world.metallo.items() if fold == 0}
    later = {accession for accession, fold in world.metallo.items() if fold in (2, 3)}
    with pytest.raises(gp.GlobalEsmPretrainError, match="not flagged"):
        gp.select_pretrain_rows(corpus, "inner1", training_accessions=later, validation_accessions=validation)
    with pytest.raises(gp.GlobalEsmPretrainError, match="inner-training protein"):
        gp.select_pretrain_rows(corpus, "inner0", training_accessions=training | {"P0000"},
                                validation_accessions=validation - {"P0000"})
    with pytest.raises(gp.GlobalEsmPretrainError, match="no validation proteins"):
        gp.select_pretrain_rows(corpus, "refit", training_accessions=training, validation_accessions=validation)
    with pytest.raises(gp.GlobalEsmPretrainError, match="another fold"):
        gp.select_pretrain_rows(corpus, "inner0", training_accessions={"X1"}, validation_accessions=validation)
    with pytest.raises(gp.GlobalEsmPretrainError, match="Unknown pretraining split"):
        gp.select_pretrain_rows(corpus, "inner4", training_accessions=training)
    # A run protein absent from the corpus (several EC classes in the official file) is counted, not refused.
    selection = gp.select_pretrain_rows(corpus, "inner0", training_accessions=training | {"Z9999"},
                                        validation_accessions=validation)
    assert selection.checks["training_absent_from_corpus"] == 1


# ---------------------------------------------------------------------------
# Pretraining


def test_pretraining_is_deterministic_and_leaves_the_global_stream_untouched():
    torch.manual_seed(7)
    before = torch.get_rng_state()
    first = _pretrain(seed=42)
    after = torch.get_rng_state()
    assert torch.equal(before, after)
    second = _pretrain(seed=42)
    assert first.branch_digest == second.branch_digest
    assert all(torch.equal(first.branch_state[name], second.branch_state[name]) for name in first.branch_state)
    assert first.history == second.history and len(first.history) == 3
    assert _pretrain(seed=43).branch_digest != first.branch_digest
    assert first.settings["pretrain_seed"] == 42 + gp.WP_PRETRAIN_SEED_OFFSET != 42 + 7919


def test_pretraining_class_weights_follow_the_campaign_rule():
    vectors, labels = _small_vectors(n=30)
    labels = torch.tensor([0] * 15 + [2] * 10 + [6] * 5)
    outcome = gp.pretrain_whole_protein_classifier(vectors, labels, seed=1, embedding_dim=DIM, hidden_dim=HIDDEN,
                                                   branch_dropout=0.2, epochs=1, batch_size=64)
    expected = class_weights_from_labels(labels.tolist(), 7, mode="inverse_frequency")
    assert outcome.class_weights == pytest.approx(expected.tolist(), abs=0, rel=0)
    assert outcome.class_counts == {"1": 15, "2": 0, "3": 10, "4": 0, "5": 0, "6": 0, "7": 5}
    assert outcome.history[0]["optimizer_steps"] == 1


def test_f6_initialisation_equals_f5_and_the_branch_is_installed_frozen():
    f5 = _f6_model(42)
    torch.manual_seed(42)
    outcome = _pretrain(seed=42)
    f6 = build_pocket_classifier(model_architecture="only_esm", esm_dim=DIM, hidden_s=HIDDEN, predict_metal=False,
                                 predict_ec=True, n_ec=7, use_global_esm_branch=True, global_esm_dim=DIM)
    f5_state, f6_state = f5.state_dict(), f6.state_dict()
    assert all(torch.equal(f5_state[name], f6_state[name]) for name in f5_state)
    record = gp.install_pretrained_global_branch(f6, outcome)
    assert record["branch_digest"] == outcome.branch_digest == gp.branch_digest(f6.state_dict(), prefix=gp.BRANCH_PREFIX)
    after = f6.state_dict()
    assert all(torch.equal(f5_state[name], after[name]) for name in f5_state if not name.startswith(gp.BRANCH_PREFIX))
    assert not all(torch.equal(f5_state[name], after[name]) for name in f5_state if name.startswith(gp.BRANCH_PREFIX))
    assert all(not param.requires_grad for param in f6.global_esm_encoder.parameters())
    assert record["frozen_tensors"] == sorted(f"{gp.BRANCH_PREFIX}{name}" for name, _ in
                                              f6.global_esm_encoder.named_parameters())
    with pytest.raises(gp.GlobalEsmPretrainError, match="no whole-protein branch"):
        gp.install_pretrained_global_branch(build_pocket_classifier(
            model_architecture="only_esm", esm_dim=DIM, hidden_s=HIDDEN, predict_metal=False, n_ec=7), outcome)


def _graphs(n: int = 12, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return [Data(num_nodes=5, x_esm=torch.randn(5, DIM, generator=generator), x_global_esm=torch.randn(1, DIM, generator=generator),
                 x_dist_raw=torch.full((5, 3), 4.0), y_ec=torch.tensor([index % 7]), y_metal=torch.tensor([-1]),
                 ec_sample_weight=torch.ones(1), ec_group_id=torch.tensor([index])) for index in range(n)]


def test_a_frozen_branch_stays_bit_identical_through_a_tiny_fit():
    from training.run import build_optimizer_param_groups, realised_optimizer_steps

    model = _f6_model(42)
    outcome = _pretrain(seed=42)
    gp.install_pretrained_global_branch(model, outcome)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    config = replace(TrainConfig(), model_architecture="only_esm", use_global_esm_branch=True, learning_rate=1e-3,
                     epochs=2, lr_schedule="cosine")
    graphs = _graphs()
    batches = [Batch.from_data_list(graphs[start:start + 4]) for start in range(0, len(graphs), 4)]
    param_groups, record = build_optimizer_param_groups(model, config, steps_per_epoch=len(batches))
    assert [group["name"] for group in record["groups"]] == ["all"]
    optimizer = torch.optim.AdamW(param_groups)
    model.train()
    for _epoch in range(2):
        for batch in batches:
            loss = torch.nn.functional.cross_entropy(model(batch)["logits_ec"], batch.y_ec)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    after = model.state_dict()
    branch = [name for name in after if name.startswith(gp.BRANCH_PREFIX)]
    assert branch and all(torch.equal(before[name], after[name]) for name in branch)
    assert gp.branch_digest(after, prefix=gp.BRANCH_PREFIX) == outcome.branch_digest
    assert any(not torch.equal(before[name], after[name]) for name in after if name not in branch)
    frozen = list(model.global_esm_encoder.parameters())
    assert all(param not in optimizer.state for param in frozen)
    assert realised_optimizer_steps(optimizer) == [record["planned_total_steps"]] == [2 * len(batches)]


# ---------------------------------------------------------------------------
# The trainer hook and the C-GF control


def test_the_hook_writes_its_receipt_and_the_control_predictions(world, tmp_path):
    from training.ec_prediction_export import EC_PROTEIN_PREDICTION_COLUMNS

    pretraining, run_dir = _run(world, tmp_path, "unit_lr1", learning_rate=1e-4, global_esm_pretrain_epochs=4)
    receipt = json.loads((run_dir / gp.RECEIPT_NAME).read_text())
    assert receipt["split"] == "inner0" and receipt["corpus"]["sha256"] == world.corpus_sha
    assert receipt["cache"]["manifest_sha256"] == world.manifest_sha and len(receipt["history"]) == 4
    assert receipt["rows_used"] == len(world.rows) - 4 and receipt["rows_excluded_by_split"] == 4
    kept = [row for row in world.rows if not row.exclude[0]]
    expected = class_weights_from_labels([EC_CLASS_TOKEN_TO_INDEX[row.ec_class] for row in kept], 7,
                                         mode="inverse_frequency")
    assert receipt["class_weights"] == pytest.approx(expected.tolist(), abs=0, rel=0)
    assert receipt["branch_digest"] == pretraining.outcome.branch_digest
    with (run_dir / gp.CONTROL_PREDICTIONS_NAME).open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(EC_PROTEIN_PREDICTION_COLUMNS)
        rows = list(reader)
    _train, val = world.pockets(0)
    assert sorted(row["structure_ids"] for row in rows) == sorted(pocket.structure_id for pocket in val)
    # C-GF scores the corpus sequence means (rule A2-F6-7), not the pockets' structure chain means.
    accessions = [pocket.structure_id.split("__")[0] for pocket in val]
    vectors = torch.stack([world.vectors[_key(world.sequences[accession])] for accession in accessions])
    chain = torch.stack([pocket.metadata[GLOBAL_ESM_METADATA_KEY] for pocket in val])
    assert not torch.allclose(vectors, chain)
    logits = pretraining.outcome.logits(vectors)
    chain_logits = pretraining.outcome.logits(chain)
    by_sid = {row["structure_ids"]: row for row in rows}
    for index, pocket in enumerate(val):
        row = by_sid[pocket.structure_id]
        assert [float(row[f"logit_{token}"]) for token in "1234567"] == logits[index].tolist()
        assert [float(row[f"logit_{token}"]) for token in "1234567"] != chain_logits[index].tolist()
        assert row["mode"] == "validation" and row["fold_index"] == "0" and row["run_id"] == "unit_lr1"
    control_input = receipt["control_predictions"]["input"]
    assert control_input["source"] == "corpus_sequence_means" and control_input["n_proteins"] == len(val)
    assert control_input["manifest_sha256"] == world.manifest_sha
    # One pretraining thread, the fit's count restored, the CPU identity recorded.
    assert receipt["hyperparameters"]["num_threads"] == 1 and receipt["runtime"]["pretrain_num_threads"] == 1
    assert receipt["runtime"]["fit_num_threads"] == torch.get_num_threads()
    assert set(receipt["runtime"]["cpu_identity"]) == {"machine", "processor", "model_name"}
    # The second learning-rate setting of the same (inner fold, seed) pretrains identically.
    second, second_dir = _run(world, tmp_path, "unit_lr3", learning_rate=3e-4, global_esm_pretrain_epochs=4)
    assert second.outcome.branch_digest == pretraining.outcome.branch_digest
    assert second.receipt["control_predictions"]["content_sha256"] == receipt["control_predictions"]["content_sha256"]
    # A refit pretrains on every row and writes no control.
    refit, refit_dir = _run(world, tmp_path, "unit_refit", split="refit", global_esm_pretrain_epochs=2)
    assert refit.receipt["rows_used"] == len(world.rows) and refit.receipt["control_predictions"] is None
    assert not (refit_dir / gp.CONTROL_PREDICTIONS_NAME).exists()


def test_pretraining_runs_on_one_thread_and_restores_the_fit_count(monkeypatch):
    seen = []
    real = gp.F.cross_entropy

    def recording(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return real(*args, **kwargs)

    before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        monkeypatch.setattr(gp.F, "cross_entropy", recording)
        outcome = _pretrain(epochs=1)
        assert torch.get_num_threads() == 2 and set(seen) == {1}
        assert outcome.settings["num_threads"] == 1 and outcome.settings["fit_num_threads"] == 2
    finally:
        torch.set_num_threads(before)


def test_the_control_refuses_a_validation_protein_absent_from_the_corpus(world, tmp_path):
    rows = [row for row in world.rows if row.accession != "P0000"]
    world.corpus.write_bytes(gp.render_corpus(rows))
    world.corpus_sha = gp._sha256_file(world.corpus)
    with pytest.raises(gp.GlobalEsmPretrainError, match="absent from the pretraining corpus"):
        _run(world, tmp_path, "absent")


def test_the_hook_refuses_a_wrong_cache_or_split(world, tmp_path):
    with pytest.raises(gp.GlobalEsmPretrainError, match="do not load"):
        _run(world, tmp_path, "bad_cache", global_esm_pretrain_manifest_sha256="0" * 64)
    with pytest.raises(gp.GlobalEsmPretrainError, match="differs from the fit's inner fold"):
        _run(world, tmp_path, "bad_split", fold_index=1)
    with pytest.raises(gp.GlobalEsmPretrainError, match="pretrained_frozen"):
        _run(world, tmp_path, "random", global_esm_branch_init="random")


def _verified_unit(world: World, tmp_path: Path, name: str = "unit"):
    pretraining, run_dir = _run(world, tmp_path, name)
    model = _f6_model(42)
    gp.install_pretrained_global_branch(model, pretraining)
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "terminal_model_checkpoint.pt")
    identity = {"inner_fold": 0, "wp_pretrain_corpus_sha256": world.corpus_sha,
                "wp_pretrain_manifest_sha256": world.manifest_sha}
    _train, val = world.pockets(0)
    return run_dir, identity, [pocket.structure_id for pocket in val], model


def test_unit_verification_passes_and_catches_each_break(world, tmp_path):
    run_dir, identity, validation, model = _verified_unit(world, tmp_path)
    checks = gp.verify_wp_pretrain_unit(run_dir, identity=identity, kind="inner", validation_structure_ids=validation)
    assert set(checks) == {"wp_pretrain_receipt", "wp_branch_frozen", "wp_control_predictions"}
    assert all(item["ok"] for item in checks.values()), checks
    assert not gp.verify_wp_pretrain_unit(run_dir, identity={**identity, "wp_pretrain_corpus_sha256": "1" * 64},
                                          kind="inner", validation_structure_ids=validation)["wp_pretrain_receipt"]["ok"]
    assert not gp.verify_wp_pretrain_unit(run_dir, identity={**identity, "inner_fold": 1}, kind="inner",
                                          validation_structure_ids=validation)["wp_pretrain_receipt"]["ok"]
    assert not gp.verify_wp_pretrain_unit(run_dir, identity=identity, kind="inner",
                                          validation_structure_ids=validation[1:])["wp_control_predictions"]["ok"]
    state = model.state_dict()
    name = next(key for key in state if key.startswith(gp.BRANCH_PREFIX))
    state[name] = state[name] + 1e-3
    torch.save({"model_state_dict": state}, run_dir / "terminal_model_checkpoint.pt")
    assert not gp.verify_wp_pretrain_unit(run_dir, identity=identity, kind="inner",
                                          validation_structure_ids=validation)["wp_branch_frozen"]["ok"]
    (run_dir / gp.CONTROL_PREDICTIONS_NAME).unlink()
    assert not gp.verify_wp_pretrain_unit(run_dir, identity=identity, kind="inner",
                                          validation_structure_ids=validation)["wp_control_predictions"]["ok"]
    (run_dir / gp.RECEIPT_NAME).unlink()
    assert not any(item["ok"] for item in gp.verify_wp_pretrain_unit(
        run_dir, identity=identity, kind="inner", validation_structure_ids=validation).values())


# ---------------------------------------------------------------------------
# Trainer flags


SHA_A, SHA_B = "a" * 64, "b" * 64


def _f6_argv(split: str = "refit", *extra: str) -> list[str]:
    argv = ["--task", "ec", "--ec-label-depth", "1", "--ec-class-vocabulary", "fixed_seven", "--model-architecture",
            "only_esm", "--use-global-esm-branch", "--global-esm-cache-dir", "cache", "--global-esm-manifest-sha256",
            SHA_A, "--global-esm-branch-init", "pretrained_frozen", "--global-esm-pretrain-cache-dir", "pre",
            "--global-esm-pretrain-manifest-sha256", SHA_A, "--global-esm-pretrain-corpus", "corpus.tsv",
            "--global-esm-pretrain-corpus-sha256", SHA_B, "--global-esm-pretrain-split", split,
            "--no-prepare-missing-esm-embeddings"]
    if split != "refit":
        argv += ["--fold-split-source", "membership", "--fold-membership-csv", "m.csv", "--fold-membership-sha256",
                 SHA_A, "--n-folds", "4", "--fold-index", split.removeprefix("inner")]
    return argv + list(extra)


def _without(argv: list[str], flag: str, n_values: int = 1) -> list[str]:
    index = argv.index(flag)
    return argv[:index] + argv[index + 1 + n_values:]


def test_trainer_flags_accept_the_f6_configuration_and_keep_defaults():
    config = parse_args(_f6_argv("refit"))
    assert config.global_esm_branch_init == "pretrained_frozen" and config.global_esm_pretrain_split == "refit"
    assert (config.global_esm_pretrain_epochs, config.global_esm_pretrain_learning_rate,
            config.global_esm_pretrain_batch_size) == (30, 1e-3, 64)
    inner = parse_args(_f6_argv("inner2"))
    assert inner.global_esm_pretrain_split == "inner2" and inner.fold_index == 2
    default = parse_args(["--task", "ec"])
    assert default.global_esm_branch_init == "random" and default.global_esm_pretrain_corpus is None
    # Rule A2-GOV-5 (review S-B1/F-B1): the F6 option group leaves the payload while every option is at its default,
    # so every earlier resolved-configuration hash is unchanged; an F6 configuration carries the whole group.
    payload = config_to_payload(default)
    from training.config import F6_OPTION_DEFAULTS

    assert not set(F6_OPTION_DEFAULTS) & set(payload)
    f6_payload = config_to_payload(config)
    assert set(F6_OPTION_DEFAULTS) <= set(f6_payload) and f6_payload["global_esm_branch_init"] == "pretrained_frozen"
    assert isinstance(f6_payload["global_esm_pretrain_corpus"], str)


def test_trainer_flags_match_the_campaign_rule():
    from ec_clean30.campaign import F6_PRETRAINING

    defaults = TrainConfig()
    assert (F6_PRETRAINING["epochs"], F6_PRETRAINING["learning_rate"], F6_PRETRAINING["batch_size"]) == \
        (gp.WP_PRETRAIN_EPOCHS, gp.WP_PRETRAIN_LEARNING_RATE, gp.WP_PRETRAIN_BATCH_SIZE) == \
        (defaults.global_esm_pretrain_epochs, defaults.global_esm_pretrain_learning_rate,
         defaults.global_esm_pretrain_batch_size)
    assert (F6_PRETRAINING["weight_decay"], F6_PRETRAINING["grad_clip_norm"]) == \
        (gp.WP_PRETRAIN_WEIGHT_DECAY, gp.WP_PRETRAIN_GRAD_CLIP_NORM)


@pytest.mark.parametrize("argv, message", [
    pytest.param(_without(_f6_argv(), "--use-global-esm-branch", 0), "requires --use-global-esm-branch", id="no_branch"),
    pytest.param(_f6_argv("refit", "--model-architecture", "only_global"), "--model-architecture only_esm",
                 id="only_global"),
    pytest.param(_without(_f6_argv(), "--global-esm-pretrain-corpus-sha256"), "--global-esm-pretrain-corpus-sha256",
                 id="missing_corpus_sha"),
    pytest.param(_without(_f6_argv(), "--global-esm-pretrain-cache-dir"), "requires --global-esm-pretrain-cache-dir",
                 id="missing_cache"),
    pytest.param(_f6_argv("refit", "--global-esm-pretrain-corpus-sha256", "XYZ"), "64-character", id="bad_sha"),
    pytest.param(_f6_argv("refit", "--val-fraction", "0.2"), "runs without validation", id="refit_with_validation"),
    pytest.param(_f6_argv("inner1")[:-1] + ["2"], "--fold-index 1", id="split_fold_mismatch"),
    pytest.param(_without(_without(_f6_argv("inner0"), "--fold-split-source"), "--fold-index"),
                 "--fold-split-source membership", id="inner_without_split"),
    pytest.param(_f6_argv("refit", "--global-esm-layer", "block26"), "--global-esm-layer final", id="block_layer"),
    pytest.param(_f6_argv("refit", "--global-esm-standardize", "train_proteins"), "--global-esm-standardize none",
                 id="standardised"),
    pytest.param(_f6_argv("refit", "--run-test-eval"), "held-out test set", id="test_eval"),
    pytest.param(_f6_argv("refit", "--global-esm-pretrain-epochs", "0"), "must be >= 1", id="zero_epochs"),
    pytest.param(_f6_argv("refit", "--ec-class-vocabulary", "observed"), "fixed_seven", id="observed_vocabulary"),
    pytest.param(["--task", "ec", "--global-esm-pretrain-corpus", "corpus.tsv"],
                 "require --global-esm-branch-init pretrained_frozen", id="option_without_init"),
    pytest.param(["--task", "ec", "--global-esm-pretrain-batch-size", "32"],
                 "require --global-esm-branch-init pretrained_frozen", id="hyperparameter_without_init"),
])
def test_trainer_flags_refuse_broken_f6_configurations(argv, message, capsys):
    with pytest.raises(SystemExit):
        parse_args(argv)
    assert message in capsys.readouterr().err
