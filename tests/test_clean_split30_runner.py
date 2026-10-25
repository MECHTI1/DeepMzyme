"""Retrained CLEAN comparator runner (synthetic inputs only; no CLEAN environment or dataset access)."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import run_clean_split30_comparator as runner  # noqa: E402
from run_clean_split30_comparator import ComparatorError  # noqa: E402

STANDARD = "ACDEFGHIKLMNPQRSTVWY"
EMBEDDING = runner.DEFAULT_EMBEDDING
NAME0 = runner.training_name(0, "r1", EMBEDDING)
FOLD0_ROWS = [
    ("Q00001", "1.1.1.1", 80), ("Q00002", "1.1.1.1", 90), ("Q00003", "2.7.1.1", 70),
    ("Q00004", "2.7.1.1", 75), ("Q00005", "3.4.21.4", 60), ("Q00006", "6.3.2.1", 120),
    ("Q00007", "3.4.21.4;1.1.1.1", 85), ("Q00008", "3.1.1.-", 66),
    ("Q00012", "2.7.1.1;5.3.1.1", 70),  # its single-protein EC is not listed first
]
FOLD1_ROWS = [
    ("Q00008", "3.1.1.-", 66), ("Q00009", "3.1.1.-", 64), ("Q00006", "6.3.2.1", 120),
    ("Q00003", "2.7.1.1", 70), ("Q00010", "4.2.1.1", 50), ("Q00011", "4.2.1.1", 52),
]


@pytest.fixture
def space(tmp_path_factory):
    # Neutral directory names: the held-out rules refuse any path part such as "test_...".
    return tmp_path_factory.mktemp("clean_runner")


def sequence_for(accession, length):
    rng = random.Random(accession)
    return "".join(rng.choice(STANDARD) for _ in range(length))


def write_official(directory, fold, rows, name=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or runner.OFFICIAL_TRAIN_NAME.format(fold=fold))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(runner.TABLE_HEADER)
        for accession, ec, length in rows:
            writer.writerow((accession, ec, sequence_for(accession, length)))
    return path


def read_fasta(path):
    records, label = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            label = line[1:]
        else:
            records[label] = line
    return records


def write_farm(directory, items, *, embedding=EMBEDDING, dim=1280, layer=33, salt=""):
    """Synthetic ``<label>.pt`` files and their farm manifest; ``items`` maps label -> (seed, sequence)."""
    directory.mkdir(parents=True)
    entries = []
    for label, (seed, sequence) in items.items():
        path = directory / f"{label}.pt"
        path.write_bytes(hashlib.sha256(f"{label}:{embedding}:{seed}:{salt}".encode()).digest())
        entries.append(runner.FarmEntry(label, seed, runner.sha256_bytes(sequence.encode()), embedding, layer, dim,
                                        runner.sha256_file(path)))
    runner.write_farm_manifest(directory, entries)
    return directory


def base_items(rows):
    return {row.accession: (None, row.sequence) for row in rows}


def masked_items(rows, seed):
    singles = set(runner.single_protein_ec_accessions(rows))
    return {copy.label: (seed, copy.sequence) for row in rows if row.accession in singles
            for copy in runner.masked_copies(row.accession, row.sequence, seed)}


def write_farms(space, rows, seed, tag="", **options):
    """Base and masked farms of ``rows`` for ``seed``, as an embedding producer would leave them."""
    return (write_farm(space / f"esm_base{tag}", base_items(rows), **options),
            write_farm(space / f"esm_masked_seed{seed}{tag}", masked_items(rows, seed), **options))


# ---------------------------------------------------------------- masked copies

def test_masked_copies_are_keyed_by_accession_and_seed():
    sequence = sequence_for("P1", 200)
    first = runner.masked_copies("P1", sequence, 42)
    assert [copy.label for copy in first] == [f"P1_{j}" for j in range(runner.MASK_COPIES)]
    assert first == runner.masked_copies("P1", sequence, 42)
    other_seed = runner.masked_copies("P1", sequence, 43)
    other_accession = runner.masked_copies("P2", sequence, 42)
    assert [c.sequence for c in first] != [c.sequence for c in other_seed]
    assert [c.sequence for c in first] != [c.sequence for c in other_accession]
    assert len({copy.sequence for copy in first}) == runner.MASK_COPIES


def test_masks_are_identical_across_folds(space):
    official = space / "official"
    manifests = {}
    for fold, rows in ((0, FOLD0_ROWS), (1, list(reversed(FOLD1_ROWS)))):
        manifests[fold] = runner.prepare_fit(
            official_train_csv=write_official(official, fold, rows), fold=fold, regime="r1", seed=42,
            work_dir=space / f"work{fold}")
    fasta = {fold: read_fasta(space / f"work{fold}" / "masks" / manifests[fold]["masked_copies"]["fasta"])
             for fold in manifests}
    shared = [f"Q00006_{j}" for j in range(runner.MASK_COPIES)]
    assert [fasta[0][label] for label in shared] == [fasta[1][label] for label in shared]
    # 2.7.1.1 has one protein only in fold 1; 4.2.1.1 has two there.
    assert "Q00003_0" in fasta[1] and "Q00003_0" not in fasta[0]
    assert not any(label.startswith(("Q00010_", "Q00005_")) for labels in fasta.values() for label in labels)
    # 3.1.1.- has one protein in fold 0, so the incomplete EC also qualifies (upstream rule).
    assert "Q00008_0" in fasta[0] and "Q00008_0" not in fasta[1]
    # Any of a protein's EC numbers qualifies it, not only the first listed.
    assert "Q00012_0" in fasta[0]
    seed43 = runner.prepare_fit(official_train_csv=official / runner.OFFICIAL_TRAIN_NAME.format(fold=0), fold=0,
                                regime="r1", seed=43, work_dir=space / "work0_seed43")
    other = read_fasta(space / "work0_seed43" / "masks" / seed43["masked_copies"]["fasta"])
    assert other["Q00006_0"] != fasta[0]["Q00006_0"]


def test_mask_statistics_follow_the_published_procedure():
    length = 250
    copies = [copy for index in range(400)
              for copy in runner.masked_copies(f"S{index:04d}", sequence_for(f"S{index}", length), 7)]
    rates = [copy.rate for copy in copies]
    assert statistics.fmean(rates) == pytest.approx(runner.MASK_RATE_MEAN, abs=0.002)
    assert statistics.pstdev(rates) == pytest.approx(runner.MASK_RATE_SD, abs=0.002)
    positions = []
    for copy in copies:
        assert copy.n_draws == math.ceil(length * copy.rate)
        assert 1 <= min(copy.positions) and max(copy.positions) <= length - 1
        assert len(copy.positions) <= copy.n_draws
        assert copy.sequence.count(runner.MASK_TOKEN) == len(copy.positions)
        assert "*" not in copy.sequence
        positions.extend(copy.positions)
    assert any(len(copy.positions) < copy.n_draws for copy in copies)  # drawn with replacement
    counts = np.bincount(np.asarray(positions), minlength=length)
    assert counts[0] == 0
    assert counts[length - 1] > 0 and counts[1:].min() > 0  # every position 1..length-1 is reachable
    per_position = counts[1:].mean()
    assert all(abs(part.mean() - per_position) < 0.05 * per_position for part in np.array_split(counts[1:], 10))
    original = sequence_for("S0", length)
    restored = runner.masked_copies("S0000", original, 7)[0]
    tokens, residues, index = restored.sequence, [], 0
    while index < len(tokens):
        if tokens.startswith(runner.MASK_TOKEN, index):
            residues.append(None)
            index += len(runner.MASK_TOKEN)
        else:
            residues.append(tokens[index])
            index += 1
    assert len(residues) == length
    assert all(r is None or r == o for r, o in zip(residues, original))
    short = runner.masked_copies("T1", "MK", 3)
    assert all(set(copy.positions) <= {1} for copy in short)
    with pytest.raises(ComparatorError):
        runner.masked_copies("T2", "M", 3)


# ---------------------------------------------------------------- held-out refusals and table

def test_refuses_test_named_inputs_and_forbidden_roots(space):
    rows = FOLD0_ROWS[:3]
    held_out = write_official(space / "official", 0, rows, name="split30_test_split_0.csv")
    cases = [
        held_out,
        write_official(space / "CLEAN_30_main" / "metadata", 0, rows),
        write_official(space / "CLEAN_30_main" / "folds", 0, rows),
        write_official(space / "split30_test", 0, rows),
        write_official(space / "official_b", 0, rows, name=runner.OFFICIAL_TRAIN_NAME.format(fold=1)),
    ]
    disguised = space / "linked" / runner.OFFICIAL_TRAIN_NAME.format(fold=0)
    disguised.parent.mkdir()
    disguised.symlink_to(held_out)
    cases.append(disguised)
    for index, path in enumerate(cases):
        with pytest.raises(ComparatorError):
            runner.prepare_fit(official_train_csv=path, fold=0, regime="r1", seed=42,
                               work_dir=space / f"refused{index}")
    blocked_root = space / "blocked"
    inside_root = write_official(blocked_root, 0, rows)
    with pytest.raises(ComparatorError, match="forbidden root"):
        runner.prepare_fit(official_train_csv=inside_root, fold=0, regime="r1", seed=42,
                           work_dir=space / "refused_root", forbidden_roots=[blocked_root])
    good = write_official(space / "official_ok", 0, rows)
    with pytest.raises(ComparatorError, match="repository"):
        runner.prepare_fit(official_train_csv=good, fold=0, regime="r1", seed=42,
                           work_dir=ROOT / "clean_fit_never_created")
    assert not (ROOT / "clean_fit_never_created").exists()
    allowlist = space / "accessions_test.txt"
    allowlist.write_text("Q00001\n")
    with pytest.raises(ComparatorError, match="held-out name"):
        runner.prepare_fit(official_train_csv=good, fold=0, regime="r2", seed=42, work_dir=space / "refused_r2",
                           accession_allowlist=allowlist)
    with pytest.raises(ComparatorError, match="held-out name"):
        runner.refuse_held_out_path(space / "data" / "CLEAN_test_set.csv", [], "a file")


def test_training_table_normalisation_and_refusals(space):
    official = space / "official"
    path = official / runner.OFFICIAL_TRAIN_NAME.format(fold=2)
    official.mkdir()
    path.write_text("Entry\tEC number\tSequence\n"
                    " A1 \t 1.1.1.1 ; 1.1.1.1;2.7.1.1; \tMKV LLA\n"
                    "A2\t6.2.1.n2\tMKVQ\n"
                    "A3\t1.-.-.-\tMKVA\n")
    rows, report = runner.read_official_training_rows(path, 2, [])
    assert rows[0] == runner.TrainingRow("A1", ("1.1.1.1", "2.7.1.1"), "MKVLLA")
    assert [row.accession for row in rows] == ["A1", "A2", "A3"]
    assert report["n_rows_whitespace_or_separator_normalized"] == 1
    assert report["sha256"] == runner.sha256_file(path)
    for body, message in (("A1\t1.1.1.1\tMKV\nA1\t1.1.1.1\tMKV\n", "Duplicate Entry"),
                          ("A1\t8.1.1.1\tMKV\n", "EC class"), ("A1\t\tMKV\n", "Missing EC"),
                          ("A1\t1.1.1.1\tMK*V\n", "Invalid sequence"), ("A/1\t1.1.1.1\tMKV\n", "Invalid accession")):
        path.write_text("Entry\tEC number\tSequence\n" + body)
        with pytest.raises(ComparatorError, match=message):
            runner.read_official_training_rows(path, 2, [])
    good = write_official(space / "official_r2", 0, FOLD0_ROWS)
    allowlist = space / "metallo_accessions.txt"
    allowlist.write_text("# metallo subset\nQ00005\nQ00001\n\n")
    manifest = runner.prepare_fit(official_train_csv=good, fold=0, regime="r2", seed=42, work_dir=space / "r2",
                                  accession_allowlist=allowlist)
    table = (space / "r2" / manifest["training_table"]["file"]).read_text().splitlines()
    assert [line.split("\t")[0] for line in table] == ["Entry", "Q00001", "Q00005"]  # official order
    assert manifest["accession_allowlist"]["sha256"] == runner.sha256_file(allowlist)
    allowlist.write_text("Q00001\nQ99999\n")
    with pytest.raises(ComparatorError, match="not in the official training file"):
        runner.prepare_fit(official_train_csv=good, fold=0, regime="r2", seed=42, work_dir=space / "r2b",
                           accession_allowlist=allowlist)
    with pytest.raises(ComparatorError, match="r2"):
        runner.prepare_fit(official_train_csv=good, fold=0, regime="r1", seed=42, work_dir=space / "r1b",
                           accession_allowlist=allowlist)


def test_prepare_records_hashes_and_links_inputs(space):
    official = write_official(space / "official", 0, FOLD0_ROWS)
    rows, _ = runner.read_official_training_rows(official, 0, [])
    singles = sorted(runner.single_protein_ec_accessions(rows))
    base_dir, masked_dir = write_farms(space, rows, 42)
    manifest = runner.prepare_fit(official_train_csv=official, fold=0, regime="r1", seed=42, work_dir=space / "w",
                                  base_esm_dir=base_dir, masked_esm_dir=masked_dir)
    work = space / "w"
    assert manifest["training_name"] == NAME0 == f"split30_fold0_r1_{EMBEDDING}"
    assert manifest["official_training_file"]["sha256"] == runner.sha256_file(official)
    assert manifest["training_table"]["sha256"] == runner.sha256_file(work / "data" / f"{NAME0}.csv")
    assert manifest["training_table"]["n_proteins"] == len(FOLD0_ROWS)
    assert manifest["masked_copies"]["n_copies"] == runner.MASK_COPIES * len(singles)
    assert manifest["masked_copies"]["fasta_sha256"] == runner.sha256_file(
        work / "masks" / manifest["masked_copies"]["fasta"])
    farm = manifest["embedding_farm"]
    assert farm["complete"] is True
    assert farm["masked"]["n_files"] == runner.MASK_COPIES * len(singles)
    assert farm["base"]["farm_sha256"] == runner.embedding_farm_digest(
        {row.accession: runner.sha256_file(base_dir / f"{row.accession}.pt") for row in rows})
    assert farm["base"]["manifest_sha256"] == runner.sha256_file(base_dir / runner.FARM_MANIFEST)
    assert (farm["base"]["layer"], farm["masked"]["layer"], farm["base"]["dim"]) == (33, 33, 1280)
    assert farm["links_sha256"] == runner.sha256_file(work / runner.FARM_LINKS)
    link = work / "data" / "esm_data" / f"{singles[0]}_3.pt"
    assert link.is_symlink() and link.resolve() == (masked_dir / f"{singles[0]}_3.pt").resolve()
    assert json.loads((work / runner.PREPARE_MANIFEST).read_text()) == manifest
    (masked_dir / f"{singles[0]}_9.pt").unlink()
    with pytest.raises(ComparatorError, match="masked-copy embedding files are missing"):
        runner.prepare_fit(official_train_csv=official, fold=0, regime="r1", seed=42, work_dir=space / "w2",
                           base_esm_dir=base_dir, masked_esm_dir=masked_dir)
    with pytest.raises(ComparatorError, match="not empty"):
        runner.prepare_fit(official_train_csv=official, fold=0, regime="r1", seed=42, work_dir=work)


def test_prepare_refuses_embeddings_of_another_fit(space):
    official = write_official(space / "official", 0, FOLD0_ROWS)
    rows, _ = runner.read_official_training_rows(official, 0, [])
    base_dir, masked42 = write_farms(space, rows, 42)
    prepare = dict(official_train_csv=official, fold=0, regime="r1")
    altered = [runner.TrainingRow(row.accession, row.ec_numbers, row.sequence[::-1]) for row in rows]
    cases = [
        # seed 42's masks offered to a seed 43 fit
        (dict(seed=43, base_esm_dir=base_dir, masked_esm_dir=masked42), "seed"),
        # base embeddings of other sequences
        (dict(seed=42, base_esm_dir=write_farm(space / "esm_other_sequences", base_items(altered)),
              masked_esm_dir=masked42), "sequence"),
        (dict(seed=42, base_esm_dir=base_dir, masked_esm_dir=masked42, input_dim=960), "width"),
        (dict(seed=42, base_esm_dir=base_dir, masked_esm_dir=masked42, embedding="esmc_300m_mean_final"),
         "embedding"),
        (dict(seed=42, base_esm_dir=base_dir,
              masked_esm_dir=write_farm(space / "esm_masked_layer8", masked_items(rows, 42), layer=8)), "layer"),
    ]
    for index, (options, message) in enumerate(cases):
        with pytest.raises(ComparatorError, match=message):
            runner.prepare_fit(**prepare, **options, work_dir=space / f"refused{index}")
    changed = write_farm(space / "esm_changed", base_items(rows))
    (changed / f"{rows[0].accession}.pt").write_bytes(b"replaced after the manifest")
    with pytest.raises(ComparatorError, match="file hash"):
        runner.prepare_fit(**prepare, seed=42, base_esm_dir=changed, masked_esm_dir=masked42,
                           work_dir=space / "refused_changed")
    bare = space / "esm_without_manifest"
    bare.mkdir()
    for row in rows:
        (bare / f"{row.accession}.pt").write_bytes(b"x")
    with pytest.raises(ComparatorError, match=runner.FARM_MANIFEST):
        runner.prepare_fit(**prepare, seed=42, base_esm_dir=bare, masked_esm_dir=masked42,
                           work_dir=space / "refused_bare")
    with pytest.raises(ComparatorError, match="unique"):
        runner.write_farm_manifest(space, [runner.FarmEntry("A1", None, "a" * 64, EMBEDDING, 33, 1280, "b" * 64)] * 2)
    for embedding in ("esm/1b", "esm_test_layer", "", "a b"):
        with pytest.raises(ComparatorError, match="Embedding name"):
            runner.training_name(0, "r1", embedding)


def test_names_separate_embeddings_and_seeds():
    final, layer8 = "esmc_300m_mean_final", "esmc_300m_mean_layer8"
    assert runner.training_name(0, "r1", final) != runner.training_name(0, "r1", layer8)
    regimes = [(regime, None) for regime in runner.OUTER_REGIMES] + [
        (regime, inner) for regime in runner.INNER_REGIMES for inner in range(runner.N_INNER_FOLDS)]
    names = {runner.model_name(fold, regime, embedding, seed, 960, inner) for fold in (0, 1) for regime, inner in regimes
             for embedding in (final, layer8) for seed in (42, 43)}
    assert len(names) == 2 * len(regimes) * 2 * 2
    assert runner.model_name(0, "r1", EMBEDDING, 42, 1280) == f"clean_split30_fold0_r1_{EMBEDDING}_seed42_d1280"
    assert runner.model_name(0, "r1_inner", EMBEDDING, 42, 1280, 3) == \
        f"clean_split30_fold0_r1_inner3_{EMBEDDING}_seed42_d1280"
    for regime, inner in (("r1", 0), ("r1_inner", None), ("r2_inner", 4), ("r2_inner", True), ("r3", None)):
        with pytest.raises(ComparatorError):
            runner.training_name(0, regime, EMBEDDING, inner)


def test_distance_map_links_need_the_same_embeddings(space):
    official = write_official(space / "official", 0, FOLD0_ROWS)
    rows, _ = runner.read_official_training_rows(official, 0, [])
    base_dir, masked42 = write_farms(space, rows, 42)
    table_sha256 = runner.prepare_fit(official_train_csv=official, fold=0, regime="r1", seed=42,
                                      work_dir=space / "w0")["training_table"]["sha256"]
    farm_sha256 = runner.embedding_farm_digest(
        {row.accession: runner.sha256_file(base_dir / f"{row.accession}.pt") for row in rows})
    prepare = dict(official_train_csv=official, fold=0, regime="r1", seed=42)
    farms = dict(base_esm_dir=base_dir, masked_esm_dir=masked42)

    good = write_fake_distance_map(space / "maps_good", NAME0, table_sha256, farm_sha256)
    with pytest.raises(ComparatorError, match="embedding farm"):
        runner.prepare_fit(**prepare, work_dir=space / "no_farm", distance_map_dir=good)
    assert not (space / "no_farm").exists()
    linked = runner.prepare_fit(**prepare, **farms, work_dir=space / "w1", distance_map_dir=good)
    assert linked["distance_map"]["outputs"]["distance_map"]["file"] == f"{NAME0}.pkl"
    assert linked["distance_map"]["embedding_farm_sha256"] == linked["embedding_farm"]["base"]["farm_sha256"]
    assert (space / "w1" / "data" / "distance_map" / f"{NAME0}.pkl").is_symlink()

    other_base = write_farm(space / "esm_base_other", base_items(rows), salt="other")
    with pytest.raises(ComparatorError, match="other base embeddings"):
        runner.prepare_fit(**prepare, base_esm_dir=other_base, masked_esm_dir=masked42, work_dir=space / "w2",
                           distance_map_dir=good)
    bad_maps = [
        (dict(table_sha256="0" * 64), "another training table"),
        (dict(dim=960), "width"),
        (dict(helper_sha256="f" * 64), "another helper"),
        (dict(hash_seed="1234"), "PYTHONHASHSEED"),
    ]
    for index, (override, message) in enumerate(bad_maps):
        options = {"table_sha256": table_sha256, "farm_sha256": farm_sha256, **override}
        directory = write_fake_distance_map(space / f"maps_bad{index}", NAME0, options.pop("table_sha256"),
                                            options.pop("farm_sha256"), **options)
        with pytest.raises(ComparatorError, match=message):
            runner.prepare_fit(**prepare, **farms, work_dir=space / f"w_bad{index}", distance_map_dir=directory)


def write_fake_distance_map(directory, name, table_sha256, farm_sha256, *, dim=1280,
                            helper_sha256=runner.DISTANCE_MAP_HELPER_SHA256, hash_seed="0"):
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for key, suffix in (("distance_map", ".pkl"), ("esm_embeddings", "_esm.pkl")):
        path = directory / f"{name}{suffix}"
        path.write_bytes(f"{key}:{name}".encode())
        outputs[key] = {"file": path.name, "sha256": runner.sha256_file(path)}
    (directory / f"{name}_receipt.json").write_text(json.dumps(
        {"training_table_sha256": table_sha256, "outputs": outputs, "pythonhashseed": hash_seed,
         "embedding_farm_sha256": farm_sha256, "esm_embeddings_shape": [12, dim], "helper_sha256": helper_sha256}))
    return directory


# ---------------------------------------------------------------- contract C4

METADATA = {"fold": 0, "regime": "r1", "seed": 42, "input_dim": 1280, "embedding": runner.DEFAULT_EMBEDDING,
            "mode": "validation", "checkpoint_sha256": "a" * 64, "training_table_sha256": "b" * 64,
            "embedding_farm_sha256": "c" * 64, "query_embeddings_sha256": "d" * 64}


def test_distance_export_round_trip_and_validation(space):
    distances = np.array([[0.1, 0.5, 0.9], [0.7, 0.2, 0.4]], dtype=np.float32)
    queries, labels = ["Q1", "Q2"], ["1.1.1.1", "2.7.1.1", "6.2.1.n2"]
    npz = space / "fold0_r1_seed42.npz"
    receipt = runner.write_distance_export(npz, distances, queries, labels, METADATA)
    assert runner.c4_receipt_path(npz).is_file()
    assert receipt["contract"] == runner.C4_CONTRACT and receipt["n_query"] == 2 and receipt["n_ec"] == 3
    export = runner.read_distance_export(npz)
    assert export.distances.dtype == np.float32 and np.array_equal(export.distances, distances)
    assert export.query_ids == ("Q1", "Q2") and export.ec_labels == tuple(labels)
    assert export.receipt["seed"] == 42
    assert {field: export.receipt[field] for field in runner.C4_METADATA_FIELDS} == METADATA
    with pytest.raises(ComparatorError, match="already exists"):
        runner.write_distance_export(npz, distances, queries, labels, METADATA)

    bad_inputs = [
        (distances.astype(np.float64), queries, labels, METADATA, "float32"),
        (np.array([[np.nan, 0, 0], [0, 0, 0]], dtype=np.float32), queries, labels, METADATA, "finite"),
        (-distances, queries, labels, METADATA, "non-negative"),
        (distances, ["Q1", "Q1"], labels, METADATA, "duplicates"),
        (distances, queries, ["1.1.1.1", "2.7.1.1", "9.9.9.9"], METADATA, "EC class"),
        (distances[:, :2], queries, labels, METADATA, "shape"),
        (distances, queries, labels, {**METADATA, "mode": "final"}, "mode"),
        (distances, queries, labels, {k: v for k, v in METADATA.items() if k != "checkpoint_sha256"}, "lacks"),
        (distances, queries, labels, {k: v for k, v in METADATA.items() if k != "query_embeddings_sha256"},
         "lacks"),
        (distances, queries, labels, {**METADATA, "embedding_farm_sha256": "xyz"}, "not a SHA-256"),
    ]
    for index, (values, ids, ecs, metadata, message) in enumerate(bad_inputs):
        with pytest.raises(ComparatorError, match=message):
            runner.write_distance_export(space / f"bad{index}.npz", values, ids, ecs, metadata)

    tampered = space / "tampered.npz"
    runner.write_distance_export(tampered, distances, queries, labels, METADATA)
    tampered.unlink()
    with tampered.open("xb") as handle:
        np.savez(handle, distances=distances * 2, query_ids=np.array(queries), ec_labels=np.array(labels))
    with pytest.raises(ComparatorError, match="does not match its receipt"):
        runner.read_distance_export(tampered)

    extra = space / "extra.npz"
    runner.write_distance_export(extra, distances, queries, labels, METADATA)
    extra.unlink()
    with extra.open("xb") as handle:
        np.savez(handle, distances=distances, query_ids=np.array(queries), ec_labels=np.array(labels),
                 labels=np.array(["3", "1"]))
    receipt_path = runner.c4_receipt_path(extra)
    payload = json.loads(receipt_path.read_text())
    payload["npz_sha256"] = runner.sha256_file(extra)
    receipt_path.write_text(json.dumps(payload))
    with pytest.raises(ComparatorError, match="exactly"):
        runner.read_distance_export(extra)


def test_distance_export_conditional_fields(space):
    distances = np.array([[0.1, 0.5]], dtype=np.float32)
    queries, labels = ["Q1"], ["1.1.1.1", "2.7.1.1"]
    inner = {**METADATA, "regime": "r1_inner", "inner_fold": 2, "n_in_sample_queries": 0}
    runner.write_distance_export(space / "inner.npz", distances, queries, labels, inner)
    assert runner.read_distance_export(space / "inner.npz").receipt["inner_fold"] == 2
    binding = {"freeze2_manifest_sha256": "e" * 64, "test_pass_output_root": str(space / "pass_root"),
               "runner_sha256": "f" * 64, "export_script_sha256": runner.EXPORT_SCRIPT_SHA256}
    bound = {**METADATA, "mode": "test", "n_in_sample_queries": 0, **binding}
    receipt = runner.write_distance_export(space / "bound.npz", distances, queries, labels, bound)
    assert {field: receipt[field] for field in binding} == binding
    assert runner.read_distance_export(space / "bound.npz").receipt == receipt
    runner.check_test_export_binding(receipt, freeze2_sha256="E" * 64, test_pass_output_root=space / "pass_root",
                                     runner_sha256="f" * 64)
    # A test export written without the binding (as a hand-made fixture would be) is readable but not bound.
    unbound = runner.write_distance_export(space / "unbound.npz", distances, queries, labels, {**METADATA, "mode": "test"})
    with pytest.raises(ComparatorError, match="no Freeze-2 binding"):
        runner.check_test_export_binding(unbound, freeze2_sha256="e" * 64, test_pass_output_root=space / "pass_root")
    with pytest.raises(ComparatorError, match="Not a test export"):
        runner.check_test_export_binding(METADATA, freeze2_sha256="e" * 64, test_pass_output_root=space / "pass_root")
    bad = [
        ({**METADATA, "regime": "r1_inner"}, "needs inner fold"),
        ({**METADATA, "inner_fold": 1}, "takes no inner fold"),
        ({**inner, "inner_fold": 7}, "outside"),
        ({**inner, "mode": "test"}, "validation exports only"),
        ({**METADATA, "mode": "test", "n_in_sample_queries": 2}, "n_in_sample_queries"),
        ({**METADATA, "n_in_sample_queries": -1}, "n_in_sample_queries"),
        ({**METADATA, **binding}, "belongs to test exports"),
        ({**bound, "runner_sha256": None}, "all fields together"),
        ({**bound, "test_pass_output_root": "pass_root"}, "absolute"),
        ({**bound, "export_script_sha256": "xyz"}, "not SHA-256"),
    ]
    for index, (metadata, message) in enumerate(bad):
        with pytest.raises(ComparatorError, match=message):
            runner.write_distance_export(space / f"bad{index}.npz", distances, queries, labels, metadata)
        assert not (space / f"bad{index}.npz").exists()


# ---------------------------------------------------------------- projection and ensemble

def test_projection_applies_the_tie_rule():
    labels = ["1.1.1.1", "2.7.1.1", "1.2.3.4", "6.2.1.n2", "1.-.-.-"]
    big = 5.0
    distances = np.array([
        [0.5, 0.9, 0.7, big, big],          # clear nearest, class 1
        [0.3, 0.3 + 5e-7, 0.9, big, big],   # different class within 1e-6: tie, wrong
        [0.3, 0.9, 0.3 + 5e-7, big, big],   # same class within 1e-6: class 1
        [0.4, 0.4 + 2e-6, 0.9, big, big],   # beyond the tolerance: class 1
        [0.2, 0.2, 0.9, big, big],          # exact tie across classes: wrong
        [big, big, big, 0.1, 0.1],          # exact tie across classes 6 and 1: wrong
        [big, big, big, 0.1, 0.3],          # class 6 from a non-numeric fourth field
    ], dtype=np.float32)
    queries = [f"Q{i}" for i in range(len(distances))]
    projection = runner.project_ec_classes(distances, queries, labels)
    assert projection.predicted_class == ("1", runner.TIE_CLASS, "1", "1", runner.TIE_CLASS, runner.TIE_CLASS, "6")
    assert projection.tie.tolist() == [False, True, False, False, True, True, False]
    assert projection.n_ties == 3
    assert projection.nearest_ec[0] == "1.1.1.1" and projection.nearest_ec[6] == "6.2.1.n2"
    truth = {query: "1" for query in queries} | {"Q6": "6"}
    assert runner.projection_correct(projection, truth).tolist() == [True, False, True, True, False, False, True]
    assert runner.ec_class_of_label("1.-.-.-") == "1" and runner.ec_class_of_label("7.1.1.1") == "7"
    for label in ("x.1.1.1", "8.1.1.1", ""):
        with pytest.raises(ComparatorError):
            runner.ec_class_of_label(label)


def make_export(seed, queries, labels, rows, **overrides):
    receipt = {**METADATA, "seed": seed, **overrides}
    return runner.DistanceExport(np.asarray(rows, dtype=np.float32), tuple(queries), tuple(labels), receipt)


def test_seed_ensemble_averages_distances_before_the_arg_min():
    labels = ["1.1.1.1", "2.7.1.1", "3.4.21.4"]
    a = make_export(42, ["Q1", "Q2"], labels, [[0.1, 0.2, 0.9], [0.8, 0.6, 0.1]])
    b = make_export(43, ["Q1", "Q2"], labels, [[0.1, 0.2, 0.9], [0.9, 0.7, 0.2]])
    # Seed 44 lists queries and labels in another order; aligned by name.
    c = make_export(44, ["Q2", "Q1"], list(reversed(labels)), [[0.3, 0.5, 0.7], [0.9, 0.2, 0.9]])
    distances, queries, ec_labels = runner.ensemble_distances([a, b, c])
    assert queries == ("Q1", "Q2") and ec_labels == tuple(labels)
    expected = np.array([[(0.1 + 0.1 + 0.9) / 3, (0.2 + 0.2 + 0.2) / 3, (0.9 + 0.9 + 0.9) / 3],
                         [(0.8 + 0.9 + 0.7) / 3, (0.6 + 0.7 + 0.5) / 3, (0.1 + 0.2 + 0.3) / 3]])
    np.testing.assert_allclose(distances, expected, rtol=1e-6)
    projection = runner.project_ensemble([a, b, c])
    # Two of three seeds put Q1 nearest to class 1, but the mean distance favours class 2.
    assert [runner.project_ec_classes(e.distances, e.query_ids, e.ec_labels).predicted_class[
        e.query_ids.index("Q1")] for e in (a, b, c)] == ["1", "1", "2"]
    assert projection.predicted_class == ("2", "3") and projection.n_ties == 0
    with pytest.raises(ComparatorError, match="repeat a seed"):
        runner.ensemble_distances([a, make_export(42, ["Q1", "Q2"], labels, a.distances)])
    for field, value in (("fold", 1), ("training_table_sha256", "e" * 64), ("embedding_farm_sha256", "e" * 64),
                         ("query_embeddings_sha256", "e" * 64)):
        with pytest.raises(ComparatorError, match=f"differ in.*{field}"):
            runner.ensemble_distances([a, make_export(43, ["Q1", "Q2"], labels, a.distances, **{field: value})])
    with pytest.raises(ComparatorError, match="other queries or EC labels"):
        runner.ensemble_distances([a, make_export(43, ["Q1", "Q2"], ["1.1.1.1", "2.7.1.1", "4.2.1.1"],
                                                  a.distances)])
    with pytest.raises(ComparatorError, match="at least two"):
        runner.ensemble_distances([a])


# ---------------------------------------------------------------- launch (subprocess mocked)

@pytest.fixture
def fit_setup(space, monkeypatch):
    clone = space / "clean_checkout"
    expected = {}
    for rel in runner.UPSTREAM_FILE_SHA256:
        path = clone / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic stand-in for {rel}\n")
        expected[rel] = runner.sha256_file(path)
    monkeypatch.setattr(runner, "UPSTREAM_FILE_SHA256", expected)
    monkeypatch.setattr(runner, "_git_head", lambda path: runner.CLEAN_COMMIT)
    scripts = space / "private"
    scripts.mkdir()
    training_script, helper = scripts / "modified_train.py", scripts / "distance_helper.py"
    training_script.write_text("# synthetic training script\n")
    helper.write_text("# synthetic helper\n")
    official = write_official(space / "official", 0, FOLD0_ROWS)
    rows, _ = runner.read_official_training_rows(official, 0, [])
    base_dir, masked_dir = write_farms(space, rows, 42)
    work = space / "fit"
    runner.prepare_fit(official_train_csv=official, fold=0, regime="r1", seed=42, work_dir=work,
                       base_esm_dir=base_dir, masked_esm_dir=masked_dir)
    python = space / "clean_env" / "bin" / "python"
    kwargs = dict(work_dir=work, clean_python=python, clean_clone=clone, training_script=training_script,
                  distance_helper=helper, training_script_sha256=runner.sha256_file(training_script),
                  distance_helper_sha256=runner.sha256_file(helper), epochs=200, adaptive_rate=100, threads=2)
    return {"work": work, "clone": clone, "python": python, "kwargs": kwargs, "space": space, "base": base_dir}


class FakeClean:
    """Records launches and writes the outputs the real CLEAN processes would leave."""

    def __init__(self, work, *, hash_seed_override=None, leftover=False):
        self.calls, self.work = [], work
        self.hash_seed_override, self.leftover = hash_seed_override, leftover

    def __call__(self, command, *, cwd, env, log_path):
        self.calls.append({"command": list(command), "cwd": Path(cwd), "env": dict(env)})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("synthetic log\n")
        manifest = json.loads((self.work / runner.PREPARE_MANIFEST).read_text())
        name = manifest["training_name"]
        args = command[5:]
        if "--model_name" not in args:
            write_fake_distance_map(self.work / "data" / "distance_map", name, manifest["training_table"]["sha256"],
                                    manifest["embedding_farm"]["base"]["farm_sha256"],
                                    dim=manifest["input_dim"], helper_sha256=runner.sha256_file(command[4]),
                                    hash_seed=env["PYTHONHASHSEED"])
            return 0
        model = args[args.index("--model_name") + 1]
        epochs = int(args[args.index("--epoch") + 1])
        model_dir = self.work / "data" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / f"{model}.pth").write_bytes(b"weights")
        (model_dir / f"{model}_loss_log.jsonl").write_text("".join(
            json.dumps({"epoch": epoch, "train_loss": 1.0 / epoch}) + "\n" for epoch in range(1, epochs + 1)))
        if self.leftover:
            (model_dir / f"{model}_{epochs}.pth").write_bytes(b"left over")
        receipt = {"seed": int(args[args.index("--seed") + 1]), "epochs": epochs,
                   "adaptive_rate": int(args[args.index("--adaptive_rate") + 1]),
                   "input_dim": int(args[args.index("--input_dim") + 1]), "model_name": model,
                   "launch_env": {"PYTHONHASHSEED": self.hash_seed_override or env["PYTHONHASHSEED"]},
                   "final_weights_sha256": runner.sha256_file(model_dir / f"{model}.pth")}
        (model_dir / f"{model}_receipt.json").write_text(json.dumps(receipt))
        return 0


def test_fit_builds_guarded_commands_and_environment(fit_setup, monkeypatch):
    monkeypatch.setenv("PYTHONSTARTUP", "/somewhere/startup.py")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    fake = FakeClean(fit_setup["work"])
    monkeypatch.setattr(runner, "_run", fake)
    work, clone, kwargs = fit_setup["work"], fit_setup["clone"], fit_setup["kwargs"]

    plan = runner.run_fit(**kwargs, dry_run=True)
    assert fake.calls == [] and [step["step"] for step in plan["steps"]] == ["distance_map", "train"]

    fit = runner.run_fit(**kwargs)
    assert [call["cwd"] for call in fake.calls] == [work, work]
    distance, train = fake.calls
    for call in fake.calls:
        command, env = call["command"], call["env"]
        assert command[:4] == [str(fit_setup["python"]), "-c", runner.GUARDED_LAUNCH, str(runner.ACCESS_GUARD_FILE)]
        assert env["PYTHONNOUSERSITE"] == "1" and env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "2" and env["MKL_THREADING_LAYER"] == "GNU"
        assert env["PYTHONPATH"] == str(clone.resolve() / "app" / "src")
        assert "PYTHONSTARTUP" not in env
        roots = env[runner.FORBIDDEN_READ_ROOTS_ENV].split(os.pathsep)
        assert str((clone / "app" / "data").resolve()) in roots and str((clone / "app" / "results").resolve()) in roots
        assert str(runner.DEFAULT_FORBIDDEN_ROOTS[0].resolve()) in roots
    assert distance["command"][4] == str(kwargs["distance_helper"])
    assert distance["command"][5:] == ["--training_data", NAME0]
    assert distance["env"]["PYTHONHASHSEED"] == str(runner.DISTANCE_MAP_HASH_SEED)
    assert train["command"][4] == str(kwargs["training_script"])
    assert train["env"]["PYTHONHASHSEED"] == "42"
    args = train["command"][5:]
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs == {"--training_data": NAME0, "--model_name": f"clean_split30_fold0_r1_{EMBEDDING}_seed42_d1280",
                     "--epoch": "200", "--learning_rate": "0.0005", "--hidden_dim": "512", "--out_dim": "128",
                     "--adaptive_rate": "100", "--seed": "42", "--input_dim": "1280"}
    written = json.loads((work / runner.FIT_MANIFEST).read_text())
    assert written == json.loads(json.dumps(fit))
    assert written["scripts"]["training_script"]["sha256"] == kwargs["training_script_sha256"]
    assert written["clean_clone"]["commit"] == runner.CLEAN_COMMIT
    assert written["prepare_manifest_sha256"] == runner.sha256_file(work / runner.PREPARE_MANIFEST)
    assert written["steps"][1]["env"]["PYTHONHASHSEED"] == "42"
    prepared = json.loads((work / runner.PREPARE_MANIFEST).read_text())
    assert written["embedding_farm_verified"] == {kind: prepared["embedding_farm"][kind]["farm_sha256"]
                                                  for kind in runner.FARM_KINDS}
    assert written["distance_map"]["embedding_farm_sha256"] == prepared["embedding_farm"]["base"]["farm_sha256"]
    with pytest.raises(ComparatorError, match="already ran"):
        runner.run_fit(**kwargs)


def test_fit_refusals(fit_setup, monkeypatch):
    kwargs = fit_setup["kwargs"]
    monkeypatch.setattr(runner, "_run", FakeClean(fit_setup["work"]))
    for epochs in (250, 50):
        with pytest.raises(ComparatorError, match="positive multiple"):
            runner.run_fit(**{**kwargs, "epochs": epochs, "adaptive_rate": 100})
    with pytest.raises(ComparatorError, match="positive multiple"):
        runner.build_train_args(name="n", model="m", epochs=2500, adaptive_rate=0, seed=1, input_dim=1280)
    with pytest.raises(ComparatorError, match="SHA-256"):
        runner.run_fit(**{**kwargs, "training_script_sha256": "f" * 64})
    with pytest.raises(ComparatorError, match="repository"):
        runner.run_fit(**{**kwargs, "clean_clone": ROOT})
    monkeypatch.setattr(runner, "_git_head", lambda path: "0" * 40)
    with pytest.raises(ComparatorError, match="expected f2bf2a4"):
        runner.run_fit(**kwargs)
    monkeypatch.setattr(runner, "_git_head", lambda path: runner.CLEAN_COMMIT)
    (fit_setup["clone"] / "app" / "src" / "CLEAN" / "utils.py").write_text("# changed\n")
    with pytest.raises(ComparatorError, match="differ from commit"):
        runner.run_fit(**kwargs)
    held_out = fit_setup["clone"] / "app" / "data"
    held_out.mkdir()
    with pytest.raises(ComparatorError, match="app/data"):
        runner.verify_clean_clone(fit_setup["clone"])
    held_out.rmdir()

    space = fit_setup["space"]
    no_farm = space / "no_farm"
    runner.prepare_fit(official_train_csv=space / "official" / runner.OFFICIAL_TRAIN_NAME.format(fold=0), fold=0,
                       regime="r1", seed=42, work_dir=no_farm)
    with pytest.raises(ComparatorError, match="embedding farm"):
        runner.run_fit(**{**kwargs, "work_dir": no_farm})
    # A linked embedding replaced after prepare is caught before any launch.
    (fit_setup["base"] / f"{FOLD0_ROWS[0][0]}.pt").write_bytes(b"replaced after prepare")
    with pytest.raises(ComparatorError, match="changed after prepare"):
        runner.run_fit(**kwargs, dry_run=True)


@pytest.mark.parametrize("fake_options, message", [
    ({"hash_seed_override": "1234"}, "PYTHONHASHSEED"),
    ({"leftover": True}, "Fit outputs differ"),
])
def test_fit_rejects_inconsistent_training_outputs(fit_setup, monkeypatch, fake_options, message):
    monkeypatch.setattr(runner, "_run", FakeClean(fit_setup["work"], **fake_options))
    with pytest.raises(ComparatorError, match=message):
        runner.run_fit(**fit_setup["kwargs"])
    assert not (fit_setup["work"] / runner.FIT_MANIFEST).exists()


def test_guarded_launch_installs_the_read_guard(space):
    forbidden = space / "blocked"
    forbidden.mkdir()
    (forbidden / "rows.csv").write_text("x\n")
    script = space / "child.py"
    script.write_text("import sys\nprint(sys.argv[1:])\nopen(sys.argv[1]).read()\n")
    command = runner.guarded_command(Path(sys.executable), script, [str(forbidden / "rows.csv")])
    env = {**os.environ, runner.FORBIDDEN_READ_ROOTS_ENV: str(forbidden.resolve())}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "PermissionError" in result.stderr
    assert str(forbidden / "rows.csv") in result.stdout  # the script ran as __main__ with its own argv
    unguarded = {key: value for key, value in os.environ.items() if key != runner.FORBIDDEN_READ_ROOTS_ENV}
    result = subprocess.run(command, env=unguarded, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "without forbidden-read roots" in result.stderr


def test_pins_and_default_roots():
    for pin in (runner.TRAINING_SCRIPT_SHA256, runner.TRAINING_PATCH_SHA256, runner.DISTANCE_MAP_HELPER_SHA256,
                runner.EXPORT_SCRIPT_SHA256):
        assert runner.SHA256_RE.match(pin) and pin != "0" * 64
    assert runner.TRAINING_PATCH_SHA256 in runner.__doc__
    # Exact constructed held-out paths (path arithmetic only; nothing is resolved or opened here).
    for fold in range(runner.N_FOLDS):
        assert runner.SPLIT30_DIR / f"split30_test_split_{fold}_curate.csv" in runner.DEFAULT_FORBIDDEN_ROOTS
        assert runner.DATA_DIR / f"CLEAN_30_train_test_split_{fold}" / "test" in runner.DEFAULT_FORBIDDEN_ROOTS


def test_runner_never_imports_clean():
    tree = ast.parse((ROOT / "scripts" / "run_clean_split30_comparator.py").read_text())
    imported = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imported += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not [name for name in imported if name.split(".")[0] == "CLEAN"]
    assert "CLEAN" not in runner.GUARDED_LAUNCH


# ---------------------------------------------------------------- training-only sequence map

# (accession, EC field, length); protein i is left out of fold (i % 5)'s file, so each is in 4 files.
SPLIT30_PROTEINS = [
    ("Q00001", "1.1.1.1", 80), ("Q00002", "1.1.1.1", 90), ("Q00003", "2.7.1.1", 70),
    ("Q00004", "2.7.1.1;5.3.1.1", 75), ("Q00005", "3.4.21.4", 60), ("Q00006", "6.3.2.1", 120),
    ("Q00007", "3.1.1.-", 66), ("Q00008", "4.2.1.1", 50), ("Q00009", "4.2.1.1", 52), ("Q00010", "6.2.1.n2", 64),
]


def write_split30(directory, proteins=SPLIT30_PROTEINS, *, edits=None):
    """Five official training files; ``edits`` maps (fold, accession) to an (EC field, sequence) override."""
    directory.mkdir(parents=True, exist_ok=True)
    edits = edits or {}
    for fold in range(runner.N_FOLDS):
        with (directory / runner.OFFICIAL_TRAIN_NAME.format(fold=fold)).open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(runner.TABLE_HEADER)
            for index, (accession, ec, length) in enumerate(proteins):
                if index % runner.N_FOLDS != fold:
                    writer.writerow((accession, *edits.get((fold, accession), (ec, sequence_for(accession, length)))))
    return directory


def split30_sequences(proteins=SPLIT30_PROTEINS):
    return {accession: sequence_for(accession, length) for accession, _, length in proteins}


def test_sequence_map_round_trip(space):
    from embed_helpers.esmc_chain_means import read_sequence_inputs, read_sequence_map

    long = sequence_for("Q00001", 80)
    # Whitespace, an empty field and a repeated EC number are normalised, so these rows agree with the rest.
    noisy = {(1, "Q00004"): (" 2.7.1.1 ;2.7.1.1;5.3.1.1; ", sequence_for("Q00004", 75)),
             (2, "Q00001"): ("1.1.1.1", f"{long[:40]} {long[40:]}")}
    split30 = write_split30(space / "split30", edits=noisy)
    out = space / "seqmap"
    receipt = runner.write_sequence_map(split30_dir=split30, out_dir=out)
    expected = split30_sequences()
    assert receipt["n_accessions"] == len(expected) and receipt["n_rows_total"] == 4 * len(expected)
    assert receipt["n_files_histogram"] == {"4": len(expected)} and receipt["n_unique_sequences"] == len(expected)
    names = [runner.OFFICIAL_TRAIN_NAME.format(fold=fold) for fold in range(runner.N_FOLDS)]
    assert [item["file"] for item in receipt["inputs"]] == names
    assert [item["sha256"] for item in receipt["inputs"]] == [runner.sha256_file(split30 / name) for name in names]
    assert [item["n_rows_whitespace_or_separator_normalized"] for item in receipt["inputs"]] == [0, 1, 1, 0, 0]
    assert receipt["runner_sha256"] == runner.sha256_file(ROOT / "scripts" / "run_clean_split30_comparator.py")
    assert json.loads((out / runner.SEQUENCE_MAP_RECEIPT).read_text()) == receipt

    rows = list(csv.reader((out / runner.SEQUENCE_MAP_FILE).read_text().splitlines(), delimiter="\t"))
    assert tuple(rows[0]) == runner.SEQUENCE_MAP_COLUMNS and [row[0] for row in rows[1:]] == sorted(expected)
    for accession, digest, length, n_files, sequence in rows[1:]:
        assert sequence == expected[accession] and digest == hashlib.sha256(sequence.encode()).hexdigest()
        assert int(length) == len(sequence) and n_files == "4"
    assert read_fasta(out / runner.SEQUENCE_MAP_FASTA) == expected
    # The ESMC 'sequences' action reads the map with the same hashing and residue rules.
    loaded = read_sequence_map(out / runner.SEQUENCE_MAP_FILE)
    assert dict(loaded.sequence_by_accession) == expected
    assert dict(loaded.sha256_by_accession) == {key: hashlib.sha256(value.encode()).hexdigest()
                                                for key, value in expected.items()}
    assert loaded.ids_sha256 == receipt["accessions_sha256"]
    assert loaded.file_sha256 == receipt["outputs"]["sequence_map"]["sha256"]
    assert runner.read_sequence_map_file(out / runner.SEQUENCE_MAP_FILE, [])[0] == expected
    # The base FASTA and a prepared fit's masked copies pass the 'sequences' input checks.
    prepared = runner.prepare_fit(official_train_csv=split30 / names[0], fold=0, regime="r1", seed=42,
                                  work_dir=space / "masks_fit")
    masked_fasta = space / "masks_fit" / "masks" / prepared["masked_copies"]["fasta"]
    _, labels, _ = read_sequence_inputs([out / runner.SEQUENCE_MAP_FASTA, masked_fasta], loaded)
    assert {row["label"] for row in labels} == set(expected) | set(read_fasta(masked_fasta))
    with pytest.raises(ComparatorError, match="not a new or empty"):
        runner.write_sequence_map(split30_dir=split30, out_dir=out)

    cli_out = space / "seqmap_cli"
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_clean_split30_comparator.py"), "sequence-map",
                             "--split30-dir", str(split30), "--out-dir", str(cli_out)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["accessions_sha256"] == receipt["accessions_sha256"]
    assert (cli_out / runner.SEQUENCE_MAP_FILE).read_bytes() == (out / runner.SEQUENCE_MAP_FILE).read_bytes()


def test_sequence_map_refuses_conflicts(space):
    q3 = sequence_for("Q00003", 70)
    other = q3[:-1] + ("A" if q3[-1] != "A" else "C")
    cases = [
        ({(3, "Q00003"): ("2.7.1.1", other)}, "sequence in"),
        ({(3, "Q00003"): ("2.7.1.2", q3)}, "EC string in"),
        # The order of a protein's EC numbers is part of its native string.
        ({(4, "Q00004"): ("5.3.1.1;2.7.1.1", sequence_for("Q00004", 75))}, "EC string in"),
    ]
    messages = []
    for index, (edits, message) in enumerate(cases):
        split30 = write_split30(space / f"conflict{index}", edits=edits)
        with pytest.raises(ComparatorError, match=message) as caught:
            runner.write_sequence_map(split30_dir=split30, out_dir=space / f"conflict_out{index}")
        assert not (space / f"conflict_out{index}").exists()
        messages.append(str(caught.value))
    assert q3 in messages[0] and other in messages[0]  # both values are printed
    assert "'2.7.1.1'" in messages[1] and "'2.7.1.2'" in messages[1]

    duplicated = write_split30(space / "duplicated")
    with (duplicated / runner.OFFICIAL_TRAIN_NAME.format(fold=2)).open("a") as handle:
        handle.write(f"Q00001\t1.1.1.1\t{sequence_for('Q00001', 80)}\n")
    with pytest.raises(ComparatorError, match="Duplicate Entry Q00001"):
        runner.write_sequence_map(split30_dir=duplicated, out_dir=space / "duplicated_out")

    ambiguous = write_split30(space / "ambiguous", SPLIT30_PROTEINS + [("Q00001_3", "1.1.1.1", 40)])
    with pytest.raises(ComparatorError, match="masked copy of another accession"):
        runner.write_sequence_map(split30_dir=ambiguous, out_dir=space / "ambiguous_out")

    good = write_split30(space / "good")
    runner.write_sequence_map(split30_dir=good, out_dir=space / "good_out")
    path = space / "good_out" / runner.SEQUENCE_MAP_FILE
    with pytest.raises(ComparatorError, match="expected"):
        runner.read_sequence_map_file(path, [], expected_sha256="0" * 64)
    lines = path.read_text().splitlines()
    lines[1] = lines[1][:-1] + ("A" if lines[1][-1] != "A" else "C")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ComparatorError, match="disagree"):
        runner.read_sequence_map_file(path, [])


def test_sequence_map_reads_training_files_only(space):
    split30 = write_split30(space / "linked")
    names = runner.official_training_paths(split30)
    held_out = split30 / "split30_test_split_0.csv"
    names[0].rename(held_out)
    names[0].symlink_to(held_out)  # a training name that resolves to a test file
    with pytest.raises(ComparatorError, match="held-out name"):
        runner.write_sequence_map(split30_dir=split30, out_dir=space / "linked_out")
    with pytest.raises(ComparatorError, match="held-out name"):
        runner.collect_sequence_map([held_out, *names[1:]], [])
    with pytest.raises(ComparatorError, match="5 official training files"):
        runner.collect_sequence_map(names[1:], [])
    assert not (space / "linked_out").exists()
    # A fresh process with the guard roots over the folder refuses before opening anything.
    guarded = write_split30(space / "guarded")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_clean_split30_comparator.py"), "sequence-map",
                             "--split30-dir", str(guarded), "--out-dir", str(space / "guarded_out"),
                             "--forbidden-root", str(guarded)], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and "forbidden root" in result.stderr
    assert not (space / "guarded_out").exists()


# ---------------------------------------------------------------- ESMC embedding farm

ESMC_BLOCKS = 4


class FakeEsmc:
    """ESMC stand-in for 'esmc_chain_means.py sequences': rows follow the tokens; '<mask>' is one token."""

    def __init__(self):
        self.transformer = SimpleNamespace(blocks=[None] * ESMC_BLOCKS)

    def parameters(self):
        import torch
        return iter([SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)])

    def encode(self, protein):
        return protein.sequence

    def logits(self, sequence, config):
        import torch
        tokens = [token for token in sequence.replace(runner.MASK_TOKEN, "#")]
        codes = torch.tensor([0.0] + [7.0 if token == "#" else float(ord(token)) for token in tokens] + [1.0])
        position = torch.arange(len(codes), dtype=torch.float32)[:, None]
        rows = torch.sin(0.01 * (torch.arange(runner.ESMC_DIM, dtype=torch.float32)[None, :] + 1) * (position + 1)
                         + codes[:, None])
        hidden = torch.stack([rows + 0.1 * block for block in range(ESMC_BLOCKS)])
        return SimpleNamespace(embeddings=(0.5 * rows)[None], hidden_states=hidden[:, None])


FAKE_ESMC_SDK = (FakeEsmc, lambda sequence: SimpleNamespace(sequence=sequence), lambda **kwargs: dict(kwargs))


def esmc_inputs(space, *, with_masks=True, cache="esmc_cache"):
    """Sequence map, a prepared fold-0 fit's masked copies (seed 42) and a sequence-mean cache (block 2)."""
    from embed_helpers.esmc_chain_means import build_sequence_means

    split30 = space / "split30"
    if not split30.exists():
        write_split30(split30)
        runner.write_sequence_map(split30_dir=split30, out_dir=space / "seqmap")
        runner.prepare_fit(official_train_csv=split30 / runner.OFFICIAL_TRAIN_NAME.format(fold=0), fold=0,
                           regime="r1", seed=42, work_dir=space / "masks_fit")
    masks = json.loads((space / "masks_fit" / runner.PREPARE_MANIFEST).read_text())["masked_copies"]
    fasta = [space / "seqmap" / runner.SEQUENCE_MAP_FASTA]
    if with_masks:
        fasta.append(space / "masks_fit" / "masks" / masks["fasta"])
    summary = build_sequence_means(fasta, space / "seqmap" / runner.SEQUENCE_MAP_FILE, space / cache, layer_index=2,
                                   device="cpu", model=FakeEsmc(), sdk=FAKE_ESMC_SDK)
    return SimpleNamespace(split30=split30, seqmap=space / "seqmap" / runner.SEQUENCE_MAP_FILE,
                           masked_manifest=space / "masks_fit" / "masks" / masks["manifest"],
                           masked_fasta=space / "masks_fit" / "masks" / masks["fasta"],
                           cache=space / cache, cache_sha256=summary["manifest_sha256"])


def test_esmc_farm_converts_sequence_means(space):
    import torch
    from training.global_esm_features import load_sequence_mean_vectors

    data = esmc_inputs(space)
    sequences = split30_sequences()
    copies = read_fasta(data.masked_fasta)
    common = dict(cache_dir=data.cache, cache_manifest_sha256=data.cache_sha256, sequence_map=data.seqmap)
    farms = {"base": runner.write_esmc_farm(**common, layer="final", kind="base", out_dir=space / "farm_base"),
             "masked42": runner.write_esmc_farm(**common, layer="final", kind="masked", seed=42,
                                                masked_manifests=[data.masked_manifest],
                                                out_dir=space / "farm_masked42"),
             "block": runner.write_esmc_farm(**common, layer="block02", kind="base", out_dir=space / "farm_block")}
    strings = {"base": sequences, "masked42": copies, "block": sequences}
    for kind, receipt in farms.items():
        directory = space / f"farm_{kind}"
        entries = runner.read_farm_manifest(directory / runner.FARM_MANIFEST)
        assert set(entries) == set(strings[kind]) and receipt["n_files"] == len(entries)  # exact coverage
        layer = "final" if kind != "block" else "block02"
        keys = {label: hashlib.sha256(strings[kind][label].encode()).hexdigest() for label in entries}
        vectors = load_sequence_mean_vectors(data.cache, data.cache_sha256, sorted(set(keys.values())),
                                             expected_layer=layer)
        for label, entry in entries.items():
            tensor = torch.load(directory / f"{label}.pt", weights_only=True)
            assert tensor.dim() == 1 and tensor.dtype == torch.float32 and tensor.numel() == runner.ESMC_DIM
            assert tensor.untyped_storage().nbytes() == 4 * runner.ESMC_DIM  # an owned copy, not a view
            assert torch.equal(tensor, vectors[keys[label]])
            assert (entry.sequence_sha256, entry.dim, entry.file_sha256) == (
                keys[label], runner.ESMC_DIM, runner.sha256_file(directory / f"{label}.pt"))
            assert entry.seed == (42 if kind == "masked42" else None)
            assert (entry.embedding, entry.layer) == (("esmc_300m_mean_final", ESMC_BLOCKS) if kind != "block"
                                                      else ("esmc_300m_mean_block02", 2))
        assert receipt["farm_sha256"] == runner.embedding_farm_digest(
            {label: entry.file_sha256 for label, entry in entries.items()})
    # The farms link into a 960-wide fit of the same embedding; a block-layer masked farm does not.
    official = data.split30 / runner.OFFICIAL_TRAIN_NAME.format(fold=0)
    prepare = dict(official_train_csv=official, fold=0, regime="r1", seed=42, input_dim=runner.ESMC_DIM,
                   embedding="esmc_300m_mean_final")
    linked = runner.prepare_fit(**prepare, work_dir=space / "esmc_fit", base_esm_dir=space / "farm_base",
                                masked_esm_dir=space / "farm_masked42")
    assert linked["embedding_farm"]["complete"] and linked["embedding_farm"]["base"]["layer"] == ESMC_BLOCKS
    runner.write_esmc_farm(**common, layer="block02", kind="masked", seed=42, masked_manifests=[data.masked_manifest],
                           out_dir=space / "farm_masked_block")
    with pytest.raises(ComparatorError, match=r"\(embedding\)"):
        runner.prepare_fit(**prepare, work_dir=space / "esmc_fit_mixed", base_esm_dir=space / "farm_base",
                           masked_esm_dir=space / "farm_masked_block")


def test_esmc_farm_refusals(space):
    data = esmc_inputs(space)
    common = dict(cache_dir=data.cache, cache_manifest_sha256=data.cache_sha256, sequence_map=data.seqmap)
    masked = dict(kind="masked", seed=42, masked_manifests=[data.masked_manifest])
    cases = [
        (dict(layer="block03", kind="base"), "Layer mismatch"),
        (dict(layer=8, kind="base"), "Layer mismatch"),
        (dict(layer="layer8", kind="base"), "Unknown layer"),
        (dict(layer="final", kind="base", seed=42), "no mask seed"),
        (dict(layer="final", kind="masked", seed=42), "at least one masked-copy manifest"),
        (dict(layer="final", **{**masked, "seed": 43}), "mask seed 42, not 43"),
        (dict(layer="final", kind="base", sequence_map_sha256="0" * 64), "expected"),
        (dict(layer="final", kind="base", cache_manifest_sha256="0" * 64), "SHA-256"),
    ]
    for index, (options, message) in enumerate(cases):
        with pytest.raises(ComparatorError, match=message):
            runner.write_esmc_farm(**{**common, **options}, out_dir=space / f"refused{index}")
    # A cache embedded without the masked FASTA cannot give the masked farm.
    bare = esmc_inputs(space, with_masks=False, cache="esmc_cache_base_only")
    with pytest.raises(ComparatorError, match="not embedded"):
        runner.write_esmc_farm(cache_dir=bare.cache, cache_manifest_sha256=bare.cache_sha256,
                               sequence_map=data.seqmap, layer="final", **masked, out_dir=space / "refused_bare")
    tampered = space / "tampered_masked_manifest.tsv"
    lines = data.masked_manifest.read_text().splitlines()
    fields = lines[1].split("\t")
    fields[-1] = "0" * 64
    tampered.write_text("\n".join([lines[0], "\t".join(fields), *lines[2:]]) + "\n")
    with pytest.raises(ComparatorError, match="not the keyed masked copy"):
        runner.write_esmc_farm(**common, layer="final", kind="masked", seed=42, masked_manifests=[tampered],
                               out_dir=space / "refused_tampered")
    runner.write_esmc_farm(**common, layer="final", kind="base", out_dir=space / "farm_once")
    with pytest.raises(ComparatorError, match="not a new or empty"):
        runner.write_esmc_farm(**common, layer="final", kind="base", out_dir=space / "farm_once")


def test_esmc_masked_farms_of_several_seeds_share_one_cache(space):
    """One cache over the masked FASTAs of seeds 42 and 43: the label '<acc>_<j>' names a different string
    in each seed, and each seed's farm takes its own strings."""
    from embed_helpers.esmc_chain_means import build_sequence_means

    data = esmc_inputs(space)
    official = data.split30 / runner.OFFICIAL_TRAIN_NAME.format(fold=0)
    seed43 = runner.prepare_fit(official_train_csv=official, fold=0, regime="r1", seed=43,
                                work_dir=space / "masks_fit43")["masked_copies"]
    masks43 = space / "masks_fit43" / "masks"
    inputs = {42: (data.masked_manifest, data.masked_fasta), 43: (masks43 / seed43["manifest"], masks43 / seed43["fasta"])}
    copies = {seed: read_fasta(fasta) for seed, (_, fasta) in inputs.items()}
    shared = set(copies[42]) & set(copies[43])
    assert shared and all(copies[42][label] != copies[43][label] for label in shared)
    summary = build_sequence_means([space / "seqmap" / runner.SEQUENCE_MAP_FASTA] + [fasta for _, fasta in inputs.values()],
                                   data.seqmap, space / "esmc_cache_two_seeds", layer_index=2, device="cpu",
                                   model=FakeEsmc(), sdk=FAKE_ESMC_SDK)
    for seed, (manifest, fasta) in inputs.items():
        receipt = runner.write_esmc_farm(cache_dir=space / "esmc_cache_two_seeds",
                                         cache_manifest_sha256=summary["manifest_sha256"], sequence_map=data.seqmap,
                                         layer="final", kind="masked", seed=seed, masked_manifests=[manifest],
                                         out_dir=space / f"farm_two_seeds_{seed}")
        entries = runner.read_farm_manifest(space / f"farm_two_seeds_{seed}" / runner.FARM_MANIFEST)
        assert {label: entry.sequence_sha256 for label, entry in entries.items()} == {
            label: hashlib.sha256(string.encode()).hexdigest() for label, string in copies[seed].items()}
        assert {entry.seed for entry in entries.values()} == {seed}
        assert receipt["source_cache"]["inputs_used"] == [fasta.name]


# ---------------------------------------------------------------- distance export (fake export script)

# Stands in for the private export script: same arguments, the runner's checks and C4 writer.
FAKE_EXPORT_SCRIPT = '''
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
parser = argparse.ArgumentParser()
for name in ("--work-dir", "--query-ids", "--query-esm-dir", "--out", "--mode", "--runner", "--threads",
             "--freeze2-manifest", "--freeze2-sha256"):
    parser.add_argument(name)
parser.add_argument("--allow-in-sample-queries", action="store_true")
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("fake_export_runner", args.runner)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
roots = [Path(part) for part in os.environ[runner.FORBIDDEN_READ_ROOTS_ENV].split(os.pathsep)]
plan = runner.plan_distance_export(work_dir=args.work_dir, query_ids_file=args.query_ids,
                                   query_esm_dir=args.query_esm_dir, out=args.out, mode=args.mode,
                                   forbidden_roots=roots, freeze2_manifest=args.freeze2_manifest,
                                   freeze2_sha256=args.freeze2_sha256,
                                   export_script_sha256=runner.sha256_file(Path(__file__)),
                                   allow_in_sample=args.allow_in_sample_queries)
behaviour = Path(__file__).with_name("fake_export_behaviour.txt")
behaviour = behaviour.read_text().strip() if behaviour.is_file() else ""
metadata = dict(plan.metadata)
if behaviour == "wrong_seed":
    metadata["seed"] += 1
labels = []
for line in plan.table_path.read_text().splitlines()[1:]:
    labels += [ec for ec in line.split("\\t")[1].split(";") if ec not in labels]
if behaviour == "reversed_labels":
    labels.reverse()
distances = np.arange(len(plan.query_ids) * len(labels), dtype=np.float32).reshape(len(plan.query_ids), -1)
runner.write_distance_export(plan.out, distances, plan.query_ids, labels, metadata)
runner.export_side_files(plan.out)["export_receipt"].write_text(json.dumps(
    {"hash_seed": os.environ["PYTHONHASHSEED"], "argv": sys.argv[1:]}))
'''


@pytest.fixture
def export_setup(fit_setup, monkeypatch):
    original = runner._run
    monkeypatch.setattr(runner, "_run", FakeClean(fit_setup["work"]))
    runner.run_fit(**fit_setup["kwargs"])
    monkeypatch.setattr(runner, "_run", original)  # the export launches a real (fake-script) process
    space = fit_setup["space"]
    script = space / "private" / "fake_export.py"
    script.write_text(FAKE_EXPORT_SCRIPT)
    queries = space / "queries.txt"
    queries.write_text("# validation proteins\nQ00003\nQ00001\nQ00005\n")
    model = f"clean_split30_fold0_r1_{EMBEDDING}_seed42_d1280"
    # The queries are training proteins of this r1 fit, so these validation exports are in-sample.
    kwargs = dict(work_dir=fit_setup["work"], query_ids_file=queries, query_esm_dir=fit_setup["base"],
                  out=space / "exports" / "fold0_r1_seed42.npz", mode="validation", clean_python=Path(sys.executable),
                  clean_clone=fit_setup["clone"], export_script=script,
                  export_script_sha256=runner.sha256_file(script), threads=1, allow_in_sample=True)
    return {**fit_setup, "kwargs": kwargs, "fit_kwargs": fit_setup["kwargs"], "script": script,
            "weights": fit_setup["work"] / "data" / "model" / f"{model}.pth"}


def test_export_launches_the_pinned_script_and_validates(export_setup):
    kwargs, work = export_setup["kwargs"], export_setup["work"]
    plan = runner.run_export(**kwargs, dry_run=True)
    assert plan["hash_seed"] == 42 and not kwargs["out"].exists()
    record = runner.run_export(**kwargs)
    out = kwargs["out"]
    export = runner.read_distance_export(out)
    assert export.query_ids == ("Q00003", "Q00001", "Q00005")  # list order
    assert export.receipt["contract_version"] == 2 and export.receipt["n_in_sample_queries"] == 3
    assert "inner_fold" not in export.receipt and "freeze2_manifest_sha256" not in export.receipt
    assert export.ec_labels == runner.table_ec_order(work / "data" / f"{NAME0}.csv")
    assert export.ec_labels == ("1.1.1.1", "2.7.1.1", "3.4.21.4", "6.3.2.1", "3.1.1.-", "5.3.1.1")  # not sorted
    prepared = json.loads((work / runner.PREPARE_MANIFEST).read_text())
    assert export.receipt["embedding_farm_sha256"] == prepared["embedding_farm"]["base"]["farm_sha256"]
    assert export.receipt["training_table_sha256"] == prepared["training_table"]["sha256"]
    assert export.receipt["checkpoint_sha256"] == runner.sha256_file(export_setup["weights"])
    assert export.receipt["query_embeddings_sha256"] == runner.embedding_farm_digest(
        {query: runner.sha256_file(export_setup["base"] / f"{query}.pt") for query in export.query_ids})
    side = runner.export_side_files(out)
    launched = json.loads(side["export_receipt"].read_text())
    assert launched["hash_seed"] == "42"  # PYTHONHASHSEED is the fit seed
    pairs = dict(zip(launched["argv"][::2], launched["argv"][1::2]))
    assert pairs["--mode"] == "validation" and "--freeze2-manifest" not in pairs
    assert pairs["--runner"] == str((ROOT / "scripts" / "run_clean_split30_comparator.py").resolve())
    written = json.loads(side["run_record"].read_text())
    assert written == json.loads(json.dumps(record))
    assert written["command"][:4] == [sys.executable, "-c", runner.GUARDED_LAUNCH, str(runner.ACCESS_GUARD_FILE)]
    assert written["command"][4] == str(export_setup["script"])
    assert written["scripts"]["export_script"]["sha256"] == kwargs["export_script_sha256"]
    assert written["env"]["PYTHONHASHSEED"] == "42" and written["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert str(runner.DEFAULT_FORBIDDEN_ROOTS[0].resolve()) in written["env"][runner.FORBIDDEN_READ_ROOTS_ENV]
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_clean_split30_comparator.py"),
                             "validate-export", "--npz", str(out)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and json.loads(result.stdout)["n_query"] == 3
    with pytest.raises(ComparatorError, match="write-once"):
        runner.run_export(**kwargs)


def test_export_test_mode_needs_the_freeze2_manifest(export_setup, monkeypatch):
    from ec_clean30.provenance import build_freeze2_manifest

    kwargs, space = export_setup["kwargs"], export_setup["space"]
    checkpoint = runner.sha256_file(export_setup["weights"])
    pass_root = space / "pass_root"
    # Frozen-pass queries are proteins outside the fit's training table: a label-free list and embeddings.
    queries = space / "frozen_queries.txt"
    queries.write_text("Z00001\nZ00002\n")
    query_farm = write_farm(space / "esm_frozen_queries", {accession: (None, sequence_for(accession, 60))
                                                            for accession in ("Z00001", "Z00002")})
    # The campaign's own Freeze-2 builder: it pins the live code and this runner.
    real = build_freeze2_manifest(checkpoints=[{"unit_id": "clean__fold0_r1_seed42", "sha256": checkpoint,
                                                "smoke": False}],
                                  test_pass_output_root=pass_root, smoke=False, provenance={"synthetic": True})

    def manifest(name, edit=None):
        payload = json.loads(json.dumps(real))
        if edit is not None:
            edit(payload)
        path = space / name
        path.write_text(json.dumps(payload))
        return {"freeze2_manifest": path, "freeze2_sha256": runner.sha256_file(path)}

    def runner_sources_only(payload):
        payload["code_sha256"] = {path: payload["code_sha256"][path] for path in runner.RUNNER_SOURCE_PATHS}

    good = manifest("freeze2.json")
    test_mode = {**kwargs, "mode": "test", "allow_in_sample": False, "query_ids_file": queries,
                 "query_esm_dir": query_farm, "out": pass_root / "pass1" / "fold0_r1_seed42.npz"}
    other_script = export_setup["script"].with_name("other_export.py")
    other_script.write_text(FAKE_EXPORT_SCRIPT + "\n# another export script\n")
    # The fake script stands in for the pinned one; another script is refused in test mode.
    monkeypatch.setattr(runner, "EXPORT_SCRIPT_SHA256", kwargs["export_script_sha256"])
    cases = [
        ({}, "needs the Freeze-2 manifest"),
        ({"freeze2_manifest": good["freeze2_manifest"]}, "needs the Freeze-2 manifest"),
        ({**good, "freeze2_sha256": "0" * 64}, "expected"),
        (manifest("schema2.json", lambda m: m.update(schema_version=2)), "schema_version 1"),
        (manifest("no_code.json", lambda m: m.update(code_sha256=None)), "code_sha256"),
        (manifest("empty_code.json", lambda m: m.update(code_sha256={})), "does not pin"),
        (manifest("runner_sources_only.json", runner_sources_only), "required code file"),
        (manifest("no_runner_table.json", lambda m: m.pop("additional_code_sha256")), "additional_code_sha256"),
        (manifest("other_runner.json", lambda m: m["additional_code_sha256"].update({runner.RUNNER_KEY: "0" * 64})),
         "Live code differs"),
        (manifest("other_model.json", lambda m: m["code_sha256"].update({"src/model.py": "0" * 64})),
         r"Live code differs.*src/model\.py"),
        (manifest("other_checkpoint.json", lambda m: m.update(checkpoint_sha256s=["e" * 64])),
         "does not list this checkpoint"),
        (manifest("relative.json", lambda m: m.update(test_pass_output_root="pass_root")), "absolute"),
        ({**good, "out": space / "elsewhere" / "fold0_r1_seed42.npz"}, "not inside"),
        ({**good, "query_ids_file": kwargs["query_ids_file"], "query_esm_dir": kwargs["query_esm_dir"]}, "in-sample"),
        ({**good, "allow_in_sample": True}, "no in-sample"),
        ({**good, "export_script": other_script, "export_script_sha256": runner.sha256_file(other_script)},
         "pinned export script"),
    ]
    for options, message in cases:
        with pytest.raises(ComparatorError, match=message):
            runner.run_export(**{**test_mode, **options})
    with pytest.raises(ComparatorError, match="takes no Freeze-2"):
        runner.run_export(**kwargs, **good)
    runner.run_export(**test_mode, **good, dry_run=True)
    assert not pass_root.exists()  # no refusal and no dry run claims the checkpoint

    record = runner.run_export(**test_mode, **good)
    assert record["inputs"]["freeze2_manifest"]["sha256"] == good["freeze2_sha256"]
    assert record["inputs"]["freeze2_manifest"]["code_files_verified"] == len(real["code_sha256"])
    export = runner.read_distance_export(test_mode["out"])
    assert export.receipt["mode"] == "test" and export.receipt["n_in_sample_queries"] == 0
    runner_sha256 = real["additional_code_sha256"][runner.RUNNER_KEY]
    assert runner_sha256 == runner.sha256_file(ROOT / runner.RUNNER_KEY)
    assert {field: export.receipt[field] for field in runner.C4_TEST_BINDING_FIELDS} == {
        "freeze2_manifest_sha256": good["freeze2_sha256"], "test_pass_output_root": str(pass_root.resolve()),
        "runner_sha256": runner_sha256, "export_script_sha256": kwargs["export_script_sha256"]}
    runner.check_test_export_binding(export.receipt, freeze2_sha256=good["freeze2_sha256"],
                                     test_pass_output_root=pass_root, runner_sha256=runner_sha256)
    for options, field in (({"freeze2_sha256": "0" * 64}, "freeze2_manifest_sha256"),
                           ({"test_pass_output_root": space / "other_root"}, "test_pass_output_root"),
                           ({"runner_sha256": "0" * 64}, "runner_sha256"),
                           ({"export_script_sha256": "0" * 64}, "export_script_sha256")):
        with pytest.raises(ComparatorError, match=field):
            runner.check_test_export_binding(export.receipt, **{"freeze2_sha256": good["freeze2_sha256"],
                                                                "test_pass_output_root": pass_root, **options})
    launched = json.loads(runner.export_side_files(test_mode["out"])["export_receipt"].read_text())
    assert dict(zip(launched["argv"][::2], launched["argv"][1::2]))["--freeze2-sha256"] == good["freeze2_sha256"]
    # One test export per checkpoint: the claim is written before the launch and never replaced.
    claim = pass_root / runner.TEST_CLAIMS_DIR / f"{checkpoint}.json"
    assert json.loads(claim.read_text())["status"] == "claimed"
    assert record["test_claim"] == {"path": str(claim), "sha256": runner.sha256_file(claim)}
    with pytest.raises(ComparatorError, match="already has a test export claim"):
        runner.run_export(**{**test_mode, "out": pass_root / "pass2" / "again.npz"}, **good)
    assert not (pass_root / "pass2").exists()


def test_export_refusals(export_setup):
    kwargs, space = export_setup["kwargs"], export_setup["space"]
    outside = space / "outside_queries.txt"
    outside.write_text("Q00010\n")  # a fold-1 training protein, not in fold 0's file
    other = write_farm(space / "esm_other_queries", {"Q00010": (None, sequence_for("Q00010", 50))})
    labelled = space / "labelled_queries.csv"
    labelled.write_text("Q00003\t2.7.1.1\tMKV\n")
    rows = {row[0]: sequence_for(row[0], row[2]) for row in FOLD0_ROWS}
    unprepared = space / "unfitted"
    runner.prepare_fit(official_train_csv=space / "official" / runner.OFFICIAL_TRAIN_NAME.format(fold=0), fold=0,
                       regime="r1", seed=42, work_dir=unprepared)
    cases = [
        (dict(query_ids_file=outside, query_esm_dir=other), "not training-side proteins of fold 0"),
        (dict(query_ids_file=labelled), "more than one field"),
        (dict(query_esm_dir=write_farm(space / "esm_960", {key: (None, rows[key]) for key in rows}, dim=960)),
         "width"),
        (dict(query_esm_dir=write_farm(space / "esm_seeded", {key: (42, rows[key]) for key in rows})),
         "not a base embedding"),
        (dict(query_esm_dir=write_farm(space / "esm_layer8", {key: (None, rows[key]) for key in rows}, layer=8)),
         "layer"),
        (dict(out=space / "exports_test" / "fold0.npz"), "held-out name"),
        (dict(out=space / "exports" / "fold0.json"), ".npz"),
        (dict(export_script_sha256="f" * 64), "SHA-256"),
        (dict(mode="final"), "mode"),
        (dict(work_dir=unprepared), "has not completed"),
        # Training proteins of the fit are in-sample: refused unless an engineering check allows them.
        (dict(allow_in_sample=False), "in-sample"),
        # Test mode runs only the pinned export script.
        (dict(mode="test", allow_in_sample=False), "pinned export script"),
    ]
    for options, message in cases:
        with pytest.raises(ComparatorError, match=message) as caught:
            runner.run_export(**{**kwargs, **options})
        assert "2.7.1.1" not in str(caught.value)  # a refused table row is never echoed
    assert not (space / "exports").exists()
    behaviour = export_setup["script"].with_name("fake_export_behaviour.txt")
    behaviour.write_text("wrong_seed\n")
    with pytest.raises(ComparatorError, match=r"differs from this fit.*'seed'"):
        runner.run_export(**kwargs)
    behaviour.write_text("reversed_labels\n")  # EC columns out of the upstream centre order
    with pytest.raises(ComparatorError, match=r"differs from this fit.*'ec_labels'"):
        runner.run_export(**{**kwargs, "out": space / "exports" / "reversed.npz"})
    behaviour.unlink()
    export_setup["weights"].write_bytes(b"replaced after the fit")
    with pytest.raises(ComparatorError, match="Final weights"):
        runner.run_export(**{**kwargs, "out": space / "exports" / "again.npz"})


def write_exclusions(directory, fold, inner_fold, reasons, *, name=None, header=runner.EXCLUSIONS_COLUMNS):
    """An identity-tool exclusions list (accession, reason) for one inner fold."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or runner.EXCLUSIONS_NAME.format(fold=fold, inner_fold=inner_fold))
    path.write_text("\t".join(header) + "\n" + "".join(f"{accession}\t{reason}\n"
                                                        for accession, reason in reasons.items()))
    return path


def table_entries(work, manifest):
    return [line.split("\t")[0] for line in (work / manifest["training_table"]["file"]).read_text().splitlines()[1:]]


def test_inner_regimes_use_the_exclusions_list(space):
    from ec_clean30 import identity

    assert (runner.EXCLUSIONS_COLUMNS, runner.N_INNER_FOLDS) == (identity.R1_EXCLUSION_COLUMNS, identity.N_INNER_FOLDS)
    assert set(runner.EXCLUSION_REASONS) == {identity.R1_REASON_VALIDATION, identity.R1_REASON_D3}
    assert runner.EXCLUSIONS_NAME.format(fold=0, inner_fold=2) == "r1_inner_exclusions_fold0_inner2.tsv"
    official = write_official(space / "official", 0, FOLD0_ROWS)
    reasons = {"Q00001": "inner_validation", "Q00004": "d3_neighbour"}
    exclusions = write_exclusions(space / "identity", 0, 1, reasons)
    common = dict(official_train_csv=official, fold=0, seed=42)
    r1 = runner.prepare_fit(**common, regime="r1_inner", inner_fold=1, accession_exclusions=exclusions,
                            work_dir=space / "r1_inner")
    assert table_entries(space / "r1_inner", r1) == [row[0] for row in FOLD0_ROWS if row[0] not in reasons]
    assert r1["training_name"] == f"split30_fold0_r1_inner1_{EMBEDDING}" and r1["inner_fold"] == 1
    assert r1["accession_exclusions"]["sha256"] == runner.sha256_file(exclusions)
    assert r1["accession_exclusions"]["n_by_reason"] == {"inner_validation": 1, "d3_neighbour": 1}
    allowlist = space / "metallo_inner_training.txt"
    allowlist.write_text("Q00005\nQ00003\n")
    r2 = runner.prepare_fit(**common, regime="r2_inner", inner_fold=1, accession_allowlist=allowlist,
                            accession_exclusions=exclusions, work_dir=space / "r2_inner")
    assert table_entries(space / "r2_inner", r2) == ["Q00003", "Q00005"]
    assert r2["training_name"] == f"split30_fold0_r2_inner1_{EMBEDDING}"
    outer = runner.prepare_fit(**common, regime="r1", work_dir=space / "r1_outer")
    assert outer["inner_fold"] is None and outer["accession_exclusions"] is None

    leaky = space / "leaky_allowlist.txt"
    leaky.write_text("Q00005\nQ00004\n")
    lists = space / "lists"
    cases = [
        (dict(regime="r2_inner", inner_fold=1, accession_allowlist=leaky, accession_exclusions=exclusions),
         "exclusions list"),
        (dict(regime="r1_inner", accession_exclusions=exclusions), "needs inner fold"),
        (dict(regime="r1_inner", inner_fold=4, accession_exclusions=exclusions), "outside"),
        (dict(regime="r1", inner_fold=1), "takes no inner fold"),
        (dict(regime="r1_inner", inner_fold=1), "exclusions list"),
        (dict(regime="r1", accession_exclusions=exclusions), "exclusions list"),
        (dict(regime="r2_inner", inner_fold=1, accession_exclusions=exclusions), "allowlist"),
        (dict(regime="r1_inner", inner_fold=2, accession_exclusions=exclusions),
         "needs r1_inner_exclusions_fold0_inner2.tsv"),
        (dict(regime="r1_inner", inner_fold=1, accession_exclusions=write_exclusions(
            lists / "header", 0, 1, reasons, header=("accession", "reason", "ec"))), "exactly"),
        (dict(regime="r1_inner", inner_fold=1, accession_exclusions=write_exclusions(
            lists / "reason", 0, 1, {"Q00001": "inner_validation", "Q00004": "neighbour"})), "Unknown exclusion"),
        (dict(regime="r1_inner", inner_fold=1, accession_exclusions=write_exclusions(
            lists / "d3_only", 0, 1, {"Q00004": "d3_neighbour"})), "lists no inner_validation"),
        (dict(regime="r1_inner", inner_fold=1, accession_exclusions=write_exclusions(
            lists / "unknown", 0, 1, {"Q00001": "inner_validation", "Q99999": "d3_neighbour"})),
         "not in the official training file"),
        (dict(regime="r1_inner", inner_fold=1, accession_exclusions=write_exclusions(
            lists / "identity_test", 0, 1, reasons)), "held-out name"),
    ]
    for index, (options, message) in enumerate(cases):
        with pytest.raises(ComparatorError, match=message):
            runner.prepare_fit(**common, **options, work_dir=space / f"refused{index}")
        assert not (space / f"refused{index}").exists()


def test_inner_regime_exports_are_out_of_sample_validation(export_setup, monkeypatch):
    space, kwargs = export_setup["space"], export_setup["kwargs"]
    official = space / "official" / runner.OFFICIAL_TRAIN_NAME.format(fold=0)
    reasons = {"Q00001": "inner_validation", "Q00003": "inner_validation", "Q00004": "d3_neighbour"}
    exclusions = write_exclusions(space / "identity", 0, 1, reasons)
    rows, _ = runner.read_official_training_rows(official, 0, [])
    base, masked = write_farms(space, [row for row in rows if row.accession not in reasons], 42, tag="_inner1")
    work = space / "fit_inner1"
    runner.prepare_fit(official_train_csv=official, fold=0, regime="r1_inner", inner_fold=1, seed=42, work_dir=work,
                       base_esm_dir=base, masked_esm_dir=masked, accession_exclusions=exclusions)
    original = runner._run
    monkeypatch.setattr(runner, "_run", FakeClean(work))
    fit = runner.run_fit(**{**export_setup["fit_kwargs"], "work_dir": work})
    assert fit["training_receipt"]["model_name"] == f"clean_split30_fold0_r1_inner1_{EMBEDDING}_seed42_d1280"
    monkeypatch.setattr(runner, "_run", original)

    validation = space / "inner_validation_queries.txt"
    validation.write_text("Q00003\nQ00001\n")
    inner = {**kwargs, "work_dir": work, "query_ids_file": validation, "allow_in_sample": False,
             "out": space / "exports_inner" / "fold0_r1_inner1_seed42.npz"}
    runner.run_export(**inner)  # the inner-validation proteins are not in the inner training table
    export = runner.read_distance_export(inner["out"])
    assert (export.receipt["regime"], export.receipt["inner_fold"], export.receipt["n_in_sample_queries"]) == (
        "r1_inner", 1, 0)
    kept = space / "inner_kept_queries.txt"
    kept.write_text("Q00002\n")
    with pytest.raises(ComparatorError, match="in-sample"):
        runner.run_export(**{**inner, "query_ids_file": kept, "out": space / "exports_inner" / "kept.npz"})
    monkeypatch.setattr(runner, "EXPORT_SCRIPT_SHA256", kwargs["export_script_sha256"])
    with pytest.raises(ComparatorError, match="development arm"):
        runner.run_export(**{**inner, "mode": "test", "freeze2_manifest": exclusions, "freeze2_sha256": "0" * 64,
                             "out": space / "exports_inner" / "frozen.npz"})
