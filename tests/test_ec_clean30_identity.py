"""Identity tools for the EC class vs CLEAN30 campaign (synthetic sequences only; no dataset access)."""
from __future__ import annotations

from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import audit_sequence_remoteness as remoteness  # noqa: E402
from ec_clean30 import identity  # noqa: E402
from ec_clean30.identity import (  # noqa: E402
    EDGE_COLUMNS, MMSEQS_BINARY_SHA256, MMSEQS_VERSION, NODE_COLUMNS, PAIR_COLUMNS, SEARCH_FIELDS,
    STRATA_COLUMNS, Hit, Link, assert_training_file, assign_components, assign_inner, build_cohort_components,
    build_components, build_edges, build_inner_components, build_inputs, build_r1_inner_exclusions,
    build_search_command, build_strata, certify_inner, check_search_command, component_id, component_summary,
    identity_bin, inner_certification_counts, node_links, read_hits, read_tsv, recount_alignment, run_search,
    sha256_file, sort_alignment_text, strata_rows, training_file, undirected_pairs, verify_binary,
)
from training.ec_membership import read_ec_fold_membership, write_ec_fold_membership  # noqa: E402

STANDARD = "ACDEFGHIKLMNPQRSTVWY"
FREEZE1 = "a" * 64
OTHER_FREEZE1 = "b" * 64
EC_TEXT = "3.4.24.17"
LENGTH = 120
TEST_FOLD = {"A00001": 1, "A00002": 2, "A00003": 3, "A00004": 4, "A00005": 1,
             "A00006": 2, "A00007": 3, "A00008": 0, "A00009": 0, "A00010": 1}
DONORS = {"A00001": "1abc", "A00002": "2DEF", "A00003": "1ABC", "A00004": "3GHI", "A00005": "4JKL",
          "A00006": "5MNO"}
HITS = {frozenset(pair): evalue for pair, evalue in {
    ("A00001", "A00002"): "1.000E-20", ("A00001", "A00007"): "1.000E-15", ("A00002", "A00007"): "1.000E-08",
    ("A00003", "A00004"): "2.000E-05", ("A00005", "A00006"): "1.000E-50", ("A00001", "A00008"): "5.000E-02",
}.items()}


def substitute(residue):
    return STANDARD[(STANDARD.index(residue) + 1) % len(STANDARD)]


def random_sequence(rng, length):
    return "".join(rng.choice(STANDARD) for _ in range(length))


def mutate(sequence, n_changes, rng):
    positions = set(rng.sample(range(len(sequence)), n_changes))
    return "".join(substitute(r) if i in positions else r for i, r in enumerate(sequence))


def synthetic_sequences():
    rng = random.Random(7)
    base, other, third = (random_sequence(rng, LENGTH) for _ in range(3))
    return {
        "A00001": base, "A00002": mutate(base, 72, rng), "A00003": other, "A00004": mutate(other, 96, rng),
        "A00005": third, "A00006": third, "A00007": mutate(base, 60, rng),
        "A00008": random_sequence(rng, LENGTH), "A00009": random_sequence(rng, LENGTH),
        "A00010": random_sequence(rng, LENGTH),
    }


def write_dataset(root, sequences=None, donors=None, official_overrides=None, fold_sequences=None):
    """Official training files (tab-separated) and site-level metallo training CSVs for 5 folds."""
    sequences = sequences or synthetic_sequences()
    fold_sequences = fold_sequences or {}
    donors = donors or {}
    official_dir, metallo_dir = root / "split30", root / "folds"
    official_dir.mkdir(parents=True)
    metallo_dir.mkdir(parents=True)
    for fold in range(5):
        training = [acc for acc in sorted(TEST_FOLD) if TEST_FOLD[acc] != fold]
        rows = [(acc, EC_TEXT, fold_sequences.get((acc, fold), sequences[acc])) for acc in training]
        rows += (official_overrides or {}).get(fold, [])
        with (official_dir / f"split30_train_split_{fold}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["Entry", "EC number", "Sequence"])
            writer.writerows(rows)
        with (metallo_dir / f"CLEAN_30_train_test_split_{fold}_train.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["structure", "chain_resi", "metaltype", "ecnumber", "uniprot_id", "alphafill_pdb_id",
                             identity.DONOR_COLUMN])
            for acc in training:
                if acc not in DONORS:
                    continue
                donor = donors.get((acc, fold), DONORS[acc])
                stems = [f"{acc}__chain_A__EC_{EC_TEXT}"] + ([f"{acc}__chain_B__EC_{EC_TEXT}"]
                                                             if acc == "A00004" else [])
                for site, stem in enumerate(stems * (2 if acc == "A00001" else 1)):
                    writer.writerow([stem, f"A_{site}", "ZN", EC_TEXT, acc, donor, donor])
    return official_dir, metallo_dir


class FakeMMseqs:
    """Stands in for the pinned binary: answers 'version' and writes gapless full-length hits."""

    def __init__(self, order="reverse", version=MMSEQS_VERSION, hits=None):
        self.calls, self.order, self.version = [], order, version
        self.hits = HITS if hits is None else hits

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if argv[1] == "version":
            return subprocess.CompletedProcess(argv, 0, stdout=self.version + "\n", stderr="")
        assert argv[1] == "easy-search"
        queries, targets = identity.read_fasta(Path(argv[2])), identity.read_fasta(Path(argv[3]))
        rows = []
        for query, q in queries.items():
            for target, t in targets.items():
                if query != target and frozenset((query, target)) not in self.hits:
                    continue
                nident = sum(a == b for a, b in zip(q, t))
                evalue = "0.000E+00" if query == target else self.hits[frozenset((query, target))]
                rows.append([query, target, nident, len(q), 1, len(q), len(q), 1, len(t), len(t), evalue,
                             f"{2.0 * nident:.1f}", q, t])
        rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        if self.order == "shuffled":
            random.Random(3).shuffle(rows)
        lines = ["\t".join(SEARCH_FIELDS)] + ["\t".join(map(str, row)) for row in rows]
        Path(argv[4]).write_text("\n".join(lines) + "\n")
        return subprocess.CompletedProcess(argv, 0)


def fake_binary(directory, version="unused"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "mmseqs"
    path.write_text(f"#!/bin/sh\necho {version}\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def pinned_fake(tmp_path, monkeypatch):
    """The fake binary stands in for the pinned build: its SHA-256 becomes the pinned value."""
    binary = fake_binary(tmp_path / "fakebin")
    monkeypatch.setattr(identity, "MMSEQS_BINARY_SHA256", sha256_file(binary))
    return binary


def run_metallo_pipeline(tmp_path, work_name, dataset, fold=0, order="reverse", allow_unpinned=True, hits=None):
    """fasta, search, edges and inner components; unpinned (engineering) unless ``pinned_fake`` is active."""
    official_dir, metallo_dir = dataset
    work = tmp_path / work_name
    build_inputs(work, fold, official_split_dir=official_dir, metallo_folds_dir=metallo_dir)
    binary = fake_binary(tmp_path / "fakebin")
    fake = FakeMMseqs(order=order, hits=hits)
    run_search(work, fold, binary=binary, expected_sha256=sha256_file(binary), runner=fake,
               allow_unpinned_binary=allow_unpinned)
    build_edges(work, fold, allow_unpinned_binary=allow_unpinned)
    build_inner_components(work, fold, allow_unpinned_binary=allow_unpinned)
    return work, fake


def run_complement_chain(work, fold, dataset, fake, freeze1=FREEZE1):
    binary = Path(fake.calls[0][0])
    build_inputs(work, fold, query_set="complement", freeze1_sha256=freeze1, official_split_dir=dataset[0],
                 metallo_folds_dir=dataset[1])
    run_search(work, fold, query_set="complement", binary=binary, expected_sha256=sha256_file(binary),
               freeze1_sha256=freeze1, runner=fake)
    build_edges(work, fold, query_set="complement", freeze1_sha256=freeze1)


def write_search_rows(path, rows):
    lines = ["\t".join(SEARCH_FIELDS)] + ["\t".join(str(row[field]) for field in SEARCH_FIELDS) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def make_alignment(*, n_cols, n_ident, target_gaps=0, query_gaps=0, ambiguous=0, q_prefix=0, q_suffix=0,
                   t_prefix=0, t_suffix=0, evalue="1.000E-10", exported=None, seed=0):
    """One MMseqs2-like row plus the sequences it must match.

    Columns: target gaps, then query gaps, then identical columns (the first ``ambiguous`` of them
    are X-X), then substituted columns.
    """
    rng = random.Random(seed)
    paired = n_cols - target_gaps - query_gaps
    assert 0 <= n_ident <= paired and ambiguous <= n_ident
    q_core = random_sequence(rng, target_gaps) + "-" * query_gaps
    t_core = "-" * target_gaps + random_sequence(rng, query_gaps)
    same = "X" * ambiguous + random_sequence(rng, n_ident - ambiguous)
    rest = random_sequence(rng, paired - n_ident)
    qaln, taln = q_core + same + rest, t_core + same + "".join(substitute(r) for r in rest)
    q_ungapped, t_ungapped = qaln.replace("-", ""), taln.replace("-", "")
    q = random_sequence(rng, q_prefix) + q_ungapped + random_sequence(rng, q_suffix)
    t = random_sequence(rng, t_prefix) + t_ungapped + random_sequence(rng, t_suffix)
    row = {"query": "Q1", "target": "T1", "nident": str(n_ident if exported is None else exported),
           "alnlen": str(n_cols), "qstart": str(q_prefix + 1), "qend": str(q_prefix + len(q_ungapped)),
           "qlen": str(len(q)), "tstart": str(t_prefix + 1), "tend": str(t_prefix + len(t_ungapped)),
           "tlen": str(len(t)), "evalue": evalue, "bits": "100.0", "qaln": qaln, "taln": taln}
    return row, {"Q1": q, "T1": t}


def output_texts(work):
    return {path.relative_to(work).as_posix(): path.read_text() for path in sorted(work.rglob("*"))
            if path.is_file()}


# ---------------------------------------------------------------- recount and flags

@pytest.mark.parametrize("value", ["acdef", " AC DE\nFG ", "ACDXBZJUO", "", "AC1", "AC*"])
def test_sequence_normalisation_matches_remoteness_module(value):
    try:
        expected = remoteness._sequence(value)
    except ValueError:
        with pytest.raises(ValueError):
            identity.normalize_sequence(value)
    else:
        assert identity.normalize_sequence(value) == expected


@pytest.mark.parametrize("seed", range(6))
def test_recount_matches_alignment_metrics_on_unambiguous_rows(seed):
    rng = random.Random(seed)
    n_cols = rng.randint(60, 200)
    target_gaps, query_gaps = rng.randint(0, 10), rng.randint(0, 10)
    n_ident = rng.randint(0, n_cols - target_gaps - query_gaps)
    row, sequences = make_alignment(n_cols=n_cols, n_ident=n_ident, target_gaps=target_gaps, query_gaps=query_gaps,
                                    q_prefix=rng.randint(0, 40), t_suffix=rng.randint(0, 40), seed=seed)
    hit = recount_alignment(row, sequences)
    reference = remoteness.alignment_metrics(row, {key: {"sequence": value} for key, value in sequences.items()})
    assert float(hit.identity) == reference["identity"]
    assert hit.residue_pairs == reference["residue_pairs"]
    assert float(hit.shorter_coverage) == reference["shorter_coverage"]
    assert hit.identity == Fraction(n_ident, n_cols)


@pytest.mark.parametrize("exported", [40, 37])
def test_identical_ambiguous_residues_count_as_pairs_not_identities(exported):
    row, sequences = make_alignment(n_cols=100, n_ident=40, ambiguous=3, exported=exported)
    # The reference refuses either way: through its ambiguity check or, when the export already
    # excludes X-X identities, through its nident check.
    with pytest.raises(ValueError, match="Ambiguous identical residue|nident disagrees"):
        remoteness.alignment_metrics(row, {key: {"sequence": value} for key, value in sequences.items()})
    hit = recount_alignment(row, sequences)
    assert (hit.nident, hit.residue_pairs, hit.ambiguous_identical_columns) == (37, 100, 3)
    assert hit.identity == Fraction(37, 100)


def test_recount_refuses_inconsistent_rows():
    row, sequences = make_alignment(n_cols=80, n_ident=30, exported=28)
    with pytest.raises(ValueError, match="nident disagrees"):
        recount_alignment(row, sequences)
    row, sequences = make_alignment(n_cols=80, n_ident=30)
    with pytest.raises(ValueError, match="does not match the input sequence"):
        recount_alignment(row, {"Q1": sequences["Q1"], "T1": sequences["T1"][::-1]})
    with pytest.raises(ValueError, match="outside the search inputs"):
        recount_alignment(row, {"Q1": sequences["Q1"]})


BOUNDARY_CASES = [
    # identity boundary at 0.30 over 1,000 columns
    (dict(n_cols=1000, n_ident=299), dict(bin_hit=True, d3=False)),
    (dict(n_cols=1000, n_ident=300), dict(bin_hit=True, d3=True)),
    # 50 residue pairs, counted as pairs rather than columns
    (dict(n_cols=49, n_ident=49), dict(any_hit=True, bin_hit=False, d3=False)),
    (dict(n_cols=50, n_ident=50), dict(bin_hit=True, d3=True)),
    (dict(n_cols=55, n_ident=48, target_gaps=4, query_gaps=3), dict(bin_hit=False, d3=False)),
    # E-value boundary, inclusive at 1e-3
    (dict(n_cols=100, n_ident=60, evalue="1.000E-03"), dict(any_hit=True, d3=True)),
    (dict(n_cols=100, n_ident=60, evalue="9.999E-04"), dict(any_hit=True, d3=True)),
    (dict(n_cols=100, n_ident=60, evalue="1.001E-03"), dict(any_hit=False, bin_hit=False, d3=False, d2=False,
                                                            d1=False)),
    # D2: 50% of the shorter sequence
    (dict(n_cols=100, n_ident=40, q_prefix=50, q_suffix=50, t_prefix=150, t_suffix=150),
     dict(d3=True, d2=True, d2_min_pairs=True, d1=False)),
    (dict(n_cols=100, n_ident=40, q_prefix=50, q_suffix=51, t_prefix=150, t_suffix=150),
     dict(d3=True, d2=False, d1=False)),
    (dict(n_cols=40, n_ident=40, q_suffix=40, t_suffix=200), dict(d3=False, d2=True, d2_min_pairs=False)),
    # D1: 80% of both sequences
    (dict(n_cols=100, n_ident=30, q_prefix=25, t_prefix=25), dict(d3=True, d2=True, d1=True)),
    (dict(n_cols=100, n_ident=30, q_prefix=25, t_prefix=25, t_suffix=1), dict(d3=True, d1=False)),
    (dict(n_cols=100, n_ident=29, q_prefix=25, t_prefix=25), dict(d3=False, d2=False, d1=False)),
]


@pytest.mark.parametrize(("kwargs", "expected"), BOUNDARY_CASES)
def test_hit_flags_at_boundaries(kwargs, expected):
    row, sequences = make_alignment(**kwargs)
    hit = recount_alignment(row, sequences)
    for name, value in expected.items():
        assert getattr(hit, name) is value, name
    assert Hit.from_row(hit.to_row()) == hit


@pytest.mark.parametrize(("kwargs", "expected"), BOUNDARY_CASES)
def test_boundary_flags_through_the_search_output_reader(kwargs, expected, tmp_path):
    row, sequences = make_alignment(**kwargs)
    self_hit = dict(row, target="Q1", tlen=row["qlen"])
    path = write_search_rows(tmp_path / "alignments.sorted.tsv", [row, self_hit])
    hits, self_hits = read_hits(path, sequences)
    assert self_hits == 1 and len(hits) == 1
    stored = hits[0].to_row()
    assert (stored["query"], stored["target"]) == ("Q1", "T1")
    for name, value in expected.items():
        assert stored[name] == ("1" if value else "0"), name
    assert Hit.from_row(stored) == hits[0]


def test_search_output_reader_refuses_a_repeated_pair(tmp_path):
    row, sequences = make_alignment(n_cols=100, n_ident=40)
    path = write_search_rows(tmp_path / "alignments.sorted.tsv", [row, row])
    with pytest.raises(ValueError, match="Several alignments"):
        read_hits(path, sequences)


def test_pair_takes_each_flag_from_either_direction():
    # q -> t has 45 residue pairs (no bin hit); t -> q has 60 pairs at identity 0.5 (D3).
    forward = _hit("P1", "P2", 40, alnlen=45, length=45)
    backward = _hit("P2", "P1", 30, alnlen=60, length=60)
    assert (forward.bin_hit, forward.d3, backward.d3) == (False, False, True)
    for hits in ([forward, backward], [backward, forward]):
        (pair,) = undirected_pairs(hits, {"P1", "P2"})
        assert (pair["a"], pair["b"], pair["n_directions"]) == ("P1", "P2", "2")
        assert (pair["bin_hit"], pair["d3"], pair["max_bin_identity"]) == ("1", "1", "0.500000")
    (only_forward,) = undirected_pairs([forward], {"P1", "P2"})
    assert (only_forward["d3"], only_forward["max_bin_identity"]) == ("0", "")


def test_stored_flags_must_match_recount():
    row, sequences = make_alignment(n_cols=100, n_ident=40)
    stored = recount_alignment(row, sequences).to_row()
    stored["d3"] = "0"
    with pytest.raises(ValueError, match="Stored flags disagree"):
        Hit.from_row(stored)


# ---------------------------------------------------------------- training inputs

def test_fasta_step_writes_sorted_training_inputs_without_labels(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    summary = build_inputs(tmp_path / "work", 0, official_split_dir=dataset[0], metallo_folds_dir=dataset[1])
    work = tmp_path / "work"
    targets = identity.read_fasta(work / "target_train_fold0.fasta")
    queries = identity.read_fasta(work / "query_metallo_train.fasta")
    assert list(targets) == sorted(acc for acc, fold in TEST_FOLD.items() if fold != 0)
    assert list(queries) == ["A00001", "A00002", "A00003", "A00004", "A00005", "A00006"]
    nodes = {row["accession"]: row for row in read_tsv(work / "nodes_metallo_train.tsv", NODE_COLUMNS)}
    assert nodes["A00001"]["donor_pdb_id"] == nodes["A00003"]["donor_pdb_id"] == "1ABC"
    assert nodes["A00004"]["n_structure_ids"] == "2"
    receipt = json.loads((work / "fasta_receipt_metallo_train.json").read_text())
    assert receipt["labels_read"] is False and receipt["freeze1_sha256"] is None
    assert receipt["counts"]["metallo_training_files_per_query"] == {"4": 6}
    for output in receipt["outputs"].values():
        assert sha256_file(work / output["file"]) == output["sha256"]
    assert summary["counts"]["query_sequences"] == 6
    for name, text in output_texts(work).items():
        assert EC_TEXT not in text and "__EC_" not in text and "__chain_" not in text, name


def test_fasta_step_refuses_conflicting_duplicates_and_counts_identical_ones(tmp_path):
    sequences = synthetic_sequences()
    dataset = write_dataset(tmp_path / "same", official_overrides={0: [("A00002", EC_TEXT, sequences["A00002"])]})
    summary = build_inputs(tmp_path / "w1", 0, official_split_dir=dataset[0], metallo_folds_dir=dataset[1])
    assert summary["counts"]["target_identical_duplicate_rows"] == 1
    dataset = write_dataset(tmp_path / "conflict", official_overrides={0: [("A00002", EC_TEXT, "ACDEFGHIK")]})
    with pytest.raises(ValueError, match="Conflicting duplicate sequences for A00002"):
        build_inputs(tmp_path / "w2", 0, official_split_dir=dataset[0], metallo_folds_dir=dataset[1])


def test_training_paths_refuse_held_out_names_and_directories(tmp_path):
    for directory in ("test", "metadata"):
        held_out = tmp_path / directory
        held_out.mkdir()
        (held_out / "split30_train_split_0.csv").write_text("Entry,Sequence\n")
        with pytest.raises(ValueError, match="held-out directory"):
            training_file(held_out, "official", 0)
    curated = tmp_path / "split30_test_split_0_curate.csv"
    curated.write_text("Entry,Sequence\n")
    with pytest.raises(ValueError, match="non-training file name"):
        assert_training_file(curated, "official", 0)
    with pytest.raises(ValueError, match="non-training file name"):
        assert_training_file(tmp_path / "CLEAN_30_train_test_split_0_test.csv", "metallo", 0)
    with pytest.raises(ValueError, match="not found"):
        training_file(tmp_path, "official", 3)


def test_training_file_symlink_to_another_name_is_refused(tmp_path):
    held_out = tmp_path / "split30_test_split_0_curate.csv"
    held_out.write_text("Entry,Sequence\n")
    (tmp_path / "split30_train_split_0.csv").symlink_to(held_out)
    with pytest.raises(ValueError, match="resolves to another name"):
        training_file(tmp_path, "official", 0)


@pytest.mark.parametrize("donor", ["NAN", "na", "-", "NONE", "1abc_A", "1ABCD", "ABCD"])
def test_invalid_donor_ids_are_refused_with_their_location(tmp_path, donor):
    dataset = write_dataset(tmp_path / "data", donors={("A00002", 3): donor})
    with pytest.raises(ValueError, match=r"Invalid single_donor_selected_alphafill_pdb_id .* "
                                         r"CLEAN_30_train_test_split_3_train\.csv:\d+"):
        build_inputs(tmp_path / "work", 0, official_split_dir=dataset[0], metallo_folds_dir=dataset[1])


def test_fasta_receipt_counts_the_largest_donor_group(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    summary = build_inputs(tmp_path / "work", 0, official_split_dir=dataset[0], metallo_folds_dir=dataset[1])
    # A00001 ("1abc") and A00003 ("1ABC") share one donor after uppercasing.
    assert summary["counts"]["largest_donor_group_size"] == 2


def test_donor_disagreement_across_training_files_is_refused(tmp_path):
    dataset = write_dataset(tmp_path / "data", donors={("A00002", 3): "9ZZZ"})
    with pytest.raises(ValueError, match="Selected donor differs .* A00002"):
        build_inputs(tmp_path / "work", 0, official_split_dir=dataset[0], metallo_folds_dir=dataset[1])


def test_complement_queries_are_locked_until_freeze1(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    kwargs = dict(query_set="complement", official_split_dir=dataset[0], metallo_folds_dir=dataset[1])
    with pytest.raises(ValueError, match="locked until Freeze 1"):
        build_inputs(tmp_path / "work", 1, **kwargs)
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        build_inputs(tmp_path / "work", 1, freeze1_sha256="abc", **kwargs)
    assert not (tmp_path / "work").exists()
    build_inputs(tmp_path / "work", 1, freeze1_sha256=FREEZE1, **kwargs)
    receipt = json.loads((tmp_path / "work" / "fasta_receipt_complement.json").read_text())
    assert receipt["freeze1_sha256"] == FREEZE1 and receipt["freeze1_verification"] == "recorded_not_verified"
    # The metallo cohort complement: A00010 is in the official complement but not in the cohort, so it is not queried.
    assert list(identity.read_fasta(tmp_path / "work" / "query_complement.fasta")) == ["A00001", "A00005"]
    assert receipt["counts"]["official_training_files_per_query"] == {"4": 2}
    assert receipt["complement_scope"] == identity.COMPLEMENT_SCOPE


def test_complement_sequences_must_agree_across_training_files(tmp_path):
    # A00001 is a fold-1 complement protein, read from the fold 0, 2, 3 and 4 training files.
    dataset = write_dataset(tmp_path / "data", fold_sequences={("A00001", 3): synthetic_sequences()["A00009"]})
    with pytest.raises(ValueError, match="Sequence of A00001 differs between official training files"):
        build_inputs(tmp_path / "w", 1, query_set="complement", freeze1_sha256=FREEZE1,
                     official_split_dir=dataset[0], metallo_folds_dir=dataset[1])


# ---------------------------------------------------------------- search

def test_search_command_pins_every_setting_once(tmp_path):
    query, target, output, tmp = (tmp_path / name for name in ("q.fasta", "t.fasta", "s/out.tsv", "tmp"))
    command = build_search_command(Path("mmseqs"), query, target, output, tmp, threads=3, split_memory_limit="1G")
    assert command[:6] == ["mmseqs", "easy-search", str(query), str(target), str(output), str(tmp)]
    pinned = {"--search-type": "1", "-s": "7.5", "-e": "1e-3", "--alignment-mode": "3", "--seq-id-mode": "0",
              "-a": "1", "--min-seq-id": "0", "--min-aln-len": "0", "-c": "0", "--cov-mode": "0", "--mask": "1",
              "--comp-bias-corr": "1", "--max-seqs": "1000000", "--max-accept": "1000000",
              "--max-rejected": "1000000", "--threads": "3", "--format-mode": "4"}
    for flag, value in pinned.items():
        assert command.count(flag) == 1 and command[command.index(flag) + 1] == value, flag
    fields = command[command.index("--format-output") + 1].split(",")
    assert {"query", "target", "nident", "alnlen", "evalue", "bits", "qaln", "taln", "qlen", "tlen"} <= set(fields)
    check_search_command(command, query, target, output)
    with pytest.raises(ValueError, match="exactly one -e"):
        check_search_command(command + ["-e", "1e-2"], query, target, output)
    altered = list(command)
    altered[altered.index("--min-aln-len") + 1] = "50"
    with pytest.raises(ValueError, match="--min-aln-len"):
        check_search_command(altered, query, target, output)


def test_search_command_refuses_options_outside_the_pinned_settings(tmp_path):
    query, target, output, tmp = (tmp_path / name for name in ("q.fasta", "t.fasta", "s/out.tsv", "tmp"))
    command = build_search_command(Path("mmseqs"), query, target, output, tmp, threads=2, split_memory_limit="2G")
    unpinned = re.escape("outside the pinned settings: ['--exhaustive-search', '--prefilter-mode']")
    with pytest.raises(ValueError, match=unpinned):
        check_search_command(command + ["--exhaustive-search", "1", "--prefilter-mode", "2"], query, target, output)
    with pytest.raises(ValueError, match="flag-value pairs"):
        check_search_command(command + ["--exhaustive-search"], query, target, output)
    with pytest.raises(ValueError, match="exactly one --split-memory-limit"):
        check_search_command(command + ["--split-memory-limit", "4G"], query, target, output)
    without_threads = list(command)
    del without_threads[without_threads.index("--threads"):without_threads.index("--threads") + 2]
    with pytest.raises(ValueError, match="exactly one --threads"):
        check_search_command(without_threads, query, target, output)
    bad_threads = list(command)
    bad_threads[bad_threads.index("--threads") + 1] = "0"
    with pytest.raises(ValueError, match="positive integer"):
        check_search_command(bad_threads, query, target, output)


def _process_alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc to inspect child processes")
def test_search_runner_kills_the_whole_process_group_on_timeout(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = f"sleep 30 & echo $! > {shlex.quote(str(pid_file))}; wait"
    with pytest.raises(subprocess.TimeoutExpired):
        identity.run_process_group(["/bin/sh", "-c", script], timeout=1)
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _process_alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_alive(child)
    completed = identity.run_process_group(["/bin/sh", "-c", "echo ok"], capture_output=True, text=True, timeout=10)
    assert (completed.returncode, completed.stdout) == (0, "ok\n")


def test_binary_checks_refuse_wrong_sha_or_version(tmp_path):
    wrong_version = fake_binary(tmp_path / "a", "0000000")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_binary(wrong_version, expected_sha256="0" * 64, expected_version=MMSEQS_VERSION)
    with pytest.raises(ValueError, match="version '0000000'"):
        verify_binary(wrong_version, expected_sha256=sha256_file(wrong_version), expected_version=MMSEQS_VERSION)
    right = fake_binary(tmp_path / "b", MMSEQS_VERSION)
    assert verify_binary(right, expected_sha256=sha256_file(right),
                         expected_version=MMSEQS_VERSION) == (sha256_file(right), MMSEQS_VERSION)


def test_search_refuses_wrong_binary_before_running(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    work = tmp_path / "work"
    build_inputs(work, 0, official_split_dir=dataset[0], metallo_folds_dir=dataset[1])
    binary = fake_binary(tmp_path / "bin")
    fake = FakeMMseqs()
    with pytest.raises(ValueError, match="SHA-256"):
        run_search(work, 0, binary=binary, runner=fake)
    with pytest.raises(ValueError, match="version"):
        run_search(work, 0, binary=binary, expected_sha256=sha256_file(binary), runner=FakeMMseqs(version="1"))
    with pytest.raises(ValueError, match="not the pinned 18-8cc5c build"):
        run_search(work, 0, binary=binary, expected_sha256=sha256_file(binary), runner=fake)
    assert all(call[1] == "version" for call in fake.calls)
    assert not (work / "search_metallo_train").exists()


def test_mocked_search_binds_sorted_output_and_argv(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    work, fake = run_metallo_pipeline(tmp_path, "work", dataset)
    version_call, search_call = fake.calls
    assert version_call[1:] == ["version"]
    assert search_call[1] == "easy-search"
    assert Path(search_call[2]).name == "query_metallo_train.fasta"
    assert Path(search_call[3]).name == "target_train_fold0.fasta"
    assert search_call[search_call.index("-e") + 1] == "1e-3"
    assert search_call[search_call.index("--threads") + 1] == "2"
    receipt = json.loads((work / "search_metallo_train" / "search_receipt.json").read_text())
    assert receipt["version"] == MMSEQS_VERSION and receipt["pinned_binary"] is False
    assert receipt["allow_unpinned_binary"] is True
    assert receipt["binary_sha256"] == sha256_file(Path(search_call[0]))
    raw = (work / "search_metallo_train" / "alignments.raw.tsv").read_text()
    ordered = (work / "search_metallo_train" / "alignments.sorted.tsv").read_text()
    assert ordered == sort_alignment_text(raw) != raw
    keys = [tuple(line.split("\t")[:2]) for line in ordered.splitlines()[1:]]
    assert keys == sorted(keys)
    assert receipt["sorted_output"]["sha256"] == sha256_file(work / "search_metallo_train" / "alignments.sorted.tsv")
    reused = run_search(work, 0, binary=Path(search_call[0]), expected_sha256=receipt["binary_sha256"], runner=fake,
                        allow_unpinned_binary=True)
    assert reused["status"] == "reused_verified_search"
    with pytest.raises(ValueError, match="unpinned MMseqs2 binary"):
        run_search(work, 0, binary=Path(search_call[0]), expected_sha256=receipt["binary_sha256"], runner=fake)


def test_unpinned_searches_do_not_feed_later_steps(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    work, fake = run_metallo_pipeline(tmp_path, "work", dataset, fold=1)
    assert json.loads((work / "edges_receipt_metallo_train.json").read_text())["allow_unpinned_binary"] is True
    assert json.loads((work / "components_receipt_inner.json").read_text())["allow_unpinned_binary"] is True
    with pytest.raises(ValueError, match="unpinned MMseqs2 binary"):
        build_edges(work, 1)
    with pytest.raises(ValueError, match="unpinned MMseqs2 binary"):
        build_inner_components(work, 1)
    with pytest.raises(ValueError, match="unpinned MMseqs2 binary"):
        build_cohort_components([work], tmp_path / "cohort", freeze1_sha256=FREEZE1)
    build_inputs(work, 1, query_set="complement", freeze1_sha256=FREEZE1, official_split_dir=dataset[0],
                 metallo_folds_dir=dataset[1])
    binary = Path(fake.calls[0][0])
    with pytest.raises(ValueError, match="always needs the pinned MMseqs2 binary"):
        run_search(work, 1, query_set="complement", binary=binary, expected_sha256=sha256_file(binary),
                   freeze1_sha256=FREEZE1, allow_unpinned_binary=True, runner=fake)
    with pytest.raises(ValueError, match="not the pinned"):
        run_search(work, 1, query_set="complement", binary=binary, expected_sha256=sha256_file(binary),
                   freeze1_sha256=FREEZE1, runner=fake)
    assert not (work / "search_complement").exists()


def test_strata_refuse_a_complement_search_that_is_no_longer_pinned(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "data")
    work, fake = run_metallo_pipeline(tmp_path, "work", dataset, fold=1)
    with monkeypatch.context() as patched:
        patched.setattr(identity, "MMSEQS_BINARY_SHA256", sha256_file(Path(fake.calls[0][0])))
        run_complement_chain(work, 1, dataset, fake)
    with pytest.raises(ValueError, match="unpinned MMseqs2 binary"):
        build_strata(work, 1, freeze1_sha256=FREEZE1)
    assert not (work / "strata_receipt.json").exists()


def test_search_and_edges_refuse_changed_inputs_and_outputs(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    for name, tampered, message in (
        ("sorted", "search_metallo_train/alignments.sorted.tsv", "Sorted search output changed after the search"),
        ("query", "query_metallo_train.fasta", "query_metallo_train.fasta changed after the fasta step"),
    ):
        work, _ = run_metallo_pipeline(tmp_path, name, dataset)
        with (work / tampered).open("a") as handle:
            handle.write("\n")
        with pytest.raises(ValueError, match=message):
            build_edges(work, 0, allow_unpinned_binary=True)
        with pytest.raises(ValueError, match=message):
            build_inner_components(work, 0, allow_unpinned_binary=True)


def test_inner_components_refuse_changed_or_mismatched_edge_tables(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    work, _ = run_metallo_pipeline(tmp_path, "pairs", dataset)
    pairs = work / "inner_pairs.tsv"
    original = pairs.read_text()
    pairs.write_text(original.replace("\t1\t", "\t0\t", 1))
    with pytest.raises(ValueError, match="inner_pairs.tsv of fold 0 changed after the edges step"):
        build_inner_components(work, 0, allow_unpinned_binary=True)
    pairs.write_text(original)
    receipt_path = work / "edges_receipt_metallo_train.json"
    receipt = json.loads(receipt_path.read_text())
    for change, message in ((dict(fold=1), "belongs to fold 1"),
                            (dict(search=dict(receipt["search"], sorted_output_sha256="0" * 64)),
                             "built from another search output")):
        receipt_path.write_text(json.dumps(receipt | change))
        with pytest.raises(ValueError, match=message):
            build_inner_components(work, 0, allow_unpinned_binary=True)


# ---------------------------------------------------------------- edges and components

def test_edges_components_and_receipts_from_mocked_search(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    sequences = synthetic_sequences()
    work, _ = run_metallo_pipeline(tmp_path, "work", dataset)
    edges = read_tsv(work / "edges_metallo_train.tsv", EDGE_COLUMNS)
    assert [(row["query"], row["target"]) for row in edges] == [
        ("A00001", "A00002"), ("A00001", "A00007"), ("A00002", "A00001"), ("A00002", "A00007"),
        ("A00003", "A00004"), ("A00004", "A00003"), ("A00005", "A00006"), ("A00006", "A00005")]
    receipt = json.loads((work / "edges_receipt_metallo_train.json").read_text())
    assert receipt["counts"]["self_hits_dropped"] == 6 and receipt["counts"]["hits"] == 8
    pairs = {(row["a"], row["b"]): row for row in read_tsv(work / "inner_pairs.tsv", PAIR_COLUMNS)}
    assert set(pairs) == {("A00001", "A00002"), ("A00003", "A00004"), ("A00005", "A00006")}
    assert pairs[("A00001", "A00002")]["n_directions"] == "2" and pairs[("A00001", "A00002")]["d3"] == "1"
    assert pairs[("A00001", "A00002")]["max_bin_identity"] == "0.400000"
    assert (pairs[("A00003", "A00004")]["bin_hit"], pairs[("A00003", "A00004")]["d3"]) == ("1", "0")
    assert pairs[("A00005", "A00006")]["max_bin_identity"] == "1.000000"
    nonmetallo = read_tsv(work / "nonmetallo_d3_hits.tsv", EDGE_COLUMNS)
    a2_a7 = sum(a == b for a, b in zip(sequences["A00002"], sequences["A00007"])) / LENGTH >= 0.3
    assert [(row["query"], row["target"]) for row in nonmetallo] == (
        [("A00001", "A00007")] + ([("A00002", "A00007")] if a2_a7 else []))
    components = {row["accession"]: row for row in read_tsv(work / "components_inner.tsv",
                                                            identity.COMPONENT_COLUMNS)}
    assert components["A00001"]["component_id"] == components["A00002"]["component_id"] == \
        components["A00003"]["component_id"] == component_id(["A00001", "A00002", "A00003"])
    assert components["A00004"]["component_size"] == "1"
    assert components["A00005"]["component_id"] == components["A00006"]["component_id"]
    component_receipt = json.loads((work / "components_receipt_inner.json").read_text())
    assert component_receipt["summary"]["links_by_type"] == {"d3": 2, "donor": 1, "exact_sequence": 1}
    assert component_receipt["summary"]["component_size_histogram"] == {"1": 1, "2": 1, "3": 1}
    for name, text in output_texts(work).items():
        assert EC_TEXT not in text and "__EC_" not in text, name


def test_outputs_and_receipts_are_deterministic(tmp_path):
    dataset = write_dataset(tmp_path / "data")
    first, _ = run_metallo_pipeline(tmp_path, "first", dataset, order="reverse")
    second, _ = run_metallo_pipeline(tmp_path, "second", dataset, order="shuffled")
    a, b = output_texts(first), output_texts(second)
    run_specific = {"search_metallo_train/search_receipt.json", "search_metallo_train/alignments.raw.tsv",
                    "search_metallo_train/search.log"}
    assert set(a) == set(b)
    assert a["search_metallo_train/alignments.raw.tsv"] != b["search_metallo_train/alignments.raw.tsv"]
    for name in sorted(set(a) - run_specific):
        assert a[name] == b[name], name
    with pytest.raises(ValueError, match="Refusing to replace"):
        identity.write_once(first / "inner_pairs.tsv", "different\n")


def test_union_find_components_with_transitive_and_empty_donor_links():
    nodes = [
        {"accession": "P1", "sequence_sha256": "s1", "donor_pdb_id": ""},
        {"accession": "P2", "sequence_sha256": "s2", "donor_pdb_id": "1ABC"},
        {"accession": "P3", "sequence_sha256": "s3", "donor_pdb_id": "1ABC"},
        {"accession": "P4", "sequence_sha256": "s3", "donor_pdb_id": ""},
        {"accession": "p4", "sequence_sha256": "s5", "donor_pdb_id": ""},
        {"accession": "P6", "sequence_sha256": "s6", "donor_pdb_id": ""},
        {"accession": "P7", "sequence_sha256": "s7", "donor_pdb_id": ""},
    ]
    links = [Link("d3", "P1", "P2")] + node_links(nodes, identity.INNER_LINK_TYPES)
    assert sorted(links) == [Link("alias", "P4", "p4"), Link("d3", "P1", "P2"), Link("donor", "P2", "P3"),
                             Link("exact_sequence", "P3", "P4")]
    components = build_components([node["accession"] for node in nodes], links)
    chain = component_id(["P1", "P2", "P3", "P4", "p4"])
    assert {node: components[node] for node in ("P1", "P2", "P3", "P4", "p4")} == dict.fromkeys(
        ("P1", "P2", "P3", "P4", "p4"), chain)
    assert components["P6"] != components["P7"]
    assert build_components(reversed(list(components)), reversed(links)) == components
    summary = component_summary(components, links)
    assert summary["largest_component_size"] == 5 and summary["largest_component_exceeds_quarter"] is True
    with pytest.raises(ValueError, match="outside the nodes"):
        build_components(["P1"], [Link("donor", "P1", "P9")])


def test_cohort_clusters_need_freeze1_and_join_folds(tmp_path, pinned_fake):
    dataset = write_dataset(tmp_path / "data")
    works = [run_metallo_pipeline(tmp_path, f"fold{fold}", dataset, fold=fold, allow_unpinned=False)[0]
             for fold in range(5)]
    with pytest.raises(ValueError, match="locked until Freeze 1"):
        build_cohort_components(works, tmp_path / "cohort", freeze1_sha256=None)
    with pytest.raises(ValueError, match="folds 0..4"):
        build_cohort_components(works[:4], tmp_path / "partial", freeze1_sha256=FREEZE1)
    build_cohort_components(works, tmp_path / "cohort", freeze1_sha256=FREEZE1)
    rows = {row["accession"]: row["component_id"] for row in
            read_tsv(tmp_path / "cohort" / "components_cohort.tsv", identity.COMPONENT_COLUMNS)}
    assert sorted(rows) == ["A00001", "A00002", "A00003", "A00004", "A00005", "A00006"]
    assert len({rows[acc] for acc in ("A00001", "A00002", "A00003", "A00004")}) == 1
    assert rows["A00005"] == rows["A00006"] != rows["A00001"]
    receipt = json.loads((tmp_path / "cohort" / "components_receipt_cohort.json").read_text())
    assert receipt["freeze1_sha256"] == FREEZE1 and receipt["link_types"] == ["any_hit", "donor"]
    assert receipt["inputs"]["0"]["binary_sha256"] == sha256_file(pinned_fake)
    edges_receipt = works[2] / "edges_receipt_metallo_train.json"
    edges_receipt.write_text(json.dumps(json.loads(edges_receipt.read_text()) | {"fold": 3}))
    with pytest.raises(ValueError, match="belongs to fold 3"):
        build_cohort_components(works, tmp_path / "again", freeze1_sha256=FREEZE1)


# ---------------------------------------------------------------- strata

@pytest.mark.parametrize(
    ("value", "label"),
    [(None, "no_hit"), (Fraction(0), "<=20"), (Fraction(1, 5), "<=20"), (Fraction(201, 1000), "(20,25]"),
     (Fraction(1, 4), "(20,25]"), (Fraction(251, 1000), "(25,30)"), (Fraction(2999, 10000), "(25,30)"),
     (Fraction(3, 10), "[30,50]"), (Fraction(1, 2), "[30,50]"), (Fraction(501, 1000), ">50"), (Fraction(1), ">50")],
)
def test_identity_bins(value, label):
    assert identity_bin(value) == label


def _hit(query, target, nident, alnlen=1000, evalue="1.000E-10", bits="50.0", span=None, length=1000):
    span = alnlen if span is None else span
    return Hit(query=query, target=target, evalue_text=evalue, bits_text=bits, alnlen=alnlen, nident=nident,
               residue_pairs=alnlen, query_span=span, qlen=length, target_span=span, tlen=length)


def test_strata_rows_flags_bins_and_nearest_hit():
    queries = [{"accession": acc, "donor_pdb_id": donor} for acc, donor in
               (("C3", ""), ("C1", "1ABC"), ("C2", "9XYZ"), ("C4", ""))]
    hits = [
        _hit("C1", "T2", 300, bits="80.0"), _hit("C1", "T1", 300, bits="80.0"), _hit("C1", "T3", 290, bits="90.0"),
        _hit("C2", "T1", 2999, alnlen=10000, length=10000), _hit("C2", "T9", 40, alnlen=40, length=40),
        _hit("C4", "T1", 900, evalue="2.000E-03"),
    ]
    rows = {row["accession"]: row for row in strata_rows(queries, hits, {"1ABC"}, {"C1": "cmp_x"})}
    assert list(rows) == ["C1", "C2", "C3", "C4"]
    c1, c2, c3, c4 = (rows[acc] for acc in ("C1", "C2", "C3", "C4"))
    assert (c1["identity_bin"], c1["max_bin_identity"], c1["max_bin_target"]) == ("[30,50]", "0.300000", "T1")
    assert (c1["has_d3"], c1["donor_shared_with_metallo_train"], c1["no_d3_hit"]) == ("1", "1", "0")
    assert (c1["nearest_target"], c1["nearest_bits"], c1["cluster_id"]) == ("T3", "90.0", "cmp_x")
    assert (c2["identity_bin"], c2["has_d3"], c2["no_d3_hit_no_shared_donor"]) == ("(25,30)", "0", "1")
    assert c2["has_d2"] == "1" and c2["has_d2_min_pairs"] == "0"
    assert (c3["identity_bin"], c3["has_any_hit"], c3["nearest_target"]) == ("no_hit", "0", "")
    assert (c4["identity_bin"], c4["has_any_hit"], c4["no_d3_hit"]) == ("no_hit", "0", "1")


def test_strata_end_to_end_after_freeze1(tmp_path, capsys, pinned_fake):
    dataset = write_dataset(tmp_path / "data")
    work, fake = run_metallo_pipeline(tmp_path, "fold1", dataset, fold=1, allow_unpinned=False)
    binary = Path(fake.calls[0][0])
    search_kwargs = dict(binary=binary, expected_sha256=sha256_file(binary), runner=fake)
    with pytest.raises(SystemExit) as refused:
        identity.main(["strata", "--fold", "1", "--work", str(work)])
    assert refused.value.code == 2 and "locked until Freeze 1" in capsys.readouterr().err
    with pytest.raises(ValueError, match="locked until Freeze 1"):
        run_search(work, 1, query_set="complement", binary=binary, runner=fake)
    # The training-side chain recorded no Freeze 1 value, so a later step cannot add one.
    with pytest.raises(ValueError, match="differs from None recorded in fasta_receipt_metallo_train.json"):
        run_search(work, 1, freeze1_sha256=FREEZE1, **search_kwargs)
    build_inputs(work, 1, query_set="complement", freeze1_sha256=FREEZE1, official_split_dir=dataset[0],
                 metallo_folds_dir=dataset[1])
    with pytest.raises(ValueError, match="recorded in fasta_receipt_complement.json"):
        run_search(work, 1, query_set="complement", freeze1_sha256=OTHER_FREEZE1, **search_kwargs)
    run_search(work, 1, query_set="complement", freeze1_sha256=FREEZE1, **search_kwargs)
    with pytest.raises(ValueError, match="locked until Freeze 1"):
        build_edges(work, 1, query_set="complement")
    with pytest.raises(ValueError, match="recorded in fasta_receipt_complement.json"):
        build_edges(work, 1, query_set="complement", freeze1_sha256=OTHER_FREEZE1)
    assert not (work / "edges_receipt_complement.json").exists()
    build_edges(work, 1, query_set="complement", freeze1_sha256=FREEZE1)
    with pytest.raises(ValueError, match="recorded in fasta_receipt_complement.json"):
        build_strata(work, 1, freeze1_sha256=OTHER_FREEZE1)
    edges_receipt = work / "edges_receipt_complement.json"
    original = edges_receipt.read_text()
    edges_receipt.write_text(json.dumps(json.loads(original) | {"freeze1_sha256": OTHER_FREEZE1}))
    with pytest.raises(ValueError, match="recorded in edges_receipt_complement.json"):
        build_strata(work, 1, freeze1_sha256=FREEZE1)
    edges_receipt.write_text(original)
    assert not (work / "strata_receipt.json").exists()
    assert identity.main(["strata", "--fold", "1", "--work", str(work), "--freeze1-sha256", FREEZE1]) == 0
    rows = {row["accession"]: row for row in read_tsv(work / "strata_fold1.tsv", STRATA_COLUMNS)}
    assert list(rows) == ["A00001", "A00005"]
    assert (rows["A00001"]["identity_bin"], rows["A00001"]["max_bin_target"]) == ("[30,50]", "A00007")
    assert rows["A00001"]["donor_shared_with_metallo_train"] == "1" and rows["A00001"]["nearest_target"] == "A00007"
    assert (rows["A00005"]["identity_bin"], rows["A00005"]["donor_shared_with_metallo_train"]) == (">50", "0")
    receipt = json.loads((work / "strata_receipt.json").read_text())
    assert receipt["freeze1_sha256"] == FREEZE1 and receipt["curated_test_membership_read"] is False
    assert receipt["identity_bin_status"] == "provisional_until_freeze1"
    assert receipt["complement_scope"] == identity.COMPLEMENT_SCOPE


def test_strata_cluster_ids_need_a_matching_cohort_receipt(tmp_path, pinned_fake):
    dataset = write_dataset(tmp_path / "data")
    runs = [run_metallo_pipeline(tmp_path, f"fold{fold}", dataset, fold=fold, allow_unpinned=False)
            for fold in range(5)]
    works = [work for work, _ in runs]
    build_cohort_components(works, tmp_path / "cohort", freeze1_sha256=FREEZE1)
    build_cohort_components(works, tmp_path / "other", freeze1_sha256=OTHER_FREEZE1)
    work, fake = runs[1]
    run_complement_chain(work, 1, dataset, fake)
    clusters = tmp_path / "cohort" / "components_cohort.tsv"
    with pytest.raises(ValueError, match="recorded in components_receipt_cohort.json"):
        build_strata(work, 1, freeze1_sha256=FREEZE1, clusters=tmp_path / "other" / "components_cohort.tsv")
    shutil.copytree(tmp_path / "cohort", tmp_path / "tampered")
    tampered = tmp_path / "tampered" / "components_cohort.tsv"
    tampered.write_text(tampered.read_text().replace("cmp_", "cmq_", 1))
    with pytest.raises(ValueError, match="changed after the cohort components step"):
        build_strata(work, 1, freeze1_sha256=FREEZE1, clusters=tampered)
    renamed = tmp_path / "cohort" / "clusters.tsv"
    shutil.copyfile(clusters, renamed)
    with pytest.raises(ValueError, match="--clusters must be a components_cohort.tsv"):
        build_strata(work, 1, freeze1_sha256=FREEZE1, clusters=renamed)
    assert not (work / "strata_receipt.json").exists()
    build_strata(work, 1, freeze1_sha256=FREEZE1, clusters=clusters)
    cohort = {row["accession"]: row["component_id"] for row in read_tsv(clusters, identity.COMPONENT_COLUMNS)}
    rows = {row["accession"]: row["cluster_id"] for row in read_tsv(work / "strata_fold1.tsv", STRATA_COLUMNS)}
    assert rows == {"A00001": cohort["A00001"], "A00005": cohort["A00005"]}
    receipt = json.loads((work / "strata_receipt.json").read_text())
    assert receipt["inputs"]["clusters_sha256"] == sha256_file(clusters)


# ---------------------------------------------------------------- inner folds

INNER_EC = {"1": "1.1.1.1", "2": "2.7.1.1", "3": "3.4.24.17", "5": "5.3.1.1", "7": "7.1.1.1"}
INNER_CLASSES = {
    "B0001": "3", "B0002": "3", "B0003": "3", "B0004": "3", "B0005": "3", "B0006": "1", "B0007": "1",
    "B0008": "2", "B0009": "2", "B0010": "7", "B0011": "7", "B0012": "5", "B0013": "5", "B0014": "5",
    "B0015": "5", "B0016": "1", "B0017": "1", "B0018": "2", "B0019": "2", "B0020": "3", "B0021": "3",
    "B0022": "3", "B0023": "1", "B0024": "2",
}
BRIDGE = "X0001"  # two EC classes: excluded from inner folds, still joins B0008 and B0009
NON_METALLO = ("N0001", "N0002", "N0003", "N0004", "N0005")
INNER_DONORS = {"B0006": "6ABC", "B0007": "6ABC", "B0010": "7AAA", "B0011": "7BBB"}
INNER_D3 = [("B0001", "B0002"), ("B0002", "B0003"), ("B0008", BRIDGE), (BRIDGE, "B0009"), ("B0012", "B0013"),
            ("B0001", "N0004"), ("B0016", "N0001"), ("B0008", "N0005")]
INNER_HITS = {frozenset(pair): "1.000E-20" for pair in INNER_D3} | {
    frozenset(("B0016", "B0017")): "1.000E-05",  # bin hit below 30% identity: no link
    frozenset(("B0016", "N0002")): "5.000E-02",  # above the E-value cut: no hit
}
# Linked through D3, exact sequence, shared donor, the excluded bridge, and D3 again.
INNER_LINKED = [{"B0001", "B0002", "B0003"}, {"B0004", "B0005"}, {"B0006", "B0007"}, {"B0008", "B0009"},
                {"B0012", "B0013"}]


def inner_sequences():
    rng = random.Random(11)
    families = {"B0001": "a", "B0002": "a", "B0003": "a", "N0004": "a", "B0008": "b", BRIDGE: "b", "B0009": "b",
                "N0005": "b", "B0012": "c", "B0013": "c", "B0016": "d", "N0001": "d"}
    bases = {name: random_sequence(rng, LENGTH) for name in "abcd"}
    sequences = {}
    for accession in [*sorted(INNER_CLASSES), BRIDGE, *NON_METALLO]:
        family = families.get(accession)
        sequences[accession] = mutate(bases[family], 24, rng) if family else random_sequence(rng, LENGTH)
    sequences["B0005"] = sequences["B0004"]
    return sequences


def write_inner_dataset(root, ec_overrides=None):
    """Fold-0 official training file (metallo and other proteins) and five identical metallo training files."""
    sequences = inner_sequences()
    ec_text = {acc: INNER_EC[cls] for acc, cls in INNER_CLASSES.items()} | {BRIDGE: "1.1.1.1;2.7.1.1"}
    ec_text |= ec_overrides or {}
    official_dir, metallo_dir = root / "split30", root / "folds"
    official_dir.mkdir(parents=True)
    metallo_dir.mkdir(parents=True)
    with (official_dir / "split30_train_split_0.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["Entry", "EC number", "Sequence"])
        writer.writerows((acc, ec_text.get(acc, EC_TEXT), sequences[acc]) for acc in sorted(sequences))
    for fold in range(5):
        with (metallo_dir / f"CLEAN_30_train_test_split_{fold}_train.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["structure", "chain_resi", "metaltype", "ecnumber", "uniprot_id", identity.DONOR_COLUMN])
            for acc in sorted(ec_text):
                writer.writerow([f"{acc}__chain_A__EC_{ec_text[acc]}", "A_1", "ZN", ec_text[acc], acc,
                                 INNER_DONORS.get(acc, "")])
    return official_dir, metallo_dir


def run_inner_pipeline(tmp_path, name="work", ec_overrides=None):
    dataset = write_inner_dataset(tmp_path / f"{name}_data", ec_overrides)
    work, _ = run_metallo_pipeline(tmp_path, name, dataset, hits=INNER_HITS)
    return work, dataset


def inner_kwargs(dataset):
    return dict(metallo_folds_dir=dataset[1], allow_unpinned_binary=True)


def read_inner_membership(work):
    receipt = json.loads((work / identity.INNER_ASSIGNMENT_RECEIPT).read_text())
    output = receipt["outputs"]["membership"]
    return read_ec_fold_membership(work / output["file"], output["sha256"]), receipt


def test_assign_components_balances_each_class_over_four_folds():
    counts = {f"common{i}": {"3": 1} for i in range(8)} | {f"rare{i}": {"1": 1} for i in range(4)}
    assignment = assign_components(counts, seed=42)
    for token, per_fold in (("3", 2), ("1", 1)):
        folds = Counter(assignment[c] for c in counts if counts[c].get(token))
        assert folds == {fold: per_fold for fold in range(4)}, token
    assert assign_components(dict(reversed(list(counts.items()))), seed=42) == assignment
    other_seed = assign_components(counts, seed=43)
    assert Counter(other_seed[c] for c in counts if "3" in counts[c]) == {fold: 2 for fold in range(4)}
    with pytest.raises(ValueError, match="without eligible proteins"):
        assign_components({"empty": {}}, seed=42)


def test_assign_components_places_larger_components_first():
    # Larger first: 3, 3, 3, 2, 2, 2, 1, 1 gives 5/4/4/4; the seeded hash order alone would give 5/5/3/4.
    sizes = {"c0": 1, "c1": 3, "c2": 2, "c3": 2, "c4": 3, "c5": 3, "c6": 2, "c7": 1}
    assignment = assign_components({name: {"3": size} for name, size in sizes.items()}, seed=42)
    assert assignment == {"c1": 0, "c5": 1, "c4": 2, "c2": 3, "c3": 3, "c6": 0, "c0": 1, "c7": 2}
    per_fold = Counter()
    for name, fold in assignment.items():
        per_fold[fold] += sizes[name]
    assert per_fold == {0: 5, 1: 4, 2: 4, 3: 4}


# Checked by hand against INNER_BALANCE_RULE: class 7 (2 proteins) first, then class 5 (4), classes 1 and 2
# (5 each), then class 3 (8); larger components first within a class; ties by the seeded hash.
INNER_GOLDEN_FOLDS = {
    "B0001": 0, "B0002": 0, "B0003": 0, "B0004": 1, "B0005": 1, "B0006": 1, "B0007": 1, "B0008": 3, "B0009": 3,
    "B0010": 1, "B0011": 0, "B0012": 2, "B0013": 2, "B0014": 0, "B0015": 3, "B0016": 3, "B0017": 2, "B0018": 2,
    "B0019": 0, "B0020": 2, "B0021": 2, "B0022": 3, "B0023": 0, "B0024": 1,
}


def test_inner_assignment_matches_the_registered_balancing_order(tmp_path):
    work, dataset = run_inner_pipeline(tmp_path)
    assign_inner(work, 0, **inner_kwargs(dataset))
    membership, receipt = read_inner_membership(work)
    assert {row["accession"]: int(row["fold"]) for row in membership.rows} == INNER_GOLDEN_FOLDS
    assert [receipt["inner_folds"][str(fold)]["n_proteins"] for fold in range(4)] == [7, 6, 6, 5]
    assert receipt["balance_rule"] == identity.INNER_BALANCE_RULE
    assert receipt["max_share_basis"] == identity.MAX_SHARE_BASIS


def test_assign_inner_keeps_components_whole_and_counts_exclusions(tmp_path):
    work, dataset = run_inner_pipeline(tmp_path)
    summary = assign_inner(work, 0, **inner_kwargs(dataset))
    membership, receipt = read_inner_membership(work)
    rows = {row["accession"]: row for row in membership.rows}
    assert sorted(rows) == sorted(INNER_CLASSES)
    assert (receipt["counts"]["nodes"], receipt["counts"]["eligible_proteins"]) == (25, 24)
    assert receipt["counts"]["excluded_ambiguous_ec_class"] == 1
    assert receipt["excluded_ambiguous_ec_class_accessions"] == [BRIDGE]
    for linked in INNER_LINKED:
        assert len({rows[acc]["fold"] for acc in linked}) == 1, linked
        assert len({rows[acc]["group_id"] for acc in linked}) == 1, linked
    components = {row["accession"]: row["component_id"]
                  for row in read_tsv(work / "components_inner.tsv", identity.COMPONENT_COLUMNS)}
    assert rows["B0008"]["group_id"] == components[BRIDGE] == component_id(["B0008", "B0009", BRIDGE])
    assert rows["B0016"]["group_id"] != rows["B0017"]["group_id"]
    assert all(rows[acc]["ec_class"] == INNER_CLASSES[acc] for acc in rows)
    assert rows["B0001"]["structure_id"] == "B0001__chain_A__EC_3.4.24.17"
    folds = receipt["inner_folds"]
    assert folds == identity.inner_fold_summary(membership.rows)
    sizes = [folds[str(fold)]["n_proteins"] for fold in range(4)]
    assert sum(sizes) == 24 and max(sizes) - min(sizes) <= 2
    assert sum(folds[str(fold)]["n_components"] for fold in range(4)) == 18
    assert receipt["counts"]["components_with_eligible_proteins"] == 18
    assert all(folds[str(fold)]["ec_class_counts"]["3"] >= 1 for fold in range(4))
    assert rows["B0010"]["fold"] != rows["B0011"]["fold"]
    assert set(receipt["inner_folds_without_ec_class"]) <= {"5", "7"}
    assert (receipt["seed"], receipt["max_component_share"], receipt["labels_read"]) == (42, "1/4", True)
    assert receipt["freeze1_sha256"] is None and receipt["status"] == "assigned"
    assert summary["outputs"]["membership"]["sha256"] == sha256_file(work / "inner_membership_fold0.csv")
    assert set(membership.rows[0]) == {"structure_id", "accession", "group_id", "fold", "ec_class"}


def test_inner_membership_round_trips_through_the_trainer_split(tmp_path):
    from training.ec_membership import ec_membership_split

    work, dataset = run_inner_pipeline(tmp_path)
    assign_inner(work, 0, **inner_kwargs(dataset))
    membership, receipt = read_inner_membership(work)
    sha = receipt["outputs"]["membership"]["sha256"]
    assert write_ec_fold_membership(membership.rows, tmp_path / "copy.csv") == sha
    pockets = [SimpleNamespace(structure_id=row["structure_id"], pocket_id=f"{row['structure_id']}_METAL_0",
                               metadata={}) for row in membership.rows]
    for inner in range(4):
        split, split_receipt = ec_membership_split(pockets, membership, fold_index=inner, n_folds=4)
        validation = {pocket.structure_id for pocket in split.val_pockets}
        assert validation == {row["structure_id"] for row in membership.rows if row["fold"] == str(inner)}
        assert len(split.train_pockets) + len(validation) == 24
        assert split_receipt["membership_file_sha256"] == sha


def test_inner_assignment_is_deterministic(tmp_path):
    dataset = write_inner_dataset(tmp_path / "data")
    works = [run_metallo_pipeline(tmp_path, name, dataset, order=order, hits=INNER_HITS)[0]
             for name, order in (("first", "reverse"), ("second", "shuffled"))]
    results = [assign_inner(work, 0, **inner_kwargs(dataset)) for work in works]
    assert results[0] == results[1]
    for name in ("inner_membership_fold0.csv", identity.INNER_ASSIGNMENT_RECEIPT):
        assert (works[0] / name).read_text() == (works[1] / name).read_text(), name
    assert assign_inner(works[0], 0, **inner_kwargs(dataset)) == results[0]
    with pytest.raises(ValueError, match="Refusing to replace"):
        assign_inner(works[0], 0, seed=43, **inner_kwargs(dataset))


def test_giant_component_is_a_refusal_with_a_receipt(tmp_path, capsys):
    work, dataset = run_inner_pipeline(tmp_path)
    with pytest.raises(ValueError, match=r"refused: 1 component\(s\) hold more than 1/10 of the 24 eligible"):
        assign_inner(work, 0, max_component_share=0.1, **inner_kwargs(dataset))
    refusal = json.loads((work / identity.INNER_ASSIGNMENT_REFUSAL).read_text())
    assert refusal["status"] == "refused" and refusal["max_component_share"] == "1/10"
    assert refusal["components_over_max_share"] == [
        {"component_id": component_id(["B0001", "B0002", "B0003"]), "eligible_size": 3, "share": "1/8"}]
    assert "Freeze 1 decision" in refusal["decision"]
    assert not (work / "inner_membership_fold0.csv").exists()
    assert not (work / identity.INNER_ASSIGNMENT_RECEIPT).exists()
    argv = ["assign-inner", "--fold", "0", "--work", str(work), "--metallo-folds-dir", str(dataset[1]),
            "--allow-unpinned-binary"]
    for extra in (["--max-component-share", "0.1"], []):
        with pytest.raises(SystemExit) as stopped:
            identity.main(argv + extra)
        assert stopped.value.code == 2 and "was refused" in capsys.readouterr().err
    assert not (work / "inner_membership_fold0.csv").exists()
    assert not (work / identity.INNER_ASSIGNMENT_RECEIPT).exists()


def test_a_refusal_and_an_assignment_never_share_a_working_directory(tmp_path):
    work, dataset = run_inner_pipeline(tmp_path)
    assign_inner(work, 0, **inner_kwargs(dataset))
    with pytest.raises(ValueError, match="already holds an assignment; no refusal written"):
        assign_inner(work, 0, max_component_share=Fraction(1, 10), **inner_kwargs(dataset))
    assert not (work / identity.INNER_ASSIGNMENT_REFUSAL).exists()
    certify_inner(work, 0, **inner_kwargs(dataset))
    (work / identity.INNER_ASSIGNMENT_REFUSAL).write_text("{}\n")
    for step in (lambda: certify_inner(work, 0, **inner_kwargs(dataset)),
                 lambda: build_r1_inner_exclusions(work, 0, 0, allow_unpinned_binary=True)):
        with pytest.raises(ValueError, match="An earlier inner assignment of fold 0 was refused"):
            step()


@pytest.mark.parametrize("value", ["1/0", "0", "-1/4", "5/4", "abc", "nan"])
def test_maximum_share_must_be_a_fraction_in_range(value, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        identity.main(["assign-inner", "--fold", "0", "--work", str(tmp_path), "--metallo-folds-dir", str(tmp_path),
                       "--max-component-share", value])
    assert stopped.value.code == 2 and "--max-component-share" in capsys.readouterr().err
    with pytest.raises(ValueError, match="maximum component share"):
        identity.max_share_value(value)


def test_maximum_share_reads_floats_by_their_decimal_text():
    assert identity.max_share_value(0.3) == Fraction(3, 10)
    assert identity.max_share_value("0.25") == identity.max_share_value(Fraction(1, 4)) == Fraction(1, 4)
    assert identity.max_share_value(1) == 1


def test_assign_inner_refuses_unusable_annotations_and_several_structures(tmp_path):
    work, dataset = run_inner_pipeline(tmp_path, ec_overrides={"B0024": "0.1.1.1"})
    with pytest.raises(ValueError, match=r"cannot give an EC class: B0024__chain_A__EC_0\.1\.1\.1 \(out_of_range\)"):
        assign_inner(work, 0, **inner_kwargs(dataset))
    legacy = write_dataset(tmp_path / "legacy")
    legacy_work, _ = run_metallo_pipeline(tmp_path, "legacy_work", legacy)
    with pytest.raises(ValueError, match=r"several structure IDs, e\.g\. \['A00004'\]"):
        assign_inner(legacy_work, 0, **inner_kwargs(legacy))


def test_inner_steps_refuse_unpinned_searches_and_held_out_or_complement_inputs(tmp_path):
    work, dataset = run_inner_pipeline(tmp_path)
    with pytest.raises(ValueError, match="unpinned MMseqs2 binary"):
        assign_inner(work, 0, metallo_folds_dir=dataset[1])
    metallo_name = "CLEAN_30_train_test_split_0_train.csv"
    held_out = tmp_path / "test"
    held_out.mkdir()
    shutil.copy(dataset[1] / metallo_name, held_out / metallo_name)
    with pytest.raises(ValueError, match="held-out directory"):
        assign_inner(work, 0, metallo_folds_dir=held_out, allow_unpinned_binary=True)
    changed = tmp_path / "changed"
    changed.mkdir()
    (changed / metallo_name).write_text((dataset[1] / metallo_name).read_text() + "\n")
    with pytest.raises(ValueError, match="not the .* the fasta step read"):
        assign_inner(work, 0, metallo_folds_dir=changed, allow_unpinned_binary=True)
    with pytest.raises(ValueError, match="differs from None recorded in fasta_receipt_metallo_train.json"):
        assign_inner(work, 0, freeze1_sha256=FREEZE1, **inner_kwargs(dataset))
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        assign_inner(work, 0, freeze1_sha256="abc", **inner_kwargs(dataset))
    fasta_receipt = work / "fasta_receipt_metallo_train.json"
    original = fasta_receipt.read_text()
    fasta_receipt.write_text(json.dumps(json.loads(original) | {"query_set": "complement"}))
    with pytest.raises(ValueError, match="set complement"):
        assign_inner(work, 0, **inner_kwargs(dataset))
    fasta_receipt.write_text(original)
    for name in ("inner_membership_fold0.csv", identity.INNER_ASSIGNMENT_RECEIPT, identity.INNER_ASSIGNMENT_REFUSAL):
        assert not (work / name).exists(), name


def test_certify_inner_certifies_a_clean_membership(tmp_path):
    work, dataset = run_inner_pipeline(tmp_path)
    assign_inner(work, 0, **inner_kwargs(dataset))
    with pytest.raises(ValueError, match="Run certify-inner first"):
        build_r1_inner_exclusions(work, 0, 0, allow_unpinned_binary=True)
    result = certify_inner(work, 0, **inner_kwargs(dataset))
    receipt = json.loads((work / identity.INNER_CERTIFICATE).read_text())
    assert result["certified"] is True and receipt["certified"] is True and receipt["failures"] == []
    assert all(receipt["counts"][key] == 0 for key in identity.CERTIFICATION_FAILURE_KEYS)
    assert receipt["counts"]["membership_rows"] == receipt["counts"]["eligible_proteins"] == 24
    assert receipt["counts"]["excluded_ambiguous_ec_class"] == 1
    assert receipt["counts"]["cross_fold_links_by_type"] == {"alias": 0, "d3": 0, "donor": 0, "exact_sequence": 0}
    assert receipt["membership"]["sha256"] == sha256_file(work / "inner_membership_fold0.csv")
    assert certify_inner(work, 0, **inner_kwargs(dataset)) == result


def test_inner_receipts_must_name_the_membership_file_and_stay_unchanged(tmp_path):
    work, dataset = run_inner_pipeline(tmp_path)
    assign_inner(work, 0, **inner_kwargs(dataset))
    assignment_path, certificate_path = work / identity.INNER_ASSIGNMENT_RECEIPT, work / identity.INNER_CERTIFICATE
    original = assignment_path.read_text()
    forged = json.loads(original)
    forged["outputs"]["membership"]["file"] = "../elsewhere.csv"
    assignment_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match=r"names the membership file '\.\./elsewhere\.csv'"):
        certify_inner(work, 0, **inner_kwargs(dataset))
    assignment_path.write_text(original)
    certify_inner(work, 0, **inner_kwargs(dataset))
    certified = certificate_path.read_text()
    forged = json.loads(certified)
    forged["membership"]["file"] = "/elsewhere.csv"
    certificate_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match=r"inner_certification_receipt.json names the membership file"):
        build_r1_inner_exclusions(work, 0, 0, allow_unpinned_binary=True)
    certificate_path.write_text(certified)
    assignment_path.write_text(json.dumps(json.loads(original)))
    with pytest.raises(ValueError, match="inner_assignment_receipt.json changed after"):
        build_r1_inner_exclusions(work, 0, 0, allow_unpinned_binary=True)
    assignment_path.write_text(original)
    assert build_r1_inner_exclusions(work, 0, 0, allow_unpinned_binary=True)["inner_fold"] == 0


def test_certify_inner_refuses_an_injected_cross_edge(tmp_path):
    work, dataset = run_inner_pipeline(tmp_path)
    assign_inner(work, 0, **inner_kwargs(dataset))
    membership, receipt = read_inner_membership(work)
    rows = [dict(row) for row in membership.rows]
    moved = next(row for row in rows if row["accession"] == "B0002")
    moved.update(fold=str((int(moved["fold"]) + 1) % 4), group_id="cmp_forged")
    receipt["outputs"]["membership"]["sha256"] = write_ec_fold_membership(
        rows, work / receipt["outputs"]["membership"]["file"])
    (work / identity.INNER_ASSIGNMENT_RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="not certified: group_id_mismatches=1, split_components=1, "
                                         "cross_fold_links=2"):
        certify_inner(work, 0, **inner_kwargs(dataset))
    certificate = json.loads((work / identity.INNER_CERTIFICATE).read_text())
    assert certificate["certified"] is False
    assert certificate["counts"]["cross_fold_links_by_type"]["d3"] == 2
    with pytest.raises(ValueError, match="is not certified"):
        build_r1_inner_exclusions(work, 0, 0, allow_unpinned_binary=True)


def test_certification_counts_catch_injected_links_and_coverage_gaps():
    eligible = {acc: (f"{acc}__chain_A__EC_3.1.1.1", "3") for acc in ("P1", "P2", "P3", "P4")}
    components = {"P1": "g1", "P2": "g1", "P3": "g3", "P4": "g4"}
    rows = [{"structure_id": eligible[acc][0], "accession": acc, "group_id": components[acc], "fold": str(fold),
             "ec_class": "3"} for acc, fold in (("P1", 0), ("P2", 0), ("P3", 1), ("P4", 2))]
    links = [Link("d3", "P1", "P2")]
    clean = inner_certification_counts(rows, eligible, components, links, n_folds=3)
    assert all(clean[key] == 0 for key in identity.CERTIFICATION_FAILURE_KEYS)
    injected = inner_certification_counts(rows, eligible, components, links + [Link("donor", "P2", "P3")], n_folds=3)
    assert (injected["cross_fold_links"], injected["cross_fold_links_by_type"]["donor"]) == (1, 1)
    gaps = inner_certification_counts(rows[:-1] + [dict(rows[0])], eligible, components, links, n_folds=3)
    assert (gaps["eligible_missing"], gaps["repeated_accessions"], gaps["repeated_structure_ids"],
            gaps["empty_inner_folds"]) == (1, 1, 1, 1)
    relabelled = [dict(rows[0], ec_class="1")] + rows[1:]
    assert inner_certification_counts(relabelled, eligible, components, links, n_folds=3)["ec_class_mismatches"] == 1


def test_r1_inner_exclusions_from_the_mocked_search(tmp_path):
    work, dataset = run_inner_pipeline(tmp_path)
    assign_inner(work, 0, **inner_kwargs(dataset))
    certify_inner(work, 0, **inner_kwargs(dataset))
    membership, _ = read_inner_membership(work)
    fold_of = {row["accession"]: int(row["fold"]) for row in membership.rows}
    seen_validation = set()
    for inner in range(4):
        validation = {acc for acc, fold in fold_of.items() if fold == inner}
        neighbours = ({b for a, b in INNER_D3 if a in validation} | {a for a, b in INNER_D3 if b in validation}) \
            - validation
        build_r1_inner_exclusions(work, 0, inner, allow_unpinned_binary=True)
        name = f"r1_inner_exclusions_fold0_inner{inner}.tsv"
        rows = read_tsv(work / name, identity.R1_EXCLUSION_COLUMNS)
        assert [row["accession"] for row in rows] == sorted(validation | neighbours)
        assert {row["accession"] for row in rows if row["reason"] == "inner_validation"} == validation
        assert {row["accession"] for row in rows if row["reason"] == "d3_neighbour"} == neighbours
        assert "N0002" not in neighbours and "B0017" not in neighbours
        receipt = json.loads((work / f"r1_inner_exclusions_receipt_fold0_inner{inner}.json").read_text())
        assert receipt["outputs"][name]["sha256"] == sha256_file(work / name)
        counts = receipt["counts"]
        assert (counts["full_training_proteins"], counts["excluded"]) == (30, len(validation | neighbours))
        assert counts["r1_inner_training_proteins"] == 30 - counts["excluded"]
        assert counts["d3_neighbours_outside_metallo_training"] == len(neighbours & set(NON_METALLO))
        assert receipt["labels_used"] is False
        for text in (work / name).read_text(), json.dumps(receipt):
            assert "__EC_" not in text and "3.4.24.17" not in text
        seen_validation |= validation
    assert seen_validation == set(INNER_CLASSES)
    inner = fold_of["B0008"]
    excluded = {row["accession"] for row in read_tsv(work / f"r1_inner_exclusions_fold0_inner{inner}.tsv",
                                                     identity.R1_EXCLUSION_COLUMNS)}
    assert {BRIDGE, "N0005"} <= excluded
    with pytest.raises(ValueError, match=r"outside 0\.\.3"):
        build_r1_inner_exclusions(work, 0, 4, allow_unpinned_binary=True)
    with pytest.raises(ValueError, match="unpinned MMseqs2 binary"):
        build_r1_inner_exclusions(work, 0, 0)


def test_inner_cli_chain_on_a_pinned_search(tmp_path, capsys, pinned_fake):
    dataset = write_inner_dataset(tmp_path / "data")
    work, _ = run_metallo_pipeline(tmp_path, "work", dataset, hits=INNER_HITS, allow_unpinned=False)
    base = ["--fold", "0", "--work", str(work)]
    assert identity.main(["assign-inner", *base, "--metallo-folds-dir", str(dataset[1])]) == 0
    assert identity.main(["certify-inner", *base, "--metallo-folds-dir", str(dataset[1])]) == 0
    assert identity.main(["r1-inner-exclusions", *base, "--inner-fold", "0"]) == 0
    assert '"step": "r1_inner_exclusions"' in capsys.readouterr().out
    certificate = json.loads((work / identity.INNER_CERTIFICATE).read_text())
    assert certificate["pinned_binary"] is True and certificate["allow_unpinned_binary"] is False
    assert certificate["freeze1_sha256"] is None and certificate["certified"] is True


# ---------------------------------------------------------------- command line and imports

@pytest.mark.parametrize("name", ["assign-inner", "certify-inner", "r1-inner-exclusions"])
def test_inner_subcommands_refuse_without_their_inputs(name, tmp_path, capsys):
    extra = ["--inner-fold", "0"] if name == "r1-inner-exclusions" else ["--metallo-folds-dir", str(tmp_path)]
    with pytest.raises(SystemExit) as stopped:
        identity.main([name, "--fold", "0", "--work", str(tmp_path), *extra])
    assert stopped.value.code == 2
    assert "Run the fasta step for metallo_train first" in capsys.readouterr().err


def test_cohort_scope_cli_refuses_without_freeze1_or_with_unpinned_searches(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        identity.main(["components", "--scope", "cohort", "--out", str(tmp_path), "--fold-work", str(tmp_path)])
    assert stopped.value.code == 2 and "locked until Freeze 1" in capsys.readouterr().err
    with pytest.raises(SystemExit) as stopped:
        identity.main(["components", "--scope", "cohort", "--out", str(tmp_path), "--fold-work", str(tmp_path),
                       "--freeze1-sha256", FREEZE1, "--allow-unpinned-binary"])
    assert stopped.value.code == 2 and "always need the pinned MMseqs2 binary" in capsys.readouterr().err


def test_script_help_and_torch_free_import():
    script = ROOT / "scripts" / "run_ec_clean30_identity.py"
    completed = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert completed.returncode == 0 and "strata" in completed.stdout
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
            "import ec_clean30.identity; print('torch' in sys.modules)")
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"


@pytest.mark.skipif(not identity.DEFAULT_MMSEQS_BINARY.is_file(), reason="pinned MMseqs2 binary not available")
def test_pinned_mmseqs2_on_synthetic_sequences(tmp_path):
    if sha256_file(identity.DEFAULT_MMSEQS_BINARY) != MMSEQS_BINARY_SHA256:
        pytest.skip("binary present but not the pinned build")
    dataset = write_dataset(tmp_path / "data")
    work = tmp_path / "work"
    build_inputs(work, 0, official_split_dir=dataset[0], metallo_folds_dir=dataset[1])
    # The pinned settings need a 2G prefilter estimate even here; measured peak RSS is about 0.9 GB.
    run_search(work, 0, threads=1, split_memory_limit="2G", timeout_seconds=300)
    build_edges(work, 0)
    receipt = json.loads((work / "search_metallo_train" / "search_receipt.json").read_text())
    assert receipt["pinned_binary"] is True and receipt["version"] == MMSEQS_VERSION
    edges = {(row["query"], row["target"]): row for row in read_tsv(work / "edges_metallo_train.tsv", EDGE_COLUMNS)}
    assert edges[("A00005", "A00006")]["identity"] == "1.000000" and edges[("A00005", "A00006")]["d3"] == "1"
    assert edges[("A00001", "A00002")]["d3"] == "1"
    assert json.loads((work / "edges_receipt_metallo_train.json").read_text())["counts"]["self_hits_dropped"] == 6


def to_accession_layout(root, metallo_dir):
    """The prepared CLEAN30 layout: 'structure' holds the accession; stems live in per-fold training manifests."""
    manifest_root = root / "manifests"
    for fold in range(5):
        path = metallo_dir / f"CLEAN_30_train_test_split_{fold}_train.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        stems = sorted({row["structure"] for row in rows})
        for row in rows:
            row["structure"] = row["uniprot_id"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        manifest = manifest_root / identity.METALLO_MANIFEST_PATH.format(fold=fold)
        manifest.parent.mkdir(parents=True)
        manifest.write_text("structure_name,relative_path\n" + "".join(f"{stem}.pdb,x/{stem}.pdb\n" for stem in stems))
    return manifest_root


def test_accession_layout_takes_stems_from_the_training_manifest(tmp_path):
    stem_layout = write_dataset(tmp_path / "stems")
    build_inputs(tmp_path / "w_stems", 0, official_split_dir=stem_layout[0], metallo_folds_dir=stem_layout[1])
    official_dir, metallo_dir = write_dataset(tmp_path / "acc")
    manifest_root = to_accession_layout(tmp_path / "acc", metallo_dir)
    with pytest.raises(ValueError, match="not a structure stem"):
        build_inputs(tmp_path / "w_none", 0, official_split_dir=official_dir, metallo_folds_dir=metallo_dir)
    summary = build_inputs(tmp_path / "w_acc", 0, official_split_dir=official_dir, metallo_folds_dir=metallo_dir,
                           metallo_manifest_root=manifest_root)
    nodes = summary["outputs"]["nodes"]["file"]
    assert (tmp_path / "w_acc" / nodes).read_text() == (tmp_path / "w_stems" / nodes).read_text()
    assert summary["counts"]["queries_with_several_structure_ids"] == 1  # A00004 has chains A and B
    receipt = json.loads((tmp_path / "w_acc" / "fasta_receipt_metallo_train.json").read_text())
    recorded = receipt["sources"]["metallo_structure_manifests"]["0"]
    assert recorded["sha256"] == sha256_file(manifest_root / identity.METALLO_MANIFEST_PATH.format(fold=0))

    chain = SimpleNamespace(inputs=receipt)
    _, entries = identity._metallo_training_entries(chain, 0, metallo_dir, manifest_root)
    assert entries["A00004"].structure_ids == {f"A00004__chain_A__EC_{EC_TEXT}", f"A00004__chain_B__EC_{EC_TEXT}"}
    manifest = manifest_root / identity.METALLO_MANIFEST_PATH.format(fold=0)
    manifest.write_text("\n".join(line for line in manifest.read_text().splitlines() if "A00001" not in line) + "\n")
    with pytest.raises(ValueError, match="differs from the one the fasta step read"):
        identity._metallo_training_entries(chain, 0, metallo_dir, manifest_root)
    with pytest.raises(ValueError, match="has no structure in structure_manifest.csv"):
        identity.read_metallo_training(metallo_dir / "CLEAN_30_train_test_split_0_train.csv", manifest)


def test_structure_manifest_path_refuses_held_out_folders(tmp_path):
    with pytest.raises(ValueError, match="held-out directory"):
        identity.metallo_structure_manifest(tmp_path / "test", 0)
    assert identity.metallo_structure_manifest(tmp_path, 3).parts[-3:] == (
        "CLEAN_30_train_test_split_3", "train", "structure_manifest.csv")
