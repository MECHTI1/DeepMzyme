"""Tests for the v3 near-copy-disjoint fold builder (plan step A1)."""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmarking import v3_folds as vf  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def random_sequence(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(AA) for _ in range(length))


def mutate(rng: random.Random, sequence: str, substitutions: int, indels: int) -> str:
    seq = list(sequence)
    for _ in range(substitutions):
        i = rng.randrange(len(seq))
        seq[i] = rng.choice([a for a in AA if a != seq[i]])
    for _ in range(indels):
        i = rng.randrange(1, len(seq) - 1)
        if rng.random() < 0.5:
            del seq[i]
        else:
            seq.insert(i, rng.choice(AA))
    return "".join(seq)


def test_identity_definition():
    rng = random.Random(1)
    base = random_sequence(rng, 60)
    assert vf.near_copy_identity(base, base)[0] == 1.0
    one_sub = base[:30] + ("A" if base[30] != "A" else "C") + base[31:]
    identity, identical, denominator = vf.near_copy_identity(base, one_sub)
    assert (identical, denominator) == (59, 60)
    # A fragment of a longer chain is a near-copy (free end gaps).
    longer = random_sequence(rng, 40) + base + random_sequence(rng, 40)
    assert vf.near_copy_identity(base, longer)[0] == 1.0
    # Each internal insertion event in the longer chain counts once in the denominator.
    inserted = base[:30] + "WWWWW" + base[30:]
    identity, identical, denominator = vf.near_copy_identity(base, inserted)
    assert identical == 60 and denominator == 61
    # A long loop insertion in an otherwise identical chain is still a near-copy.
    chain = random_sequence(rng, 227)
    looped = chain[:110] + random_sequence(rng, 29) + chain[110:]
    assert vf.near_copy_identity(chain, looped)[0] >= vf.IDENTITY_THRESHOLD
    # The result does not depend on argument order, also for equal lengths.
    twin = mutate(rng, base, substitutions=4, indels=0)
    assert vf.near_copy_identity(base, twin) == vf.near_copy_identity(twin, base)
    # X is never an identical residue.
    x_seq = "X" * 40
    assert vf.near_copy_identity(x_seq, x_seq)[1] == 0


def test_prefilter_never_misses_a_qualifying_pair():
    rng = random.Random(7)
    qualifying = 0
    for case in range(400):
        length = rng.randrange(30, 220)
        base = random_sequence(rng, length)
        other = mutate(rng, base, substitutions=rng.randrange(0, max(1, length // 9)),
                       indels=rng.randrange(0, 3))
        if rng.random() < 0.3:  # fragment versus longer chain
            other = random_sequence(rng, rng.randrange(0, 50)) + other + random_sequence(rng, rng.randrange(0, 50))
        if len(other) < 30:
            continue
        identity, _, _ = vf.near_copy_identity(base, other)
        if identity >= vf.IDENTITY_THRESHOLD:
            qualifying += 1
            short, long_ = (base, other) if len(base) <= len(other) else (other, base)
            assert vf.shared_kmer_positions(short, long_) >= vf.prefilter_threshold(len(short)), case
    assert qualifying > 100


def test_pair_search_matches_brute_force():
    rng = random.Random(11)
    sequences = []
    for _family in range(6):
        base = random_sequence(rng, rng.randrange(40, 120))
        sequences.append(base)
        for _ in range(3):
            sequences.append(mutate(rng, base, substitutions=rng.randrange(0, 12), indels=rng.randrange(0, 2)))
    sequences += [random_sequence(rng, rng.randrange(30, 120)) for _ in range(8)]
    sequences.append(sequences[0][:25])  # too short to ever link
    sequences = sorted(set(sequences))
    found = {(r["a"], r["b"]) for r in vf.find_near_copy_pairs(sequences)}
    brute = set()
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if min(len(sequences[i]), len(sequences[j])) < vf.MIN_CHAIN_LENGTH:
                continue
            if vf.near_copy_identity(sequences[i], sequences[j])[0] >= vf.IDENTITY_THRESHOLD:
                brute.add((i, j))
    assert found == brute and brute


def test_grouping_is_transitive_and_ignores_short_chains():
    rng = random.Random(3)
    a = random_sequence(rng, 100)
    b = a[:50] + mutate(rng, a[50:], substitutions=5, indels=0)
    c = mutate(rng, b[:50], substitutions=5, indels=0) + b[50:]
    short = random_sequence(rng, 20)
    pdbs = {"1aaa": {a}, "1bbb": {b}, "1ccc": {c}, "1ddd": {short}, "1eee": {short},
            "1fff": {random_sequence(rng, 80), a}}
    pairs = [(a, b), (b, c)]
    groups = vf.group_pdbs(pdbs, pairs)
    assert groups["1aaa"] == groups["1bbb"] == groups["1ccc"] == groups["1fff"] == "v3g_1aaa"
    assert groups["1ddd"] != groups["1eee"]  # chains under 30 residues never link entries


def synthetic_groups(n_per_element: int = 100, seed: int = 5):
    rng = random.Random(seed)
    groups = []
    for element in vf.ELEMENTS:
        for k in range(n_per_element):
            size = rng.choice([1, 1, 1, 2, 3, 4, 8])
            groups.append(vf.GroupStats(f"{element}{k:03d}", size, {element: size}))
    return groups


def test_assignment_is_balanced_and_deterministic():
    groups = synthetic_groups()
    first = vf.choose_assignment(groups, seed=42, starts=20)
    second = vf.choose_assignment(groups, seed=42, starts=20)
    assert first["accepted"] and first["assignment"] == second["assignment"]
    assert set(first["assignment"].values()) == set(range(vf.N_FOLDS))
    assert first["balance"]["failures"] == []


def test_acceptance_reports_failures_for_an_infeasible_cohort():
    groups = [vf.GroupStats("big", 500, {"CU": 500})] + synthetic_groups(n_per_element=10)
    result = vf.choose_assignment(groups, seed=1, starts=5)
    assert not result["accepted"] and result["balance"]["failures"]


def test_cross_fold_pairs_are_detected():
    sequence_pdbs = {"A" * 40: {"1aaa"}, "C" * 40: {"1bbb"}}
    pairs = [("A" * 40, "C" * 40)]
    assert vf.cross_fold_pair_violations(pairs, sequence_pdbs, {"1aaa": 0, "1bbb": 0}) == []
    assert vf.cross_fold_pair_violations(pairs, sequence_pdbs, {"1aaa": 0, "1bbb": 1})


def synthetic_cohort(seed: int = 9):
    rng = random.Random(seed)
    bindings, pdb_sequences = [], {}
    uid = 0
    families = [random_sequence(rng, 70) for _ in range(30)]
    for element in vf.ELEMENTS:
        for k in range(100):
            pdb = f"{element.lower()}{k:03d}"
            seq = random_sequence(rng, 60)
            if k < 15:  # some entries are near-copies of shared families
                seq = mutate(rng, families[(k + len(element)) % len(families)], substitutions=3, indels=0)
            pdb_sequences[pdb] = {seq}
            for _ in range(rng.choice([1, 1, 2, 3])):
                uid += 1
                bindings.append(SimpleNamespace(source_uid=f"uid{uid}", physical_ion_id=f"{pdb}|{uid}",
                                                pdbid=pdb, native_element=element))
    return bindings, pdb_sequences


def test_end_to_end_is_reproducible_and_never_overwrites(tmp_path):
    bindings, pdb_sequences = synthetic_cohort()
    digests = []
    for name in ("run1", "run2"):
        result = vf.build_folds(bindings, pdb_sequences, seed=42, starts=10)
        assert result["violations"] == []
        folds_of_group = {}
        for row in result["fold_rows"]:
            folds_of_group.setdefault(row["group_id"], set()).add(row["fold"])
        assert all(len(f) == 1 for f in folds_of_group.values())
        receipt = vf.write_outputs(tmp_path / name, result, inputs={"synthetic": True}, seed=42, starts=10)
        assert receipt["accepted"]
        digests.append(hashlib.sha256((tmp_path / name / "fold_membership.csv").read_bytes()).hexdigest())
    assert digests[0] == digests[1]
    with pytest.raises(FileExistsError):
        vf.write_outputs(tmp_path / "run1", result, inputs={}, seed=42, starts=10)


def test_cross_fold_violation_blocks_the_fold_file(tmp_path):
    bindings, pdb_sequences = synthetic_cohort()
    result = vf.build_folds(bindings, pdb_sequences, seed=42, starts=10)
    result["violations"] = ["synthetic violation"]
    receipt = vf.write_outputs(tmp_path / "bad", result, inputs={}, seed=42, starts=10)
    assert not receipt["accepted"]
    assert not (tmp_path / "bad" / "fold_membership.csv").exists()
    assert (tmp_path / "bad" / "fold_balance_report.json").exists()


def test_receipt_records_code_identity(tmp_path):
    bindings, pdb_sequences = synthetic_cohort()
    result = vf.build_folds(bindings, pdb_sequences, seed=42, starts=5)
    receipt = vf.write_outputs(tmp_path / "out", result, inputs={}, seed=42, starts=5)
    files = receipt["code"]["files_sha256"]
    assert "src/benchmarking/v3_folds.py" in files and "scripts/build_v3_folds.py" in files
    assert receipt["builder_version"] == vf.BUILDER_VERSION
