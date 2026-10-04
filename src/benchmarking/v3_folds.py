"""Near-copy-disjoint grouped folds for the PMM ion metal v3 campaign (plan step A1).

The unit is the PDB entry. PDB entries whose context chains are near-copies are
merged transitively into one group, and groups are assigned to five folds in a
seeded random order by a greedy balance rule (no size sorting; see TECH-020).

Near-copy definition. Two chains of at least ``MIN_CHAIN_LENGTH`` residues are
aligned globally with free end gaps (BLOSUM62; Biopython convention, so an internal
gap of length n scores -11 - (n - 1)); the first optimal alignment returned is used.
The shorter chain ``s`` (length L; for equal lengths the lexicographically smaller
sequence) is the alignment target. With ``E`` the number of insertion events, i.e.
places between consecutive aligned blocks of ``s`` where the longer chain has extra
residues, identity = identical aligned standard residues / (L + E). A pair with
identity of at least ``IDENTITY_THRESHOLD`` is a near-copy. Each insertion event
counts once whatever its length, so a long loop insertion in an otherwise identical
chain still qualifies, while the k-mer prefilter below stays provably lossless.

Prefilter proof (k = 5, threshold 0.9). Let D = identical, N = L + E >= L >= 30.
Identical columns form runs contiguous in both chains; a run ends only at a
non-identical residue of ``s`` (at most L - D of them) or at an insertion event
(E of them), so there are at most (L - D) + E + 1 = (N - D) + 1 <= 0.1 N + 1 runs.
A run of length r holds r - k + 1 start positions of ``s`` whose k-mer also occurs
in the longer chain, and runs contain standard residues only. Summed over runs this
is at least 0.9 N - (k - 1)(0.1 N + 1) = 0.5 N - 4 >= 0.5 L - 4. Pairs below that
count of shared k-mer start positions cannot qualify and are skipped unaligned.
Insertions before the first or after the last aligned block never split a run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

BUILDER_VERSION = "v3-near-copy-folds-2"
N_FOLDS = 5
MIN_CHAIN_LENGTH = 30
IDENTITY_THRESHOLD = 0.90
KMER = 5
GAP_OPEN = -11.0
GAP_EXTEND = -1.0
DEFAULT_SEED = 42
DEFAULT_STARTS = 200
ELEMENTS = ("MN", "FE", "CO", "NI", "CU", "ZN")
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
ACCEPTANCE = {
    "min_groups_per_element_per_fold": 15,
    "fold_ion_tolerance": 0.05,
    "element_ion_tolerance": 0.15,
    "cu_group_tolerance": 0.20,
}
FOLD_COLUMNS = ("source_uid", "physical_ion_id", "group_id", "native_element", "fold", "pdbid")
PAIR_COLUMNS = ("sequence_a_sha256", "sequence_b_sha256", "length_a", "length_b",
                "shared_kmer_positions", "identical", "denominator", "identity")


# ---------------------------------------------------------------------------
# Near-copy identity and the provable k-mer prefilter
# ---------------------------------------------------------------------------

def _aligner():
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = GAP_OPEN
    aligner.extend_gap_score = GAP_EXTEND
    aligner.end_gap_score = 0.0  # all end gaps free; internal gaps keep -11/-1
    return aligner


_ALIGNER = None


def _matrix_safe(sequence: str) -> str:
    """Letters outside BLOSUM62 (for example U or O) align as X; X is never identical."""
    allowed = set("ARNDCQEGHILKMFPSTWYVBZX")
    return "".join(ch if ch in allowed else "X" for ch in sequence)


def near_copy_identity(first: str, second: str) -> tuple[float, int, int]:
    """Return (identity, identical, denominator) under the module's near-copy definition."""
    global _ALIGNER
    if _ALIGNER is None:
        _ALIGNER = _aligner()
    short, long_ = (first, second) if (len(first), first) <= (len(second), second) else (second, first)
    alignment = _ALIGNER.align(_matrix_safe(short), _matrix_safe(long_))[0]
    blocks_short, blocks_long = alignment.aligned
    identical = 0
    for (s0, s1), (l0, _l1) in zip(blocks_short, blocks_long):
        for offset in range(s1 - s0):
            residue = short[s0 + offset]
            if residue in STANDARD_RESIDUES and residue == long_[l0 + offset]:
                identical += 1
    insertion_events = sum(1 for index in range(1, len(blocks_long))
                           if blocks_long[index][0] > blocks_long[index - 1][1])
    denominator = len(short) + insertion_events
    return identical / denominator, identical, denominator


def prefilter_threshold(shorter_length: int) -> int:
    """Minimum shared k-mer start positions any qualifying pair must have (see module proof)."""
    return max(1, math.ceil(0.5 * shorter_length - (KMER - 1)))


def _kmers(sequence: str) -> list[str | None]:
    out: list[str | None] = []
    for index in range(len(sequence) - KMER + 1):
        word = sequence[index:index + KMER]
        out.append(word if set(word) <= STANDARD_RESIDUES else None)
    return out


def shared_kmer_positions(shorter: str, longer: str) -> int:
    words = {word for word in _kmers(longer) if word is not None}
    return sum(1 for word in _kmers(shorter) if word is not None and word in words)


def find_near_copy_pairs(sequences: Sequence[str], *, progress: bool = False) -> list[dict]:
    """All near-copy pairs among unique sequences of at least MIN_CHAIN_LENGTH residues."""
    eligible = [i for i, seq in enumerate(sequences) if len(seq) >= MIN_CHAIN_LENGTH]
    postings: dict[str, list[int]] = defaultdict(list)
    word_lists: dict[int, list[str | None]] = {}
    for i in eligible:
        words = _kmers(sequences[i])
        word_lists[i] = words
        for word in set(w for w in words if w is not None):
            postings[word].append(i)
    posting_arrays = {word: np.asarray(ids, dtype=np.int64) for word, ids in postings.items()}
    lengths = np.asarray([len(seq) for seq in sequences], dtype=np.int64)
    pairs: list[dict] = []
    n = len(sequences)
    for count_index, i in enumerate(eligible, start=1):
        arrays = [posting_arrays[w] for w in word_lists[i] if w is not None]
        if not arrays:
            continue
        counts = np.bincount(np.concatenate(arrays), minlength=n)
        counts[i] = 0
        # i is the shorter (or equal-length, lower-index) member of each evaluated pair.
        partner_ok = (lengths > len(sequences[i])) | ((lengths == len(sequences[i])) & (np.arange(n) > i))
        candidates = np.nonzero(partner_ok & (counts >= prefilter_threshold(len(sequences[i]))))[0]
        for j in candidates.tolist():
            identity, identical, denominator = near_copy_identity(sequences[i], sequences[j])
            if identity >= IDENTITY_THRESHOLD:
                a, b = sorted((i, j))
                pairs.append({"a": a, "b": b, "shared_kmer_positions": int(counts[j]),
                              "identical": identical, "denominator": denominator, "identity": identity})
        if progress and count_index % 500 == 0:
            print(f"[V3-FOLDS] near-copy search {count_index}/{len(eligible)} sequences, "
                  f"{len(pairs)} pairs", flush=True)
    pairs.sort(key=lambda row: (row["a"], row["b"]))
    return pairs


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def group_pdbs(pdb_sequences: dict[str, set[str]], pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Map each PDB to its group ID: PDBs sharing a sequence or a near-copy pair merge transitively.

    Sequences shorter than MIN_CHAIN_LENGTH never link entries. The group ID is
    ``v3g_`` plus the lexicographically smallest member PDB ID.
    """
    finder = _UnionFind(pdb_sequences)
    by_sequence: dict[str, list[str]] = defaultdict(list)
    for pdb, seqs in pdb_sequences.items():
        for seq in seqs:
            if len(seq) >= MIN_CHAIN_LENGTH:
                by_sequence[seq].append(pdb)
    for members in by_sequence.values():
        for other in members[1:]:
            finder.union(members[0], other)
    for first, second in pairs:
        if len(first) < MIN_CHAIN_LENGTH or len(second) < MIN_CHAIN_LENGTH:
            continue
        for pdb_a in by_sequence.get(first, []):
            for pdb_b in by_sequence.get(second, []):
                finder.union(pdb_a, pdb_b)
    roots = {pdb: finder.find(pdb) for pdb in pdb_sequences}
    smallest: dict[str, str] = {}
    for pdb, root in roots.items():
        smallest[root] = min(smallest.get(root, pdb), pdb)
    return {pdb: f"v3g_{smallest[root]}" for pdb, root in roots.items()}


# ---------------------------------------------------------------------------
# Fold assignment and acceptance
# ---------------------------------------------------------------------------

@dataclass
class GroupStats:
    group_id: str
    n_ions: int
    elements: dict[str, int] = field(default_factory=dict)


def _targets(groups: Sequence[GroupStats]) -> dict:
    total = sum(g.n_ions for g in groups)
    element_ions = {e: sum(g.elements.get(e, 0) for g in groups) for e in ELEMENTS}
    element_groups = {e: sum(1 for g in groups if g.elements.get(e, 0) > 0) for e in ELEMENTS}
    return {"ions": total / N_FOLDS,
            "element_ions": {e: element_ions[e] / N_FOLDS for e in ELEMENTS},
            "element_groups": {e: element_groups[e] / N_FOLDS for e in ELEMENTS}}


def _fold_score(state: dict, targets: dict) -> float:
    def rel(value: float, target: float) -> float:
        return ((value - target) / target) ** 2 if target > 0 else 0.0

    score = rel(state["ions"], targets["ions"])
    for e in ELEMENTS:
        score += rel(state["element_ions"][e], targets["element_ions"][e])
        score += rel(state["element_groups"][e], targets["element_groups"][e])
    return score


def _empty_state() -> dict:
    return {"ions": 0, "element_ions": {e: 0 for e in ELEMENTS}, "element_groups": {e: 0 for e in ELEMENTS}}


def _add(state: dict, group: GroupStats) -> dict:
    new = {"ions": state["ions"] + group.n_ions,
           "element_ions": dict(state["element_ions"]), "element_groups": dict(state["element_groups"])}
    for e, count in group.elements.items():
        new["element_ions"][e] += count
        new["element_groups"][e] += 1 if count > 0 else 0
    return new


def assign_once(groups: Sequence[GroupStats], rng: random.Random) -> tuple[dict[str, int], float]:
    """One seeded start: shuffle groups, then place each where it least increases the score."""
    targets = _targets(groups)
    order = sorted(groups, key=lambda g: g.group_id)
    rng.shuffle(order)
    states = [_empty_state() for _ in range(N_FOLDS)]
    assignment: dict[str, int] = {}
    for group in order:
        best = None
        for fold in range(N_FOLDS):
            trial = _add(states[fold], group)
            delta = _fold_score(trial, targets) - _fold_score(states[fold], targets)
            key = (delta, states[fold]["ions"], fold)
            if best is None or key < best[0]:
                best = (key, fold, trial)
        _key, fold, trial = best
        states[fold] = trial
        assignment[group.group_id] = fold
    return assignment, sum(_fold_score(state, targets) for state in states)


def fold_balance(groups: Sequence[GroupStats], assignment: dict[str, int]) -> dict:
    """Per-fold counts plus the acceptance verdict of plan step A1."""
    targets = _targets(groups)
    folds = []
    for fold in range(N_FOLDS):
        members = [g for g in groups if assignment[g.group_id] == fold]
        ions = sum(g.n_ions for g in members)
        element_ions = {e: sum(g.elements.get(e, 0) for g in members) for e in ELEMENTS}
        element_groups = {e: sum(1 for g in members if g.elements.get(e, 0) > 0) for e in ELEMENTS}
        largest_share = {e: (max((g.elements.get(e, 0) for g in members), default=0) / element_ions[e])
                         if element_ions[e] else 0.0 for e in ELEMENTS}
        sizes = sorted((g.n_ions for g in members), reverse=True)
        histogram = {"1": 0, "2-4": 0, "5-9": 0, "10-49": 0, "50+": 0}
        for size in sizes:
            key = "1" if size == 1 else "2-4" if size < 5 else "5-9" if size < 10 else "10-49" if size < 50 else "50+"
            histogram[key] += 1
        folds.append({"fold": fold, "ions": ions, "groups": len(members),
                      "element_ions": element_ions, "element_groups": element_groups,
                      "largest_group_share": {e: round(v, 4) for e, v in largest_share.items()},
                      "largest_group_sizes": sizes[:5], "group_size_histogram": histogram})
    failures = []
    a = ACCEPTANCE
    for row in folds:
        f = row["fold"]
        for e in ELEMENTS:
            if row["element_groups"][e] < a["min_groups_per_element_per_fold"]:
                failures.append(f"fold {f}: {e} has {row['element_groups'][e]} groups "
                                f"(< {a['min_groups_per_element_per_fold']})")
            target = targets["element_ions"][e]
            if target and abs(row["element_ions"][e] - target) > a["element_ion_tolerance"] * target:
                failures.append(f"fold {f}: {e} ions {row['element_ions'][e]} outside "
                                f"±{a['element_ion_tolerance']:.0%} of {target:.1f}")
        if abs(row["ions"] - targets["ions"]) > a["fold_ion_tolerance"] * targets["ions"]:
            failures.append(f"fold {f}: {row['ions']} ions outside ±{a['fold_ion_tolerance']:.0%} "
                            f"of {targets['ions']:.1f}")
    cu_groups = [row["element_groups"]["CU"] for row in folds]
    cu_mean = sum(cu_groups) / N_FOLDS
    for row in folds:
        if cu_mean and abs(row["element_groups"]["CU"] - cu_mean) > a["cu_group_tolerance"] * cu_mean:
            failures.append(f"fold {row['fold']}: Cu groups {row['element_groups']['CU']} outside "
                            f"±{a['cu_group_tolerance']:.0%} of {cu_mean:.1f}")
    return {"targets": targets, "folds": folds, "failures": failures, "accepted": not failures}


def choose_assignment(groups: Sequence[GroupStats], *, seed: int = DEFAULT_SEED,
                      starts: int = DEFAULT_STARTS) -> dict:
    """Run seeded starts; keep the lowest-score start that passes acceptance."""
    best = None
    passing = 0
    for start in range(starts):
        assignment, score = assign_once(groups, random.Random(f"{seed}:{start}"))
        balance = fold_balance(groups, assignment)
        if balance["accepted"]:
            passing += 1
            if best is None or score < best["score"]:
                best = {"start": start, "score": score, "assignment": assignment, "balance": balance}
    if best is None:
        # Report the best-scoring failure so the user can see why acceptance failed.
        fallback = min((assign_once(groups, random.Random(f"{seed}:{s}")) + (s,) for s in range(starts)),
                       key=lambda item: item[1])
        return {"accepted": False, "passing_starts": 0, "start": fallback[2], "score": fallback[1],
                "assignment": fallback[0], "balance": fold_balance(groups, fallback[0])}
    best.update({"accepted": True, "passing_starts": passing})
    return best


def cross_fold_pair_violations(pair_rows: Iterable[tuple[str, str]], sequence_pdbs: dict[str, set[str]],
                               pdb_fold: dict[str, int]) -> list[str]:
    """Recheck from the saved pair list that no near-copy pair or shared sequence crosses folds."""
    violations = []
    for first, second in pair_rows:
        folds = {pdb_fold[p] for p in sequence_pdbs.get(first, set()) | sequence_pdbs.get(second, set())}
        if len(folds) > 1:
            violations.append(f"{first[:12]}~{second[:12]} spans folds {sorted(folds)}")
    for seq in sorted(sequence_pdbs):
        if len(seq) >= MIN_CHAIN_LENGTH and len({pdb_fold[p] for p in sequence_pdbs[seq]}) > 1:
            violations.append(f"shared sequence {seq[:12]} spans folds")
    return violations


# ---------------------------------------------------------------------------
# Cohort inputs and outputs
# ---------------------------------------------------------------------------

def sequence_sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _csv_bytes(columns: Sequence[str], rows: Iterable[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write(path: Path, payload: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _guard_worker(roots: list[str]) -> None:
    from training.access_guard import install_forbidden_read_guard

    install_forbidden_read_guard(roots)


def _parse_sequences(path_text: str) -> dict[str, str]:
    from embed_helpers.esmc import extract_chain_sequences, parse_structure

    return extract_chain_sequences(parse_structure(Path(path_text)))


def load_inputs(v2_root: Path, train_dir: Path, *, workers: int = 2) -> dict:
    """Read the frozen v2 cohort and ESMC plan; rebuild and hash-check every context-chain sequence."""
    from training.access_guard import install_forbidden_read_guard
    from training.data import _cohort_structure_files
    from training.source_cohort import read_cohort_csv, sha256_file
    from benchmarking.pmm_ion_campaign import forbidden_read_roots

    guard_roots = [str(path) for path in forbidden_read_roots(train_dir)]
    install_forbidden_read_guard(guard_roots)
    manifest = json.loads((v2_root / "campaign_manifest.json").read_text(encoding="utf-8"))
    cohort_path = v2_root / "train_cohort.csv"
    bindings = read_cohort_csv(cohort_path, expected_sha256=manifest["cohort"]["sha256"])
    plan_path = v2_root / "esm_generation_plan.csv"
    with plan_path.open(encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    files = {path.name: path for path in _cohort_structure_files(train_dir, bindings)}
    names = sorted({row["structure_name"] for row in plan})
    missing = [name for name in names if name not in files]
    if missing:
        raise ValueError(f"{len(missing)} planned structures are not cohort files, e.g. {missing[:3]}")
    if workers > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(workers, initializer=_guard_worker, initargs=(guard_roots,)) as pool:
            parsed = dict(zip(names, pool.map(_parse_sequences, [str(files[n]) for n in names], chunksize=16)))
    else:
        parsed = {name: _parse_sequences(str(files[name])) for name in names}
    structure_pdb = {b.structure_name: b.pdbid for b in bindings}
    pdb_sequences: dict[str, set[str]] = defaultdict(set)
    for row in plan:
        sequence = parsed[row["structure_name"]].get(row["chain"])
        if sequence is None or sequence_sha(sequence) != row["sequence_sha256"] \
                or len(sequence) != int(row["sequence_length"]):
            raise ValueError(f"Sequence mismatch for {row['structure_name']} chain {row['chain']}")
        pdb_sequences[structure_pdb[row["structure_name"]]].add(sequence)
    for binding in bindings:
        pdb_sequences.setdefault(binding.pdbid, set())
    return {"bindings": bindings, "pdb_sequences": dict(pdb_sequences),
            "inputs": {"v2_root": str(v2_root), "train_dir": str(train_dir),
                       "cohort_sha256": manifest["cohort"]["sha256"],
                       "esm_plan_sha256": sha256_file(plan_path), "n_plan_rows": len(plan),
                       "structure_content_sha256": manifest.get("structure_content_sha256")}}


def build_folds(bindings, pdb_sequences: dict[str, set[str]], *, seed: int = DEFAULT_SEED,
                starts: int = DEFAULT_STARTS, progress: bool = False) -> dict:
    """Pure core: near-copy pairs, groups, fold assignment and checks (no file I/O)."""
    unique = sorted({seq for seqs in pdb_sequences.values() for seq in seqs})
    raw_pairs = find_near_copy_pairs(unique, progress=progress)
    pair_sequences = [(unique[row["a"]], unique[row["b"]]) for row in raw_pairs]
    pdb_group = group_pdbs(pdb_sequences, pair_sequences)
    stats: dict[str, GroupStats] = {}
    for binding in bindings:
        gid = pdb_group[binding.pdbid]
        group = stats.setdefault(gid, GroupStats(gid, 0, {}))
        group.n_ions += 1
        group.elements[binding.native_element] = group.elements.get(binding.native_element, 0) + 1
    groups = sorted(stats.values(), key=lambda g: g.group_id)
    chosen = choose_assignment(groups, seed=seed, starts=starts)
    unassigned = sorted(pdb for pdb, gid in pdb_group.items() if gid not in chosen["assignment"])
    if unassigned:
        raise ValueError(f"PDB entries without cohort ions cannot be assigned: {unassigned[:5]}")
    pdb_fold = {pdb: chosen["assignment"][gid] for pdb, gid in pdb_group.items()}
    sequence_pdbs: dict[str, set[str]] = defaultdict(set)
    for pdb, seqs in pdb_sequences.items():
        for seq in seqs:
            sequence_pdbs[seq].add(pdb)
    violations = cross_fold_pair_violations(pair_sequences, sequence_pdbs, pdb_fold)
    fold_rows = [{"source_uid": b.source_uid, "physical_ion_id": b.physical_ion_id,
                  "group_id": pdb_group[b.pdbid], "native_element": b.native_element,
                  "fold": pdb_fold[b.pdbid], "pdbid": b.pdbid} for b in bindings]
    pair_rows = [{"sequence_a_sha256": sequence_sha(unique[r["a"]]), "sequence_b_sha256": sequence_sha(unique[r["b"]]),
                  "length_a": len(unique[r["a"]]), "length_b": len(unique[r["b"]]),
                  "shared_kmer_positions": r["shared_kmer_positions"], "identical": r["identical"],
                  "denominator": r["denominator"], "identity": f"{r['identity']:.6f}"} for r in raw_pairs]
    return {"fold_rows": fold_rows, "pair_rows": pair_rows, "pdb_group": pdb_group, "groups": groups,
            "chosen": chosen, "violations": violations,
            "summary": {"n_ions": len(bindings), "n_pdbs": len(pdb_sequences), "n_groups": len(groups),
                        "n_unique_sequences": len(unique), "n_near_copy_pairs": len(raw_pairs)}}


def code_identity() -> dict:
    """SHA-256 of the builder files plus the git commit and dirty flag of their checkout."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), root / "scripts" / "build_v3_folds.py"]
    identity = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files if path.is_file()}
    try:
        commit = subprocess.run(["git", "--no-optional-locks", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "--no-optional-locks", "-C", str(root), "status", "--porcelain", "--",
                                *identity], capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"files_sha256": identity, "git_commit": commit, "builder_files_dirty": dirty}


def write_outputs(out_dir: Path, result: dict, *, inputs: dict, seed: int, starts: int) -> dict:
    """Write fold file, pair list, group table, balance report and receipt; never overwrite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = result["chosen"]
    accepted = chosen["accepted"] and not result["violations"]
    hashes = {}
    hashes["near_copy_pairs.csv"] = _write(out_dir / "near_copy_pairs.csv", _csv_bytes(PAIR_COLUMNS, result["pair_rows"]))
    group_sizes: dict[str, int] = defaultdict(int)
    for pdb, gid in result["pdb_group"].items():
        group_sizes[gid] += 1
    group_rows = [{"pdbid": pdb, "group_id": gid, "n_pdbs_in_group": group_sizes[gid]}
                  for pdb, gid in sorted(result["pdb_group"].items())]
    hashes["pdb_groups.csv"] = _write(out_dir / "pdb_groups.csv",
                                      _csv_bytes(("pdbid", "group_id", "n_pdbs_in_group"), group_rows))
    balance = {"accepted": accepted, "acceptance": ACCEPTANCE, "passing_starts": chosen["passing_starts"],
               "chosen_start": chosen["start"], "score": round(chosen["score"], 9),
               "violations": result["violations"], **chosen["balance"]}
    hashes["fold_balance_report.json"] = _write(out_dir / "fold_balance_report.json",
                                                (json.dumps(balance, indent=2, sort_keys=True) + "\n").encode())
    if accepted:
        hashes["fold_membership.csv"] = _write(out_dir / "fold_membership.csv",
                                               _csv_bytes(FOLD_COLUMNS, result["fold_rows"]))
    import Bio

    receipt = {
        "builder_version": BUILDER_VERSION, "accepted": accepted, "seed": seed, "starts": starts,
        "chosen_start": chosen["start"],
        "parameters": {"n_folds": N_FOLDS, "min_chain_length": MIN_CHAIN_LENGTH,
                       "identity_threshold": IDENTITY_THRESHOLD, "kmer": KMER,
                       "alignment": "global, free end gaps, BLOSUM62, open -11, extend -1, first optimal",
                       "identity": "identical standard residues / (shorter length + internal insertion events)",
                       "acceptance": ACCEPTANCE},
        "inputs": inputs, "summary": result["summary"], "outputs_sha256": hashes,
        "code": code_identity(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "biopython": Bio.__version__},
    }
    _write(out_dir / "fold_receipt.json", (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt
