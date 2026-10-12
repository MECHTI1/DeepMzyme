#!/usr/bin/env python3
"""Label-blind test-row preparation for v3 step F (draft v3 section F3b; rules in step_f_spec.json).

Every source row of the PMM test side receives exactly one disposition (scored, scored_link_flagged or unscorable with
a reason) from identifiers and structures only:

- the site manifest and the source CSV are read through column allowlists: the label-bearing cells (``element``,
  ``metaltype``, ``label_metal`` and every other column outside the allowlist) pass through the CSV reader and are
  discarded record by record; they are never kept, used or written;
- a target is scorable when exactly one hetero residue at the row's chain and residue number in model 0 is a supported
  transition-metal ion residue with one atom without alternate locations and at least one residue within 10 A, as in the
  training cohort; the metal is used in memory only to ask whether it is supported, and *which* metal it is is never
  read into a decision or written to an output;
- stored outputs carry no element, residue name, atom name or readable physical ion ID (for metal ions these spell the
  element); a physical ion is identified by ``ion_key``, the SHA-256 of its structure content, model, chain, residue
  number, insertion code and alternate location (stored identifier columns only, so the key reveals nothing more);
- symmetry-LINK rows are flagged, never dropped; rows resolving to the same ion are aliases of the first one, and the
  row accounting (one disposition per row, aliases, physical ions) is checked;
- the clean subset applies the explicit criteria C1-C3 of the specification (training PDB ID, A1 near-copy or identical
  chain, previously opened PDB ID).

This module also plans and generates the test-side ESMC embeddings for the scored rows' context chains with the
training files' payload format (``<structure stem>_chain_<chain>_esmc.pt``) and model dtype, and it validates the
records that gate the test-side actions (the recorded Freeze 1 and Freeze 2 and the campaign-wide prediction claim; used
by run_pmm_v3_step_f.py and the report). It never uses a label, never trains and never predicts. It is run only after
Freeze 1 (step F section F3a); before that it is exercised on synthetic or training-side data only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SPEC_PATH = ROOT / "docs" / "campaigns" / "pmm_ion_metal_v3" / "step_f_spec.json"
# The step F rules as corrected on 2026-10-12 (log v3-043; first prepared in v3-042); a different file is refused (a
# change needs a new dated log entry, the user's decision and new pins here and in pmm_v3_step_f.py).
SPEC_SHA256 = "04d22cc432de6aaece0ea5a140cbaac0799d1194f79eb37f50aa1bec24d9b77c"
if hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest() != SPEC_SHA256:
    raise ValueError(f"{SPEC_PATH} differs from the pinned step F specification")
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
SITE_ALLOWLIST = tuple(SPEC["test_rows"]["label_blind_inputs"]["site_manifest_allowlist"])
FORBIDDEN_COLUMNS = tuple(SPEC["test_rows"]["label_blind_inputs"]["forbidden_site_manifest_columns"])
SOURCE_ALLOWLIST = tuple(SPEC["test_rows"]["label_blind_inputs"]["source_csv_allowlist"])
UNSCORABLE_REASONS = tuple(SPEC["test_rows"]["unscorable_reasons"])
SCORED, LINKED, UNSCORABLE = "scored", "scored_link_flagged", "unscorable"
ROW_COLUMNS = ("source_uid", "source_row", "pdbid", "structure_name", "structure_sha256", "chain", "resseq", "icode",
               "altloc", "coord_x", "coord_y", "coord_z", "ion_key", "alias_primary_source_uid", "disposition", "reason",
               "link_flagged", "context_chains", "n_context_residues", "clean_subset", "clean_exclusions")
PLAN_COLUMNS = ("structure_name", "chain", "sequence_sha256", "sequence_length", "n_ions_using_chain")
MEMBERSHIP_FILE_NAMES = ("prepared_rows.csv", "scored_source_uids.txt", "link_flagged_source_uids.txt",
                         "unscorable_rows.csv", "clean_subset_source_uids.txt", "clean_exclusions.csv",
                         "previously_opened_pdb_ids.txt", "training_pdb_ids.txt", "test_esm_plan.csv")
TEST_ESM = SPEC["test_esm"]
# The records that gate the test side, at fixed paths under the campaign root (written by pmm_v3_step_f.py, whose
# EVIDENCE_DIR and record names must equal these).
EVIDENCE_DIR = "step_f_evidence"
RECORD_NAMES = {"freeze1": "freeze1.json", "freeze2": "freeze2.json", "prediction": "f4_prediction_claim.json"}
LOG_ENTRY = re.compile(r"v3-\d{3}")
PLACEHOLDER = "<fill in>"


class PreparationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha(sequence: str) -> str:
    return sha256_bytes(sequence.encode("utf-8"))


# ---------------------------------------------------------------------------
# Allowlisted reads (cells outside the allowlist are discarded per record, never kept)
# ---------------------------------------------------------------------------

def read_allowlisted(path: Path, allowlist: Iterable[str]) -> list[dict[str, str]]:
    """Rows of a CSV restricted to ``allowlist`` columns; other cells are dropped while reading each record."""
    allowlist = tuple(allowlist)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        missing = [column for column in allowlist if column not in header]
        require(not missing, f"{path.name}: allowlisted columns missing: {missing}")
        indices = [header.index(column) for column in allowlist]
        rows = []
        for record in reader:
            require(len(record) == len(header), f"{path.name}: a record has {len(record)} fields, not {len(header)}")
            rows.append({column: record[index] for column, index in zip(allowlist, indices)})
    return rows


def ion_key(structure_sha256: str, model_index: int, chain: str, resseq: int, icode: str, altloc: str) -> str:
    """Label-free identity of one physical ion from identifier fields only. No atom name: for a metal ion it spells the
    element, which six guesses would recover from the hash. The residue is unique: a scorable target is the only
    supported-metal atom at (chain, resseq) with one insertion code and no alternate locations (resolve_target)."""
    return sha256_bytes("|".join([structure_sha256, str(model_index), chain, str(resseq), icode, altloc])
                        .encode("utf-8"))


# ---------------------------------------------------------------------------
# Resolution of one structure (worker; label-free)
# ---------------------------------------------------------------------------

def resolve_target(structure, chain: str, resseq: int) -> tuple[Any | None, str]:
    """The unique supported-metal ion atom at (chain, resseq) in model 0, or (None, reason). The metal's identity is
    used only to ask whether the residue is one of the supported transition-metal ion residues."""
    from training.source_cohort import target_atom_candidates

    candidates = target_atom_candidates(structure, model_index=0, chain=chain, resseq=resseq)
    if not candidates:
        return None, "target_residue_absent_in_model_0"
    metal = [candidate for candidate in candidates if candidate.symbol is not None]
    if not metal:
        return None, "target_not_a_supported_metal_ion"
    if len({(candidate.icode, candidate.resname) for candidate in metal}) > 1:
        return None, "ambiguous_target_residue"
    if len(metal) != 1:
        return None, "multiple_atoms_in_target_residue"
    if len(metal[0].altloc_coords) > 1:
        return None, "target_atom_alternate_locations"
    return metal[0], ""


def resolve_structure(task: tuple[str, str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Resolve every requested row of one structure; per-row outcomes, never an exception."""
    import torch

    from benchmarking.pmm_ion_cohort import pdb_symmetry_facts, protein_symmetry_links
    from data_structures import DEFAULT_POCKET_RADIUS
    from embed_helpers.esmc import extract_chain_sequences
    from graph.structure_parsing import (MetalAtomRecord, collect_structure_residues_and_metals,
                                         find_pocket_residues_near_metal_cluster, parse_structure_file)
    from training.source_cohort import format_coord

    path_text, structure_sha, requests = task
    path = Path(path_text)
    torch.set_num_threads(1)
    try:
        structure = parse_structure_file(str(path), structure_id=path.stem)
        residues, _metals = collect_structure_residues_and_metals(structure, model_index=0)
        residue_coords = [torch.stack([coord.float() for coord in r.atoms.values()], dim=0) for r in residues]
        facts = pdb_symmetry_facts(path)[1] if path.suffix.lower() == ".pdb" else []
        sequences = extract_chain_sequences(structure)
    except Exception as exc:  # recorded per row
        return {"results": [{"source_row": r["source_row"], "reason": "structure_parse_error",
                             "detail": type(exc).__name__} for r in requests]}
    results = []
    for request in requests:
        chain, resseq = request["chain"], int(request["resseq"])
        candidate, reason = resolve_target(structure, chain, resseq)
        if candidate is None:
            results.append({"source_row": request["source_row"], "reason": reason})
            continue
        altloc = candidate.altloc_coords[0][0]
        coord = torch.tensor(candidate.coord, dtype=torch.float32)
        target = MetalAtomRecord(coord=coord, symbol="X", site_id=(chain, resseq, candidate.icode))
        nearby = find_pocket_residues_near_metal_cluster(residues, [target], pocket_radius=DEFAULT_POCKET_RADIUS,
                                                         residue_coord_tensors=residue_coords)
        if not nearby:
            results.append({"source_row": request["source_row"], "reason": "no_context_residues"})
            continue
        links = protein_symmetry_links(facts, (chain, resseq, candidate.icode, candidate.resname, candidate.atom_name))
        context = sorted({residue.chain_id for residue in nearby})
        results.append({
            "source_row": request["source_row"], "reason": "", "icode": candidate.icode, "altloc": altloc,
            "coord": [format_coord(value) for value in candidate.coord],
            "ion_key": ion_key(structure_sha, 0, chain, resseq, candidate.icode, altloc),
            "link_flagged": bool(links), "context_chains": context, "n_context_residues": len(nearby),
            "context_sequences": {c: sequences.get(c) for c in context},
        })
    return {"results": results}


# ---------------------------------------------------------------------------
# Row dispositions
# ---------------------------------------------------------------------------

def structure_files(test_dir: Path) -> dict[str, tuple[Path, str]]:
    rows = read_allowlisted(test_dir / "structure_manifest.csv", ("structure_name", "relative_path", "sha256"))
    return {row["structure_name"]: (test_dir / row["relative_path"], row["sha256"]) for row in rows}


def prepare_rows(test_dir: Path, source_csv: Path, *, expected_source_sha256: str | None, expected_side: str = "test",
                 row_limit: int | None = None, only_rows: set[int] | None = None,
                 workers: int = 1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One label-free disposition per source row (plus context sequences for planning, not written as rows).
    ``only_rows`` restricts the output to those source rows (readiness checks on training-side data only)."""
    test_dir, source_csv = Path(test_dir), Path(source_csv)
    source_sha = sha256_file(source_csv)
    require(expected_source_sha256 is None or source_sha == expected_source_sha256,
            f"The source CSV has SHA-256 {source_sha}, not the pinned {expected_source_sha256}")
    source = read_allowlisted(source_csv, SOURCE_ALLOWLIST)
    if row_limit is not None:
        source = source[:row_limit]
    manifest = read_allowlisted(test_dir / "site_manifest.csv", SITE_ALLOWLIST)
    by_row: dict[int, dict[str, str]] = {}
    for row in manifest:
        require(row["source_side"] == expected_side, f"site manifest row {row['source_uid']} is not on the "
                                                     f"{expected_side} side")
        number = int(row["source_row"])
        require(row["source_uid"] == f"sha256:{source_sha}:row:{number}",
                f"site manifest row {row['source_uid']} does not bind the source CSV")
        require(number not in by_row, f"repeated manifest source_row {number}")
        by_row[number] = row
    files = structure_files(test_dir)
    verified: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for number, src in enumerate(source, start=1):
        if only_rows is not None and number not in only_rows:
            continue
        row = {column: "" for column in ROW_COLUMNS}
        row.update(source_uid=f"sha256:{source_sha}:row:{number}", source_row=number, pdbid=src["pdbid"].lower(),
                   disposition=UNSCORABLE, link_flagged=False, clean_subset=False)
        rows.append(row)
        site = by_row.get(number)
        if site is None:
            row["reason"] = "missing_reconstruction_row"
            continue
        if (site["pdbid"].lower(), site["residueid_ion"].strip()) != (src["pdbid"].lower(), src["residueid_ion"].strip()):
            row["reason"] = "manifest_source_identifier_mismatch"
            continue
        name = site["structure_name"]
        row.update(structure_name=name, chain=site["chain"], resseq=site["resseq"])
        if name not in files:
            row["reason"] = "structure_unverified"
            continue
        path, expected = files[name]
        if name not in verified:
            verified[name] = path.is_file() and sha256_file(path) == expected
        if not verified[name]:
            row["reason"] = "structure_unverified"
            continue
        row["structure_sha256"] = expected
        tasks[name].append({"source_row": number, "chain": site["chain"], "resseq": int(site["resseq"])})
    jobs = [(str(files[name][0]), files[name][1], requests) for name, requests in sorted(tasks.items())]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(resolve_structure, jobs, chunksize=4))
    else:
        outcomes = [resolve_structure(job) for job in jobs]
    sequences: dict[tuple[str, str], str] = {}
    by_number = {row["source_row"]: row for row in rows}
    for job, outcome in zip(jobs, outcomes):
        name = Path(job[0]).name
        for result in outcome["results"]:
            row = by_number[result["source_row"]]
            if result["reason"]:
                row["reason"] = result["reason"]
                continue
            row.update(icode=result["icode"], altloc=result["altloc"], coord_x=result["coord"][0],
                       coord_y=result["coord"][1], coord_z=result["coord"][2], ion_key=result["ion_key"],
                       disposition=LINKED if result["link_flagged"] else SCORED, link_flagged=result["link_flagged"],
                       context_chains=";".join(result["context_chains"]),
                       n_context_residues=result["n_context_residues"])
            for chain, sequence in result["context_sequences"].items():
                if sequence is not None:
                    sequences[(name, chain)] = sequence
    primaries: dict[str, str] = {}
    for row in rows:
        if row["disposition"] != UNSCORABLE:
            row["alias_primary_source_uid"] = primaries.setdefault(row["ion_key"], row["source_uid"])
    unknown = sorted({row["reason"] for row in rows if row["disposition"] == UNSCORABLE} - set(UNSCORABLE_REASONS))
    require(not unknown, f"Unspecified unscorable reasons: {unknown}")
    expected_rows = len(source) if only_rows is None else len(only_rows & set(range(1, len(source) + 1)))
    accounting = check_row_accounting(rows, expected_rows=expected_rows)
    counts = Counter(row["disposition"] for row in rows)
    summary = {"source_csv_sha256": source_sha, "source_rows": len(rows), "dispositions": dict(sorted(counts.items())),
               "row_subset": None if only_rows is None else len(only_rows),
               "unscorable_reasons": dict(sorted(Counter(r["reason"] for r in rows if r["reason"]).items())),
               "physical_ions": accounting["physical_ions"], "aliases": accounting["aliases"],
               "structures_resolved": len(jobs)}
    return rows, {"summary": summary, "sequences": sequences}


def check_row_accounting(rows: list[dict[str, Any]], *, expected_rows: int) -> dict[str, int]:
    """Every source row exactly once with one specified disposition; each physical ion (distinct ion_key) has one
    primary row, the earliest; an alias points to that primary and has the same structure, identifiers and coordinates;
    unscorable rows carry no ion key or alias."""
    numbers = [int(row["source_row"]) for row in rows]
    require(len(rows) == expected_rows and len(set(numbers)) == len(numbers) and numbers == sorted(numbers),
            f"{len(rows)} prepared rows for {expected_rows} source rows (each once, in source order)")
    require(all(row["disposition"] in (SCORED, LINKED, UNSCORABLE) for row in rows), "an unspecified disposition")
    identity = ("structure_sha256", "chain", "resseq", "icode", "altloc", "coord_x", "coord_y", "coord_z")
    first: dict[str, dict[str, Any]] = {}
    aliases = 0
    for row in rows:
        if row["disposition"] == UNSCORABLE:
            require(not row["ion_key"] and not row["alias_primary_source_uid"] and row["reason"],
                    f"{row['source_uid']}: an unscorable row carries an ion key, an alias or no reason")
            continue
        require(len(str(row["ion_key"])) == 64 and row["ion_key"] == ion_key(
            row["structure_sha256"], 0, row["chain"], int(row["resseq"]), row["icode"], row["altloc"]),
            f"{row['source_uid']}: the ion key is not the hash of its stored identifiers")
        primary = first.setdefault(row["ion_key"], row)
        require(row["alias_primary_source_uid"] == primary["source_uid"],
                f"{row['source_uid']}: the alias does not point to the first row of its physical ion")
        if primary is not row:
            aliases += 1
            require(all(str(row[key]) == str(primary[key]) for key in identity),
                    f"{row['source_uid']}: an alias differs from its primary ion")
    scored = sum(1 for row in rows if row["disposition"] != UNSCORABLE)
    require(len(first) + aliases == scored, "physical ions and aliases do not add up to the scored rows")
    return {"rows": len(rows), "scored": scored, "physical_ions": len(first), "aliases": aliases}


# ---------------------------------------------------------------------------
# Clean subset (criteria C1-C3 of the specification)
# ---------------------------------------------------------------------------

def near_copy_hits(queries: list[str], references: list[str]) -> list[bool]:
    """For each query sequence: is it identical to, or a near-copy (A1 rule) of, any reference sequence of at least
    MIN_CHAIN_LENGTH residues? Identical sequences always match, as A1 grouping links them (v3_folds.group_pdbs): a
    chain with more than 10% non-standard residues never reaches the identity threshold, not even against itself. The
    5-mer prefilter is applied in the direction the A1 proof requires (shared positions of the shorter member in the
    longer one)."""
    import numpy as np

    from benchmarking.v3_folds import IDENTITY_THRESHOLD, MIN_CHAIN_LENGTH, _kmers, near_copy_identity, prefilter_threshold

    eligible_references = {reference for reference in references if len(reference) >= MIN_CHAIN_LENGTH}

    def postings(sequences):
        table: dict[str, list[int]] = defaultdict(list)
        for index, sequence in enumerate(sequences):
            if len(sequence) >= MIN_CHAIN_LENGTH:
                for word in {w for w in _kmers(sequence) if w is not None}:
                    table[word].append(index)
        return {word: np.asarray(ids, dtype=np.int64) for word, ids in table.items()}

    def counts_in(sequence, table, size):
        arrays = [table[w] for w in _kmers(sequence) if w is not None and w in table]
        return np.bincount(np.concatenate(arrays), minlength=size) if arrays else np.zeros(size, dtype=np.int64)

    ref_table, query_table = postings(references), postings(queries)
    query_in_ref = [counts_in(q, ref_table, len(references)) if len(q) >= MIN_CHAIN_LENGTH else None for q in queries]
    ref_in_query = np.zeros((len(queries), len(references)), dtype=np.int64)
    for j, reference in enumerate(references):
        if len(reference) >= MIN_CHAIN_LENGTH:
            ref_in_query[:, j] = counts_in(reference, query_table, len(queries))
    hits = []
    for i, query in enumerate(queries):
        found = len(query) >= MIN_CHAIN_LENGTH and query in eligible_references
        if not found and query_in_ref[i] is not None:
            for j in np.nonzero((query_in_ref[i] > 0) | (ref_in_query[i] > 0))[0].tolist():
                reference = references[j]
                if len(reference) < MIN_CHAIN_LENGTH:
                    continue
                query_shorter = (len(query), query) <= (len(reference), reference)
                shorter = query if query_shorter else reference
                shared = query_in_ref[i][j] if query_shorter else ref_in_query[i, j]
                if shared < prefilter_threshold(len(shorter)):
                    continue
                if near_copy_identity(query, reference)[0] >= IDENTITY_THRESHOLD:
                    found = True
                    break
        hits.append(found)
    return hits


def apply_clean_subset(rows: list[dict[str, Any]], sequences: dict[tuple[str, str], str], *,
                       training_pdbids: set[str], training_sequences: Iterable[str],
                       previously_opened: set[str]) -> dict[str, Any]:
    """Annotate every row with its clean-subset membership and the criteria that exclude it."""
    from benchmarking.v3_folds import MIN_CHAIN_LENGTH

    unique = sorted({seq for seq in sequences.values() if len(seq) >= MIN_CHAIN_LENGTH})
    references = sorted(set(training_sequences))
    flagged = dict(zip(unique, near_copy_hits(unique, references)))
    short_chain_only = 0
    for row in rows:
        reasons = []
        if row["pdbid"] in training_pdbids:
            reasons.append("C1_training_pdb_id")
        chains = [c for c in row["context_chains"].split(";") if c] if row["context_chains"] else []
        if not chains or any((row["structure_name"], c) not in sequences for c in chains):
            reasons.append("C2_unverifiable")  # no context chain, or one without a first-model sequence
        elif any(flagged.get(sequences[(row["structure_name"], c)], False) for c in chains):
            reasons.append("C2_near_copy_chain")
        elif all(len(sequences[(row["structure_name"], c)]) < MIN_CHAIN_LENGTH for c in chains):
            short_chain_only += 1  # the disclosed A1 limitation: chains under 30 residues are never compared
        if row["pdbid"] in previously_opened:
            reasons.append("C3_previously_opened_pdb_id")
        row["clean_exclusions"] = ";".join(reasons)
        row["clean_subset"] = not reasons
    eligible_references = set(seq for seq in references if len(seq) >= MIN_CHAIN_LENGTH)
    return {"clean_subset_rows": sum(1 for r in rows if r["clean_subset"]),
            "exclusion_counts": dict(sorted(Counter(reason for r in rows for reason in
                                                     filter(None, r["clean_exclusions"].split(";"))).items())),
            "unique_context_sequences": len(unique), "near_copy_sequences": sum(flagged.values()),
            "identical_to_a_reference": sum(1 for seq in unique if seq in eligible_references),
            "rows_with_only_short_context_chains": short_chain_only}


# ---------------------------------------------------------------------------
# ESMC plan and generation for the scored rows' context chains
# ---------------------------------------------------------------------------

def esm_plan(rows: list[dict[str, Any]], sequences: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    needed: Counter = Counter()
    for row in rows:
        if row["disposition"] != UNSCORABLE and row["alias_primary_source_uid"] == row["source_uid"]:
            for chain in row["context_chains"].split(";"):
                needed[(row["structure_name"], chain)] += 1
    plan = []
    for (name, chain), count in sorted(needed.items()):
        sequence = sequences.get((name, chain))
        require(sequence is not None, f"{name}: context chain {chain!r} has no first-model sequence")
        plan.append({"structure_name": name, "chain": chain, "sequence_sha256": sequence_sha(sequence),
                     "sequence_length": len(sequence), "n_ions_using_chain": count})
    return plan


def embedding_path(out_dir: Path, structure_name: str, chain: str) -> Path:
    return Path(out_dir) / f"{Path(structure_name).stem}_chain_{chain}_esmc.pt"


def generate_embeddings(plan: list[dict[str, Any]], test_dir: Path, out_dir: Path, *,
                        embed: Callable[[str], Any] | None = None, device: str | None = None,
                        freeze1_sha256: str | None = None) -> dict[str, Any]:
    """One payload per planned (structure, chain) in the training files' format (pmm_ion_features.generate_embeddings
    core); each unique sequence is embedded once. The model runs in the training embeddings' model dtype (spec test_esm:
    bfloat16; the SDK keeps float32 on CPU, so it is cast). ``embed`` replaces the ESMC model in tests."""
    import torch

    from benchmarking.pmm_ion_features import ESM_DIM, ESM_MODEL_NAME
    from embed_helpers.esmc import extract_chain_sequences, parse_structure, utc_timestamp
    from training.esm_feature_loading import (build_embedding_payload, embedding_metadata_from_payload,
                                              residue_keys_for_structure_chain, write_embedding_metadata_sidecar)

    require(ESM_MODEL_NAME == TEST_ESM["model"], f"The ESMC model is {ESM_MODEL_NAME}, not {TEST_ESM['model']}")
    model_name, model_dtype, emitted = ESM_MODEL_NAME, "injected", set()
    if embed is None:
        from embed_helpers.esmc import _esmc_sdk, clean_embedding_length, load_esmc_model

        model, device = load_esmc_model(device, model_name=ESM_MODEL_NAME)
        dtype = getattr(torch, TEST_ESM["model_dtype"])
        model = model.to(dtype)  # the SDK casts to bfloat16 on CUDA only (as for the training embeddings)
        model_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        require(model_dtype == TEST_ESM["model_dtype"], f"The ESMC model runs in {model_dtype}")
        _ESMC, ESMProtein, LogitsConfig = _esmc_sdk()

        def embed(sequence: str):
            # The SDK's CUDA path runs under bfloat16 autocast; the same is set here for the CPU.
            with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
                output = model.logits(model.encode(ESMProtein(sequence=sequence)),
                                      LogitsConfig(sequence=False, return_embeddings=True))
            embedding = clean_embedding_length(output.embeddings, len(sequence)).detach().cpu()
            emitted.add(str(embedding.dtype).removeprefix("torch."))
            return embedding.to(getattr(torch, TEST_ESM["storage_dtype"]))  # the training files store float32
    else:
        model_name = "injected-test-embedder"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    files = structure_files(Path(test_dir))
    cache: dict[str, Any] = {}
    inventory = []
    by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plan:
        by_structure[row["structure_name"]].append(row)
    with torch.no_grad():
        for name, rows in sorted(by_structure.items()):
            path, expected = files[name]
            require(sha256_file(path) == expected, f"{name}: structure differs from its manifest")
            structure = parse_structure(path)
            sequences = extract_chain_sequences(structure)
            for row in rows:
                sequence = sequences[row["chain"]]
                require(sequence_sha(sequence) == row["sequence_sha256"] and len(sequence) == int(row["sequence_length"]),
                        f"{name}:{row['chain']} sequence differs from the plan")
                if row["sequence_sha256"] not in cache:
                    embedding = embed(sequence)
                    require(embedding.dim() == 2 and embedding.size(0) == len(sequence) and embedding.size(1) == ESM_DIM
                            and bool(torch.isfinite(embedding).all()), f"{name}:{row['chain']} invalid embedding")
                    cache[row["sequence_sha256"]] = embedding.contiguous()
                embedding = cache[row["sequence_sha256"]]
                residues, _ = residue_keys_for_structure_chain(structure, row["chain"])
                payload = build_embedding_payload(
                    embedding.float(), residues, structure_id=Path(name).stem, chain_id=row["chain"],
                    source_path=str(path),
                    metadata={"esm_model_name": model_name, "esm_checkpoint_name": model_name, "embedding_dim": ESM_DIM,
                              "n_residues": int(embedding.size(0)), "generated_at": utc_timestamp(),
                              "source_structure_id": Path(name).stem, "source_chain_id": row["chain"],
                              "source_sequence_length": len(sequence), "source_sequence_sha256": row["sequence_sha256"],
                              "campaign": "pmm_ion_metal_v3 step F test side (label-free)",
                              "model_dtype": model_dtype, "device": str(device),
                              "storage_dtype": str(embedding.dtype).removeprefix("torch."),
                              "context_note": "full-length sequence, no truncation"})
                payload["embeddings"] = embedding
                target = embedding_path(out_dir, name, row["chain"])
                torch.save(payload, target)
                write_embedding_metadata_sidecar(target, embedding_metadata_from_payload(payload))
                inventory.append({"structure_name": name, "chain": row["chain"], "file": target.name,
                                  "storage_dtype": str(embedding.dtype).removeprefix("torch."),
                                  "sha256": sha256_file(target),
                                  "sidecar_sha256": sha256_file(target.with_name(target.name + ".json"))})
    receipt = {"model": model_name, "model_dtype": model_dtype, "emitted_dtypes": sorted(emitted),
               "storage_dtypes": sorted({item["storage_dtype"] for item in inventory}), "device": str(device),
               "files": len(inventory),
               "unique_sequences": len(cache), "freeze1_sha256": freeze1_sha256, "tool_sha256": sha256_file(Path(__file__)),
               "inventory": inventory, "inventory_sha256": inventory_sha256(inventory)}
    (out_dir / "test_esm_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    return receipt


# ---------------------------------------------------------------------------
# Outputs (written once; every list hashed for Freeze 2)
# ---------------------------------------------------------------------------

def _csv_bytes(columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> bytes:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in columns})
    return buffer.getvalue().encode("utf-8")


def membership_files(rows: list[dict[str, Any]], plan: list[dict[str, Any]], *, previously_opened: set[str],
                     training_pdbids: set[str]) -> dict[str, bytes]:
    """Every Freeze 2 artifact as bytes (deterministic for the same inputs)."""
    def uid_lines(predicate):
        return ("".join(f"{r['source_uid']}\n" for r in rows if predicate(r))).encode("utf-8")

    files = {
        "prepared_rows.csv": _csv_bytes(ROW_COLUMNS, rows),
        "scored_source_uids.txt": uid_lines(lambda r: r["disposition"] != UNSCORABLE),
        "link_flagged_source_uids.txt": uid_lines(lambda r: r["disposition"] == LINKED),
        "unscorable_rows.csv": _csv_bytes(("source_uid", "reason"), [r for r in rows if r["disposition"] == UNSCORABLE]),
        "clean_subset_source_uids.txt": uid_lines(lambda r: r["clean_subset"]),
        "clean_exclusions.csv": _csv_bytes(("source_uid", "clean_exclusions"), [r for r in rows if not r["clean_subset"]]),
        "previously_opened_pdb_ids.txt": "".join(f"{p}\n" for p in sorted(previously_opened)).encode("utf-8"),
        "training_pdb_ids.txt": "".join(f"{p}\n" for p in sorted(training_pdbids)).encode("utf-8"),
        "test_esm_plan.csv": _csv_bytes(PLAN_COLUMNS, plan),
    }
    require(tuple(files) == MEMBERSHIP_FILE_NAMES, "The membership files are not the specified set")
    return files


def write_preparation(out_dir: Path, files: dict[str, bytes], summary: dict[str, Any], *,
                      freeze1_sha256: str | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, data in sorted(files.items()):
        (out_dir / name).write_bytes(data)
        hashes[name] = sha256_bytes(data)
    record = {**summary, "files_sha256": hashes, "spec_sha256": sha256_file(SPEC_PATH),
              "tool_sha256": sha256_file(Path(__file__)), "freeze1_sha256": freeze1_sha256, "labels_read": False,
              "label_cells": "passed through the CSV reader and discarded record by record, unused",
              "held_out_labels_accessed": False, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "preparation_summary.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                                      encoding="utf-8")
    return record


def prepare(test_dir: Path, source_csv: Path, out_dir: Path, *, expected_source_sha256: str | None,
            training_pdbids: set[str], training_sequences: Iterable[str] | Callable[[], Iterable[str]],
            previously_opened: set[str], expected_side: str = "test", row_limit: int | None = None,
            workers: int = 1, freeze1_sha256: str | None = None) -> dict[str, Any]:
    """The whole label-blind preparation: rows, clean subset, ESMC plan and hashed outputs. A callable
    ``training_sequences`` is called after the test rows are read (loading the training side installs the held-out
    read guard for the rest of the process)."""
    rows, extra = prepare_rows(test_dir, source_csv, expected_source_sha256=expected_source_sha256,
                               expected_side=expected_side, row_limit=row_limit, workers=workers)
    if callable(training_sequences):
        training_sequences = training_sequences()
    clean = apply_clean_subset(rows, extra["sequences"], training_pdbids=training_pdbids,
                               training_sequences=training_sequences, previously_opened=previously_opened)
    plan = esm_plan(rows, extra["sequences"])
    files = membership_files(rows, plan, previously_opened=previously_opened, training_pdbids=training_pdbids)
    summary = {**extra["summary"], "clean_subset": clean, "esm_plan_chains": len(plan),
               "previously_opened_pdb_ids": len(previously_opened), "training_pdb_ids": len(training_pdbids)}
    return write_preparation(out_dir, files, summary, freeze1_sha256=freeze1_sha256)


# ---------------------------------------------------------------------------
# The records that gate the test side (shared by run_pmm_v3_step_f.py and the report)
# ---------------------------------------------------------------------------

def inventory_sha256(inventory: list[dict[str, Any]]) -> str:
    return sha256_bytes(json.dumps(inventory, sort_keys=True).encode())


def evidence_paths(campaign_dir: Path) -> dict[str, Path]:
    """The canonical record paths under the campaign root; no other location is read."""
    evidence = Path(campaign_dir) / EVIDENCE_DIR
    return {key: evidence / name for key, name in RECORD_NAMES.items()}


def has_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(has_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(has_placeholder(item) for item in value)
    return value == PLACEHOLDER


def _recorded(path: Path, kind: str, what: str) -> tuple[dict[str, Any], str]:
    """A recorded freeze at its canonical path: the user's logged go-ahead (v3-NNN), no template placeholder, and this
    specification."""
    require(path.is_file(), f"{what} is not recorded ({path})")
    record = json.loads(path.read_text(encoding="utf-8"))
    require(record.get("kind") == kind and LOG_ENTRY.fullmatch(str(record.get("user_go_ahead_log_entry"))) is not None,
            f"{path} is not a {what} record with the user's logged go-ahead (v3-NNN)")
    require(not has_placeholder(record), f"{path} has a '{PLACEHOLDER}' field: a template is not a record")
    require(record.get("step_f_spec_sha256") == sha256_file(SPEC_PATH), f"{what} binds another specification")
    return record, sha256_file(path)


def check_freeze1(campaign_dir: Path, *, tools: dict[str, Path]) -> tuple[dict[str, Any], str]:
    """The recorded Freeze 1 and the frozen code: every named tool file must have the SHA-256 Freeze 1 recorded (a code
    change after Freeze 1 needs an amended Freeze 1, spec freeze1_amendment). Returns the record and its SHA-256."""
    record, digest = _recorded(evidence_paths(campaign_dir)["freeze1"], "step_f_freeze_1", "Freeze 1")
    frozen = record.get("tool_sha256") or {}
    changed = sorted(name for name, path in tools.items() if frozen.get(name) != sha256_file(path))
    require(not changed, f"{changed} differ from the code Freeze 1 recorded; a code change after Freeze 1 needs an "
                         "amended Freeze 1 (spec freeze1_amendment)")
    require(re.fullmatch(r"[0-9a-f]{64}", str(record.get("checkpoint_sha256"))) is not None,
            "Freeze 1 binds no refit checkpoint")
    return record, digest


def check_freeze2(campaign_dir: Path, *, freeze1_sha256: str, prepared_dir: Path,
                  esm_dir: Path | None = None) -> tuple[dict[str, Any], str]:
    """The recorded Freeze 2, bound to this Freeze 1: every membership file and the preparation summary by SHA-256, a
    preparation made under this Freeze 1 and, with ``esm_dir``, the recomputed embedding inventory and every payload."""
    paths = evidence_paths(campaign_dir)
    require(paths["freeze1"].is_file() and sha256_file(paths["freeze1"]) == freeze1_sha256,
            "The recorded Freeze 1 is not the one checked")
    record, digest = _recorded(paths["freeze2"], "step_f_freeze_2", "Freeze 2")
    require(record.get("freeze1_sha256") == freeze1_sha256, "Freeze 2 binds another Freeze 1 record")
    files = record.get("prepared_files_sha256")
    require(isinstance(files, dict) and set(files) == set(MEMBERSHIP_FILE_NAMES) | {"preparation_summary.json"},
            "Freeze 2 does not bind every prepared membership file and the preparation summary")
    for name, value in files.items():
        require(sha256_file(Path(prepared_dir) / name) == value, f"{name} differs from Freeze 2")
    summary = json.loads((Path(prepared_dir) / "preparation_summary.json").read_text(encoding="utf-8"))
    require(summary.get("freeze1_sha256") == freeze1_sha256 and summary.get("row_subset") is None,
            "The preparation was not made under this Freeze 1 for every source row")
    if esm_dir is not None:
        receipt = json.loads((Path(esm_dir) / "test_esm_receipt.json").read_text(encoding="utf-8"))
        require(receipt.get("freeze1_sha256") == freeze1_sha256, "The test embeddings were not made under this Freeze 1")
        require(inventory_sha256(receipt["inventory"]) == receipt["inventory_sha256"]
                == record.get("test_esm_inventory_sha256"), "The test embedding inventory differs from Freeze 2")
        for item in receipt["inventory"]:
            payload = Path(esm_dir) / item["file"]
            require(sha256_file(payload) == item["sha256"]
                    and sha256_file(payload.with_name(payload.name + ".json")) == item["sidecar_sha256"],
                    f"{item['file']} changed")
    return record, digest


def claim_prediction(campaign_dir: Path, claim: dict[str, Any]) -> tuple[Path, str]:
    """The campaign-wide write-once claim of the single prediction pass, made before any test structure is opened in
    F4: refused when a claim exists, whatever the output directory."""
    path = evidence_paths(campaign_dir)["prediction"]
    data = (json.dumps({**claim, "kind": "step_f_prediction_claim"}, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        raise PreparationError(f"The single prediction pass was already claimed ({path}); a second pass is refused "
                               "campaign-wide") from None
    return path, sha256_bytes(data)


def check_prediction_claim(campaign_dir: Path, *, freeze2_sha256: str, predictions_dir: Path) -> tuple[dict, str]:
    """The predictions are those of the single claimed pass (its go-ahead, its Freeze 2 and its output directory)."""
    path = evidence_paths(campaign_dir)["prediction"]
    require(path.is_file(), f"No prediction claim ({path}): these are not the predictions of the single recorded pass")
    claim = json.loads(path.read_text(encoding="utf-8"))
    require(claim.get("kind") == "step_f_prediction_claim" and claim.get("freeze2_sha256") == freeze2_sha256
            and Path(claim.get("out_dir", "")).resolve() == Path(predictions_dir).resolve(),
            "The predictions directory is not the one the single prediction pass claimed under this Freeze 2")
    require(LOG_ENTRY.fullmatch(str(claim.get("user_go_ahead_log_entry"))) is not None,
            "The prediction claim names no logged go-ahead for the single test pass")
    return claim, sha256_file(path)
