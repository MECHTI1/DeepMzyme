#!/usr/bin/env python3
"""Prepare a CLEAN-derived metalloenzyme subset for DeepMzyme.

This pipeline is intentionally separate from prepare_training_and_test_set/.
CLEAN starts from UniProt IDs and sequences, so AlphaFill/AlphaFold is used as
an independent structure-plus-metal source before MAHOMES catalytic filtering.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from Bio.PDB import MMCIFParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEAN_SPLITS_ROOT = PROJECT_ROOT / "DeepMzyme_Data" / "CLEAN_all_train_valid_splits"
DEFAULT_WORK_ROOT = Path("/media/Data/clean_sets/split30/fold0")
DEFAULT_OUTPUT_ROOT = None
ALPHAFILL_ENTRY_URL = "https://alphafill.eu/v1/aff/{accession}"
ALPHAFILL_JSON_URL = "https://alphafill.eu/v1/aff/{accession}/json"
UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
SUPPORTED_TRANSITION_METALS = frozenset({"MN", "FE", "CO", "NI", "CU", "ZN"})
SUMMARY_CSV_NAME = "final_data_summarazing_table_transition_metals_only_catalytic.csv"
WHETHER_CATALYTIC_CSV_NAME = "data_summarazing_table_transition_metals_whether_catalytic.csv"
PREDICTION_RESULTS_CSV_NAME = "prediction_results_summary.csv"
CANDIDATE_SITE_CSV_NAME = "candidate_site_summary.csv"
STRUCTURE_ID_RE = re.compile(r"^(?P<structure>[^_]+)__chain_(?P<chain>[^_]+)__EC_(?P<ec>.+)$")
COFACTOR_SYMBOL_PATTERNS = {
    "CO": (r"^CO(?:\(\d\+\))?$", r"\bCO CATION\b", r"COBALT"),
    "CU": (r"^CU(?:\(\d\+\))?$", r"\bCU CATION\b", r"COPPER"),
    "FE": (r"^FE(?:\(\d\+\))?$", r"\bFE CATION\b", r"IRON"),
    "MG": (r"^MG(?:\(\d\+\))?$", r"\bMG CATION\b", r"MAGNESIUM"),
    "MN": (r"^MN(?:\(\d\+\))?$", r"\bMN CATION\b", r"MANGANESE"),
    "NI": (r"^NI(?:\(\d\+\))?$", r"\bNI CATION\b", r"NICKEL"),
    "ZN": (r"^ZN(?:\(\d\+\))?$", r"\bZN CATION\b", r"ZINC"),
}

MANIFEST_FIELDS = [
    "clean_identity",
    "clean_fold",
    "split",
    "uniprot_id",
    "ecnumber",
    "sequence",
    "source_file",
    "source_row",
]

CANDIDATE_FIELDS = [
    "structure",
    "chain_resi",
    "metaltype",
    "ecnumber",
    "whether_catalytic",
    "uniprot_id",
    "clean_identity",
    "clean_fold",
    "clean_split",
    "sequence_length",
    "output_pdb",
    "alphafill_identity",
    "alphafill_alignment_length",
    "alphafill_pdb_id",
    "alphafill_pdb_resolution",
    "alphafill_pdb_asym_id",
    "alphafill_compound_id",
    "alphafill_analogue_id",
    "alphafill_asym_id",
    "alphafill_local_rmsd",
    "alphafill_binding_site_rmsd",
    "alphafill_local_environment_rmsd",
    "alphafill_pae_mean",
    "alphafill_binding_site_atom_count",
    "alphafill_transplant_atom_count",
    "uniprot_supported_transition_metals",
    "selected_by_uniprot_annotation",
    "selection_reason",
    "cluster_supported_metals",
    "cluster_size",
]

FINAL_SUMMARY_FIELDS = [
    "structure",
    "chain_resi",
    "metaltype",
    "ecnumber",
    "whether_catalytic",
    "uniprot_id",
    "clean_identity",
    "clean_fold",
    "clean_split",
    "alphafill_identity",
    "alphafill_alignment_length",
    "alphafill_pdb_id",
    "alphafill_pdb_resolution",
    "alphafill_compound_id",
    "alphafill_local_rmsd",
    "alphafill_binding_site_rmsd",
    "alphafill_local_environment_rmsd",
    "alphafill_pae_mean",
    "uniprot_supported_transition_metals",
    "selected_by_uniprot_annotation",
    "selection_reason",
]

PREDICTION_REQUIRED_COLUMNS = {
    "input file",
    "prediction",
    "Name1",
    "Name2",
    "Name3",
    "Name4",
    "Res#1",
    "Res#2",
    "Res#3",
    "Res#4",
}


@dataclass(frozen=True)
class CleanRow:
    clean_identity: str
    clean_fold: int
    split: str
    uniprot_id: str
    ecnumber: str
    sequence: str
    source_file: str
    source_row: int


@dataclass(frozen=True)
class MetalCandidate:
    metaltype: str
    coord: tuple[float, float, float]
    alphafill_identity: float
    alphafill_alignment_length: int | None
    alphafill_pdb_id: str
    alphafill_pdb_resolution: float | None
    alphafill_pdb_asym_id: str
    alphafill_compound_id: str
    alphafill_analogue_id: str
    alphafill_asym_id: str
    alphafill_local_rmsd: float | None
    alphafill_binding_site_rmsd: float | None
    alphafill_local_environment_rmsd: float | None
    alphafill_pae_mean: float | None
    alphafill_binding_site_atom_count: int | None
    alphafill_transplant_atom_count: int | None
    selected_by_uniprot_annotation: bool = False
    selection_reason: str = ""
    cluster_supported_metals: tuple[str, ...] = ()
    cluster_size: int = 1


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_ec_number_list(value: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for ec in re.split(r"[;,]", str(value)):
        normalized = ec.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return ";".join(values)


def sanitize_filename_fragment(value: str) -> str:
    sanitized = str(value).strip()
    sanitized = re.sub(r"\s+", "", sanitized)
    sanitized = sanitized.replace("/", "-")
    sanitized = sanitized.replace(";", ",")
    sanitized = re.sub(r"[&|<>$`'\"(){}\\]", "-", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "unknown"


def canonicalize_metal(value: str) -> str | None:
    letters_only = "".join(ch for ch in str(value).strip().upper() if ch.isalpha())
    for metal in sorted(SUPPORTED_TRANSITION_METALS, key=len, reverse=True):
        if letters_only == metal:
            return metal
    return None


def extract_annotated_metal_symbols_from_names(names: Iterable[str]) -> list[str]:
    symbols: set[str] = set()
    for name in names:
        upper_name = str(name).upper().strip()
        for symbol, patterns in COFACTOR_SYMBOL_PATTERNS.items():
            if any(re.search(pattern, upper_name) for pattern in patterns):
                symbols.add(symbol)
    return sorted(symbol for symbol in symbols if symbol in SUPPORTED_TRANSITION_METALS)


def extract_uniprot_supported_transition_metals(uniprot_json: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for comment in uniprot_json.get("comments", []) or []:
        if comment.get("commentType") != "COFACTOR":
            continue
        for cofactor in comment.get("cofactors", []) or []:
            name = cofactor.get("name")
            if name:
                names.append(str(name))
    for feature in uniprot_json.get("features", []) or []:
        ligand = feature.get("ligand", {}) or {}
        name = ligand.get("name")
        if name:
            names.append(str(name))
    return extract_annotated_metal_symbols_from_names(names)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() in {"", ".", "?"}:
            return None
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    numeric = as_float(value)
    if numeric is None:
        return None
    try:
        return int(numeric)
    except (TypeError, ValueError):
        return None


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def require_columns(fieldnames: Sequence[str] | None, required: Iterable[str], csv_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"Could not read CSV header from {csv_path}")
    missing = sorted(set(required) - set(fieldnames))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")


def read_tsv_or_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read CSV header from {path}")
        return list(reader.fieldnames), list(reader)


def resolve_clean_columns(fieldnames: Sequence[str], path: Path) -> tuple[str, str, str]:
    normalized = {field.strip().lower(): field for field in fieldnames}
    id_col = normalized.get("entry") or normalized.get("id")
    ec_col = normalized.get("ec number") or normalized.get("ec")
    sequence_col = normalized.get("sequence") or normalized.get("sequences")
    missing = []
    if id_col is None:
        missing.append("Entry/ID")
    if ec_col is None:
        missing.append("EC number/EC")
    if sequence_col is None:
        missing.append("Sequence/Sequences")
    if missing:
        raise ValueError(f"Missing CLEAN columns in {path}: {missing}")
    return id_col, ec_col, sequence_col


def read_clean_split_file(path: Path, *, clean_identity: str, clean_fold: int, split: str) -> list[CleanRow]:
    fieldnames, raw_rows = read_tsv_or_csv(path)
    id_col, ec_col, sequence_col = resolve_clean_columns(fieldnames, path)
    rows: list[CleanRow] = []
    seen: set[str] = set()
    for row_index, row in enumerate(raw_rows, start=2):
        accession = str(row.get(id_col, "")).strip()
        ecnumber = normalize_ec_number_list(row.get(ec_col, ""))
        sequence = str(row.get(sequence_col, "")).strip().replace(" ", "")
        if not accession or not ecnumber or not sequence:
            continue
        if accession in seen:
            raise ValueError(f"Duplicate UniProt ID {accession!r} in {path}")
        seen.add(accession)
        rows.append(
            CleanRow(
                clean_identity=clean_identity,
                clean_fold=clean_fold,
                split=split,
                uniprot_id=accession,
                ecnumber=ecnumber,
                sequence=sequence,
                source_file=str(path),
                source_row=row_index,
            )
        )
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read CSV header from {path}")
        return list(reader)


def clean_row_to_dict(row: CleanRow) -> dict[str, Any]:
    return {
        "clean_identity": row.clean_identity,
        "clean_fold": row.clean_fold,
        "split": row.split,
        "uniprot_id": row.uniprot_id,
        "ecnumber": row.ecnumber,
        "sequence": row.sequence,
        "source_file": row.source_file,
        "source_row": row.source_row,
    }


def manifest_paths(work_root: Path, *, clean_identity: str, clean_fold: int) -> dict[str, Path]:
    manifest_dir = work_root / "manifests"
    return {
        "train": manifest_dir / f"clean_split{clean_identity}_fold{clean_fold}_train.csv",
        "test": manifest_dir / f"clean_split{clean_identity}_fold{clean_fold}_test.csv",
    }


def load_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    require_columns(MANIFEST_FIELDS, MANIFEST_FIELDS, path)
    return rows


def command_audit_split(args: argparse.Namespace) -> None:
    clean_root = resolve_path(args.clean_splits_root)
    work_root = resolve_path(args.work_root)
    split_dir = clean_root / f"split{args.identity}"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"CLEAN split directory not found: {split_dir}")

    audit_rows: list[dict[str, Any]] = []
    for fold in args.fold:
        train_path = split_dir / f"split{args.identity}_train_split_{fold}.csv"
        test_path = split_dir / f"split{args.identity}_test_split_{fold}_curate.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing CLEAN train split file: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing CLEAN test split file: {test_path}")

        train_rows = read_clean_split_file(
            train_path,
            clean_identity=str(args.identity),
            clean_fold=fold,
            split="train",
        )
        test_rows = read_clean_split_file(
            test_path,
            clean_identity=str(args.identity),
            clean_fold=fold,
            split="test",
        )
        train_ids = {row.uniprot_id for row in train_rows}
        test_ids = {row.uniprot_id for row in test_rows}
        overlap = sorted(train_ids & test_ids)
        if overlap:
            preview = ", ".join(overlap[:10])
            raise ValueError(f"Fold {fold} train/test UniProt overlap: {preview}")

        paths = manifest_paths(work_root, clean_identity=str(args.identity), clean_fold=fold)
        write_csv(paths["train"], MANIFEST_FIELDS, (clean_row_to_dict(row) for row in train_rows))
        write_csv(paths["test"], MANIFEST_FIELDS, (clean_row_to_dict(row) for row in test_rows))

        audit_rows.append(
            {
                "clean_identity": args.identity,
                "clean_fold": fold,
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "train_unique_uniprot": len(train_ids),
                "test_unique_uniprot": len(test_ids),
                "train_test_overlap": len(overlap),
                "train_manifest": str(paths["train"]),
                "test_manifest": str(paths["test"]),
            }
        )
        print(
            f"[OK] split{args.identity} fold {fold}: "
            f"train={len(train_rows)} test={len(test_rows)} overlap=0"
        )

    audit_path = work_root / "manifests" / f"clean_split{args.identity}_audit.csv"
    write_csv(audit_path, audit_rows[0].keys() if audit_rows else [], audit_rows)
    print(f"Wrote audit: {audit_path}")


def download_url(url: str, out_path: Path, *, timeout: int, retries: int, overwrite: bool) -> str:
    if out_path.exists() and not overwrite:
        return "SKIPPED"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read()
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_bytes(payload)
            tmp_path.replace(out_path)
            return "OK"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return "HTTP_404"
            last_error = f"HTTP_{exc.code}"
        except Exception as exc:  # noqa: BLE001 - command-line tool records network failures.
            last_error = type(exc).__name__
        if attempt <= retries:
            time.sleep(min(2.0 * attempt, 10.0))
    return f"FAILED:{last_error or 'unknown'}"


def cached_json_request(url: str, cache_path: Path, *, timeout: int, retries: int) -> Any | None:
    if not cache_path.exists():
        status = download_url(url, cache_path, timeout=timeout, retries=retries, overwrite=False)
        if status not in {"OK", "SKIPPED"}:
            return None
    try:
        return load_json(cache_path)
    except Exception:
        return None


def donor_pdb_resolution(
    pdb_id: str,
    *,
    work_root: Path,
    timeout: int,
    retries: int,
) -> float | None:
    cleaned = str(pdb_id).strip().upper()
    if not cleaned:
        return None
    cache_path = work_root / "rcsb" / "entry" / f"{cleaned}.json"
    data = cached_json_request(
        RCSB_ENTRY_URL.format(pdb_id=cleaned),
        cache_path,
        timeout=timeout,
        retries=retries,
    )
    if not isinstance(data, Mapping):
        return None
    resolutions = data.get("rcsb_entry_info", {}).get("resolution_combined") or []
    parsed = [value for value in (as_float(item) for item in resolutions) if value is not None]
    return min(parsed) if parsed else None


def iter_selected_manifest_rows(
    work_root: Path,
    *,
    clean_identity: str,
    fold: int,
    splits: Sequence[str],
    limit_per_split: int | None,
) -> Iterable[tuple[str, dict[str, str]]]:
    paths = manifest_paths(work_root, clean_identity=clean_identity, clean_fold=fold)
    for split in splits:
        manifest = paths[split]
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}. Run audit-split first.")
        rows = load_manifest(manifest)
        if limit_per_split is not None:
            rows = rows[:limit_per_split]
        for row in rows:
            yield split, row


def command_fetch_alphafill(args: argparse.Namespace) -> None:
    work_root = resolve_path(args.work_root)
    if args.prefilter_uniprot_supported_metals and args.skip_uniprot:
        raise ValueError("--prefilter-uniprot-supported-metals requires UniProt fetching; remove --skip-uniprot.")
    summary_rows: list[dict[str, Any]] = []
    for split, row in iter_selected_manifest_rows(
        work_root,
        clean_identity=str(args.identity),
        fold=args.fold,
        splits=args.splits,
        limit_per_split=args.limit_per_split,
    ):
        accession = row["uniprot_id"].strip()
        json_path = work_root / "alphafill" / split / "json" / f"{accession}.json"
        cif_path = work_root / "alphafill" / split / "cif" / f"{accession}.cif"
        uniprot_path = work_root / "uniprot" / split / f"{accession}.json"
        uniprot_status = "NOT_REQUESTED"
        uniprot_metals: list[str] = []
        if not args.skip_uniprot:
            uniprot_status = download_url(
                UNIPROT_JSON_URL.format(accession=accession),
                uniprot_path,
                timeout=args.timeout,
                retries=args.retries,
                overwrite=args.overwrite,
            )
            if uniprot_status in {"OK", "SKIPPED"} and uniprot_path.exists():
                try:
                    uniprot_metals = extract_uniprot_supported_transition_metals(load_json(uniprot_path))
                except Exception:
                    uniprot_metals = []
        if args.prefilter_uniprot_supported_metals and not uniprot_metals:
            json_status = "SKIPPED_NO_UNIPROT_SUPPORTED_TRANSITION_METAL"
            cif_status = "NOT_REQUESTED"
        else:
            json_status = download_url(
                ALPHAFILL_JSON_URL.format(accession=accession),
                json_path,
                timeout=args.timeout,
                retries=args.retries,
                overwrite=args.overwrite,
            )
            cif_status = "NOT_REQUESTED"
            if json_status in {"OK", "SKIPPED"}:
                cif_status = download_url(
                    ALPHAFILL_ENTRY_URL.format(accession=accession),
                    cif_path,
                    timeout=args.timeout,
                    retries=args.retries,
                    overwrite=args.overwrite,
                )
        summary_rows.append(
            {
                "split": split,
                "uniprot_id": accession,
                "json_status": json_status,
                "json_path": str(json_path) if json_path.exists() else "",
                "cif_status": cif_status,
                "cif_path": str(cif_path) if cif_path.exists() else "",
                "uniprot_status": uniprot_status,
                "uniprot_path": str(uniprot_path) if uniprot_path.exists() else "",
                "uniprot_supported_transition_metals": ";".join(uniprot_metals),
            }
        )
        print(
            f"[{split}] {accession}: json={json_status} cif={cif_status} "
            f"uniprot={uniprot_status} metals={';'.join(uniprot_metals) or '-'}"
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary_path = work_root / "alphafill" / "fetch_summary.csv"
    write_csv(
        summary_path,
        [
            "split",
            "uniprot_id",
            "json_status",
            "json_path",
            "cif_status",
            "cif_path",
            "uniprot_status",
            "uniprot_path",
            "uniprot_supported_transition_metals",
        ],
        summary_rows,
    )
    print(f"Wrote fetch summary: {summary_path}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_model(structure: Any) -> Any:
    return next(structure.get_models())


def parse_mmcif(path: Path, structure_id: str) -> Any:
    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    return parser.get_structure(structure_id, str(path))


def residue_is_polymer_atom_record(residue: Any) -> bool:
    hetflag = residue.id[0]
    return str(hetflag).strip() == ""


def get_atom_element(atom: Any) -> str:
    element = getattr(atom, "element", "") or ""
    if str(element).strip():
        return str(element).strip().upper()
    name = atom.get_name().strip()
    return "".join(ch for ch in name if ch.isalpha())[:2].upper()


def metal_coord_from_structure(structure: Any, candidate: Mapping[str, Any]) -> tuple[float, float, float] | None:
    model = first_model(structure)
    asym_id = str(candidate.get("alphafill_asym_id", "")).strip()
    metal = str(candidate.get("metaltype", "")).strip().upper()
    chains_to_try = [asym_id] if asym_id else []
    chains_to_try.extend(chain.id for chain in model if chain.id not in chains_to_try)
    for chain_id in chains_to_try:
        if not chain_id or chain_id not in model:
            continue
        chain = model[chain_id]
        for residue in chain:
            residue_metal = canonicalize_metal(residue.resname)
            if residue_metal != metal:
                continue
            for atom in residue.get_atoms():
                atom_metal = canonicalize_metal(get_atom_element(atom)) or residue_metal
                if atom_metal != metal:
                    continue
                x, y, z = map(float, atom.coord)
                return (x, y, z)
    return None


def passes_optional_max(value: float | None, max_value: float | None) -> bool:
    return max_value is None or value is None or value <= max_value


def extract_candidate_dicts(json_data: Mapping[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit_index, hit in enumerate(json_data.get("hits", []) or []):
        alignment = hit.get("alignment", {}) or {}
        identity = as_float(alignment.get("identity"))
        if identity is None or identity < args.min_alphafill_identity:
            continue
        alignment_length = as_int(alignment.get("length"))
        if args.min_alignment_length is not None and (alignment_length is None or alignment_length < args.min_alignment_length):
            continue
        for transplant_index, transplant in enumerate(hit.get("transplants", []) or []):
            metal = canonicalize_metal(transplant.get("compound_id") or transplant.get("analogue_id") or "")
            if metal is None:
                continue
            validation = transplant.get("validation", {}) or {}
            pae = transplant.get("pae", {}) or {}
            local_rmsd = as_float(transplant.get("local_rmsd"))
            binding_site_rmsd = as_float(validation.get("binding_site_rmsd"))
            local_environment_rmsd = as_float(validation.get("local_environment_rmsd"))
            pae_mean = as_float(pae.get("mean"))
            binding_site_atom_count = as_int(validation.get("binding_site_atom_count"))
            transplant_atom_count = as_int(validation.get("transplant_atom_count"))
            if not passes_optional_max(local_rmsd, args.max_local_rmsd):
                continue
            if not passes_optional_max(binding_site_rmsd, args.max_binding_site_rmsd):
                continue
            if not passes_optional_max(local_environment_rmsd, args.max_local_environment_rmsd):
                continue
            if not passes_optional_max(pae_mean, args.max_pae_mean):
                continue
            if args.min_binding_site_atom_count is not None and (
                binding_site_atom_count is None or binding_site_atom_count < args.min_binding_site_atom_count
            ):
                continue
            rows.append(
                {
                    "metaltype": metal,
                    "alphafill_identity": identity,
                    "alphafill_alignment_length": alignment_length,
                    "alphafill_pdb_id": str(hit.get("pdb_id", "")).strip(),
                    "alphafill_pdb_asym_id": str(hit.get("pdb_asym_id", "")).strip(),
                    "alphafill_compound_id": str(transplant.get("compound_id", "")).strip(),
                    "alphafill_analogue_id": str(transplant.get("analogue_id", "")).strip(),
                    "alphafill_asym_id": str(transplant.get("asym_id", "")).strip(),
                    "alphafill_local_rmsd": local_rmsd,
                    "alphafill_binding_site_rmsd": binding_site_rmsd,
                    "alphafill_local_environment_rmsd": local_environment_rmsd,
                    "alphafill_pae_mean": pae_mean,
                    "alphafill_binding_site_atom_count": binding_site_atom_count,
                    "alphafill_transplant_atom_count": transplant_atom_count,
                    "hit_index": hit_index,
                    "transplant_index": transplant_index,
                }
            )
    return rows


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def quality_key(candidate: MetalCandidate) -> tuple[float, float, float, float, float, int]:
    return (
        candidate.alphafill_pdb_resolution if candidate.alphafill_pdb_resolution is not None else float("inf"),
        -candidate.alphafill_identity,
        candidate.alphafill_local_environment_rmsd if candidate.alphafill_local_environment_rmsd is not None else float("inf"),
        candidate.alphafill_local_rmsd if candidate.alphafill_local_rmsd is not None else float("inf"),
        candidate.alphafill_binding_site_rmsd if candidate.alphafill_binding_site_rmsd is not None else float("inf"),
        -(candidate.alphafill_binding_site_atom_count or 0),
    )


def select_biological_candidates(
    candidates: Sequence[MetalCandidate],
    *,
    dedup_distance: float,
    annotated_metals: set[str],
    uniprot_metal_policy: str,
) -> tuple[list[MetalCandidate], int]:
    clusters: list[list[MetalCandidate]] = []
    for candidate in candidates:
        assigned = False
        for cluster in clusters:
            if any(distance(candidate.coord, other.coord) < dedup_distance for other in cluster):
                cluster.append(candidate)
                assigned = True
                break
        if not assigned:
            clusters.append([candidate])

    selected: list[MetalCandidate] = []
    filtered_by_uniprot = 0
    for cluster in clusters:
        metals = tuple(sorted({candidate.metaltype for candidate in cluster}))
        uniprot_supported = [candidate for candidate in cluster if candidate.metaltype in annotated_metals]
        if uniprot_metal_policy == "require_supported":
            if not uniprot_supported:
                filtered_by_uniprot += len(cluster)
                continue
            pool = uniprot_supported
            reason = "uniprot_annotation_then_resolution"
        elif uniprot_metal_policy == "prefer_supported":
            if uniprot_supported:
                pool = uniprot_supported
                reason = "uniprot_annotation_then_resolution"
            else:
                pool = list(cluster)
                reason = "resolution_fallback_no_uniprot_match"
        elif uniprot_metal_policy == "ignore":
            pool = list(cluster)
            reason = "resolution_only"
        else:
            raise ValueError(f"Unsupported UniProt metal policy: {uniprot_metal_policy}")

        filtered_by_uniprot += len(cluster) - len(pool)
        best = sorted(pool, key=quality_key)[0]
        selected.append(
            MetalCandidate(
                **{
                    **best.__dict__,
                    "selected_by_uniprot_annotation": best.metaltype in annotated_metals,
                    "selection_reason": reason,
                    "cluster_supported_metals": metals,
                    "cluster_size": len(cluster),
                }
            )
        )
    return selected, filtered_by_uniprot


def candidate_from_dict(structure: Any, row: Mapping[str, Any]) -> MetalCandidate | None:
    coord = metal_coord_from_structure(structure, row)
    if coord is None:
        return None
    return MetalCandidate(
        metaltype=str(row["metaltype"]),
        coord=coord,
        alphafill_identity=float(row["alphafill_identity"]),
        alphafill_alignment_length=row.get("alphafill_alignment_length"),
        alphafill_pdb_id=str(row.get("alphafill_pdb_id", "")),
        alphafill_pdb_resolution=row.get("alphafill_pdb_resolution"),
        alphafill_pdb_asym_id=str(row.get("alphafill_pdb_asym_id", "")),
        alphafill_compound_id=str(row.get("alphafill_compound_id", "")),
        alphafill_analogue_id=str(row.get("alphafill_analogue_id", "")),
        alphafill_asym_id=str(row.get("alphafill_asym_id", "")),
        alphafill_local_rmsd=row.get("alphafill_local_rmsd"),
        alphafill_binding_site_rmsd=row.get("alphafill_binding_site_rmsd"),
        alphafill_local_environment_rmsd=row.get("alphafill_local_environment_rmsd"),
        alphafill_pae_mean=row.get("alphafill_pae_mean"),
        alphafill_binding_site_atom_count=row.get("alphafill_binding_site_atom_count"),
        alphafill_transplant_atom_count=row.get("alphafill_transplant_atom_count"),
    )


def format_pdb_atom_line(
    record_name: str,
    serial: int,
    atom_name: str,
    altloc: str,
    resname: str,
    chain_id: str,
    resseq: int,
    icode: str,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    bfactor: float,
    element: str,
    charge: str = "",
) -> str:
    atom_name = atom_name[:4]
    altloc = (altloc or " ")[:1]
    resname = (resname or "")[:3].rjust(3)
    chain_id = (chain_id or " ")[:1]
    icode = (icode or " ")[:1]
    element = (element or "").strip().upper()[:2].rjust(2)
    charge = (charge or "")[:2].rjust(2)
    if len(atom_name.strip()) < 4 and not atom_name[:1].isdigit():
        atom_field = f" {atom_name.strip():<3}"
    else:
        atom_field = f"{atom_name:<4}"
    return (
        f"{record_name:<6}"
        f"{serial:>5} "
        f"{atom_field}"
        f"{altloc}"
        f"{resname} "
        f"{chain_id}"
        f"{resseq:>4}"
        f"{icode}   "
        f"{x:>8.3f}"
        f"{y:>8.3f}"
        f"{z:>8.3f}"
        f"{occupancy:>6.2f}"
        f"{bfactor:>6.2f}"
        f"          "
        f"{element}"
        f"{charge}"
        "\n"
    )


def write_reduced_pdb(
    *,
    structure: Any,
    out_path: Path,
    uniprot_id: str,
    ecnumber: str,
    protein_chain_id: str,
    selected_metals: Sequence[MetalCandidate],
    metal_resseq_start: int,
) -> list[tuple[MetalCandidate, str]]:
    model = first_model(structure)
    if protein_chain_id not in model:
        raise ValueError(f"Protein chain {protein_chain_id!r} not found in AlphaFill structure for {uniprot_id}")
    chain = model[protein_chain_id]
    assigned_sites: list[tuple[MetalCandidate, str]] = []
    serial = 1
    max_resseq = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("HEADER    CLEAN ALPHAFILL REDUCED STRUCTURE\n")
        handle.write(f"COMPND   1 MOL_ID: 1;\n")
        handle.write(f"COMPND   2 MOLECULE: CLEAN_UNIPROT_{uniprot_id};\n")
        handle.write(f"COMPND   3 CHAIN: {protein_chain_id};\n")
        handle.write(f"COMPND   4 EC: {ecnumber};\n")
        handle.write(f"COMPND   5 UNIPROT: {uniprot_id};\n")
        handle.write("REMARK Source: AlphaFill REST entry filtered for supported transition metals\n")
        for residue in chain:
            if not residue_is_polymer_atom_record(residue):
                continue
            _hetflag, resseq, icode = residue.id
            try:
                resseq_int = int(resseq)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Non-integer residue number {resseq!r} in {uniprot_id}") from exc
            max_resseq = max(max_resseq, resseq_int)
            if resseq_int > 9999:
                raise ValueError(f"Residue number {resseq_int} exceeds PDB limit in {uniprot_id}")
            for atom in residue.get_atoms():
                if serial > 99999:
                    raise ValueError(f"Atom serial exceeds PDB limit in {uniprot_id}")
                x, y, z = map(float, atom.coord)
                occupancy = atom.occupancy if atom.occupancy is not None else 1.0
                bfactor = atom.bfactor if atom.bfactor is not None else 0.0
                atom_name = atom.fullname.strip() if getattr(atom, "fullname", None) else atom.get_name().strip()
                element = get_atom_element(atom)
                altloc = atom.get_altloc() if hasattr(atom, "get_altloc") else " "
                handle.write(
                    format_pdb_atom_line(
                        record_name="ATOM",
                        serial=serial,
                        atom_name=atom_name,
                        altloc=altloc if altloc not in {"", "?"} else " ",
                        resname=residue.resname.strip(),
                        chain_id=protein_chain_id,
                        resseq=resseq_int,
                        icode=str(icode).strip() if str(icode).strip() else " ",
                        x=x,
                        y=y,
                        z=z,
                        occupancy=float(occupancy),
                        bfactor=float(bfactor),
                        element=element,
                    )
                )
                serial += 1

        next_metal_resseq = max(metal_resseq_start, max_resseq + 1)
        if next_metal_resseq + len(selected_metals) - 1 > 9999:
            raise ValueError(f"Metal residue numbers would exceed PDB limit in {uniprot_id}")
        for candidate in selected_metals:
            if serial > 99999:
                raise ValueError(f"Atom serial exceeds PDB limit in {uniprot_id}")
            resseq = next_metal_resseq
            next_metal_resseq += 1
            x, y, z = candidate.coord
            metal = candidate.metaltype.upper()
            handle.write(
                format_pdb_atom_line(
                    record_name="HETATM",
                    serial=serial,
                    atom_name=metal,
                    altloc=" ",
                    resname=metal,
                    chain_id=protein_chain_id,
                    resseq=resseq,
                    icode=" ",
                    x=x,
                    y=y,
                    z=z,
                    occupancy=1.0,
                    bfactor=0.0,
                    element=metal,
                )
            )
            serial += 1
            assigned_sites.append((candidate, f"{protein_chain_id}_{resseq}"))
        handle.write("END\n")
    return assigned_sites


def candidate_to_summary_row(
    *,
    manifest_row: Mapping[str, str],
    candidate: MetalCandidate,
    chain_resi: str,
    output_pdb: Path,
) -> dict[str, Any]:
    return {
        "structure": manifest_row["uniprot_id"],
        "chain_resi": chain_resi,
        "metaltype": candidate.metaltype,
        "ecnumber": manifest_row["ecnumber"],
        "whether_catalytic": 0,
        "uniprot_id": manifest_row["uniprot_id"],
        "clean_identity": manifest_row["clean_identity"],
        "clean_fold": manifest_row["clean_fold"],
        "clean_split": manifest_row["split"],
        "sequence_length": len(manifest_row["sequence"]),
        "output_pdb": str(output_pdb),
        "alphafill_identity": format_optional(candidate.alphafill_identity),
        "alphafill_alignment_length": format_optional(candidate.alphafill_alignment_length),
        "alphafill_pdb_id": candidate.alphafill_pdb_id,
        "alphafill_pdb_resolution": format_optional(candidate.alphafill_pdb_resolution),
        "alphafill_pdb_asym_id": candidate.alphafill_pdb_asym_id,
        "alphafill_compound_id": candidate.alphafill_compound_id,
        "alphafill_analogue_id": candidate.alphafill_analogue_id,
        "alphafill_asym_id": candidate.alphafill_asym_id,
        "alphafill_local_rmsd": format_optional(candidate.alphafill_local_rmsd),
        "alphafill_binding_site_rmsd": format_optional(candidate.alphafill_binding_site_rmsd),
        "alphafill_local_environment_rmsd": format_optional(candidate.alphafill_local_environment_rmsd),
        "alphafill_pae_mean": format_optional(candidate.alphafill_pae_mean),
        "alphafill_binding_site_atom_count": format_optional(candidate.alphafill_binding_site_atom_count),
        "alphafill_transplant_atom_count": format_optional(candidate.alphafill_transplant_atom_count),
        "uniprot_supported_transition_metals": manifest_row.get("uniprot_supported_transition_metals", ""),
        "selected_by_uniprot_annotation": int(candidate.selected_by_uniprot_annotation),
        "selection_reason": candidate.selection_reason,
        "cluster_supported_metals": ";".join(candidate.cluster_supported_metals),
        "cluster_size": candidate.cluster_size,
    }


def command_build_mahomes_inputs(args: argparse.Namespace) -> None:
    work_root = resolve_path(args.work_root)
    all_build_rows: list[dict[str, Any]] = []
    for split in args.splits:
        candidate_rows: list[dict[str, Any]] = []
        build_summary_rows: list[dict[str, Any]] = []
        for _, manifest_row in iter_selected_manifest_rows(
            work_root,
            clean_identity=str(args.identity),
            fold=args.fold,
            splits=[split],
            limit_per_split=args.limit_per_split,
        ):
            accession = manifest_row["uniprot_id"]
            json_path = work_root / "alphafill" / split / "json" / f"{accession}.json"
            cif_path = work_root / "alphafill" / split / "cif" / f"{accession}.cif"
            status = "OK"
            detail = ""
            selected_count = 0
            raw_supported_count = 0
            filtered_by_uniprot = 0
            annotated_metals: set[str] = set()
            output_pdb = ""
            try:
                if not json_path.exists() or not cif_path.exists():
                    raise FileNotFoundError("missing AlphaFill json/cif")
                json_data = load_json(json_path)
                raw_candidate_dicts = extract_candidate_dicts(json_data, args)
                uniprot_path = work_root / "uniprot" / split / f"{accession}.json"
                if not uniprot_path.exists() and args.fetch_missing_uniprot:
                    download_url(
                        UNIPROT_JSON_URL.format(accession=accession),
                        uniprot_path,
                        timeout=args.metadata_timeout,
                        retries=args.metadata_retries,
                        overwrite=False,
                    )
                uniprot_json = load_json(uniprot_path) if uniprot_path.exists() else {}
                annotated_metals = set(extract_uniprot_supported_transition_metals(uniprot_json))
                for candidate_dict in raw_candidate_dicts:
                    candidate_dict["alphafill_pdb_resolution"] = donor_pdb_resolution(
                        str(candidate_dict.get("alphafill_pdb_id", "")),
                        work_root=work_root,
                        timeout=args.metadata_timeout,
                        retries=args.metadata_retries,
                    )
                structure = parse_mmcif(cif_path, accession)
                candidates = [candidate_from_dict(structure, row) for row in raw_candidate_dicts]
                candidates = [candidate for candidate in candidates if candidate is not None]
                raw_supported_count = len(candidates)
                selected, filtered_by_uniprot = select_biological_candidates(
                    candidates,
                    dedup_distance=args.site_dedup_distance,
                    annotated_metals=annotated_metals,
                    uniprot_metal_policy=args.uniprot_metal_policy,
                )
                if not selected:
                    status = (
                        "NO_UNIPROT_SUPPORTED_TRANSITION_METAL"
                        if args.uniprot_metal_policy == "require_supported" and not annotated_metals
                        else "NO_ACCEPTED_SUPPORTED_METALS"
                    )
                else:
                    safe_ec = sanitize_filename_fragment(manifest_row["ecnumber"])
                    out_name = f"{sanitize_filename_fragment(accession)}__chain_{args.protein_chain}__EC_{safe_ec}.pdb"
                    out_path = work_root / "mahomes_inputs" / split / out_name
                    assigned_sites = write_reduced_pdb(
                        structure=structure,
                        out_path=out_path,
                        uniprot_id=accession,
                        ecnumber=manifest_row["ecnumber"],
                        protein_chain_id=args.protein_chain,
                        selected_metals=selected,
                        metal_resseq_start=args.metal_resseq_start,
                    )
                    selected_count = len(assigned_sites)
                    output_pdb = str(out_path)
                    for candidate, chain_resi in assigned_sites:
                        summary_manifest_row = {
                            **manifest_row,
                            "uniprot_supported_transition_metals": ";".join(sorted(annotated_metals)),
                        }
                        candidate_rows.append(
                            candidate_to_summary_row(
                                manifest_row=summary_manifest_row,
                                candidate=candidate,
                                chain_resi=chain_resi,
                                output_pdb=out_path,
                            )
                        )
            except Exception as exc:  # noqa: BLE001 - records per-entry failure and continues.
                status = "FAILED"
                detail = str(exc)
            build_row = {
                "split": split,
                "uniprot_id": accession,
                "status": status,
                "raw_supported_candidate_count": raw_supported_count,
                "selected_site_count": selected_count,
                "filtered_by_uniprot_candidate_count": filtered_by_uniprot,
                "uniprot_supported_transition_metals": ";".join(sorted(annotated_metals)),
                "output_pdb": output_pdb,
                "detail": detail,
            }
            build_summary_rows.append(build_row)
            all_build_rows.append(build_row)
            print(
                f"[{split}] {accession}: {status}; "
                f"raw_supported={raw_supported_count} selected={selected_count}"
            )
        candidate_csv = work_root / "mahomes_inputs" / split / CANDIDATE_SITE_CSV_NAME
        write_csv(candidate_csv, CANDIDATE_FIELDS, candidate_rows)
        summary_csv = work_root / "mahomes_inputs" / split / "build_summary.csv"
        write_csv(summary_csv, build_summary_rows[0].keys() if build_summary_rows else [], build_summary_rows)
        print(f"Wrote candidates for {split}: {candidate_csv} ({len(candidate_rows)} rows)")
    all_summary = work_root / "mahomes_inputs" / "build_summary_all.csv"
    write_csv(all_summary, all_build_rows[0].keys() if all_build_rows else [], all_build_rows)
    print(f"Wrote build summary: {all_summary}")


def parse_prediction_label(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"catalytic", "1", "true", "yes", "y"}:
        return 1
    if normalized in {"non-catalytic", "non catalytic", "not catalytic", "0", "false", "no", "n"}:
        return 0
    return None


def parse_structure_identity(structure_id: str) -> tuple[str, str, str]:
    stem = str(structure_id).strip()
    if stem.lower().endswith(".pdb"):
        stem = stem[:-4]
    match = STRUCTURE_ID_RE.match(stem)
    if match is None:
        raise ValueError(f"Could not parse structure identity from {structure_id!r}")
    return (
        match.group("structure").strip(),
        match.group("chain").strip(),
        normalize_ec_number_list(match.group("ec")),
    )


def candidate_lookup_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row["structure"]).strip().lower(),
        normalize_ec_number_list(row["ecnumber"]),
        str(row["chain_resi"]).strip(),
        str(row["metaltype"]).strip().upper(),
    )


def load_candidate_lookup(candidate_csv: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    rows = read_csv(candidate_csv)
    require_columns(CANDIDATE_FIELDS, CANDIDATE_FIELDS, candidate_csv)
    return {candidate_lookup_key(row): row for row in rows}


def iter_mahomes_prediction_rows(job_root: Path) -> Iterable[tuple[Path, dict[str, str]]]:
    for pred_path in sorted(job_root.glob("job_*/predictions.csv")):
        with pred_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require_columns(reader.fieldnames, PREDICTION_REQUIRED_COLUMNS, pred_path)
            for row in reader:
                yield pred_path, row


def final_row_from_candidate(candidate: Mapping[str, str], catalytic_value: int) -> dict[str, Any]:
    return {
        "structure": candidate["structure"],
        "chain_resi": candidate["chain_resi"],
        "metaltype": candidate["metaltype"],
        "ecnumber": candidate["ecnumber"],
        "whether_catalytic": catalytic_value,
        "uniprot_id": candidate.get("uniprot_id", candidate["structure"]),
        "clean_identity": candidate.get("clean_identity", ""),
        "clean_fold": candidate.get("clean_fold", ""),
        "clean_split": candidate.get("clean_split", ""),
        "alphafill_identity": candidate.get("alphafill_identity", ""),
        "alphafill_alignment_length": candidate.get("alphafill_alignment_length", ""),
        "alphafill_pdb_id": candidate.get("alphafill_pdb_id", ""),
        "alphafill_pdb_resolution": candidate.get("alphafill_pdb_resolution", ""),
        "alphafill_compound_id": candidate.get("alphafill_compound_id", ""),
        "alphafill_local_rmsd": candidate.get("alphafill_local_rmsd", ""),
        "alphafill_binding_site_rmsd": candidate.get("alphafill_binding_site_rmsd", ""),
        "alphafill_local_environment_rmsd": candidate.get("alphafill_local_environment_rmsd", ""),
        "alphafill_pae_mean": candidate.get("alphafill_pae_mean", ""),
        "uniprot_supported_transition_metals": candidate.get("uniprot_supported_transition_metals", ""),
        "selected_by_uniprot_annotation": candidate.get("selected_by_uniprot_annotation", ""),
        "selection_reason": candidate.get("selection_reason", ""),
    }


def command_summarize_mahomes(args: argparse.Namespace) -> None:
    work_root = resolve_path(args.work_root)
    for split in args.splits:
        candidate_csv = args.candidate_csv or (work_root / "mahomes_inputs" / split / CANDIDATE_SITE_CSV_NAME)
        job_root = args.job_root or (work_root / "mahomes" / split)
        output_dir = args.output_dir or (work_root / "mahomes_outputs" / split)
        candidate_csv = resolve_path(Path(candidate_csv))
        job_root = resolve_path(Path(job_root))
        output_dir = resolve_path(Path(output_dir))
        if not candidate_csv.exists():
            raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv}")
        if not job_root.exists():
            raise FileNotFoundError(f"MAHOMES job root not found: {job_root}")
        lookup = load_candidate_lookup(candidate_csv)
        expanded_rows: list[dict[str, Any]] = []
        whether_rows_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        unmatched = 0
        prediction_count = 0
        for pred_path, row in iter_mahomes_prediction_rows(job_root):
            prediction_count += 1
            catalytic_value = parse_prediction_label(row.get("prediction", ""))
            if catalytic_value is None:
                continue
            try:
                structure, chain, ecnumber = parse_structure_identity(row["input file"])
            except ValueError:
                unmatched += 1
                continue
            for idx in range(1, 5):
                metal = canonicalize_metal(row.get(f"Name{idx}", ""))
                resseq = as_int(row.get(f"Res#{idx}", ""))
                if metal is None or resseq is None:
                    continue
                chain_resi = f"{chain}_{resseq}"
                key = (structure.lower(), normalize_ec_number_list(ecnumber), chain_resi, metal)
                candidate = lookup.get(key)
                if candidate is None:
                    unmatched += 1
                    continue
                final_row = final_row_from_candidate(candidate, catalytic_value)
                expanded_rows.append(
                    {
                        **final_row,
                        "job_root": str(job_root),
                        "prediction_csv": str(pred_path),
                        "input file": row.get("input file", ""),
                        "prediction": row.get("prediction", ""),
                        "percent catalytic predictions": row.get("percent catalytic predictions", ""),
                    }
                )
                existing = whether_rows_by_key.get(key)
                if existing is None or catalytic_value > int(existing["whether_catalytic"]):
                    whether_rows_by_key[key] = final_row
        if prediction_count == 0:
            raise ValueError(f"No MAHOMES predictions.csv files found under {job_root}")
        whether_rows = sorted(
            whether_rows_by_key.values(),
            key=lambda row: (str(row["structure"]), str(row["chain_resi"]), str(row["metaltype"]), str(row["ecnumber"])),
        )
        catalytic_rows = [row for row in whether_rows if int(row["whether_catalytic"]) == 1]
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / PREDICTION_RESULTS_CSV_NAME, list(expanded_rows[0].keys()) if expanded_rows else FINAL_SUMMARY_FIELDS, expanded_rows)
        write_csv(output_dir / WHETHER_CATALYTIC_CSV_NAME, FINAL_SUMMARY_FIELDS, whether_rows)
        write_csv(output_dir / SUMMARY_CSV_NAME, FINAL_SUMMARY_FIELDS, catalytic_rows)
        summary = {
            "split": split,
            "candidate_csv": str(candidate_csv),
            "job_root": str(job_root),
            "prediction_rows_seen": prediction_count,
            "expanded_site_rows": len(expanded_rows),
            "unique_site_rows": len(whether_rows),
            "catalytic_site_rows": len(catalytic_rows),
            "unmatched_prediction_site_rows": unmatched,
        }
        (output_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{split}] predictions={prediction_count} sites={len(whether_rows)} "
            f"catalytic={len(catalytic_rows)} unmatched={unmatched}"
        )
        print(f"Wrote MAHOMES summary: {output_dir / SUMMARY_CSV_NAME}")


def clear_output_split_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {path}. Use --overwrite to replace generated files.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def structure_stem_for_summary_row(row: Mapping[str, str]) -> str:
    return f"{sanitize_filename_fragment(row['structure'])}__chain_A__EC_{sanitize_filename_fragment(row['ecnumber'])}"


def collect_required_structure_names(summary_csv: Path) -> set[str]:
    rows = read_csv(summary_csv)
    require_columns(FINAL_SUMMARY_FIELDS, FINAL_SUMMARY_FIELDS, summary_csv)
    names = {structure_stem_for_summary_row(row) + ".pdb" for row in rows}
    return names


def command_export_dataset(args: argparse.Namespace) -> None:
    work_root = resolve_path(args.work_root)
    output_root = (
        resolve_path(args.output_root)
        if args.output_root is not None
        else PROJECT_ROOT / "DeepMzyme_Data" / f"CLEAN_{args.identity}_train_test_split_{args.fold}"
    )
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, Any] = {}
    for split in args.splits:
        summary_csv = work_root / "mahomes_outputs" / split / SUMMARY_CSV_NAME
        source_pdb_dir = work_root / "mahomes_inputs" / split
        if not summary_csv.exists():
            raise FileNotFoundError(f"Final catalytic summary CSV not found: {summary_csv}")
        if not source_pdb_dir.exists():
            raise FileNotFoundError(f"Source PDB dir not found: {source_pdb_dir}")
        dest_dir = output_root / split
        clear_output_split_dir(dest_dir, overwrite=args.overwrite)
        required_names = collect_required_structure_names(summary_csv)
        copied = 0
        missing: list[str] = []
        for name in sorted(required_names):
            source = source_pdb_dir / name
            if not source.exists():
                missing.append(name)
                continue
            shutil.copy2(source, dest_dir / name)
            copied += 1
        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(f"Missing {len(missing)} structure(s) for {split}: {preview}")
        shutil.copy2(summary_csv, dest_dir / SUMMARY_CSV_NAME)
        candidate_csv = source_pdb_dir / CANDIDATE_SITE_CSV_NAME
        metadata_dir = output_root / "metadata" / split
        metadata_dir.mkdir(parents=True, exist_ok=True)
        if candidate_csv.exists():
            shutil.copy2(candidate_csv, metadata_dir / CANDIDATE_SITE_CSV_NAME)
        stats_json = work_root / "mahomes_outputs" / split / "summary_stats.json"
        if stats_json.exists():
            shutil.copy2(stats_json, metadata_dir / "mahomes_summary_stats.json")
        split_stats[split] = {
            "summary_csv": str(dest_dir / SUMMARY_CSV_NAME),
            "structure_count": copied,
            "site_count": len(read_csv(summary_csv)),
        }
        print(f"[{split}] copied {copied} structures and {split_stats[split]['site_count']} site rows")

    metadata = {
        "split_name": f"CLEAN split{args.identity} fold{args.fold} AlphaFill-MAHOMES catalytic metalloenzyme subset",
        "clean_identity": str(args.identity),
        "clean_fold": int(args.fold),
        "source_work_root": str(work_root),
        "output_root": str(output_root),
        "note": (
            "Computational CLEAN-derived subset. AlphaFill provides transferred metal ions; "
            "MAHOMES filters candidate sites as catalytic. This is not experimental validation."
        ),
        "splits": split_stats,
    }
    (output_root / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    readme_lines = [
        "# CLEAN AlphaFill-MAHOMES Split",
        "",
        "This dataset is a CLEAN-derived, computationally filtered metalloenzyme subset.",
        "It is intentionally separate from all PinMyMetal split directories.",
        "",
        "AlphaFill supplies transferred metal ions on AlphaFold models; MAHOMES is used to keep catalytic candidate sites.",
        "Do not describe this as experimentally validated metalloenzymes without additional evidence.",
        "",
        "## Contents",
        "",
    ]
    for split, stats in split_stats.items():
        readme_lines.append(f"- `{split}/`: {stats['structure_count']} structures, {stats['site_count']} catalytic site rows")
    readme_lines.extend(["", "See `split_metadata.json` and `metadata/` for source evidence.", ""])
    (output_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Wrote dataset metadata: {output_root / 'split_metadata.json'}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--identity", type=str, default="30", help="CLEAN identity split, e.g. 30")
    parser.add_argument("--fold", type=int, default=0, help="CLEAN fold index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CLEAN AlphaFill-MAHOMES inputs for DeepMzyme.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-split", help="Normalize and audit CLEAN split files.")
    audit.add_argument("--clean-splits-root", type=Path, default=DEFAULT_CLEAN_SPLITS_ROOT)
    audit.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    audit.add_argument("--identity", type=str, default="30")
    audit.add_argument("--fold", type=int, action="append", default=None, help="Fold index; repeat for multiple folds.")
    audit.set_defaults(func=lambda args: (setattr(args, "fold", args.fold or [0]), command_audit_split(args))[1])

    fetch = subparsers.add_parser("fetch-alphafill", help="Download AlphaFill JSON and mmCIF files.")
    add_common_args(fetch)
    fetch.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    fetch.add_argument("--limit-per-split", type=int, default=None)
    fetch.add_argument("--timeout", type=int, default=45)
    fetch.add_argument("--retries", type=int, default=2)
    fetch.add_argument("--sleep-seconds", type=float, default=0.0)
    fetch.add_argument("--overwrite", action="store_true")
    fetch.add_argument("--skip-uniprot", action="store_true", help="Do not fetch UniProt JSON annotations.")
    fetch.add_argument(
        "--prefilter-uniprot-supported-metals",
        action="store_true",
        help="Fetch UniProt first and skip AlphaFill downloads unless UniProt annotates a supported transition metal.",
    )
    fetch.set_defaults(func=command_fetch_alphafill)

    build = subparsers.add_parser("build-mahomes-inputs", help="Build reduced PDBs and candidate-site CSVs.")
    add_common_args(build)
    build.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    build.add_argument("--limit-per-split", type=int, default=None)
    build.add_argument("--min-alphafill-identity", type=float, default=0.30)
    build.add_argument("--min-alignment-length", type=int, default=85)
    build.add_argument("--max-local-rmsd", type=float, default=None)
    build.add_argument("--max-binding-site-rmsd", type=float, default=None)
    build.add_argument("--max-local-environment-rmsd", type=float, default=None)
    build.add_argument("--max-pae-mean", type=float, default=None)
    build.add_argument("--min-binding-site-atom-count", type=int, default=None)
    build.add_argument(
        "--site-dedup-distance",
        type=float,
        default=1.0,
        help="Near-duplicate AlphaFill metal alternatives closer than this distance are collapsed to one site.",
    )
    build.add_argument(
        "--uniprot-metal-policy",
        choices=("require_supported", "prefer_supported", "ignore"),
        default="require_supported",
        help=(
            "require_supported keeps only UniProt-supported transition metals; "
            "prefer_supported falls back to donor resolution if no cluster candidate matches UniProt; "
            "ignore uses donor resolution only."
        ),
    )
    build.add_argument("--fetch-missing-uniprot", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--metadata-timeout", type=int, default=45)
    build.add_argument("--metadata-retries", type=int, default=2)
    build.add_argument("--protein-chain", default="A")
    build.add_argument("--metal-resseq-start", type=int, default=9001)
    build.set_defaults(func=command_build_mahomes_inputs)

    summarize = subparsers.add_parser("summarize-mahomes", help="Create catalytic-only summary CSVs from MAHOMES predictions.")
    summarize.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    summarize.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    summarize.add_argument("--candidate-csv", type=Path, default=None, help="Single split override; normally omit.")
    summarize.add_argument("--job-root", type=Path, default=None, help="Single split override; normally omit.")
    summarize.add_argument("--output-dir", type=Path, default=None, help="Single split override; normally omit.")
    summarize.set_defaults(func=command_summarize_mahomes)

    export = subparsers.add_parser("export-dataset", help="Copy catalytic structures/CSVs into DeepMzyme_Data.")
    export.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    export.add_argument("--identity", type=str, default="30")
    export.add_argument("--fold", type=int, default=0)
    export.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    export.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(func=command_export_dataset)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
