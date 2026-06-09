#!/usr/bin/env python3
"""Prepare a CARE Task 1 <30% identity metalloenzyme subset for DeepMzyme.

This pipeline is intentionally separate from the CLEAN and PinMyMetal
preparation workflows. CARE starts from protein/EC rows, UniProt IDs, and
sequences. AlphaFill/AlphaFold is used as the structure-plus-metal source, then
MAHOMES is used to keep candidate sites predicted as catalytic.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARE_ROOT = PROJECT_ROOT / "DeepMzyme_Data" / "CARE_dataset"
DEFAULT_WORK_ROOT = Path("/media/Data/care_sets/task1_30")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "DeepMzyme_Data" / "CARE_task1_30_train_test_metallo"
ALPHAFILL_ENTRY_URL = "https://alphafill.eu/v1/aff/{accession}"
ALPHAFILL_JSON_URL = "https://alphafill.eu/v1/aff/{accession}/json"
UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
UNIPROT_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
SUPPORTED_TRANSITION_METALS = frozenset({"MN", "FE", "CO", "NI", "CU", "ZN"})
MANIFEST_PREFIX = "care_task1_30"
SUMMARY_CSV_NAME = "final_data_summarazing_table_transition_metals_only_catalytic.csv"
WHETHER_CATALYTIC_CSV_NAME = "data_summarazing_table_transition_metals_whether_catalytic.csv"
PREDICTION_RESULTS_CSV_NAME = "prediction_results_summary.csv"
CANDIDATE_SITE_CSV_NAME = "candidate_site_summary.csv"
STRUCTURE_ID_RE = re.compile(r"^(?P<structure>[^_]+)__chain_(?P<chain>[^_]+)__EC_(?P<ec>.+)$")
COFACTOR_SYMBOL_PATTERNS = {
    "CO": (r"\bCO(?:\(\d\+\))?\b", r"\bCO CATION\b", r"COBALT"),
    "CU": (r"\bCU(?:\(\d\+\))?\b", r"\bCU CATION\b", r"COPPER"),
    "FE": (r"\bFE(?:\(\d\+\))?\b", r"\bFE CATION\b", r"IRON", r"\bFE[- ]?S\b", r"\b\dFE-\dS\b"),
    "MN": (r"\bMN(?:\(\d\+\))?\b", r"\bMN CATION\b", r"MANGANESE"),
    "NI": (r"\bNI(?:\(\d\+\))?\b", r"\bNI CATION\b", r"NICKEL"),
    "ZN": (r"\bZN(?:\(\d\+\))?\b", r"\bZN CATION\b", r"ZINC"),
}
UNIPROT_ANNOTATION_CACHE_NAME = "uniprot_annotation_cache.csv"
UNIPROT_ANNOTATION_CACHE_FIELDS = [
    "split",
    "uniprot_id",
    "status",
    "uniprot_supported_transition_metals",
    "cofactor",
    "binding_site",
]

ACCESSION_COLUMN_CANDIDATES = (
    "Entry",
    "ID",
    "UniProt",
    "UniProt ID",
    "uniprot_id",
    "accession",
    "protein_id",
    "protein",
)
EC_COLUMN_CANDIDATES = (
    "EC number",
    "EC",
    "ecnumber",
    "ec_number",
    "label",
)
SEQUENCE_COLUMN_CANDIDATES = (
    "Sequence",
    "Sequences",
    "protein_sequence",
    "amino_acid_sequence",
)

PAIR_MANIFEST_FIELDS = [
    "source_dataset",
    "source_task",
    "source_split_name",
    "source_split",
    "split",
    "source_record_id",
    "protein_id",
    "uniprot_id",
    "ecnumber",
    "sequence",
    "source_file",
    "source_row",
]

PROTEIN_MANIFEST_FIELDS = [
    *PAIR_MANIFEST_FIELDS,
    "source_record_ids",
    "n_source_rows",
]

CANDIDATE_FIELDS = [
    "structure",
    "chain_resi",
    "metaltype",
    "ecnumber",
    "whether_catalytic",
    "uniprot_id",
    "source_dataset",
    "source_task",
    "source_split_name",
    "source_split",
    "source_record_ids",
    "n_source_rows",
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
    "source_dataset",
    "source_task",
    "source_split_name",
    "source_split",
    "source_record_ids",
    "n_source_rows",
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


class CareColumnError(ValueError):
    """Raised when CARE input columns cannot support accession-based fetching."""


@dataclass(frozen=True)
class CarePairRow:
    source_dataset: str
    source_task: str
    source_split_name: str
    source_split: str
    split: str
    source_record_id: str
    protein_id: str
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


def normalize_ec_number_list(value: Any) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for ec in re.split(r"[;,|]", str(value or "")):
        normalized = ec.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return ";".join(values)


def ec_sort_key(value: str) -> tuple[Any, ...]:
    parts: list[Any] = []
    for part in str(value).split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part))
    return tuple(parts)


def merge_ec_numbers(values: Iterable[Any]) -> str:
    merged: set[str] = set()
    for value in values:
        for ec in normalize_ec_number_list(value).split(";"):
            if ec:
                merged.add(ec)
    return ";".join(sorted(merged, key=ec_sort_key))


def sanitize_filename_fragment(value: Any) -> str:
    sanitized = str(value or "").strip()
    sanitized = re.sub(r"\s+", "", sanitized)
    sanitized = sanitized.replace("/", "-")
    sanitized = sanitized.replace(";", ",")
    sanitized = re.sub(r"[&|<>$`'\"(){}\\]", "-", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "unknown"


def canonicalize_metal(value: Any) -> str | None:
    letters_only = "".join(ch for ch in str(value or "").strip().upper() if ch.isalpha())
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


def extract_uniprot_tsv_cofactor_names(cofactor_text: str) -> list[str]:
    """Extract structured UniProt cofactor Name= values, excluding free-text notes."""
    return [match.group(1).strip() for match in re.finditer(r"(?:^|;\s*|COFACTOR:\s*)Name=([^;]+)", cofactor_text or "")]


def extract_uniprot_tsv_binding_ligand_names(binding_site_text: str) -> list[str]:
    """Extract structured UniProt binding-site ligand names and ligand parts."""
    names: list[str] = []
    for match in re.finditer(r"/(?:ligand|ligand_part)=(?:\"([^\"]+)\"|([^;]+))", binding_site_text or ""):
        names.append((match.group(1) or match.group(2) or "").strip())
    return [name for name in names if name]


def extract_uniprot_supported_transition_metals_from_tsv_fields(
    *,
    cofactor_text: str,
    binding_site_text: str,
) -> list[str]:
    names = [
        *extract_uniprot_tsv_cofactor_names(cofactor_text),
        *extract_uniprot_tsv_binding_ligand_names(binding_site_text),
    ]
    return extract_annotated_metal_symbols_from_names(names)


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


def annotation_cache_path(work_root: Path) -> Path:
    return work_root / "uniprot" / UNIPROT_ANNOTATION_CACHE_NAME


def load_uniprot_annotation_cache(work_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = annotation_cache_path(work_root)
    if not path.exists():
        return {}
    rows = read_csv(path)
    cache: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        split = str(row.get("split", "")).strip()
        accession = str(row.get("uniprot_id", "")).strip()
        if not split or not accession:
            continue
        cache[(split, accession)] = row
    return cache


def cached_uniprot_supported_metals(
    cache: Mapping[tuple[str, str], Mapping[str, str]],
    *,
    split: str,
    accession: str,
) -> list[str] | None:
    row = cache.get((split, accession))
    if row is None:
        return None
    return [
        metal.strip().upper()
        for metal in str(row.get("uniprot_supported_transition_metals", "")).split(";")
        if metal.strip().upper() in SUPPORTED_TRANSITION_METALS
    ]


def query_uniprot_accession_batch(
    accessions: Sequence[str],
    *,
    timeout: int,
    retries: int,
) -> dict[str, dict[str, str]]:
    if not accessions:
        return {}
    query = urllib.parse.urlencode(
        {
            "accessions": ",".join(accessions),
            "fields": "accession,cc_cofactor,ft_binding",
            "format": "tsv",
        }
    )
    url = f"{UNIPROT_ACCESSIONS_URL}?{query}"
    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "DeepMzyme-CARE-prep/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
            reader = csv.DictReader(text.splitlines(), delimiter="\t")
            result: dict[str, dict[str, str]] = {}
            for row in reader:
                accession = str(row.get("Entry", "")).strip()
                if not accession:
                    continue
                result[accession] = {
                    "cofactor": row.get("Cofactor", "") or "",
                    "binding_site": row.get("Binding site", "") or "",
                }
            return result
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP_{exc.code}"
        except Exception as exc:  # noqa: BLE001 - command-line tool records network failures.
            last_error = type(exc).__name__
        if attempt <= retries:
            time.sleep(min(2.0 * attempt, 10.0))
    raise RuntimeError(f"UniProt batch annotation request failed for {len(accessions)} accessions: {last_error or 'unknown'}")


def unique_manifest_rows_for_splits(
    work_root: Path,
    *,
    splits: Sequence[str],
    limit_per_split: int | None,
) -> list[tuple[str, dict[str, str]]]:
    return list(
        iter_selected_manifest_rows(
            work_root,
            splits=splits,
            limit_per_split=limit_per_split,
        )
    )


def command_prefetch_uniprot_annotations(args: argparse.Namespace) -> None:
    work_root = resolve_path(args.work_root)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.checkpoint_batches <= 0:
        raise ValueError("--checkpoint-batches must be positive")
    selected_rows = unique_manifest_rows_for_splits(
        work_root,
        splits=args.splits,
        limit_per_split=args.limit_per_split,
    )
    existing = load_uniprot_annotation_cache(work_root)
    accessions = sorted({row["uniprot_id"].strip() for _, row in selected_rows if row.get("uniprot_id", "").strip()})
    annotation_by_accession: dict[str, dict[str, str]] = {}
    cache_path = annotation_cache_path(work_root)
    if existing and not args.overwrite:
        existing_by_accession: dict[str, Mapping[str, str]] = {}
        for (_, cached_accession), row in existing.items():
            existing_by_accession.setdefault(cached_accession, row)
        for accession in accessions:
            existing_row = existing_by_accession.get(accession)
            if existing_row is not None:
                annotation_by_accession[accession] = {
                    "cofactor": existing_row.get("cofactor", ""),
                    "binding_site": existing_row.get("binding_site", ""),
                    "status": existing_row.get("status", "CACHED"),
                }
    missing_accessions = [accession for accession in accessions if accession not in annotation_by_accession]
    print(
        f"[INFO] UniProt annotation cache target={cache_path} selected_accessions={len(accessions)} "
        f"cached={len(annotation_by_accession)} to_fetch={len(missing_accessions)}",
        flush=True,
    )

    def build_cache_rows(*, include_missing: bool) -> list[dict[str, str]]:
        rows_by_key: dict[tuple[str, str], dict[str, str]] = {
            (split, accession): {
                "split": str(row.get("split", "")),
                "uniprot_id": str(row.get("uniprot_id", "")),
                "status": str(row.get("status", "")),
                "uniprot_supported_transition_metals": str(row.get("uniprot_supported_transition_metals", "")),
                "cofactor": str(row.get("cofactor", "")),
                "binding_site": str(row.get("binding_site", "")),
            }
            for (split, accession), row in existing.items()
        }
        for split, row in selected_rows:
            accession = row["uniprot_id"].strip()
            payload = annotation_by_accession.get(accession)
            if payload is None:
                if not include_missing:
                    continue
                payload = {"status": "NOT_FOUND", "cofactor": "", "binding_site": ""}
            metals = extract_uniprot_supported_transition_metals_from_tsv_fields(
                cofactor_text=payload.get("cofactor", ""),
                binding_site_text=payload.get("binding_site", ""),
            )
            rows_by_key[(split, accession)] = {
                "split": split,
                "uniprot_id": accession,
                "status": payload.get("status", ""),
                "uniprot_supported_transition_metals": ";".join(metals),
                "cofactor": payload.get("cofactor", ""),
                "binding_site": payload.get("binding_site", ""),
            }
        return sorted(rows_by_key.values(), key=lambda row: (row["split"], row["uniprot_id"]))

    def write_annotation_cache(*, include_missing: bool, label: str) -> None:
        output_rows = build_cache_rows(include_missing=include_missing)
        write_csv(cache_path, UNIPROT_ANNOTATION_CACHE_FIELDS, output_rows)
        supported = sum(1 for row in output_rows if row["uniprot_supported_transition_metals"])
        print(
            f"[INFO] wrote {label} UniProt annotation cache rows={len(output_rows)} "
            f"supported_transition_metal_rows={supported}: {cache_path}",
            flush=True,
        )

    for start in range(0, len(missing_accessions), args.batch_size):
        batch = missing_accessions[start : start + args.batch_size]
        try:
            batch_result = query_uniprot_accession_batch(batch, timeout=args.timeout, retries=args.retries)
        except RuntimeError as exc:
            if args.allow_batch_failures:
                print(f"[WARN] {exc}")
                batch_result = {}
            else:
                raise
        for accession in batch:
            payload = batch_result.get(accession)
            if payload is None:
                annotation_by_accession[accession] = {
                    "cofactor": "",
                    "binding_site": "",
                    "status": "NOT_FOUND",
                }
            else:
                annotation_by_accession[accession] = {
                    "cofactor": payload.get("cofactor", ""),
                    "binding_site": payload.get("binding_site", ""),
                    "status": "OK",
                }
        print(
            f"[INFO] fetched UniProt annotation batch "
            f"{min(start + len(batch), len(missing_accessions))}/{len(missing_accessions)}",
            flush=True,
        )
        if ((start // args.batch_size) + 1) % args.checkpoint_batches == 0:
            write_annotation_cache(include_missing=False, label="checkpoint")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_annotation_cache(include_missing=True, label="final")


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
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read CSV header from {path}")
        return list(reader.fieldnames), list(reader)


def column_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    by_key = {column_key(field): field for field in fieldnames}
    for candidate in candidates:
        match = by_key.get(column_key(candidate))
        if match is not None:
            return match
    return None


def fieldname_context(path: Path) -> dict[str, Any]:
    try:
        fieldnames, rows = read_tsv_or_csv(path)
        return {"path": str(path), "fieldnames": fieldnames, "row_count": len(rows)}
    except Exception as exc:  # noqa: BLE001 - audit context should survive malformed files.
        return {"path": str(path), "error": str(exc)}


def care_no_candidate_error(care_root: Path) -> FileNotFoundError:
    return FileNotFoundError(
        "Could not find protein_train.csv / 30_protein_test.csv under DeepMzyme_Data/CARE_dataset. "
        "Check the unzip layout or set CARE_TRAIN_CSV and CARE_TEST_CSV explicitly. "
        f"Checked: {care_root}."
    )


def choose_unambiguous_care_file(care_root: Path, filename: str, candidates: Sequence[Path]) -> Path:
    if not candidates:
        raise care_no_candidate_error(care_root)
    unique = sorted({path.resolve() for path in candidates}, key=lambda path: str(path))
    preferred = [path for path in unique if "/splits/task1/" in path.as_posix().lower()]
    pool = preferred or unique
    if len(pool) == 1:
        return pool[0]
    lines = "\n".join(f"  - {path}" for path in pool)
    raise ValueError(
        f"Multiple ambiguous candidates found for {filename} under {care_root}:\n{lines}\n"
        "Set CARE_TRAIN_CSV and CARE_TEST_CSV explicitly."
    )


def resolve_care_task1_files(
    care_root: Path,
    train_csv: Path | None,
    test_csv: Path | None,
) -> tuple[Path, Path]:
    care_root = resolve_path(care_root)
    train_csv = resolve_path(train_csv) if train_csv is not None else None
    test_csv = resolve_path(test_csv) if test_csv is not None else None
    if (train_csv is None) != (test_csv is None):
        raise ValueError("Set both CARE_TRAIN_CSV and CARE_TEST_CSV, or leave both empty for discovery.")
    if train_csv is not None and test_csv is not None:
        missing = [str(path) for path in (train_csv, test_csv) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Explicit CARE CSV override(s) not found: {missing}")
        return train_csv.resolve(), test_csv.resolve()
    if not care_root.is_dir():
        raise FileNotFoundError(f"CARE root does not exist or is not a directory: {care_root}")
    train_candidates = list(care_root.rglob("protein_train.csv"))
    test_candidates = list(care_root.rglob("30_protein_test.csv"))
    if not train_candidates or not test_candidates:
        raise care_no_candidate_error(care_root)
    return (
        choose_unambiguous_care_file(care_root, "protein_train.csv", train_candidates),
        choose_unambiguous_care_file(care_root, "30_protein_test.csv", test_candidates),
    )


def source_record_id_for_row(row: Mapping[str, str], split: str, source_row: int) -> str:
    for candidate in ("source_record_id", "record_id", "id", ""):
        value = str(row.get(candidate, "")).strip()
        if value:
            return value
    return f"{split}:{source_row}"


def read_care_split_file(
    path: Path,
    *,
    source_task: str,
    source_split_name: str,
    split: str,
    representative_column: str | None = None,
) -> tuple[list[CarePairRow], dict[str, Any]]:
    fieldnames, raw_rows = read_tsv_or_csv(path)
    accession_col = resolve_column(fieldnames, ACCESSION_COLUMN_CANDIDATES)
    ec_col = resolve_column(fieldnames, EC_COLUMN_CANDIDATES)
    sequence_col = resolve_column(fieldnames, SEQUENCE_COLUMN_CANDIDATES)
    representative_col = resolve_column(fieldnames, (representative_column,)) if representative_column else None
    if accession_col is None:
        raise CareColumnError(
            "CARE file does not contain a UniProt/accession-like column required for AlphaFill accession-based fetching. "
            "Add a mapping step or provide CARE files with UniProt IDs."
        )
    if ec_col is None:
        raise ValueError(f"CARE file does not contain an EC/label-like column required for EC labeling: {path}")
    if representative_column and representative_col is None:
        raise ValueError(f"CARE file does not contain requested representative column {representative_column!r}: {path}")

    indexed_rows = list(enumerate(raw_rows, start=2))
    read_stats: dict[str, Any] = {
        f"{split}_source_rows_before_representative_filter": len(indexed_rows),
    }
    if representative_col is not None:
        rows_by_entry: dict[str, list[tuple[int, dict[str, str]]]] = {}
        representatives: set[str] = set()
        for row_index, row in indexed_rows:
            accession = str(row.get(accession_col, "")).strip()
            representative = str(row.get(representative_col, "")).strip()
            if accession:
                rows_by_entry.setdefault(accession, []).append((row_index, row))
            if representative:
                representatives.add(representative)
        missing_representatives = sorted(representatives - set(rows_by_entry))
        indexed_rows = [
            item
            for representative in sorted(representatives)
            for item in rows_by_entry.get(representative, [])
        ]
        read_stats.update(
            {
                f"{split}_representative_column": representative_col,
                f"{split}_unique_representatives": len(representatives),
                f"{split}_representatives_with_entry_rows": len(representatives) - len(missing_representatives),
                f"{split}_representatives_missing_entry_rows": len(missing_representatives),
                f"{split}_representatives_missing_entry_rows_preview": ";".join(missing_representatives[:25]),
                f"{split}_source_rows_after_representative_filter": len(indexed_rows),
            }
        )

    rows: list[CarePairRow] = []
    for row_index, row in indexed_rows:
        accession = str(row.get(accession_col, "")).strip()
        ecnumber = normalize_ec_number_list(row.get(ec_col, ""))
        sequence = str(row.get(sequence_col, "")).strip().replace(" ", "") if sequence_col else ""
        if not accession and not ecnumber and not sequence:
            continue
        rows.append(
            CarePairRow(
                source_dataset="CARE",
                source_task=source_task,
                source_split_name=source_split_name,
                source_split=split,
                split=split,
                source_record_id=source_record_id_for_row(row, split, row_index),
                protein_id=accession,
                uniprot_id=accession,
                ecnumber=ecnumber,
                sequence=sequence,
                source_file=str(path),
                source_row=row_index,
            )
        )
    read_stats[f"{split}_rows_read"] = len(rows)
    return rows, read_stats


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


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def care_pair_to_dict(row: CarePairRow) -> dict[str, Any]:
    return {
        "source_dataset": row.source_dataset,
        "source_task": row.source_task,
        "source_split_name": row.source_split_name,
        "source_split": row.source_split,
        "split": row.split,
        "source_record_id": row.source_record_id,
        "protein_id": row.protein_id,
        "uniprot_id": row.uniprot_id,
        "ecnumber": row.ecnumber,
        "sequence": row.sequence,
        "source_file": row.source_file,
        "source_row": row.source_row,
    }


def manifest_paths(work_root: Path) -> dict[str, Path]:
    manifest_dir = work_root / "manifests"
    return {
        "train_pairs": manifest_dir / f"{MANIFEST_PREFIX}_train_pairs.csv",
        "test_pairs": manifest_dir / f"{MANIFEST_PREFIX}_test_pairs.csv",
        "train_proteins": manifest_dir / f"{MANIFEST_PREFIX}_train_proteins.csv",
        "test_proteins": manifest_dir / f"{MANIFEST_PREFIX}_test_proteins.csv",
        "audit_csv": manifest_dir / f"{MANIFEST_PREFIX}_audit.csv",
        "audit_json": manifest_dir / f"{MANIFEST_PREFIX}_audit.json",
    }


def load_unique_protein_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    require_columns(PROTEIN_MANIFEST_FIELDS, PROTEIN_MANIFEST_FIELDS, path)
    return rows


def collapse_to_unique_proteins(rows: Sequence[CarePairRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CarePairRow]] = {}
    for row in rows:
        accession = row.uniprot_id.strip()
        if not accession:
            continue
        grouped.setdefault(accession, []).append(row)

    unique_rows: list[dict[str, Any]] = []
    for accession in sorted(grouped):
        group = sorted(grouped[accession], key=lambda item: (item.source_row, item.source_record_id))
        first = group[0]
        source_files = sorted({item.source_file for item in group})
        source_record_ids = [item.source_record_id for item in group]
        sequence = next((item.sequence for item in group if item.sequence), "")
        ecnumber = merge_ec_numbers(item.ecnumber for item in group)
        unique_rows.append(
            {
                "source_dataset": first.source_dataset,
                "source_task": first.source_task,
                "source_split_name": first.source_split_name,
                "source_split": first.source_split,
                "split": first.split,
                "source_record_id": first.source_record_id,
                "protein_id": first.protein_id,
                "uniprot_id": accession,
                "ecnumber": ecnumber,
                "sequence": sequence,
                "source_file": ";".join(source_files),
                "source_row": first.source_row,
                "source_record_ids": ";".join(source_record_ids),
                "n_source_rows": len(group),
            }
        )
    return unique_rows


def ec_set(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        for ec in normalize_ec_number_list(row.get("ecnumber", "")).split(";"):
            if ec:
                values.add(ec)
    return values


def audit_stats(
    *,
    train_pairs: Sequence[CarePairRow],
    test_pairs: Sequence[CarePairRow],
    train_proteins: Sequence[Mapping[str, Any]],
    test_proteins: Sequence[Mapping[str, Any]],
    train_csv: Path,
    test_csv: Path,
    read_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    train_pair_dicts = [care_pair_to_dict(row) for row in train_pairs]
    test_pair_dicts = [care_pair_to_dict(row) for row in test_pairs]
    train_ecs = ec_set(train_pair_dicts)
    test_ecs = ec_set(test_pair_dicts)
    stats = {
        "source_dataset": "CARE",
        "source_task": "task1",
        "source_split_name": "30_identity",
        "rows_train_pairs": len(train_pairs),
        "rows_test_pairs": len(test_pairs),
        "unique_train_proteins": len(train_proteins),
        "unique_test_proteins": len(test_proteins),
        "unique_train_ecs": len(train_ecs),
        "unique_test_ecs": len(test_ecs),
        "test_ecs_missing_from_train_ecs": ";".join(sorted(test_ecs - train_ecs, key=ec_sort_key)),
        "proteins_with_multiple_ecs_train": sum(1 for row in train_proteins if len(normalize_ec_number_list(row.get("ecnumber", "")).split(";")) > 1),
        "proteins_with_multiple_ecs_test": sum(1 for row in test_proteins if len(normalize_ec_number_list(row.get("ecnumber", "")).split(";")) > 1),
        "missing_uniprot_ids_train": sum(1 for row in train_pairs if not row.uniprot_id.strip()),
        "missing_uniprot_ids_test": sum(1 for row in test_pairs if not row.uniprot_id.strip()),
        "resolved_train_csv": str(train_csv),
        "resolved_test_csv": str(test_csv),
    }
    if read_stats:
        stats.update(read_stats)
    return stats


def write_audit_error(paths: Mapping[str, Path], payload: Mapping[str, Any]) -> None:
    audit_path = paths["audit_json"]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def command_audit_care_task1(args: argparse.Namespace) -> None:
    care_root = resolve_path(args.care_root)
    work_root = resolve_path(args.work_root)
    paths = manifest_paths(work_root)
    train_csv: Path | None = None
    test_csv: Path | None = None
    try:
        train_csv, test_csv = resolve_care_task1_files(care_root, args.train_csv, args.test_csv)
        train_pairs, train_read_stats = read_care_split_file(
            train_csv,
            source_task=args.care_task,
            source_split_name=args.care_split_name,
            split="train",
            representative_column=args.train_representative_column,
        )
        test_pairs, test_read_stats = read_care_split_file(
            test_csv,
            source_task=args.care_task,
            source_split_name=args.care_split_name,
            split="test",
        )
    except Exception as exc:
        context = {
            "source_dataset": "CARE",
            "care_root": str(care_root),
            "resolved_train_csv": str(train_csv or ""),
            "resolved_test_csv": str(test_csv or ""),
            "error": str(exc),
        }
        if train_csv is not None:
            context["train_context"] = fieldname_context(train_csv)
        if test_csv is not None:
            context["test_context"] = fieldname_context(test_csv)
        write_audit_error(paths, context)
        raise

    train_proteins = collapse_to_unique_proteins(train_pairs)
    test_proteins = collapse_to_unique_proteins(test_pairs)
    write_csv(paths["train_pairs"], PAIR_MANIFEST_FIELDS, (care_pair_to_dict(row) for row in train_pairs))
    write_csv(paths["test_pairs"], PAIR_MANIFEST_FIELDS, (care_pair_to_dict(row) for row in test_pairs))
    write_csv(paths["train_proteins"], PROTEIN_MANIFEST_FIELDS, train_proteins)
    write_csv(paths["test_proteins"], PROTEIN_MANIFEST_FIELDS, test_proteins)

    stats = audit_stats(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        train_proteins=train_proteins,
        test_proteins=test_proteins,
        train_csv=train_csv,
        test_csv=test_csv,
        read_stats={**train_read_stats, **test_read_stats},
    )
    stats["source_task"] = args.care_task
    stats["source_split_name"] = args.care_split_name
    stats["care_root"] = str(care_root)
    stats["work_root"] = str(work_root)
    write_csv(paths["audit_csv"], list(stats.keys()), [stats])
    paths["audit_json"].write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(
        f"[OK] CARE {args.care_task} {args.care_split_name}: "
        f"train_pairs={len(train_pairs)} test_pairs={len(test_pairs)} "
        f"train_proteins={len(train_proteins)} test_proteins={len(test_proteins)}"
    )
    print(f"Wrote manifests under: {work_root / 'manifests'}")


def download_url(url: str, out_path: Path, *, timeout: int, retries: int, overwrite: bool) -> str:
    if out_path.exists() and not overwrite:
        return "SKIPPED"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DeepMzyme-CARE-prep/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    splits: Sequence[str],
    limit_per_split: int | None,
) -> Iterable[tuple[str, dict[str, str]]]:
    paths = manifest_paths(work_root)
    split_to_path = {
        "train": paths["train_proteins"],
        "test": paths["test_proteins"],
    }
    for split in splits:
        manifest = split_to_path[split]
        if not manifest.exists():
            raise FileNotFoundError(f"Unique-protein manifest not found: {manifest}. Run audit-care-task1 first.")
        rows = load_unique_protein_manifest(manifest)
        if limit_per_split is not None:
            rows = rows[:limit_per_split]
        for row in rows:
            yield split, row


def fetch_alphafill_manifest_row(
    *,
    args: argparse.Namespace,
    work_root: Path,
    annotation_cache: Mapping[tuple[str, str], Mapping[str, str]],
    processed: int,
    split: str,
    row: Mapping[str, str],
) -> tuple[int, dict[str, Any], bool, str]:
    accession = row["uniprot_id"].strip()
    json_path = work_root / "alphafill" / split / "json" / f"{accession}.json"
    cif_path = work_root / "alphafill" / split / "cif" / f"{accession}.cif"
    uniprot_path = work_root / "uniprot" / split / f"{accession}.json"
    uniprot_status = "NOT_REQUESTED"
    cached_metals = cached_uniprot_supported_metals(annotation_cache, split=split, accession=accession)
    uniprot_metals: list[str] = cached_metals or []
    if cached_metals is not None:
        cached_row = annotation_cache.get((split, accession), {})
        uniprot_status = f"CACHED_{cached_row.get('status', 'OK')}"
    elif not args.skip_uniprot:
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
        json_candidate_count = ""
        json_supported_candidate_count = ""
    else:
        json_status = download_url(
            ALPHAFILL_JSON_URL.format(accession=accession),
            json_path,
            timeout=args.timeout,
            retries=args.retries,
            overwrite=args.overwrite,
        )
        cif_status = "NOT_REQUESTED"
        json_candidate_count = ""
        json_supported_candidate_count = ""
        if json_status in {"OK", "SKIPPED"}:
            should_download_cif = True
            if args.download_cif_only_if_json_has_supported_candidate:
                should_download_cif = False
                try:
                    candidate_dicts = extract_candidate_dicts(load_json(json_path), args)
                    json_candidate_count = str(len(candidate_dicts))
                    supported_metals = set(uniprot_metals)
                    if supported_metals:
                        candidate_dicts = [
                            candidate for candidate in candidate_dicts if candidate.get("metaltype") in supported_metals
                        ]
                    json_supported_candidate_count = str(len(candidate_dicts))
                    should_download_cif = bool(candidate_dicts)
                except Exception as exc:  # noqa: BLE001 - fetch summary should record malformed JSON.
                    cif_status = f"SKIPPED_JSON_PARSE_FAILED:{type(exc).__name__}"
                if not should_download_cif and cif_status == "NOT_REQUESTED":
                    cif_status = "SKIPPED_NO_JSON_SUPPORTED_CANDIDATE"
            if should_download_cif:
                cif_status = download_url(
                    ALPHAFILL_ENTRY_URL.format(accession=accession),
                    cif_path,
                    timeout=args.timeout,
                    retries=args.retries,
                    overwrite=args.overwrite,
                )

    summary_row = {
        "split": split,
        "uniprot_id": accession,
        "source_dataset": row.get("source_dataset", "CARE"),
        "source_task": row.get("source_task", "task1"),
        "source_split_name": row.get("source_split_name", "30_identity"),
        "source_record_ids": row.get("source_record_ids", ""),
        "n_source_rows": row.get("n_source_rows", ""),
        "json_status": json_status,
        "json_path": str(json_path) if json_path.exists() else "",
        "cif_status": cif_status,
        "cif_path": str(cif_path) if cif_path.exists() else "",
        "json_candidate_count": json_candidate_count,
        "json_supported_candidate_count": json_supported_candidate_count,
        "uniprot_status": uniprot_status,
        "uniprot_path": str(uniprot_path) if uniprot_path.exists() else "",
        "uniprot_supported_transition_metals": ";".join(uniprot_metals),
    }
    should_print = (
        args.progress_every <= 1
        or processed % args.progress_every == 0
        or str(json_status).startswith("FAILED")
        or str(cif_status).startswith("FAILED")
    )
    log_line = (
        f"[{processed}] [{split}] {accession}: json={json_status} cif={cif_status} "
        f"uniprot={uniprot_status} metals={';'.join(uniprot_metals) or '-'} "
        f"json_supported_candidates={json_supported_candidate_count or '-'}"
    )
    return processed, summary_row, should_print, log_line


def command_fetch_alphafill(args: argparse.Namespace) -> None:
    work_root = resolve_path(args.work_root)
    annotation_cache = load_uniprot_annotation_cache(work_root) if args.use_uniprot_annotation_cache else {}
    if args.prefilter_uniprot_supported_metals and args.skip_uniprot and not annotation_cache:
        raise ValueError(
            "--prefilter-uniprot-supported-metals with --skip-uniprot requires a populated "
            f"{UNIPROT_ANNOTATION_CACHE_NAME}. Run prefetch-uniprot-annotations first."
        )
    selected_rows = list(iter_selected_manifest_rows(
        work_root,
        splits=args.splits,
        limit_per_split=args.limit_per_split,
    ))
    results: list[tuple[int, dict[str, Any]]] = []

    if args.n_jobs <= 1:
        skipped_not_supported = 0
        for processed, (split, row) in enumerate(selected_rows, start=1):
            if not row.get("uniprot_id", "").strip():
                continue
            cached_metals = cached_uniprot_supported_metals(annotation_cache, split=split, accession=row["uniprot_id"].strip())
            if args.only_uniprot_supported and cached_metals is not None and not cached_metals:
                skipped_not_supported += 1
                continue
            if skipped_not_supported and not results:
                print(f"[INFO] skipped_no_uniprot_supported={skipped_not_supported}", flush=True)
            result = fetch_alphafill_manifest_row(
                args=args,
                work_root=work_root,
                annotation_cache=annotation_cache,
                processed=processed,
                split=split,
                row=row,
            )
            index, summary_row, should_print, log_line = result
            results.append((index, summary_row))
            if should_print:
                print(log_line, flush=True)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        if skipped_not_supported:
            print(f"[INFO] skipped_no_uniprot_supported={skipped_not_supported}", flush=True)
    else:
        futures: dict[concurrent.futures.Future[tuple[int, dict[str, Any], bool, str]], int] = {}
        skipped_not_supported = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
            for processed, (split, row) in enumerate(selected_rows, start=1):
                accession = row.get("uniprot_id", "").strip()
                if not accession:
                    continue
                cached_metals = cached_uniprot_supported_metals(annotation_cache, split=split, accession=accession)
                if args.only_uniprot_supported and cached_metals is not None and not cached_metals:
                    skipped_not_supported += 1
                    continue
                futures[
                    executor.submit(
                        fetch_alphafill_manifest_row,
                        args=args,
                        work_root=work_root,
                        annotation_cache=annotation_cache,
                        processed=processed,
                        split=split,
                        row=row,
                    )
                ] = processed
            print(
                f"[INFO] submitted AlphaFill fetch jobs={len(futures)} "
                f"skipped_no_uniprot_supported={skipped_not_supported} n_jobs={args.n_jobs}",
                flush=True,
            )
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                index, summary_row, should_print, log_line = future.result()
                completed += 1
                results.append((index, summary_row))
                if should_print or completed % max(args.progress_every, 1) == 0:
                    print(f"[completed {completed}/{len(futures)}] {log_line}", flush=True)

    summary_path = work_root / "alphafill" / "fetch_summary.csv"
    summary_rows = [row for _, row in sorted(results, key=lambda item: item[0])]
    write_csv(
        summary_path,
        [
            "split",
            "uniprot_id",
            "source_dataset",
            "source_task",
            "source_split_name",
            "source_record_ids",
            "n_source_rows",
            "json_status",
            "json_path",
            "cif_status",
            "cif_path",
            "json_candidate_count",
            "json_supported_candidate_count",
            "uniprot_status",
            "uniprot_path",
            "uniprot_supported_transition_metals",
        ],
        summary_rows,
    )
    print(f"Wrote fetch summary: {summary_path}")


def first_model(structure: Any) -> Any:
    return next(structure.get_models())


def parse_mmcif(path: Path, structure_id: str) -> Any:
    from Bio.PDB import MMCIFParser

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
        handle.write("HEADER    CARE ALPHAFILL REDUCED STRUCTURE\n")
        handle.write("COMPND   1 MOL_ID: 1;\n")
        handle.write(f"COMPND   2 MOLECULE: CARE_UNIPROT_{uniprot_id};\n")
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


def effective_ecnumber(manifest_row: Mapping[str, str]) -> str:
    return normalize_ec_number_list(manifest_row.get("ecnumber", "")) or "unknown"


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
        "ecnumber": effective_ecnumber(manifest_row),
        "whether_catalytic": 0,
        "uniprot_id": manifest_row["uniprot_id"],
        "source_dataset": manifest_row.get("source_dataset", "CARE"),
        "source_task": manifest_row.get("source_task", "task1"),
        "source_split_name": manifest_row.get("source_split_name", "30_identity"),
        "source_split": manifest_row.get("source_split", manifest_row.get("split", "")),
        "source_record_ids": manifest_row.get("source_record_ids", manifest_row.get("source_record_id", "")),
        "n_source_rows": manifest_row.get("n_source_rows", ""),
        "sequence_length": len(manifest_row.get("sequence", "")),
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
    annotation_cache = load_uniprot_annotation_cache(work_root) if args.use_uniprot_annotation_cache else {}
    all_build_rows: list[dict[str, Any]] = []
    build_parameters = {
        "min_alphafill_identity": args.min_alphafill_identity,
        "min_alignment_length": args.min_alignment_length,
        "site_dedup_distance": args.site_dedup_distance,
        "uniprot_metal_policy": args.uniprot_metal_policy,
        "supported_transition_metals": sorted(SUPPORTED_TRANSITION_METALS),
    }
    for split in args.splits:
        candidate_rows: list[dict[str, Any]] = []
        build_summary_rows: list[dict[str, Any]] = []
        for _, manifest_row in iter_selected_manifest_rows(
            work_root,
            splits=[split],
            limit_per_split=args.limit_per_split,
        ):
            accession = manifest_row["uniprot_id"].strip()
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
                cached_metals = cached_uniprot_supported_metals(annotation_cache, split=split, accession=accession)
                if cached_metals is not None:
                    annotated_metals = set(cached_metals)
                else:
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

                if args.uniprot_metal_policy == "require_supported" and not annotated_metals:
                    status = "NO_UNIPROT_SUPPORTED_TRANSITION_METAL"
                    detail = "skipped before AlphaFill parse by UniProt annotation"
                elif not json_path.exists() or not cif_path.exists():
                    status = "MISSING_ALPHAFILL_JSON_OR_CIF"
                    missing = []
                    if not json_path.exists():
                        missing.append("json")
                    if not cif_path.exists():
                        missing.append("cif")
                    detail = "missing " + "/".join(missing)
                else:
                    json_data = load_json(json_path)
                    raw_candidate_dicts = extract_candidate_dicts(json_data, args)
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
                        status = "NO_ACCEPTED_SUPPORTED_METALS"
                    else:
                        safe_ec = sanitize_filename_fragment(effective_ecnumber(manifest_row))
                        out_name = f"{sanitize_filename_fragment(accession)}__chain_{args.protein_chain}__EC_{safe_ec}.pdb"
                        out_path = work_root / "mahomes_inputs" / split / out_name
                        assigned_sites = write_reduced_pdb(
                            structure=structure,
                            out_path=out_path,
                            uniprot_id=accession,
                            ecnumber=effective_ecnumber(manifest_row),
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
                "source_dataset": manifest_row.get("source_dataset", "CARE"),
                "source_task": manifest_row.get("source_task", "task1"),
                "source_split_name": manifest_row.get("source_split_name", "30_identity"),
                "source_record_ids": manifest_row.get("source_record_ids", ""),
                "n_source_rows": manifest_row.get("n_source_rows", ""),
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
    (work_root / "mahomes_inputs" / "build_parameters.json").write_text(
        json.dumps(build_parameters, indent=2) + "\n",
        encoding="utf-8",
    )
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
        "source_dataset": candidate.get("source_dataset", "CARE"),
        "source_task": candidate.get("source_task", "task1"),
        "source_split_name": candidate.get("source_split_name", "30_identity"),
        "source_split": candidate.get("source_split", ""),
        "source_record_ids": candidate.get("source_record_ids", ""),
        "n_source_rows": candidate.get("n_source_rows", ""),
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
    return {structure_stem_for_summary_row(row) + ".pdb" for row in rows}


def count_existing_pdbs(path: Path) -> int:
    return len(list(path.glob("*.pdb"))) if path.exists() else 0


def load_build_parameters(work_root: Path) -> dict[str, Any]:
    defaults = {
        "min_alphafill_identity": 0.30,
        "min_alignment_length": 85,
        "site_dedup_distance": 1.0,
        "uniprot_metal_policy": "require_supported",
        "supported_transition_metals": sorted(SUPPORTED_TRANSITION_METALS),
    }
    path = work_root / "mahomes_inputs" / "build_parameters.json"
    if not path.exists():
        return defaults
    try:
        return {**defaults, **load_json(path)}
    except Exception:
        return defaults


def load_audit_metadata(work_root: Path) -> dict[str, Any]:
    path = manifest_paths(work_root)["audit_json"]
    if not path.exists():
        return {}
    try:
        data = load_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def fetch_summary_counts(work_root: Path, split: str) -> dict[str, int]:
    rows = [row for row in read_csv_if_exists(work_root / "alphafill" / "fetch_summary.csv") if row.get("split") == split]
    supported = {
        row.get("uniprot_id", "")
        for row in rows
        if row.get("uniprot_id", "") and row.get("uniprot_supported_transition_metals", "").strip()
    }
    fetched = {
        row.get("uniprot_id", "")
        for row in rows
        if row.get("uniprot_id", "")
        and row.get("json_status") in {"OK", "SKIPPED"}
        and row.get("cif_status") in {"OK", "SKIPPED"}
    }
    return {
        "uniprot_supported_transition_metal_proteins": len(supported),
        "alphafill_fetched_proteins": len(fetched),
    }


def count_rows(path: Path) -> int:
    return len(read_csv(path)) if path.exists() else 0


def assert_safe_export_root(output_root: Path) -> None:
    resolved = output_root.resolve()
    forbidden = {
        PROJECT_ROOT.resolve(),
        (PROJECT_ROOT / "DeepMzyme_Data").resolve(),
        DEFAULT_CARE_ROOT.resolve(),
    }
    if resolved in forbidden:
        raise ValueError(f"Refusing to export CARE dataset into unsafe root: {resolved}")


def command_export_dataset(args: argparse.Namespace) -> None:
    work_root = resolve_path(args.work_root)
    output_root = resolve_path(args.output_root)
    assert_safe_export_root(output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = manifest_paths(work_root)
    audit = load_audit_metadata(work_root)
    build_parameters = load_build_parameters(work_root)
    split_stats: dict[str, Any] = {}
    split_counts: dict[str, Any] = {}
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
        metadata_dir = output_root / "metadata" / split
        metadata_dir.mkdir(parents=True, exist_ok=True)
        for metadata_source, metadata_name in [
            (source_pdb_dir / CANDIDATE_SITE_CSV_NAME, CANDIDATE_SITE_CSV_NAME),
            (source_pdb_dir / "build_summary.csv", "build_summary.csv"),
            (work_root / "mahomes_outputs" / split / "summary_stats.json", "mahomes_summary_stats.json"),
            (work_root / "mahomes_outputs" / split / WHETHER_CATALYTIC_CSV_NAME, WHETHER_CATALYTIC_CSV_NAME),
            (work_root / "mahomes_outputs" / split / PREDICTION_RESULTS_CSV_NAME, PREDICTION_RESULTS_CSV_NAME),
        ]:
            if metadata_source.exists():
                shutil.copy2(metadata_source, metadata_dir / metadata_name)
        pair_manifest = paths[f"{split}_pairs"]
        protein_manifest = paths[f"{split}_proteins"]
        fetch_counts = fetch_summary_counts(work_root, split)
        split_counts[split] = {
            "input_care_rows": count_rows(pair_manifest),
            "unique_proteins": count_rows(protein_manifest),
            "uniprot_supported_transition_metal_proteins": fetch_counts["uniprot_supported_transition_metal_proteins"],
            "alphafill_fetched_proteins": fetch_counts["alphafill_fetched_proteins"],
            "mahomes_input_structures": count_existing_pdbs(source_pdb_dir),
            "mahomes_catalytic_structures": len(required_names),
            "catalytic_sites": count_rows(summary_csv),
            "exported_structures": copied,
        }
        split_stats[split] = {
            "summary_csv": str(dest_dir / SUMMARY_CSV_NAME),
            "structure_count": copied,
            "site_count": count_rows(summary_csv),
        }
        print(f"[{split}] copied {copied} structures and {split_stats[split]['site_count']} site rows")

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for source in [
        paths["train_pairs"],
        paths["test_pairs"],
        paths["train_proteins"],
        paths["test_proteins"],
        paths["audit_csv"],
        paths["audit_json"],
        work_root / "alphafill" / "fetch_summary.csv",
        work_root / "mahomes_inputs" / "build_summary_all.csv",
        work_root / "mahomes_inputs" / "build_parameters.json",
    ]:
        if source.exists():
            shutil.copy2(source, metadata_dir / source.name)

    total_counts = {
        "input_care_rows": sum(split_counts.get(split, {}).get("input_care_rows", 0) for split in args.splits),
        "unique_proteins": sum(split_counts.get(split, {}).get("unique_proteins", 0) for split in args.splits),
        "uniprot_supported_transition_metal_proteins": sum(split_counts.get(split, {}).get("uniprot_supported_transition_metal_proteins", 0) for split in args.splits),
        "alphafill_fetched_proteins": sum(split_counts.get(split, {}).get("alphafill_fetched_proteins", 0) for split in args.splits),
        "mahomes_input_structures": sum(split_counts.get(split, {}).get("mahomes_input_structures", 0) for split in args.splits),
        "mahomes_catalytic_structures": sum(split_counts.get(split, {}).get("mahomes_catalytic_structures", 0) for split in args.splits),
        "catalytic_sites": sum(split_counts.get(split, {}).get("catalytic_sites", 0) for split in args.splits),
        "exported_train_structures": split_counts.get("train", {}).get("exported_structures", 0),
        "exported_test_structures": split_counts.get("test", {}).get("exported_structures", 0),
    }
    metadata = {
        "source_dataset": "CARE",
        "source_task": audit.get("source_task", "task1"),
        "source_split_name": audit.get("source_split_name", "30_identity"),
        "care_train_file": audit.get("resolved_train_csv", ""),
        "care_test_file": audit.get("resolved_test_csv", ""),
        "work_root": str(work_root),
        "output_root": str(output_root),
        "supported_transition_metals": sorted(SUPPORTED_TRANSITION_METALS),
        "min_alphafill_identity": build_parameters.get("min_alphafill_identity", 0.30),
        "min_alignment_length": build_parameters.get("min_alignment_length", 85),
        "site_dedup_distance": build_parameters.get("site_dedup_distance", 1.0),
        "uniprot_metal_policy": build_parameters.get("uniprot_metal_policy", "require_supported"),
        "counts": {
            "by_split": split_counts,
            "total": total_counts,
        },
        "note": (
            "Computational CARE-derived AlphaFill-MAHOMES catalytic metalloenzyme subset. "
            "It is not the full CARE benchmark and is not experimental validation."
        ),
        "splits": split_stats,
    }
    (output_root / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    readme_lines = [
        "# CARE Task 1 30% AlphaFill-MAHOMES Metalloenzyme Subset",
        "",
        "This is a CARE-derived AlphaFill-MAHOMES catalytic metalloenzyme subset for DeepMzyme.",
        "It is not the full CARE benchmark.",
        "",
        "AlphaFill transferred metals are computational hypotheses.",
        "MAHOMES catalytic filtering is computational catalytic-site evidence.",
        "These structures should not be called experimentally validated metalloenzymes without independent evidence.",
        "",
        "Do not tune or select models on the exported CARE test split. Use only the exported CARE train split for internal train/validation splitting, HPO, seed repeats, and model selection.",
        "",
        "## Contents",
        "",
    ]
    for split, stats in split_stats.items():
        readme_lines.append(f"- `{split}/`: {stats['structure_count']} structures, {stats['site_count']} catalytic site rows")
    readme_lines.extend(["", "See `split_metadata.json` and `metadata/` for source evidence.", ""])
    (output_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Wrote dataset metadata: {output_root / 'split_metadata.json'}")


def add_work_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CARE Task 1 30% AlphaFill-MAHOMES inputs for DeepMzyme.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-care-task1", help="Discover, normalize, and audit CARE Task 1 train/test files.")
    audit.add_argument("--care-root", type=Path, default=DEFAULT_CARE_ROOT)
    audit.add_argument("--train-csv", type=Path, default=None)
    audit.add_argument("--test-csv", type=Path, default=None)
    audit.add_argument("--care-task", default="task1")
    audit.add_argument("--care-split-name", default="30_identity")
    audit.add_argument(
        "--train-representative-column",
        default=None,
        help=(
            "Optional CARE train CSV column whose unique representative accessions define the train manifest. "
            "Only representatives that also appear as train Entry rows are kept."
        ),
    )
    audit.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    audit.set_defaults(func=command_audit_care_task1)

    fetch = subparsers.add_parser("fetch-alphafill", help="Download AlphaFill JSON/mmCIF and UniProt annotations.")
    add_work_root_arg(fetch)
    fetch.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    fetch.add_argument("--limit-per-split", type=int, default=None)
    fetch.add_argument("--timeout", type=int, default=45)
    fetch.add_argument("--retries", type=int, default=2)
    fetch.add_argument("--sleep-seconds", type=float, default=0.0)
    fetch.add_argument("--progress-every", type=int, default=1, help="Print every N processed rows plus all requested rows.")
    fetch.add_argument("--n-jobs", type=int, default=1, help="Number of parallel AlphaFill fetch workers.")
    fetch.add_argument(
        "--only-uniprot-supported",
        action="store_true",
        help="Only write fetch-summary rows for proteins with UniProt-supported transition-metal annotations.",
    )
    fetch.add_argument("--overwrite", action="store_true")
    fetch.add_argument("--min-alphafill-identity", type=float, default=0.30)
    fetch.add_argument("--min-alignment-length", type=int, default=85)
    fetch.add_argument("--max-local-rmsd", type=float, default=None)
    fetch.add_argument("--max-binding-site-rmsd", type=float, default=None)
    fetch.add_argument("--max-local-environment-rmsd", type=float, default=None)
    fetch.add_argument("--max-pae-mean", type=float, default=None)
    fetch.add_argument("--min-binding-site-atom-count", type=int, default=None)
    fetch.add_argument("--skip-uniprot", action="store_true", help="Do not fetch UniProt JSON annotations.")
    fetch.add_argument(
        "--use-uniprot-annotation-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Use {UNIPROT_ANNOTATION_CACHE_NAME} when present instead of per-accession UniProt JSON.",
    )
    fetch.add_argument(
        "--download-cif-only-if-json-has-supported-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download AlphaFill CIF only when JSON has a threshold-passing candidate for the UniProt-supported metal.",
    )
    fetch.add_argument(
        "--prefilter-uniprot-supported-metals",
        action="store_true",
        help="Fetch UniProt first and skip AlphaFill downloads unless UniProt annotates a supported transition metal.",
    )
    fetch.set_defaults(func=command_fetch_alphafill)

    prefetch = subparsers.add_parser(
        "prefetch-uniprot-annotations",
        help="Batch-prefetch UniProt cofactor/binding-site annotations into a shared CARE cache.",
    )
    add_work_root_arg(prefetch)
    prefetch.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    prefetch.add_argument("--limit-per-split", type=int, default=None)
    prefetch.add_argument("--batch-size", type=int, default=200)
    prefetch.add_argument("--timeout", type=int, default=60)
    prefetch.add_argument("--retries", type=int, default=2)
    prefetch.add_argument("--sleep-seconds", type=float, default=0.0)
    prefetch.add_argument(
        "--checkpoint-batches",
        type=int,
        default=10,
        help="Write a resumable partial annotation cache after this many fetched batches.",
    )
    prefetch.add_argument("--overwrite", action="store_true")
    prefetch.add_argument(
        "--allow-batch-failures",
        action="store_true",
        help="Record failed UniProt batches as NOT_FOUND instead of aborting.",
    )
    prefetch.set_defaults(func=command_prefetch_uniprot_annotations)

    build = subparsers.add_parser("build-mahomes-inputs", help="Build reduced PDBs and candidate-site CSVs.")
    add_work_root_arg(build)
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
    build.add_argument(
        "--use-uniprot-annotation-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Use {UNIPROT_ANNOTATION_CACHE_NAME} when present for biological metal filtering.",
    )
    build.add_argument("--metadata-timeout", type=int, default=45)
    build.add_argument("--metadata-retries", type=int, default=2)
    build.add_argument("--protein-chain", default="A")
    build.add_argument("--metal-resseq-start", type=int, default=9001)
    build.set_defaults(func=command_build_mahomes_inputs)

    summarize = subparsers.add_parser("summarize-mahomes", help="Create catalytic-only summary CSVs from MAHOMES predictions.")
    add_work_root_arg(summarize)
    summarize.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    summarize.add_argument("--candidate-csv", type=Path, default=None, help="Single split override; normally omit.")
    summarize.add_argument("--job-root", type=Path, default=None, help="Single split override; normally omit.")
    summarize.add_argument("--output-dir", type=Path, default=None, help="Single split override; normally omit.")
    summarize.set_defaults(func=command_summarize_mahomes)

    export = subparsers.add_parser("export-dataset", help="Copy catalytic structures/CSVs into DeepMzyme_Data.")
    add_work_root_arg(export)
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
