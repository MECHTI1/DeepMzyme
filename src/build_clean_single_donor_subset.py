from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "DeepMzyme_Data" / "CLEAN_30_shared"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "DeepMzyme_Data" / "CLEAN_30_shared_single_donor_supported_metal_conservative"
)

REQUIRED_COLUMNS = (
    "structure",
    "chain_resi",
    "metaltype",
    "uniprot_id",
    "alphafill_pdb_id",
    "alphafill_pdb_resolution",
    "alphafill_identity",
    "alphafill_alignment_length",
    "alphafill_local_rmsd",
    "uniprot_supported_transition_metals",
)

AUDIT_COLUMNS = (
    "single_donor_selected_alphafill_pdb_id",
    "single_donor_selection_status",
    "single_donor_selection_reason",
    "single_donor_candidate_count",
    "single_donor_original_site_count",
    "single_donor_selected_site_count_raw",
    "single_donor_selected_site_count_deduplicated",
    "single_donor_sites_removed_by_dedup",
    "single_donor_dedup_distance_angstrom",
    "single_donor_dedup_cluster_id",
    "single_donor_dedup_cluster_size",
    "single_donor_coordinate_status",
    "single_donor_coordinate_x",
    "single_donor_coordinate_y",
    "single_donor_coordinate_z",
    "stoichiometry_status",
    "stoichiometry_target_counts",
    "selected_donor_metal_counts",
    "stoichiometry_mismatch",
)


@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class DeduplicatedRows:
    rows: list[dict[str, str]]
    cluster_ids: dict[int, int]
    cluster_sizes: dict[int, int]
    coordinate_status_by_index: dict[int, str]
    removed_count: int


@dataclass(frozen=True)
class DonorCandidate:
    donor_id: str
    raw_rows: list[dict[str, str]]
    deduplicated: DeduplicatedRows
    metal_counts: Counter[str]
    score: tuple
    status: str
    reason: str
    mismatch: str
    target_counts_text: str
    selected_counts_text: str
    unsupported_metal_count: int
    count_distance: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a conservative CLEAN shared subset by selecting one AlphaFill donor "
            "per UniProt target and keeping that donor's deduplicated site rows."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--stoichiometry-csv",
        type=Path,
        default=None,
        help=(
            "Optional exact UniProt stoichiometry table. Supported formats: "
            "columns uniprot_id, metal/count; or uniprot_id, metal_counts with "
            "values like CU:2;ZN:1."
        ),
    )
    parser.add_argument(
        "--dedup-distance",
        type=float,
        default=2.0,
        help="Distance threshold, in Angstrom, for deduplicating close metal rows within one donor group.",
    )
    parser.add_argument(
        "--exclude-stoichiometry-mismatches",
        action="store_true",
        help=(
            "When exact stoichiometry is available for a target, drop the target if "
            "no donor group matches the exact counts. By default, keep the best "
            "quality donor and flag the mismatch."
        ),
    )
    parser.add_argument(
        "--exclude-no-clear-supported-metal",
        action="store_true",
        help=(
            "Drop targets that have neither exact stoichiometry nor a non-empty "
            "uniprot_supported_transition_metals field. The current CLEAN input "
            "normally has supported-metal annotations."
        ),
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to populate the output shared structures directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output root.",
    )
    return parser.parse_args()


def parse_float(value: str | None, *, missing: float = math.inf) -> float:
    if value is None:
        return missing
    text = str(value).strip()
    if not text:
        return missing
    try:
        return float(text)
    except ValueError:
        return missing


def parse_int(value: str | None, *, missing: int = -1) -> int:
    if value is None:
        return missing
    text = str(value).strip()
    if not text:
        return missing
    try:
        return int(float(text))
    except ValueError:
        return missing


def parse_bool_text(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def split_metal_list(value: str | None) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    return {token.strip().upper() for token in text.replace(",", ";").split(";") if token.strip()}


def format_counts(counts: Counter[str] | dict[str, int]) -> str:
    return ";".join(f"{metal}:{counts[metal]}" for metal in sorted(counts) if metal and counts[metal])


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def parse_counts_text(value: str | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    text = str(value or "").strip()
    if not text:
        return counts
    for token in text.replace(",", ";").split(";"):
        part = token.strip()
        if not part:
            continue
        if ":" in part:
            metal, count_text = part.split(":", 1)
        elif "=" in part:
            metal, count_text = part.split("=", 1)
        else:
            raise ValueError(f"Could not parse stoichiometry token {part!r}; expected METAL:COUNT.")
        metal = metal.strip().upper()
        if not metal:
            raise ValueError(f"Could not parse stoichiometry token {part!r}; empty metal.")
        counts[metal] = int(float(count_text.strip()))
    return counts


def normalized_header_map(fieldnames: Iterable[str] | None) -> dict[str, str]:
    return {field.strip().lower(): field for field in (fieldnames or []) if field}


def load_stoichiometry(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Stoichiometry CSV not found: {path}")

    result: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = normalized_header_map(reader.fieldnames)
        uniprot_col = header.get("uniprot_id") or header.get("uniprot") or header.get("entry")
        counts_col = header.get("metal_counts") or header.get("stoichiometry") or header.get("stoichiometry_counts")
        metal_col = header.get("metaltype") or header.get("metal") or header.get("metal_type")
        count_col = header.get("count") or header.get("stoichiometry_count") or header.get("metal_count")
        if uniprot_col is None:
            raise ValueError(f"{path} needs a uniprot_id/uniprot/entry column.")
        if counts_col is None and (metal_col is None or count_col is None):
            raise ValueError(
                f"{path} needs either a metal_counts column or metal/count columns."
            )

        for row in reader:
            uniprot_id = row.get(uniprot_col, "").strip()
            if not uniprot_id:
                continue
            if counts_col is not None and row.get(counts_col, "").strip():
                for metal, count in parse_counts_text(row[counts_col]).items():
                    result[uniprot_id][metal] = count
                continue
            metal = row.get(metal_col or "", "").strip().upper()
            count_text = row.get(count_col or "", "").strip()
            if metal and count_text:
                result[uniprot_id][metal] = int(float(count_text))
    return dict(result)


def structure_id_from_path(path: Path) -> str:
    return path.stem.split("__chain_", 1)[0]


def normalize_chain_resi(chain: str, resseq: str, icode: str = "") -> str:
    chain_text = chain.strip() or "_"
    resseq_text = resseq.strip()
    try:
        resseq_text = str(int(resseq_text))
    except ValueError:
        pass
    icode_text = icode.strip()
    if icode_text:
        resseq_text = f"{resseq_text}{icode_text}"
    return f"{chain_text}_{resseq_text}"


def parse_structure_coordinates(structures_dir: Path) -> tuple[dict[tuple[str, str], Coordinate], dict[str, Path]]:
    if not structures_dir.exists():
        raise FileNotFoundError(f"Structures directory not found: {structures_dir}")

    coordinates: dict[tuple[str, str], Coordinate] = {}
    structure_paths: dict[str, Path] = {}
    for path in sorted(structures_dir.glob("*.pdb")):
        structure_id = structure_id_from_path(path)
        structure_paths[structure_id] = path
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("HETATM"):
                    continue
                if len(line) < 54:
                    continue
                chain_resi = normalize_chain_resi(line[21:22], line[22:26], line[26:27])
                try:
                    coord = Coordinate(
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                    )
                except ValueError:
                    continue
                coordinates[(structure_id, chain_resi)] = coord
    return coordinates, structure_paths


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        return fieldnames, list(reader)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distance(a: Coordinate, b: Coordinate) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def row_quality_key(row: dict[str, str], supported_metals: set[str]) -> tuple:
    metal = row.get("metaltype", "").strip().upper()
    unsupported_penalty = 0 if not supported_metals or metal in supported_metals else 1
    return (
        unsupported_penalty,
        0 if parse_bool_text(row.get("selected_by_uniprot_annotation")) else 1,
        parse_float(row.get("alphafill_binding_site_rmsd")),
        parse_float(row.get("alphafill_local_environment_rmsd")),
        parse_float(row.get("alphafill_local_rmsd")),
        parse_float(row.get("alphafill_pae_mean")),
        parse_float(row.get("alphafill_pdb_resolution")),
        -parse_float(row.get("alphafill_identity"), missing=-math.inf),
        -parse_int(row.get("alphafill_alignment_length")),
        row.get("chain_resi", ""),
    )


def deduplicate_rows(
    rows: list[dict[str, str]],
    *,
    coordinates: dict[tuple[str, str], Coordinate],
    distance_threshold: float,
    supported_metals: set[str],
) -> DeduplicatedRows:
    if not rows:
        return DeduplicatedRows([], {}, {}, {}, 0)

    parent = list(range(len(rows)))
    coordinate_status_by_index: dict[int, str] = {}
    row_coordinates: dict[int, Coordinate] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for index, row in enumerate(rows):
        key = (row["structure"].strip(), row["chain_resi"].strip())
        coord = coordinates.get(key)
        if coord is None:
            coordinate_status_by_index[index] = "missing_coordinate_kept"
            continue
        coordinate_status_by_index[index] = "coordinate_found"
        row_coordinates[index] = coord

    coordinate_indices = sorted(row_coordinates)
    for offset, left in enumerate(coordinate_indices):
        for right in coordinate_indices[offset + 1 :]:
            if distance(row_coordinates[left], row_coordinates[right]) <= distance_threshold:
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)

    kept_indices: list[int] = []
    cluster_ids: dict[int, int] = {}
    cluster_sizes: dict[int, int] = {}
    for cluster_index, members in enumerate(sorted(groups.values(), key=lambda item: min(item)), start=1):
        best_index = min(members, key=lambda idx: row_quality_key(rows[idx], supported_metals))
        kept_indices.append(best_index)
        cluster_ids[best_index] = cluster_index
        cluster_sizes[best_index] = len(members)
        if len(members) > 1:
            coordinate_status_by_index[best_index] = "deduplicated_coordinate_cluster"

    kept_indices.sort()
    kept_rows = [rows[index] for index in kept_indices]
    removed_count = len(rows) - len(kept_rows)
    return DeduplicatedRows(
        rows=kept_rows,
        cluster_ids=cluster_ids,
        cluster_sizes=cluster_sizes,
        coordinate_status_by_index=coordinate_status_by_index,
        removed_count=removed_count,
    )


def mean_or_inf(values: Iterable[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return math.inf
    return sum(finite_values) / len(finite_values)


def donor_quality_tuple(rows: list[dict[str, str]]) -> tuple:
    return (
        min(parse_float(row.get("alphafill_pdb_resolution")) for row in rows),
        -max(parse_float(row.get("alphafill_identity"), missing=-math.inf) for row in rows),
        -max(parse_int(row.get("alphafill_alignment_length")) for row in rows),
        mean_or_inf(parse_float(row.get("alphafill_binding_site_rmsd")) for row in rows),
        mean_or_inf(parse_float(row.get("alphafill_local_environment_rmsd")) for row in rows),
        mean_or_inf(parse_float(row.get("alphafill_local_rmsd")) for row in rows),
        mean_or_inf(parse_float(row.get("alphafill_pae_mean")) for row in rows),
    )


def count_distance(observed: Counter[str], target: dict[str, int]) -> int:
    metals = set(observed).union(target)
    return sum(abs(observed.get(metal, 0) - target.get(metal, 0)) for metal in metals)


def build_donor_candidate(
    donor_id: str,
    rows: list[dict[str, str]],
    *,
    target_counts: dict[str, int] | None,
    supported_metals: set[str],
    coordinates: dict[tuple[str, str], Coordinate],
    dedup_distance: float,
) -> DonorCandidate:
    deduplicated = deduplicate_rows(
        rows,
        coordinates=coordinates,
        distance_threshold=dedup_distance,
        supported_metals=supported_metals,
    )
    metal_counts = Counter(row.get("metaltype", "").strip().upper() for row in deduplicated.rows)
    unsupported_metal_count = (
        sum(count for metal, count in metal_counts.items() if metal not in supported_metals)
        if supported_metals
        else 0
    )

    quality = donor_quality_tuple(deduplicated.rows or rows)
    target_counts_text = format_counts(target_counts or {})
    selected_counts_text = format_counts(metal_counts)

    if target_counts is not None:
        diff = count_distance(metal_counts, target_counts)
        exact_match = diff == 0
        status = "stoichiometry_exact" if exact_match else "stoichiometry_mismatch"
        reason = (
            "exact_stoichiometry_match_then_quality"
            if exact_match
            else "closest_stoichiometry_then_quality"
        )
        mismatch = "0" if exact_match else "1"
        score = (
            0 if exact_match else 1,
            diff,
            unsupported_metal_count,
            *quality,
            donor_id,
        )
        return DonorCandidate(
            donor_id=donor_id,
            raw_rows=rows,
            deduplicated=deduplicated,
            metal_counts=metal_counts,
            score=score,
            status=status,
            reason=reason,
            mismatch=mismatch,
            target_counts_text=target_counts_text,
            selected_counts_text=selected_counts_text,
            unsupported_metal_count=unsupported_metal_count,
            count_distance=diff,
        )

    if supported_metals:
        status = "metal_supported_but_count_unknown"
        reason = "supported_metal_annotation_then_quality"
        mismatch = "unknown"
    else:
        status = "no_clear_supported_metal"
        reason = "no_stoichiometry_or_supported_metal_then_quality"
        mismatch = "unknown"

    score = (
        unsupported_metal_count,
        *quality,
        donor_id,
    )
    return DonorCandidate(
        donor_id=donor_id,
        raw_rows=rows,
        deduplicated=deduplicated,
        metal_counts=metal_counts,
        score=score,
        status=status,
        reason=reason,
        mismatch=mismatch,
        target_counts_text=target_counts_text,
        selected_counts_text=selected_counts_text,
        unsupported_metal_count=unsupported_metal_count,
        count_distance=None,
    )


def supported_metals_for_target(rows: list[dict[str, str]]) -> set[str]:
    supported: set[str] = set()
    for row in rows:
        supported.update(split_metal_list(row.get("uniprot_supported_transition_metals")))
    return supported


def choose_candidate(
    target_rows: list[dict[str, str]],
    *,
    target_counts: dict[str, int] | None,
    coordinates: dict[tuple[str, str], Coordinate],
    dedup_distance: float,
) -> DonorCandidate:
    supported_metals = supported_metals_for_target(target_rows)
    by_donor: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in target_rows:
        by_donor[row["alphafill_pdb_id"].strip()].append(row)

    candidates = [
        build_donor_candidate(
            donor_id,
            rows,
            target_counts=target_counts,
            supported_metals=supported_metals,
            coordinates=coordinates,
            dedup_distance=dedup_distance,
        )
        for donor_id, rows in by_donor.items()
    ]
    return min(candidates, key=lambda candidate: candidate.score)


def add_audit_columns(
    row: dict[str, str],
    *,
    candidate: DonorCandidate,
    original_site_count: int,
    candidate_count: int,
    dedup_index: int,
    coordinates: dict[tuple[str, str], Coordinate],
    dedup_distance: float,
) -> dict[str, str]:
    output = dict(row)
    coord = coordinates.get((row["structure"].strip(), row["chain_resi"].strip()))
    original_index = next(
        index for index, candidate_row in enumerate(candidate.raw_rows) if candidate_row is row
    )
    coordinate_status = candidate.deduplicated.coordinate_status_by_index.get(original_index, "not_checked")

    output.update(
        {
            "single_donor_selected_alphafill_pdb_id": candidate.donor_id,
            "single_donor_selection_status": candidate.status,
            "single_donor_selection_reason": candidate.reason,
            "single_donor_candidate_count": str(candidate_count),
            "single_donor_original_site_count": str(original_site_count),
            "single_donor_selected_site_count_raw": str(len(candidate.raw_rows)),
            "single_donor_selected_site_count_deduplicated": str(len(candidate.deduplicated.rows)),
            "single_donor_sites_removed_by_dedup": str(candidate.deduplicated.removed_count),
            "single_donor_dedup_distance_angstrom": f"{dedup_distance:g}",
            "single_donor_dedup_cluster_id": str(candidate.deduplicated.cluster_ids.get(original_index, dedup_index)),
            "single_donor_dedup_cluster_size": str(candidate.deduplicated.cluster_sizes.get(original_index, 1)),
            "single_donor_coordinate_status": coordinate_status,
            "single_donor_coordinate_x": "" if coord is None else f"{coord.x:.3f}",
            "single_donor_coordinate_y": "" if coord is None else f"{coord.y:.3f}",
            "single_donor_coordinate_z": "" if coord is None else f"{coord.z:.3f}",
            "stoichiometry_status": candidate.status,
            "stoichiometry_target_counts": candidate.target_counts_text,
            "selected_donor_metal_counts": candidate.selected_counts_text,
            "stoichiometry_mismatch": candidate.mismatch,
        }
    )
    return output


def process_fold_csv(
    source_csv: Path,
    output_csv: Path,
    *,
    coordinates: dict[tuple[str, str], Coordinate],
    stoichiometry: dict[str, dict[str, int]],
    dedup_distance: float,
    exclude_stoichiometry_mismatches: bool,
    exclude_no_clear_supported_metal: bool,
) -> tuple[dict[str, int | str], list[dict[str, str]], set[str]]:
    source_fieldnames, source_rows = read_csv_rows(source_csv)
    output_fieldnames = source_fieldnames + [column for column in AUDIT_COLUMNS if column not in source_fieldnames]

    rows_by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        target_id = row.get("uniprot_id", "").strip() or row.get("structure", "").strip()
        rows_by_target[target_id].append(row)

    selected_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    selected_structure_ids: set[str] = set()
    excluded_targets = 0
    dedup_removed = 0
    multi_donor_targets = 0
    status_counts: Counter[str] = Counter()

    for target_id in sorted(rows_by_target):
        target_rows = rows_by_target[target_id]
        donor_ids = {row["alphafill_pdb_id"].strip() for row in target_rows}
        if len(donor_ids) > 1:
            multi_donor_targets += 1

        candidate = choose_candidate(
            target_rows,
            target_counts=stoichiometry.get(target_id),
            coordinates=coordinates,
            dedup_distance=dedup_distance,
        )

        if exclude_stoichiometry_mismatches and candidate.status == "stoichiometry_mismatch":
            excluded_targets += 1
            continue
        if exclude_no_clear_supported_metal and candidate.status == "no_clear_supported_metal":
            excluded_targets += 1
            continue

        status_counts[candidate.status] += 1
        dedup_removed += candidate.deduplicated.removed_count
        for dedup_index, row in enumerate(candidate.deduplicated.rows, start=1):
            selected_rows.append(
                add_audit_columns(
                    row,
                    candidate=candidate,
                    original_site_count=len(target_rows),
                    candidate_count=len(donor_ids),
                    dedup_index=dedup_index,
                    coordinates=coordinates,
                    dedup_distance=dedup_distance,
                )
            )
            selected_structure_ids.add(row["structure"].strip())

        audit_rows.append(
            {
                "source_csv": source_csv.name,
                "target_id": target_id,
                "candidate_donor_count": str(len(donor_ids)),
                "selected_alphafill_pdb_id": candidate.donor_id,
                "selection_status": candidate.status,
                "selection_reason": candidate.reason,
                "original_site_count": str(len(target_rows)),
                "selected_site_count_raw": str(len(candidate.raw_rows)),
                "selected_site_count_deduplicated": str(len(candidate.deduplicated.rows)),
                "sites_removed_by_dedup": str(candidate.deduplicated.removed_count),
                "target_counts": candidate.target_counts_text,
                "selected_counts": candidate.selected_counts_text,
                "stoichiometry_mismatch": candidate.mismatch,
                "unsupported_metal_count": str(candidate.unsupported_metal_count),
                "count_distance": "" if candidate.count_distance is None else str(candidate.count_distance),
                "score": json.dumps(json_safe(candidate.score)),
            }
        )

    selected_rows.sort(key=lambda row: (row.get("structure", ""), row.get("chain_resi", "")))
    write_csv_rows(output_csv, output_fieldnames, selected_rows)
    stats: dict[str, int | str] = {
        "source_csv": source_csv.name,
        "output_csv": output_csv.name,
        "source_site_count": len(source_rows),
        "output_site_count": len(selected_rows),
        "source_target_count": len(rows_by_target),
        "output_target_count": len({row["uniprot_id"] for row in selected_rows}),
        "multi_donor_target_count": multi_donor_targets,
        "excluded_target_count": excluded_targets,
        "deduplicated_site_count": dedup_removed,
    }
    for status, count in sorted(status_counts.items()):
        stats[f"status_{status}"] = count
    return stats, audit_rows, selected_structure_ids


def prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    (output_root / "folds").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "structures").mkdir(parents=True, exist_ok=True)


def link_or_copy_structure(source: Path, destination: Path, *, link_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "copy":
        shutil.copy2(source, destination)
        return "copied"
    try:
        os.link(source, destination)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied_after_hardlink_failed"


def populate_structures(
    *,
    selected_structure_ids: set[str],
    source_structure_paths: dict[str, Path],
    output_structures_dir: Path,
    metadata_csv: Path,
    link_mode: str,
) -> dict[str, int]:
    fieldnames = ("structure_file", "shared_structure_path", "source_structure_path", "link_status")
    status_counts: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    missing = sorted(selected_structure_ids.difference(source_structure_paths))
    if missing:
        raise FileNotFoundError(f"Missing structure files for selected structures: {missing[:10]}")

    for structure_id in sorted(selected_structure_ids):
        source_path = source_structure_paths[structure_id]
        destination = output_structures_dir / source_path.name
        status = link_or_copy_structure(source_path, destination, link_mode=link_mode)
        status_counts[status] += 1
        rows.append(
            {
                "structure_file": source_path.name,
                "shared_structure_path": str(destination.resolve()),
                "source_structure_path": str(source_path.resolve()),
                "link_status": status,
            }
        )

    write_csv_rows(metadata_csv, list(fieldnames), rows)
    return dict(status_counts)


def write_audit_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = (
        "source_csv",
        "target_id",
        "candidate_donor_count",
        "selected_alphafill_pdb_id",
        "selection_status",
        "selection_reason",
        "original_site_count",
        "selected_site_count_raw",
        "selected_site_count_deduplicated",
        "sites_removed_by_dedup",
        "target_counts",
        "selected_counts",
        "stoichiometry_mismatch",
        "unsupported_metal_count",
        "count_distance",
        "score",
    )
    write_csv_rows(path, list(fieldnames), rows)


def write_readme(output_root: Path, *, source_root: Path, metadata: dict) -> None:
    readme = f"""# CLEAN 30 Single-Donor Supported-Metal Conservative Layout

This dataset was derived from `{source_root}` by selecting one AlphaFill donor
structure (`alphafill_pdb_id`) per UniProt target within each CLEAN fold split,
then keeping only that donor's site rows after close-metal deduplication.

The original `CLEAN_30_shared` directory is not modified.

For new CLEAN-30 training/evaluation runs, prefer the stable alias
`DeepMzyme_Data/CLEAN_30_main`, which points to this conservative dataset.
Keep `CLEAN_30_shared` as the original multi-donor reference/source dataset.

## Selection Rule

1. Group rows by target protein (`uniprot_id`).
2. Group each target's rows by `alphafill_pdb_id`.
3. Deduplicate close metal rows within each donor group at
   `{metadata["dedup_distance_angstrom"]}` Angstrom using coordinates from the
   shared PDB files.
4. If exact stoichiometry is supplied through `--stoichiometry-csv`, prefer
   donor groups whose deduplicated metal counts match it.
5. If exact counts are unavailable but UniProt-supported metal identities are
   present, choose the best-quality single donor group and mark rows as
   `metal_supported_but_count_unknown`.
6. If neither exact counts nor supported-metal identities are available, choose
   the best-quality donor group and mark rows as `no_clear_supported_metal`
   unless that exclusion option was enabled.

Quality tie-breakers are donor PDB resolution, AlphaFill identity, alignment
length, binding-site/local RMSD fields when present, local RMSD, PAE, and donor
ID for deterministic ordering.

## Contents

- `structures/`: selected shared structure files.
- `folds/`: conservative site-level fold CSVs with the original CLEAN fold file
  names and additional audit columns.
- `metadata/structure_sources.csv`: source path and link/copy status for each
  structure file.
- `metadata/single_donor_selection_audit.csv`: one decision row per target per
  fold split.
- `split_metadata.json`: generation settings and per-fold source/output counts.

## Supported-Metal / Stoichiometry Caveat

The source CLEAN CSVs used here contain `uniprot_supported_transition_metals`,
not exact UniProt metal counts. That is why this dataset name uses
`supported_metal` rather than `stoich`.

Unless a `--stoichiometry-csv` is supplied, this builder does not invent exact
stoichiometry. It records the count status in `stoichiometry_status` and
`stoichiometry_mismatch`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    folds_dir = source_root / "folds"
    structures_dir = source_root / "structures"
    if not folds_dir.exists():
        raise FileNotFoundError(f"Source folds directory not found: {folds_dir}")

    prepare_output_root(output_root, overwrite=args.overwrite)
    coordinates, source_structure_paths = parse_structure_coordinates(structures_dir)
    stoichiometry = load_stoichiometry(args.stoichiometry_csv)

    fold_stats: dict[str, dict[str, int | str]] = {}
    all_audit_rows: list[dict[str, str]] = []
    all_selected_structure_ids: set[str] = set()
    for source_csv in sorted(folds_dir.glob("*.csv")):
        output_csv = output_root / "folds" / source_csv.name
        stats, audit_rows, selected_structure_ids = process_fold_csv(
            source_csv,
            output_csv,
            coordinates=coordinates,
            stoichiometry=stoichiometry,
            dedup_distance=args.dedup_distance,
            exclude_stoichiometry_mismatches=args.exclude_stoichiometry_mismatches,
            exclude_no_clear_supported_metal=args.exclude_no_clear_supported_metal,
        )
        fold_stats[source_csv.stem] = stats
        all_audit_rows.extend(audit_rows)
        all_selected_structure_ids.update(selected_structure_ids)

    structure_link_status_counts = populate_structures(
        selected_structure_ids=all_selected_structure_ids,
        source_structure_paths=source_structure_paths,
        output_structures_dir=output_root / "structures",
        metadata_csv=output_root / "metadata" / "structure_sources.csv",
        link_mode=args.link_mode,
    )
    write_audit_csv(output_root / "metadata" / "single_donor_selection_audit.csv", all_audit_rows)

    metadata = {
        "layout_name": "CLEAN 30 shared single AlphaFill donor supported-metal conservative subset",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "stoichiometry_csv": str(args.stoichiometry_csv.resolve()) if args.stoichiometry_csv else None,
        "stoichiometry_exact_counts_available": bool(stoichiometry),
        "dedup_distance_angstrom": args.dedup_distance,
        "exclude_stoichiometry_mismatches": args.exclude_stoichiometry_mismatches,
        "exclude_no_clear_supported_metal": args.exclude_no_clear_supported_metal,
        "selected_structure_count": len(all_selected_structure_ids),
        "structure_link_status_counts": structure_link_status_counts,
        "fold_stats": fold_stats,
        "note": (
            "Exact stoichiometry is used only when supplied via --stoichiometry-csv. "
            "The bundled CLEAN_30_shared CSVs provide supported metal identities, not exact metal counts."
        ),
    }
    (output_root / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_readme(output_root, source_root=source_root, metadata=metadata)

    print(f"Wrote conservative single-donor CLEAN dataset to {output_root}")
    print(f"Selected structures: {len(all_selected_structure_ids)}")
    print(f"Audit rows: {len(all_audit_rows)}")


if __name__ == "__main__":
    main()
