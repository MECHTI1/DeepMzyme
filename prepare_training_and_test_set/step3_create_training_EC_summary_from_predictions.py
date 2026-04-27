#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import (
    CATALYTIC_ONLY_SUMMARY_CSV,
    MAHOMES_TRAIN_SET_DIR,
    PREDICTION_RESULTS_SUMMARY_CSV,
    WHETHER_CATALYTIC_SUMMARY_CSV,
)

JOB_ROOT = MAHOMES_TRAIN_SET_DIR
COMPLEMENTARY_JOB_ROOT = MAHOMES_TRAIN_SET_DIR / "train_set_complementary"
JOB_ROOTS = (JOB_ROOT, COMPLEMENTARY_JOB_ROOT)

OUTPUT_FIELDNAMES = ["structure", "chain_resi", "metaltype", "ecnumber", "whether_catalytic"]
PREDICTION_FIELDNAMES = [
    "job_root",
    "job_name",
    "jobid",
    "input file",
    "site#",
    "structure",
    "chain",
    "ecnumber",
    "metaltype",
    "chain_resi",
    "percent catalytic predictions",
    "prediction",
    "whether_catalytic",
]

PREDICTION_INPUT_RE = re.compile(r"^(?P<pdbid>[^_]+)__chain_(?P<chain>[^_]+)__EC_(?P<ec>.+)$")
PREDICTION_REQUIRED_COLUMNS = {
    "jobid",
    "input file",
    "site#",
    "prediction",
    "percent catalytic predictions",
    "Name1",
    "Name2",
    "Name3",
    "Name4",
    "Res#1",
    "Res#2",
    "Res#3",
    "Res#4",
}


def normalize_ec_number_list(value: str) -> str:
    values = []
    seen = set()
    for ec in re.split(r"[;,]", value):
        normalized = ec.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return ";".join(values)


def parse_prediction_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"catalytic", "1", "true"}:
        return 1
    if normalized in {"non-catalytic", "non catalytic", "not catalytic", "0", "false"}:
        return 0
    raise ValueError(f"Unsupported prediction label: {value!r}")


def parse_prediction_input_file(value: str) -> tuple[str, str, str]:
    match = PREDICTION_INPUT_RE.match(value.strip())
    if match is None:
        raise ValueError(f"Could not parse prediction input file: {value!r}")
    return (
        match.group("pdbid").strip().lower(),
        match.group("chain").strip(),
        normalize_ec_number_list(match.group("ec")),
    )


def require_columns(fieldnames: list[str] | None, required: set[str], csv_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"Could not read CSV header from: {csv_path}")
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")


def format_chain_resi(chain_id: str, residue_number: int) -> str:
    return f"{chain_id}_{residue_number}"


def iter_prediction_rows(job_root: Path):
    for pred_path in sorted(job_root.glob("job_*/predictions.csv")):
        job_name = pred_path.parent.name
        with pred_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require_columns(reader.fieldnames, PREDICTION_REQUIRED_COLUMNS, pred_path)

            for row in reader:
                structure, chain_id, ecnumber = parse_prediction_input_file(row["input file"])
                try:
                    catalytic_value = parse_prediction_label(row["prediction"])
                except ValueError:
                    continue

                for idx in range(1, 5):
                    metaltype = (row.get(f"Name{idx}") or "").strip().upper()
                    residue_number = (row.get(f"Res#{idx}") or "").strip()
                    if not metaltype or not residue_number:
                        continue

                    try:
                        residue_number_int = int(float(residue_number))
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid residue number {residue_number!r} in {pred_path} for row jobid={row['jobid']!r}"
                        ) from exc

                    yield {
                        "job_root": str(job_root),
                        "job_name": job_name,
                        "jobid": row["jobid"].strip(),
                        "input file": row["input file"].strip(),
                        "site#": row["site#"].strip(),
                        "structure": structure,
                        "chain": chain_id,
                        "ecnumber": ecnumber,
                        "metaltype": metaltype,
                        "chain_resi": format_chain_resi(chain_id, residue_number_int),
                        "percent catalytic predictions": row["percent catalytic predictions"].strip(),
                        "prediction": row["prediction"].strip(),
                        "whether_catalytic": catalytic_value,
                    }


def dedupe_rows(rows, fieldnames: list[str]) -> tuple[list[dict[str, str | int]], int]:
    unique_rows: list[dict[str, str | int]] = []
    seen_keys: set[tuple[str, ...]] = set()
    dropped_count = 0

    for row in rows:
        key = tuple(str(row[name]).strip() for name in fieldnames)
        if key in seen_keys:
            dropped_count += 1
            continue
        seen_keys.add(key)
        unique_rows.append(dict(row))

    return unique_rows, dropped_count


def build_output_rows(prediction_rows: list[dict[str, str | int]]) -> tuple[list[dict[str, str | int]], int]:
    catalytic_by_key: dict[tuple[str, str, str, str], int] = {}
    conflict_count = 0

    for row in prediction_rows:
        key = (
            str(row["structure"]).strip(),
            str(row["chain_resi"]).strip(),
            str(row["metaltype"]).strip(),
            str(row["ecnumber"]).strip(),
        )
        value = int(row["whether_catalytic"])

        if key in catalytic_by_key and catalytic_by_key[key] != value:
            conflict_count += 1
            catalytic_by_key[key] = max(catalytic_by_key[key], value)
            continue

        catalytic_by_key[key] = value

    output_rows = [
        {
            "structure": structure,
            "chain_resi": chain_resi,
            "metaltype": metaltype,
            "ecnumber": ecnumber,
            "whether_catalytic": catalytic_value,
        }
        for (structure, chain_resi, metaltype, ecnumber), catalytic_value in sorted(catalytic_by_key.items())
    ]
    return output_rows, conflict_count


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    missing_job_roots = [job_root for job_root in JOB_ROOTS if not job_root.exists()]
    if missing_job_roots:
        missing_paths = ", ".join(str(path) for path in missing_job_roots)
        raise FileNotFoundError(f"Job root not found: {missing_paths}")

    raw_prediction_rows = []
    for job_root in JOB_ROOTS:
        raw_prediction_rows.extend(iter_prediction_rows(job_root))
    if not raw_prediction_rows:
        searched_roots = ", ".join(str(path) for path in JOB_ROOTS)
        raise ValueError(f"No prediction rows found under: {searched_roots}")

    prediction_rows, duplicate_prediction_rows = dedupe_rows(raw_prediction_rows, PREDICTION_FIELDNAMES)
    output_rows, conflict_count = build_output_rows(prediction_rows)
    catalytic_only_rows = [row for row in output_rows if int(row["whether_catalytic"]) == 1]

    write_csv(PREDICTION_RESULTS_SUMMARY_CSV, PREDICTION_FIELDNAMES, prediction_rows)
    write_csv(WHETHER_CATALYTIC_SUMMARY_CSV, OUTPUT_FIELDNAMES, output_rows)
    write_csv(CATALYTIC_ONLY_SUMMARY_CSV, OUTPUT_FIELDNAMES, catalytic_only_rows)

    print(f"Wrote {len(prediction_rows)} unique expanded prediction rows to {PREDICTION_RESULTS_SUMMARY_CSV}")
    print(f"Dropped {duplicate_prediction_rows} duplicate expanded prediction rows")
    print(f"Resolved {conflict_count} conflicting output rows by preferring catalytic (1)")
    print(f"Wrote {len(output_rows)} rows to {WHETHER_CATALYTIC_SUMMARY_CSV}")
    print(f"Wrote {len(catalytic_only_rows)} catalytic rows to {CATALYTIC_ONLY_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
