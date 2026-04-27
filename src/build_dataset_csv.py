from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from label_schemes import METAL_TARGET_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a structure-level CSV with EC number(s) and metal type from a labeled structure set."
    )
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--allow-multi-metal-structures", action="store_true")
    parser.add_argument("--ec-label-depth", type=int, default=1)
    return parser.parse_args()


def build_structure_rows(
    *,
    structure_dir: Path,
    summary_csv: Path,
    allow_multi_metal_structures: bool,
    ec_label_depth: int,
) -> list[dict[str, str]]:
    from training.data import load_training_pockets_with_report_from_dir

    load_result = load_training_pockets_with_report_from_dir(
        structure_dir=structure_dir,
        require_full_labels=False,
        required_targets=(),
        summary_csv=summary_csv,
        require_esm_embeddings=False,
        require_external_features=False,
        ec_label_depth=ec_label_depth,
    )

    metals_by_structure: dict[str, set[str]] = defaultdict(set)
    ec_numbers_by_structure: dict[str, set[str]] = defaultdict(set)
    for pocket in load_result.pockets:
        if pocket.y_metal is None:
            raise ValueError(f"Structure {pocket.structure_id!r} is missing an inferred metal label.")
        metals_by_structure[pocket.structure_id].add(METAL_TARGET_LABELS[int(pocket.y_metal)])
        for ec_number in pocket.metadata.get("ec_numbers", []):
            ec_numbers_by_structure[pocket.structure_id].add(str(ec_number))

    rows: list[dict[str, str]] = []
    for structure_name in sorted(metals_by_structure):
        metal_labels = sorted(metals_by_structure[structure_name])
        if len(metal_labels) > 1 and not allow_multi_metal_structures:
            raise ValueError(
                f"Structure {structure_name!r} maps to multiple metal labels {metal_labels}. "
                "Pass --allow-multi-metal-structures to write a joined label instead."
            )
        rows.append(
            {
                "structure_name": structure_name,
                "ec_numbers": ";".join(sorted(ec_numbers_by_structure.get(structure_name, set()))),
                "metal_type": ";".join(metal_labels),
            }
        )
    return rows


def write_rows(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("structure_name", "ec_numbers", "metal_type"))
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows: list[dict[str, str]]) -> None:
    seen_structure_names: set[str] = set()
    for row in rows:
        structure_name = row["structure_name"].strip()
        if not structure_name:
            raise ValueError("Encountered an empty structure_name while validating the dataset CSV rows.")
        if structure_name in seen_structure_names:
            raise ValueError(f"Duplicate structure_name detected in CSV rows: {structure_name!r}")
        seen_structure_names.add(structure_name)
        if not row["metal_type"].strip():
            raise ValueError(f"Row for structure {structure_name!r} is missing metal_type.")


def main() -> None:
    args = parse_args()
    rows = build_structure_rows(
        structure_dir=args.structure_dir,
        summary_csv=args.summary_csv,
        allow_multi_metal_structures=args.allow_multi_metal_structures,
        ec_label_depth=args.ec_label_depth,
    )
    validate_rows(rows)
    write_rows(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
