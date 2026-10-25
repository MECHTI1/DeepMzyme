"""MMseqs2 nearest-hit baseline for the EC class vs CLEAN30 campaign (campaign design §4.3).

Each test protein takes the EC class of its nearest hit in fold k's full official training file.
The hit is the ``nearest_target`` of the fold's strata table (``ec_clean30.identity``: the pinned
search, E <= 1e-3, highest bits, then lowest E-value, then target accession). The training file's
EC column is the only label read; no test label is read.

Statuses (the class-table contract of ``ec_clean30.scoring``; every status but ``predicted`` is an
error there):
  - ``predicted``: the nearest training protein has one EC class;
  - ``no_hit``: the test protein has no hit at E <= 1e-3;
  - ``tie``: the nearest training protein's annotation has several EC classes;
  - ``failure``: its annotation cannot give an EC class (missing, out of range, malformed).

The baseline is training-free, but its test predictions are produced only in the Phase 7 pass:
the step needs the Freeze-2 manifest and its SHA-256, and Freeze 1 through the strata receipts.
The class table and its receipt are written once (mode ``x``).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

from ec_clean30 import identity
from project_paths import PROJECT_ROOT
from training.ec_classes import (
    STATUS_AMBIGUOUS, STATUS_OK, classify_ec_class, normalize_ec_number_list,
)

SCHEMA_VERSION = 1
STEP = "mmseqs2_nearest_hit_baseline"
CLASS_TABLE_COLUMNS = ("fold", "accession", "predicted_ec_class", "status")
MEMBERSHIP_COLUMNS = ("fold", "accession")
STATUS_PREDICTED = "predicted"
STATUS_NO_HIT = "no_hit"
STATUS_TIE = "tie"
STATUS_FAILURE = "failure"
STATUSES = (STATUS_PREDICTED, STATUS_NO_HIT, STATUS_TIE, STATUS_FAILURE)
STRATA_NAME = "strata_fold{fold}.tsv"
STRATA_RECEIPT_NAME = "strata_receipt.json"
RECEIPT_NAME = "baseline_receipt.json"
SEVERAL_CLASS_RULE = "a nearest training protein with several EC classes is a tie (an error), never a vote"


class BaselineError(ValueError):
    """The baseline step refuses its inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineError(message)


def _check_path(path: Path, what: str) -> Path:
    path = Path(path)
    for parts in (path.parts[:-1], path.resolve().parts[:-1]):
        blocked = [part for part in parts if part.lower() in identity.FORBIDDEN_PATH_PARTS]
        _require(not blocked, f"Refusing {what} through a held-out directory {blocked}: {path}")
    _require(path.is_file(), f"{what} not found: {path}")
    return path


def code_sha256() -> dict[str, Optional[str]]:
    paths = {
        "src/ec_clean30/baseline.py": Path(__file__).resolve(),
        "src/ec_clean30/identity.py": Path(identity.__file__).resolve(),
        "src/training/ec_classes.py": PROJECT_ROOT / "src" / "training" / "ec_classes.py",
        "scripts/run_ec_clean30_baseline.py": PROJECT_ROOT / "scripts" / "run_ec_clean30_baseline.py",
    }
    return {name: identity.sha256_file(path) if path.is_file() else None for name, path in paths.items()}


def read_membership(path: Path) -> dict[int, list[str]]:
    """Fold -> test accessions from the curated test membership (IDs only)."""
    path = _check_path(path, "The test membership")
    reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
    _require(tuple(reader.fieldnames or ()) == MEMBERSHIP_COLUMNS,
             f"{path.name} must have exactly the columns {list(MEMBERSHIP_COLUMNS)}")
    membership: dict[int, list[str]] = {}
    seen = set()
    for line, row in enumerate(reader, start=2):
        fold_text = (row["fold"] or "").strip()
        _require(fold_text.isdigit() and int(fold_text) < identity.N_FOLDS, f"{path.name}:{line}: bad fold")
        accession = identity._accession(row["accession"], f"{path.name}:{line}")
        _require((int(fold_text), accession) not in seen, f"{path.name}:{line}: repeated row")
        seen.add((int(fold_text), accession))
        membership.setdefault(int(fold_text), []).append(accession)
    _require(bool(membership), f"{path.name} lists no test protein")
    return {fold: sorted(accessions) for fold, accessions in sorted(membership.items())}


def read_official_ec_numbers(path: Path) -> dict[str, str]:
    """Accession -> normalised native EC string list from an official training file."""
    labels: dict[str, str] = {}
    for line, row in enumerate(identity._read_allowlisted(path, {"accession": ("entry", "id"),
                                                                 "ec_numbers": ("ec number", "ec_number", "ec")}),
                               start=2):
        accession = identity._accession(row["accession"], f"{path.name}:{line}")
        value = normalize_ec_number_list(row["ec_numbers"])
        _require(labels.get(accession, value) == value, f"{path.name} gives {accession} two EC annotations")
        labels[accession] = value
    return labels


def load_strata(path: Path, fold: int, freeze1_sha256: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Accession -> nearest training target from a receipted strata table of this fold and Freeze 1."""
    path = _check_path(path, f"The fold-{fold} strata table")
    _require(path.name == STRATA_NAME.format(fold=fold), f"Expected {STRATA_NAME.format(fold=fold)}, got {path.name}")
    receipt_path = _check_path(path.parent / STRATA_RECEIPT_NAME, f"The fold-{fold} strata receipt")
    receipt = json.loads(receipt_path.read_text())
    _require(receipt.get("step") == "strata" and receipt.get("fold") == fold,
             f"{receipt_path} is not the fold-{fold} strata receipt")
    _require(receipt.get("freeze1_sha256") == freeze1_sha256,
             f"The fold-{fold} strata receipt records another Freeze 1")
    _require(receipt["outputs"][path.name]["sha256"] == identity.sha256_file(path),
             f"{path.name} changed after the strata step")
    nearest = {row["accession"]: row["nearest_target"] for row in identity.read_tsv(path, identity.STRATA_COLUMNS)}
    record = {"strata_sha256": identity.sha256_file(path), "strata_receipt_sha256": identity.sha256_file(receipt_path),
              "nearest_hit_order": receipt.get("nearest_hit_order")}
    return nearest, record


def predict(accession: str, nearest: Mapping[str, str], training_labels: Mapping[str, str],
            fold: int) -> dict[str, str]:
    _require(accession in nearest, f"Fold-{fold} test protein {accession} has no strata row")
    target = nearest[accession]
    if not target:
        return {"fold": str(fold), "accession": accession, "predicted_ec_class": "", "status": STATUS_NO_HIT}
    _require(target in training_labels,
             f"The nearest target {target} of {accession} is not in the fold-{fold} official training file")
    result = classify_ec_class(training_labels[target].split(";"))
    if result.status == STATUS_OK:
        return {"fold": str(fold), "accession": accession, "predicted_ec_class": result.token,
                "status": STATUS_PREDICTED}
    status = STATUS_TIE if result.status == STATUS_AMBIGUOUS else STATUS_FAILURE
    return {"fold": str(fold), "accession": accession, "predicted_ec_class": "", "status": status}


def build_baseline(*, membership: Path, strata: Mapping[int, Path], official_split_dir: Path, out: Path,
                   freeze1_sha256: Optional[str], freeze2_manifest: Path, freeze2_sha256: str) -> dict[str, Any]:
    """Write the baseline class table and its receipt (Phase 7 only)."""
    freeze1_sha256 = identity.require_freeze1(freeze1_sha256, "The nearest-hit baseline")
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", freeze2_sha256 or "")),
             "--freeze2-sha256 must be 64 lowercase hexadecimal characters")
    freeze2_manifest = _check_path(freeze2_manifest, "The Freeze-2 manifest")
    _require(identity.sha256_file(freeze2_manifest) == freeze2_sha256,
             "The Freeze-2 manifest does not hash to --freeze2-sha256")
    out = Path(out)
    receipt_path = out.parent / RECEIPT_NAME
    _require(not out.exists() and not receipt_path.exists(),
             f"Refusing to replace an existing baseline in {out.parent}")
    folds = read_membership(membership)
    _require(set(folds) == set(strata), f"Strata are given for folds {sorted(strata)}, the membership has "
                                        f"folds {sorted(folds)}")
    rows, inputs, counts = [], {}, {}
    for fold, accessions in folds.items():
        nearest, record = load_strata(strata[fold], fold, freeze1_sha256)
        training_path = identity.training_file(official_split_dir, "official", fold)
        training_labels = read_official_ec_numbers(training_path)
        fold_rows = [predict(accession, nearest, training_labels, fold) for accession in accessions]
        rows.extend(fold_rows)
        inputs[str(fold)] = {**record, "official_training_file": training_path.name,
                             "official_training_sha256": identity.sha256_file(training_path)}
        tally = Counter(row["status"] for row in fold_rows)
        counts[str(fold)] = {status: tally.get(status, 0) for status in STATUSES}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CLASS_TABLE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    text = buffer.getvalue()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(text)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "freeze1_sha256": freeze1_sha256,
        "freeze2_manifest": {"path": str(freeze2_manifest), "sha256": freeze2_sha256},
        "rules": {"nearest_hit_order": identity.NEAREST_HIT_ORDER, "max_evalue": identity.MAX_EVALUE,
                  "several_ec_classes": SEVERAL_CLASS_RULE, "statuses": list(STATUSES)},
        "inputs": {"membership_sha256": identity.sha256_file(Path(membership)), "folds": inputs},
        "counts": counts,
        "outputs": {out.name: {"sha256": identity.sha256_text(text), "n_rows": len(rows)}},
        "code_sha256": code_sha256(),
        "labels_read": "official_training_files_only",
        "test_labels_read": False,
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(identity._json_text(receipt))
    return {"step": STEP, "counts": counts, "outputs": receipt["outputs"]}


def _strata_arg(text: str) -> tuple[int, Path]:
    fold, separator, path = text.partition("=")
    if not separator or not fold.isdigit():
        raise argparse.ArgumentTypeError("--strata takes FOLD=PATH")
    return int(fold), Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--membership", type=Path, required=True, help="Test membership CSV (fold, accession).")
    parser.add_argument("--strata", type=_strata_arg, action="append", required=True,
                        help="FOLD=PATH of strata_fold<k>.tsv next to its strata_receipt.json; once per fold.")
    parser.add_argument("--official-split-dir", type=Path, default=identity.DEFAULT_OFFICIAL_SPLIT_DIR)
    parser.add_argument("--out", type=Path, required=True, help="Class table to write (CSV, written once).")
    parser.add_argument("--freeze1-sha256", required=True)
    parser.add_argument("--freeze2-manifest", type=Path, required=True)
    parser.add_argument("--freeze2-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    strata = dict(args.strata)
    if len(strata) != len(args.strata):
        parser.exit(2, "Baseline refused: a fold is given twice in --strata\n")
    try:
        result = build_baseline(membership=args.membership, strata=strata, official_split_dir=args.official_split_dir,
                                out=args.out, freeze1_sha256=args.freeze1_sha256,
                                freeze2_manifest=args.freeze2_manifest, freeze2_sha256=args.freeze2_sha256)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Baseline refused: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
