#!/usr/bin/env python3
"""Validate paired PinMyMetal refits and write their same-test comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from training.evaluation_protocols import require_paired_refit_configs
from training.final_test_reporting import paired_metal_route_comparison


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _prediction_path(run_dir: Path, report: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    test_report = report.get("test_report")
    if not isinstance(test_report, dict):
        test_report_path = run_dir / "test_report.json"
        test_report = _read_json(test_report_path) if test_report_path.is_file() else {}
    recorded = test_report.get("prediction_artifact_path")
    if recorded:
        candidate = Path(str(recorded))
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        if candidate.is_file():
            return candidate.resolve()
    for name in ("test_predictions.pt", "ensemble_predictions.pt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate a held-out prediction artifact in {run_dir}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that primary non-overlap and secondary exact PinMyMetal refits differ only in "
            "data/run fields, then compare their aligned predictions with paired bootstrap CIs."
        )
    )
    parser.add_argument("--primary-run-dir", type=Path, required=True)
    parser.add_argument("--secondary-run-dir", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, default=None)
    parser.add_argument("--secondary-predictions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260518)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    primary_run_dir = args.primary_run_dir.resolve()
    secondary_run_dir = args.secondary_run_dir.resolve()
    primary_config_path = primary_run_dir / "run_config.json"
    secondary_config_path = secondary_run_dir / "run_config.json"
    primary_run_config = _read_json(primary_config_path)
    secondary_run_config = _read_json(secondary_config_path)
    config_report = require_paired_refit_configs(primary_run_config, secondary_run_config)

    primary_predictions = _prediction_path(primary_run_dir, primary_run_config, args.primary_predictions)
    secondary_predictions = _prediction_path(secondary_run_dir, secondary_run_config, args.secondary_predictions)
    prediction_report = paired_metal_route_comparison(
        primary_predictions,
        secondary_predictions,
        n_bootstrap=args.bootstrap_resamples,
        confidence_level=args.bootstrap_confidence_level,
        seed=args.bootstrap_seed,
    )
    report = {
        "evaluation_protocol_id": "metal_pinmymetal_shared_config_dual_v1",
        "paired_refit_configuration": config_report,
        "paired_prediction_comparison": prediction_report,
        "primary_run_dir": str(primary_run_dir),
        "secondary_run_dir": str(secondary_run_dir),
        "primary_prediction_artifact": str(primary_predictions),
        "secondary_prediction_artifact": str(secondary_predictions),
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else primary_run_dir.parent / "metal_pinmymetal_paired_route_comparison.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved validated paired-route comparison: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
