from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.config import TrainConfig
from training.evaluation_protocols import (
    EXACT_PINMYMETAL_SECONDARY_REFERENCE_POLICY,
    METAL_PINMYMETAL_DUAL_PROTOCOL_ID,
    paired_refit_config_comparison,
)
from training.final_test_reporting import paired_metal_route_comparison, prediction_artifact_payload
from training.run import validate_held_out_structure_disjointness


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _materialize_synthetic_exact_protocol(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    exact_root = tmp_path / "train_and_test_sets_structures_exact_pinmymetal"
    nonoverlap_root = tmp_path / "train_and_test_sets_structures_non_overlapped_pinmymetal"
    train_dir = exact_root / "train"
    test_dir = exact_root / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir()
    nonoverlap_root.mkdir()
    pdbids = [f"1{i:03x}" for i in range(177)]
    for pdbid in pdbids:
        (train_dir / f"{pdbid}__chain_A__EC_1.1.1.1.pdb").write_bytes(b"train")
        (test_dir / f"{pdbid}__chain_B__EC_1.1.1.1.pdb").write_bytes(b"test")
    train_summary = train_dir / "summary.csv"
    test_summary = test_dir / "summary.csv"
    train_summary.write_text("structure,chain_resi,metaltype\n", encoding="utf-8")
    test_summary.write_text("structure,chain_resi,metaltype\n", encoding="utf-8")
    _write_json(
        exact_root / "split_metadata.json",
        {
            "split_type": "metal_split_pinmymetal_possibly_overlapped",
            "n_exact_overlap_pdbids": 177,
        },
    )
    membership_payload = "".join(f"{pdbid}\n" for pdbid in sorted(pdbids))
    membership_path = nonoverlap_root / "source_exact_overlap_pdbids.txt"
    membership_path.write_text(membership_payload, encoding="utf-8")
    _write_json(
        nonoverlap_root / "split_metadata.json",
        {
            "split_type": "non_overlapped_pinmymetal",
            "validation": {
                "final_pdbid_overlap_is_zero": True,
                "test_tree_byte_identical_to_exact": True,
            },
            "membership_files": {
                membership_path.name: {
                    "count": 177,
                    "sha256": hashlib.sha256(membership_payload.encode("utf-8")).hexdigest(),
                }
            },
            "test_membership_relationship": "synthetic same membership",
        },
    )
    return train_dir, test_dir, train_summary, test_summary


def test_exact_overlap_exception_is_narrow_and_manifest_validated(tmp_path: Path) -> None:
    train_dir, test_dir, train_summary, test_summary = _materialize_synthetic_exact_protocol(tmp_path)
    config = TrainConfig(
        task="metal",
        structure_dir=train_dir,
        summary_csv=train_summary,
        test_structure_dir=test_dir,
        test_summary_csv=test_summary,
        run_test_eval=True,
        allow_final_refit_test_eval=True,
        selection_metric="train_loss",
        evaluation_protocol_id=METAL_PINMYMETAL_DUAL_PROTOCOL_ID,
        held_out_overlap_policy=EXACT_PINMYMETAL_SECONDARY_REFERENCE_POLICY,
        final_test_result_role="secondary_diagnostic_report",
        final_test_selected_config_id="selected-on-primary-validation",
    )
    report = validate_held_out_structure_disjointness(config)
    assert report is not None
    assert report["overlap_exception_applied"] is True
    assert report["overlap_counts"]["pdb_id"] == 177
    assert report["validated_protocol_manifest"]["audited_overlap_pdbid_count"] == 177

    forbidden = replace(config, held_out_overlap_policy="forbid", final_test_result_role="primary_final_report")
    with pytest.raises(RuntimeError, match="active held-out overlap policy is 'forbid'"):
        validate_held_out_structure_disjointness(forbidden)

    with pytest.raises(ValueError, match="secondary_diagnostic_report"):
        replace(config, final_test_result_role="primary_final_report")


def test_paired_refit_guard_rejects_model_field_drift() -> None:
    primary = TrainConfig(
        task="metal",
        evaluation_protocol_id=METAL_PINMYMETAL_DUAL_PROTOCOL_ID,
        held_out_overlap_policy="forbid",
        final_test_result_role="primary_final_report",
        final_test_selected_config_id="frozen-config",
    )
    secondary = replace(
        primary,
        structure_dir=Path("exact/train"),
        summary_csv=Path("exact/train/summary.csv"),
        test_structure_dir=Path("exact/test"),
        test_summary_csv=Path("exact/test/summary.csv"),
        run_name="exact-secondary-refit",
        final_test_result_role="secondary_diagnostic_report",
        held_out_overlap_policy=EXACT_PINMYMETAL_SECONDARY_REFERENCE_POLICY,
    )
    report = paired_refit_config_comparison(primary, secondary)
    assert report["valid"] is True
    drifted = replace(secondary, learning_rate=secondary.learning_rate * 2.0)
    drift_report = paired_refit_config_comparison(primary, drifted)
    assert drift_report["valid"] is False
    assert "learning_rate" in drift_report["unexpected_difference_fields"]


def test_paired_route_report_requires_aligned_prediction_ids() -> None:
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    primary = prediction_artifact_payload(
        task="metal",
        metal_y=targets,
        metal_probabilities=torch.tensor([[0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]]),
        sample_ids=["a", "b", "c", "d"],
        pdb_ids=["1aaa", "1bbb", "1ccc", "1ddd"],
    )
    secondary = prediction_artifact_payload(
        task="metal",
        metal_y=targets,
        metal_probabilities=torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]),
        sample_ids=["a", "b", "c", "d"],
        pdb_ids=["1aaa", "1bbb", "1ccc", "1ddd"],
    )
    report = paired_metal_route_comparison(primary, secondary, n_bootstrap=40, seed=7)
    assert report["sample_ids_aligned"] is True
    assert report["secondary_minus_primary"]["balanced_accuracy"] > 0.0
    assert len(report["secondary_minus_primary"]["balanced_accuracy_paired_bootstrap_ci"]) == 2

    misaligned = dict(secondary)
    misaligned["sample_ids"] = ["b", "a", "c", "d"]
    with pytest.raises(ValueError, match="sample_id ordering"):
        paired_metal_route_comparison(primary, misaligned, n_bootstrap=5)
