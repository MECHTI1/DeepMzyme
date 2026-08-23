"""Validated held-out evaluation protocols and paired-refit safeguards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


STANDARD_DISJOINT_PROTOCOL_ID = "standard_disjoint_v1"
METAL_PINMYMETAL_DUAL_PROTOCOL_ID = "metal_pinmymetal_shared_config_dual_v1"
VALID_EVALUATION_PROTOCOL_IDS = (
    STANDARD_DISJOINT_PROTOCOL_ID,
    METAL_PINMYMETAL_DUAL_PROTOCOL_ID,
)

FORBID_HELD_OUT_OVERLAP = "forbid"
EXACT_PINMYMETAL_SECONDARY_REFERENCE_POLICY = "exact_pinmymetal_secondary_reference"
VALID_HELD_OUT_OVERLAP_POLICIES = (
    FORBID_HELD_OUT_OVERLAP,
    EXACT_PINMYMETAL_SECONDARY_REFERENCE_POLICY,
)

EXACT_SPLIT_TYPE = "metal_split_pinmymetal_possibly_overlapped"
NONOVERLAP_SPLIT_TYPE = "non_overlapped_pinmymetal"
EXPECTED_EXACT_OVERLAP_PDBIDS = 177
EXACT_ROOT_NAME = "train_and_test_sets_structures_exact_pinmymetal"
NONOVERLAP_ROOT_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"

# These fields are expected to differ between paired refits because they locate
# data or generated run artifacts. Every modeling/training/selection field must
# remain identical.
PAIRED_REFIT_ALLOWED_DIFFERENCE_FIELDS = frozenset(
    {
        "structure_dir",
        "summary_csv",
        "test_structure_dir",
        "test_summary_csv",
        "runs_dir",
        "run_name",
        "dataset_bundle_id",
        "dataset_bundle_sha256",
        "final_test_result_role",
        "held_out_overlap_policy",
        "final_test_source_run_dirs",
        "final_test_checkpoint_paths",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required evaluation-protocol manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Evaluation-protocol manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Evaluation-protocol manifest must contain a JSON object: {path}")
    return payload


def _config_mapping(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        payload = asdict(config)
        return {key: _jsonable(value) for key, value in payload.items()}
    if isinstance(config, Mapping):
        payload = dict(config)
        nested = payload.get("config")
        if isinstance(nested, Mapping):
            return {key: _jsonable(value) for key, value in nested.items()}
        return {key: _jsonable(value) for key, value in payload.items()}
    raise TypeError(f"Expected a dataclass or mapping configuration, got {type(config).__name__}.")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def validate_evaluation_protocol_configuration(config: Any) -> None:
    protocol_id = str(getattr(config, "evaluation_protocol_id", STANDARD_DISJOINT_PROTOCOL_ID))
    overlap_policy = str(getattr(config, "held_out_overlap_policy", FORBID_HELD_OUT_OVERLAP))
    if protocol_id not in VALID_EVALUATION_PROTOCOL_IDS:
        raise ValueError(
            f"Unsupported --evaluation-protocol-id {protocol_id!r}. "
            f"Expected one of: {', '.join(VALID_EVALUATION_PROTOCOL_IDS)}."
        )
    if overlap_policy not in VALID_HELD_OUT_OVERLAP_POLICIES:
        raise ValueError(
            f"Unsupported --held-out-overlap-policy {overlap_policy!r}. "
            f"Expected one of: {', '.join(VALID_HELD_OUT_OVERLAP_POLICIES)}."
        )
    if protocol_id == METAL_PINMYMETAL_DUAL_PROTOCOL_ID and getattr(config, "task", None) != "metal":
        raise ValueError(f"{METAL_PINMYMETAL_DUAL_PROTOCOL_ID!r} is valid only for --task metal.")
    if overlap_policy == EXACT_PINMYMETAL_SECONDARY_REFERENCE_POLICY:
        if protocol_id != METAL_PINMYMETAL_DUAL_PROTOCOL_ID:
            raise ValueError(
                f"--held-out-overlap-policy {EXACT_PINMYMETAL_SECONDARY_REFERENCE_POLICY} requires "
                f"--evaluation-protocol-id {METAL_PINMYMETAL_DUAL_PROTOCOL_ID}."
            )
        if getattr(config, "task", None) != "metal":
            raise ValueError("The exact PinMyMetal overlap exception is valid only for --task metal.")
        if getattr(config, "final_test_result_role", None) != "secondary_diagnostic_report":
            raise ValueError(
                "The exact PinMyMetal overlap exception requires "
                "--final-test-result-role secondary_diagnostic_report."
            )
        if not getattr(config, "final_test_selected_config_id", None):
            raise ValueError(
                "The exact PinMyMetal overlap exception requires --final-test-selected-config-id "
                "for the configuration selected on non-overlap validation only."
            )
    if (
        protocol_id == METAL_PINMYMETAL_DUAL_PROTOCOL_ID
        and getattr(config, "final_test_result_role", None) in {"primary_final_report", "primary_preselected"}
        and overlap_policy != FORBID_HELD_OUT_OVERLAP
    ):
        raise ValueError("The primary route of the dual PinMyMetal protocol must forbid held-out overlap.")


def _require_exact_reference_manifest(config: Any, overlap_report: Mapping[str, Any]) -> dict[str, Any]:
    train_root = Path(config.structure_dir).resolve().parent
    test_root = Path(config.test_structure_dir).resolve().parent
    if train_root != test_root or train_root.name != EXACT_ROOT_NAME:
        raise RuntimeError(
            "The exact PinMyMetal overlap exception requires train and test directories under the same "
            f"{EXACT_ROOT_NAME!r} root; got train_root={train_root}, test_root={test_root}."
        )
    if Path(config.summary_csv).resolve().parent != Path(config.structure_dir).resolve():
        raise RuntimeError("Exact-reference training summary CSV must be inside its declared train structure directory.")
    if Path(config.test_summary_csv).resolve().parent != Path(config.test_structure_dir).resolve():
        raise RuntimeError("Exact-reference test summary CSV must be inside its declared test structure directory.")

    exact_metadata_path = train_root / "split_metadata.json"
    exact_metadata = _read_json(exact_metadata_path)
    if exact_metadata.get("split_type") != EXACT_SPLIT_TYPE:
        raise RuntimeError(f"Exact split manifest has unexpected split_type: {exact_metadata.get('split_type')!r}.")
    if int(exact_metadata.get("n_exact_overlap_pdbids", -1)) != EXPECTED_EXACT_OVERLAP_PDBIDS:
        raise RuntimeError("Exact split manifest does not record the audited 177-PDB-ID overlap.")

    nonoverlap_root = train_root.parent / NONOVERLAP_ROOT_NAME
    nonoverlap_metadata_path = nonoverlap_root / "split_metadata.json"
    nonoverlap_metadata = _read_json(nonoverlap_metadata_path)
    if nonoverlap_metadata.get("split_type") != NONOVERLAP_SPLIT_TYPE:
        raise RuntimeError("Primary non-overlap manifest has an unexpected split_type.")
    validation = nonoverlap_metadata.get("validation")
    if not isinstance(validation, Mapping) or not validation.get("final_pdbid_overlap_is_zero"):
        raise RuntimeError("Primary non-overlap manifest does not certify zero final PDB-ID overlap.")
    if not validation.get("test_tree_byte_identical_to_exact"):
        raise RuntimeError("Primary non-overlap manifest does not certify exact-test byte identity.")

    membership_path = nonoverlap_root / "source_exact_overlap_pdbids.txt"
    membership_metadata = nonoverlap_metadata.get("membership_files", {})
    expected_membership = membership_metadata.get(membership_path.name, {}) if isinstance(membership_metadata, Mapping) else {}
    if not membership_path.is_file() or not isinstance(expected_membership, Mapping):
        raise RuntimeError(f"Audited overlap membership is missing from the primary manifest: {membership_path}")
    actual_membership_sha256 = _sha256_file(membership_path)
    if actual_membership_sha256 != expected_membership.get("sha256"):
        raise RuntimeError("Audited exact-overlap membership hash does not match the primary manifest.")
    expected_pdbids = {
        line.strip().lower()
        for line in membership_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(expected_pdbids) != EXPECTED_EXACT_OVERLAP_PDBIDS:
        raise RuntimeError("Audited exact-overlap membership does not contain exactly 177 unique PDB IDs.")

    identities = overlap_report.get("overlap_identities", {})
    observed_pdbids = {
        str(value).strip().lower()
        for value in identities.get("pdb_id", [])
    } if isinstance(identities, Mapping) else set()
    if observed_pdbids != expected_pdbids:
        missing = sorted(expected_pdbids - observed_pdbids)[:10]
        unexpected = sorted(observed_pdbids - expected_pdbids)[:10]
        raise RuntimeError(
            "Observed held-out PDB-ID overlap does not equal the audited exact PinMyMetal membership: "
            f"expected={len(expected_pdbids)}, observed={len(observed_pdbids)}, "
            f"missing_examples={missing}, unexpected_examples={unexpected}."
        )

    return {
        "exact_split_metadata_path": str(exact_metadata_path),
        "exact_split_metadata_sha256": _sha256_file(exact_metadata_path),
        "nonoverlap_split_metadata_path": str(nonoverlap_metadata_path),
        "nonoverlap_split_metadata_sha256": _sha256_file(nonoverlap_metadata_path),
        "audited_overlap_membership_path": str(membership_path),
        "audited_overlap_membership_sha256": actual_membership_sha256,
        "audited_overlap_pdbid_count": len(expected_pdbids),
        "test_membership_relationship": nonoverlap_metadata.get("test_membership_relationship"),
    }


def enforce_held_out_overlap_policy(
    config: Any,
    overlap_report: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Return an annotated report or raise before any held-out inference."""

    validate_evaluation_protocol_configuration(config)
    report = dict(overlap_report)
    report.update(
        {
            "evaluation_protocol_id": config.evaluation_protocol_id,
            "held_out_overlap_policy": config.held_out_overlap_policy,
            "overlap_validation_phase": phase,
            "overlap_exception_applied": False,
            "overlap_policy_decision": "no_overlap_detected",
        }
    )
    if not report.get("train_test_overlap_detected"):
        return report
    if config.held_out_overlap_policy == FORBID_HELD_OUT_OVERLAP:
        raise RuntimeError(
            "Held-out evaluation blocked before inference because train/test structure groups overlap. "
            f"overlap_counts={report.get('overlap_counts')}; "
            f"overlap_examples={report.get('overlap_examples')}. "
            "The active held-out overlap policy is 'forbid'."
        )
    manifest = _require_exact_reference_manifest(config, report)
    report.update(
        {
            "overlap_exception_applied": True,
            "overlap_policy_decision": "allowed_exact_pinmymetal_secondary_reference",
            "overlap_warning": (
                "Known 177-PDB-ID train/test overlap allowed only for the exact PinMyMetal secondary-reference "
                "route. This report is non-independent, non-primary, and must not drive model selection."
            ),
            "validated_protocol_manifest": manifest,
        }
    )
    return report


def paired_refit_config_comparison(primary_config: Any, secondary_config: Any) -> dict[str, Any]:
    primary = _config_mapping(primary_config)
    secondary = _config_mapping(secondary_config)
    keys = sorted(set(primary) | set(secondary))
    differences = {
        key: {"primary": primary.get(key), "secondary": secondary.get(key)}
        for key in keys
        if primary.get(key) != secondary.get(key)
    }
    unexpected = sorted(set(differences) - PAIRED_REFIT_ALLOWED_DIFFERENCE_FIELDS)
    primary_selected = primary.get("final_test_selected_config_id")
    secondary_selected = secondary.get("final_test_selected_config_id")
    errors: list[str] = []
    if unexpected:
        errors.append(f"model/training fields differ: {unexpected}")
    if not primary_selected or primary_selected != secondary_selected:
        errors.append("both refits must record the same non-empty final_test_selected_config_id")
    if primary.get("evaluation_protocol_id") != METAL_PINMYMETAL_DUAL_PROTOCOL_ID:
        errors.append(f"primary refit must use {METAL_PINMYMETAL_DUAL_PROTOCOL_ID}")
    if secondary.get("evaluation_protocol_id") != METAL_PINMYMETAL_DUAL_PROTOCOL_ID:
        errors.append(f"secondary refit must use {METAL_PINMYMETAL_DUAL_PROTOCOL_ID}")
    if primary.get("held_out_overlap_policy") != FORBID_HELD_OUT_OVERLAP:
        errors.append("primary refit must forbid held-out overlap")
    if secondary.get("held_out_overlap_policy") != EXACT_PINMYMETAL_SECONDARY_REFERENCE_POLICY:
        errors.append("secondary refit must use the narrow exact-reference overlap policy")
    if primary.get("final_test_result_role") not in {"primary_final_report", "primary_preselected"}:
        errors.append("primary refit must have a primary final-test result role")
    if secondary.get("final_test_result_role") != "secondary_diagnostic_report":
        errors.append("secondary refit must have secondary_diagnostic_report role")
    return {
        "valid": not errors,
        "selected_config_id": primary_selected if primary_selected == secondary_selected else None,
        "allowed_difference_fields": sorted(PAIRED_REFIT_ALLOWED_DIFFERENCE_FIELDS),
        "differences": differences,
        "unexpected_difference_fields": unexpected,
        "errors": errors,
        "selection_policy": "configuration selected only on primary non-overlap validation; no test-metric selection",
    }


def require_paired_refit_configs(primary_config: Any, secondary_config: Any) -> dict[str, Any]:
    report = paired_refit_config_comparison(primary_config, secondary_config)
    if not report["valid"]:
        raise ValueError("Paired refit configurations are incompatible: " + "; ".join(report["errors"]))
    return report
