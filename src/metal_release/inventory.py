"""Bind the fixed release catalog to existing development artifacts only."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .catalog import release_catalog
from .manifest import ReleaseError, candidate_member_ids, is_held_out_path, sha256_file, validate_manifest

REFIT_DIRECTORY = "stage6b_refit__gvp_late_fusion__four_class__baseline__e50__seed42"
BASELINE_PROBE = "probe-full-fp32__gvp_late_fusion__four_class__baseline__fold0__seed42"
_HASH = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_FILES = {
    "checkpoint": "terminal_model_checkpoint.pt",
    "run_config": "run_config.json",
    "selected_checkpoint": "selected_checkpoint.json",
    "validation_predictions": "val_predictions.csv",
}
_METADATA_FILES = ("dataset_summary.json", "run_metadata.json", "prepare_status.json")
_NOT_RUN_INPUTS = {"tests", "cache", "caches", "failed_attempts", "independent_validation_replay", ".git", ".cache"}


def excluded_from_discovery(path: Path) -> bool:
    """Do not descend into held-out inputs, failed runs, or cached predictions."""
    return is_held_out_path(path) or any(
        part.lower() in _NOT_RUN_INPUTS or part.lower().endswith("_cache") for part in Path(path).parts)


def _allowed_file(path: Path) -> Path:
    if excluded_from_discovery(path) or excluded_from_discovery(path.resolve()):
        raise ReleaseError(f"Held-out/cache/failed-run paths are not inventory inputs: {path}")
    if not path.is_file():
        raise ReleaseError(f"Missing development artifact: {path}")
    return path


def _read_json(path: Path) -> tuple[dict, str]:
    raw = _allowed_file(path).read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReleaseError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"JSON artifact must be an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _ref(path: Path, output_parent: Path, digest: str | None = None) -> dict:
    path = _allowed_file(path).resolve()
    return {"path": os.path.relpath(path, output_parent), "sha256": digest or sha256_file(path)}


def _discover(run_roots: list[Path], members: dict) -> dict[str, list[Path]]:
    names = {name: name for name in members if name != "stage6b_refit"}
    if "stage6b_refit" in members:
        names[REFIT_DIRECTORY] = "stage6b_refit"
    baseline = "gvp_late_fusion__four_class__baseline__fold0__seed42"
    if baseline in members:
        names[BASELINE_PROBE] = baseline
    for name, member in members.items():
        if name.startswith(("mc_P_", "mc_S_")):
            recipe = name.split("_")[1]
            phase = "discovery" if member["seed"] == 42 else "confirmation"
            names[f"metal_completion__{phase}__{recipe}__{member['target']}"
                  f"__fold{member['fold']}__seed{member['seed']}"] = name
    found: dict[str, set[Path]] = {name: set() for name in members}
    for supplied in run_roots:
        root = Path(supplied).expanduser().resolve()
        if excluded_from_discovery(root):
            raise ReleaseError(f"Not an allowed development run root: {root}")
        if not root.is_dir():
            continue
        for current, dirs, _files in os.walk(root, followlinks=False):
            directory = Path(current)
            if directory.name in names:
                found[names[directory.name]].add(directory)
                dirs[:] = []
                continue
            dirs[:] = sorted(d for d in dirs if not excluded_from_discovery(directory / d)
                             and not (directory / d).is_symlink())
    return {name: sorted(paths) for name, paths in found.items()}


def _check_identity(identity: dict, member: dict, run: Path) -> None:
    for key, expected in (("fold", member["fold"]), ("model_seed", member["seed"]),
                          ("target_scheme", member["target"])):
        if identity.get(key) != expected:
            raise ReleaseError(f"Campaign {key} differs from catalog in {run}")


def _inspect_run(run: Path, member: dict, output_parent: Path, fold_sha: str) -> dict:
    result = {"run_directory": str(run), "artifacts": {}, "metadata": {}, "blockers": []}
    files = dict(_ARTIFACT_FILES)
    if member["fold"] is None:
        files.pop("validation_predictions")
        files["selected_checkpoint"] = "refit_reload_check.json"
    for key, filename in files.items():
        try:
            result["artifacts"][key] = _ref(run / filename, output_parent)
        except (ReleaseError, OSError) as exc:
            result["blockers"].append(str(exc))
    for filename in _METADATA_FILES:
        path = run / filename
        if path.exists():
            try:
                result["metadata"][filename] = _ref(path, output_parent)
            except (ReleaseError, OSError) as exc:
                result["blockers"].append(str(exc))
    try:
        receipt, _ = _read_json(run / files["selected_checkpoint"])
        config, _ = _read_json(run / "run_config.json")
        config_body = config.get("config")
        if not isinstance(config_body, dict):
            raise ReleaseError(f"Missing configuration object in {run}")
        config_identity = config_body.get("campaign_run_identity")
        if isinstance(config_identity, str):
            config_identity = json.loads(config_identity)
        is_refit = member["fold"] is None
        identity = config_identity if is_refit else receipt.get("campaign_run_identity")
        if not isinstance(identity, dict):
            raise ReleaseError(f"Missing campaign identity in selected checkpoint: {run}")
        _check_identity(identity, member, run)
        if not is_refit and identity.get("fold_membership_sha256") != fold_sha:
            raise ReleaseError(f"Frozen fold membership hash differs in {run}")
        if is_refit:
            if (receipt.get("passed") is not True or identity.get("validation") != "none"
                    or identity.get("phase") != "stage6b_full_non_test_refit"
                    or config.get("checkpoint_rule") != "terminal"):
                raise ReleaseError(f"Refit receipt/configuration does not certify the frozen full-data fit: {run}")
        elif receipt.get("selected_checkpoint") != "terminal_model_checkpoint.pt":
            raise ReleaseError(f"Receipt selects a different checkpoint in {run}")
        if not is_refit and (receipt.get("checkpoint_rule") != "terminal" or receipt.get("fit_status") != "completed"):
            raise ReleaseError(f"Receipt does not certify a completed terminal fit in {run}")
        bound = receipt.get("checkpoint_sha256" if is_refit else "selected_checkpoint_sha256")
        actual = result["artifacts"].get("checkpoint", {}).get("sha256")
        if not isinstance(bound, str) or not _HASH.fullmatch(bound) or bound != actual:
            raise ReleaseError(f"Recorded checkpoint SHA-256 mismatch in {run}")
        if member["fold"] is not None:
            pred = receipt.get("validation_predictions", {})
            if not isinstance(pred, dict) or pred.get("path") != "val_predictions.csv":
                raise ReleaseError(f"Receipt validation path differs in {run}")
            if pred.get("sha256") != result["artifacts"].get("validation_predictions", {}).get("sha256"):
                raise ReleaseError(f"Recorded validation prediction SHA-256 mismatch in {run}")
        if not isinstance(config_identity, dict) or config_identity != identity:
            raise ReleaseError(f"Run config and selected checkpoint campaign identities differ in {run}")
        result["receipt_kind"] = "refit_reload_check" if is_refit else "selected_checkpoint"
        result["recorded_campaign_identity"] = identity
        result["recorded_environment"] = config.get("runtime_environment", {})
        result["embedding_metadata"] = config.get("esm_embedding_metadata", {})
        result["receipt_binding_verified"] = True
    except (ReleaseError, OSError, ValueError, TypeError) as exc:
        result["blockers"].append(str(exc))
        result["receipt_binding_verified"] = False
    return result


def bind_catalog(run_roots: list[Path], fold_membership: Path, output_path: Path, *,
                 source_inventory: Path | None = None, candidate: str | None = None) -> dict:
    """Write a new catalog and inventory report, retaining every missing member.

    Receipt comparisons verify historical byte bindings, not the scientific
    validity of source, normalization, environment, or validation eligibility.
    The latter are separate audit/replay gates.
    """
    output_path = Path(output_path).expanduser().resolve()
    report_path = output_path.with_name(output_path.stem + ".inventory.json")
    if output_path.exists() or report_path.exists():
        raise ReleaseError("Inventory outputs already exist; choose a new output path")
    manifest = release_catalog()
    selected = [candidate] if candidate is not None else list(manifest["candidates"])
    required_ids = {mid for name in selected for mid in candidate_member_ids(manifest, name)}
    # Keep the frozen catalog intact; only the explicitly requested scope can run.
    if candidate is not None:
        manifest["binding_scope"] = {"candidates": selected}
    required_members = {mid: member for mid, member in manifest["members"].items()
                        if mid in required_ids}
    manifest["fold_membership"] = _ref(Path(fold_membership), output_path.parent)
    manifest["embedding_model"] = "esmc_600m"
    if source_inventory is not None:
        manifest["source_inventory"] = _ref(Path(source_inventory), output_path.parent)
    locations = _discover(run_roots, required_members)
    report = {"schema": "deepmzyme-metal-inventory-v1", "run_roots": [str(Path(p).resolve()) for p in run_roots],
              "members": {}, "blockers": [], "claim_boundary":
              "Metadata and historical receipt bindings only; source, environment, OOF eligibility and replay are not certified."}
    if candidate is not None:
        report["binding_scope"] = manifest["binding_scope"]
    for name, member in required_members.items():
        copies = [_inspect_run(path, member, output_path.parent, manifest["fold_membership"]["sha256"])
                  for path in locations[name]]
        problems = []
        if not copies:
            problems.append("Missing exact run directory")
        for key in _ARTIFACT_FILES:
            hashes = {c["artifacts"][key]["sha256"] for c in copies if key in c["artifacts"]}
            if len(hashes) > 1:
                problems.append(f"Conflicting copies of {key}; no copy chosen")
        for filename in _METADATA_FILES:
            hashes = {c["metadata"][filename]["sha256"] for c in copies if filename in c["metadata"]}
            if len(hashes) > 1:
                problems.append(f"Conflicting copies of {filename}; no copy chosen")
        # A mismatched receipt is not repaired by binding a fresh hash or silently
        # choosing another copy. Incomplete copies remain visible as blockers.
        for copy in copies:
            problems.extend(copy["blockers"])
        if not problems:
            chosen = copies[0]
            member.update(chosen["artifacts"])
            member["metadata"] = chosen["metadata"]
            member["run_directory"] = chosen["run_directory"]
            member["recorded_campaign_identity"] = chosen["recorded_campaign_identity"]
            member["recorded_environment"] = chosen["recorded_environment"]
            member["receipt_binding_verified"] = True
            member["receipt_kind"] = chosen["receipt_kind"]
        report["members"][name] = {"copies": copies, "blockers": problems, "bound": not problems}
        report["blockers"].extend({"member": name, "reason": problem} for problem in problems)
    report["candidates"] = {
        name: {"bound": all(report["members"][mid]["bound"] for mid in candidate_member_ids(manifest, name)),
               "missing_or_invalid_members": [mid for mid in candidate_member_ids(manifest, name)
                                              if not report["members"][mid]["bound"]]}
        for name in selected
    }
    manifest["release_status"] = "blocked" if report["blockers"] else "provisional"
    manifest["inventory_blockers"] = report["blockers"]
    manifest["provenance_status"] = "receipt_bindings_only_pending_eligibility_and_replay"
    validate_manifest(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive writes preserve previous inventories and their historical claims.
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest["inventory_report"] = _ref(report_path, output_path.parent)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
