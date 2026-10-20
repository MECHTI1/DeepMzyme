"""Hash-bound artifacts and explicit weighted ensembles; no training or test I/O."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "deepmzyme-metal-release-v1"
COMMON4 = ("Mn", "Cu", "Zn", "Class VIII")
NATIVE = {"four_class": COMMON4, "five_class": ("Mn", "Cu", "Zn", "Fe", "Class VIII"),
          "six_class": ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")}
ARTIFACT_KEYS = ("checkpoint", "run_config", "selected_checkpoint", "validation_predictions")
HELD_OUT_PARTS = {"test", "heldout", "held_out", "held-out", "step_f_test"}
HELD_OUT_PREFIXES = ("test_", "heldout_", "held_out_", "held-out_")
HELD_OUT_SUFFIXES = ("_test", "_test_set")
HELD_OUT_FILES = {"test_predictions.csv", "test_checkpoint_probs.npz", "prepared_rows.csv",
                  "classmodel_test_set", "test_pockets_and_graphs.pt"}


class ReleaseError(ValueError):
    """The requested release cannot be used without changing its frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def is_held_out_path(path: Path | str) -> bool:
    """True when a path, or its symlink target, names a held-out or test input."""
    path = Path(path)
    for candidate in (path, path.resolve()):
        parts = [part.lower() for part in candidate.parts]
        if (HELD_OUT_PARTS.intersection(parts) or (parts and parts[-1] in HELD_OUT_FILES)
                or any(part.startswith(HELD_OUT_PREFIXES) or part.endswith(HELD_OUT_SUFFIXES) for part in parts)):
            return True
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(ref: dict | None) -> Path:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not ref["path"]:
        raise ReleaseError("Missing artifact path")
    expected = ref.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ReleaseError(f"Missing or invalid SHA-256: {ref['path']}")
    path = Path(ref["path"])
    if not path.is_file():
        raise ReleaseError(f"Missing artifact: {path}")
    if sha256_file(path) != expected:
        raise ReleaseError(f"Artifact SHA-256 mismatch: {path}")
    return path


def development_artifact(ref: dict | None, role: str) -> Path:
    """Refuse held-out paths before hashing or opening a supplied artifact."""
    require(isinstance(ref, dict) and bool(ref.get("path")), f"Missing {role} artifact")
    require(not is_held_out_path(ref["path"]), f"Refusing held-out path for {role}: {Path(ref['path'])}")
    return verify_artifact(ref)


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ReleaseError(f"Expected manifest schema {SCHEMA}")
    members, candidates = manifest.get("members"), manifest.get("candidates")
    if not isinstance(members, dict) or not members or not isinstance(candidates, dict) or not candidates:
        raise ReleaseError("Manifest requires members and candidates")
    if manifest.get("reference") not in candidates:
        raise ReleaseError("Reference candidate is missing")
    if "class_order" in manifest and tuple(manifest["class_order"]) != COMMON4:
        raise ReleaseError("Reporting class order must be Mn, Cu, Zn, Class VIII")
    if manifest.get("release_status", "provisional") not in {"blocked", "provisional", "verified"}:
        raise ReleaseError("Unknown release status")
    if "binding_scope" in manifest:
        scope = manifest["binding_scope"]
        selected = scope.get("candidates") if isinstance(scope, dict) else None
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(name, str) or name not in candidates for name in selected)
                or len(selected) != len(set(selected))):
            raise ReleaseError("Binding scope requires distinct known candidates")
    n_folds = manifest.get("n_folds", 5)
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ReleaseError("n_folds must be an integer of at least two")
    for member_id, member in members.items():
        if not isinstance(member, dict) or member.get("target") not in NATIVE:
            raise ReleaseError(f"Invalid member target: {member_id}")
        fold, seed = member.get("fold"), member.get("seed")
        if fold is not None and (isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < n_folds):
            raise ReleaseError(f"Invalid member fold: {member_id}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ReleaseError(f"Invalid member seed: {member_id}")
    for name, candidate in candidates.items():
        groups = candidate.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ReleaseError(f"Missing weighted groups: {name}")
        seen_groups, seen_members = set(), set()
        for group in groups:
            weight, ids, group_id = group.get("weight"), group.get("members"), group.get("id")
            if (not isinstance(group_id, str) or not group_id or group_id in seen_groups
                    or isinstance(weight, bool) or not isinstance(weight, (int, float))
                    or not math.isfinite(weight) or weight <= 0):
                raise ReleaseError(f"Invalid group identity/weight: {name}")
            if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
                    or len(set(ids)) != len(ids) or any(i not in members for i in ids)):
                raise ReleaseError(f"Missing, unknown or duplicate group members: {name}/{group_id}")
            if seen_members.intersection(ids):
                raise ReleaseError(f"Repeated checkpoint identity across groups: {name}")
            seen_groups.add(group_id)
            seen_members.update(ids)
            fold_sets: dict[int, set[int]] = {}
            targets = {members[i]["target"] for i in ids}
            if len(targets) != 1:
                raise ReleaseError(f"A family group must use one native target: {name}/{group_id}")
            if any(members[i]["fold"] is None for i in ids):
                if len(ids) != 1 or len(groups) != 1:
                    raise ReleaseError("A full-data refit cannot be mixed into a fold ensemble")
            else:
                for member_id in ids:
                    member = members[member_id]
                    folds = fold_sets.setdefault(member["seed"], set())
                    if member["fold"] in folds:
                        raise ReleaseError(f"Duplicate fold/seed in family: {name}/{group_id}")
                    folds.add(member["fold"])
                if any(folds != set(range(n_folds)) for folds in fold_sets.values()):
                    raise ReleaseError(f"Incomplete fold set in family: {name}/{group_id}")
        proxy = candidate.get("oof_proxy")
        if proxy is not None and (proxy == name or proxy not in candidates or candidates[proxy].get("oof_proxy")):
            raise ReleaseError(f"Invalid OOF configuration proxy: {name}")
        if "blocks" in candidate:
            block_groups = []
            by_id = {group["id"]: group for group in groups}
            for block in candidate["blocks"]:
                ids, weight = block.get("groups"), block.get("weight")
                if (not isinstance(ids, list) or not ids or any(i not in by_id for i in ids)
                        or isinstance(weight, bool) or not isinstance(weight, (int, float))
                        or not math.isfinite(weight) or weight <= 0):
                    raise ReleaseError(f"Invalid aggregation block: {name}")
                if any(not math.isclose(by_id[i]["weight"], weight / len(ids), rel_tol=1e-12) for i in ids):
                    raise ReleaseError(f"Block and effective group weights disagree: {name}")
                block_groups.extend(ids)
            if len(block_groups) != len(set(block_groups)) or set(block_groups) != seen_groups:
                raise ReleaseError(f"Aggregation blocks must cover every group once: {name}")
    from .catalog import CATALOG_VERSIONS, release_catalog
    if manifest.get("catalog_id") in CATALOG_VERSIONS:
        expected = release_catalog(manifest["catalog_id"])
        if manifest["reference"] != expected["reference"] or n_folds != 5:
            raise ReleaseError("Frozen Phase 2 reference or folds changed")
        if set(candidates) != set(expected["candidates"]):
            raise ReleaseError("Frozen Phase 2 shortlist changed")
        if manifest.get("class_order") != list(COMMON4) or manifest.get("embedding_model") != "esmc_600m":
            raise ReleaseError("Frozen Phase 2 class order or embedding model changed")
        for name, candidate in expected["candidates"].items():
            actual = candidates.get(name, {})
            if (actual.get("groups") != candidate["groups"] or actual.get("oof_proxy") != candidate.get("oof_proxy")
                    or actual.get("blocks") != candidate.get("blocks")):
                raise ReleaseError(f"Frozen Phase 2 aggregation changed: {name}")
        if set(members) != set(expected["members"]):
            raise ReleaseError("Frozen Phase 2 member inventory changed")
        for member_id, member in expected["members"].items():
            if any(members[member_id].get(key) != member[key] for key in ("fold", "seed", "target")):
                raise ReleaseError(f"Frozen Phase 2 member identity changed: {member_id}")
    if "binding_scope" in manifest and manifest.get("release_status") != "blocked":
        for name in manifest["binding_scope"]["candidates"]:
            for member_id in candidate_member_ids(manifest, name):
                member = members[member_id]
                required = ARTIFACT_KEYS if member["fold"] is not None else ARTIFACT_KEYS[:-1]
                if (member.get("receipt_binding_verified") is not True
                        or any(not isinstance(member.get(key), dict) for key in required)):
                    raise ReleaseError(f"Incomplete binding for scoped candidate {name}: {member_id}")


def load_manifest(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Cannot read release manifest: {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReleaseError("Manifest must be a JSON object")
    validate_manifest(manifest)
    manifest = copy.deepcopy(manifest)

    def resolve_refs(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value and isinstance(value["path"], str):
                artifact_path = Path(value["path"]).expanduser()
                value["path"] = str((path.parent / artifact_path).resolve() if not artifact_path.is_absolute()
                                    else artifact_path.resolve())
            for item in value.values():
                resolve_refs(item)
        elif isinstance(value, list):
            for item in value:
                resolve_refs(item)

    resolve_refs(manifest)
    manifest["manifest_sha256"] = sha256_file(path)
    manifest["manifest_path"] = str(path)
    return manifest


def candidate_member_ids(manifest: dict, candidate: str) -> list[str]:
    if candidate not in manifest["candidates"]:
        raise ReleaseError(f"Unknown candidate: {candidate}")
    return [member for group in manifest["candidates"][candidate]["groups"] for member in group["members"]]


def prediction_member_ids(manifest: dict, candidate: str) -> list[str]:
    """Require a usable inventory and honor its explicit candidate binding scope."""
    if manifest.get("release_status") == "blocked":
        raise ReleaseError("Release manifest is blocked; audit issues must be resolved before prediction")
    member_ids = candidate_member_ids(manifest, candidate)
    scope = manifest.get("binding_scope")
    if scope is not None and candidate not in scope["candidates"]:
        raise ReleaseError(f"Candidate is outside this manifest's binding scope: {candidate}; bind it separately")
    return member_ids


def native_to_common(probabilities: np.ndarray, target: str) -> np.ndarray:
    if target not in NATIVE:
        raise ReleaseError(f"Unknown native target: {target}")
    values = np.asarray(probabilities)
    if values.ndim < 1 or values.shape[-1] != len(NATIVE[target]):
        raise ReleaseError(f"Native probability width does not match {target}")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ReleaseError("Nonfinite or out-of-range probabilities")
    if not np.allclose(values.sum(axis=-1), 1, atol=2e-6, rtol=0):
        raise ReleaseError("Native probabilities must sum to one")
    return np.concatenate((values[..., :3], values[..., 3:].sum(axis=-1, keepdims=True)), axis=-1)


def aggregate_probabilities(manifest: dict, candidate: str, probs: dict[str, np.ndarray],
                            fold: int | None = None) -> np.ndarray:
    """Mean within each family then weighted mean across families; never drop missing members."""
    candidate_member_ids(manifest, candidate)
    if fold is not None and (isinstance(fold, bool) or fold not in range(manifest.get("n_folds", 5))):
        raise ReleaseError("Invalid OOF fold")
    total = None
    weight_sum = 0.0
    shape = None
    group_means = {}
    for group in manifest["candidates"][candidate]["groups"]:
        ids = [i for i in group["members"] if fold is None or manifest["members"][i]["fold"] == fold]
        if not ids:
            raise ReleaseError(f"No eligible OOF member in {candidate}/{group['id']}")
        missing = [i for i in ids if i not in probs]
        if missing:
            raise ReleaseError(f"Missing probabilities for {candidate}: {missing}")
        arrays = []
        for member_id in ids:
            values = np.asarray(probs[member_id])
            if values.ndim != 2 or values.shape[-1] != 4 or not np.isfinite(values).all():
                raise ReleaseError(f"Invalid common-four probabilities: {member_id}")
            if np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(-1), 1, atol=2e-6, rtol=0):
                raise ReleaseError(f"Common-four probabilities must sum to one: {member_id}")
            if shape is not None and values.shape != shape:
                raise ReleaseError("Ensemble member row shapes differ")
            shape = values.shape
            arrays.append(values)
        group_means[group["id"]] = np.mean(np.stack(arrays), axis=0)
        contribution = group_means[group["id"]] * group["weight"]
        total = contribution if total is None else total + contribution
        weight_sum += group["weight"]
    if "blocks" in manifest["candidates"][candidate]:
        total, weight_sum = None, 0.0
        for block in manifest["candidates"][candidate]["blocks"]:
            block_total = group_means[block["groups"][0]].copy()
            for group_id in block["groups"][1:]:
                block_total += group_means[group_id]
            contribution = (block_total / len(block["groups"])) * block["weight"]
            total = contribution if total is None else total + contribution
            weight_sum += block["weight"]
    return total / weight_sum
