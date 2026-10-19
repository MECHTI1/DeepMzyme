"""Nested re-selection of OOF ensembles to size selection optimism.

For each outer fold k, an ensemble recipe is chosen from the other folds' OOF
predictions only and then scored on fold k. Only `val_predictions.csv` and
`selected_checkpoint.json` files are read; no model, held-out file or network is used.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .audit import (COMMON4, PROBABILITY_TOLERANCE, _membership, _target_index, related_group_deltas,
                    related_groups)
from .inventory import BASELINE_PROBE, excluded_from_discovery
from .manifest import ReleaseError, candidate_member_ids, load_manifest, sha256_file, verify_artifact

MAX_TOP_N = 6
ESM_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
GVP_WEIGHTS = (0.25, 0.5, 0.75, 0.95, 1.0)
BRANCHES = {"gvp_late_fusion": "late_fusion", "only_esm": "only_esm", "only_gvp": "only_gvp"}
PROBABILITY_COLUMNS = [f"p_common4_{name.replace(' ', '_')}" for name in COMMON4]
_RUN = re.compile(r"(?P<family>.+)__fold(?P<fold>\d+)__seed(?P<seed>\d+)")
_COMPLETION = re.compile(r"metal_completion__(?:discovery|confirmation)__(?P<recipe>[A-Z]+)__(?P<target>\w+_class)")
# The seed-42 fold-0 baseline control was trained under a probe name, and completion recipe B is the
# unchanged v3 baseline; both belong to the baseline family, as in the release inventory.
FAMILY_ALIASES = {_RUN.fullmatch(BASELINE_PROBE)["family"]: "gvp_late_fusion__four_class__baseline",
                  "mc_B_four_class": "gvp_late_fusion__four_class__baseline"}
NOTE = ("Measures selection optimism at the OOF level: each outer fold's recipe is chosen from the other folds' "
        "OOF predictions and applied to that fold. The other folds' OOF predictions come from models whose "
        "training data included the outer fold (standard approximation for selection on OOF predictions); the "
        "outer fold's own predictions never come from a model trained on it. Families average their complete "
        "seeds, so multi-seed families score more members per OOF example. Test data are not used.")


def family_identity(directory_name: str) -> tuple[str, int, int] | None:
    """Map a run directory or member name to (family, fold, seed)."""
    match = _RUN.fullmatch(directory_name)
    if match is None:
        return None
    family = match["family"]
    completion = _COMPLETION.fullmatch(family)
    if completion:
        family = f"mc_{completion['recipe']}_{completion['target']}"
    return FAMILY_ALIASES.get(family, family), int(match["fold"]), int(match["seed"])


def balanced_accuracy(y: np.ndarray, probabilities: np.ndarray) -> float:
    correct = probabilities.argmax(axis=1) == y
    support = np.bincount(y, minlength=4)
    if (support == 0).any():
        raise ReleaseError("Balanced accuracy needs every common-four class in the selection data")
    return float(np.mean(np.bincount(y, weights=correct, minlength=4) / support))


def select_late_fusion(families: dict[str, np.ndarray], y: np.ndarray, mask: np.ndarray,
                       max_n: int = MAX_TOP_N) -> dict:
    """Rank families by selection BA, then choose the equal-family Top-N; ties keep the smaller N."""
    scores = {name: balanced_accuracy(y[mask], probs[mask]) for name, probs in families.items()}
    ranking = sorted(scores, key=lambda name: (-scores[name], name))
    best = None
    for n in range(1, min(max_n, len(ranking)) + 1):
        block = np.mean([families[name][mask] for name in ranking[:n]], axis=0)
        score = balanced_accuracy(y[mask], block)
        if best is None or score > best["selection_ba"]:
            best = {"top_n": n, "families": ranking[:n], "selection_ba": score}
    best["ranking"] = [{"family": name, "selection_ba": scores[name]} for name in ranking[:max_n]]
    return best


def _options(families: dict[str, np.ndarray], weights: tuple[float, ...]) -> list[tuple[tuple[str, ...], float]]:
    names = sorted(families)
    subsets = [subset for size in range(1, len(names) + 1) for subset in itertools.combinations(names, size)]
    return [((), 0.0)] + [(subset, weight) for subset in subsets for weight in weights]


def _block(families: dict[str, np.ndarray], subset: tuple[str, ...]) -> np.ndarray:
    return np.mean([families[name] for name in subset], axis=0)


def combine(late_fusion: np.ndarray, esm: dict[str, np.ndarray], gvp: dict[str, np.ndarray],
            choice: dict) -> np.ndarray:
    """Weighted block mean with late-fusion weight 1; empty blocks contribute nothing."""
    total, weight = late_fusion.copy(), 1.0
    for families, key in ((esm, "esm"), (gvp, "gvp")):
        if choice[f"{key}_families"]:
            total += choice[f"{key}_weight"] * _block(families, tuple(choice[f"{key}_families"]))
            weight += choice[f"{key}_weight"]
    return total / weight


def select_trimodal(late_fusion: np.ndarray, esm: dict[str, np.ndarray], gvp: dict[str, np.ndarray],
                    y: np.ndarray, mask: np.ndarray, esm_weights: tuple[float, ...] = ESM_WEIGHTS,
                    gvp_weights: tuple[float, ...] = GVP_WEIGHTS) -> dict:
    """Choose ESM/GVP subsets and weights by selection BA; ties keep fewer families, then lower weights."""
    labels, base = y[mask], late_fusion[mask]
    esm_options, gvp_options = _options(esm, esm_weights), _options(gvp, gvp_weights)
    esm_blocks = {subset: _block(esm, subset)[mask] for subset, _ in esm_options if subset}
    gvp_blocks = {subset: _block(gvp, subset)[mask] for subset, _ in gvp_options if subset}
    best_key, best = None, None
    for esm_subset, esm_weight in esm_options:
        partial = base + esm_weight * esm_blocks[esm_subset] if esm_subset else base
        for gvp_subset, gvp_weight in gvp_options:
            combined = partial + gvp_weight * gvp_blocks[gvp_subset] if gvp_subset else partial
            score = balanced_accuracy(labels, combined)
            key = (-score, len(esm_subset) + len(gvp_subset), esm_weight + gvp_weight)
            if best_key is None or key < best_key:
                best_key = key
                best = {"esm_families": list(esm_subset), "esm_weight": esm_weight,
                        "gvp_families": list(gvp_subset), "gvp_weight": gvp_weight, "selection_ba": score}
    return best


def nested_selection(families: dict[str, dict], y: np.ndarray, folds: np.ndarray, max_n: int = MAX_TOP_N) -> dict:
    """Repeat both selection procedures inside each outer fold, then once on all folds (apparent choice)."""
    pools = {branch: {name: item["probs"] for name, item in families.items() if item["branch"] == branch}
             for branch in BRANCHES.values()}
    if not pools["late_fusion"]:
        raise ReleaseError("Nested selection needs at least one late-fusion family")
    nested = {"late_fusion": np.empty((len(y), 4)), "trimodal": np.empty((len(y), 4))}

    def choose(mask):
        late = select_late_fusion(pools["late_fusion"], y, mask, max_n)
        block = np.mean([pools["late_fusion"][name] for name in late["families"]], axis=0)
        tri = select_trimodal(block, pools["only_esm"], pools["only_gvp"], y, mask)
        return late, block, tri

    per_fold = []
    for fold in sorted(set(folds.tolist())):
        held_out = folds == fold
        late, block, tri = choose(~held_out)
        nested["late_fusion"][held_out] = block[held_out]
        nested["trimodal"][held_out] = combine(block, pools["only_esm"], pools["only_gvp"], tri)[held_out]
        late["outer_fold_ba"] = balanced_accuracy(y[held_out], block[held_out])
        tri["outer_fold_ba"] = balanced_accuracy(y[held_out], nested["trimodal"][held_out])
        per_fold.append({"outer_fold": fold, "selection_examples": int((~held_out).sum()),
                         "evaluation_examples": int(held_out.sum()), "late_fusion": late, "trimodal": tri})
    late, _, tri = choose(np.ones(len(y), dtype=bool))
    return {"per_fold": per_fold, "predictions": nested, "full_data_choice": {"late_fusion": late, "trimodal": tri}}


def _discover(run_roots: list[Path]) -> list[tuple[tuple[str, int, int], Path]]:
    found = []
    for supplied in run_roots:
        root = Path(supplied).expanduser().resolve()
        if excluded_from_discovery(root) or not root.is_dir():
            raise ReleaseError(f"Run root is missing or not an allowed development path: {root}")
        for current, dirs, files in os.walk(root, followlinks=False):
            directory = Path(current)
            identity = family_identity(directory.name)
            if identity and {"val_predictions.csv", "selected_checkpoint.json"} <= set(files):
                found.append((identity, directory))
                dirs[:] = []
                continue
            dirs[:] = sorted(d for d in dirs
                             if not excluded_from_discovery(directory / d) and not (directory / d).is_symlink())
    return found


def _read_run(predictions: Path, receipt_path: Path, fold: int, seed: int, expected: dict[str, tuple[int, int]],
              membership_sha: str) -> tuple[str, np.ndarray, str]:
    """Return (branch, fold probabilities in membership order, file SHA-256) after development-only checks."""
    if predictions.name != "val_predictions.csv" or receipt_path.name != "selected_checkpoint.json":
        raise ReleaseError(f"Only validation exports are readable: {predictions}")
    digest = sha256_file(predictions)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = receipt.get("campaign_run_identity") or {}
    if (receipt.get("checkpoint_rule") != "terminal" or receipt.get("fit_status") != "completed"
            or receipt.get("reconciliation_status") != "match"
            or receipt.get("validation_predictions", {}).get("sha256") != digest):
        raise ReleaseError("receipt does not certify a completed terminal fit with these predictions")
    if (identity.get("fold") != fold or identity.get("model_seed") != seed
            or identity.get("fold_membership_sha256") != membership_sha):
        raise ReleaseError("receipt fold, seed or fold membership differs")
    branch = BRANCHES.get(identity.get("family"))
    if branch is None:
        raise ReleaseError(f"unknown model family {identity.get('family')!r}")
    with predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    uids = [row.get("source_uid") for row in rows]
    if len(set(uids)) != len(uids) or set(uids) != set(expected):
        raise ReleaseError("predictions do not cover exactly the frozen validation fold")
    values = np.empty((len(rows), 4))
    for row in rows:
        position, label = expected[row["source_uid"]]
        if int(row["fold"]) != fold or int(row["y_common4"]) != label:
            raise ReleaseError("prediction fold or label differs from the frozen membership")
        values[position] = [float(row[column]) for column in PROBABILITY_COLUMNS]
    if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or np.any(np.abs(values.sum(axis=1) - 1) > PROBABILITY_TOLERANCE)):
        raise ReleaseError("invalid common-four probabilities")
    return branch, values, digest


def collect_families(manifest: dict, run_roots: list[Path]) -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    """Load every family with complete five-fold OOF predictions for at least one seed."""
    rows = _membership(manifest)
    membership_sha = manifest["fold_membership"]["sha256"]
    fold_of = np.array([int(row["fold"]) for row in rows])
    labels = np.array([_target_index(row["native_element"], "four_class") for row in rows])
    indices = {fold: np.flatnonzero(fold_of == fold) for fold in range(manifest.get("n_folds", 5))}
    # Position within a fold's predictions follows the sorted membership order used everywhere else.
    expected = {fold: {rows[i]["source_uid"]: (position, int(labels[i])) for position, i in enumerate(index)}
                for fold, index in indices.items()}

    copies: dict[tuple[str, int, int], dict[Path, Path]] = {}
    for member_id, member in manifest["members"].items():
        identity = family_identity(member_id)
        if member.get("fold") is not None and identity is not None:
            predictions = verify_artifact(member.get("validation_predictions"))
            copies.setdefault(identity, {})[predictions] = verify_artifact(member.get("selected_checkpoint"))
    for identity, directory in _discover(run_roots):
        copies.setdefault(identity, {})[directory / "val_predictions.csv"] = directory / "selected_checkpoint.json"

    by_seed: dict[tuple[str, int], dict[int, dict[Path, Path]]] = {}
    for (family, fold, seed), paths in copies.items():
        by_seed.setdefault((family, seed), {})[fold] = paths
    seeds: dict[str, dict[int, np.ndarray]] = {}
    branches: dict[str, set[str]] = {}
    digests: dict[str, list[str]] = {}
    problems: dict[str, list[str]] = {}
    for (family, seed), fold_paths in sorted(by_seed.items()):
        if set(fold_paths) != set(expected):
            problems.setdefault(family, []).append(f"seed {seed}: folds {sorted(fold_paths)} only")
            continue
        values = np.empty((len(rows), 4))
        try:
            for fold, paths in sorted(fold_paths.items()):
                if len({sha256_file(path) for path in paths}) > 1:
                    raise ReleaseError(f"fold {fold} has conflicting copies; no copy chosen")
                predictions, receipt = next(iter(sorted(paths.items())))
                branch, fold_values, digest = _read_run(predictions, receipt, fold, seed, expected[fold],
                                                        membership_sha)
                values[indices[fold]] = fold_values
                branches.setdefault(family, set()).add(branch)
                digests.setdefault(family, []).append(digest)
        except (ReleaseError, OSError, ValueError, KeyError, TypeError) as error:
            problems.setdefault(family, []).append(f"seed {seed}: {error}")
            continue
        seeds.setdefault(family, {})[seed] = values

    families, excluded = {}, {}
    for family in sorted(set(seeds) | set(problems)):
        if family not in seeds:
            excluded[family] = "; ".join(problems[family])
        elif len(branches[family]) != 1:
            excluded[family] = "runs disagree on the model family"
        else:
            families[family] = {"branch": branches[family].pop(), "seeds": seeds[family],
                                "probs": np.mean(list(seeds[family].values()), axis=0),
                                "inputs_sha256": hashlib.sha256("".join(sorted(digests[family])).encode()).hexdigest(),
                                "ignored": problems.get(family, [])}
    return rows, families, excluded


def fixed_recipe(manifest: dict, candidate: str, families: dict[str, dict]) -> np.ndarray:
    """Rebuild a catalog candidate from the loaded per-seed OOF arrays (same arithmetic as the manifest)."""
    spec = manifest["candidates"][candidate]
    means = {}
    for group in spec["groups"]:
        keys = set()
        for member in group["members"]:
            identity = family_identity(member)
            if identity is None or manifest["members"][member].get("fold") is None:
                raise ReleaseError(f"{candidate} has no fold-level OOF members to rebuild")
            keys.add((identity[0], identity[2]))
        try:
            means[group["id"]] = np.mean([families[family]["seeds"][seed] for family, seed in sorted(keys)], axis=0)
        except KeyError as error:
            raise ReleaseError(f"{candidate}/{group['id']} lacks complete OOF predictions: {error}") from error
    blocks = spec.get("blocks") or [{"groups": [group["id"]], "weight": group["weight"]} for group in spec["groups"]]
    total = sum(block["weight"] * np.mean([means[g] for g in block["groups"]], axis=0) for block in blocks)
    return total / sum(block["weight"] for block in blocks)


def _pp(value: float) -> float:
    return round(100 * value, 4)


def _choice_view(choice: dict) -> dict:
    view = {}
    for key, value in choice.items():
        if key == "ranking":
            view[key] = [{"family": item["family"], "selection_ba_percent": _pp(item["selection_ba"])}
                         for item in value]
        elif key.endswith("_ba"):
            view[f"{key}_percent"] = _pp(value)
        else:
            view[key] = value
    return view


def _paired_delta(y: np.ndarray, left: np.ndarray, right: np.ndarray, group_ids: np.ndarray,
                  samples: int, rng: np.random.Generator) -> dict:
    deltas, invalid = related_group_deltas(y, left.argmax(1) == y, right.argmax(1) == y, group_ids, samples, rng)
    return {"pooled_delta_pp": _pp(balanced_accuracy(y, left) - balanced_accuracy(y, right)),
            "related_group_ci95_pp": [_pp(v) for v in np.quantile(deltas, [0.025, 0.975])],
            "missing_class_resamples": invalid}


def run_nested_selection(manifest_path: Path, run_roots: list[Path], output: Path, *,
                         expected_families: Sequence[str] = (), late_fusion_candidate: str = "top4_lf",
                         bootstrap_samples: int = 10000, seed: int = 42) -> dict:
    output = Path(output)
    if output.exists():
        raise ReleaseError(f"Output already exists: {output}")
    manifest = load_manifest(manifest_path)
    candidate_member_ids(manifest, late_fusion_candidate)  # Fail on an unknown candidate before reading runs.
    rows, families, excluded = collect_families(manifest, run_roots)
    y = np.array([_target_index(row["native_element"], "four_class") for row in rows])
    folds = np.array([int(row["fold"]) for row in rows])
    fold_ids = sorted(set(folds.tolist()))
    group_field, group_ids = related_groups(rows)
    result = nested_selection(families, y, folds)
    nested = result["predictions"]
    rng = np.random.default_rng(seed)

    def fold_scores(probabilities):
        return [balanced_accuracy(y[folds == k], probabilities[folds == k]) for k in fold_ids]

    summary = {}
    for procedure, candidate in (("late_fusion", late_fusion_candidate), ("trimodal", manifest["reference"])):
        fixed = fixed_recipe(manifest, candidate, families)
        summary[procedure] = {
            "fixed_candidate": candidate,
            "fixed_reported_oof_ba_percent": manifest["candidates"][candidate].get("reported_oof_ba_percent"),
            "fixed_pooled_ba_percent": _pp(balanced_accuracy(y, fixed)),
            "fixed_mean_fold_ba_percent": _pp(np.mean(fold_scores(fixed))),
            "fixed_outer_fold_ba_percent": [_pp(value) for value in fold_scores(fixed)],
            "full_data_reselection_apparent_ba_percent": _pp(result["full_data_choice"][procedure]["selection_ba"]),
            "nested_pooled_ba_percent": _pp(balanced_accuracy(y, nested[procedure])),
            "nested_mean_fold_ba_percent": _pp(np.mean(fold_scores(nested[procedure]))),
            "nested_minus_fixed": _paired_delta(y, nested[procedure], fixed, group_ids, bootstrap_samples, rng),
        }
    summary["nested_trimodal_minus_nested_late_fusion"] = _paired_delta(
        y, nested["trimodal"], nested["late_fusion"], group_ids, bootstrap_samples, rng)
    report = {
        "schema": "deepmzyme-metal-nested-selection-v1", "held_out_accessed": False, "note": NOTE,
        "manifest_sha256": manifest["manifest_sha256"], "catalog_id": manifest.get("catalog_id"),
        "fold_membership_sha256": manifest["fold_membership"]["sha256"],
        "n_examples": len(rows), "related_group_field": group_field, "n_related_groups": int(group_ids.max()) + 1,
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "procedure": {
            "late_fusion": f"Rank late-fusion families by selection pooled common-four BA; choose N in 1..{MAX_TOP_N} "
                           "for an equal-family Top-N mean by selection BA (ties keep the smaller N).",
            "trimodal": "On the chosen late-fusion block (weight 1), choose an Only-ESM family subset or none and an "
                        "Only-GVP subset or none, each family-equal, with block weights from the grids, by selection "
                        "BA (ties keep fewer families, then lower weights).",
            "esm_weights": list(ESM_WEIGHTS), "gvp_weights": list(GVP_WEIGHTS),
            "family": "One recipe; its complete seeds are averaged equally.",
        },
        "summary": summary,
        "per_fold": [{"outer_fold": item["outer_fold"], "selection_examples": item["selection_examples"],
                      "evaluation_examples": item["evaluation_examples"],
                      "late_fusion": _choice_view(item["late_fusion"]), "trimodal": _choice_view(item["trimodal"])}
                     for item in result["per_fold"]],
        "full_data_choice": {key: _choice_view(value) for key, value in result["full_data_choice"].items()},
        "families": [{"family": name, "branch": item["branch"], "seeds": sorted(item["seeds"]),
                      "pooled_oof_ba_percent": _pp(balanced_accuracy(y, item["probs"])),
                      "inputs_sha256": item["inputs_sha256"],
                      **({"ignored_seeds": item["ignored"]} if item["ignored"] else {})}
                     for name, item in sorted(families.items())],
        "excluded_families": excluded,
        "expected_families": {name: ("included" if name in families else excluded.get(name, "no run directory found"))
                              for name in expected_families},
        "selection_or_promotion": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return {"status": "complete", "reference": manifest["reference"], "output": str(output)}
