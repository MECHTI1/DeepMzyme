"""Bounded characterization of the amended S1 selector, without changing its alpha.

The input noise models are copied from hash-verified historical simulation records,
not re-estimated from held-out data. This is a source-sealed development experiment:
it can characterize an unpublished amendment but cannot authorize training. Original
simulation files and the published Freeze 1 remain read-only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

import numpy as np

from ec_clean30 import selection as sel
from ec_clean30.provenance import REPO_ROOT, git_state, refuse_test_paths, sha256_file
from ec_clean30.simulation import (
    MAX_CHALLENGER_SHIFT, challenger_shift_for_gain, expected_ba_points, is_campaign_identity_path,
)
from ec_clean30.stats import BootstrapAbort

SPEC_FORMAT = "deepmzyme_s1_characterization_spec_v1"
SUMMARY_FORMAT = "deepmzyme_s1_characterization_summary_v1"
FROZEN_BASE_SHA256 = "28077f4c3701820c9aa20f53b570d12f8b70a46514211fc63e62976e393a3c0c"
CORNERS = ("fnL-rcL-csL-psH", "fnH-rcH-csH-psH")
EFFECTS = {
    "null": {"F3": [0, 0], "F5": [0, 0], "F4": [0, 0]},
    "f4_only_2": {"F3": [0, 0], "F5": [0, 0], "F4": [2, 2]},
    "f4_only_3": {"F3": [0, 0], "F5": [0, 0], "F4": [3, 3]},
    "mixed_f3": {"F3": [0, 2], "F5": [0, 0], "F4": [-2, -2]},
    "mixed_f5_f4": {"F3": [-2, 0], "F5": [2, 2], "F4": [1, 3]},
}
SOURCE_FILES = (
    "src/ec_clean30/s1_simulation.py", "src/ec_clean30/selection.py", "src/ec_clean30/stats.py",
    "src/ec_clean30/simulation.py", "src/ec_clean30/provenance.py", "src/ec_clean30/identity.py",
    "src/ec_clean30/campaign.py", "src/ec_clean30/logit_adjustment.py", "src/project_paths.py",
    "src/training/ec_classes.py", "src/training/ec_class_metrics.py", "src/training/ec_membership.py",
    "src/audit_sequence_remoteness.py",
)


class S1SimulationError(ValueError):
    """Input, provenance, or source identity is unsuitable for characterization."""


def json_bytes(value: Any) -> bytes:
    return (json.dumps(sel.to_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _new_or_equal(path: Path, value: Any) -> str:
    content = json_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise S1SimulationError(f"Existing {path.name} differs; never overwrite characterization evidence.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    return hashlib.sha256(content).hexdigest()


def make_spec() -> dict[str, Any]:
    return {
        "format": SPEC_FORMAT, "policy": sel.s1_policy_record(), "base_freeze1_sha256": FROZEN_BASE_SHA256,
        "simulation_seed": 20261102, "n_replicates": 200, "n_draws": 10000,
        "max_cpu_hours": 8.0, "forecast_multiplier": 1.25,
        "purpose": "Fixed-rule characterization, not alpha selection or a pass/fail calibration.",
        "limitations": ["Synthetic pooled OOF predictions; no actual model training or inner-fold fits.",
                       "Two declared corners only; not uniform error control over all noise models.",
                       "Correlated scenarios share simulated populations; intervals are not pooled.",
                       "Two LR settings and two screening seeds per family are simulated."],
        "scenarios": [{"id": f"fold{fold}__{corner}__{effect}", "fold": fold, "corner": corner,
                       "effect": effect, "gain_points_by_lr": {"F2": [0, 0], **gains}}
                      for fold in range(5) for corner in CORNERS for effect, gains in EFFECTS.items()],
    }


def validate_spec(spec: Mapping[str, Any]) -> None:
    # One immutable characterization, not a parameter search interface.
    if dict(spec) != make_spec():
        raise S1SimulationError("The specification differs from the predeclared fixed S1 characterization.")


def _safe_path(path: Path) -> None:
    refuse_test_paths([path], what="S1 characterization path")
    if is_campaign_identity_path(path):
        raise S1SimulationError("Campaign identity outputs are not characterization inputs.")


def _inside(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise S1SimulationError("An evidence reference escapes its root.")
    path = root / rel
    _safe_path(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise S1SimulationError("An evidence link escapes its root.")
    return path


def load_noise_models(base_freeze: Path, campaign_root: Path, spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Reuse exactly ten historical models, authenticated by the published receipt hash."""
    validate_spec(spec)
    for path in (base_freeze, campaign_root):
        _safe_path(path)
    if sha256_file(base_freeze) != FROZEN_BASE_SHA256:
        raise S1SimulationError("Unexpected base Freeze-1 bytes.")
    base = json.loads(base_freeze.read_text())
    bound = base["bound_files"]["simulation_sweep_receipt"]
    receipt_path = _inside(campaign_root, bound["path"])
    if sha256_file(receipt_path) != bound["sha256"]:
        raise S1SimulationError("Historical simulation receipt hash differs.")
    receipt = json.loads(receipt_path.read_text())
    models = {}
    for point in spec["scenarios"]:
        key = f"fold{point['fold']}__{point['corner']}"
        if key in models:
            continue
        point_id = (f"final-look1__fold{point['fold']}__ordinary-a0.04__looks1__"
                    f"fold{point['fold']}-corner-{point['corner']}")
        entry = receipt["records"][point_id]
        source = _inside(receipt_path.parent, entry["file"])
        if sha256_file(source) != entry["sha256"]:
            raise S1SimulationError("Historical noise record hash differs.")
        record = json.loads(source.read_text())
        if sel.sha256_canonical(record["result"]) != record["result_sha256"]:
            raise S1SimulationError("Historical simulation result hash differs.")
        if record["result_sha256"] != entry["result_sha256"] or record["smoke"]:
            raise S1SimulationError("Historical record does not match its non-smoke receipt.")
        model = sel.NullNoiseModel(**record["result"]["noise_model"])
        config = sel.PromotionConfig(**record["result"]["promotion_config"])
        if config != sel.PromotionConfig():
            raise S1SimulationError("Historical promotion settings differ from the unchanged frozen defaults.")
        models[key] = {"model": model, "source_file_sha256": entry["sha256"],
                       "source_result_sha256": entry["result_sha256"],
                       "calibration": record["result"]["calibration"]}
    return models


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def shift_for_gain(model: sel.NullNoiseModel, gain: float) -> float:
    if gain == 0:
        return 0.0
    if gain > 0:
        return challenger_shift_for_gain(model, gain)
    base = expected_ba_points(model)
    low, high = -MAX_CHALLENGER_SHIFT, 0.0
    if expected_ba_points(model, low) - base > gain:
        raise S1SimulationError("Negative effect is outside the noise model's reachable range.")
    for _ in range(100):
        middle = (low + high) / 2
        if expected_ba_points(model, middle) - base < gain:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def simulate_scenario(model: sel.NullNoiseModel, point: Mapping[str, Any], *, n_replicates: int,
                      n_draws: int = 10000, simulation_seed: int = 20261102) -> dict[str, Any]:
    """Run the real LR chooser and amended selector on paired synthetic pooled OOF tables."""
    if n_replicates < 1 or n_draws < 1:
        raise S1SimulationError("Replicates and bootstrap draws must be positive.")
    gains = point["gain_points_by_lr"]
    shifts = {gain: shift_for_gain(model, float(gain))
              for values in gains.values() for gain in values}
    shifts[0] = 0.0
    winner_counts = Counter({family: 0 for family in ("F2", "F3", "F5", "F4")})
    lr_second = Counter({family: 0 for family in sel.FAMILIES})
    harmful = nonbeneficial = aborted = 0
    selected_gains = []
    config = replace(sel.PromotionConfig(), n_draws=n_draws)
    ids = tuple(f"p{index:05d}" for index in range(model.n_proteins))
    corner_number = CORNERS.index(point["corner"])
    for replicate in range(n_replicates):
        # Per-fit substreams keep pairing intact when an effect changes wrong-prediction counts.
        children = np.random.SeedSequence([simulation_seed, point["fold"], corner_number, replicate]).spawn(25)
        rng = np.random.default_rng(children[0])
        true, components, difficulty = sel._simulated_population(rng, model)
        component_of = {pid: f"c{value}" for pid, value in zip(ids, components.tolist())}
        counts = np.bincount(true, minlength=7)
        winners, chosen_gains = {}, {}
        child_index = 1
        for family in sel.FAMILIES:
            labels = sel.s1_learning_rate_labels(family)
            settings = {}
            for lr_index, label in enumerate(labels):
                gain = gains.get(family, [0, 0])[lr_index]
                predictions = []
                for _ in sel.SCREENING_SEEDS:
                    fit_rng = np.random.default_rng(children[child_index])
                    child_index += 1
                    predictions.append(sel._simulated_fit(fit_rng, model, true, difficulty, shifts[gain]))
                settings[label] = sel.PooledOof(f"{family}/{lr_index}", ids, true, sel.SCREENING_SEEDS,
                                               sel._one_hot(np.stack(predictions)))
            choice = sel.s1_choose_learning_rate(family, settings)
            chosen_index = labels.index(choice.chosen)
            lr_second[family] += chosen_index == 1
            winners[family] = choice.winner
            chosen_gains[family] = gains.get(family, [0, 0])[chosen_index]
        try:
            result = sel.s1_family_screen(winners, component_of=component_of,
                                          training_class_counts=counts, config=config)
            winner = result.winner_family
        except BootstrapAbort:
            aborted += 1
            winner = "F2"
        winner_counts[winner] += 1
        gain = chosen_gains[winner]
        selected_gains.append(gain)
        nonbeneficial += winner != "F2" and gain <= 0
        harmful += winner != "F2" and gain < 0
    return {
        "n_replicates": n_replicates, "winner_counts": dict(winner_counts),
        "winner_rates": {key: value / n_replicates for key, value in winner_counts.items()},
        "winner_wilson_95": {key: wilson(value, n_replicates) for key, value in winner_counts.items()},
        "non_anchor_selected": n_replicates - winner_counts["F2"],
        "non_anchor_wilson_95": wilson(n_replicates - winner_counts["F2"], n_replicates),
        "selected_nonbeneficial": nonbeneficial, "selected_harmful": harmful,
        "bootstrap_aborts": aborted, "mean_selected_true_gain_points": float(np.mean(selected_gains)),
        "second_lr_selected": dict(lr_second),
    }


def _source_hashes(repo: Path) -> dict[str, str]:
    return {name: sha256_file(repo / name) for name in SOURCE_FILES}


def _source_snapshot(out: Path, repo: Path, hashes: Mapping[str, str]) -> None:
    for name, digest in hashes.items():
        target = out / "source" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write((repo / name).read_bytes())
        if sha256_file(target) != digest:
            raise S1SimulationError("Source snapshot differs from current simulation source.")


def run_characterization(*, spec_path: Path, base_freeze: Path, campaign_root: Path, out: Path,
                         first_only: bool = False, repo: Path = REPO_ROOT) -> tuple[int, dict[str, Any]]:
    for path in (spec_path, base_freeze, campaign_root, out):
        _safe_path(path)
    if out.resolve().is_relative_to((campaign_root / "overlay").resolve()):
        raise S1SimulationError("The immutable overlay cannot hold simulation outputs.")
    if out.resolve().is_relative_to((campaign_root / "simulation" / "final_v1").resolve()):
        raise S1SimulationError("Historical simulation outputs must remain unchanged.")
    if campaign_root.resolve().is_relative_to(out.resolve()):
        raise S1SimulationError("Output must be a new dedicated characterization directory.")
    spec = json.loads(spec_path.read_text())
    validate_spec(spec)
    models = load_noise_models(base_freeze, campaign_root, spec)
    hashes = _source_hashes(repo)
    spec_hash = _new_or_equal(out / "spec.json", spec)
    if spec_hash != sha256_file(spec_path):
        raise S1SimulationError("Specification must use the canonical JSON rendering.")
    _new_or_equal(out / "source_manifest.json", {"source_sha256": hashes})
    _source_snapshot(out, repo, hashes)
    source_state_path = out / "source_state.json"
    if not source_state_path.exists():
        _new_or_equal(source_state_path, {"git": git_state(repo), "source_sha256": hashes,
                                        "training_authority": False,
                                        "status": "source_sealed_development_characterization"})
    results, cpu_used = {}, 0.0
    for point in spec["scenarios"]:
        path = out / "records" / (point["id"] + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            payload = {key: value for key, value in record.items() if key != "record_sha256"}
            if (record.get("record_sha256") != sel.sha256_canonical(payload)
                    or record.get("source_sha256") != hashes or record.get("spec_sha256") != spec_hash
                    or record.get("scenario") != point):
                raise S1SimulationError("Existing scenario failed its source/specification/record check.")
            results[point["id"]] = record
            cpu_used += record["cpu_seconds"]
    for point in spec["scenarios"]:
        if point["id"] in results:
            continue
        if first_only and results:
            break
        remaining = len(spec["scenarios"]) - len(results)
        if results:
            forecast = cpu_used + spec["forecast_multiplier"] * max(r["cpu_seconds"] for r in results.values()) * remaining
            if forecast > spec["max_cpu_hours"] * 3600:
                return 4, {"status": "budget_decision_required", "completed": len(results),
                           "cpu_hours_used": cpu_used / 3600, "forecast_cpu_hours": forecast / 3600,
                           "ceiling_cpu_hours": spec["max_cpu_hours"]}
        if _source_hashes(repo) != hashes:
            raise S1SimulationError("Source changed before a scenario; stop and preserve prior evidence.")
        noise = models[f"fold{point['fold']}__{point['corner']}"]
        before_wall, before_cpu = time.monotonic(), time.process_time()
        result = simulate_scenario(noise["model"], point, n_replicates=spec["n_replicates"],
                                   n_draws=spec["n_draws"], simulation_seed=spec["simulation_seed"])
        cpu_seconds, wall_seconds = time.process_time() - before_cpu, time.monotonic() - before_wall
        if _source_hashes(repo) != hashes:
            raise S1SimulationError("Source changed during a scenario; its output is not certified.")
        record = {"scenario": point, "spec_sha256": spec_hash, "source_sha256": hashes,
                  "historical_noise_record_sha256": noise["source_file_sha256"],
                  "result": result, "cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds,
                  "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        record["record_sha256"] = sel.sha256_canonical(record)
        _new_or_equal(out / "records" / (point["id"] + ".json"), record)
        results[point["id"]] = record
        cpu_used += cpu_seconds
        print(json.dumps({"scenario": point["id"], "complete": len(results), "cpu_seconds": round(cpu_seconds, 2)}),
              flush=True)
    if len(results) != len(spec["scenarios"]):
        largest = max(record["cpu_seconds"] for record in results.values())
        return 0, {"status": "timed_first_scenario", "completed": len(results), "cpu_hours_used": cpu_used / 3600,
                   "forecast_cpu_hours": (cpu_used + spec["forecast_multiplier"] * largest *
                                          (len(spec["scenarios"]) - len(results))) / 3600}
    summary = {"format": SUMMARY_FORMAT, "status": "complete", "policy": sel.s1_policy_record(),
               "spec_sha256": spec_hash, "source_sha256": hashes, "base_freeze1_sha256": FROZEN_BASE_SHA256,
               "historical_receipt_sha256": json.loads(base_freeze.read_text())["bound_files"]
               ["simulation_sweep_receipt"]["sha256"], "cpu_hours": cpu_used / 3600,
               "wall_seconds": sum(r["wall_seconds"] for r in results.values()),
               "peak_rss_kib": max(r["peak_rss_kib"] for r in results.values()),
               "n_scenarios": len(results), "n_replicates": sum(r["result"]["n_replicates"] for r in results.values()),
               "limitations": spec["limitations"], "calibrated_error_control_claim": False,
               "results": [{"id": point["id"], "fold": point["fold"], "corner": point["corner"],
                            "effect": point["effect"], "record_sha256": results[point["id"]]["record_sha256"],
                            **results[point["id"]]["result"]} for point in spec["scenarios"]]}
    _new_or_equal(out / "summary.json", summary)
    return 0, {"status": "complete", "n_scenarios": len(results), "n_replicates": summary["n_replicates"],
               "cpu_hours": summary["cpu_hours"], "summary_sha256": sha256_file(out / "summary.json")}


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--spec", required=True)
    parser.add_argument("--freeze1", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--first-only", action="store_true", help="Time and retain the first declared scenario.")
    return parser


def action(args: argparse.Namespace) -> int:
    try:
        code, record = run_characterization(spec_path=Path(args.spec), base_freeze=Path(args.freeze1),
                                           campaign_root=Path(args.campaign_root), out=Path(args.out),
                                           first_only=args.first_only)
    except (S1SimulationError, ValueError, OSError, KeyError) as exc:
        print(f"S1 characterization refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record, indent=2, sort_keys=True))
    return code
