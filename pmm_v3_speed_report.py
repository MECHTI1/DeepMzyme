#!/usr/bin/env python3
"""Plan step B: speed report over completed step-B probes (CPU; reads campaign artifacts only).

Probes run one unit (gvp_late_fusion four_class baseline fold 0 seed 42) with the fixed settings of
pmm_v3_probes (tag -> AMP, epochs, lane, batch); the runner refuses any other combination:
  w1-fp32-r1, w1-fp32-r2       two serial FP32 runs (one after the other)
  w2-fp32-a/b, w3-fp32-a/b/c   FP32 batches started together in distinct lanes
  w1-amp-r1                    serial AMP run (AMP speed gate)
  w1-amp-r2, full-amp          only if the AMP speed gate passes (AMP agreement, AMP accuracy)
  wK-amp-*                     only if AMP passes and FP32 adopts K > 1 lanes: the combined setting
  full-fp32                    50-epoch FP32 run (AMP comparison; reused as the step C cell)
  <batch tag>-retry            one whole-batch retry, only after an invalid first attempt

Rules (plan step B as revised by the user on 2026-10-05, log v3-009):
- Speed = completed, replay-verified fits per hour, end to end: pre-admission checks, preparation,
  training, validation, independent replay and measured persistence. A probe is projected to a full
  fit from its runtime profile. Serial probes and full runs use the warm (smallest serial) preparation,
  pre-admission and outside-trainer times, so the cold first probe does not slow the serial rate; batch
  members use their own measured times, so contention between lanes is charged to the batch. A
  probe without measured persistence has incomplete timing and blocks the decision.
- Each (precision, lanes) combination is a separate candidate with its own evidence.
- Batch attempt validity (decided without looking at any result): every member ran and completed
  in its own lane, admissions within 10% of the shortest member's elapsed time, and host samples
  cover the batch window (no gap over 30 s, edges within 30 s). An invalid first attempt must be
  retried once as a whole (tags + "-retry"); the retry then decides, valid or not. A retry after a
  valid first attempt makes the report refuse. Completed probes are never rerun or discarded.
- Agreement (fixed limits, never widened): each member of the deciding attempt is within 1.0
  percentage point common-four BA and 3 points per common-four class recall of serial r1 of the same
  precision. The two serial runs must agree within the same limits; if they do not, nothing is
  decided and the report asks for diagnosis. The only pre-declared closure (step_b/diagnosis.json,
  written after a dated user decision) records serial FP32 without concurrency or AMP. Probability
  and history differences are diagnostics.
- Host pressure over the batch window: memory available never below 10%, GPU memory never above
  90% (missing GPU samples fail), median load1/CPU at most 1.0, no CPU overload (load1/CPU above 1.0)
  longer than 300 s, and in the steady training phase (every member training, the first 60 s skipped
  for the load1 lag, at least 60 s judged) overloaded at most half the time; in a 50-epoch fit that
  phase is about five times longer, so this is the check that sustained pressure cannot hide behind
  preparation and replay. Median, peak, longest and total overload are reported.
- A batch passes if valid, agreeing, without host pressure and at least 1.2x the serial fits/hour.
- AMP: fits per hour >= 1.3x FP32 serial, and the full AMP run within 1.0 common-four BA point and 3
  points per class recall of full FP32. AMP with k > 1 lanes also needs its own AMP batch.
- The decision is ready only when every required probe has an outcome with complete timing
  (full runs included), no probe is still running, no failed serial probe or full run still has its
  one rerun, and no diagnosis is pending. The fastest fully evidenced candidate is
  chosen; the report prints the set-execution command, which re-checks it, and never runs anything.

Limits of short probes: a 10-epoch probe is an early, still-changing model and its fold-0 recalls are
coarse (Cu has 73 validation ions, so one ion moves Cu recall by 1.4 points and BA by 0.34 points).
Agreement at 5 epochs is a guard against wrong inputs, settings or interference between lanes, not
evidence that concurrent and serial 50-epoch fits are equivalent; GPU training is not bitwise
deterministic, so the serial pair itself differs. Timing from short probes over-weights preparation,
start-up and replay relative to a 50-epoch fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_probes as probes_manifest  # noqa: E402

PROBE_UNIT = v3.Unit.parse(probes_manifest.PROBE_UNIT_NAME)
FULL_EPOCHS = int(v3.PROFILE["epochs"])
SHORT_EPOCHS = probes_manifest.SHORT_EPOCHS
CONCURRENCY_MIN_GAIN = 1.2
AMP_MIN_SPEEDUP = 1.3
AMP_MAX_BA_DIFFERENCE = 0.01
AMP_MAX_RECALL_DIFFERENCE = 0.03
AGREEMENT_MAX_BA_DIFFERENCE = 0.01  # fixed; never widened when the serial runs disagree
AGREEMENT_MAX_RECALL_DIFFERENCE = 0.03
MAX_START_SPREAD_FRACTION = 0.10
COLD_PREPARE_FACTOR = 1.2
MEMORY_PRESSURE = {"min_mem_available_fraction": 0.10, "max_gpu_memory_fraction": 0.90}
CPU_PRESSURE = {"overload_load_per_cpu": 1.0, "max_median_load_per_cpu": 1.0,
                "max_sustained_overload_seconds": 300.0}
CPU_TRAINING = {"max_overloaded_fraction": 0.5, "load_lag_seconds": 60.0, "min_judged_seconds": 60.0}
HOST_COVERAGE = {"max_gap_seconds": 30.0, "max_edge_seconds": 30.0}
TOLERANCE = 1e-12
BATCHES = probes_manifest.BATCHES
FULL = probes_manifest.FULL
ALL_TAGS = probes_manifest.all_tags()
HOST_SAMPLES = Path("step_b") / "host.jsonl"
REPORT = Path("step_b") / "speed_report.json"
DIAGNOSIS = Path("step_b") / "diagnosis.json"  # written only after a dated user decision in the log
GATES = {"concurrency_min_gain": CONCURRENCY_MIN_GAIN, "amp_min_speedup": AMP_MIN_SPEEDUP,
         "amp_max_ba_difference": AMP_MAX_BA_DIFFERENCE, "amp_max_recall_difference": AMP_MAX_RECALL_DIFFERENCE,
         "agreement_max_ba_difference": AGREEMENT_MAX_BA_DIFFERENCE,
         "agreement_max_recall_difference": AGREEMENT_MAX_RECALL_DIFFERENCE,
         "max_start_spread_fraction": MAX_START_SPREAD_FRACTION, "memory_pressure": MEMORY_PRESSURE,
         "cpu_pressure": CPU_PRESSURE, "cpu_training_phase": CPU_TRAINING, "host_coverage": HOST_COVERAGE,
         "short_epochs": SHORT_EPOCHS,
         "full_epochs": FULL_EPOCHS, "retry": "one whole-batch retry after an invalid first attempt"}
v3.require(probes_manifest.FULL_EPOCHS == FULL_EPOCHS, "Probe manifest and profile disagree on full epochs")


def probe_name(tag: str) -> str:
    return v3.run_name_for(PROBE_UNIT, tag)


def tag_precision(tag: str) -> str:
    return probes_manifest.probe_spec(tag)["precision"]


def _events(lane_root: Path) -> list[dict[str, Any]]:
    path = lane_root / "execution_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def persistence_seconds(lane_root: Path, run_name: str) -> dict[str, Any]:
    """Measured persistence of one unit on its actual route; None when it cannot be measured."""
    events = _events(lane_root)
    terminal = [e for e in events if e.get("event") == "unit_terminal" and e.get("run_name") == run_name]
    if not terminal:
        return {"seconds": None, "route": None, "reason": "no terminal event"}
    t0 = terminal[-1]["at_unix"]
    after = [e for e in events if e["at_unix"] >= t0 and e.get("event") in {"artifacts_verified", "host_pull_requested"}]
    if not after:
        return {"seconds": None, "route": None, "reason": "no persistence event after the unit ended"}
    first = after[0]
    if first["event"] == "artifacts_verified":
        return {"seconds": first["at_unix"] - t0, "route": "mounted"}
    ack_path = lane_root / str(first["manifest"]).replace(".json", ".ack.json")
    if not ack_path.is_file():
        return {"seconds": None, "route": "host_pull", "reason": "host acknowledgment not yet uploaded"}
    ack = json.loads(ack_path.read_text())
    if not isinstance(ack.get("verified_unix"), (int, float)):
        return {"seconds": None, "route": "host_pull", "reason": "acknowledgment lacks verified_unix"}
    return {"seconds": max(0.0, float(ack["verified_unix"]) - t0), "route": "host_pull",
            "note": "host verification time minus worker unit end (host and VM clocks assumed NTP-synced)"}


def load_probe(paths: v3.V3Paths, tag: str, statuses: dict[str, Any] | None = None) -> dict[str, Any]:
    """Outcome of one probe: untested, running, failed, or completed (replay-verified) with timing."""
    name = probe_name(tag)
    spec = probes_manifest.probe_spec(tag)
    statuses = v3.completed_units(paths) if statuses is None else statuses
    record = statuses.get(name)
    archived = len(v3.archived_attempts(paths, name))
    if record is None:
        if (paths.claims / f"{name}.json").exists():
            return {"tag": tag, "outcome": "running", "note": "claimed without a terminal status (running or "
                    "interrupted; an interrupted unit is archived with archive-failed)"}
        if archived and spec["kind"] == "batch":  # batch members are never rerun individually
            return {"tag": tag, "outcome": "failed", "status": "archived", "lane": spec["lane"],
                    "archived_attempts": archived}
        if archived:
            return {"tag": tag, "outcome": "untested", "archived_attempts": archived,
                    "note": "first attempt archived; its one unchanged rerun is pending"}
        return {"tag": tag, "outcome": "untested"}
    if record["status"] != "completed":
        return {"tag": tag, "outcome": "failed", "status": record["status"], "lane": record.get("lane"),
                "archived_attempts": archived}
    identity = record["identity"]
    v3.require(bool(identity.get("amp")) == spec["amp"], f"{name} ran with amp={identity.get('amp')}")
    v3.require(int(identity["epochs"]) == spec["epochs"],
               f"{name} has {identity['epochs']} epochs; the manifest fixes {spec['epochs']}")
    lane_root = Path(record["status_path"]).parent
    run_dir = lane_root / "runs" / name
    receipt = v3.verify_completed_unit(run_dir, identity, v3.resolve_recipe(PROBE_UNIT.recipe))
    v3.verify_independent_replay(run_dir, receipt)
    profile = json.loads((run_dir / "runtime_profile.json").read_text())
    admitted = [e for e in _events(lane_root) if e.get("event") == "unit_admitted" and e.get("run_name") == name]
    with (run_dir / "epoch_metrics.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    with (run_dir / receipt["validation_predictions"]["path"]).open(encoding="utf-8", newline="") as handle:
        predictions = {row["source_uid"]: row for row in csv.DictReader(handle)}
    return {"tag": tag, "outcome": "completed", "name": name, "lane": record["lane"],
            "epochs": int(identity["epochs"]), "amp": bool(identity.get("amp")),
            "elapsed_seconds": float(record["elapsed_seconds"]),
            "preflight_seconds": record.get("preflight_seconds"),
            "persistence": persistence_seconds(lane_root, name),
            "admitted_unix": admitted[-1]["started_unix"] if admitted else None,
            "phases": profile["phase_seconds"], "peak_rss_bytes": profile.get("process_peak_rss_bytes"),
            "cuda_peak_reserved_bytes": profile.get("cuda_peak_reserved_bytes"),
            "history": history, "predictions": predictions}


def _metric_evaluations(epochs: int) -> int:
    every = int(v3.PROFILE["train_metrics_every_n_epochs"])
    return sum(1 for epoch in range(1, epochs + 1) if epoch % every == 0 or epoch == epochs)


def projected_fit_seconds(probe: dict[str, Any], *, warm_prepare: float | None = None,
                          warm: dict[str, float] | None = None, epochs: int = FULL_EPOCHS) -> dict[str, Any]:
    """Projected full-fit seconds end to end; incomplete (None) without measured persistence or the
    recorded pre-admission time. ``warm`` caps preparation, pre-admission and outside-trainer time at the
    warm serial values (serial probes and full runs only); ``warm_prepare`` caps preparation alone."""
    warm = dict(warm or {})
    if warm_prepare is not None:
        warm["prepare"] = warm_prepare
    phases, n = probe["phases"], probe["epochs"]
    in_trainer = sum(float(value) for value in phases.values())
    prepare = float(phases.get("prepare", 0))
    if warm.get("prepare") is not None:
        prepare = min(prepare, warm["prepare"])
    per_epoch = (float(phases.get("train_epoch", 0)) + float(phases.get("validation", 0))
                 + float(phases.get("checkpoint_save", 0))) / n
    per_metric_eval = float(phases.get("train_metric_evaluation", 0)) / max(1, _metric_evaluations(n))
    outside = max(0.0, probe["elapsed_seconds"] - in_trainer)  # replay, imports, start-up
    if warm.get("outside") is not None:
        outside = min(outside, warm["outside"])
    persistence = probe["persistence"]["seconds"]
    preflight = probe.get("preflight_seconds")
    if preflight is not None and warm.get("preflight") is not None:
        preflight = min(float(preflight), warm["preflight"])
    result = {"per_epoch_seconds": per_epoch, "prepare_seconds": prepare, "outside_trainer_seconds": outside,
              "preflight_seconds": preflight, "persistence_seconds": persistence,
              "persistence_route": probe["persistence"].get("route")}
    if persistence is None:
        return {**result, "seconds": None, "complete": False,
                "reason": probe["persistence"].get("reason", "persistence not measured")}
    if preflight is None:
        return {**result, "seconds": None, "complete": False, "reason": "pre-admission time not recorded"}
    total = (float(preflight) + prepare + per_epoch * epochs + per_metric_eval * _metric_evaluations(epochs)
             + float(phases.get("selected_export", 0)) + outside + persistence)
    return {**result, "seconds": total, "complete": True}


def max_difference(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Diagnostics: largest absolute difference in validation history and terminal probabilities, and the
    number of ions whose common-four prediction differs."""
    history = 0.0
    for row_a, row_b in zip(a["history"], b["history"]):
        for key in ("val_metal_collapsed4_balanced_acc", "val_loss"):
            if row_a.get(key) not in (None, "") and row_b.get(key) not in (None, ""):
                history = max(history, abs(float(row_a[key]) - float(row_b[key])))
    v3.require(a["predictions"].keys() == b["predictions"].keys(), f"{a['tag']} and {b['tag']} cover different ions")
    probabilities, labels = 0.0, 0
    for uid, row in a["predictions"].items():
        other = b["predictions"][uid]
        labels += row["pred_common4"] != other["pred_common4"]
        for key, value in row.items():
            if key.startswith("p_"):
                probabilities = max(probabilities, abs(float(value) - float(other[key])))
    return {"history": history, "probabilities": probabilities, "changed_predictions": labels}


def common_four(probe: dict[str, Any]) -> tuple[float, dict[str, float | None]]:
    import pmm_v3_assessment as assess

    metrics = assess.prediction_metrics(list(probe["predictions"].values()), PROBE_UNIT.target)["common4"]
    return metrics["balanced_accuracy"], metrics["recall"]


def agreement(reference: dict[str, Any], member: dict[str, Any]) -> dict[str, Any]:
    """Fixed-limit agreement of one probe with its serial reference (1.0 BA point, 3 recall points)."""
    ba_ref, recall_ref = common_four(reference)
    ba, recall = common_four(member)
    recall_difference = {k: (abs(recall[k] - recall_ref[k]) if recall[k] is not None and recall_ref[k] is not None
                             else None) for k in recall_ref}
    ba_difference = abs(ba - ba_ref) if ba is not None and ba_ref is not None else None
    within = (ba_difference is not None and ba_difference <= AGREEMENT_MAX_BA_DIFFERENCE + TOLERANCE
              and all(v is not None and v <= AGREEMENT_MAX_RECALL_DIFFERENCE + TOLERANCE
                      for v in recall_difference.values()))
    return {"reference": reference["tag"], "member": member["tag"], "ba_difference": ba_difference,
            "recall_difference": recall_difference, "within_fixed_limits": within,
            "diagnostics": max_difference(reference, member)}


def read_host_samples(path: Path) -> list[dict[str, Any]] | None:
    """Host samples, or None when the file does not exist (every batch is then invalid)."""
    if not path.is_file():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def host_coverage(samples: list[dict[str, Any]] | None, start: float, end: float) -> dict[str, Any]:
    if samples is None:
        return {"complete": False, "reason": "no host sample file"}
    times = sorted(r["t"] for r in samples if start <= r["t"] <= end)
    if not times:
        return {"complete": False, "reason": "no host samples inside the batch window"}
    gaps = [b - a for a, b in zip([start, *times], [*times, end])]
    edge, gap = max(gaps[0], gaps[-1]), max(gaps[1:-1], default=0.0)
    complete = edge <= HOST_COVERAGE["max_edge_seconds"] and gap <= HOST_COVERAGE["max_gap_seconds"]
    return {"complete": complete, "samples": len(times), "largest_edge_seconds": edge, "largest_gap_seconds": gap,
            **({} if complete else {"reason": "host samples leave a gap in the batch window"})}


def _overloads(rows: list[dict[str, Any]], end: float) -> list[tuple[float, float]]:
    """(begin, stop) of each CPU overload: from the first overloaded sample until the next sample that is
    not overloaded, or the window end."""
    limit = CPU_PRESSURE["overload_load_per_cpu"]
    loads = [r["load1"] / r["cpus"] for r in rows]
    runs, index = [], 0
    while index < len(rows):
        if loads[index] > limit:
            first = index
            while index + 1 < len(rows) and loads[index + 1] > limit:
                index += 1
            runs.append((rows[first]["t"], rows[index + 1]["t"] if index + 1 < len(rows) else end))
        index += 1
    return runs


def training_phase_overload(samples: list[dict[str, Any]] | None, start: float, end: float) -> dict[str, Any]:
    """Share of the steady training phase (all lanes training, load1 lag skipped) spent CPU-overloaded."""
    start += CPU_TRAINING["load_lag_seconds"]
    if end - start < CPU_TRAINING["min_judged_seconds"]:
        return {"judged": False, "reason": f"steady training phase {max(0.0, end - start):.0f} s is too short"}
    rows = sorted((r for r in samples or [] if start <= r["t"] <= end), key=lambda r: r["t"])
    coverage = host_coverage(rows, start, end)
    if not coverage["complete"]:
        return {"judged": False, "reason": f"host samples incomplete in the training phase: {coverage['reason']}"}
    overloaded = sum(max(0.0, min(stop, end) - max(begin, start)) for begin, stop in _overloads(rows, end))
    fraction = overloaded / (end - start)
    return {"judged": True, "window": [start, end], "overloaded_fraction": fraction,
            "median_load_per_cpu": statistics.median(r["load1"] / r["cpus"] for r in rows),
            "pressure": fraction > CPU_TRAINING["max_overloaded_fraction"]}


def host_pressure(samples: list[dict[str, Any]] | None, start: float, end: float,
                  training: tuple[float, float] | None = None) -> dict[str, Any]:
    """Memory, GPU-memory and CPU pressure over one window. CPU: median, peak, longest and total overload
    over the window, plus the overloaded share of the steady training phase when ``training`` is given.
    Missing GPU-memory samples count as unmeasured (fail closed)."""
    rows = sorted((r for r in samples or [] if start <= r["t"] <= end), key=lambda r: r["t"])
    if not rows:
        return {"measured": False, "pressure": None, "reason": "no samples inside the batch's run time"}
    loads = [r["load1"] / r["cpus"] for r in rows]
    runs = [stop - begin for begin, stop in _overloads(rows, end)]
    mem = min(r["mem_available_kb"] / r["mem_total_kb"] for r in rows)
    gpu_rows = [r for r in rows if r.get("gpu_mem_total_mb")]
    gpu = max(r["gpu_mem_used_mb"] / r["gpu_mem_total_mb"] for r in gpu_rows) if gpu_rows else None
    median, longest = statistics.median(loads), max(runs, default=0.0)
    memory_pressure = (gpu is None or mem < MEMORY_PRESSURE["min_mem_available_fraction"]
                       or gpu > MEMORY_PRESSURE["max_gpu_memory_fraction"])
    phase = training_phase_overload(samples, *training) if training else {"judged": False, "reason": "not requested"}
    cpu_pressure = (median > CPU_PRESSURE["max_median_load_per_cpu"]
                    or longest > CPU_PRESSURE["max_sustained_overload_seconds"]
                    or (training is not None and (not phase["judged"] or phase["pressure"])))
    return {"measured": True, "samples": len(rows), "gpu_samples": len(gpu_rows), "window_seconds": end - start,
            "min_mem_available_fraction": mem, "max_gpu_memory_fraction": gpu,
            "median_load_per_cpu": median, "peak_load_per_cpu": max(loads),
            "longest_overload_seconds": longest, "total_overload_seconds": sum(runs), "training_phase": phase,
            "memory_pressure": memory_pressure, "cpu_pressure": cpu_pressure,
            "pressure": memory_pressure or cpu_pressure}


def _serial(probes: dict[str, Any], precision: str, warm: dict[str, float] | None) -> dict[str, Any]:
    tags = BATCHES[(precision, 1)]
    members = [probes[t] for t in tags]
    if any(p["outcome"] == "failed" for p in members):
        return {"status": "failed", "failed": [p["tag"] for p in members if p["outcome"] == "failed"]}
    completed = [p for p in members if p["outcome"] == "completed"]
    if not completed:
        return {"status": "untested"}
    out: dict[str, Any] = {"members": [p["tag"] for p in completed]}
    if len(completed) == 2:
        out["agreement"] = agreement(completed[0], completed[1])
        out["serial_disagreement"] = not out["agreement"]["within_fixed_limits"]
    projections = [projected_fit_seconds(p, warm=warm) for p in completed]
    if not all(x["complete"] for x in projections):
        return {**out, "status": "incomplete timing",
                "reasons": [x.get("reason") for x in projections if not x["complete"]]}
    return {**out, "status": "measured",
            "rate_fits_per_hour": 3600.0 / (sum(x["seconds"] for x in projections) / len(projections))}


def batch_attempt(probes: dict[str, Any], tags: tuple[str, ...], lanes: int,
                  samples: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Outcome-blind validity of one attempt: who ran, where, when, and whether the host was sampled."""
    members = [probes[t] for t in tags]
    outcomes = {p["tag"]: p["outcome"] for p in members}
    base = {"tags": list(tags), "outcomes": outcomes}
    if all(o == "untested" for o in outcomes.values()):
        return {**base, "state": "untested"}
    if any(o == "running" for o in outcomes.values()):
        return {**base, "state": "in_progress"}
    reasons = []
    if any(o == "untested" for o in outcomes.values()):
        reasons.append("not every member ran: " + ", ".join(t for t, o in outcomes.items() if o == "untested"))
    if any(o == "failed" for o in outcomes.values()):
        reasons.append("failed member(s): " + ", ".join(t for t, o in outcomes.items() if o == "failed"))
    window = None
    completed = [p for p in members if p["outcome"] == "completed"]
    if len(completed) == lanes:
        starts = [p["admitted_unix"] for p in completed]
        if None in starts:
            reasons.append("an admission time is missing")
        else:
            if len({p["lane"] for p in completed}) != lanes:
                reasons.append("members did not run in distinct lanes")
            spread = max(starts) - min(starts)
            allowed = MAX_START_SPREAD_FRACTION * min(p["elapsed_seconds"] for p in completed)
            if spread > allowed:
                reasons.append(f"admissions {spread:.1f} s apart; at most {allowed:.1f} s allowed")
            window = (min(starts), max(p["admitted_unix"] + p["elapsed_seconds"] for p in completed))
            coverage = host_coverage(samples, *window)
            if not coverage["complete"]:
                reasons.append(f"host samples incomplete: {coverage['reason']}")
    return {**base, "state": "complete", "valid": not reasons, "invalid_reasons": reasons, "window": window}


def resolve_attempts(probes: dict[str, Any], precision: str, lanes: int,
                     samples: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Pre-declared choice of the deciding attempt: the first if valid, else the one retry."""
    first = batch_attempt(probes, BATCHES[(precision, lanes)], lanes, samples)
    retry = batch_attempt(probes, probes_manifest.retry_tags(precision, lanes), lanes, samples)
    first_invalid = first["state"] == "complete" and not first["valid"]
    v3.require(retry["state"] == "untested" or first_invalid,
               f"{precision} x{lanes}: a retry ran although the first attempt is "
               f"{'valid' if first['state'] == 'complete' else first['state']}; diagnose (retry rule)")
    deciding = 2 if first_invalid else 1
    return {"first": first, "retry": retry, "deciding_attempt": deciding,
            "deciding": retry if deciding == 2 else first}


def require_retry_permitted(paths: v3.V3Paths, batch: tuple[str, int]) -> dict[str, Any]:
    """Runner guard: a whole-batch retry needs a complete, invalid first attempt."""
    precision, lanes = batch
    statuses = v3.completed_units(paths)
    probes = {tag: load_probe(paths, tag, statuses) for tag in BATCHES[(precision, lanes)]}
    first = batch_attempt(probes, BATCHES[(precision, lanes)], lanes, read_host_samples(paths.root / HOST_SAMPLES))
    v3.require(first["state"] == "complete" and not first["valid"],
               f"A retry of {precision} x{lanes} needs a complete, invalid first attempt (state {first['state']}"
               f"{', valid' if first.get('valid') else ''})")
    return first


def _training_seconds(probe: dict[str, Any]) -> float:
    phases = probe["phases"]
    return sum(float(phases.get(key, 0)) for key in ("train_epoch", "validation", "checkpoint_save",
                                                      "train_metric_evaluation"))


def _batch(probes: dict[str, Any], precision: str, lanes: int, serial: dict[str, Any],
           samples: list[dict[str, Any]] | None) -> dict[str, Any]:
    attempts = resolve_attempts(probes, precision, lanes, samples)
    decision = attempts["deciding"]
    info = {"attempts": {"first": attempts["first"], "retry": attempts["retry"]},
            "deciding_attempt": attempts["deciding_attempt"], "required_tags": decision["tags"]}
    if decision["state"] == "untested":
        return {**info, "status": "untested", "passed": False}
    if decision["state"] == "in_progress":
        return {**info, "status": "in progress", "passed": False}
    if not decision["valid"]:
        return {**info, "status": "failed", "passed": False,
                "reason": "the retry is invalid too" if attempts["deciding_attempt"] == 2 else "invalid"}
    members = [probes[t] for t in decision["tags"]]
    if serial.get("status") != "measured" or "agreement" not in serial:
        return {**info, "status": "untested", "passed": False, "reason": f"needs both serial {precision} probes"}
    if serial["serial_disagreement"]:
        return {**info, "status": "blocked: serial runs disagree beyond the fixed limits; diagnose",
                "passed": False}
    projections = [projected_fit_seconds(p) for p in members]  # own preparation: contention counts
    if not all(x["complete"] for x in projections):
        return {**info, "status": "incomplete timing", "passed": False,
                "reasons": [x.get("reason") for x in projections if not x["complete"]]}
    rate = lanes * 3600.0 / max(x["seconds"] for x in projections)
    reference = probes[BATCHES[(precision, 1)][0]]
    agreements = [agreement(reference, p) for p in members]
    # Steady training phase: every member past its preparation and before its last epoch ended.
    training = (max(p["admitted_unix"] + float(p["phases"].get("prepare", 0)) for p in members),
                min(p["admitted_unix"] + float(p["phases"].get("prepare", 0)) + _training_seconds(p) for p in members))
    pressure = host_pressure(samples, *decision["window"], training=training)
    gates = {"rate_gain_at_least_1.2x": rate >= CONCURRENCY_MIN_GAIN * serial["rate_fits_per_hour"],
             "agrees_within_fixed_limits": all(a["within_fixed_limits"] for a in agreements),
             "no_memory_pressure": pressure["measured"] and pressure["memory_pressure"] is False,  # incl. GPU
             "no_cpu_pressure": pressure["measured"] and pressure["cpu_pressure"] is False}
    passed = all(gates.values())
    return {**info, "status": "passed" if passed else "failed", "passed": passed, "rate_fits_per_hour": rate,
            "gain": rate / serial["rate_fits_per_hour"], "agreement": agreements, "host": pressure,
            "gates": gates}


def speed_report(paths: v3.V3Paths, *, host_samples: Path | None = None) -> dict[str, Any]:
    statuses = v3.completed_units(paths)
    probes = {tag: load_probe(paths, tag, statuses) for tag in ALL_TAGS}
    samples = read_host_samples(host_samples if host_samples is not None else paths.root / HOST_SAMPLES)
    completed = {t: p for t, p in probes.items() if p["outcome"] == "completed"}
    serial_tags = set(BATCHES[("fp32", 1)]) | set(BATCHES[("amp", 1)])
    warm_probes = [completed[t] for t in sorted(serial_tags) if t in completed]
    warm = {"prepare": min((float(p["phases"].get("prepare", 0)) for p in warm_probes), default=None),
            "preflight": min((float(p["preflight_seconds"]) for p in warm_probes
                              if p.get("preflight_seconds") is not None), default=None),
            "outside": min((max(0.0, p["elapsed_seconds"] - sum(float(v) for v in p["phases"].values()))
                            for p in warm_probes), default=None)}
    capped = serial_tags | set(FULL.values())
    report: dict[str, Any] = {
        "gates_definition": GATES, "report_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assessment_code_sha256": hashlib.sha256((ROOT / "pmm_v3_assessment.py").read_bytes()).hexdigest(),
        "host_samples": str(host_samples if host_samples is not None else paths.root / HOST_SAMPLES),
        "warm_caps_seconds": warm, "warm_prepare_seconds": warm["prepare"],
        "cold_prepare_probes": sorted(t for t, p in completed.items()
                                      if warm["prepare"]
                                      and float(p["phases"].get("prepare", 0)) > COLD_PREPARE_FACTOR * warm["prepare"]),
        "probes": {t: ({"outcome": p["outcome"], "lane": p["lane"], "epochs": p["epochs"], "amp": p["amp"],
                        "elapsed_seconds": p["elapsed_seconds"], "preflight_seconds": p["preflight_seconds"],
                        "persistence": p["persistence"],
                        "peak_rss_gib": (p["peak_rss_bytes"] or 0) / 2**30,
                        "cuda_peak_reserved_gib": (p["cuda_peak_reserved_bytes"] or 0) / 2**30,
                        "projection": projected_fit_seconds(p, warm=warm if t in capped else None)}
                       if p["outcome"] == "completed" else {k: v for k, v in p.items() if k != "tag"})
                   for t, p in probes.items() if p["outcome"] != "untested"}}
    serial = {prec: _serial(probes, prec, warm) for prec in ("fp32", "amp")}
    batches = {f"fp32x{k}": _batch(probes, "fp32", k, serial["fp32"], samples) for k in (2, 3)}
    candidates: dict[str, dict[str, Any]] = {}
    if serial["fp32"].get("status") == "measured" and not serial["fp32"].get("serial_disagreement"):
        candidates["fp32x1"] = {"amp": False, "lanes": 1, "rate": serial["fp32"]["rate_fits_per_hour"], "passed": True}
    for k in (2, 3):
        b = batches[f"fp32x{k}"]
        candidates[f"fp32x{k}"] = {"amp": False, "lanes": k, "rate": b.get("rate_fits_per_hour"), "passed": b["passed"],
                                   "status": b["status"]}
    fp32_lanes = max((c for c in candidates.values() if not c["amp"] and c["passed"]),
                     key=lambda c: c["rate"], default={"lanes": 1})["lanes"]
    # AMP: speed (fits per hour) against FP32 serial, accuracy on full runs, own concurrency evidence.
    amp: dict[str, Any] = {}
    p_amp, p_full_fp32, p_full_amp = probes["w1-amp-r1"], probes[FULL["fp32"]], probes[FULL["amp"]]
    if p_amp["outcome"] == "failed":
        amp["speed_gate"] = False
        amp["reason"] = "w1-amp-r1 failed"
    elif p_amp["outcome"] == "completed" and serial["fp32"].get("status") == "measured":
        amp_alone = projected_fit_seconds(p_amp, warm=warm)
        if amp_alone["complete"]:
            amp["speedup_fits_per_hour"] = (3600.0 / amp_alone["seconds"]) / serial["fp32"]["rate_fits_per_hour"]
            amp["speed_gate"] = amp["speedup_fits_per_hour"] >= AMP_MIN_SPEEDUP
    required = [*BATCHES[("fp32", 1)], *batches["fp32x2"]["required_tags"], *batches["fp32x3"]["required_tags"],
                "w1-amp-r1", FULL["fp32"]]
    if amp.get("speed_gate"):
        required += ["w1-amp-r2", FULL["amp"]]
        if p_full_fp32["outcome"] == "completed" and p_full_amp["outcome"] == "completed":
            ba_fp32, recall_fp32 = common_four(p_full_fp32)
            ba_amp, recall_amp = common_four(p_full_amp)
            recall_diff = {k: abs(recall_amp[k] - recall_fp32[k]) for k in recall_fp32}
            amp.update(ba_difference=abs(ba_amp - ba_fp32), recall_difference=recall_diff,
                       accuracy_gate=abs(ba_amp - ba_fp32) <= AMP_MAX_BA_DIFFERENCE + TOLERANCE
                       and all(v <= AMP_MAX_RECALL_DIFFERENCE + TOLERANCE for v in recall_diff.values()))
        elif "failed" in (p_full_fp32["outcome"], p_full_amp["outcome"]):
            amp["accuracy_gate"] = False
            amp["reason"] = "a full run failed, so AMP accuracy cannot be shown"
    amp_ok = bool(amp.get("speed_gate") and amp.get("accuracy_gate"))
    if amp_ok and serial["amp"].get("status") == "measured" and not serial["amp"].get("serial_disagreement"):
        candidates["ampx1"] = {"amp": True, "lanes": 1, "rate": serial["amp"]["rate_fits_per_hour"], "passed": True}
        if fp32_lanes > 1:  # the combined setting needs its own AMP batch evidence
            b = _batch(probes, "amp", fp32_lanes, serial["amp"], samples)
            required += b["required_tags"]
            batches[f"ampx{fp32_lanes}"] = b
            candidates[f"ampx{fp32_lanes}"] = {"amp": True, "lanes": fp32_lanes, "rate": b.get("rate_fits_per_hour"),
                                               "passed": b["passed"], "status": b["status"]}
    elif amp_ok:
        required += list(BATCHES[("amp", 1)])
    diagnosis = [f"serial {prec} runs disagree beyond the fixed limits" for prec in ("fp32", "amp")
                 if serial[prec].get("serial_disagreement")]
    closure = read_diagnosis(paths) if diagnosis else None
    if closure is not None:
        # The one pre-declared outcome of a diagnosis (dated user decision): serial FP32, no concurrency, no
        # AMP. It never widens a limit; probes not yet run are no longer required.
        v3.require(serial["fp32"].get("status") == "measured", "a diagnosis closure needs the serial FP32 rate")
        candidates = {"fp32x1": {"amp": False, "lanes": 1, "rate": serial["fp32"]["rate_fits_per_hour"],
                                 "passed": True, "via_diagnosis": closure}}
        required = [*BATCHES[("fp32", 1)], FULL["fp32"]]
        diagnosis = []
    report.update(serial=serial, batches=batches, amp={**amp, "passed": amp_ok}, candidates=candidates,
                  diagnosis_closure=closure)
    required = list(dict.fromkeys(required))
    untested = [t for t in required if probes[t]["outcome"] == "untested"]
    running = [t for t in required if probes[t]["outcome"] == "running"]
    failed = [t for t in required if probes[t]["outcome"] == "failed"]
    # A failed serial probe or full run still has its one unchanged rerun (archive-failed, then run).
    rerun_available = [t for t in failed if probes_manifest.probe_spec(t)["kind"] in ("serial", "full")
                       and int(probes[t].get("archived_attempts", 0)) < v3.MAX_RERUNS]
    incomplete = [t for t in required if probes[t]["outcome"] == "completed"
                  and not projected_fit_seconds(probes[t], warm=warm if t in capped else None)["complete"]]
    retry_required = [name for name, b in batches.items()
                      if b["deciding_attempt"] == 2 and b["attempts"]["retry"]["state"] == "untested"]
    report.update(required=required, untested=untested, running=running, failed=failed,
                  rerun_available=rerun_available, incomplete_timing=incomplete, retry_required=retry_required,
                  diagnosis_required=diagnosis)
    report["decision_ready"] = (not untested and not running and not incomplete and not diagnosis
                                and not rerun_available)
    if report["decision_ready"]:
        chosen = max((c for c in candidates.values() if c["passed"] and c["rate"]), key=lambda c: c["rate"], default=None)
        v3.require(chosen is not None, "no measured setting: the serial FP32 probes failed; diagnose before step C")
        precision = "amp" if chosen["amp"] else "fp32"
        full = FULL[precision]
        reuse = {PROBE_UNIT.name: probe_name(full)} if probes[full]["outcome"] == "completed" else {}
        report["chosen"] = {**chosen, "reuse_full_run": bool(reuse), "reuse": reuse}
        report["set_execution_command"] = (
            f"run_pmm_v3_campaign.py --action set-execution --campaign-dir {paths.root} "
            f"--amp {'on' if chosen['amp'] else 'off'} --lanes {chosen['lanes']} "
            f"--evidence {paths.root / REPORT}" + "".join(f" --reuse {k}={v}" for k, v in reuse.items()))
    return report


def read_diagnosis(paths: v3.V3Paths) -> dict[str, Any] | None:
    """The closure of a serial disagreement, written only after the user records the decision in the log."""
    path = paths.root / DIAGNOSIS
    if not path.is_file():
        return None
    closure = json.loads(path.read_text())
    v3.require(closure.get("outcome") == "serial_fp32", "The only diagnosis outcome is serial_fp32")
    for key in ("cause", "user_decision"):
        v3.require(isinstance(closure.get(key), str) and bool(closure[key].strip()), f"diagnosis needs {key}")
    return {key: closure[key] for key in ("outcome", "cause", "user_decision")}


def report_digest(report: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()


def verify_execution_choice(paths: v3.V3Paths, *, amp: bool, lanes: int, evidence: str,
                            reuse: dict[str, str]) -> dict[str, Any]:
    """set-execution guard: recompute the report from the artifacts; the request must equal its choice and
    the evidence file must be that report."""
    report = speed_report(paths)
    v3.require(report["decision_ready"], "The step-B report is not decision ready")
    chosen = report["chosen"]
    v3.require(bool(amp) == bool(chosen["amp"]) and lanes == chosen["lanes"],
               f"Requested amp={amp} lanes={lanes}; the report chose amp={chosen['amp']} lanes={chosen['lanes']}")
    v3.require(dict(reuse) == chosen["reuse"], f"Requested reuse {reuse}; the report names {chosen['reuse']}")
    evidence_path = Path(evidence)
    v3.require(evidence_path.resolve() == (paths.root / REPORT).resolve() and evidence_path.is_file(),
               f"--evidence must be the written report {paths.root / REPORT}")
    written = json.loads(evidence_path.read_text())
    v3.require(report_digest(written) == report_digest(json.loads(json.dumps(report, default=str))),
               "The written report differs from a recomputation; rerun the report first")
    return {"report_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "report_digest": report_digest(written), "report_code_sha256": report["report_code_sha256"]}


def sample_host(out: Path, seconds: float, interval: float) -> None:
    """Append host samples (memory, load, GPU memory) as JSON lines until ``seconds`` have passed."""
    import subprocess

    out.parent.mkdir(parents=True, exist_ok=True)
    end = time.time() + seconds
    cpus = os.cpu_count() or 1
    while time.time() < end:
        meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        row = {"t": time.time(), "cpus": cpus, "load1": os.getloadavg()[0],
               "mem_total_kb": int(meminfo["MemTotal"].split()[0]),
               "mem_available_kb": int(meminfo["MemAvailable"].split()[0])}
        try:
            used, total = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                         timeout=10).stdout.split("\n")[0].split(",")
            row.update(gpu_mem_used_mb=float(used), gpu_mem_total_mb=float(total))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
        time.sleep(interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--sample-host", action="store_true",
                        help=f"append host samples to <campaign>/{HOST_SAMPLES} instead of reporting")
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args(argv)
    paths = v3.V3Paths(args.campaign_dir)
    if args.sample_host:
        sample_host(paths.root / HOST_SAMPLES, args.seconds, args.interval)
        return 0
    report = speed_report(paths)
    out = paths.root / REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "decision_ready": report["decision_ready"], "untested": report["untested"],
                      "running": report["running"], "failed": report["failed"],
                      "incomplete_timing": report["incomplete_timing"], "retry_required": report["retry_required"],
                      "diagnosis_required": report["diagnosis_required"], "chosen": report.get("chosen"),
                      "set_execution_command": report.get("set_execution_command")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"speed report refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
