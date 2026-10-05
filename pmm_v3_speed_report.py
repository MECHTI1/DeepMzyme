#!/usr/bin/env python3
"""Plan step B: speed report over completed step-B probes (CPU; reads campaign artifacts only).

Probes run one unit (gvp_late_fusion four_class baseline fold 0 seed 42) through
`run_pmm_v3_campaign.py --action run --probe TAG --amp on|off [--epochs N]`:
  w1-fp32-r1, w1-fp32-r2       two serial FP32 runs (one after the other)
  w2-fp32-a/b, w3-fp32-a/b/c   FP32 batches started together in distinct lanes
  w1-amp-r1                    serial AMP run (AMP speed gate)
  w1-amp-r2, full-amp          only if the AMP speed gate passes (AMP noise, AMP accuracy)
  wK-amp-*                     only if AMP passes and FP32 adopts K > 1 lanes: the combined setting
  full-fp32                    50-epoch FP32 run (AMP comparison; reused as the step C cell)
Short probes share one --epochs value below 50; full probes use 50 epochs.

Rules (plan step B with the user's 2026-10-05 conditions):
- Speed = completed, replay-verified fits per hour including preparation, replay and persistence.
  A probe is projected to a full fit from its runtime profile, with the warm (cached) preparation
  time; the measured persistence comes from the actual route (host-pull acknowledgment or mounted
  readback). A probe without measured persistence has incomplete timing and blocks the decision.
- Each (precision, lanes) combination is a separate candidate with its own evidence. A batch of k
  must run in k distinct lanes with near-simultaneous starts; it passes if its rate is >= 1.2x the
  serial rate of the same precision, every member agrees with serial r1 no worse than serial r2
  does (validation history and terminal probabilities), and measured host samples over the batch's
  real run time show no memory, CPU-load or GPU-memory pressure.
- AMP: fits per hour >= 1.3x FP32 serial, and the full AMP run within 1.0 common-four BA point and 3
  points per class recall of full FP32. AMP with k > 1 lanes also needs its own AMP batch.
- The decision is ready only when every required probe has an outcome (completed or failed) with
  complete timing; failed and untested probes are reported separately. The fastest fully evidenced
  candidate is chosen; the report prints the set-execution command and never runs anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_campaign as v3  # noqa: E402

PROBE_UNIT = v3.Unit("gvp_late_fusion", "four_class", "baseline", 0, 42)
FULL_EPOCHS = int(v3.PROFILE["epochs"])
CONCURRENCY_MIN_GAIN = 1.2
AMP_MIN_SPEEDUP = 1.3
AMP_MAX_BA_DIFFERENCE = 0.01
AMP_MAX_RECALL_DIFFERENCE = 0.03
MAX_START_SPREAD_FRACTION = 0.10
COLD_PREPARE_FACTOR = 1.2
PRESSURE = {"min_mem_available_fraction": 0.10, "max_load_per_cpu": 1.0, "max_gpu_memory_fraction": 0.90}
BATCHES = {("fp32", 1): ("w1-fp32-r1", "w1-fp32-r2"), ("fp32", 2): ("w2-fp32-a", "w2-fp32-b"),
           ("fp32", 3): ("w3-fp32-a", "w3-fp32-b", "w3-fp32-c"),
           ("amp", 1): ("w1-amp-r1", "w1-amp-r2"), ("amp", 2): ("w2-amp-a", "w2-amp-b"),
           ("amp", 3): ("w3-amp-a", "w3-amp-b", "w3-amp-c")}
FULL = {"fp32": "full-fp32", "amp": "full-amp"}
ALL_TAGS = tuple(dict.fromkeys([t for tags in BATCHES.values() for t in tags] + list(FULL.values())))


def probe_name(tag: str) -> str:
    return v3.run_name_for(PROBE_UNIT, tag)


def tag_precision(tag: str) -> str:
    return "amp" if "-amp" in tag else "fp32"


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
    """Outcome of one probe: untested, failed, or completed (replay-verified) with timing."""
    name = probe_name(tag)
    statuses = v3.completed_units(paths) if statuses is None else statuses
    record = statuses.get(name)
    if record is None:
        return {"tag": tag, "outcome": "untested"}
    if record["status"] != "completed":
        return {"tag": tag, "outcome": "failed", "status": record["status"]}
    identity = record["identity"]
    precision, full = tag_precision(tag), tag.startswith("full-")
    v3.require(bool(identity.get("amp")) == (precision == "amp"), f"{name} ran with amp={identity.get('amp')}")
    v3.require((int(identity["epochs"]) == FULL_EPOCHS) == full,
               f"{name} has {identity['epochs']} epochs; full probes need {FULL_EPOCHS}, short probes fewer")
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
            "persistence": persistence_seconds(lane_root, name),
            "admitted_unix": admitted[-1]["started_unix"] if admitted else None,
            "phases": profile["phase_seconds"], "peak_rss_bytes": profile.get("process_peak_rss_bytes"),
            "cuda_peak_reserved_bytes": profile.get("cuda_peak_reserved_bytes"),
            "history": history, "predictions": predictions}


def _metric_evaluations(epochs: int) -> int:
    every = int(v3.PROFILE["train_metrics_every_n_epochs"])
    return sum(1 for epoch in range(1, epochs + 1) if epoch % every == 0 or epoch == epochs)


def projected_fit_seconds(probe: dict[str, Any], *, warm_prepare: float | None = None,
                          epochs: int = FULL_EPOCHS) -> dict[str, Any]:
    """Projected full-fit seconds; incomplete (None) without measured persistence."""
    phases, n = probe["phases"], probe["epochs"]
    in_trainer = sum(float(value) for value in phases.values())
    prepare = float(phases.get("prepare", 0))
    if warm_prepare is not None:
        prepare = min(prepare, warm_prepare)
    per_epoch = (float(phases.get("train_epoch", 0)) + float(phases.get("validation", 0))
                 + float(phases.get("checkpoint_save", 0))) / n
    per_metric_eval = float(phases.get("train_metric_evaluation", 0)) / max(1, _metric_evaluations(n))
    outside = max(0.0, probe["elapsed_seconds"] - in_trainer)  # replay, imports, start-up
    persistence = probe["persistence"]["seconds"]
    result = {"per_epoch_seconds": per_epoch, "prepare_seconds": prepare, "outside_trainer_seconds": outside,
              "persistence_seconds": persistence, "persistence_route": probe["persistence"].get("route")}
    if persistence is None:
        return {**result, "seconds": None, "complete": False,
                "reason": probe["persistence"].get("reason", "persistence not measured")}
    total = (prepare + per_epoch * epochs + per_metric_eval * _metric_evaluations(epochs)
             + float(phases.get("selected_export", 0)) + outside + persistence)
    return {**result, "seconds": total, "complete": True}


def max_difference(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Largest absolute difference in validation history and terminal predicted probabilities."""
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


def read_host_samples(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    v3.require(path.is_file() and path.stat().st_size > 0, f"host samples {path} are missing or empty")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def host_pressure(samples: list[dict[str, Any]] | None, start: float, end: float) -> dict[str, Any]:
    if samples is None:
        return {"measured": False, "pressure": None}
    rows = [r for r in samples if start <= r["t"] <= end]
    if not rows:
        return {"measured": False, "pressure": None, "reason": "no samples inside the batch's run time"}
    mem = min(r["mem_available_kb"] / r["mem_total_kb"] for r in rows)
    load = max(r["load1"] / r["cpus"] for r in rows)
    gpu_rows = [r for r in rows if r.get("gpu_mem_total_mb")]
    gpu = max(r["gpu_mem_used_mb"] / r["gpu_mem_total_mb"] for r in gpu_rows) if gpu_rows else None
    pressure = (mem < PRESSURE["min_mem_available_fraction"] or load > PRESSURE["max_load_per_cpu"]
                or (gpu is not None and gpu > PRESSURE["max_gpu_memory_fraction"]))
    return {"measured": True, "samples": len(rows), "min_mem_available_fraction": mem, "max_load_per_cpu": load,
            "max_gpu_memory_fraction": gpu, "pressure": pressure}


def common_four(probe: dict[str, Any]) -> tuple[float, dict[str, float]]:
    import pmm_v3_assessment as assess

    metrics = assess.prediction_metrics(list(probe["predictions"].values()), PROBE_UNIT.target)["common4"]
    return metrics["balanced_accuracy"], metrics["recall"]


def _serial(probes: dict[str, Any], precision: str, warm: float | None) -> dict[str, Any]:
    tags = BATCHES[(precision, 1)]
    members = [probes[t] for t in tags]
    if any(p["outcome"] == "failed" for p in members):
        return {"status": "failed", "failed": [p["tag"] for p in members if p["outcome"] == "failed"]}
    completed = [p for p in members if p["outcome"] == "completed"]
    if not completed:
        return {"status": "untested"}
    projections = [projected_fit_seconds(p, warm_prepare=warm) for p in completed]
    if not all(x["complete"] for x in projections):
        return {"status": "incomplete timing", "reasons": [x.get("reason") for x in projections if not x["complete"]]}
    out = {"status": "measured", "rate_fits_per_hour": 3600.0 / (sum(x["seconds"] for x in projections) / len(projections)),
           "members": [p["tag"] for p in completed]}
    if len(completed) == 2:
        out["noise"] = max_difference(completed[0], completed[1])
    return out


def _batch(probes: dict[str, Any], precision: str, lanes: int, serial: dict[str, Any], warm: float | None,
           samples: list[dict[str, Any]] | None) -> dict[str, Any]:
    members = [probes[t] for t in BATCHES[(precision, lanes)]]
    if any(p["outcome"] == "failed" for p in members):
        return {"status": "failed", "passed": False, "failed": [p["tag"] for p in members if p["outcome"] == "failed"]}
    if any(p["outcome"] == "untested" for p in members):
        return {"status": "untested", "passed": False, "untested": [p["tag"] for p in members if p["outcome"] == "untested"]}
    if serial.get("status") != "measured" or "noise" not in serial:
        return {"status": "untested", "passed": False, "reason": f"needs both serial {precision} probes measured"}
    projections = [projected_fit_seconds(p, warm_prepare=warm) for p in members]
    if not all(x["complete"] for x in projections):
        return {"status": "incomplete timing", "passed": False,
                "reasons": [x.get("reason") for x in projections if not x["complete"]]}
    starts = [p["admitted_unix"] for p in members]
    concurrent = (None not in starts and len({p["lane"] for p in members}) == lanes
                  and max(starts) - min(starts) <= MAX_START_SPREAD_FRACTION * min(p["elapsed_seconds"] for p in members))
    rate = lanes * 3600.0 / max(x["seconds"] for x in projections)
    reference = probes[BATCHES[(precision, 1)][0]]
    diffs = [max_difference(reference, p) for p in members]
    noise = serial["noise"]
    within = all(d["history"] <= noise["history"] + 1e-12 and d["probabilities"] <= noise["probabilities"] + 1e-12
                 for d in diffs)
    window = (min(starts), max(p["admitted_unix"] + p["elapsed_seconds"] for p in members)) if None not in starts else (0, -1)
    pressure = host_pressure(samples, *window)
    gates = {"ran_concurrently_in_distinct_lanes": concurrent,
             "rate_gain_at_least_1.2x": rate >= CONCURRENCY_MIN_GAIN * serial["rate_fits_per_hour"],
             "within_serial_noise": within,
             "no_host_pressure": pressure["measured"] and pressure["pressure"] is False}
    passed = all(gates.values())
    return {"status": "passed" if passed else "failed", "passed": passed, "rate_fits_per_hour": rate,
            "gain": rate / serial["rate_fits_per_hour"], "difference_to_serial": diffs, "host": pressure,
            "gates": gates}


def speed_report(paths: v3.V3Paths, *, host_samples: Path | None = None) -> dict[str, Any]:
    statuses = v3.completed_units(paths)
    probes = {tag: load_probe(paths, tag, statuses) for tag in ALL_TAGS}
    samples = read_host_samples(host_samples)
    completed = {t: p for t, p in probes.items() if p["outcome"] == "completed"}
    short_epochs = {p["epochs"] for t, p in completed.items() if not t.startswith("full-")}
    v3.require(len(short_epochs) <= 1, f"short probes use different epoch counts: {sorted(short_epochs)}")
    prepares = [float(p["phases"].get("prepare", 0)) for t, p in completed.items() if not t.startswith("full-")]
    warm = min(prepares) if prepares else None
    report: dict[str, Any] = {
        "warm_prepare_seconds": warm,
        "cold_prepare_probes": sorted(t for t, p in completed.items()
                                      if warm and float(p["phases"].get("prepare", 0)) > COLD_PREPARE_FACTOR * warm),
        "probes": {t: ({"outcome": p["outcome"], "lane": p["lane"], "epochs": p["epochs"], "amp": p["amp"],
                        "elapsed_seconds": p["elapsed_seconds"], "persistence": p["persistence"],
                        "peak_rss_gib": (p["peak_rss_bytes"] or 0) / 2**30,
                        "cuda_peak_reserved_gib": (p["cuda_peak_reserved_bytes"] or 0) / 2**30,
                        "projection": projected_fit_seconds(p, warm_prepare=warm)}
                       if p["outcome"] == "completed" else {k: v for k, v in p.items() if k != "tag"})
                   for t, p in probes.items()}}
    serial = {prec: _serial(probes, prec, warm) for prec in ("fp32", "amp")}
    batches = {f"fp32x{k}": _batch(probes, "fp32", k, serial["fp32"], warm, samples) for k in (2, 3)}
    candidates: dict[str, dict[str, Any]] = {}
    if serial["fp32"].get("status") == "measured":
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
        amp_alone = projected_fit_seconds(p_amp, warm_prepare=warm)
        if amp_alone["complete"]:
            amp["speedup_fits_per_hour"] = (3600.0 / amp_alone["seconds"]) / serial["fp32"]["rate_fits_per_hour"]
            amp["speed_gate"] = amp["speedup_fits_per_hour"] >= AMP_MIN_SPEEDUP
    required = ["w1-fp32-r1", "w1-fp32-r2", *BATCHES[("fp32", 2)], *BATCHES[("fp32", 3)], "w1-amp-r1", FULL["fp32"]]
    if amp.get("speed_gate"):
        required += ["w1-amp-r2", FULL["amp"]]
        if p_full_fp32["outcome"] == "completed" and p_full_amp["outcome"] == "completed":
            ba_fp32, recall_fp32 = common_four(p_full_fp32)
            ba_amp, recall_amp = common_four(p_full_amp)
            recall_diff = {k: abs(recall_amp[k] - recall_fp32[k]) for k in recall_fp32}
            amp.update(ba_difference=abs(ba_amp - ba_fp32), recall_difference=recall_diff,
                       accuracy_gate=abs(ba_amp - ba_fp32) <= AMP_MAX_BA_DIFFERENCE + 1e-12
                       and all(v <= AMP_MAX_RECALL_DIFFERENCE + 1e-12 for v in recall_diff.values()))
        elif "failed" in (p_full_fp32["outcome"], p_full_amp["outcome"]):
            amp["accuracy_gate"] = False
            amp["reason"] = "a full run failed, so AMP accuracy cannot be shown"
    amp_ok = bool(amp.get("speed_gate") and amp.get("accuracy_gate"))
    if amp_ok and serial["amp"].get("status") == "measured":
        candidates["ampx1"] = {"amp": True, "lanes": 1, "rate": serial["amp"]["rate_fits_per_hour"], "passed": True}
        if fp32_lanes > 1:  # the combined setting needs its own AMP batch evidence
            required += list(BATCHES[("amp", fp32_lanes)])
            b = _batch(probes, "amp", fp32_lanes, serial["amp"], warm, samples)
            batches[f"ampx{fp32_lanes}"] = b
            candidates[f"ampx{fp32_lanes}"] = {"amp": True, "lanes": fp32_lanes, "rate": b.get("rate_fits_per_hour"),
                                               "passed": b["passed"], "status": b["status"]}
    elif amp_ok:
        required += list(BATCHES[("amp", 1)])
    report.update(serial=serial, batches=batches, amp={**amp, "passed": amp_ok}, candidates=candidates)
    required = list(dict.fromkeys(required))
    untested = [t for t in required if probes[t]["outcome"] == "untested"]
    failed = [t for t in required if probes[t]["outcome"] == "failed"]
    incomplete = [t for t in required if probes[t]["outcome"] == "completed" and not t.startswith("full-")
                  and not projected_fit_seconds(probes[t], warm_prepare=warm)["complete"]]
    report.update(required=required, untested=untested, failed=failed, incomplete_timing=incomplete)
    report["decision_ready"] = not untested and not incomplete
    if report["decision_ready"]:
        chosen = max((c for c in candidates.values() if c["passed"] and c["rate"]), key=lambda c: c["rate"], default=None)
        v3.require(chosen is not None, "no measured setting: the serial FP32 probes failed; diagnose before step C")
        precision = "amp" if chosen["amp"] else "fp32"
        full = FULL[precision]
        reuse = (f" --reuse {PROBE_UNIT.name}={probe_name(full)}" if probes[full]["outcome"] == "completed" else "")
        report["chosen"] = {**chosen, "reuse_full_run": bool(reuse)}
        report["set_execution_command"] = (
            f"run_pmm_v3_campaign.py --action set-execution --campaign-dir {paths.root} "
            f"--amp {'on' if chosen['amp'] else 'off'} --lanes {chosen['lanes']} "
            f"--evidence {paths.root / 'step_b' / 'speed_report.json'}{reuse}")
    return report


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
        time.sleep(interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--host-samples", type=Path, help="JSON lines written by --sample-host during the probes")
    parser.add_argument("--sample-host", type=Path, help="write host samples to this file instead of reporting")
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args(argv)
    paths = v3.V3Paths(args.campaign_dir)
    if args.sample_host:
        sample_host(args.sample_host, args.seconds, args.interval)
        return 0
    report = speed_report(paths, host_samples=args.host_samples)
    out = paths.root / "step_b" / "speed_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "decision_ready": report["decision_ready"], "untested": report["untested"],
                      "failed": report["failed"], "incomplete_timing": report["incomplete_timing"],
                      "chosen": report.get("chosen"), "set_execution_command": report.get("set_execution_command")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"speed report refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
