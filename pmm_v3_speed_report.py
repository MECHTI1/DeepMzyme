#!/usr/bin/env python3
"""Plan step B: speed report over completed step-B probes (CPU; reads campaign artifacts only).

Probe design (one unit, gvp_late_fusion four_class baseline fold 0 seed 42; tags passed to
`run_pmm_v3_campaign.py --action run --probe TAG`):
  w1-fp32-r1, w1-fp32-r2   two serial FP32 runs (lane 0, one after the other)
  w2-fp32-a, w2-fp32-b     two FP32 runs started together in lanes 0 and 1
  w3-fp32-a, -b, -c        three FP32 runs started together in lanes 0, 1 and 2
  w1-amp-r1                one serial AMP run
  full-fp32, full-amp      50-epoch runs (full-amp only if the AMP speed gate passes)
Short probes use the same --epochs value.

Rules (plan step B, defaults recorded in the plan):
- Speed = completed, replay-verified fits per hour including preparation, replay and persistence.
  A probe's full-fit time is projected from its runtime profile: preparation + per-epoch cost x 50 +
  train-metric evaluations x 5 + export, plus the measured overhead outside the trainer (replay,
  start-up) and the measured persistence time. A batch of k concurrent probes completes k fits in the
  time of its slowest member.
- Concurrency k is adopted if its rate is at least 1.2x the serial rate, the host showed no memory or
  CPU pressure while it ran (host samples), and every same-seed concurrent probe differs from serial
  r1 by no more than serial r2 differs from serial r1 (validation history and terminal predictions).
- AMP is adopted only if the AMP per-epoch time is at least 1.3x faster and the full AMP run stays
  within 1.0 common-four BA point and 3 points per common-four class recall of full FP32.
The report prints the set-execution command; it never runs anything.
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
PRESSURE = {"min_mem_available_fraction": 0.10, "max_load_per_cpu": 1.0, "max_gpu_memory_fraction": 0.90}
SHORT_TAGS = {1: ("w1-fp32-r1",), 2: ("w2-fp32-a", "w2-fp32-b"), 3: ("w3-fp32-a", "w3-fp32-b", "w3-fp32-c")}


def probe_name(tag: str) -> str:
    return v3.run_name_for(PROBE_UNIT, tag)


def _events(lane_root: Path) -> list[dict[str, Any]]:
    path = lane_root / "execution_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_probe(paths: v3.V3Paths, tag: str) -> dict[str, Any] | None:
    """A completed, replay-verified probe with its timing, or None when it is absent."""
    name = probe_name(tag)
    record = v3.completed_units(paths).get(name)
    if record is None:
        return None
    v3.require(record["status"] == "completed", f"{name} did not complete ({record['status']})")
    lane_root = Path(record["status_path"]).parent
    run_dir = lane_root / "runs" / name
    receipt = v3.verify_completed_unit(run_dir, record["identity"], v3.resolve_recipe(PROBE_UNIT.recipe))
    v3.verify_independent_replay(run_dir, receipt)
    profile = json.loads((run_dir / "runtime_profile.json").read_text())
    events = _events(lane_root)
    terminal = [e for e in events if e.get("event") == "unit_terminal" and e.get("run_name") == name]
    persisted = None
    if terminal:
        later = [e for e in events if e.get("event") == "artifacts_verified" and e["at_unix"] >= terminal[-1]["at_unix"]]
        persisted = later[0]["at_unix"] - terminal[-1]["at_unix"] if later else None
    admitted = [e for e in events if e.get("event") == "unit_admitted" and e.get("run_name") == name]
    with (run_dir / "epoch_metrics.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    with (run_dir / receipt["validation_predictions"]["path"]).open(encoding="utf-8", newline="") as handle:
        predictions = {row["source_uid"]: row for row in csv.DictReader(handle)}
    return {"tag": tag, "name": name, "lane": record["lane"], "epochs": int(record["identity"]["epochs"]),
            "amp": bool(record["identity"].get("amp")), "elapsed_seconds": float(record["elapsed_seconds"]),
            "persistence_seconds": persisted, "admitted_unix": admitted[-1]["started_unix"] if admitted else None,
            "phases": profile["phase_seconds"], "peak_rss_bytes": profile.get("process_peak_rss_bytes"),
            "cuda_peak_reserved_bytes": profile.get("cuda_peak_reserved_bytes"),
            "history": history, "predictions": predictions, "receipt_metrics": receipt["metrics"]}


def _metric_evaluations(epochs: int) -> int:
    every = int(v3.PROFILE["train_metrics_every_n_epochs"])
    return sum(1 for epoch in range(1, epochs + 1) if epoch % every == 0 or epoch == epochs)


def projected_fit_seconds(probe: dict[str, Any], *, epochs: int = FULL_EPOCHS) -> dict[str, float]:
    phases, n = probe["phases"], probe["epochs"]
    in_trainer = sum(float(value) for value in phases.values())
    per_epoch = (float(phases.get("train_epoch", 0)) + float(phases.get("validation", 0))
                 + float(phases.get("checkpoint_save", 0))) / n
    per_metric_eval = float(phases.get("train_metric_evaluation", 0)) / max(1, _metric_evaluations(n))
    outside = max(0.0, probe["elapsed_seconds"] - in_trainer)  # replay, imports, start-up
    persistence = float(probe["persistence_seconds"] or 0.0)
    total = (float(phases.get("prepare", 0)) + per_epoch * epochs + per_metric_eval * _metric_evaluations(epochs)
             + float(phases.get("selected_export", 0)) + outside + persistence)
    return {"seconds": total, "per_epoch_seconds": per_epoch, "outside_trainer_seconds": outside,
            "persistence_seconds": persistence, "persistence_measured": probe["persistence_seconds"] is not None}


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


def host_pressure(samples_path: Path | None, start: float | None, end: float | None) -> dict[str, Any]:
    """Pressure verdict from host samples (JSON lines written by --sample-host) within [start, end]."""
    if samples_path is None or not samples_path.is_file() or start is None or end is None:
        return {"measured": False, "pressure": None}
    rows = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if start <= r["t"] <= end]
    if not rows:
        return {"measured": False, "pressure": None}
    mem = min(r["mem_available_kb"] / r["mem_total_kb"] for r in rows)
    load = max(r["load1"] / r["cpus"] for r in rows)
    gpu = max((r.get("gpu_mem_used_mb", 0) / r["gpu_mem_total_mb"]) for r in rows if r.get("gpu_mem_total_mb")) \
        if any(r.get("gpu_mem_total_mb") for r in rows) else None
    pressure = (mem < PRESSURE["min_mem_available_fraction"] or load > PRESSURE["max_load_per_cpu"]
                or (gpu is not None and gpu > PRESSURE["max_gpu_memory_fraction"]))
    return {"measured": True, "samples": len(rows), "min_mem_available_fraction": mem, "max_load_per_cpu": load,
            "max_gpu_memory_fraction": gpu, "pressure": pressure}


def common_four(probe: dict[str, Any]) -> tuple[float, dict[str, float]]:
    import pmm_v3_assessment as assess

    rows = list(probe["predictions"].values())
    metrics = assess.prediction_metrics(rows, PROBE_UNIT.target)["common4"]
    return metrics["balanced_accuracy"], metrics["recall"]


def speed_report(paths: v3.V3Paths, *, host_samples: Path | None = None) -> dict[str, Any]:
    probes = {tag: load_probe(paths, tag) for tag in
              ("w1-fp32-r1", "w1-fp32-r2", *SHORT_TAGS[2], *SHORT_TAGS[3], "w1-amp-r1", "full-fp32", "full-amp")}
    present = {tag: p for tag, p in probes.items() if p is not None}
    short_epochs = {p["epochs"] for tag, p in present.items() if not tag.startswith("full")}
    v3.require(len(short_epochs) <= 1, f"short probes use different epoch counts: {sorted(short_epochs)}")
    report: dict[str, Any] = {"probes": {tag: {"lane": p["lane"], "epochs": p["epochs"], "amp": p["amp"],
                                              "elapsed_seconds": p["elapsed_seconds"],
                                              "peak_rss_gib": (p["peak_rss_bytes"] or 0) / 2**30,
                                              "cuda_peak_reserved_gib": (p["cuda_peak_reserved_bytes"] or 0) / 2**30,
                                              "projection": projected_fit_seconds(p)}
                                        for tag, p in present.items()},
                              "missing": sorted(tag for tag, p in probes.items() if p is None)}
    serial = [present[t] for t in ("w1-fp32-r1", "w1-fp32-r2") if t in present]
    concurrency: dict[str, Any] = {"adopted_lanes": 1, "candidates": {}}
    if len(serial) == 2:
        serial_rate = 3600.0 / (sum(projected_fit_seconds(p)["seconds"] for p in serial) / 2)
        noise = max_difference(serial[0], serial[1])
        concurrency["serial_rate_fits_per_hour"] = serial_rate
        concurrency["serial_vs_serial_difference"] = noise
        for lanes in (2, 3):
            batch = [present.get(tag) for tag in SHORT_TAGS[lanes]]
            if not all(batch):
                concurrency["candidates"][lanes] = {"status": "not measured"}
                continue
            slowest = max(projected_fit_seconds(p)["seconds"] for p in batch)
            rate = lanes * 3600.0 / slowest
            diffs = [max_difference(serial[0], p) for p in batch]
            within = all(d["history"] <= noise["history"] + 1e-12 and d["probabilities"] <= noise["probabilities"] + 1e-12
                         for d in diffs)
            starts = [p["admitted_unix"] for p in batch if p["admitted_unix"] is not None]
            window = (min(starts), min(starts) + slowest) if starts else (None, None)
            pressure = host_pressure(host_samples, *window)
            gates = {"rate_gain_at_least_1.2x": rate >= CONCURRENCY_MIN_GAIN * serial_rate,
                     "within_serial_noise": within,
                     "no_host_pressure": pressure["measured"] and pressure["pressure"] is False}
            concurrency["candidates"][lanes] = {"rate_fits_per_hour": rate, "gain": rate / serial_rate,
                                                "difference_to_serial": diffs, "host": pressure, "gates": gates,
                                                "passed": all(gates.values())}
        passing = [k for k, c in concurrency["candidates"].items() if c.get("passed")]
        if passing:
            concurrency["adopted_lanes"] = max(passing, key=lambda k: concurrency["candidates"][k]["rate_fits_per_hour"])
    else:
        concurrency["status"] = "needs both serial probes"
    report["concurrency"] = concurrency
    amp: dict[str, Any] = {"adopted": False}
    if "w1-fp32-r1" in present and "w1-amp-r1" in present:
        speedup = (projected_fit_seconds(present["w1-fp32-r1"])["per_epoch_seconds"]
                   / projected_fit_seconds(present["w1-amp-r1"])["per_epoch_seconds"])
        amp["per_epoch_speedup"] = speedup
        amp["speed_gate"] = speedup >= AMP_MIN_SPEEDUP
        if not amp["speed_gate"]:
            amp["next"] = "AMP rejected on speed; do not run full-amp"
        elif "full-fp32" in present and "full-amp" in present:
            ba_fp32, recall_fp32 = common_four(present["full-fp32"])
            ba_amp, recall_amp = common_four(present["full-amp"])
            recall_diff = {k: abs(recall_amp[k] - recall_fp32[k]) for k in recall_fp32}
            amp.update(ba_difference=abs(ba_amp - ba_fp32), recall_difference=recall_diff,
                       accuracy_gate=abs(ba_amp - ba_fp32) <= AMP_MAX_BA_DIFFERENCE + 1e-12
                       and all(v <= AMP_MAX_RECALL_DIFFERENCE + 1e-12 for v in recall_diff.values()))
            amp["adopted"] = bool(amp["accuracy_gate"])
        else:
            amp["next"] = "run full-fp32 and full-amp (50 epochs) for the accuracy gate"
    else:
        amp["status"] = "needs w1-fp32-r1 and w1-amp-r1"
    report["amp"] = amp
    full_tag = "full-amp" if amp["adopted"] else "full-fp32"
    ready = (concurrency.get("serial_rate_fits_per_hour") is not None and full_tag in present
             and (amp["adopted"] or amp.get("speed_gate") is False or amp.get("accuracy_gate") is False))
    report["decision_ready"] = ready
    if ready:
        report["set_execution_command"] = (
            f"run_pmm_v3_campaign.py --action set-execution --campaign-dir {paths.root} "
            f"--amp {'on' if amp['adopted'] else 'off'} --lanes {concurrency['adopted_lanes']} "
            f"--evidence {paths.root / 'step_b' / 'speed_report.json'} "
            f"--reuse {PROBE_UNIT.name}={probe_name(full_tag)}")
    return report


def sample_host(out: Path, seconds: float, interval: float) -> None:
    """Append host samples (memory, load, GPU memory) as JSON lines until ``seconds`` have passed."""
    import subprocess

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
    parser.add_argument("--seconds", type=float, default=3600)
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
    summary = {"report": str(out), "missing": report["missing"], "lanes": report["concurrency"]["adopted_lanes"],
               "amp": report["amp"], "decision_ready": report["decision_ready"],
               "set_execution_command": report.get("set_execution_command")}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"speed report refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
