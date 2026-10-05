"""Tests for the step-B speed report (pmm_v3_speed_report.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_speed_report as speed  # noqa: E402
from test_v3_campaign import campaign, policy  # noqa: E402,F401  (fixture reuse)

FP32_BASE = ["w1-fp32-r1", "w1-fp32-r2", *speed.BATCHES[("fp32", 2)], *speed.BATCHES[("fp32", 3)], "full-fp32"]


def probe(tag, *, per_epoch=10.0, lane=0, start=1000.0, shift=0.0, recalls=(0.8, 0.8, 0.8, 0.8), prepare=60.0,
          persistence=20.0):
    """A synthetic completed probe: 30 s outside the trainer; persistence measured on the host-pull route."""
    full = tag.startswith("full-")
    epochs = speed.FULL_EPOCHS if full else 3
    phases = {"prepare": prepare, "train_epoch": per_epoch * epochs * 0.8, "validation": per_epoch * epochs * 0.2,
              "train_metric_evaluation": 5.0 * speed._metric_evaluations(epochs), "checkpoint_save": 0.0,
              "selected_export": 2.0}
    labels = ("Mn", "Cu", "Zn", "Class VIII")
    rows = {}
    for label_index, recall in enumerate(recalls):
        for k in range(10):
            pred = label_index if k < round(recall * 10) else (label_index + 1) % 4
            row = {"source_uid": f"{label_index}-{k}", "y_common4": str(label_index), "pred_common4": str(pred),
                   "y_native": str(label_index), "pred_native": str(pred)}
            for j, name in enumerate(labels):
                row[f"p_common4_{name.replace(' ', '_')}"] = f"{(0.7 if j == pred else 0.1) + shift:.8f}"
            rows[row["source_uid"]] = row
    history = [{"val_metal_collapsed4_balanced_acc": str(0.5 + 0.01 * e + shift), "val_loss": str(1.0 - 0.01 * e)}
               for e in range(epochs)]
    return {"tag": tag, "outcome": "completed", "name": speed.probe_name(tag), "lane": lane, "epochs": epochs,
            "amp": speed.tag_precision(tag) == "amp", "elapsed_seconds": sum(phases.values()) + 30.0,
            "persistence": {"seconds": persistence, "route": "host_pull"} if persistence is not None
            else {"seconds": None, "route": "host_pull", "reason": "host acknowledgment not yet uploaded"},
            "admitted_unix": start, "phases": phases, "peak_rss_bytes": 2**30, "cuda_peak_reserved_bytes": 2**30,
            "history": history, "predictions": rows}


def batch(precision, lanes, **kw):
    return {t: probe(t, lane=i, **kw) for i, t in enumerate(speed.BATCHES[(precision, lanes)])}


def fp32_base(*, w2=12.5, w3=35.0):
    """Serial FP32 at 10 s/epoch; two lanes about 1.65x the serial rate; three lanes about 1.0x."""
    return {"w1-fp32-r1": probe("w1-fp32-r1"), "w1-fp32-r2": probe("w1-fp32-r2", shift=1e-6),
            **batch("fp32", 2, per_epoch=w2), **batch("fp32", 3, per_epoch=w3), "full-fp32": probe("full-fp32")}


def install(monkeypatch, probes):
    monkeypatch.setattr(speed, "load_probe", lambda paths, tag, statuses=None:
                        probes.get(tag, {"tag": tag, "outcome": "untested"}))


def samples(path, *, mem_fraction=0.5, times=range(900, 5000, 50)):
    path.write_text("\n".join(json.dumps({"t": t, "cpus": 8, "load1": 4.0, "mem_total_kb": 1000,
                                          "mem_available_kb": mem_fraction * 1000, "gpu_mem_used_mb": 10,
                                          "gpu_mem_total_mb": 100}) for t in times) + "\n")
    return path


def paths_for(tmp_path):
    return v3.V3Paths(tmp_path / "v3")


# ---------------------------------------------------------------------------
# Timing: warm preparation and measured persistence (never zero)
# ---------------------------------------------------------------------------

def test_projection_uses_warm_preparation_and_needs_measured_persistence():
    p = probe("w1-fp32-r1", prepare=600.0)
    warm = speed.projected_fit_seconds(p, warm_prepare=60.0)
    # 60 warm prepare + 10 x 50 epochs + 5 x 5 metric evaluations + 2 export + 30 outside + 20 persistence
    assert warm["complete"] and warm["seconds"] == pytest.approx(60 + 500 + 25 + 2 + 30 + 20)
    missing = speed.projected_fit_seconds(probe("w1-fp32-r1", persistence=None))
    assert not missing["complete"] and missing["seconds"] is None and "acknowledgment" in missing["reason"]


def test_persistence_is_measured_on_the_actual_route(tmp_path):
    lane = tmp_path / "lane0"
    (lane / "persistence_receipts").mkdir(parents=True)
    name = speed.probe_name("w1-fp32-r1")
    events = [{"event": "unit_terminal", "run_name": name, "at_unix": 100.0},
              {"event": "host_pull_requested", "manifest": "persistence_receipts/t1.json", "at_unix": 101.0}]
    (lane / "execution_events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    pending = speed.persistence_seconds(lane, name)
    assert pending["seconds"] is None and pending["route"] == "host_pull"  # no acknowledgment yet: not zero
    (lane / "persistence_receipts" / "t1.ack.json").write_text(json.dumps({"verified_unix": 160.0}))
    assert speed.persistence_seconds(lane, name) == pytest.approx(
        {**speed.persistence_seconds(lane, name), "seconds": 60.0})
    mounted = tmp_path / "lane1"
    mounted.mkdir()
    (mounted / "execution_events.jsonl").write_text("\n".join(json.dumps(e) for e in (
        {"event": "unit_terminal", "run_name": name, "at_unix": 100.0},
        {"event": "artifacts_verified", "at_unix": 130.0})) + "\n")
    assert speed.persistence_seconds(mounted, name)["seconds"] == pytest.approx(30.0)


def test_incomplete_timing_blocks_the_decision(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, lane=1, persistence=None)
    install(monkeypatch, probes)
    report = speed.speed_report(paths_for(tmp_path), host_samples=samples(tmp_path / "host.jsonl"))
    assert not report["decision_ready"] and report["incomplete_timing"] == ["w2-fp32-b"]
    assert report["batches"]["fp32x2"]["status"] == "incomplete timing"


# ---------------------------------------------------------------------------
# Decision readiness: untested vs failed
# ---------------------------------------------------------------------------

def test_untested_required_probes_block_and_failed_ones_are_outcomes(monkeypatch, tmp_path):
    host = samples(tmp_path / "host.jsonl")
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}  # AMP too slow: no AMP extras
    for tag in speed.BATCHES[("fp32", 3)]:
        probes.pop(tag)
    install(monkeypatch, probes)
    report = speed.speed_report(paths_for(tmp_path), host_samples=host)
    assert not report["decision_ready"] and set(report["untested"]) == set(speed.BATCHES[("fp32", 3)])
    assert "set_execution_command" not in report
    probes.update(batch("fp32", 3, per_epoch=35.0))
    probes["w3-fp32-b"] = {"tag": "w3-fp32-b", "outcome": "failed", "status": "failed"}
    report = speed.speed_report(paths_for(tmp_path), host_samples=host)
    assert report["decision_ready"] and report["failed"] == ["w3-fp32-b"] and report["untested"] == []
    assert report["batches"]["fp32x3"]["status"] == "failed"
    assert report["chosen"]["lanes"] == 2 and not report["chosen"]["amp"]
    assert "--amp off --lanes 2" in report["set_execution_command"] and "probe-full-fp32__" in report["set_execution_command"]


# ---------------------------------------------------------------------------
# Concurrency gates
# ---------------------------------------------------------------------------

def test_concurrency_needs_gain_noise_real_overlap_and_no_pressure(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    install(monkeypatch, probes)
    paths = paths_for(tmp_path)
    report = speed.speed_report(paths, host_samples=samples(tmp_path / "host.jsonl"))
    two, three = report["batches"]["fp32x2"], report["batches"]["fp32x3"]
    assert two["passed"] and two["gain"] > 1.5
    assert not three["gates"]["rate_gain_at_least_1.2x"]
    busy = speed.speed_report(paths, host_samples=samples(tmp_path / "busy.jsonl", mem_fraction=0.05))
    assert not busy["batches"]["fp32x2"]["gates"]["no_host_pressure"] and busy["chosen"]["lanes"] == 1
    # Pressure after the batch has finished (a later batch) is not charged to it.
    end = 1000 + max(p["elapsed_seconds"] for p in batch("fp32", 2, per_epoch=12.5).values())
    late = samples(tmp_path / "late.jsonl", times=range(900, 5000, 50))
    rows = [json.loads(line) for line in late.read_text().splitlines()]
    late.write_text("\n".join(json.dumps({**r, "mem_available_kb": 10 if r["t"] > end + 1 else 500}) for r in rows) + "\n")
    assert speed.speed_report(paths, host_samples=late)["batches"]["fp32x2"]["gates"]["no_host_pressure"]
    with pytest.raises(ValueError, match="missing or empty"):
        speed.speed_report(paths, host_samples=tmp_path / "absent.jsonl")
    probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, lane=0)  # same lane: not a concurrent batch
    serial_pair = speed.speed_report(paths, host_samples=tmp_path / "host.jsonl")["batches"]["fp32x2"]
    assert not serial_pair["gates"]["ran_concurrently_in_distinct_lanes"]
    probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, lane=1, shift=1e-3)  # beyond serial noise
    assert not speed.speed_report(paths, host_samples=tmp_path / "host.jsonl")["batches"]["fp32x2"]["gates"][
        "within_serial_noise"]


def test_a_cold_first_serial_probe_does_not_inflate_the_gain(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes["w1-fp32-r1"] = probe("w1-fp32-r1", prepare=1200.0)  # built the caches
    install(monkeypatch, probes)
    report = speed.speed_report(paths_for(tmp_path), host_samples=samples(tmp_path / "host.jsonl"))
    assert report["cold_prepare_probes"] == ["w1-fp32-r1"] and report["warm_prepare_seconds"] == 60.0
    serial_seconds = 3600.0 / report["serial"]["fp32"]["rate_fits_per_hour"]
    assert serial_seconds == pytest.approx(60 + 500 + 25 + 2 + 30 + 20)


# ---------------------------------------------------------------------------
# AMP and the combined setting
# ---------------------------------------------------------------------------

def test_amp_speed_is_measured_in_fits_per_hour(monkeypatch, tmp_path):
    install(monkeypatch, {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)})
    slow = speed.speed_report(paths_for(tmp_path), host_samples=samples(tmp_path / "host.jsonl"))
    assert slow["amp"]["speed_gate"] is False and "full-amp" not in slow["required"]
    assert slow["decision_ready"] and "--amp off" in slow["set_execution_command"]


def test_amp_with_lanes_needs_its_own_amp_batch(monkeypatch, tmp_path):
    host = samples(tmp_path / "host.jsonl")
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=6.0),
              "w1-amp-r2": probe("w1-amp-r2", per_epoch=6.0, shift=1e-6), "full-amp": probe("full-amp")}
    install(monkeypatch, probes)
    paths = paths_for(tmp_path)
    pending = speed.speed_report(paths, host_samples=host)
    assert pending["amp"]["passed"] and set(speed.BATCHES[("amp", 2)]) <= set(pending["untested"])
    assert not pending["decision_ready"]  # FP32 concurrency + serial AMP do not establish AMP x 2
    probes.update(batch("amp", 2, per_epoch=7.5))
    combined = speed.speed_report(paths, host_samples=host)
    assert combined["decision_ready"] and combined["chosen"]["amp"] and combined["chosen"]["lanes"] == 2
    assert "--amp on --lanes 2" in combined["set_execution_command"]
    assert "probe-full-amp__" in combined["set_execution_command"]
    probes["w2-amp-b"] = probe("w2-amp-b", per_epoch=7.5, lane=1, shift=1e-3)  # AMP batch beyond AMP noise
    fallback = speed.speed_report(paths, host_samples=host)
    assert not fallback["candidates"]["ampx2"]["passed"]
    best = max((c for c in fallback["candidates"].values() if c["passed"]), key=lambda c: c["rate"])
    assert (fallback["chosen"]["amp"], fallback["chosen"]["lanes"]) == (best["amp"], best["lanes"])


def test_amp_accuracy_gate_rejects_a_worse_full_run(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=6.0),
              "w1-amp-r2": probe("w1-amp-r2", per_epoch=6.0),
              "full-amp": probe("full-amp", recalls=(0.8, 0.8, 0.8, 0.7))}
    install(monkeypatch, probes)
    report = speed.speed_report(paths_for(tmp_path), host_samples=samples(tmp_path / "host.jsonl"))
    assert report["amp"]["accuracy_gate"] is False and not report["amp"]["passed"]
    assert report["decision_ready"] and "--amp off" in report["set_execution_command"]


# ---------------------------------------------------------------------------
# Real artifacts, labels and the host sampler
# ---------------------------------------------------------------------------

def test_real_probes_are_read_and_mislabeled_ones_refused(campaign, tmp_path):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    common = dict(lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                  execution_policy=policy(tmp_path), load_workers=1, epochs=2)
    assert v3.run_unit(paths, speed.PROBE_UNIT, probe="w1-fp32-r1", amp=False, **common)["status"] == "completed"
    loaded = speed.load_probe(paths, "w1-fp32-r1")
    assert loaded["outcome"] == "completed" and loaded["epochs"] == 2 and loaded["admitted_unix"]
    assert loaded["persistence"]["route"] == "mounted" and loaded["persistence"]["seconds"] is not None
    assert speed.projected_fit_seconds(loaded)["complete"]
    assert speed.load_probe(paths, "w1-fp32-r2")["outcome"] == "untested"
    assert v3.run_unit(paths, speed.PROBE_UNIT, probe="w1-amp-r1", amp=False, **common)["status"] == "completed"
    with pytest.raises(ValueError, match="amp=False"):
        speed.load_probe(paths, "w1-amp-r1")


def test_the_host_sampler_creates_its_folder(tmp_path):
    out = tmp_path / "v3" / "step_b" / "host.jsonl"
    speed.sample_host(out, seconds=0.05, interval=0.01)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and {"t", "cpus", "load1", "mem_total_kb", "mem_available_kb"} <= set(rows[0])
