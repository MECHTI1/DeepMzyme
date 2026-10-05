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


def probe(tag, *, per_epoch=10.0, epochs=3, lane=0, amp=False, start=1000.0, shift=0.0, recalls=(0.8, 0.8, 0.8, 0.8)):
    """A synthetic probe: 60 s preparation, per-epoch cost, 30 s outside the trainer, 20 s persistence."""
    phases = {"prepare": 60.0, "train_epoch": per_epoch * epochs * 0.8, "validation": per_epoch * epochs * 0.2,
              "train_metric_evaluation": 5.0, "checkpoint_save": 0.0, "selected_export": 2.0}
    labels = ("Mn", "Cu", "Zn", "Class VIII")
    rows = {}
    for label_index, recall in enumerate(recalls):
        for k in range(10):  # 10 ions per class; recall decides how many are predicted correctly
            correct = k < round(recall * 10)
            pred = label_index if correct else (label_index + 1) % 4
            row = {"source_uid": f"{label_index}-{k}", "y_common4": str(label_index), "pred_common4": str(pred),
                   "y_native": str(label_index), "pred_native": str(pred)}
            for j, name in enumerate(labels):
                row[f"p_common4_{name.replace(' ', '_')}"] = f"{(0.7 if j == pred else 0.1) + shift:.8f}"
            rows[row["source_uid"]] = row
    history = [{"val_metal_collapsed4_balanced_acc": str(0.5 + 0.01 * e + shift), "val_loss": str(1.0 - 0.01 * e)}
               for e in range(epochs)]
    return {"tag": tag, "name": speed.probe_name(tag), "lane": lane, "epochs": epochs, "amp": amp,
            "elapsed_seconds": sum(phases.values()) + 30.0, "persistence_seconds": 20.0, "admitted_unix": start,
            "phases": phases, "peak_rss_bytes": 2**30, "cuda_peak_reserved_bytes": 2**30, "history": history,
            "predictions": rows, "receipt_metrics": {}}


def install(monkeypatch, probes):
    monkeypatch.setattr(speed, "load_probe", lambda paths, tag: probes.get(tag))


def samples(path, *, mem_fraction=0.5, load=0.5):
    path.write_text("\n".join(json.dumps({"t": t, "cpus": 8, "load1": load * 8, "mem_total_kb": 1000,
                                          "mem_available_kb": mem_fraction * 1000, "gpu_mem_used_mb": 10,
                                          "gpu_mem_total_mb": 100}) for t in range(900, 5000, 50)))
    return path


def test_projection_scales_epochs_and_keeps_overheads():
    p = probe("w1-fp32-r1", per_epoch=10.0, epochs=3)
    projected = speed.projected_fit_seconds(p)
    # 60 prepare + 10 x 50 epochs + 5 x 5 metric evaluations + 2 export + 30 outside + 20 persistence
    assert projected["seconds"] == pytest.approx(60 + 500 + 25 + 2 + 30 + 20)
    assert projected["per_epoch_seconds"] == pytest.approx(10.0)


def test_concurrency_needs_gain_noise_and_no_pressure(monkeypatch, tmp_path):
    probes = {"w1-fp32-r1": probe("w1-fp32-r1"), "w1-fp32-r2": probe("w1-fp32-r2", shift=1e-6),
              # two lanes at 1.25x per-fit time: 2 / 1.25 = 1.6x the serial rate
              "w2-fp32-a": probe("w2-fp32-a", per_epoch=12.5, lane=0), "w2-fp32-b": probe("w2-fp32-b", per_epoch=12.5, lane=1),
              # three lanes at 3.5x the per-epoch time: about 1.0x the serial rate after overheads
              **{t: probe(t, per_epoch=35.0, lane=i) for i, t in enumerate(speed.SHORT_TAGS[3])}}
    install(monkeypatch, probes)
    paths = v3.V3Paths(tmp_path / "v3")
    report = speed.speed_report(paths, host_samples=samples(tmp_path / "host.jsonl"))
    two, three = report["concurrency"]["candidates"][2], report["concurrency"]["candidates"][3]
    assert two["passed"] and two["gain"] > 1.5
    assert not three["gates"]["rate_gain_at_least_1.2x"]
    assert report["concurrency"]["adopted_lanes"] == 2
    pressured = speed.speed_report(paths, host_samples=samples(tmp_path / "busy.jsonl", mem_fraction=0.05))
    assert not pressured["concurrency"]["candidates"][2]["gates"]["no_host_pressure"]
    assert pressured["concurrency"]["adopted_lanes"] == 1
    unmeasured = speed.speed_report(paths)  # no host samples: pressure is unknown, so not adopted
    assert unmeasured["concurrency"]["adopted_lanes"] == 1
    probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, lane=1, shift=1e-3)  # beyond serial noise
    noisy = speed.speed_report(paths, host_samples=tmp_path / "host.jsonl")
    assert not noisy["concurrency"]["candidates"][2]["gates"]["within_serial_noise"]


def test_amp_needs_speed_and_full_run_accuracy(monkeypatch, tmp_path):
    base = {"w1-fp32-r1": probe("w1-fp32-r1"), "w1-fp32-r2": probe("w1-fp32-r2"),
            "full-fp32": probe("full-fp32", epochs=50)}
    paths = v3.V3Paths(tmp_path / "v3")
    install(monkeypatch, {**base, "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0, amp=True)})
    slow = speed.speed_report(paths)
    assert slow["amp"]["speed_gate"] is False and not slow["amp"]["adopted"]
    assert slow["decision_ready"] and "--amp off" in slow["set_execution_command"]
    assert "probe-full-fp32__" in slow["set_execution_command"]
    fast = {**base, "w1-amp-r1": probe("w1-amp-r1", per_epoch=6.0, amp=True)}
    install(monkeypatch, fast)
    pending = speed.speed_report(paths)
    assert pending["amp"]["speed_gate"] and not pending["decision_ready"]
    install(monkeypatch, {**fast, "full-amp": probe("full-amp", epochs=50, amp=True, recalls=(0.8, 0.8, 0.8, 0.7))})
    worse = speed.speed_report(paths)  # Class VIII recall 10 points lower: rejected
    assert worse["amp"]["accuracy_gate"] is False and "--amp off" in worse["set_execution_command"]
    install(monkeypatch, {**fast, "full-amp": probe("full-amp", epochs=50, amp=True)})
    adopted = speed.speed_report(paths)
    assert adopted["amp"]["adopted"] and "--amp on" in adopted["set_execution_command"]
    assert "probe-full-amp__" in adopted["set_execution_command"]


def test_a_real_probe_is_read_from_its_artifacts(campaign, tmp_path):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    result = v3.run_unit(paths, speed.PROBE_UNIT, lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable,
                         device="cpu", execution_policy=policy(tmp_path), load_workers=1, epochs=2,
                         probe="w1-fp32-r1", amp=False)
    assert result["status"] == "completed", result
    loaded = speed.load_probe(paths, "w1-fp32-r1")
    assert loaded["epochs"] == 2 and loaded["persistence_seconds"] is not None and loaded["admitted_unix"]
    assert loaded["phases"]["train_epoch"] > 0 and loaded["predictions"]
    assert speed.projected_fit_seconds(loaded)["seconds"] > loaded["elapsed_seconds"]
    report = speed.speed_report(paths)
    assert "w1-fp32-r2" in report["missing"] and report["concurrency"]["adopted_lanes"] == 1
