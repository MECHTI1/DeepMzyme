"""Tests for the step-B speed report (pmm_v3_speed_report.py), revised gates of 2026-10-05 (log v3-009)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_probes as probes_manifest  # noqa: E402
import pmm_v3_speed_report as speed  # noqa: E402
from test_v3_campaign import campaign, policy  # noqa: E402,F401  (fixture reuse)

START = 1000.0


def probe(tag, *, per_epoch=10.0, lane=None, start=START, shift=0.0, recalls=(0.8, 0.8, 0.8, 0.8), prepare=60.0,
          persistence=20.0, preflight=10.0, per_class=10):
    """A synthetic completed probe: 30 s outside the trainer; persistence measured on the host-pull route."""
    spec = probes_manifest.probe_spec(tag)
    epochs = spec["epochs"]
    phases = {"prepare": prepare, "train_epoch": per_epoch * epochs * 0.8, "validation": per_epoch * epochs * 0.2,
              "train_metric_evaluation": 5.0 * speed._metric_evaluations(epochs), "checkpoint_save": 0.0,
              "selected_export": 2.0}
    labels = ("Mn", "Cu", "Zn", "Class VIII")
    rows = {}
    for label_index, recall in enumerate(recalls):
        for k in range(per_class):
            pred = label_index if k < round(recall * per_class) else (label_index + 1) % 4
            row = {"source_uid": f"{label_index}-{k}", "y_common4": str(label_index), "pred_common4": str(pred),
                   "y_native": str(label_index), "pred_native": str(pred)}
            for j, name in enumerate(labels):
                row[f"p_common4_{name.replace(' ', '_')}"] = f"{(0.7 if j == pred else 0.1) + shift:.8f}"
            rows[row["source_uid"]] = row
    history = [{"val_metal_collapsed4_balanced_acc": str(0.5 + 0.01 * e + shift), "val_loss": str(1.0 - 0.01 * e)}
               for e in range(epochs)]
    return {"tag": tag, "outcome": "completed", "name": speed.probe_name(tag),
            "lane": spec["lane"] if lane is None else lane, "epochs": epochs, "amp": spec["amp"],
            "elapsed_seconds": sum(phases.values()) + 30.0, "preflight_seconds": preflight,
            "persistence": {"seconds": persistence, "route": "host_pull"} if persistence is not None
            else {"seconds": None, "route": "host_pull", "reason": "host acknowledgment not yet uploaded"},
            "admitted_unix": start, "phases": phases, "peak_rss_bytes": 2**30, "cuda_peak_reserved_bytes": 2**30,
            "history": history, "predictions": rows}


def batch(precision, lanes, *, retry=False, **kw):
    tags = probes_manifest.retry_tags(precision, lanes) if retry else probes_manifest.BATCHES[(precision, lanes)]
    return {t: probe(t, **kw) for t in tags}


def fp32_base(*, w2=12.5, w3=35.0):
    """Serial FP32 at 10 s/epoch; two lanes about 1.6x the serial rate; three lanes below 1.2x."""
    return {"w1-fp32-r1": probe("w1-fp32-r1"), "w1-fp32-r2": probe("w1-fp32-r2", shift=1e-6),
            **batch("fp32", 2, per_epoch=w2), **batch("fp32", 3, per_epoch=w3), "full-fp32": probe("full-fp32")}


def install(monkeypatch, probes):
    monkeypatch.setattr(speed, "load_probe", lambda paths, tag, statuses=None:
                        probes.get(tag, {"tag": tag, "outcome": "untested"}))
    monkeypatch.setattr(v3, "completed_units", lambda paths: {})


def samples(path, *, mem_fraction=0.5, load=4.0, times=None):
    times = list(range(900, 9000, 5)) if times is None else list(times)
    rows = [{"t": t, "cpus": 8, "load1": load, "mem_total_kb": 1000, "mem_available_kb": mem_fraction * 1000,
             "gpu_mem_used_mb": 10, "gpu_mem_total_mb": 100} for t in times]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return Path(path)


def paths_for(tmp_path):
    return v3.V3Paths(tmp_path / "v3")


def report(tmp_path, host):
    return speed.speed_report(paths_for(tmp_path), host_samples=host)


# ---------------------------------------------------------------------------
# Timing: end to end, warm preparation for serial probes only, measured persistence
# ---------------------------------------------------------------------------

def test_projection_is_end_to_end_and_needs_measured_persistence_and_preflight():
    p = probe("w1-fp32-r1", prepare=600.0)
    warm = speed.projected_fit_seconds(p, warm_prepare=60.0)
    # 10 preflight + 60 warm prepare + 10 x 50 epochs + 5 x 5 metric evaluations + 2 export + 30 outside + 20 persistence
    assert warm["complete"] and warm["seconds"] == pytest.approx(10 + 60 + 500 + 25 + 2 + 30 + 20)
    missing = speed.projected_fit_seconds(probe("w1-fp32-r1", persistence=None))
    assert not missing["complete"] and missing["seconds"] is None and "acknowledgment" in missing["reason"]
    no_preflight = speed.projected_fit_seconds(probe("w1-fp32-r1", preflight=None))
    assert not no_preflight["complete"] and "pre-admission" in no_preflight["reason"]


def test_a_cold_first_serial_probe_does_not_inflate_the_gain(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes["w1-fp32-r1"] = probe("w1-fp32-r1", prepare=1200.0)  # built the caches
    install(monkeypatch, probes)
    out = report(tmp_path, samples(tmp_path / "host.jsonl"))
    assert out["cold_prepare_probes"] == ["w1-fp32-r1"] and out["warm_prepare_seconds"] == 60.0
    assert 3600.0 / out["serial"]["fp32"]["rate_fits_per_hour"] == pytest.approx(10 + 60 + 500 + 25 + 2 + 30 + 20)


def test_batch_members_are_charged_their_own_contended_preparation(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes.update(batch("fp32", 2, per_epoch=12.5, prepare=400.0))  # contention during preparation
    install(monkeypatch, probes)
    two = report(tmp_path, samples(tmp_path / "host.jsonl"))["batches"]["fp32x2"]
    assert two["rate_fits_per_hour"] == pytest.approx(2 * 3600 / (10 + 400 + 12.5 * 50 + 25 + 2 + 30 + 20))


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
    assert speed.persistence_seconds(lane, name)["seconds"] == pytest.approx(60.0)


def test_incomplete_timing_blocks_the_decision(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, persistence=None)
    install(monkeypatch, probes)
    out = report(tmp_path, samples(tmp_path / "host.jsonl"))
    assert not out["decision_ready"] and out["incomplete_timing"] == ["w2-fp32-b"]
    assert out["batches"]["fp32x2"]["status"] == "incomplete timing"


# ---------------------------------------------------------------------------
# Agreement: fixed limits (1.0 BA point, 3 recall points); serial disagreement -> diagnosis
# ---------------------------------------------------------------------------

def test_agreement_uses_fixed_limits_not_the_serial_difference():
    reference = probe("w1-fp32-r1", per_class=100)
    same = probe("w2-fp32-a", per_class=100, shift=1e-3)  # probabilities differ, classes do not: agrees
    assert speed.agreement(reference, same)["within_fixed_limits"]
    assert speed.agreement(reference, same)["diagnostics"]["probabilities"] == pytest.approx(1e-3)
    edge = probe("w2-fp32-a", per_class=100, recalls=(0.8, 0.77, 0.8, 0.8))  # 3 points Cu, 0.75 BA points
    assert speed.agreement(reference, edge)["within_fixed_limits"]
    beyond = probe("w2-fp32-a", per_class=100, recalls=(0.8, 0.76, 0.8, 0.8))  # 4 points Cu
    assert not speed.agreement(reference, beyond)["within_fixed_limits"]
    ba = probe("w2-fp32-a", per_class=100, recalls=(0.78, 0.78, 0.78, 0.82))  # every class 2 points, BA 1.0 point
    assert speed.agreement(reference, ba)["ba_difference"] == pytest.approx(0.01)
    assert speed.agreement(reference, ba)["within_fixed_limits"]
    worse = probe("w2-fp32-a", per_class=100, recalls=(0.78, 0.78, 0.78, 0.81))  # BA 1.25 points
    assert not speed.agreement(reference, worse)["within_fixed_limits"]


def test_a_disagreeing_member_fails_the_batch(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, recalls=(0.8, 0.6, 0.8, 0.8))
    install(monkeypatch, probes)
    two = report(tmp_path, samples(tmp_path / "host.jsonl"))["batches"]["fp32x2"]
    assert two["status"] == "failed" and not two["gates"]["agrees_within_fixed_limits"]


def test_serial_disagreement_blocks_every_decision_and_is_never_absorbed(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes["w1-fp32-r2"] = probe("w1-fp32-r2", recalls=(0.8, 0.6, 0.8, 0.8))  # the serial pair disagrees
    probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, recalls=(0.8, 0.6, 0.8, 0.8))
    install(monkeypatch, probes)
    out = report(tmp_path, samples(tmp_path / "host.jsonl"))
    assert out["serial"]["fp32"]["serial_disagreement"] and not out["decision_ready"]
    assert out["diagnosis_required"] == ["serial fp32 runs disagree beyond the fixed limits"]
    assert out["batches"]["fp32x2"]["status"].startswith("blocked: serial runs disagree")
    assert "fp32x1" not in out["candidates"] and "set_execution_command" not in out


# ---------------------------------------------------------------------------
# Host pressure: memory, GPU memory, median and sustained CPU overload; coverage
# ---------------------------------------------------------------------------

def test_cpu_pressure_reports_median_peak_and_sustained_overload():
    rows = [{"t": t, "cpus": 8, "load1": 12.0 if 100 <= t < 250 else 4.0, "mem_total_kb": 10,
             "mem_available_kb": 5} for t in range(0, 1000, 5)]
    burst = speed.host_pressure(rows, 0, 995)
    assert burst["peak_load_per_cpu"] == 1.5 and burst["median_load_per_cpu"] == 0.5
    assert burst["longest_overload_seconds"] == pytest.approx(150) and not burst["cpu_pressure"]
    sustained = speed.host_pressure([{**r, "load1": 12.0 if 100 <= r["t"] < 450 else 4.0} for r in rows], 0, 995)
    assert sustained["longest_overload_seconds"] == pytest.approx(350) and sustained["cpu_pressure"]
    assert sustained["median_load_per_cpu"] <= 1.0  # a median alone would have hidden it
    busy = speed.host_pressure([{**r, "load1": 9.0} for r in rows], 0, 995)
    assert busy["median_load_per_cpu"] > 1.0 and busy["cpu_pressure"]
    tail = speed.host_pressure([{**r, "load1": 12.0 if r["t"] >= 700 else 4.0} for r in rows], 0, 995)
    assert tail["longest_overload_seconds"] == pytest.approx(295) and not tail["cpu_pressure"]  # to the window end
    assert tail["total_overload_seconds"] == pytest.approx(295)


def test_memory_pressure_is_separate_and_late_pressure_is_not_charged(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    install(monkeypatch, probes)
    busy = report(tmp_path, samples(tmp_path / "busy.jsonl", mem_fraction=0.05))
    gates = busy["batches"]["fp32x2"]["gates"]
    assert not gates["no_memory_pressure"] and gates["no_cpu_pressure"] and busy["chosen"]["lanes"] == 1
    end = START + max(p["elapsed_seconds"] for p in batch("fp32", 2, per_epoch=12.5).values())
    late = samples(tmp_path / "late.jsonl")
    rows = [json.loads(line) for line in late.read_text().splitlines()]
    late.write_text("\n".join(json.dumps({**r, "mem_available_kb": 10 if r["t"] > end + 1 else 500}) for r in rows) + "\n")
    assert report(tmp_path, late)["batches"]["fp32x2"]["gates"]["no_memory_pressure"]


def test_coverage_needs_samples_across_the_whole_window():
    rows = [{"t": t} for t in range(0, 1000, 5)]
    assert speed.host_coverage(rows, 10, 990)["complete"]
    assert not speed.host_coverage([r for r in rows if not 400 < r["t"] < 450], 10, 990)["complete"]
    assert not speed.host_coverage([r for r in rows if r["t"] < 900], 10, 990)["complete"]  # ends early
    assert not speed.host_coverage(None, 10, 990)["complete"]


# ---------------------------------------------------------------------------
# Batch validity and the one whole-batch retry
# ---------------------------------------------------------------------------

def test_valid_batches_decide_and_the_fastest_passing_candidate_is_chosen(monkeypatch, tmp_path):
    install(monkeypatch, {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)})
    out = report(tmp_path, samples(tmp_path / "host.jsonl"))
    two, three = out["batches"]["fp32x2"], out["batches"]["fp32x3"]
    assert two["passed"] and two["gain"] > 1.5 and two["deciding_attempt"] == 1
    assert three["status"] == "failed" and not three["gates"]["rate_gain_at_least_1.2x"]
    assert out["decision_ready"] and out["chosen"]["lanes"] == 2 and not out["chosen"]["amp"]
    assert "--amp off --lanes 2" in out["set_execution_command"] and "probe-full-fp32__" in out["set_execution_command"]
    assert out["gates_definition"]["agreement_max_recall_difference"] == 0.03


@pytest.mark.parametrize("problem", ["failed_member", "same_lane", "late_start", "no_samples", "sample_gap",
                                     "member_missing"])
def test_an_invalid_first_attempt_requires_the_retry_which_then_decides(monkeypatch, tmp_path, problem):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    retry_start = START + 5000
    times = list(range(900, 9000, 5))
    if problem == "failed_member":
        probes["w2-fp32-b"] = {"tag": "w2-fp32-b", "outcome": "failed", "status": "failed_independent_replay"}
    elif problem == "same_lane":
        probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, lane=0)
    elif problem == "late_start":
        probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, start=START + 200)
    elif problem == "no_samples":  # the sampler started only after the first attempt
        times = list(range(int(retry_start) - 100, 9000, 5))
    elif problem == "sample_gap":
        times = [t for t in times if not START + 100 < t < START + 160]
    elif problem == "member_missing":
        probes.pop("w2-fp32-b")
    install(monkeypatch, probes)
    host = samples(tmp_path / "host.jsonl", times=times)
    out = report(tmp_path, host)
    two = out["batches"]["fp32x2"]
    assert two["deciding_attempt"] == 2 and two["attempts"]["first"]["valid"] is False
    # Host coverage is judged per batch window; the fixture's 3-lane batch shares the uncovered window.
    expected = ["fp32x2", "fp32x3"] if problem in ("no_samples", "sample_gap") else ["fp32x2"]
    assert out["retry_required"] == expected and not out["decision_ready"]
    assert set(probes_manifest.retry_tags("fp32", 2)) <= set(out["untested"])
    probes.update(batch("fp32", 2, retry=True, per_epoch=12.5, start=retry_start))
    if "fp32x3" in expected:
        probes.update(batch("fp32", 3, retry=True, per_epoch=35.0, start=retry_start + 1000))
    out = report(tmp_path, host)
    two = out["batches"]["fp32x2"]
    assert two["deciding_attempt"] == 2 and two["passed"] and out["decision_ready"] and out["retry_required"] == []
    assert out["chosen"]["lanes"] == 2 and two["attempts"]["first"]["invalid_reasons"]  # first attempt kept


def test_a_retry_after_a_valid_first_attempt_is_refused(monkeypatch, tmp_path):
    install(monkeypatch, {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0),
                          **batch("fp32", 2, retry=True, per_epoch=12.5)})
    with pytest.raises(ValueError, match="retry ran although the first attempt is valid"):
        report(tmp_path, samples(tmp_path / "host.jsonl"))


def test_an_invalid_retry_fails_the_batch_without_a_third_attempt(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes["w2-fp32-b"] = {"tag": "w2-fp32-b", "outcome": "failed", "status": "failed"}
    probes.update(batch("fp32", 2, retry=True, per_epoch=12.5))
    probes["w2-fp32-b-retry"] = {"tag": "w2-fp32-b-retry", "outcome": "failed", "status": "failed"}
    install(monkeypatch, probes)
    out = report(tmp_path, samples(tmp_path / "host.jsonl"))
    two = out["batches"]["fp32x2"]
    assert two["status"] == "failed" and two["reason"] == "the retry is invalid too"
    assert out["decision_ready"] and out["chosen"]["lanes"] == 1 and out["retry_required"] == []


def test_running_members_block_decisions_and_retries(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes["w2-fp32-b"] = {"tag": "w2-fp32-b", "outcome": "running"}
    install(monkeypatch, probes)
    out = report(tmp_path, samples(tmp_path / "host.jsonl"))
    assert out["batches"]["fp32x2"]["status"] == "in progress" and out["running"] == ["w2-fp32-b"]
    assert not out["decision_ready"] and out["retry_required"] == []
    with pytest.raises(ValueError, match="complete, invalid first attempt"):
        speed.require_retry_permitted(paths_for(tmp_path), ("fp32", 2))


def test_retry_permission_follows_validity(monkeypatch, tmp_path):
    probes = fp32_base()
    install(monkeypatch, probes)
    paths = paths_for(tmp_path)
    samples(paths.root / speed.HOST_SAMPLES)
    with pytest.raises(ValueError, match="valid"):
        speed.require_retry_permitted(paths, ("fp32", 2))
    probes["w2-fp32-b"] = probe("w2-fp32-b", per_epoch=12.5, lane=0)
    assert speed.require_retry_permitted(paths, ("fp32", 2))["valid"] is False


# ---------------------------------------------------------------------------
# AMP and the combined setting
# ---------------------------------------------------------------------------

def test_amp_speed_is_measured_in_fits_per_hour(monkeypatch, tmp_path):
    install(monkeypatch, {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)})
    slow = report(tmp_path, samples(tmp_path / "host.jsonl"))
    assert slow["amp"]["speed_gate"] is False and "full-amp" not in slow["required"]
    assert slow["decision_ready"] and "--amp off" in slow["set_execution_command"]


def test_amp_with_lanes_needs_its_own_amp_batch(monkeypatch, tmp_path):
    host = samples(tmp_path / "host.jsonl")
    # AMP serial 6.5 s/epoch (1.37x FP32); the AMP pair 12 s/epoch keeps a judgeable steady training phase.
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=6.5),
              "w1-amp-r2": probe("w1-amp-r2", per_epoch=6.5, shift=1e-6), "full-amp": probe("full-amp")}
    install(monkeypatch, probes)
    pending = report(tmp_path, host)
    assert pending["amp"]["passed"] and set(probes_manifest.BATCHES[("amp", 2)]) <= set(pending["untested"])
    assert not pending["decision_ready"]  # FP32 concurrency + serial AMP do not establish AMP x 2
    probes.update(batch("amp", 2, per_epoch=12.0))
    combined = report(tmp_path, host)
    assert combined["decision_ready"] and combined["chosen"]["amp"] and combined["chosen"]["lanes"] == 2
    assert "--amp on --lanes 2" in combined["set_execution_command"]
    assert "probe-full-amp__" in combined["set_execution_command"]
    probes["w2-amp-b"] = probe("w2-amp-b", per_epoch=12.0, recalls=(0.8, 0.6, 0.8, 0.8))  # AMP batch disagrees
    fallback = report(tmp_path, host)
    assert not fallback["candidates"]["ampx2"]["passed"]
    best = max((c for c in fallback["candidates"].values() if c["passed"]), key=lambda c: c["rate"])
    assert (fallback["chosen"]["amp"], fallback["chosen"]["lanes"]) == (best["amp"], best["lanes"])


def test_amp_accuracy_gate_rejects_a_worse_full_run(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=6.0),
              "w1-amp-r2": probe("w1-amp-r2", per_epoch=6.0),
              "full-amp": probe("full-amp", recalls=(0.8, 0.8, 0.8, 0.7))}
    install(monkeypatch, probes)
    out = report(tmp_path, samples(tmp_path / "host.jsonl"))
    assert out["amp"]["accuracy_gate"] is False and not out["amp"]["passed"]
    assert out["decision_ready"] and "--amp off" in out["set_execution_command"]


# ---------------------------------------------------------------------------
# set-execution evidence
# ---------------------------------------------------------------------------

def test_set_execution_must_equal_the_recomputed_report(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    install(monkeypatch, probes)
    paths = paths_for(tmp_path)
    samples(paths.root / speed.HOST_SAMPLES)
    written = speed.speed_report(paths)
    evidence = paths.root / speed.REPORT
    evidence.write_text(json.dumps(written, indent=2, sort_keys=True, default=str) + "\n")
    reuse = written["chosen"]["reuse"]
    ok = speed.verify_execution_choice(paths, amp=False, lanes=2, evidence=str(evidence), reuse=reuse)
    assert ok["report_code_sha256"] == written["report_code_sha256"]
    with pytest.raises(ValueError, match="the report chose"):
        speed.verify_execution_choice(paths, amp=False, lanes=3, evidence=str(evidence), reuse=reuse)
    with pytest.raises(ValueError, match="reuse"):
        speed.verify_execution_choice(paths, amp=False, lanes=2, evidence=str(evidence), reuse={})
    with pytest.raises(ValueError, match="written report"):
        speed.verify_execution_choice(paths, amp=False, lanes=2, evidence=str(tmp_path / "other.json"), reuse=reuse)
    probes["w2-fp32-a"] = probe("w2-fp32-a", per_epoch=12.0)  # artifacts changed since the report was written
    with pytest.raises(ValueError, match="differs from a recomputation"):
        speed.verify_execution_choice(paths, amp=False, lanes=2, evidence=str(evidence), reuse=reuse)


# ---------------------------------------------------------------------------
# Real artifacts, labels and the host sampler
# ---------------------------------------------------------------------------

def test_real_probes_are_read_with_the_replay_policy_and_preflight(campaign, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(probes_manifest, "SHORT_EPOCHS", 2)
    train, esm_dir, paths, _, _ = campaign
    common = dict(lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                  execution_policy=policy(tmp_path), load_workers=1, epochs=2)
    assert v3.run_unit(paths, speed.PROBE_UNIT, probe="w1-fp32-r1", amp=False, **common)["status"] == "completed"
    loaded = speed.load_probe(paths, "w1-fp32-r1")
    assert loaded["outcome"] == "completed" and loaded["epochs"] == 2 and loaded["admitted_unix"]
    assert loaded["preflight_seconds"] > 0
    assert loaded["persistence"]["route"] == "mounted" and loaded["persistence"]["seconds"] is not None
    assert speed.projected_fit_seconds(loaded)["complete"]
    assert speed.load_probe(paths, "w1-fp32-r2")["outcome"] == "untested"
    with pytest.raises(ValueError, match="runs with AMP on"):  # refused before any GPU time
        v3.run_unit(paths, speed.PROBE_UNIT, probe="w1-amp-r1", amp=False, **common)
    assert speed.load_probe(paths, "w1-amp-r1")["outcome"] == "untested"


def test_the_host_sampler_writes_the_canonical_file(tmp_path):
    out = tmp_path / "v3" / speed.HOST_SAMPLES
    speed.sample_host(out, seconds=0.05, interval=0.01)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and {"t", "cpus", "load1", "mem_total_kb", "mem_available_kb"} <= set(rows[0])


# ---------------------------------------------------------------------------
# Review fixes: reruns, archived members, training-phase CPU, GPU samples, full-run timing, diagnosis
# ---------------------------------------------------------------------------

def test_a_failed_serial_or_full_probe_waits_for_its_one_rerun(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": {"tag": "w1-amp-r1", "outcome": "failed", "status": "failed"}}
    install(monkeypatch, probes)
    out = report(tmp_path, samples(tmp_path / "host.jsonl"))
    assert out["rerun_available"] == ["w1-amp-r1"] and not out["decision_ready"]
    probes["w1-amp-r1"] = {"tag": "w1-amp-r1", "outcome": "failed", "status": "failed", "archived_attempts": 1}
    final = report(tmp_path, samples(tmp_path / "host.jsonl"))  # the rerun failed too: AMP is rejected
    assert final["rerun_available"] == [] and final["decision_ready"] and final["amp"]["speed_gate"] is False
    probes["w1-amp-r1"] = probe("w1-amp-r1", per_epoch=9.0)
    probes["full-fp32"] = {"tag": "full-fp32", "outcome": "failed", "status": "failed_independent_replay"}
    assert report(tmp_path, samples(tmp_path / "host.jsonl"))["rerun_available"] == ["full-fp32"]


def test_a_full_run_without_its_acknowledgment_blocks_the_decision(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    probes["full-fp32"] = probe("full-fp32", persistence=None)
    install(monkeypatch, probes)
    out = report(tmp_path, samples(tmp_path / "host.jsonl"))
    assert out["incomplete_timing"] == ["full-fp32"] and not out["decision_ready"]


def test_archived_batch_members_count_as_failed_and_are_never_relaunched(tmp_path):
    paths = paths_for(tmp_path)
    name = speed.probe_name("w2-fp32-b")
    (paths.root / "failed_attempts" / name / "attempt1").mkdir(parents=True)
    loaded = speed.load_probe(paths, "w2-fp32-b", statuses={})
    assert loaded["outcome"] == "failed" and loaded["status"] == "archived"
    serial = speed.probe_name("w1-fp32-r2")
    (paths.root / "failed_attempts" / serial / "attempt1").mkdir(parents=True)
    assert speed.load_probe(paths, "w1-fp32-r2", statuses={})["outcome"] == "untested"  # its rerun is pending


def test_sustained_overload_during_training_fails_the_cpu_gate(monkeypatch, tmp_path):
    probes = {**fp32_base(), "w1-amp-r1": probe("w1-amp-r1", per_epoch=9.0)}
    two = batch("fp32", 2, per_epoch=40.0)  # long training phase: 1060 .. 1465 (steady from 1120)
    probes.update(two)
    install(monkeypatch, probes)
    host = tmp_path / "host.jsonl"
    rows = [{"t": t, "cpus": 8, "load1": 12.0 if 1150 <= t < 1380 else 2.0, "mem_total_kb": 1000,
             "mem_available_kb": 500, "gpu_mem_used_mb": 10, "gpu_mem_total_mb": 100} for t in range(900, 9000, 5)]
    host.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = report(tmp_path, host)["batches"]["fp32x2"]
    phase = out["host"]["training_phase"]
    assert phase["judged"] and phase["overloaded_fraction"] > 0.5 and not out["gates"]["no_cpu_pressure"]
    assert out["host"]["longest_overload_seconds"] <= 300 and out["host"]["median_load_per_cpu"] <= 1.0


def test_missing_gpu_samples_fail_the_memory_gate():
    rows = [{"t": t, "cpus": 8, "load1": 2.0, "mem_total_kb": 10, "mem_available_kb": 5} for t in range(0, 600, 5)]
    out = speed.host_pressure(rows, 0, 595)
    assert out["gpu_samples"] == 0 and out["memory_pressure"]


def test_a_short_training_phase_cannot_pass_the_cpu_gate():
    rows = [{"t": t, "cpus": 8, "load1": 2.0, "mem_total_kb": 10, "mem_available_kb": 5, "gpu_mem_used_mb": 1,
             "gpu_mem_total_mb": 10} for t in range(0, 600, 5)]
    out = speed.host_pressure(rows, 0, 595, training=(100, 200))  # 40 s left after the load1 lag
    assert not out["training_phase"]["judged"] and out["cpu_pressure"]


def test_the_only_diagnosis_closure_records_serial_fp32(monkeypatch, tmp_path):
    probes = {"w1-fp32-r1": probe("w1-fp32-r1"), "w1-fp32-r2": probe("w1-fp32-r2", recalls=(0.8, 0.6, 0.8, 0.8)),
              "full-fp32": probe("full-fp32")}
    install(monkeypatch, probes)
    paths = paths_for(tmp_path)
    samples(paths.root / speed.HOST_SAMPLES)
    blocked = speed.speed_report(paths)
    assert blocked["diagnosis_required"] and not blocked["decision_ready"]
    (paths.root / speed.DIAGNOSIS).write_text(json.dumps({"outcome": "two_lanes", "cause": "x", "user_decision": "y"}))
    with pytest.raises(ValueError, match="serial_fp32"):
        speed.speed_report(paths)
    (paths.root / speed.DIAGNOSIS).write_text(json.dumps({"outcome": "serial_fp32", "cause": "nondeterministic Cu",
                                                          "user_decision": "log v3-0xx 2026-10-07"}))
    closed = speed.speed_report(paths)
    assert closed["decision_ready"] and closed["chosen"]["lanes"] == 1 and not closed["chosen"]["amp"]
    assert closed["required"] == ["w1-fp32-r1", "w1-fp32-r2", "full-fp32"] and "probe-full-fp32__" in \
        closed["set_execution_command"]
