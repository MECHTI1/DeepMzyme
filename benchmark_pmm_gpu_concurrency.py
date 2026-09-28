"""Bounded, disposable two-process GPU throughput probe; never a campaign fit.

Uses trusted, audited normalized graph snapshots read-only. The first version
supports the frozen ordinary GVP recipe only. No accuracy, model selection,
checkpoint promotion, MPS, cache construction or provider operation occurs.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "pmm-gpu-concurrency-probe-v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def check_options(args):
    if not 1 <= args.steps <= 1000 or not 1 <= args.warmup <= 100:
        raise ValueError("Require 1..1000 measured steps and 1..100 warmup steps")
    if not 2 <= args.batches <= 16 or not 1 <= args.threads <= 4:
        raise ValueError("Require 2..16 input batches and 1..4 CPU threads per worker")
    if not math.isfinite(args.max_seconds) or not 10 <= args.max_seconds <= 900:
        raise ValueError("Require 10..900 seconds total bounded probe time")


def select_batches(batches, count):
    full = [i for i, batch in enumerate(batches) if batch.num_graphs == 16]
    if len(full) < count:
        raise ValueError("Not enough complete batch-size-16 inputs")
    return [full[round(i * (len(full) - 1) / (count - 1))] for i in range(count)]


def prepare_plan(args):
    """CPU-only content checks. Execute this entry point from the frozen checkout."""
    import torch
    from audit_pmm_replay import descriptor, file_sha, graph_hash, stable_hash, verify_source

    check_options(args)
    prepared = args.prepared_dir.resolve()
    manifest_path = prepared / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "predictive_inputs_matched":
        raise ValueError("An audited matched input snapshot is required")
    identity = manifest["campaign_run_identity"]
    if identity["family"] != "only_gvp" or identity["readout"] != "none":
        raise ValueError("This initial probe only supports ordinary Only-GVP")
    source = verify_source(identity["source_tree_sha256"])
    checkpoint_path = args.checkpoint.resolve()
    if checkpoint_path.parent.parent != args.campaign_root.resolve() / "runs":
        raise ValueError("Checkpoint must belong to this campaign's runs directory; lock identity cannot differ")
    checkpoint_hashes = [v for k, v in manifest["input_files"].items()
                         if Path(k).name == "best_model_checkpoint.pt"]
    if checkpoint_hashes != [file_sha(checkpoint_path)]:
        raise ValueError("Checkpoint differs from audited snapshot identity")
    snapshot_path = prepared / "input_snapshot.pt"
    if file_sha(snapshot_path) != manifest["snapshot_sha256"]:
        raise ValueError("Input snapshot changed")
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if snapshot["schema"] != manifest["schema"] or [graph_hash(b) for b in snapshot["batches"]] != [
            b["fresh_sha256"] for b in manifest["batches"]]:
        raise ValueError("Snapshot batch content/order differs from audited manifest")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if json.loads(config["campaign_run_identity"]) != identity:
        raise ValueError("Checkpoint scientific identity differs from snapshot")
    if (config["batch_size"] != 16 or config["grad_accum_steps"] != 1 or config["use_amp"]
            or config["deterministic"] or config.get("gvp_learning_rate") is not None
            or config["lr_schedule"] != "fixed" or config["task"] != "metal"
            or config.get("run_test_eval") or config.get("test_structure_dir")):
        raise ValueError("Unsupported recipe; probe must preserve frozen FP32 GVP settings")
    if stable_hash(descriptor(checkpoint["normalization_stats"])) != manifest["normalization_sha256"]:
        raise ValueError("Checkpoint normalization differs from audited inputs")
    indices = select_batches(snapshot["batches"], args.batches)
    for protected in (prepared, checkpoint_path.parent, args.campaign_root.resolve() / "runs",
                      args.campaign_root.resolve() / "raw_graph_cache"):
        output = args.output_dir.resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Probe output overlaps protected scientific inputs")
    return dict(schema=SCHEMA, status="planned", certifies_fit=False, diagnostic_only=True,
                source_tree_sha256=source, tool_sha256=file_sha(__file__),
                input_manifest_sha256=file_sha(manifest_path),
                snapshot_sha256=manifest["snapshot_sha256"], checkpoint_sha256=checkpoint_hashes[0],
                checkpoint=str(checkpoint_path), snapshot=str(snapshot_path),
                prepared_dir=str(prepared), campaign_root=str(args.campaign_root.resolve()),
                batch_indices=indices, batch_sha256=[graph_hash(snapshot["batches"][i]) for i in indices],
                input_partition="development validation; disposable performance-only optimizer steps",
                no_validation_metrics=True, no_promotable_checkpoint=True,
                steps=args.steps, warmup=args.warmup, threads=args.threads, max_seconds=args.max_seconds,
                recipe={k: config[k] for k in ("learning_rate", "weight_decay", "grad_clip_norm",
                                               "batch_size", "use_amp", "deterministic")},
                state_comparison_atol=1e-5, state_comparison_rtol=1e-4,
                limitations=["Only GVP graph microsteps; not ESMC or late-fusion training",
                             "No graph preparation, full resident dataset, replay or backup cost",
                             "Two CPU threads per worker by default; production thread count may differ",
                             "One serial/concurrent pair; not an end-to-end speedup or accuracy claim",
                             "Numerical comparison tolerance is diagnostic, never a replay gate"])


@contextlib.contextmanager
def campaign_lock(campaign_root):
    root = Path(campaign_root)
    if not root.is_dir():
        raise ValueError("Existing campaign root is required")
    with (root / "execution.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "execution_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("active_child") or state.get("pending_transfer") or (
                state.get("pending_unit") and not state["pending_unit"].get("persisted")):
            raise ValueError("Campaign ownership or persistence remains unresolved")
        # Children inherit this same lock FD; the single-worker rule is not bypassed.
        yield lock.fileno()


def set_child_lifetime(parent_pid, deadline):
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot enforce Linux parent-death signal")
    if os.getppid() != parent_pid:
        raise RuntimeError("Probe coordinator exited before worker initialization")
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Probe deadline expired")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Probe deadline")))
    signal.setitimer(signal.ITIMER_REAL, remaining)


def optimize_steps(model, batches, optimizer, *, steps, device, clip):
    import torch

    losses = []
    model.train()
    for i in range(steps):
        batch = batches[i % len(batches)].clone().to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch)["loss"]
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Non-finite diagnostic loss")
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    set_child_lifetime(spec["parent_pid"], spec["deadline"])
    inherited = os.fstat(spec["lock_fd"])
    expected_lock = (Path(spec["campaign_root"]) / "execution.lock").stat()
    if (inherited.st_dev, inherited.st_ino) != (expected_lock.st_dev, expected_lock.st_ino):
        raise ValueError("Worker did not inherit this campaign's ownership lock")
    import random
    import resource
    import numpy as np
    import torch
    from audit_pmm_replay import descriptor, file_sha, graph_hash, stable_hash, verify_source
    from training.campaign_runtime import load_campaign_prediction_components

    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(False)
    verify_source(spec["source_tree_sha256"])
    if file_sha(spec["checkpoint"]) != spec["checkpoint_sha256"] or file_sha(spec["snapshot"]) != spec["snapshot_sha256"]:
        raise ValueError("Worker inputs changed after planning")
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    snapshot = torch.load(spec["snapshot"], map_location="cpu", weights_only=False)
    batches = [snapshot["batches"][i] for i in spec["batch_indices"]]
    del snapshot
    if [graph_hash(b) for b in batches] != spec["batch_sha256"]:
        raise ValueError("Worker selected batches changed")
    device = "cuda:0"
    torch.cuda.set_device(device)
    model, _, _ = load_campaign_prediction_components(checkpoint, device=device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def reset():
        model.load_state_dict(initial, strict=True)
        random.seed(spec["seed"])
        np.random.seed(spec["seed"])
        torch.manual_seed(spec["seed"])
        torch.cuda.manual_seed_all(spec["seed"])
        return torch.optim.AdamW(model.parameters(), lr=spec["recipe"]["learning_rate"],
                                 weight_decay=spec["recipe"]["weight_decay"])

    optimizer = reset()
    optimize_steps(model, batches, optimizer, steps=spec["warmup"], device=device,
                   clip=spec["recipe"]["grad_clip_norm"])
    del optimizer
    optimizer = reset()  # Warmup cannot alter the measured initial condition.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    output = Path(spec["worker_output"])
    write_json(output / "ready.json", {"pid": os.getpid(), "initial_state_sha256": stable_hash(descriptor(initial))})
    while not Path(spec["go_file"]).exists():
        time.sleep(0.02)
    torch.cuda.synchronize()
    started = time.monotonic()
    losses = optimize_steps(model, batches, optimizer, steps=spec["steps"], device=device,
                            clip=spec["recipe"]["grad_clip_norm"])
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise FloatingPointError("Non-finite diagnostic model state")
    # A flat diagnostic vector has no checkpoint config, tensor names or optimizer.
    vector = torch.cat([v.reshape(-1).double() for k, v in sorted(state.items())])
    torch.save(vector, output / "disposable_state_vector.pt")
    report = dict(status="complete", pid=os.getpid(), seed=spec["seed"], seconds=elapsed,
                  measured_steps=spec["steps"], examples=spec["steps"] * 16,
                  initial_state_sha256=stable_hash(descriptor(initial)),
                  final_state_sha256=stable_hash(descriptor(state)),
                  input_hashes_before=spec["batch_sha256"],
                  input_hashes_after=[graph_hash(b) for b in batches],
                  finite_losses=True, losses=losses,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), cpu_threads=torch.get_num_threads(),
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    write_json(output / "result.json", report)


def terminate_processes(processes):
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def wait_until(predicate, processes, deadline, telemetry=None):
    next_sample = 0.0
    while not predicate():
        if time.time() >= deadline:
            raise TimeoutError("Probe total deadline expired")
        failed = [p.returncode for p in processes if p.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"Probe subprocess failed: {failed}")
        if telemetry is not None and time.monotonic() >= next_sample:
            result = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            telemetry.write(result.stdout)
            telemetry.flush()
            next_sample = time.monotonic() + 1
        time.sleep(0.05)


def compare_vectors(left, right, *, atol, rtol):
    import torch

    if left.shape != right.shape or not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("Diagnostic vectors have invalid shape or non-finite values")
    delta = (left - right).abs()
    return {"allclose": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
            "bitwise_equal": bool(torch.equal(left, right)), "max_absolute_difference": float(delta.max()),
            "rms_difference": float(delta.square().mean().sqrt()), "atol": atol, "rtol": rtol,
            "comparison_role": "diagnostic only; never scientific replay certification"}


def execute(plan, output):
    import torch
    from audit_pmm_replay import file_sha

    if output.exists():
        raise FileExistsError("Probe output must be new")
    with campaign_lock(plan["campaign_root"]) as lock_fd:
        existing = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, check=True, timeout=5)
        if existing.stdout.strip():
            raise ValueError("GPU compute processes already exist; reconcile before probing")
        if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") or os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
            raise ValueError("Initial probe does not support MPS")
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
        deadline = time.time() + plan["max_seconds"]
        all_processes, reports, phases = [], {}, {}
        started = time.monotonic()
        old_handlers = {}

        def interrupted(signum, _frame):
            raise InterruptedError(f"Probe interrupted by signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        try:
            with (output / "gpu_telemetry.csv").open("w") as telemetry:
                for phase, workers in (("serial_a", [0]), ("serial_b", [1]), ("concurrent", [0, 1])):
                    phase_start = time.monotonic()
                    children, dirs = [], []
                    go = output / f"{phase}.go"
                    for index in workers:
                        directory = output / f"{phase}_{index}"
                        directory.mkdir()
                        dirs.append(directory)
                        spec = dict(plan, parent_pid=os.getpid(), deadline=deadline, seed=42 + index, lock_fd=lock_fd,
                                    worker_output=str(directory), go_file=str(go))
                        write_json(directory / "worker_spec.json", spec)
                        env = dict(os.environ, OMP_NUM_THREADS=str(plan["threads"]), MKL_NUM_THREADS=str(plan["threads"]))
                        with (directory / "worker.log").open("w") as log:
                            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-spec",
                                                      str(directory / "worker_spec.json")], stdout=log,
                                                     stderr=subprocess.STDOUT, env=env,
                                                     start_new_session=True, pass_fds=(lock_fd,))
                        children.append(child)
                        all_processes.append(child)
                    wait_until(lambda: all((d / "ready.json").exists() for d in dirs), children, deadline, telemetry)
                    ready = time.monotonic()
                    go.touch()
                    wait_until(lambda: all(p.poll() is not None for p in children), children, deadline, telemetry)
                    if any(p.returncode for p in children):
                        raise RuntimeError("Probe worker exited unsuccessfully")
                    phases[phase] = {"launch_to_finish_seconds": time.monotonic() - phase_start,
                                     "barrier_to_finish_seconds": time.monotonic() - ready}
                    for index, directory in zip(workers, dirs):
                        result = json.loads((directory / "result.json").read_text())
                        if result["input_hashes_after"] != result["input_hashes_before"]:
                            raise ValueError("Diagnostic mutated input tensors")
                        reports[f"{phase}_{index}"] = result
            comparisons = {}
            for index, serial in ((0, "serial_a"), (1, "serial_b")):
                names = [f"{serial}_{index}", f"concurrent_{index}"]
                if reports[names[0]]["initial_state_sha256"] != reports[names[1]]["initial_state_sha256"]:
                    raise ValueError("Serial/concurrent initial weights differ")
                arrays = [torch.load(output / name / "disposable_state_vector.pt", weights_only=True) for name in names]
                comparisons[str(index)] = compare_vectors(*arrays, atol=plan["state_comparison_atol"],
                                                         rtol=plan["state_comparison_rtol"])
            serial_seconds = reports["serial_a_0"]["seconds"] + reports["serial_b_1"]["seconds"]
            concurrent_seconds = max(reports[f"concurrent_{i}"]["seconds"] for i in (0, 1))
            report = dict(plan, status="diagnostic_complete", workers=reports, phases=phases,
                          elapsed_seconds=time.monotonic() - started, state_comparisons=comparisons,
                          timing_warning="Microstep throughput excludes preparation and shutdown; concurrent window uses maximum worker time with a common start barrier",
                          measured_step_speedup=serial_seconds / concurrent_seconds,
                          serial_steps_per_second=2 * plan["steps"] / serial_seconds,
                          concurrent_steps_per_second=2 * plan["steps"] / concurrent_seconds,
                          process_launch_wall_speedup=(phases["serial_a"]["launch_to_finish_seconds"] +
                                                      phases["serial_b"]["launch_to_finish_seconds"]) /
                                                     phases["concurrent"]["launch_to_finish_seconds"],
                          concurrent_peak_reserved_upper_bound_bytes=sum(reports[f"concurrent_{i}"]["peak_reserved_bytes"] for i in (0, 1)),
                          concurrent_peak_rss_upper_bound_bytes=sum(reports[f"concurrent_{i}"]["peak_rss_bytes"] for i in (0, 1)),
                          production_parallelism_authorized=False,
                          caution="Sum of individual peaks is an upper bound; small snapshot RSS excludes full training datasets")
            for path in output.glob("*/disposable_state_vector.pt"):
                path.unlink()  # Report numerical comparisons, never retain disposable trained weights.
            report["original_input_hashes_unchanged"] = (
                file_sha(plan["snapshot"]) == plan["snapshot_sha256"] and
                file_sha(plan["checkpoint"]) == plan["checkpoint_sha256"])
            if not report["original_input_hashes_unchanged"]:
                raise ValueError("Original scientific input changed during probe")
            write_json(output / "report.json", report)
            return report
        except BaseException as exc:
            write_json(output / "failure.json", {"schema": SCHEMA, "status": "failed",
                                                "error": repr(exc), "deadline": deadline})
            raise
        finally:
            terminate_processes(all_processes)
            for path in output.glob("*/disposable_state_vector.pt"):
                path.unlink()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", action="store_true", help="Verify CPU inputs and print plan; no GPU or output writes")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.worker_spec:
        worker(args.worker_spec)
        return
    if any(getattr(args, key) is None for key in ("prepared_dir", "checkpoint", "campaign_root", "output_dir")):
        parser.error("--prepared-dir, --checkpoint, --campaign-root and --output-dir are required")
    plan = prepare_plan(args)
    result = plan if args.plan else execute(plan, args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
