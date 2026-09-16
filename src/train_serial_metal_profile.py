"""Profile one serial campaign fit while preserving the ordinary training CLI."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time


def _option(argv, name):
    value = None
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith(name + "="):
            value = token.split("=", 1)[1]
    return value


def profile_training(entrypoint, argv=None, *, torch_module=None):
    """Call the unmodified entrypoint once; record partial profiles on failure."""
    argv = list(sys.argv[1:] if argv is None else argv)
    started_epoch, started_clock = time.time(), time.perf_counter()
    runs_dir, run_name = _option(argv, "--runs-dir"), _option(argv, "--run-name")
    directory = Path(runs_dir) / run_name if runs_dir and run_name else None
    record = {"status": "started", "started_epoch": started_epoch,
              "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
              "prepare_seconds": None, "setup_seconds": None, "training_seconds": None}
    cuda_device = _option(argv, "--device") or "cpu"
    cuda_profiled = False
    try:
        if torch_module is None:
            import torch as torch_module
        try:
            if cuda_device.startswith("cuda") and torch_module.cuda.is_available():
                torch_module.cuda.reset_peak_memory_stats(cuda_device)
                cuda_profiled = True
        except Exception as exc:
            record["cuda_profile_error"] = str(exc)
        result = entrypoint()
        record["status"] = "completed"
        return result
    except BaseException as exc:
        record.update(status="failed", exception_type=type(exc).__name__, exception=str(exc))
        if isinstance(exc, SystemExit):
            record["exit_code"] = exc.code
        raise
    finally:
        record.update(ended_epoch=time.time(), elapsed_seconds=max(0., time.perf_counter() - started_clock))
        record["total_seconds"] = record["elapsed_seconds"]
        if cuda_profiled:
            try:
                record["cuda_peak_allocated_bytes"] = int(torch_module.cuda.max_memory_allocated(cuda_device))
                record["cuda_peak_reserved_bytes"] = int(torch_module.cuda.max_memory_reserved(cuda_device))
            except Exception as exc:
                record["cuda_profile_error"] = str(exc)
        try:
            if directory is not None:
                status_path = directory / "prepare_status.json"
                if status_path.is_file():
                    status = json.loads(status_path.read_text())
                    ready_epoch = status_path.stat().st_mtime
                    if status.get("status") == "ready" and started_epoch <= ready_epoch <= record["ended_epoch"]:
                        setup = min(record["elapsed_seconds"], ready_epoch - started_epoch)
                        record.update(prepare_seconds=setup, setup_seconds=setup,
                                      training_seconds=max(0., record["elapsed_seconds"] - setup),
                                      preparation_timing_source="same_process_prepare_status_ready_mtime")
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "performance_profile.json"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                temporary.replace(path)
        except Exception as exc:
            # A profiling problem must not conceal the original training error.
            print(f"Unable to complete performance profile: {exc}", file=sys.stderr)


def main():
    def train_entrypoint():
        from train import main as train_main
        return train_main()
    return profile_training(train_entrypoint)


if __name__ == "__main__":
    main()
