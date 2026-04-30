from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from training.config import parse_args
from training.run import validate_training_configuration


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_help(script_path: Path) -> str:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [PYTHON, str(script_path), "--help"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def check_training_cli_help() -> None:
    help_text = run_help(REPO_ROOT / "src" / "train.py")
    expected_options = (
        "--deterministic",
        "--metal-loss-weight",
        "--ec-loss-weight",
        "--allow-train-loss-test-eval-debug",
    )
    missing = [option for option in expected_options if option not in help_text]
    if missing:
        raise AssertionError(f"Training CLI help is missing expected options: {missing}")


def check_test_eval_safety() -> None:
    unsafe_config = parse_args(
        [
            "--task",
            "metal",
            "--test-structure-dir",
            "/tmp/deepmzyme_missing_test_structures",
            "--test-summary-csv",
            "/tmp/deepmzyme_missing_test_summary.csv",
            "--run-test-eval",
        ]
    )
    try:
        validate_training_configuration(unsafe_config)
    except ValueError as exc:
        message = str(exc)
        if "--run-test-eval is for held-out reporting" not in message:
            raise AssertionError(f"Unsafe test-eval config failed with an unexpected error: {message}") from exc
    else:
        raise AssertionError("Unsafe test-eval config without validation was not rejected.")

    debug_config = parse_args(
        [
            "--task",
            "metal",
            "--test-structure-dir",
            "/tmp/deepmzyme_missing_test_structures",
            "--test-summary-csv",
            "/tmp/deepmzyme_missing_test_summary.csv",
            "--run-test-eval",
            "--allow-train-loss-test-eval-debug",
        ]
    )
    validate_training_configuration(debug_config)


def check_loss_weight_validation() -> None:
    default_config = parse_args([])
    if default_config.metal_loss_weight != 1.0:
        raise AssertionError(f"Expected default metal_loss_weight=1.0, got {default_config.metal_loss_weight}")
    if default_config.ec_loss_weight != 1.0:
        raise AssertionError(f"Expected default ec_loss_weight=1.0, got {default_config.ec_loss_weight}")

    for option in ("--metal-loss-weight", "--ec-loss-weight"):
        config = parse_args([option, "-0.1"])
        try:
            validate_training_configuration(config)
        except ValueError as exc:
            if option not in str(exc):
                raise AssertionError(f"{option} failed with an unexpected error: {exc}") from exc
        else:
            raise AssertionError(f"{option} accepted a negative value.")


def check_bundle_cli_help() -> None:
    run_help(REPO_ROOT / "src" / "build_colab_bundle.py")


def check_docs_do_not_use_broken_training_command() -> None:
    broken_module = ".".join(("src", "training", "run"))
    broken_patterns = (f"python -m {broken_module}", broken_module)
    for relative_path in ("README.md", "list_train_commands.md"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        matches = [pattern for pattern in broken_patterns if pattern in text]
        if matches:
            raise AssertionError(f"{relative_path} still contains broken command patterns: {matches}")


def main() -> int:
    checks = (
        check_training_cli_help,
        check_test_eval_safety,
        check_loss_weight_validation,
        check_bundle_cli_help,
        check_docs_do_not_use_broken_training_command,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
