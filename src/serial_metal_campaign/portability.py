"""Export an unstarted local plan for fixed worker paths without moving state."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shlex

from serial_metal_campaign import budget, profile

base = profile.base


def _absolute(value, name):
    path = Path(value)
    base.require(path.is_absolute() and ".." not in path.parts, f"{name} must be an absolute normalized path")
    return str(path)


def _remap(value, mappings, python_executable):
    if isinstance(value, dict):
        result = {key: _remap(item, mappings, python_executable) for key, item in value.items()}
        if isinstance(result.get("command"), list) and result["command"]:
            result["command"][0] = python_executable
        return result
    if isinstance(value, list):
        return [_remap(item, mappings, python_executable) for item in value]
    if isinstance(value, str):
        # PYTHONPATH-like environment variables can contain multiple paths.
        if ":" in value and all(part.startswith("/") for part in value.split(":")):
            return ":".join(_remap(part, mappings, python_executable) for part in value.split(":"))
        for old, new in mappings:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
    return value


def export_runtime_plan(local_output, destination, root, data, worker_output, external, python_executable):
    """Write a reviewable worker plan; no worker filesystem or GPU is required."""
    source, destination = Path(local_output).resolve(), Path(destination).resolve()
    base.require(source != destination and not source.is_relative_to(destination)
                 and not destination.is_relative_to(source), "Export destination must be separate from the source campaign")
    manifest = profile.verify_manifest(source)
    queue = base.read(source / "queue.json", {})
    base.require(queue == {"runs": manifest["runs"], "phase": "screen"},
                 "Export requires the untouched initial discovery queue")
    base.require(len(manifest["runs"]) == 65 and manifest["runs"] == profile.initial_runs(manifest),
                 "Export requires the complete initial 65-run plan")
    for name in ("attempts.json", "sessions.json", "confirmation_manifest.json", "discovery_closed.json",
                 "refinement_decisions.json", "chain_decisions.json", "comparisons.json"):
        base.require(not base.read(source / name), "Cannot export a started campaign: " + name)
    for name in ("runs", "reuse", "readiness"):
        base.require(not (source / name).is_dir() or not any((source / name).iterdir()),
                     "Cannot export existing execution evidence: " + name)
    replacements = {
        manifest["root"]: _absolute(root, "root"),
        manifest["data_root"]: _absolute(data, "data"),
        manifest["output_dir"]: _absolute(worker_output, "worker_output"),
        manifest["external_features_root_dir"]: _absolute(external, "external"),
    }
    base.require(len(replacements) == 4, "Source repository, data, output and feature roots must be distinct")
    python_executable = _absolute(python_executable, "python_executable")
    mappings = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
    exported = _remap(manifest, mappings, python_executable)
    # Only paths and the interpreter change. Reparse command flags to certify
    # that every stored run configuration still describes the actual command.
    for original, run in zip(manifest["runs"], exported["runs"]):
        parsed = base.parse_config(run["command"])
        profile.validate(parsed, epochs=run["epochs"], fold_index=run["fold_index"], ring=run["ring"])
        payload = profile.canonical(asdict(parsed))
        base.require(payload == _remap(original["config"], mappings, python_executable),
                     f"Export changed scientific CLI semantics: {run['id']}")
        run["config"] = payload
        run["config_sha256"] = base.fingerprint(payload)
        for key in ("id", "recipe_id", "parameters", "seed", "epochs", "family", "scheme", "ring", "fold_index"):
            base.require(run[key] == original[key], f"Export changed {key}")
    for key in ("source_files", "dataset_files", "policy", "bundle_sha256", "arms"):
        base.require(exported[key] == manifest[key], f"Export changed frozen scientific identity: {key}")
    allowed = {"campaign_manifest.json", "queue.json", "commands.txt", "run_matrix.csv",
               "budget_forecast.json", "budget_forecast.md", "runtime_export.json"}
    base.require(not destination.exists() or all(path.name in allowed and path.is_file() for path in destination.iterdir()),
                 "Export destination contains execution state or unrelated files")
    # Check frozen values before writing any output, including idempotent reexports.
    exported_queue = {"runs": exported["runs"], "phase": "screen"}
    for name, payload in (("campaign_manifest.json", exported), ("queue.json", exported_queue)):
        existing = base.read(destination / name)
        base.require(existing is None or existing == payload, "Existing exported plan differs: " + name)
    destination.mkdir(parents=True, exist_ok=True)
    profile.freeze(destination / "campaign_manifest.json", exported)
    profile.freeze(destination / "queue.json", exported_queue)
    (destination / "commands.txt").write_text("\n\n".join(shlex.join(row["command"]) for row in exported["runs"]) + "\n")
    base.csv_save(destination / "run_matrix.csv", exported["runs"],
                  ["id", "stage", "block", "arm", "family", "scheme", "seed", "epochs", "parameters", "run_dir"])
    budget.write_preview(destination, budget.historical_preview())
    profile.freeze(destination / "runtime_export.json", dict(
        status="exported_initial_plan_not_prepared", source_campaign=str(source),
        source_manifest_sha256=base.digest(source / "campaign_manifest.json"),
        exported_manifest_sha256=base.digest(destination / "campaign_manifest.json"),
        worker_output=exported["output_dir"], initial_runs=len(exported["runs"]),
        worker_preparation_required=True, execution_state_copied=False, certified_reuse=0,
        instruction="Copy these files unchanged to worker_output, then run CPU prepare there before GPU readiness."))
    return exported
