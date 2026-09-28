"""Verify a completed host backup, then optionally copy small portable evidence.

Default is verification/preview only. --copy creates a NEW direct child of the
fixed portable execution directory; canonical evidence is never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
C = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context")
CODE = C.parent / "_code/pmm_core_scope_v2"
TRAIN = Path("/media/mechti/Data1/DeepMzyme_PMM_Zenodo_Exact_Dataset/dataset/train")
DEST = Path("/home/mechti/PycharmProjects/DeepMzyme/docs/notebook_outputs/raw/pmm_five_class_screen_20260928/execution")
SOURCE_SHA = "adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23"
ADAPTER_SHA = "5c038c30d3540ed88c893a973dd66ec11425d7f28e60c2aa6605b4b1fa508c8d"
PROTOCOL_SHA = "c6d1da7fa0b7aa32b147f0bfbc4914a95ed5d127cdbf22d11716e26c42321607"
RUN_FILES = (
    "selected_checkpoint.json", "epoch_metrics.csv", "train_metrics.csv", "val_metrics.csv",
    "val_predictions.csv", "runtime_profile.json", "split_diagnostics.json",
    "independent_validation_replay/selected_checkpoint.json",
    "independent_validation_replay/replay_receipt.json",
    "independent_validation_replay/val_predictions.csv",
)
OMISSIONS = {"run_config.json": ("dataset_summary", "history"),
             "run_metadata.json": ("dataset_summary",),
             "dataset_summary.json": ("retained_split_identity", "feature_load_report")}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_source(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe backup path")
    path = (C / relative).resolve()
    require(path.is_relative_to(C.resolve()) and path.is_file(), f"Missing or escaping backup file: {relative}")
    return path


def verify_backup(unit, manifest_id):
    require(re.fullmatch(r"(?:only_esm|only_gvp|gvp_late_fusion)__five_class__none__fold0__seed42", unit),
            "Only a planned five-class fold-0 unit is allowed")
    require(re.fullmatch(r"[0-9a-f]{32}", manifest_id), "Invalid host manifest ID")
    prefix = "persistence_receipts/" + manifest_id
    manifest_path = safe_source(prefix + ".json")
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    ack_relative = prefix + ".ack.json"
    require(manifest["schema_version"] == 1 and manifest["persistence_mode"] == "host_pull"
            and manifest["ack_path"] == ack_relative and Path(manifest["destination_root"]).resolve() == C,
            "Wrong backup schema, destination or acknowledgment path")
    ack_path, state_path = safe_source(ack_relative), safe_source(prefix + ".state.json")
    ack_raw = ack_path.read_bytes()
    ack = json.loads(ack_raw)
    bindings = {str(manifest_path): hashlib.sha256(manifest_raw).hexdigest(),
                str(ack_path): hashlib.sha256(ack_raw).hexdigest()}
    require(ack["schema_version"] == 1 and ack["manifest_sha256"] == bindings[str(manifest_path)]
            and ack["file_count"] == len(manifest["files"]) and Path(ack["verified_root"]).resolve() == C
            and bool(ack["verified_by"]) and math.isfinite(ack["verified_unix"])
            and ack["verified_unix"] >= manifest["created_unix"], "Host acknowledgment does not verify this manifest")
    indexed = {}
    for item in manifest["files"]:
        relative = item["path"]
        require(relative not in indexed, "Duplicate backup entry")
        path = safe_source(relative)
        require(type(item["size_bytes"]) is int and item["size_bytes"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                and path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"],
                f"Backup content/hash mismatch: {relative}")
        indexed[relative] = item
        bindings[str(path)] = item["sha256"]
    require(prefix + ".state.json" in indexed and prefix + ".events.jsonl" in indexed,
            "Backup lacks its immutable state/events snapshot")
    state = read(state_path)
    pending, transfer = state["pending_unit"], state["pending_transfer"]
    require(pending["run_name"] == unit and pending["status"] == "completed"
            and pending["session_id"] == manifest["session_id"] == state["active_session_id"]
            and transfer["manifest_path"] == prefix + ".json" and transfer["ack_path"] == ack_relative
            and Path(transfer["destination_root"]).resolve() == C, "Backup snapshot is not this completed unit")
    # persisted=false in this immutable pre-ack snapshot is expected; the later
    # host acknowledgment above supplies independent hash verification.
    return manifest, ack, state, indexed, bindings


def excerpt(source, omitted, original):
    raw = source.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == original["sha256"], f"Excerpt source changed: {source}")
    payload = json.loads(raw)
    actual_omissions = [key for key in omitted if key in payload]
    return {"derived_excerpt": True, "source_path": str(source.relative_to(C)),
            "source_sha256": original["sha256"], "source_size_bytes": original["size_bytes"],
            "omitted_top_level_fields": actual_omissions,
            "note": "This is a derived excerpt, not the original file. Full files and binary checkpoints remain in the canonical verified host backup. Full epoch history is copied as epoch_metrics.csv. No labels/configuration values are rewritten.",
            "payload": {key: value for key, value in payload.items() if key not in omitted}}


def prepare(unit, manifest_id):
    manifest, ack, state, indexed, bindings = verify_backup(unit, manifest_id)
    adapter, protocol = CODE / "run_pmm_five_class_screen.py", CODE / "docs/plans/pmm_five_class_screen_v1.json"
    require(sha(adapter) == ADAPTER_SHA and sha(protocol) == PROTOCOL_SHA, "Frozen adapter/protocol changed")
    bindings.update({str(adapter): ADAPTER_SHA, str(protocol): PROTOCOL_SHA})
    sys.path[:0] = [str(CODE), str(CODE / "src")]
    spec = importlib.util.spec_from_file_location("five_screen", adapter)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
    from benchmarking import pmm_ion_campaign as campaign
    from training.access_guard import install_forbidden_read_guard

    install_forbidden_read_guard(campaign.forbidden_read_roots(TRAIN))
    require(screen.source_tree_sha256() == SOURCE_SHA, "Frozen scientific source changed")
    screen.load_protocol(protocol)
    paths = campaign.CampaignPaths(C)
    family = unit.split("__")[0]
    identity = screen.build_command(paths, SimpleNamespace(python_bin=sys.executable, train_dir=TRAIN,
                                    action="plan", load_workers=None), family, PROTOCOL_SHA)[2]
    command_relative = f"commands/{unit}.json"
    status_relative = f"run_status_five_screen_{unit}.json"
    required = [f"runs/{unit}/{relative}" for relative in (*RUN_FILES, *OMISSIONS)]
    required += [command_relative, status_relative]
    required += [f"runs/{unit}/best_model_checkpoint.pt"]
    require(set(required) <= set(indexed), "Backup manifest omits required scientific artifacts")
    actual_run_files = {str(path.relative_to(C)) for path in (C / "runs" / unit).rglob("*") if path.is_file()}
    require(actual_run_files <= set(indexed), "Run contains files outside its verified terminal backup")
    require(read(safe_source(command_relative))["identity"] == identity, "Command identity differs from frozen recipe")
    terminal = read(safe_source(status_relative))["units"]
    require(len(terminal) == 1 and terminal[0]["run_name"] == unit and terminal[0]["status"] == "completed"
            and terminal[0]["five_class_validation"]["status"] == "passed"
            and terminal[0]["five_class_validation"]["campaign_run_identity"] == identity,
            "Adapter did not certify this terminal unit")
    validation = screen.validate_completed_unit(paths, family, identity)
    require(validation["status"] == "passed", "Strict five-class validation failed")
    prefix = "persistence_receipts/" + manifest_id
    copies = {relative: safe_source(f"runs/{unit}/{relative}") for relative in RUN_FILES}
    copies.update({"command_record.json": safe_source(command_relative), "terminal_status.json": safe_source(status_relative),
                   "provenance/five_class_screen_protocol.json": protocol,
                   "provenance/run_pmm_five_class_screen.py": adapter})
    copies.update({"persistence_receipts/" + Path(prefix + suffix).name: safe_source(prefix + suffix)
                   for suffix in (".json", ".ack.json", ".state.json", ".events.jsonl")})
    derived = {Path(name).stem + "_excerpt.json": excerpt(safe_source(f"runs/{unit}/{name}"), omitted,
               indexed[f"runs/{unit}/{name}"]) for name, omitted in OMISSIONS.items()}
    record = {"schema_version": 1, "run_name": unit, "source_root": str(C / "runs" / unit),
              "host_manifest_id": manifest_id, "host_manifest_sha256": ack["manifest_sha256"],
              "complete_backup_file_count_verified": len(indexed), "host_ack": ack,
              "campaign_run_identity": identity, "strict_five_class_validation": validation,
              "fit_and_replay_seconds": state["pending_unit"]["elapsed_seconds"],
              "source_tree_sha256": SOURCE_SHA, "adapter_sha256": ADAPTER_SHA, "protocol_sha256": PROTOCOL_SHA,
              "helper_sha256": sha(__file__), "held_out_access": False,
              "copy_contract": "Exact portable artifacts plus explicitly marked JSON excerpts. selected_checkpoint.json is the exact selection receipt; binary .pt weights, full cohort/split expansions and full original JSON files remain in canonical verified backups. This copy does not certify VM shutdown or full-grid completion."}
    copy_hashes = {relative: bindings[str(path)] for relative, path in copies.items()}
    for relative, path in copies.items():
        require(sha(path) == copy_hashes[relative], f"Portable source changed after backup verification: {path}")
    return copies, derived, record, copy_hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy after verification; default is a write-free preview")
    args = parser.parse_args(argv)
    output = args.out_dir.resolve()
    require(output.parent == DEST.resolve() and not output.exists() and not args.out_dir.is_symlink(),
            "Output must be a NEW direct child of the portable execution directory")
    copies, derived, record, copy_hashes = prepare(args.unit, args.manifest_id)
    descriptors = {relative: {"source_path": str(path), "sha256": copy_hashes[relative], "size_bytes": path.stat().st_size,
                              "exact_copy": True} for relative, path in copies.items()}
    encoded = {name: (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
               for name, payload in derived.items()}
    require(all(len(data) < 256_000 for data in encoded.values()), "An excerpt still contains a large expansion")
    result = {"run_name": args.unit, "backup_verified_files": record["complete_backup_file_count_verified"],
              "exact_files": len(copies), "derived_excerpts": len(derived),
              "portable_bytes_without_manifest": sum(item["size_bytes"] for item in descriptors.values()) + sum(map(len, encoded.values())),
              "out_dir": str(output), "files_written": False}
    if args.copy:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
        for relative, source in copies.items():
            expected = descriptors[relative]["sha256"]
            require(sha(source) == expected, f"Source changed after verification: {source}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            require(sha(destination) == expected and sha(source) == expected, f"Copy changed: {relative}")
        for relative, content in encoded.items():
            with (output / relative).open("xb") as stream:
                stream.write(content)
            descriptors[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
                                     "exact_copy": False, "derived_excerpt": True,
                                     "source_sha256": derived[relative]["source_sha256"]}
        record.update(copied_at_utc=datetime.now(timezone.utc).isoformat(), artifacts=descriptors)
        with (output / "portable_evidence_manifest.json").open("x") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
        result["files_written"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
