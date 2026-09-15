"""Materialize pinned inputs on the already owned VM; no training or resume."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

from ring_common import allocation_deadline, read, require, save, sha, validate_handoff


def verify_source(root):
    snapshot = read(root/"code_snapshot_manifest.json")
    for name, expected in snapshot["files"].items():
        require(sha(root/name) == expected, "Frozen source changed: "+name)
    return snapshot


def main():
    cfg = read("/content/ring_session_config.json")
    output, root, budget = Path(cfg["output_dir"]), Path(cfg["repo_root"]), Path(cfg["budget_root"])
    deadline = min(cfg["allocation_started_epoch"]+cfg["setup_limit_seconds"], allocation_deadline(cfg, training=True))
    def remaining():
        value = deadline-time.time()
        require(value > 0, "Continuation setup allowance exhausted")
        return value
    def extract(path, checksum, destination):
        remaining()
        require(sha(path) == checksum, "Input archive checksum mismatch: "+Path(path).name)
        with tarfile.open(path) as stream:
            for member in stream:
                remaining()
                stream.extract(member, destination, filter="data")
    for name in ("ring_common.py", "ring_control.py", "ring_bootstrap.py"):
        require(sha(Path("/content")/name) == cfg["operator_files"][name], "Uploaded operator source changed: "+name)
    handoff_path = Path("/content/ring_budget_handoff.json")
    require(sha(handoff_path) == cfg["budget_handoff_sha256"], "Immutable budget handoff changed")
    handoff = validate_handoff(read(handoff_path))
    budget.mkdir(parents=True, exist_ok=True)
    existing = budget/"budget_handoff.json"
    require(not existing.exists() or sha(existing) == cfg["budget_handoff_sha256"], "Different existing budget handoff")
    shutil.copyfile(handoff_path, existing)
    if not (budget/"allocation_ledger.json").exists():
        save(budget/"allocation_ledger.json", dict(intervals=handoff["prior_intervals"]))
    receipt = read("/content/metal_ring_persistence_receipt.json", {})
    require(receipt.get("method") == "verified_archive_transfer" and receipt.get("drive_probe_verified")
            and receipt.get("local_write_verified") and receipt.get("drive_folder_id") == cfg["drive_folder_id"],
            "Persistence evidence is missing or differs")
    output.mkdir(parents=True, exist_ok=True)
    started = cfg["allocation_started_epoch"]
    result = dict(status="running", allocation_started_epoch=started, held_out_evaluation=False)
    try:
        extract(cfg["source_archive"], cfg["source_sha256"], root)
        snapshot = verify_source(root)
        sys.path.insert(0, str(root/"scripts"))
        from colab_metal_pilot_bootstrap import cuda_check
        before = cuda_check()
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(root/"requirements/colab-overlay.txt")],
                       check=True, timeout=remaining())
        code = "import sys;sys.path.insert(0,"+repr(str(root/"scripts"))+");from colab_metal_pilot_bootstrap import cuda_check;cuda_check()"
        after = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True, timeout=remaining()))
        require(after["torch"] == before["torch"], "Dependency installation replaced stock PyTorch")
        bundle = Path("/content/pilot_data_bundle.tar.gz")
        if not bundle.is_file() or sha(bundle) != cfg["bundle_sha256"]:
            temporary = bundle.with_suffix(".partial")
            with urllib.request.urlopen(cfg["bundle_url"], timeout=min(60, remaining())) as response, temporary.open("wb") as target:
                while chunk := response.read(4*1024*1024):
                    remaining()
                    target.write(chunk)
            temporary.replace(bundle)
        extract(bundle, cfg["bundle_sha256"], "/content/deepmzyme_bundle")
        save(Path("/content/v12_ready.json"), dict(sha256=cfg["bundle_sha256"], verified_epoch=time.time(), url=cfg["bundle_url"]))
        extract(cfg["overlay_archive"], cfg["overlay_sha256"], "/content/metal_architecture_pilot_features")
        extract(cfg["parent_reference_archive"], cfg["parent_reference_sha256"], cfg["parent_reference_dir"])
        parent = Path(cfg["parent_reference_dir"])
        require(sha(parent/"campaign_manifest.json") == handoff["parent_manifest_sha256"]
                and sha(parent/"expected_split.json") == handoff["expected_split_sha256"], "Parent feature/cohort proof changed")
        manifest = read(parent/"campaign_manifest.json")
        data = Path(cfg["data_root"])
        for name, checksum in manifest["dataset_files"].items():
            require(sha(data/name) == checksum, "Pinned training data differs from parent proof: "+name)
        require(sha(Path(cfg["feature_overlay_manifest"])) == manifest["feature_overlay_manifest"]["sha256"], "Repaired overlay proof differs")
        sys.path.insert(0, str(root/"src"))
        import run_metal_architecture_pilot as base
        base.validate_feature_overlay(data, cfg["external_features_root_dir"], cfg["feature_overlay_manifest"])
        # Source-specific readiness and RING coverage are separate bounded
        # preflight attempts. This bootstrap never prepares graphs or trains.
        result.update(status="ready", source_sha256=cfg["source_sha256"], source_files=snapshot["files"],
                      parent_manifest_sha256=handoff["parent_manifest_sha256"], bundle_sha256=cfg["bundle_sha256"],
                      overlay_sha256=cfg["overlay_sha256"], cuda=after, data_materialized=True,
                      old_checkpoint_restoration_performed=False)
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        elapsed = time.time()-started
        save(budget/"bootstrap_budget_usage.json", dict(normal_elapsed_seconds=elapsed, started_epoch=started,
              ended_epoch=time.time(), status=result["status"], includes_provisioning_upload_and_materialization=True))
        save(output/"bootstrap_provenance.json", dict(result, elapsed_seconds=elapsed, session_name=cfg["session_name"]))
    print("RING_BOOTSTRAP_READY="+json.dumps(dict(status="ready", elapsed_seconds=elapsed)))


if __name__ == "__main__":
    main()
