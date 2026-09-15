"""Prepare one already allocated Colab VM from a verified pilot setup file.

Run on the remote VM, with /content/pilot_setup.json supplied by the host.
Allocation and teardown remain the host operator's responsibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.request

BOOTSTRAP_DEADLINE = None


def remaining():
    seconds = BOOTSTRAP_DEADLINE - time.time() if BOOTSTRAP_DEADLINE else 3600
    if seconds <= 0:
        raise TimeoutError("The pilot setup/profile allowance is exhausted")
    return seconds


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            remaining()
            h.update(block)
    return h.hexdigest()


def verified_extract(archive, expected, destination):
    if digest(archive) != expected:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    with tarfile.open(archive) as stream:
        for member in stream:
            remaining()
            stream.extract(member, destination, filter="data")


def cuda_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    required = "sm_%d%d" % torch.cuda.get_device_capability(0)
    result = dict(python=sys.version, executable=sys.executable,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), required_architecture=required,
                  compiled_architectures=torch.cuda.get_arch_list())
    print(json.dumps(result), flush=True)
    if required not in result["compiled_architectures"]:
        raise RuntimeError("Stock PyTorch does not support the assigned CUDA architecture")
    x = torch.ones(32, device="cuda", requires_grad=True)
    x.square().sum().backward()
    torch.cuda.synchronize()
    return result


def main():
    global BOOTSTRAP_DEADLINE
    config = json.loads(Path("/content/pilot_setup.json").read_text())
    BOOTSTRAP_DEADLINE = config["allocation_started_epoch"] + 3600
    remaining()
    receipt = json.loads(Path("/content/persistence_receipt.json").read_text())
    if not (receipt.get("method") == "verified_archive_transfer" and receipt.get("drive_probe_verified")
            and receipt.get("local_write_verified") and receipt.get("drive_folder_id")):
        raise ValueError("Verified persistence is required before provisioning work")
    root = Path(config["repo_root"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    before = cuda_check()
    verified_extract(config["source_archive"], config["source_sha256"], root)
    snapshot = json.loads((root / "code_snapshot_manifest.json").read_text())
    for name, expected in snapshot["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Working snapshot mismatch: {name}")
    expected = json.loads((root / "colab_expected_metal_inputs.json").read_text())
    if expected["dataset"] != "train_and_test_sets_structures_non_overlapped_pinmymetal":
        raise ValueError("The snapshot does not certify the pilot's non-overlap training inputs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(root / "requirements/colab-overlay.txt")], check=True, timeout=remaining())
    # Recheck in a fresh process to detect any accidental package replacement.
    after_text = subprocess.check_output([sys.executable, "-c",
                    "import sys; sys.path.insert(0, " + repr(str(root / "scripts")) + "); "
                    "from colab_metal_pilot_bootstrap import cuda_check; cuda_check()"], text=True, timeout=remaining())
    after = json.loads(after_text)
    if after["torch"] != before["torch"]:
        raise RuntimeError("Dependency installation replaced the stock PyTorch build")
    bundle = Path("/content/pilot_data_bundle.tar.gz")
    if not bundle.is_file() or digest(bundle) != config["bundle_sha256"]:
        temporary = bundle.with_suffix(".partial")
        print("Downloading pinned v12 data bundle", flush=True)
        with urllib.request.urlopen(config["bundle_url"], timeout=min(60, remaining())) as response, temporary.open("wb") as stream:
            while chunk := response.read(4 * 1024 * 1024):
                remaining()
                stream.write(chunk)
        temporary.replace(bundle)
    verified_extract(bundle, config["bundle_sha256"], "/content/deepmzyme_bundle")
    Path("/content/v12_ready.json").write_text(json.dumps(dict(
        sha256=config["bundle_sha256"], verified_epoch=time.time(), url=config["bundle_url"])))
    verified_extract(config["overlay_archive"], config["overlay_sha256"],
                     "/content/metal_architecture_pilot_features")
    data = Path(config["data_root"])
    sys.path.insert(0, str(root / "src"))
    from structure_store import read_structure_manifest
    train = data / expected["dataset"] / "train"
    if digest(train / "final_data_summarazing_table_transition_metals_only_catalytic.csv") != expected["summary_sha256"]:
        raise ValueError("Remote training summary differs from the reviewed local input")
    actual = {ref.structure_name: ref.sha256 for ref in read_structure_manifest(train)}
    if actual != expected["train_structures"]:
        raise ValueError("Remote training structure identities differ from the reviewed local input")
    (output / "bootstrap_provenance.json").write_text(json.dumps(dict(
        setup=config, cuda=before, snapshot=snapshot, expected_training_inputs=expected), indent=2))
    subprocess.run([sys.executable, str(root / "src/run_metal_architecture_pilot.py"), "plan",
                    "--data-root", str(data), "--output-dir", str(output),
                    "--source-commit", snapshot["base_commit"] + ":snapshot:" + config["source_sha256"],
                    "--external-features-root-dir", config["external_features_root_dir"],
                    "--feature-overlay-manifest", config["feature_overlay_manifest"]], check=True)
    subprocess.run([sys.executable, str(root / "src/run_metal_architecture_pilot.py"), "preflight",
                    "--output-dir", str(output), "--allocation-started-epoch", str(config["allocation_started_epoch"]),
                    "--persistence-receipt", "/content/persistence_receipt.json"], check=True)
    print("PILOT_BOOTSTRAP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
