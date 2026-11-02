"""CPU-only transfer and execution contracts for the EC Phase-3 runner."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_clean30 import phase3, views  # noqa: E402
from ec_clean30.campaign import CAMPAIGN_ID  # noqa: E402
from ec_clean30_fixtures import build_smoke_view, make_writable, unlock, write_clean30_data_root  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def transfer_view(tmp_path_factory):
    """A real synthetic View with both layer caches; no model or real-data read."""
    import torch
    from training.global_esm_features import build_manifest, chain_mean_payload, encode_cache_file, write_manifest

    base = tmp_path_factory.mktemp("phase3_transfer")
    source = write_clean30_data_root(base / "source")
    root = base / "campaign" / CAMPAIGN_ID / "smoke_views" / "tiny"
    view = build_smoke_view(source, root)
    final = json.loads((view.chain_means_dir("final") / "manifest.json").read_text())
    files = []
    for entry in final["files"]:
        payload = torch.load(view.chain_means_dir("final") / entry["file"], weights_only=True)
        item = encode_cache_file(chain_mean_payload(
            payload["mean"], structure_id=payload["structure_id"], layer="block26",
            n_residues=payload["n_residues"], source={"kind": "synthetic_fixture"},
            source_sequence_sha256=payload["source_sequence_sha256"], embedding_dim=payload["embedding_dim"],
        ))
        (view.chain_means_dir("block26") / item.file_name).write_bytes(item.data)
        files.append(item)
    write_manifest(view.chain_means_dir("block26"), build_manifest(
        files, kind="chain_mean", allowlist=final["allowlist"], build={"action": "synthetic_fixture"},
        embedding_dim=final["embedding_dim"],
    ))
    dry = {"status": "passed", "view_manifest_sha256": view.manifest_sha256, "verified_view": True,
           "cap_prefix": ["synthetic-cap"], "verification": {"chain_means": {
               layer: {"manifest_sha256": view.chain_means_manifest_sha256(layer)}
               for layer in ("final", "block26")}}}
    (view.receipts / views.DRY_RUN_RECEIPT_NAME).write_text(json.dumps(dry) + "\n")
    (view.caches / "dry_run" / "old-probe.txt").write_text("not transferred\n")
    yield view
    make_writable(root)
    unlock(source)


def _destination(tmp_path: Path) -> Path:
    return tmp_path / "relocated" / CAMPAIGN_ID / "smoke_views" / "tiny"


def test_transfer_round_trip_preserves_receipts_layers_and_relocated_guards(transfer_view, tmp_path):
    view = transfer_view
    original = (view.receipts / views.DRY_RUN_RECEIPT_NAME).read_bytes()
    archive = tmp_path / "view.tar"
    phase3.pack_view(view.root, archive, smoke=True)
    destination = _destination(tmp_path)
    phase3.unpack_view(archive, destination, expected_sha256=_sha(archive), smoke=True)
    relocated = views.View.load(destination)
    assert (relocated.receipts / views.DRY_RUN_RECEIPT_NAME).read_bytes() == original
    assert (view.receipts / views.DRY_RUN_RECEIPT_NAME).read_bytes() == original
    assert relocated.manifest_sha256 == view.manifest_sha256
    assert not (relocated.caches / "dry_run" / "old-probe.txt").exists()
    for layer in ("final", "block26"):
        assert relocated.chain_means_manifest_sha256(layer) == view.chain_means_manifest_sha256(layer)
    assert str(relocated.campaign_root / "identity") in relocated.forbidden_read_roots
    for entry in relocated.manifest["files"]:
        assert not (relocated.root / entry["path"]).stat().st_mode & 0o222
    assert stat.S_IMODE(relocated.structures.stat().st_mode) == 0o555
    assert stat.S_IMODE(relocated.receipts.stat().st_mode) & stat.S_IWUSR
    views.verify_view(destination, require_chain_means=("final", "block26"), write_receipt=False)


def test_transfer_refuses_wrong_checksum_before_unpacking(transfer_view, tmp_path):
    archive = tmp_path / "view.tar"
    phase3.pack_view(transfer_view.root, archive, smoke=True)
    destination = _destination(tmp_path)
    with pytest.raises(ValueError):
        phase3.unpack_view(archive, destination, expected_sha256="0" * 64, smoke=True)
    assert not destination.exists()


def _malformed_archive(source: Path, destination: Path, case: str) -> None:
    with tarfile.open(source, "r:*") as old, tarfile.open(destination, "w") as new:
        members = old.getmembers()
        changed = False
        for member in members:
            if (not changed and case in ("permissions", "member_checksum") and member.isfile()
                    and member.name != phase3.TRANSFER_MANIFEST):
                payload = old.extractfile(member).read()
                if case == "permissions":
                    member.mode = 0o777
                else:
                    payload = bytes([payload[0] ^ 1]) + payload[1:]
                new.addfile(member, io.BytesIO(payload))
                changed = True
                continue
            new.addfile(member, old.extractfile(member) if member.isfile() else None)
        if changed:
            return
        if case == "duplicate":
            member = next(member for member in members if member.isfile())
            new.addfile(member, old.extractfile(member))
            return
        names = {"traversal": "../escaped", "absolute": "/escaped", "extra": "unexpected.txt",
                 "symlink": "linked.txt", "hardlink": "hardlinked.txt"}
        member = tarfile.TarInfo(names[case])
        member.mode = 0o444
        if case in ("symlink", "hardlink"):
            member.type = tarfile.SYMTYPE if case == "symlink" else tarfile.LNKTYPE
            member.linkname = "../outside"
            new.addfile(member)
        else:
            member.size = 4
            new.addfile(member, io.BytesIO(b"oops"))


@pytest.mark.parametrize("case", ["traversal", "absolute", "symlink", "hardlink", "duplicate", "extra",
                                  "permissions", "member_checksum"])
def test_transfer_refuses_unsafe_or_unlisted_members(transfer_view, tmp_path, case):
    archive = tmp_path / "view.tar"
    phase3.pack_view(transfer_view.root, archive, smoke=True)
    malformed = tmp_path / "malformed.tar"
    _malformed_archive(archive, malformed, case)
    destination = _destination(tmp_path)
    with pytest.raises(ValueError):
        phase3.unpack_view(malformed, destination, expected_sha256=_sha(malformed), smoke=True)
    assert not destination.exists()
    assert not (destination.parent / "escaped").exists()


def test_transfer_refuses_existing_destination(transfer_view, tmp_path):
    archive = tmp_path / "view.tar"
    phase3.pack_view(transfer_view.root, archive, smoke=True)
    destination = _destination(tmp_path)
    destination.mkdir(parents=True)
    sentinel = destination / "owned.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError):
        phase3.unpack_view(archive, destination, expected_sha256=_sha(archive), smoke=True)
    assert sentinel.read_text() == "preserve"


def _training_rows(n=1004):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for index in range(n):
        sequence = "M" + "".join(alphabet[(index // 20**power) % 20] for power in range(3)) + "K"
        rows.append((f"P{index:06d}", "1.1.1.1", sequence))
    if n >= 2:
        rows[-1] = (rows[-1][0], rows[-1][1], rows[-2][2])
    return rows


def _official(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("Entry", "EC number", "Sequence"))
        writer.writerows(rows)
    return path


def test_timing_inputs_are_exactly_the_hash_ordered_training_rows(tmp_path_factory):
    from scripts.run_clean_split30_comparator import read_sequence_map_file
    from embed_helpers.esmc_chain_means import read_sequence_map

    tmp_path = tmp_path_factory.mktemp("phase3_timing")
    rows = _training_rows()
    first = _official(tmp_path / "a" / "split30_train_split_0.csv", rows)
    second = _official(tmp_path / "b" / "split30_train_split_0.csv", reversed(rows))
    out1, out2 = tmp_path / "timing_a", tmp_path / "timing_b"
    record1 = phase3.timing_inputs(first, out1, expected_sha256=_sha(first))
    record2 = phase3.timing_inputs(second, out2, expected_sha256=_sha(second))
    assert record1["n_rows"] == record2["n_rows"] == 1000
    assert record1["source_sha256"] != record2["source_sha256"]
    expected = sorted(rows, key=lambda row: (hashlib.sha256(row[2].encode()).hexdigest(), row[0]))[:1000]
    with (out1 / "sequence_map.tsv").open() as handle:
        observed = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["accession"], row["sequence"]) for row in observed] == [(r[0], r[2]) for r in expected]
    assert all(row["sha256"] == hashlib.sha256(row["sequence"].encode()).hexdigest() for row in observed)
    assert (out1 / "timing.fasta").read_text() == "".join(f">{row[0]}\n{row[2]}\n" for row in expected)
    for name in ("timing.fasta", "sequence_map.tsv"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
        assert record1["files"][name] == _sha(out1 / name)
    sequence_path = out1 / "sequence_map.tsv"
    clean_sequences, clean_receipt = read_sequence_map_file(sequence_path, [], expected_sha256=_sha(sequence_path))
    esmc_sequences = read_sequence_map(sequence_path, expected_sha256=_sha(sequence_path))
    assert clean_receipt["n_accessions"] == 1000
    assert clean_sequences == esmc_sequences.sequence_by_accession == {row[0]: row[2] for row in expected}


@pytest.mark.parametrize("case", ["checksum", "wrong_fold", "test_name", "too_small", "duplicate_accession"])
def test_timing_inputs_refuse_invalid_source_without_output(tmp_path_factory, case):
    tmp_path = tmp_path_factory.mktemp("phase3_timing_invalid")
    names = {"wrong_fold": "split30_train_split_1.csv", "test_name": "split30_test_split_0.csv"}
    rows = _training_rows(999 if case == "too_small" else 1004)
    if case == "duplicate_accession":
        rows[-1] = rows[0]
    source = _official(tmp_path / names.get(case, "split30_train_split_0.csv"), rows)
    out = tmp_path / "timing"
    with pytest.raises((phase3.Phase3Error, RuntimeError)):
        phase3.timing_inputs(source, out, expected_sha256="0" * 64 if case == "checksum" else _sha(source))
    assert not out.exists()


@pytest.fixture
def esmc_command_inputs(tmp_path, monkeypatch):
    inputs = tmp_path / "g2_inputs"
    inputs.mkdir()
    rows = _training_rows(1000)
    (inputs / "timing.fasta").write_text("".join(f">{row[0]}\n{row[2]}\n" for row in rows))
    with (inputs / "sequence_map.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("accession", "sha256", "length", "n_files", "sequence"))
        for accession, _, sequence in rows:
            writer.writerow((accession, hashlib.sha256(sequence.encode()).hexdigest(), len(sequence), 1, sequence))
    receipt = {"n_rows": 1000, "fold": 0, "source_sha256": phase3.OFFICIAL_FOLD0_SHA256,
               "files": {name: _sha(inputs / name) for name in ("timing.fasta", "sequence_map.tsv")}}
    _json_write(inputs / "timing_inputs.json", receipt)
    cache = tmp_path / "hf_home"
    weights = (cache / "hub" / "models--EvolutionaryScale--esmc-300m-2024-12" / "snapshots" /
               ("0" * 40) / "data" / "weights" / "esmc_300m_2024_12_v0.pth")
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"synthetic frozen weights")
    monkeypatch.setattr(phase3, "_esmc_cached_weight", lambda cache_root: weights.resolve())
    request = {"kind": "esmc", "python": sys.executable, "device": "cuda", "timing_inputs": str(inputs),
               "hf_home": str(cache), "esmc_weights": str(weights)}
    source = {"timing_inputs_sha256": _sha(inputs / "timing_inputs.json"),
              "official_fold0_training_sha256": phase3.OFFICIAL_FOLD0_SHA256,
              "frozen_esmc_weight_sha256": _sha(weights)}
    return request, source, tmp_path / "output"


def test_esmc_command_pins_offline_cache_and_uses_existing_producer(esmc_command_inputs, transfer_view, monkeypatch):
    from embed_helpers.esmc_chain_means import build_arg_parser
    from training.access_guard import FORBIDDEN_READ_ROOTS_ENV

    request, source, output = esmc_command_inputs
    monkeypatch.setenv("INFRA_PROVIDER", "synthetic-provider")
    argv, env = phase3.embedding_command(request, source, transfer_view, output)
    args = build_arg_parser().parse_args(argv[4:])
    assert args.action == "sequences" and args.layer_index == 26 and args.device == "cuda"
    assert args.sequence_map == Path(request["timing_inputs"]) / "sequence_map.tsv"
    assert args.sequence_map_sha256 == _sha(args.sequence_map)
    assert args.out == output
    cache = str(Path(request["hf_home"]).resolve())
    assert env["HF_HOME"] == cache
    assert env["HF_HUB_CACHE"] == env["HUGGINGFACE_HUB_CACHE"] == str(Path(cache) / "hub")
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert "INFRA_PROVIDER" not in env
    assert json.loads(env[FORBIDDEN_READ_ROOTS_ENV]) == transfer_view.forbidden_read_roots
    assert "install_guard_from_environment()" in argv[2]


@pytest.mark.parametrize("case", ["changed_weight", "missing_pin", "outside_cache", "wrong_sdk_snapshot"])
def test_esmc_command_refuses_unbound_weights(esmc_command_inputs, transfer_view, monkeypatch, case):
    request, source, output = esmc_command_inputs
    if case == "changed_weight":
        Path(request["esmc_weights"]).write_bytes(b"different weights")
    elif case == "missing_pin":
        source.pop("frozen_esmc_weight_sha256")
    elif case == "outside_cache":
        outside = output.parent / "weights.pth"
        shutil.copyfile(request["esmc_weights"], outside)
        request["esmc_weights"] = str(outside)
    else:
        other_snapshot = Path(request["hf_home"]) / "hub" / "other_snapshot" / "weights.pth"
        monkeypatch.setattr(phase3, "_esmc_cached_weight", lambda cache_root: other_snapshot)
    with pytest.raises(phase3.Phase3Error, match="ESMC weights"):
        phase3.embedding_command(request, source, transfer_view, output)


@pytest.mark.parametrize("case", ["inputs", "output", "symlink_inputs"])
def test_embedding_rejects_forbidden_timing_namespace_before_reading(esmc_command_inputs, transfer_view, monkeypatch, case):
    request, source, output = esmc_command_inputs
    forbidden = transfer_view.campaign_root / "timing" / "fold0"
    assert str(transfer_view.campaign_root / "timing") in transfer_view.forbidden_read_roots
    if case == "inputs":
        request["timing_inputs"] = str(forbidden)
    elif case == "output":
        output = forbidden
    else:
        alias = output.parent / "input_alias"
        alias.symlink_to(forbidden)
        request["timing_inputs"] = str(alias)

    def no_hash_before_guard(path):
        raise AssertionError(f"Read happened before input guard: {path}")

    monkeypatch.setattr(phase3, "_sha", no_hash_before_guard)
    with pytest.raises(phase3.Phase3Error, match="forbidden-read root"):
        phase3.embedding_command(request, source, transfer_view, output)


@pytest.mark.parametrize("profile", ["gcp_l4", "runpod_rtx4090"])
def test_esm1b_cuda_preflight_uses_clean_interpreter_and_resource_cap(tmp_path, monkeypatch, profile):
    request = {"cap_prefix": ["fixture-cap", "--"], "clean_python": "/fixture/clean/bin/python",
               "unit_timeout_seconds": 120}
    if profile != "gcp_l4":
        request["hardware_profile"] = profile
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    evidence = {"hardware_profile": profile, "device_name": phase3.hardware_profile(profile)["device_name"],
                "device_count": 1, "total_memory_bytes": 24 * 1024 ** 3,
                "torch": "fixture", "cuda": "fixture", "kernel_value": 8.0, "tf32_override": "0"}
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        json.dump(evidence, kwargs["stdout"])
        kwargs["stderr"].write('{"cap_status":"started"}\n{"cap_status":"finished"}\n')

    monkeypatch.setattr(phase3.subprocess, "run", run)
    result = phase3._esm1b_cuda_preflight(request, env, tmp_path)
    argv, kwargs = calls[0]
    assert argv[:4] == ["fixture-cap", "--", "/fixture/clean/bin/python", "-c"]
    assert "torch.cuda.is_available()" in argv[4] and "device='cuda'" in argv[4]
    assert kwargs["env"] == env and kwargs["check"] is True and kwargs["timeout"] == 60
    assert kwargs["cwd"] == phase3.REPO_ROOT
    assert json.loads((tmp_path / "cuda_preflight.json").read_text()) == evidence
    assert result == {"cuda_preflight_sha256": _sha(tmp_path / "cuda_preflight.json"),
                      "cuda_preflight_log_sha256": _sha(tmp_path / "cuda_preflight.stderr.log")}
    assert "cap_status" in (tmp_path / "cuda_preflight.stderr.log").read_text()


@pytest.mark.parametrize("profile", ["gcp_l4", "runpod_rtx4090"])
@pytest.mark.parametrize("case", ["profile", "device", "cpu", "multiple_devices", "small_memory", "large_memory", "kernel"])
def test_cuda_profile_evidence_refuses_unapproved_or_unusable_hardware(profile, case):
    evidence = {"hardware_profile": profile, "device_name": phase3.hardware_profile(profile)["device_name"],
                "device_count": 1, "total_memory_bytes": 24 * 1024 ** 3,
                "torch": "fixture", "cuda": "fixture", "kernel_value": 8.0, "tf32_override": "0"}
    mutations = {"profile": ("hardware_profile", "unapproved"), "device": ("device_name", "NVIDIA A100"),
                 "cpu": ("cuda", None), "multiple_devices": ("device_count", 2),
                 "small_memory": ("total_memory_bytes", 16 * 1024 ** 3),
                 "large_memory": ("total_memory_bytes", 48 * 1024 ** 3), "kernel": ("kernel_value", 0.0)}
    key, value = mutations[case]
    evidence[key] = value
    with pytest.raises(phase3.Phase3Error, match="CUDA preflight"):
        phase3.validate_cuda_evidence(evidence, profile)


@pytest.mark.parametrize("profile", ["gcp_l4", "runpod_rtx4090"])
def test_cuda_probe_program_executes_kernel_and_reports_actual_device(monkeypatch, capsys, profile):
    if profile.startswith("runpod_"):
        monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    calls = []

    class Tensor:
        def __mul__(self, value):
            assert value == 2
            return self

        def sum(self):
            return self

        def item(self):
            return 8.0

    def ones(size, *, device):
        calls.append((size, device))
        return Tensor()

    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_name=lambda index: phase3.hardware_profile(profile)["device_name"],
                           get_device_properties=lambda index: SimpleNamespace(total_memory=24 * 1024 ** 3),
                           synchronize=lambda: calls.append("synchronized"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="fixture"),
                                                             __version__="fixture", ones=ones))
    exec(phase3.cuda_probe_program(profile), {})
    evidence = json.loads(capsys.readouterr().out)
    assert phase3.validate_cuda_evidence(evidence, profile) == evidence
    assert calls == [(4, "cuda"), "synchronized"]


def test_cuda_profile_defaults_to_l4_and_refuses_unknown_hardware():
    assert phase3.hardware_profile()["device_name"] == "NVIDIA L4"
    with pytest.raises(phase3.Phase3Error, match="hardware profile"):
        phase3.cuda_probe_program("any_cuda_gpu")


@pytest.mark.parametrize("case", ["cpu_build", "child_failure"])
def test_esm1b_cuda_preflight_refuses_cpu_or_failed_check(tmp_path, monkeypatch, case):
    request = {"cap_prefix": [], "clean_python": "/fixture/clean/bin/python", "unit_timeout_seconds": 20}

    def run(argv, **kwargs):
        if case == "child_failure":
            raise subprocess.CalledProcessError(1, argv)
        json.dump({"device_name": "NVIDIA L4", "cuda": None, "kernel_value": 8.0}, kwargs["stdout"])

    monkeypatch.setattr(phase3.subprocess, "run", run)
    error = subprocess.CalledProcessError if case == "child_failure" else phase3.Phase3Error
    with pytest.raises(error):
        phase3._esm1b_cuda_preflight(request, {}, tmp_path)


@pytest.mark.parametrize("case", ["passed", "missing_log", "tampered_log", "cpu_fallback"])
def test_esm1b_gpu_evidence_requires_hash_bound_actual_transfer(tmp_path, case):
    log = tmp_path / "extract.log"
    log.write_text("Loaded model\nTransferred model to GPU\nFinished\n")
    extraction = {"device": "cuda", "returncode": 0, "log": "extract.log", "log_sha256": _sha(log)}
    if case == "missing_log":
        log.unlink()
    elif case == "tampered_log":
        log.write_text(log.read_text() + "changed\n")
    elif case == "cpu_fallback":
        log.write_text("Loaded model\nFinished\n")
        extraction["log_sha256"] = _sha(log)
    _json_write(tmp_path / "farm_receipt.json", {"extraction": extraction})
    if case == "passed":
        assert phase3._esm1b_gpu_evidence(tmp_path) == {"extraction_log_sha256": _sha(log)}
    else:
        with pytest.raises(phase3.Phase3Error, match="actually used CUDA"):
            phase3._esm1b_gpu_evidence(tmp_path)


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def _json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def execution_source(tmp_path, monkeypatch):
    """Isolate WP-13's live-source layer; the frozen verifier has its own suite."""
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Fixture Maintainer")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    for name in ("src/frozen.py", "src/amended.py", *phase3.EXECUTION_FILES):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture {name}\n")
    docs = repo / "docs" / "campaigns" / CAMPAIGN_ID
    base_path = docs / "freeze1.json"
    base = {"code": {"tracked_file_sha256": {
        "src/frozen.py": _sha(repo / "src/frozen.py"), "src/amended.py": "0" * 64}}}
    _json_write(base_path, base)
    amendment_path = docs / "s1_amendment" / "amendment.json"
    amendment = {"base_freeze1": {"path": base_path.relative_to(repo).as_posix(), "sha256": _sha(base_path)},
                 "source_changes": {"src/amended.py": {"before": "0" * 64, "after": _sha(repo / "src/amended.py")}}}
    _json_write(amendment_path, amendment)
    _git(repo, "add", "src", "scripts", "docs")
    _git(repo, "commit", "--quiet", "-m", "Fixture baseline")
    commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", commit)
    trusted = {"document_sha256": _sha(amendment_path), "amendment_chain": [_sha(amendment_path)],
               "document": amendment}
    calls = []

    def authenticate(current_repo, current_commit, name, digest):
        calls.append((current_repo, current_commit, name, digest))
        assert current_commit == commit
        assert name == amendment_path.relative_to(repo).as_posix()
        assert digest == amendment["base_freeze1"]["sha256"]
        return trusted

    monkeypatch.setattr(phase3, "_baseline_authentication", authenticate)
    manifest = docs / "wp13_execution.json"
    phase3.make_execution_manifest(amendment_path, manifest, amendment_commit=commit,
                                   freeze1b_sha256=phase3.FREEZE1B_SHA256, repo=repo)
    _git(repo, "add", "docs")
    _git(repo, "commit", "--quiet", "-m", "Fixture execution")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo, manifest, base_path, calls


def test_execution_source_requires_published_baseline_and_exact_live_extension(execution_source):
    repo, manifest, _, calls = execution_source
    result = phase3.verify_execution_source(manifest, repo=repo)
    assert result["published"] is True
    assert result["source_sha256"] == {name: _sha(repo / name) for name in phase3.EXECUTION_FILES}
    assert result["execution_manifest_sha256"] == _sha(manifest)
    assert calls


@pytest.mark.parametrize("case", ["dirty", "unpushed", "frozen_source", "execution_source", "extra_source",
                                  "freeze1b", "amendment_chain", "amendment_bytes", "base_and_source"])
def test_execution_source_refuses_unpublished_or_tampered_state(execution_source, case):
    repo, manifest, base_path, _ = execution_source
    if case in ("dirty", "unpushed"):
        (repo / "note.txt").write_text("pending")
        if case == "unpushed":
            _git(repo, "add", "note.txt")
            _git(repo, "commit", "--quiet", "-m", "Unpublished fixture change")
    elif case in ("frozen_source", "execution_source", "extra_source"):
        name = {"frozen_source": "src/frozen.py", "execution_source": phase3.EXECUTION_FILES[0],
                "extra_source": "src/extra.py"}[case]
        (repo / name).write_text("changed\n")
    elif case in ("freeze1b", "amendment_chain"):
        record = json.loads(manifest.read_text())
        record["freeze1b_sha256" if case == "freeze1b" else "amendment_chain"] = "0" * 64 if case == "freeze1b" else []
        _json_write(manifest, record)
    elif case == "amendment_bytes":
        path = repo / json.loads(manifest.read_text())["amendment_path"]
        path.write_text(path.read_text() + "\n")
    elif case == "base_and_source":
        (repo / "src/frozen.py").write_text("changed\n")
        base = json.loads(base_path.read_text())
        base["code"]["tracked_file_sha256"]["src/frozen.py"] = _sha(repo / "src/frozen.py")
        _json_write(base_path, base)
    with pytest.raises(phase3.Phase3Error):
        phase3.verify_execution_source(manifest, repo=repo, require_published=case in ("dirty", "unpushed"))


def test_vm_probe_uses_read_only_dry_run_and_keeps_original_receipt(transfer_view, tmp_path, monkeypatch):
    original = (transfer_view.receipts / views.DRY_RUN_RECEIPT_NAME).read_bytes()
    observed = {}

    def dry_run(view, **kwargs):
        observed.update(kwargs)
        assert view.root == transfer_view.root
        return {"status": "passed", "n_commands_checked": 1}

    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: {"published": False})
    monkeypatch.setattr(phase3.commands, "dry_run", dry_run)
    output = tmp_path / "probe.json"
    result = phase3.vm_probe(transfer_view.root, tmp_path / "execution.json", output,
                             cap_prefix=("fixture-cap",), smoke=True)
    assert observed["write"] is False and observed["cap_prefix"] == ("fixture-cap",)
    assert result["status"] == "passed" and output.is_file()
    assert (transfer_view.receipts / views.DRY_RUN_RECEIPT_NAME).read_bytes() == original


def test_vm_probe_does_not_publish_failed_probe(transfer_view, tmp_path, monkeypatch):
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: {"published": False})
    monkeypatch.setattr(phase3.commands, "dry_run", lambda *args, **kwargs: {"status": "failed"})
    output = tmp_path / "probe.json"
    with pytest.raises(phase3.Phase3Error, match="probe failed"):
        phase3.vm_probe(transfer_view.root, tmp_path / "execution.json", output, cap_prefix=(), smoke=True)
    assert not output.exists()


def _source_identity():
    return {"freeze1_sha256": "1" * 64, "freeze1b_sha256": phase3.FREEZE1B_SHA256,
            "amendment_chain": ["2" * 64], "git_head": "3" * 40, "source_tree_sha256": "4" * 64,
            "execution_manifest_sha256": "5" * 64}


def test_fixed_timing_recipes_reuse_builder_and_stay_outside_selection(transfer_view):
    plan = phase3.timing_plan()
    assert set(plan) == {"F1", "F2", "F3", "F4", "F5", "C-G", "F4_L3",
                         "lane_serial_1", "lane_serial_2", "lane_parallel_1", "lane_parallel_2", "lane_parallel_3"}
    for name, recipe in plan.items():
        request = {"recipe": name, "attempt": 1, "execution_root": str(transfer_view.default_runs_dir / "g2"),
                   "device": "cpu"}
        command = phase3._build_timing_command(request, _source_identity(), transfer_view)
        assert command.identity["step"] == "G2_TIMING"
        assert command.seed == 42 and command.inner_fold == 0
        assert command.recipe.epochs == (30 if name.startswith("lane_") else 5)
        assert command.recipe.family == ("F4" if name.startswith("lane_") or name == "F4_L3" else name)
        assert command.recipe.lever_configs == (("L3",) if name == "F4_L3" else ())
        assert command.identity["freeze1_amendment_chain"] == ["2" * 64]
        assert command.identity["freeze1b_sha256"] == phase3.FREEZE1B_SHA256


@pytest.mark.parametrize("case", ["view", "dry_run", "layer"])
def test_real_view_must_match_the_original_freeze(transfer_view, case):
    view = SimpleNamespace(smoke=False, manifest_sha256=transfer_view.manifest_sha256,
                           receipts=transfer_view.receipts, chain_means_manifest_sha256=transfer_view.chain_means_manifest_sha256)
    frozen = {"view_manifest_sha256": view.manifest_sha256,
              "dry_run_receipt": {"sha256": _sha(view.receipts / views.DRY_RUN_RECEIPT_NAME)},
              "chain_means": {layer: view.chain_means_manifest_sha256(layer) for layer in ("final", "block26")}}
    phase3._bind_view(view, {"frozen_fold0_view": frozen})
    if case == "view":
        frozen["view_manifest_sha256"] = "0" * 64
    elif case == "dry_run":
        frozen["dry_run_receipt"]["sha256"] = "0" * 64
    else:
        frozen["chain_means"]["block26"] = "0" * 64
    with pytest.raises(phase3.Phase3Error, match="original Freeze 1"):
        phase3._bind_view(view, {"frozen_fold0_view": frozen})


@pytest.mark.parametrize("case", ["probe", "cap", "timeout", "allocation", "root", "persistence", "smoke"])
def test_runtime_refuses_missing_probe_or_relaxed_limits(transfer_view, tmp_path, monkeypatch, case):
    source = _source_identity()
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: source)
    probe_path = tmp_path / "probe.json"
    probe = {"status": "passed", "view_manifest_sha256": transfer_view.manifest_sha256,
             "source": {"execution_manifest_sha256": source["execution_manifest_sha256"]}}
    _json_write(probe_path, probe)
    request = {"smoke": True, "execution_manifest": str(tmp_path / "execution.json"), "view": str(transfer_view.root),
               "vm_probe": str(probe_path), "cap_prefix": ["synthetic-cap"], "device": "cpu",
               "memory_limit_bytes": 3_000_000_000, "unit_timeout_seconds": 30, "forecast_seconds": 60,
               "deadline_unix": time.time()+7200, "max_total_seconds": 7200,
               "allocation_started_unix": time.time()-1, "execution_root": str(transfer_view.default_runs_dir / "g2"),
               "persistence_mode": "host_pull"}
    _, checked_view, policy = phase3._runtime(request)
    assert checked_view.root == transfer_view.root and policy.reserve_seconds == 900 and policy.forecast_margin == 1.25
    if case == "probe":
        probe["source"]["execution_manifest_sha256"] = "0" * 64
        _json_write(probe_path, probe)
    else:
        name, value = {"cap": ("cap_prefix", []), "timeout": ("unit_timeout_seconds", 61),
                       "allocation": ("max_total_seconds", 7201), "root": ("execution_root", str(tmp_path)),
                       "persistence": ("persistence_mode", "mounted"), "smoke": ("smoke", "true")}[case]
        request[name] = value
    with pytest.raises(phase3.Phase3Error):
        phase3._runtime(request)


def test_prediction_table_reads_the_exporters_protein_columns(tmp_path):
    from training.ec_prediction_export import EC_PROTEIN_PREDICTION_COLUMNS

    path = tmp_path / "protein.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EC_PROTEIN_PREDICTION_COLUMNS)
        writer.writeheader()
        for i in range(3):
            writer.writerow({"accession": f"P{i}", "ec_group_key": f"P{i}", "y": str(i), "pred": str((i+1)%3)})
    receipt = tmp_path / "ec_prediction_receipt.json"
    _json_write(receipt, {"protein_predictions": {"path": path.name, "sha256": _sha(path)}})
    result = phase3._prediction_table(tmp_path)
    assert result == {"ids": ["P0", "P1", "P2"], "true": [0, 1, 2], "predicted": [1, 2, 0]}
    path.write_text(path.read_text() + "\n")
    with pytest.raises(phase3.Phase3Error, match="Prediction bytes"):
        phase3._prediction_table(tmp_path)


def _lane_records():
    labels = [label for label in range(6) for _ in range(100)]

    def row(name, seconds):
        return {"ids": [f"P{i}" for i in range(len(labels))], "true": labels.copy(), "predicted": labels.copy(),
                "recipe": name, "wall_seconds": seconds, "execution_manifest_sha256": "a" * 64}

    serial = [row("lane_serial_1", 100), row("lane_serial_2", 120)]
    parallel = [row(f"lane_parallel_{i}", 115) for i in (1, 2, 3)]
    limits = {"cpu_percent": 800, "ram_bytes": 32_000_000_000, "vram_bytes": 24_000_000_000}
    observed = {"cpu_percent": 700, "ram_bytes": 10_000_000_000, "vram_bytes": 12_000_000_000,
                "swap_bytes": 0, "oom": False, "read_guard_breach": False}
    return serial, parallel, limits, observed


def test_lane_gate_uses_equal_work_and_all_six_reference_pairs():
    serial, parallel, limits, observed = _lane_records()
    result = phase3.lane_gate(serial, parallel, batch_wall_seconds=275, limits=limits, observed=observed)
    assert result["status"] == "passed" and result["lanes"] == 3
    assert result["speedup"] == pytest.approx(1.2)
    assert result["supported_classes"] == list(range(6))
    assert len(result["comparisons"]) == 6
    parallel[2]["predicted"][0:4] = [1] * 4
    result = phase3.lane_gate(serial, parallel, batch_wall_seconds=270, limits=limits, observed=observed)
    assert result["status"] == "failed" and result["lanes"] == 1


@pytest.mark.parametrize("case", ["speed", "cpu", "ram", "vram", "swap", "oom", "guard", "no_support"])
def test_lane_failure_selects_serial_without_a_new_lane_search(case):
    serial, parallel, limits, observed = _lane_records()
    wall = 276 if case == "speed" else 270
    if case in ("cpu", "ram", "vram"):
        key = {"cpu": "cpu_percent", "ram": "ram_bytes", "vram": "vram_bytes"}[case]
        observed[key] = limits[key] + 1
    elif case in ("swap", "oom", "guard"):
        key = {"swap": "swap_bytes", "oom": "oom", "guard": "read_guard_breach"}[case]
        observed[key] = 1 if case == "swap" else True
    elif case == "no_support":
        for row in [*serial, *parallel]:
            for key in ("ids", "true", "predicted"):
                row[key] = row[key][:9]
    result = phase3.lane_gate(serial, parallel, batch_wall_seconds=wall, limits=limits, observed=observed)
    assert result["status"] == "failed" and result["lanes"] == 1


@pytest.mark.parametrize("case", ["source", "membership", "duplicate_ids", "recipe", "missing_peak", "nonfinite"])
def test_lane_gate_refuses_incomparable_or_unmeasured_inputs(case):
    serial, parallel, limits, observed = _lane_records()
    if case == "source":
        parallel[-1]["execution_manifest_sha256"] = "b" * 64
    elif case == "membership":
        parallel[-1]["ids"][0] = "extra"
    elif case == "duplicate_ids":
        serial[0]["ids"][1] = serial[0]["ids"][0]
    elif case == "recipe":
        parallel[0]["recipe"] = "F4"
    elif case == "missing_peak":
        del observed["vram_bytes"]
    elif case == "nonfinite":
        observed["ram_bytes"] = float("nan")
    with pytest.raises(phase3.Phase3Error):
        phase3.lane_gate(serial, parallel, batch_wall_seconds=270, limits=limits, observed=observed)


@pytest.fixture
def timing_execution(tmp_path, monkeypatch):
    """Run the real ownership/persistence wrapper around a tiny artifact-only child."""
    from benchmarking.pmm_execution import ExecutionPolicy

    source = _source_identity()
    request = {"recipe": "F2", "attempt": 1, "execution_root": str(tmp_path / "work"),
               "durable_root": str(tmp_path / "independent"), "session_id": "synthetic-owned-session",
               "python": sys.executable, "forecast_seconds": 1, "memory_limit_bytes": 3_000_000_000,
               "unit_timeout_seconds": 10, "fixture_source": source}
    request["fixture_unit_name"] = phase3._unit_name(request)
    started = time.time() - 1
    policy = [ExecutionPolicy(deadline_unix=started + 4000, max_total_seconds=4000,
                               allocation_started_unix=started)]
    monkeypatch.setattr(phase3, "_runtime", lambda req: (source, SimpleNamespace(), policy[0]))
    monkeypatch.setattr(phase3.commands, "verify_unit", lambda *args, **kwargs: {"passed": True})
    launched = []
    program = (
        "import json,pathlib,sys; r=json.load(open(sys.argv[1])); "
        "p=pathlib.Path(r['execution_root'])/'runs'/r['fixture_unit_name']; "
        "p.mkdir(parents=True); "
        "(p/'g2_timing.json').write_text(json.dumps({'status':'passed','source':r['fixture_source'],"
        "'unit_id':p.name,'wall_seconds':1.0}))"
    )

    def child(path, action, python):
        launched.append((path, action, python))
        return [python, "-c", program, str(path)]

    monkeypatch.setattr(phase3, "_child_command", child)
    return request, policy, launched


def _acknowledge(request, transfer):
    root, durable = Path(request["execution_root"]), Path(request["durable_root"])
    manifest_path = root / transfer["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["files"]:
        target = durable / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / row["path"], target)
    copied_manifest = durable / transfer["manifest_path"]
    copied_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, copied_manifest)
    phase3.host_ack(copied_manifest, durable, root / transfer["ack_path"], transfer_seconds=.5)


def test_timing_wrapper_requires_ack_then_reuses_verified_result(timing_execution):
    from benchmarking.pmm_execution import ExecutionBlocked

    request, _, launched = timing_execution
    first = phase3.run_timing_unit(request)
    assert first["status"] == "awaiting_host_ack" and len(launched) == 1
    assert launched[0][1] == "_timing-worker"
    with pytest.raises(ExecutionBlocked, match="download/verify"):
        phase3.run_timing_unit(request)
    assert len(launched) == 1
    _acknowledge(request, first["persistence"]["transfer"])
    reused = phase3.run_timing_unit(request)
    assert reused["status"] == "reused_verified" and len(launched) == 1


def test_timing_wrapper_refuses_wrong_ack_without_relaunch(timing_execution):
    from benchmarking.pmm_execution import PersistenceError

    request, _, launched = timing_execution
    first = phase3.run_timing_unit(request)
    transfer = first["persistence"]["transfer"]
    _json_write(Path(request["execution_root"]) / transfer["ack_path"], {"manifest_sha256": "0" * 64})
    with pytest.raises(PersistenceError, match="acknowledgment"):
        phase3.run_timing_unit(request)
    assert len(launched) == 1


def test_timing_wrapper_deadline_refuses_before_child_launch(timing_execution):
    from benchmarking.pmm_execution import ExecutionBlocked, ExecutionPolicy

    request, policy, launched = timing_execution
    started = time.time() - 1
    policy[0] = ExecutionPolicy(deadline_unix=started + 600, max_total_seconds=600,
                                allocation_started_unix=started)
    with pytest.raises(ExecutionBlocked, match="needs"):
        phase3.run_timing_unit(request)
    assert launched == []


def test_timing_wrapper_saves_failed_child_before_new_seed_attempt(timing_execution, monkeypatch):
    from benchmarking.pmm_execution import ExecutionBlocked

    request, _, launched = timing_execution
    monkeypatch.setattr(phase3, "_child_command", lambda *args: [sys.executable, "-c", "raise SystemExit(7)"])
    with pytest.raises(phase3.Phase3Error, match="Timing failed"):
        phase3.run_timing_unit(request)
    root = Path(request["execution_root"])
    state = json.loads((root / "execution_state.json").read_text())
    assert state["pending_unit"]["status"] == "failed_timing"
    assert state["pending_transfer"] and not state["pending_unit"]["persisted"]
    with pytest.raises(ExecutionBlocked):
        phase3.run_timing_unit({**request, "attempt": 2})


@pytest.fixture
def freeze1c_evidence(tmp_path, monkeypatch):
    source = _source_identity()
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: source)
    common = {"status": "passed", "execution_manifest_sha256": source["execution_manifest_sha256"],
              "host_transfer": {"manifest_sha256": "b" * 64, "ack_sha256": "c" * 64}, "wall_seconds": 100.0}
    records = []
    for name in ("F1", "F2", "F3", "F4", "F5", "C-G", "F4_L3"):
        records.append({**common, "format": "deepmzyme_ec_clean30_completed_timing_v1", "recipe": name})
    for kind in ("esmc", "esm1b"):
        records.append({**common, "format": "deepmzyme_ec_clean30_completed_embedding_v1", "kind": kind})
    serial, parallel, limits, observed = _lane_records()
    for row in [*serial, *parallel]:
        row["execution_manifest_sha256"] = source["execution_manifest_sha256"]
    gate = phase3.lane_gate(serial, parallel, batch_wall_seconds=276, limits=limits, observed=observed)
    gate["host_transfer"] = dict(common["host_transfer"])
    records.append(gate)
    evidence = []
    for index, record in enumerate(records):
        path = tmp_path / "evidence" / f"record{index}.json"
        _json_write(path, record)
        evidence.append(path)
    return source, evidence


def test_freeze1c_uses_the_existing_selector_ceiling_mapping(freeze1c_evidence, tmp_path):
    from test_ec_clean30_selector import FREEZE1, _cost_world, _reseal_cost, _verify_cost

    source, evidence = freeze1c_evidence
    source["freeze1_sha256"] = FREEZE1
    out = tmp_path / "completed_freeze1c.json"
    result = phase3.build_freeze1c(evidence, out, execution_manifest_path=tmp_path / "execution.json",
                                    l3_vm_hours_per_fold={str(k): 1.0 for k in range(5)})
    assert result["campaign_id"] == CAMPAIGN_ID and result["freeze1_sha256"] == FREEZE1
    assert result["lanes"] == 1 and result["budget_authority"] == "separate_G3_approval_required"
    assert result["l3_vm_hours_per_fold"] == {str(k): 1.0 for k in range(5)}
    (tmp_path / "selector").mkdir()
    world = _cost_world(tmp_path / "selector", 0)
    frozen = world[-2]
    frozen.write_bytes(out.read_bytes())
    for name in ("accounting.json", "authority.json", "l3_cost_disposition.json"):
        path = frozen.parent / name
        record = json.loads(path.read_text())
        record["freeze1c_sha256"] = _sha(frozen)
        _json_write(path, record)
    _reseal_cost(frozen.parent)
    assert _verify_cost(world)["ceiling_vm_hours"] == 1.0


@pytest.mark.parametrize("case", ["missing_timing", "failed_timing", "foreign_source", "missing_ack", "missing_embedding",
                                  "missing_fold", "nonpositive_ceiling"])
def test_freeze1c_refuses_incomplete_evidence_or_invalid_ceilings(freeze1c_evidence, tmp_path, case):
    _, evidence = freeze1c_evidence
    ceilings = {str(k): 1.0 for k in range(5)}
    if case == "missing_timing":
        evidence = evidence[1:]
    elif case == "missing_embedding":
        evidence = [path for path in evidence if json.loads(path.read_text()).get("kind") != "esm1b"]
    elif case in ("failed_timing", "foreign_source", "missing_ack"):
        record = json.loads(evidence[0].read_text())
        if case == "failed_timing":
            record["status"] = "failed"
        elif case == "foreign_source":
            record["execution_manifest_sha256"] = "0" * 64
        else:
            record["host_transfer"].pop("ack_sha256")
        _json_write(evidence[0], record)
    elif case == "missing_fold":
        del ceilings["4"]
    elif case == "nonpositive_ceiling":
        ceilings["0"] = 0
    out = tmp_path / "freeze1c.json"
    with pytest.raises(phase3.Phase3Error):
        phase3.build_freeze1c(evidence, out, execution_manifest_path=tmp_path / "execution.json",
                                l3_vm_hours_per_fold=ceilings)
    assert not out.exists()
