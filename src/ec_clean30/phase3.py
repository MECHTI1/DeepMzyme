"""Small transfer, measurement and persistence tools for the frozen CLEAN30 campaign.

This module owns no cloud lifecycle. Measurements have separate G2 identities and
cannot be selected as S1 fits. Existing views, command builders, replay and execution
hooks remain authoritative; no preparation or scientific policy is regenerated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Mapping, Sequence

from ec_clean30 import commands, freeze, provenance, views
from ec_clean30.campaign import CAMPAIGN_ID, FAMILIES, KIND_INNER, default_smoke_recipe, recipe_with_lever
from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy, verify_host_pull

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_FILES = ("src/ec_clean30/phase3.py", "scripts/run_ec_clean30_phase3.py")
TRANSFER_MANIFEST = "phase3_transfer.json"
RECEIPT_NAMES = (views.DRY_RUN_RECEIPT_NAME, views.VERIFY_RECEIPT_NAME,
                 views.FILE_MODES_RECEIPT_NAME, views.FILE_MODES_APPLIED_RECEIPT_NAME)
TIMING_ROWS = 1000
FREEZE1B_SHA256 = "fc6f032805f084d7aef4f704a42149e92fb7894af3699b2ad5c12e86c87bcbdb"
OFFICIAL_FOLD0_SHA256 = "ab8a8a21219aec19ba03dd7205dec8d8c1a156ff5cfad23aad344c7d7fbdd9c1"
EXECUTION_FORMAT = "deepmzyme_ec_clean30_execution_v1"
AUTOMATION_FORMAT = "deepmzyme_ec_clean30_execution_v2"
AUTOMATION_FILES = (*EXECUTION_FILES, "src/ec_clean30/phase4.py", "scripts/run_ec_clean30_phase4.py",
                    "src/ec_clean30/workstation.py", "scripts/run_ec_clean30_workstation.py")
TRANSFER_FORMAT = "deepmzyme_ec_clean30_fold0_transfer_v1"


class Phase3Error(ValueError):
    """The frozen inputs or runtime evidence do not permit the requested action."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise Phase3Error(message)


def _sha(path: Path) -> str:
    return provenance.sha256_file(Path(path))


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_bytes(value))
    return _sha(path)


def _relative(name: str) -> str:
    path = PurePosixPath(name)
    _require(bool(name) and not path.is_absolute() and ".." not in path.parts
             and "." not in path.parts and "\\" not in name and str(path) == name,
             "Unsafe relative member name.")
    return name


def _safe_file(root: Path, name: str) -> Path:
    name = _relative(name)
    path = root / name
    current = root
    for part in PurePosixPath(name).parts:
        current /= part
        _require(not current.is_symlink(), "Input contains a symlink.")
    _require(path.is_file(), f"Required regular file is absent: {name}")
    return path


def _no_restricted_path(path: Path) -> None:
    for form in (Path(path).absolute(), Path(path).resolve()):
        parts = {part.lower() for part in form.parts}
        _require(not parts.intersection({"identity", "test", "held-out", "held_out"})
                 and not any(part.startswith("identity_after_") or "_test_split_" in part for part in parts),
                 "Restricted identity/test paths cannot be used by Phase 3.")


def _outside_guard(path: Path, roots: Sequence[str]) -> None:
    _no_restricted_path(path)
    for form in (Path(path).absolute(), Path(path).resolve()):
        for root in roots:
            for forbidden in (Path(root).absolute(), Path(root).resolve()):
                _require(form != forbidden and forbidden not in form.parents,
                         "Timing input, output or weights lie within a frozen forbidden-read root.")


def _transfer_names(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    names = {views.VIEW_MANIFEST_NAME}
    names.update(_relative(row["path"]) for row in manifest["files"])
    _require(not any(PurePosixPath(name).parts[0] in views.WRITABLE_DIRS for name in names),
             "The immutable view manifest lists a writable/cache location.")
    for directory in views.CHAIN_MEAN_DIRS.values():
        cache = _json(_safe_file(root, f"{directory}/manifest.json"))
        names.add(f"{directory}/manifest.json")
        for row in cache["files"]:
            _require(PurePosixPath(row["file"]).name == row["file"], "Unsafe chain-cache file.")
            names.add(f"{directory}/{_relative(row['file'])}")
    for name in RECEIPT_NAMES:
        relative = f"{views.RECEIPTS_DIR}/{name}"
        if (root / relative).exists():
            names.add(relative)
    _require(f"{views.RECEIPTS_DIR}/{views.DRY_RUN_RECEIPT_NAME}" in names,
             "Transfer requires the original completed dry-run receipt.")
    return sorted(names)


def _member_mode(name: str) -> int:
    return 0o644 if name.startswith(views.RECEIPTS_DIR + "/") else 0o444


def pack_view(view_root: Path, archive_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    """Write only the exact fold-0 view closure; stale probes and caches never travel."""
    _no_restricted_path(view_root)
    view = views.View.load(view_root)
    _require(view.fold == 0 and view.smoke == smoke, "Only the declared fold-0 view may travel.")
    views.verify_view(view.root, require_chain_means=("final", "block26"), write_receipt=False)
    receipt = views.read_dry_run_receipt(view)
    _require(receipt is not None and receipt.get("status") == "passed"
             and receipt.get("view_manifest_sha256") == view.manifest_sha256,
             "The original dry-run receipt does not certify this view.")
    names = _transfer_names(view.root, view.manifest)
    files = [{"path": name, "sha256": _sha(_safe_file(view.root, name)),
              "size_bytes": (view.root / name).stat().st_size, "mode": _member_mode(name)} for name in names]
    manifest = {"format": TRANSFER_FORMAT, "campaign_id": CAMPAIGN_ID, "fold": 0, "smoke": smoke,
                "view_manifest_sha256": view.manifest_sha256, "files": files,
                "original_dry_run_sha256": _sha(view.receipts / views.DRY_RUN_RECEIPT_NAME)}
    archive_path = Path(archive_path)
    _no_restricted_path(archive_path)
    _require(not archive_path.exists(), "Transfer archive already exists.")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(archive_path.name + ".partial")
    _require(not partial.exists(), "A previous partial transfer archive needs reconciliation.")
    try:
        with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
            payload = _bytes(manifest)
            info = tarfile.TarInfo(TRANSFER_MANIFEST)
            info.size, info.mode, info.mtime = len(payload), 0o444, 0
            archive.addfile(info, io.BytesIO(payload))
            for row in files:
                path = _safe_file(view.root, row["path"])
                info = tarfile.TarInfo(row["path"])
                info.size, info.mode, info.mtime = row["size_bytes"], row["mode"], 0
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
                _require(_sha(path) == row["sha256"], "View changed while it was being archived.")
        partial.rename(archive_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"status": "passed", "archive_sha256": _sha(archive_path), "files": len(files),
            "archive_bytes": archive_path.stat().st_size, "unpacked_bytes": sum(row["size_bytes"] for row in files),
            "view_manifest_sha256": view.manifest_sha256,
            "original_dry_run_sha256": manifest["original_dry_run_sha256"]}


def unpack_view(archive_path: Path, destination: Path, *, expected_sha256: str,
                smoke: bool = False) -> dict[str, Any]:
    """Validate every tar member before extracting into an isolated staging view."""
    archive_path, destination = Path(archive_path), Path(destination)
    _no_restricted_path(archive_path)
    _no_restricted_path(destination)
    _require(_sha(archive_path) == expected_sha256, "Transfer archive checksum differs.")
    _require(not destination.exists(), "Transfer destination must not exist.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        names = [_relative(member.name) for member in members]
        _require(len(names) == len(set(names)), "Duplicate archive member.")
        _require(all(member.isreg() and not member.linkname for member in members),
                 "Only regular files are accepted; links and special members are forbidden.")
        _require(TRANSFER_MANIFEST in names, "Transfer manifest is missing.")
        descriptor = archive.getmember(TRANSFER_MANIFEST)
        _require(descriptor.size <= 16 * 1024 * 1024 and descriptor.mode == 0o444,
                 "Invalid transfer manifest size or permissions.")
        manifest = json.load(archive.extractfile(descriptor))
        _require(manifest.get("format") == TRANSFER_FORMAT and manifest.get("campaign_id") == CAMPAIGN_ID
                 and manifest.get("fold") == 0 and manifest.get("smoke") is smoke,
                 "Transfer is not the declared fold-0 view.")
        records = manifest.get("files", [])
        expected = {row["path"]: row for row in records}
        _require(len(expected) == len(records) and set(names) == {TRANSFER_MANIFEST, *expected},
                 "Archive members differ from its exact transfer manifest.")
        for name, row in expected.items():
            _relative(name)
            member = archive.getmember(name)
            _require(row["mode"] == _member_mode(name) == member.mode
                     and row["size_bytes"] == member.size, "Archive size or permission differs.")
        with tempfile.TemporaryDirectory(prefix="ec30-unpack-", dir=destination.parent) as temporary:
            stage = Path(temporary) / CAMPAIGN_ID / ("smoke_views" if smoke else "views") / destination.name
            stage.mkdir(parents=True)
            for name, row in expected.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(name) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output)
                _require(_sha(path) == row["sha256"], "Extracted member checksum differs.")
                path.chmod(row["mode"])
            embedded = _json(stage / views.VIEW_MANIFEST_NAME)
            _require(set(_transfer_names(stage, embedded)) == set(expected),
                     "Transfer includes files outside the view and cache manifests.")
            _require(_sha(stage / views.VIEW_MANIFEST_NAME) == manifest["view_manifest_sha256"],
                     "View manifest checksum differs.")
            for name in (*views.WRITABLE_DIRS, views.RING_DIR):
                (stage / name).mkdir(exist_ok=True)
            for name in views.CACHE_SUBDIRS:
                (stage / views.CACHE_DIR / name).mkdir(exist_ok=True)
            # Existing permission and verification hooks enforce the relocated view contract.
            try:
                modes = views._set_permissions(stage)
                _require(modes["outcome"] == views.FILE_MODES_ENFORCED,
                         "The extraction filesystem does not enforce required view permissions.")
                views.verify_view(stage, require_chain_means=("final", "block26"), write_receipt=False)
            except BaseException:
                views._remove_tree(stage)
                raise
            _require(_sha(stage / views.RECEIPTS_DIR / views.DRY_RUN_RECEIPT_NAME)
                     == manifest["original_dry_run_sha256"], "Original dry-run receipt changed.")
            # Moving a directory between parents updates its '..' entry on POSIX.
            stage.chmod(0o755)
            stage.rename(destination)
            destination.chmod(views.DATA_DIR_MODE)
    return {"status": "passed", "archive_sha256": expected_sha256,
            "view_manifest_sha256": manifest["view_manifest_sha256"], "files": len(expected),
            "original_dry_run_sha256": manifest["original_dry_run_sha256"]}


def timing_inputs(training_file: Path, out: Path, *, expected_sha256: str) -> dict[str, Any]:
    """The deterministic 1,000-row fold-0 sequence timing set; no other fold is read."""
    from scripts.run_clean_split30_comparator import SEQUENCE_MAP_COLUMNS, read_official_training_rows

    training_file, out = Path(training_file), Path(out)
    _no_restricted_path(training_file)
    _no_restricted_path(out)
    _require(_sha(training_file) == expected_sha256, "Official training input checksum differs.")
    rows, _ = read_official_training_rows(training_file, 0, [])
    _require(len(rows) >= TIMING_ROWS, "Timing input needs at least 1,000 official training rows.")
    chosen = sorted(rows, key=lambda row: (hashlib.sha256(row.sequence.encode()).hexdigest(), row.accession))[:TIMING_ROWS]
    _require(not out.exists(), "Timing input output already exists.")
    out.mkdir(parents=True)
    with (out / "timing.fasta").open("x") as fasta, (out / "sequence_map.tsv").open("x", newline="") as table:
        writer = csv.writer(table, delimiter="\t", lineterminator="\n")
        writer.writerow(SEQUENCE_MAP_COLUMNS)
        for row in chosen:
            fasta.write(f">{row.accession}\n{row.sequence}\n")
            writer.writerow((row.accession, hashlib.sha256(row.sequence.encode()).hexdigest(), len(row.sequence), 1, row.sequence))
    receipt = {"format": "deepmzyme_ec_clean30_timing_inputs_v1", "fold": 0, "n_rows": TIMING_ROWS,
               "source_name": training_file.name, "source_sha256": expected_sha256,
               "rule": "ascending sequence SHA-256, then accession; first 1000 rows",
               "files": {name: _sha(out / name) for name in ("timing.fasta", "sequence_map.tsv")}}
    _write(out / "timing_inputs.json", receipt)
    return receipt


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE).strip()


def _baseline_authentication(repo: Path, commit: str, amendment_name: str, base_digest: str) -> dict[str, Any]:
    """Run the unchanged verifier in a sparse checkout of its published source version."""
    _require(re.fullmatch(r"[0-9a-f]{40}", commit), "Amendment commit must be a full Git object ID.")
    _require(_git(repo, "merge-base", commit, "refs/remotes/origin/main") == commit,
             "The amendment source commit is not published.")
    with tempfile.TemporaryDirectory(prefix="ec30-amendment-source-") as temporary:
        baseline = Path(temporary) / "source"
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(baseline)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _git(baseline, "sparse-checkout", "set", "--no-cone", "/src/", "/scripts/",
             "/" + freeze.CAMPAIGN_DOCS_DIR + "/")
        _git(baseline, "checkout", "--quiet", "--detach", commit)
        return freeze.verify_s1_amendment(baseline / amendment_name, expected_freeze1_sha256=base_digest,
                                         repo=baseline, require_published=True)


def make_execution_manifest(amendment_path: Path, out: Path, *, amendment_commit: str,
                            freeze1b_sha256: str, timing_inputs_sha256: str | None = None,
                            repo: Path = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo).resolve()
    amendment_path = Path(amendment_path).resolve()
    amendment_name = amendment_path.relative_to(repo).as_posix()
    amendment = _json(amendment_path)
    base_digest = amendment["base_freeze1"]["sha256"]
    verified = _baseline_authentication(repo, amendment_commit, amendment_name, base_digest)
    _require(_sha(amendment_path) == verified["document_sha256"], "Live amendment differs from published bytes.")
    _require(freeze1b_sha256 == FREEZE1B_SHA256, "Freeze-1b binding differs from the published audit.")
    document = {"format": EXECUTION_FORMAT, "campaign_id": CAMPAIGN_ID,
                "freeze1_sha256": base_digest, "freeze1b_sha256": freeze1b_sha256,
                "amendment_path": amendment_name, "amendment_sha256": verified["document_sha256"],
                "amendment_commit": amendment_commit, "amendment_chain": verified["amendment_chain"],
                "timing_inputs_sha256": timing_inputs_sha256,
                "official_fold0_training_sha256": OFFICIAL_FOLD0_SHA256,
                "source_sha256": {name: _sha(_safe_file(repo, name)) for name in EXECUTION_FILES},
                "purpose": "G2 measurements only; separate from S1 scientific fits"}
    _write(out, document)
    verify_execution_source(out, repo=repo, require_published=False)
    return document


def verify_execution_source(manifest_path: Path, *, repo: Path = REPO_ROOT,
                            require_published: bool = True) -> dict[str, Any]:
    repo, manifest_path = Path(repo).resolve(), Path(manifest_path).absolute()
    _require(repo in manifest_path.parents and not manifest_path.is_symlink()
             and manifest_path.resolve() == manifest_path,
             "Execution manifest must be a regular repository document.")
    document = _json(manifest_path)
    _require(document.get("format") in {EXECUTION_FORMAT, AUTOMATION_FORMAT}
             and document.get("campaign_id") == CAMPAIGN_ID,
             "Unknown execution manifest.")
    _require(document.get("freeze1b_sha256") == FREEZE1B_SHA256, "Freeze-1b binding is missing or changed.")
    verified = _baseline_authentication(repo, document["amendment_commit"], document["amendment_path"],
                                        document["freeze1_sha256"])
    _require(document.get("amendment_sha256") == verified["document_sha256"]
             and document.get("amendment_chain") == verified["amendment_chain"], "Amendment binding differs.")
    amendment = verified["document"]
    base_path = _safe_file(repo, amendment["base_freeze1"]["path"])
    _require(_sha(base_path) == document["freeze1_sha256"] == amendment["base_freeze1"]["sha256"],
             "Live original Freeze 1 changed.")
    base = _json(base_path)
    _require(_sha(repo / document["amendment_path"]) == verified["document_sha256"], "Live amendment changed.")
    expected = dict(base["code"]["tracked_file_sha256"])
    expected.update({name: row["after"] for name, row in amendment["source_changes"].items()})
    pins = document.get("source_sha256", {})
    if document["format"] == AUTOMATION_FORMAT:
        _verify_execution_base(repo, document)
        allowed = AUTOMATION_FILES
    else:
        allowed = EXECUTION_FILES
    _require(set(pins) == set(allowed), "Execution pins must name exactly the versioned execution files.")
    expected.update(pins)
    _require(freeze._amendment_code(repo) == expected, "Live source differs from frozen and execution pins.")
    head = _git(repo, "rev-parse", "HEAD")
    published = (_git(repo, "merge-base", head, "refs/remotes/origin/main") == head
                 and not _git(repo, "status", "--porcelain"))
    if require_published:
        _require(published, "Real G2 work requires a clean, published execution checkout.")
        content = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{manifest_path.relative_to(repo)}"])
        _require(content == manifest_path.read_bytes(), "Execution manifest is not published with these bytes.")
    return {**document, "execution_manifest_sha256": _sha(manifest_path), "published": published,
            "git_head": head, "source_tree_sha256": provenance.source_tree_sha256(repo),
            "frozen_fold0_view": base.get("views", {}).get("0"),
            "frozen_esmc_weight_sha256": base.get("pins", {}).get("esmc_300m_weights")}


def _verify_execution_base(repo: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """The automation extension preserves its published WP-13 manifest and scientific bindings."""
    reference = document.get("base_execution", {})
    commit = reference.get("commit", "")
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
             "Automation requires a full published WP-13 commit.")
    _require(_git(repo, "merge-base", commit, "refs/remotes/origin/main") == commit
             and _git(repo, "merge-base", commit, "HEAD") == commit,
             "WP-13 base must be published and an ancestor of the execution checkout.")
    name = _relative(reference.get("path", ""))
    path = _safe_file(repo, name)
    published = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"])
    _require(_sha(path) == reference.get("sha256") == hashlib.sha256(published).hexdigest(),
             "Original published WP-13 manifest changed.")
    base = json.loads(published)
    _require(base.get("format") == EXECUTION_FORMAT and set(base.get("source_sha256", {})) == set(EXECUTION_FILES),
             "Automation base must be the original two-file WP-13 execution manifest.")
    for key in ("campaign_id", "freeze1_sha256", "freeze1b_sha256", "amendment_path", "amendment_sha256",
                "amendment_commit", "amendment_chain", "official_fold0_training_sha256", "timing_inputs_sha256"):
        _require(document.get(key) == base.get(key), f"Automation changed the WP-13 {key} binding.")
    for name, digest in base["source_sha256"].items():
        content = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"])
        _require(hashlib.sha256(content).hexdigest() == digest, "Published WP-13 source pin differs.")
    return base


def make_automation_manifest(base_execution: Path, out: Path, *, base_commit: str,
                             repo: Path = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo).resolve()
    base_execution = Path(base_execution).resolve()
    document = {**_json(base_execution), "format": AUTOMATION_FORMAT,
                "base_execution": {"path": base_execution.relative_to(repo).as_posix(),
                                   "sha256": _sha(base_execution), "commit": base_commit},
                "source_sha256": {name: _sha(_safe_file(repo, name)) for name in AUTOMATION_FILES},
                "purpose": "G2 automation and Fold-0 S1/CLEAN dispatch; Phase 4 requires numerical G3 authority"}
    _verify_execution_base(repo, document)
    _write(out, document)
    verify_execution_source(out, repo=repo, require_published=False)
    return document


def _bind_view(view: views.View, source: Mapping[str, Any]) -> None:
    if view.smoke:
        return
    frozen = source.get("frozen_fold0_view") or {}
    _require(frozen.get("view_manifest_sha256") == view.manifest_sha256
             and frozen.get("dry_run_receipt", {}).get("sha256") == _sha(view.receipts / views.DRY_RUN_RECEIPT_NAME)
             and all(frozen.get("chain_means", {}).get(layer) == view.chain_means_manifest_sha256(layer)
                     for layer in ("final", "block26")), "Fold-0 view differs from original Freeze 1.")


def vm_probe(view_root: Path, execution_manifest_path: Path, out: Path, *,
             cap_prefix: Sequence[str], python: str = sys.executable, smoke: bool = False) -> dict[str, Any]:
    source = verify_execution_source(execution_manifest_path, require_published=not smoke)
    view = views.View.load(view_root)
    _require(view.fold == 0 and view.smoke == smoke, "Probe view differs from its declared role.")
    _bind_view(view, source)
    original = _sha(view.receipts / views.DRY_RUN_RECEIPT_NAME)
    verification = views.verify_view(view.root, require_chain_means=("final", "block26"), write_receipt=False)
    report = commands.dry_run(view, cap_prefix=cap_prefix, python=python, write=False)
    _require(_sha(view.receipts / views.DRY_RUN_RECEIPT_NAME) == original, "Probe rewrote the original dry run.")
    _require(report.get("status") == "passed", "Relocated VM dry-run probe failed.")
    receipt = {"format": "deepmzyme_ec_clean30_vm_probe_v1", "status": "passed", "source": source,
               "view_manifest_sha256": view.manifest_sha256, "original_dry_run_sha256": original,
               "verification": verification, "dry_run": report}
    _write(out, receipt)
    return receipt


def timing_plan() -> dict[str, dict[str, Any]]:
    result = {name: {"family": name, "epochs": 5, "lever": None} for name in FAMILIES}
    result["F4_L3"] = {"family": "F4", "epochs": 5, "lever": "L3"}
    for name in ("lane_serial_1", "lane_serial_2", "lane_parallel_1", "lane_parallel_2", "lane_parallel_3"):
        result[name] = {"family": "F4", "epochs": 30, "lever": None}
    return result


def _runtime(request: Mapping[str, Any]) -> tuple[dict[str, Any], views.View, ExecutionPolicy]:
    smoke = request.get("smoke", False)
    _require(type(smoke) is bool, "smoke must be explicit boolean.")
    source = verify_execution_source(Path(request["execution_manifest"]), require_published=not smoke)
    view = views.View.load(request["view"])
    _require(view.fold == 0 and view.smoke == smoke, "Runtime view role differs.")
    _bind_view(view, source)
    probe = _json(Path(request["vm_probe"]))
    _require(probe.get("status") == "passed" and probe.get("view_manifest_sha256") == view.manifest_sha256
             and probe.get("source", {}).get("execution_manifest_sha256") == source["execution_manifest_sha256"],
             "Runtime requires its matching completed VM probe.")
    _require(isinstance(request.get("cap_prefix"), list) and request["cap_prefix"], "A nonempty runtime cap is required.")
    _require(request.get("device") in {"cpu", "cuda"} and (smoke or request["device"] == "cuda"),
             "Real G2 timing requires the declared CUDA device.")
    _require(int(request["memory_limit_bytes"]) > 0, "Memory limit must be positive.")
    _require(0 < float(request["unit_timeout_seconds"]) <= float(request["forecast_seconds"]),
             "The bounded probe timeout must fit its full-unit admission forecast.")
    policy = ExecutionPolicy(deadline_unix=float(request["deadline_unix"]),
                             max_total_seconds=float(request["max_total_seconds"]),
                             allocation_started_unix=float(request["allocation_started_unix"]),
                             reserve_seconds=900, forecast_margin=1.25)
    _require(policy.max_total_seconds <= 7200, "This G2 package permits at most a two-hour allocation.")
    execution_root = Path(request["execution_root"]).resolve()
    _require(execution_root == (view.default_runs_dir / "g2").resolve(), "G2 uses its dedicated fold-0 execution root.")
    _require(request.get("persistence_mode") == "host_pull", "G2 uses independent host-pull acknowledgment.")
    return source, view, policy


def _unit_name(request: Mapping[str, Any]) -> str:
    name, attempt = request["recipe"], request.get("attempt", 1)
    _require(name in timing_plan() and type(attempt) is int and attempt > 0, "Unknown timing recipe or attempt.")
    return f"ecc30timing_{name}_attempt{attempt}"


def _build_timing_command(request: Mapping[str, Any], source: Mapping[str, Any], view: views.View):
    spec = timing_plan()[request["recipe"]]
    recipe = default_smoke_recipe(spec["family"], epochs=spec["epochs"], lr_index=1)
    if spec["lever"]:
        recipe = recipe_with_lever(recipe, spec["lever"])
    return commands.build_unit_command(view, recipe, unit_id=_unit_name(request), kind=KIND_INNER,
        seed=42, runs_dir=Path(request["execution_root"]) / "runs", step="G2_TIMING", inner_fold=0,
        device=request["device"], freeze1_sha256=source["freeze1_sha256"],
        freeze1b_sha256=source["freeze1b_sha256"], freeze1_amendment_chain=source["amendment_chain"],
        git_head=source["git_head"], tree_sha256=source["source_tree_sha256"])


def _prediction_table(run_dir: Path) -> dict[str, list[Any]]:
    prediction = _json(run_dir / "ec_prediction_receipt.json")
    path = run_dir / prediction["protein_predictions"]["path"]
    _require(_sha(path) == prediction["protein_predictions"]["sha256"], "Prediction bytes changed.")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # The exporter owns these column names and the protein-level averaging.
    ids = [row["ec_group_key"] for row in rows]
    _require(bool(ids) and len(ids) == len(set(ids)), "Protein export has duplicate or empty identities.")
    return {"ids": ids, "true": [int(row["y"]) for row in rows],
            "predicted": [int(row["pred"]) for row in rows]}


def timing_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    """One owned child does fit -> verify -> independent replay, with a fixed timing identity."""
    started = time.monotonic()
    source, view, _ = _runtime(request)
    command = _build_timing_command(request, source, view)
    prepared = time.monotonic()
    runner = commands.run_unit(command, view=view, cap_prefix=request["cap_prefix"], python=request["python"])
    _require(runner.get("status") == "completed", "Timing fit failed; preserve its attempt and restart from seed.")
    fitted = time.monotonic()
    verified = commands.verify_unit(Path(command.runs_dir), command.unit_id, view=view,
                                   memory_limit_bytes=int(request["memory_limit_bytes"]))
    _require(verified.get("passed"), "Timing unit verification failed.")
    checked = time.monotonic()
    run_dir = Path(command.runs_dir) / command.unit_id
    replay_argv = [request["python"], "-m", "training.ec_prediction_export", "replay",
                   "--run-dir", str(run_dir), "--device", request["device"], "--logit-atol", "1e-6"]
    for root in view.forbidden_read_roots:
        replay_argv += ["--forbidden-read-root", root]
    env = command.environment()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    with (run_dir / "g2_replay.log").open("x") as log:
        subprocess.run([*request["cap_prefix"], *replay_argv], cwd=command.runs_dir, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    replay = run_dir / "ec_prediction_replay" / "ec_prediction_replay_receipt.json"
    _require(replay.is_file(), "Independent replay receipt is missing.")
    finished = time.monotonic()
    record = {"format": "deepmzyme_ec_clean30_timing_unit_v1", "status": "passed", "recipe": request["recipe"],
              "unit_id": command.unit_id, "source": source, "view_manifest_sha256": view.manifest_sha256,
              "execution_manifest_sha256": source["execution_manifest_sha256"],
              "runner_sha256": _sha(commands.runner_receipt_path(Path(command.runs_dir), command.unit_id)),
              "replay_sha256": _sha(replay), "verification": verified,
              "seconds": {"preparation": prepared-started, "fit_and_export": fitted-prepared,
                          "verification": checked-fitted, "independent_replay": finished-checked},
              "trainer_runtime_profile": _json(run_dir / "runtime_profile.json"),
              "wall_seconds": finished-started, **_prediction_table(run_dir)}
    _write(run_dir / "g2_timing.json", record)
    return record


def _execution(request: Mapping[str, Any], policy: ExecutionPolicy) -> CampaignExecution:
    return CampaignExecution(Path(request["execution_root"]), policy=policy,
                             session_id=request["session_id"], durable_root=Path(request["durable_root"]),
                             persistence_mode="host_pull")


def _child_command(request_file: Path, action: str, python: str) -> list[str]:
    return [python, str(REPO_ROOT / "scripts/run_ec_clean30_phase3.py"), action, "--request", str(request_file)]


def run_timing_unit(request: Mapping[str, Any]) -> dict[str, Any]:
    source, view, policy = _runtime(request)
    name = _unit_name(request)
    root = Path(request["execution_root"])
    output = root / "runs" / name / "g2_timing.json"
    with _execution(request, policy) as execution:
        if output.exists():
            record = _json(output)
            _require(record.get("status") == "passed" and record.get("source") == source,
                     "Existing timing result has another source or failed.")
            verification = commands.verify_unit(root / "runs", name, view=view,
                                                memory_limit_bytes=int(request["memory_limit_bytes"]))
            _require(verification["passed"], "Existing timing result no longer verifies.")
            _require(execution.state.get("pending_unit", {}).get("persisted"),
                     "Existing timing result awaits persistence acknowledgment.")
            complete = _complete_timing(record, execution, output)
            return {"status": "reused_verified", "unit_id": name, "timing_sha256": _sha(output),
                    "complete": complete}
        request_file = root / "commands" / f"{name}.json"
        _write(request_file, dict(request))
        execution.admit(name, float(request["forecast_seconds"]))
        started = time.monotonic()
        with (root / "commands" / f"{name}.log").open("x") as log:
            result = execution.run_subprocess(_child_command(request_file, "_timing-worker", request["python"]),
                                              cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                              timeout=float(request["unit_timeout_seconds"]))
        elapsed = time.monotonic()-started
        passed = result.returncode == 0 and output.is_file() and _json(output).get("status") == "passed"
        execution.record_result(name, "completed" if passed else "failed_timing", elapsed)
        artifacts = [request_file, root / "commands" / f"{name}.log"]
        artifacts += [path for path in (root / "runs" / name, root / "runs" / f"{name}.log",
                      commands.runner_receipt_path(root / "runs", name)) if path.exists()]
        persisted = execution.persist(artifacts)
        _require(passed, "Timing failed; saved artifacts must be acknowledged before an original-seed retry.")
        return {"status": "awaiting_host_ack", "unit_id": name, "timing_sha256": _sha(output),
                "wall_seconds": elapsed, "persistence": persisted}


def host_ack(manifest_path: Path, downloaded_root: Path, ack_path: Path, *, transfer_seconds: float) -> dict[str, Any]:
    _require(math.isfinite(transfer_seconds) and transfer_seconds >= 0, "Record actual completed transfer seconds.")
    _require(not Path(ack_path).exists(), "Host acknowledgment already exists.")
    started = time.monotonic()
    record = verify_host_pull(Path(manifest_path), Path(downloaded_root), Path(ack_path))
    record.update(transfer_seconds=transfer_seconds, verification_seconds=time.monotonic()-started)
    Path(ack_path).write_bytes(_bytes(record))
    return record


def _persistence_seconds(execution: CampaignExecution) -> tuple[float, dict[str, Any]]:
    transfer = execution.state.get("last_host_transfer") or {}
    _require(bool(transfer.get("ack_path")), "A measured independent host acknowledgment is required.")
    ack = _json(execution.root / transfer["ack_path"])
    values = [ack.get(name) for name in ("transfer_seconds", "verification_seconds")]
    _require(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values),
             "Host acknowledgment lacks actual transfer/readback timing.")
    return float(sum(values)), transfer


def _complete_timing(record: Mapping[str, Any], execution: CampaignExecution, output: Path) -> dict[str, Any]:
    destination = output.with_name("g2_complete.json")
    if destination.exists():
        result = _json(destination)
        _require(result.get("timing_sha256") == _sha(output), "Completed timing receipt changed.")
        return result
    seconds, transfer = _persistence_seconds(execution)
    _require(execution.state.get("pending_unit", {}).get("run_name") == record["unit_id"],
             "Acknowledgment belongs to a different timing unit.")
    complete = {**record, "format": "deepmzyme_ec_clean30_completed_timing_v1",
                "timing_sha256": _sha(output), "persistence_seconds": seconds,
                "wall_seconds": float(record["wall_seconds"])+seconds,
                "host_transfer": transfer}
    _write(destination, complete)
    return complete


def lane_gate(serial: Sequence[Mapping[str, Any]], parallel: Sequence[Mapping[str, Any]], *,
              batch_wall_seconds: float, limits: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    """The fixed operational lane diagnostic; it never selects an EC model."""
    import numpy as np

    _require(len(serial) == 2 and len(parallel) == 3, "The lane gate needs exactly two serial and three parallel fits.")
    all_runs = [*serial, *parallel]
    _require([row["recipe"] for row in serial] == ["lane_serial_1", "lane_serial_2"]
             and [row["recipe"] for row in parallel] == [f"lane_parallel_{i}" for i in (1, 2, 3)],
             "Lane recipe identities differ from the fixed plan.")
    ids, truth = list(serial[0]["ids"]), np.asarray(serial[0]["true"], dtype=int)
    _require(ids and len(set(ids)) == len(ids) and len(truth) == len(ids), "Lane query identities are invalid.")
    binding = serial[0].get("execution_manifest_sha256")
    _require(bool(binding) and all(row.get("execution_manifest_sha256") == binding for row in all_runs),
             "Lane source identities differ.")
    for row in all_runs:
        _require(list(row["ids"]) == ids and np.array_equal(row["true"], truth)
                 and len(row["predicted"]) == len(ids), "Lane query/label membership differs.")
        _require(math.isfinite(float(row["wall_seconds"])) and float(row["wall_seconds"]) > 0,
                 "Lane timing must be finite and positive.")
    _require(math.isfinite(batch_wall_seconds) and batch_wall_seconds > 0, "Batch wall time must be positive.")
    supported = [label for label in sorted(set(truth)) if int(np.sum(truth == label)) >= 10]
    comparisons = []
    for lane in parallel:
        for reference in serial:
            a, b = np.asarray(reference["predicted"]), np.asarray(lane["predicted"])
            recall_a = [float(np.mean(a[truth == label] == label)) for label in supported]
            recall_b = [float(np.mean(b[truth == label] == label)) for label in supported]
            delta_ba = abs(float(np.mean(recall_a))-float(np.mean(recall_b)))*100 if supported else None
            delta_recall = max((abs(a-b)*100 for a, b in zip(recall_a, recall_b)), default=None)
            agreement = float(np.mean(a == b))
            comparisons.append({"lane": lane["recipe"], "reference": reference["recipe"],
                                "ba_difference_points": delta_ba, "max_recall_difference_points": delta_recall,
                                "top1_agreement": agreement, "passed": bool(supported) and delta_ba <= 1.0+1e-12
                                and delta_recall <= 3.0+1e-12 and agreement >= .95-1e-12})
    resource_pass = True
    for name in ("cpu_percent", "ram_bytes", "vram_bytes"):
        _require(name in limits and name in observed and math.isfinite(float(limits[name]))
                 and math.isfinite(float(observed[name])) and float(limits[name]) > 0 and float(observed[name]) >= 0,
                 "The lane gate requires measured numeric host limits and peaks.")
        resource_pass &= float(observed[name]) <= float(limits[name])
    resource_pass &= observed.get("swap_bytes") == 0 and observed.get("oom") is False and observed.get("read_guard_breach") is False
    speedup = 3 * sum(float(row["wall_seconds"]) for row in serial) / 2 / batch_wall_seconds
    passed = resource_pass and all(row["passed"] for row in comparisons) and speedup >= 1.2
    return {"format": "deepmzyme_ec_clean30_lane_gate_v1", "status": "passed" if passed else "failed",
            "execution_manifest_sha256": binding, "lanes": 3 if passed else 1, "speedup": speedup,
            "supported_classes": [int(label) for label in supported], "minimum_validation_support": 10,
            "comparisons": comparisons, "resource_passed": resource_pass, "limits": dict(limits),
            "observed": dict(observed), "batch_wall_seconds": batch_wall_seconds,
            "timing_scope": "fit/export, verification, independent replay and active host transfer/readback"}


def _host_sample() -> dict[str, float]:
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.split()[0]) * 1024
    cpu = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  text=True, timeout=5).splitlines()
    _require(len(gpu) == 1, "The lane gate expects one GPU.")
    return {"ram_bytes": memory["MemTotal"]-memory["MemAvailable"],
            "swap_bytes": memory["SwapTotal"]-memory["SwapFree"], "vram_bytes": float(gpu[0])*1024*1024,
            "cpu_total": sum(cpu), "cpu_idle": cpu[3]+cpu[4]}


def lane_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    """One owned process starts precisely the three declared parallel children."""
    _runtime(request)
    root = Path(request["execution_root"])
    batch = root / "runs" / request["batch_name"]
    batch.mkdir(parents=True)
    lock_fd = int(os.environ["DEEPMZYME_EXECUTION_LOCK_FD"])
    os.fstat(lock_fd)
    processes, logs = [], []
    started = time.monotonic()
    previous = _host_sample()
    peaks = {"cpu_percent": 0.0, **{key: previous[key] for key in ("ram_bytes", "swap_bytes", "vram_bytes")},
             "oom": False, "read_guard_breach": False}
    try:
        for index in (1, 2, 3):
            child = {**request, "recipe": f"lane_parallel_{index}"}
            path = batch / f"lane{index}.json"
            _write(path, child)
            log = (batch / f"lane{index}.log").open("x")
            logs.append(log)
            processes.append(subprocess.Popen(_child_command(path, "_timing-worker", request["python"]),
                                               cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               pass_fds=(lock_fd,)))
        while any(process.poll() is None for process in processes):
            time.sleep(.5)
            current = _host_sample()
            total = current["cpu_total"]-previous["cpu_total"]
            if total > 0:
                busy = total-(current["cpu_idle"]-previous["cpu_idle"])
                peaks["cpu_percent"] = max(peaks["cpu_percent"], 100*(os.cpu_count() or 1)*busy/total)
            for name in ("ram_bytes", "swap_bytes", "vram_bytes"):
                peaks[name] = max(peaks[name], current[name])
            previous = current
        codes = [process.wait() for process in processes]
        records = []
        for index in (1, 2, 3):
            path = root / "runs" / _unit_name({**request, "recipe": f"lane_parallel_{index}"}) / "g2_timing.json"
            if path.is_file():
                records.append(_json(path))
        record = {"status": "passed" if all(code == 0 for code in codes) and len(records) == 3 else "failed",
                  "records": records, "return_codes": codes, "observed": peaks,
                  "batch_wall_seconds": time.monotonic()-started}
        _write(batch / "batch_timing.json", record)
        return record
    finally:
        active = [process for process in processes if process.poll() is None]
        for process in active:
            process.terminate()
        for process in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for log in logs:
            log.close()


def run_lane_gate(request: Mapping[str, Any]) -> dict[str, Any]:
    source, _view, policy = _runtime(request)
    serial_paths = [Path(path) for path in request["serial_receipts"]]
    serial = [_json(path) for path in serial_paths]
    _require(len(serial) == 2 and all(row.get("format") == "deepmzyme_ec_clean30_completed_timing_v1"
             and row.get("status") == "passed" and row.get("execution_manifest_sha256") == source["execution_manifest_sha256"]
             for row in serial), "Lane gate requires two verified and persisted serial repeats.")
    _require(float(request["forecast_seconds"]) >= 3*max(row["wall_seconds"] for row in serial),
             "Batch admission may not assume the speedup under test.")
    root = Path(request["execution_root"])
    attempt = request.get("attempt", 1)
    _require(type(attempt) is int and attempt > 0, "Invalid batch attempt.")
    name = f"ecc30timing_lane_batch_attempt{attempt}"
    batch = root / "runs" / name
    measured = batch / "batch_timing.json"
    with _execution(request, policy) as execution:
        if measured.exists():
            result = _json(measured)
            persistence_seconds, transfer = _persistence_seconds(execution)
            _require(execution.state.get("pending_unit", {}).get("run_name") == name
                     and execution.state["pending_unit"].get("persisted"), "Batch acknowledgment is missing.")
            gate_path = batch / "lane_gate.json"
            if gate_path.exists():
                return _json(gate_path)
            if result.get("status") == "passed":
                # Both serial units and the parallel batch include exactly their own transfer/readback work.
                gate = lane_gate(serial, result["records"], batch_wall_seconds=result["batch_wall_seconds"]+persistence_seconds,
                                 limits=request["limits"], observed=result["observed"])
            else:
                gate = {"format": "deepmzyme_ec_clean30_lane_gate_v1", "status": "failed", "lanes": 1,
                        "reason": "parallel_batch_failed", "execution_manifest_sha256": source["execution_manifest_sha256"]}
            gate.update(host_transfer=transfer, serial_receipts=[{"path": str(path), "sha256": _sha(path)} for path in serial_paths],
                        batch_timing_sha256=_sha(measured), persistence_seconds=persistence_seconds)
            _write(gate_path, gate)
            return gate
        local_request = {**request, "batch_name": name}
        request_file = root / "commands" / f"{name}.json"
        _write(request_file, local_request)
        execution.admit(name, float(request["forecast_seconds"]))
        env = dict(os.environ)
        env["DEEPMZYME_EXECUTION_LOCK_FD"] = str(execution._lock.fileno())
        started = time.monotonic()
        with (root / "commands" / f"{name}.log").open("x") as log:
            result = execution.run_subprocess(_child_command(request_file, "_lane-worker", request["python"]),
                cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=float(request["unit_timeout_seconds"]))
        passed = result.returncode == 0 and measured.is_file() and _json(measured).get("status") == "passed"
        if not measured.exists():
            _write(measured, {"status": "failed", "reason": "batch_worker_failed", "records": []})
        execution.record_result(name, "completed" if passed else "failed_lane", time.monotonic()-started)
        artifacts = [batch, request_file, root / "commands" / f"{name}.log"]
        for index in (1, 2, 3):
            unit = _unit_name({**request, "recipe": f"lane_parallel_{index}"})
            artifacts += [path for path in (root / "runs" / unit, root / "runs" / f"{unit}.log",
                          commands.runner_receipt_path(root / "runs", unit)) if path.exists()]
        persistence = execution.persist(artifacts)
        return {"status": "awaiting_host_ack", "batch_passed": passed, "persistence": persistence}


def build_freeze1c(evidence: Sequence[Path], out: Path, *, execution_manifest_path: Path,
                   l3_vm_hours_per_fold: Mapping[str, Any]) -> dict[str, Any]:
    source = verify_execution_source(execution_manifest_path)
    _require(set(l3_vm_hours_per_fold) == {str(index) for index in range(5)}
             and all(not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
                     for value in l3_vm_hours_per_fold.values()), "Freeze 1c needs five positive numerical L3 ceilings.")
    records, bindings = [], []
    for path in evidence:
        path = Path(path)
        _no_restricted_path(path)
        row = _json(path)
        _require(row.get("execution_manifest_sha256") == source["execution_manifest_sha256"],
                 "Measurement belongs to another execution source.")
        transfer = row.get("host_transfer") or {}
        _require(all(re.fullmatch(r"[0-9a-f]{64}", str(transfer.get(key, "")))
                     for key in ("manifest_sha256", "ack_sha256")), "Measurement lacks verified host persistence.")
        records.append(row)
        bindings.append({"path": str(path), "sha256": _sha(path)})
    timings = [row for row in records if row.get("format") == "deepmzyme_ec_clean30_completed_timing_v1"]
    _require({row.get("recipe") for row in timings} == {*FAMILIES, "F4_L3"}
             and len(timings) == 7 and all(row.get("status") == "passed" for row in timings),
             "Freeze 1c needs all six family timings and the L3 timing.")
    embedding = [row for row in records if row.get("format") == "deepmzyme_ec_clean30_completed_embedding_v1"]
    _require(len(embedding) == 2 and {row.get("kind") for row in embedding} == {"esmc", "esm1b"}
             and all(row.get("status") == "passed" for row in embedding), "Both embedding timings are required.")
    gates = [row for row in records if row.get("format") == "deepmzyme_ec_clean30_lane_gate_v1"]
    _require(len(records) == 10 and len(gates) == 1 and gates[0].get("status") in {"passed", "failed"}
             and gates[0].get("lanes") == (3 if gates[0]["status"] == "passed" else 1),
             "Freeze 1c needs one complete lane decision.")
    document = {"format": "deepmzyme_ec_clean30_freeze1c_v1", "campaign_id": CAMPAIGN_ID,
                "freeze1_sha256": source["freeze1_sha256"], "freeze1b_sha256": source["freeze1b_sha256"],
                "execution_manifest_sha256": source["execution_manifest_sha256"],
                "l3_vm_hours_per_fold": dict(l3_vm_hours_per_fold), "lanes": gates[0]["lanes"],
                "evidence": bindings, "budget_authority": "separate_G3_approval_required"}
    _write(out, document)
    return document


def _esmc_cached_weight(cache: Path) -> Path:
    """Resolve the same default snapshot/file used by the installed ESMC-300m SDK, offline."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12",
                                 cache_dir=str(cache / "hub"), local_files_only=True)
    return (Path(snapshot) / "data/weights/esmc_300m_2024_12_v0.pth").resolve()


def embedding_command(request: Mapping[str, Any], source: Mapping[str, Any], view: views.View,
                      output: Path) -> tuple[list[str], dict[str, str]]:
    """Only the two existing embedding producers, on the pinned timing subset."""
    inputs = Path(request["timing_inputs"])
    roots = view.forbidden_read_roots
    _outside_guard(inputs, roots)
    _outside_guard(output, roots)
    receipt_path = inputs / "timing_inputs.json"
    _require(source.get("timing_inputs_sha256") == _sha(receipt_path), "Timing subset is not bound by the execution manifest.")
    receipt = _json(receipt_path)
    _require(receipt.get("n_rows") == TIMING_ROWS and receipt.get("fold") == 0
             and receipt.get("source_sha256") == source.get("official_fold0_training_sha256") == OFFICIAL_FOLD0_SHA256,
             "Embedding timing is not the exact official fold-0 subset.")
    for name in ("timing.fasta", "sequence_map.tsv"):
        _require(receipt.get("files", {}).get(name) == _sha(_safe_file(inputs, name)), "Timing input changed.")
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT))),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    if request["kind"] == "esmc":
        from training.access_guard import FORBIDDEN_READ_ROOTS_ENV
        cache, weights = Path(request["hf_home"]).resolve(), Path(request["esmc_weights"]).resolve()
        _outside_guard(Path(request["hf_home"]), roots)
        _outside_guard(Path(request["esmc_weights"]), roots)
        _require(cache in weights.parents and weights.is_file()
                 and _esmc_cached_weight(cache) == weights
                 and re.fullmatch(r"[0-9a-f]{64}", str(source.get("frozen_esmc_weight_sha256", "")))
                 and _sha(weights) == source["frozen_esmc_weight_sha256"],
                 "ESMC weights must match Freeze 1 inside the explicit offline cache.")
        env.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"), HUGGINGFACE_HUB_CACHE=str(cache / "hub"))
        env.pop("INFRA_PROVIDER", None)  # The SDK otherwise bypasses its HF cache and loads relative data/weights.
        env[FORBIDDEN_READ_ROOTS_ENV] = json.dumps(roots)
        wrapper = ("import runpy,sys; from training.access_guard import install_guard_from_environment; "
                   "install_guard_from_environment(); sys.argv=sys.argv[1:]; "
                   "runpy.run_module('embed_helpers.esmc_chain_means',run_name='__main__')")
        argv = [request["python"], "-c", wrapper, "esmc_chain_means", "sequences", "--fasta", str(inputs / "timing.fasta"),
                "--sequence-map", str(inputs / "sequence_map.tsv"), "--sequence-map-sha256", receipt["files"]["sequence_map.tsv"],
                "--out", str(output), "--layer-index", "26", "--device", request["device"]]
    elif request["kind"] == "esm1b":
        pins = {"harness": "73c25c5396f89a603a2ca97cf2adf82bb6b220bccf4d9f45801d1bdcf60bede4",
                "extract_script": "902a69cffa0b35ab16798ba655d7b0e357d0e8efd5a268558cc4b064d2e7719d"}
        for name, expected in pins.items():
            path = Path(request[name]); _outside_guard(path, roots)
            _require(path.is_file() and _sha(path) == expected, "Private embedding helper differs from its existing pin.")
        weights = Path(request["weights"])
        _outside_guard(weights, roots)
        contact = Path(str(weights.with_suffix("")) + "-contact-regression.pt")
        _outside_guard(contact, roots)
        _require(weights.is_file() and contact.is_file(), "Local ESM-1b weights and contact-regression companion are required.")
        for path, key in ((weights, "weights_sha256"), (contact, "contact_weights_sha256")):
            _require(re.fullmatch(r"[0-9a-f]{64}", str(request.get(key, ""))) and _sha(path) == request[key],
                     "Private local weight checksum differs.")
        argv = [request["clean_python"], str(request["harness"]), "--runner",
                str(REPO_ROOT / "scripts/run_clean_split30_comparator.py"), "--extract-script", str(request["extract_script"]),
                "--extract-sha256", pins["extract_script"], "--weights", str(weights), "--weights-sha256", request["weights_sha256"],
                "--sequence-map", str(inputs / "sequence_map.tsv"), "--sequence-map-sha256", receipt["files"]["sequence_map.tsv"],
                "--kind", "base", "--out-dir", str(output), "--device", request["device"], "--threads", str(request["threads"])]
        for root in roots:
            argv += ["--forbidden-root", root]
    else:
        raise Phase3Error("Embedding kind must be esmc or esm1b.")
    return argv, env


def _esm1b_cuda_preflight(request: Mapping[str, Any], env: Mapping[str, str], directory: Path) -> dict[str, str]:
    """The private extractor can silently use CPU, so prove this interpreter can use the L4."""
    program = (
        "import json,torch; "
        "assert torch.version.cuda and torch.cuda.is_available(), 'ESM-1b interpreter has no usable CUDA'; "
        "name=torch.cuda.get_device_name(0); assert name=='NVIDIA L4', 'G2 requires the approved L4'; "
        "value=(torch.ones(4,device='cuda')*2).sum().item(); torch.cuda.synchronize(); "
        "assert value==8.0, 'CUDA kernel check failed'; "
        "print(json.dumps({'device_name':name,'torch':torch.__version__,'cuda':torch.version.cuda,'kernel_value':value}))"
    )
    path = directory / "cuda_preflight.json"
    with path.open("x") as handle:
        subprocess.run([*request["cap_prefix"], request["clean_python"], "-c", program],
                       cwd=REPO_ROOT, env=dict(env), stdout=handle, stderr=subprocess.STDOUT,
                       check=True, timeout=min(60.0, float(request["unit_timeout_seconds"])))
    evidence = _json(path)
    _require(evidence.get("device_name") == "NVIDIA L4" and evidence.get("cuda")
             and evidence.get("kernel_value") == 8.0, "ESM-1b CUDA preflight did not pass.")
    return {"cuda_preflight_sha256": _sha(path)}


def _esm1b_gpu_evidence(output: Path) -> dict[str, str]:
    extraction = _json(output / "farm_receipt.json").get("extraction", {})
    path = output / "extract.log"
    _require(extraction.get("device") == "cuda" and extraction.get("returncode") == 0
             and extraction.get("log") == "extract.log" and path.is_file()
             and extraction.get("log_sha256") == _sha(path)
             and "Transferred model to GPU" in path.read_text().splitlines(),
             "ESM-1b extraction lacks hash-bound evidence that its model actually used CUDA.")
    return {"extraction_log_sha256": _sha(path)}


def embedding_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    source, view, _ = _runtime(request)
    name = f"ecc30timing_embed_{request['kind']}_attempt{request.get('attempt', 1)}"
    directory = Path(request["execution_root"]) / "runs" / name
    directory.mkdir(parents=True)
    started = time.monotonic()
    argv, env = embedding_command(request, source, view, directory / "embeddings")
    device_evidence = {}
    if request["kind"] == "esm1b" and request["device"] == "cuda":
        device_evidence.update(_esm1b_cuda_preflight(request, env, directory))
    with (directory / "embedding.log").open("x") as log:
        subprocess.run([*request["cap_prefix"], *argv], cwd=REPO_ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = directory / "embeddings" / ("build_receipt.json" if request["kind"] == "esmc" else "farm_receipt.json")
    _require(receipt.is_file(), "Embedding producer did not leave its completed receipt.")
    if request["kind"] == "esm1b" and request["device"] == "cuda":
        device_evidence.update(_esm1b_gpu_evidence(directory / "embeddings"))
    result = {"format": "deepmzyme_ec_clean30_embedding_timing_v1", "status": "passed", "kind": request["kind"],
              "unit_id": name, "source": source, "execution_manifest_sha256": source["execution_manifest_sha256"],
              "producer_receipt_sha256": _sha(receipt), "wall_seconds": time.monotonic()-started,
              "n_input_sequences": TIMING_ROWS, "device_evidence": device_evidence}
    _write(directory / "embedding_timing.json", result)
    return result


def embed_timing(request: Mapping[str, Any]) -> dict[str, Any]:
    source, _view, policy = _runtime(request)
    _require(request.get("kind") in {"esmc", "esm1b"} and type(request.get("attempt", 1)) is int
             and request.get("attempt", 1) > 0, "Unknown embedding timing or attempt.")
    name = f"ecc30timing_embed_{request['kind']}_attempt{request.get('attempt', 1)}"
    root = Path(request["execution_root"])
    directory = root / "runs" / name
    timing = directory / "embedding_timing.json"
    with _execution(request, policy) as execution:
        if timing.exists():
            row = _json(timing)
            _require(row.get("status") == "passed" and row.get("source") == source, "Embedding source or status differs.")
            persistence_seconds, transfer = _persistence_seconds(execution)
            _require(execution.state.get("pending_unit", {}).get("run_name") == name
                     and execution.state["pending_unit"].get("persisted"), "Embedding persistence is not acknowledged.")
            complete = directory / "embedding_complete.json"
            if not complete.exists():
                _write(complete, {**row, "format": "deepmzyme_ec_clean30_completed_embedding_v1",
                                 "persistence_seconds": persistence_seconds, "host_transfer": transfer,
                                 "wall_seconds": float(row["wall_seconds"])+persistence_seconds,
                                 "timing_sha256": _sha(timing)})
            return _json(complete)
        request_file = root / "commands" / f"{name}.json"
        _write(request_file, dict(request))
        execution.admit(name, float(request["forecast_seconds"]))
        started = time.monotonic()
        with (root / "commands" / f"{name}.log").open("x") as log:
            result = execution.run_subprocess(_child_command(request_file, "_embedding-worker", request["python"]),
                cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=float(request["unit_timeout_seconds"]))
        passed = result.returncode == 0 and timing.is_file() and _json(timing).get("status") == "passed"
        execution.record_result(name, "completed" if passed else "failed_embedding", time.monotonic()-started)
        paths = [request_file, root / "commands" / f"{name}.log"] + ([directory] if directory.exists() else [])
        persistence = execution.persist(paths)
        _require(passed, "Embedding timing failed; acknowledge its saved attempt before retrying.")
        return {"status": "awaiting_host_ack", "kind": request["kind"], "persistence": persistence}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    pack = actions.add_parser("pack-view")
    pack.add_argument("--view", type=Path, required=True)
    pack.add_argument("--archive", type=Path, required=True)
    pack.add_argument("--smoke", action="store_true")
    unpack = actions.add_parser("unpack-view")
    unpack.add_argument("--archive", type=Path, required=True)
    unpack.add_argument("--destination", type=Path, required=True)
    unpack.add_argument("--sha256", required=True)
    unpack.add_argument("--smoke", action="store_true")
    inputs = actions.add_parser("timing-inputs")
    inputs.add_argument("--training-file", type=Path, required=True)
    inputs.add_argument("--sha256", required=True)
    inputs.add_argument("--out", type=Path, required=True)
    source = actions.add_parser("execution-manifest")
    source.add_argument("--amendment", type=Path, required=True)
    source.add_argument("--amendment-commit", required=True)
    source.add_argument("--freeze1b-sha256", required=True)
    source.add_argument("--timing-inputs-sha256", required=True)
    source.add_argument("--out", type=Path, required=True)
    automation = actions.add_parser("automation-manifest")
    automation.add_argument("--base-execution", type=Path, required=True)
    automation.add_argument("--base-commit", required=True)
    automation.add_argument("--out", type=Path, required=True)
    verify = actions.add_parser("verify-execution")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--draft", action="store_true")
    probe = actions.add_parser("vm-probe")
    probe.add_argument("--view", type=Path, required=True)
    probe.add_argument("--execution-manifest", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--cap-json", type=Path, required=True, help="JSON array of the approved runtime cap argv.")
    probe.add_argument("--python", default=sys.executable)
    probe.add_argument("--smoke", action="store_true")
    for name in ("timing-unit", "lane-gate", "embed-timing", "_timing-worker", "_lane-worker", "_embedding-worker"):
        action = actions.add_parser(name)
        action.add_argument("--request", type=Path, required=True)
    ack = actions.add_parser("ack-host")
    ack.add_argument("--manifest", type=Path, required=True)
    ack.add_argument("--downloaded-root", type=Path, required=True)
    ack.add_argument("--ack", type=Path, required=True)
    ack.add_argument("--transfer-seconds", type=float, required=True)
    final = actions.add_parser("freeze1c")
    final.add_argument("--execution-manifest", type=Path, required=True)
    final.add_argument("--evidence", type=Path, action="append", required=True)
    final.add_argument("--l3-ceilings", type=Path, required=True, help="JSON object with fold keys 0..4 and proposed hours.")
    final.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "pack-view":
            result = pack_view(args.view, args.archive, smoke=args.smoke)
        elif args.action == "unpack-view":
            result = unpack_view(args.archive, args.destination, expected_sha256=args.sha256, smoke=args.smoke)
        elif args.action == "timing-inputs":
            result = timing_inputs(args.training_file, args.out, expected_sha256=args.sha256)
        elif args.action == "execution-manifest":
            result = make_execution_manifest(args.amendment, args.out, amendment_commit=args.amendment_commit,
                freeze1b_sha256=args.freeze1b_sha256, timing_inputs_sha256=args.timing_inputs_sha256)
        elif args.action == "automation-manifest":
            result = make_automation_manifest(args.base_execution, args.out, base_commit=args.base_commit)
        elif args.action == "verify-execution":
            result = verify_execution_source(args.manifest, require_published=not args.draft)
        elif args.action == "vm-probe":
            result = vm_probe(args.view, args.execution_manifest, args.out, cap_prefix=_json(args.cap_json),
                              python=args.python, smoke=args.smoke)
        elif args.action == "ack-host":
            result = host_ack(args.manifest, args.downloaded_root, args.ack, transfer_seconds=args.transfer_seconds)
        elif args.action == "freeze1c":
            result = build_freeze1c(args.evidence, args.out, execution_manifest_path=args.execution_manifest,
                                    l3_vm_hours_per_fold=_json(args.l3_ceilings))
        else:
            _no_restricted_path(args.request)
            action = {"timing-unit": run_timing_unit, "lane-gate": run_lane_gate, "embed-timing": embed_timing,
                      "_timing-worker": timing_worker, "_lane-worker": lane_worker, "_embedding-worker": embedding_worker}[args.action]
            result = action(_json(args.request))
        public = {key: result[key] for key in ("status", "format", "kind", "unit_id", "recipe", "archive_sha256",
                  "archive_bytes", "unpacked_bytes", "files", "n_rows", "published", "execution_manifest_sha256",
                  "lanes", "speedup") if key in result and (key != "files" or isinstance(result[key], int))}
        print(json.dumps(public, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"Phase 3 refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
