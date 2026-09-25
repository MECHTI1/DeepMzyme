"""Build the transfer bundle for the frozen PMM ion campaign on a remote GPU host.

Three archives plus a manifest, written to ``--out-dir``:

- ``train_side.tar.gz``: the dataset's ``train/`` directory only (manifests and
  structures, hard links preserved, so the 6,443 chain-named files store 4,191
  contents once). The public dataset tarball also contains the held-out test side
  and must not be used for this development campaign.
- ``campaign_frozen.tar.gz``: the frozen campaign files (cohort, dispositions, audit,
  manifest, folds, weights, ESM plan, comparator outputs). Runs, caches, generated
  embeddings, smoke campaigns and environments are excluded.
- ``code_snapshot.tar.gz``: allowlisted working-tree source, scripts, requirements
  and the pinned training source only; uncommitted campaign code travels exactly.
- ``bundle_manifest.json``: SHA-256 of every archive, member counts, the code-tree
  hash that run identities bind, and the test-exclusion check.

Every archive member is checked: a path component ``test`` or a crosswalk file name
aborts the build.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pmm_source_release import sha256_file  # noqa: E402

DATASET_NAME = "train_and_test_sets_structures_zenodo_pmm_exact"
CAMPAIGN_EXCLUDED_TOP_LEVEL = {"runs", "parse_cache", "smoke", "commands", "analysis", "_incomplete_attempts"}
CAMPAIGN_EXCLUDED_INPUT_PREFIXES = ("esm_embeddings",)
FORBIDDEN_NAMES = {"site_crosswalk.csv", "classmodel_test_set", "classmodel_test_set.csv"}
TRAIN_METADATA = {"site_manifest.csv", "structure_manifest.csv", "structure_chain_manifest.csv",
                  "final_data_summarazing_table.csv"}
CAMPAIGN_FILES = {"campaign_manifest.json", "train_cohort.csv", "train_row_dispositions.csv", "train_audit.json",
                  "train_context_audit.json", "fold_membership.csv", "fold_class_weights.json",
                  "feature_inventory.json", "esm_generation_plan.csv", "esm_generation_plan.json"}
TRAIN_SOURCE = "prepare_training_and_test_set/pinmymetal_files/classmodel_train_set"


class BundleError(RuntimeError):
    pass


def _check_member(arcname: str) -> None:
    parts = Path(arcname).parts
    if Path(arcname).is_absolute() or ".." in parts or "test" in parts or any(part in FORBIDDEN_NAMES for part in parts):
        raise BundleError(f"Refusing to bundle held-out or mixed-side path: {arcname}")


def _code_member_allowed(name: str) -> bool:
    path = Path(name)
    return (name == TRAIN_SOURCE or name in {"pyproject.toml", "requirements.txt"}
            or (path.parts[0] == "src" and path.suffix == ".py")
            or (path.parts[0] == "scripts" and len(path.parts) == 2 and path.suffix in {".py", ".sh"})
            or (path.parts[0] == "requirements" and path.suffix == ".txt"))


def _member_allowed(name: str, kind: str, root: str) -> bool:
    parts = Path(name).parts
    root_parts = Path(root).parts
    if parts[:len(root_parts)] != root_parts:
        return False
    relative = Path(*parts[len(root_parts):])
    if str(relative) == ".":
        return True
    if kind == "code":
        return _code_member_allowed(relative.as_posix())
    if kind == "train_side":
        return (relative.as_posix() in TRAIN_METADATA or relative.as_posix() == "structures"
                or (len(relative.parts) == 2 and relative.parts[0] == "structures"
                    and relative.suffix.lower() in {".pdb", ".cif", ".mmcif"}))
    return (relative.as_posix() in CAMPAIGN_FILES
            or (len(relative.parts) == 2 and relative.parts[0] == "pmm_comparator"
                and (relative.name == "pmm_comparator_manifest.json"
                     or relative.name in {f"fold{k}_predictions.csv" for k in range(5)})))


def verify_archive(path: Path, *, kind: str, root: str) -> int:
    """Inspect every archive header and link target without extracting member contents."""
    seen: set[str] = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            _check_member(member.name)
            if member.name in seen or not _member_allowed(member.name, kind, root):
                raise BundleError(f"Unexpected or repeated {kind} member: {member.name}")
            if member.issym() or not (member.isfile() or member.isdir() or member.islnk()):
                raise BundleError(f"Unsupported archive member type: {member.name}")
            if member.islnk():
                target = posixpath.normpath(member.linkname)
                _check_member(member.linkname)
                if target not in seen or not _member_allowed(target, kind, root):
                    raise BundleError(f"Unsafe hard-link target: {member.name} -> {member.linkname}")
            seen.add(member.name)
    return len(seen)


def _add(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    _check_member(arcname)
    if path.is_symlink():
        raise BundleError(f"Symbolic links are not permitted in a development bundle: {path}")
    tar.add(str(path), arcname=arcname, recursive=False)


def build_train_side(train_dir: Path, out: Path) -> dict:
    train_dir = Path(train_dir).resolve()
    if train_dir.name != "train":
        raise BundleError(f"Expected the dataset's train/ directory, got {train_dir}")
    prefix = f"{DATASET_NAME}/train"
    members = 0
    with tarfile.open(out, "w:gz", compresslevel=1, dereference=False) as tar:
        _add(tar, train_dir, prefix)
        for path in sorted(train_dir.iterdir()):
            if path.name not in TRAIN_METADATA:
                continue
            _add(tar, path, f"{prefix}/{path.name}")
            members += 1
        structures = train_dir / "structures"
        _add(tar, structures, f"{prefix}/structures")
        for path in sorted(structures.iterdir()):
            if not _member_allowed(f"{prefix}/structures/{path.name}", "train_side", prefix):
                raise BundleError(f"Unexpected training structure path: {path}")
            _add(tar, path, f"{prefix}/structures/{path.name}")  # later hard links become LNKTYPE entries
            members += 1
    members = verify_archive(out, kind="train_side", root=prefix)
    return {"path": out.name, "sha256": sha256_file(out), "members": members, "bytes": out.stat().st_size,
            "root": prefix}


def build_campaign(campaign_dir: Path, out: Path) -> dict:
    campaign_dir = Path(campaign_dir).resolve()
    members = 0
    with tarfile.open(out, "w:gz", compresslevel=6) as tar:
        for path in sorted(campaign_dir.rglob("*")):
            relative = path.relative_to(campaign_dir)
            if relative.parts[0] in CAMPAIGN_EXCLUDED_TOP_LEVEL:
                continue
            if relative.parts[0] == "inputs" and len(relative.parts) > 1 and \
                    relative.parts[1].startswith(CAMPAIGN_EXCLUDED_INPUT_PREFIXES):
                continue
            if path.is_dir() or not _member_allowed(f"{campaign_dir.name}/{relative.as_posix()}", "campaign", campaign_dir.name):
                continue
            _add(tar, path, f"{campaign_dir.name}/{relative.as_posix()}")
            members += 1
    members = verify_archive(out, kind="campaign", root=campaign_dir.name)
    return {"path": out.name, "sha256": sha256_file(out), "members": members, "bytes": out.stat().st_size,
            "root": campaign_dir.name}


def build_code_snapshot(out: Path) -> dict:
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=REPO_ROOT, check=True, capture_output=True).stdout.decode().split("\0")
    files = sorted(name for name in listed if name and _code_member_allowed(name) and (REPO_ROOT / name).is_file())
    with tarfile.open(out, "w:gz", compresslevel=6) as tar:
        for name in files:
            _add(tar, REPO_ROOT / name, f"DeepMzyme/{name}")
    from benchmarking.pmm_ion_campaign import source_tree_sha256

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                            text=True).stdout.strip()
    members = verify_archive(out, kind="code", root="DeepMzyme")
    return {"path": out.name, "sha256": sha256_file(out), "members": members, "root": "DeepMzyme", "bytes": out.stat().st_size,
            "git_head": head, "git_branch": branch, "source_tree_sha256": source_tree_sha256()}


def build_bundle(train_dir: Path, campaign_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    campaign_id = json.loads((campaign_dir / "campaign_manifest.json").read_text(encoding="utf-8"))["campaign_id"]
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "campaign_id": campaign_id,
        "purpose": f"{campaign_id} development-only transfer (no held-out test files)",
        "train_side": build_train_side(train_dir, out_dir / "train_side.tar.gz"),
        "campaign": build_campaign(campaign_dir, out_dir / "campaign_frozen.tar.gz"),
        "code": build_code_snapshot(out_dir / "code_snapshot.tar.gz"),
        "test_exclusion_checked": True,
    }
    manifest["build_seconds"] = round(time.time() - started, 1)
    (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_bundle(out_dir: Path) -> dict:
    manifest = json.loads((out_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    for key in ("train_side", "campaign", "code"):
        entry = manifest[key]
        expected_name = {"train_side": "train_side.tar.gz", "campaign": "campaign_frozen.tar.gz",
                         "code": "code_snapshot.tar.gz"}[key]
        if entry["path"] != expected_name:
            raise BundleError(f"Unexpected bundle archive path: {entry['path']}")
        if sha256_file(out_dir / entry["path"]) != entry["sha256"]:
            raise BundleError(f"{entry['path']} hash differs from the bundle manifest")
        count = verify_archive(out_dir / entry["path"], kind=key, root=entry.get("root", "DeepMzyme"))
        if count != entry["members"]:
            raise BundleError(f"{entry['path']} member count differs from the bundle manifest")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = verify_bundle(args.out_dir) if args.verify_only else build_bundle(args.train_dir, args.campaign_dir, args.out_dir)
    print(json.dumps({key: manifest[key] for key in ("train_side", "campaign", "code")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
