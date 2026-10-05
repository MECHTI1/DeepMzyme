#!/usr/bin/env python3
"""Transfer bundle for the v3 campaign on the GPU host (step B onwards).

build (workstation): two archives and a manifest from the COMMITTED tree only:
  v3_code.tar.gz   allowlisted source: src/**/*.py, scripts/*.py|*.sh, requirements/*.txt, the v3
                   runtime files at the repository root, and the frozen A4 specification;
  v3_folds.tar.gz  the frozen fold set (fold file, receipt and reports);
  v3_bundle_manifest.json  SHA-256 of both archives, member lists, git commit, the source-tree hash
                   (pmm_ion_campaign rule), the runner hash and the specification hash.
Any member whose path has a component named "test", a crosswalk file or the PMM classmodel test set
aborts the build: held-out data never travels.

apply (GPU host; standard library only): verifies the archive hashes, extracts the code into a NEW
directory and the folds into another, and re-verifies the source-tree, runner, specification and fold
hashes before anything can run. It never starts compute.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = ("pmm_v3_campaign.py", "run_pmm_v3_campaign.py", "pmm_v3_assessment.py", "pmm_v3_speed_report.py",
                 "pmm_v3_bundle.py")
SPEC_FILES = ("docs/campaigns/pmm_ion_metal_v3/assessment_spec.json", "docs/campaigns/pmm_ion_metal_v3/assessment_spec.md")
FOLD_FILES = ("fold_membership.csv", "fold_receipt.json", "pdb_groups.csv", "fold_balance_report.json",
              "near_copy_pairs.csv")
FORBIDDEN_PARTS = {"test", "site_crosswalk.csv", "classmodel_test_set"}
DEFAULT_FOLDS = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/folds/v3-seqid90-s42-b2")
CODE_ROOT = "DeepMzyme_v3"


class BundleError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def allowed(name: str) -> bool:
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe path {name}")
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        return False
    return (name in RUNTIME_FILES or name in SPEC_FILES
            or (path.parts[0] == "src" and path.suffix == ".py")
            or (path.parts[0] == "scripts" and len(path.parts) == 2 and path.suffix in {".py", ".sh"})
            or (path.parts[0] == "requirements" and path.suffix == ".txt"))


def tree_sha256(files: dict[str, bytes]) -> str:
    """The pmm_ion_campaign.source_tree_sha256 rule over relative path -> content."""
    digest = hashlib.sha256()
    for name in sorted(n for n in files if (n.startswith("src/") and n.endswith(".py") and "__pycache__" not in n)
                       or (n.startswith("scripts/") and n.count("/") == 1 and n.endswith(".py"))):
        digest.update(name.encode() + b"\0" + files[name])
    return digest.hexdigest()


def runner_sha256(files: dict[str, bytes]) -> dict[str, str]:
    return {name: sha256_bytes(files[name]) for name in ("pmm_v3_campaign.py", "run_pmm_v3_campaign.py")}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def build(out_dir: Path, *, folds: Path = DEFAULT_FOLDS) -> dict:
    tracked = [n for n in _git("ls-files", "-z").split("\0") if n]
    names = sorted(n for n in tracked if allowed(n))
    for required in (*RUNTIME_FILES, *SPEC_FILES):
        require(required in names, f"{required} is not committed")
    dirty = _git("status", "--porcelain", "--", *names).strip()
    require(not dirty, f"uncommitted changes in bundled files:\n{dirty}")
    head = _git("rev-parse", "HEAD").strip()
    files = {n: subprocess.run(["git", "show", f"{head}:{n}"], cwd=ROOT, check=True, capture_output=True).stdout
             for n in names}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    code = out_dir / "v3_code.tar.gz"
    with tarfile.open(code, "w:gz", compresslevel=6) as tar:
        for name in names:
            info = tarfile.TarInfo(f"{CODE_ROOT}/{name}")
            info.size, info.mode, info.mtime = len(files[name]), 0o644, 0
            tar.addfile(info, io.BytesIO(files[name]))
    fold_files = {}
    for name in FOLD_FILES:
        path = Path(folds) / name
        require(path.is_file(), f"fold set file {path} is missing")
        fold_files[name] = path.read_bytes()
    receipt = json.loads(fold_files["fold_receipt.json"])
    require(receipt.get("accepted") is True and not (Path(folds) / "SUPERSEDED.md").exists(), "fold set not accepted")
    require(receipt["outputs_sha256"]["fold_membership.csv"] == sha256_bytes(fold_files["fold_membership.csv"]),
            "fold file differs from its receipt")
    fold_archive = out_dir / "v3_folds.tar.gz"
    with tarfile.open(fold_archive, "w:gz", compresslevel=6) as tar:
        for name, data in fold_files.items():
            info = tarfile.TarInfo(f"{Path(folds).name}/{name}")
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    import pmm_v3_assessment as assessment
    from benchmarking.pmm_ion_campaign import source_tree_sha256

    spec_sha = sha256_bytes(files[SPEC_FILES[0]])
    require(spec_sha == assessment.FROZEN_SPEC_SHA256, "bundled specification is not the frozen one")
    src_sha = tree_sha256(files)
    require(src_sha == source_tree_sha256(), "bundled source differs from the working tree")
    manifest = {"schema_version": 1, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": head,
                "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(), "code_root": CODE_ROOT,
                "code": {"path": code.name, "sha256": sha256_file(code), "members": names},
                "folds": {"path": fold_archive.name, "sha256": sha256_file(fold_archive), "fold_set": Path(folds).name,
                          "fold_membership_sha256": sha256_bytes(fold_files["fold_membership.csv"])},
                "source_tree_sha256": src_sha, "runner_sha256": runner_sha256(files), "spec_sha256": spec_sha,
                "held_out_check": f"no member path has a component in {sorted(FORBIDDEN_PARTS)}"}
    (out_dir / "v3_bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _members(tar: tarfile.TarFile, root: str) -> dict[str, bytes]:
    out = {}
    for member in tar.getmembers():
        require(member.isfile(), f"unexpected archive member type: {member.name}")
        path = PurePosixPath(member.name)
        require(path.parts[0] == root and ".." not in path.parts and not path.is_absolute(), f"unsafe member {member.name}")
        relative = str(PurePosixPath(*path.parts[1:]))
        require(not any(part in FORBIDDEN_PARTS for part in PurePosixPath(relative).parts), f"held-out member {relative}")
        out[relative] = tar.extractfile(member).read()
    return out


def apply(bundle_dir: Path, *, code_dest: Path, folds_parent: Path) -> dict:
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "v3_bundle_manifest.json").read_text())
    for key in ("code", "folds"):
        require(sha256_file(bundle_dir / manifest[key]["path"]) == manifest[key]["sha256"], f"{key} archive changed")
    code_dest, folds_dest = Path(code_dest), Path(folds_parent) / manifest["folds"]["fold_set"]
    require(not code_dest.exists(), f"{code_dest} exists; apply only into a new directory")
    require(not folds_dest.exists(), f"{folds_dest} exists; apply only into a new directory")
    with tarfile.open(bundle_dir / manifest["code"]["path"], "r:gz") as tar:
        files = _members(tar, manifest["code_root"])
    require(sorted(files) == sorted(manifest["code"]["members"]), "code archive members differ from the manifest")
    require(tree_sha256(files) == manifest["source_tree_sha256"], "source-tree hash differs")
    require(runner_sha256(files) == manifest["runner_sha256"], "runner hash differs")
    require(sha256_bytes(files[SPEC_FILES[0]]) == manifest["spec_sha256"], "specification hash differs")
    with tarfile.open(bundle_dir / manifest["folds"]["path"], "r:gz") as tar:
        fold_files = _members(tar, manifest["folds"]["fold_set"])
    require(sha256_bytes(fold_files["fold_membership.csv"]) == manifest["folds"]["fold_membership_sha256"],
            "fold file hash differs")
    for dest, content in ((code_dest, files), (folds_dest, fold_files)):
        for relative, data in content.items():
            path = dest / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    on_disk = {str(p.relative_to(code_dest)): p.read_bytes() for p in code_dest.rglob("*") if p.is_file()}
    require(tree_sha256(on_disk) == manifest["source_tree_sha256"], "extracted source-tree hash differs")
    receipt = {"applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_dest": str(code_dest),
               "folds_dest": str(folds_dest), "manifest_sha256": sha256_file(bundle_dir / "v3_bundle_manifest.json"),
               "source_tree_sha256": manifest["source_tree_sha256"], "git_commit": manifest["git_commit"]}
    (code_dest / "v3_bundle_apply_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    a = sub.add_parser("apply")
    a.add_argument("--bundle-dir", type=Path, required=True)
    a.add_argument("--code-dest", type=Path, required=True)
    a.add_argument("--folds-parent", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (build(args.out_dir, folds=args.folds) if args.cmd == "build"
              else apply(args.bundle_dir, code_dest=args.code_dest, folds_parent=args.folds_parent))
    print(json.dumps({k: v for k, v in result.items() if k != "code"}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleError as exc:
        print(f"v3 bundle refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
