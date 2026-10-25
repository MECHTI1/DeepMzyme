"""Provenance and input refusals for the EC class versus CLEAN30 campaign runner (torch-free).

Every receipt the runner writes carries ``provenance_record``: git HEAD and branch, the
tree status (``git status --porcelain=v1 --untracked-files=all``), SHA-256 hashes of the
campaign code and an aggregate hash of every ``src/**/*.py`` file, the Python runtime and
package versions (read from installed metadata, without importing them), the host and the
runner's own command line.

Real (non-smoke) actions call ``require_clean_tree`` first: an uncommitted change or an
untracked file anywhere in the tree refuses the action, so every campaign artifact names
a commit that reproduces it. Engineering smokes record the tree state instead.

``refuse_test_arguments`` and ``refuse_test_paths`` keep test-side inputs out: no trainer
flag that evaluates or names a held-out test set is accepted, and no path whose
components name a test split, a mixed metadata folder or the PinMyMetal test files.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PROVENANCE_SCHEMA_VERSION = 1

# Campaign code whose hashes every receipt records (repository-relative; globs allowed).
CAMPAIGN_CODE_PATHS: tuple[str, ...] = (
    "src/ec_clean30/*.py",
    "scripts/run_ec_clean30_campaign.py",
    "src/train.py",
    "src/training/access_guard.py",
    "src/training/config.py",
    "src/training/data.py",
    "src/training/ec_class_metrics.py",
    "src/training/ec_classes.py",
    "src/training/ec_membership.py",
    "src/training/ec_prediction_export.py",
    "src/training/global_esm_features.py",
    "src/training/run.py",
    "src/training/runtime_preparation.py",
    "src/training/site_filter.py",
    "src/training/structure_loading.py",
    "src/embed_helpers/esmc_chain_means.py",
)
PACKAGE_DISTRIBUTIONS: tuple[str, ...] = (
    "torch", "torch-geometric", "numpy", "biopython", "esm", "scikit-learn", "pandas",
)
RECORDED_ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES", "MKL_THREADING_LAYER", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED",
    "DEEPMZYME_PARSE_CACHE_DIR", "DEEPMZYME_GRAPH_CACHE_DIR", "DEEPMZYME_FORBIDDEN_READ_ROOTS",
)

# Trainer flags that evaluate, name or calibrate on a held-out test set.
TEST_FLAGS: frozenset[str] = frozenset({
    "--run-test-eval", "--test-structure-dir", "--test-summary-csv", "--allow-train-loss-test-eval-debug",
    "--allow-final-refit-test-eval",
})
TEST_FLAG_PREFIXES: tuple[str, ...] = ("--test-", "--final-test-")
# Any flag name with a "test" word (e.g. --enable-final-test-calibration, --no-final-test-bootstrap-ci).
_TEST_FLAG_WORD_RE = re.compile(r"(^|-)test(-|$)", re.IGNORECASE)
# Path components that hold held-out or mixed train/test files.
FORBIDDEN_PATH_PARTS: frozenset[str] = frozenset({"test", "tests", "metadata", "pinmymetal_files"})
_TEST_TOKEN_RE = re.compile(r"(^|[_.\-])test([_.\-]|$)", re.IGNORECASE)
# The official fold folders and files are named ``CLEAN_30_train_test_split_<k>[...]``; that dataset
# name is not a test marker, so it is removed before the token check (``..._k_test.csv`` still fails).
_DATASET_NAME_RE = re.compile(r"train_test_split", re.IGNORECASE)


class DirtyTreeError(RuntimeError):
    """A real campaign action was asked to run from a tree with uncommitted or untracked changes."""


class HeldOutInputError(ValueError):
    """A command or path names a held-out test input."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Git state
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        result = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return result.returncode, result.stdout


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """HEAD, branch and full tree status (tracked changes and untracked files)."""
    code, head = _git(repo, "rev-parse", "HEAD")
    if code != 0:
        return {"repository_root": str(Path(repo).resolve()), "head": None, "branch": None, "clean": None,
                "status_entries": None, "status_sample": [], "error": "not a git checkout"}
    _code, branch = _git(repo, "branch", "--show-current")
    status_code, status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    lines = [line for line in status.splitlines() if line.strip()] if status_code == 0 else None
    return {
        "repository_root": str(Path(repo).resolve()),
        "head": head.strip(),
        "branch": branch.strip() or None,
        "clean": (not lines) if lines is not None else None,
        "status_entries": len(lines) if lines is not None else None,
        "status_sample": (lines or [])[:50],
        "status_sha256": sha256_bytes(status.encode("utf-8")) if status_code == 0 else None,
    }


def require_clean_tree(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Return the git state, or raise ``DirtyTreeError`` unless the tree is a clean checkout."""
    state = git_state(repo)
    if state["head"] is None:
        raise DirtyTreeError(f"{repo} is not a git checkout; real campaign actions need a committed tree.")
    if not state["clean"]:
        sample = "; ".join(state["status_sample"][:5])
        raise DirtyTreeError(
            f"The tree at {repo} has {state['status_entries']} uncommitted or untracked entries (e.g. {sample}). "
            "Commit first: real campaign actions run only from a clean tree (engineering smokes use --smoke)."
        )
    return state


# ---------------------------------------------------------------------------
# Code and environment
# ---------------------------------------------------------------------------

def expand_code_paths(patterns: Iterable[str], repo: Path = REPO_ROOT) -> list[str]:
    """Repository-relative files matching ``patterns`` (globs only inside code folders)."""
    files: set[str] = set()
    for pattern in patterns:
        matches = sorted(Path(repo).glob(pattern)) if any(ch in pattern for ch in "*?[") else [Path(repo) / pattern]
        for path in matches:
            if path.is_file():
                files.add(path.relative_to(repo).as_posix())
    return sorted(files)


def code_sha256(patterns: Iterable[str] = CAMPAIGN_CODE_PATHS, repo: Path = REPO_ROOT) -> dict[str, str]:
    return {relative: sha256_file(Path(repo) / relative) for relative in expand_code_paths(patterns, repo)}


def source_tree_sha256(repo: Path = REPO_ROOT) -> str:
    """Aggregate hash over every ``src/**/*.py`` path and its bytes (the trainer's cache identity scope)."""
    digest = hashlib.sha256()
    src = Path(repo) / "src"
    for path in sorted(src.rglob("*.py")):
        digest.update(path.relative_to(src).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def package_versions(distributions: Sequence[str] = PACKAGE_DISTRIBUTIONS) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_record() -> dict[str, Any]:
    return {
        "python": {"executable": sys.executable, "version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "environment": {name: os.environ.get(name) for name in RECORDED_ENVIRONMENT_VARIABLES},
    }


def provenance_record(*, smoke: bool, argv: Optional[Sequence[str]] = None, repo: Path = REPO_ROOT,
                      git: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The provenance block of every receipt."""
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "smoke": bool(smoke),
        "git": dict(git) if git is not None else git_state(repo),
        "code_sha256": code_sha256(repo=repo),
        "source_tree_sha256": source_tree_sha256(repo),
        "environment": environment_record(),
        "argv": list(sys.argv if argv is None else argv),
    }


def real_action_git_state(*, smoke: bool, repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Clean-tree check for real actions; smokes only record the state."""
    return git_state(repo) if smoke else require_clean_tree(repo)


# ---------------------------------------------------------------------------
# Test-input refusals
# ---------------------------------------------------------------------------

def names_test_split(part: str) -> bool:
    """Whether one path component names a test split or a mixed metadata folder."""
    lowered = part.lower()
    return lowered in FORBIDDEN_PATH_PARTS or bool(_TEST_TOKEN_RE.search(_DATASET_NAME_RE.sub("", lowered)))


def is_test_named_path(path: str | os.PathLike, *, relative_to: Optional[str | os.PathLike] = None) -> bool:
    """Whether ``path`` names a test input.

    With ``relative_to`` every component below that root is checked (a source layout, a
    manifest's relative path, a file inside a view). Without it only the final component
    is checked, so an operator-chosen absolute location is judged by its own name, not by
    the folders above it.
    """
    path = Path(os.fspath(path))
    if relative_to is None:
        parts = (path.name,)
    else:
        try:
            parts = path.relative_to(Path(os.fspath(relative_to))).parts
        except ValueError:
            parts = path.parts
    return any(names_test_split(part) for part in parts if part not in ("/", "", "."))


def refuse_test_paths(paths: Iterable[Optional[str | os.PathLike]], *, what: str = "path",
                      relative_to: Optional[str | os.PathLike] = None) -> None:
    for path in paths:
        if path is None:
            continue
        if is_test_named_path(path, relative_to=relative_to):
            raise HeldOutInputError(f"{what} {os.fspath(path)!r} names a test split or mixed metadata folder; refused.")


def refuse_test_arguments(argv: Sequence[str]) -> None:
    """Refuse trainer flags that evaluate or name a held-out test set."""
    for token in argv:
        if not token.startswith("-"):
            continue
        flag = token.split("=", 1)[0]
        if flag in TEST_FLAGS or flag.startswith(TEST_FLAG_PREFIXES) or _TEST_FLAG_WORD_RE.search(flag.lstrip("-")):
            raise HeldOutInputError(f"Trainer flag {flag} names or evaluates a held-out test set; refused.")


# ---------------------------------------------------------------------------
# Freeze manifests
# ---------------------------------------------------------------------------

FREEZE1_FORMAT = "deepmzyme_ec_clean30_freeze1_inputs_v1"
FREEZE2_FORMAT = "deepmzyme_ec_clean30_freeze2_v1"
# An engineering smoke manifest carries this schema version and no test_pass_output_root, so the
# label-free test pass (training.ec_prediction_export, schema_version 1 only) refuses it.
FREEZE2_SMOKE_SCHEMA_VERSION = "smoke-1"
# Campaign code under src/ that the Freeze-2 manifest pins beside the test pass's required files.
FREEZE2_CAMPAIGN_SOURCE_PATHS: tuple[str, ...] = (
    "src/ec_clean30/*.py",
    "src/training/__init__.py",
    "src/training/ec_classes.py",
    "src/training/ec_class_metrics.py",
    "src/training/ec_membership.py",
    "src/training/access_guard.py",
    "src/embed_helpers/esmc_chain_means.py",
    "src/audit_sequence_remoteness.py",
    "src/project_paths.py",
    "src/train.py",
)
# Code outside src/ (the test pass pins only src/...py paths, so these go in a separate table).
FREEZE2_ADDITIONAL_CODE_PATHS: tuple[str, ...] = (
    "scripts/run_ec_clean30_campaign.py",
    "scripts/run_ec_clean30_identity.py",
    "scripts/run_ec_clean30_baseline.py",
    "scripts/score_ec_clean30_comparison.py",
    "scripts/run_clean_split30_comparator.py",
)


class FreezeManifestError(ValueError):
    """A Freeze manifest cannot be written from the given inputs."""


def write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Write a JSON document that must not exist yet; returns its SHA-256."""
    import json

    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
    return sha256_bytes(data)


def build_freeze1_manifest(
    *,
    plans: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]] = (),
    smoke: bool,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Inputs of the Freeze-1 pre-registration: plan hashes, constants, provisional settings, views and code."""
    from ec_clean30 import selection, stats
    from ec_clean30.campaign import CAMPAIGN_ID, PROVISIONAL_LEVER_SETTINGS, campaign_constants

    folds = sorted(int(plan["fold"]) for plan in plans)
    if len(folds) != len(set(folds)):
        raise FreezeManifestError("Each outer fold may appear once.")
    selection_constants = {name: getattr(selection, name) for name in (
        "MIN_GAIN_POINTS", "PROMOTION_ALPHA", "LR_TIE_BAND_POINTS", "GUARD_MIN_TRAINING_PROTEINS",
        "GUARD_RECALL_DROP_POINTS", "GUARD_NET_PROTEINS", "NULL_ADOPTION_TARGET", "DEFAULT_SIMULATION_SEED",
        "SUPCON_MIN_POSITIVE_STEP_FRACTION")}
    selection_constants.update({name: getattr(stats, name) for name in (
        "DEFAULT_BOOTSTRAP_SEED", "DEFAULT_N_DRAWS", "DEFAULT_MAX_SKIP_FRACTION") if hasattr(stats, name)})
    return {
        "format": FREEZE1_FORMAT,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "kind": "freeze1_inputs",
        "smoke": bool(smoke),
        "plans": {str(plan["fold"]): {"plan_sha256": plan["plan_sha256"], "counts": plan["counts"],
                                      "filters": plan["filters"]} for plan in plans},
        "constants": campaign_constants(),
        "provisional_lever_settings": {key: list(value) for key, value in PROVISIONAL_LEVER_SETTINGS.items()},
        "selection_constants": selection_constants,
        "views": [dict(view) for view in views],
        "code_sha256": code_sha256((*CAMPAIGN_CODE_PATHS, "src/ec_clean30/*.py",
                                    "scripts/run_ec_clean30_identity.py")),
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=smoke),
    }


def build_freeze2_manifest(
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    test_pass_output_root: Optional[str | os.PathLike],
    smoke: bool,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The Freeze-2 manifest in the format the label-free test pass reads.

    ``schema_version`` 1, ``checkpoint_sha256s``, ``code_sha256`` (``src/...py`` paths covering
    ``training.ec_prediction_export.FREEZE2_REQUIRED_CODE_PATHS`` plus the campaign's own
    source files) and an absolute ``test_pass_output_root``; checkpoint records and code
    outside ``src/`` are kept in separate fields the test pass ignores.

    A smoke manifest (engineering checkpoints) has the same content under
    ``schema_version`` ``"smoke-1"`` and without ``test_pass_output_root``, so the test pass
    refuses it; a real manifest refuses smoke checkpoint records.
    """
    from training.ec_prediction_export import FREEZE2_SCHEMA_VERSION, freeze2_required_code_files
    from ec_clean30.campaign import CAMPAIGN_ID

    root = None
    if test_pass_output_root is not None:
        root = Path(os.fspath(test_pass_output_root))
        if not root.is_absolute():
            raise FreezeManifestError(f"test_pass_output_root must be an absolute path, got {root}.")
    elif not smoke:
        raise FreezeManifestError("A real Freeze-2 manifest needs an absolute test_pass_output_root.")
    if not checkpoints:
        raise FreezeManifestError("A Freeze-2 manifest needs at least one checkpoint.")
    digests = []
    for item in checkpoints:
        digest = str(item.get("sha256") or "")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise FreezeManifestError(f"Checkpoint record {item.get('unit_id')!r} has no valid SHA-256.")
        if not smoke and item.get("smoke") is not False:
            raise FreezeManifestError(f"{item.get('unit_id')} is not recorded as a real (non-smoke) unit; refused in "
                                      "a real manifest.")
        digests.append(digest)
    required = freeze2_required_code_files()
    pinned = sorted(set(required) | {path for path in expand_code_paths(FREEZE2_CAMPAIGN_SOURCE_PATHS)
                                     if path.startswith("src/") and path.endswith(".py")})
    manifest = {
        "schema_version": FREEZE2_SMOKE_SCHEMA_VERSION if smoke else FREEZE2_SCHEMA_VERSION,
        "format": FREEZE2_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "kind": "freeze2",
        "smoke": bool(smoke),
        "checkpoint_sha256s": sorted(set(digests)),
        "checkpoints": [dict(item) for item in sorted(checkpoints, key=lambda item: str(item.get("unit_id")))],
        "code_sha256": {path: sha256_file(REPO_ROOT / path) for path in pinned},
        "additional_code_sha256": code_sha256(FREEZE2_ADDITIONAL_CODE_PATHS),
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=smoke),
    }
    if smoke:
        manifest["smoke_output_root"] = None if root is None else str(root)
    else:
        manifest["test_pass_output_root"] = str(root)
    return manifest
