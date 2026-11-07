"""General amendment-chain verifier of the EC class versus CLEAN30 campaign (science side; torch-free).

``verify_chain(tip_path, *, expected_freeze1_sha256, expected_freeze1b_sha256, repo, require_published=True,
expected_chain=None)`` authenticates the Freeze-1 amendment chain that ends at ``tip_path`` on a full clone and
returns exactly ``RESULT_KEYS``. ``chain_receipt_record`` runs it and returns the hash-bound receipt that
``amendment2.write_chain_receipt`` writes once; ``check_chain_receipt(receipt_path, *, repo, expected_chain=None)``
re-validates that receipt cheaply on a pod or per unit, also in a blobless partial clone, and returns the same keys.
The checks are C1-C16 of ``docs/campaigns/ec_class_vs_clean30_2026-10/amendment2/interface_contract.md``; the
machine form is ``tests/fixtures/ec_clean30_amendment2_contract.json``, whose ``amendment_record.record_shape``
fixes every record field this verifier reads. A failed check raises ``AmendmentChainError``; nothing is downgraded
to a warning. The module imports only ``amendment_rules``, ``freeze`` and ``provenance`` of the campaign package;
the partition sets, function pins, change classes, role table, log-line format, ``r_cmd_v2_check`` and
``scientific_identity`` are called from ``amendment_rules``. Two names of the record builder are used at call time
only: ``amendment2.render_amendment2`` (C2, looked up by name) and the two receipt formats, which this module
repeats (a test keeps them equal).

How a chain is read
- Links. Link 1 is the S1 amendment (``amendment_rules.AMENDMENT1``). A link of sequence n >= 2 lives at
  ``amendment_rules.link_path(n)``, names its predecessor in ``base_amendment`` and repeats its SHA-256 as
  ``previous_sha256``; ``chain_before`` lists the earlier links.
- Source commits. Link 1's is ``AMENDMENT1["commit"]``. The source commit of a link of sequence n >= 2 is the
  commit that introduced its strict campaign-log line ``amendment_rules.link_log_line(n, sha256)``; the link file,
  its rendering and that line are committed together there. Every link file is written once: on the history
  reachable from HEAD its path was added once with these bytes and never modified or deleted. The campaign log at
  HEAD names each link of sequence 2 to the tip once and no other link (``amendment_rules.link_log_problems``), so
  a fork, a rewrite or a stale tip is refused.
- Fit chain: the chain up to its last link that changed a scientific file (links 1 and 2 always do; a later link
  does when its ``source_changes`` hold a scientific file).

How a chain is authenticated
- ``verify_chain`` refuses a partial clone at once (use ``check_chain_receipt`` there). It runs only the code of the
  repository it verifies: for a ``repo`` whose verifier is not this file it runs that repository's ``verify_chain``
  in a fresh interpreter, and with ``require_published`` only when that repository's verifier files
  (``RECEIPT_CODE_FILES``) are byte-identical to this one's.
- Per link (C4), in a detached sparse checkout of the link's source commit (``/src/``, ``/scripts/`` and the campaign
  docs folder; a class-(a2) link adds the top-level test files and the amendment-2 fixtures). Before the published
  code there runs, every ``src/`` and ``scripts/`` blob of that commit is hashed and must equal the composite pins
  of the authenticated records (so the code that judges a link is the code its records pin). Link 1:
  ``freeze.verify_s1_amendment`` as published at link 1's commit; it is never run on the live tree. Link n >= 2:
  ``verify_chain`` as published at link n's commit, which authenticates link n-1 the same way, down to link 1; its
  result is cross-checked against the records and must carry a scientific identity. A repository already at the
  link's commit with a clean tree runs the published code in place. An uncommitted tip is authenticated in place
  only without ``require_published`` (draft review; never for real use).
- Per-link checks at the link's commit: C1 format and keys, C2 canonical bytes and the builder's rendering, C3
  order, C5 scope, C7 partition, private names, analysis channels, the static import closure of every scientific
  file and a fresh-interpreter import probe, C8 bound files with the required roles, the decisions file and the
  rule text, C9 rule policies (every rule, the analysis rule included, recomputed in a fresh interpreter), the
  function pins and ``live_constants`` SHA-256, C10 v1 plans, C11 dry-run receipts and ``r_cmd_v2_check`` up to
  the link's commit, C13 data state, change class and bound futility files, C16 for class (a2) with its fixture
  reruns.
- Tip only, on the live tree: C6 (scientific files equal the composite scientific pins, post-fit analysis files the
  analysis pins, the pinned function closures their pins, every link file and rendering its published bytes and
  every file the tip binds its SHA-256), C7 on the live tree, C10 and C11 (``r_cmd_v2_check`` from the dry-run
  commit to HEAD, v1 and v2 plans), C12 publication and C14 (``amendment_rules.scientific_identity`` over exactly
  ``SCIENTIFIC_IDENTITY_KEYS``). Execution files are pinned by the execution manifest in force (manifest v5),
  which this verifier does not read; it fixes their closed list and the pinned gate functions inside them.
- Every git call and fresh interpreter runs without the caller's ``GIT_*`` variables, without system or global git
  configuration and in its own process group under one shrinking time budget; a timed-out group is killed and its
  temporary files go with it.
- Private bindings (``amendment_rules.PRIVATE_ROLES``: the dry-run-v2 receipts, the CLEAN tie exposure and the F6
  hold-out lists and corpus; the repository is public). C8 and C11 read them from the working tree of the
  verification repository, never from git, and refuse one that is absent, does not hash, has a symbolic link in its
  path, lies outside ``amendment_rules.PRIVATE_BINDING_ROOT`` or is tracked at the link's commit; a private role
  bound publicly and a public binding flagged private are refused (``amendment_rules.binding_privacy_problems``).
  A nested verification in a checkout of a link's commit reads them from the outermost verification repository,
  whose root the parent passes as an argument of the fixed chain probe (never through the environment, so a
  top-level call always reads its own working tree). Public bindings are
  read from git at the link's commit. ``check_chain_receipt`` never reads a private file: the receipt lists the
  private bindings apart, and a pod holds none of them.

Helpers for consumers: ``check_chain_receipt``, ``record_precedence_problems`` (C15 by git ancestry and
publication), ``fold_dry_run_evidence``, ``identity_record``, ``import_isolation_problems``,
``analysis_channel_problems`` and ``execution_closure_problems``.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from ec_clean30 import amendment_rules as rules
from ec_clean30 import freeze, provenance

RESULT_KEYS: tuple[str, ...] = (
    "links", "chain", "fit_chain", "tip_sequence", "composite_scientific_pins", "analysis_pins",
    "execution_file_set", "post_fit_analysis_files", "pinned_execution_functions", "pinned_analysis_functions",
    "policies", "plan_records", "dry_run_evidence", "scientific_identity_sha256", "published",
)
VERIFY_KEYWORDS: tuple[str, ...] = (
    "expected_freeze1_sha256", "expected_freeze1b_sha256", "repo", "require_published", "expected_chain",
)
RECEIPT_CHECK_KEYWORDS: tuple[str, ...] = ("repo", "expected_chain")
# C1: every key a link of sequence 2 or higher carries (the contract fixture's amendment_record.required_keys).
LINK_REQUIRED_KEYS: tuple[str, ...] = (
    "format", "schema_version", "amendment_id", "sequence", "change_class", "base_freeze1", "base_freeze1b_sha256",
    "base_amendment", "previous_sha256", "chain_before", "timestamps", "data_state", "decisions", "rules",
    "supersedes", "plans", "dry_runs", "source_parent_commit", "execution_file_set", "pinned_execution_functions",
    "post_fit_analysis_files", "pinned_analysis_functions", "live_constants_sha256", "source_changes",
    "unchanged_scientific_files", "inputs", "deferred_bindings", "fixtures", "evidence", "deviations", "publication",
)
# Keys the record builder adds when it builds a link from the reviewed draft (C8, C9).
BUILT_RECORD_KEYS: tuple[str, ...] = (
    "confirmation_status", "provisional_markers_settled", "decisions_file", "bound_files", "rule_text",
    "rule_policy_sha256",
)
BASE_AMENDMENT_KEYS: tuple[str, ...] = ("sequence", "amendment_id", "path", "sha256", "commit")
FREEZE1_PATH = f"{freeze.CAMPAIGN_DOCS_DIR}/freeze1.json"
SPARSE_CHECKOUT_PATHS: tuple[str, ...] = ("/src/", "/scripts/", f"/{freeze.CAMPAIGN_DOCS_DIR}/")
# A class-(a2) link reruns the named refusal tests and the decision-fixture test at its commit: its checkout adds
# the top-level test files and the amendment-2 fixtures (no test subfolder is materialised).
FIXTURE_RERUN_SPARSE_PATHS: tuple[str, ...] = ("/tests/*.py", "/tests/fixtures/ec_clean30_amendment2_*")
# C7: a fresh interpreter that imports these modules loads no execution file and no post-fit analysis file.
ISOLATED_IMPORTS: tuple[str, ...] = tuple(f"ec_clean30.{name}" for name in (
    "campaign", "commands", "views", "provenance", "identity", "selection", "selector", "stats", "futility",
    "clean_sanity", "amendment_rules", "amendment_chain", "runtime_precision"))
DRY_RUN_V2_RECEIPT_FORMAT = "deepmzyme_ec_clean30_dry_run_v2_receipt_v1"  # amendment2.DRY_RUN_V2_RECEIPT_FORMAT
CHAIN_RECEIPT_FORMAT = "deepmzyme_ec_clean30_amendment_chain_receipt_v1"  # amendment2.CHAIN_RECEIPT_FORMAT
CHAIN_RECEIPT_SCHEMA_VERSION = 2
# The receipt is keyed by the SHA-256 of these files (the verifier and the modules it trusts).
RECEIPT_CODE_FILES: tuple[str, ...] = (
    "src/ec_clean30/amendment_chain.py", "src/ec_clean30/amendment_rules.py", "src/ec_clean30/freeze.py",
    "src/ec_clean30/provenance.py",
)
RECORD_RENDERER = "ec_clean30.amendment2:render_amendment2"
R_CMD_V2_POLICY_FUNCTION = "ec_clean30.amendment_rules:r_cmd_v2_check"
PUBLICATION = "required_before_execution"
CONFIRMED = "confirmed"
LINK2_DATA_STATE: Mapping[str, Any] = {"fold0_records_at_draft": [], "g2_completed_units": 0}
FIXTURE_ROLES: tuple[str, ...] = ("refusals", "decisions")
DECISIONS_BINDING = "decisions"
EVIDENCE_SECTIONS: tuple[str, ...] = ("bound_files", "fixtures", "evidence", "inputs", "plans", "dry_runs",
                                      "data_state")
FUT_KIND = rules.FUT_KIND
FUTILITY_CONTINUE, FUTILITY_STOP = rules.FUTILITY_DECISIONS
# One budget for a whole verification; every nested level hands its children less (a deadline that shrinks with
# the nesting depth), so a parent always outlives its children and can clean up after them.
CHAIN_BUDGET_SECONDS = 3600.0
NESTING_MARGIN_SECONDS = 20.0
MAX_CHAIN_LENGTH = 64
_DEPTH_VARIABLE = "DEEPMZYME_EC_CLEAN30_CHAIN_DEPTH"
_DEADLINE_VARIABLE = "DEEPMZYME_EC_CLEAN30_CHAIN_DEADLINE"
_GIT_SAFE_ENVIRONMENT: Mapping[str, str] = {
    "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_TEST_SPEC_RE = re.compile(r"^(?P<file>test_[A-Za-z0-9_]+\.py)::(?P<name>test_[A-Za-z0-9_]+)$")
_NULL_OID = "0" * 40


class AmendmentChainError(ValueError):
    """The amendment chain does not authenticate; nothing that depends on it may run."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AmendmentChainError(message)


def _refuse(message: str, problems: Sequence[str]) -> None:
    if problems:
        raise AmendmentChainError(message + "\n  " + "\n  ".join(list(problems)[:40]))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(_COMMIT_RE.fullmatch(value))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _folds() -> list[str]:
    return [str(fold) for fold in range(freeze.N_OUTER_FOLDS)]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# The verification context: git environment and time budget


_DEADLINES: list[float] = []
_PRIVATE_ROOTS: list[Path] = []


def _environment_number(name: str, kind: type) -> Any:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = kind(raw)
    except ValueError:
        raise AmendmentChainError(f"The variable {name} is malformed ({raw[:40]!r}); the verification is refused.") \
            from None
    _require(value >= 0, f"The variable {name} is negative; the verification is refused.")
    return value


@contextmanager
def _verification(budget: Optional[float] = None) -> Iterator[None]:
    """A verification: no ``GIT_*`` variable of the caller reaches any git call or fresh interpreter (a hook's
    ``GIT_DIR`` would point git at the verified repository), no system or global git configuration, no terminal
    credential request; one deadline for the whole call, inherited from a parent verification when nested."""
    saved = dict(os.environ)
    inherited = _environment_number(_DEADLINE_VARIABLE, float)
    if _DEADLINES:
        deadline = _DEADLINES[-1]
    elif inherited is not None:
        deadline = inherited
    else:
        deadline = time.time() + (CHAIN_BUDGET_SECONDS if budget is None else float(budget))
    if budget is not None:
        deadline = min(deadline, time.time() + float(budget))
    _DEADLINES.append(deadline)
    try:
        for key in [key for key in os.environ if key.startswith("GIT_")]:
            del os.environ[key]
        os.environ.update(_GIT_SAFE_ENVIRONMENT)
        yield
    finally:
        _DEADLINES.pop()
        os.environ.clear()
        os.environ.update(saved)


def _remaining(what: str) -> float:
    deadline = _DEADLINES[-1] if _DEADLINES else time.time() + CHAIN_BUDGET_SECONDS
    remaining = deadline - time.time()
    _require(remaining > 0, f"{what}: the verification budget is spent; the chain is refused.")
    return remaining


def _child_environment(extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_") and key != "PYTHONPATH"}
    env.update(_GIT_SAFE_ENVIRONMENT)
    env.update(extra or {})
    return env


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_process(argv: Sequence[str], *, cwd: Optional[Path], what: str, env: Optional[Mapping[str, str]] = None,
                 input_bytes: Optional[bytes] = None) -> subprocess.CompletedProcess:
    """Run ``argv`` in its own process group within the remaining budget; on a timeout the whole group is killed.
    The child's temporary files live in a folder of this process, removed whatever happens to the child."""
    timeout = _remaining(what)
    with tempfile.TemporaryDirectory(prefix="ec30-chain-run-") as temporary:
        child_env = dict(env if env is not None else _child_environment())
        child_env.update({"TMPDIR": temporary, "TMP": temporary, "TEMP": temporary})
        try:
            process = subprocess.Popen(list(argv), cwd=None if cwd is None else str(cwd), env=child_env,
                                       stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        except OSError as exc:
            raise AmendmentChainError(f"{what} could not start ({type(exc).__name__}).") from None
        try:
            out, err = process.communicate(input=input_bytes, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            process.communicate()
            raise AmendmentChainError(f"{what} exceeded the verification budget; its process group was stopped.") \
                from None
        except BaseException:
            _kill_group(process)
            process.communicate()
            raise
        finally:
            _kill_group(process)  # no grandchild outlives the call
    return subprocess.CompletedProcess(list(argv), process.returncode, out, err)


def _git(repo: Path, *args: str) -> tuple[int, str]:
    return provenance._git(Path(repo), *args)


def _git_bytes(repo: Optional[Path], *args: str, what: str, input_bytes: Optional[bytes] = None) -> bytes:
    completed = _run_process(["git", "-c", f"core.hooksPath={os.devnull}", *args], cwd=repo, what=what,
                             input_bytes=input_bytes)
    reason = (completed.stderr.decode("utf-8", "replace").strip().splitlines() or ["no output"])[-1][:200]
    _require(completed.returncode == 0, f"{what} failed ({reason}).")
    return completed.stdout


def _run_git(*args: str) -> None:
    _git_bytes(None, *args, what=f"git {args[2] if args[:1] == ('-C',) else args[0]}")


def _head(repo: Path) -> Optional[str]:
    return freeze._resolve_commit(repo, "HEAD")


def _origin(repo: Path) -> Optional[str]:
    return freeze._resolve_commit(repo, freeze.ORIGIN_MAIN_REF)


def _commit_time(repo: Path, commit: Optional[str]) -> Optional[str]:
    if commit is None:
        return None
    code, out = _git(repo, "show", "-s", "--format=%cI", commit)
    return out.strip() if code == 0 and out.strip() else None


def _status_text(repo: Path) -> str:
    code, out = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    _require(code == 0, f"git status failed in {repo}.")
    return out


def _safe_relative(path: Any) -> bool:
    return (isinstance(path, str) and bool(path) and not path.startswith(("/", "~")) and "\\" not in path
            and ".." not in Path(path).parts and ":" not in path.split("/")[0])


def _working_tree_bytes(repo: Path, path: str) -> Optional[bytes]:
    if not _safe_relative(path):
        return None
    target = Path(repo) / path
    cursor = Path(repo)
    for part in Path(path).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return None
    return target.read_bytes() if target.is_file() else None


def _probe_private_root(raw: Any) -> Path:
    """The private root a parent verification passed to the chain probe: an absolute directory."""
    _require(isinstance(raw, str) and os.path.isabs(raw) and Path(raw).is_dir(),
             f"The chain probe names no private root directory ({str(raw)[:40]!r}); the verification is refused.")
    return Path(os.path.realpath(raw))


def _private_root(repo: Path) -> Path:
    return _PRIVATE_ROOTS[-1] if _PRIVATE_ROOTS else Path(repo)


def _private_file(root: Path, path: Any) -> tuple[Optional[bytes], Optional[str]]:
    """The bytes of a private binding's file in the working tree of ``root`` (never from git), or why there are
    none."""
    if not rules.is_private_path(path):
        return None, f"lies outside the private binding root {rules.PRIVATE_BINDING_ROOT}"
    cursor = Path(root)
    for part in Path(path).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return None, "has a symbolic link in its path in the verification repository's working tree"
    data = _working_tree_bytes(Path(root), path)
    if data is None:
        return None, ("is absent from the verification repository's working tree (a private input is never read "
                      "from git)")
    return data, None


def _private_reader(root: Path) -> Callable[[str], tuple[Optional[bytes], Optional[str]]]:
    return lambda path: _private_file(root, path)


def _tracked_check(repo: Path, commit: Optional[str]) -> Callable[[str], Optional[bool]]:
    """Whether a path is tracked at ``commit`` (without one: at HEAD or in the index); None when git cannot tell."""
    def tracked(path: str) -> Optional[bool]:
        if not _safe_relative(path):
            return None
        revisions = [commit] if commit is not None else ["HEAD"]
        for revision in revisions:
            code, out = _git(repo, "--literal-pathspecs", "ls-tree", "-z", "--name-only", revision, "--", path)
            if code == 0 and out:
                return True
            if code != 0 and commit is not None:
                return None
        if commit is None:
            code, out = _git(repo, "--literal-pathspecs", "ls-files", "-z", "--", path)
            if code != 0:
                return None
            return bool(out)
        return False

    return tracked


def _reader(repo: Path, commit: Optional[str]) -> Callable[[str], Optional[bytes]]:
    """Bytes of a repository file at ``commit`` (from the object store), or in the working tree without one."""
    if commit is None:
        return lambda path: _working_tree_bytes(repo, path)
    return lambda path: freeze._show_bytes(repo, commit, path) if _safe_relative(path) else None


def _runs_from(repo: Path) -> bool:
    """Whether this file is the verifier of ``repo`` (so in-process checks run the repository's own code)."""
    target = Path(repo) / "src" / "ec_clean30" / "amendment_chain.py"
    return target.is_file() and Path(__file__).resolve() == target.resolve()


def _code_sha256(root: Path) -> dict[str, Optional[str]]:
    return {path: (_sha256(data) if (data := _working_tree_bytes(root, path)) is not None else None)
            for path in RECEIPT_CODE_FILES}


def partial_clone_problem(repo: Path) -> Optional[str]:
    """Why ``repo`` is a partial clone (promisor remote, partial-clone filter or promisor packs, also behind an
    alternate object store), or None for a full clone."""
    code, out = _git(repo, "config", "--get-regexp",
                     r"^(extensions\.partialclone|remote\..*\.promisor|remote\..*\.partialclonefilter)$")
    if code == 0 and out.strip():
        return f"its git configuration names a promisor remote ({out.split()[0]})"
    code, objects = _git(repo, "rev-parse", "--git-path", "objects")
    if code != 0 or not objects.strip():
        return None
    folder = Path(objects.strip())
    folder = folder if folder.is_absolute() else Path(repo) / folder
    stores = [folder]
    alternates = folder / "info" / "alternates"
    if alternates.is_file():
        for line in alternates.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip() and not line.startswith("#"):
                path = Path(line.strip())
                stores.append(path if path.is_absolute() else folder / path)
    for store in stores:
        if (store / "pack").is_dir() and any((store / "pack").glob("*.promisor")):
            return f"its object store {store.name} holds promisor packs"
    return None


_PRELUDE = "import json, sys\nsys.path.insert(0, sys.argv[1])\nargs = json.loads(sys.argv[2])\n"
_FAIL = ("except Exception as exc:\n"
         "    print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}))\n"
         "    sys.exit(3)\n")
# The parent's private root (``_PRIVATE_ROOTS``) is an argument of this fixed probe: the working tree whose private
# files the nested verification reads. It is never taken from the environment.
_CHAIN_PROBE = _PRELUDE + (
    "from ec_clean30 import amendment_chain\n"
    "try:\n"
    "    amendment_chain._PRIVATE_ROOTS.append(amendment_chain._probe_private_root(args['private_root']))\n"
    "    result = amendment_chain.verify_chain(args['tip'], expected_freeze1_sha256=args['freeze1'],\n"
    "        expected_freeze1b_sha256=args['freeze1b'], repo=args['repo'],\n"
    "        require_published=args['require_published'], expected_chain=args['chain'])\n"
    + _FAIL + "print(json.dumps({'result': result}, sort_keys=True))\n")
_S1_PROBE = _PRELUDE + (
    "from ec_clean30 import freeze\n"
    "try:\n"
    "    verified = freeze.verify_s1_amendment(args['path'], expected_freeze1_sha256=args['freeze1'],\n"
    "        repo=args['repo'], require_published=args['require_published'], expected_chain=args['chain'])\n"
    + _FAIL + "print(json.dumps({'result': {key: verified[key] for key in\n"
    "    ('document_sha256', 'amendment_chain', 's1_rule', 'policy', 'published')}}, sort_keys=True))\n")
_IMPORT_PROBE = _PRELUDE + (
    "import importlib, os\n"
    "try:\n"
    "    for name in args['modules']:\n"
    "        importlib.import_module(name)\n"
    + _FAIL + "print(json.dumps({'result': sorted({os.path.realpath(module.__file__) for module in\n"
    "    list(sys.modules.values()) if isinstance(getattr(module, '__file__', None), str)})}))\n")
_POLICY_PROBE = _PRELUDE + (
    "from ec_clean30 import amendment_rules\n"
    "try:\n"
    "    result = amendment_rules.rule_policy_sha256(include_analysis=True)\n"
    + _FAIL + "print(json.dumps({'result': result}, sort_keys=True))\n")


def _run_fresh(root: Path, code: str, args: Mapping[str, Any], *, what: str) -> Any:
    """Run ``code`` in a fresh, isolated interpreter whose first import root is ``<root>/src``; the child gets a
    shorter deadline than this process."""
    depth = _environment_number(_DEPTH_VARIABLE, int) or 0
    _require(depth < 4 * MAX_CHAIN_LENGTH, f"{what}: the nested verification is deeper than any chain allows.")
    child_deadline = (_DEADLINES[-1] if _DEADLINES else time.time() + CHAIN_BUDGET_SECONDS) - NESTING_MARGIN_SECONDS
    _require(child_deadline > time.time(), f"{what}: the verification budget is spent; the chain is refused.")
    env = _child_environment({_DEPTH_VARIABLE: str(depth + 1), _DEADLINE_VARIABLE: repr(child_deadline),
                              "PYTHONDONTWRITEBYTECODE": "1"})
    completed = _run_process([sys.executable, "-I", "-B", "-c", code, str(Path(root) / "src"),
                              json.dumps(dict(args))], cwd=Path(root), what=what, env=env)
    lines = completed.stdout.decode("utf-8", "replace").strip().splitlines()
    payload: Any = None
    if lines:
        try:
            payload = json.loads(lines[-1])
        except ValueError:
            payload = None
    if completed.returncode != 0 or not isinstance(payload, dict) or "result" not in payload:
        reason = payload.get("error") if isinstance(payload, dict) and payload.get("error") else (
            (completed.stderr.decode("utf-8", "replace").strip().splitlines() or ["no output"])[-1][:400])
        raise AmendmentChainError(f"{what} refused: {reason}")
    return payload["result"]


@contextmanager
def _checkout(repo: Path, commit: str, *, extra_paths: Sequence[str] = ()) -> Iterator[Path]:
    """A detached sparse checkout of ``commit`` (``phase3._baseline_authentication`` pattern) whose ``origin/main``
    is the verified repository's ``origin/main``."""
    _require(_is_commit(commit) and freeze._resolve_commit(repo, commit) == commit,
             f"The source commit {str(commit)[:12]!r} is not a full commit of this repository.")
    origin = _origin(repo)
    with tempfile.TemporaryDirectory(prefix="ec30-chain-link-") as temporary:
        checkout = Path(temporary) / "source"
        _run_git("clone", "--quiet", "--shared", "--no-checkout", str(repo), str(checkout))
        _run_git("-C", str(checkout), "sparse-checkout", "set", "--no-cone", *SPARSE_CHECKOUT_PATHS, *extra_paths)
        if origin is not None:
            _run_git("-C", str(checkout), "update-ref", freeze.ORIGIN_MAIN_REF, origin)
        else:
            _run_git("-C", str(checkout), "update-ref", "-d", freeze.ORIGIN_MAIN_REF)
        _run_git("-C", str(checkout), "checkout", "--quiet", "--detach", commit)
        yield checkout


def _commit_tree_pins(repo: Path, commit: str) -> dict[str, str]:
    """SHA-256 of every ``src/`` and ``scripts/`` blob at ``commit``, read from the object store (one
    ``git cat-file --batch``)."""
    listing = _git_bytes(repo, "ls-tree", "-r", "-z", "--full-tree", commit, "--", *rules.TRACKED_CODE_FOLDERS,
                         what=f"git ls-tree of {commit[:12]}")
    entries = []
    for item in listing.split(b"\0"):
        if not item:
            continue
        meta, _, name = item.partition(b"\t")
        mode, kind, oid = meta.decode().split()
        _require(kind == "blob" and mode in ("100644", "100755"),
                 f"[C4] {name.decode('utf-8', 'replace')} at {commit[:12]} is not a regular file.")
        entries.append((name.decode("utf-8"), oid))
    if not entries:
        return {}
    data = _git_bytes(repo, "cat-file", "--batch", what=f"git cat-file of {commit[:12]}",
                      input_bytes="".join(f"{oid}\n" for _path, oid in entries).encode())
    pins, cursor = {}, 0
    for path, oid in entries:
        end = data.index(b"\n", cursor)
        header = data[cursor:end].decode().split()
        _require(len(header) == 3 and header[0] == oid and header[1] == "blob",
                 f"[C4] The object store does not hold {path} at {commit[:12]}.")
        size = int(header[2])
        pins[path] = _sha256(data[end + 1:end + 1 + size])
        cursor = end + 1 + size + 1
    return dict(sorted(pins.items()))


# ---------------------------------------------------------------------------
# Links (C1, C3)


@dataclass
class _Link:
    sequence: int
    path: str
    data: bytes
    document: dict
    sha256: str
    commit: Optional[str]


def _parse(data: bytes, what: str) -> dict:
    try:
        document = json.loads(data.decode("utf-8"))
    except ValueError:
        raise AmendmentChainError(f"{what} is not JSON.") from None
    _require(isinstance(document, dict), f"{what} is not a JSON object.")
    return document


def link_header_problems(document: Mapping[str, Any], sequence: int) -> list[str]:
    """C1 for a link of sequence 2 or higher: format, schema version, campaign, sequence, every required key and the
    base Freeze-1 bindings; link 2 also has the amendment-2 identity and change class."""
    problems = []
    if document.get("format") != rules.AMENDMENT2_FORMAT:
        problems.append(f"[C1] link {sequence} has the unknown format {document.get('format')!r}")
    if not (_is_int(document.get("schema_version")) and document["schema_version"] == 1):
        problems.append(f"[C1] link {sequence} does not have schema_version 1")
    if document.get("campaign_id") != rules.CAMPAIGN_ID:
        problems.append(f"[C1] link {sequence} names another campaign")
    if not (_is_int(document.get("sequence")) and document["sequence"] == sequence):
        problems.append(f"[C1] link {sequence} states the sequence {document.get('sequence')!r}")
    missing = [key for key in (*LINK_REQUIRED_KEYS, *BUILT_RECORD_KEYS) if key not in document]
    if missing:
        problems.append(f"[C1] link {sequence} lacks {missing}")
    if sequence == rules.AMENDMENT2_SEQUENCE:
        if document.get("amendment_id") != rules.AMENDMENT2_ID:
            problems.append(f"[C1] link 2 is not {rules.AMENDMENT2_ID}")
        if document.get("change_class") != rules.CHANGE_PRE_DATA:
            problems.append(f"[C1] link 2 does not have the change class {rules.CHANGE_PRE_DATA}")
    elif document.get("change_class") == rules.CHANGE_PRE_DATA:
        problems.append(f"[C1] link {sequence} has the change class {rules.CHANGE_PRE_DATA} of link 2")
    if document.get("base_freeze1") != {"path": FREEZE1_PATH, "sha256": rules.FREEZE1_SHA256}:
        problems.append(f"[C1] link {sequence} names another base Freeze 1")
    if document.get("base_freeze1b_sha256") != rules.FREEZE1B_SHA256:
        problems.append(f"[C1] link {sequence} names another Freeze 1b")
    if document.get("publication") != PUBLICATION:
        problems.append(f"[C1] link {sequence} does not require publication before execution")
    return problems


def _introducing_commit(repo: Path, sequence: int, digest: str) -> Optional[str]:
    """The earliest commit reachable from HEAD that introduced the link's strict campaign-log line."""
    code, out = _git(repo, "log", f"-S{rules.link_log_line(sequence, digest)}", "--reverse", "--format=%H", "HEAD",
                     "--", freeze.LOG_PATH)
    commits = out.split() if code == 0 else []
    return commits[0] if commits else None


def _collect_links(repo: Path, tip: str) -> list[_Link]:
    """The links from the tip down to link 1 (C1 headers, C3 order, fixed paths), oldest first."""
    data = _working_tree_bytes(repo, tip)
    _require(data is not None, f"The chain tip {tip} is not a regular file of the repository.")
    path, document, commit_hint = tip, _parse(data, f"The chain tip {tip}"), None
    links: list[_Link] = []
    while True:
        _require(len(links) < MAX_CHAIN_LENGTH, "[C3] The chain is longer than any campaign chain.")
        digest = provenance.sha256_bytes(data)
        sequence = document.get("sequence")
        _require(_is_int(sequence) and sequence >= 1, f"[C1] {path} states no sequence.")
        if links:
            _require(sequence == links[-1].sequence - 1,
                     f"[C3] link {links[-1].sequence} names a predecessor of sequence {sequence} (a gap, a reorder or "
                     "a second link of one sequence).")
        if document.get("format") == freeze.S1_AMENDMENT_FORMAT:
            _require(sequence == 1 and path == rules.AMENDMENT1["path"] and digest == rules.AMENDMENT1["sha256"],
                     "[C3] Link 1 must be the S1 amendment of amendment_rules.AMENDMENT1.")
            _require(commit_hint in (None, rules.AMENDMENT1["commit"]),
                     "[C3] Link 2 names another source commit for link 1.")
            links.append(_Link(1, path, data, document, digest, rules.AMENDMENT1["commit"]))
            break
        _require(sequence >= 2 and document.get("format") == rules.AMENDMENT2_FORMAT,
                 f"[C1] {path} has the unknown format {document.get('format')!r}.")
        _refuse(f"[C1] Link {sequence} has the wrong form.", link_header_problems(document, sequence))
        _require(path == rules.link_path(sequence), f"[C1] Link {sequence} must be {rules.link_path(sequence)} "
                                                    "(one fixed path per sequence).")
        introduced = _introducing_commit(repo, sequence, digest)
        if commit_hint is None:
            commit = introduced
        else:
            _require(introduced == commit_hint, f"[C3] Link {sequence + 1} names a source commit for link {sequence} "
                                                "that did not enter its log line in the campaign log.")
            commit = commit_hint
        if commit is not None:
            committed = freeze._show_bytes(repo, commit, path)
            _require(committed is not None and provenance.sha256_bytes(committed) == digest,
                     f"[C12] Link {sequence} is not committed with these bytes at the commit that entered its log "
                     "line (commit the link, its rendering and its log line together).")
            code, parents = _git(repo, "rev-list", "--parents", "-n", "1", commit)
            prior = [freeze._show_bytes(repo, parent, path) for parent in parents.split()[1:]]
            _require(code == 0 and not any(data is not None and provenance.sha256_bytes(data) == digest
                                           for data in prior),
                     f"[C12] Link {sequence} was committed before the commit that entered its log line (commit the "
                     "link, its rendering and its log line together).")
        links.append(_Link(sequence, path, data, document, digest, commit))
        base = document.get("base_amendment")
        _require(isinstance(base, Mapping) and set(BASE_AMENDMENT_KEYS) <= set(base)
                 and _is_int(base.get("sequence")) and _safe_relative(base.get("path"))
                 and _is_sha256(base.get("sha256")) and _is_commit(base.get("commit")),
                 f"[C3] Link {sequence} names its predecessor incompletely.")
        _require(base["sequence"] == sequence - 1,
                 f"[C3] Link {sequence} names a predecessor of sequence {base['sequence']} (a gap or a reorder).")
        _require(document.get("previous_sha256") == base["sha256"],
                 f"[C3] previous_sha256 of link {sequence} is not its predecessor's SHA-256.")
        if sequence == rules.AMENDMENT2_SEQUENCE:
            _require({key: base[key] for key in BASE_AMENDMENT_KEYS}
                     == {key: rules.AMENDMENT1[key] for key in BASE_AMENDMENT_KEYS},
                     "[C3] Link 2 must name the S1 amendment of amendment_rules.AMENDMENT1 as its predecessor.")
        previous = freeze._show_bytes(repo, base["commit"], base["path"])
        _require(previous is not None and provenance.sha256_bytes(previous) == base["sha256"],
                 f"[C3] The predecessor of link {sequence} is not committed with its SHA-256 at its source commit.")
        _require(_working_tree_bytes(repo, base["path"]) == previous,
                 f"[C2] The live file of link {sequence - 1} differs from its published bytes (links are written once).")
        path, data, commit_hint = base["path"], previous, base["commit"]
        document = _parse(data, f"Link {sequence - 1}")
        _require(document.get("amendment_id") == base["amendment_id"],
                 f"[C3] Link {sequence} names another amendment ID for its predecessor.")
    links.reverse()
    chain = [link.sha256 for link in links]
    _require(len(set(chain)) == len(chain), "[C3] A link occurs twice in the chain.")
    for link in links[1:]:
        _require(link.document.get("chain_before") == chain[:link.sequence - 1],
                 f"[C3] chain_before of link {link.sequence} differs from the earlier links (a fork or a reorder).")
    return links


def written_once_problems(repo: Path, links: Sequence[_Link]) -> list[str]:
    """C3/C12: on the history reachable from HEAD, every committed link file was added once with its published
    bytes and never modified or deleted (a rewrite that is published again is refused, also after a restore)."""
    problems = []
    for link in links:
        if link.commit is None:
            continue
        code, oid = _git(repo, "rev-parse", f"{link.commit}:{link.path}")
        if code != 0 or not oid.strip():
            problems.append(f"[C3] link {link.sequence} has no blob at its source commit")
            continue
        code, out = _git(repo, "log", "--full-history", "-m", "--no-renames", "--raw", "--no-abbrev", "--format=",
                         "HEAD", "--", link.path)
        if code != 0:
            problems.append(f"[C3] the history of link {link.sequence} cannot be read")
            continue
        statuses, oids = set(), set()
        for line in out.splitlines():
            if not line.startswith(":"):
                continue
            fields = line.split("\t", 1)[0].split()
            if len(fields) < 5:
                continue
            statuses.add(fields[4][:1])
            oids |= {value for value in fields[2:4] if value != _NULL_OID}
        if statuses - {"A"} or oids != {oid.strip()}:
            problems.append(f"[C3] link {link.sequence} at {link.path} was rewritten or deleted on the history of "
                            "HEAD (links are written once)")
    return problems


def _execution_files(document: Mapping[str, Any]) -> list[str]:
    if document.get("format") == freeze.S1_AMENDMENT_FORMAT:
        return []
    item = document.get("execution_file_set")
    _require(isinstance(item, Mapping) and isinstance(item.get("files"), list)
             and all(isinstance(path, str) for path in item["files"]),
             "[C7] execution_file_set must hold the closed list of execution files and the SHA-256 of the sorted "
             "name list.")
    return list(item["files"])


def _is_scientific_link(link: _Link) -> bool:
    if link.sequence <= rules.AMENDMENT2_SEQUENCE:
        return True
    split = rules.changed_set(link.document.get("source_changes") or {}, _execution_files(link.document))
    return bool(split["scientific"])


def _fit_chain(links: Sequence[_Link]) -> list[str]:
    last = max(index for index, link in enumerate(links) if _is_scientific_link(link))
    return [link.sha256 for link in links[:last + 1]]


def _freeze1(repo: Path, expected: str) -> dict:
    data = _working_tree_bytes(repo, FREEZE1_PATH)
    _require(data is not None and provenance.sha256_bytes(data) == expected,
             "The base Freeze 1 is missing or does not hash to the expected SHA-256.")
    document = _parse(data, "Freeze 1")
    pins = (document.get("code") or {}).get("tracked_file_sha256")
    _require(document.get("format") == freeze.FREEZE1_DOCUMENT_FORMAT and document.get("campaign_id") == rules.CAMPAIGN_ID
             and document.get("amendments") == [] and isinstance(pins, dict) and pins
             and all(_safe_relative(path) and _is_sha256(value) for path, value in pins.items()),
             "The base Freeze 1 is not the original, unamended Freeze-1 document with code pins.")
    return document


def _link1_pins(freeze1: Mapping[str, Any], link1: Mapping[str, Any]) -> dict[str, str]:
    pins = dict(freeze1["code"]["tracked_file_sha256"])
    for path, change in (link1.get("source_changes") or {}).items():
        pins[path] = change["after"]
    return dict(sorted(pins.items()))


def _record_composite(links: Sequence[_Link], freeze1: Mapping[str, Any]) -> dict[str, str]:
    """The composite pins of the non-execution files after the last link, from the authenticated records."""
    pins = _link1_pins(freeze1, links[0].document)
    for link in links[1:]:
        execution = set(_execution_files(link.document))
        pins = {path: value for path, value in pins.items() if path not in execution}
        for path, change in (link.document.get("source_changes") or {}).items():
            pins[path] = change.get("after") if isinstance(change, Mapping) else None
    return dict(sorted(pins.items()))


def _split_pins(pins: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    analysis = {path: value for path, value in pins.items() if path in rules.POST_FIT_ANALYSIS_FILES}
    scientific = {path: value for path, value in pins.items() if path not in rules.POST_FIT_ANALYSIS_FILES}
    return dict(sorted(scientific.items())), dict(sorted(analysis.items()))


# ---------------------------------------------------------------------------
# Per-link checks (pure where possible; the verifier passes the values found at the link's commit)


def scope_problems(previous_pins: Mapping[str, str], tree_pins: Mapping[str, str],
                   source_changes: Mapping[str, Any], execution_files: Iterable[str],
                   unchanged_files: Iterable[str] = (), *, sequence: int) -> tuple[list[str], dict[str, str]]:
    """C5: ``source_changes`` equals the set of non-execution files whose bytes differ between the previous composite
    pins and the link's tree; each ``before`` is the previous composite pin (None for a new file) and each ``after``
    the link's bytes; no pinned file is deleted; no execution file is in scope; from sequence 3 on no pinned file
    joins the execution set; every ``unchanged_scientific_files`` entry (a file or a folder ending in ``/``) names
    tracked files outside the scope. Returns the problems and the composite pins after the link."""
    execution = set(execution_files)
    problems: list[str] = []
    if not isinstance(source_changes, Mapping):
        return ["[C5] source_changes is not a mapping"], {}
    if sequence > rules.AMENDMENT2_SEQUENCE:
        problems += [f"[C5] the pinned file {path} cannot join the execution set after amendment 2"
                     for path in sorted(execution & set(previous_pins))]
    previous = {path: value for path, value in previous_pins.items() if path not in execution}
    tree = {path: value for path, value in tree_pins.items() if path not in execution}
    problems += [f"[C5] the pinned file {path} is deleted" for path in sorted(set(previous) - set(tree))]
    problems += [f"[C5] the execution file {path} is in scope" for path in sorted(set(source_changes) & execution)]
    differ = {path for path in set(previous) | set(tree) if previous.get(path) != tree.get(path)}
    problems += [f"[C5] {path} changed but is not in source_changes"
                 for path in sorted(differ - set(source_changes) - execution)]
    problems += [f"[C5] {path} is in source_changes but did not change"
                 for path in sorted(set(source_changes) - differ - execution)]
    for path, change in sorted(source_changes.items()):
        if not isinstance(change, Mapping):
            problems.append(f"[C5] the change of {path} is malformed")
            continue
        if change.get("before") != previous.get(path):
            problems.append(f"[C5] the before pin of {path} is not the previous composite pin")
        if not _is_sha256(change.get("after")) or change.get("after") != tree.get(path):
            problems.append(f"[C5] the after pin of {path} is not its bytes at the link's commit")
    for entry in unchanged_files:
        matched = [path for path in tree if path == entry or (entry.endswith("/") and path.startswith(entry))]
        if not isinstance(entry, str) or not matched:
            problems.append(f"[C5] the unchanged file entry {entry!r} names no tracked non-execution file")
        problems += [f"[C5] {path} is declared unchanged but is in source_changes"
                     for path in matched if path in source_changes]
    composite = dict(previous)
    for path, change in source_changes.items():
        if isinstance(change, Mapping):
            composite[path] = change.get("after")
    return problems, dict(sorted(composite.items()))


def live_source_problems(live_pins: Mapping[str, str], scientific_pins: Mapping[str, str],
                         analysis_pins: Mapping[str, str], execution_files: Iterable[str]) -> list[str]:
    """C6: every non-execution tracked file equals its composite pin (scientific or post-fit analysis), none is
    missing and none is added without a link."""
    execution = set(execution_files)
    live = {path: value for path, value in live_pins.items() if path not in execution}
    problems = []
    for pins, what in ((scientific_pins, "scientific"), (analysis_pins, "post-fit analysis")):
        problems += [f"[C6] the {what} file {path} is missing from the live tree" for path in sorted(set(pins) - set(live))]
        problems += [f"[C6] the {what} file {path} differs from the composite {what} pins"
                     for path in sorted(path for path in pins if path in live and live[path] != pins[path])]
    problems += [f"[C6] {path} is neither pinned by the chain nor an execution file"
                 for path in sorted(set(live) - set(scientific_pins) - set(analysis_pins))]
    return problems


def evidence_entries(record: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """Every ``{"path", "sha256"}`` binding below the record's evidence sections, as (where, path, sha256)."""
    return [entry for section in EVIDENCE_SECTIONS for entry in rules.file_bindings(record.get(section), section)]


def _default_private_reader(read: Callable[[str], Optional[bytes]]
                            ) -> Callable[[str], tuple[Optional[bytes], Optional[str]]]:
    def private_read(path: str) -> tuple[Optional[bytes], Optional[str]]:
        data = read(path)
        return data, None if data is not None else "does not exist"

    return private_read


def binding_hash_problems(read: Callable[[str], Optional[bytes]], record: Mapping[str, Any], *,
                          where: str = "at the link's commit",
                          private_read: Optional[Callable[[str], tuple[Optional[bytes], Optional[str]]]] = None,
                          tracked: Optional[Callable[[str], Optional[bool]]] = None) -> list[str]:
    """C8: every file binding of the record's evidence sections has the privacy of its role
    (``amendment_rules.binding_privacy_problems``) and hashes to its recorded SHA-256. A public binding is read with
    ``read`` (git at the link's commit); a private binding with ``private_read`` (``(bytes, reason)`` from the
    verification repository's working tree; the verifier always passes it) and, with ``tracked``, must not be
    tracked ``where``."""
    private_read = private_read or _default_private_reader(read)
    problems = [f"[C8] {problem}" for problem in rules.binding_privacy_problems(record, EVIDENCE_SECTIONS)]
    for section in EVIDENCE_SECTIONS:
        for at, item in rules.binding_items(record.get(section), section):
            path, digest = item.get("path"), item.get("sha256")
            if not _safe_relative(path) or not _is_sha256(digest):
                problems.append(f"[C8] {at} is a malformed file binding")
                continue
            if rules.is_private_role(at):
                if not rules.is_private_binding(item):
                    continue  # refused above; a public copy of a private input is never read
                data, reason = private_read(path)
                if data is None:
                    problems.append(f"[C8] {at}: the private file {path} {reason}")
                elif provenance.sha256_bytes(data) != digest:
                    problems.append(f"[C8] {at}: the private file {path} does not hash to its recorded SHA-256 in "
                                    "the verification repository's working tree")
                state = tracked(path) if tracked is not None else False
                if state is None:
                    problems.append(f"[C8] {at}: git cannot tell whether the private file {path} is tracked {where}")
                elif state:
                    problems.append(f"[C8] {at}: the private file {path} is tracked {where} (a private input was "
                                    "published)")
                continue
            data = read(path)
            if data is None:
                problems.append(f"[C8] {at}: {path} does not exist {where}")
            elif provenance.sha256_bytes(data) != digest:
                problems.append(f"[C8] {at}: {path} does not hash to its recorded SHA-256 {where}")
    return problems


def decisions_file_problems(data: Optional[bytes], recorded: Any) -> list[str]:
    """C8: the bound decisions file itself answers exactly ``amendment_rules.CONFIRMATION_IDS`` (choice and
    ``accepted_default``), names the maintainer, holds no placeholder token and no absolute path, and equals the
    record's ``decisions_file`` (its SHA-256, maintainer and answers)."""
    if data is None:
        return ["[C8] the decisions file does not exist at the link's commit"]
    text = data.decode("utf-8", errors="replace")
    problems = []
    if freeze._PLACEHOLDER_TOKEN_RE.search(text):
        problems.append("[C8] the decisions file holds a placeholder token")
    if freeze._ABSOLUTE_PATH_RE.search(text):
        problems.append("[C8] the decisions file holds an absolute path")
    try:
        payload = json.loads(text)
    except ValueError:
        return problems + ["[C8] the decisions file is not JSON"]
    answers = payload.get("answers") if isinstance(payload, dict) else None
    maintainer = payload.get("maintainer") if isinstance(payload, dict) else None
    if not (isinstance(payload, dict) and payload.get("format") == rules.DECISIONS_FORMAT):
        problems.append(f"[C8] the decisions file is not a {rules.DECISIONS_FORMAT} file")
    if not (isinstance(answers, Mapping) and sorted(answers) == sorted(rules.CONFIRMATION_IDS) and all(
            isinstance(answer, Mapping) and isinstance(answer.get("choice"), str) and answer["choice"].strip()
            and isinstance(answer.get("accepted_default"), bool) for answer in answers.values())):
        problems.append(f"[C8] the decisions file does not settle exactly {list(rules.CONFIRMATION_IDS)} with a "
                        "choice and accepted_default each")
    if not (isinstance(maintainer, str) and maintainer.strip()):
        problems.append("[C8] the decisions file names no maintainer")
    expected = {"sha256": provenance.sha256_bytes(data),
                "maintainer": maintainer.strip() if isinstance(maintainer, str) else None,
                "answers": {key: dict(value) for key, value in sorted(answers.items())}
                if isinstance(answers, Mapping) and all(isinstance(value, Mapping) for value in answers.values())
                else None}
    if not isinstance(recorded, Mapping) or {key: recorded.get(key) for key in expected} != expected:
        problems.append("[C8] the record's decisions_file differs from the bound decisions file")
    return problems


def rule_text_problems(read: Callable[[str], Optional[bytes]], record: Mapping[str, Any]) -> list[str]:
    """C8: ``rule_text`` holds the text, its SHA-256 and its source, the reviewed document
    ``amendment_rules.rule_text_source(n)``, which holds that text exactly once at the link's commit."""
    item = record.get("rule_text")
    sequence = record.get("sequence")
    if not (isinstance(item, Mapping) and isinstance(item.get("text"), str) and item["text"].strip()):
        return ["[C8] rule_text holds no rule text"]
    problems = []
    if item.get("sha256") != _sha256(item["text"].encode("utf-8")):
        problems.append("[C8] the rule text does not hash to rule_text.sha256")
    expected = rules.rule_text_source(sequence) if _is_int(sequence) and sequence >= 2 else None
    if item.get("source") != expected:
        problems.append(f"[C8] rule_text.source is not the reviewed document {expected}")
        return problems
    data = read(expected)
    if data is None or data.decode("utf-8", errors="replace").count(item["text"]) != 1:
        problems.append(f"[C8] the reviewed document {expected} does not hold the bound rule text exactly once")
    return problems


def evidence_problems(read: Callable[[str], Optional[bytes]], record: Mapping[str, Any], *,
                      private_read: Optional[Callable[[str], tuple[Optional[bytes], Optional[str]]]] = None,
                      tracked: Optional[Callable[[str], Optional[bool]]] = None) -> list[str]:
    """C8: every bound file hashes to its recorded SHA-256 (public bindings at the link's commit, private bindings in
    the verification repository's working tree and untracked at the link's commit, see ``binding_hash_problems``);
    every role of ``amendment_rules.REQUIRED_BINDINGS`` is bound with the privacy of its role; the decisions file is
    bound, settles every confirmation and equals the record's ``decisions_file``; the record states that its
    confirmations and provisional markers are settled; the rule text is the reviewed document's; the refusal and
    decision fixtures are bound."""
    problems = binding_hash_problems(read, record, private_read=private_read, tracked=tracked)
    problems += [f"[C8] {problem}" for problem in rules.required_binding_problems(record)]
    if record.get("confirmation_status") != CONFIRMED or record.get("provisional_markers_settled") is not True:
        problems.append("[C8] the record does not state that its confirmations and provisional markers are settled")
    decisions = record.get("decisions_file")
    binding = (record.get("bound_files") or {}).get(DECISIONS_BINDING) if isinstance(record.get("bound_files"),
                                                                                    Mapping) else None
    if not (isinstance(decisions, Mapping) and isinstance(binding, Mapping)
            and _is_sha256(decisions.get("sha256")) and binding.get("sha256") == decisions["sha256"]):
        problems.append("[C8] the decisions file is not bound under bound_files.decisions with the recorded SHA-256")
    else:
        data = read(binding.get("path")) if _safe_relative(binding.get("path")) else None
        problems += decisions_file_problems(data, decisions)
    problems += rule_text_problems(read, record)
    fixtures = record.get("fixtures")
    for role in FIXTURE_ROLES:
        item = fixtures.get(role) if isinstance(fixtures, Mapping) else None
        if not (isinstance(item, Mapping) and _safe_relative(item.get("path")) and _is_sha256(item.get("sha256"))):
            problems.append(f"[C8] the {role} fixture is not bound")
    return problems


def policy_problems(record: Mapping[str, Any], live_rule_hashes: Mapping[str, str]) -> list[str]:
    """C9: every rule names its registered policy function and analysis flag; ``rule_policy_sha256`` binds every
    registered rule, and each policy given in ``live_rule_hashes`` equals the canonical output of its registered
    function at the link's commit (the verifier passes every rule, the analysis rule computed in a fresh
    interpreter so that no scientific process loads analysis code)."""
    problems = []
    entries = record.get("rules")
    if not isinstance(entries, Mapping):
        return ["[C9] rules is not a mapping"]
    expected_ids = set(rules.RULE_POLICY_FUNCTIONS) | {rules.R_CMD_V2_RULE}
    if set(entries) != expected_ids:
        problems.append(f"[C9] the rules differ from the registry (extra {sorted(set(entries) - expected_ids)}, "
                        f"missing {sorted(expected_ids - set(entries))})")
    for rule_id, registered in rules.RULE_POLICY_FUNCTIONS.items():
        item = entries.get(rule_id)
        if not (isinstance(item, Mapping) and item.get("policy_function") == registered["function"]
                and item.get("analysis") is registered["analysis"]):
            problems.append(f"[C9] the rule {rule_id} names another policy function or analysis flag")
    item = entries.get(rules.R_CMD_V2_RULE)
    if not (isinstance(item, Mapping) and item.get("policy_function") == R_CMD_V2_POLICY_FUNCTION):
        problems.append(f"[C9] the rule {rules.R_CMD_V2_RULE} names another function")
    hashes = record.get("rule_policy_sha256")
    if not (isinstance(hashes, Mapping) and set(hashes) == set(rules.RULE_POLICY_FUNCTIONS)
            and all(_is_sha256(value) for value in hashes.values())):
        problems.append("[C9] rule_policy_sha256 does not bind the policy of every registered rule")
        return problems
    for rule_id, digest in sorted(live_rule_hashes.items()):
        if hashes.get(rule_id) != digest:
            problems.append(f"[C9] the policy of {rule_id} differs from the canonical output of its registered "
                            "function at the link's commit")
    return problems


def plan_records(record: Mapping[str, Any], freeze1: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """C10 (record side): the v1 plan SHA-256 values equal Freeze 1 and every fold has a v2 plan SHA-256 (the v2
    records are compared with ``campaign.build_fold_plan_v2`` by ``amendment_rules.r_cmd_v2_check``)."""
    folds = _folds()
    plans = record.get("plans") if isinstance(record.get("plans"), Mapping) else {}
    v1 = {fold: (freeze1.get("plans") or {}).get(fold, {}).get("plan_sha256") for fold in folds}
    v2 = plans.get("v2_plan_sha256")
    problems = []
    if plans.get("v1_plan_sha256") != v1 or not all(_is_sha256(value) for value in v1.values()):
        problems.append("[C10] the v1 plan SHA-256 values differ from Freeze 1")
    if not (isinstance(v2, Mapping) and sorted(v2) == folds and all(_is_sha256(value) for value in v2.values())):
        problems.append("[C10] the record does not hold a v2 plan SHA-256 for every fold")
        v2 = {}
    return problems, {"v1": v1, "v2": dict(sorted(v2.items()))}


def _in_view_receipts(freeze1: Mapping[str, Any]) -> dict[str, Any]:
    return {fold: ((freeze1.get("views") or {}).get(fold) or {}).get("dry_run_receipt", {}).get("sha256")
            for fold in _folds()}


def _frozen_view_manifests(freeze1: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze 1's view manifest SHA-256 of each fold (the view each dry-run-v2 receipt must name)."""
    return {fold: ((freeze1.get("views") or {}).get(fold) or {}).get("view_manifest_sha256") for fold in _folds()}


def dry_run_problems(read: Callable[[str], Optional[bytes]], record: Mapping[str, Any],
                     freeze1: Mapping[str, Any], *, tracked: Optional[Callable[[str], Optional[bool]]] = None
                     ) -> tuple[list[str], dict[str, Any]]:
    """C11 (record side): five dry-run-v2 receipts are bound privately (``{path, sha256, private: true}`` below
    ``amendment_rules.PRIVATE_BINDING_ROOT``; ``read`` gives their bytes from the verification repository's working
    tree, never from git, and with ``tracked`` none is tracked at the link's commit), exist, hash, passed and ran on a
    clean tree at the recorded source commit, on the view whose manifest SHA-256 Freeze 1 records for that fold; the
    Freeze-1 in-view receipt SHA-256 values are unchanged. The returned evidence gives each fold's receipt binding
    ``{path, sha256}`` with that view manifest SHA-256 (a digest, never a count), so that the per-unit view binding
    (``fold_dry_run_evidence``) needs no receipt bytes on a pod; the in-view bytes are compared there."""
    folds = _folds()
    manifests = _frozen_view_manifests(freeze1)
    dry = record.get("dry_runs") if isinstance(record.get("dry_runs"), Mapping) else {}
    problems = []
    source = dry.get("source_commit")
    if dry.get("rule") != rules.R_CMD_V2_RULE:
        problems.append(f"[C11] the dry runs do not name the rule {rules.R_CMD_V2_RULE}")
    if not _is_commit(source):
        problems.append("[C11] the dry runs name no full source commit")
    receipts = dry.get("receipts")
    if not (isinstance(receipts, Mapping) and sorted(receipts) == folds):
        problems.append("[C11] the record does not bind one dry-run-v2 receipt per fold")
        receipts = {}
    bound = {}
    for fold, binding in sorted(receipts.items()):
        if not (isinstance(binding, Mapping) and _safe_relative(binding.get("path")) and _is_sha256(binding.get("sha256"))):
            problems.append(f"[C11] the fold-{fold} receipt binding is malformed")
            continue
        bound[fold] = {"path": binding["path"], "sha256": binding["sha256"],
                       "view_manifest_sha256": manifests.get(fold)}
        if not rules.is_private_binding(binding):
            problems.append(f"[C11] the fold-{fold} receipt is not bound privately ({{path, sha256, private: true}} "
                            f"below {rules.PRIVATE_BINDING_ROOT})")
            continue
        state = tracked(binding["path"]) if tracked is not None else False
        if state is not False:
            problems.append(f"[C11] the fold-{fold} dry-run-v2 receipt is tracked at the link's commit (a private "
                            "receipt was published)" if state else
                            f"[C11] git cannot tell whether the fold-{fold} dry-run-v2 receipt is tracked")
        data = read(binding["path"])
        if data is None or provenance.sha256_bytes(data) != binding["sha256"]:
            problems.append(f"[C11] the fold-{fold} dry-run-v2 receipt is missing or does not hash")
            continue
        try:
            receipt = json.loads(data.decode("utf-8"))
        except ValueError:
            problems.append(f"[C11] the fold-{fold} dry-run-v2 receipt is not JSON")
            continue
        git = ((receipt.get("provenance") or {}).get("git") or {}) if isinstance(receipt, dict) else {}
        if not (isinstance(receipt, dict) and receipt.get("format") == DRY_RUN_V2_RECEIPT_FORMAT
                and receipt.get("status") == "passed" and receipt.get("fold") == int(fold)
                and receipt.get("plan_version") == 2 and (receipt.get("provenance") or {}).get("smoke") is False):
            problems.append(f"[C11] the fold-{fold} receipt is not a passed, real plan-v2 dry run of that fold")
        if git.get("head") != source or git.get("clean") is not True:
            problems.append(f"[C11] the fold-{fold} dry run did not run on a clean tree at the recorded source commit")
        if not (_is_sha256(manifests.get(fold)) and isinstance(receipt, dict)
                and receipt.get("view_manifest_sha256") == manifests[fold]):
            problems.append(f"[C11] the fold-{fold} dry run did not run on the fold view that Freeze 1 records")
    in_view = _in_view_receipts(freeze1)
    if dry.get("in_view_receipts_unchanged") != in_view:
        problems.append("[C11] the Freeze-1 in-view dry-run receipts differ from Freeze 1")
    return problems, {"source_commit": source, "receipts": bound, "in_view_receipts_unchanged": in_view}


def data_state_problems(record: Mapping[str, Any]) -> list[str]:
    """C13 for link 2: an empty fold-0 record list and zero completed G2 units."""
    state = record.get("data_state")
    if not isinstance(state, Mapping):
        return ["[C13] link 2 has no data_state"]
    problems = [f"[C13] link 2 records {key} = {state.get(key)!r}, not {expected!r}"
                for key, expected in LINK2_DATA_STATE.items()
                if type(state.get(key)) is not type(expected) or state.get(key) != expected]
    if "fold0_records" in record and record["fold0_records"] != []:
        problems.append("[C13] link 2 lists fold-0 records")
    return problems


def futility_flags(records: Sequence[Mapping[str, Any]]) -> tuple[bool, bool, list[str]]:
    """``(fut_record_exists, stop_futility_exists, problems)`` from the ``fold0_records`` lists of links of sequence 3
    or higher (``{"kind", "sha256"}``; a ``FUT`` entry also names its ``decision`` and binds its pointer and
    futility record, which ``fut_evidence_problems`` verifies at the link's commit)."""
    exists = stopped = False
    problems = []
    for record in records:
        entries = record.get("fold0_records")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not (isinstance(entry, Mapping) and isinstance(entry.get("kind"), str) and _is_sha256(entry.get("sha256"))):
                problems.append(f"[C13] link {record.get('sequence')} lists a malformed fold-0 record")
                continue
            if entry["kind"] == FUT_KIND:
                if entry.get("decision") not in rules.FUTILITY_DECISIONS:
                    problems.append(f"[C13] link {record.get('sequence')} lists a FUT record without its decision")
                    continue
                if not (rules.is_file_binding(entry.get("pointer")) and rules.is_file_binding(entry.get("record"))):
                    problems.append(f"[C13] link {record.get('sequence')} lists a FUT record without its bound "
                                    "pointer and futility record")
                    continue
                exists = True
                stopped = stopped or entry.get("decision") == FUTILITY_STOP
    return exists, stopped, problems


def fut_evidence_problems(read: Callable[[str], Optional[bytes]], record: Mapping[str, Any],
                          chain_before: Sequence[str]) -> list[str]:
    """C13/C16: every FUT entry of a link's ``fold0_records`` binds the binding pointer file ``fut_seq<k>.json`` and
    the futility record it names, as repository files at the link's commit: the pointer has the futility pointer
    format, is binding, names a fit chain that is a prefix of the chain before the link (amendments 1 and 2 at
    least) and the record's SHA-256; the record has the futility record format, the same decision, is binding and
    names the same fit chain. A FUT record is never taken from the link's own statement."""
    problems = []
    sequence = record.get("sequence")
    for index, entry in enumerate(record.get("fold0_records") or []):
        if not (isinstance(entry, Mapping) and entry.get("kind") == FUT_KIND):
            continue
        where = f"[C13] link {sequence} FUT entry {index}"
        pointer_binding, record_binding = entry.get("pointer"), entry.get("record")
        if not (rules.is_file_binding(pointer_binding) and rules.is_file_binding(record_binding)):
            problems.append(f"{where} binds no pointer file and futility record")
            continue
        pointer_data, record_data = read(pointer_binding["path"]), read(record_binding["path"])
        if pointer_data is None or provenance.sha256_bytes(pointer_data) != pointer_binding["sha256"]:
            problems.append(f"{where}: the pointer file is missing or does not hash at the link's commit")
            continue
        if record_data is None or provenance.sha256_bytes(record_data) != record_binding["sha256"]:
            problems.append(f"{where}: the futility record is missing or does not hash at the link's commit")
            continue
        try:
            pointer, fut = json.loads(pointer_data.decode("utf-8")), json.loads(record_data.decode("utf-8"))
        except ValueError:
            problems.append(f"{where}: the pointer file or the futility record is not JSON")
            continue
        fit_chain = pointer.get("fit_chain") if isinstance(pointer, dict) else None
        decision = (fut.get("decision") or {}) if isinstance(fut, dict) else {}
        checks = (
            (isinstance(pointer, dict) and pointer.get("format") == rules.FUTILITY_POINTER_FORMAT,
             "the pointer file has another format"),
            (isinstance(pointer, dict) and pointer.get("binding") is True, "the pointer file is not binding"),
            (isinstance(pointer, dict) and Path(pointer_binding["path"]).name
             == f"fut_seq{pointer.get('sequence')}.json" and _is_int(pointer.get("sequence"))
             and rules.AMENDMENT2_SEQUENCE <= pointer["sequence"] < (sequence if _is_int(sequence) else 0),
             "the pointer file does not name its chain sequence before the link"),
            (isinstance(pointer, dict) and pointer.get("decision") == entry.get("decision"),
             "the pointer file gives another decision"),
            (isinstance(pointer, dict) and pointer.get("futility_record_sha256") == record_binding["sha256"]
             == entry.get("sha256"), "the pointer file names another futility record"),
            (isinstance(fit_chain, list) and len(fit_chain) >= rules.AMENDMENT2_SEQUENCE
             and list(chain_before[:len(fit_chain)]) == fit_chain, "the pointer file names another fit chain"),
            (isinstance(fut, dict) and fut.get("format") == rules.FUTILITY_RECORD_FORMAT,
             "the futility record has another format"),
            (isinstance(decision, Mapping) and decision.get("decision") == entry.get("decision")
             and decision.get("binding") is True, "the futility record gives another or a non-binding decision"),
            (isinstance(fut, dict) and fut.get("freeze1_amendment_chain") == fit_chain,
             "the futility record names another fit chain"),
        )
        problems += [f"{where}: {message}" for ok, message in checks if not ok]
    return problems


def later_link_problems(record: Mapping[str, Any], *, changed_files: Iterable[str], execution_files: Iterable[str],
                        fut_record_exists: bool, stop_futility_exists: bool,
                        previous: Optional[Mapping[str, Any]] = None) -> list[str]:
    """C13 for a link of sequence 3 or higher: ``amendment_rules.check_change_class`` on the files the link changes
    and, with ``previous`` (the previous link's record), its pinned functions."""
    return [f"[C13] {problem}" for problem in rules.check_change_class(
        record, changed_files=changed_files, execution_files=execution_files, fut_record_exists=fut_record_exists,
        stop_futility_exists=stop_futility_exists, previous=previous)]


def analysis_completion_problems(previous: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """C16: ``amendment_rules.check_analysis_completion`` between the states before and after a class-(a2) link."""
    return [f"[C16] {problem}" for problem in rules.check_analysis_completion(previous=previous, current=current)]


def identity_record(*, fit_chain: Sequence[str], composite_scientific_pins: Mapping[str, str],
                    pinned_execution_functions: Mapping[str, str], plans: Mapping[str, Mapping[str, str]],
                    freeze1_views: Mapping[str, Any], dry_run_evidence: Mapping[str, Any],
                    deferred_input_specifications: Any) -> dict[str, Any]:
    """C14: the scientific-identity record, exactly ``amendment_rules.SCIENTIFIC_IDENTITY_KEYS``."""
    return {
        "campaign_id": rules.CAMPAIGN_ID, "freeze1_sha256": rules.FREEZE1_SHA256,
        "freeze1b_sha256": rules.FREEZE1B_SHA256, "fit_chain": list(fit_chain),
        "composite_scientific_pins": dict(sorted(composite_scientific_pins.items())),
        "pinned_execution_functions": dict(sorted(pinned_execution_functions.items())),
        "plan_sha256_v1": dict(sorted((plans.get("v1") or {}).items())),
        "plan_sha256_v2": dict(sorted((plans.get("v2") or {}).items())),
        "freeze1_view_records": freeze1_views,
        "dry_run_receipts": {fold: item["sha256"] for fold, item in sorted((dry_run_evidence.get("receipts")
                                                                             or {}).items())},
        "precision_policy": rules.PRECISION_POLICY,
        "deferred_input_specifications": deferred_input_specifications,
    }


def _identity(state: Mapping[str, Any], freeze1: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    return rules.scientific_identity(identity_record(
        fit_chain=state["fit_chain"], composite_scientific_pins=state["composite_scientific_pins"],
        pinned_execution_functions=state["pinned_execution_functions"], plans=state["plan_records"],
        freeze1_views=freeze1.get("views") or {}, dry_run_evidence=state["dry_run_evidence"] or {},
        deferred_input_specifications=record.get("deferred_bindings")))


# ---------------------------------------------------------------------------
# Partition, imports and analysis channels (C7)


def _module_name(path: str) -> Optional[str]:
    return rules.module_name(path)


def import_isolation_problems(repo: Path, execution_files: Iterable[str],
                              tracked: Optional[Iterable[str]] = None) -> list[str]:
    """C7: a fresh interpreter that imports ``ISOLATED_IMPORTS`` from ``<repo>/src`` loads no execution file, no
    post-fit analysis file and, with ``tracked``, no ``src/`` or ``scripts/`` file that git does not track (a
    gitignored extension module found before its source would otherwise run unpinned)."""
    try:
        loaded = _run_fresh(Path(repo), _IMPORT_PROBE, {"modules": list(ISOLATED_IMPORTS)},
                            what="The import-isolation probe")
    except AmendmentChainError as exc:
        return [f"[C7] a fresh interpreter could not import the scientific modules ({exc})"]
    root = Path(os.path.realpath(repo))
    files = set()
    for item in loaded:
        path = Path(item)
        if root in path.parents:
            files.add(path.relative_to(root).as_posix())
    problems = [f"[C7] importing the scientific modules loads the execution file {path}"
                for path in sorted(files & set(execution_files))]
    problems += [f"[C7] importing the scientific modules loads the post-fit analysis file {path}"
                 for path in sorted(files & set(rules.POST_FIT_ANALYSIS_FILES))]
    if tracked is not None:
        known = set(tracked)
        problems += [f"[C7] importing the scientific modules loads the untracked file {path}"
                     for path in sorted(files) if path.startswith(("src/", "scripts/")) and path not in known]
    return problems


@dataclass
class _Scan:
    names: dict[str, set[str]]
    targets: dict[str, set[str]]
    problems: list[str]


def source_scan(repo: Path, files: Iterable[str]) -> _Scan:
    """One parse of every Python file of ``files``: the module names each imports (at any depth of the file) and
    the tracked files those names load."""
    files = sorted(set(files))
    known = set(files)
    names: dict[str, set[str]] = {}
    targets: dict[str, set[str]] = {}
    problems = []
    for path in files:
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse((Path(repo) / path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            problems.append(f"[C7] {path} cannot be parsed for its imports")
            continue
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    imported |= {".".join(parts[:end]) for end in range(1, len(parts) + 1)}
            elif isinstance(node, ast.ImportFrom):
                module = rules.resolve_import_from(path, node)
                if module:
                    parts = module.split(".")
                    imported |= {".".join(parts[:end]) for end in range(1, len(parts) + 1)}
                imported |= {f"{module}.{alias.name}" if module else alias.name for alias in node.names}
        names[path] = imported
        targets[path] = {target for name in imported if (target := rules.module_file(name, known)) and target != path}
    return _Scan(names, targets, problems)


def analysis_channel_problems(repo: Path, scientific_files: Iterable[str], *, scan: Optional[_Scan] = None) -> list[str]:
    """C7: only the files of ``amendment_rules.ANALYSIS_CHANNELS`` import a post-fit analysis module."""
    channel_files = {path for path, _channel in rules.ANALYSIS_CHANNELS}
    modules = {name for name in map(_module_name, rules.POST_FIT_ANALYSIS_FILES) if name}
    scientific_files = sorted(set(scientific_files))
    scan = scan or source_scan(repo, scientific_files)
    problems = [problem for problem in scan.problems
                if any(problem.startswith(f"[C7] {path} ") for path in scientific_files)]
    for path in scientific_files:
        if not path.endswith(".py") or path in channel_files or path not in scan.names:
            continue
        names = scan.names[path]
        problems += [f"[C7] {path} imports the post-fit analysis module {name} outside the pinned channels"
                     for name in sorted(modules) if any(item == name or item.startswith(name + ".") for item in names)]
    return problems


def execution_closure_problems(repo: Path, tracked: Iterable[str], execution_files: Iterable[str], *,
                               scan: Optional[_Scan] = None) -> list[str]:
    """C7: the static import closure of no scientific or post-fit analysis file (``training.run``,
    ``training.ec_prediction_export``, the embedding modules and every campaign module included) holds an execution
    file, and no execution file imports a post-fit analysis module."""
    tracked = sorted(set(tracked))
    execution = set(execution_files)
    scan = scan or source_scan(repo, tracked)
    reverse: dict[str, set[str]] = {}
    for path, targets in scan.targets.items():
        for target in targets:
            reverse.setdefault(target, set()).add(path)
    reaches: dict[str, str] = {}
    pending = [(path, path) for path in sorted(execution)]
    while pending:
        current, execution_file = pending.pop()
        for importer in sorted(reverse.get(current, ())):
            if importer not in reaches:
                reaches[importer] = execution_file
                pending.append((importer, execution_file))
    problems = [f"[C7] the scientific file {path} reaches the execution file {reaches[path]} through its imports"
                for path in sorted(reaches) if path not in execution]
    analysis = set(rules.POST_FIT_ANALYSIS_FILES)
    problems += [f"[C7] the execution file {path} imports the post-fit analysis file {target}"
                 for path in sorted(execution) for target in sorted(scan.targets.get(path, set()) & analysis)]
    return problems


def _partition_problems(repo: Path, record: Mapping[str, Any], tracked: Sequence[str],
                        execution: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    try:
        part = rules.partition(tracked, execution)
    except rules.AmendmentRuleError as exc:
        return [f"[C7] {exc}"], {"scientific_files": [], "post_fit_analysis_files": []}
    problems = []
    item = record.get("execution_file_set") or {}
    if list(execution) != part["execution_files"]:
        problems.append("[C7] the execution file list is not sorted or holds a name twice")
    if item.get("names_sha256") != part["execution_names_sha256"]:
        problems.append("[C7] the SHA-256 of the sorted execution name list differs")
    if record.get("post_fit_analysis_files") != list(rules.POST_FIT_ANALYSIS_FILES):
        problems.append("[C7] post_fit_analysis_files differs from the closed post-fit analysis set")
    return problems, part


def _private_import_problems(repo: Path, execution: Iterable[str], tracked: Iterable[str]) -> list[str]:
    execution = set(execution)
    known = set(tracked)

    def protected(name: str) -> bool:
        target = rules.module_file(name, known)
        return target is not None and target not in execution

    problems = []
    for path in sorted(execution):
        if path.endswith(".py") and (Path(repo) / path).is_file():
            try:
                problems += [f"[C7] {problem}" for problem in rules.private_import_problems(
                    (Path(repo) / path).read_text(encoding="utf-8"), path, protected=protected)]
            except (SyntaxError, ValueError, UnicodeDecodeError):
                problems.append(f"[C7] the execution file {path} does not parse")
    return problems


def _function_pin_problems(repo: Path, recorded: Any, registry: Mapping[str, Sequence[str]], allowed: Iterable[str],
                           what: str, check: str) -> tuple[list[str], dict[str, str]]:
    try:
        live = rules.pinned_function_hashes(repo, registry)
    except (rules.AmendmentRuleError, OSError, SyntaxError, ValueError) as exc:
        return [f"[{check}] the pinned {what} functions cannot be hashed ({exc})"], {}
    problems = []
    allowed = set(allowed)
    problems += [f"[{check}] the pinned {what} function {name} is not in a {what} file"
                 for name in sorted(live) if name.split("::")[0] not in allowed]
    if not isinstance(recorded, Mapping) or set(recorded) != set(live):
        problems.append(f"[{check}] the pinned {what} functions differ from the registry")
    else:
        problems += [f"[{check}] the closure of {name} differs from its pin" for name in sorted(live)
                     if recorded[name] != live[name]]
    return problems, live


def _tracked_files(repo: Path) -> list[str]:
    out = _git_bytes(repo, "ls-files", "-z", "--", *rules.TRACKED_CODE_FOLDERS, what="git ls-files")
    return sorted(item for item in out.decode("utf-8").split("\0") if item)


def _tracked_pins(repo: Path) -> tuple[list[str], dict[str, str], list[str]]:
    tracked = _tracked_files(repo)
    missing = [path for path in tracked if _working_tree_bytes(repo, path) is None]
    pins = {path: provenance.sha256_file(Path(repo) / path) for path in tracked if path not in missing}
    return tracked, pins, [f"[C6] the tracked file {path} is missing from the tree or a link" for path in missing]


def _fixture_payload(read: Callable[[str], Optional[bytes]], record: Mapping[str, Any], role: str) -> Any:
    fixtures = record.get("fixtures") or {}
    item = fixtures.get(role) if isinstance(fixtures, Mapping) else None
    data = read(item["path"]) if isinstance(item, Mapping) and _safe_relative(item.get("path")) else None
    try:
        return json.loads(data.decode("utf-8")) if data is not None else None
    except ValueError:
        return None


def _fixture_state(read: Callable[[str], Optional[bytes]], record: Mapping[str, Any]) -> dict[str, Any]:
    """The refusal-fixture IDs and the decision-fixture content bound by a link (the state class (a2) keeps)."""
    refusals = _fixture_payload(read, record, "refusals")
    decisions = _fixture_payload(read, record, "decisions")
    entries = refusals.get("refusals") if isinstance(refusals, dict) else None
    return {"refused_fixtures": sorted(str(entry.get("id")) for entry in entries or [] if isinstance(entry, Mapping)),
            "decision_fields": decisions if isinstance(decisions, dict) else {}}


def fixture_tests(read: Callable[[str], Optional[bytes]], record: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The tests a class-(a2) link reruns: every test the bound refusal fixture names and
    ``amendment_rules.DECISION_FIXTURE_TEST``, as ``"<file>::<name>"`` below ``tests/``."""
    payload = _fixture_payload(read, record, "refusals")
    entries = payload.get("refusals") if isinstance(payload, dict) else None
    problems = [] if isinstance(entries, list) and entries else ["[C16] the bound refusal fixture names no refusal"]
    tests = {rules.DECISION_FIXTURE_TEST}
    for entry in entries or []:
        spec = entry.get("test") if isinstance(entry, Mapping) else None
        if isinstance(spec, str) and _TEST_SPEC_RE.fullmatch(spec):
            tests.add(spec)
        else:
            problems.append(f"[C16] the refusal fixture entry {entry.get('id') if isinstance(entry, Mapping) else entry!r} "
                            "names no test")
    return sorted(tests), problems


def named_test_pins(read: Callable[[str], Optional[bytes]], tests: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    """``test -> closure SHA-256`` of each named test function (with the module-level names it references) as read
    by ``read``."""
    pins, problems = {}, []
    for spec in sorted(set(tests)):
        match = _TEST_SPEC_RE.fullmatch(spec)
        data = read(f"tests/{match['file']}") if match else None
        if data is None:
            problems.append(f"[C16] the named test {spec} does not exist")
            continue
        try:
            pins[spec] = rules.sha256_canonical(rules.function_closure(data.decode("utf-8"), [match["name"]]))
        except (rules.AmendmentRuleError, SyntaxError, ValueError, UnicodeDecodeError) as exc:
            problems.append(f"[C16] the named test {spec} cannot be pinned ({exc})")
    return pins, problems


def fixture_rerun_problems(root: Path, tests: Sequence[str]) -> list[str]:
    """C16: every named test passes at the link's commit (a fresh pytest in ``root``, no cache, no bytecode)."""
    if not tests:
        return ["[C16] no fixture test to rerun"]
    nodes = [f"tests/{spec}" for spec in tests]
    env = _child_environment({"PYTHONDONTWRITEBYTECODE": "1"})
    completed = _run_process([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                              "--no-header", *nodes], cwd=Path(root), what="The fixture rerun", env=env)
    if completed.returncode == 0:
        return []
    tail = [line for line in completed.stdout.decode("utf-8", "replace").splitlines() if line.strip()][-3:]
    return [f"[C16] the bound refusal and decision fixtures do not pass at the link's commit ({' | '.join(tail)[:400]})"]


# ---------------------------------------------------------------------------
# Authentication


def _anchor_problems(tree_pins: Mapping[str, str], links: Sequence[_Link], freeze1: Mapping[str, Any]) -> list[str]:
    """The code at the commit of ``links[-1]`` is the code the authenticated records pin (before it is run)."""
    if links[-1].sequence == 1:
        expected = _link1_pins(freeze1, links[0].document)
        return [f"[C4] {path} at link 1's commit differs from its Freeze-1 or S1-amendment pin"
                for path, digest in sorted(expected.items()) if tree_pins.get(path) != digest]
    scientific, analysis = _split_pins(_record_composite(links, freeze1))
    return [f"[C4] at link {links[-1].sequence}'s commit: {problem[5:]}" for problem in live_source_problems(
        tree_pins, scientific, analysis, _execution_files(links[-1].document))]


def _authenticate_link1(repo: Path, link: _Link, freeze1: Mapping[str, Any], require_published: bool) -> dict[str, Any]:
    """Link 1 by ``freeze.verify_s1_amendment`` as published at its commit, in a sparse checkout of that commit."""
    _refuse("[C4] The code at link 1's commit is not the code its records pin.",
            _anchor_problems(_commit_tree_pins(repo, link.commit), [link], freeze1))
    with _checkout(repo, link.commit) as checkout:
        result = _run_fresh(checkout, _S1_PROBE, {
            "path": str(checkout / link.path), "freeze1": rules.FREEZE1_SHA256, "repo": str(checkout),
            "require_published": require_published, "chain": [link.sha256]},
            what=f"[C4] Link 1 at its source commit {link.commit[:12]}")
    _require(isinstance(result, dict) and result.get("document_sha256") == link.sha256
             and result.get("amendment_chain") == [link.sha256],
             "[C4] The S1-amendment verifier at link 1's commit authenticated another document.")
    return result


def _authenticate_in_checkout(repo: Path, links: Sequence[_Link], freeze1: Mapping[str, Any],
                              require_published: bool) -> dict[str, Any]:
    """The last of ``links`` by the ``verify_chain`` published at its commit, in a sparse checkout of that commit,
    after the code there is matched with the records' pins; the result is cross-checked against the records and
    its scientific identity recomputed."""
    target = links[-1]
    _require(target.commit is not None, f"[C4] Link {target.sequence} has no source commit to authenticate it at.")
    _refuse(f"[C4] The code at link {target.sequence}'s commit is not the code its records pin.",
            _anchor_problems(_commit_tree_pins(repo, target.commit), links, freeze1))
    chain = [link.sha256 for link in links]
    extra = FIXTURE_RERUN_SPARSE_PATHS if target.document.get("change_class") == rules.CHANGE_ANALYSIS_COMPLETION \
        else ()
    with _checkout(repo, target.commit, extra_paths=extra) as checkout:
        result = _run_fresh(checkout, _CHAIN_PROBE, {
            "tip": str(checkout / target.path), "freeze1": rules.FREEZE1_SHA256, "freeze1b": rules.FREEZE1B_SHA256,
            "repo": str(checkout), "require_published": require_published, "chain": chain,
            "private_root": str(_private_root(repo))},
            what=f"[C4] Link {target.sequence} at its source commit {target.commit[:12]}")
    _require(isinstance(result, dict) and set(result) == set(RESULT_KEYS) and result.get("chain") == chain
             and result.get("tip_sequence") == target.sequence,
             f"[C4] The verifier published at link {target.sequence}'s commit returned another chain.")
    _cross_check(result, links, freeze1)
    return result


def _link_entries(repo: Path, links: Sequence[_Link], published: Mapping[int, bool]) -> list[dict[str, Any]]:
    return [{"sequence": link.sequence, "amendment_id": link.document.get("amendment_id"),
             "format": link.document.get("format"), "path": link.path, "sha256": link.sha256,
             "source_commit": link.commit, "source_commit_time": _commit_time(repo, link.commit),
             "change_class": link.document.get("change_class"), "scientific": _is_scientific_link(link),
             "published": bool(published.get(link.sequence))} for link in links]


def _publication(repo: Path, links: Sequence[_Link]) -> tuple[dict[int, bool], list[str]]:
    """C12: each link's source commit is an ancestor of ``origin/main`` and HEAD is published too. Link 1's log at
    its fixed commit holds its SHA-256. A link of sequence n >= 2 has a source commit only as the commit that entered
    its strict log line (``_introducing_commit``), and its own authentication at that commit applies the log rule
    there (``amendment_rules.link_log_problems``), so the line is not looked up a second time here."""
    origin, head = _origin(repo), _head(repo)
    published, problems = {}, []
    for link in links:
        reasons = []
        if link.commit is None:
            reasons.append(f"[C12] link {link.sequence} is not committed with its log line in the campaign log")
        else:
            if link.sequence == 1:
                log = freeze._show(repo, link.commit, freeze.LOG_PATH)
                if log is None or not freeze._hash_in_text(link.sha256, log):
                    reasons.append("[C12] the campaign log at link 1's source commit does not hold its SHA-256")
            if origin is None or not freeze._is_ancestor(repo, link.commit, origin):
                reasons.append(f"[C12] link {link.sequence}'s source commit is not an ancestor of "
                               f"{freeze.ORIGIN_MAIN_REF}")
        published[link.sequence] = not reasons
        problems += reasons
    if head is None or origin is None or not freeze._is_ancestor(repo, head, origin):
        problems.append(f"[C12] HEAD is not published on {freeze.ORIGIN_MAIN_REF}")
    return published, problems


def _c7_problems(repo: Path, record: Mapping[str, Any], tracked: Sequence[str], execution: Sequence[str]
                 ) -> tuple[list[str], dict[str, Any]]:
    """C7 at a tree: the partition, private names, the import probe, the analysis channels and the static import
    closure (one parse of every tracked Python file)."""
    problems, part = _partition_problems(repo, record, tracked, execution)
    problems += _private_import_problems(repo, execution, tracked)
    problems += import_isolation_problems(repo, execution, tracked)
    scan = source_scan(repo, tracked)
    problems += analysis_channel_problems(repo, part["scientific_files"], scan=scan)
    problems += execution_closure_problems(repo, tracked, execution, scan=scan)
    return problems, part


def _policy_probe(root: Path) -> tuple[dict[str, str], list[str]]:
    try:
        result = _run_fresh(root, _POLICY_PROBE, {}, what="The rule-policy probe")
    except AmendmentChainError as exc:
        return {}, [f"[C9] the rule policies cannot be computed in a fresh interpreter ({exc})"]
    if not (isinstance(result, dict) and all(_is_sha256(value) for value in result.values())):
        return {}, ["[C9] the rule-policy probe returned malformed policies"]
    return dict(result), []


def _authenticate_here(repo: Path, links: Sequence[_Link], freeze1: Mapping[str, Any],
                       require_published: bool) -> dict[str, Any]:
    """The tip link n >= 2 in a repository that is at its source commit (or, for an unpublished draft tip, in its
    working tree), with the code published there; link n-1 is authenticated at its own commit."""
    tip, record = links[-1], links[-1].document
    sequence = tip.sequence
    read = _reader(repo, tip.commit)
    private_read = _private_reader(_private_root(repo))
    is_tracked = _tracked_check(repo, tip.commit)
    before = links[-2]
    if sequence == rules.AMENDMENT2_SEQUENCE:
        _authenticate_link1(repo, links[0], freeze1, require_published)
        previous_pins = _link1_pins(freeze1, links[0].document)
        previous: dict[str, Any] = {"fit_chain": [links[0].sha256]}
    else:
        inner = _authenticate_in_checkout(repo, links[:-1], freeze1, require_published)
        previous_pins = {**inner["composite_scientific_pins"], **inner["analysis_pins"]}
        previous = {"fit_chain": inner["fit_chain"], "pinned_analysis_functions": inner["pinned_analysis_functions"],
                    "scientific_identity_sha256": inner["scientific_identity_sha256"],
                    "live_constants_sha256": before.document.get("live_constants_sha256"),
                    "rule_policy_sha256": before.document.get("rule_policy_sha256"),
                    **_fixture_state(_reader(repo, before.commit), before.document)}
    problems: list[str] = []
    # C2: canonical bytes and the builder's rendering at the link's commit.
    if tip.data != freeze.canonical_bytes(record):
        problems.append(f"[C2] link {sequence} is not the canonical serialisation of its builder")
    rendering = read(rules.rendering_path(tip.path))
    try:
        module, name = RECORD_RENDERER.split(":")
        rendered = getattr(importlib.import_module(module), name)(record).encode("utf-8")
    except Exception as exc:  # the builder refuses to render the record
        problems.append(f"[C2] the builder cannot render link {sequence} ({type(exc).__name__}: {exc})")
    else:
        if rendering != rendered:
            problems.append(f"[C2] the rendering of link {sequence} differs from the builder's rendering of it")
    # C7 at the link's commit.
    execution = _execution_files(record)
    tracked, tree_pins, missing = _tracked_pins(repo)
    problems += missing
    c7, part = _c7_problems(repo, record, tracked, execution)
    problems += c7
    # C5.
    unchanged = record.get("unchanged_scientific_files")
    if not (isinstance(unchanged, list) and all(isinstance(item, str) for item in unchanged)):
        problems.append("[C5] unchanged_scientific_files is not a list of files and folders")
        unchanged = []
    scope, composite = scope_problems(previous_pins, tree_pins, record.get("source_changes"), execution, unchanged,
                                      sequence=sequence)
    problems += scope
    parent, head = record.get("source_parent_commit"), _head(repo)
    if not (_is_commit(parent) and head is not None and freeze._resolve_commit(repo, parent) == parent
            and freeze._is_ancestor(repo, parent, head)):
        problems.append(f"[C5] the source parent commit of link {sequence} is not an ancestor of its commit")
    # C8, C9, pins and live constants.
    problems += evidence_problems(read, record, private_read=private_read, tracked=is_tracked)
    live_policies, policy_probe_problems = _policy_probe(repo)
    problems += policy_probe_problems
    problems += policy_problems(record, {**rules.rule_policy_sha256(include_analysis=False), **live_policies})
    exec_problems, exec_pins = _function_pin_problems(repo, record.get("pinned_execution_functions"),
                                                      rules.PINNED_EXECUTION_FUNCTIONS, execution, "execution", "C6")
    analysis_problems, analysis_function_pins = _function_pin_problems(
        repo, record.get("pinned_analysis_functions"), rules.PINNED_ANALYSIS_FUNCTIONS, rules.POST_FIT_ANALYSIS_FILES,
        "analysis", "C7")
    problems += exec_problems + analysis_problems
    live_constants = rules.live_constants_sha256()
    if record.get("live_constants_sha256") != live_constants:
        problems.append("[C9] live_constants_sha256 differs from freeze.live_constants() at the link's commit")
    # C10, C11.
    plan_problems, plans = plan_records(record, freeze1)
    receipt_problems, dry_run = dry_run_problems(lambda path: private_read(path)[0], record, freeze1,
                                                 tracked=is_tracked)
    problems += plan_problems + receipt_problems
    if _is_commit(dry_run["source_commit"]):
        if head is None or not freeze._is_ancestor(repo, dry_run["source_commit"], head):
            problems.append("[C11] the dry-run source commit is not an ancestor of the link's commit")
        else:
            check = rules.r_cmd_v2_check(dry_run["source_commit"], repo, plans, execution)
            problems += [f"[C11] {problem}" for problem in check["problems"]]
    # C13 and C16.
    scientific, analysis = _split_pins(composite)
    fit_chain = _fit_chain(links)
    identity = None
    if sequence == rules.AMENDMENT2_SEQUENCE:
        problems += data_state_problems(record)
    else:
        problems += fut_evidence_problems(read, record, [link.sha256 for link in links[:-1]])
        exists, stopped, flag_problems = futility_flags([link.document for link in links[2:]])
        problems += flag_problems
        problems += later_link_problems(record, changed_files=sorted(record.get("source_changes") or {}),
                                        execution_files=execution, fut_record_exists=exists,
                                        stop_futility_exists=stopped, previous=before.document)
    if not problems:
        identity = rules.scientific_identity(identity_record(
            fit_chain=fit_chain, composite_scientific_pins=scientific, pinned_execution_functions=exec_pins,
            plans=plans, freeze1_views=freeze1.get("views") or {}, dry_run_evidence=dry_run,
            deferred_input_specifications=record.get("deferred_bindings")))
        if sequence > rules.AMENDMENT2_SEQUENCE and record.get("change_class") == rules.CHANGE_ANALYSIS_COMPLETION:
            problems += analysis_completion_problems(previous, {
                "fit_chain": fit_chain, "pinned_analysis_functions": analysis_function_pins,
                "scientific_identity_sha256": identity, "live_constants_sha256": live_constants,
                "rule_policy_sha256": record.get("rule_policy_sha256"), **_fixture_state(read, record)})
            if read(rules.FREEZE2_PATH) is not None:
                problems.append("[C16] an analysis completion is published only before Freeze 2")
            tests, test_problems = fixture_tests(read, record)
            before_tests, _ = fixture_tests(_reader(repo, before.commit), before.document)
            current_pins, current_problems = named_test_pins(read, tests)
            previous_test_pins, previous_problems = named_test_pins(_reader(repo, before.commit), before_tests)
            problems += test_problems + current_problems + previous_problems
            problems += [f"[C16] the named test {spec} changed in an analysis completion"
                         for spec, digest in sorted(previous_test_pins.items()) if current_pins.get(spec) != digest]
            if not problems:
                problems += fixture_rerun_problems(repo, tests)
    _refuse(f"Link {sequence} does not authenticate at its source commit.", problems)
    return {
        "chain": [link.sha256 for link in links], "fit_chain": fit_chain, "tip_sequence": sequence,
        "composite_scientific_pins": scientific, "analysis_pins": analysis, "execution_file_set": list(execution),
        "post_fit_analysis_files": list(rules.POST_FIT_ANALYSIS_FILES), "pinned_execution_functions": exec_pins,
        "pinned_analysis_functions": analysis_function_pins, "policies": rules.policy_records(),
        "plan_records": plans, "dry_run_evidence": dry_run, "scientific_identity_sha256": identity,
    }


def _cross_check(state: Mapping[str, Any], links: Sequence[_Link], freeze1: Mapping[str, Any]) -> None:
    """The values of the verifier published at a link's commit agree with the authenticated records, and its
    scientific identity is present and recomputes."""
    record = links[-1].document
    scientific, analysis = _split_pins(_record_composite(links, freeze1))
    _, plans = plan_records(record, freeze1)
    dry = record.get("dry_runs") if isinstance(record.get("dry_runs"), Mapping) else {}
    receipts = dry.get("receipts") if isinstance(dry.get("receipts"), Mapping) else {}
    expected = {"fit_chain": _fit_chain(links), "composite_scientific_pins": scientific, "analysis_pins": analysis,
                "execution_file_set": _execution_files(record), "plan_records": plans,
                "post_fit_analysis_files": list(rules.POST_FIT_ANALYSIS_FILES),
                "pinned_execution_functions": dict(record.get("pinned_execution_functions") or {}),
                "pinned_analysis_functions": dict(record.get("pinned_analysis_functions") or {}),
                "dry_run_evidence": {"source_commit": dry.get("source_commit"),
                                     "receipts": {fold: {"path": item.get("path"), "sha256": item.get("sha256"),
                                                         "view_manifest_sha256": _frozen_view_manifests(freeze1).get(
                                                             fold)}
                                                  for fold, item in sorted(receipts.items())
                                                  if isinstance(item, Mapping)},
                                     "in_view_receipts_unchanged": _in_view_receipts(freeze1)}}
    differ = sorted(key for key, value in expected.items() if state.get(key) != value)
    hashes = record.get("rule_policy_sha256") if isinstance(record.get("rule_policy_sha256"), Mapping) else {}
    policies = state.get("policies")
    scientific_rules = {rule_id for rule_id, item in rules.RULE_POLICY_FUNCTIONS.items() if not item["analysis"]}
    if not (isinstance(policies, Mapping) and set(policies) == scientific_rules and all(
            isinstance(item, Mapping) and item.get("rule_id") == rule_id
            and rules.sha256_canonical(item.get("policy")) == hashes.get(rule_id)
            for rule_id, item in policies.items())):
        differ.append("policies")
    _require(not differ, f"[C4] The verifier published at link {links[-1].sequence}'s commit returned values that "
                         f"differ from the authenticated records: {differ}.")
    identity = state.get("scientific_identity_sha256")
    _require(_is_sha256(identity) and identity == _identity(state, freeze1, record),
             f"[C14] The verifier published at link {links[-1].sequence}'s commit returned no scientific identity or "
             "another one than its records give.")


def _live_problems(repo: Path, links: Sequence[_Link], state: Mapping[str, Any], *, probes: bool) -> list[str]:
    """C6, C7, C8 (bound files, when HEAD is past the tip's commit), C10 and C11 at the tip, on the live tree."""
    execution = list(state["execution_file_set"])
    tracked, live, problems = _tracked_pins(repo)
    problems += live_source_problems(live, state["composite_scientific_pins"], state["analysis_pins"], execution)
    for link in links:
        if _working_tree_bytes(repo, link.path) != link.data:
            problems.append(f"[C2] the live file of link {link.sequence} differs from its published bytes")
        rendering = rules.rendering_path(link.path)
        if link.commit is not None and _working_tree_bytes(repo, rendering) != freeze._show_bytes(repo, link.commit,
                                                                                                  rendering):
            problems.append(f"[C2] the live rendering of link {link.sequence} differs from its published rendering")
    if links[-1].sequence == 1:
        return problems
    record = links[-1].document
    partition_problems, part = _partition_problems(repo, record, tracked, execution)
    problems += partition_problems + _private_import_problems(repo, execution, tracked)
    exec_problems, _ = _function_pin_problems(repo, state["pinned_execution_functions"],
                                              rules.PINNED_EXECUTION_FUNCTIONS, execution, "execution", "C6")
    analysis_problems, _ = _function_pin_problems(repo, state["pinned_analysis_functions"],
                                                  rules.PINNED_ANALYSIS_FUNCTIONS, rules.POST_FIT_ANALYSIS_FILES,
                                                  "analysis", "C6")
    problems += exec_problems + analysis_problems
    if probes:
        # HEAD is past the tip's commit: the files the tip binds must still hash on the live tree (bound files under
        # docs/ and tests/ are free for r_cmd_v2). At the tip's commit with a clean tree C8 has proved it already.
        problems += binding_hash_problems(lambda path: _working_tree_bytes(repo, path), record,
                                          where="on the live tree", private_read=_private_reader(_private_root(repo)),
                                          tracked=_tracked_check(repo, None))
        problems += import_isolation_problems(repo, execution, tracked)
        scan = source_scan(repo, tracked)
        problems += analysis_channel_problems(repo, part["scientific_files"], scan=scan)
        problems += execution_closure_problems(repo, tracked, execution, scan=scan)
        dry_run = state["dry_run_evidence"] or {}
        check = rules.r_cmd_v2_check(dry_run.get("source_commit"), repo, state["plan_records"], execution)
        problems += [f"[C11] {problem}" for problem in check["problems"]]
    return problems


def _authenticates_here(repo: Path, tip: _Link, require_published: bool) -> bool:
    if tip.commit is None:
        return not require_published
    return _head(repo) == tip.commit and provenance.git_state(repo).get("clean") is True


def _expected_chain(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    _require(isinstance(value, (list, tuple)) and all(_is_sha256(item) for item in value),
             "expected_chain must be a list of SHA-256 values.")
    return list(value)


def verify_chain(tip_path: str | os.PathLike, *, expected_freeze1_sha256: str, expected_freeze1b_sha256: str,
                 repo: str | os.PathLike, require_published: bool = True,
                 expected_chain: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Authenticate the amendment chain that ends at ``tip_path`` in the full clone ``repo`` (checks C1-C16) and
    return exactly ``RESULT_KEYS``. ``require_published=False`` is for draft review and smoke use only: it never makes
    a chain eligible for real selection, futility, scoring or execution (``published`` is then reported, not
    required). A partial clone is refused at once: verify on the workstation's full clone and check the receipt
    there with ``check_chain_receipt``."""
    with _verification():
        try:
            return _verify_chain(tip_path, expected_freeze1_sha256=expected_freeze1_sha256,
                                 expected_freeze1b_sha256=expected_freeze1b_sha256, repo=repo,
                                 require_published=require_published, expected_chain=expected_chain)
        except AmendmentChainError:
            raise
        except Exception as exc:  # fail closed: any other failure refuses the chain
            raise AmendmentChainError(f"The chain does not verify ({type(exc).__name__}: {exc}).") from exc


def _verify_chain(tip_path: Any, *, expected_freeze1_sha256: Any, expected_freeze1b_sha256: Any, repo: Any,
                  require_published: Any, expected_chain: Any) -> dict[str, Any]:
    _require(isinstance(require_published, bool), "require_published must be a boolean.")
    expected = _expected_chain(expected_chain)
    repo = Path(os.path.realpath(os.fspath(repo)))
    problem = partial_clone_problem(repo)
    _require(problem is None, f"verify_chain needs a full clone, and {repo} is a partial clone ({problem}). Verify on "
                              "the workstation's full clone, write the receipt with verify-amendment-chain and check "
                              "it here with amendment_chain.check_chain_receipt.")
    try:
        _absolute, tip = freeze._amendment_repo_file(tip_path, repo, "The chain tip", document=True)
    except freeze.Freeze1Error as exc:
        raise AmendmentChainError(str(exc)) from None
    with _private_files_of(repo):
        return _verify_located(repo, tip, tip_path, expected_freeze1_sha256=expected_freeze1_sha256,
                               expected_freeze1b_sha256=expected_freeze1b_sha256,
                               require_published=require_published, expected=expected)


def _verify_located(repo: Path, tip: str, tip_path: Any, *, expected_freeze1_sha256: Any,
                    expected_freeze1b_sha256: Any, require_published: bool, expected: Optional[list[str]]
                    ) -> dict[str, Any]:
    if not _runs_from(repo):
        own, foreign = _code_sha256(Path(__file__).resolve().parents[2]), _code_sha256(repo)
        _require(not require_published or own == foreign,
                 f"[C4] {repo} holds other verifier code than this verifier ("
                 f"{sorted(path for path in RECEIPT_CODE_FILES if own[path] != foreign[path])}); a real verification "
                 "runs inside the verified repository.")
        result = _run_fresh(repo, _CHAIN_PROBE, {
            "tip": str(repo / tip), "freeze1": expected_freeze1_sha256, "freeze1b": expected_freeze1b_sha256,
            "repo": str(repo), "require_published": require_published, "chain": expected,
            "private_root": str(_private_root(repo))},
            what="The verifier of the verified repository")
        _require(isinstance(result, dict) and set(result) == set(RESULT_KEYS),
                 "The verifier of the verified repository returned another result shape.")
        _require(not (_is_int(result.get("tip_sequence")) and result["tip_sequence"] >= rules.AMENDMENT2_SEQUENCE)
                 or _is_sha256(result.get("scientific_identity_sha256")),
                 "[C14] The verifier of the verified repository returned no scientific identity.")
        return result
    _require(expected_freeze1_sha256 == rules.FREEZE1_SHA256,
             "The chain starts from another Freeze 1 than amendment_rules.FREEZE1_SHA256.")
    _require(expected_freeze1b_sha256 == rules.FREEZE1B_SHA256,
             "The chain names another Freeze 1b than amendment_rules.FREEZE1B_SHA256.")
    return _verify_here(repo, tip, require_published, expected)


@contextmanager
def _private_files_of(repo: Path) -> Iterator[None]:
    """Private bindings are read from ``repo``'s working tree, or, in a nested verification, from the root that the
    parent passed to the chain probe (``_probe_private_root``); nothing is taken from the environment."""
    _PRIVATE_ROOTS.append(_PRIVATE_ROOTS[-1] if _PRIVATE_ROOTS else Path(repo))
    try:
        yield
    finally:
        _PRIVATE_ROOTS.pop()


def _verify_here(repo: Path, tip: str, require_published: bool, expected_chain: Optional[list[str]]) -> dict[str, Any]:
    links = _collect_links(repo, tip)
    chain = [link.sha256 for link in links]
    _require(expected_chain is None or expected_chain == chain,
             "[C3] The chain differs from the expected chain (a fork, a gap or a reorder).")
    published, publication_problems = _publication(repo, links)
    if require_published:
        provenance.require_clean_tree(repo)
        _refuse("[C12] The amendment chain is not published; real use is refused.", publication_problems)
    _refuse("[C3] A link file was rewritten.", written_once_problems(repo, links))
    log = _working_tree_bytes(repo, freeze.LOG_PATH)
    _refuse("[C3] The campaign log at HEAD does not publish exactly this chain.", [
        f"[C3] {problem}" for problem in rules.link_log_problems(
            (log or b"").decode("utf-8", errors="replace"), chain,
            tip_logged=require_published or links[-1].commit is not None)])
    freeze1 = _freeze1(repo, rules.FREEZE1_SHA256)
    if links[-1].sequence == 1:
        result = _authenticate_link1(repo, links[0], freeze1, require_published)
        scientific, analysis = _split_pins(_link1_pins(freeze1, links[0].document))
        state: dict[str, Any] = {
            "chain": chain, "fit_chain": chain, "tip_sequence": 1, "composite_scientific_pins": scientific,
            "analysis_pins": analysis, "execution_file_set": [],
            "post_fit_analysis_files": list(rules.POST_FIT_ANALYSIS_FILES), "pinned_execution_functions": {},
            "pinned_analysis_functions": {},
            "policies": {result["s1_rule"]: {"rule_id": result["s1_rule"], "policy": result["policy"]}},
            "plan_records": {"v1": plan_records({}, freeze1)[1]["v1"], "v2": {}}, "dry_run_evidence": None,
            "scientific_identity_sha256": None}
        problems = _live_problems(repo, links, state, probes=False)
    elif _authenticates_here(repo, links[-1], require_published):
        state = _authenticate_here(repo, links, freeze1, require_published)
        problems = _live_problems(repo, links, state, probes=False)
    else:
        state = dict(_authenticate_in_checkout(repo, links, freeze1, require_published))
        problems = _live_problems(repo, links, state, probes=True)
    if links[-1].sequence >= rules.AMENDMENT2_SEQUENCE:
        _require(_is_sha256(state.get("scientific_identity_sha256")),
                 "[C14] A chain of two or more links has no scientific identity.")
    _refuse("[C6] The live tree does not match the authenticated chain tip.", problems)
    state["links"] = _link_entries(repo, links, published)
    state["published"] = not publication_problems
    return {key: state[key] for key in RESULT_KEYS}


# ---------------------------------------------------------------------------
# Chain receipts: verified once on the workstation's full clone, checked cheaply on a pod and per unit


def receipt_key(repo: Path, tip: str, *, require_published: bool, expected_chain: Optional[Sequence[str]]
                ) -> dict[str, Any]:
    """What a chain receipt is keyed by: the tip's SHA-256, HEAD, the SHA-256 of
    ``git status --porcelain=v1 --untracked-files=all``, ``origin/main``, the SHA-256 of ``RECEIPT_CODE_FILES``,
    ``require_published`` and ``expected_chain``."""
    data = _working_tree_bytes(repo, tip)
    return {"tip_sha256": _sha256(data) if data is not None else None, "head": _head(repo),
            "status_sha256": _sha256(_status_text(repo).encode("utf-8")), "origin_main": _origin(repo),
            "code_sha256": _code_sha256(repo), "require_published": bool(require_published),
            "expected_chain": None if expected_chain is None else list(expected_chain)}


def chain_receipt_record(tip_path: str | os.PathLike, *, expected_freeze1_sha256: str,
                         expected_freeze1b_sha256: str, repo: str | os.PathLike, require_published: bool = True,
                         expected_chain: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """``verify_chain`` on a full clone and its receipt: format ``CHAIN_RECEIPT_FORMAT``, the receipt key (taken
    before and after the verification, which must agree), the full result, the bytes of every link and rendering,
    every public file the tip binds (``bindings``, re-hashed by ``check_chain_receipt``), the private bindings that
    ``verify_chain`` checked in the working tree (``private_bindings``: where, path and SHA-256 only, never re-read
    on a pod) and ``receipt_sha256`` over the canonical rest. ``amendment2.write_chain_receipt`` writes it once;
    ``check_chain_receipt`` refuses one made without ``require_published``."""
    with _verification():
        try:
            repo = Path(os.path.realpath(os.fspath(repo)))
            _require(isinstance(require_published, bool), "require_published must be a boolean.")
            expected = _expected_chain(expected_chain)
            try:
                _absolute, tip = freeze._amendment_repo_file(tip_path, repo, "The chain tip", document=True)
            except freeze.Freeze1Error as exc:
                raise AmendmentChainError(str(exc)) from None
            before = receipt_key(repo, tip, require_published=require_published, expected_chain=expected)
            result = verify_chain(repo / tip, expected_freeze1_sha256=expected_freeze1_sha256,
                                  expected_freeze1b_sha256=expected_freeze1b_sha256, repo=repo,
                                  require_published=require_published, expected_chain=expected)
            after = receipt_key(repo, tip, require_published=require_published, expected_chain=expected)
            _require(before == after, "The repository changed while the chain was verified; no receipt is written.")
            links = []
            for link in result["links"]:
                rendering = rules.rendering_path(link["path"])
                rendered = _working_tree_bytes(repo, rendering)
                links.append({"sequence": link["sequence"], "path": link["path"], "sha256": link["sha256"],
                              "rendering": None if rendered is None else {"path": rendering,
                                                                          "sha256": _sha256(rendered)}})
            document = _parse(_working_tree_bytes(repo, tip) or b"", "The chain tip")
            bindings, private_bindings = _receipt_bindings(document, result["tip_sequence"])
            receipt = {
                "format": CHAIN_RECEIPT_FORMAT, "schema_version": CHAIN_RECEIPT_SCHEMA_VERSION, "verified": True,
                "published": bool(result["published"]), "draft": not require_published,
                "require_published": require_published, "expected_chain": expected,
                "chain": list(result["chain"]), "fit_chain": list(result["fit_chain"]),
                "tip_sequence": int(result["tip_sequence"]),
                "scientific_identity_sha256": result["scientific_identity_sha256"],
                "tip": {"path": tip, "sha256": before["tip_sha256"]}, "tip_sha256": before["tip_sha256"],
                "freeze1_sha256": rules.FREEZE1_SHA256, "freeze1b_sha256": rules.FREEZE1B_SHA256,
                "key": before, "links": links, "bindings": bindings, "private_bindings": private_bindings,
                "result": {key: result[key] for key in RESULT_KEYS},
            }
            receipt["receipt_sha256"] = rules.sha256_canonical(receipt)
            return receipt
        except AmendmentChainError:
            raise
        except Exception as exc:  # fail closed
            raise AmendmentChainError(f"No chain receipt ({type(exc).__name__}: {exc}).") from exc


def _receipt_bindings(document: Mapping[str, Any], tip_sequence: Any) -> tuple[list[dict], list[dict]]:
    """The tip's file bindings split into the public ones (re-hashed on the live tree by ``check_chain_receipt``)
    and the private ones (``amendment_rules.PRIVATE_ROLES``; recorded by path and SHA-256, never re-read)."""
    if tip_sequence == 1:
        return [], []
    entries = [{"where": where, "path": path, "sha256": digest} for where, path, digest in evidence_entries(document)]
    return ([entry for entry in entries if not rules.is_private_role(entry["where"])],
            [entry for entry in entries if rules.is_private_role(entry["where"])])


def check_chain_receipt(receipt_path: str | os.PathLike, *, repo: str | os.PathLike,
                        expected_chain: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """The ``verify_chain`` result of a published chain, read from its receipt (``chain_receipt_record``, written
    by ``verify-amendment-chain`` on the workstation's full clone) after a cheap re-check of ``repo``: HEAD is the
    receipt's HEAD, or a published descendant of it that changes only execution files; the tree is clean; every
    link and rendering has its receipt bytes; the live scientific and analysis pins and the pinned execution
    functions hold; the receipt's public and private bindings are exactly those of the tip link file; every public
    file the tip binds hashes; the verifier files and the receipt's own SHA-256 agree. Private bindings were checked
    by ``verify_chain`` on the workstation and are never read here (a pod holds none of them). Works in a blobless
    partial clone (it reads only the checked-out files, commits and trees). A receipt made without
    ``require_published`` is refused. Returns exactly ``RESULT_KEYS``."""
    with _verification():
        try:
            path = Path(os.fspath(receipt_path))
            _require(path.is_file() and not path.is_symlink(), f"The chain receipt {path.name} is missing or a link.")
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise AmendmentChainError(f"The chain receipt {path.name} is not JSON.") from None
            return _check_receipt(receipt, Path(os.path.realpath(os.fspath(repo))), _expected_chain(expected_chain))
        except AmendmentChainError:
            raise
        except Exception as exc:  # fail closed
            raise AmendmentChainError(f"The chain receipt does not check ({type(exc).__name__}: {exc}).") from exc


def check_chain_receipt_record(receipt: Mapping[str, Any], *, repo: str | os.PathLike,
                               expected_chain: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """``check_chain_receipt`` on a receipt held in memory (a consumer that verified once in this process)."""
    with _verification():
        try:
            return _check_receipt(receipt, Path(os.path.realpath(os.fspath(repo))), _expected_chain(expected_chain))
        except AmendmentChainError:
            raise
        except Exception as exc:  # fail closed
            raise AmendmentChainError(f"The chain receipt does not check ({type(exc).__name__}: {exc}).") from exc


def _check_receipt(receipt: Any, repo: Path, expected_chain: Optional[list[str]]) -> dict[str, Any]:
    _require(isinstance(receipt, Mapping) and receipt.get("format") == CHAIN_RECEIPT_FORMAT
             and receipt.get("schema_version") == CHAIN_RECEIPT_SCHEMA_VERSION,
             f"The chain receipt is not a {CHAIN_RECEIPT_FORMAT} receipt of schema {CHAIN_RECEIPT_SCHEMA_VERSION}.")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    _require(receipt.get("receipt_sha256") == rules.sha256_canonical(body),
             "The chain receipt does not hash to its receipt_sha256 (it was edited).")
    key, result = receipt.get("key"), receipt.get("result")
    _require(receipt.get("require_published") is True and receipt.get("draft") is False
             and isinstance(key, Mapping) and key.get("require_published") is True,
             "The chain receipt was made without require_published (draft review); real use is refused.")
    _require(receipt.get("verified") is True and receipt.get("published") is True and isinstance(result, Mapping)
             and set(result) == set(RESULT_KEYS) and result.get("published") is True,
             "The chain receipt is not the verified result of a published chain.")
    chain = result["chain"]
    _require(isinstance(chain, list) and chain and receipt.get("chain") == chain
             and receipt.get("fit_chain") == result["fit_chain"]
             and receipt.get("scientific_identity_sha256") == result["scientific_identity_sha256"]
             and receipt.get("freeze1_sha256") == rules.FREEZE1_SHA256
             and receipt.get("freeze1b_sha256") == rules.FREEZE1B_SHA256,
             "The chain receipt names another chain, identity or Freeze 1 than its result.")
    _require(expected_chain is None or expected_chain == chain,
             "[C3] The chain receipt's chain differs from the expected chain.")
    _require(key.get("expected_chain") is None or key["expected_chain"] == chain,
             "[C3] The chain receipt's chain differs from the chain it was verified against.")
    _require(len(chain) < rules.AMENDMENT2_SEQUENCE or _is_sha256(result["scientific_identity_sha256"]),
             "[C14] The chain receipt carries no scientific identity.")
    problems: list[str] = []
    live_code = _code_sha256(repo)
    problems += [f"[C4] the verifier file {path} differs from the one that wrote the receipt"
                 for path in RECEIPT_CODE_FILES if live_code.get(path) != (key.get("code_sha256") or {}).get(path)]
    head, origin = _head(repo), _origin(repo)
    execution = set(result["execution_file_set"])
    if head is None:
        problems.append("[C12] the repository has no HEAD")
    elif head != key.get("head"):
        base = key.get("head")
        if not (_is_commit(base) and freeze._resolve_commit(repo, base) == base
                and freeze._is_ancestor(repo, base, head)):
            problems.append("[C12] HEAD is neither the receipt's HEAD nor a descendant of it")
        else:
            changed = _git_bytes(repo, "diff", "--name-only", "--no-renames", "-z", base, head,
                                 what="git diff since the receipt").decode("utf-8").split("\0")
            others = sorted(path for path in changed if path and path not in execution)
            problems += [f"[C12] {path} changed after the receipt and is not an execution file" for path in others[:20]]
            if origin is None or not freeze._is_ancestor(repo, head, origin):
                problems.append(f"[C12] HEAD is not published on {freeze.ORIGIN_MAIN_REF}")
    if _status_text(repo).strip():
        problems.append("[C12] the tree is not clean")
    for link in receipt.get("links") or []:
        data = _working_tree_bytes(repo, link.get("path")) if isinstance(link, Mapping) else None
        if data is None or _sha256(data) != link.get("sha256") or link.get("sha256") not in chain:
            problems.append(f"[C2] the live file of link {link.get('sequence') if isinstance(link, Mapping) else link} "
                            "differs from its receipt")
            continue
        rendering = link.get("rendering")
        if rendering is not None:
            rendered = _working_tree_bytes(repo, rendering.get("path")) if isinstance(rendering, Mapping) else None
            if rendered is None or _sha256(rendered) != rendering.get("sha256"):
                problems.append(f"[C2] the live rendering of link {link.get('sequence')} differs from its receipt")
    _require(len(receipt.get("links") or []) == len(chain), "The chain receipt does not list every link.")
    tracked, live, missing = _tracked_pins(repo)
    problems += missing
    problems += live_source_problems(live, result["composite_scientific_pins"], result["analysis_pins"], execution)
    if len(chain) >= rules.AMENDMENT2_SEQUENCE:
        exec_problems, _ = _function_pin_problems(repo, result["pinned_execution_functions"],
                                                  rules.PINNED_EXECUTION_FUNCTIONS, execution, "execution", "C6")
        problems += exec_problems
        bindings, private_bindings = receipt.get("bindings"), receipt.get("private_bindings")
        if not isinstance(bindings, list) or not isinstance(private_bindings, list):
            problems.append("[C8] the chain receipt lists no bound files")
            bindings, private_bindings = [], []
        tip_item = receipt.get("tip")
        tip_data = _working_tree_bytes(repo, tip_item.get("path")) if isinstance(tip_item, Mapping) else None
        if tip_data is None or _sha256(tip_data) != receipt.get("tip_sha256") or _sha256(tip_data) != chain[-1]:
            problems.append("[C2] the live tip link file differs from the receipt's tip")
        else:
            expected_bindings = _receipt_bindings(_parse(tip_data, "The chain tip"), len(chain))
            if [bindings, private_bindings] != list(expected_bindings):
                problems.append("[C8] the receipt's public and private bindings differ from those of the tip link")
        for item in private_bindings:
            if not (isinstance(item, Mapping) and rules.is_private_role(item.get("where"))
                    and rules.is_private_path(item.get("path")) and _is_sha256(item.get("sha256"))):
                label = item.get("where") if isinstance(item, Mapping) else repr(item)
                problems.append(f"[C8] the receipt names a malformed private binding {label}")
        for item in bindings:
            data = _working_tree_bytes(repo, item.get("path")) if isinstance(item, Mapping) else None
            if data is None or _sha256(data) != item.get("sha256"):
                problems.append(f"[C8] {item.get('where') if isinstance(item, Mapping) else item}: the bound file does "
                                "not hash on the live tree")
    _refuse("The chain receipt does not hold for this repository.", problems)
    return {name: result[name] for name in RESULT_KEYS}


# ---------------------------------------------------------------------------
# Consumers


def _utc(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def record_head(record: Mapping[str, Any]) -> Optional[str]:
    """The commit a record was made at: ``git_head``, or ``provenance.git.head`` (``provenance_record``)."""
    value = record.get("git_head")
    if value is None:
        value = (((record.get("provenance") or {}).get("git") or {}).get("head")
                 if isinstance(record.get("provenance"), Mapping) else None)
    return value if _is_commit(value) else None


def record_precedence_problems(verified: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *,
                               repo: str | os.PathLike, replay: bool = False) -> list[str]:
    """C15: every record presented under ``verified`` carries its fit chain and was made at the commit that
    published the chain's last scientific link or at a descendant of it, and that commit is published on
    ``origin/main`` (git ancestry, never a clock); as a secondary check its ``created_utc`` (or
    ``provenance.created_utc``) is later than that commit's time. A record bound to a shorter prefix is accepted
    only in replay mode; records of different chains are never mixed."""
    repo = Path(os.path.realpath(os.fspath(repo)))
    with _verification():
        fit_chain = list(verified.get("fit_chain") or [])
        last = next((link for link in verified.get("links") or []
                     if fit_chain and link.get("sha256") == fit_chain[-1]), None)
        publishing = last.get("source_commit") if last else None
        published_at = _utc(last.get("source_commit_time")) if last else None
        origin = _origin(repo)
        problems = []
        chains = [list(record.get("fit_chain", record.get("freeze1_amendment_chain")) or []) for record in records]
        if len({tuple(chain) for chain in chains}) > 1:
            problems.append("[C15] records of different fit chains are mixed")
        for index, (record, chain) in enumerate(zip(records, chains)):
            if chain == fit_chain and fit_chain:
                head = record_head(record)
                created = _utc(record.get("created_utc", (record.get("provenance") or {}).get("created_utc")
                                          if isinstance(record.get("provenance"), Mapping) else None))
                if not _is_commit(publishing) or published_at is None:
                    problems.append("[C15] the chain's last scientific link has no publishing commit")
                elif head is None or freeze._resolve_commit(repo, head) != head:
                    problems.append(f"[C15] record {index} names no commit of this repository it was made at")
                elif not freeze._is_ancestor(repo, publishing, head):
                    problems.append(f"[C15] record {index} was made before the commit that published the chain's last "
                                    "scientific link")
                elif origin is None or not freeze._is_ancestor(repo, head, origin):
                    problems.append(f"[C15] record {index} was made at a commit that is not published")
                elif created is None or created <= published_at:
                    problems.append(f"[C15] record {index} was created before the commit that published the chain's "
                                    "last scientific link")
            elif chain and fit_chain[:len(chain)] == chain:
                if not replay:
                    problems.append(f"[C15] record {index} is bound to a shorter prefix of the chain (replay only)")
            else:
                problems.append(f"[C15] record {index} carries another fit chain")
        return problems


def require_record_precedence(verified: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *,
                              repo: str | os.PathLike, replay: bool = False) -> None:
    _refuse("[C15] Records do not belong to the authenticated chain.",
            record_precedence_problems(verified, records, repo=repo, replay=replay))


def fold_dry_run_evidence(verified: Mapping[str, Any], fold: int) -> dict[str, Any]:
    """The fold's amendment-2 dry-run evidence that the per-unit view binding requires in addition to the Freeze-1
    in-view receipt (C11): the dry-run-v2 receipt binding ``{path, sha256, view_manifest_sha256}`` (the private
    receipt was read and checked by ``verify_chain`` on the workstation; a pod holds no copy), its source commit and
    the Freeze-1 in-view receipt SHA-256 the view must still hold. A consumer compares its view's manifest SHA-256
    and the in-view receipt bytes it reads against these values."""
    evidence = verified.get("dry_run_evidence")
    key = str(fold)
    receipt = (evidence.get("receipts") or {}).get(key) if isinstance(evidence, Mapping) else None
    in_view = (evidence.get("in_view_receipts_unchanged") or {}).get(key) if isinstance(evidence, Mapping) else None
    _require(isinstance(receipt, Mapping) and _is_sha256(receipt.get("sha256"))
             and _is_sha256(receipt.get("view_manifest_sha256")) and _is_sha256(in_view)
             and _is_commit(evidence.get("source_commit")),
             f"[C11] The chain holds no amendment-2 dry-run evidence for fold {key}; the view binding is refused.")
    return {"fold": int(fold), "source_commit": evidence["source_commit"], "receipt": dict(receipt),
            "in_view_receipt_sha256": in_view}


__all__: Sequence[str] = (
    "AmendmentChainError", "CHAIN_RECEIPT_FORMAT", "RESULT_KEYS", "analysis_channel_problems", "chain_receipt_record",
    "check_chain_receipt", "check_chain_receipt_record", "execution_closure_problems", "fold_dry_run_evidence",
    "identity_record", "import_isolation_problems", "partial_clone_problem", "record_head",
    "record_precedence_problems", "require_record_precedence", "verify_chain",
)
