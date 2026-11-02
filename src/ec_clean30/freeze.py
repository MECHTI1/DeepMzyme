"""Freeze-1 completeness checks, the Freeze-1 document, ``verify-freeze1`` and Freeze 1b (WP-9, WP-10; torch-free).

This module wraps ``provenance.build_freeze1_manifest`` and adds to its output; ``provenance``,
``commands`` and ``views`` stay unchanged (rule R-CMD). Only the campaign runner's handlers import
it: ``campaign``, ``commands``, ``views``, ``provenance`` and ``identity`` never do, and the R-CMD
check below proves that for every module added after the dry runs.

Real Freeze-1 inputs manifest (``manifest --kind freeze1 --plans-dir DIR``, P2-10 step 3):
- exactly the fold views 0-4 of one campaign root, each verified now with its final-layer chain
  means and a passed dry run of that exact view (``views.freeze1_view_record``);
- each view's inner membership has the SHA-256 and source name of the output its fold's
  assignment receipt records, and its certificate equals the copy under the overlay's
  ``identity/fold<k>``, whose SHA-256 the overlay manifest records. Nothing under
  ``<campaign root>/identity`` is read, and such a path is refused;
- one overlay ID and manifest SHA-256 across the five views, and a real overlay: its manifest and
  the cohort audit and R2 lists it records state ``smoke: false``;
- rule R-CMD (``r_cmd_check``): from each view's dry-run HEAD to HEAD, ``git diff --name-status``
  shows only the allowlist, the twelve pinned runner functions are unchanged (source from the first
  decorator line, as ``inspect.getsource`` gives it, and no other module-scope binding of the name,
  including one inside a top-level ``if``, ``try`` or ``with`` block), a fresh interpreter
  that imports the protected modules loads none of the files added since, and each fold's live
  plan equals its record in ``--plans-dir``. Any problem means: rerun the dry runs;
- ``file_modes_enforced`` per view, and the SHA-256 of every tracked ``src/`` and ``scripts/`` file.

Freeze-1 document (``manifest --kind freeze1-document``, P2-10 step 4): a canonical ``freeze1.json``
and its ``freeze1.md`` rendering (same stem), written once. It holds the decisions record, every
required decision answered (``REQUIRED_DECISION_IDS``), the live constants of the campaign code,
the plan, view, membership, overlay and code records of the inputs manifest, and the SHA-256 of
every bound receipt or record (label table, sequence map, transfer list, cohort audit, assignments,
R2 lists, simulation records, environment exports, ``--pin`` values). No content of a bound file is
copied, so the document holds no per-fold EC-class count or proportion. The transfer list is
required unless ``--no-transfer-list`` states that none was approved (decision A8); the R2 lists
receipt and the overlay must be real (non-smoke). Paths are relative (to the campaign root, or to
the repository for the decisions record) and are checked before they are read; an absolute path
anywhere (``file:`` URIs included), an unanswered decision, a placeholder token in the decisions
record (``<a name>``, ``[a name]``, ``{name}``, TBD) or a placeholder character in the
acknowledgement name (E6, the scorer's rule) is refused. ``amendments`` starts empty.

``verify-freeze1``: the document hashes to F, F appears in ``log.md`` at HEAD, the document is
committed at HEAD with these bytes, and the commit that introduced F into the log is an ancestor of
HEAD. Whether ``refs/remotes/origin/main`` contains that commit is recorded (nothing is fetched);
``--require-pushed`` fails otherwise.

Freeze 1b (``manifest --kind freeze1b``, P2-12): each fold's complement fasta, search and edges
receipts, its strata receipt and the cohort components receipt are re-verified with the identity
loaders and must record F; every identity receipt of the fold (inner and complement) must record the
identity code SHA-256 that Freeze 1 pins; ``--identity-head`` is an ancestor of HEAD, is named in
``log.md`` at HEAD, and the six R-ID files are unchanged from it to HEAD; a passed, pushed
``verify-freeze1`` receipt is required (and the check is repeated). Receipts are opened by their
exact names, never by listing a folder, and only counts are printed.

Runner wiring (``scripts/run_ec_clean30_campaign.py``): ``add_manifest_freeze_arguments`` on the
``manifest`` sub-parser (``MANIFEST_KINDS`` are its kinds); ``route_manifest(args)`` returns
``action_manifest_freeze1`` (every real ``freeze1``; it needs ``--plans-dir``),
``action_manifest_freeze1_document`` or ``action_manifest_freeze1b``, or ``None`` for the runner's
own path (``freeze2`` and the smoke ``freeze1``, unchanged); ``add_verify_freeze1_arguments`` and
``action_verify_freeze1`` for the top-level ``verify-freeze1`` action. Refusals raise
``Freeze1Error`` (a ``FreezeManifestError``, exit 2); a failed ``verify-freeze1`` returns 1 and
writes no receipt.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, is_dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence

from ec_clean30 import provenance
from ec_clean30.campaign import CAMPAIGN_ID, N_OUTER_FOLDS
from ec_clean30.provenance import FreezeManifestError, sha256_bytes, sha256_file


REPO_ROOT = provenance.REPO_ROOT
SCHEMA_VERSION = 1

FREEZE1_INPUTS_COMPLETENESS_FORMAT = "deepmzyme_ec_clean30_freeze1_inputs_completeness_v1"
FREEZE1_DOCUMENT_FORMAT = "deepmzyme_ec_clean30_freeze1_document_v1"
FREEZE1_DECISIONS_FORMAT = "deepmzyme_ec_clean30_freeze1_decisions_v1"
VERIFY_FREEZE1_FORMAT = "deepmzyme_ec_clean30_verify_freeze1_v1"
FREEZE1B_FORMAT = "deepmzyme_ec_clean30_freeze1b_v1"

KIND_FREEZE1 = "freeze1"
KIND_FREEZE1_DOCUMENT = "freeze1-document"
KIND_FREEZE1B = "freeze1b"
MANIFEST_KINDS: tuple[str, ...] = (KIND_FREEZE1, "freeze2", KIND_FREEZE1_DOCUMENT, KIND_FREEZE1B)
VERIFY_FREEZE1_HELP = ("Check that the Freeze-1 document hashes to F, that F is in the campaign log at HEAD and "
                       "that the commit that logged it is an ancestor of HEAD; record whether origin/main holds it.")

CAMPAIGN_DOCS_DIR = f"docs/campaigns/{CAMPAIGN_ID}"
LOG_PATH = f"{CAMPAIGN_DOCS_DIR}/log.md"
ORIGIN_MAIN_REF = "refs/remotes/origin/main"
RERUN_DRY_RUNS = "rerun the dry runs"

# Rule R-ID: these six files do not change from the P2-1 HEAD to Freeze 1b.
R_ID_FILES: tuple[str, ...] = (
    "src/ec_clean30/identity.py",
    "scripts/run_ec_clean30_identity.py",
    "src/audit_sequence_remoteness.py",
    "src/training/ec_classes.py",
    "src/training/ec_membership.py",
    "src/project_paths.py",
)
# The files every identity receipt hashes (identity.code_sha256).
IDENTITY_CODE_FILES: tuple[str, ...] = (
    "src/ec_clean30/identity.py",
    "src/audit_sequence_remoteness.py",
    "scripts/run_ec_clean30_identity.py",
)

# Rule R-CMD: what may differ between a dry run's HEAD and HEAD.
R_CMD_RUNNER = "scripts/run_ec_clean30_campaign.py"
R_CMD_ALLOWED_MODIFIED: frozenset[str] = frozenset({
    "src/ec_clean30/selection.py",
    "src/ec_clean30/stats.py",
    "src/ec_clean30/logit_adjustment.py",
    "src/ec_clean30/scoring.py",
    "src/ec_clean30/baseline.py",
    "src/ec_clean30/simulation.py",
    R_CMD_RUNNER,
    "scripts/run_ec_clean30_baseline.py",
    "scripts/run_clean_split30_comparator.py",
})
R_CMD_FREE_FOLDERS: tuple[str, ...] = ("tests/", "docs/", "tools/")
_ADDED_MODULE_RE = re.compile(r"^src/ec_clean30/[A-Za-z_][A-Za-z0-9_]*\.py$")
_ADDED_SCRIPT_RE = re.compile(r"^scripts/[A-Za-z0-9_][A-Za-z0-9_.-]*$")
# The runner functions whose source must not change after the dry runs (the dry-run command path).
PINNED_RUNNER_FUNCTIONS: tuple[str, ...] = (
    "_csv", "_git_for", "_refuse_paths", "_layout", "action_plan", "_build", "action_build_view",
    "action_smoke_view", "_load_view", "action_verify_view", "action_apply_view_modes", "action_dry_run",
)
PROTECTED_IMPORTERS: tuple[str, ...] = tuple(f"ec_clean30.{name}" for name in (
    "campaign", "commands", "views", "provenance", "identity"))
TRACKED_CODE_FOLDERS: tuple[str, ...] = ("src", "scripts")
IMPORT_CHECK_TIMEOUT_SECONDS = 300

PLAN_FILE_NAME = "fold{fold}.json"
PLAN_RECEIPT_SUFFIX = ".receipt.json"

# Files bound by SHA-256, by exact name below each folder argument (no folder is listed).
LABEL_FILES: Mapping[str, str] = {
    "label_table": "label_table/native_label_table.tsv",
    "label_table_receipt": "label_table/label_table_receipt.json",
    "sequence_map": "sequence_map/sequence_map.tsv",
    "sequence_map_receipt": "sequence_map/sequence_map_receipt.json",
}
# The transfer list (decision A8, P2-9d): both files are required unless --no-transfer-list (then neither may exist).
TRANSFER_LIST_FILES: Mapping[str, str] = {
    "transfer_list": "ec_transfer_list/ec_transfer_list.csv",
    "transfer_list_receipt": "ec_transfer_list/transfer_list_receipt.json",
}
TRANSFER_LIST_ABSENT = "no list supplied"
COHORT_FILES: Mapping[str, str] = {"cohort_audit_receipt": "cohort_audit_receipt.json"}
R2_FILES: Mapping[str, str] = {"r2_lists_receipt": "r2_lists_receipt.json"}
SIMULATION_FILES: Mapping[str, str] = {"sweep": "sweep.json", "sweep_receipt": "sweep_receipt.json",
                                       "summary": "summary.tsv"}
ENVIRONMENT_FILES: Mapping[str, str] = {"conda_explicit": "env_conda_explicit.txt",
                                        "pip_freeze": "env_pip_freeze.txt"}
OVERLAY_COPIES: Mapping[str, str] = {"cohort_audit_receipt": "cohort/cohort_audit_receipt.json",
                                     "r2_lists_receipt": "r2_lists/r2_lists_receipt.json"}
SIMULATION_RECEIPT_FORMAT = "deepmzyme_ec_clean30_simulation_sweep_receipt_v1"
SIMULATION_RECORD_FORMAT = "deepmzyme_ec_clean30_simulation_record_v1"
_SIMULATION_RECORD_RE = re.compile(r"^records/[A-Za-z0-9][A-Za-z0-9._+=-]*\.json$")

# --pin NAME=SHA256: the CLEAN harness identity (F6), the ESMC-300m weights and the MMseqs2 binary (G5).
REQUIRED_PINS: tuple[str, ...] = ("clean_patch_manifest", "clean_environment_lock", "esmc_300m_weights",
                                  "mmseqs2_binary")
CLEAN_PATCH_MANIFEST_PREFIX = "f331019d"
_PIN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_TOKEN_RE = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")

# Decision ids of the plan's section 4: 4.0 (decided; recorded as is) and the tables of rounds 1a, 1b and 2.
DECIDED_DECISION_IDS: tuple[str, ...] = (*(f"U{index}" for index in range(1, 15)), "U17")
ROUND_1A_CHOICE_IDS: tuple[str, ...] = ("M4", "M6", "M3", "M8", "M7", "M1", "M2", "M5")
ROUND_1A_CONFIRMATION_IDS: tuple[str, ...] = (
    "A3", "A4", "A7", "A9", "A10", "A11", "A12", "A13", "B6", "B8", "B9", "B10",
    *(f"C{index}" for index in range(1, 15)),
    "BP-D3", "BP item 20", "D7", "G4", "BP-D9", "Push", "Lock", "Counts", "Map run",
)
ROUND_1B_IDS: tuple[str, ...] = ("A1", "A2", "A5", "A6", "P2-9")
ROUND_2_IDS: tuple[str, ...] = (
    "A8", "B1", "B2", "B3", "B4", "B5", "B7", "B11", "B12",
    *(f"D{index}" for index in range(1, 7)),
    *(f"E{index}" for index in range(1, 11)),
    "BP-D7", "BP-D11",
    *(f"F{index}" for index in range(1, 10)),
    "G1", "G2", "G3", "G5", "G6", "G7", "G8", "G9",
)
AMENDMENT_DECISION_IDS: tuple[str, ...] = ("SA1", "SA1-c", "P2-L5", "P2-9-A1", "A8-a", "A8-b", "A8-c", "P2-9b")
# The hyphen distinguishes the final implementation choices from the original lever decisions C1-C14.
ROUND_2_IMPLEMENTATION_IDS: tuple[str, ...] = tuple(f"C-{index}" for index in range(1, 15))
REQUIRED_DECISION_IDS: tuple[str, ...] = (*DECIDED_DECISION_IDS, *ROUND_1A_CHOICE_IDS, *ROUND_1A_CONFIRMATION_IDS,
                                          *ROUND_1B_IDS, *ROUND_2_IDS, *AMENDMENT_DECISION_IDS,
                                          *ROUND_2_IMPLEMENTATION_IDS)
UNANSWERED_ANSWERS: frozenset[str] = frozenset({"", "tbd", "todo", "pending", "open", "unanswered", "?"})
# The scorer's placeholder rule (scoring._PLACEHOLDER_CHARACTERS; a test keeps the two equal). It applies to the
# decisions that become the scorer's acknowledgement name (E6), as the scorer applies it to that setting only.
PLACEHOLDER_CHARACTERS: frozenset[str] = frozenset("[]{}<>")
ACKNOWLEDGEMENT_DECISION_IDS: tuple[str, ...] = ("E6",)
# Placeholder tokens refused in every other text: <a name>, [a name], {name}, and TBD, TODO or FIXME as words.
# Sets, intervals and brace lists such as {-1, 0, 1}, (20,25] or {identity,cohort} are not placeholders.
_PLACEHOLDER_TOKEN_RE = re.compile(r"<[A-Za-z][A-Za-z0-9 _.'-]*>|\[[A-Za-z][A-Za-z _.'-]*\]|\{[A-Za-z_][A-Za-z0-9_]*\}"
                                   r"|\b(?:TBD|TODO|FIXME)\b")
# A path that starts at a root, a home folder or a drive letter (after a separator, quote, bracket or colon), and
# any file: URI. Other URLs (https://host/...) are not absolute paths.
_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s\"'`=(,;:\[{<])(?:~?/|[A-Za-z]:[\\/])[A-Za-z0-9._~-]|(?i:\bfile:/)")
# Keys that hold per-fold class numbers in the receipts the document binds; never copied.
CLASS_COUNT_KEYS: frozenset[str] = frozenset({
    "eligible_by_ec_class", "ec_class_counts", "inner_folds", "inner_folds_without_ec_class", "class_proportions",
    "proportions", "n_rows", "n_eligible", "per_fold", "counts_by_ec_class",
})


class Freeze1Error(FreezeManifestError):
    """A Freeze-1 or Freeze-1b record cannot be written from the given inputs; ``problems`` lists why."""

    def __init__(self, message: str, problems: Sequence[str] = ()) -> None:
        self.problems = list(problems)
        text = message if not self.problems else message + "\n  " + "\n  ".join(self.problems[:40])
        super().__init__(text)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Freeze1Error(message)


def _refuse_problems(message: str, problems: Sequence[str]) -> None:
    if problems:
        raise Freeze1Error(message, problems)


def _sha256_argument(value: Optional[str], what: str) -> str:
    text = str(value or "").strip().lower()
    _require(bool(_SHA256_RE.match(text)), f"{what} must be 64 hexadecimal characters.")
    return text


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def canonical_bytes(payload: Any) -> bytes:
    """The bytes ``provenance.write_new_json`` writes (sorted keys, two-space indent, no NaN, final newline)."""
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def jsonable(value: Any) -> Any:
    """A JSON-ready copy of a constant: mappings with string keys, lists for sequences and sets, text fractions."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, re.Pattern):
        return value.pattern
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    raise TypeError(f"Constant of type {type(value).__name__} has no JSON form.")


# ---------------------------------------------------------------------------
# Paths and git
# ---------------------------------------------------------------------------

def _real(path: str | os.PathLike) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_under(path: str | os.PathLike, root: str | os.PathLike) -> bool:
    path, root = _real(path), _real(root)
    return path == root or root in path.parents


def campaign_identity_problem(path: str | os.PathLike, campaign_root: str | os.PathLike, what: str) -> Optional[str]:
    """Why ``path`` may not be read before Freeze 1b (it lies under ``<campaign root>/identity``), or ``None``."""
    if _is_under(path, Path(campaign_root) / "identity"):
        return (f"{what} lies under <campaign root>/identity, which holds test-side content after Freeze 1; "
                "Freeze-1 inputs come from the overlay's identity copies")
    return None


def campaign_identity_name_problem(path: str | os.PathLike, what: str) -> Optional[str]:
    """Why ``path`` may not be read, judged by its name alone: a folder named ``identity`` directly below a folder
    named after the campaign ID (a real campaign root carries that name) is on its path, as given or resolved."""
    for candidate in (Path(os.path.abspath(os.fspath(path))), _real(path)):
        parts = candidate.parts
        if any(parts[index] == CAMPAIGN_ID and parts[index + 1] == "identity" for index in range(len(parts) - 1)):
            return (f"{what} lies under <campaign root>/identity, which holds test-side content after Freeze 1; "
                    "it is refused before it is read")
    return None


def campaign_relative(path: str | os.PathLike, campaign_root: str | os.PathLike, what: str) -> str:
    """``path`` relative to the campaign root (POSIX); refused outside it or under its identity folder."""
    problem = campaign_identity_problem(path, campaign_root, what)
    _require(problem is None, problem or "")
    _require(_is_under(path, campaign_root), f"{what} lies outside the campaign root; the Freeze-1 document records "
                                             "campaign paths relative to that root only.")
    return _real(path).relative_to(_real(campaign_root)).as_posix()


def _git(repo: Path, *args: str) -> tuple[int, str]:
    return provenance._git(Path(repo), *args)


def _git_out(repo: Path, *args: str, what: str) -> str:
    code, out = _git(repo, *args)
    _require(code == 0, f"git {args[0]} failed while reading {what} in {repo}.")
    return out


def _resolve_commit(repo: Path, revision: str) -> Optional[str]:
    code, out = _git(repo, "rev-parse", "--verify", "-q", f"{revision}^{{commit}}")
    return out.strip() if code == 0 and out.strip() else None


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", ancestor, descendant)[0] == 0


def _show(repo: Path, revision: str, path: str) -> Optional[str]:
    code, out = _git(repo, "show", f"{revision}:{path}")
    return out if code == 0 else None


def _show_bytes(repo: Path, revision: str, path: str) -> Optional[bytes]:
    try:
        result = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=str(repo), capture_output=True,
                                timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def tracked_code_sha256(repo: Path = REPO_ROOT) -> dict[str, str]:
    """SHA-256 of every tracked file under ``src/`` and ``scripts/`` (``git ls-files``), read from the tree."""
    repo = Path(repo)
    out = _git_out(repo, "ls-files", "-z", "--", *TRACKED_CODE_FOLDERS, what="the tracked code files")
    paths = sorted(path for path in out.split("\0") if path)
    _require(bool(paths), f"No tracked file under {', '.join(TRACKED_CODE_FOLDERS)} in {repo}.")
    missing = [path for path in paths if not (repo / path).is_file()]
    _require(not missing, f"{len(missing)} tracked code files are missing from the tree, e.g. {missing[:5]}.")
    return {path: sha256_file(repo / path) for path in paths}


# ---------------------------------------------------------------------------
# Rule R-CMD
# ---------------------------------------------------------------------------

def _function_source(text: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """A function's source from its first decorator line to its end, as ``inspect.getsource`` gives it."""
    start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    span = SimpleNamespace(lineno=start, col_offset=0, end_lineno=node.end_lineno, end_col_offset=node.end_col_offset)
    return ast.get_source_segment(text, span) or ""


def _module_scope_nodes(statements: Sequence[ast.stmt]) -> Iterable[ast.AST]:
    """Every node evaluated in module scope, depth first: the bodies of functions, classes and lambdas are skipped
    (their decorators, defaults and class bases are kept), the blocks of ``if``, ``try``, ``with``, loops and
    ``match`` are entered."""
    stack: list[ast.AST] = list(reversed(statements))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            children = [*getattr(node, "decorator_list", []), *arguments.defaults,
                        *(default for default in arguments.kw_defaults if default is not None)]
        elif isinstance(node, ast.ClassDef):
            children = [*node.decorator_list, *node.bases, *node.keywords]
        else:
            children = list(ast.iter_child_nodes(node))
        stack.extend(reversed(children))


def runner_function_sha256(text: str, names: Sequence[str] = PINNED_RUNNER_FUNCTIONS
                           ) -> tuple[dict[str, Optional[str]], list[str]]:
    """SHA-256 of each named top-level function's source and the problems found.

    The source runs from the first decorator line to the end of the function (what
    ``inspect.getsource`` returns), so an added, removed or changed decorator changes the hash.
    A missing name, a second definition, or any other module-scope binding of the name is a
    problem (the name would not resolve to the hashed source): an assignment, ``del``, import,
    class, loop, ``with`` or ``except`` target, or a ``def`` inside a top-level ``if``, ``try``,
    ``with``, loop or ``match`` block.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {name: None for name in names}, [f"the runner does not parse ({exc.msg}, line {exc.lineno})"]
    hashes: dict[str, Optional[str]] = {name: None for name in names}
    defined: Counter = Counter()
    problems = []
    direct = {id(node) for node in tree.body}
    for node in _module_scope_nodes(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in hashes:
            if id(node) in direct:
                defined[node.name] += 1
                hashes[node.name] = sha256_bytes(_function_source(text, node).encode("utf-8"))
            else:
                problems.append(f"the runner defines the pinned name {node.name} inside a top-level block")
        elif isinstance(node, ast.Name) and node.id in hashes and isinstance(node.ctx, (ast.Store, ast.Del)):
            verb = "deletes" if isinstance(node.ctx, ast.Del) else "assigns"
            problems.append(f"the runner {verb} the pinned name {node.id} at top level")
        elif isinstance(node, ast.ExceptHandler) and node.name in hashes:
            problems.append(f"the runner assigns the pinned name {node.name} at top level")
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in hashes:
            problems.append(f"the runner assigns the pinned name {node.name} at top level")
        elif isinstance(node, ast.MatchMapping) and node.rest in hashes:
            problems.append(f"the runner assigns the pinned name {node.rest} at top level")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in hashes:
                    problems.append(f"the runner imports the pinned name {bound} at top level")
        elif isinstance(node, ast.ClassDef) and node.name in hashes:
            problems.append(f"the runner defines a class named after the pinned name {node.name}")
    for name in names:
        if defined[name] == 0:
            problems.append(f"the runner defines no top-level function {name}")
        elif defined[name] > 1:
            problems.append(f"the runner defines {name} {defined[name]} times")
            hashes[name] = None
    return hashes, problems


def _name_status(repo: Path, base: str) -> list[tuple[str, str]]:
    out = _git_out(repo, "diff", "--name-status", "--no-renames", "-z", base, "HEAD", what="the change list")
    fields = [field for field in out.split("\0") if field]
    _require(len(fields) % 2 == 0, "Unexpected output of git diff --name-status.")
    return [(fields[index][:1], fields[index + 1]) for index in range(0, len(fields), 2)]


def r_cmd_change_problems(changes: Iterable[tuple[str, str]]) -> list[str]:
    """Changes (status, path) that rule R-CMD does not allow after the dry run."""
    problems = []
    for status, path in changes:
        if path.startswith(R_CMD_FREE_FOLDERS):
            continue
        if status == "M" and path in R_CMD_ALLOWED_MODIFIED:
            continue
        if status == "A" and (_ADDED_MODULE_RE.match(path) or _ADDED_SCRIPT_RE.match(path)):
            continue
        what = {"A": "added", "D": "deleted", "M": "modified", "T": "changed in type"}.get(status, f"status {status}")
        problems.append(f"{path} was {what} after the dry run; rule R-CMD does not allow it")
    return problems


def _import_probe_code() -> str:
    return "\n".join([
        "import json, os, sys",
        "sys.path.insert(0, sys.argv[1])",
        *(f"import {name}" for name in PROTECTED_IMPORTERS),
        "files = sorted({os.path.realpath(module.__file__) for module in list(sys.modules.values())",
        "                if isinstance(getattr(module, '__file__', None), str)})",
        "print(json.dumps(files))",
    ])


def protected_import_files(repo: Path = REPO_ROOT) -> tuple[Optional[set[str]], Optional[str]]:
    """Repository-relative files a fresh interpreter loads (``sys.modules``) when it imports the protected modules.

    Returns (files, None), or (None, problem) when the interpreter fails.
    """
    repo = Path(repo)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # the check writes nothing into the checked tree
    try:
        completed = subprocess.run([sys.executable, "-c", _import_probe_code(), str(repo / "src")], cwd=str(repo),
                                   capture_output=True, text=True, env=env, timeout=IMPORT_CHECK_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"a fresh interpreter could not import the protected modules ({type(exc).__name__})"
    lines = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not lines:
        reason = (completed.stderr.strip().splitlines() or ["no output"])[-1][:200]
        return None, f"a fresh interpreter could not import the protected modules ({reason})"
    root = _real(repo)
    files = set()
    for item in json.loads(lines[-1]):
        path = Path(item)
        if path == root or root in path.parents:
            files.add(path.relative_to(root).as_posix())
    return files, None


def plans_dir_check(plans_dir: str | os.PathLike) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Each fold's live plan against ``<plans dir>/fold<k>.json`` and its ``.receipt.json`` (the runner's ``plan``).

    The live plan (``campaign.build_fold_plan(k)``, unfiltered) must have the recorded
    ``plan_sha256`` and, written as the runner writes it, the recorded file SHA-256.
    """
    from ec_clean30.campaign import build_fold_plan

    plans_dir = Path(plans_dir)
    records: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for fold in range(N_OUTER_FOLDS):
        name = PLAN_FILE_NAME.format(fold=fold)
        path, receipt_path = plans_dir / name, plans_dir / (name + PLAN_RECEIPT_SUFFIX)
        if any(not candidate.is_file() or candidate.is_symlink() for candidate in (path, receipt_path)):
            problems.append(f"fold {fold}: {name} or its {PLAN_RECEIPT_SUFFIX} is not a regular file in --plans-dir")
            continue
        data = path.read_bytes()
        try:
            recorded, receipt = json.loads(data.decode("utf-8")), json.loads(receipt_path.read_text(encoding="utf-8"))
        except ValueError:
            problems.append(f"fold {fold}: {name} or its receipt is not JSON")
            continue
        live = build_fold_plan(fold)
        live_sha = sha256_bytes(canonical_bytes(live))
        file_sha = sha256_bytes(data)
        if recorded.get("plan_sha256") != live["plan_sha256"] or receipt.get("plan_sha256") != live["plan_sha256"]:
            problems.append(f"fold {fold}: the live plan SHA-256 differs from the record in --plans-dir")
        if file_sha != live_sha or receipt.get("plan_file_sha256") != file_sha:
            problems.append(f"fold {fold}: {name} differs from the live plan written as the runner writes it, or "
                            "from its receipt")
        if receipt.get("plan") != name:
            problems.append(f"fold {fold}: the plan receipt names {receipt.get('plan')!r}, not {name}")
        records[str(fold)] = {"plan": name, "plan_sha256": live["plan_sha256"], "plan_file_sha256": file_sha,
                              "receipt_sha256": sha256_file(receipt_path)}
    return records, problems


def r_cmd_check(dry_run_heads: Mapping[Any, str] | Iterable[str], repo: Path = REPO_ROOT,
                plans_dir: Optional[str | os.PathLike] = None) -> dict[str, Any]:
    """Rule R-CMD from each dry-run HEAD to HEAD; ``problems`` is empty when the dry runs still stand.

    ``dry_run_heads`` maps folds to the HEAD their dry-run receipts record (or lists HEADs).
    Checks: the ``git diff --name-status`` allowlist (``r_cmd_change_problems``); the pinned runner
    functions, hashed on the tree and at each dry-run HEAD; the files added since (``git diff
    --diff-filter=A``) are not loaded by a fresh interpreter that imports the protected modules;
    and, with ``plans_dir``, every fold's live plan against its record (``plans_dir_check``).
    """
    repo = Path(repo)
    if isinstance(dry_run_heads, Mapping):
        by_fold = {str(fold): str(head) for fold, head in sorted(dry_run_heads.items(), key=lambda item: str(item[0]))}
    else:
        by_fold = {str(index): str(head) for index, head in enumerate(dry_run_heads)}
    problems: list[str] = []
    head = _resolve_commit(repo, "HEAD")
    _require(head is not None, f"{repo} has no commit; rule R-CMD needs a git checkout.")
    heads = sorted(set(by_fold.values()))
    if not heads:
        problems.append("no dry-run HEAD was given")
    live_text = (repo / R_CMD_RUNNER).read_text(encoding="utf-8") if (repo / R_CMD_RUNNER).is_file() else None
    if live_text is None:
        problems.append(f"{R_CMD_RUNNER} is missing from the tree")
        live_hashes: dict[str, Optional[str]] = {}
    else:
        live_hashes, live_problems = runner_function_sha256(live_text)
        problems += [f"{problem} (tree)" for problem in live_problems]
    changes: dict[str, list[dict[str, str]]] = {}
    added: set[str] = set()
    for base in heads:
        full = _resolve_commit(repo, base) if re.fullmatch(r"[0-9a-f]{40}", base) else None
        if full is None:
            problems.append(f"the dry-run HEAD {base[:12]!r} is not a full commit SHA of this repository")
            continue
        if not _is_ancestor(repo, full, head):
            problems.append(f"the dry-run HEAD {full[:12]} is not an ancestor of HEAD")
            continue
        entries = _name_status(repo, full)
        changes[full] = [{"status": status, "path": path} for status, path in entries]
        problems += [f"{problem} (since {full[:12]})" for problem in r_cmd_change_problems(entries)]
        added |= {path for status, path in entries if status == "A" and path.endswith(".py")
                  and not path.startswith(R_CMD_FREE_FOLDERS)}
        old_text = _show(repo, full, R_CMD_RUNNER)
        if old_text is None:
            problems.append(f"{R_CMD_RUNNER} does not exist at the dry-run HEAD {full[:12]}")
            continue
        old_hashes, old_problems = runner_function_sha256(old_text)
        problems += [f"{problem} (at {full[:12]})" for problem in old_problems]
        for name in PINNED_RUNNER_FUNCTIONS:
            if old_hashes.get(name) is not None and old_hashes.get(name) != live_hashes.get(name):
                problems.append(f"the pinned runner function {name} changed after the dry run at {full[:12]}")
    loaded, import_problem = protected_import_files(repo)
    if import_problem is not None:
        problems.append(import_problem)
    else:
        for path in sorted(added & loaded):
            problems.append(f"{path}, added after the dry run, is loaded when the protected modules are imported")
    plans: dict[str, dict[str, str]] = {}
    if plans_dir is not None:
        plans, plan_problems = plans_dir_check(plans_dir)
        problems += plan_problems
    return {
        "status": "passed" if not problems else "failed",
        "head": head,
        "dry_run_heads": by_fold,
        "changes": changes,
        "added_files": sorted(added),
        "pinned_runner_function_sha256": dict(sorted(live_hashes.items())),
        "protected_importers": list(PROTECTED_IMPORTERS),
        "plans": plans,
        "problems": problems,
    }


def r_cmd_problems(dry_run_heads: Mapping[Any, str] | Iterable[str], repo: Path = REPO_ROOT,
                   plans_dir: Optional[str | os.PathLike] = None) -> list[str]:
    """The problems of ``r_cmd_check``; any one means the five dry runs are repeated."""
    return r_cmd_check(dry_run_heads, repo, plans_dir)["problems"]


# ---------------------------------------------------------------------------
# Real Freeze-1 inputs manifest (completeness)
# ---------------------------------------------------------------------------

def promotion_constants() -> dict[str, Any]:
    """Decision D1 as coded: the promotion levels and the configurations each lever uses."""
    from ec_clean30 import selection

    return jsonable({
        "PROMOTION_ALPHA": selection.PROMOTION_ALPHA,
        "L9_PROMOTION_ALPHA": selection.L9_PROMOTION_ALPHA,
        "LEVER_PROMOTION_ALPHA": dict(selection.LEVER_PROMOTION_ALPHA),
        "promotion_config_default": selection.PromotionConfig().to_record(),
        "promotion_config_by_lever": {lever: selection.promotion_config_for(lever).to_record()
                                      for lever in selection.LEVERS},
        "promotion_config_steps_without_lever": selection.promotion_config_for(None).to_record(),
    })


def _overlay_files(manifest: Mapping[str, Any]) -> dict[str, Optional[str]]:
    return {str(item.get("path")): item.get("sha256") for item in manifest.get("files") or []
            if isinstance(item, Mapping)}


def inner_membership_problems(view: Any, overlay_root: Path, overlay_files: Mapping[str, Optional[str]],
                              campaign_root: Path) -> tuple[dict[str, Any], list[str]]:
    """Cross-check one view's inner membership against its fold's identity copies in the overlay.

    The membership SHA-256 and source name equal the output of the assignment receipt
    (``identity._load_assignment_receipt`` on ``<overlay>/identity/fold<k>``); the view's certificate
    SHA-256 equals that folder's ``inner_certification_receipt.json``; the overlay manifest lists the
    certificate, the assignment receipt and the membership file with the SHA-256 found.
    """
    from ec_clean30 import identity

    fold = view.fold
    membership = dict(view.membership or {})
    folder = f"identity/fold{fold}"
    work = Path(overlay_root) / folder
    problems: list[str] = []
    if membership.get("role") != "inner":
        problems.append(f"fold {fold}: the view has no inner membership (role {membership.get('role')!r})")
        return {}, problems
    found: dict[str, Optional[str]] = {}
    for name in (identity.INNER_CERTIFICATE, identity.INNER_ASSIGNMENT_RECEIPT):
        relative = f"{folder}/{name}"
        path = Path(overlay_root) / relative
        if not path.is_file() or path.is_symlink():
            problems.append(f"fold {fold}: the overlay holds no regular file {relative}")
            found[name] = None
            continue
        found[name] = sha256_file(path)
        if overlay_files.get(relative) != found[name]:
            problems.append(f"fold {fold}: the overlay manifest does not record {relative} with its SHA-256")
    certificate_sha = found.get(identity.INNER_CERTIFICATE)
    if certificate_sha is not None and membership.get("certificate_sha256") != certificate_sha:
        problems.append(f"fold {fold}: the view's inner certificate differs from the overlay's identity copy")
    output: Mapping[str, Any] = {}
    if found.get(identity.INNER_ASSIGNMENT_RECEIPT) is not None:
        try:
            output = identity._load_assignment_receipt(work, fold)["outputs"]["membership"]
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"fold {fold}: the overlay's assignment receipt does not load ({type(exc).__name__})")
    if output:
        if membership.get("sha256") != output.get("sha256"):
            problems.append(f"fold {fold}: the view's inner membership SHA-256 differs from the assignment receipt's "
                            "output")
        if membership.get("source_name") != output.get("file"):
            problems.append(f"fold {fold}: the view's inner membership source {membership.get('source_name')!r} is "
                            f"not the assignment receipt's output {output.get('file')!r}")
        if overlay_files.get(f"{folder}/{output.get('file')}") != output.get("sha256"):
            problems.append(f"fold {fold}: the overlay manifest does not record the membership file with the "
                            "assignment receipt's SHA-256")
    overlay_relative = _real(overlay_root).relative_to(_real(campaign_root)).as_posix() \
        if _is_under(overlay_root, campaign_root) else None
    record = {
        "membership_sha256": membership.get("sha256"),
        "identity_sha256": membership.get("identity_sha256"),
        "source_name": membership.get("source_name"),
        "assignment_receipt": {"path": f"{overlay_relative}/{folder}/{identity.INNER_ASSIGNMENT_RECEIPT}",
                               "sha256": found.get(identity.INNER_ASSIGNMENT_RECEIPT)},
        "certificate": {"path": f"{overlay_relative}/{folder}/{identity.INNER_CERTIFICATE}",
                        "sha256": certificate_sha},
    }
    return record, problems


def overlay_smoke_problems(manifest: Mapping[str, Any]) -> list[str]:
    """Why an overlay manifest is not a real overlay: its ``smoke`` flag, or that of the cohort audit or R2 lists it
    was built from, is not ``false`` (a smoke overlay may carry smoke inputs and another D2 option)."""
    problems = []
    if manifest.get("smoke") is not False:
        problems.append("the overlay is an engineering smoke (its manifest does not state smoke: false)")
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), Mapping) else {}
    for name in ("cohort", "r2_lists"):
        entry = inputs.get(name) if isinstance(inputs.get(name), Mapping) else {}
        if entry.get("smoke") is not False:
            problems.append(f"the overlay's {name} input is not recorded as real (smoke: false)")
    return problems


def _dry_run_head(view: Any) -> Optional[str]:
    from ec_clean30.views import read_dry_run_receipt

    receipt = read_dry_run_receipt(view)
    git = ((receipt or {}).get("provenance") or {}).get("git") or {}
    head = git.get("head")
    return head if isinstance(head, str) and head else None


def build_freeze1_inputs(view_roots: Sequence[str | os.PathLike], *, plans_dir: str | os.PathLike,
                         repo: Path = REPO_ROOT, provenance_block: Optional[Mapping[str, Any]] = None
                         ) -> dict[str, Any]:
    """The real Freeze-1 inputs manifest: completeness checks, then ``provenance.build_freeze1_manifest`` plus
    the ``completeness`` block (see the module docstring). Raises ``Freeze1Error`` with every problem found."""
    from ec_clean30.campaign import build_fold_plan
    from ec_clean30.overlay import OverlayError, read_overlay_manifest
    from ec_clean30.views import View, ViewError, freeze1_view_record

    _require(bool(view_roots), "A real Freeze-1 manifest needs the five fold views (--view, one per fold).")
    loaded = [View.load(root) for root in view_roots]
    problems = [f"{view.root.name} is a smoke view; refused in a real Freeze-1 manifest" for view in loaded
                if view.smoke]
    folds = sorted(view.fold for view in loaded)
    if folds != list(range(N_OUTER_FOLDS)):
        problems.append(f"a real Freeze-1 manifest needs exactly one view per fold 0-{N_OUTER_FOLDS - 1}, got folds "
                        f"{folds}")
    _refuse_problems("The views do not form a complete Freeze-1 set.", problems)
    views = sorted(loaded, key=lambda view: view.fold)
    roots = {_real(view.campaign_root) for view in views}
    _require(len(roots) == 1, "The five views belong to different campaign roots.")
    campaign_root = views[0].campaign_root
    candidates = [(plans_dir, "--plans-dir"), *((view.root, f"the fold-{view.fold} view") for view in views)]
    problems = [problem for path, what in candidates
                if (problem := campaign_identity_problem(path, campaign_root, what)) is not None]
    overlays = {None if view.overlay is None else (view.overlay.get("id"), view.overlay.get("manifest_sha256"))
                for view in views}
    if None in overlays:
        problems.append("a view was not built from a data overlay; Freeze 1 needs one overlay for all folds")
    elif len(overlays) != 1:
        problems.append(f"the views name {len(overlays)} different overlays (ID or manifest SHA-256); Freeze 1 needs "
                        "one overlay for all folds")
    data_roots = {_real(view.manifest["source"]["data_root"]) for view in views}
    if len(data_roots) != 1:
        problems.append("the views were built from different data roots")
    _refuse_problems("The views do not share one overlay.", problems)
    overlay_id, overlay_sha = overlays.pop()
    overlay_root = data_roots.pop()
    problem = campaign_identity_problem(overlay_root, campaign_root, "the overlay")
    _require(problem is None, problem or "")
    _require(_is_under(overlay_root, campaign_root), "The overlay lies outside the campaign root.")
    try:
        overlay_manifest, manifest_sha = read_overlay_manifest(overlay_root)
    except OverlayError as exc:
        raise Freeze1Error(f"The overlay manifest does not load: {exc}") from None
    _require(manifest_sha == overlay_sha and overlay_manifest.get("overlay_id") == overlay_id,
             "The overlay manifest on disk is not the one the views recorded (ID or SHA-256 differs).")
    _refuse_problems("A real Freeze-1 manifest needs a real (non-smoke) overlay.",
                     overlay_smoke_problems(overlay_manifest))
    files = _overlay_files(overlay_manifest)
    memberships: dict[str, dict[str, Any]] = {}
    for view in views:
        record, view_problems = inner_membership_problems(view, overlay_root, files, campaign_root)
        memberships[str(view.fold)] = record
        problems += view_problems
    _refuse_problems("The inner memberships do not match the overlay's identity copies.", problems)

    heads = {view.fold: _dry_run_head(view) for view in views}
    missing = [fold for fold, head in heads.items() if head is None]
    _require(not missing, f"The dry-run receipts of folds {missing} record no git HEAD; {RERUN_DRY_RUNS}.")
    r_cmd = r_cmd_check(heads, repo=repo, plans_dir=plans_dir)
    _refuse_problems(f"Rule R-CMD does not hold; {RERUN_DRY_RUNS} before Freeze 1.", r_cmd["problems"])

    records = []
    for view in views:
        try:
            records.append(freeze1_view_record(view.root, require_dry_run=True, provenance=provenance_block))
        except ViewError as exc:
            problems.append(f"fold {view.fold}: {exc}")
    _refuse_problems("A view does not verify or has no passed dry run of itself.", problems)
    plans = [build_fold_plan(fold) for fold in range(N_OUTER_FOLDS)]
    manifest = provenance.build_freeze1_manifest(plans=plans, views=records, smoke=False,
                                                 provenance=provenance_block)
    manifest["completeness"] = {
        "format": FREEZE1_INPUTS_COMPLETENESS_FORMAT,
        "status": "passed",
        "campaign_root": str(campaign_root),
        "folds": list(range(N_OUTER_FOLDS)),
        "overlay": {"id": overlay_id, "manifest_sha256": overlay_sha,
                    "root": _real(overlay_root).relative_to(_real(campaign_root)).as_posix()},
        "inner_membership": memberships,
        "file_modes_enforced": {str(record["fold"]): record["file_modes_enforced"] for record in records},
        "r_cmd": r_cmd,
        "promotion": promotion_constants(),
        "tracked_code_sha256": tracked_code_sha256(repo),
    }
    return manifest


# ---------------------------------------------------------------------------
# Freeze-1 document
# ---------------------------------------------------------------------------

def _walk_strings(value: Any, where: str = "") -> Iterable[tuple[str, str]]:
    """(location, text) for every string key and value, depth first."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            location = f"{where}.{key}" if where else str(key)
            yield location, str(key)
            yield from _walk_strings(item, location)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{where}[{index}]")
    elif isinstance(value, str):
        yield where, value


def absolute_path_problems(value: Any, what: str) -> list[str]:
    """Locations in ``value`` whose text holds an absolute path (a root, home-folder or drive-letter path)."""
    return [f"{what} {location or '(top)'} holds an absolute path" for location, text in _walk_strings(value)
            if _ABSOLUTE_PATH_RE.search(text)]


def placeholder_problems(value: Any, what: str) -> list[str]:
    """Locations in ``value`` whose text holds a placeholder token (``_PLACEHOLDER_TOKEN_RE``: ``<a name>``,
    ``[a name]``, ``{name}``, TBD, TODO, FIXME). Set, interval and brace-list notation is allowed."""
    return [f"{what} {location or '(top)'} holds a placeholder" for location, text in _walk_strings(value)
            if _PLACEHOLDER_TOKEN_RE.search(text)]


def placeholder_character_problems(value: Any, what: str, where: str = "") -> list[str]:
    """Locations in ``value`` (below ``where``) whose text holds a placeholder character (the scorer's rule: any of
    ``[]{}<>``)."""
    return [f"{what} {location or '(top)'} holds a placeholder character"
            for location, text in _walk_strings(value, where) if set(text) & PLACEHOLDER_CHARACTERS]


def class_count_problems(value: Any, what: str) -> list[str]:
    """Locations in ``value`` under a key that holds per-fold class numbers (``CLASS_COUNT_KEYS``)."""
    problems = []

    def walk(item: Any, where: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                location = f"{where}.{key}" if where else str(key)
                if str(key) in CLASS_COUNT_KEYS:
                    problems.append(f"{what} {location} would copy per-fold class numbers")
                walk(child, location)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{where}[{index}]")

    walk(value, "")
    return problems


def decision_answer(value: Any) -> Optional[str]:
    """The answer text of one decision entry (a string, or a mapping with an ``answer`` string)."""
    answer = value.get("answer") if isinstance(value, Mapping) else value
    if not isinstance(answer, str) or answer.strip().lower() in UNANSWERED_ANSWERS:
        return None
    return answer.strip()


def decision_problems(record: Any) -> list[str]:
    """Why a ``freeze1_decisions.json`` record cannot be frozen (empty when it can).

    It needs the decisions format and campaign ID, ``provisional_markers_settled: true`` (the
    code's "provisional" markers are settled by this record), and an answer for every id of
    ``REQUIRED_DECISION_IDS`` and every other id it lists. No text may hold a placeholder token
    (``placeholder_problems``) or an absolute path; the acknowledgement-name decisions
    (``ACKNOWLEDGEMENT_DECISION_IDS``) also take the scorer's character rule (none of ``[]{}<>``),
    while other answers may use set and interval notation such as ``{-1, 0, 1}`` or ``(20,25]``.
    """
    if not isinstance(record, Mapping):
        return ["the decisions record is not a JSON object"]
    problems = []
    if record.get("format") != FREEZE1_DECISIONS_FORMAT:
        problems.append(f"the decisions record's format is not {FREEZE1_DECISIONS_FORMAT}")
    if record.get("campaign_id") != CAMPAIGN_ID:
        problems.append(f"the decisions record is not for campaign {CAMPAIGN_ID}")
    if record.get("provisional_markers_settled") is not True:
        problems.append("the decisions record does not state provisional_markers_settled: true")
    decisions = record.get("decisions")
    if not isinstance(decisions, Mapping):
        return problems + ["the decisions record has no 'decisions' object"]
    for decision_id in REQUIRED_DECISION_IDS:
        if decision_id not in decisions:
            problems.append(f"decision {decision_id} is missing")
    for decision_id, value in decisions.items():
        if decision_answer(value) is None:
            problems.append(f"decision {decision_id} is unanswered")
    problems += placeholder_problems(record, "the decisions record at")
    for decision_id in ACKNOWLEDGEMENT_DECISION_IDS:
        if decision_id in decisions:
            problems += placeholder_character_problems(decisions[decision_id], "the decisions record at",
                                                       f"decisions.{decision_id}")
    problems += absolute_path_problems(record, "the decisions record at")
    return problems


def load_decisions(path: str | os.PathLike) -> tuple[dict[str, Any], str]:
    """The decisions record and its SHA-256; ``Freeze1Error`` when ``decision_problems`` finds anything."""
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"--decisions is not a regular file: {path}")
    data = path.read_bytes()
    try:
        record = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise Freeze1Error(f"--decisions is not JSON: {exc}") from None
    _refuse_problems("The decisions record cannot be frozen.", decision_problems(record))
    return record, sha256_bytes(data)


def parse_pins(values: Optional[Sequence[str]]) -> dict[str, str]:
    """``--pin NAME=SHA256`` values; every name of ``REQUIRED_PINS`` once, the MMseqs2 binary as coded."""
    from ec_clean30 import identity

    pins: dict[str, str] = {}
    for value in values or ():
        name, separator, digest = str(value).partition("=")
        name, digest = name.strip(), digest.strip().lower()
        _require(bool(separator) and bool(_PIN_NAME_RE.match(name)) and bool(_SHA256_RE.match(digest)),
                 f"--pin takes NAME=SHA256 (lower-case name, 64 hexadecimal characters), got {value!r}")
        _require(name not in pins, f"--pin names {name} twice")
        pins[name] = digest
    missing = [name for name in REQUIRED_PINS if name not in pins]
    _require(not missing, f"--pin is missing {missing} (the CLEAN harness, ESMC-300m weights and MMseqs2 binary).")
    _require(pins["mmseqs2_binary"] == identity.MMSEQS_BINARY_SHA256,
             "--pin mmseqs2_binary differs from the MMseqs2 binary the identity code pins.")
    _require(pins["clean_patch_manifest"].startswith(CLEAN_PATCH_MANIFEST_PREFIX),
             f"--pin clean_patch_manifest is not the patch manifest {CLEAN_PATCH_MANIFEST_PREFIX}... of decision F6.")
    return dict(sorted(pins.items()))


def _bind_files(folder: str | os.PathLike, names: Mapping[str, str], campaign_root: Path, what: str,
                *, optional: bool = False) -> dict[str, dict[str, str]]:
    """{role: {path, sha256}} for files at exact names below ``folder`` (campaign-relative paths)."""
    folder = Path(folder)
    campaign_relative(folder, campaign_root, what)
    bound: dict[str, dict[str, str]] = {}
    missing = []
    for role, name in names.items():
        path = folder / name
        if not path.is_file() or path.is_symlink():
            missing.append(name)
            continue
        bound[role] = {"path": campaign_relative(path, campaign_root, f"{what} file {name}"),
                       "sha256": sha256_file(path)}
    if optional and len(missing) == len(names):
        return {}
    _require(not missing, f"{what} holds no regular file {missing} (bound by exact name).")
    return bound


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Freeze1Error(f"{what} is not readable JSON ({type(exc).__name__}).") from None


def _simulation_problems(folder: Path, bound: Mapping[str, Mapping[str, str]]) -> tuple[int, list[str]]:
    """The final sweep's receipt names the bound sweep and summary, and every record it lists is a real
    (non-smoke) record with the recorded SHA-256. Records are read for these checks only; nothing is copied."""
    receipt = _read_json(folder / SIMULATION_FILES["sweep_receipt"], "The simulation receipt")
    problems = []
    if not isinstance(receipt, Mapping) or receipt.get("format") != SIMULATION_RECEIPT_FORMAT:
        return 0, [f"the simulation receipt is not a {SIMULATION_RECEIPT_FORMAT} receipt"]
    if (receipt.get("provenance") or {}).get("smoke") is not False:
        problems.append("the simulation receipt is not from a real (non-smoke) sweep")
    if receipt.get("sweep_file_sha256") != bound["sweep"]["sha256"]:
        problems.append("the simulation receipt names another sweep file")
    if (receipt.get("summary") or {}).get("sha256") != bound["summary"]["sha256"]:
        problems.append("the simulation receipt names another summary")
    records = receipt.get("records")
    if not isinstance(records, Mapping) or not records:
        return 0, problems + ["the simulation receipt lists no records"]
    for point_id, entry in sorted(records.items()):
        name = entry.get("file") if isinstance(entry, Mapping) else None
        if not isinstance(name, str) or not _SIMULATION_RECORD_RE.match(name):
            problems.append(f"simulation record {point_id!r} has no plain records/<id>.json name")
            continue
        path = folder / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != entry.get("sha256"):
            problems.append(f"simulation record {name} is missing or differs from the receipt")
            continue
        record = _read_json(path, f"Simulation record {name}")
        if not isinstance(record, Mapping) or record.get("format") != SIMULATION_RECORD_FORMAT \
                or record.get("smoke") is not False or record.get("point_id") != point_id:
            problems.append(f"simulation record {name} is not a real record of point {point_id!r}")
    return len(records), problems


def live_constants() -> dict[str, Any]:
    """The constants of the campaign code that Freeze 1 states (read from the modules now)."""
    from ec_clean30 import baseline, campaign, identity, logit_adjustment, scoring, selection, stats
    from ec_clean30.commands import _comparator_published_recipe

    selection_names = (
        "POINTS", "FLOAT_TOLERANCE", "SCREENING_SEEDS", "REFIT_SEEDS", "N_INNER_FOLDS", "MIN_GAIN_POINTS",
        "PROMOTION_ALPHA", "L9_PROMOTION_ALPHA", "LR_TIE_BAND_POINTS", "GUARD_MIN_TRAINING_PROTEINS",
        "GUARD_RECALL_DROP_POINTS", "GUARD_NET_PROTEINS", "ZERO_RECALL_UNITS", "GAIN_CONTRAST", "FAMILIES",
        "ANCHOR_FAMILY", "CONTROL_FAMILIES", "GVP_FAMILIES", "GLOBAL_BRANCH_FAMILIES", "S1_LEARNING_RATE_GRID",
        "LEVERS", "EPOCH_LEVER", "LOGIT_ADJUSTMENT_LEVER", "DEFAULT_EPOCHS", "EPOCH_LOOKS",
        "SUPCON_MIN_POSITIVE_STEP_FRACTION", "SUPCON_LEVER", "NULL_ADOPTION_TARGET", "DEFAULT_SIMULATION_SEED",
        "LEVER_PROMOTION_ALPHA",
    )
    stats_names = ("DEFAULT_N_DRAWS", "DEFAULT_BOOTSTRAP_SEED", "DEFAULT_MAX_SKIP_FRACTION", "DRAW_BATCH_SIZE",
                   "BOOTSTRAP_METRICS", "P_VALUE_TOLERANCE", "CO_PRIMARY_FAMILY_ALPHA", "DEFAULT_SUBSET",
                   "CLASS_7_ONLY_STRATUM", "UNSTRATIFIED_STRATUM")
    identity_names = (
        "MMSEQS_RELEASE", "MMSEQS_VERSION", "MMSEQS_BINARY_SHA256", "SEARCH_SETTINGS", "SEARCH_RUN_FLAGS",
        "MAX_TARGET_SEQUENCES", "MAX_EVALUE", "MIN_RESIDUE_PAIRS", "IDENTITY_THRESHOLD", "D2_SHORTER_COVERAGE",
        "D1_BOTH_COVERAGE", "AMBIGUOUS_RESIDUE_POLICY", "IDENTITY_RULE", "DONOR_ID_RE", "COMPLEMENT_SCOPE",
        "IDENTITY_BIN_LABELS", "IDENTITY_BIN_STATUS", "NEAREST_HIT_ORDER", "INNER_LINK_TYPES", "COHORT_LINK_TYPES",
        "N_INNER_FOLDS", "INNER_SPLIT_SEED", "DEFAULT_MAX_COMPONENT_SHARE", "INNER_ELIGIBILITY_RULE",
        "INNER_BALANCE_RULE", "MAX_SHARE_RULE", "MAX_SHARE_BASIS", "RERUN_RULE", "R1_EXCLUSION_RULE",
    )
    baseline_names = ("STEP", "STATUSES", "SEVERAL_CLASS_RULE", "CLASS_TABLE_COLUMNS")
    return jsonable({
        "campaign": {
            "campaign_constants": campaign.campaign_constants(),
            "provisional_lever_settings": dict(campaign.PROVISIONAL_LEVER_SETTINGS),
            "s2_sub_ceilings": dict(campaign.S2_SUB_CEILINGS),
            "l1_target_shrink": campaign.L1_TARGET_SHRINK,
            "l8_layer": campaign.L8_LAYER,
        },
        "selection": {name: getattr(selection, name) for name in selection_names},
        "promotion": promotion_constants(),
        "stats": {name: getattr(stats, name) for name in stats_names},
        "scoring": {"FROZEN_SETTINGS": dict(scoring.FROZEN_SETTINGS),
                    "MANIFEST_SETTINGS": list(scoring.MANIFEST_SETTINGS),
                    "CO_PRIMARY_POPULATIONS": dict(scoring.CO_PRIMARY_POPULATIONS),
                    "ABSTENTION_RULE": scoring.ABSTENTION_RULE},
        "identity": {**{name: getattr(identity, name) for name in identity_names},
                     "rules": identity.rules_record()},
        "logit_adjustment": {"TAUS": logit_adjustment.TAUS, "PRIOR_SMOOTHING": logit_adjustment.PRIOR_SMOOTHING},
        "baseline": {name: getattr(baseline, name) for name in baseline_names},
        "comparator_published_recipe": _comparator_published_recipe(),
    })


def build_freeze1_document(
    *,
    inputs: str | os.PathLike,
    inputs_sha256: str,
    decisions: str | os.PathLike,
    overlay_manifest: str | os.PathLike,
    labels: str | os.PathLike,
    cohort: str | os.PathLike,
    r2_lists: str | os.PathLike,
    simulation: str | os.PathLike,
    environment: str | os.PathLike,
    pins: Optional[Sequence[str]],
    no_transfer_list: bool = False,
    repo: Path = REPO_ROOT,
) -> dict[str, Any]:
    """The canonical Freeze-1 document (see the module docstring); raises ``Freeze1Error`` on any problem.

    Every path is checked before it is read: ``--inputs`` may not lie under an ``identity`` folder
    of a folder named after the campaign ID (its campaign root is known only from its content),
    ``--decisions`` must lie in the repository, and ``--overlay`` and the bound folders in the
    campaign root outside ``<campaign root>/identity``. The transfer list (decision A8, P2-9d) is
    required below ``labels`` unless ``no_transfer_list`` states that none was approved; then
    neither of its files may exist and the document records ``TRANSFER_LIST_ABSENT``.
    """
    from ec_clean30.campaign import build_fold_plan, campaign_constants
    from ec_clean30.r2_lists import R2ListError, check_cohort_receipt

    repo = Path(repo)
    inputs_sha256 = _sha256_argument(inputs_sha256, "--inputs-sha256")
    inputs_path = Path(inputs)
    problem = campaign_identity_name_problem(inputs_path, "--inputs")
    _require(problem is None, problem or "")
    _require(inputs_path.is_file() and not inputs_path.is_symlink(), f"--inputs is not a regular file: {inputs_path}")
    inputs_bytes = inputs_path.read_bytes()
    _require(sha256_bytes(inputs_bytes) == inputs_sha256, "--inputs does not hash to --inputs-sha256.")
    manifest = json.loads(inputs_bytes.decode("utf-8"))
    completeness = manifest.get("completeness") if isinstance(manifest, Mapping) else None
    _require(isinstance(manifest, Mapping) and manifest.get("format") == provenance.FREEZE1_FORMAT
             and manifest.get("kind") == "freeze1_inputs" and manifest.get("smoke") is False
             and isinstance(completeness, Mapping)
             and completeness.get("format") == FREEZE1_INPUTS_COMPLETENESS_FORMAT
             and completeness.get("status") == "passed",
             "--inputs is not a real, complete Freeze-1 inputs manifest (manifest --kind freeze1 --plans-dir).")
    campaign_root = Path(completeness["campaign_root"])
    inputs_relative = campaign_relative(inputs_path, campaign_root, "--inputs")
    problems: list[str] = []
    if manifest.get("constants") != jsonable(campaign_constants()):
        problems.append("the live campaign constants differ from the inputs manifest")
    for fold in range(N_OUTER_FOLDS):
        recorded = (manifest.get("plans") or {}).get(str(fold)) or {}
        if recorded.get("plan_sha256") != build_fold_plan(fold)["plan_sha256"]:
            problems.append(f"fold {fold}: the live plan SHA-256 differs from the inputs manifest")
    tracked = tracked_code_sha256(repo)
    pinned = completeness.get("tracked_code_sha256") or {}
    changed = sorted(path for path in set(tracked) | set(pinned) if tracked.get(path) != pinned.get(path))
    if changed:
        problems.append(f"{len(changed)} tracked src/ or scripts/ files differ from the inputs manifest's pins, e.g. "
                        f"{changed[:5]}")
    _refuse_problems("The inputs manifest no longer describes this code.", problems)

    decisions_path = Path(decisions)
    _require(_is_under(decisions_path, repo), "--decisions must be a file of this repository (the campaign docs).")
    decisions_record, decisions_sha = load_decisions(decisions_path)
    overlay_path = Path(overlay_manifest)
    overlay_relative = campaign_relative(overlay_path, campaign_root, "--overlay")
    _require(overlay_path.is_file() and not overlay_path.is_symlink(), "--overlay is not a regular file.")
    overlay_record = completeness["overlay"]
    _require(sha256_file(overlay_path) == overlay_record["manifest_sha256"],
             "--overlay is not the overlay manifest the inputs manifest recorded.")
    overlay_manifest_record = _read_json(overlay_path, "--overlay")
    _require(isinstance(overlay_manifest_record, Mapping), "--overlay is not a JSON object.")
    overlay_files = _overlay_files(overlay_manifest_record)

    bound: dict[str, dict[str, str]] = {}
    bound.update(_bind_files(labels, LABEL_FILES, campaign_root, "--labels"))
    if no_transfer_list:
        present = [name for name in TRANSFER_LIST_FILES.values() if os.path.lexists(Path(labels) / name)]
        _require(not present, f"--no-transfer-list states that no transfer list was approved (decision A8), but "
                              f"--labels holds {present}; bind it (drop --no-transfer-list) or remove it.")
        transfer: dict[str, dict[str, str]] = {}
    else:
        transfer = _bind_files(labels, TRANSFER_LIST_FILES, campaign_root,
                               "--labels (transfer list, decision A8; --no-transfer-list when none was approved)")
    bound.update(transfer)
    bound.update(_bind_files(cohort, COHORT_FILES, campaign_root, "--cohort"))
    bound.update(_bind_files(r2_lists, R2_FILES, campaign_root, "--r2-lists"))
    simulation_bound = _bind_files(simulation, SIMULATION_FILES, campaign_root, "--simulation")
    bound.update({f"simulation_{role}": entry for role, entry in simulation_bound.items()})
    bound.update({f"environment_{role}": entry for role, entry in
                  _bind_files(environment, ENVIRONMENT_FILES, campaign_root, "--environment").items()})
    try:
        check_cohort_receipt(_read_json(Path(cohort) / COHORT_FILES["cohort_audit_receipt"], "--cohort's receipt"),
                             smoke=False)
    except R2ListError as exc:
        problems.append(f"the cohort audit receipt: {exc}")
    r2_receipt = _read_json(Path(r2_lists) / R2_FILES["r2_lists_receipt"], "--r2-lists's receipt")
    if not isinstance(r2_receipt, Mapping) or ((r2_receipt.get("provenance") or {}).get("smoke") is not False):
        problems.append("the R2 lists receipt is not from real (non-smoke) R2 lists")
    problems += overlay_smoke_problems(overlay_manifest_record)
    for role, relative in OVERLAY_COPIES.items():
        if overlay_files.get(relative) != bound[role]["sha256"]:
            problems.append(f"the overlay manifest does not record {relative} with the SHA-256 of the bound {role}")
    n_records, simulation_problems = _simulation_problems(Path(simulation), simulation_bound)
    problems += simulation_problems
    pin_values = parse_pins(pins)
    _refuse_problems("The bound records do not agree.", problems)

    head = _resolve_commit(repo, "HEAD")
    document = {
        "format": FREEZE1_DOCUMENT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "kind": "freeze1",
        "amendments": [],
        "source_commit": head,
        "inputs_manifest": {"path": inputs_relative, "sha256": inputs_sha256},
        "decisions": {"path": _real(decisions_path).relative_to(_real(repo)).as_posix(), "sha256": decisions_sha,
                      "provisional_markers_settled": True,
                      "answers": {str(key): value for key, value in sorted(decisions_record["decisions"].items())}},
        "plans": {fold: {"plan_sha256": entry["plan_sha256"], "unit_counts": entry["counts"],
                         "filters": entry["filters"]} for fold, entry in sorted(manifest["plans"].items())},
        "plans_dir_records": completeness["r_cmd"]["plans"],
        "views": {str(view["fold"]): {key: view.get(key) for key in (
            "kind", "bundle_id", "view_manifest_sha256", "allowlist_sha256", "eligible_ids_sha256", "probe_sha256",
            "membership", "chain_means", "file_modes_enforced", "overlay", "dry_run_receipt")}
            for view in manifest["views"]},
        "file_modes_enforced": completeness["file_modes_enforced"],
        "inner_membership": completeness["inner_membership"],
        "overlay": {"id": overlay_record["id"], "root": overlay_record["root"],
                    "manifest": {"path": overlay_relative, "sha256": overlay_record["manifest_sha256"]}},
        "r_cmd": {"status": completeness["r_cmd"]["status"], "head": completeness["r_cmd"]["head"],
                  "dry_run_heads": completeness["r_cmd"]["dry_run_heads"],
                  "pinned_runner_function_sha256": completeness["r_cmd"]["pinned_runner_function_sha256"]},
        "code": {"campaign_code_sha256": manifest["code_sha256"], "tracked_file_sha256": pinned},
        "bound_files": dict(sorted(bound.items())),
        "transfer_list": "bound" if transfer else TRANSFER_LIST_ABSENT,
        "simulation_records": {"n_records": n_records},
        "pins": pin_values,
        "constants": live_constants(),
        "inputs_selection_constants": manifest["selection_constants"],
    }
    document = jsonable(document)
    problems = absolute_path_problems(document, "the Freeze-1 document at")
    problems += class_count_problems(document, "the Freeze-1 document at")
    _refuse_problems("The Freeze-1 document would break its own rules.", problems)
    return document


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return " ".join(str(text).split()).replace("|", "\\|")


def render_freeze1_markdown(document: Mapping[str, Any]) -> str:
    """The deterministic ``freeze1.md`` rendering of a Freeze-1 document (the JSON file stays the record)."""
    lines = [
        f"# Freeze 1: {document['campaign_id']}",
        "",
        "This page renders `freeze1.json`; that file and its SHA-256 (logged in `log.md`) are the record.",
        "",
        f"- Format: `{document['format']}` (schema {document['schema_version']})",
        f"- Source commit: `{document['source_commit']}`",
        f"- Inputs manifest: `{document['inputs_manifest']['path']}`, "
        f"SHA-256 `{document['inputs_manifest']['sha256']}`",
        f"- Decisions record: `{document['decisions']['path']}`, SHA-256 `{document['decisions']['sha256']}`",
        f"- Overlay: `{document['overlay']['id']}`, manifest SHA-256 `{document['overlay']['manifest']['sha256']}`",
        f"- Transfer list: {document['transfer_list']}",
        f"- Amendments: {len(document['amendments'])}",
        "",
        "## Decisions",
        "",
        "| ID | Answer |",
        "|---|---|",
    ]
    for decision_id, value in document["decisions"]["answers"].items():
        lines.append(f"| {_cell(decision_id)} | {_cell(decision_answer(value) or '')} |")
    lines += ["", "## Plans", "", "| Fold | Plan SHA-256 | Planned fits |", "|---|---|---|"]
    for fold, entry in document["plans"].items():
        lines.append(f"| {fold} | `{entry['plan_sha256']}` | {entry['unit_counts'].get('total')} |")
    lines += ["", "## Views", "",
              "| Fold | View manifest SHA-256 | Inner membership SHA-256 | Certificate SHA-256 "
              "| Dry-run receipt SHA-256 | File modes enforced |", "|---|---|---|---|---|---|"]
    for fold, view in document["views"].items():
        membership = view.get("membership") or {}
        dry = view.get("dry_run_receipt") or {}
        lines.append(f"| {fold} | `{view['view_manifest_sha256']}` | `{membership.get('sha256')}` | "
                     f"`{membership.get('certificate_sha256')}` | `{dry.get('sha256')}` | "
                     f"{_cell(view.get('file_modes_enforced'))} |")
    lines += ["", "## Rule R-CMD", "", f"- Status: {document['r_cmd']['status']} at `{document['r_cmd']['head']}`"]
    for fold, head in document["r_cmd"]["dry_run_heads"].items():
        lines.append(f"- Fold {fold} dry-run HEAD: `{head}`")
    lines += ["", "## Bound files", "", "| Role | Path | SHA-256 |", "|---|---|---|"]
    for role, entry in document["bound_files"].items():
        lines.append(f"| {_cell(role)} | `{entry['path']}` | `{entry['sha256']}` |")
    for fold, entry in document["inner_membership"].items():
        for role in ("assignment_receipt", "certificate"):
            label = f"fold {fold} {role.replace('_', ' ')}"
            lines.append(f"| {label} | `{entry[role]['path']}` | `{entry[role]['sha256']}` |")
    lines += ["", "## Pins", "", "| Name | SHA-256 |", "|---|---|"]
    lines += [f"| {_cell(name)} | `{digest}` |" for name, digest in document["pins"].items()]
    lines += ["", "## Live constants", ""]
    for group, values in document["constants"].items():
        lines += [f"### {group}", "", "```json", json.dumps(values, indent=2, sort_keys=True), "```", ""]
    tracked = document["code"]["tracked_file_sha256"]
    lines += [f"## Code pins ({len(tracked)} tracked src/ and scripts/ files)", "", "| Path | SHA-256 |", "|---|---|"]
    lines += [f"| `{path}` | `{digest}` |" for path, digest in sorted(tracked.items())]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_freeze1_document(document: Mapping[str, Any], out: str | os.PathLike) -> tuple[str, Path]:
    """Write ``freeze1.json`` and its ``.md`` rendering once; returns the JSON SHA-256 and the rendering's path."""
    out = Path(out)
    rendering = out.with_suffix(".md")
    _require(out.suffix == ".json", "--out must name the .json document (the rendering gets the .md suffix).")
    for path in (out, rendering):
        if os.path.lexists(path):
            raise FileExistsError(f"{path} exists; the Freeze-1 document is written once.")
    text = render_freeze1_markdown(document)
    _refuse_problems("The rendering would hold an absolute path.", absolute_path_problems(text, "freeze1.md"))
    digest = provenance.write_new_json(out, document)
    with rendering.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return digest, rendering


# ---------------------------------------------------------------------------
# verify-freeze1
# ---------------------------------------------------------------------------

def _hash_in_text(digest: str, text: str) -> bool:
    return re.search(rf"(?<![0-9A-Za-z]){re.escape(digest)}(?![0-9A-Za-z])", text) is not None


def verify_freeze1(*, sha256: str, document: str | os.PathLike, require_pushed: bool,
                   repo: Path = REPO_ROOT) -> dict[str, Any]:
    """The ``verify-freeze1`` record; ``status`` is ``passed`` only when ``problems`` is empty (no fetch)."""
    repo = Path(repo)
    digest = _sha256_argument(sha256, "--sha256")
    document = Path(document)
    _require(document.is_file() and not document.is_symlink(), f"--document is not a regular file: {document}")
    head = _resolve_commit(repo, "HEAD")
    _require(head is not None, f"{repo} has no commit.")
    problems = []
    if sha256_file(document) != digest:
        problems.append("the document does not hash to the given SHA-256")
    relative = _real(document).relative_to(_real(repo)).as_posix() if _is_under(document, repo) else None
    if relative is None:
        problems.append("the document is not a file of this repository")
    else:
        committed = _show_bytes(repo, "HEAD", relative)
        if committed is None or sha256_bytes(committed) != digest:
            problems.append("the document is not committed at HEAD with these bytes")
    log_text = _show(repo, "HEAD", LOG_PATH)
    introducing = commit_time = None
    if log_text is None or not _hash_in_text(digest, log_text):
        problems.append(f"the SHA-256 does not appear in {LOG_PATH} at HEAD")
    else:
        out = _git_out(repo, "log", f"-S{digest}", "--reverse", "--format=%H", "HEAD", "--", LOG_PATH,
                       what="the log history")
        introducing = (out.split() or [None])[0]
        if introducing is None or not _is_ancestor(repo, introducing, head):
            problems.append("the commit that logged the SHA-256 is not an ancestor of HEAD")
        else:
            commit_time = _git_out(repo, "show", "-s", "--format=%cI", introducing, what="the commit time").strip()
    origin = _resolve_commit(repo, ORIGIN_MAIN_REF)
    pushed = bool(introducing and origin and _is_ancestor(repo, introducing, origin))
    if require_pushed and not pushed:
        problems.append(f"{ORIGIN_MAIN_REF} does not contain the commit that logged the SHA-256 (no fetch is made; "
                        "push, then fetch, then check again)")
    return {
        "format": VERIFY_FREEZE1_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "kind": "verify_freeze1",
        "status": "passed" if not problems else "failed",
        "problems": problems,
        "freeze1_sha256": digest,
        "document": relative,
        "log": LOG_PATH,
        "head": head,
        "introducing_commit": introducing,
        "introducing_commit_time": commit_time,
        "origin_main": {"ref": ORIGIN_MAIN_REF, "commit": origin, "contains_introducing_commit": pushed,
                        "fetched": False},
        "pushed": pushed,
        "require_pushed": bool(require_pushed),
    }


# ---------------------------------------------------------------------------
# Freeze 1b
# ---------------------------------------------------------------------------

def _identity_head_problems(repo: Path, identity_head: str, head: str, log_text: Optional[str]
                            ) -> tuple[Optional[str], list[str]]:
    token = str(identity_head or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", token):
        return None, ["--identity-head must be a commit SHA (7 to 40 hexadecimal characters)"]
    full = _resolve_commit(repo, token)
    if full is None:
        return None, ["--identity-head names no commit of this repository"]
    problems = []
    if not _is_ancestor(repo, full, head):
        problems.append("--identity-head is not an ancestor of HEAD")
    named = log_text is not None and any(full.startswith(match.group(0)) and len(match.group(0)) >= 7
                                         for match in _COMMIT_TOKEN_RE.finditer(log_text))
    if not named:
        problems.append(f"--identity-head is not named in {LOG_PATH} at HEAD")
    changed = [path for path in _git_out(repo, "diff", "--name-only", "--no-renames", full, "HEAD", "--",
                                         *R_ID_FILES, what="the R-ID files").splitlines() if path.strip()]
    if changed:
        problems.append(f"the R-ID files {changed} changed between --identity-head and HEAD")
    return full, problems


def _receipt_problems(receipt: Mapping[str, Any], name: str, *, fold: Optional[int], expected_code: Mapping[str, Any],
                      freeze1_sha256: Optional[str]) -> list[str]:
    where = f"fold {fold}: {name}" if fold is not None else name
    problems = []
    if dict(receipt.get("code_sha256") or {}) != dict(expected_code):
        problems.append(f"{where} records another identity code SHA-256 than Freeze 1 pins")
    if freeze1_sha256 is not None and receipt.get("freeze1_sha256") != freeze1_sha256:
        problems.append(f"{where} does not record the Freeze-1 SHA-256")
    return problems


def _outputs_problems(work: Path, receipt: Mapping[str, Any], where: str) -> tuple[dict[str, str], list[str]]:
    outputs = receipt.get("outputs")
    found: dict[str, str] = {}
    problems = []
    if not isinstance(outputs, Mapping):
        return found, [f"{where} records no outputs"]
    for name, entry in sorted(outputs.items()):
        path = work / str(name)
        if "/" in str(name) or not path.is_file() or not isinstance(entry, Mapping):
            problems.append(f"{where}: output {name} is missing")
            continue
        found[str(name)] = sha256_file(path)
        if found[str(name)] != entry.get("sha256"):
            problems.append(f"{where}: output {name} changed after its step")
    return found, problems


def _fold_identity_records(work: Path, fold: int, *, freeze1_sha256: str, expected_code: Mapping[str, Any],
                           frozen_membership: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Re-verify one fold's identity receipts (exact names only); returns their SHA-256 values and the problems."""
    from ec_clean30 import identity

    receipts: dict[str, str] = {}
    outputs: dict[str, str] = {}
    problems: list[str] = []
    complement, metallo = identity.SET_COMPLEMENT, identity.SET_METALLO_TRAIN

    def record(name: str) -> None:
        receipts[name] = sha256_file(work / name)

    def load(what: str, loader) -> Optional[dict[str, Any]]:
        try:
            return loader()
        except (ValueError, KeyError, TypeError, OSError) as exc:
            reason = exc if isinstance(exc, ValueError) else type(exc).__name__
            problems.append(f"fold {fold}: {what} does not verify ({reason})")
            return None

    for query_set, f_required in ((metallo, None), (complement, freeze1_sha256)):
        fasta_name = f"fasta_receipt_{query_set}.json"
        fasta = load(fasta_name, lambda: identity._load_fasta_receipt(work, fold, query_set))
        search_name = f"search_{query_set}/search_receipt.json"
        search = load(search_name, lambda: identity.verify_search_receipt(work, fold, query_set, require_pinned=True))
        edges_name = f"edges_receipt_{query_set}.json"
        edges = load(edges_name, lambda: identity.load_edges_receipt(work, fold, query_set, search)) \
            if search is not None else None
        for name, receipt in ((fasta_name, fasta), (search_name, search), (edges_name, edges)):
            if receipt is not None:
                record(name)
                problems += _receipt_problems(receipt, name, fold=fold, expected_code=expected_code,
                                              freeze1_sha256=f_required)
        if fasta is not None:
            outputs.update({entry["file"]: entry["sha256"] for entry in fasta["outputs"].values()})
        if search is not None:
            outputs[search["sorted_output"]["file"]] = search["sorted_output"]["sha256"]
        if edges is not None:
            outputs.update({name: entry["sha256"] for name, entry in edges["outputs"].items()})
        if query_set == metallo:
            metallo_fasta = fasta
        else:
            complement_fasta, complement_search, complement_edges = fasta, search, edges

    components_name = f"components_receipt_{identity.SCOPE_INNER}.json"
    components = load(components_name, lambda: json.loads((work / components_name).read_text(encoding="utf-8")))
    if components is not None:
        record(components_name)
        problems += _receipt_problems(components, components_name, fold=fold, expected_code=expected_code,
                                      freeze1_sha256=None)
        found, output_problems = _outputs_problems(work, components, f"fold {fold}: {components_name}")
        outputs.update(found)
        problems += output_problems
    assignment = load(identity.INNER_ASSIGNMENT_RECEIPT, lambda: identity._load_assignment_receipt(work, fold))
    for name, receipt in ((identity.INNER_ASSIGNMENT_RECEIPT, assignment),):
        if receipt is not None:
            record(name)
            problems += _receipt_problems(receipt, name, fold=fold, expected_code=expected_code, freeze1_sha256=None)
    certificate = load(identity.INNER_CERTIFICATE,
                       lambda: json.loads((work / identity.INNER_CERTIFICATE).read_text(encoding="utf-8")))
    if certificate is not None:
        record(identity.INNER_CERTIFICATE)
        problems += _receipt_problems(certificate, identity.INNER_CERTIFICATE, fold=fold,
                                      expected_code=expected_code, freeze1_sha256=None)
        if certificate.get("certified") is not True or certificate.get("fold") != fold:
            problems.append(f"fold {fold}: {identity.INNER_CERTIFICATE} is not a passed certificate of this fold")
    for role, name in (("assignment_receipt", identity.INNER_ASSIGNMENT_RECEIPT),
                       ("certificate", identity.INNER_CERTIFICATE)):
        frozen = ((frozen_membership.get(role) or {}).get("sha256"))
        if name in receipts and receipts[name] != frozen:
            problems.append(f"fold {fold}: {name} differs from the copy Freeze 1 bound")
    for inner in range(identity.N_INNER_FOLDS):
        name = f"r1_inner_exclusions_receipt_fold{fold}_inner{inner}.json"
        if not (work / name).is_file():
            if fold == 0:
                problems.append(f"fold 0: {name} is missing")
            continue
        receipt = load(name, lambda: json.loads((work / name).read_text(encoding="utf-8")))
        if receipt is not None:
            record(name)
            problems += _receipt_problems(receipt, name, fold=fold, expected_code=expected_code, freeze1_sha256=None)
            found, output_problems = _outputs_problems(work, receipt, f"fold {fold}: {name}")
            outputs.update(found)
            problems += output_problems

    strata_name = "strata_receipt.json"
    strata = load(strata_name, lambda: json.loads((work / strata_name).read_text(encoding="utf-8")))
    if strata is not None:
        record(strata_name)
        problems += _receipt_problems(strata, strata_name, fold=fold, expected_code=expected_code,
                                      freeze1_sha256=freeze1_sha256)
        if strata.get("fold") != fold or strata.get("step") != "strata":
            problems.append(f"fold {fold}: {strata_name} is not this fold's strata receipt")
        expected_inputs = {
            "complement_nodes_sha256": (complement_fasta or {}).get("outputs", {}).get("nodes", {}).get("sha256"),
            "metallo_train_nodes_sha256": (metallo_fasta or {}).get("outputs", {}).get("nodes", {}).get("sha256"),
            "complement_search_sorted_output_sha256": ((complement_search or {}).get("sorted_output") or {}).get(
                "sha256"),
            "complement_edges_sha256": ((complement_edges or {}).get("outputs") or {}).get(
                f"edges_{complement}.tsv", {}).get("sha256"),
        }
        recorded_inputs = strata.get("inputs") or {}
        for key, value in expected_inputs.items():
            if value is None or recorded_inputs.get(key) != value:
                problems.append(f"fold {fold}: {strata_name} was built from another {key.replace('_sha256', '')}")
        strata_table = f"strata_fold{fold}.tsv"
        found, output_problems = _outputs_problems(work, strata, f"fold {fold}: {strata_name}")
        if strata_table not in found:
            output_problems.append(f"fold {fold}: {strata_name} records no {strata_table}")
        outputs.update(found)
        problems += output_problems
    nodes_sha = (metallo_fasta or {}).get("outputs", {}).get("nodes", {}).get("sha256")
    clusters_sha = (strata or {}).get("inputs", {}).get("clusters_sha256")
    return {"receipts": dict(sorted(receipts.items())), "outputs": dict(sorted(outputs.items())),
            "metallo_nodes_sha256": nodes_sha, "clusters_sha256": clusters_sha}, problems


def build_freeze1b(*, freeze1: str | os.PathLike, freeze1_sha256: str, identity_root: str | os.PathLike,
                   fold_work: Any = None, verify_receipt: str | os.PathLike, identity_head: str,
                   repo: Path = REPO_ROOT, provenance_block: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The Freeze-1b manifest (see the module docstring); raises ``Freeze1Error`` with every problem found."""
    from ec_clean30 import identity
    from ec_clean30.r2_lists import R2ListError, fold_work_dirs

    repo = Path(repo)
    digest = _sha256_argument(freeze1_sha256, "--freeze1-sha256")
    document_path = Path(freeze1)
    _require(document_path.is_file() and not document_path.is_symlink(), "--freeze1 is not a regular file.")
    _require(sha256_file(document_path) == digest, "--freeze1 does not hash to --freeze1-sha256.")
    document = _read_json(document_path, "--freeze1")
    _require(isinstance(document, Mapping) and document.get("format") == FREEZE1_DOCUMENT_FORMAT
             and document.get("kind") == "freeze1" and document.get("campaign_id") == CAMPAIGN_ID,
             f"--freeze1 is not a {FREEZE1_DOCUMENT_FORMAT} document of {CAMPAIGN_ID}.")
    problems: list[str] = []
    receipt_path = Path(verify_receipt)
    _require(receipt_path.is_file() and not receipt_path.is_symlink(), "--verify-freeze1 is not a regular file.")
    receipt = _read_json(receipt_path, "--verify-freeze1")
    if not (isinstance(receipt, Mapping) and receipt.get("format") == VERIFY_FREEZE1_FORMAT
            and receipt.get("status") == "passed" and receipt.get("freeze1_sha256") == digest
            and receipt.get("pushed") is True
            and (receipt.get("origin_main") or {}).get("contains_introducing_commit") is True):
        problems.append("--verify-freeze1 is not a passed verify-freeze1 receipt of this SHA-256 with the commit on "
                        "origin/main")
    live = verify_freeze1(sha256=digest, document=document_path, require_pushed=True, repo=repo)
    problems += [f"verify-freeze1 now: {problem}" for problem in live["problems"]]
    if isinstance(receipt, Mapping) and (receipt.get("document") != live["document"]
                                         or receipt.get("introducing_commit") != live["introducing_commit"]):
        problems.append("--verify-freeze1 names another document or logging commit than the check made now")

    head = _resolve_commit(repo, "HEAD")
    pins = ((document.get("code") or {}).get("tracked_file_sha256")) or {}
    expected_code = {path: pins.get(path) for path in IDENTITY_CODE_FILES}
    if any(value is None for value in expected_code.values()):
        problems.append("Freeze 1 pins no SHA-256 for every identity code file")
    r_id_live = {path: sha256_file(repo / path) if (repo / path).is_file() else None for path in R_ID_FILES}
    for path in R_ID_FILES:
        if pins.get(path) is None or r_id_live[path] != pins.get(path):
            problems.append(f"the R-ID file {path} differs from its Freeze-1 pin")
    full_head, head_problems = _identity_head_problems(repo, identity_head, head, _show(repo, "HEAD", LOG_PATH))
    problems += head_problems

    identity_root = Path(identity_root)
    _require(identity_root.is_dir(), f"--identity-root is not a folder: {identity_root}")
    try:
        works = fold_work_dirs(identity_root, fold_work)
    except R2ListError as exc:
        raise Freeze1Error(f"--fold-work: {exc}") from None
    provenance.refuse_test_paths(works.values(), what="--fold-work folder")
    frozen_memberships = document.get("inner_membership") or {}
    folds: dict[str, Any] = {}
    for fold, work in sorted(works.items()):
        records, fold_problems = _fold_identity_records(
            work, fold, freeze1_sha256=digest, expected_code=expected_code,
            frozen_membership=frozen_memberships.get(str(fold)) or {})
        records["work"] = (_real(work).relative_to(_real(identity_root.parent)).as_posix()
                           if _is_under(work, identity_root.parent) else work.name)
        folds[str(fold)] = records
        problems += fold_problems

    cohort_dir = identity_root / "cohort"
    cohort_receipt_name = f"components_receipt_{identity.SCOPE_COHORT}.json"
    clusters_name = f"components_{identity.SCOPE_COHORT}.tsv"
    cohort_record: dict[str, Any] = {}
    if not (cohort_dir / cohort_receipt_name).is_file() or not (cohort_dir / clusters_name).is_file():
        problems.append(f"the cohort folder holds no {cohort_receipt_name} or {clusters_name}")
    else:
        cohort_receipt = _read_json(cohort_dir / cohort_receipt_name, cohort_receipt_name)
        problems += _receipt_problems(cohort_receipt, cohort_receipt_name, fold=None, expected_code=expected_code,
                                      freeze1_sha256=digest)
        clusters_sha = sha256_file(cohort_dir / clusters_name)
        for fold, records in folds.items():
            try:
                identity._load_cohort_clusters(cohort_dir / clusters_name, digest, int(fold),
                                               records["metallo_nodes_sha256"])
            except (ValueError, KeyError, TypeError) as exc:
                problems.append(f"fold {fold}: the cohort components do not verify "
                                f"({exc if isinstance(exc, ValueError) else type(exc).__name__})")
            if records["clusters_sha256"] != clusters_sha:
                problems.append(f"fold {fold}: the strata receipt was built from other cohort clusters")
        cohort_record = {"receipts": {cohort_receipt_name: sha256_file(cohort_dir / cohort_receipt_name)},
                         "outputs": {clusters_name: clusters_sha}}
    _refuse_problems("Freeze 1b does not hold.", problems)
    for records in folds.values():
        records.pop("metallo_nodes_sha256", None)
        records.pop("clusters_sha256", None)
    return {
        "format": FREEZE1B_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "kind": "freeze1b",
        "status": "passed",
        "freeze1_sha256": digest,
        "freeze1_document": live["document"],
        "verify_freeze1_receipt_sha256": sha256_file(receipt_path),
        "verify_freeze1_introducing_commit": live["introducing_commit"],
        "identity_head": full_head,
        "head": head,
        "identity_code_sha256": expected_code,
        "r_id_files_sha256": r_id_live,
        "folds": folds,
        "cohort": cohort_record,
        "provenance": dict(provenance_block) if provenance_block is not None else None,
    }


def freeze1b_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    """What the action prints: counts of folds, receipts and output files (no names, no per-fold numbers)."""
    folds = manifest["folds"].values()
    return {"folds": len(manifest["folds"]),
            "receipts": sum(len(item["receipts"]) for item in folds) + len(manifest["cohort"].get("receipts", {})),
            "outputs": sum(len(item["outputs"]) for item in folds) + len(manifest["cohort"].get("outputs", {}))}


# ---------------------------------------------------------------------------
# Runner arguments and handlers
# ---------------------------------------------------------------------------

def add_manifest_freeze_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``manifest`` flags of the real ``freeze1``, ``freeze1-document`` and ``freeze1b`` kinds."""
    parser.add_argument("--plans-dir", default=None,
                        help="Plan records <dir>/fold<k>.json and their .receipt.json (real freeze1; checked against "
                             "the live plans).")
    parser.add_argument("--inputs", default=None, help="Real Freeze-1 inputs manifest (freeze1-document).")
    parser.add_argument("--inputs-sha256", default=None, help="Its SHA-256 (freeze1-document).")
    parser.add_argument("--decisions", default=None, help="freeze1_decisions.json in the campaign docs "
                                                          "(freeze1-document).")
    parser.add_argument("--overlay", default=None, help="The overlay manifest (freeze1-document).")
    parser.add_argument("--labels", default=None, help="<campaign root>/labels: label table, sequence map and transfer "
                                                       "list folders (freeze1-document).")
    parser.add_argument("--cohort", default=None, help="Cohort audit folder (freeze1-document).")
    parser.add_argument("--r2-lists", default=None, help="R2 lists folder (freeze1-document).")
    parser.add_argument("--simulation", default=None, help="Final simulation sweep folder (freeze1-document).")
    parser.add_argument("--environment", default=None, help="Environment exports folder (freeze1-document).")
    parser.add_argument("--pin", action="append", default=[], metavar="NAME=SHA256",
                        help=f"A pinned SHA-256 (freeze1-document; repeatable; needs {', '.join(REQUIRED_PINS)}).")
    parser.add_argument("--no-transfer-list", action="store_true",
                        help="No transfer list was approved (decision A8): record 'no list supplied' instead of "
                             "requiring <labels>/ec_transfer_list (freeze1-document).")
    parser.add_argument("--freeze1", default=None, help="The committed Freeze-1 document (freeze1b).")
    parser.add_argument("--freeze1-sha256", default=None, help="Its SHA-256 F (freeze1b).")
    parser.add_argument("--identity-root", default=None, help="<campaign root>/identity (freeze1b).")
    parser.add_argument("--fold-work", action="append", default=[], metavar="K=PATH",
                        help="Identity folder of fold K instead of <identity root>/fold<K> (freeze1b; repeatable).")
    parser.add_argument("--verify-freeze1", default=None, help="A passed, pushed verify-freeze1 receipt (freeze1b).")
    parser.add_argument("--identity-head", default=None, help="The P2-1 HEAD named in the campaign log (freeze1b).")
    return parser


def add_verify_freeze1_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``verify-freeze1`` action's arguments; the handler is ``action_verify_freeze1``."""
    parser.add_argument("--sha256", required=True, help="The SHA-256 F of the Freeze-1 document.")
    parser.add_argument("--document", required=True, help="The committed Freeze-1 document.")
    parser.add_argument("--require-pushed", action="store_true",
                        help="Fail unless refs/remotes/origin/main contains the commit that logged F (no fetch).")
    parser.add_argument("--out", required=True, help="New receipt file (written only when the check passes).")
    return parser


FREEZE1_DOCUMENT_FLAGS: tuple[str, ...] = ("inputs", "inputs_sha256", "decisions", "overlay", "labels", "cohort",
                                           "r2_lists", "simulation", "environment", "pin", "no_transfer_list")
FREEZE1B_FLAGS: tuple[str, ...] = ("freeze1", "freeze1_sha256", "identity_root", "fold_work", "verify_freeze1",
                                   "identity_head")
_FLAGS_BY_HANDLER: Mapping[str, tuple[str, ...]] = {
    "freeze1": ("plans_dir",), "freeze1_document": FREEZE1_DOCUMENT_FLAGS, "freeze1b": FREEZE1B_FLAGS}


def route_manifest(args: argparse.Namespace):
    """The handler of this module for a ``manifest`` call, or ``None`` for the runner's own path.

    ``None`` (the existing ``action_manifest``) for ``freeze2`` and for a smoke ``freeze1``
    without ``--plans-dir``; ``action_manifest_freeze1`` for every real ``freeze1`` (it needs
    ``--plans-dir``, so a real manifest never skips the completeness checks) and for any
    ``freeze1`` given ``--plans-dir``; ``action_manifest_freeze1_document`` and
    ``action_manifest_freeze1b`` for their kinds. A flag of another kind is refused.
    """
    kind = getattr(args, "kind", None)
    if kind == KIND_FREEZE1_DOCUMENT:
        handler, name = action_manifest_freeze1_document, "freeze1_document"
    elif kind == KIND_FREEZE1B:
        handler, name = action_manifest_freeze1b, "freeze1b"
    elif kind == KIND_FREEZE1 and (getattr(args, "plans_dir", None) or not getattr(args, "smoke", False)):
        handler, name = action_manifest_freeze1, "freeze1"
    else:
        handler, name = None, None
    allowed = _FLAGS_BY_HANDLER.get(name, ())
    foreign = [f"--{flag.replace('_', '-')}" for flags in _FLAGS_BY_HANDLER.values() for flag in flags
               if flag not in allowed and getattr(args, flag, None)]
    _require(not foreign, f"--kind {kind}{' --smoke' if getattr(args, 'smoke', False) else ''} does not take "
                          f"{' '.join(sorted(set(foreign)))}.")
    return handler


def _refuse_test_named(args: argparse.Namespace, names: Sequence[str]) -> None:
    for name in names:
        value = getattr(args, name, None)
        values = value if isinstance(value, list) else [value]
        provenance.refuse_test_paths(values, what=f"--{name.replace('_', '-')}")


def _needs(args: argparse.Namespace, names: Sequence[str], kind: str) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if not getattr(args, name, None)]
    _require(not missing, f"--kind {kind} needs {' '.join(missing)}.")


def _real_provenance() -> dict[str, Any]:
    git = provenance.real_action_git_state(smoke=False, repo=REPO_ROOT)
    return provenance.provenance_record(smoke=False, git=git, repo=REPO_ROOT)


def action_manifest_freeze1(args: argparse.Namespace) -> int:
    """``manifest --kind freeze1 --plans-dir DIR --view ... --out FILE``: the real, complete inputs manifest."""
    _require(not getattr(args, "smoke", False), "--plans-dir builds a real Freeze-1 manifest; a smoke manifest takes "
                                                "one view and no --plans-dir.")
    _needs(args, ["plans_dir", "out"], KIND_FREEZE1)
    _refuse_test_named(args, ["out", "view", "plans_dir"])
    manifest = build_freeze1_inputs(args.view or [], plans_dir=args.plans_dir, repo=REPO_ROOT,
                                    provenance_block=_real_provenance())
    campaign_root = manifest["completeness"]["campaign_root"]
    problem = campaign_identity_problem(args.out, campaign_root, "--out")
    _require(problem is None, problem or "")
    digest = provenance.write_new_json(Path(args.out), manifest)
    _print({"manifest": str(args.out), "kind": KIND_FREEZE1, "sha256": digest, "folds": len(manifest["views"]),
            "r_cmd": manifest["completeness"]["r_cmd"]["status"],
            "tracked_code_files": len(manifest["completeness"]["tracked_code_sha256"])})
    return 0


def action_manifest_freeze1_document(args: argparse.Namespace) -> int:
    """``manifest --kind freeze1-document ... --out freeze1.json``: the document and its rendering."""
    names = ["inputs", "inputs_sha256", "decisions", "overlay", "labels", "cohort", "r2_lists", "simulation",
             "environment", "out"]
    _needs(args, names, KIND_FREEZE1_DOCUMENT)
    _refuse_test_named(args, [name for name in names if name != "inputs_sha256"])
    _real_provenance()  # the clean-tree check; the document itself records no machine detail
    document = build_freeze1_document(
        inputs=args.inputs, inputs_sha256=args.inputs_sha256, decisions=args.decisions, overlay_manifest=args.overlay,
        labels=args.labels, cohort=args.cohort, r2_lists=args.r2_lists, simulation=args.simulation,
        environment=args.environment, pins=args.pin, no_transfer_list=bool(getattr(args, "no_transfer_list", False)),
        repo=REPO_ROOT)
    digest, rendering = write_freeze1_document(document, args.out)
    _print({"document": str(args.out), "rendering": str(rendering), "kind": KIND_FREEZE1_DOCUMENT, "sha256": digest,
            "decisions": len(document["decisions"]["answers"]), "bound_files": len(document["bound_files"]),
            "transfer_list": document["transfer_list"]})
    return 0


def action_manifest_freeze1b(args: argparse.Namespace) -> int:
    """``manifest --kind freeze1b ... --out FILE``: the Freeze-1b manifest; prints counts only."""
    names = ["freeze1", "freeze1_sha256", "identity_root", "verify_freeze1", "identity_head", "out"]
    _needs(args, names, KIND_FREEZE1B)
    _refuse_test_named(args, ["freeze1", "identity_root", "verify_freeze1", "out"])
    provenance.refuse_test_paths([str(value).partition("=")[2].strip() or str(value) for value in args.fold_work or ()],
                                 what="--fold-work")
    manifest = build_freeze1b(freeze1=args.freeze1, freeze1_sha256=args.freeze1_sha256,
                              identity_root=args.identity_root, fold_work=args.fold_work or None,
                              verify_receipt=args.verify_freeze1, identity_head=args.identity_head, repo=REPO_ROOT,
                              provenance_block=_real_provenance())
    digest = provenance.write_new_json(Path(args.out), manifest)
    _print({"manifest": str(args.out), "kind": KIND_FREEZE1B, "sha256": digest, **freeze1b_counts(manifest)})
    return 0


def action_verify_freeze1(args: argparse.Namespace) -> int:
    """``verify-freeze1``: 0 and a new receipt when the check passes, 1 (no receipt) when it fails."""
    _refuse_test_named(args, ["document", "out"])
    block = _real_provenance()
    record = verify_freeze1(sha256=args.sha256, document=args.document, require_pushed=args.require_pushed,
                            repo=REPO_ROOT)
    shown = {key: record[key] for key in ("status", "freeze1_sha256", "document", "introducing_commit", "pushed",
                                          "require_pushed", "problems")}
    if record["status"] != "passed":
        _print(shown)
        return 1
    if os.path.lexists(args.out):
        raise FileExistsError(f"{args.out} exists; a verify-freeze1 receipt is written once.")
    digest = provenance.write_new_json(Path(args.out), {**record, "provenance": block})
    _print({**shown, "receipt": str(args.out), "receipt_sha256": digest})
    return 0


# ---------------------------------------------------------------------------
# First, narrowly scoped S1 amendment. Original freeze documents remain immutable.
# ---------------------------------------------------------------------------

S1_AMENDMENT_FORMAT = "deepmzyme_ec_clean30_s1_amendment_v1"
S1_AMENDMENT_ID = "s1-fixed-anchor-holm-v1"
S1_AMENDMENT_SOURCE_FILES = (
    "src/ec_clean30/selection.py", "src/ec_clean30/selector.py", "src/ec_clean30/freeze.py",
    "scripts/run_ec_clean30_campaign.py", "src/ec_clean30/s1_simulation.py",
)
S1_AMENDMENT_EVIDENCE = ("simulation_spec", "simulation_summary", "verification")
S1_AMENDMENT_CHECKS = ("focused_tests", "cpu_smoke", "regression", "docs_contract", "dry_run_reuse")


def _amendment_repo_file(path: str | os.PathLike, repo: Path, what: str, *, exists: bool = True,
                         document: bool = False) -> tuple[Path, str]:
    """Check repository containment and symlinks before any amendment input is read."""
    repo = _real(repo)
    given = Path(path)
    absolute = Path(os.path.abspath(os.fspath(given if given.is_absolute() else repo / given)))
    _require(absolute == repo or repo in absolute.parents, f"{what} must be inside the repository.")
    relative = absolute.relative_to(repo)
    _require(relative.parts and ".." not in relative.parts, f"{what} has an unsafe path.")
    cursor = repo
    for part in relative.parts:
        cursor /= part
        _require(not cursor.is_symlink(), f"{what} must not traverse a symlink.")
    name = relative.as_posix()
    if document:
        _require(name.startswith(CAMPAIGN_DOCS_DIR + "/") and absolute.suffix == ".json",
                 f"{what} must be a JSON file under {CAMPAIGN_DOCS_DIR}.")
    else:
        _require(any(name.startswith(folder + "/") for folder in TRACKED_CODE_FOLDERS),
                 f"{what} must be a src/ or scripts/ file.")
    if exists:
        _require(absolute.is_file(), f"{what} is not a regular file: {name}")
    return absolute, name


def _amendment_code(repo: Path) -> dict[str, str]:
    names = _git_out(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
                     *TRACKED_CODE_FOLDERS, what="amendment source files")
    return {name: sha256_file(_amendment_repo_file(name, repo, "source file")[0])
            for name in sorted(set(filter(None, names.split("\0"))))}


def _amendment_base(path: Path, digest: str, repo: Path) -> dict[str, Any]:
    path, name = _amendment_repo_file(path, repo, "base Freeze 1", document=True)
    _require(name == f"{CAMPAIGN_DOCS_DIR}/freeze1.json", "The amendment must bind the original Freeze-1 path.")
    _require(sha256_file(path) == digest, "The base Freeze-1 hash differs.")
    base = _read_json(path, "base Freeze 1")
    _require(isinstance(base, dict) and base.get("format") == FREEZE1_DOCUMENT_FORMAT
             and base.get("campaign_id") == CAMPAIGN_ID and base.get("amendments") == [],
             "The amendment needs the original, unamended Freeze-1 document.")
    published = verify_freeze1(sha256=digest, document=path, require_pushed=True, repo=repo)
    _refuse_problems("The base Freeze 1 is not published.", published["problems"])
    pins = (base.get("code") or {}).get("tracked_file_sha256")
    _require(isinstance(pins, dict) and bool(pins), "The base Freeze 1 has no code pins.")
    for name, value in pins.items():
        _amendment_repo_file(name, repo, "base source pin")
        _sha256_argument(value, f"base source pin {name}")
    return base


def _amendment_evidence(entries: Mapping[str, Any], repo: Path, policy: Mapping[str, Any],
                        sources: Mapping[str, str], *, base_sha256: str,
                        all_sources: Mapping[str, str]) -> None:
    _require(isinstance(entries, Mapping) and set(entries) == set(S1_AMENDMENT_EVIDENCE),
             "The amendment needs simulation_spec, simulation_summary and verification evidence.")
    loaded = {}
    for role, entry in entries.items():
        _require(isinstance(entry, Mapping) and set(entry) == {"path", "sha256"}, f"Malformed {role} evidence.")
        path, _ = _amendment_repo_file(entry["path"], repo, role, document=True)
        _require(sha256_file(path) == _sha256_argument(entry["sha256"], role), f"{role} evidence hash differs.")
        loaded[role] = _read_json(path, role)
        _require(isinstance(loaded[role], dict), f"{role} evidence must be an object.")
    spec, summary, verification = (loaded[name] for name in S1_AMENDMENT_EVIDENCE)
    _require(spec.get("policy") == policy and isinstance(spec.get("scenarios"), list) and bool(spec["scenarios"]),
             "Simulation specification does not declare this policy and its scenarios.")
    _require(summary.get("format") == "deepmzyme_s1_characterization_summary_v1"
             and summary.get("status") == "complete" and summary.get("policy") == policy,
             "Simulation summary must completely characterize this fixed policy.")
    _require(spec.get("base_freeze1_sha256") == base_sha256
             and summary.get("base_freeze1_sha256") == base_sha256,
             "Simulation evidence names another base Freeze 1.")
    _require(summary.get("spec_sha256") == entries["simulation_spec"]["sha256"],
             "Simulation summary names another specification.")
    simulation_sources = summary.get("source_sha256") or {}
    _require(isinstance(simulation_sources, dict)
             and all(simulation_sources.get(name) == sources[name]
                     for name in ("src/ec_clean30/selection.py", "src/ec_clean30/s1_simulation.py"))
             and all(name in all_sources and digest == all_sources[name]
                     for name, digest in simulation_sources.items()),
             "Simulation summary does not bind the current selection and simulation source.")
    scenarios, results = spec["scenarios"], summary.get("results")
    _require(all(isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] for row in scenarios)
             and isinstance(results, list)
             and all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in results),
             "Simulation summary must contain the specified scenario results.")
    scenario_ids, result_ids = [row["id"] for row in scenarios], [row["id"] for row in results]
    _require(len(set(scenario_ids)) == len(scenario_ids) and len(set(result_ids)) == len(result_ids)
             and set(result_ids) == set(scenario_ids),
             "Simulation summary has missing, duplicate, or extra scenarios.")
    _require(type(spec.get("n_replicates")) is int and spec["n_replicates"] == 200
             and type(summary.get("n_scenarios")) is int and summary["n_scenarios"] == len(scenario_ids),
             "Simulation scenario or replicate totals differ from the fixed specification.")
    total = 0
    for row in results:
        count, winners = row.get("n_replicates"), row.get("winner_counts")
        _require(type(count) is int and count == spec["n_replicates"]
                 and isinstance(winners, dict) and set(winners) == {"F2", "F3", "F5", "F4"}
                 and all(type(value) is int and value >= 0 for value in winners.values())
                 and sum(winners.values()) == count,
                 "Simulation result counts do not cover every replicate exactly once.")
        _sha256_argument(row.get("record_sha256"), "simulation result record hash")
        total += count
    _require(type(summary.get("n_replicates")) is int and summary["n_replicates"] == total,
             "Simulation summary replicate total differs from its results.")
    checks = verification.get("checks")
    _require(verification.get("status") == "passed" and verification.get("source_sha256") == dict(sources)
             and isinstance(checks, dict) and all(isinstance(checks.get(name), dict)
                                                 and checks[name].get("status") == "passed"
                                                 for name in S1_AMENDMENT_CHECKS),
             "Verification evidence must pass and bind every amended source file.")


def build_s1_amendment(*, freeze1_path: str | os.PathLike, evidence: Mapping[str, Path],
                        out: str | os.PathLike, repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Write a source-pinned draft once. Publication is checked separately before real execution."""
    from ec_clean30.selection import AMENDED_S1_RULE, s1_policy_record

    repo = _real(repo)
    base_path, base_name = _amendment_repo_file(freeze1_path, repo, "base Freeze 1", document=True)
    out_path, _ = _amendment_repo_file(out, repo, "amendment output", exists=False, document=True)
    rendering = out_path.with_suffix(".md")
    _require(not os.path.lexists(out_path) and not os.path.lexists(rendering), "Amendment files are written once.")
    digest = sha256_file(base_path)
    base = _amendment_base(base_path, digest, repo)
    base_pins = base["code"]["tracked_file_sha256"]
    current = _amendment_code(repo)
    _require(set(S1_AMENDMENT_SOURCE_FILES) <= set(current), "An amended source file is missing.")
    changes = {name: {"before": base_pins.get(name), "after": current[name]}
               for name in S1_AMENDMENT_SOURCE_FILES}
    expected = {**base_pins, **{name: row["after"] for name, row in changes.items()}}
    _require(current == expected, "Unamended source files changed, disappeared, or were added outside the S1 scope.")
    policy = s1_policy_record()
    records = {}
    for role, path in evidence.items():
        checked, name = _amendment_repo_file(path, repo, f"{role} evidence", document=True)
        records[role] = {"path": name, "sha256": sha256_file(checked)}
    _amendment_evidence(records, repo, policy, {name: current[name] for name in S1_AMENDMENT_SOURCE_FILES},
                        base_sha256=digest, all_sources=current)
    parent = _resolve_commit(repo, "HEAD")
    _require(parent is not None, "Amendment source needs a repository parent commit.")
    document = {
        "format": S1_AMENDMENT_FORMAT, "schema_version": 1, "campaign_id": CAMPAIGN_ID,
        "amendment_id": S1_AMENDMENT_ID, "base_freeze1": {"path": base_name, "sha256": digest},
        "previous_sha256": digest, "sequence": 1, "s1_rule": AMENDED_S1_RULE, "policy": policy,
        "source_parent_commit": parent, "source_changes": changes, "evidence": records,
        "publication": "required_before_execution",
        "reused_artifacts": ["original Freeze 1 and Freeze 1b", "five fit plans and dry runs",
                             "overlay and five views", "inner memberships and identity audit",
                             "final and block-26 embeddings", "historical simulations"],
    }
    provenance.write_new_json(out_path, document)
    with rendering.open("x", encoding="utf-8") as handle:
        handle.write(render_s1_amendment_markdown(document))
    return document


def verify_s1_amendment(path: str | os.PathLike, *, expected_freeze1_sha256: str,
                         repo: Path = REPO_ROOT, require_published: bool = True,
                         expected_chain: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Authenticate the first S1 amendment, all original/changed source pins and bound evidence.

    ``require_published=False`` is only for draft review and synthetic tests. It never makes a
    draft eligible for real selection or execution. No private identity artifact is read.
    """
    from ec_clean30.selection import AMENDED_S1_RULE, s1_policy_record

    repo = _real(repo)
    path, _ = _amendment_repo_file(path, repo, "amendment", document=True)
    document = _read_json(path, "S1 amendment")
    _require(isinstance(document, dict) and document.get("format") == S1_AMENDMENT_FORMAT
             and document.get("schema_version") == 1 and document.get("campaign_id") == CAMPAIGN_ID
             and document.get("amendment_id") == S1_AMENDMENT_ID, "Unknown S1 amendment format or identity.")
    _require(path.read_bytes() == canonical_bytes(document), "S1 amendment is not canonical JSON.")
    digest = sha256_file(path)
    chain = [digest]
    _require(expected_chain is None or list(expected_chain) == chain, "S1 amendment chain differs or is reordered.")
    base_digest = _sha256_argument(expected_freeze1_sha256, "expected Freeze-1 SHA-256")
    base_record = document.get("base_freeze1") or {}
    _require(base_record.get("sha256") == base_digest and document.get("previous_sha256") == base_digest
             and document.get("sequence") == 1, "S1 amendment has the wrong base or predecessor.")
    base_path, _ = _amendment_repo_file(base_record.get("path", ""), repo, "base Freeze 1", document=True)
    base = _amendment_base(base_path, base_digest, repo)
    _require(document.get("s1_rule") == AMENDED_S1_RULE and document.get("policy") == s1_policy_record(),
             "S1 amendment policy differs from the implemented fixed policy.")
    _require(document.get("publication") == "required_before_execution", "S1 publication requirement differs.")
    parent = document.get("source_parent_commit")
    head = _resolve_commit(repo, "HEAD")
    _require(isinstance(parent, str) and _resolve_commit(repo, parent) == parent
             and head is not None and _is_ancestor(repo, parent, head), "S1 amendment source parent is not an ancestor.")
    changes = document.get("source_changes")
    _require(isinstance(changes, dict) and set(changes) == set(S1_AMENDMENT_SOURCE_FILES),
             "S1 amendment source scope differs.")
    expected = dict(base["code"]["tracked_file_sha256"])
    sources = {}
    for name, change in changes.items():
        _require(isinstance(change, dict) and set(change) == {"before", "after"}
                 and change["before"] == expected.get(name), f"S1 amendment original source pin differs: {name}")
        sources[name] = _sha256_argument(change["after"], f"amended source pin {name}")
        expected[name] = sources[name]
    _require(_amendment_code(repo) == expected, "Live source differs from the authenticated amendment scope.")
    _amendment_evidence(document.get("evidence"), repo, document["policy"], sources,
                        base_sha256=base_digest, all_sources=expected)
    rendering = path.with_suffix(".md")
    _require(rendering.is_file() and not rendering.is_symlink()
             and rendering.read_text(encoding="utf-8") == render_s1_amendment_markdown(document),
             "S1 amendment rendering differs from the canonical JSON.")
    publication = verify_freeze1(sha256=digest, document=path, require_pushed=True, repo=repo)
    origin = _resolve_commit(repo, ORIGIN_MAIN_REF)
    if origin is None or not _is_ancestor(repo, head, origin):
        publication["problems"].append("the current source HEAD is not published on refs/remotes/origin/main")
        publication["status"] = "failed"
    published = publication["status"] == "passed"
    if require_published:
        provenance.require_clean_tree(repo)
        _refuse_problems("The S1 amendment is not published; real execution is refused.", publication["problems"])
    return {"document_sha256": digest, "amendment_chain": chain, "s1_rule": AMENDED_S1_RULE,
            "policy": document["policy"], "published": published, "document": document}


def render_s1_amendment_markdown(document: Mapping[str, Any]) -> str:
    """Compact public rendering; the canonical JSON and existing Freeze 1 own all pins."""
    digest = sha256_bytes(canonical_bytes(document))
    lines = ["# S1 fixed-anchor family selection amendment", "",
             f"- Amendment SHA-256: `{digest}`",
             f"- Original Freeze 1: `{document['base_freeze1']['sha256']}`",
             f"- Rule: `{document['s1_rule']}`; publication is required before execution.", "",
             "The original freeze documents and prepared data remain unchanged. This amendment changes only S1",
             "family selection and the evidence/authentication needed to execute that rule. Historical selection",
             "remains available for replay. Simulation describes the fixed rule; it does not tune alpha or establish",
             "real-model superiority. R2 scoring and all secondary multiplicity settings remain unchanged.", "",
             "## Fixed policy", "", "```json", json.dumps(document["policy"], indent=2, sort_keys=True), "```", "",
             "## Evidence", "", "| Role | File | SHA-256 |", "|---|---|---|"]
    for role, entry in document["evidence"].items():
        lines.append(f"| {role} | `{entry['path']}` | `{entry['sha256']}` |")
    return "\n".join(lines) + "\n"
