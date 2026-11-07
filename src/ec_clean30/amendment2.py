"""Amendment-2 record tools of the EC class versus CLEAN30 campaign (science side; rules A2-GOV-1, A2-GOV-4, A2-SEL-9).

- ``write_chain_receipt`` (runner action ``verify-amendment-chain``): authenticates the amendment chain on the
  workstation's full clone through the general chain verifier (``ec_clean30.amendment_chain.chain_receipt_record``
  around ``verify_chain``, science side, contract X2; refused without it) and writes the receipt once. It is keyed
  by the tip's SHA-256, HEAD, the SHA-256 of ``git status --porcelain=v1 --untracked-files=all``, ``origin/main``,
  the SHA-256 of the verifier files, ``require_published`` and ``expected_chain``, holds the full verifier result
  and its own SHA-256, and is re-validated on a pod and per unit by ``amendment_chain.check_chain_receipt``. A
  refusal of the verifier is a refusal of the receipt. The scoring pass reads it as declared JSON
  (``scoring.check_amendment2_bindings``).
- ``write_refit_recipe_receipt`` (``refit-recipe``): the verified refit recipe of one outer fold
  (``selector.verified_refit_recipe``) with, in fold 0, the binding futility ``continue``; the scoring pass refuses a
  refit or a tau that differs from it (audit P7).
- ``dry_run_v2`` (``dry-run-v2``): the repeated plan-v2 dry run of one view (``commands.dry_run`` with
  ``plan_version=2``, ``write=False`` and the declared pretraining-input stub); its receipt is written once outside
  the view.
- ``check_decisions_file``, ``build_amendment2`` and ``render_amendment2`` (``amend-2``): the maintainer's decisions
  file (C1-C8 and C6b), the canonical record ``amendment.json`` (written once from the reviewed draft record
  ``amendment_draft.json``, the rule text of the reviewed document ``amendment_draft.md``, the decisions file, the
  fixtures, the evidence and the build inputs, each bound by SHA-256) and its rendering ``amendment.md``. The
  builder reuses ``freeze._PLACEHOLDER_TOKEN_RE`` and ``freeze._ABSOLUTE_PATH_RE``.

The record has the shape of the contract fixture (``tests/fixtures/ec_clean30_amendment2_contract.json``,
``amendment_record.required_keys`` and ``record_shape``; ``record_builder`` for the draft keys, the values
computed at build, the required bindings and the build inputs). Values that depend on execution-side code (the
execution file set and the pinned function closures) are computed from the live tree at build, never taken from
the draft; the derived execution file set must equal the draft's declared closed list. The plan-v2 records, the
dry-run-v2 receipts and every role of ``amendment_rules.REQUIRED_BINDINGS`` are build inputs, and the builder
refuses without them.

Private bindings (the repository is public): the files of ``amendment_rules.PRIVATE_ROLES`` (the dry-run-v2
receipts, the CLEAN tie exposure and the F6 hold-out lists and corpus) are bound as
``{"path", "sha256", "private": true}``. Each is a regular file below ``amendment_rules.PRIVATE_BINDING_ROOT`` of
the repository, reached without a symbolic link, untracked and ignored by git; only its path and SHA-256 enter the
record and the rendering, never its content. Every other bound file is public and committed with the link.

Receipt and record formats are fixed here; ``scoring`` repeats the two receipt format names (a test keeps them
equal) so that the scorer imports nothing from this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

from ec_clean30 import amendment_rules
from ec_clean30.provenance import (
    provenance_record, real_action_git_state, refuse_test_paths, sha256_file, write_new_json,
)

CHAIN_RECEIPT_FORMAT = "deepmzyme_ec_clean30_amendment_chain_receipt_v1"
REFIT_RECIPE_RECEIPT_FORMAT = "deepmzyme_ec_clean30_refit_recipe_receipt_v1"
DRY_RUN_V2_RECEIPT_FORMAT = "deepmzyme_ec_clean30_dry_run_v2_receipt_v1"
DECISIONS_FORMAT = amendment_rules.DECISIONS_FORMAT
RECORD_FORMAT = "deepmzyme_ec_clean30_amendment_v2"
DRAFT_FORMAT = "deepmzyme_ec_clean30_amendment_v2_draft"
CONFIRMATION_IDS: tuple[str, ...] = amendment_rules.CONFIRMATION_IDS
RULE_TEXT_START = "## Rule text"
RULE_TEXT_END = "## Policy deviations"
SCHEMA_VERSION = 1
PLAN_V2_FORMAT = "deepmzyme_ec_clean30_fold_plan_v2"
N_FOLDS = 5
# The record (contract fixture: amendment_record.required_keys and record_shape.built_keys; a test keeps them equal).
RECORD_REQUIRED_KEYS: tuple[str, ...] = (
    "format", "schema_version", "amendment_id", "sequence", "change_class", "base_freeze1", "base_freeze1b_sha256",
    "base_amendment", "previous_sha256", "chain_before", "timestamps", "data_state", "decisions", "rules",
    "supersedes", "plans", "dry_runs", "source_parent_commit", "execution_file_set", "pinned_execution_functions",
    "post_fit_analysis_files", "pinned_analysis_functions", "live_constants_sha256", "source_changes",
    "unchanged_scientific_files", "inputs", "deferred_bindings", "fixtures", "evidence", "deviations", "publication",
)
RECORD_BUILT_KEYS: tuple[str, ...] = (
    "confirmation_status", "provisional_markers_settled", "decisions_file", "bound_files", "rule_text",
    "rule_policy_sha256",
)
# Sections whose {"path", "sha256"} entries are file bindings (record_shape.file_binding_sections).
FILE_BINDING_SECTIONS: tuple[str, ...] = ("bound_files", "fixtures", "evidence", "inputs", "plans", "dry_runs",
                                         "data_state")
DECISIONS_BINDING = "decisions"
# The reviewed draft (contract fixture: record_builder).
DRAFT_KEYS: tuple[str, ...] = (
    "schema_version", "campaign_id", "amendment_id", "sequence", "change_class", "base_freeze1",
    "base_freeze1b_sha256", "base_amendment", "previous_sha256", "chain_before", "timestamps", "data_state",
    "decisions", "rules", "supersedes", "plans", "dry_runs", "source_parent_commit", "pinned_execution_functions",
    "post_fit_analysis_files", "pinned_analysis_functions", "source_changes", "unchanged_scientific_files", "inputs",
    "deferred_bindings", "fixtures", "evidence", "deviations", "publication", "execution_file_set",
)
DRAFT_SUBKEYS: Mapping[str, tuple[str, ...]] = {
    "base_freeze1": ("path", "sha256"),
    "base_amendment": ("sequence", "amendment_id", "path", "sha256", "commit"),
    "data_state": ("fold0_records_at_draft", "g2_completed_units"),
    "plans": ("v1_plan_sha256",),
    "dry_runs": ("rule", "in_view_receipts_unchanged"),
    "fixtures": ("refusals", "decisions"),
    "execution_file_set": ("declared", "pending"),
}
DRAFT_FUNCTION_NAMES: Mapping[str, str] = {
    "pinned_execution_functions": "PINNED_EXECUTION_FUNCTIONS",
    "pinned_analysis_functions": "PINNED_ANALYSIS_FUNCTIONS",
}
REMOVED_DRAFT_KEYS: tuple[str, ...] = ("status_note", "source_changes_pending")
# The reviewed document whose rule text link 2 binds, and the rendering the builder writes (two files, two roles).
RULE_TEXT_SOURCE = amendment_rules.rule_text_source(amendment_rules.AMENDMENT2_SEQUENCE)
RENDERING_PATH = amendment_rules.rendering_path(amendment_rules.AMENDMENT2_PATH)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class AmendmentRecordError(ValueError):
    """An amendment-2 receipt, decisions file or record input breaks its contract; nothing is written."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AmendmentRecordError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Receipts read by the scoring pass


def write_chain_receipt(tip: Path, freeze1_sha256: str, out: Path, *, draft: bool = False,
                        expected_chain: Optional[Sequence[str]] = None, repo: Optional[Path] = None) -> dict[str, Any]:
    """Authenticate the chain ending at ``tip`` on the full clone ``repo`` (default: this repository) and write the
    ``verify-amendment-chain`` receipt once (``amendment_chain.chain_receipt_record``). A ``draft`` receipt
    (``require_published=False``) serves review only; ``check_chain_receipt`` refuses it."""
    from ec_clean30.provenance import REPO_ROOT

    refuse_test_paths([tip, out], what="--tip/--out")
    _require(isinstance(freeze1_sha256, str) and bool(_SHA256_RE.fullmatch(freeze1_sha256)),
             "--freeze1-sha256 must be a lower-case SHA-256.")
    try:
        from ec_clean30 import amendment_chain
    except ImportError as exc:
        raise AmendmentRecordError("The chain receipt needs the general amendment-chain verifier "
                                   "(ec_clean30.amendment_chain), which is not available.") from exc
    try:
        receipt = amendment_chain.chain_receipt_record(
            Path(tip), expected_freeze1_sha256=freeze1_sha256,
            expected_freeze1b_sha256=amendment_rules.FREEZE1B_SHA256, repo=Path(repo or REPO_ROOT),
            require_published=not draft, expected_chain=expected_chain)
    except amendment_chain.AmendmentChainError as exc:
        raise AmendmentRecordError(str(exc)) from exc
    _require(draft or receipt["published"] is True, "A real chain receipt needs a published chain.")
    receipt = dict(receipt)
    receipt["sha256"] = write_new_json(Path(out), receipt)
    return receipt


def write_refit_recipe_receipt(*, fold: int, step_chain: Path, freeze1_sha256: str, amendment_tip: Path,
                               campaign_root: Path, plan_path: Path, out: Path, smoke: bool = False,
                               chain_receipt: Optional[Path] = None) -> dict[str, Any]:
    """The verified refit recipe of ``fold`` (and, in fold 0, the binding futility ``continue``), written once. The
    chain is authenticated once (or its ``chain_receipt`` checked) and every later check expects that chain."""
    from ec_clean30 import futility as futility_module
    from ec_clean30 import selector

    refuse_test_paths([step_chain, amendment_tip, campaign_root, plan_path, out], what="refit-recipe inputs")
    receipt_option = {} if chain_receipt is None else {"chain_receipt": Path(chain_receipt)}
    try:
        policy = selector._selection_policy(Path(amendment_tip), freeze1_sha256, smoke=smoke, **receipt_option)
        expected = {"expected_chain": list(policy.get("amendment_tip_chain") or policy["freeze1_amendment_chain"])}
        spec = selector.verified_refit_recipe(Path(step_chain), int(fold), freeze1_sha256=freeze1_sha256,
                                              amendment_tip=Path(amendment_tip), campaign_root=Path(campaign_root),
                                              plan_path=Path(plan_path), smoke=smoke, **receipt_option, **expected)
        futility = None
        if int(fold) == futility_module.FUTILITY_FOLD:
            decision = futility_module.verify_futility_decision(
                Path(step_chain), freeze1_sha256=freeze1_sha256, amendment_tip=Path(amendment_tip),
                campaign_root=Path(campaign_root), smoke=smoke, **receipt_option, **expected)
            futility = {"decision": decision["decision"], "record_sha256": decision["record_sha256"],
                        "pointer_sequence": decision["pointer_sequence"], "binding": not smoke}
    except (selector.SelectionInputError, futility_module.FutilityInputError) as exc:
        raise AmendmentRecordError(str(exc)) from exc
    receipt = {"format": REFIT_RECIPE_RECEIPT_FORMAT, "schema_version": SCHEMA_VERSION, "fold": int(fold),
               "smoke": bool(smoke), "fit_chain": list(policy["freeze1_amendment_chain"]),
               "scientific_identity_sha256": policy.get("scientific_identity_sha256"),
               "refit": spec.record(), "futility": futility}
    receipt["sha256"] = write_new_json(Path(out), json.loads(json.dumps(receipt)))
    return receipt


def dry_run_v2(view_root: Path, out: Path, *, runs_dir: Optional[Path] = None,
               provenance: Optional[Mapping[str, Any]] = None, **options: Any) -> dict[str, Any]:
    """The repeated plan-v2 dry run of one view (rule A2-GOV-4): ``commands.dry_run(plan_version=2, write=False,
    pretrain_stub=True)``; the receipt is written once outside the view and the in-view receipts are untouched."""
    from ec_clean30 import commands
    from ec_clean30.views import View

    view = View.load(view_root)
    out = Path(os.path.abspath(os.fspath(out)))
    _require(Path(view.root).resolve() not in out.resolve().parents,
             "The dry-run-v2 receipt is written outside the view (the in-view receipts stay as Freeze 1 binds them).")
    receipt = commands.dry_run(view, runs_dir=runs_dir, provenance=provenance, write=False, plan_version=2,
                               pretrain_stub=True, **options)
    record = {"format": DRY_RUN_V2_RECEIPT_FORMAT, "schema_version": SCHEMA_VERSION, **receipt}
    record["sha256"] = write_new_json(out, json.loads(json.dumps(record, default=str)))
    return record


# ---------------------------------------------------------------------------
# Decisions file, record builder and rendering


def check_decisions_file(path: Path) -> dict[str, Any]:
    """The maintainer's decisions file: every confirmation C1-C8 and C6b answered, a maintainer named, no
    placeholder token and no absolute path (``freeze._PLACEHOLDER_TOKEN_RE``, ``freeze._ABSOLUTE_PATH_RE``)."""
    from ec_clean30.freeze import _ABSOLUTE_PATH_RE, _PLACEHOLDER_TOKEN_RE

    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"The decisions file {path.name} is missing or a link.")
    data = path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise AmendmentRecordError(f"The decisions file is not JSON ({exc}).") from None
    _require(isinstance(payload, dict) and payload.get("format") == DECISIONS_FORMAT,
             f"The decisions file is not a {DECISIONS_FORMAT} file.")
    answers = payload.get("answers")
    _require(isinstance(answers, Mapping) and sorted(answers) == sorted(CONFIRMATION_IDS),
             f"The decisions file must answer exactly {list(CONFIRMATION_IDS)}.")
    for key, answer in answers.items():
        _require(isinstance(answer, Mapping) and isinstance(answer.get("choice"), str) and answer["choice"].strip()
                 and isinstance(answer.get("accepted_default"), bool),
                 f"Confirmation {key} needs a choice and accepted_default.")
    maintainer = payload.get("maintainer")
    _require(isinstance(maintainer, str) and maintainer.strip(), "The decisions file names the maintainer.")
    text = data.decode("utf-8")
    _require(not _PLACEHOLDER_TOKEN_RE.search(text), "The decisions file holds a placeholder token.")
    _require(not _ABSOLUTE_PATH_RE.search(text), "The decisions file holds an absolute path.")
    return {"sha256": _sha256_bytes(data), "answers": {key: dict(value) for key, value in sorted(answers.items())},
            "maintainer": maintainer.strip()}


def rule_text(amendment_draft_md: Path) -> str:
    """The rule text of the reviewed document ``amendment_draft.md``: from ``## Rule text`` up to
    ``## Policy deviations``."""
    text = Path(amendment_draft_md).read_text(encoding="utf-8")
    start, end = text.find(RULE_TEXT_START), text.find(RULE_TEXT_END)
    _require(0 <= start < end, f"{Path(amendment_draft_md).name} has no rule-text section.")
    return text[start:end]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(_COMMIT_RE.fullmatch(value))


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AmendmentRecordError(f"{what} cannot be read as JSON ({type(exc).__name__}).") from None


def _repo_relative(repo: Path, path: Path, what: str) -> str:
    """The repository-relative POSIX path of a regular file inside ``repo`` (bound files are read at the link's
    commit, so a file outside the repository cannot be bound)."""
    absolute = Path(os.path.realpath(os.fspath(path)))
    root = Path(os.path.realpath(os.fspath(repo)))
    _require(Path(path).is_file() and not Path(path).is_symlink(), f"{what} is missing or a link.")
    _require(root in absolute.parents, f"{what} is outside the repository; only repository files are bound.")
    return absolute.relative_to(root).as_posix()


def _git_status(repo: Path, *args: str, literal: bool = True, stdin: Optional[str] = None) -> tuple[int, str]:
    """``git <args>`` in ``repo`` without the caller's ``GIT_*`` variables (a hook's ``GIT_DIR`` would point git at
    another repository) and without system or global git configuration, with literal pathspecs unless ``literal`` is
    false (``check-ignore`` takes plain paths)."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
    try:
        completed = subprocess.run(["git", *(["--literal-pathspecs"] if literal else []), *args], cwd=str(repo),
                                   env=env, input=stdin, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout


def _private_binding(repo: Path, path: Path, what: str) -> dict[str, Any]:
    """The private binding ``{"path", "sha256", "private": true}`` of a file of a private role: a regular file inside
    ``repo`` (lexically and by its real path), reached without a symbolic link in any component, below
    ``amendment_rules.PRIVATE_BINDING_ROOT``, not tracked by git (in the index or at HEAD) and ignored by a
    ``.gitignore`` committed at HEAD (``_committed_ignore``; a local exclude holds in one clone only). Only the path
    and the SHA-256 are recorded."""
    root = Path(os.path.realpath(os.fspath(repo)))
    given = Path(os.path.abspath(os.fspath(path)))
    # The repository root as the path names it lexically (links above the root are allowed, none below it).
    anchor = next((parent for parent in reversed(given.parents) if Path(os.path.realpath(parent)) == root), None)
    _require(anchor is not None, f"{what} is outside the repository; a private input lives below "
                                 f"{amendment_rules.PRIVATE_BINDING_ROOT} of the repository.")
    relative = given.relative_to(anchor).as_posix()
    cursor = anchor
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{what} ({relative}) has a symbolic link in its path; a private input is "
                                          "a regular file reached without links.")
    _require(given.is_file(), f"{what} ({relative}) is missing or not a regular file.")
    real = Path(os.path.realpath(given))
    _require(root in real.parents and real.relative_to(root).as_posix() == relative,
             f"{what} ({relative}) does not resolve to itself inside the repository.")
    _require(amendment_rules.is_private_path(relative),
             f"{what} ({relative}) is not below the private binding root {amendment_rules.PRIVATE_BINDING_ROOT}.")
    code_index, index = _git_status(root, "ls-files", "-z", "--", relative)
    code_head, head = _git_status(root, "ls-tree", "-z", "--name-only", "HEAD", "--", relative)
    _require(code_index == 0 and not index and (code_head != 0 or not head),
             f"{what} ({relative}) is tracked by git; a private input is never committed (the repository is "
             "public).")
    _require(_committed_ignore(root, relative),
             f"{what} ({relative}) is not ignored by a .gitignore committed at HEAD; the private binding root must be "
             "gitignored so that a private input is never committed (.git/info/exclude and core.excludesFile hold in "
             "one clone only).")
    return {"path": relative, "sha256": sha256_file(given), "private": True}


def _committed_ignore(root: Path, relative: str) -> bool:
    """Whether git ignores ``relative`` by a pattern of a ``.gitignore`` file that is tracked at HEAD, unmodified in
    the working tree and in a folder that contains ``relative``, so that every clone ignores it too. Global and
    repository ``core.excludesFile`` are switched off; a match in ``.git/info/exclude`` or a negated pattern is not
    accepted."""
    code, out = _git_status(root, "-c", f"core.excludesFile={os.devnull}", "check-ignore", "-v", "-z", "--stdin",
                            literal=False, stdin=relative + "\0")
    fields = out.split("\0")
    if code != 0 or len(fields) != 5 or fields[3] != relative or fields[4] != "" or fields[2].startswith("!"):
        return False
    source = PurePosixPath(fields[0])
    folder = source.parent.as_posix()
    if (source.is_absolute() or source.name != ".gitignore" or ".." in source.parts
            or not (folder == "." or relative.startswith(folder + "/"))):
        return False
    code_head, head = _git_status(root, "ls-tree", "-z", "--name-only", "HEAD", "--", source.as_posix())
    code_diff, _diff = _git_status(root, "diff", "--quiet", "HEAD", "--", source.as_posix())
    return code_head == 0 and head == source.as_posix() + "\0" and code_diff == 0


def draft_shape_problems(draft: Mapping[str, Any]) -> list[str]:
    """What the reviewed draft lacks against the contract's ``record_builder`` section: the draft keys, the
    sub-keys of the listed sections, the pinned-function name lists (path -> top-level names) and the scope list."""
    problems = [f"{key} is missing" for key in DRAFT_KEYS if key not in draft]
    for key, subkeys in DRAFT_SUBKEYS.items():
        value = draft.get(key)
        if key in draft and not isinstance(value, Mapping):
            problems.append(f"{key} is not an object")
        elif isinstance(value, Mapping):
            problems += [f"{key}.{subkey} is missing" for subkey in subkeys if subkey not in value]
    for key in DRAFT_FUNCTION_NAMES:
        value = draft.get(key)
        if key in draft and not (isinstance(value, Mapping) and all(
                isinstance(names, list) and names and all(isinstance(name, str) and name.isidentifier()
                                                          for name in names) for names in value.values())):
            problems.append(f"{key} is not a map of file paths to top-level function names")
    changes = draft.get("source_changes")
    if "source_changes" in draft and not (isinstance(changes, Mapping) and all(
            isinstance(item, Mapping) and "before" in item and "after" in item for item in changes.values())):
        problems.append("source_changes does not give each file its before and after SHA-256")
    unchanged = draft.get("unchanged_scientific_files")
    if "unchanged_scientific_files" in draft and not (isinstance(unchanged, list)
                                                      and all(isinstance(item, str) for item in unchanged)):
        problems.append("unchanged_scientific_files is not a list of files and folders")
    declared = draft.get("execution_file_set")
    if isinstance(declared, Mapping) and not all(
            isinstance(declared.get(key), list) and all(isinstance(item, str) for item in declared[key])
            for key in DRAFT_SUBKEYS["execution_file_set"] if key in declared):
        problems.append("execution_file_set.declared and .pending are not lists of repository paths")
    return problems


def _fold_inputs(inputs: Optional[Mapping[Any, Path]], what: str) -> dict[str, Path]:
    folds = [str(fold) for fold in range(N_FOLDS)]
    result = {str(fold): Path(path) for fold, path in (inputs or {}).items()}
    _require(set(result) <= set(folds), f"{what} names folds outside 0-{N_FOLDS - 1}: {sorted(set(result) - set(folds))}.")
    return result


def _file_binding_problems(repo: Path, record: Mapping[str, Any]) -> list[str]:
    """Every ``{"path", "sha256"}`` entry below ``FILE_BINDING_SECTIONS`` has the privacy of its role
    (``amendment_rules.binding_privacy_problems``) and names a repository file with that SHA-256; a private one also
    passes ``_private_binding`` (below the private root, no link, untracked, ignored)."""
    problems: list[str] = list(amendment_rules.binding_privacy_problems(record, FILE_BINDING_SECTIONS))

    def walk(value: Any, where: str) -> None:
        if isinstance(value, Mapping):
            if "path" in value and "sha256" in value:
                path = value.get("path")
                target = Path(repo) / path if isinstance(path, str) and not os.path.isabs(path) else None
                if amendment_rules.is_private_role(where) and amendment_rules.is_private_binding(value):
                    try:
                        binding = _private_binding(repo, Path(repo) / path, where)
                    except AmendmentRecordError as exc:
                        problems.append(str(exc))
                    else:
                        if binding["sha256"] != value.get("sha256"):
                            problems.append(f"{where}: {path} does not hash to its recorded SHA-256")
                elif target is None or ".." in Path(path).parts or not target.is_file():
                    problems.append(f"{where}: {path} is not a repository file")
                elif sha256_file(target) != value.get("sha256"):
                    problems.append(f"{where}: {path} does not hash to its recorded SHA-256")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                walk(item, f"{where}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{where}[{index}]")

    for section in FILE_BINDING_SECTIONS:
        walk(record.get(section), section)
    return problems


def _base_document(repo: Path, binding: Mapping[str, Any], what: str) -> dict[str, Any]:
    """A base document (Freeze 1, link 1) named by the draft, refused unless its bytes have the recorded SHA-256."""
    path = binding.get("path")
    _require(isinstance(path, str) and (Path(repo) / path).is_file()
             and sha256_file(Path(repo) / path) == binding.get("sha256"),
             f"The {what} named by the draft is missing or does not hash to its recorded SHA-256.")
    document = _read_json(Path(repo) / path, f"The {what}")
    _require(isinstance(document, dict), f"The {what} is not a JSON object.")
    return document


def _previous_pins(repo: Path, draft: Mapping[str, Any]) -> dict[str, Any]:
    """The composite pins before link 2: the Freeze-1 code pins with the source changes of link 1."""
    freeze1 = _base_document(repo, draft["base_freeze1"], "base Freeze 1")
    link1 = _base_document(repo, draft["base_amendment"], "base amendment (link 1)")
    pins = (freeze1.get("code") or {}).get("tracked_file_sha256")
    changes = link1.get("source_changes")
    _require(isinstance(pins, Mapping) and isinstance(changes, Mapping),
             "The base Freeze 1 or link 1 holds no code pins.")
    previous = dict(pins)
    for path, change in changes.items():
        previous[path] = change.get("after") if isinstance(change, Mapping) else None
    return previous


def _scope(repo: Path, draft: Mapping[str, Any], previous: Mapping[str, Any]) -> dict[str, Any]:
    """The live tree against the scope list: each ``source_changes`` entry has the previous composite pin as
    ``before`` and the live bytes as ``after``, and no pinned file changed or vanished outside it. The execution file
    set is then every tracked file that neither the previous pins nor the scope list name
    (``amendment_rules.partition`` refuses a scientific or post-fit analysis role), and it must equal the draft's
    declared closed list ``execution_file_set.declared`` (an unlisted file is refused, never taken as execution)."""
    tracked = amendment_rules.tracked_source_files(repo)
    live = {path: sha256_file(Path(repo) / path) for path in tracked if (Path(repo) / path).is_file()}
    changes = draft["source_changes"]
    differ = [path for path, item in sorted(changes.items())
              if live.get(path) is None or item.get("after") != live[path] or item.get("before") != previous.get(path)]
    _require(not differ, f"Source files differ from the record's scope list: {differ[:8]}.")
    unlisted = sorted(path for path, pin in previous.items() if path not in changes and live.get(path) != pin)
    _require(not unlisted, f"Pinned files changed or vanished outside the record's scope list: {unlisted[:8]}.")
    execution = sorted(path for path in tracked if path not in previous and path not in changes)
    try:
        part = amendment_rules.partition(tracked, execution)
    except amendment_rules.AmendmentRuleError as exc:
        raise AmendmentRecordError(f"The execution file set derived from the live tree is refused ({exc}); a "
                                   "scientific file belongs in the record's scope list.") from None
    declared = sorted(set(draft["execution_file_set"]["declared"]))
    unlisted = sorted(set(execution) - set(declared))
    absent = sorted(set(declared) - set(execution))
    _require(not unlisted and not absent,
             f"The execution file set derived from the live tree differs from the draft's declared closed list "
             f"(not declared: {unlisted[:8]}; declared but not derived: {absent[:8]}); a file outside the declared "
             "list is never an execution file.")

    def names(entry: str, path: str) -> bool:
        return path == entry or (entry.endswith("/") and path.startswith(entry))

    unchanged = [entry for entry in draft["unchanged_scientific_files"]
                 if not any(names(entry, path) for path in previous) or any(names(entry, path) for path in changes)]
    _require(not unchanged, f"Unchanged scientific entries name no pinned file or a changed one: {unchanged[:8]}.")
    return part


def _function_pins(repo: Path, draft: Mapping[str, Any], execution_files: Sequence[str]) -> dict[str, dict[str, str]]:
    """``"<path>::<name>" -> amendment_rules.function_closure_sha256`` in the live tree for the draft's pinned
    execution and analysis functions, whose names must equal the ``amendment_rules`` registries."""
    pins: dict[str, dict[str, str]] = {}
    for key, registry_name in DRAFT_FUNCTION_NAMES.items():
        registry = getattr(amendment_rules, registry_name)
        named = {f"{path}::{name}" for path, names in draft[key].items() for name in names}
        expected = {f"{path}::{name}" for path, names in registry.items() for name in names}
        _require(named == expected, f"The draft's {key} differ from amendment_rules.{registry_name} (draft only: "
                                    f"{sorted(named - expected)}; registry only: {sorted(expected - named)}); a "
                                    "function joins the registry once its source exists.")
        allowed = set(execution_files) if key == "pinned_execution_functions" else set(
            amendment_rules.POST_FIT_ANALYSIS_FILES)
        outside = sorted(item for item in named if item.split("::")[0] not in allowed)
        _require(not outside, f"The draft's {key} lie outside their file set: {outside}.")
        try:
            pins[key] = {item: amendment_rules.function_closure_sha256(Path(repo) / item.split("::")[0],
                                                                       [item.split("::")[1]])
                         for item in sorted(named)}
        except (amendment_rules.AmendmentRuleError, OSError, SyntaxError) as exc:
            raise AmendmentRecordError(f"The {key} cannot be hashed in the live tree ({exc}).") from None
    return pins


def _plan_v2_hashes(plan_paths: Mapping[str, Path]) -> dict[str, str]:
    """``fold -> plan_sha256`` of the plan-v2 records, each equal to ``campaign.build_fold_plan_v2`` live."""
    from ec_clean30 import campaign

    hashes = {}
    for fold, path in sorted(plan_paths.items()):
        plan = _read_json(path, f"The plan-v2 record of fold {fold}")
        _require(isinstance(plan, dict) and plan.get("format") == PLAN_V2_FORMAT and plan.get("fold") == int(fold)
                 and _is_sha256(plan.get("plan_sha256")),
                 f"The plan-v2 record of fold {fold} is not a {PLAN_V2_FORMAT} record of that fold.")
        _require(campaign.build_fold_plan_v2(int(fold))["plan_sha256"] == plan["plan_sha256"],
                 f"The plan-v2 record of fold {fold} differs from campaign.build_fold_plan_v2({fold}).")
        hashes[fold] = plan["plan_sha256"]
    return hashes


def _dry_run_evidence(repo: Path, receipt_paths: Mapping[str, Path], freeze1: Mapping[str, Any]) -> dict[str, Any]:
    """The dry-run-v2 receipts (passed, real plan-v2 dry runs of their folds on a clean tree, each on the fold view
    whose manifest SHA-256 Freeze 1 records), bound privately below ``amendment_rules.PRIVATE_BINDING_ROOT`` (they
    hold absolute paths, the host name and per-fold counts), and their common HEAD, which must be an ancestor of
    HEAD."""
    from ec_clean30 import freeze

    receipts, heads = {}, set()
    for fold, path in sorted(receipt_paths.items()):
        what = f"The dry-run-v2 receipt of fold {fold}"
        binding = _private_binding(repo, Path(path), what)
        receipt = _read_json(path, what)
        provenance = receipt.get("provenance") if isinstance(receipt, dict) else None
        provenance = provenance if isinstance(provenance, Mapping) else {}
        git = provenance.get("git") if isinstance(provenance.get("git"), Mapping) else {}
        _require(isinstance(receipt, dict) and receipt.get("format") == DRY_RUN_V2_RECEIPT_FORMAT
                 and receipt.get("status") == "passed" and receipt.get("fold") == int(fold)
                 and receipt.get("plan_version") == 2 and provenance.get("smoke") is False
                 and git.get("clean") is True and _is_commit(git.get("head")),
                 f"{what} is not a passed, real plan-v2 dry run of that fold on a clean tree.")
        frozen = (((freeze1.get("views") or {}).get(fold) or {}).get("view_manifest_sha256")
                  if isinstance(freeze1.get("views"), Mapping) else None)
        _require(_is_sha256(frozen) and receipt.get("view_manifest_sha256") == frozen,
                 f"{what} names another view manifest than the fold-{fold} view of Freeze 1.")
        heads.add(git["head"])
        _require(sha256_file(path) == binding["sha256"], f"{what} changed while it was read.")
        receipts[fold] = binding
    _require(len(heads) == 1, f"The dry-run-v2 receipts name {len(heads)} source commits; the five dry runs run at one.")
    (source,) = heads
    head = freeze._resolve_commit(Path(repo), "HEAD")
    _require(freeze._resolve_commit(Path(repo), source) == source and head is not None
             and freeze._is_ancestor(Path(repo), source, head),
             "The dry-run source commit is not an ancestor of HEAD in this repository.")
    return {"source_commit": source, "receipts": receipts}


def _place_binding(record: dict[str, Any], role: str, binding: Mapping[str, str]) -> None:
    """Put a file binding at the dotted record path ``role`` (a note string there is replaced)."""
    keys = role.split(".")
    cursor: Any = record
    for key in keys[:-1]:
        _require(isinstance(cursor, dict), f"--bind {role}: {key} is not an object of the record.")
        if not isinstance(cursor.get(key), dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = dict(binding)


def build_amendment2(*, draft_path: Path, decisions_path: Path, amendment_draft_md: Path, repo: Path,
                     bound_files: Mapping[str, Path], out: Path, plan_v2: Optional[Mapping[Any, Path]] = None,
                     dry_run_receipts: Optional[Mapping[Any, Path]] = None) -> dict[str, Any]:
    """The canonical record ``amendment.json`` (link 2), written once, in the contract fixture's record shape.

    It keeps the reviewed draft's content, settles its confirmations from the decisions file (bound as
    ``bound_files.decisions``), binds the rule text of the reviewed document ``amendment_draft.md`` (its
    repository path as ``rule_text.source``), every ``bound_files`` entry and the live rule policies by SHA-256,
    and computes from the live tree what depends on execution-side code: the execution file set (equal to the
    draft's declared closed list) and the closure SHA-256 of every pinned function. A ``bound_files`` name with a
    dot is a role of ``amendment_rules.REQUIRED_BINDINGS`` (or a key below one, such as ``inputs.f6_holdout.0``)
    and is placed at that record path; any other name goes to ``bound_files``. ``plan_v2`` (fold -> plan-v2 record)
    and ``dry_run_receipts`` (fold -> dry-run-v2 receipt below ``amendment_rules.PRIVATE_BINDING_ROOT``) are build
    inputs for all five folds. A name below ``amendment_rules.PRIVATE_ROLES`` and every dry-run-v2 receipt are bound
    privately (``_private_binding``); every other file is bound publicly and is committed with the link.

    Refused, in this order: another draft format; a draft without its contract shape; an invalid decisions file;
    missing build inputs; a pending source change; a pending execution file; base documents that do not hash; a
    scope list that differs from the live tree; a derived execution set with a scientific role or other than the
    declared closed list; pinned-function names that differ from the registries; build inputs that do not check; a
    rule-text document other than ``amendment_draft.md``; a required binding that is absent; file bindings that do
    not hash or do not have the privacy of their role.
    """
    from ec_clean30 import selection

    repo = Path(os.path.realpath(os.fspath(repo)))
    draft = _read_json(Path(draft_path), "The draft record")
    _require(isinstance(draft, dict) and draft.get("format") == DRAFT_FORMAT, "The draft record has another format.")
    shape = draft_shape_problems(draft)
    _require(not shape, "The draft record does not have the contract's shape: " + "; ".join(shape) + ".")
    decisions = check_decisions_file(Path(decisions_path))
    plans = _fold_inputs(plan_v2, "plan_v2")
    receipt_paths = _fold_inputs(dry_run_receipts, "dry_run_receipts")
    folds = [str(fold) for fold in range(N_FOLDS)]
    missing = []
    if sorted(plans) != folds:
        missing.append("plans.v2_plan_sha256 needs the plan-v2 records of folds "
                       f"{[int(fold) for fold in folds if fold not in plans]}")
    if sorted(receipt_paths) != folds:
        missing.append("dry_runs.source_commit and dry_runs.receipts need the dry-run-v2 receipts of folds "
                       f"{[int(fold) for fold in folds if fold not in receipt_paths]}")
    _require(not missing, "Build inputs are missing: " + "; ".join(missing) + ". They are written by plan-v2 and "
                          "dry-run-v2 before the record build and are never fabricated.")
    pending = sorted(f"{path} ({(item or {}).get('written_by') if isinstance(item, Mapping) else None})"
                     for path, item in (draft.get("source_changes_pending") or {}).items())
    _require(not pending, f"Source changes are still pending: {pending}; the draft lists them in source_changes "
                          "once they are written.")
    pending_execution = sorted(draft["execution_file_set"]["pending"])
    _require(not pending_execution, f"Execution files are still pending: {pending_execution}; the draft lists them in "
                                    "execution_file_set.declared once they exist.")
    part = _scope(repo, draft, _previous_pins(repo, draft))
    pins = _function_pins(repo, draft, part["execution_files"])
    v2_plans = _plan_v2_hashes(plans)
    dry_runs = _dry_run_evidence(repo, receipt_paths, _base_document(repo, draft["base_freeze1"], "base Freeze 1"))
    source = _repo_relative(repo, Path(amendment_draft_md), "The reviewed document")
    _require(source == RULE_TEXT_SOURCE, f"The rule text is read from the reviewed document {RULE_TEXT_SOURCE}, "
                                         f"not from {source} (the rendering {RENDERING_PATH} is written from the "
                                         "record).")
    bound, roles = {}, {}
    for name, path in sorted(bound_files.items()):
        _require(name != DECISIONS_BINDING, f"bound_files.{DECISIONS_BINDING} is the decisions file itself.")
        if "." in name:
            _require(any(name == role or name.startswith(role + ".") for role in amendment_rules.REQUIRED_BINDINGS),
                     f"--bind {name} names no role of amendment_rules.REQUIRED_BINDINGS.")
        if amendment_rules.is_private_role(name):
            roles[name] = _private_binding(repo, Path(path), f"Bound file {name}")
            continue
        binding = {"path": _repo_relative(repo, Path(path), f"Bound file {name}"), "sha256": sha256_file(path)}
        if "." in name:
            roles[name] = binding
        else:
            bound[name] = binding
    bound[DECISIONS_BINDING] = {"path": _repo_relative(repo, Path(decisions_path), "The decisions file"),
                                "sha256": decisions["sha256"]}
    text = rule_text(Path(amendment_draft_md))
    record = json.loads(json.dumps({key: value for key, value in draft.items() if key not in REMOVED_DRAFT_KEYS}))
    record.pop("execution_file_set", None)
    for name, binding in sorted(roles.items()):
        _place_binding(record, name, binding)
    record.update({
        "format": RECORD_FORMAT, "status": "built", "confirmation_status": "confirmed",
        "provisional_markers_settled": True,
        "decisions_file": {"sha256": decisions["sha256"], "maintainer": decisions["maintainer"],
                           "answers": decisions["answers"]},
        "rule_text": {"source": source, "sha256": _sha256_bytes(text.encode("utf-8")), "text": text},
        "bound_files": bound,
        "plans": {**record["plans"], "v2_plan_sha256": v2_plans},
        "dry_runs": {**record["dry_runs"], **dry_runs},
        "execution_file_set": {"files": part["execution_files"], "names_sha256": part["execution_names_sha256"]},
        **pins,
        "rule_policy_sha256": amendment_rules.rule_policy_sha256(),
        "selection_policy_sha256_v2": selection.selection_policy_sha256_v2(),
        "live_constants_sha256": amendment_rules.live_constants_sha256(),
    })
    unbound = amendment_rules.required_binding_problems(record)
    _require(not unbound, f"Required bindings are absent or malformed: {unbound[:9]}. Each role of "
                          "amendment_rules.REQUIRED_BINDINGS is bound by SHA-256 (--bind ROLE=PATH, or a "
                          "{path, sha256} entry in the draft) before the record is built.")
    bindings = _file_binding_problems(repo, record)
    _require(not bindings, f"File bindings of the record do not hold: {bindings[:8]}.")
    absent = [key for key in (*RECORD_REQUIRED_KEYS, *RECORD_BUILT_KEYS) if key not in record]
    _require(not absent, f"The built record lacks {absent}.")
    record["sha256_of_content"] = _sha256_bytes(_canonical(record))
    write_new_json(Path(out), record)
    return record


def render_amendment2(record: Mapping[str, Any]) -> str:
    """The rendering ``amendment.md`` of a built record: header, bindings, confirmations and the bound rule text
    (read at build from the reviewed document ``amendment_draft.md``)."""
    _require(record.get("format") == RECORD_FORMAT, "Only a built amendment-2 record is rendered.")
    text = record["rule_text"]["text"]
    _require(_sha256_bytes(text.encode("utf-8")) == record["rule_text"]["sha256"], "The rule text differs from its hash.")
    lines = ["# Amendment 2: pre-G2 science amendment", "",
             f"Record `{record['amendment_id']}`, sequence {record['sequence']}, change class "
             f"`{record['change_class']}`; built from the reviewed draft. It takes effect when its SHA-256 is entered "
             "in the campaign log and it is published (rule A2-GOV-1).", "",
             "## Confirmations", "", "| ID | Choice | Default accepted |", "|---|---|---|"]
    for key, answer in sorted(record["decisions_file"]["answers"].items()):
        lines.append(f"| {key} | {answer['choice']} | {'yes' if answer['accepted_default'] else 'no'} |")
    lines += ["", "## Bound files", "", "| Name | Path | SHA-256 |", "|---|---|---|"]
    lines += [f"| {name} | `{item['path']}` | `{item['sha256']}` |" for name, item in sorted(record["bound_files"].items())]
    private = [(where, item) for section in FILE_BINDING_SECTIONS
               for where, item in amendment_rules.binding_items(record.get(section), section)
               if amendment_rules.is_private_role(where)]
    if private:
        lines += ["", "## Private bindings", "",
                  f"Gitignored files below `{amendment_rules.PRIVATE_BINDING_ROOT}`, bound by path and SHA-256 only; "
                  "they are not published.", "", "| Role | Path | SHA-256 |", "|---|---|---|"]
        lines += [f"| `{where}` | `{item.get('path')}` | `{item.get('sha256')}` |" for where, item in private]
    lines += ["", "## Rule policies", "", "| Rule | Policy SHA-256 |", "|---|---|"]
    lines += [f"| `{rule}` | `{digest}` |" for rule, digest in sorted(record["rule_policy_sha256"].items())]
    lines += ["", text.rstrip("\n"), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner actions (registered by scripts/run_ec_clean30_campaign.py; the pinned runner functions are unchanged)


def add_amendment2_arguments(sub: Any) -> None:
    """Sub-parsers of ``amend-2``, ``verify-amendment-chain``, ``dry-run-v2`` and ``refit-recipe``."""
    amend = sub.add_parser("amend-2", help="Build amendment.json once from the reviewed draft and the decisions file.")
    for flag in ("draft", "decisions", "amendment-draft-md", "out"):
        amend.add_argument("--" + flag, required=True)
    amend.add_argument("--bind", action="append", default=[],
                       help="NAME=PATH of a bound file (repeatable); a dotted NAME is a required role, such as "
                            "evidence.verification or inputs.f6_holdout.0; a private role's file lies below "
                            f"{amendment_rules.PRIVATE_BINDING_ROOT} and is bound privately.")
    amend.add_argument("--plan-v2", action="append", default=[],
                       help="FOLD=PATH of a plan-v2 record (build input; one per fold 0-4).")
    amend.add_argument("--dry-run-receipt", action="append", default=[],
                       help="FOLD=PATH of a dry-run-v2 receipt below the gitignored private binding root "
                            f"{amendment_rules.PRIVATE_BINDING_ROOT} (build input; one per fold 0-4; bound privately).")
    amend.add_argument("--render", default=None, help=f"Write the rendering here, {RENDERING_PATH} (written once).")
    amend.set_defaults(handler=action_amend2)
    chain = sub.add_parser("verify-amendment-chain", help="Authenticate the amendment chain; write its receipt.")
    chain.add_argument("--tip", required=True)
    chain.add_argument("--freeze1-sha256", required=True)
    chain.add_argument("--out", required=True)
    chain.add_argument("--draft", action="store_true", help="Review only: an unpublished chain (never for scoring).")
    chain.add_argument("--expected-chain", action="append", default=None,
                       help="SHA-256 of each link in order (repeatable); the receipt is refused for another chain.")
    chain.set_defaults(handler=action_verify_amendment_chain)
    dry = sub.add_parser("dry-run-v2", help="Plan-v2 dry run of one view against the pretraining-input stub.")
    dry.add_argument("--view", required=True)
    dry.add_argument("--out", required=True, help="Receipt outside the view (written once).")
    dry.add_argument("--runs-dir", default=None)
    dry.set_defaults(handler=action_dry_run_v2)
    refit = sub.add_parser("refit-recipe", help="The verified refit recipe of one fold, for the scoring pass.")
    for flag in ("step-chain", "freeze1-sha256", "freeze1-amendment", "campaign-root", "plan", "out"):
        refit.add_argument("--" + flag, required=True)
    refit.add_argument("--fold", type=int, required=True)
    refit.add_argument("--amendment-chain-receipt", default=None,
                       help="The verify-amendment-chain receipt to check instead of verifying the chain again.")
    refit.add_argument("--smoke", action="store_true")
    refit.set_defaults(handler=action_refit_recipe)


def _report(callable_, *args: Any, **kwargs: Any) -> int:
    try:
        result = callable_(*args, **kwargs)
    except (AmendmentRecordError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: result.get(key) for key in ("format", "sha256", "status", "fold", "fit_chain")
                      if key in result}, indent=2, sort_keys=True, default=str))
    return 0


def action_amend2(args: argparse.Namespace) -> int:
    from ec_clean30.provenance import REPO_ROOT

    real_action_git_state(smoke=False)

    def pairs(items: Sequence[str], flag: str) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for item in items:
            name, _, path = item.partition("=")
            _require(name and path and name not in result, f"{flag} needs distinct NAME=PATH values, got {item!r}")
            result[name] = Path(path)
        return result

    def build() -> dict[str, Any]:
        record = build_amendment2(draft_path=Path(args.draft), decisions_path=Path(args.decisions),
                                  amendment_draft_md=Path(args.amendment_draft_md), repo=REPO_ROOT,
                                  bound_files=pairs(args.bind, "--bind"), out=Path(args.out),
                                  plan_v2=pairs(args.plan_v2, "--plan-v2"),
                                  dry_run_receipts=pairs(args.dry_run_receipt, "--dry-run-receipt"))
        if args.render:
            with Path(args.render).open("x", encoding="utf-8") as handle:
                handle.write(render_amendment2(record))
        return {"format": record["format"], "sha256": sha256_file(Path(args.out)), "status": record["status"]}

    return _report(build)


def action_verify_amendment_chain(args: argparse.Namespace) -> int:
    real_action_git_state(smoke=args.draft)
    return _report(write_chain_receipt, Path(args.tip), args.freeze1_sha256, Path(args.out), draft=args.draft,
                   expected_chain=getattr(args, "expected_chain", None))


def action_dry_run_v2(args: argparse.Namespace) -> int:
    git = real_action_git_state(smoke=False)
    return _report(dry_run_v2, Path(args.view), Path(args.out),
                   runs_dir=Path(args.runs_dir) if args.runs_dir else None,
                   provenance=provenance_record(smoke=False, git=git))


def action_refit_recipe(args: argparse.Namespace) -> int:
    real_action_git_state(smoke=args.smoke)
    return _report(write_refit_recipe_receipt, fold=args.fold, step_chain=Path(args.step_chain),
                   freeze1_sha256=args.freeze1_sha256, amendment_tip=Path(args.freeze1_amendment),
                   campaign_root=Path(args.campaign_root), plan_path=Path(args.plan), out=Path(args.out),
                   smoke=args.smoke, chain_receipt=Path(args.amendment_chain_receipt)
                   if getattr(args, "amendment_chain_receipt", None) else None)


def receipt_formats() -> dict[str, str]:
    return {"chain": CHAIN_RECEIPT_FORMAT, "refit_recipe": REFIT_RECIPE_RECEIPT_FORMAT}


__all__: Sequence[str] = (
    "AmendmentRecordError", "build_amendment2", "check_decisions_file", "draft_shape_problems", "dry_run_v2",
    "render_amendment2", "write_chain_receipt", "write_refit_recipe_receipt",
)
