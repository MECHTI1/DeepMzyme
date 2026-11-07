"""Shared rules of the Freeze-1 amendment chain from amendment 2 on (torch-free).

The amendment-2 record builder (``ec_clean30.amendment2``) and the general chain verifier
(``ec_clean30.amendment_chain.verify_chain``) both import this module, so the rules are written once
(contract X2: the verifier calls these functions and does not reimplement them).

- Chain constants: the Freeze-1, Freeze-1b and amendment-1 identities the chain starts from; the fixed path of
  every link (``link_path``) and the strict campaign-log line that publishes it (``link_log_line``,
  ``link_log_problems``).
- Partition (rule A2-GOV-2): every tracked ``src/`` and ``scripts/`` file is exactly one of an execution file
  (a closed list recorded by the amendment), a post-fit analysis file (exactly ``POST_FIT_ANALYSIS_FILES``) or a
  scientific file (every other file). ``partition`` refuses an execution list that names a forbidden role;
  ``private_import_problems`` refuses an execution file that reaches a private name of another set's module.
- Function pins: ``function_closure_sha256`` hashes the source of named top-level functions together with every
  module-level name they reference, transitively within their module. ``PINNED_ANALYSIS_FUNCTIONS`` are the
  channels through which scientific code reaches post-fit analysis code; ``PINNED_EXECUTION_FUNCTIONS`` are the
  frozen gates inside execution files.
- Bindings: ``REQUIRED_BINDINGS`` is the role table of the files every link of sequence 2 or higher binds by
  SHA-256 (``required_binding_problems``); ``DECISIONS_FORMAT`` and ``CONFIRMATION_IDS`` fix the decisions file.
  ``PRIVATE_ROLES`` are the roles whose files stay private (the repository is public): every binding below one is
  ``{"path", "sha256", "private": true}`` with a path below the gitignored ``PRIVATE_BINDING_ROOT``, matched by
  digest in the working tree and never committed; every other binding is public ``{"path", "sha256"}`` and committed
  (``is_file_binding``, ``is_private_binding``, ``binding_privacy_problems``).
- Change classes (rule A2-GOV-7): ``check_change_class`` checks a link of sequence 3 or higher against the files
  it changes and the previous link's function pins; ``check_analysis_completion`` checks the extra limits of class
  (a2); ``FUTILITY_POINTER_FORMAT`` and ``FUTILITY_RECORD_FORMAT`` name the fold-0 futility files such a link binds.
- ``r_cmd_v2_check`` (rule A2-GOV-4): what may change between the amendment-2 dry-run HEAD and a later HEAD.
- ``scientific_identity``: the SHA-256 of the canonical scientific-identity record; it refuses execution pins,
  commits, providers, hardware and session identifiers at any depth of the deferred-input specifications.
- ``RULE_POLICY_FUNCTIONS``: rule ID to the function whose canonical output is the bound policy record.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

CAMPAIGN_ID = "ec_class_vs_clean30_2026-10"
FREEZE1_SHA256 = "28077f4c3701820c9aa20f53b570d12f8b70a46514211fc63e62976e393a3c0c"
FREEZE1B_SHA256 = "fc6f032805f084d7aef4f704a42149e92fb7894af3699b2ad5c12e86c87bcbdb"
AMENDMENT1 = {
    "sequence": 1,
    "amendment_id": "s1-fixed-anchor-holm-v1",
    "path": "docs/campaigns/ec_class_vs_clean30_2026-10/s1_amendment/amendment.json",
    "sha256": "d07dbb6a68de7f441a0030661f110f2de9dabd4786a702a1e23d92a3a7a242ea",
    "commit": "3154a7f3b0ad44e8dffbc66072990cfb7d9c9030",
}
CAMPAIGN_DOCS = f"docs/campaigns/{CAMPAIGN_ID}"
AMENDMENT2_FORMAT = "deepmzyme_ec_clean30_amendment_v2"
AMENDMENT2_ID = "amendment-2-pre-g2-v1"
AMENDMENT2_PATH = "docs/campaigns/ec_class_vs_clean30_2026-10/amendment2/amendment.json"
AMENDMENT2_SEQUENCE = 2
# Freeze 2 of this campaign (a class-(a2) link is published before it exists, rule A2-GOV-7).
FREEZE2_PATH = f"{CAMPAIGN_DOCS}/freeze2.json"
# The strict campaign-log line that publishes a link of sequence 2 or higher (rule A2-GOV-1(3)); any log line
# that holds the marker must have exactly this form (an optional list dash, nothing after the SHA-256).
LINK_LOG_MARKER = "Amendment chain link"
_LINK_LOG_LINE_RE = re.compile(r"^(?:- )?Amendment chain link (?P<sequence>[1-9][0-9]*): `(?P<sha256>[0-9a-f]{64})`$")
CHANGE_PRE_DATA = "pre_data"
CHANGE_EXECUTION = "execution"
CHANGE_ANALYSIS_COMPLETION = "analysis_completion"
CHANGE_DEFECT = "defect"
CHANGE_CLASSES: tuple[str, ...] = (CHANGE_PRE_DATA, CHANGE_EXECUTION, CHANGE_ANALYSIS_COMPLETION, CHANGE_DEFECT)
LATER_LINK_REQUIRED_FIELDS: tuple[str, ...] = ("change_class", "timestamps", "fold0_records")
DEFECT_REQUIRED_FIELDS: tuple[str, ...] = ("defect", "failing_test", "found_through_record_sha256",
                                           "inner_score_viewed_before_identification", "reruns")
# Classes whose links keep every pinned function of the previous link (rule A2-GOV-2, A2-GOV-7(a) and (a2)).
PIN_KEEPING_CLASSES: tuple[str, ...] = (CHANGE_EXECUTION, CHANGE_ANALYSIS_COMPLETION)
PINNED_FUNCTION_KEYS: tuple[str, ...] = ("pinned_execution_functions", "pinned_analysis_functions")
# The fold-0 futility files a link of sequence 3 or higher binds for a FUT entry of its fold-0 record list
# (``futility.FUTILITY_POINTER_FORMAT`` and ``FUTILITY_RECORD_FORMAT``; a test keeps them equal).
FUTILITY_POINTER_FORMAT = "deepmzyme_ec_clean30_futility_pointer_v1"
FUTILITY_RECORD_FORMAT = "deepmzyme_ec_clean30_futility_record_v1"
FUT_KIND = "FUT"
FUTILITY_DECISIONS: tuple[str, ...] = ("continue", "stop_futility")
# The test that reproduces the bound decision fixture (class (a2) reruns it with every named refusal test).
DECISION_FIXTURE_TEST = "test_ec_clean30_amendment2_bound_fixtures.py::test_the_decision_fixture_is_reproduced"
# The maintainer's decisions file (``amendment2.check_decisions_file`` and the verifier's C8 parse it).
DECISIONS_FORMAT = "deepmzyme_ec_clean30_amendment2_decisions_v1"
CONFIRMATION_IDS: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6", "C6b", "C7", "C8")
# Role table (contract fixture ``record_builder.required_bindings``): dotted record path -> binding shape. Every
# link of sequence 2 or higher binds these files by SHA-256; the builder refuses a record without them.
REQUIRED_BINDINGS: Mapping[str, str] = {
    "data_state.attestation_at_build": "file_binding",
    "dry_runs.v1_resolved_config": "file_binding",
    "evidence.futility_oc": "file_binding",
    "evidence.interface_contract": "file_binding",
    "evidence.power_simulation": "file_binding",
    "evidence.verification": "file_binding",
    "inputs.clean_tie_exposure": "file_binding",
    "inputs.f6_corpus": "corpus_bindings",
    "inputs.f6_holdout": "fold_bindings",
}
BINDING_SHAPES: Mapping[str, str] = {
    "file_binding": ("{path, sha256}: a committed repository file and its SHA-256; below a private role "
                     "{path, sha256, private: true}: a gitignored, untracked file below PRIVATE_BINDING_ROOT"),
    "fold_bindings": "folds '0' to '4' exactly, each a file binding or a non-empty object of file bindings",
    "corpus_bindings": "{tables, union_fasta}: tables are fold_bindings, union_fasta is a file binding",
}
# Private bindings (the repository is public). The files of these roles hold per-fold counts, accessions or host
# details that the campaign does not publish before Stage 7 (five per-fold counts give the test-side counts by
# subtraction). They live below the gitignored campaign data root, are never committed and are matched by SHA-256
# in the working tree; only their path and SHA-256 enter the record. Every binding below a private role is private
# and every other binding is public (contract fixture: record_shape.required_bindings.private_roles).
PRIVATE_BINDING_ROOT = f"DeepMzyme_Data/campaigns/{CAMPAIGN_ID}/"
# Data-root links (``private_link_problem``): the campaign data may live on another disk, so each component of the
# private binding root itself ("DeepMzyme_Data", "DeepMzyme_Data/campaigns" and the campaign folder) may be a symbolic
# link to a directory outside the repository; every folder below the campaign folder and the file itself are real.
PRIVATE_ROOT_LINK_COMPONENTS: tuple[str, ...] = tuple(
    "/".join(PRIVATE_BINDING_ROOT.rstrip("/").split("/")[:count])
    for count in range(1, len(PRIVATE_BINDING_ROOT.rstrip("/").split("/")) + 1))
PRIVATE_ROLES: tuple[str, ...] = (
    "dry_runs.receipts", "inputs.clean_tie_exposure", "inputs.f6_corpus", "inputs.f6_holdout",
)
PRIVATE_BINDING_KEYS: tuple[str, ...] = ("path", "private", "sha256")
N_OUTER_FOLDS = 5

TRACKED_CODE_FOLDERS: tuple[str, ...] = ("src", "scripts")
# Rule A2-GOV-2: the closed post-fit analysis set (no other file joins it without a scientific amendment).
POST_FIT_ANALYSIS_FILES: tuple[str, ...] = (
    "scripts/run_ec_clean30_baseline.py",
    "scripts/score_ec_clean30_comparison.py",
    "src/ec_clean30/baseline.py",
    "src/ec_clean30/scoring.py",
)
# Scientific files that the execution side writes (classed scientific, listed in ``source_changes``); the science
# side writes every other scientific file, the chain verifier ``amendment_chain.py`` included.
EXECUTION_WRITTEN_SCIENTIFIC_FILES: tuple[str, ...] = (
    "scripts/run_clean_split30_comparator.py",
    "src/ec_clean30/runtime_precision.py",
)
# Files and folders an execution file may never be (roles of A2-GOV-2 and contract X3).
FORBIDDEN_EXECUTION_FILES: frozenset[str] = frozenset({
    *(f"src/ec_clean30/{name}.py" for name in (
        "campaign", "commands", "views", "provenance", "identity", "cohort", "transfers", "overlay", "r2_lists",
        "selection", "selector", "freeze", "amendment_chain", "amendment_rules", "amendment2", "runtime_precision",
        "futility", "stats", "scoring", "clean_sanity", "logit_adjustment", "baseline", "simulation",
        "s1_simulation", "pretrain_corpus")),
    "scripts/run_clean_split30_comparator.py",
    "scripts/run_ec_clean30_campaign.py",
    "scripts/run_ec_clean30_clean_sanity.py",
    *POST_FIT_ANALYSIS_FILES,
})
FORBIDDEN_EXECUTION_PREFIXES: tuple[str, ...] = ("src/training/", "src/model", "src/embed_helpers/")
R_ID_FILES: tuple[str, ...] = (
    "src/ec_clean30/identity.py", "scripts/run_ec_clean30_identity.py", "src/audit_sequence_remoteness.py",
    "src/training/ec_classes.py", "src/training/ec_membership.py", "src/project_paths.py",
)
# The channels through which scientific code reaches post-fit analysis code (pinned by function source).
PINNED_ANALYSIS_FUNCTIONS: Mapping[str, tuple[str, ...]] = {
    "src/ec_clean30/scoring.py": ("_load_comparator", "parse_label_table", "parse_transfer_list"),
    "src/ec_clean30/baseline.py": ("read_official_ec_numbers",),
}
ANALYSIS_CHANNELS: tuple[tuple[str, str], ...] = (
    ("scripts/run_clean_split30_comparator.py", "scoring.parse_label_table"),
    ("scripts/run_clean_split30_comparator.py", "baseline.read_official_ec_numbers"),
    ("src/ec_clean30/cohort.py", "scoring.parse_label_table"),
    ("src/ec_clean30/transfers.py", "scoring.parse_transfer_list"),
    ("src/ec_clean30/freeze.py", "live_constants (scoring and baseline constants)"),
)
# Frozen gates inside execution files (the CPU replay comparison is named by the execution lane in the contract
# fixture; its source is pinned once it exists).
PINNED_EXECUTION_FUNCTIONS: Mapping[str, tuple[str, ...]] = {
    "src/ec_clean30/phase3.py": ("lane_gate",),
    "src/ec_clean30/phase4.py": ("replay_cpu_logits",),
}
CPU_REPLAY_FUNCTION_CONTRACT = {"path": "src/ec_clean30/phase4.py", "qualname": "replay_cpu_logits",
                                "absolute_tolerance": 1e-5, "near_tie_tolerance": 2e-5, "argmax_agreement": True}
# Keys a scientific identity record may hold (rule A2-GOV-2); anything else is refused.
SCIENTIFIC_IDENTITY_KEYS: tuple[str, ...] = (
    "campaign_id", "freeze1_sha256", "freeze1b_sha256", "fit_chain", "composite_scientific_pins",
    "pinned_execution_functions", "plan_sha256_v1", "plan_sha256_v2", "freeze1_view_records", "dry_run_receipts",
    "precision_policy", "deferred_input_specifications",
)
FORBIDDEN_IDENTITY_KEYS: tuple[str, ...] = (
    "execution_file_pins", "execution_manifest_sha256", "commit", "git_head", "provider", "hardware", "gpu",
    "session", "session_id", "pod_id", "host",
)
PRECISION_POLICY = {
    "tf32": "off for every GPU DeepMzyme fit, embedding, CLEAN fit, export and farm child",
    "recorded_by": "ec_clean30.runtime_precision.configure_and_record",
    "esmc_whole_protein": ("float32 parameters set after loading; forward called directly without the SDK logits "
                           "wrapper, no autocast, no flash attention; emitted dtypes verified as float32"),
    "cpu_replay": {"absolute_tolerance": 1e-5, "near_tie_tolerance": 2e-5, "argmax_agreement": True},
    "lane_gate": "G3 keeps its frozen balanced-accuracy limits; the primary-metric difference is descriptive",
}
R_CMD_V2_RULE = "r_cmd_v2"

# Rule ID -> "module:function" whose canonical output is the bound policy record. ``analysis`` rules belong to
# post-fit analysis code: the scorer checks them itself, so scientific code never loads it to recompute them.
RULE_POLICY_FUNCTIONS: Mapping[str, Mapping[str, Any]] = {
    "amendment2_selection_v1": {"function": "ec_clean30.selection:selection_policy_record_v2", "analysis": False},
    "futility_predictive_cluster_bootstrap_v1": {"function": "ec_clean30.futility:policy_record", "analysis": False},
    "l9_only_no_s3_v1": {"function": "ec_clean30.campaign:search_policy_record_v2", "analysis": False},
    "controls_v2": {"function": "ec_clean30.campaign:controls_policy_record_v2", "analysis": False},
    "stratified_rao_wu_v2": {"function": "ec_clean30.stats:engine_policy_record_v2", "analysis": False},
    "clean_sanity_v1": {"function": "ec_clean30.clean_sanity:clean_sanity_policy_record", "analysis": False},
    "amendment2_accuracy_v1": {"function": "ec_clean30.scoring:scoring_policy_record_v2", "analysis": True},
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AmendmentRuleError(ValueError):
    """A link, partition, pin or identity record breaks a rule of the amendment chain."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AmendmentRuleError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Link paths and the campaign-log lines that publish links (rule A2-GOV-1)


def link_path(sequence: int) -> str:
    """The fixed path of the link of sequence n >= 2: ``amendment<n>/amendment.json`` in the campaign docs folder
    (one path per sequence, so a second link of one sequence cannot sit beside the first)."""
    _require(_is_int(sequence) and sequence >= AMENDMENT2_SEQUENCE, f"A link of sequence {sequence!r} has no path.")
    return f"{CAMPAIGN_DOCS}/amendment{sequence}/amendment.json"


def rendering_path(path: str) -> str:
    """The rendering ``amendment.md`` beside a link file ``amendment.json``."""
    _require(isinstance(path, str) and path.endswith(".json"), f"{path!r} is not a link file.")
    return path[:-len(".json")] + ".md"


def rule_text_source(sequence: int) -> str:
    """The reviewed document whose rule text a link of sequence n >= 2 binds: ``amendment<n>/amendment_draft.md``
    beside the draft record (never the rendering ``amendment.md``, which the builder writes from the record)."""
    return link_path(sequence)[:-len("amendment.json")] + "amendment_draft.md"


def link_log_line(sequence: int, sha256: str) -> str:
    """The campaign-log line that publishes a link; it is committed together with the link and its rendering."""
    _require(_is_int(sequence) and sequence >= 1 and isinstance(sha256, str) and bool(_SHA256_RE.fullmatch(sha256)),
             "A link log line needs a sequence and a SHA-256.")
    return f"{LINK_LOG_MARKER} {sequence}: `{sha256}`"


def link_log_entries(text: str) -> tuple[list[tuple[int, str]], list[str]]:
    """``(sequence, sha256)`` of every link line of a campaign log, and the lines that hold the marker in another
    form."""
    entries: list[tuple[int, str]] = []
    problems: list[str] = []
    for number, line in enumerate(str(text).splitlines(), 1):
        if LINK_LOG_MARKER not in line:
            continue
        match = _LINK_LOG_LINE_RE.fullmatch(line.strip())
        if match is None:
            problems.append(f"log line {number} names an amendment chain link outside the strict line format")
        else:
            entries.append((int(match["sequence"]), match["sha256"]))
    return entries, problems


def link_log_problems(text: str, chain: Sequence[str], *, tip_logged: bool = True) -> list[str]:
    """The log names each link of sequence 2 to the tip exactly once with the chain's SHA-256, and no other link:
    no second line of one sequence (a fork or a rewrite), no link beyond the tip (a stale tip) and no line of
    another form. Link 1 needs no line (it was published before this format); a line for it must name it.
    ``tip_logged=False`` (draft review of an uncommitted tip) lets the tip's line be absent."""
    entries, problems = link_log_entries(text)
    tip = len(chain)
    by_sequence: dict[int, list[str]] = {}
    for sequence, digest in entries:
        by_sequence.setdefault(sequence, []).append(digest)
    for sequence, digests in sorted(by_sequence.items()):
        if len(digests) > 1:
            problems.append(f"the log names {len(digests)} links of sequence {sequence} (a fork or a rewrite)")
        if sequence > tip:
            problems.append(f"the log names a link of sequence {sequence} beyond the tip of sequence {tip} "
                            "(a stale tip)")
        elif any(digest != chain[sequence - 1] for digest in digests):
            problems.append(f"the log names another link of sequence {sequence} than the chain")
    for sequence in range(AMENDMENT2_SEQUENCE, tip + 1):
        if sequence not in by_sequence and (tip_logged or sequence < tip):
            problems.append(f"the log has no line for link {sequence}")
    return problems


# ---------------------------------------------------------------------------
# Function-closure pins


def _module_definitions(tree: ast.Module) -> dict[str, list[ast.AST]]:
    """Top-level bindings by name (functions, classes, assignments, imports)."""
    names: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        bound: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound = [item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name)]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        for name in bound:
            names.setdefault(name, []).append(node)
    return names


def _source(text: str, node: ast.AST) -> str:
    lines = text.splitlines(keepends=True)
    start = min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", []))])
    return "".join(lines[start - 1:node.end_lineno])


def function_closure(text: str, qualnames: Sequence[str]) -> dict[str, str]:
    """``name -> source`` of the named top-level functions and every module-level name they reference,
    transitively within the module (a name bound twice at module level is refused)."""
    tree = ast.parse(text)
    definitions = _module_definitions(tree)
    pending, closure = list(qualnames), {}
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        nodes = definitions.get(name)
        _require(nodes, f"{name} is not defined at module level.")
        _require(len(nodes) == 1, f"{name} is bound {len(nodes)} times at module level.")
        node = nodes[0]
        closure[name] = _source(text, node)
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and item.id in definitions and item.id not in closure:
                pending.append(item.id)
    return dict(sorted(closure.items()))


def function_closure_sha256(path: str | Path, qualnames: Sequence[str]) -> str:
    """SHA-256 of the canonical ``function_closure`` of ``qualnames`` in the module file ``path``."""
    return sha256_canonical(function_closure(Path(path).read_text(encoding="utf-8"), qualnames))


def pinned_function_hashes(repo: str | Path, pins: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """``"<path>::<name>" -> closure SHA-256`` for every pinned function (the closure of each name alone)."""
    repo = Path(repo)
    return {f"{path}::{name}": function_closure_sha256(repo / path, [name])
            for path, names in sorted(pins.items()) for name in names}


def live_constants_sha256() -> str:
    """SHA-256 of the canonical ``freeze.live_constants()`` (class (a2) leaves it unchanged)."""
    from ec_clean30.freeze import live_constants

    return sha256_canonical(live_constants())


# ---------------------------------------------------------------------------
# Partition


def execution_file_problems(path: str) -> list[str]:
    if path in FORBIDDEN_EXECUTION_FILES or path.startswith(FORBIDDEN_EXECUTION_PREFIXES) or path in R_ID_FILES:
        return [f"{path} has a scientific or post-fit analysis role and cannot be an execution file"]
    if not path.startswith(tuple(f"{folder}/" for folder in TRACKED_CODE_FOLDERS)):
        return [f"{path} is not a tracked src/ or scripts/ file"]
    return []


def partition(files: Iterable[str], execution_files: Iterable[str]) -> dict[str, Any]:
    """Rule A2-GOV-2 on a list of tracked ``src/`` and ``scripts/`` files; raises on a forbidden execution file."""
    files = sorted(set(files))
    execution = sorted(set(execution_files))
    problems = [problem for path in execution for problem in execution_file_problems(path)]
    problems += [f"{path} is not a tracked file" for path in execution if path not in files]
    _require(not problems, "; ".join(problems))
    analysis = [path for path in POST_FIT_ANALYSIS_FILES if path in files]
    scientific = [path for path in files if path not in execution and path not in analysis]
    return {
        "execution_files": execution,
        "execution_names_sha256": sha256_canonical(execution),
        "post_fit_analysis_files": analysis,
        "post_fit_analysis_names_sha256": sha256_canonical(list(POST_FIT_ANALYSIS_FILES)),
        "scientific_files": scientific,
    }


def tracked_source_files(repo: str | Path) -> list[str]:
    """Tracked files below ``src/`` and ``scripts/`` (pathspec-limited listing)."""
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--", *TRACKED_CODE_FOLDERS],
                         capture_output=True, check=True).stdout.decode("utf-8")
    return sorted(item for item in out.split("\0") if item)


def module_name(path: str) -> Optional[str]:
    """The import name of a tracked ``src/`` or ``scripts/`` Python file (``src/a/b.py`` -> ``a.b``,
    ``scripts/x.py`` -> ``scripts.x``, a package ``__init__.py`` -> the package)."""
    if not (isinstance(path, str) and path.endswith(".py")):
        return None
    if path.startswith("src/"):
        name = path[len("src/"):-len(".py")].replace("/", ".")
    elif path.startswith("scripts/"):
        name = path[:-len(".py")].replace("/", ".")
    else:
        return None
    return name[:-len(".__init__")] if name.endswith(".__init__") else name


def module_file(name: str, files: Iterable[str]) -> Optional[str]:
    """The tracked file that the import name ``name`` loads from the repository (``src/`` first, then the
    repository root, as the campaign's import roots are), or None for a module outside the repository."""
    files = files if isinstance(files, (set, frozenset)) else set(files)
    parts = str(name).split(".")
    for candidate in (f"src/{'/'.join(parts)}.py", f"src/{'/'.join(parts)}/__init__.py", f"{'/'.join(parts)}.py",
                      f"{'/'.join(parts)}/__init__.py"):
        if candidate in files and candidate.startswith(tuple(f"{folder}/" for folder in TRACKED_CODE_FOLDERS)):
            return candidate
    return None


def resolve_import_from(path: str, node: ast.ImportFrom) -> str:
    """The absolute module name of a ``from ... import`` statement in the file ``path`` (relative levels
    resolved against the file's package)."""
    if not node.level:
        return node.module or ""
    package = (module_name(path) or "").split(".")
    if not path.endswith("__init__.py"):
        package = package[:-1]
    base = package[:len(package) - (node.level - 1)] if node.level - 1 <= len(package) else []
    return ".".join(part for part in (*base, node.module or "") if part)


def _default_protected(name: str) -> bool:
    parts = str(name).split(".")
    candidates = (f"src/{'/'.join(parts)}.py", f"src/{'/'.join(parts)}/__init__.py", f"{'/'.join(parts)}.py")
    return any(path in FORBIDDEN_EXECUTION_FILES or path in R_ID_FILES or path.startswith(FORBIDDEN_EXECUTION_PREFIXES)
               for path in candidates)


_ALLOWED_DUNDERS = frozenset({"__name__", "__file__", "__doc__"})
_NAME_FUNCTIONS = frozenset({"getattr", "setattr", "delattr", "hasattr"})


def _private(name: Any) -> bool:
    return isinstance(name, str) and name.startswith("_") and name not in _ALLOWED_DUNDERS


def _dotted(node: ast.AST) -> Optional[list[str]]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return [node.id, *reversed(parts)]


def private_import_problems(text: str, path: str, *, protected: Optional[Callable[[str], bool]] = None) -> list[str]:
    """An execution file reaches no private name (a leading ``_``, except ``__name__``, ``__file__`` and
    ``__doc__``) of a scientific or post-fit analysis module: no ``from m import _x`` (absolute or relative), no
    attribute ``m._x`` on a name bound to such a module (``import m``, ``import a.m as m``, ``from a import m``,
    ``m = importlib.import_module("a.m")``) and no ``getattr``/``setattr``/``delattr``/``hasattr`` of a private
    literal on it. ``protected(name)`` tells whether an import name is such a module (default: the campaign modules
    of ``FORBIDDEN_EXECUTION_FILES``, ``R_ID_FILES`` and the forbidden prefixes)."""
    protected = protected or _default_protected
    tree = ast.parse(text)
    problems: list[str] = []
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    bound[alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            module = resolve_import_from(path, node)
            for alias in node.names:
                if module and protected(module) and _private(alias.name):
                    problems.append(f"{path} imports the private name {alias.name} of {module}")
                full = f"{module}.{alias.name}" if module else alias.name
                if protected(full):
                    bound[alias.asname or alias.name] = full
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and isinstance(node.value, ast.Call) and node.value.args
              and isinstance(node.value.args[0], ast.Constant) and isinstance(node.value.args[0].value, str)
              and (_dotted(node.value.func) or [None])[-1] in ("import_module", "__import__")):
            bound[node.targets[0].id] = node.value.args[0].value

    def target(expression: ast.AST) -> Optional[str]:
        parts = _dotted(expression)
        if not parts or parts[0] not in bound:
            return None
        name = ".".join([bound[parts[0]], *parts[1:]])
        return name if protected(name) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _private(node.attr):
            module = target(node.value)
            if module is not None:
                problems.append(f"{path} reaches the private name {node.attr} of {module}")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _NAME_FUNCTIONS
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and _private(node.args[1].value)):
            module = target(node.args[0])
            if module is not None:
                problems.append(f"{path} reaches the private name {node.args[1].value} of {module} through "
                                f"{node.func.id}")
    return sorted(set(problems))


# ---------------------------------------------------------------------------
# Required bindings (contract fixture: record_builder.required_bindings)


def _safe_relative(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and not value.startswith(("/", "~")) and "\\" not in value
            and ".." not in value.split("/") and ":" not in value.split("/")[0])


def is_file_binding(value: Any) -> bool:
    """A public file binding ``{path, sha256}`` (further descriptive keys allowed, never ``private``)."""
    return (isinstance(value, Mapping) and "private" not in value and _safe_relative(value.get("path"))
            and isinstance(value.get("sha256"), str) and bool(_SHA256_RE.fullmatch(value["sha256"])))


def is_private_role(where: Any) -> bool:
    """Whether the dotted record path ``where`` is a private role or lies below one."""
    return isinstance(where, str) and any(where == role or where.startswith(role + ".") for role in PRIVATE_ROLES)


def is_private_path(value: Any) -> bool:
    """A safe repository-relative POSIX path of a file below ``PRIVATE_BINDING_ROOT`` (no ``..``, ``.`` or empty
    component, not absolute)."""
    if not (_safe_relative(value) and value.startswith(PRIVATE_BINDING_ROOT)
            and len(value) > len(PRIVATE_BINDING_ROOT)):
        return False
    return all(part not in ("", ".", "..") for part in value.split("/"))


def is_private_binding(value: Any) -> bool:
    """A private file binding: exactly ``{path, sha256, private: true}`` with ``is_private_path`` (no other key, so a
    private binding carries no private content into the record)."""
    return (isinstance(value, Mapping) and sorted(value) == list(PRIVATE_BINDING_KEYS) and value.get("private") is True
            and is_private_path(value.get("path")) and isinstance(value.get("sha256"), str)
            and bool(_SHA256_RE.fullmatch(value["sha256"])))


def _below_private_root(path: Any) -> bool:
    if not isinstance(path, str):
        return False
    normal = posixpath.normpath(path.lstrip("/"))
    return (normal + "/").startswith(PRIVATE_BINDING_ROOT)


def private_link_problem(root: Any, path: Any) -> Optional[str]:
    """Why the private file ``path`` (``is_private_path``) of the working tree ``root`` breaks the data-root link
    rule, as a phrase that follows the file's name, or None. The rule: a symbolic link is allowed only on a component
    of ``PRIVATE_ROOT_LINK_COMPONENTS`` (the data root itself), and such a link points to an existing directory
    outside the repository ``root`` (by its real path), so a private file is never a repository file reached by
    another name; every component below the campaign folder (its subfolders and the file itself) is not a link; a
    file reached through data-root links resolves outside the repository, one reached without a link resolves to
    itself. The record builder (``amendment2._private_binding``) and the chain verifier (``amendment_chain``, C8 and
    C11) both apply it, so the two cannot drift."""
    if not is_private_path(path):
        return f"is not below the private binding root {PRIVATE_BINDING_ROOT}"
    base = Path(os.path.realpath(os.fspath(root)))
    lexical = Path(os.fspath(root))
    parts = path.split("/")
    linked = False
    for count in range(1, len(parts) + 1):
        component = "/".join(parts[:count])
        cursor = lexical.joinpath(*parts[:count])
        if not cursor.is_symlink():
            continue
        if component not in PRIVATE_ROOT_LINK_COMPONENTS:
            return (f"has a symbolic link at {component}, below the campaign folder; only the data-root components "
                    f"{list(PRIVATE_ROOT_LINK_COMPONENTS)} may be links (every folder below the campaign folder and "
                    "the file itself are real)")
        if not cursor.is_dir():
            return (f"is reached through the data-root link {component}, which is dangling or not a directory; a "
                    "data-root link points to an existing directory outside the repository")
        target = Path(os.path.realpath(cursor))
        if target == base or base in target.parents:
            return (f"is reached through the data-root link {component}, which points into the repository; a "
                    "data-root link points outside the repository (a private file is never a repository file "
                    "reached by another name)")
        linked = True
    real = Path(os.path.realpath(lexical.joinpath(*parts)))
    inside = real == base or base in real.parents
    if (linked and inside) or (not linked and real != base.joinpath(*parts)):
        return "does not resolve to itself inside the repository or to a file outside it through a data-root link"
    return None


def binding_at(record: Mapping[str, Any], role: str) -> Any:
    """The value at the dotted record path ``role`` (None when a step is absent)."""
    value: Any = record
    for key in role.split("."):
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _shape_problems(value: Any, shape: str, where: str, private: bool = False) -> list[str]:
    """``value`` against a binding shape; with ``private`` every file binding of it is a private binding."""
    leaf = is_private_binding if private else is_file_binding
    kind = (f"private file binding {{path, sha256, private: true}} below {PRIVATE_BINDING_ROOT}" if private
            else "file binding {path, sha256}")
    if shape == "file_binding":
        return [] if leaf(value) else [f"{where} is not a {kind}"]
    if shape == "fold_bindings":
        folds = [str(fold) for fold in range(N_OUTER_FOLDS)]
        if not (isinstance(value, Mapping) and sorted(value) == folds):
            return [f"{where} does not bind exactly the folds 0-{N_OUTER_FOLDS - 1}"]
        problems = []
        for fold, item in sorted(value.items()):
            if leaf(item):
                continue
            if not (isinstance(item, Mapping) and item and "path" not in item
                    and all(leaf(entry) for entry in item.values())):
                problems.append(f"{where}.{fold} is neither a {kind} nor an object of them")
        return problems
    if shape == "corpus_bindings":
        if not isinstance(value, Mapping):
            return [f"{where} is not an object of corpus bindings {{tables, union_fasta}}"]
        return (_shape_problems(value.get("tables"), "fold_bindings", f"{where}.tables", private)
                + _shape_problems(value.get("union_fasta"), "file_binding", f"{where}.union_fasta", private))
    return [f"{where} has the unknown binding shape {shape!r}"]


def required_binding_problems(record: Mapping[str, Any]) -> list[str]:
    """Every role of ``REQUIRED_BINDINGS`` holds a binding of its shape, private below ``PRIVATE_ROLES`` and public
    elsewhere (the files are hashed by the caller)."""
    return [problem for role, shape in sorted(REQUIRED_BINDINGS.items())
            for problem in _shape_problems(binding_at(record, role), shape, role, is_private_role(role))]


def file_bindings(value: Any, where: str) -> list[tuple[str, Any, Any]]:
    """Every ``{"path", "sha256"}`` object below ``value`` as ``(where, path, sha256)``."""
    entries: list[tuple[str, Any, Any]] = []

    def walk(item: Any, at: str) -> None:
        if isinstance(item, Mapping):
            if "path" in item and "sha256" in item:
                entries.append((at, item.get("path"), item.get("sha256")))
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                walk(child, f"{at}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{at}[{index}]")

    walk(value, where)
    return entries


def binding_items(value: Any, where: str) -> list[tuple[str, Mapping[str, Any]]]:
    """Every object below ``value`` that holds ``path`` and ``sha256``, as ``(where, object)`` (the order and the
    dotted ``where`` of ``file_bindings``)."""
    items: list[tuple[str, Mapping[str, Any]]] = []

    def walk(item: Any, at: str) -> None:
        if isinstance(item, Mapping):
            if "path" in item and "sha256" in item:
                items.append((at, item))
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                walk(child, f"{at}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{at}[{index}]")

    walk(value, where)
    return items


def binding_privacy_problems(record: Mapping[str, Any], sections: Iterable[str]) -> list[str]:
    """Every file binding below the record ``sections`` has the privacy of its role: below ``PRIVATE_ROLES`` exactly
    ``{path, sha256, private: true}`` with a safe path below ``PRIVATE_BINDING_ROOT``; anywhere else a public
    ``{path, sha256}`` without ``private`` and outside the private root. Each problem starts with the binding's
    dotted record path."""
    problems: list[str] = []
    for section in sections:
        for where, item in binding_items(record.get(section), section):
            path = item.get("path")
            if is_private_role(where):
                if item.get("private") is not True:
                    problems.append(f"{where} binds a private input publicly; a private role holds "
                                    f"{{path, sha256, private: true}} below {PRIVATE_BINDING_ROOT}")
                elif not is_private_path(path):
                    problems.append(f"{where}: the private path {path!r} is not a safe repository-relative path "
                                    f"below {PRIVATE_BINDING_ROOT}")
                elif not is_private_binding(item):
                    problems.append(f"{where} is not a private file binding {{path, sha256, private: true}} (no "
                                    "other key)")
            elif "private" in item:
                problems.append(f"{where} flags a public binding private; only {list(PRIVATE_ROLES)} are private")
            elif _below_private_root(path):
                problems.append(f"{where} binds {path!r} below the private binding root publicly; a public binding "
                                "is a committed repository file")
    return problems


# ---------------------------------------------------------------------------
# Change classes


def changed_set(changes: Iterable[str], execution_files: Iterable[str]) -> dict[str, list[str]]:
    """Changed tracked files split by the partition sets."""
    execution = set(execution_files)
    split = {"execution": [], "post_fit_analysis": [], "scientific": [], "other": []}
    for path in sorted(set(changes)):
        if not path.startswith(tuple(f"{folder}/" for folder in TRACKED_CODE_FOLDERS)):
            split["other"].append(path)
        elif path in execution:
            split["execution"].append(path)
        elif path in POST_FIT_ANALYSIS_FILES:
            split["post_fit_analysis"].append(path)
        else:
            split["scientific"].append(path)
    return split


def check_change_class(link: Mapping[str, Any], *, changed_files: Iterable[str],
                       execution_files: Iterable[str], fut_record_exists: bool = False,
                       stop_futility_exists: bool = False, previous: Optional[Mapping[str, Any]] = None) -> list[str]:
    """Problems of a link of sequence 3 or higher (rule A2-GOV-7); an empty list means the class holds.

    (a) ``execution``: only execution files change. (a2) ``analysis_completion``: only post-fit analysis files
    change, after a fold-0 FUT record exists (``check_analysis_completion`` adds its other limits). (b)
    ``defect``: a scientific change with its defect fields, never after a stop. Anything else is class (c).
    With ``previous`` (the previous link's record), classes (a) and (a2) keep its pinned execution and analysis
    functions: an execution change never alters a frozen gate (rule A2-GOV-2).
    """
    problems: list[str] = []
    sequence = link.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 3:
        return [f"a later link has sequence {sequence!r}; links 1 and 2 are fixed"]
    for name in LATER_LINK_REQUIRED_FIELDS:
        if name not in link:
            problems.append(f"the link of sequence {sequence} lacks {name}")
    change_class = link.get("change_class")
    split = changed_set(changed_files, execution_files)
    if change_class == CHANGE_EXECUTION:
        if split["scientific"] or split["post_fit_analysis"]:
            problems.append("an execution change may change only execution files")
    elif change_class == CHANGE_ANALYSIS_COMPLETION:
        if split["scientific"] or split["execution"]:
            problems.append("an analysis completion may change only the four post-fit analysis files")
        if not fut_record_exists:
            problems.append("an analysis completion is published only after the fold-0 FUT record")
    elif change_class == CHANGE_DEFECT:
        problems += [f"a defect amendment lacks {name}" for name in DEFECT_REQUIRED_FIELDS if name not in link]
        if stop_futility_exists:
            problems.append("after stop_futility no defect amendment may rerun S1, S2 or FUT (rule A2-GOV-7(d))")
    else:
        problems.append(f"change class {change_class!r} is outside the campaign (class (c): stop or successor)")
    if previous is not None and change_class in PIN_KEEPING_CLASSES:
        problems += [f"an {change_class} link changes {key} (the pinned functions stay those of the previous link)"
                     for key in PINNED_FUNCTION_KEYS if link.get(key) != previous.get(key)]
    return problems


def check_analysis_completion(*, previous: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """The limits of class (a2) between the record state before and after the link: unchanged analysis-function
    pins, ``live_constants``, scientific identity and fit chain; every rule's policy SHA-256 unchanged (all
    ``RULE_POLICY_FUNCTIONS`` entries, the analysis rule ``amendment2_accuracy_v1`` included, so no decision rule,
    metric, population, threshold, seed, draw count, family composition or decision sentence moves); every refusal
    fixture still refused (the refusal set never shrinks) and every decision fixture with identical decision
    fields. ``rule_policy_sha256`` maps each rule ID to the SHA-256 of its canonical policy record."""
    problems = []
    for key in ("pinned_analysis_functions", "live_constants_sha256", "scientific_identity_sha256", "fit_chain"):
        if previous.get(key) != current.get(key):
            problems.append(f"{key} changed in an analysis completion")
    before_rules, after_rules = previous.get("rule_policy_sha256"), current.get("rule_policy_sha256")
    if not isinstance(before_rules, Mapping) or set(before_rules) != set(RULE_POLICY_FUNCTIONS):
        problems.append("the earlier state does not bind the policy SHA-256 of every rule")
    elif not isinstance(after_rules, Mapping) or dict(after_rules) != dict(before_rules):
        changed = sorted(rule for rule in before_rules if (after_rules or {}).get(rule) != before_rules[rule])
        problems.append(f"rule policies changed in an analysis completion: {changed}")
    before, after = set(previous.get("refused_fixtures") or ()), set(current.get("refused_fixtures") or ())
    if not before <= after:
        problems.append(f"{len(before - after)} refusal fixtures are no longer refused")
    if (previous.get("decision_fields") or {}) != (current.get("decision_fields") or {}):
        problems.append("a decision fixture gives other decision fields")
    return problems


# ---------------------------------------------------------------------------
# Rule r_cmd_v2


def r_cmd_v2_change_problems(changes: Iterable[tuple[str, str]], execution_files: Iterable[str]) -> list[str]:
    """Changes (status, path) that ``r_cmd_v2`` does not allow without new dry runs: the Freeze-1 R-CMD allowlist
    and the execution files may be modified; added modules and scripts are allowed (their import is checked
    separately); everything else needs a new amendment and new dry runs."""
    from ec_clean30.freeze import R_CMD_ALLOWED_MODIFIED, r_cmd_change_problems

    execution = set(execution_files)
    remaining = [(status, path) for status, path in changes if not (status == "M" and path in execution)]
    problems = r_cmd_change_problems(remaining)
    allowed = set(R_CMD_ALLOWED_MODIFIED) | execution
    return [problem.replace("rule R-CMD", R_CMD_V2_RULE) for problem in problems
            if not any(problem.startswith(f"{path} was modified") for path in allowed)]


def r_cmd_v2_check(dry_run_head: str, repo: str | Path, plans: Mapping[str, Mapping[str, str]],
                   execution_file_set: Iterable[str]) -> dict[str, Any]:
    """Rule ``r_cmd_v2`` from the amendment-2 dry-run HEAD to HEAD (rule A2-GOV-4): the allowlist above, the pinned
    runner functions unchanged, added files not loaded by the protected importers, and the v1 and v2 plans equal to
    their records (``plans``: ``{"v1": {fold: plan_sha256}, "v2": {fold: plan_sha256}}``)."""
    from ec_clean30 import freeze
    from ec_clean30.campaign import build_fold_plan, build_fold_plan_v2

    repo = Path(repo)
    problems: list[str] = []
    head = freeze._resolve_commit(repo, "HEAD")
    base = freeze._resolve_commit(repo, dry_run_head) if re.fullmatch(r"[0-9a-f]{40}", str(dry_run_head)) else None
    _require(head is not None, f"{repo} has no commit.")
    changes: list[tuple[str, str]] = []
    if base is None:
        problems.append(f"the dry-run HEAD {str(dry_run_head)[:12]!r} is not a full commit SHA of this repository")
    else:
        changes = freeze._name_status(repo, base)
        problems += r_cmd_v2_change_problems(changes, execution_file_set)
        live_text = (repo / freeze.R_CMD_RUNNER).read_text(encoding="utf-8")
        old_text = freeze._show(repo, base, freeze.R_CMD_RUNNER)
        live, live_problems = freeze.runner_function_sha256(live_text)
        problems += live_problems
        if old_text is None:
            problems.append(f"{freeze.R_CMD_RUNNER} does not exist at the dry-run HEAD")
        else:
            old, _old_problems = freeze.runner_function_sha256(old_text)
            problems += [f"the pinned runner function {name} changed after the dry run"
                         for name in freeze.PINNED_RUNNER_FUNCTIONS if old.get(name) != live.get(name)]
        added = {path for status, path in changes if status == "A" and path.endswith(".py")
                 and not path.startswith(freeze.R_CMD_FREE_FOLDERS)}
        loaded, import_problem = freeze.protected_import_files(repo)
        problems += [import_problem] if import_problem else [
            f"{path}, added after the dry run, is loaded by the protected importers" for path in sorted(added & loaded)]
    for version, builder in (("v1", build_fold_plan), ("v2", build_fold_plan_v2)):
        for fold, expected in sorted((plans.get(version) or {}).items()):
            if builder(int(fold))["plan_sha256"] != expected:
                problems.append(f"the live {version} plan of fold {fold} differs from its record")
    return {"rule": R_CMD_V2_RULE, "status": "passed" if not problems else "failed", "head": head,
            "dry_run_head": base, "changes": [{"status": status, "path": path} for status, path in changes],
            "problems": problems}


# ---------------------------------------------------------------------------
# Scientific identity and policies


def scientific_identity_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical scientific-identity record: exactly ``SCIENTIFIC_IDENTITY_KEYS``, none of the forbidden keys."""
    forbidden = sorted(key for key in record if key in FORBIDDEN_IDENTITY_KEYS)

    def nested(value: Any, where: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in FORBIDDEN_IDENTITY_KEYS:
                    forbidden.append(f"{where}.{key}")
                nested(item, f"{where}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                nested(item, f"{where}[{index}]")

    nested(record.get("deferred_input_specifications"), "deferred_input_specifications")
    _require(not forbidden, f"A scientific identity holds no execution, commit, provider, hardware or session "
                            f"field; got {forbidden}.")
    unknown = sorted(set(record) - set(SCIENTIFIC_IDENTITY_KEYS))
    missing = sorted(set(SCIENTIFIC_IDENTITY_KEYS) - set(record))
    _require(not unknown and not missing, f"Scientific identity keys differ (unknown {unknown}, missing {missing}).")
    _require(record["freeze1_sha256"] == FREEZE1_SHA256 and record["freeze1b_sha256"] == FREEZE1B_SHA256,
             "The scientific identity names another Freeze 1 or Freeze 1b.")
    chain = record["fit_chain"]
    _require(isinstance(chain, list) and chain and chain[0] == AMENDMENT1["sha256"]
             and all(isinstance(item, str) and _SHA256_RE.fullmatch(item) for item in chain),
             "The fit chain must start with amendment 1 and hold SHA-256 values.")
    return json.loads(canonical_json(dict(record)))


def scientific_identity(record: Mapping[str, Any]) -> str:
    return sha256_canonical(scientific_identity_record(record))


def _resolve(spec: str) -> Callable[[], Any]:
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def rule_policy_sha256(*, include_analysis: bool = True) -> dict[str, str]:
    """``rule_id -> SHA-256`` of each rule's canonical policy record (the state ``check_analysis_completion``
    compares; the analysis rule is included by default)."""
    return {rule_id: sha256_canonical(item["policy"])
            for rule_id, item in policy_records(include_analysis=include_analysis).items()}


def policy_records(*, include_analysis: bool = False) -> dict[str, dict[str, Any]]:
    """``rule_id -> {"rule_id", "policy"}`` from the live functions (analysis rules only on request)."""
    records = {}
    for rule_id, entry in RULE_POLICY_FUNCTIONS.items():
        if entry["analysis"] and not include_analysis:
            continue
        records[rule_id] = {"rule_id": rule_id, "policy": json.loads(canonical_json(_resolve(entry["function"])()))}
    return records
