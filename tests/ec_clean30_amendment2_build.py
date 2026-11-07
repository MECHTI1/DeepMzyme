"""Helpers of the amendment-2 record tests (synthetic only; no campaign record, view or data folder is read).

- ``record_shape_problems`` and ``draft_problems`` check a record or a draft against the contract fixture
  (``amendment_record.record_shape`` and ``record_builder``); they read nothing from the builder or the verifier.
- ``build_world_with_builder`` builds link 2 with the real ``amendment2.build_amendment2`` on the synthetic published
  repository of ``test_ec_clean30_amendment_chain`` (execution files first added after link 1, as in the campaign
  repository), with every required role bound by a dotted ``--bind`` name and the rule text read from the reviewed
  document ``amendment_draft.md``, renders it to ``amendment.md`` with ``amendment2.render_amendment2``, and
  commits and publishes it with its strict log line. The builder runs in the synthetic repository's own
  interpreter, so every live value is that repository's.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from typing import Any, Optional

import test_ec_clean30_amendment_chain as harness
from ec_clean30 import amendment2, freeze
from ec_clean30 import amendment_rules as rules
from ec_clean30.campaign import PLAN_FORMAT_V2

CONTRACT = harness.CONTRACT
SHAPE = CONTRACT["amendment_record"]["record_shape"]
BUILDER = CONTRACT["record_builder"]
FOLDS = [str(fold) for fold in range(5)]
DOCS = harness.DOCS
DRAFT_PATH = f"{DOCS}/amendment2/amendment_draft.json"
DECISIONS_PATH = f"{DOCS}/amendment2/maintainer_decisions.json"
RECEIPT_PATH = f"{DOCS}/amendment2/dry_runs/fold{{fold}}.json"
RULE_TEXT = harness.RULE_TEXT
R_CMD_V2_FUNCTION = next(spec for spec in CONTRACT["science_api"]["rules"] if spec.endswith(":r_cmd_v2_check"))

# ---------------------------------------------------------------------------
# The contract fixture's shapes as a checker

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _relative(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and not value.startswith(("/", "~")) and "\\" not in value
            and ".." not in value.split("/"))


_TOKENS = {
    "sha256": lambda value: isinstance(value, str) and bool(_SHA256.fullmatch(value)),
    "sha256_or_null": lambda value: value is None or (isinstance(value, str) and bool(_SHA256.fullmatch(value))),
    "commit": lambda value: isinstance(value, str) and bool(_COMMIT.fullmatch(value)),
    "relative_path": _relative,
    "string": lambda value: isinstance(value, str) and bool(value.strip()),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "object": lambda value: isinstance(value, Mapping),
    "list": lambda value: isinstance(value, list),
}
_KEY_KINDS = {
    "fold": lambda key: key in FOLDS,
    "function": lambda key: (isinstance(key, str) and key.count("::") == 1 and _relative(key.split("::")[0])
                             and key.split("::")[1].isidentifier()),
    "rule_id": lambda key: isinstance(key, str) and bool(key.strip()),
    "relative_path": _relative,
}


def shape_problems(value: Any, spec: Any, where: str) -> list[str]:
    """``value`` against one shape of the fixture (a type token, a named shape, keys, map, list or const)."""
    if isinstance(spec, str):
        if spec in SHAPE["named"]:
            return shape_problems(value, SHAPE["named"][spec], where)
        return [] if _TOKENS[spec](value) else [f"{where} is not {spec}"]
    if "const" in spec:
        expected = spec["const"]
        return [] if value == expected and type(value) is type(expected) else [f"{where} is not {expected!r}"]
    if "keys" in spec:
        if not isinstance(value, Mapping):
            return [f"{where} is not an object"]
        problems: list[str] = []
        for key, item in spec["keys"].items():
            problems += shape_problems(value[key], item, f"{where}.{key}") if key in value else [
                f"{where}.{key} is missing"]
        return problems
    if "map" in spec:
        if not isinstance(value, Mapping):
            return [f"{where} is not an object"]
        problems = [] if spec["key"] != "fold" or sorted(value) == FOLDS else [f"{where} does not hold the folds 0-4"]
        for key, item in value.items():
            if not _KEY_KINDS[spec["key"]](key):
                problems.append(f"{where} has the key {key!r}, which is not a {spec['key']}")
            problems += shape_problems(item, spec["map"], f"{where}[{key!r}]")
        return problems
    if "list" in spec:
        if not isinstance(value, list):
            return [f"{where} is not a list"]
        return [problem for index, item in enumerate(value)
                for problem in shape_problems(item, spec["list"], f"{where}[{index}]")]
    raise ValueError(f"unknown shape {spec!r}")


def _role_value(record: Mapping[str, Any], role: str) -> Any:
    value: Any = record
    for key in role.split("."):
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _binding_shape_problems(value: Any, kind: str, where: str) -> list[str]:
    """One role of ``record_shape.required_bindings`` against its shape kind (read from the fixture's words)."""
    if kind == "file_binding":
        return shape_problems(value, "file_binding", where)
    if kind == "fold_bindings":
        if not (isinstance(value, Mapping) and sorted(value) == FOLDS):
            return [f"{where} does not bind the folds 0-4"]
        problems: list[str] = []
        for fold, item in value.items():
            if isinstance(item, Mapping) and "path" in item:
                problems += shape_problems(item, "file_binding", f"{where}.{fold}")
            elif isinstance(item, Mapping) and item:
                problems += [problem for name, entry in item.items()
                             for problem in shape_problems(entry, "file_binding", f"{where}.{fold}.{name}")]
            else:
                problems.append(f"{where}.{fold} binds no file")
        return problems
    if kind == "corpus_bindings":
        if not isinstance(value, Mapping):
            return [f"{where} is not an object"]
        return (_binding_shape_problems(value.get("tables"), "fold_bindings", f"{where}.tables")
                + _binding_shape_problems(value.get("union_fasta"), "file_binding", f"{where}.union_fasta"))
    raise ValueError(f"unknown binding shape {kind!r}")


def record_shape_problems(record: Mapping[str, Any]) -> list[str]:
    """A link-2 record against ``required_keys``, ``record_shape.fields``, the required bindings and the link-2
    data state."""
    problems = [f"lacks {key}" for key in (*CONTRACT["amendment_record"]["required_keys"], *SHAPE["built_keys"])
                if key not in record]
    for key, spec in SHAPE["fields"].items():
        problems += shape_problems(record[key], spec, key) if key in record else [f"lacks {key}"]
    for role, kind in SHAPE["required_bindings"]["roles"].items():
        problems += _binding_shape_problems(_role_value(record, role), kind, role)
    state = record.get("data_state") or {}
    problems += [f"data_state.{key} is not {value!r}" for key, value in SHAPE["link2"]["data_state"].items()
                 if state.get(key) != value or type(state.get(key)) is not type(value)]
    if record.get("fold0_records", []) != []:
        problems.append("link 2 lists fold-0 records")
    return sorted(set(problems))


def draft_problems(draft: Mapping[str, Any]) -> list[str]:
    """A reviewed draft against ``record_builder``: format, draft keys, sub-keys and pinned-function name lists."""
    problems = [] if draft.get("format") == BUILDER["draft_format"] else ["the draft has another format"]
    problems += [f"lacks {key}" for key in BUILDER["draft_keys"] if key not in draft]
    for key, subkeys in BUILDER["draft_subkeys"].items():
        problems += [f"{key} lacks {subkey}" for subkey in subkeys if subkey not in (draft.get(key) or {})]
    for key in BUILDER["draft_function_names"]:
        problems += shape_problems(draft.get(key), {"map": {"list": "string"}, "key": "relative_path"}, key)
    return problems


# ---------------------------------------------------------------------------
# Link 2 built by the real builder on the synthetic repository

_BUILD_PROBE = textwrap.dedent('''\
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from ec_clean30 import amendment2
    args = json.loads(sys.argv[2])


    def paths(items):
        return None if items is None else {key: Path(value) for key, value in items.items()}


    try:
        record = amendment2.build_amendment2(
            draft_path=Path(args["draft"]), decisions_path=Path(args["decisions"]),
            amendment_draft_md=Path(args["amendment_draft_md"]), repo=Path(args["repo"]),
            bound_files=paths(args["bind"]),
            out=Path(args["out"]), plan_v2=paths(args["plan_v2"]), dry_run_receipts=paths(args["dry_run_receipts"]))
    except (amendment2.AmendmentRecordError, FileExistsError) as exc:
        print(json.dumps({"error": type(exc).__name__ + ": " + str(exc)}))
    else:
        if args["render"]:
            Path(args["render"]).write_text(amendment2.render_amendment2(record), encoding="utf-8")
        print(json.dumps({"record": record}))
    ''')


def role_binds(roles: dict[str, Any], repo: Path) -> dict[str, Path]:
    """``--bind`` names (dotted roles of ``amendment_rules.REQUIRED_BINDINGS``, a key below one for each file of a
    fold or corpus role) and the repository files they bind."""
    binds: dict[str, Path] = {}
    for role, value in roles.items():
        for where, relative, _digest in rules.file_bindings(value, role):
            binds[where] = repo / relative
    return binds


def write_build_inputs(world: dict) -> dict[str, Any]:
    """The maintainer's inputs on the source world: five dry-run-v2 receipts of the dry-run commit D, five plan-v2
    records in a campaign root outside the repository, the decisions file, the bound fixtures, the files of every
    required role (bound with dotted ``--bind`` names), the reviewed document ``amendment_draft.md`` with the rule
    text, and the reviewed draft, whose keys are exactly the fixture's draft keys and whose required roles hold notes
    only, as the campaign's draft does."""
    repo, root, commits = world["repo"], world["root"], world["commits"]
    receipts, plans = {}, {}
    for fold in FOLDS:
        relative = RECEIPT_PATH.format(fold=fold)
        harness._write(repo, relative, json.dumps({
            "format": amendment2.DRY_RUN_V2_RECEIPT_FORMAT, "schema_version": 1, "status": "passed", "fold": int(fold),
            "plan_version": 2, "provenance": {"smoke": False, "git": {"head": commits["D"], "clean": True}}},
            sort_keys=True))
        receipts[fold] = repo / relative
        plan = root / "campaign" / "plans_v2" / f"fold{fold}.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps({"format": PLAN_FORMAT_V2, "fold": int(fold),
                                    "plan_sha256": world["values"]["v2"][fold]}, sort_keys=True))
        plans[fold] = plan
    answers = {key: {"choice": "the recommended default", "accepted_default": True}
               for key in amendment2.CONFIRMATION_IDS}
    harness._write(repo, DECISIONS_PATH, json.dumps({"format": amendment2.DECISIONS_FORMAT, "maintainer": "synthetic",
                                                     "answers": answers}, indent=2, sort_keys=True))
    harness._write(repo, harness.REFUSALS_FIXTURE, json.dumps(harness.SYNTHETIC_REFUSALS))
    harness._write(repo, harness.DECISIONS_FIXTURE, json.dumps(harness.SYNTHETIC_DECISIONS))
    harness.write_rule_text(repo, rules.AMENDMENT2_SEQUENCE)
    rule_text = repo / rules.rule_text_source(rules.AMENDMENT2_SEQUENCE)
    binds = role_binds(harness.write_role_files(repo), repo)
    previous = world["previous_pins"]
    changes = {path: {"before": previous.get(path), "after": digest, "written_by": "science"}
               for path, digest in sorted(harness._tree_pins(repo).items())
               if path not in harness.EXECUTION_FILES and previous.get(path) != digest}
    values = {
        "schema_version": 1, "campaign_id": rules.CAMPAIGN_ID, "amendment_id": rules.AMENDMENT2_ID,
        "sequence": rules.AMENDMENT2_SEQUENCE, "change_class": rules.CHANGE_PRE_DATA,
        "base_freeze1": {"path": harness.FREEZE1_PATH, "sha256": world["freeze1_sha256"]},
        "base_freeze1b_sha256": rules.FREEZE1B_SHA256,
        "base_amendment": {"sequence": 1, "amendment_id": freeze.S1_AMENDMENT_ID, "path": rules.AMENDMENT1["path"],
                           "sha256": world["a1"], "commit": commits["B"]},
        "previous_sha256": world["a1"], "chain_before": [world["a1"]], "timestamps": {"draft_written": "2026-11-04"},
        "data_state": {**SHAPE["link2"]["data_state"], "attested_by_log": "synthetic",
                       "attestation_at_build": "required (rule A2-GOV-6); bound at build"}, "decisions": {},
        "rules": {**{rule_id: {"policy_function": entry["function"], "analysis": entry["analysis"]}
                     for rule_id, entry in rules.RULE_POLICY_FUNCTIONS.items()},
                  rules.R_CMD_V2_RULE: {"policy_function": R_CMD_V2_FUNCTION, "analysis": False}},
        "supersedes": [], "plans": {"v1_plan_sha256": world["values"]["v1"], "format_v2": PLAN_FORMAT_V2},
        "dry_runs": {"rule": rules.R_CMD_V2_RULE, "receipts": "written once by dry-run-v2; bound at build",
                     "in_view_receipts_unchanged": {fold: world["freeze1_doc"]["views"][fold]["dry_run_receipt"]["sha256"]
                                                    for fold in FOLDS},
                     "v1_resolved_config": "digests at the base and the dry-run HEAD; bound at build"},
        "source_parent_commit": commits["B"],
        "pinned_execution_functions": {path: list(names) for path, names in rules.PINNED_EXECUTION_FUNCTIONS.items()},
        "post_fit_analysis_files": list(rules.POST_FIT_ANALYSIS_FILES),
        "pinned_analysis_functions": {path: list(names) for path, names in rules.PINNED_ANALYSIS_FUNCTIONS.items()},
        "source_changes": changes,
        "unchanged_scientific_files": ["src/ec_clean30/freeze.py", "src/ec_clean30/provenance.py"],
        "inputs": {"clean_tie_exposure": "bound at build", "f6_corpus": "bound at build", "f6_holdout": "bound at build"},
        "deferred_bindings": {"f6_inputs": {"bound_by": "freeze1c_v2", "spec": "A2-F6-4"}},
        "fixtures": {name: {"path": path, "sha256": harness._file_sha(repo, path)}
                     for name, path in (("refusals", harness.REFUSALS_FIXTURE),
                                        ("decisions", harness.DECISIONS_FIXTURE))},
        "evidence": {"futility_oc": "bound at build", "interface_contract": "bound at build",
                     "power_simulation": "bound at build", "verification": "bound at build"},
        "deviations": [], "publication": SHAPE["fields"]["publication"]["const"],
        "execution_file_set": {"declared": sorted(harness.EXECUTION_FILES), "pending": [],
                               "status": "derived at build; must equal the declared closed list"},
    }
    draft = {"format": BUILDER["draft_format"], "status": "draft_for_review", "status_note": "synthetic review draft",
             "confirmation_status": "pending_maintainer_confirmation", "provisional_markers_settled": False,
             "source_changes_pending": {}, **{key: values[key] for key in BUILDER["draft_keys"]}}
    harness._write(repo, DRAFT_PATH, json.dumps(draft, indent=2, sort_keys=True) + "\n")
    return {"draft": repo / DRAFT_PATH, "decisions": repo / DECISIONS_PATH, "amendment_draft_md": rule_text,
            "plan_v2": plans, "dry_run_receipts": receipts, "bind": binds}


def run_builder(world: dict, *, out: Path, render: Optional[Path] = None, bind: Optional[dict] = None,
                drop_binds: tuple[str, ...] = (), **overrides: Any) -> dict[str, Any]:
    """``amendment2.build_amendment2`` in the synthetic repository's interpreter: ``{"record": ...}`` or
    ``{"error": "<exception>: <message>"}``. ``overrides`` replace the inputs of ``world["inputs"]`` (None drops a
    build input); ``bind`` adds bound files to the required-role binds and ``drop_binds`` removes binds whose name
    starts with one of its prefixes."""
    inputs = {**world["inputs"], **overrides}
    binds = {name: path for name, path in {**inputs.get("bind", {}), **(bind or {})}.items()
             if not any(name == prefix or name.startswith(prefix + ".") for prefix in drop_binds)}
    args = {"draft": str(inputs["draft"]), "decisions": str(inputs["decisions"]),
            "amendment_draft_md": str(inputs["amendment_draft_md"]), "repo": str(world["repo"]), "out": str(out),
            "render": str(render) if render else None,
            "bind": {name: str(path) for name, path in binds.items()},
            "plan_v2": None if inputs["plan_v2"] is None else {k: str(v) for k, v in inputs["plan_v2"].items()},
            "dry_run_receipts": None if inputs["dry_run_receipts"] is None else {
                k: str(v) for k, v in inputs["dry_run_receipts"].items()}}
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", _BUILD_PROBE, str(world["repo"] / "src"),
                                json.dumps(args)], capture_output=True, text=True, cwd=str(world["repo"]))
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def publish_link2(world: dict, message: str = "Link 2") -> str:
    """Enter the link-2 log line in the campaign log, commit it with the link and its rendering and push; returns
    the link-2 SHA-256."""
    digest = harness._file_sha(world["repo"], rules.AMENDMENT2_PATH)
    harness.log_link(world["repo"], rules.AMENDMENT2_SEQUENCE, digest)
    world["commits"] = {**world["commits"], "S2": harness._commit(world["repo"], message)}
    return digest


def build_world_with_builder(root: Path) -> dict:
    """The source world, the maintainer's inputs, link 2 by the real builder with its rendering, published."""
    world = harness.build_source_world(root, execution_after_link1=True)
    world["inputs"] = write_build_inputs(world)
    out = world["repo"] / rules.AMENDMENT2_PATH
    result = run_builder(world, out=out, render=world["repo"] / rules.rendering_path(rules.AMENDMENT2_PATH))
    assert "record" in result, result
    world["record"] = result["record"]
    world["a2"] = publish_link2(world)
    return world
