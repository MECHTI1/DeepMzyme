"""General amendment-chain verifier (``ec_clean30.amendment_chain``) on synthetic published repositories.

Each synthetic repository is built below the pytest temporary root: ``git init``, a bare ``origin`` remote, copies of
the real verifier, rule, freeze, provenance and record-builder modules, and small synthetic stand-ins for every
other campaign module. Its history is that of the campaign: Freeze 1 (commit A), link 1 by the real
``freeze.build_s1_amendment`` (B), several execution-only commits with a merge (E1, E2 on a side branch, E3, M),
the amendment-2 source with the synthetic refusal and decision-fixture tests (dry-run commit D) and link 2 (S2)
with every required binding filled, its reviewed document, rendering and strict log line. The verifier always runs
in the synthetic repository's own interpreter (``verify``), as a real verification runs inside the verified
repository. Expected contract values are read from ``tests/fixtures/ec_clean30_amendment2_contract.json``. No
campaign record, view or data folder is read. ``tests/test_ec_clean30_amendment_chain_attacks.py`` holds the
end-to-end refusals, the sequence-3 links of every class, receipts and pod partial clones on this harness.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from typing import Any, Callable, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ec_clean30 import amendment2, amendment_chain, freeze, selection  # noqa: E402
from ec_clean30 import amendment_rules as rules  # noqa: E402
from ec_clean30.amendment_chain import AmendmentChainError  # noqa: E402

CONTRACT = json.loads((ROOT / "tests" / "fixtures" / "ec_clean30_amendment2_contract.json").read_text())
DOCS = freeze.CAMPAIGN_DOCS_DIR
FREEZE1_PATH = f"{DOCS}/freeze1.json"
EXECUTION_FILES = ["src/ec_clean30/phase3.py", "src/ec_clean30/phase4.py", "src/ec_clean30/workstation.py"]
REFUSALS_FIXTURE = "tests/fixtures/ec_clean30_amendment2_refusals.json"
DECISIONS_FIXTURE = "tests/fixtures/ec_clean30_amendment2_decisions.json"
DECISIONS_PATH = f"{DOCS}/amendment2/maintainer_decisions.json"
RULE_TEXT = "## Rule text\n\n**A2-X-1** A synthetic rule.\n\n"
FOLDS = [str(fold) for fold in range(5)]


# ---------------------------------------------------------------------------
# The synthetic repository

_CONSTANTS = '''

def __getattr__(name):
    if name.isupper():
        return "synthetic " + name
    raise AttributeError(name)
'''

_STUBS = {
    "src/ec_clean30/__init__.py": "",
    "src/ec_clean30/campaign.py": textwrap.dedent('''\
        """Synthetic campaign module."""
        import hashlib
        import json

        CAMPAIGN_ID = "ec_class_vs_clean30_2026-10"
        N_OUTER_FOLDS = 5
        PROVISIONAL_LEVER_SETTINGS = {"L9": "synthetic"}
        S2_SUB_CEILINGS = {"S2": 0}
        L1_TARGET_SHRINK = 0.5
        L8_LAYER = 8


        def _sha(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


        def build_fold_plan(fold):
            return {"plan_sha256": _sha({"fold": fold, "plan": 1})}


        def build_fold_plan_v2(fold):
            return {"plan_sha256": _sha({"fold": fold, "plan": 2})}


        def campaign_constants():
            return {"campaign_id": CAMPAIGN_ID}


        def search_policy_record_v2():
            return {"rule": "synthetic search"}


        def controls_policy_record_v2():
            return {"rule": "synthetic controls"}
        '''),
    "src/ec_clean30/selection.py": textwrap.dedent('''\
        """Synthetic selection module."""
        import hashlib
        import json

        AMENDED_S1_RULE = __S1_RULE__
        LEVER_PROMOTION_ALPHA = {}
        LEVERS = ("L9",)
        S1_POLICY = json.loads(__S1_POLICY__)


        class PromotionConfig:
            def to_record(self):
                return {"config": "synthetic"}


        def promotion_config_for(lever):
            return PromotionConfig()


        def s1_policy_record():
            return json.loads(json.dumps(S1_POLICY))


        def selection_policy_record_v2():
            return {"rule": "synthetic selection"}


        def selection_policy_sha256_v2():
            return hashlib.sha256(json.dumps(selection_policy_record_v2(), sort_keys=True).encode()).hexdigest()
        ''') + _CONSTANTS,
    "src/ec_clean30/selector.py": '"""Synthetic selector module."""\nSTEP = "S1"\n',
    "src/ec_clean30/s1_simulation.py": '"""Synthetic S1 simulation module."""\nSEED = 1\n',
    "src/ec_clean30/stats.py": ('"""Synthetic statistics module."""\n\n\ndef engine_policy_record_v2():\n'
                                '    return {"rule": "synthetic engine"}\n' + _CONSTANTS),
    "src/ec_clean30/futility.py": ('"""Synthetic futility module."""\n\n\ndef policy_record():\n'
                                   '    return {"rule": "synthetic futility"}\n'),
    "src/ec_clean30/clean_sanity.py": ('"""Synthetic CLEAN sanity module."""\n\n\ndef clean_sanity_policy_record():\n'
                                       '    return {"rule": "synthetic sanity"}\n'),
    "src/ec_clean30/scoring.py": textwrap.dedent('''\
        """Synthetic scoring module (post-fit analysis)."""
        FROZEN_SETTINGS = {"setting": 1}
        MANIFEST_SETTINGS = ("setting",)
        CO_PRIMARY_POPULATIONS = {"B1-all": "all"}
        ABSTENTION_RULE = "synthetic"
        LABEL_COLUMNS = ("ec",)


        def _load_comparator():
            return "comparator"


        def parse_label_table(data):
            return [*LABEL_COLUMNS, *data.split()]


        def parse_transfer_list(data):
            return []


        def scoring_policy_record_v2():
            return {"rule": "synthetic accuracy"}


        def refuses(value):
            return value != "bound input"


        def decision_for(name):
            return "d"
        '''),
    "src/ec_clean30/baseline.py": ('"""Synthetic baseline module (post-fit analysis)."""\n\n\n'
                                   'def read_official_ec_numbers(path):\n    return []\n' + _CONSTANTS),
    "src/ec_clean30/identity.py": ('"""Synthetic identity module."""\n\n\ndef rules_record():\n'
                                   '    return {"identity": "synthetic"}\n' + _CONSTANTS),
    "src/ec_clean30/logit_adjustment.py": '"""Synthetic logit adjustment."""\nTAUS = (0.0, 0.5)\nPRIOR_SMOOTHING = 1.0\n',
    "src/ec_clean30/commands.py": ('"""Synthetic commands module."""\n\n\ndef _comparator_published_recipe():\n'
                                   '    return {"recipe": "synthetic"}\n'),
    "src/ec_clean30/views.py": '"""Synthetic views module."""\nVIEW = "synthetic"\n',
    "src/ec_clean30/phase3.py": ('"""Synthetic execution file with the pinned gate."""\nLIMIT = 3\n\n\n'
                                 'def _helper(value):\n    return value + LIMIT\n\n\n'
                                 'def lane_gate(value):\n    return _helper(value)\n'),
    "src/ec_clean30/phase4.py": ('"""Synthetic execution file with the pinned CPU replay comparison."""\n'
                                 'ATOL = 1e-5\n\n\ndef replay_cpu_logits(saved, replayed):\n'
                                 '    return abs(saved - replayed) <= ATOL\n'),
    "src/ec_clean30/workstation.py": '"""Synthetic execution file."""\nHOST = "synthetic"\n',
    "src/training/__init__.py": "",
    "src/training/run.py": '"""Synthetic trainer entry point."""\nEPOCHS = 30\n',
    "scripts/run_ec_clean30_baseline.py": '"""Synthetic baseline runner (post-fit analysis)."""\n',
    "scripts/score_ec_clean30_comparison.py": '"""Synthetic scorer runner (post-fit analysis)."""\n',
    "scripts/run_ec_clean30_campaign.py": '"""Synthetic campaign runner."""\n' + "".join(
        f"\n\ndef {name}():\n    return None\n" for name in freeze.PINNED_RUNNER_FUNCTIONS),
    ".gitignore": "__pycache__/\n",
}
_AMENDMENT2_STUBS = {
    "src/ec_clean30/runtime_precision.py": '"""Synthetic runtime-precision module."""\nPOLICY = "tf32 off"\n',
}
# The synthetic refusal tests and decision-fixture test that the bound fixtures name (class (a2) reruns them).
_SYNTHETIC_TESTS = {
    "tests/test_synthetic_refusals.py": textwrap.dedent('''\
        """Synthetic refusal tests (named by the bound refusal fixture)."""
        from pathlib import Path
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

        from ec_clean30 import scoring


        def test_r1():
            assert scoring.refuses("unbound input")


        def test_r2():
            assert not scoring.refuses("bound input")
        '''),
    "tests/test_ec_clean30_amendment2_bound_fixtures.py": textwrap.dedent('''\
        """Synthetic reproduction of the bound decision fixture."""
        import json
        from pathlib import Path
        import sys

        ROOT = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(ROOT / "src"))

        from ec_clean30 import scoring

        FIXTURE = ROOT / "tests" / "fixtures" / "ec_clean30_amendment2_decisions.json"


        def test_the_decision_fixture_is_reproduced():
            expected = {"format": "synthetic", "cases": [{"name": "c", "decision": scoring.decision_for("c")}]}
            assert json.loads(FIXTURE.read_text()) == expected
        '''),
}
SYNTHETIC_REFUSALS = {"format": "synthetic", "refusals": [
    {"id": "r1", "test": "test_synthetic_refusals.py::test_r1"},
    {"id": "r2", "test": "test_synthetic_refusals.py::test_r2"}]}
SYNTHETIC_DECISIONS = {"format": "synthetic", "cases": [{"name": "c", "decision": "d"}]}

_VALUES_PROBE = textwrap.dedent('''\
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    repo = Path(sys.argv[2])
    from ec_clean30 import campaign
    out = {"v1": {str(k): campaign.build_fold_plan(k)["plan_sha256"] for k in range(5)},
           "v2": {str(k): campaign.build_fold_plan_v2(k)["plan_sha256"] for k in range(5)}}
    if (repo / "src" / "ec_clean30" / "amendment_rules.py").is_file():
        from ec_clean30 import amendment_rules as r, selection
        out.update(rule_policy_sha256=r.rule_policy_sha256(), live_constants_sha256=r.live_constants_sha256(),
                   pinned_execution=r.pinned_function_hashes(repo, r.PINNED_EXECUTION_FUNCTIONS),
                   pinned_analysis=r.pinned_function_hashes(repo, r.PINNED_ANALYSIS_FUNCTIONS),
                   selection_policy_sha256_v2=r.sha256_canonical(selection.selection_policy_record_v2()))
    print(json.dumps(out))
    ''')

# Runs the synthetic repository's own verifier (``verify_chain``, ``chain_receipt_record``,
# ``check_chain_receipt``) or a snippet in its interpreter and prints its result or its refusal.
_WORLD_PROBE = textwrap.dedent('''\
    import json, sys
    sys.path.insert(0, sys.argv[1])
    args = json.loads(sys.argv[2])
    from ec_clean30 import amendment_chain
    from ec_clean30 import amendment_rules as rules
    try:
        if args["action"] == "verify":
            result = amendment_chain.verify_chain(args["tip"], expected_freeze1_sha256=args["freeze1"],
                                                  expected_freeze1b_sha256=rules.FREEZE1B_SHA256, repo=args["repo"],
                                                  **args["options"])
        elif args["action"] == "receipt":
            result = amendment_chain.chain_receipt_record(args["tip"], expected_freeze1_sha256=args["freeze1"],
                                                          expected_freeze1b_sha256=rules.FREEZE1B_SHA256,
                                                          repo=args["repo"], **args["options"])
        elif args["action"] == "check":
            result = amendment_chain.check_chain_receipt(args["receipt"], repo=args["repo"], **args["options"])
        else:
            namespace = {"amendment_chain": amendment_chain, "rules": rules, "args": args}
            exec(args["code"], namespace)
            result = namespace["result"]
    except amendment_chain.AmendmentChainError as exc:
        print(json.dumps({"refused": str(exc)}))
    else:
        print(json.dumps({"result": result}, sort_keys=True, default=str))
    ''')


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid", "-c",
         "commit.gpgsign=false", "-c", f"core.hooksPath={os.devnull}", *args],
        cwd=str(repo), capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def _commit(repo: Path, message: str, *, push: bool = True) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "-m", message)
    if push:
        _git(repo, "push", "--quiet", "origin", "main")
    return _git(repo, "rev-parse", "HEAD")


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(repo: Path, relative: str) -> str:
    return _sha((repo / relative).read_bytes())


def _binding(repo: Path, relative: str) -> dict:
    return {"path": relative, "sha256": _file_sha(repo, relative)}


def _tree_pins(repo: Path) -> dict[str, str]:
    return {path: _file_sha(repo, path) for path in rules.tracked_source_files(repo)}


def _values(repo: Path) -> dict:
    out = subprocess.run([sys.executable, "-I", "-B", "-c", _VALUES_PROBE, str(repo / "src"), str(repo)],
                         capture_output=True, text=True, check=True, cwd=str(repo))
    return json.loads(out.stdout.strip().splitlines()[-1])


def _log(repo: Path, line: str) -> None:
    path = repo / freeze.LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def log_link(repo: Path, sequence: int, digest: str) -> None:
    """Enter a link's strict log line (``amendment_rules.link_log_line``) in the campaign log."""
    _log(repo, f"- {rules.link_log_line(sequence, digest)}")


def place(record: dict, role: str, value: Any) -> None:
    keys = role.split(".")
    cursor = record
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def write_role_files(repo: Path, folder: str = f"{DOCS}/amendment2") -> dict[str, Any]:
    """Synthetic files for every role of ``amendment_rules.REQUIRED_BINDINGS``, as ``role -> binding``."""
    def put(name: str, text: str) -> dict:
        _write(repo, f"{folder}/{name}", text)
        return _binding(repo, f"{folder}/{name}")

    roles = {
        "data_state.attestation_at_build": put("data_state_attestation.json",
                                               '{"fold0_records": [], "g2_completed_units": 0}\n'),
        "dry_runs.v1_resolved_config": put("v1_resolved_config.json", '{"digests": "synthetic"}\n'),
        "evidence.futility_oc": put("futility_oc.json", '{"oc": "synthetic"}\n'),
        "evidence.interface_contract": put("interface_contract.md", "# Synthetic contract\n"),
        "evidence.power_simulation": put("power_simulation.md", "# Synthetic power simulation\n"),
        "evidence.verification": put("verification.json", '{"status": "passed"}\n'),
        "inputs.clean_tie_exposure": put("clean_tie_exposure.json", '{"ties": 0}\n'),
        "inputs.f6_corpus": {"tables": {fold: put(f"f6_corpus/fold{fold}.tsv", f"corpus {fold}\n") for fold in FOLDS},
                             "union_fasta": put("f6_corpus/union.fasta", ">p\nMKV\n")},
        "inputs.f6_holdout": {fold: put(f"f6_holdout/fold{fold}.tsv", f"holdout {fold}\n") for fold in FOLDS},
    }
    assert set(roles) == set(rules.REQUIRED_BINDINGS)
    return roles


def write_rule_text(repo: Path, sequence: int, text: str = RULE_TEXT) -> dict:
    """The reviewed document ``amendment<n>/amendment_draft.md`` and the record's ``rule_text``."""
    source = rules.rule_text_source(sequence)
    _write(repo, source, f"# Amendment {sequence}\n\n{text}## Policy deviations\n\n1. none\n")
    return {"source": source, "sha256": _sha(text.encode()), "text": text}


def _link2_record(repo: Path, *, values: dict, freeze1_sha: str, freeze1_doc: dict, a1: str, link1_commit: str,
                  dry_run_commit: str, previous_pins: dict) -> dict:
    """The link-2 record as the record builder writes it, from the synthetic repository's own values, with every
    required binding, the decisions file, the fixtures and the reviewed rule-text document."""
    receipts = {}
    for fold in FOLDS:
        relative = f"{DOCS}/amendment2/dry_runs/fold{fold}.json"
        _write(repo, relative, json.dumps({
            "format": amendment2.DRY_RUN_V2_RECEIPT_FORMAT, "schema_version": 1, "status": "passed",
            "fold": int(fold), "plan_version": 2,
            "provenance": {"smoke": False, "git": {"head": dry_run_commit, "clean": True}}}, sort_keys=True))
        receipts[fold] = _binding(repo, relative)
    answers = {key: {"choice": "the recommended default", "accepted_default": True}
               for key in amendment2.CONFIRMATION_IDS}
    _write(repo, DECISIONS_PATH, json.dumps({"format": amendment2.DECISIONS_FORMAT, "maintainer": "synthetic",
                                             "answers": answers}, indent=2, sort_keys=True))
    _write(repo, REFUSALS_FIXTURE, json.dumps(SYNTHETIC_REFUSALS))
    _write(repo, DECISIONS_FIXTURE, json.dumps(SYNTHETIC_DECISIONS))
    roles = write_role_files(repo)
    rule_text = write_rule_text(repo, rules.AMENDMENT2_SEQUENCE)
    tree = _tree_pins(repo)
    changes = {path: {"before": previous_pins.get(path), "after": tree.get(path)}
               for path in sorted(set(previous_pins) | set(tree))
               if path not in EXECUTION_FILES and previous_pins.get(path) != tree.get(path)}
    record = {
        "format": rules.AMENDMENT2_FORMAT, "schema_version": 1, "campaign_id": rules.CAMPAIGN_ID,
        "amendment_id": rules.AMENDMENT2_ID, "sequence": rules.AMENDMENT2_SEQUENCE,
        "change_class": rules.CHANGE_PRE_DATA,
        "base_freeze1": {"path": FREEZE1_PATH, "sha256": freeze1_sha},
        "base_freeze1b_sha256": rules.FREEZE1B_SHA256,
        "base_amendment": {"sequence": 1, "amendment_id": freeze.S1_AMENDMENT_ID, "path": rules.AMENDMENT1["path"],
                           "sha256": a1, "commit": link1_commit},
        "previous_sha256": a1, "chain_before": [a1], "timestamps": {"draft_written": "2026-11-04"},
        "data_state": {**amendment_chain.LINK2_DATA_STATE, "attested_by_log": "synthetic"}, "decisions": {},
        "rules": {**{rule_id: {"policy_function": entry["function"], "analysis": entry["analysis"]}
                     for rule_id, entry in rules.RULE_POLICY_FUNCTIONS.items()},
                  rules.R_CMD_V2_RULE: {"policy_function": amendment_chain.R_CMD_V2_POLICY_FUNCTION,
                                        "analysis": False}},
        "supersedes": [], "plans": {"v1_plan_sha256": values["v1"], "v2_plan_sha256": values["v2"]},
        "dry_runs": {"rule": rules.R_CMD_V2_RULE, "source_commit": dry_run_commit, "receipts": receipts,
                     "in_view_receipts_unchanged": {fold: freeze1_doc["views"][fold]["dry_run_receipt"]["sha256"]
                                                    for fold in FOLDS}},
        "source_parent_commit": link1_commit,
        "execution_file_set": {"files": EXECUTION_FILES, "names_sha256": rules.sha256_canonical(EXECUTION_FILES)},
        "pinned_execution_functions": values["pinned_execution"],
        "post_fit_analysis_files": list(rules.POST_FIT_ANALYSIS_FILES),
        "pinned_analysis_functions": values["pinned_analysis"], "live_constants_sha256": values["live_constants_sha256"],
        "source_changes": changes,
        "unchanged_scientific_files": ["src/ec_clean30/freeze.py", "src/ec_clean30/provenance.py"],
        "inputs": {}, "deferred_bindings": {"f6_inputs": {"bound_by": "freeze1c_v2", "spec": "A2-F6-4"}},
        "fixtures": {"refusals": _binding(repo, REFUSALS_FIXTURE), "decisions": _binding(repo, DECISIONS_FIXTURE)},
        "evidence": {}, "deviations": [], "publication": amendment_chain.PUBLICATION,
        "status": "built", "confirmation_status": amendment_chain.CONFIRMED, "provisional_markers_settled": True,
        "decisions_file": {"sha256": _file_sha(repo, DECISIONS_PATH), "maintainer": "synthetic", "answers": answers},
        "rule_text": rule_text,
        "bound_files": {"decisions": _binding(repo, DECISIONS_PATH),
                        **{f"dry_run_v2_fold{fold}": item for fold, item in receipts.items()}},
        "rule_policy_sha256": values["rule_policy_sha256"],
        "selection_policy_sha256_v2": values["selection_policy_sha256_v2"],
    }
    for role, binding in roles.items():
        place(record, role, binding)
    return record


def _execution_history(repo: Path, *, add_files: bool) -> dict[str, str]:
    """Execution-only commits between link 1 and amendment 2: E1 (adds or changes the execution files), E2 on a side
    branch, E3 on main and the merge M, all pushed."""
    if add_files:
        for relative in EXECUTION_FILES:
            _write(repo, relative, _STUBS[relative])
    else:
        with (repo / "src/ec_clean30/phase3.py").open("a", encoding="utf-8") as handle:
            handle.write("\nRETRIES = 1\n")
    commits = {"E1": _commit(repo, "Execution change E1")}
    _git(repo, "checkout", "--quiet", "-b", "side")
    with (repo / "src/ec_clean30/workstation.py").open("a", encoding="utf-8") as handle:
        handle.write("PORT = 22\n")
    commits["E2"] = _commit(repo, "Execution change E2 on a side branch", push=False)
    _git(repo, "checkout", "--quiet", "main")
    with (repo / "src/ec_clean30/phase3.py").open("a", encoding="utf-8") as handle:
        handle.write("TIMEOUT = 60\n")
    commits["E3"] = _commit(repo, "Execution change E3", push=False)
    _git(repo, "merge", "--quiet", "--no-ff", "side", "-m", "Merge the side branch")
    _git(repo, "branch", "--quiet", "-D", "side")
    _git(repo, "push", "--quiet", "origin", "main")
    commits["M"] = _git(repo, "rev-parse", "HEAD")
    return commits


def build_source_world(root: Path, *, execution_after_link1: bool = False, drift_at_link1: bool = False,
                       log_link1: bool = True) -> dict:
    """Freeze 1 (commit A) and link 1 (commit B), each pushed to a bare origin, the execution-only history E1, E2, E3
    and the merge M, then the amendment-2 source (the dry-run commit D, not pushed). With ``execution_after_link1``
    the execution files are first added at E1, as in the campaign repository, so Freeze 1 and link 1 pin no
    execution file. With ``drift_at_link1`` the link-1 commit also holds a scientific file that differs from its
    Freeze-1 pin, restored in the next commit (only the authentication of link 1 at its own commit can see it).
    Without ``log_link1`` the campaign log never holds link 1's SHA-256."""
    repo, origin = root / "repo", root / "origin.git"
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "init", "--quiet", "-b", "main", str(repo)], check=True)
    _git(repo, "remote", "add", "origin", str(origin))
    s1_policy = selection.s1_policy_record()
    stubs = {relative: text for relative, text in _STUBS.items()
             if not (execution_after_link1 and relative in EXECUTION_FILES)}
    for relative, text in stubs.items():
        text = text.replace("__S1_RULE__", repr(selection.AMENDED_S1_RULE)).replace(
            "__S1_POLICY__", repr(json.dumps(s1_policy, sort_keys=True)))
        _write(repo, relative, text)
    for name in ("freeze.py", "provenance.py"):
        shutil.copyfile(ROOT / "src" / "ec_clean30" / name, repo / "src" / "ec_clean30" / name)
    _git(repo, "add", "--all")
    values = _values(repo)
    freeze1_doc = {"format": freeze.FREEZE1_DOCUMENT_FORMAT, "schema_version": 1, "campaign_id": rules.CAMPAIGN_ID,
                   "amendments": [], "code": {"tracked_file_sha256": _tree_pins(repo)},
                   "plans": {fold: {"plan_sha256": values["v1"][fold]} for fold in FOLDS},
                   "views": {fold: {"dry_run_receipt": {"sha256": _sha(f"in-view receipt {fold}".encode())}}
                             for fold in FOLDS}}
    (repo / FREEZE1_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / FREEZE1_PATH).write_bytes(freeze.canonical_bytes(freeze1_doc))
    freeze1_sha = _file_sha(repo, FREEZE1_PATH)
    _log(repo, f"- Freeze 1 `{freeze1_sha}`")
    commit_a = _commit(repo, "Freeze 1")
    # Link 1 by the real S1-amendment builder.
    with (repo / "src/ec_clean30/selection.py").open("a", encoding="utf-8") as handle:
        handle.write("\nS1_AMENDED = True\n")
    sources = {name: _file_sha(repo, name) for name in freeze.S1_AMENDMENT_SOURCE_FILES}
    s1 = f"{DOCS}/s1_amendment"
    spec = {"policy": s1_policy, "scenarios": [{"id": "synthetic"}], "base_freeze1_sha256": freeze1_sha,
            "n_replicates": 200}
    _write(repo, f"{s1}/simulation_spec.json", json.dumps(spec, sort_keys=True))
    summary = {"format": "deepmzyme_s1_characterization_summary_v1", "status": "complete", "policy": s1_policy,
               "base_freeze1_sha256": freeze1_sha, "spec_sha256": _file_sha(repo, f"{s1}/simulation_spec.json"),
               "source_sha256": {name: sources[name] for name in ("src/ec_clean30/selection.py",
                                                                  "src/ec_clean30/s1_simulation.py")},
               "results": [{"id": "synthetic", "n_replicates": 200, "record_sha256": "0" * 64,
                            "winner_counts": {"F2": 200, "F3": 0, "F5": 0, "F4": 0}}],
               "n_scenarios": 1, "n_replicates": 200}
    _write(repo, f"{s1}/simulation_summary.json", json.dumps(summary, sort_keys=True))
    _write(repo, f"{s1}/verification.json", json.dumps({
        "status": "passed", "source_sha256": sources,
        "checks": {name: {"status": "passed"} for name in freeze.S1_AMENDMENT_CHECKS}}, sort_keys=True))
    freeze.build_s1_amendment(freeze1_path=repo / FREEZE1_PATH, out=repo / rules.AMENDMENT1["path"], repo=repo,
                              evidence={role: repo / f"{s1}/{role}.json" for role in freeze.S1_AMENDMENT_EVIDENCE})
    a1 = _file_sha(repo, rules.AMENDMENT1["path"])
    if log_link1:
        _log(repo, f"- S1 amendment `{a1}`")
    views = repo / "src/ec_clean30/views.py"
    original_views = views.read_text(encoding="utf-8")
    if drift_at_link1:
        views.write_text(original_views + "DRIFT = 1\n", encoding="utf-8")
    commit_b = _commit(repo, "Link 1")
    views.write_text(original_views, encoding="utf-8")
    previous_pins = _tree_pins(repo)
    history = _execution_history(repo, add_files=execution_after_link1)
    # The amendment-2 source: shared rules with this repository's chain constants, the verifier, the builder and the
    # synthetic refusal and decision-fixture tests.
    text = (ROOT / "src/ec_clean30/amendment_rules.py").read_text(encoding="utf-8")
    for old, new in ((rules.FREEZE1_SHA256, freeze1_sha), (rules.AMENDMENT1["sha256"], a1),
                     (rules.AMENDMENT1["commit"], commit_b)):
        assert text.count(old) == 1
        text = text.replace(old, new)
    _write(repo, "src/ec_clean30/amendment_rules.py", text)
    for name in ("amendment_chain.py", "amendment2.py"):
        shutil.copyfile(ROOT / "src" / "ec_clean30" / name, repo / "src" / "ec_clean30" / name)
    for relative, text in {**_AMENDMENT2_STUBS, **_SYNTHETIC_TESTS}.items():
        _write(repo, relative, text)
    with (repo / "src/ec_clean30/phase3.py").open("a", encoding="utf-8") as handle:
        handle.write("\nNOTE = 'execution change'\n")
    commit_d = _commit(repo, "Amendment-2 source", push=False)
    return {"root": root, "repo": repo, "origin": origin, "freeze1_sha256": freeze1_sha, "freeze1_doc": freeze1_doc,
            "a1": a1, "values": _values(repo), "previous_pins": previous_pins,
            "commits": {"A": commit_a, "B": commit_b, **history, "D": commit_d}}


def build_world(root: Path, *, bad_rendering: bool = False, drift_at_link1: bool = False, log_link1: bool = True,
                before_record: Optional[Callable[[Path], None]] = None,
                mutate_record: Optional[Callable[[dict, Path], None]] = None,
                before_link2_commit: Optional[Callable[[Path, dict], None]] = None) -> dict:
    """The source world and link 2 (commit S2), published with its strict log line. Adversarial worlds:
    ``before_record(repo)`` changes the tree after the dry-run commit and before the record is assembled (so the
    record's scope holds the change), ``mutate_record(record, repo)`` edits the assembled record before it is
    written, and ``before_link2_commit(repo, record)`` changes the tree just before the link-2 commit."""
    source = build_source_world(root, drift_at_link1=drift_at_link1, log_link1=log_link1)
    repo = source["repo"]
    if before_record is not None:
        before_record(repo)
    record = _link2_record(repo, values=source["values"], freeze1_sha=source["freeze1_sha256"],
                           freeze1_doc=source["freeze1_doc"], a1=source["a1"], link1_commit=source["commits"]["B"],
                           dry_run_commit=source["commits"]["D"], previous_pins=source["previous_pins"])
    if mutate_record is not None:
        mutate_record(record, repo)
    (repo / rules.AMENDMENT2_PATH).write_bytes(freeze.canonical_bytes(record))
    rendering = amendment2.render_amendment2(record) + ("drift\n" if bad_rendering else "")
    (repo / rules.rendering_path(rules.AMENDMENT2_PATH)).write_text(rendering, encoding="utf-8")
    a2 = _file_sha(repo, rules.AMENDMENT2_PATH)
    log_link(repo, 2, a2)
    if before_link2_commit is not None:
        before_link2_commit(repo, record)
    commit_s2 = _commit(repo, "Link 2")
    return {**source, "a2": a2, "record": record, "commits": {**source["commits"], "S2": commit_s2}}


def copy_world(world: dict, factory) -> dict:
    root = factory.mktemp("chain_copy") / "world"
    shutil.copytree(world["root"], root, symlinks=True)
    copied = {**world, "root": root, "repo": root / "repo", "origin": root / "origin.git"}
    _git(copied["repo"], "remote", "set-url", "origin", str(copied["origin"]))
    return copied


def commit_change(world: dict, relative: str, text: str, *, push: bool = True) -> str:
    _write(world["repo"], relative, text)
    return _commit(world["repo"], f"Change {relative}", push=push)


def run_world(world: dict, action: str, *, repo: Optional[Path] = None, env: Optional[dict] = None,
              **args: Any) -> Any:
    """``action`` in the synthetic repository's own interpreter (its ``src`` first, as a real verification runs
    inside the verified repository); a refusal raises ``AmendmentChainError``."""
    repo = Path(repo or world["repo"])
    payload = {"action": action, "repo": str(repo), "freeze1": world["freeze1_sha256"], "options": {}, **args}
    child_env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    child_env.update(env or {})
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", _WORLD_PROBE, str(repo / "src"),
                                json.dumps(payload, default=str)], capture_output=True, text=True, cwd=str(repo),
                               env=child_env, timeout=1800)
    lines = completed.stdout.strip().splitlines()
    assert completed.returncode == 0 and lines, completed.stderr[-3000:]
    output = json.loads(lines[-1])
    if "refused" in output:
        raise AmendmentChainError(output["refused"])
    return output["result"]


def verify(world: dict, *, tip: str = rules.AMENDMENT2_PATH, repo: Optional[Path] = None,
           env: Optional[dict] = None, **options) -> dict:
    root = Path(repo or world["repo"])
    return run_world(world, "verify", repo=root, env=env, tip=str(root / tip), options=options)


def write_receipt(world: dict, out: Path, *, tip: str = rules.AMENDMENT2_PATH, **options) -> dict:
    receipt = run_world(world, "receipt", tip=str(world["repo"] / tip), options=options)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def check_receipt(world: dict, receipt: Path, *, repo: Optional[Path] = None, env: Optional[dict] = None,
                  **options) -> dict:
    return run_world(world, "check", repo=repo, env=env, receipt=str(receipt), options=options)


def partial_clone(world: dict, name: str, *, offline: bool = False) -> Path:
    """A pod-style blobless partial clone of ``origin`` (``git clone --filter=blob:none`` through a ``file://``
    promisor, as the workstation stages pods), detached at the repository's HEAD. ``offline`` points the promisor
    at a missing path afterwards (a pod whose network or credentials are gone)."""
    origin = world["origin"]
    for key in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):
        subprocess.run(["git", "-C", str(origin), "config", key, "true"], check=True)
    pod = world["root"] / name
    head = _git(world["repo"], "rev-parse", "HEAD")
    subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", f"file://{origin}", str(pod)], check=True)
    subprocess.run(["git", "-C", str(pod), "checkout", "--quiet", "--detach", head], check=True)
    if offline:
        subprocess.run(["git", "-C", str(pod), "remote", "set-url", "origin", f"file://{world['root']}/gone.git"],
                       check=True)
    return pod


def link_record(world: dict, *, sequence: int, change_class: str, files: Optional[dict] = None,
                fold0: Optional[list] = None, extra: Optional[dict] = None, base: Optional[dict] = None,
                write_fut: bool = False) -> dict:
    """Write the files of a link of sequence 3 or higher (changed sources, reviewed document, optional fold-0 FUT
    files) and return its record, built from the previous link's record as the builder would build it."""
    repo = world["repo"]
    base = base or {"sequence": 2, "record": world["record"], "sha256": world["a2"],
                    "commit": world["commits"]["S2"], "chain": [world["a1"], world["a2"]]}
    before = verify(world, tip=rules.link_path(base["sequence"]), require_published=False)
    previous = {**before["composite_scientific_pins"], **before["analysis_pins"]}
    for path, text in (files or {}).items():
        _write(repo, path, text)
    fold0 = list(fold0 or [])
    if write_fut:
        fold0.append(write_fut_files(world, sequence, fit_chain=before["fit_chain"]))
    rule_text = write_rule_text(repo, sequence, f"## Rule text\n\n**A{sequence}-X-1** Link {sequence}.\n\n")
    _git(repo, "add", "--all")
    values = _values(repo)
    tree = _tree_pins(repo)
    execution = set(base["record"]["execution_file_set"]["files"])
    changes = {path: {"before": previous.get(path), "after": tree.get(path)}
               for path in sorted(set(previous) | set(tree))
               if path not in execution and previous.get(path) != tree.get(path)}
    record = json.loads(json.dumps(base["record"]))
    record.update({
        "sequence": sequence, "amendment_id": f"synthetic-link-{sequence}-{change_class}",
        "change_class": change_class,
        "base_amendment": {"sequence": base["sequence"], "amendment_id": base["record"]["amendment_id"],
                           "path": rules.link_path(base["sequence"]), "sha256": base["sha256"],
                           "commit": base["commit"]},
        "previous_sha256": base["sha256"], "chain_before": list(base["chain"]),
        "timestamps": {"written": f"link {sequence}"}, "fold0_records": fold0, "source_changes": changes,
        "source_parent_commit": base["commit"], "rule_text": rule_text,
        "pinned_analysis_functions": values["pinned_analysis"],
        "pinned_execution_functions": values["pinned_execution"],
        "live_constants_sha256": values["live_constants_sha256"],
        "rule_policy_sha256": values["rule_policy_sha256"],
    })
    record.update(extra or {})
    return record


def write_fut_files(world: dict, sequence: int, *, fit_chain: list, decision: str = "continue",
                    pointer_sequence: int = 2, folder: Optional[str] = None) -> dict:
    """The fold-0 futility record and its binding pointer ``fut_seq<k>.json`` as repository files of the link, and
    the ``fold0_records`` FUT entry that binds them."""
    repo = world["repo"]
    folder = folder or f"{DOCS}/amendment{sequence}/fold0"
    record_path = f"{folder}/futility_record.json"
    _write(repo, record_path, json.dumps({"format": rules.FUTILITY_RECORD_FORMAT, "freeze1_amendment_chain": fit_chain,
                                          "decision": {"decision": decision, "binding": True}}, sort_keys=True))
    record_sha = _file_sha(repo, record_path)
    pointer_path = f"{folder}/fut_seq{pointer_sequence}.json"
    _write(repo, pointer_path, json.dumps({
        "format": rules.FUTILITY_POINTER_FORMAT, "sequence": pointer_sequence, "decision": decision, "binding": True,
        "futility_record_sha256": record_sha, "fit_chain": fit_chain}, sort_keys=True))
    return {"kind": rules.FUT_KIND, "sha256": record_sha, "decision": decision,
            "pointer": _binding(repo, pointer_path), "record": _binding(repo, record_path)}


def publish_link(world: dict, record: dict, *, message: Optional[str] = None, push: bool = True,
                 log: bool = True) -> tuple[str, str, str]:
    """Write a link at its fixed path with the builder's rendering, enter its strict log line and commit all three
    together; returns (path, SHA-256, commit)."""
    repo = world["repo"]
    path = rules.link_path(record["sequence"])
    _write(repo, path, freeze.canonical_bytes(record).decode())
    _write(repo, rules.rendering_path(path), amendment2.render_amendment2(record))
    digest = _file_sha(repo, path)
    if log:
        log_link(repo, record["sequence"], digest)
    commit = _commit(repo, message or f"Link {record['sequence']}", push=push)
    return path, digest, commit


def defect_fields() -> dict:
    return {"defect": "the published rule is not implemented", "failing_test": "tests/test_synthetic_refusals.py",
            "found_through_record_sha256": "0" * 64, "inner_score_viewed_before_identification": False,
            "reruns": ["S1"]}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("chain_world") / "world")


@pytest.fixture(scope="module")
def verified(world):
    return verify(world)


@pytest.fixture(scope="module")
def world_after_execution_change(world, tmp_path_factory):
    """The published chain with a later, published change of an execution file only."""
    copied = copy_world(world, tmp_path_factory)
    commit_change(copied, "src/ec_clean30/workstation.py", '"""Synthetic execution file."""\nHOST = "other"\n')
    return copied


# ---------------------------------------------------------------------------
# Contract names


def test_contract_names_are_the_fixture_names():
    verifier = CONTRACT["verifier"]
    module, name = verifier["function"].split(":")
    assert module == amendment_chain.__name__ and getattr(amendment_chain, name) is amendment_chain.verify_chain
    assert list(amendment_chain.RESULT_KEYS) == verifier["returns"]
    parameters = inspect.signature(amendment_chain.verify_chain).parameters
    assert [key for key, item in parameters.items() if item.kind is item.KEYWORD_ONLY] == verifier["keyword_arguments"]
    assert list(amendment_chain.VERIFY_KEYWORDS) == verifier["keyword_arguments"]
    receipt = verifier["receipt_check"]
    module, name = receipt["function"].split(":")
    assert getattr(amendment_chain, name) is amendment_chain.check_chain_receipt
    parameters = inspect.signature(amendment_chain.check_chain_receipt).parameters
    assert [key for key, item in parameters.items() if item.kind is item.KEYWORD_ONLY] == receipt["keyword_arguments"]
    assert list(amendment_chain.RECEIPT_CHECK_KEYWORDS) == receipt["keyword_arguments"]
    assert receipt["format"] == amendment_chain.CHAIN_RECEIPT_FORMAT == amendment2.CHAIN_RECEIPT_FORMAT
    assert receipt["schema_version"] == amendment_chain.CHAIN_RECEIPT_SCHEMA_VERSION
    assert receipt["code_files"] == list(amendment_chain.RECEIPT_CODE_FILES)
    record = CONTRACT["amendment_record"]
    assert list(amendment_chain.LINK_REQUIRED_KEYS) == record["required_keys"]
    assert record["path"] == rules.AMENDMENT2_PATH == rules.link_path(2)
    assert record["link_path"] == rules.link_path(7).replace("7", "<n>")
    assert record["log_line"] == rules.link_log_line(2, "0" * 64).replace("2", "<n>", 1).replace("0" * 64, "<sha256>")
    assert record["header"] == {"amendment_id": rules.AMENDMENT2_ID, "change_class": rules.CHANGE_PRE_DATA,
                                "format": rules.AMENDMENT2_FORMAT, "schema_version": 1,
                                "sequence": rules.AMENDMENT2_SEQUENCE}
    assert record["base_amendment"] == {key: rules.AMENDMENT1[key] for key in record["base_amendment"]}
    assert amendment_chain.DRY_RUN_V2_RECEIPT_FORMAT == amendment2.DRY_RUN_V2_RECEIPT_FORMAT
    module, name = amendment_chain.RECORD_RENDERER.split(":")
    assert module == amendment2.__name__ and getattr(amendment2, name) is amendment2.render_amendment2


def test_the_futility_formats_and_the_decisions_file_are_named_once():
    from ec_clean30 import futility

    assert rules.FUTILITY_POINTER_FORMAT == futility.FUTILITY_POINTER_FORMAT
    assert rules.FUTILITY_RECORD_FORMAT == futility.FUTILITY_RECORD_FORMAT
    assert list(rules.FUTILITY_DECISIONS) == [selection.FUTILITY_CONTINUE, selection.FUTILITY_STOP]
    assert futility.POINTER_NAME.format(sequence=2) == "fut_seq2.json"
    assert amendment2.DECISIONS_FORMAT == rules.DECISIONS_FORMAT
    assert amendment2.CONFIRMATION_IDS == rules.CONFIRMATION_IDS


def test_the_verifier_is_torch_free_and_imports_only_its_three_campaign_modules():
    tree = ast.parse((ROOT / "src" / "ec_clean30" / "amendment_chain.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ec_clean30"):
            imported |= {node.module.split(".")[1]} if "." in node.module else {alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[1] for alias in node.names if alias.name.startswith("ec_clean30.")}
    assert imported == {"amendment_rules", "freeze", "provenance"}
    probe = ("import sys\nsys.modules['torch'] = None\nsys.path.insert(0, sys.argv[1])\n"
             "import ec_clean30.amendment_chain\nprint('torch' in sys.modules and sys.modules['torch'] is not None)")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", probe, str(ROOT / "src")], capture_output=True,
                         text=True, check=True, cwd=str(ROOT))
    assert out.stdout.strip() == "False"
    assert CONTRACT["verifier"]["torch_free"] is True


def test_the_verifier_calls_the_shared_rules_and_defines_none_of_them():
    tree = ast.parse((ROOT / "src" / "ec_clean30" / "amendment_chain.py").read_text(encoding="utf-8"))
    referenced = {node.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "rules"}
    defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    helper = inspect.getsource(rules.pinned_function_hashes)
    for spec in CONTRACT["science_api"]["rules"]:
        name = spec.split(":")[1]
        assert name not in defined
        assert name in referenced or name in helper, name
    assert "verify_s1_amendment" not in {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def test_the_real_scientific_files_reach_analysis_and_execution_code_only_through_the_channels():
    tracked = rules.tracked_source_files(ROOT)
    draft = json.loads((ROOT / DOCS / "amendment2" / "amendment_draft.json").read_text(encoding="utf-8"))
    execution = [path for path in draft["execution_file_set"]["declared"] if path in tracked]
    part = rules.partition(tracked, execution)
    scan = amendment_chain.source_scan(ROOT, tracked)
    assert amendment_chain.analysis_channel_problems(ROOT, part["scientific_files"], scan=scan) == []
    assert amendment_chain.execution_closure_problems(ROOT, tracked, execution, scan=scan) == []


# ---------------------------------------------------------------------------
# The two-link chain on a synthetic published repository


def test_the_synthetic_link_2_record_has_the_contract_record_shape(world):
    import ec_clean30_amendment2_build as build

    assert build.record_shape_problems(world["record"]) == []
    assert rules.required_binding_problems(world["record"]) == []


def test_the_synthetic_history_has_execution_commits_and_a_merge_between_link_1_and_amendment_2(world):
    repo = world["repo"]
    merge = world["commits"]["M"]
    assert len(_git(repo, "rev-list", "--parents", "-n", "1", merge).split()) == 3
    between = _git(repo, "rev-list", f"{world['commits']['B']}..{world['commits']['D']}").split()
    assert len(between) == 5 and merge in between


def test_link_1_and_link_2_verify_at_their_own_commits(world, verified):
    assert set(verified) == set(CONTRACT["verifier"]["returns"])
    assert verified["chain"] == [world["a1"], world["a2"]] == verified["fit_chain"]
    assert verified["tip_sequence"] == 2 and verified["published"] is True
    assert [link["source_commit"] for link in verified["links"]] == [world["commits"]["B"], world["commits"]["S2"]]
    assert all(link["published"] and link["scientific"] for link in verified["links"])
    assert verified["policies"] == {rule_id: {"rule_id": rule_id, "policy": {"rule": text}} for rule_id, text in (
        ("amendment2_selection_v1", "synthetic selection"), ("futility_predictive_cluster_bootstrap_v1",
                                                             "synthetic futility"),
        ("l9_only_no_s3_v1", "synthetic search"), ("controls_v2", "synthetic controls"),
        ("stratified_rao_wu_v2", "synthetic engine"), ("clean_sanity_v1", "synthetic sanity"))}
    assert verified["plan_records"] == {"v1": world["record"]["plans"]["v1_plan_sha256"],
                                        "v2": world["record"]["plans"]["v2_plan_sha256"]}
    assert verified["dry_run_evidence"]["source_commit"] == world["commits"]["D"]
    assert len(verified["scientific_identity_sha256"]) == 64


def test_the_partition_is_closed_disjoint_and_equal_to_the_live_pins(world, verified):
    repo = world["repo"]
    execution = verified["execution_file_set"]
    scientific, analysis = verified["composite_scientific_pins"], verified["analysis_pins"]
    assert execution == sorted(set(execution)) == EXECUTION_FILES
    assert not (set(execution) & set(scientific) or set(execution) & set(analysis) or set(scientific) & set(analysis))
    assert set(execution) | set(scientific) | set(analysis) == set(rules.tracked_source_files(repo))
    assert set(analysis) == set(rules.POST_FIT_ANALYSIS_FILES) == set(verified["post_fit_analysis_files"])
    assert {**scientific, **analysis} == {path: value for path, value in _tree_pins(repo).items()
                                          if path not in execution}
    assert verified["pinned_execution_functions"] == rules.pinned_function_hashes(repo, rules.PINNED_EXECUTION_FUNCTIONS)
    assert verified["pinned_analysis_functions"] == rules.pinned_function_hashes(repo, rules.PINNED_ANALYSIS_FUNCTIONS)


def test_link_2_is_authenticated_in_a_checkout_of_its_commit_after_an_execution_change(
        world, verified, world_after_execution_change):
    copied = world_after_execution_change
    assert _git(copied["repo"], "rev-parse", "HEAD") != world["commits"]["S2"]
    result = verify(copied)
    assert result["chain"] == verified["chain"] and result["fit_chain"] == verified["fit_chain"]
    # An execution-only change keeps the scientific identity and the composite pins.
    assert result["scientific_identity_sha256"] == verified["scientific_identity_sha256"]
    assert result["composite_scientific_pins"] == verified["composite_scientific_pins"]


def test_a_scientific_change_without_a_link_is_refused(world, tmp_path_factory):
    copied = copy_world(world, tmp_path_factory)
    path = copied["repo"] / "src/ec_clean30/selector.py"
    commit_change(copied, "src/ec_clean30/selector.py", path.read_text() + "CHANGED = True\n")
    with pytest.raises(AmendmentChainError, match=r"\[C6\] the scientific file src/ec_clean30/selector.py differs"):
        verify(copied)


def test_an_execution_file_that_imports_a_private_scientific_name_is_refused(world, tmp_path_factory):
    copied = copy_world(world, tmp_path_factory)
    path = copied["repo"] / "src/ec_clean30/workstation.py"
    commit_change(copied, "src/ec_clean30/workstation.py",
                  "from ec_clean30.selection import _hidden  # noqa\n" + path.read_text())
    with pytest.raises(AmendmentChainError, match=r"\[C7\] .*imports the private name _hidden"):
        verify(copied)


def test_a_changed_pinned_execution_function_is_refused(world, tmp_path_factory):
    copied = copy_world(world, tmp_path_factory)
    path = copied["repo"] / "src/ec_clean30/phase3.py"
    commit_change(copied, "src/ec_clean30/phase3.py", path.read_text().replace("LIMIT = 3", "LIMIT = 4"))
    with pytest.raises(AmendmentChainError, match=r"closure of src/ec_clean30/phase3.py::lane_gate differs"):
        verify(copied)


def test_an_unpublished_link_is_refused_for_real_use(world, tmp_path_factory):
    copied = copy_world(world, tmp_path_factory)
    _git(copied["repo"], "update-ref", freeze.ORIGIN_MAIN_REF, world["commits"]["B"])
    with pytest.raises(AmendmentChainError, match=r"\[C12\].*not published"):
        verify(copied)
    result = verify(copied, require_published=False)
    assert result["published"] is False and result["chain"] == [world["a1"], world["a2"]]
    assert [link["published"] for link in result["links"]] == [True, False]


def test_a_live_rendering_drift_is_refused(world, tmp_path_factory):
    copied = copy_world(world, tmp_path_factory)
    rendering = rules.rendering_path(rules.AMENDMENT2_PATH)
    commit_change(copied, rendering, (copied["repo"] / rendering).read_text() + "edited\n")
    with pytest.raises(AmendmentChainError, match=r"\[C2\] the live rendering of link 2 differs"):
        verify(copied)


def test_a_rendering_that_is_not_the_builders_is_refused_at_the_links_commit(tmp_path_factory):
    drifted = build_world(tmp_path_factory.mktemp("chain_drift") / "world", bad_rendering=True)
    with pytest.raises(AmendmentChainError, match=r"\[C2\] the rendering of link 2 differs from the builder's"):
        verify(drifted)


def test_a_fork_a_gap_and_a_reorder_are_refused(world, tmp_path_factory):
    with pytest.raises(AmendmentChainError, match=r"\[C3\] The chain differs from the expected chain"):
        verify(world, expected_chain=[world["a2"], world["a1"]])
    with pytest.raises(AmendmentChainError, match=r"\[C3\] The chain differs from the expected chain"):
        verify(world, expected_chain=[world["a1"], "f" * 64])
    copied = copy_world(world, tmp_path_factory)
    repo = copied["repo"]
    gap = {**world["record"], "sequence": 3, "amendment_id": "synthetic-later-link", "change_class": "execution"}
    _write(repo, rules.link_path(3), freeze.canonical_bytes(gap).decode())
    with pytest.raises(AmendmentChainError, match=r"\[C3\] Link 3 names a predecessor of sequence 1"):
        verify(copied, tip=rules.link_path(3), require_published=False)
    base = {"sequence": 2, "amendment_id": rules.AMENDMENT2_ID, "path": rules.AMENDMENT2_PATH,
            "sha256": world["a2"], "commit": world["commits"]["S2"]}
    reordered = {**gap, "base_amendment": base, "previous_sha256": world["a2"], "chain_before": [world["a2"],
                                                                                                  world["a1"]]}
    _write(repo, rules.link_path(3), freeze.canonical_bytes(reordered).decode())
    with pytest.raises(AmendmentChainError, match=r"\[C3\] chain_before of link 3 differs"):
        verify(copied, tip=rules.link_path(3), require_published=False)
    second = {**world["record"], "timestamps": {"draft_written": "another day"}}
    _write(repo, f"{DOCS}/amendment2b/amendment.json", freeze.canonical_bytes(second).decode())
    with pytest.raises(AmendmentChainError, match=r"\[C1\] Link 2 must be"):
        verify(copied, tip=f"{DOCS}/amendment2b/amendment.json", require_published=False)


def test_a_tip_below_sequence_2_does_not_verify_on_the_amendment2_tree(world):
    with pytest.raises(AmendmentChainError, match=r"\[C3\].*beyond the tip of sequence 1 \(a stale tip\)"):
        verify(world, tip=rules.AMENDMENT1["path"])


def test_the_import_probe_refuses_execution_analysis_and_untracked_loads(world, tmp_path_factory):
    copied = copy_world(world, tmp_path_factory)
    repo = copied["repo"]
    tracked = rules.tracked_source_files(repo)
    assert amendment_chain.import_isolation_problems(repo, EXECUTION_FILES, tracked) == []
    _write(repo, "src/ec_clean30/selector.py", "import ec_clean30.phase3\nimport ec_clean30.scoring\n")
    problems = amendment_chain.import_isolation_problems(repo, EXECUTION_FILES, tracked)
    assert any("execution file src/ec_clean30/phase3.py" in problem for problem in problems)
    assert any("post-fit analysis file src/ec_clean30/scoring.py" in problem for problem in problems)
    channel = amendment_chain.analysis_channel_problems(repo, ["src/ec_clean30/selector.py"])
    assert channel and "outside the pinned channels" in channel[0]
    untracked = [path for path in tracked if path != "src/ec_clean30/stats.py"]
    _write(repo, "src/ec_clean30/selector.py", "import ec_clean30.stats\n")
    assert any("loads the untracked file src/ec_clean30/stats.py" in problem
               for problem in amendment_chain.import_isolation_problems(repo, EXECUTION_FILES, untracked))


# ---------------------------------------------------------------------------
# Per-link checks on synthetic values


def _reader(files: dict):
    return lambda path: files.get(path)


def _evidence_files(world: dict) -> dict:
    record = world["record"]
    paths = {path for _where, path, _digest in amendment_chain.evidence_entries(record)}
    paths |= {record["rule_text"]["source"]}
    return {path: (world["repo"] / path).read_bytes() for path in paths}


def test_evidence_hash_drift_and_an_unsettled_decisions_file_are_refused(world):
    record, files = world["record"], _evidence_files(world)
    assert amendment_chain.evidence_problems(_reader(files), record) == []
    receipt = record["dry_runs"]["receipts"]["0"]["path"]
    drifted = {**files, receipt: files[receipt] + b" "}
    assert any("does not hash to its recorded SHA-256" in problem
               for problem in amendment_chain.evidence_problems(_reader(drifted), record))
    decisions = record["bound_files"]["decisions"]["path"]
    for text in ('{"maintainer": "<a name>"}', '{"note": "see /home/someone/file"}'):
        unsettled = {**files, decisions: text.encode()}
        changed = json.loads(json.dumps(record))
        changed["bound_files"]["decisions"]["sha256"] = changed["decisions_file"]["sha256"] = _sha(text.encode())
        problems = amendment_chain.evidence_problems(_reader(unsettled), changed)
        assert any("placeholder token" in problem or "absolute path" in problem for problem in problems)
    pending = {**record, "provisional_markers_settled": False}
    assert any("provisional markers" in problem for problem in amendment_chain.evidence_problems(_reader(files), pending))


def test_the_required_roles_the_decisions_file_and_the_rule_text_are_read_from_their_files(world):
    record, files = world["record"], _evidence_files(world)
    for role in rules.REQUIRED_BINDINGS:
        changed = json.loads(json.dumps(record))
        place(changed, role, "bound at build")
        assert any(problem.startswith(f"[C8] {role} ") for problem in
                   amendment_chain.evidence_problems(_reader(files), changed)), role
    decisions = record["bound_files"]["decisions"]["path"]
    payload = json.loads(files[decisions])
    del payload["answers"]["C6b"]
    data = json.dumps(payload).encode()
    changed = json.loads(json.dumps(record))
    changed["bound_files"]["decisions"]["sha256"] = changed["decisions_file"]["sha256"] = _sha(data)
    problems = amendment_chain.evidence_problems(_reader({**files, decisions: data}), changed)
    assert any("does not settle exactly" in problem for problem in problems)
    assert any("decisions_file differs from the bound decisions file" in problem for problem in problems)
    changed = json.loads(json.dumps(record))
    changed["decisions_file"]["answers"]["C1"]["choice"] = "another choice"
    assert any("decisions_file differs" in problem for problem in amendment_chain.evidence_problems(_reader(files),
                                                                                                   changed))
    source = record["rule_text"]["source"]
    assert any("does not hold the bound rule text" in problem for problem in amendment_chain.evidence_problems(
        _reader({**files, source: b"# another document\n"}), record))
    changed = json.loads(json.dumps(record))
    changed["rule_text"]["source"] = rules.rendering_path(rules.AMENDMENT2_PATH)
    assert any("is not the reviewed document" in problem for problem in amendment_chain.evidence_problems(
        _reader(files), changed))
    changed = json.loads(json.dumps(record))
    changed["rule_text"]["text"] += "added"
    assert any("does not hash to rule_text.sha256" in problem for problem in amendment_chain.evidence_problems(
        _reader(files), changed))


def test_a_tampered_policy_is_refused(world):
    record = world["record"]
    live = dict(record["rule_policy_sha256"])
    assert amendment_chain.policy_problems(record, live) == []
    tampered = {**live, "futility_predictive_cluster_bootstrap_v1": "0" * 64}
    assert any("futility_predictive_cluster_bootstrap_v1 differs" in problem
               for problem in amendment_chain.policy_problems(record, tampered))
    analysis = {**live, "amendment2_accuracy_v1": "0" * 64}
    assert any("amendment2_accuracy_v1 differs" in problem
               for problem in amendment_chain.policy_problems(record, analysis))
    renamed = json.loads(json.dumps(record))
    renamed["rules"]["controls_v2"]["policy_function"] = "ec_clean30.campaign:other"
    assert any("controls_v2 names another policy function" in problem
               for problem in amendment_chain.policy_problems(renamed, live))


def test_dry_run_receipts_and_plans_are_checked_against_freeze1(world):
    record, files = world["record"], _evidence_files(world)
    freeze1_doc = json.loads((world["repo"] / FREEZE1_PATH).read_text())
    problems, evidence = amendment_chain.dry_run_problems(_reader(files), record, freeze1_doc)
    assert problems == [] and evidence["source_commit"] == world["commits"]["D"]
    receipt = record["dry_runs"]["receipts"]["1"]["path"]
    failed = json.loads(files[receipt])
    failed["provenance"]["git"]["clean"] = False
    changed = json.loads(json.dumps(record))
    changed["dry_runs"]["receipts"]["1"]["sha256"] = _sha(json.dumps(failed).encode())
    problems, _ = amendment_chain.dry_run_problems(_reader({**files, receipt: json.dumps(failed).encode()}), changed,
                                                   freeze1_doc)
    assert any("did not run on a clean tree" in problem for problem in problems)
    changed = json.loads(json.dumps(record))
    changed["dry_runs"]["in_view_receipts_unchanged"]["2"] = "0" * 64
    problems, _ = amendment_chain.dry_run_problems(_reader(files), changed, freeze1_doc)
    assert any("in-view dry-run receipts differ" in problem for problem in problems)
    changed = json.loads(json.dumps(record))
    changed["plans"]["v1_plan_sha256"]["0"] = "0" * 64
    assert amendment_chain.plan_records(changed, freeze1_doc)[0]


def test_the_view_binding_requires_the_folds_amendment2_dry_run_evidence(verified):
    binding = amendment_chain.fold_dry_run_evidence(verified, 3)
    assert binding["receipt"] == verified["dry_run_evidence"]["receipts"]["3"]
    view_binding_without_dry_run_evidence = {**verified, "dry_run_evidence": None}
    with pytest.raises(AmendmentChainError, match=r"no amendment-2 dry-run evidence for fold 3"):
        amendment_chain.fold_dry_run_evidence(view_binding_without_dry_run_evidence, 3)


def test_scope_refuses_unlisted_changes_execution_files_and_deletions():
    previous = {"src/a.py": "1" * 64, "src/b.py": "2" * 64, "src/x.py": "3" * 64}
    tree = {"src/a.py": "4" * 64, "src/b.py": "2" * 64, "src/new.py": "5" * 64, "src/x.py": "6" * 64}
    good = {"src/a.py": {"before": "1" * 64, "after": "4" * 64}, "src/new.py": {"before": None, "after": "5" * 64}}
    problems, composite = amendment_chain.scope_problems(previous, tree, good, ["src/x.py"], ["src/b.py"], sequence=2)
    assert problems == [] and composite == {"src/a.py": "4" * 64, "src/b.py": "2" * 64, "src/new.py": "5" * 64}
    for changes, message in (
            ({"src/a.py": good["src/a.py"]}, "src/new.py changed but is not in source_changes"),
            ({**good, "src/x.py": {"before": "3" * 64, "after": "6" * 64}}, "execution file src/x.py is in scope"),
            ({**good, "src/a.py": {"before": "9" * 64, "after": "4" * 64}}, "before pin of src/a.py")):
        problems, _ = amendment_chain.scope_problems(previous, tree, changes, ["src/x.py"], sequence=2)
        assert any(message in problem for problem in problems), message
    deleted = {key: value for key, value in tree.items() if key != "src/b.py"}
    problems, _ = amendment_chain.scope_problems(previous, deleted, good, ["src/x.py"], sequence=2)
    assert any("pinned file src/b.py is deleted" in problem for problem in problems)
    problems, _ = amendment_chain.scope_problems(previous, tree, good, ["src/x.py"], ["src/a.py"], sequence=2)
    assert any("src/a.py is declared unchanged" in problem for problem in problems)
    problems, _ = amendment_chain.scope_problems(previous, tree, good, ["src/x.py"], sequence=3)
    assert any("src/x.py cannot join the execution set" in problem for problem in problems)


def test_data_state_rules_of_link_2_and_of_later_links(world):
    assert amendment_chain.data_state_problems(world["record"]) == []
    for key, value in (("fold0_records_at_draft", ["0" * 64]), ("g2_completed_units", 1),
                       ("g2_completed_units", False)):
        changed = {**world["record"], "data_state": {**world["record"]["data_state"], key: value}}
        assert amendment_chain.data_state_problems(changed)
    for example in CONTRACT["change_class_examples"]:
        problems = amendment_chain.later_link_problems(
            example["link"], changed_files=example["changed_files"],
            execution_files=example.get("execution_files", []), fut_record_exists=example["fut_record_exists"],
            stop_futility_exists=example["stop_futility_exists"], previous=example.get("previous"))
        assert len(problems) == example["expected_problems"], example["name"]
    binding = {"path": f"{DOCS}/amendment3/fold0/fut_seq2.json", "sha256": "b" * 64}
    fut = {"kind": amendment_chain.FUT_KIND, "sha256": "a" * 64, "decision": amendment_chain.FUTILITY_STOP,
           "pointer": binding, "record": {**binding, "path": f"{DOCS}/amendment3/fold0/futility_record.json"}}
    assert amendment_chain.futility_flags([{"sequence": 3, "fold0_records": [fut]}]) == (True, True, [])
    later = {"sequence": 3, "fold0_records": [{**fut, "decision": None}, {"kind": "S1"},
                                              {key: value for key, value in fut.items() if key != "pointer"}]}
    exists, stopped, problems = amendment_chain.futility_flags([later])
    assert not exists and not stopped and len(problems) == 3


def test_analysis_completion_keeps_every_analysis_limit():
    state = {"pinned_analysis_functions": {"p": "1" * 64}, "live_constants_sha256": "2" * 64,
             "scientific_identity_sha256": "3" * 64, "fit_chain": ["4" * 64],
             "rule_policy_sha256": {rule_id: "5" * 64 for rule_id in rules.RULE_POLICY_FUNCTIONS},
             "refused_fixtures": ["r1", "r2"], "decision_fields": {"cases": []}}
    assert amendment_chain.analysis_completion_problems(state, dict(state, refused_fixtures=["r1", "r2", "r3"])) == []
    for key, value in (("live_constants_sha256", "6" * 64), ("fit_chain", ["7" * 64]),
                       ("refused_fixtures", ["r1"]), ("decision_fields", {"cases": [1]})):
        problems = amendment_chain.analysis_completion_problems(state, dict(state, **{key: value}))
        assert problems and all(problem.startswith("[C16]") for problem in problems), key


def test_the_scientific_identity_moves_with_scientific_pins_only(verified):
    components = {
        "fit_chain": [rules.AMENDMENT1["sha256"], "b" * 64],
        "composite_scientific_pins": {"src/ec_clean30/selection.py": "1" * 64},
        "pinned_execution_functions": {"src/ec_clean30/phase3.py::lane_gate": "2" * 64},
        "plans": verified["plan_records"], "freeze1_views": {"0": {"dry_run_receipt": {"sha256": "3" * 64}}},
        "dry_run_evidence": verified["dry_run_evidence"], "deferred_input_specifications": {"f6_inputs": {}}}
    record = amendment_chain.identity_record(**components)
    assert tuple(record) == rules.SCIENTIFIC_IDENTITY_KEYS
    first = rules.scientific_identity(record)
    assert rules.scientific_identity(amendment_chain.identity_record(**components)) == first
    changed = dict(components, composite_scientific_pins={"src/ec_clean30/selection.py": "4" * 64})
    assert rules.scientific_identity(amendment_chain.identity_record(**changed)) != first
    with pytest.raises(rules.AmendmentRuleError, match="no execution, commit"):
        rules.scientific_identity({**record, "commit": "5" * 40})
    with pytest.raises(rules.AmendmentRuleError, match="deferred_input_specifications.f6_inputs.host"):
        rules.scientific_identity({**record, "deferred_input_specifications": {"f6_inputs": {"host": "pod-1"}}})


def _record_at(world: dict, commit: str, created: datetime) -> dict:
    return {"fit_chain": [world["a1"], world["a2"]], "created_utc": created.isoformat(),
            "provenance": {"git": {"head": commit}}}


def test_records_are_bound_by_git_ancestry_and_publication_not_by_a_clock(world, verified, tmp_path_factory):
    repo = world["repo"]
    published = datetime.fromisoformat(verified["links"][-1]["source_commit_time"])
    later = published + timedelta(hours=1)
    at_s2 = _record_at(world, world["commits"]["S2"], later)
    assert amendment_chain.record_precedence_problems(verified, [at_s2], repo=repo) == []
    assert amendment_chain.record_precedence_problems(
        verified, [{**at_s2, "provenance": None, "git_head": world["commits"]["S2"]}], repo=repo) == []
    # A record made at the dry-run commit D (before amendment 2 was published) is refused whatever its clock says.
    early = _record_at(world, world["commits"]["D"], later + timedelta(days=30))
    assert any("made before the commit that published" in problem
               for problem in amendment_chain.record_precedence_problems(verified, [early], repo=repo))
    # The clock stays a secondary check.
    clock = _record_at(world, world["commits"]["S2"], published - timedelta(hours=1))
    assert any("created before the commit" in problem
               for problem in amendment_chain.record_precedence_problems(verified, [clock], repo=repo))
    headless = {"fit_chain": at_s2["fit_chain"], "created_utc": at_s2["created_utc"]}
    assert any("names no commit" in problem
               for problem in amendment_chain.record_precedence_problems(verified, [headless], repo=repo))
    copied = copy_world(world, tmp_path_factory)
    local = commit_change(copied, "src/ec_clean30/workstation.py", '"""Synthetic execution file."""\nHOST = "x"\n',
                          push=False)
    assert any("not published" in problem for problem in amendment_chain.record_precedence_problems(
        verified, [_record_at(world, local, later)], repo=copied["repo"]))
    chain = verified["fit_chain"]
    prefix = {"fit_chain": chain[:1], "created_utc": at_s2["created_utc"]}
    assert amendment_chain.record_precedence_problems(verified, [prefix], repo=repo)
    assert amendment_chain.record_precedence_problems(verified, [prefix], repo=repo, replay=True) == []
    assert any("mixed" in problem for problem in amendment_chain.record_precedence_problems(
        verified, [at_s2, prefix], repo=repo, replay=True))
    with pytest.raises(AmendmentChainError, match="another fit chain"):
        amendment_chain.require_record_precedence(verified, [{"fit_chain": ["c" * 64], "created_utc": "x"}],
                                                  repo=repo)
