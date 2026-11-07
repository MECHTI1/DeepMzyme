"""Shared rules of the amendment chain (``ec_clean30.amendment_rules``) and the contract fixture (synthetic only)."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ec_clean30 import amendment_rules as rules  # noqa: E402

CONTRACT = json.loads((ROOT / "tests" / "fixtures" / "ec_clean30_amendment2_contract.json").read_text())


def test_chain_constants_match_the_tracked_records():
    import hashlib

    amendment = ROOT / rules.AMENDMENT1["path"]
    assert hashlib.sha256(amendment.read_bytes()).hexdigest() == rules.AMENDMENT1["sha256"]
    assert CONTRACT["amendment_record"]["base_amendment"]["sha256"] == rules.AMENDMENT1["sha256"]
    assert CONTRACT["amendment_record"]["header"]["format"] == rules.AMENDMENT2_FORMAT
    from ec_clean30 import selector

    assert selector.AMENDMENT_V2_FORMAT == rules.AMENDMENT2_FORMAT


def test_function_closure_pins_are_stable_and_follow_referenced_helpers(tmp_path):
    source = ("import os\nLIMIT = 3\n\n\ndef helper(x):\n    return x + LIMIT\n\n\ndef unused():\n    return 1\n\n\n"
              "def target(x):\n    return helper(x) * 2\n")
    path = tmp_path / "module.py"
    path.write_text(source)
    closure = rules.function_closure(source, ["target"])
    assert sorted(closure) == ["LIMIT", "helper", "target"]
    first = rules.function_closure_sha256(path, ["target"])
    assert rules.function_closure_sha256(path, ["target"]) == first
    path.write_text(source.replace("def unused():\n    return 1", "def unused():\n    return 2"))
    assert rules.function_closure_sha256(path, ["target"]) == first  # an unreferenced name does not matter
    path.write_text(source.replace("LIMIT = 3", "LIMIT = 4"))
    assert rules.function_closure_sha256(path, ["target"]) != first  # a referenced constant does
    path.write_text(source.replace("x + LIMIT", "x - LIMIT"))
    assert rules.function_closure_sha256(path, ["target"]) != first  # a referenced helper does
    path.write_text(source + "\n\ndef helper(x):\n    return x\n")
    with pytest.raises(rules.AmendmentRuleError, match="bound 2 times"):
        rules.function_closure_sha256(path, ["target"])
    with pytest.raises(rules.AmendmentRuleError, match="not defined"):
        rules.function_closure(source, ["missing"])


def test_pinned_analysis_functions_exist_and_hash():
    hashes = rules.pinned_function_hashes(ROOT, rules.PINNED_ANALYSIS_FUNCTIONS)
    assert set(hashes) == {"src/ec_clean30/baseline.py::read_official_ec_numbers",
                           "src/ec_clean30/scoring.py::_load_comparator", "src/ec_clean30/scoring.py::parse_label_table",
                           "src/ec_clean30/scoring.py::parse_transfer_list"}
    assert all(len(value) == 64 for value in hashes.values())
    assert set(rules.POST_FIT_ANALYSIS_FILES) == {
        "scripts/run_ec_clean30_baseline.py", "scripts/score_ec_clean30_comparison.py", "src/ec_clean30/baseline.py",
        "src/ec_clean30/scoring.py"}


def test_the_partition_is_closed_and_refuses_forbidden_execution_files():
    files = ["src/ec_clean30/phase4.py", "src/ec_clean30/selection.py", "src/ec_clean30/scoring.py",
             "src/ec_clean30/baseline.py", "scripts/run_ec_clean30_phase4.py", "src/training/run.py"]
    split = rules.partition(files, ["src/ec_clean30/phase4.py", "scripts/run_ec_clean30_phase4.py"])
    assert split["execution_files"] == ["scripts/run_ec_clean30_phase4.py", "src/ec_clean30/phase4.py"]
    assert split["post_fit_analysis_files"] == ["src/ec_clean30/baseline.py", "src/ec_clean30/scoring.py"]
    assert split["scientific_files"] == ["src/ec_clean30/selection.py", "src/training/run.py"]
    for forbidden in ("src/ec_clean30/selection.py", "src/training/run.py", "src/ec_clean30/scoring.py",
                      "scripts/run_clean_split30_comparator.py", "src/ec_clean30/runtime_precision.py",
                      "src/ec_clean30/amendment_chain.py", "src/ec_clean30/transfers.py", "src/embed_helpers/x.py"):
        with pytest.raises(rules.AmendmentRuleError, match="cannot be an execution file"):
            rules.partition([*files, forbidden], [forbidden])
    assert set(rules.EXECUTION_WRITTEN_SCIENTIFIC_FILES) <= rules.FORBIDDEN_EXECUTION_FILES
    problems = rules.private_import_problems("from ec_clean30.selection import _check_family\n", "phase4.py")
    assert problems and "_check_family" in problems[0]
    assert not rules.private_import_problems("from ec_clean30.selection import FAMILIES\n", "phase4.py")


@pytest.mark.parametrize("example", CONTRACT["change_class_examples"], ids=lambda item: item["name"])
def test_change_classes_of_later_links(example):
    problems = rules.check_change_class(example["link"], changed_files=example["changed_files"],
                                        execution_files=example.get("execution_files", ()),
                                        fut_record_exists=example["fut_record_exists"],
                                        stop_futility_exists=example["stop_futility_exists"],
                                        previous=example.get("previous"))
    assert len(problems) == example["expected_problems"], problems


def test_analysis_completion_never_shrinks_the_refusal_set_or_moves_the_identity():
    policies = {rule: f"{index:064x}" for index, rule in enumerate(rules.RULE_POLICY_FUNCTIONS)}
    state = {"pinned_analysis_functions": {"a": "1"}, "live_constants_sha256": "c", "scientific_identity_sha256": "s",
             "fit_chain": ["x"], "refused_fixtures": ["r1", "r2"], "decision_fields": {"d": 1},
             "rule_policy_sha256": policies}
    assert rules.check_analysis_completion(previous=state, current=dict(state, refused_fixtures=["r1", "r2", "r3"])) == []
    assert rules.check_analysis_completion(previous=state, current=dict(state, refused_fixtures=["r1"]))
    assert rules.check_analysis_completion(previous=state, current=dict(state, decision_fields={"d": 2}))
    for key in ("pinned_analysis_functions", "live_constants_sha256", "scientific_identity_sha256", "fit_chain"):
        assert rules.check_analysis_completion(previous=state, current=dict(state, **{key: "moved"}))
    # Every rule's policy, the analysis rule included, is frozen under class (a2).
    moved = dict(policies, amendment2_accuracy_v1="f" * 64)
    problems = rules.check_analysis_completion(previous=state, current=dict(state, rule_policy_sha256=moved))
    assert problems and "amendment2_accuracy_v1" in problems[0]
    assert rules.check_analysis_completion(previous={k: v for k, v in state.items() if k != "rule_policy_sha256"},
                                           current=state)


def test_the_live_rule_policies_cover_every_rule_and_the_scoring_policy_holds_the_v2_decision_text():
    from ec_clean30 import scoring

    live = rules.rule_policy_sha256()
    assert set(live) == set(rules.RULE_POLICY_FUNCTIONS) and "amendment2_accuracy_v1" in live
    policy = scoring.scoring_policy_record_v2()
    assert policy["settings"] == scoring.FROZEN_SETTINGS_V2 and "scoring_rules" in policy
    assert policy["decision_sentences"]["unavailable"] == scoring.DECISION_UNAVAILABLE_V2
    assert policy["secondary_family"] == [list(item) for item in scoring.SECONDARY_FAMILY_V2]
    assert policy["held_out_consequence"]["rule_id"] == "held_out_arm_consequence_v1"


def test_r_cmd_v2_allows_execution_and_allowlisted_files_only():
    execution = ["src/ec_clean30/phase4.py"]
    allowed = [("M", "src/ec_clean30/phase4.py"), ("M", "src/ec_clean30/scoring.py"), ("A", "src/ec_clean30/new.py"),
               ("M", "tests/test_x.py"), ("M", "docs/x.md")]
    assert rules.r_cmd_v2_change_problems(allowed, execution) == []
    problems = rules.r_cmd_v2_change_problems([("M", "src/ec_clean30/commands.py"), ("D", "src/ec_clean30/stats.py"),
                                               ("M", "src/training/run.py")], execution)
    assert len(problems) == 3 and all("r_cmd_v2" in problem for problem in problems)


def test_scientific_identity_refuses_execution_and_host_fields():
    record = {key: None for key in rules.SCIENTIFIC_IDENTITY_KEYS}
    record.update(campaign_id=rules.CAMPAIGN_ID, freeze1_sha256=rules.FREEZE1_SHA256,
                  freeze1b_sha256=rules.FREEZE1B_SHA256, fit_chain=[rules.AMENDMENT1["sha256"], "a2" * 32],
                  precision_policy=rules.PRECISION_POLICY)
    digest = rules.scientific_identity(record)
    assert digest == rules.scientific_identity(dict(record)) and len(digest) == 64
    for key in ("git_head", "provider", "execution_manifest_sha256", "session_id"):
        with pytest.raises(rules.AmendmentRuleError, match="no execution, commit, provider"):
            rules.scientific_identity({**record, key: "x"})
    with pytest.raises(rules.AmendmentRuleError, match="keys differ"):
        rules.scientific_identity({key: value for key, value in record.items() if key != "plan_sha256_v2"})
    with pytest.raises(rules.AmendmentRuleError, match="start with amendment 1"):
        rules.scientific_identity({**record, "fit_chain": ["a2" * 32]})


def test_rule_policies_resolve_and_analysis_rules_stay_out_by_default():
    records = rules.policy_records()
    assert "amendment2_accuracy_v1" not in records and records["amendment2_selection_v1"]["policy"]["selection_pipeline"]
    assert set(rules.policy_records(include_analysis=True)) == set(rules.RULE_POLICY_FUNCTIONS)
    from ec_clean30 import selection

    assert records["amendment2_selection_v1"]["policy"] == json.loads(json.dumps(selection.selection_policy_record_v2()))


def test_contract_names_exist_in_the_science_files():
    for group, names in CONTRACT["science_api"].items():
        for name in names:
            module, attribute = name.split(":")
            assert hasattr(importlib.import_module(module), attribute), (group, name)


def test_the_rules_module_is_torch_free_and_loads_no_post_fit_analysis_file():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import ec_clean30.amendment_rules; "
            "import json; print(json.dumps(sorted(sys.modules)))")
    loaded = json.loads(subprocess.run([sys.executable, "-c", code, str(ROOT / "src")], capture_output=True, text=True,
                                       check=True).stdout.strip().splitlines()[-1])
    assert "torch" not in loaded and "ec_clean30.scoring" not in loaded and "ec_clean30.baseline" not in loaded


def test_private_names_are_refused_through_every_import_form():
    path = "src/ec_clean30/workstation.py"
    files = {"src/ec_clean30/__init__.py", "src/ec_clean30/selection.py", "src/ec_clean30/views.py",
             "src/ec_clean30/phase3.py", path, "src/training/__init__.py", "src/training/ec_prediction_export.py",
             "scripts/run_clean_split30_comparator.py"}
    execution = {path, "src/ec_clean30/phase3.py"}

    def protected(name):
        target = rules.module_file(name, files)
        return target is not None and target not in execution

    refused = {
        "from .selection import _hidden\n": "imports the private name _hidden of ec_clean30.selection",
        "from . import selection as s\nX = s._hidden\n": "reaches the private name _hidden of ec_clean30.selection",
        "from ec_clean30 import views\nviews._remove_tree(1)\n": "reaches the private name _remove_tree",
        "import ec_clean30.views\nec_clean30.views._secret = 1\n": "reaches the private name _secret",
        "import ec_clean30.selection as sel\ndel sel._cache\n": "reaches the private name _cache",
        "from training import ec_prediction_export as export\nexport._predict()\n": "_predict of training.",
        "from scripts import run_clean_split30_comparator as clean\nclean._run()\n": "_run of scripts.",
        "import importlib\nm = importlib.import_module('ec_clean30.views')\nm._x()\n": "reaches the private name _x",
        "from ec_clean30 import views\nsetattr(views, '_x', 1)\n": "through setattr",
        "from ec_clean30 import views\nX = views.__dict__\n": "reaches the private name __dict__",
    }
    for text, message in refused.items():
        problems = rules.private_import_problems(text, path, protected=protected)
        assert any(message in problem for problem in problems), (text, problems)
    allowed = ("from ec_clean30 import phase3\nphase3._helper()\n", "from ec_clean30 import views\nX = views.__name__\n",
               "import os\nos._exit\n", "from ec_clean30 import views\nX = views.VIEW\n",
               "host = object()\nhost._reserve()\n")
    for text in allowed:
        assert rules.private_import_problems(text, path, protected=protected) == [], text
    assert rules.resolve_import_from("src/ec_clean30/workstation.py", ast.parse("from ..training.run import main").body[0]) \
        == "training.run"
    assert rules.module_name("src/ec_clean30/__init__.py") == "ec_clean30"
    assert rules.module_name("scripts/run_clean_split30_comparator.py") == "scripts.run_clean_split30_comparator"


def test_link_paths_log_lines_and_the_role_table():
    assert rules.link_path(2) == rules.AMENDMENT2_PATH
    assert rules.link_path(3).endswith("/amendment3/amendment.json")
    assert rules.rendering_path(rules.link_path(3)).endswith("/amendment3/amendment.md")
    assert rules.rule_text_source(2).endswith("/amendment2/amendment_draft.md")
    with pytest.raises(rules.AmendmentRuleError):
        rules.link_path(1)
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    log = f"- S1 amendment `{a}`\n- {rules.link_log_line(2, b)}\n"
    assert rules.link_log_problems(log, [a, b]) == []
    assert rules.link_log_problems(log.replace(f"- {rules.link_log_line(2, b)}\n", ""), [a, b])
    assert rules.link_log_problems(log.replace(f"- {rules.link_log_line(2, b)}\n", ""), [a, b], tip_logged=False) == []
    assert any("beyond the tip" in problem for problem in rules.link_log_problems(log, [a]))
    assert any("a fork or a rewrite" in problem
               for problem in rules.link_log_problems(log + rules.link_log_line(2, c) + "\n", [a, b]))
    assert any("another link of sequence 2" in problem for problem in rules.link_log_problems(log, [a, c]))
    assert any("outside the strict line format" in problem
               for problem in rules.link_log_problems(log + f"Amendment chain link 3 is `{c}`\n", [a, b]))
    record = {}
    assert sorted(problem.split(" ")[0] for problem in rules.required_binding_problems(record)) == sorted(
        rules.REQUIRED_BINDINGS)
    binding = {"path": "docs/x.json", "sha256": a}
    folds = {str(fold): binding for fold in range(5)}
    full = {"data_state": {"attestation_at_build": binding}, "dry_runs": {"v1_resolved_config": binding},
            "evidence": {name: binding for name in ("futility_oc", "interface_contract", "power_simulation",
                                                    "verification")},
            "inputs": {"clean_tie_exposure": binding, "f6_holdout": {**folds, "4": {"inner0": binding}},
                       "f6_corpus": {"tables": folds, "union_fasta": binding}}}
    assert rules.required_binding_problems(full) == []
    for role, value in (("inputs.f6_holdout", {"0": binding}), ("inputs.f6_corpus", {"tables": folds}),
                        ("evidence.verification", {"path": "/abs.json", "sha256": a})):
        broken = json.loads(json.dumps(full))
        target = broken
        for key in role.split(".")[:-1]:
            target = target[key]
        target[role.split(".")[-1]] = value
        assert any(problem.startswith(role) for problem in rules.required_binding_problems(broken)), role


def test_execution_and_analysis_links_keep_the_previous_function_pins():
    pins = {"pinned_execution_functions": {"p::lane_gate": "1" * 64}, "pinned_analysis_functions": {"s::f": "2" * 64}}
    link = {"change_class": "execution", "sequence": 3, "timestamps": {}, "fold0_records": [], **pins}
    assert rules.check_change_class(link, changed_files=["p"], execution_files=["p"], previous=pins) == []
    moved = {**link, "pinned_execution_functions": {"p::lane_gate": "3" * 64}}
    assert rules.check_change_class(moved, changed_files=["p"], execution_files=["p"], previous=pins) == [
        "an execution link changes pinned_execution_functions (the pinned functions stay those of the previous link)"]
    defect = {**moved, "change_class": "defect", "defect": "d", "failing_test": "t",
              "found_through_record_sha256": "0" * 64, "inner_score_viewed_before_identification": False,
              "reruns": []}
    assert rules.check_change_class(defect, changed_files=["src/ec_clean30/selection.py"], execution_files=["p"],
                                    previous=pins) == []
