"""The amendment-2 refusal and decision fixtures that class (a2) relies on (rule A2-GOV-7; synthetic only).

``tests/fixtures/ec_clean30_amendment2_decisions.json`` holds the decision fields of fixed synthetic scoring tables
(co-primary decisions, availability of every co-primary and secondary step); a class-(a2) link must reproduce them.
``tests/fixtures/ec_clean30_amendment2_refusals.json`` names the tests that prove each amendment-2 refusal; every
named test must exist (and the suite runs it), so the refusal set never shrinks. Both files are bound by SHA-256 in
the amendment-2 record.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]

from ec_clean30 import scoring  # noqa: E402

DECISIONS = ROOT / "tests" / "fixtures" / "ec_clean30_amendment2_decisions.json"
REFUSALS = ROOT / "tests" / "fixtures" / "ec_clean30_amendment2_refusals.json"
DECISIONS_FORMAT = "deepmzyme_ec_clean30_amendment2_decision_fixtures_v1"
REFUSALS_FORMAT = "deepmzyme_ec_clean30_amendment2_refusal_fixtures_v1"
CASES = {"clear_gain": (), "r1_disclosed": ("r1",), "r2_disclosed": ("r2",)}


def decision_fields() -> dict:
    """Decision fields of the fixed synthetic tables under the amendment-2 scoring rule (test draw count)."""
    import test_ec_clean30_scoring as helpers

    specs = helpers.specs_for({"nh30": 10, "high": 10})
    table = helpers.synthetic_table(specs, helpers.truth,
                                    lambda index, spec: spec[1] if index % 2 else helpers.wrong(spec[1]))
    cases = {}
    for name, disclosed in CASES.items():
        result = scoring.analyse_comparison(table, helpers.ANALYSIS_SETTINGS, clean_disclosed_arms=disclosed)
        cases[name] = {"decisions": dict(result["decisions"]),
                       "co_primary_available": [step["available"] for step in result["co_primary"]["steps"]],
                       "secondary_available": {step["name"]: step["available"]
                                               for step in result["secondary_family"]["steps"]}}
    return {"format": DECISIONS_FORMAT, "rule_id": scoring.SCORING_RULE_AMENDMENT2,
            "settings": "ANALYSIS_SETTINGS of tests/test_ec_clean30_scoring.py", "cases": cases}


def test_the_decision_fixture_is_reproduced():
    assert json.loads(DECISIONS.read_text(encoding="utf-8")) == json.loads(json.dumps(decision_fields()))


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def test_every_refusal_fixture_names_an_existing_test():
    fixture = json.loads(REFUSALS.read_text(encoding="utf-8"))
    assert fixture["format"] == REFUSALS_FORMAT and fixture["refusals"]
    ids = [item["id"] for item in fixture["refusals"]]
    assert len(set(ids)) == len(ids)
    for item in fixture["refusals"]:
        file_name, _, name = item["test"].partition("::")
        assert name in _test_names(ROOT / "tests" / file_name), item
        assert item["rule"].startswith("A2-") and item["refuses"]
