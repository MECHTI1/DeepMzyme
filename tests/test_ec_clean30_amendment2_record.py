"""Amendment-2 record tools: receipts the scoring pass binds, the decisions file, the builder and its rendering.

Synthetic inputs only (no campaign record, view or data folder is read).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]

from ec_clean30 import amendment2, scoring, selector  # noqa: E402
from ec_clean30.amendment2 import AmendmentRecordError  # noqa: E402

FREEZE1 = "28077f4c3701820c9aa20f53b570d12f8b70a46514211fc63e62976e393a3c0c"


def test_the_scorer_names_the_same_receipt_formats_without_importing_this_module():
    assert amendment2.receipt_formats() == {"chain": scoring.CHAIN_RECEIPT_FORMAT,
                                            "refit_recipe": scoring.REFIT_RECIPE_RECEIPT_FORMAT}
    source = (ROOT / "src" / "ec_clean30" / "scoring.py").read_text()
    assert "amendment2 import" not in source and "import amendment2" not in source


def _decisions(tmp_path, **changes):
    answers = {key: {"choice": "the recommended default", "accepted_default": True}
               for key in amendment2.CONFIRMATION_IDS}
    payload = {"format": amendment2.DECISIONS_FORMAT, "maintainer": "the maintainer", "answers": answers}
    payload.update(changes)
    path = tmp_path / f"decisions_{len(list(tmp_path.iterdir()))}.json"
    path.write_text(json.dumps(payload))
    return path


def test_the_decisions_file_answers_every_confirmation_without_placeholders_or_paths(tmp_path):
    assert set(amendment2.check_decisions_file(_decisions(tmp_path))["answers"]) == set(amendment2.CONFIRMATION_IDS)
    assert "C6b" in amendment2.CONFIRMATION_IDS
    missing = {key: {"choice": "x", "accepted_default": True} for key in amendment2.CONFIRMATION_IDS if key != "C6b"}
    with pytest.raises(AmendmentRecordError, match="answer exactly"):
        amendment2.check_decisions_file(_decisions(tmp_path, answers=missing))
    with pytest.raises(AmendmentRecordError, match="placeholder"):
        amendment2.check_decisions_file(_decisions(tmp_path, maintainer="[name]"))
    with pytest.raises(AmendmentRecordError, match="absolute path"):
        amendment2.check_decisions_file(_decisions(tmp_path, note="see /home/somebody/file"))


def test_the_builder_binds_its_inputs_once_and_the_rendering_carries_the_rule_text(tmp_path_factory):
    """The real builder on the synthetic published repository (``ec_clean30_amendment2_build``): every input bound
    once, the rendering carries the rule text; a pending source change and a source that differs from the scope list
    are refused and nothing is written."""
    import ec_clean30_amendment2_build as build
    import test_ec_clean30_amendment_chain as harness

    world = build.build_world_with_builder(tmp_path_factory.mktemp("record_builder") / "world")
    repo, record = world["repo"], world["record"]
    assert record["format"] == amendment2.RECORD_FORMAT and record["confirmation_status"] == "confirmed"
    assert "amendment2_accuracy_v1" in record["rule_policy_sha256"] and "source_changes_pending" not in record
    fixture = harness.REFUSALS_FIXTURE
    out = world["root"] / "scratch" / "bound.json"
    bound = build.run_builder(world, out=out, bind={"refusals_copy": repo / fixture})["record"]
    assert bound["bound_files"]["refusals_copy"] == {"path": fixture,
                                                     "sha256": hashlib.sha256((repo / fixture).read_bytes()).hexdigest()}
    assert bound["bound_files"]["decisions"] == record["bound_files"]["decisions"]
    assert {key: value for key, value in bound.items() if key not in ("bound_files", "sha256_of_content")} == {
        key: value for key, value in record.items() if key not in ("bound_files", "sha256_of_content")}
    rendered = (repo / "docs/campaigns/ec_class_vs_clean30_2026-10/amendment2/amendment.md").read_text()
    assert rendered == amendment2.render_amendment2(record)
    assert "**A2-X-1** A synthetic rule." in rendered and "## Policy deviations" not in rendered
    assert "| C6b | the recommended default | yes |" in rendered
    assert build.run_builder(world, out=out)["error"].startswith("FileExistsError")
    draft = json.loads(world["inputs"]["draft"].read_text())
    pending = world["root"] / "drafts" / "pending.json"
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps(dict(draft, source_changes_pending={
        "src/ec_clean30/runtime_precision.py": {"written_by": "execution v5", "after": None}})))
    refused = world["root"] / "scratch" / "refused.json"
    error = build.run_builder(world, out=refused, draft=pending)["error"]
    assert "still pending" in error and "src/ec_clean30/runtime_precision.py (execution v5)" in error
    changed = dict(draft, source_changes={**draft["source_changes"], "src/ec_clean30/amendment2.py": {
        **draft["source_changes"]["src/ec_clean30/amendment2.py"], "after": "0" * 64}})
    pending.write_text(json.dumps(changed))
    error = build.run_builder(world, out=refused, draft=pending)["error"]
    assert "differ from the record's scope list: ['src/ec_clean30/amendment2.py']" in error
    unlisted = {key: value for key, value in draft["source_changes"].items() if key != "src/ec_clean30/amendment2.py"}
    pending.write_text(json.dumps(dict(draft, source_changes=unlisted)))
    error = build.run_builder(world, out=refused, draft=pending)["error"]
    assert "execution file set derived from the live tree is refused" in error and "amendment2.py" in error
    assert not refused.exists()


def test_the_chain_receipt_needs_the_general_verifier_and_a_published_chain(tmp_path, monkeypatch):
    tip = tmp_path / "amendment.json"
    tip.write_text("{}\n")
    import ec_clean30

    # Hide the verifier: once imported, the submodule is also an attribute of the package.
    monkeypatch.setitem(sys.modules, "ec_clean30.amendment_chain", None)
    monkeypatch.delattr(ec_clean30, "amendment_chain", raising=False)
    with pytest.raises(AmendmentRecordError, match="general amendment-chain verifier"):
        amendment2.write_chain_receipt(tip, FREEZE1, tmp_path / "receipt.json")
    monkeypatch.undo()
    from ec_clean30 import amendment_chain

    result = {"format": scoring.CHAIN_RECEIPT_FORMAT, "verified": True, "chain": ["a" * 64, "b" * 64],
              "fit_chain": ["a" * 64, "b" * 64], "tip_sequence": 2, "scientific_identity_sha256": "c" * 64,
              "published": False, "require_published": True, "draft": False}
    calls = []

    def receipt(tip_path, **kwargs):
        calls.append(kwargs)
        return dict(result)

    monkeypatch.setattr(amendment_chain, "chain_receipt_record", receipt)
    with pytest.raises(AmendmentRecordError, match="published chain"):
        amendment2.write_chain_receipt(tip, FREEZE1, tmp_path / "receipt.json")
    result["published"] = True
    written = amendment2.write_chain_receipt(tip, FREEZE1, tmp_path / "receipt.json", expected_chain=result["chain"])
    assert written["format"] == scoring.CHAIN_RECEIPT_FORMAT and written["verified"] is True
    assert json.loads((tmp_path / "receipt.json").read_text())["fit_chain"] == result["fit_chain"]
    assert calls[-1]["require_published"] is True and calls[-1]["expected_chain"] == result["chain"]
    amendment2.write_chain_receipt(tip, FREEZE1, tmp_path / "draft_receipt.json", draft=True)
    assert calls[-1]["require_published"] is False


def test_a_chain_refusal_exits_with_an_error_line_and_code_2(tmp_path, monkeypatch, capsys):
    from ec_clean30.amendment_chain import AmendmentChainError

    tip = tmp_path / "amendment.json"
    tip.write_text("{}\n")
    out = tmp_path / "receipt.json"

    from ec_clean30 import amendment_chain

    def refuse(tip_path, **kwargs):
        raise AmendmentChainError("[C12] The amendment chain is not published; real use is refused.")

    monkeypatch.setattr(amendment_chain, "chain_receipt_record", refuse)
    with pytest.raises(AmendmentRecordError, match=r"^\[C12\] The amendment chain is not published"):
        amendment2.write_chain_receipt(tip, FREEZE1, out)
    args = argparse.Namespace(tip=str(tip), freeze1_sha256=FREEZE1, out=str(out), draft=True)
    assert amendment2.action_verify_amendment_chain(args) == 2
    captured = capsys.readouterr()
    assert captured.err == "error: [C12] The amendment chain is not published; real use is refused.\n"
    assert captured.out == "" and not out.exists()


def test_the_refit_recipe_receipt_holds_what_the_scoring_pass_compares(tmp_path_factory, monkeypatch):
    from ec_clean30 import futility

    tmp_path = tmp_path_factory.mktemp("refit_recipe")
    from ec_clean30.campaign import Recipe

    spec = selector.RefitSpec(0, Recipe("F3", (("gvp_learning_rate", 3e-4), ("learning_rate", 1e-4)), 30, (), "raw"),
                              0.5, (42, 43, 44, 45, 46), None, "d" * 64, tuple(f"u{seed}" for seed in range(5)))
    monkeypatch.setattr(selector, "verified_refit_recipe", lambda *args, **kwargs: spec)
    monkeypatch.setattr(selector, "_selection_policy", lambda *args, **kwargs: {
        "freeze1_amendment_chain": ["a" * 64, "b" * 64], "scientific_identity_sha256": "c" * 64})
    monkeypatch.setattr(futility, "verify_futility_decision", lambda *args, **kwargs: {
        "decision": "continue", "record_sha256": "e" * 64, "pointer_sequence": 2})
    out = tmp_path / "refit_fold0.json"
    receipt = amendment2.write_refit_recipe_receipt(fold=0, step_chain=tmp_path / "c.jsonl", freeze1_sha256=FREEZE1,
                                                    amendment_tip=tmp_path / "a.json", campaign_root=tmp_path,
                                                    plan_path=tmp_path / "p.json", out=out)
    written = json.loads(out.read_text())
    assert written["format"] == scoring.REFIT_RECIPE_RECEIPT_FORMAT and written["fold"] == 0
    assert written["futility"] == {"decision": "continue", "record_sha256": "e" * 64, "pointer_sequence": 2,
                                   "binding": True}
    assert written["refit"]["tau"] == 0.5 and written["refit"]["recipe"]["rbf_distances"] == "raw"
    assert set(written["refit"]) >= {"recipe", "tau", "seeds", "refit_unit_ids", "s4_record_sha256"}
    assert receipt["fit_chain"] == ["a" * 64, "b" * 64]


def test_the_runner_registers_the_record_actions():
    import importlib.util

    spec = importlib.util.spec_from_file_location("amendment2_record_runner", ROOT / "scripts" /
                                                  "run_ec_clean30_campaign.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    parser = runner.build_parser()
    assert parser.parse_args(["amend-2", "--draft", "d", "--decisions", "c", "--amendment-draft-md", "m", "--out",
                              "o"]).handler is amendment2.action_amend2
    with pytest.raises(SystemExit):
        parser.parse_args(["amend-2", "--draft", "d", "--decisions", "c", "--amendment-md", "m", "--out", "o"])
    chain = parser.parse_args(["verify-amendment-chain", "--tip", "t", "--freeze1-sha256", FREEZE1, "--out", "o",
                               "--expected-chain", "a" * 64, "--expected-chain", "b" * 64])
    assert chain.expected_chain == ["a" * 64, "b" * 64]
    assert parser.parse_args(["refit-recipe", "--fold", "1", "--step-chain", "c", "--freeze1-sha256", FREEZE1,
                              "--freeze1-amendment", "a", "--campaign-root", "r", "--plan", "p", "--out", "o",
                              "--amendment-chain-receipt", "x"]).amendment_chain_receipt == "x"
    assert parser.parse_args(["verify-amendment-chain", "--tip", "t", "--freeze1-sha256", FREEZE1, "--out",
                              "o"]).handler is amendment2.action_verify_amendment_chain
    assert parser.parse_args(["dry-run-v2", "--view", "v", "--out", "o"]).handler is amendment2.action_dry_run_v2
    assert parser.parse_args(["refit-recipe", "--fold", "1", "--step-chain", "c", "--freeze1-sha256", FREEZE1,
                              "--freeze1-amendment", "a", "--campaign-root", "r", "--plan", "p",
                              "--out", "o"]).handler is amendment2.action_refit_recipe
    assert parser.parse_args(["futility-oc", "--out", "o"]).handler.__name__ == "action_futility_oc"
    for argv in (["pretrain-corpus", "build", "--campaign-root", "r", "--fold", "0", "--official-split-dir", "d",
                  "--identity-work", "w"], ["pretrain-corpus", "fasta", "--campaign-root", "r",
                                            "--official-split-dir", "d"]):
        assert parser.parse_args(argv).handler.__name__ == "action_pretrain_corpus"
    assert isinstance(parser, argparse.ArgumentParser)
