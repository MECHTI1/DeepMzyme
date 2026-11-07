"""Link 2 built by the real record builder and judged by the real chain verifier.

The synthetic published repository is the harness of ``test_ec_clean30_amendment_chain``, with the execution files
first added at the amendment-2 source commit, as in the campaign repository. ``amendment2.build_amendment2`` runs in
that repository's own interpreter from a synthetic draft, decisions file, plan-v2 records and dry-run-v2 receipts; its
output is rendered by ``amendment2.render_amendment2``, committed and published, and ``amendment_chain.verify_chain``
accepts it. Expected shapes and names come from the contract fixture only. One test reads the staged review draft
and decisions file of the campaign (no campaign record, view or data folder).
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import ec_clean30_amendment2_build as build  # noqa: E402
import test_ec_clean30_amendment_chain as harness  # noqa: E402
from ec_clean30 import amendment2, amendment_chain, freeze  # noqa: E402
from ec_clean30 import amendment_rules as rules  # noqa: E402
from ec_clean30.amendment2 import AmendmentRecordError  # noqa: E402
from ec_clean30.amendment_chain import AmendmentChainError  # noqa: E402

CONTRACT, SHAPE, BUILDER = build.CONTRACT, build.SHAPE, build.BUILDER
REQUIRED = CONTRACT["amendment_record"]["required_keys"]
MISSING_INPUTS = ("Build inputs are missing: plans.v2_plan_sha256 needs the plan-v2 records of folds {plans}; "
                  "dry_runs.source_commit and dry_runs.receipts need the dry-run-v2 receipts of folds {receipts}. "
                  "They are written by plan-v2 and dry-run-v2 before the record build and are never fabricated.")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build.build_world_with_builder(tmp_path_factory.mktemp("built_chain") / "world")


def _publish_record(world: dict, record: dict) -> None:
    repo = world["repo"]
    (repo / rules.AMENDMENT2_PATH).write_bytes(freeze.canonical_bytes(record))
    (repo / rules.AMENDMENT2_PATH).with_suffix(".md").write_text(amendment2.render_amendment2(record),
                                                                  encoding="utf-8")
    build.publish_link2(world, "Link 2 without a required key")


# ---------------------------------------------------------------------------
# Both sides name the fixture's record shape


def test_the_builder_and_the_verifier_name_the_fixtures_record_shape():
    fields = SHAPE["fields"]
    assert set(fields) == {*REQUIRED, *SHAPE["built_keys"], "campaign_id"}
    # The builder.
    assert list(amendment2.RECORD_REQUIRED_KEYS) == REQUIRED
    assert list(amendment2.RECORD_BUILT_KEYS) == SHAPE["built_keys"]
    assert list(amendment2.FILE_BINDING_SECTIONS) == SHAPE["file_binding_sections"]
    assert amendment2.RECORD_FORMAT == fields["format"]["const"] and amendment2.DRAFT_FORMAT == BUILDER["draft_format"]
    assert list(amendment2.DRAFT_KEYS) == BUILDER["draft_keys"]
    assert {key: list(value) for key, value in amendment2.DRAFT_SUBKEYS.items()} == BUILDER["draft_subkeys"]
    assert {key: f"ec_clean30.amendment_rules:{name}" for key, name in amendment2.DRAFT_FUNCTION_NAMES.items()} == \
        BUILDER["draft_function_names"]
    assert list(amendment2.REMOVED_DRAFT_KEYS) == BUILDER["removed_draft_keys"]
    assert amendment2.RULE_TEXT_SOURCE == BUILDER["rule_text_source"] == rules.rule_text_source(2)
    assert amendment2.RENDERING_PATH == BUILDER["rendering"] == rules.rendering_path(rules.AMENDMENT2_PATH)
    assert BUILDER["rule_text_source"] != BUILDER["rendering"]
    assert [amendment2.DECISIONS_BINDING] == list(fields["bound_files"]["keys"])
    module, name = BUILDER["function"].split(":")
    assert module == amendment2.__name__ and getattr(amendment2, name) is amendment2.build_amendment2
    parameters = inspect.signature(amendment2.build_amendment2).parameters
    assert [key for key, item in parameters.items() if item.kind is item.KEYWORD_ONLY] == BUILDER["keyword_arguments"]
    # The verifier.
    assert list(amendment_chain.LINK_REQUIRED_KEYS) == REQUIRED
    assert list(amendment_chain.BUILT_RECORD_KEYS) == SHAPE["built_keys"]
    assert list(amendment_chain.EVIDENCE_SECTIONS) == SHAPE["file_binding_sections"]
    assert dict(amendment_chain.LINK2_DATA_STATE) == SHAPE["link2"]["data_state"]
    assert sorted(amendment_chain.FIXTURE_ROLES) == sorted(fields["fixtures"]["keys"])
    assert [amendment_chain.DECISIONS_BINDING] == list(fields["bound_files"]["keys"])
    assert amendment_chain.PUBLICATION == fields["publication"]["const"]
    assert amendment_chain.CONFIRMED == fields["confirmation_status"]["const"]
    later = SHAPE["later_links"]
    assert amendment_chain.FUT_KIND == later["fut_kind"]
    assert [amendment_chain.FUTILITY_CONTINUE, amendment_chain.FUTILITY_STOP] == later["fut_decisions"]
    # The shared rules.
    assert dict(rules.REQUIRED_BINDINGS) == SHAPE["required_bindings"]["roles"]
    assert dict(rules.BINDING_SHAPES) == SHAPE["required_bindings"]["shapes"]
    assert list(rules.PIN_KEEPING_CLASSES) == later["pin_keeping_classes"]
    assert rules.FREEZE2_PATH == later["analysis_completion"]["freeze2_path"]
    assert rules.DECISION_FIXTURE_TEST == later["analysis_completion"]["decision_fixture_test"]
    assert list(amendment_chain.FIXTURE_RERUN_SPARSE_PATHS) == later["analysis_completion"]["checkout_extra_paths"]
    assert list(rules.LATER_LINK_REQUIRED_FIELDS) == later["required_fields"]
    assert rules.CAMPAIGN_ID == fields["campaign_id"]["const"] and rules.AMENDMENT2_FORMAT == fields["format"]["const"]
    assert rules.R_CMD_V2_RULE == fields["dry_runs"]["keys"]["rule"]["const"]


# ---------------------------------------------------------------------------
# Link 2 by the real builder, accepted by the real verifier


def test_the_builder_writes_the_contract_record_shape_from_the_draft_and_the_live_tree(built):
    draft = json.loads(built["inputs"]["draft"].read_text())
    record = built["record"]
    assert build.draft_problems(draft) == []
    assert build.record_shape_problems(record) == []
    assert (built["repo"] / rules.AMENDMENT2_PATH).read_bytes() == freeze.canonical_bytes(record)
    # Computed at build from the live tree, never taken from the draft.
    assert record["execution_file_set"]["files"] == sorted(harness.EXECUTION_FILES)
    assert record["execution_file_set"]["names_sha256"] == rules.sha256_canonical(sorted(harness.EXECUTION_FILES))
    assert record["pinned_execution_functions"] == rules.pinned_function_hashes(built["repo"],
                                                                                rules.PINNED_EXECUTION_FUNCTIONS)
    assert record["pinned_analysis_functions"] == rules.pinned_function_hashes(built["repo"],
                                                                               rules.PINNED_ANALYSIS_FUNCTIONS)
    assert record["live_constants_sha256"] == built["values"]["live_constants_sha256"]
    assert record["rule_policy_sha256"] == built["values"]["rule_policy_sha256"]
    # The build inputs.
    assert record["plans"]["v2_plan_sha256"] == built["values"]["v2"]
    assert record["dry_runs"]["source_commit"] == built["commits"]["D"]
    assert record["dry_runs"]["receipts"] == {fold: {"path": build.RECEIPT_PATH.format(fold=fold),
                                                     "sha256": harness._file_sha(built["repo"],
                                                                                 build.RECEIPT_PATH.format(fold=fold))}
                                              for fold in build.FOLDS}
    assert record["bound_files"] == {"decisions": {"path": build.DECISIONS_PATH,
                                                   "sha256": harness._file_sha(built["repo"], build.DECISIONS_PATH)}}
    assert not set(BUILDER["removed_draft_keys"]) & set(record)
    # Every required role is bound at its record path (the draft held notes there), the rule text comes from the
    # reviewed document and the rendering is the other file.
    assert rules.required_binding_problems(record) == []
    assert record["inputs"]["f6_holdout"]["3"] == {"path": f"{harness.DOCS}/amendment2/f6_holdout/fold3.tsv",
                                                   "sha256": harness._file_sha(
                                                       built["repo"], f"{harness.DOCS}/amendment2/f6_holdout/fold3.tsv")}
    assert record["rule_text"]["source"] == BUILDER["rule_text_source"]
    assert record["execution_file_set"]["files"] == draft["execution_file_set"]["declared"]
    rendering = (built["repo"] / BUILDER["rendering"]).read_text()
    assert "**A2-X-1** A synthetic rule." in rendering and "## Policy deviations" not in rendering
    assert "| C6b | the recommended default | yes |" in rendering


def test_verify_chain_accepts_link_2_built_by_the_real_builder(built):
    verified = harness.verify(built)
    record = built["record"]
    assert set(verified) == set(CONTRACT["verifier"]["returns"])
    assert verified["chain"] == [built["a1"], built["a2"]] == verified["fit_chain"]
    assert verified["tip_sequence"] == 2 and verified["published"] is True
    assert [link["source_commit"] for link in verified["links"]] == [built["commits"]["B"], built["commits"]["S2"]]
    assert verified["execution_file_set"] == record["execution_file_set"]["files"]
    assert verified["pinned_execution_functions"] == record["pinned_execution_functions"]
    assert verified["pinned_analysis_functions"] == record["pinned_analysis_functions"]
    assert verified["plan_records"] == {"v1": record["plans"]["v1_plan_sha256"],
                                        "v2": record["plans"]["v2_plan_sha256"]}
    assert verified["dry_run_evidence"]["receipts"] == record["dry_runs"]["receipts"]
    assert verified["dry_run_evidence"]["source_commit"] == built["commits"]["D"]
    assert len(verified["scientific_identity_sha256"]) == 64


def test_the_builder_refuses_an_absent_required_binding_an_undeclared_execution_file_and_another_rule_text(
        built, tmp_path_factory):
    copied = harness.copy_world(built, tmp_path_factory)
    copied["inputs"] = {key: (Path(str(value).replace(str(built["root"]), str(copied["root"])))
                              if isinstance(value, Path) else
                              {name: Path(str(path).replace(str(built["root"]), str(copied["root"])))
                               for name, path in value.items()})
                        for key, value in built["inputs"].items()}
    out = copied["root"] / "scratch" / "refused.json"
    for role in rules.REQUIRED_BINDINGS:
        error = _refused(copied, out, drop_binds=(role,))
        assert "Required bindings are absent or malformed" in error and role in error, role
    error = _refused(copied, out, bind={"inputs.unknown_role": copied["inputs"]["decisions"]})
    assert "names no role of amendment_rules.REQUIRED_BINDINGS" in error
    # A tracked file outside the declared closed list is never taken as an execution file.
    harness._write(copied["repo"], "src/data_structures_extra.py", "VALUE = 1\n")
    harness._git(copied["repo"], "add", "src/data_structures_extra.py")
    error = _refused(copied, out)
    assert "differs from the draft's declared closed list" in error and "src/data_structures_extra.py" in error
    harness._git(copied["repo"], "rm", "--quiet", "--cached", "src/data_structures_extra.py")
    (copied["repo"] / "src/data_structures_extra.py").unlink()
    draft = json.loads(copied["inputs"]["draft"].read_text())
    pending = copied["root"] / "drafts" / "pending_execution.json"
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps({**draft, "execution_file_set": {**draft["execution_file_set"],
                                                                   "pending": ["src/ec_clean30/execution_v5.py"]}}))
    assert "Execution files are still pending: ['src/ec_clean30/execution_v5.py']" in _refused(copied, out,
                                                                                               draft=pending)
    other = copied["repo"] / harness.DOCS / "amendment2" / "review_copy.md"
    other.write_text(copied["inputs"]["amendment_draft_md"].read_text())
    error = _refused(copied, out, amendment_draft_md=other)
    assert "The rule text is read from the reviewed document" in error


@pytest.mark.parametrize("key", [*REQUIRED, *SHAPE["built_keys"]])
def test_a_builder_output_without_a_required_key_has_the_wrong_link_form(built, key):
    record = {name: value for name, value in built["record"].items() if name != key}
    assert any(problem == f"[C1] link 2 lacks {[key]}" for problem in amendment_chain.link_header_problems(record, 2))


def test_a_published_builder_output_without_one_required_key_is_refused(built, tmp_path_factory):
    copied = harness.copy_world(built, tmp_path_factory)
    _publish_record(copied, {key: value for key, value in built["record"].items()
                             if key != "unchanged_scientific_files"})
    with pytest.raises(AmendmentChainError, match=r"\[C1\] link 2 lacks \['unchanged_scientific_files'\]"):
        harness.verify(copied)


# ---------------------------------------------------------------------------
# Refusals of the builder


def _refused(world: dict, out: Path, **overrides) -> str:
    result = build.run_builder(world, out=out, **overrides)
    assert "error" in result and not out.exists(), result
    return result["error"]


def test_the_builder_refuses_missing_build_inputs_and_never_fabricates_them(built):
    out = built["root"] / "scratch" / "missing.json"
    error = _refused(built, out, plan_v2=None, dry_run_receipts=None)
    assert error == "AmendmentRecordError: " + MISSING_INPUTS.format(plans=[0, 1, 2, 3, 4], receipts=[0, 1, 2, 3, 4])
    plans = {fold: path for fold, path in built["inputs"]["plan_v2"].items() if fold != "2"}
    error = _refused(built, out, plan_v2=plans)
    assert error == ("AmendmentRecordError: Build inputs are missing: plans.v2_plan_sha256 needs the plan-v2 records "
                     "of folds [2]. They are written by plan-v2 and dry-run-v2 before the record build and are never "
                     "fabricated.")
    receipts = {fold: path for fold, path in built["inputs"]["dry_run_receipts"].items() if fold != "4"}
    assert "dry-run-v2 receipts of folds [4]" in _refused(built, out, dry_run_receipts=receipts)


def test_the_builder_refuses_build_inputs_that_do_not_check(built, tmp_path_factory):
    copied = harness.copy_world(built, tmp_path_factory)
    copied["inputs"] = {key: (Path(str(value).replace(str(built["root"]), str(copied["root"])))
                              if isinstance(value, Path) else
                              {fold: Path(str(path).replace(str(built["root"]), str(copied["root"])))
                               for fold, path in value.items()})
                        for key, value in built["inputs"].items()}
    out = copied["root"] / "scratch" / "refused.json"
    # A plan-v2 record that is not the live plan of its fold.
    plan = json.loads(copied["inputs"]["plan_v2"]["1"].read_text())
    other = copied["root"] / "campaign" / "other_fold1.json"
    other.write_text(json.dumps({**plan, "plan_sha256": "0" * 64}))
    error = _refused(copied, out, plan_v2={**copied["inputs"]["plan_v2"], "1": other})
    assert "differs from campaign.build_fold_plan_v2(1)" in error
    # Dry-run receipts of two source commits, and a receipt outside the repository.
    receipt = json.loads(copied["inputs"]["dry_run_receipts"]["3"].read_text())
    receipt["provenance"]["git"]["head"] = copied["commits"]["B"]
    moved = copied["repo"] / build.RECEIPT_PATH.format(fold="3x")
    moved.write_text(json.dumps(receipt))
    error = _refused(copied, out, dry_run_receipts={**copied["inputs"]["dry_run_receipts"], "3": moved})
    assert "receipts name 2 source commits" in error
    outside = copied["root"] / "campaign" / "fold3.json"
    outside.write_text(copied["inputs"]["dry_run_receipts"]["3"].read_text())
    error = _refused(copied, out, dry_run_receipts={**copied["inputs"]["dry_run_receipts"], "3": outside})
    assert "receipt of fold 3 is outside the repository" in error
    # A record that exists already is never rewritten.
    existing = copied["repo"] / rules.AMENDMENT2_PATH
    before = existing.read_bytes()
    assert build.run_builder(copied, out=existing)["error"].startswith("FileExistsError")
    assert existing.read_bytes() == before


def test_the_builder_refuses_a_draft_without_its_shape_or_with_unregistered_function_names(built):
    out = built["root"] / "scratch" / "draft.json"
    draft = json.loads(built["inputs"]["draft"].read_text())
    renamed = {key: value for key, value in draft.items() if key != "unchanged_scientific_files"}
    renamed["unchanged_scientific_files_explicit"] = draft["unchanged_scientific_files"]
    path = built["root"] / "drafts" / "renamed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(renamed))
    assert _refused(built, out, draft=path) == ("AmendmentRecordError: The draft record does not have the contract's "
                                                "shape: unchanged_scientific_files is missing.")
    hashes = dict(draft, pinned_execution_functions=built["record"]["pinned_execution_functions"])
    path.write_text(json.dumps(hashes))
    assert "pinned_execution_functions is not a map of file paths to top-level function names" in _refused(
        built, out, draft=path)
    replay = CONTRACT["execution_api"]["cpu_replay"]
    assert draft["pinned_execution_functions"][replay["path"]] == [replay["qualname"]]  # registered and pinned
    without_replay = dict(draft, pinned_execution_functions={key: value for key, value in
                                                             draft["pinned_execution_functions"].items()
                                                             if key != replay["path"]})
    path.write_text(json.dumps(without_replay))
    error = _refused(built, out, draft=path)
    assert "differ from amendment_rules.PINNED_EXECUTION_FUNCTIONS" in error
    assert f"{replay['path']}::{replay['qualname']}" in error


# ---------------------------------------------------------------------------
# The staged review draft of the campaign


def test_the_staged_draft_is_refused_only_for_its_missing_build_inputs(tmp_path):
    draft_path = ROOT / build.DRAFT_PATH
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    assert build.draft_problems(draft) == [] and amendment2.draft_shape_problems(draft) == []
    out = tmp_path / "amendment.json"
    with pytest.raises(AmendmentRecordError) as refused:
        amendment2.build_amendment2(draft_path=draft_path, decisions_path=ROOT / build.DECISIONS_PATH,
                                    amendment_draft_md=ROOT / BUILDER["rule_text_source"], repo=ROOT,
                                    bound_files={}, out=out)
    assert str(refused.value) == MISSING_INPUTS.format(plans=[0, 1, 2, 3, 4], receipts=[0, 1, 2, 3, 4])
    assert not out.exists()
    # Ownership: the execution side writes exactly EXECUTION_WRITTEN_SCIENTIFIC_FILES; the science side writes every
    # other scientific file, the chain verifier included, which stays scientific.
    writers = {path: item["written_by"] for section in ("source_changes", "source_changes_pending")
               for path, item in draft[section].items()}
    assert {path for path, writer in writers.items() if writer != "science"} == set(
        rules.EXECUTION_WRITTEN_SCIENTIFIC_FILES)
    assert writers["src/ec_clean30/amendment_chain.py"] == "science"
    assert "src/ec_clean30/amendment_chain.py" in rules.FORBIDDEN_EXECUTION_FILES
    # The declared closed execution list: the execution-v4 files now, the execution-v5 files pending; no declared
    # file has a scientific role, and every required role still holds its note until the build binds it.
    declared = draft["execution_file_set"]
    assert declared["declared"] == sorted(set(declared["declared"])) and not set(declared["declared"]) & set(
        declared["pending"])
    assert all(rules.execution_file_problems(path) == [] for path in [*declared["declared"], *declared["pending"]])
    assert "src/ec_clean30/runtime_precision.py" not in [*declared["declared"], *declared["pending"]]
    bound_now = {"evidence.interface_contract", "evidence.power_simulation"}
    unbound = {problem.split(" ")[0].split(".tables")[0].split(".union_fasta")[0]
               for problem in rules.required_binding_problems(draft)}
    assert unbound == set(rules.REQUIRED_BINDINGS) - bound_now
    for role in sorted(bound_now):
        binding = rules.binding_at(draft, role)
        assert rules.is_file_binding(binding) and harness._file_sha(ROOT, binding["path"]) == binding["sha256"], role
