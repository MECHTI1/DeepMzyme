"""End-to-end refusals of the amendment-chain verifier on synthetic published repositories.

Every scenario runs the real verifier of the synthetic repository (``test_ec_clean30_amendment_chain`` harness: a
history with execution commits and a merge between link 1 and amendment 2, filled evidence, inputs and required
bindings, synthetic refusal and decision-fixture tests, bound futility pointer files). Covered: rewritten and forked
links and stale tips (C3), a lenient verifier at a link's commit and foreign verifier code (C4), the pod's blobless
partial clone and the chain receipt (C12, receipts), exported ``GIT_*`` variables, the class-(a2) limits with their
bound futility files and fixture reruns (C16), frozen gates in execution links, private names, the required bindings
and the decisions file (C8), policies (C9), scope (C5), dry runs (C11), the import closure (C7), time budgets and
process groups, and a sequence-3 link of each class, accepted from a later HEAD. No campaign record, view or data
folder is read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import test_ec_clean30_amendment_chain as harness  # noqa: E402
from ec_clean30 import amendment2, amendment_chain, freeze  # noqa: E402
from ec_clean30 import amendment_rules as rules  # noqa: E402
from ec_clean30.amendment_chain import AmendmentChainError  # noqa: E402

DOCS = harness.DOCS
WORKSTATION = "src/ec_clean30/workstation.py"
EXECUTION_TEXT = '"""Synthetic execution file."""\nHOST = "later"\n'


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return harness.build_world(tmp_path_factory.mktemp("attack_world") / "world")


@pytest.fixture(scope="module")
def base_verified(base):
    return harness.verify(base)


def copy(world: dict, factory) -> dict:
    return harness.copy_world(world, factory)


def _state(repo: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()

    return {"head": git("rev-parse", "HEAD"), "symbolic": git("symbolic-ref", "-q", "HEAD"),
            "sparse": git("config", "--get", "core.sparseCheckout"), "status": git("status", "--porcelain"),
            "sparse_file": (repo / ".git" / "info" / "sparse-checkout").exists()}


# ---------------------------------------------------------------------------
# Blocking 1: a published link is written once


def _rewrite_link2(world: dict, *, replace_log_line: bool) -> str:
    repo = world["repo"]
    record = json.loads(json.dumps(world["record"]))
    record["timestamps"] = {"draft_written": "rewritten after G2 data"}
    record["deferred_bindings"] = {"f6_inputs": {"bound_by": "freeze1c_v2", "spec": "A2-F6-4 (changed post hoc)"}}
    (repo / rules.AMENDMENT2_PATH).write_bytes(freeze.canonical_bytes(record))
    (repo / rules.rendering_path(rules.AMENDMENT2_PATH)).write_text(amendment2.render_amendment2(record),
                                                                     encoding="utf-8")
    digest = harness._file_sha(repo, rules.AMENDMENT2_PATH)
    log = repo / freeze.LOG_PATH
    if replace_log_line:
        log.write_text(log.read_text().replace(world["a2"], digest), encoding="utf-8")
    else:
        harness.log_link(repo, 2, digest)
    harness._commit(repo, "Rewrite link 2 after publication")
    return digest


def test_a_rewritten_link_2_is_refused_in_place_and_from_a_later_head(base, tmp_path_factory):
    logged_twice = copy(base, tmp_path_factory)
    digest = _rewrite_link2(logged_twice, replace_log_line=False)
    with pytest.raises(AmendmentChainError, match=r"link 2 at .* was rewritten or deleted"):
        harness.verify(logged_twice)
    # The log rule refuses the same rewrite on its own: two lines of sequence 2.
    log = (logged_twice["repo"] / freeze.LOG_PATH).read_text()
    assert any("names 2 links of sequence 2 (a fork or a rewrite)" in problem
               for problem in rules.link_log_problems(log, [logged_twice["a1"], digest]))
    relogged = copy(base, tmp_path_factory)
    _rewrite_link2(relogged, replace_log_line=True)
    with pytest.raises(AmendmentChainError, match=r"link 2 at .* was rewritten or deleted"):
        harness.verify(relogged)
    harness.commit_change(relogged, WORKSTATION, EXECUTION_TEXT)
    with pytest.raises(AmendmentChainError, match=r"was rewritten or deleted"):
        harness.verify(relogged)


def test_a_rewrite_that_restores_the_published_bytes_is_still_refused(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    repo = copied["repo"]
    original = {path: (repo / path).read_bytes() for path in (rules.AMENDMENT2_PATH, freeze.LOG_PATH,
                                                              rules.rendering_path(rules.AMENDMENT2_PATH))}
    _rewrite_link2(copied, replace_log_line=True)
    for path, data in original.items():
        (repo / path).write_bytes(data)
    harness._commit(repo, "Restore link 2")
    with pytest.raises(AmendmentChainError, match=r"was rewritten or deleted"):
        harness.verify(copied)


def test_a_link_committed_before_its_log_line_is_refused(tmp_path_factory):
    def unlogged(repo, record):
        log = repo / freeze.LOG_PATH
        log.write_text(log.read_text().replace(f"- {rules.link_log_line(2, harness._file_sha(repo, rules.AMENDMENT2_PATH))}\n",
                                               ""), encoding="utf-8")

    world = harness.build_world(tmp_path_factory.mktemp("link_first") / "world", before_link2_commit=unlogged)
    harness.log_link(world["repo"], 2, world["a2"])
    harness._commit(world["repo"], "Log line after the link")
    for options in ({}, {"require_published": False}):
        with pytest.raises(AmendmentChainError):
            harness.verify(world, **options)


def test_a_log_line_removed_after_publication_is_refused(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    log = copied["repo"] / freeze.LOG_PATH
    line = f"- {rules.link_log_line(2, copied['a2'])}\n"
    assert line in log.read_text()
    harness.commit_change(copied, freeze.LOG_PATH, log.read_text().replace(line, ""))
    for options in ({}, {"require_published": False}):
        with pytest.raises(AmendmentChainError):
            harness.verify(copied, **options)
    harness.commit_change(copied, freeze.LOG_PATH, log.read_text() + line)
    assert harness.verify(copied)["chain"] == [copied["a1"], copied["a2"]]


def test_link_1_is_published_only_with_its_sha256_in_the_campaign_log(tmp_path_factory):
    world = harness.build_world(tmp_path_factory.mktemp("link1_unlogged") / "world", log_link1=False)
    draft = harness.verify(world, require_published=False)
    assert [link["published"] for link in draft["links"]] == [False, True] and draft["published"] is False
    with pytest.raises(AmendmentChainError):
        harness.verify(world)


def test_the_log_line_must_be_committed_with_the_link(tmp_path_factory):
    def log_first(repo, record):
        harness._git(repo, "add", freeze.LOG_PATH)
        harness._git(repo, "commit", "--quiet", "-m", "Log line before the link")

    world = harness.build_world(tmp_path_factory.mktemp("log_first") / "world", before_link2_commit=log_first)
    with pytest.raises(AmendmentChainError, match=r"commit the link, its rendering and its log line together"):
        harness.verify(world)


# ---------------------------------------------------------------------------
# Blocking 2: the nested verifier runs the pinned code; foreign code never judges a real chain


def test_a_lenient_verifier_at_the_link_commit_is_refused_after_the_pinned_file_is_restored(tmp_path_factory):
    old = '        raise AmendmentChainError(message + "\\n  " + "\\n  ".join(list(problems)[:40]))'

    def lenient(repo, record):
        path = repo / "src/ec_clean30/amendment_chain.py"
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1
        path.write_text(text.replace(old, "        return None"), encoding="utf-8")

    world = harness.build_world(tmp_path_factory.mktemp("lenient") / "world", before_link2_commit=lenient)
    repo = world["repo"]
    shutil.copyfile(ROOT / "src/ec_clean30/amendment_chain.py", repo / "src/ec_clean30/amendment_chain.py")
    harness._commit(repo, "Restore the pinned verifier")
    with pytest.raises(AmendmentChainError, match=r"(?s)\[C4\] The code at link 2's commit is not the code its "
                                                  r"records pin.*src/ec_clean30/amendment_chain.py differs"):
        harness.verify(world)


def test_link_1_is_authenticated_at_its_own_commit_against_its_pins(tmp_path_factory):
    world = harness.build_world(tmp_path_factory.mktemp("link1_drift") / "world", drift_at_link1=True)
    with pytest.raises(AmendmentChainError, match=r"(?s)\[C4\] The code at link 1's commit is not the code its "
                                                  r"records pin.*src/ec_clean30/views.py at link 1's commit differs"):
        harness.verify(world)


def test_a_real_verification_never_runs_another_repositorys_verifier_code(base):
    tip = base["repo"] / rules.AMENDMENT2_PATH
    with pytest.raises(AmendmentChainError, match=r"holds other verifier code.*amendment_rules.py"):
        amendment_chain.verify_chain(tip, expected_freeze1_sha256=base["freeze1_sha256"],
                                     expected_freeze1b_sha256=rules.FREEZE1B_SHA256, repo=base["repo"])
    draft = amendment_chain.verify_chain(tip, expected_freeze1_sha256=base["freeze1_sha256"],
                                         expected_freeze1b_sha256=rules.FREEZE1B_SHA256, repo=base["repo"],
                                         require_published=False)
    assert draft["chain"] == [base["a1"], base["a2"]]


_NO_IDENTITY = '''
amendment_chain.rules.scientific_identity = lambda record: None
try:
    verified = amendment_chain.verify_chain(args["tip"], expected_freeze1_sha256=args["freeze1"],
                                            expected_freeze1b_sha256=rules.FREEZE1B_SHA256, repo=args["repo"])
    result = {"accepted": verified["scientific_identity_sha256"]}
except amendment_chain.AmendmentChainError:
    result = "refused"
'''


def test_verify_chain_never_returns_a_null_scientific_identity_for_a_tip_of_two_or_more(base, base_verified):
    """Fault injection: the identity computation yields nothing; the published two-link chain is refused, in place
    (HEAD at link 2's commit) and from a later HEAD."""
    assert len(base_verified["scientific_identity_sha256"]) == 64
    tip = str(base["repo"] / rules.AMENDMENT2_PATH)
    assert harness.run_world(base, "exec", code=_NO_IDENTITY, tip=tip) == "refused"


def _identityless_verifier(repo, record):
    path = repo / "src/ec_clean30/amendment_chain.py"
    text = path.read_text(encoding="utf-8")
    for old, new in (('        raise AmendmentChainError(message + "\\n  " + "\\n  ".join(list(problems)[:40]))',
                      "        return None"),
                     ('        _require(_is_sha256(state.get("scientific_identity_sha256")),', "        _require(True,")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def test_a_delegated_verifier_that_returns_no_scientific_identity_is_refused_even_in_draft_review(
        tmp_path_factory):
    world = harness.build_world(tmp_path_factory.mktemp("no_identity") / "world",
                                before_link2_commit=_identityless_verifier)
    tip = world["repo"] / rules.AMENDMENT2_PATH
    draft = harness.verify(world, require_published=False)
    assert draft["tip_sequence"] == 2 and draft["scientific_identity_sha256"] is None
    for require_published in (True, False):
        with pytest.raises(AmendmentChainError):
            amendment_chain.verify_chain(tip, expected_freeze1_sha256=world["freeze1_sha256"],
                                         expected_freeze1b_sha256=rules.FREEZE1B_SHA256, repo=world["repo"],
                                         require_published=require_published)


_CROSS_CHECK = '''
from pathlib import Path
repo = Path(args["repo"])
links = amendment_chain._collect_links(repo, args["tip"])
freeze1 = amendment_chain._freeze1(repo, rules.FREEZE1_SHA256)
state = args["state"]
amendment_chain._cross_check(state, links, freeze1)
result = []
for key, value in args["tampers"]:
    try:
        amendment_chain._cross_check({**state, key: value}, links, freeze1)
        result.append([key, "accepted"])
    except amendment_chain.AmendmentChainError as exc:
        result.append([key, str(exc)])
'''


def test_the_nested_result_is_cross_checked_against_the_records(base, base_verified):
    state = {key: value for key, value in base_verified.items() if key not in ("links", "published")}
    policies = json.loads(json.dumps(state["policies"]))
    policies["controls_v2"]["policy"] = {"rule": "relaxed controls"}
    dry = json.loads(json.dumps(state["dry_run_evidence"]))
    dry["source_commit"] = base["commits"]["B"]
    tampers = [
        ["fit_chain", [base["a1"]]], ["composite_scientific_pins", {}], ["analysis_pins", {}],
        ["execution_file_set", [harness.EXECUTION_FILES[0]]], ["plan_records", {"v1": {}, "v2": {}}],
        ["post_fit_analysis_files", []], ["pinned_execution_functions", {}], ["pinned_analysis_functions", {}],
        ["dry_run_evidence", dry], ["policies", policies], ["scientific_identity_sha256", None],
        ["scientific_identity_sha256", "f" * 64],
    ]
    result = harness.run_world(base, "exec", code=_CROSS_CHECK, tip=rules.AMENDMENT2_PATH, state=state,
                               tampers=tampers)
    assert [key for key, _message in result] == [key for key, _value in tampers]
    assert all(message != "accepted" and ("[C4]" in message or "[C14]" in message)
               for _key, message in result), result


# ---------------------------------------------------------------------------
# Blocking 3 and should-fix 10: the pod's partial clone and the chain receipt


def test_verify_chain_refuses_a_partial_clone_early_and_names_the_receipt_check(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT)
    pod = harness.partial_clone(copied, "pod")
    with pytest.raises(AmendmentChainError, match=r"partial clone.*check_chain_receipt"):
        harness.verify(copied, repo=pod)
    assert amendment_chain.partial_clone_problem(copied["repo"]) is None
    assert amendment_chain.partial_clone_problem(pod)


_TIMED_CHECK = '''
import time
times = []
for _ in range(args["repeats"]):
    started = time.perf_counter()
    checked = amendment_chain.check_chain_receipt(args["receipt"], repo=args["repo"])
    times.append(time.perf_counter() - started)
result = {"seconds": sorted(times), "chain": checked["chain"]}
'''


def test_a_receipt_written_on_the_full_clone_checks_in_a_pod_partial_clone_and_offline(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT)
    out = copied["root"] / "receipt.json"
    receipt = harness.write_receipt(copied, out, expected_chain=[copied["a1"], copied["a2"]])
    assert receipt["key"]["head"] == harness._git(copied["repo"], "rev-parse", "HEAD")
    assert receipt["key"]["expected_chain"] == [copied["a1"], copied["a2"]]
    assert receipt["require_published"] is True and receipt["draft"] is False
    assert set(receipt["key"]) == {"tip_sha256", "head", "status_sha256", "origin_main", "code_sha256",
                                   "require_published", "expected_chain"}
    verified = harness.verify(copied)
    for repo in (copied["repo"], harness.partial_clone(copied, "pod"),
                 harness.partial_clone(copied, "pod_offline", offline=True)):
        checked = harness.check_receipt(copied, out, repo=repo, expected_chain=[copied["a1"], copied["a2"]])
        assert checked == {key: verified[key] for key in amendment_chain.RESULT_KEYS} | {"links": checked["links"]}
        assert [link["sha256"] for link in checked["links"]] == verified["chain"]
    pod = copied["root"] / "pod_offline"
    timed = harness.run_world(copied, "exec", repo=pod, code=_TIMED_CHECK, receipt=str(out), repeats=5)
    assert timed["chain"] == verified["chain"] and timed["seconds"][0] < 2.0


def test_the_receipt_check_refuses_drafts_edits_other_chains_moved_heads_and_dirty_trees(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    repo = copied["repo"]
    draft = harness.write_receipt(copied, copied["root"] / "draft.json", require_published=False)
    assert draft["draft"] is True
    with pytest.raises(AmendmentChainError, match=r"made without require_published"):
        harness.check_receipt(copied, copied["root"] / "draft.json")
    out = copied["root"] / "receipt.json"
    receipt = harness.write_receipt(copied, out)
    edited = json.loads(json.dumps(receipt))
    edited["result"]["fit_chain"] = [copied["a1"]]
    (copied["root"] / "edited.json").write_text(json.dumps(edited))
    with pytest.raises(AmendmentChainError, match=r"does not hash to its receipt_sha256"):
        harness.check_receipt(copied, copied["root"] / "edited.json")
    # An edited receipt with a recomputed self-hash is caught by the live re-check.
    forged = json.loads(json.dumps(receipt))
    forged["result"]["composite_scientific_pins"]["src/ec_clean30/selection.py"] = "0" * 64
    del forged["receipt_sha256"]
    forged["receipt_sha256"] = rules.sha256_canonical(forged)
    (copied["root"] / "forged.json").write_text(json.dumps(forged))
    with pytest.raises(AmendmentChainError, match=r"\[C6\] the scientific file src/ec_clean30/selection.py differs"):
        harness.check_receipt(copied, copied["root"] / "forged.json")
    with pytest.raises(AmendmentChainError, match=r"differs from the expected chain"):
        harness.check_receipt(copied, out, expected_chain=[copied["a1"], "f" * 64])
    # A re-hashed receipt without a scientific identity.
    identityless = json.loads(json.dumps(receipt))
    identityless["result"]["scientific_identity_sha256"] = identityless["scientific_identity_sha256"] = None
    del identityless["receipt_sha256"]
    identityless["receipt_sha256"] = rules.sha256_canonical(identityless)
    (copied["root"] / "identityless.json").write_text(json.dumps(identityless))
    with pytest.raises(AmendmentChainError):
        harness.check_receipt(copied, copied["root"] / "identityless.json")
    assert harness.check_receipt(copied, out)["scientific_identity_sha256"] == receipt["scientific_identity_sha256"]
    # A published execution-only descendant is accepted; a dirty tree, an unpublished or a non-execution change is not.
    harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT)
    assert harness.check_receipt(copied, out)["chain"] == [copied["a1"], copied["a2"]]
    (repo / WORKSTATION).write_text(EXECUTION_TEXT + "DIRTY = True\n")
    with pytest.raises(AmendmentChainError, match=r"the tree is not clean"):
        harness.check_receipt(copied, out)
    harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT + "DIRTY = True\n", push=False)
    with pytest.raises(AmendmentChainError, match=r"HEAD is not published"):
        harness.check_receipt(copied, out)
    harness._git(repo, "push", "--quiet", "origin", "main")
    receipt_path = receipt["result"]["dry_run_evidence"]["receipts"]["0"]["path"]
    harness.commit_change(copied, receipt_path, (repo / receipt_path).read_text() + " ")
    with pytest.raises(AmendmentChainError, match=r"changed after the receipt and is not an execution file"):
        harness.check_receipt(copied, out)


# ---------------------------------------------------------------------------
# Should-fix 4: exported GIT_* variables


def test_exported_git_variables_never_reach_the_verified_repository(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT)
    repo = copied["repo"]
    before = _state(repo)
    env = {"GIT_DIR": str(repo / ".git"), "GIT_WORK_TREE": str(repo), "GIT_INDEX_FILE": str(repo / ".git" / "index"),
           "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": "/nonexistent"}
    result = harness.verify(copied, env=env)
    assert result["chain"] == [copied["a1"], copied["a2"]]
    assert _state(repo) == before and before["sparse"] == "" and not before["sparse_file"]


# ---------------------------------------------------------------------------
# Should-fix 5: one path per sequence, the tip is the latest logged link


def test_a_second_link_of_one_sequence_and_a_stale_tip_are_refused(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    repo = copied["repo"]
    record = harness.link_record(copied, sequence=3, change_class=rules.CHANGE_EXECUTION,
                                 files={WORKSTATION: EXECUTION_TEXT})
    path, digest, _commit = harness.publish_link(copied, record)
    assert harness.verify(copied, tip=path)["chain"] == [copied["a1"], copied["a2"], digest]
    with pytest.raises(AmendmentChainError, match=r"a link of sequence 3 beyond the tip of sequence 2 \(a stale tip\)"):
        harness.verify(copied)
    fork = {**record, "timestamps": {"written": "a fork"}}
    fork_path = f"{DOCS}/amendment3_fork/amendment.json"
    harness._write(repo, fork_path, freeze.canonical_bytes(fork).decode())
    harness.log_link(repo, 3, harness._file_sha(repo, fork_path))
    harness._commit(repo, "A second link of sequence 3")
    with pytest.raises(AmendmentChainError, match=r"\[C1\] Link 3 must be .*amendment3/amendment.json"):
        harness.verify(copied, tip=fork_path)
    with pytest.raises(AmendmentChainError, match=r"names 2 links of sequence 3 \(a fork or a rewrite\)"):
        harness.verify(copied, tip=path)


# ---------------------------------------------------------------------------
# Should-fix 6: class (a2) rests on bound files, a fresh policy recomputation and fixture reruns


def test_an_analysis_completion_needs_bound_futility_files(base, tmp_path_factory):
    scoring = "src/ec_clean30/scoring.py"
    for fold0, message in (
            ([{"kind": "FUT", "sha256": "a" * 64, "decision": "continue"}], r"binds no pointer file"),
            ([{"kind": "FUT", "sha256": "a" * 64, "decision": "continue",
               "pointer": {"path": f"{DOCS}/amendment3/fold0/fut_seq2.json", "sha256": "b" * 64},
               "record": {"path": f"{DOCS}/amendment3/fold0/futility_record.json", "sha256": "a" * 64}}],
             r"the pointer file is missing or does not hash")):
        copied = copy(base, tmp_path_factory)
        text = (copied["repo"] / scoring).read_text() + "\n\ndef descriptive_table():\n    return []\n"
        record = harness.link_record(copied, sequence=3, change_class=rules.CHANGE_ANALYSIS_COMPLETION,
                                     files={scoring: text}, fold0=fold0)
        path, _digest, _commit = harness.publish_link(copied, record)
        with pytest.raises(AmendmentChainError, match=message):
            harness.verify(copied, tip=path)
    copied = copy(base, tmp_path_factory)
    other_chain = harness.write_fut_files(copied, 3, fit_chain=[copied["a1"], "c" * 64])
    text = (copied["repo"] / scoring).read_text() + "\n\ndef descriptive_table():\n    return []\n"
    record = harness.link_record(copied, sequence=3, change_class=rules.CHANGE_ANALYSIS_COMPLETION,
                                 files={scoring: text}, fold0=[other_chain])
    path, _digest, _commit = harness.publish_link(copied, record)
    with pytest.raises(AmendmentChainError, match=r"the pointer file names another fit chain"):
        harness.verify(copied, tip=path)


def _analysis_link(world: dict, *, scoring_text=None, extra=None, files=None):
    scoring = "src/ec_clean30/scoring.py"
    text = scoring_text or ((world["repo"] / scoring).read_text() + "\n\ndef descriptive_table():\n    return []\n")
    record = harness.link_record(world, sequence=3, change_class=rules.CHANGE_ANALYSIS_COMPLETION,
                                 files={scoring: text, **(files or {})}, write_fut=True, extra=extra)
    return harness.publish_link(world, record)


@pytest.mark.parametrize("attack", ["redefined_rule", "changed_decision", "weakened_test", "after_freeze2",
                                    "scientific_file"])
def test_an_analysis_completion_keeps_every_policy_fixture_and_limit(base, tmp_path_factory, attack):
    copied = copy(base, tmp_path_factory)
    repo = copied["repo"]
    scoring = (repo / "src/ec_clean30/scoring.py").read_text()
    if attack == "redefined_rule":
        # The accuracy rule is redefined and link 2's policy hashes are copied into the record.
        changed = scoring.replace('return {"rule": "synthetic accuracy"}', 'return {"rule": "redefined after FUT"}')
        path, _digest, _commit = _analysis_link(copied, scoring_text=changed,
                                                extra={"rule_policy_sha256": copied["record"]["rule_policy_sha256"]})
        message = r"\[C9\] the policy of amendment2_accuracy_v1 differs"
    elif attack == "changed_decision":
        changed = scoring.replace('def decision_for(name):\n    return "d"', 'def decision_for(name):\n    return "e"')
        assert changed != scoring
        path, _digest, _commit = _analysis_link(copied, scoring_text=changed)
        message = r"\[C16\] the bound refusal and decision fixtures do not pass"
    elif attack == "weakened_test":
        test_file = "tests/test_synthetic_refusals.py"
        weakened = (repo / test_file).read_text().replace('assert scoring.refuses("unbound input")', "assert True")
        path, _digest, _commit = _analysis_link(copied, files={test_file: weakened})
        message = r"\[C16\] the named test test_synthetic_refusals.py::test_r1 changed"
    elif attack == "after_freeze2":
        path, _digest, _commit = _analysis_link(copied, files={rules.FREEZE2_PATH: "{}\n"})
        message = r"\[C16\] an analysis completion is published only before Freeze 2"
    else:
        selector = (repo / "src/ec_clean30/selector.py").read_text() + "EXTRA = 1\n"
        path, _digest, _commit = _analysis_link(copied, files={"src/ec_clean30/selector.py": selector})
        message = r"an analysis completion may change only the four post-fit analysis files"
    with pytest.raises(AmendmentChainError, match=message):
        harness.verify(copied, tip=path)


def test_no_defect_amendment_follows_a_bound_stop_futility(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    scoring = "src/ec_clean30/scoring.py"
    stop = harness.write_fut_files(copied, 3, fit_chain=[copied["a1"], copied["a2"]], decision="stop_futility")
    text = (copied["repo"] / scoring).read_text() + "\n\ndef close_out_table():\n    return []\n"
    record3 = harness.link_record(copied, sequence=3, change_class=rules.CHANGE_ANALYSIS_COMPLETION,
                                  files={scoring: text}, fold0=[stop])
    path3, digest3, commit3 = harness.publish_link(copied, record3)
    assert harness.verify(copied, tip=path3)["tip_sequence"] == 3
    base3 = {"sequence": 3, "record": record3, "sha256": digest3, "commit": commit3,
             "chain": [copied["a1"], copied["a2"], digest3]}
    selection = (copied["repo"] / "src/ec_clean30/selection.py").read_text() + "\n\ndef _fixed():\n    return 1\n"
    record4 = harness.link_record(copied, sequence=4, change_class=rules.CHANGE_DEFECT, base=base3,
                                  files={"src/ec_clean30/selection.py": selection}, extra=harness.defect_fields())
    path4, _digest4, _commit4 = harness.publish_link(copied, record4)
    with pytest.raises(AmendmentChainError, match=r"after stop_futility no defect amendment may rerun"):
        harness.verify(copied, tip=path4)


# ---------------------------------------------------------------------------
# Should-fix 7: an execution link never changes a frozen gate


def test_an_execution_link_that_changes_the_frozen_gate_is_refused(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    phase3 = (copied["repo"] / "src/ec_clean30/phase3.py").read_text().replace("LIMIT = 3", "LIMIT = 30")
    record = harness.link_record(copied, sequence=3, change_class=rules.CHANGE_EXECUTION,
                                 files={"src/ec_clean30/phase3.py": phase3})
    assert record["pinned_execution_functions"] != copied["record"]["pinned_execution_functions"]
    path, _digest, _commit = harness.publish_link(copied, record)
    with pytest.raises(AmendmentChainError, match=r"an execution link changes pinned_execution_functions"):
        harness.verify(copied, tip=path)


# ---------------------------------------------------------------------------
# Should-fix 8 and nits: private names, analysis imports and the import closure of scientific files


@pytest.mark.parametrize("prefix, message", [
    ("from .selection import _hidden  # noqa\n", r"imports the private name _hidden of ec_clean30.selection"),
    ("from ec_clean30 import selection as _sel\nVALUE = _sel._hidden\n",
     r"reaches the private name _hidden of ec_clean30.selection"),
    ("import ec_clean30.views\nVALUE = getattr(ec_clean30.views, '_secret', None)\n",
     r"reaches the private name _secret of ec_clean30.views through getattr"),
    ("import importlib\nmodule = importlib.import_module('ec_clean30.stats')\nmodule._patched = True\n",
     r"reaches the private name _patched of ec_clean30.stats"),
    ("import ec_clean30.scoring  # noqa\n", r"the execution file src/ec_clean30/workstation.py imports the post-fit "
                                            r"analysis file src/ec_clean30/scoring.py"),
])
def test_an_execution_file_that_reaches_scientific_internals_or_analysis_code_is_refused(
        base, tmp_path_factory, prefix, message):
    copied = copy(base, tmp_path_factory)
    text = (copied["repo"] / WORKSTATION).read_text()
    harness.commit_change(copied, WORKSTATION, prefix + text)
    with pytest.raises(AmendmentChainError, match=message):
        harness.verify(copied)


def _dynamic_execution_import(repo):
    path = repo / "src/ec_clean30/selector.py"
    path.write_text(path.read_text() + 'import importlib\nimportlib.import_module("ec_clean30." + "phase3")\n')


def test_a_scientific_module_that_loads_an_execution_file_at_run_time_is_refused_at_the_links_commit(
        tmp_path_factory):
    world = harness.build_world(tmp_path_factory.mktemp("dynamic_import") / "world",
                                before_record=_dynamic_execution_import)
    with pytest.raises(AmendmentChainError, match=r"importing the scientific modules loads the execution file "
                                                  r"src/ec_clean30/phase3.py"):
        harness.verify(world)


def test_a_scientific_file_that_imports_an_execution_file_is_refused(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    harness.commit_change(copied, "src/training/run.py", '"""Synthetic trainer."""\nfrom ec_clean30 import phase3\n')
    with pytest.raises(AmendmentChainError, match=r"the scientific file src/training/run.py reaches the execution file "
                                                  r"src/ec_clean30/phase3.py"):
        harness.verify(copied)


# ---------------------------------------------------------------------------
# Should-fix 9 and 14: C5, C8, C9 and C11 at the link's commit


def _missing_role(record, repo):
    record["evidence"]["verification"] = "verification.json, written after the final checks"


def _other_answers(record, repo):
    record["decisions_file"]["answers"]["C4"]["choice"] = "an override toward continuing"


def _rendered_rule_text(record, repo):
    record["rule_text"]["source"] = rules.rendering_path(rules.AMENDMENT2_PATH)


def _tampered_policy(record, repo):
    record["rule_policy_sha256"]["controls_v2"] = "0" * 64


def _unlisted_change(record, repo):
    record["source_changes"].pop("src/ec_clean30/amendment2.py")


def _other_gate_pin(record, repo):
    record["pinned_execution_functions"]["src/ec_clean30/phase3.py::lane_gate"] = "0" * 64


def _g2_units(record, repo):
    record["data_state"]["g2_completed_units"] = 1


def _other_live_constants(record, repo):
    record["live_constants_sha256"] = "0" * 64


def _failed_dry_run(record, repo):
    binding = record["dry_runs"]["receipts"]["0"]
    receipt = json.loads((repo / binding["path"]).read_text())
    receipt["status"] = "failed"
    (repo / binding["path"]).write_text(json.dumps(receipt, sort_keys=True))
    digest = harness._file_sha(repo, binding["path"])
    binding["sha256"] = record["bound_files"]["dry_run_v2_fold0"]["sha256"] = digest


def _foreign_parent(record, repo):
    tree = harness._git(repo, "rev-parse", "HEAD^{tree}")
    record["source_parent_commit"] = harness._git(repo, "commit-tree", tree, "-m", "An unrelated commit")


def _protected_change(repo):
    path = repo / "src/ec_clean30/campaign.py"
    path.write_text(path.read_text() + "\nEXTRA_CONSTANT = 1\n")
    harness._git(repo, "add", "--all")
    harness._git(repo, "commit", "--quiet", "-m", "A protected change after the dry runs")


@pytest.mark.parametrize("hooks, message", [
    ({"mutate_record": _missing_role}, r"\[C8\] evidence.verification is not a file binding"),
    ({"mutate_record": _other_answers}, r"\[C8\] the record's decisions_file differs from the bound decisions file"),
    ({"mutate_record": _rendered_rule_text}, r"\[C8\] rule_text.source is not the reviewed document"),
    ({"mutate_record": _tampered_policy}, r"\[C9\] the policy of controls_v2 differs"),
    ({"mutate_record": _unlisted_change}, r"\[C5\] src/ec_clean30/amendment2.py changed but is not in source_changes"),
    ({"before_record": _protected_change}, r"\[C11\] src/ec_clean30/campaign.py was modified after the dry run"),
    ({"mutate_record": _other_gate_pin}, r"\[C6\] the closure of src/ec_clean30/phase3.py::lane_gate differs from its "
                                         r"pin"),
    ({"mutate_record": _g2_units}, r"\[C13\] link 2 records g2_completed_units = 1"),
    ({"mutate_record": _other_live_constants}, r"\[C9\] live_constants_sha256 differs"),
    ({"mutate_record": _failed_dry_run}, r"\[C11\] the fold-0 receipt is not a passed, real plan-v2 dry run"),
    ({"mutate_record": _foreign_parent}, r"\[C5\] the source parent commit of link 2 is not an ancestor of its commit"),
], ids=["missing_role", "decisions_answers", "rule_text_source", "policy", "scope", "r_cmd_v2", "gate_pin",
        "data_state", "live_constants", "dry_run", "source_parent"])
def test_a_published_link_2_that_breaks_a_link_check_is_refused_at_its_commit(tmp_path_factory, hooks, message):
    world = harness.build_world(tmp_path_factory.mktemp("bad_link2") / "world", **hooks)
    with pytest.raises(AmendmentChainError, match=message):
        harness.verify(world)


def test_a_link_that_names_another_source_commit_for_its_predecessor_is_refused(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    later = harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT)
    base2 = {"sequence": 2, "record": copied["record"], "sha256": copied["a2"], "commit": later,
             "chain": [copied["a1"], copied["a2"]]}
    record = harness.link_record(copied, sequence=3, change_class=rules.CHANGE_EXECUTION, base=base2,
                                 files={WORKSTATION: EXECUTION_TEXT + "MORE = 1\n"})
    path, _digest, _commit = harness.publish_link(copied, record)
    with pytest.raises(AmendmentChainError, match=r"\[C3\] Link 3 names a source commit for link 2 that did not "
                                                  r"enter its log line"):
        harness.verify(copied, tip=path)


def _bound_execution_file(record, repo):
    record["bound_files"]["workstation_at_link2"] = harness._binding(repo, WORKSTATION)


def test_a_receipt_refuses_a_bound_file_that_changed_since_it_was_written(tmp_path_factory):
    world = harness.build_world(tmp_path_factory.mktemp("bound_execution") / "world",
                                mutate_record=_bound_execution_file)
    out = world["root"] / "receipt.json"
    harness.write_receipt(world, out)
    harness.commit_change(world, WORKSTATION, EXECUTION_TEXT)
    with pytest.raises(AmendmentChainError, match=r"\[C8\] bound_files.workstation_at_link2: the bound file does not "
                                                  r"hash on the live tree"):
        harness.check_receipt(world, out)


def test_a_bound_file_edited_after_publication_is_refused_on_the_live_tree(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    receipt = copied["record"]["dry_runs"]["receipts"]["2"]["path"]
    harness.commit_change(copied, receipt, (copied["repo"] / receipt).read_text() + " ")
    with pytest.raises(AmendmentChainError, match=r"\[C8\] dry_runs.receipts.2: .* does not hash to its recorded "
                                                  r"SHA-256 on the live tree"):
        harness.verify(copied)


# ---------------------------------------------------------------------------
# Should-fix 11: time budget, process groups and malformed variables


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/status")
    return not (status.is_file() and "\nState:\tZ" in status.read_text())


def test_a_timed_out_process_group_is_killed_with_its_grandchildren_and_its_temporary_files(tmp_path):
    marker, folder = tmp_path / "pid", tmp_path / "tmpdir"
    script = f'sleep 60 & echo $! > "{marker}"; echo "$TMPDIR" > "{folder}"; touch "$TMPDIR/left"; wait'
    started = time.monotonic()
    with amendment_chain._verification(budget=1.5):
        with pytest.raises(AmendmentChainError, match=r"exceeded the verification budget"):
            amendment_chain._run_process(["sh", "-c", script], cwd=tmp_path, what="The probe")
    assert time.monotonic() - started < 20
    pid = int(marker.read_text())
    deadline = time.monotonic() + 10
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(pid)
    assert not Path(folder.read_text().strip()).exists()


_ENV_PROBE = amendment_chain._PRELUDE + (
    "import os\nprint(json.dumps({'result': {name: os.environ.get(name) for name in args['names']}}))\n")


def test_nested_interpreters_get_a_shorter_deadline_no_git_variable_and_malformed_variables_refuse(monkeypatch):
    names = [amendment_chain._DEADLINE_VARIABLE, amendment_chain._DEPTH_VARIABLE, "GIT_DIR", "GIT_CONFIG_GLOBAL"]
    monkeypatch.setenv("GIT_DIR", "/nonexistent")
    with amendment_chain._verification(budget=600):
        deadline = amendment_chain._DEADLINES[-1]
        seen = amendment_chain._run_fresh(ROOT, _ENV_PROBE, {"names": names}, what="The probe")
    assert float(seen[amendment_chain._DEADLINE_VARIABLE]) <= deadline - amendment_chain.NESTING_MARGIN_SECONDS + 1e-6
    assert seen[amendment_chain._DEPTH_VARIABLE] == "1" and seen["GIT_DIR"] is None
    assert seen["GIT_CONFIG_GLOBAL"] == os.devnull and os.environ["GIT_DIR"] == "/nonexistent"
    monkeypatch.setenv(amendment_chain._DEPTH_VARIABLE, "two")
    with pytest.raises(AmendmentChainError, match=r"is malformed"):
        amendment_chain._run_fresh(ROOT, _ENV_PROBE, {"names": names}, what="The probe")
    monkeypatch.delenv(amendment_chain._DEPTH_VARIABLE)
    monkeypatch.setenv(amendment_chain._DEADLINE_VARIABLE, "soon")
    with pytest.raises(AmendmentChainError, match=r"is malformed"):
        amendment_chain.verify_chain(ROOT / rules.AMENDMENT2_PATH, expected_freeze1_sha256=rules.FREEZE1_SHA256,
                                     expected_freeze1b_sha256=rules.FREEZE1B_SHA256, repo=ROOT)


# ---------------------------------------------------------------------------
# Should-fix 14: a sequence-3 link of each class, accepted from a later HEAD and through its receipt


@pytest.mark.parametrize("change_class", [rules.CHANGE_EXECUTION, rules.CHANGE_ANALYSIS_COMPLETION,
                                          rules.CHANGE_DEFECT])
def test_a_sequence_3_link_of_each_class_verifies_from_a_later_head(base, base_verified, tmp_path_factory,
                                                                     change_class):
    copied = copy(base, tmp_path_factory)
    repo = copied["repo"]
    if change_class == rules.CHANGE_EXECUTION:
        record = harness.link_record(copied, sequence=3, change_class=change_class,
                                     files={WORKSTATION: EXECUTION_TEXT})
    elif change_class == rules.CHANGE_ANALYSIS_COMPLETION:
        scoring = (repo / "src/ec_clean30/scoring.py").read_text() + "\n\ndef descriptive_table():\n    return []\n"
        record = harness.link_record(copied, sequence=3, change_class=change_class,
                                     files={"src/ec_clean30/scoring.py": scoring}, write_fut=True)
    else:
        selection = (repo / "src/ec_clean30/selection.py").read_text() + "\n\ndef _fixed():\n    return 1\n"
        record = harness.link_record(copied, sequence=3, change_class=change_class,
                                     files={"src/ec_clean30/selection.py": selection}, extra=harness.defect_fields())
    path, digest, commit = harness.publish_link(copied, record)
    in_place = harness.verify(copied, tip=path)
    harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT + "LATER = True\n")
    later = harness.verify(copied, tip=path)
    chain = [copied["a1"], copied["a2"], digest]
    assert later["chain"] == in_place["chain"] == chain and later["tip_sequence"] == 3
    assert [link["source_commit"] for link in later["links"]][-1] == commit
    assert all(link["published"] for link in later["links"]) and later["links"][-1]["change_class"] == change_class
    if change_class == rules.CHANGE_DEFECT:
        assert later["fit_chain"] == chain
        assert later["scientific_identity_sha256"] != base_verified["scientific_identity_sha256"]
    else:
        assert later["fit_chain"] == chain[:2]
        assert later["scientific_identity_sha256"] == base_verified["scientific_identity_sha256"]
    out = copied["root"] / "receipt3.json"
    harness.write_receipt(copied, out, tip=path, expected_chain=chain)
    pod = harness.partial_clone(copied, "pod3")
    assert harness.check_receipt(copied, out, repo=pod)["chain"] == chain


_RACE = '''
from pathlib import Path
original = amendment_chain.verify_chain


def racing(*positional, **keywords):
    result = original(*positional, **keywords)
    (Path(args["repo"]) / "src" / "ec_clean30" / "late_file.py").write_text("LATE = 1\\n")
    return result


amendment_chain.verify_chain = racing
try:
    amendment_chain.chain_receipt_record(args["tip"], expected_freeze1_sha256=args["freeze1"],
                                         expected_freeze1b_sha256=rules.FREEZE1B_SHA256, repo=args["repo"])
    result = "written"
except amendment_chain.AmendmentChainError as exc:
    result = str(exc)
'''


def test_no_receipt_is_written_when_the_repository_changes_during_the_verification(base, tmp_path_factory):
    copied = copy(base, tmp_path_factory)
    result = harness.run_world(copied, "exec", code=_RACE, tip=str(copied["repo"] / rules.AMENDMENT2_PATH))
    assert result == "The repository changed while the chain was verified; no receipt is written."
