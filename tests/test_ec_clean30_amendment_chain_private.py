"""Private bindings of the amendment chain (the repository is public) on synthetic published repositories.

The files of ``amendment_rules.PRIVATE_ROLES`` (the dry-run-v2 receipts, the CLEAN tie exposure and the F6 hold-out
lists and corpus) hold per-fold counts, accessions or host details that the campaign does not publish before Stage 7.
They live below the gitignored ``amendment_rules.PRIVATE_BINDING_ROOT`` of the synthetic repository, are bound as
``{path, sha256, private: true}`` and are never committed. Covered: the shared shape rules; the record builder's
refusals (a committed or public copy, a path outside the root or the repository, a symbolic link, a file that git
tracks or does not ignore, a dry-run receipt of a dirty tree) and its rendering, which carries paths and SHA-256 values
only; the verifier's refusals at link 2's commit and from a later HEAD (a private role bound publicly, a public role
flagged private, a path outside the root or with ``..``, a private file tracked at the link's commit, altered, absent
or behind a symbolic link, an altered or dirty dry-run receipt); the chain receipt, which lists the private bindings
apart and is checked on a pod without them; and the digest match of the execution side. Each refusal is matched by
its exact message. No campaign record, view or data folder is read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import ec_clean30_amendment2_build as build  # noqa: E402
import test_ec_clean30_amendment_chain as harness  # noqa: E402
from ec_clean30 import amendment2, amendment_chain  # noqa: E402
from ec_clean30 import amendment_rules as rules  # noqa: E402
from ec_clean30.amendment_chain import AmendmentChainError  # noqa: E402

PRIVATE = harness.PRIVATE
TIE = f"{PRIVATE}/clean_tie_exposure.json"
HOLDOUT3 = f"{PRIVATE}/f6_holdout/fold3.tsv"
CORPUS0 = f"{PRIVATE}/f6_corpus/fold0.tsv"
WORKSTATION = "src/ec_clean30/workstation.py"
EXECUTION_TEXT = '"""Synthetic execution file."""\nHOST = "later"\n'


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return harness.build_world(tmp_path_factory.mktemp("private_world") / "world")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build.build_world_with_builder(tmp_path_factory.mktemp("private_built") / "world")


def _copy(world: dict, factory) -> dict:
    copied = harness.copy_world(world, factory)
    if "inputs" in world:
        def move(value):
            return Path(str(value).replace(str(world["root"]), str(copied["root"])))

        copied["inputs"] = {key: move(value) if isinstance(value, Path) else {name: move(path)
                                                                               for name, path in value.items()}
                            for key, value in world["inputs"].items()}
    return copied


def _refused(world: dict, out: Path, **overrides) -> str:
    result = build.run_builder(world, out=out, **overrides)
    assert "error" in result and not out.exists(), result
    return result["error"]


def _lines(error: AmendmentChainError) -> list[str]:
    return [line.strip() for line in str(error).splitlines()]


# ---------------------------------------------------------------------------
# The shared rules


def test_the_private_root_and_roles_are_the_contract_fixtures():
    shape = build.SHAPE["required_bindings"]
    assert rules.PRIVATE_BINDING_ROOT == shape["private_root"] == build.BUILDER["private_bindings"]["root"]
    assert list(rules.PRIVATE_ROLES) == shape["private_roles"] == build.BUILDER["private_bindings"]["roles"]
    assert rules.PRIVATE_BINDING_ROOT == f"DeepMzyme_Data/campaigns/{rules.CAMPAIGN_ID}/"
    assert set(rules.PRIVATE_ROLES) - {"dry_runs.receipts"} <= set(rules.REQUIRED_BINDINGS)
    assert all(rules.is_private_role(role) for role in ("inputs.f6_corpus.tables.0", "inputs.f6_holdout.4.inner0",
                                                        "dry_runs.receipts.2", "inputs.clean_tie_exposure"))
    assert not any(rules.is_private_role(role) for role in ("dry_runs.v1_resolved_config", "evidence.verification",
                                                            "inputs.f6_corpus_extra", "bound_files.decisions"))


@pytest.mark.parametrize("path, ok", [
    (f"{rules.PRIVATE_BINDING_ROOT}amendment2/x.json", True),
    (rules.PRIVATE_BINDING_ROOT, False),
    (f"{rules.PRIVATE_BINDING_ROOT}amendment2/../../../../docs/x.json", False),
    (f"{rules.PRIVATE_BINDING_ROOT}./x.json", False),
    (f"{rules.PRIVATE_BINDING_ROOT}a//x.json", False),
    (f"/{rules.PRIVATE_BINDING_ROOT}x.json", False),
    ("docs/x.json", False),
    ("DeepMzyme_Data/campaigns/other_campaign/x.json", False),
])
def test_a_private_path_is_a_safe_relative_path_below_the_root(path, ok):
    assert rules.is_private_path(path) is ok
    binding = {"path": path, "sha256": "a" * 64, "private": True}
    assert rules.is_private_binding(binding) is ok
    assert not rules.is_file_binding(binding)


def test_privacy_follows_the_role_and_a_private_binding_carries_nothing_else():
    hidden = {"path": f"{PRIVATE}/x.json", "sha256": "a" * 64, "private": True}
    public = {"path": "docs/x.json", "sha256": "a" * 64}
    record = {"inputs": {"clean_tie_exposure": hidden}, "evidence": {"verification": public},
              "dry_runs": {"receipts": {"0": hidden}}}
    sections = ("inputs", "evidence", "dry_runs")
    assert rules.binding_privacy_problems(record, sections) == []
    cases = [
        ({"inputs": {"clean_tie_exposure": public}},
         "inputs.clean_tie_exposure binds a private input publicly; a private role holds {path, sha256, private: "
         f"true}} below {rules.PRIVATE_BINDING_ROOT}"),
        ({"evidence": {"verification": {**public, "private": True}}},
         f"evidence.verification flags a public binding private; only {list(rules.PRIVATE_ROLES)} are private"),
        ({"dry_runs": {"receipts": {"0": {**hidden, "path": "docs/x.json"}}}},
         "dry_runs.receipts.0: the private path 'docs/x.json' is not a safe repository-relative path below "
         f"{rules.PRIVATE_BINDING_ROOT}"),
        ({"inputs": {"clean_tie_exposure": {**hidden, "n_ties_fold3": 7}}},
         "inputs.clean_tie_exposure is not a private file binding {path, sha256, private: true} (no other key)"),
        ({"evidence": {"verification": {"path": f"{PRIVATE}/verification.json", "sha256": "a" * 64}}},
         f"evidence.verification binds '{PRIVATE}/verification.json' below the private binding root publicly; a "
         "public binding is a committed repository file"),
    ]
    for change, message in cases:
        changed = json.loads(json.dumps(record))
        for section, value in change.items():
            changed[section].update(value)
        assert rules.binding_privacy_problems(changed, sections) == [message]


# ---------------------------------------------------------------------------
# The record builder


def test_the_builder_binds_the_private_roles_privately_and_publishes_paths_and_digests_only(built):
    record, repo = built["record"], built["repo"]
    private = {where: item for section in amendment2.FILE_BINDING_SECTIONS
               for where, item in rules.binding_items(record.get(section), section) if rules.is_private_role(where)}
    assert sorted(private) == sorted([*(f"dry_runs.receipts.{fold}" for fold in build.FOLDS),
                                      "inputs.clean_tie_exposure", "inputs.f6_corpus.union_fasta",
                                      *(f"inputs.f6_corpus.tables.{fold}" for fold in build.FOLDS),
                                      *(f"inputs.f6_holdout.{fold}" for fold in build.FOLDS)])
    assert all(rules.is_private_binding(item) and item["sha256"] == harness._file_sha(repo, item["path"])
               for item in private.values())
    assert rules.binding_privacy_problems(record, amendment2.FILE_BINDING_SECTIONS) == []
    # Never committed: ignored by git and absent from the link's commit.
    assert all(harness._git(repo, "ls-tree", "--name-only", built["commits"]["S2"], "--", item["path"]) == ""
               for item in private.values())
    # The record and its rendering carry paths and digests, never the private content.
    rendering = (repo / rules.rendering_path(rules.AMENDMENT2_PATH)).read_text()
    text = json.dumps(record)
    for item in private.values():
        content = (repo / item["path"]).read_text().strip()
        assert content not in text and content not in rendering
    assert "## Private bindings" in rendering
    digest = private["inputs.clean_tie_exposure"]["sha256"]
    assert f"| `inputs.clean_tie_exposure` | `{TIE}` | `{digest}` |" in rendering
    # The bound files are public and committed.
    assert record["bound_files"]["decisions"] == {"path": build.DECISIONS_PATH,
                                                  "sha256": harness._file_sha(repo, build.DECISIONS_PATH)}


def test_the_builder_refuses_a_public_copy_a_link_a_tracked_or_unignored_file_and_a_dirty_receipt(
        built, tmp_path_factory):
    copied = _copy(built, tmp_path_factory)
    repo = copied["repo"]
    out = copied["root"] / "scratch" / "refused.json"
    # A private role bound to an in-repository public copy (committed).
    public_copy = f"{harness.DOCS}/amendment2/clean_tie_exposure.json"
    shutil.copyfile(repo / TIE, repo / public_copy)
    harness._commit(repo, "A public copy of a private input")
    error = _refused(copied, out, bind={"inputs.clean_tie_exposure": repo / public_copy})
    assert error == (f"AmendmentRecordError: Bound file inputs.clean_tie_exposure ({public_copy}) is not below the "
                     f"private binding root {rules.PRIVATE_BINDING_ROOT}.")
    # A file outside the repository.
    outside = copied["root"] / "campaign" / "clean_tie_exposure.json"
    shutil.copyfile(repo / TIE, outside)
    error = _refused(copied, out, bind={"inputs.clean_tie_exposure": outside})
    assert error == ("AmendmentRecordError: Bound file inputs.clean_tie_exposure is outside the repository; a "
                     f"private input lives below {rules.PRIVATE_BINDING_ROOT} of the repository.")
    # A symbolic link in the path below the repository root.
    real = repo / PRIVATE / "f6_corpus_real"
    shutil.copytree(repo / PRIVATE / "f6_corpus", real)
    linked = repo / PRIVATE / "f6_corpus_linked"
    linked.symlink_to(real, target_is_directory=True)
    relative = f"{PRIVATE}/f6_corpus_linked/fold0.tsv"
    error = _refused(copied, out, bind={"inputs.f6_corpus.tables.0": repo / relative})
    assert error == (f"AmendmentRecordError: Bound file inputs.f6_corpus.tables.0 ({relative}) has a symbolic link "
                     "in its path; a private input is a regular file reached without links.")
    # A private file that git tracks (force-added below the ignored root).
    harness._git(repo, "add", "-f", "--", HOLDOUT3)
    error = _refused(copied, out)
    assert error == (f"AmendmentRecordError: Bound file inputs.f6_holdout.3 ({HOLDOUT3}) is tracked by git; a private "
                     "input is never committed (the repository is public).")
    harness._git(repo, "rm", "--quiet", "--cached", "--", HOLDOUT3)
    # A private root that git does not ignore.
    gitignore = repo / ".gitignore"
    original = gitignore.read_text()
    unignored = (r"AmendmentRecordError: The dry-run-v2 receipt of fold 0 \(.*/dry_runs/fold0\.json\) is not "
                 r"ignored by a \.gitignore committed at HEAD; the private binding root must be gitignored so that a "
                 r"private input is never committed \(\.git/info/exclude and core\.excludesFile hold in one clone "
                 r"only\)\.")
    gitignore.write_text(original.replace("DeepMzyme_Data\n", ""))
    error = _refused(copied, out)
    assert re.fullmatch(unignored, error), error
    # Only a local exclude of this clone ignores the root (.git/info/exclude, or core.excludesFile in the local
    # configuration): another clone, such as a fresh clone of origin, would see the private files as untracked.
    exclude = repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(exist_ok=True)
    exclude.write_text("DeepMzyme_Data\n")
    assert harness._git(repo, "check-ignore", "--", TIE).strip() == TIE
    assert re.fullmatch(unignored, _refused(copied, out)), "info/exclude"
    exclude.write_text("")
    local_excludes = copied["root"] / "local_excludes"
    local_excludes.write_text("DeepMzyme_Data\n")
    harness._git(repo, "config", "core.excludesFile", str(local_excludes))
    assert re.fullmatch(unignored, _refused(copied, out)), "core.excludesFile"
    harness._git(repo, "config", "--unset", "core.excludesFile")
    # The committed .gitignore lacks the root and only the working-tree copy adds it back.
    harness._git(repo, "add", "--", ".gitignore")
    harness._git(repo, "commit", "--quiet", "-m", "A .gitignore without the private root")
    gitignore.write_text(original)
    assert re.fullmatch(unignored, _refused(copied, out)), "modified .gitignore"
    harness._git(repo, "add", "--", ".gitignore")
    harness._git(repo, "commit", "--quiet", "-m", "The .gitignore with the private root again")
    assert harness._git(repo, "status", "--porcelain") == ""
    # A dry-run receipt written on a dirty tree.
    receipt_path = copied["inputs"]["dry_run_receipts"]["1"]
    receipt = json.loads(receipt_path.read_text())
    receipt["provenance"]["git"]["clean"] = False
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    error = _refused(copied, out)
    assert error == ("AmendmentRecordError: The dry-run-v2 receipt of fold 1 is not a passed, real plan-v2 dry run of "
                     "that fold on a clean tree.")


def test_a_private_binding_in_the_draft_is_checked_like_a_bound_one(built, tmp_path_factory):
    copied = _copy(built, tmp_path_factory)
    repo = copied["repo"]
    out = copied["root"] / "scratch" / "refused.json"
    draft = json.loads(copied["inputs"]["draft"].read_text())
    draft["inputs"]["clean_tie_exposure"] = {"path": TIE, "sha256": "0" * 64, "private": True}
    path = copied["root"] / "drafts" / "private_in_draft.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(draft))
    error = _refused(copied, out, draft=path, drop_binds=("inputs.clean_tie_exposure",))
    assert error == ("AmendmentRecordError: File bindings of the record do not hold: "
                     f"['inputs.clean_tie_exposure: {TIE} does not hash to its recorded SHA-256'].")
    draft["inputs"]["clean_tie_exposure"] = {"path": TIE, "sha256": harness._file_sha(repo, TIE)}
    path.write_text(json.dumps(draft))
    error = _refused(copied, out, draft=path, drop_binds=("inputs.clean_tie_exposure",))
    assert error.startswith("AmendmentRecordError: Required bindings are absent or malformed: "
                            "['inputs.clean_tie_exposure is not a private file binding {path, sha256, private: true} "
                            f"below {rules.PRIVATE_BINDING_ROOT}']"), error


# ---------------------------------------------------------------------------
# The verifier at link 2's commit


def _public_copy_of_tie_exposure(record, repo):
    relative = f"{harness.DOCS}/amendment2/clean_tie_exposure.json"
    shutil.copyfile(repo / TIE, repo / relative)
    record["inputs"]["clean_tie_exposure"] = harness._binding(repo, relative)


def _public_role_flagged_private(record, repo):
    record["evidence"]["verification"]["private"] = True


def _private_flag_outside_the_root(record, repo):
    relative = f"{harness.DOCS}/amendment2/clean_tie_exposure.json"
    shutil.copyfile(repo / TIE, repo / relative)
    record["inputs"]["clean_tie_exposure"] = {**harness._binding(repo, relative), "private": True}


def _dotted_private_path(record, repo):
    record["inputs"]["clean_tie_exposure"]["path"] = f"{PRIVATE}/../../../../../{harness.DOCS}/x.json"


def _absolute_private_path(record, repo):
    record["inputs"]["clean_tie_exposure"]["path"] = f"/{TIE}"


def _tracked_private_file(repo, record):
    harness._git(repo, "add", "-f", "--", TIE)


def _dirty_receipt(record, repo):
    binding = record["dry_runs"]["receipts"]["1"]
    receipt = json.loads((repo / binding["path"]).read_text())
    receipt["provenance"]["git"]["clean"] = False
    (repo / binding["path"]).write_text(json.dumps(receipt, sort_keys=True))
    binding["sha256"] = harness._file_sha(repo, binding["path"])


def _public_receipt(record, repo):
    del record["dry_runs"]["receipts"]["0"]["private"]


ROOT_TEXT = rules.PRIVATE_BINDING_ROOT


@pytest.mark.parametrize("hooks, message", [
    ({"mutate_record": _public_copy_of_tie_exposure},
     "[C8] inputs.clean_tie_exposure binds a private input publicly; a private role holds {path, sha256, private: "
     f"true}} below {ROOT_TEXT}"),
    ({"mutate_record": _public_role_flagged_private},
     f"[C8] evidence.verification flags a public binding private; only {list(rules.PRIVATE_ROLES)} are private"),
    ({"mutate_record": _private_flag_outside_the_root},
     f"[C8] inputs.clean_tie_exposure: the private path '{harness.DOCS}/amendment2/clean_tie_exposure.json' is not a "
     f"safe repository-relative path below {ROOT_TEXT}"),
    ({"mutate_record": _dotted_private_path},
     f"[C8] inputs.clean_tie_exposure: the private path '{PRIVATE}/../../../../../{harness.DOCS}/x.json' is not a safe "
     f"repository-relative path below {ROOT_TEXT}"),
    ({"mutate_record": _absolute_private_path},
     f"[C8] inputs.clean_tie_exposure: the private path '/{TIE}' is not a safe repository-relative path below "
     f"{ROOT_TEXT}"),
    ({"before_link2_commit": _tracked_private_file},
     f"[C8] inputs.clean_tie_exposure: the private file {TIE} is tracked at the link's commit (a private input was "
     "published)"),
    ({"mutate_record": _dirty_receipt},
     "[C11] the fold-1 dry run did not run on a clean tree at the recorded source commit"),
    ({"mutate_record": _public_receipt},
     f"[C11] the fold-0 receipt is not bound privately ({{path, sha256, private: true}} below {ROOT_TEXT})"),
], ids=["private_role_bound_publicly", "public_role_flagged_private", "outside_the_root", "dotted_path",
        "absolute_path", "tracked_at_the_link_commit", "dirty_receipt", "public_receipt"])
def test_a_published_link_2_whose_private_bindings_break_the_rules_is_refused(tmp_path_factory, hooks, message):
    world = harness.build_world(tmp_path_factory.mktemp("private_bad_link2") / "world", **hooks)
    with pytest.raises(AmendmentChainError) as refused:
        harness.verify(world)
    assert message in _lines(refused.value), str(refused.value)


def test_a_private_file_altered_absent_or_behind_a_link_is_refused_in_place_and_from_a_later_head(
        base, tmp_path_factory):
    nested = f"[C4] Link 2 at its source commit {base['commits']['S2'][:12]} refused: "
    for label, change, line in (
            ("altered", lambda repo: (repo / HOLDOUT3).write_text("holdout 3 altered\n"),
             f"[C8] inputs.f6_holdout.3: the private file {HOLDOUT3} does not hash to its recorded SHA-256 in the "
             "verification repository's working tree"),
            ("absent", lambda repo: (repo / TIE).unlink(),
             f"[C8] inputs.clean_tie_exposure: the private file {TIE} is absent from the verification repository's "
             "working tree (a private input is never read from git)"),
            ("linked", lambda repo: _link_corpus_folder(repo),
             f"[C8] inputs.f6_corpus.tables.0: the private file {CORPUS0} has a symbolic link in its path in the "
             "verification repository's working tree")):
        copied = _copy(base, tmp_path_factory)
        change(copied["repo"])
        assert harness._git(copied["repo"], "status", "--porcelain") == "", label  # private files never dirty
        with pytest.raises(AmendmentChainError) as refused:
            harness.verify(copied)
        assert line in _lines(refused.value), (label, str(refused.value))
        # From a later HEAD, link 2 is authenticated in a checkout of its commit, which holds no private file: the
        # nested verifier reads them from this repository's working tree and refuses the same way.
        harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT)
        with pytest.raises(AmendmentChainError) as refused:
            harness.verify(copied)
        assert nested in str(refused.value) and line in str(refused.value), (label, str(refused.value))


def _link_corpus_folder(repo: Path) -> None:
    folder = repo / PRIVATE / "f6_corpus"
    moved = repo / PRIVATE / "f6_corpus_real"
    folder.rename(moved)
    folder.symlink_to(moved, target_is_directory=True)


def test_an_altered_dry_run_receipt_is_refused_by_c8_and_c11(base, tmp_path_factory):
    copied = _copy(base, tmp_path_factory)
    receipt = copied["repo"] / copied["record"]["dry_runs"]["receipts"]["0"]["path"]
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(AmendmentChainError) as refused:
        harness.verify(copied)
    lines = _lines(refused.value)
    assert "[C11] the fold-0 dry-run-v2 receipt is missing or does not hash" in lines
    assert (f"[C8] dry_runs.receipts.0: the private file {PRIVATE}/dry_runs/fold0.json does not hash to its recorded "
            "SHA-256 in the verification repository's working tree") in lines


def test_a_private_file_published_after_the_link_is_refused_on_the_live_tree(base, tmp_path_factory):
    copied = _copy(base, tmp_path_factory)
    harness._git(copied["repo"], "add", "-f", "--", TIE)
    harness._commit(copied["repo"], "A private input published by mistake")
    with pytest.raises(AmendmentChainError) as refused:
        harness.verify(copied)
    assert (f"[C8] inputs.clean_tie_exposure: the private file {TIE} is tracked on the live tree (a private input was "
            "published)") in _lines(refused.value)


def test_a_top_level_call_reads_its_own_working_tree_whatever_the_environment_names(base, tmp_path_factory):
    # The nested verifier's private root is an argument of the fixed chain probe, never an environment variable: a
    # top-level call that exports a nesting depth and a root naming a pristine copy still reads its own altered file.
    copied = _copy(base, tmp_path_factory)
    (copied["repo"] / HOLDOUT3).write_text("holdout 3 altered\n")
    elsewhere = _copy(base, tmp_path_factory)
    line = (f"[C8] inputs.f6_holdout.3: the private file {HOLDOUT3} does not hash to its recorded SHA-256 in the "
            "verification repository's working tree")
    env = {amendment_chain._DEPTH_VARIABLE: "1", "DEEPMZYME_EC_CLEAN30_CHAIN_PRIVATE_ROOT": str(elsewhere["repo"])}
    for exported in (None, env):
        with pytest.raises(AmendmentChainError) as refused:
            harness.verify(copied, env=exported)
        assert line in _lines(refused.value), str(refused.value)
    # From a later HEAD the nested verifier at link 2's commit gets this repository's root from the probe argument.
    harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT)
    with pytest.raises(AmendmentChainError) as refused:
        harness.verify(copied, env=env)
    assert line in str(refused.value), str(refused.value)
    assert "DEEPMZYME_EC_CLEAN30_CHAIN_PRIVATE_ROOT" not in Path(amendment_chain.__file__).read_text(encoding="utf-8")
    for raw in ("relative/root", None, str(copied["root"] / "missing")):
        with pytest.raises(AmendmentChainError, match=r"The chain probe names no private root directory"):
            amendment_chain._probe_private_root(raw)
    assert amendment_chain._probe_private_root(str(elsewhere["repo"])) == Path(os.path.realpath(elsewhere["repo"]))


# ---------------------------------------------------------------------------
# The chain receipt and the execution side


def test_the_receipt_lists_private_bindings_apart_and_checks_in_a_pod_without_them(base, tmp_path_factory):
    copied = _copy(base, tmp_path_factory)
    harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT)
    out = copied["root"] / "receipt.json"
    receipt = harness.write_receipt(copied, out)
    assert receipt["format"] == amendment_chain.CHAIN_RECEIPT_FORMAT and receipt["schema_version"] == 2
    assert receipt["private_bindings"] and all(rules.is_private_role(item["where"])
                                               and rules.is_private_path(item["path"])
                                               for item in receipt["private_bindings"])
    assert not any(rules.is_private_role(item["where"]) for item in receipt["bindings"])
    pod = harness.partial_clone(copied, "pod")
    assert not (pod / "DeepMzyme_Data").exists()
    assert harness.check_receipt(copied, out, repo=pod)["chain"] == [copied["a1"], copied["a2"]]
    # A re-hashed receipt that hides a public binding among the private ones (never re-read) is refused.
    forged = json.loads(json.dumps(receipt))
    moved = forged["bindings"].pop(0)
    forged["private_bindings"].append(moved)
    del forged["receipt_sha256"]
    forged["receipt_sha256"] = rules.sha256_canonical(forged)
    (copied["root"] / "forged.json").write_text(json.dumps(forged))
    with pytest.raises(AmendmentChainError) as refused:
        harness.check_receipt(copied, copied["root"] / "forged.json", repo=pod)
    lines = _lines(refused.value)
    assert "[C8] the receipt's public and private bindings differ from those of the tip link" in lines
    assert f"[C8] the receipt names a malformed private binding {moved['where']}" in lines


def test_the_execution_side_matches_a_private_binding_by_digest(built, monkeypatch):
    from ec_clean30 import phase3

    repo, record = built["repo"], built["record"]
    monkeypatch.setattr(phase3, "REPO_ROOT", repo)
    source = {"amendment_path": rules.AMENDMENT2_PATH}
    table = record["inputs"]["f6_corpus"]["tables"]["2"]
    assert phase3._bound_pretrain_file(source, "inputs.f6_corpus.tables.2", repo / table["path"]) == {
        "path": table["path"], "sha256": table["sha256"]}
    verified = harness.verify(built)
    evidence = amendment_chain.fold_dry_run_evidence(verified, 4)
    receipt = record["dry_runs"]["receipts"]["4"]
    assert evidence["receipt"] == {"path": receipt["path"], "sha256": receipt["sha256"],
                                   "view_manifest_sha256": built["freeze1_doc"]["views"]["4"]["view_manifest_sha256"]}
    assert harness._file_sha(repo, evidence["receipt"]["path"]) == receipt["sha256"]


def _fold_view(folder: Path, fold: int, freeze1_doc: dict, *, manifest: str | None = None):
    """A stand-in fold view whose manifest and Freeze-1 in-view receipt are the synthetic Freeze 1's."""
    from types import SimpleNamespace

    from ec_clean30 import views

    receipts = folder / "receipts"
    receipts.mkdir(parents=True)
    (receipts / views.DRY_RUN_RECEIPT_NAME).write_bytes(f"in-view receipt {fold}".encode())
    layers = {"final": "f" * 64, "block26": "b" * 64}
    frozen = {**freeze1_doc["views"][str(fold)], "chain_means": layers}
    view = SimpleNamespace(smoke=False, fold=fold, receipts=receipts,
                           manifest_sha256=manifest or frozen["view_manifest_sha256"],
                           chain_means_manifest_sha256=lambda layer: layers[layer])
    return view, {"format": "execution-v5 stand-in", "scientific_identity_sha256": "c" * 64,
                  "frozen_views": {str(fold): frozen}}


def test_the_view_binding_holds_on_a_pod_clone_without_the_private_receipts(base, tmp_path_factory, monkeypatch):
    from ec_clean30 import phase3

    copied = _copy(base, tmp_path_factory)
    harness.commit_change(copied, WORKSTATION, EXECUTION_TEXT)
    out = copied["root"] / "receipt.json"
    harness.write_receipt(copied, out)
    pod = harness.partial_clone(copied, "pod")
    assert not (pod / "DeepMzyme_Data").exists()
    verified = harness.check_receipt(copied, out, repo=pod)
    monkeypatch.setattr(phase3, "REPO_ROOT", pod)
    monkeypatch.setattr(phase3, "authenticated_chain", lambda source, **_: verified)
    for fold in range(5):
        view, source = _fold_view(tmp_path_factory.mktemp(f"pod_view{fold}"), fold, copied["freeze1_doc"])
        phase3._bind_view(view, source)
    # Another view than the one the dry run ran on (Freeze 1 agrees with the view, the receipt does not).
    view, source = _fold_view(tmp_path_factory.mktemp("pod_view_other"), 2, copied["freeze1_doc"],
                              manifest="9" * 64)
    source["frozen_views"]["2"]["view_manifest_sha256"] = "9" * 64
    with pytest.raises(phase3.Phase3Error,
                       match=r"^Amendment-2 dry-run evidence does not bind this fold view \(C11\)\.$"):
        phase3._bind_view(view, source)
