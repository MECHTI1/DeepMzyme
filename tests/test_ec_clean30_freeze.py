"""Freeze-1 completeness checks, the Freeze-1 document, verify-freeze1 and Freeze 1b (ec_clean30.freeze).

Synthetic data only: git repositories are built in pytest's tmp_path; the campaign world (five folds, a passed
cohort audit, the training-side identity chain with a fake pinned search, R2 lists) comes from the overlay tests'
builder, and the overlay, the fold views and their chain means are built from it here. Dry-run receipts are
synthetic (the probes are covered by the campaign tests).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from ec_clean30 import campaign, freeze, identity, provenance, views  # noqa: E402
from ec_clean30.campaign import CAMPAIGN_ID, N_OUTER_FOLDS  # noqa: E402
from ec_clean30.freeze import Freeze1Error  # noqa: E402
from ec_clean30.identity import (  # noqa: E402
    build_cohort_components, build_edges, build_inputs, build_strata, run_search, sha256_file,
)
from ec_clean30.overlay import DEFAULT_OVERLAY_ID, build_overlay  # noqa: E402
from ec_clean30.views import SourceLayout, View, build_view  # noqa: E402
from ec_clean30_fixtures import build_chain_means  # noqa: E402
from test_ec_clean30_identity import FakeMMseqs  # noqa: E402
from test_ec_clean30_overlay import REAL_PROVENANCE, World, build_world  # noqa: E402

OTHER_F = "b" * 64
LOG = freeze.LOG_PATH
DOCUMENT = f"{freeze.CAMPAIGN_DOCS_DIR}/freeze1.json"


# ---------------------------------------------------------------------------
# Git helpers


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                                "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false", *args], cwd=repo,
                               check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _write(repo: Path, files: dict[str, str | bytes]) -> None:
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")


def _commit(repo: Path, files: dict[str, str | bytes], message: str = "change", remove: tuple[str, ...] = ()) -> str:
    _write(repo, files)
    for relative in remove:
        _git(repo, "rm", "-q", relative)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _init(repo: Path, files: dict[str, str | bytes]) -> str:
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    return _commit(repo, files, "initial")


def _copy_repo(repo: Path, target: Path) -> Path:
    shutil.copytree(repo, target)
    return target


# ---------------------------------------------------------------------------
# A small repository in the shape rule R-CMD checks

PINNED_RUNNER = "\n\n".join(f"def {name}(args):\n    return {index}\n"
                            for index, name in enumerate(freeze.PINNED_RUNNER_FUNCTIONS))
RUNNER = f'"""Stand-in runner."""\n\n\n{PINNED_RUNNER}\n\ndef action_manifest(args):\n    return 0\n'
BASE_FILES = {
    "src/ec_clean30/__init__.py": "",
    "src/ec_clean30/campaign.py": "from ec_clean30 import selection  # noqa: F401\n",
    "src/ec_clean30/selection.py": "VALUE = 1\n",
    "src/ec_clean30/commands.py": "VALUE = 1\n",
    "src/ec_clean30/views.py": "VALUE = 1\n",
    "src/ec_clean30/provenance.py": "VALUE = 1\n",
    "src/ec_clean30/identity.py": "VALUE = 1\n",
    "src/training/loop.py": "VALUE = 1\n",
    freeze.R_CMD_RUNNER: RUNNER,
    "tests/test_stand_in.py": "def test_nothing():\n    pass\n",
    LOG: "# Campaign log\n",
}
# The allowed changes of a CP-C-like commit: an allowlisted module, a new module nothing imports, the runner
# outside its pinned functions, tests and docs.
CP_C_FILES = {
    "src/ec_clean30/selection.py": "VALUE = 2\n",
    "src/ec_clean30/freeze.py": "VALUE = 3\n",
    freeze.R_CMD_RUNNER: RUNNER.replace("def action_manifest(args):\n    return 0\n",
                                        "def action_manifest(args):\n    return 1\n"),
    "tests/test_stand_in.py": "def test_nothing():\n    assert True\n",
    f"{freeze.CAMPAIGN_DOCS_DIR}/notes.md": "notes\n",
}


def rcmd_repo(base: Path) -> tuple[Path, str]:
    """A repository with a dry-run commit and an allowed CP-C commit on top; returns it and the dry-run HEAD."""
    repo = base / "repo"
    dry_run_head = _init(repo, BASE_FILES)
    _commit(repo, CP_C_FILES, "cp-c")
    return repo, dry_run_head


def write_plans_dir(path: Path, *, change_fold: int | None = None) -> Path:
    """Plan records as the runner's ``plan --out`` writes them; ``change_fold`` gets a plan that differs."""
    path.mkdir(parents=True)
    for fold in range(N_OUTER_FOLDS):
        plan = campaign.build_fold_plan(fold)
        if fold == change_fold:
            plan = campaign.build_fold_plan(fold, levers=["L1"])
        out = path / f"fold{fold}.json"
        digest = provenance.write_new_json(out, plan)
        provenance.write_new_json(out.with_name(out.name + ".receipt.json"),
                                  {"plan": out.name, "plan_file_sha256": digest, "plan_sha256": plan["plan_sha256"],
                                   "counts": plan["counts"], "provenance": {"note": "synthetic"}})
    return path


@pytest.fixture(scope="module")
def plans_dir(tmp_path_factory):
    return write_plans_dir(tmp_path_factory.mktemp("plans") / "plans")


# ---------------------------------------------------------------------------
# Rule R-CMD


def test_r_cmd_passes_for_the_allowed_changes(tmp_path, plans_dir):
    repo, dry_run_head = rcmd_repo(tmp_path)
    result = freeze.r_cmd_check({0: dry_run_head, 1: dry_run_head}, repo, plans_dir)
    assert result["status"] == "passed", result["problems"]
    assert result["added_files"] == ["src/ec_clean30/freeze.py"]
    assert result["dry_run_heads"] == {"0": dry_run_head, "1": dry_run_head}
    assert sorted(result["plans"]) == [str(fold) for fold in range(N_OUTER_FOLDS)]
    assert result["plans"]["0"]["plan_sha256"] == campaign.build_fold_plan(0)["plan_sha256"]
    assert set(result["pinned_runner_function_sha256"]) == set(freeze.PINNED_RUNNER_FUNCTIONS)
    assert freeze.r_cmd_problems([dry_run_head], repo) == []


@pytest.mark.parametrize("files, remove, expected", [
    ({"src/ec_clean30/commands.py": "VALUE = 2\n"}, (), "src/ec_clean30/commands.py was modified"),
    ({"src/training/loop.py": "VALUE = 2\n"}, (), "src/training/loop.py was modified"),
    ({"src/model_extra.py": "VALUE = 1\n"}, (), "src/model_extra.py was added"),
    ({"src/ec_clean30/sub/new.py": "VALUE = 1\n"}, (), "src/ec_clean30/sub/new.py was added"),
    ({}, ("src/ec_clean30/views.py",), "src/ec_clean30/views.py was deleted"),
])
def test_r_cmd_refuses_a_changed_protected_file(tmp_path, files, remove, expected):
    repo, dry_run_head = rcmd_repo(tmp_path)
    _commit(repo, files, remove=remove)
    problems = freeze.r_cmd_problems({0: dry_run_head}, repo)
    assert any(expected in problem for problem in problems), problems


def test_r_cmd_refuses_a_changed_pinned_runner_function(tmp_path):
    repo, dry_run_head = rcmd_repo(tmp_path)
    runner = (repo / freeze.R_CMD_RUNNER).read_text()
    _commit(repo, {freeze.R_CMD_RUNNER: runner.replace("def _csv(args):\n    return 0",
                                                       "def _csv(args):\n    return 99")})
    problems = freeze.r_cmd_problems({0: dry_run_head}, repo)
    assert problems == [f"the pinned runner function _csv changed after the dry run at {dry_run_head[:12]}"]


@pytest.mark.parametrize("addition, expected", [
    ("\n_layout = action_manifest\n", "the runner assigns the pinned name _layout at top level (tree)"),
    ("\nfrom os.path import join as _build\n", "the runner imports the pinned name _build at top level (tree)"),
    ("\nclass action_plan:\n    pass\n", "the runner defines a class named after the pinned name action_plan (tree)"),
])
def test_r_cmd_refuses_a_redefined_pinned_name(tmp_path, addition, expected):
    repo, dry_run_head = rcmd_repo(tmp_path)
    runner = (repo / freeze.R_CMD_RUNNER).read_text()
    _commit(repo, {freeze.R_CMD_RUNNER: runner + addition})
    problems = freeze.r_cmd_problems({0: dry_run_head}, repo)
    assert expected in problems


def test_r_cmd_refuses_an_added_decorator_on_a_pinned_runner_function(tmp_path):
    repo, dry_run_head = rcmd_repo(tmp_path)
    runner = (repo / freeze.R_CMD_RUNNER).read_text()
    decorated = runner.replace("def action_dry_run(args):", "def _wrap(function):\n    return function\n\n\n"
                                                            "@_wrap\ndef action_dry_run(args):")
    _commit(repo, {freeze.R_CMD_RUNNER: decorated})
    problems = freeze.r_cmd_problems({0: dry_run_head}, repo)
    assert problems == [f"the pinned runner function action_dry_run changed after the dry run at {dry_run_head[:12]}"]
    hashes, _problems = freeze.runner_function_sha256(decorated)
    plain = "@_wrap\ndef action_dry_run(args):\n    return 11"
    assert hashes["action_dry_run"] == provenance.sha256_bytes(plain.encode())


@pytest.mark.parametrize("addition, expected", [
    ("\nif True:\n    def _csv(args):\n        return 99\n",
     "the runner defines the pinned name _csv inside a top-level block (tree)"),
    ("\ntry:\n    import os\nexcept ImportError:\n    _layout = None\n",
     "the runner assigns the pinned name _layout at top level (tree)"),
    ("\nwith open(__file__) as action_plan:\n    pass\n",
     "the runner assigns the pinned name action_plan at top level (tree)"),
    ("\nfor _build in ():\n    pass\n", "the runner assigns the pinned name _build at top level (tree)"),
    ("\ndel _git_for\n", "the runner deletes the pinned name _git_for at top level (tree)"),
    ("\n_csv, other = 1, 2\n", "the runner assigns the pinned name _csv at top level (tree)"),
    ("\nif True:\n    from os.path import join as _load_view\n",
     "the runner imports the pinned name _load_view at top level (tree)"),
])
def test_r_cmd_refuses_a_pinned_name_rebound_inside_a_top_level_block(tmp_path, addition, expected):
    repo, dry_run_head = rcmd_repo(tmp_path)
    runner = (repo / freeze.R_CMD_RUNNER).read_text()
    _commit(repo, {freeze.R_CMD_RUNNER: runner + addition})
    problems = freeze.r_cmd_problems({0: dry_run_head}, repo)
    assert expected in problems, problems


def test_runner_function_hash_ignores_names_local_to_functions():
    text = RUNNER + "\n\ndef helper(args):\n    _csv = 1\n    return _csv\n\n\nclass Holder:\n    _layout = 2\n"
    _hashes, problems = freeze.runner_function_sha256(text)
    assert problems == []


def test_r_cmd_allows_the_listed_modifications_new_scripts_and_free_folders(tmp_path):
    repo = tmp_path / "repo"
    listed = ["src/ec_clean30/stats.py", "src/ec_clean30/scoring.py", "src/ec_clean30/simulation.py",
              "scripts/run_ec_clean30_baseline.py", "scripts/run_clean_split30_comparator.py"]
    dry_run_head = _init(repo, {**BASE_FILES, **{path: "VALUE = 1\n" for path in listed}})
    later = _commit(repo, {**{path: "VALUE = 2\n" for path in listed}, "scripts/run_ec_clean30_phase3.py": "X = 1\n",
                           "tools/check.py": "VALUE = 1\n", "docs/notes.md": "notes\n"})
    result = freeze.r_cmd_check({0: dry_run_head}, repo)
    assert result["problems"] == [] and result["head"] == later
    assert {entry["path"] for entry in result["changes"][dry_run_head] if entry["status"] == "M"} == set(listed)
    assert result["added_files"] == ["scripts/run_ec_clean30_phase3.py"]  # tests/, docs/ and tools/ are free


def test_r_cmd_refuses_a_new_module_loaded_by_a_protected_module(tmp_path):
    repo, dry_run_head = rcmd_repo(tmp_path)
    # selection.py may change (allowlist), but campaign imports it, so it may not import a new module.
    _commit(repo, {"src/ec_clean30/selection.py": "from ec_clean30 import freeze  # noqa: F401\nVALUE = 2\n"})
    problems = freeze.r_cmd_problems({0: dry_run_head}, repo)
    assert problems == ["src/ec_clean30/freeze.py, added after the dry run, is loaded when the protected modules are "
                        "imported"]


def test_r_cmd_reports_a_protected_module_that_does_not_import(tmp_path):
    repo, dry_run_head = rcmd_repo(tmp_path)
    _commit(repo, {"src/ec_clean30/selection.py": "raise ImportError('broken')\n"})
    problems = freeze.r_cmd_problems({0: dry_run_head}, repo)
    assert len(problems) == 1 and "could not import the protected modules" in problems[0]


def test_r_cmd_refuses_unknown_or_non_ancestor_dry_run_heads(tmp_path):
    repo, dry_run_head = rcmd_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "side")
    side = _commit(repo, {"docs/side.md": "side\n"})
    _git(repo, "checkout", "-q", "main")
    problems = freeze.r_cmd_problems({0: side, 1: "f" * 40, 2: "abc"}, repo)
    assert f"the dry-run HEAD {side[:12]} is not an ancestor of HEAD" in problems
    assert sum("is not a full commit SHA" in problem for problem in problems) == 2
    assert freeze.r_cmd_problems({}, repo) == ["no dry-run HEAD was given"]


def test_r_cmd_refuses_a_plan_that_differs_from_its_record(tmp_path):
    repo, dry_run_head = rcmd_repo(tmp_path)
    changed = write_plans_dir(tmp_path / "changed_plans", change_fold=3)
    problems = freeze.r_cmd_problems({0: dry_run_head}, repo, changed)
    assert "fold 3: the live plan SHA-256 differs from the record in --plans-dir" in problems
    assert all(problem.startswith("fold 3:") for problem in problems)
    missing = write_plans_dir(tmp_path / "missing_plans")
    (missing / "fold1.json.receipt.json").unlink()
    assert freeze.r_cmd_problems({0: dry_run_head}, repo, missing) == [
        "fold 1: fold1.json or its .receipt.json is not a regular file in --plans-dir"]


def test_pinned_runner_functions_of_the_working_tree_match_head():
    """The runner of this checkout: its twelve pinned functions are those of HEAD (no runner change touched them)."""
    code, committed = provenance._git(ROOT, "show", f"HEAD:{freeze.R_CMD_RUNNER}")
    if code != 0:
        pytest.skip("not a git checkout")
    live, live_problems = freeze.runner_function_sha256((ROOT / freeze.R_CMD_RUNNER).read_text(encoding="utf-8"))
    head, head_problems = freeze.runner_function_sha256(committed)
    assert live_problems == head_problems == [] and live == head
    assert all(value is not None for value in live.values())


def test_protected_modules_never_load_this_module_and_it_loads_no_torch():
    files, problem = freeze.protected_import_files(ROOT)
    assert problem is None
    assert {"src/ec_clean30/campaign.py", "src/ec_clean30/identity.py"} <= files
    assert "src/ec_clean30/freeze.py" not in files
    code = "\n".join([
        "import sys",
        "sys.path.insert(0, 'src')",
        "import ec_clean30.campaign, ec_clean30.commands, ec_clean30.views, ec_clean30.provenance, ec_clean30.identity",
        "print('ec_clean30.freeze' in sys.modules)",
        "import ec_clean30.freeze",
        "names = ('torch', 'ec_clean30.simulation', 'ec_clean30.scoring')",
        "print(sorted(name for name in names if name in sys.modules))",
    ])
    env = {**os.environ, "MKL_THREADING_LAYER": "GNU"}
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env,
                               timeout=300)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.splitlines() == ["False", "[]"]


def test_tracked_code_pins_every_tracked_src_and_scripts_file(tmp_path):
    repo, _head = rcmd_repo(tmp_path)
    _write(repo, {"src/untracked.py": "VALUE = 1\n"})
    pins = freeze.tracked_code_sha256(repo)
    assert "src/untracked.py" not in pins and "tests/test_stand_in.py" not in pins and LOG not in pins
    assert pins["src/ec_clean30/freeze.py"] == sha256_file(repo / "src/ec_clean30/freeze.py")
    assert freeze.R_CMD_RUNNER in pins and "src/training/loop.py" in pins


def test_constants_match_the_modules_they_mirror():
    from ec_clean30 import scoring, simulation, transfers

    assert freeze.PLACEHOLDER_CHARACTERS == scoring._PLACEHOLDER_CHARACTERS
    assert freeze.SIMULATION_RECEIPT_FORMAT == simulation.RECEIPT_FORMAT
    assert freeze.SIMULATION_RECORD_FORMAT == simulation.RECORD_FORMAT
    assert freeze.SIMULATION_FILES == {"sweep": simulation.SWEEP_FILE, "sweep_receipt": simulation.RECEIPT_FILE,
                                       "summary": simulation.SUMMARY_FILE}
    assert freeze.TRANSFER_LIST_FILES == {"transfer_list": f"ec_transfer_list/{transfers.OUTPUT_NAME}",
                                          "transfer_list_receipt": f"ec_transfer_list/{transfers.RECEIPT_NAME}"}
    assert set(freeze.IDENTITY_CODE_FILES) == set(identity.code_sha256()) and set(freeze.IDENTITY_CODE_FILES) <= set(
        freeze.R_ID_FILES)
    from ec_clean30 import cohort, r2_lists

    assert freeze.COHORT_FILES == {"cohort_audit_receipt": cohort.RECEIPT_NAME}
    assert freeze.R2_FILES == {"r2_lists_receipt": r2_lists.RECEIPT_NAME}
    assert issubclass(Freeze1Error, provenance.FreezeManifestError)
    assert len(freeze.REQUIRED_DECISION_IDS) == len(set(freeze.REQUIRED_DECISION_IDS)) == 129
    assert {"D1", "D7", "G8", "BP item 20", "Map run", "U17", "P2-9"} <= set(freeze.REQUIRED_DECISION_IDS)
    assert "G4" in freeze.ROUND_1A_CONFIRMATION_IDS and "G4" not in freeze.ROUND_2_IDS


def _manifest_args(**values) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=freeze.MANIFEST_KINDS, required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--view", action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    freeze.add_manifest_freeze_arguments(parser)
    args = parser.parse_args(["--kind", values.pop("kind")])
    for name, value in values.items():
        setattr(args, name, value)
    return args


def test_route_manifest_keeps_the_runner_paths_and_never_skips_the_completeness_checks():
    assert freeze.route_manifest(_manifest_args(kind="freeze1", smoke=True)) is None
    assert freeze.route_manifest(_manifest_args(kind="freeze2")) is None
    assert freeze.route_manifest(_manifest_args(kind="freeze1")) is freeze.action_manifest_freeze1
    assert freeze.route_manifest(_manifest_args(kind="freeze1", plans_dir="p")) is freeze.action_manifest_freeze1
    assert freeze.route_manifest(_manifest_args(kind="freeze1-document")) is freeze.action_manifest_freeze1_document
    assert freeze.route_manifest(_manifest_args(kind="freeze1b")) is freeze.action_manifest_freeze1b
    with pytest.raises(Freeze1Error, match="--kind freeze1 --smoke does not take --identity-root"):
        freeze.route_manifest(_manifest_args(kind="freeze1", smoke=True, identity_root="x"))
    with pytest.raises(Freeze1Error, match="--kind freeze2 does not take --pin --plans-dir"):
        freeze.route_manifest(_manifest_args(kind="freeze2", plans_dir="p", pin=["a=b"]))
    with pytest.raises(Freeze1Error, match="--kind freeze1b does not take --decisions"):
        freeze.route_manifest(_manifest_args(kind="freeze1b", decisions="d.json"))
    with pytest.raises(Freeze1Error, match="--kind freeze1b does not take --no-transfer-list"):
        freeze.route_manifest(_manifest_args(kind="freeze1b", no_transfer_list=True))
    with pytest.raises(Freeze1Error, match="needs --plans-dir"):
        freeze.action_manifest_freeze1(_manifest_args(kind="freeze1", out="o.json"))
    with pytest.raises(Freeze1Error, match="smoke manifest takes one view and no --plans-dir"):
        freeze.action_manifest_freeze1(_manifest_args(kind="freeze1", smoke=True, plans_dir="p", out="o.json"))


# ---------------------------------------------------------------------------
# Decisions record


def decisions_record(**changes) -> dict:
    record = {"format": freeze.FREEZE1_DECISIONS_FORMAT, "campaign_id": CAMPAIGN_ID,
              "provisional_markers_settled": True,
              "decisions": {decision_id: {"answer": "Confirmed as recommended in the round sheet.", "round": "r"}
                            for decision_id in freeze.REQUIRED_DECISION_IDS}}
    record["decisions"]["E6"] = "The maintainer's full name."
    record["decisions"]["A8-b"] = {"answer": "Comma-separated, as the scorer reads it."}
    record.update(changes)
    return record


def test_a_complete_decisions_record_passes():
    assert freeze.decision_problems(decisions_record()) == []


@pytest.mark.parametrize("decision_id", [
    "SA1", "SA1-c", "P2-L5", "P2-9-A1", "A8-a", "A8-b", "A8-c", "P2-9b",
    *(f"C-{index}" for index in range(1, 15)),
])
def test_amendments_and_final_implementation_choices_are_required(decision_id):
    record = decisions_record()
    record["decisions"].pop(decision_id)
    assert f"decision {decision_id} is missing" in freeze.decision_problems(record)


def test_original_lever_decisions_do_not_replace_final_implementation_choices():
    record = decisions_record()
    record["decisions"].pop("C-1")
    assert "C1" in record["decisions"]
    assert "decision C-1 is missing" in freeze.decision_problems(record)
    record = decisions_record()
    record["decisions"].pop("C1")
    assert "C-1" in record["decisions"]
    assert "decision C1 is missing" in freeze.decision_problems(record)


@pytest.mark.parametrize("change, expected", [
    (lambda record: record["decisions"].pop("D3"), "decision D3 is missing"),
    (lambda record: record["decisions"].update({"F6": {"answer": "TBD"}}), "decision F6 is unanswered"),
    (lambda record: record["decisions"].update({"A8-a": "  "}), "decision A8-a is unanswered"),
    (lambda record: record["decisions"].update({"E6": "<name>"}),
     "the decisions record at decisions.E6 holds a placeholder character"),
    (lambda record: record["decisions"].update({"B1": {"answer": "bins as coded", "note": "see {bins}"}}),
     "the decisions record at decisions.B1.note holds a placeholder"),
    (lambda record: record["decisions"].update({"D4": "Approved by [the maintainer]"}),
     "the decisions record at decisions.D4 holds a placeholder"),
    (lambda record: record["decisions"].update({"D5": {"answer": "Confirmed; wording TBD"}}),
     "the decisions record at decisions.D5.answer holds a placeholder"),
    (lambda record: record["decisions"].update({"E6": "A. Name {2026}"}),
     "the decisions record at decisions.E6 holds a placeholder character"),
    (lambda record: record["decisions"].update({"E6": {"answer": "A. Name (lab [x])"}}),
     "the decisions record at decisions.E6.answer holds a placeholder character"),
    (lambda record: record["decisions"].update({"M2": "Mounted at Data1:/media/someone/Data1"}),
     "the decisions record at decisions.M2 holds an absolute path"),
    (lambda record: record["decisions"].update({"M4": "see file:///home/someone/x"}),
     "the decisions record at decisions.M4 holds an absolute path"),
    (lambda record: record["decisions"].update({"M3": "Fits under /media/someone/Data1/campaign"}),
     "the decisions record at decisions.M3 holds an absolute path"),
    (lambda record: record["decisions"].update({"M5": "copied from ~/data/views"}),
     "the decisions record at decisions.M5 holds an absolute path"),
    (lambda record: record.update({"provisional_markers_settled": False}),
     "the decisions record does not state provisional_markers_settled: true"),
    (lambda record: record.update({"format": "other"}),
     f"the decisions record's format is not {freeze.FREEZE1_DECISIONS_FORMAT}"),
])
def test_decisions_refusals(change, expected):
    record = decisions_record()
    change(record)
    assert expected in freeze.decision_problems(record)


def test_recommended_answers_with_set_and_interval_notation_pass():
    """The plan's recommended answers for C10, F9 and B1 use braces and brackets; they are not placeholders."""
    record = decisions_record()
    record["decisions"].update({
        "C10": {"answer": "τ ∈ {−1, −0.5, 0, 0.5, 1}, add-one smoothing, per-fit priors; ties to smaller |τ|"},
        "F9": "Every CLEAN call passes `--forbidden-root` for `$C/{identity,identity_stdout,labels/label_table}`",
        "B1": "no_hit, ≤20, (20,25], (25,30), [30,50], >50; confirmed",
        "D2": "Strictly a <= b and c >= d; x < y, z > w",
    })
    assert freeze.decision_problems(record) == []


def test_relative_paths_and_plain_slashes_are_not_absolute():
    text = {"a": "freeze/freeze1_inputs.json", "b": "identical columns / alignment columns", "c": "1/3 and/or",
            "d": "https://ftp.example.org/enzyme.dat", "e": "ratio 3:1, time 10:30, key: value/other"}
    assert freeze.absolute_path_problems(text, "x") == []
    assert freeze.absolute_path_problems({"f": "Data1:/media/x", "g": "FILE:///home/x"}, "x") == [
        "x f holds an absolute path", "x g holds an absolute path"]
    assert freeze.absolute_path_problems({"e": "(at C:\\data\\x)"}, "x") == ["x e holds an absolute path"]


# ---------------------------------------------------------------------------
# verify-freeze1


def verify_repo(base: Path, *, log_hash: bool = True) -> tuple[Path, str, str]:
    """A repository whose second commit adds the document and (by default) its SHA-256 to the log."""
    repo = base / "repo"
    first = _init(repo, {LOG: "# Campaign log\n\nIdentity chain at `PLACEHOLDER`.\n"})
    document = json.dumps({"format": "synthetic"}, sort_keys=True) + "\n"
    digest = provenance.sha256_bytes(document.encode())
    log = f"# Campaign log\n\nIdentity chain at `{first[:7]}`.\n\n## evc-008\n\nFreeze 1 SHA-256 "
    log += f"`{digest}`.\n" if log_hash else "pending.\n"
    _commit(repo, {DOCUMENT: document, LOG: log}, "freeze 1")
    return repo, digest, first


def test_verify_freeze1_passes_and_records_push_state(tmp_path):
    repo, digest, _first = verify_repo(tmp_path)
    introducing = _git(repo, "rev-parse", "HEAD")
    record = freeze.verify_freeze1(sha256=digest, document=repo / DOCUMENT, require_pushed=False, repo=repo)
    assert record["status"] == "passed" and record["problems"] == []
    assert record["introducing_commit"] == introducing and record["pushed"] is False
    assert record["origin_main"] == {"ref": freeze.ORIGIN_MAIN_REF, "commit": None,
                                     "contains_introducing_commit": False, "fetched": False}
    assert record["document"] == DOCUMENT and record["introducing_commit_time"]
    refused = freeze.verify_freeze1(sha256=digest, document=repo / DOCUMENT, require_pushed=True, repo=repo)
    assert refused["status"] == "failed" and "does not contain the commit" in refused["problems"][0]
    _git(repo, "update-ref", freeze.ORIGIN_MAIN_REF, introducing)
    _commit(repo, {"docs/later.md": "later\n"})
    pushed = freeze.verify_freeze1(sha256=digest, document=repo / DOCUMENT, require_pushed=True, repo=repo)
    assert pushed["status"] == "passed" and pushed["pushed"] is True
    assert pushed["origin_main"]["commit"] == introducing and pushed["introducing_commit"] == introducing


def test_verify_freeze1_refuses_an_absent_hash(tmp_path):
    repo, digest, _first = verify_repo(tmp_path, log_hash=False)
    record = freeze.verify_freeze1(sha256=digest, document=repo / DOCUMENT, require_pushed=False, repo=repo)
    assert record["status"] == "failed" and record["problems"] == [f"the SHA-256 does not appear in {LOG} at HEAD"]
    record = freeze.verify_freeze1(sha256=OTHER_F, document=repo / DOCUMENT, require_pushed=False, repo=repo)
    assert "the document does not hash to the given SHA-256" in record["problems"]


def test_verify_freeze1_refuses_a_hash_logged_in_a_non_ancestor_commit(tmp_path):
    repo, digest, _first = verify_repo(tmp_path, log_hash=False)
    _git(repo, "checkout", "-q", "-b", "side")
    log = (repo / LOG).read_text().replace("pending.", f"`{digest}`.")
    side = _commit(repo, {LOG: log})
    _git(repo, "checkout", "-q", "main")
    _git(repo, "update-ref", freeze.ORIGIN_MAIN_REF, side)
    record = freeze.verify_freeze1(sha256=digest, document=repo / DOCUMENT, require_pushed=True, repo=repo)
    assert record["status"] == "failed" and record["introducing_commit"] is None
    assert f"the SHA-256 does not appear in {LOG} at HEAD" in record["problems"]


def test_verify_freeze1_needs_the_document_committed_with_these_bytes(tmp_path):
    repo, digest, _first = verify_repo(tmp_path)
    copy = tmp_path / "outside.json"
    shutil.copyfile(repo / DOCUMENT, copy)
    record = freeze.verify_freeze1(sha256=digest, document=copy, require_pushed=False, repo=repo)
    assert record["problems"] == ["the document is not a file of this repository"]
    with pytest.raises(Freeze1Error, match="64 hexadecimal"):
        freeze.verify_freeze1(sha256="abc", document=repo / DOCUMENT, require_pushed=False, repo=repo)


def test_verify_freeze1_action_writes_a_receipt_only_when_it_passes(tmp_path, monkeypatch, capsys):
    repo, digest, _first = verify_repo(tmp_path)
    monkeypatch.setattr(freeze, "REPO_ROOT", repo)
    parser = freeze.add_verify_freeze1_arguments(argparse.ArgumentParser())
    out = tmp_path / "receipts" / "verify_freeze1.json"
    args = parser.parse_args(["--sha256", digest, "--document", str(repo / DOCUMENT), "--require-pushed",
                              "--out", str(out)])
    assert freeze.action_verify_freeze1(args) == 1 and not out.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    _git(repo, "update-ref", freeze.ORIGIN_MAIN_REF, "HEAD")
    assert freeze.action_verify_freeze1(args) == 0
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "passed" and receipt["pushed"] is True and receipt["freeze1_sha256"] == digest
    assert receipt["provenance"]["git"]["head"] == _git(repo, "rev-parse", "HEAD")
    with pytest.raises(FileExistsError):
        freeze.action_verify_freeze1(args)


# ---------------------------------------------------------------------------
# The campaign world: overlay, five fold views, plans, synthetic dry runs


def _overlay_layout(root: Path) -> SourceLayout:
    return SourceLayout.from_data_root(root, esm_dir=root / "esm", external_root=root / "external")


def write_dry_run_receipt(view_root: Path, head: str) -> None:
    view = View.load(view_root)
    final = {"manifest_sha256": view.chain_means_manifest_sha256("final")}
    receipt = {"status": "passed", "view_manifest_sha256": view.manifest_sha256, "verified_view": True,
               "cap_prefix": ["systemd-run", "--user", "--scope"], "verification": {"chain_means": {"final": final}},
               "provenance": {"git": {"head": head}}}
    (view.receipts / views.DRY_RUN_RECEIPT_NAME).write_text(json.dumps(receipt, sort_keys=True))


def write_bound_inputs(campaign_root: Path) -> None:
    """Label, simulation and environment records (synthetic), with class numbers that must never be copied."""
    class_numbers = {"eligible_by_ec_class": {"1": 11, "2": 13}, "ec_class_counts": {"1": 7},
                     "proportions": {"1": 0.123456789}}
    labels = campaign_root / "labels"
    _write(labels, {
        "label_table/native_label_table.tsv": "accession\tec_numbers\nP1000\t1.1.1.1\n",
        "label_table/label_table_receipt.json": json.dumps({"counts": class_numbers}),
        "sequence_map/sequence_map.tsv": "accession\tsha256\tlength\tn_files\tsequence\n",
        "sequence_map/sequence_map_receipt.json": json.dumps({"n_rows": 1}),
        "ec_transfer_list/ec_transfer_list.csv": "ec_number,transferred_to\n",
        "ec_transfer_list/transfer_list_receipt.json": json.dumps({"counts": {"rows": 0}}),
    })
    write_simulation(campaign_root / "simulation" / "final_v1", class_numbers)
    _write(campaign_root / "freeze" / "env", {"env_conda_explicit.txt": "@EXPLICIT\n",
                                              "env_pip_freeze.txt": "numpy==1.26.4\n"})


def write_simulation(simulation: Path, class_numbers: dict, *, smoke: bool = False) -> Path:
    """A synthetic final sweep folder: sweep, summary, one record and the receipt (with class numbers)."""
    record = {"format": freeze.SIMULATION_RECORD_FORMAT, "point_id": "p1", "smoke": smoke,
              "result": {"calibration": class_numbers}}
    _write(simulation, {"sweep.json": "{}\n", "summary.tsv": "point\tadoption\n",
                        "records/p1.json": json.dumps(record, sort_keys=True) + "\n"})
    receipt = {"format": freeze.SIMULATION_RECEIPT_FORMAT,
               "sweep_file_sha256": sha256_file(simulation / "sweep.json"),
               "summary": {"file": "summary.tsv", "sha256": sha256_file(simulation / "summary.tsv")},
               "records": {"p1": {"file": "records/p1.json", "sha256": sha256_file(simulation / "records/p1.json")}},
               "calibrations": {"0": {"class_proportions": [0.987654321], **class_numbers}},
               "provenance": {"smoke": smoke}}
    _write(simulation, {"sweep_receipt.json": json.dumps(receipt)})
    return simulation


def build_freeze_world(base: Path) -> dict:
    world = build_world(base)
    campaign_root = world.campaign
    repo, dry_run_head = rcmd_repo(base)
    overlay_root = campaign_root / "overlay" / DEFAULT_OVERLAY_ID
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(identity, "MMSEQS_BINARY_SHA256", world.binary_sha)
        build_overlay(out_dir=overlay_root, cohort_dir=world.cohort, identity_root=world.identity_root,
                      r2_lists_dir=world.r2, d2_option="d", provenance=REAL_PROVENANCE)
    for fold in range(N_OUTER_FOLDS):
        work = overlay_root / "identity" / f"fold{fold}"
        output = json.loads((work / identity.INNER_ASSIGNMENT_RECEIPT).read_text())["outputs"]["membership"]
        view_root = campaign_root / views.VIEWS_DIR / f"fold{fold}"
        build_view(fold=fold, layout=_overlay_layout(overlay_root), out_dir=view_root,
                   inner_membership=(work / output["file"], output["sha256"]), provenance=REAL_PROVENANCE)
        build_chain_means(view_root)
        write_dry_run_receipt(view_root, dry_run_head)
    write_plans_dir(campaign_root / "plans")
    write_bound_inputs(campaign_root)
    decisions = repo / freeze.CAMPAIGN_DOCS_DIR / "freeze1_decisions.json"
    _commit(repo, {decisions.relative_to(repo).as_posix(): json.dumps(decisions_record(), indent=2) + "\n"},
            "decisions")
    return {"world": world, "campaign": campaign_root, "overlay": overlay_root, "repo": repo,
            "dry_run_head": dry_run_head, "decisions": decisions,
            "views": [campaign_root / views.VIEWS_DIR / f"fold{fold}" for fold in range(N_OUTER_FOLDS)]}


@pytest.fixture(scope="module")
def freeze_world(tmp_path_factory):
    return build_freeze_world(tmp_path_factory.mktemp("freezeworld"))


@pytest.fixture(scope="module")
def inputs(freeze_world):
    """The real inputs manifest, written through the action (the clean-tree check runs on the small repository)."""
    world = freeze_world
    out = world["campaign"] / "freeze" / "freeze1_inputs.json"
    args = argparse.Namespace(kind="freeze1", plans_dir=str(world["campaign"] / "plans"),
                              view=[str(path) for path in world["views"]], out=str(out), smoke=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freeze, "REPO_ROOT", world["repo"])
        assert freeze.action_manifest_freeze1(args) == 0
    return {"path": out, "sha256": sha256_file(out), "manifest": json.loads(out.read_text())}


def test_real_freeze1_inputs_record_completeness(freeze_world, inputs):
    manifest = inputs["manifest"]
    completeness = manifest["completeness"]
    assert manifest["format"] == provenance.FREEZE1_FORMAT and manifest["smoke"] is False
    assert completeness["status"] == "passed" and completeness["folds"] == [0, 1, 2, 3, 4]
    assert completeness["campaign_root"] == str(freeze_world["campaign"])
    assert completeness["overlay"]["id"] == DEFAULT_OVERLAY_ID
    assert completeness["overlay"]["root"] == f"overlay/{DEFAULT_OVERLAY_ID}"
    assert completeness["file_modes_enforced"] == {str(fold): True for fold in range(N_OUTER_FOLDS)}
    assert completeness["r_cmd"]["status"] == "passed"
    assert set(completeness["r_cmd"]["dry_run_heads"].values()) == {freeze_world["dry_run_head"]}
    assert completeness["promotion"]["PROMOTION_ALPHA"] == 0.04
    assert completeness["promotion"]["promotion_config_by_lever"]["L9"]["alpha"] == 0.01
    assert completeness["promotion"]["promotion_config_by_lever"]["L7"]["alpha"] == 0.04
    assert completeness["tracked_code_sha256"] == freeze.tracked_code_sha256(freeze_world["repo"])
    for fold in range(N_OUTER_FOLDS):
        work = freeze_world["overlay"] / "identity" / f"fold{fold}"
        output = json.loads((work / identity.INNER_ASSIGNMENT_RECEIPT).read_text())["outputs"]["membership"]
        record = completeness["inner_membership"][str(fold)]
        assert record["membership_sha256"] == output["sha256"] and record["source_name"] == output["file"]
        assert record["certificate"]["sha256"] == sha256_file(work / identity.INNER_CERTIFICATE)
        assert record["certificate"]["path"] == (f"overlay/{DEFAULT_OVERLAY_ID}/identity/fold{fold}/"
                                                 f"{identity.INNER_CERTIFICATE}")
        assert manifest["views"][fold]["membership"]["certificate_sha256"] == record["certificate"]["sha256"]
    assert [view["fold"] for view in manifest["views"]] == [0, 1, 2, 3, 4]


def test_real_freeze1_needs_exactly_folds_zero_to_four(freeze_world):
    roots = freeze_world["views"]
    for chosen in (roots[:4], [roots[0], *roots[:4]]):
        with pytest.raises(Freeze1Error, match="exactly one view per fold"):
            freeze.build_freeze1_inputs(chosen, plans_dir=freeze_world["campaign"] / "plans",
                                        repo=freeze_world["repo"])


def _patched_load(monkeypatch, fold: int, change):
    original = View.load

    def load(cls, root):
        view = original(root)
        if view.fold != fold:
            return view
        manifest = json.loads(json.dumps(view.manifest))
        change(manifest)
        return replace(view, manifest=manifest)

    monkeypatch.setattr(View, "load", classmethod(load))


@pytest.mark.parametrize("fold, change, expected", [
    (3, lambda manifest: manifest["overlay"].update(manifest_sha256="0" * 64), "different overlays"),
    (2, lambda manifest: manifest.update(overlay=None), "not built from a data overlay"),
    (2, lambda manifest: manifest["membership"].update(sha256="0" * 64),
     "fold 2: the view's inner membership SHA-256 differs from the assignment receipt's output"),
    (1, lambda manifest: manifest["membership"].update(certificate_sha256="0" * 64),
     "fold 1: the view's inner certificate differs from the overlay's identity copy"),
    (4, lambda manifest: manifest["membership"].update(source_name="other.csv"),
     "fold 4: the view's inner membership source 'other.csv' is not the assignment receipt's output"),
    (0, lambda manifest: manifest.update(membership={"role": "smoke"}), "fold 0: the view has no inner membership"),
])
def test_real_freeze1_completeness_refusals(freeze_world, monkeypatch, fold, change, expected):
    _patched_load(monkeypatch, fold, change)
    with pytest.raises(Freeze1Error) as refused:
        freeze.build_freeze1_inputs(freeze_world["views"], plans_dir=freeze_world["campaign"] / "plans",
                                    repo=freeze_world["repo"])
    assert expected in str(refused.value)


@pytest.mark.parametrize("change, expected", [
    (lambda manifest: manifest.update(smoke=True), "the overlay is an engineering smoke"),
    (lambda manifest: manifest.pop("smoke"), "the overlay is an engineering smoke"),
    (lambda manifest: manifest["inputs"]["r2_lists"].update(smoke=True),
     "the overlay's r2_lists input is not recorded as real"),
    (lambda manifest: manifest["inputs"]["cohort"].update(smoke=None),
     "the overlay's cohort input is not recorded as real"),
])
def test_real_freeze1_refuses_a_smoke_overlay(freeze_world, monkeypatch, change, expected):
    from ec_clean30 import overlay

    original = overlay.read_overlay_manifest

    def smoke_manifest(root):
        manifest, digest = original(root)
        manifest = json.loads(json.dumps(manifest))
        change(manifest)
        return manifest, digest

    monkeypatch.setattr(overlay, "read_overlay_manifest", smoke_manifest)
    with pytest.raises(Freeze1Error) as refused:
        freeze.build_freeze1_inputs(freeze_world["views"], plans_dir=freeze_world["campaign"] / "plans",
                                    repo=freeze_world["repo"])
    assert "needs a real (non-smoke) overlay" in str(refused.value) and expected in str(refused.value)


def test_the_world_overlay_is_real(freeze_world):
    from ec_clean30.overlay import read_overlay_manifest

    manifest, _digest = read_overlay_manifest(freeze_world["overlay"])
    assert freeze.overlay_smoke_problems(manifest) == []


def test_real_freeze1_refuses_a_campaign_identity_path(freeze_world):
    with pytest.raises(Freeze1Error, match=r"--plans-dir lies under <campaign root>/identity"):
        freeze.build_freeze1_inputs(freeze_world["views"], plans_dir=freeze_world["campaign"] / "identity" / "plans",
                                    repo=freeze_world["repo"])
    problem = freeze.campaign_identity_problem(freeze_world["campaign"] / "identity" / "fold0" / "x.json",
                                               freeze_world["campaign"], "a path")
    assert problem is not None and "identity" in problem


def test_real_freeze1_refuses_an_r_cmd_violation(freeze_world, tmp_path):
    repo = _copy_repo(freeze_world["repo"], tmp_path / "repo")
    _commit(repo, {"src/ec_clean30/views.py": "VALUE = 5\n"})
    with pytest.raises(Freeze1Error) as refused:
        freeze.build_freeze1_inputs(freeze_world["views"], plans_dir=freeze_world["campaign"] / "plans", repo=repo)
    assert freeze.RERUN_DRY_RUNS in str(refused.value) and "src/ec_clean30/views.py was modified" in str(refused.value)


def test_real_freeze1_refuses_a_view_without_its_dry_run(freeze_world, monkeypatch):
    def without_dry_run(view_root, *, require_dry_run, provenance=None):
        raise views.ViewError("the view has no dry_run_receipt.json; run dry-run first")

    monkeypatch.setattr(views, "freeze1_view_record", without_dry_run)
    with pytest.raises(Freeze1Error, match="run dry-run first"):
        freeze.build_freeze1_inputs(freeze_world["views"], plans_dir=freeze_world["campaign"] / "plans",
                                    repo=freeze_world["repo"])


def test_inner_membership_check_on_the_overlay_copies(freeze_world):
    from types import SimpleNamespace

    from ec_clean30.overlay import read_overlay_manifest

    overlay_root = freeze_world["overlay"]
    manifest, _sha = read_overlay_manifest(overlay_root)
    files = freeze._overlay_files(manifest)
    view = View.load(freeze_world["views"][2])
    record, problems = freeze.inner_membership_problems(view, overlay_root, files, freeze_world["campaign"])
    assert problems == [] and record["assignment_receipt"]["path"].endswith("identity/fold2/"
                                                                            "inner_assignment_receipt.json")
    tampered = dict(files)
    tampered[f"identity/fold2/{identity.INNER_CERTIFICATE}"] = "0" * 64
    other = SimpleNamespace(fold=2, membership=view.membership)
    _record, problems = freeze.inner_membership_problems(other, overlay_root, tampered, freeze_world["campaign"])
    assert problems == ["fold 2: the overlay manifest does not record identity/fold2/inner_certification_receipt.json "
                        "with its SHA-256"]


# ---------------------------------------------------------------------------
# Freeze-1 document


def document_options(world: dict, written: dict, /, **changes) -> dict:
    campaign_root = world["campaign"]
    options = {
        "inputs": written["path"], "inputs_sha256": written["sha256"], "decisions": world["decisions"],
        "overlay_manifest": world["overlay"] / "overlay_manifest.json", "labels": campaign_root / "labels",
        "cohort": campaign_root / "cohort", "r2_lists": campaign_root / "r2_lists",
        "simulation": campaign_root / "simulation" / "final_v1", "environment": campaign_root / "freeze" / "env",
        "pins": ["clean_patch_manifest=f331019d" + "1" * 56, "clean_environment_lock=" + "2" * 64,
                 "esmc_300m_weights=" + "3" * 64, f"mmseqs2_binary={identity.MMSEQS_BINARY_SHA256}",
                 "clean_private_script.train=" + "4" * 64],
        "repo": world["repo"],
    }
    options.update(changes)
    return options


@pytest.fixture(scope="module")
def document(freeze_world, inputs):
    return freeze.build_freeze1_document(**document_options(freeze_world, inputs))


def test_document_binds_records_by_sha256_with_relative_paths(freeze_world, inputs, document):
    campaign_root = freeze_world["campaign"]
    assert document["format"] == freeze.FREEZE1_DOCUMENT_FORMAT and document["amendments"] == []
    assert document["inputs_manifest"] == {"path": "freeze/freeze1_inputs.json", "sha256": inputs["sha256"]}
    assert document["decisions"]["path"] == f"{freeze.CAMPAIGN_DOCS_DIR}/freeze1_decisions.json"
    assert set(document["decisions"]["answers"]) >= set(freeze.REQUIRED_DECISION_IDS)
    assert document["overlay"]["manifest"]["path"] == f"overlay/{DEFAULT_OVERLAY_ID}/overlay_manifest.json"
    bound = document["bound_files"]
    assert bound["label_table_receipt"] == {"path": "labels/label_table/label_table_receipt.json",
                                            "sha256": sha256_file(campaign_root / "labels/label_table/"
                                                                                  "label_table_receipt.json")}
    assert bound["cohort_audit_receipt"]["path"] == "cohort/cohort_audit_receipt.json"
    assert bound["r2_lists_receipt"]["path"] == "r2_lists/r2_lists_receipt.json"
    assert bound["simulation_sweep_receipt"]["path"] == "simulation/final_v1/sweep_receipt.json"
    assert bound["environment_pip_freeze"]["path"] == "freeze/env/env_pip_freeze.txt"
    assert bound["transfer_list"]["path"] == "labels/ec_transfer_list/ec_transfer_list.csv"
    assert document["transfer_list"] == "bound" and document["simulation_records"] == {"n_records": 1}
    assert document["pins"]["mmseqs2_binary"] == identity.MMSEQS_BINARY_SHA256
    assert "clean_private_script.train" in document["pins"]
    assert document["inner_membership"] == inputs["manifest"]["completeness"]["inner_membership"]
    assert document["code"]["tracked_file_sha256"] == inputs["manifest"]["completeness"]["tracked_code_sha256"]
    constants = document["constants"]
    assert constants["selection"]["PROMOTION_ALPHA"] == 0.04 and constants["selection"]["L9_PROMOTION_ALPHA"] == 0.01
    assert constants["selection"]["LEVER_PROMOTION_ALPHA"] == {"L9": 0.01}
    assert constants["promotion"]["promotion_config_by_lever"]["L9"]["alpha"] == 0.01
    assert constants["promotion"]["promotion_config_default"]["alpha"] == 0.04
    assert constants["stats"]["DEFAULT_BOOTSTRAP_SEED"] == 20261023 and constants["stats"]["DEFAULT_N_DRAWS"] == 10_000
    assert constants["scoring"]["FROZEN_SETTINGS"]["d2_flag"] == "has_d2"
    assert constants["identity"]["MMSEQS_BINARY_SHA256"] == identity.MMSEQS_BINARY_SHA256
    assert constants["identity"]["IDENTITY_BIN_LABELS"] == list(identity.IDENTITY_BIN_LABELS)
    assert constants["logit_adjustment"] == {"TAUS": [-1.0, -0.5, 0.0, 0.5, 1.0], "PRIOR_SMOOTHING": 1.0}
    assert constants["campaign"]["campaign_constants"] == json.loads(json.dumps(campaign.campaign_constants()))
    assert set(constants["comparator_published_recipe"]) >= {"learning_rate", "epochs"}
    assert constants["baseline"]["STATUSES"] == ["predicted", "no_hit", "tie", "failure"]
    assert freeze.absolute_path_problems(document, "document") == []


def test_document_copies_no_class_counts_or_proportions(document):
    text = json.dumps(document, sort_keys=True) + freeze.render_freeze1_markdown(document)
    for forbidden in ("eligible_by_ec_class", "ec_class_counts", "class_proportions", "proportions",
                      "0.123456789", "0.987654321", "inner_folds_without_ec_class", '"n_rows"', "n_eligible"):
        assert forbidden not in text, forbidden
    assert freeze.class_count_problems(document, "document") == []
    assert freeze.class_count_problems({"a": {"ec_class_counts": {}}}, "x") == ["x a.ec_class_counts would copy "
                                                                                 "per-fold class numbers"]


def test_document_and_rendering_are_deterministic_and_written_once(freeze_world, inputs, document, tmp_path):
    again = freeze.build_freeze1_document(**document_options(freeze_world, inputs))
    assert again == document
    assert freeze.render_freeze1_markdown(again) == freeze.render_freeze1_markdown(document)
    out = tmp_path / "docs" / "freeze1.json"
    digest, rendering = freeze.write_freeze1_document(document, out)
    assert digest == sha256_file(out) and out.read_bytes() == freeze.canonical_bytes(document)
    assert rendering == out.with_suffix(".md") and rendering.read_text() == freeze.render_freeze1_markdown(document)
    text = rendering.read_text()
    assert "## Decisions" in text and "| D1 |" in text and "| BP item 20 |" in text and "## Code pins" in text
    with pytest.raises(FileExistsError):
        freeze.write_freeze1_document(document, out)


def _tmp_decisions(tmp_path: Path, record: dict) -> Path:
    path = tmp_path / "freeze1_decisions.json"
    path.write_text(json.dumps(record))
    return path


def test_document_refusals(freeze_world, inputs, tmp_path):
    campaign_root = freeze_world["campaign"]
    repo = _copy_repo(freeze_world["repo"], tmp_path / "repo")
    drafts = repo / "docs" / "drafts"
    missing = decisions_record()
    missing["decisions"].pop("G8")
    (tmp_path / "a").mkdir()
    (drafts / "a").mkdir(parents=True)
    cases = [
        ({"decisions": _tmp_decisions(drafts / "a", missing), "repo": repo}, "decision G8 is missing"),
        ({"inputs_sha256": OTHER_F}, "--inputs does not hash to --inputs-sha256"),
        ({"cohort": campaign_root / "identity" / "cohort"}, "--cohort lies under <campaign root>/identity"),
        ({"labels": tmp_path / "labels"}, "--labels lies outside the campaign root"),
        ({"environment": campaign_root / "freeze"}, "--environment holds no regular file"),
        ({"pins": ["clean_environment_lock=" + "2" * 64]}, "--pin is missing"),
        ({"pins": document_options(freeze_world, inputs)["pins"][:3] + ["mmseqs2_binary=" + "5" * 64]},
         "--pin mmseqs2_binary differs"),
        ({"overlay_manifest": campaign_root / "cohort" / "cohort_audit_receipt.json"},
         "--overlay is not the overlay manifest the inputs manifest recorded"),
    ]
    for changes, expected in cases:
        with pytest.raises(Freeze1Error) as refused:
            freeze.build_freeze1_document(**document_options(freeze_world, inputs, **changes))
        assert expected in str(refused.value), (changes, str(refused.value))
    for change, expected in ((lambda record: record["decisions"].update({"E6": "<the name>"}), "placeholder"),
                             (lambda record: record["decisions"].update({"M3": "/home/someone/data"}),
                              "absolute path")):
        record = decisions_record()
        change(record)
        folder = drafts / expected.replace(" ", "_")
        folder.mkdir()
        with pytest.raises(Freeze1Error, match=expected):
            freeze.build_freeze1_document(**document_options(freeze_world, inputs, repo=repo,
                                                             decisions=_tmp_decisions(folder, record)))
    with pytest.raises(Freeze1Error, match="must be a file of this repository"):
        freeze.build_freeze1_document(**document_options(
            freeze_world, inputs, decisions=_tmp_decisions(tmp_path / "a", decisions_record())))


def test_document_refuses_a_smoke_simulation_and_a_cohort_receipt_the_overlay_did_not_copy(freeze_world, inputs):
    campaign_root = freeze_world["campaign"]
    smoke = write_simulation(campaign_root / "simulation" / "smoke_sweep", {"ec_class_counts": {"1": 1}}, smoke=True)
    with pytest.raises(Freeze1Error) as refused:
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, simulation=smoke))
    assert "the simulation receipt is not from a real (non-smoke) sweep" in str(refused.value)
    assert "simulation record records/p1.json is not a real record of point 'p1'" in str(refused.value)
    other = campaign_root / "cohort_other"
    other.mkdir()
    receipt = json.loads((campaign_root / "cohort" / "cohort_audit_receipt.json").read_text())
    (other / "cohort_audit_receipt.json").write_text(json.dumps(receipt, indent=1))
    with pytest.raises(Freeze1Error) as refused:
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, cohort=other))
    assert ("the overlay manifest does not record cohort/cohort_audit_receipt.json with the SHA-256 of the bound "
            "cohort_audit_receipt") in str(refused.value)


def test_document_refuses_smoke_r2_lists(freeze_world, inputs):
    campaign_root = freeze_world["campaign"]
    other = campaign_root / "r2_lists_smoke"
    other.mkdir()
    receipt = json.loads((campaign_root / "r2_lists" / "r2_lists_receipt.json").read_text())
    assert receipt["provenance"]["smoke"] is False
    receipt["provenance"]["smoke"] = True
    (other / "r2_lists_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(Freeze1Error) as refused:
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, r2_lists=other))
    assert "the R2 lists receipt is not from real (non-smoke) R2 lists" in str(refused.value)


def _labels_without_transfer_list(campaign_root: Path, name: str) -> Path:
    labels = campaign_root / name
    shutil.copytree(campaign_root / "labels", labels, ignore=shutil.ignore_patterns("ec_transfer_list"))
    return labels


def test_document_requires_the_transfer_list_unless_none_was_approved(freeze_world, inputs):
    campaign_root = freeze_world["campaign"]
    labels = _labels_without_transfer_list(campaign_root, "labels_no_transfer")
    with pytest.raises(Freeze1Error, match="transfer list, decision A8; --no-transfer-list when none was approved"):
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, labels=labels))
    document = freeze.build_freeze1_document(**document_options(freeze_world, inputs, labels=labels,
                                                                no_transfer_list=True))
    assert document["transfer_list"] == freeze.TRANSFER_LIST_ABSENT
    assert "transfer_list" not in document["bound_files"] and "transfer_list_receipt" not in document["bound_files"]
    assert f"- Transfer list: {freeze.TRANSFER_LIST_ABSENT}" in freeze.render_freeze1_markdown(document)
    with pytest.raises(Freeze1Error, match="--no-transfer-list states that no transfer list was approved"):
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, no_transfer_list=True))
    partial = _labels_without_transfer_list(campaign_root, "labels_partial_transfer")
    _write(partial, {"ec_transfer_list/ec_transfer_list.csv": "ec_number,transferred_to\n"})
    with pytest.raises(Freeze1Error, match="holds no regular file"):
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, labels=partial))


def test_document_checks_paths_before_reading_them(freeze_world, inputs, tmp_path, monkeypatch):
    campaign_root = freeze_world["campaign"]
    opened: list[str] = []
    original = freeze.sha256_file

    def recording(path):
        opened.append(str(path))
        return original(path)

    monkeypatch.setattr(freeze, "sha256_file", recording)
    identity_overlay = campaign_root / "identity" / "fold0" / "overlay_manifest.json"
    with pytest.raises(Freeze1Error, match="--overlay lies under <campaign root>/identity"):
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, overlay_manifest=identity_overlay))
    assert str(identity_overlay) not in opened
    outside = tmp_path / "freeze1_decisions.json"
    outside.write_text("not json")
    with pytest.raises(Freeze1Error, match="--decisions must be a file of this repository"):
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, decisions=outside))
    named = tmp_path / CAMPAIGN_ID / "identity" / "freeze1_inputs.json"
    with pytest.raises(Freeze1Error, match="--inputs lies under <campaign root>/identity"):
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, inputs=named))
    assert freeze.campaign_identity_name_problem(tmp_path / CAMPAIGN_ID / "freeze" / "x.json", "--inputs") is None
    assert freeze.campaign_identity_name_problem(tmp_path / "other" / "identity" / "x.json", "--inputs") is None


def test_document_refuses_inputs_that_no_longer_describe_the_code(freeze_world, inputs, tmp_path):
    repo = _copy_repo(freeze_world["repo"], tmp_path / "repo")
    _commit(repo, {"scripts/new_tool.py": "VALUE = 1\n"})
    with pytest.raises(Freeze1Error, match="tracked src/ or scripts/ files differ"):
        freeze.build_freeze1_document(**document_options(
            freeze_world, inputs, repo=repo, decisions=repo / freeze.CAMPAIGN_DOCS_DIR / "freeze1_decisions.json"))


def test_document_refuses_a_smoke_or_incomplete_inputs_manifest(freeze_world, inputs, tmp_path):
    manifest = dict(inputs["manifest"])
    manifest.pop("completeness")
    path = freeze_world["campaign"] / "freeze" / "incomplete_inputs.json"
    digest = provenance.write_new_json(path, manifest)
    with pytest.raises(Freeze1Error, match="not a real, complete Freeze-1 inputs manifest"):
        freeze.build_freeze1_document(**document_options(freeze_world, inputs, inputs=path, inputs_sha256=digest))


def test_document_action_writes_both_files(freeze_world, inputs, tmp_path, monkeypatch, capsys):
    repo = _copy_repo(freeze_world["repo"], tmp_path / "repo")
    monkeypatch.setattr(freeze, "REPO_ROOT", repo)
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=freeze.MANIFEST_KINDS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--view", action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    freeze.add_manifest_freeze_arguments(parser)
    options = document_options(freeze_world, inputs, decisions=repo / freeze.CAMPAIGN_DOCS_DIR /
                               "freeze1_decisions.json")
    out = tmp_path / "outside" / "freeze1.json"
    argv = ["--kind", "freeze1-document", "--inputs", str(options["inputs"]), "--inputs-sha256",
            options["inputs_sha256"], "--decisions", str(options["decisions"]), "--overlay",
            str(options["overlay_manifest"]), "--labels", str(options["labels"]), "--cohort", str(options["cohort"]),
            "--r2-lists", str(options["r2_lists"]), "--simulation", str(options["simulation"]), "--environment",
            str(options["environment"]), *(token for pin in options["pins"] for token in ("--pin", pin)),
            "--out", str(out)]
    args = parser.parse_args(argv)
    assert args.pin == options["pins"] and args.plans_dir is None and args.fold_work == []
    assert freeze.action_manifest_freeze1_document(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["sha256"] == sha256_file(out) and out.with_suffix(".md").is_file()
    assert printed["decisions"] == len(json.loads(out.read_text())["decisions"]["answers"])
    assert printed["transfer_list"] == "bound" and args.no_transfer_list is False
    with pytest.raises(Freeze1Error, match="needs --inputs-sha256"):
        freeze.action_manifest_freeze1_document(parser.parse_args([token for token in argv
                                                                   if token != options["inputs_sha256"]
                                                                   and token != "--inputs-sha256"]))


# ---------------------------------------------------------------------------
# Freeze 1b


def _freeze1b_document(identity_root: Path) -> dict:
    pins = {path: sha256_file(ROOT / path) for path in freeze.R_ID_FILES}
    works = {str(fold): identity_root / f"fold{fold}" for fold in range(N_OUTER_FOLDS)}
    memberships = {fold: {"assignment_receipt": {"sha256": sha256_file(work / identity.INNER_ASSIGNMENT_RECEIPT)},
                          "certificate": {"sha256": sha256_file(work / identity.INNER_CERTIFICATE)}}
                   for fold, work in works.items()}
    return {"format": freeze.FREEZE1_DOCUMENT_FORMAT, "schema_version": 1, "campaign_id": CAMPAIGN_ID,
            "kind": "freeze1", "amendments": [], "code": {"tracked_file_sha256": pins},
            "inner_membership": memberships}


def run_identity_audit(world: World, freeze1_sha256: str) -> None:
    """The P2-11 identity audit on the world's identity folders: complement chains, cohort clusters and strata."""
    split30 = world.base / "source" / "split30"
    folds_dir = world.data_root / "CLEAN_30_main" / "folds"
    binary = world.base / "fakebin" / "mmseqs"
    works = [world.work(fold) for fold in range(N_OUTER_FOLDS)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(identity, "MMSEQS_BINARY_SHA256", world.binary_sha)
        for fold, work in enumerate(works):
            build_inputs(work, fold, query_set="complement", freeze1_sha256=freeze1_sha256,
                         official_split_dir=split30, metallo_folds_dir=folds_dir, metallo_manifest_root=world.data_root)
            run_search(work, fold, query_set="complement", binary=binary, expected_sha256=sha256_file(binary),
                       freeze1_sha256=freeze1_sha256, runner=FakeMMseqs(hits={}))
            build_edges(work, fold, query_set="complement", freeze1_sha256=freeze1_sha256)
        build_cohort_components(works, world.identity_root / "cohort", freeze1_sha256=freeze1_sha256)
        for fold, work in enumerate(works):
            build_strata(work, fold, freeze1_sha256=freeze1_sha256,
                         clusters=world.identity_root / "cohort" / "components_cohort.tsv")


@pytest.fixture(scope="module")
def freeze1b_world(freeze_world, tmp_path_factory):
    """A copy of the campaign world after Freeze 1: a repository whose log names the P2-1 HEAD and F (pushed), the
    committed document, a passed verify-freeze1 receipt, and the identity audit run with F."""
    base = tmp_path_factory.mktemp("freeze1b")
    template = freeze_world["world"]
    shutil.copytree(template.base, base / "world")
    world = template.moved(base / "world")
    repo = base / "repo"
    identity_head = _init(repo, {path: (ROOT / path).read_bytes() for path in freeze.R_ID_FILES})
    document_bytes = freeze.canonical_bytes(_freeze1b_document(world.identity_root))
    digest = provenance.sha256_bytes(document_bytes)
    log = (f"# Campaign log\n\n## evc-005\n\nIdentity chain at the clean commit `{identity_head[:7]}`.\n\n"
           f"## evc-008\n\nFreeze 1 SHA-256 `{digest}`.\n")
    _commit(repo, {DOCUMENT: document_bytes, LOG: log}, "freeze 1")
    _git(repo, "update-ref", freeze.ORIGIN_MAIN_REF, "HEAD")
    record = freeze.verify_freeze1(sha256=digest, document=repo / DOCUMENT, require_pushed=True, repo=repo)
    assert record["status"] == "passed", record["problems"]
    receipt = base / "verify_freeze1.json"
    receipt.write_text(json.dumps(record))
    run_identity_audit(world, digest)
    return {"world": world, "repo": repo, "F": digest, "identity_head": identity_head, "receipt": receipt,
            "document": repo / DOCUMENT}


@pytest.fixture
def pinned(freeze1b_world, monkeypatch):
    monkeypatch.setattr(identity, "MMSEQS_BINARY_SHA256", freeze1b_world["world"].binary_sha)


def freeze1b_options(context: dict, **changes) -> dict:
    options = {"freeze1": context["document"], "freeze1_sha256": context["F"],
               "identity_root": context["world"].identity_root, "fold_work": None,
               "verify_receipt": context["receipt"], "identity_head": context["identity_head"][:7],
               "repo": context["repo"]}
    options.update(changes)
    return options


def _freeze1b_problems(context: dict, **changes) -> str:
    with pytest.raises(Freeze1Error) as refused:
        freeze.build_freeze1b(**freeze1b_options(context, **changes))
    return str(refused.value)


def test_freeze1b_records_every_identity_receipt(freeze1b_world, pinned):
    context = freeze1b_world
    manifest = freeze.build_freeze1b(**freeze1b_options(context))
    assert manifest["status"] == "passed" and manifest["freeze1_sha256"] == context["F"]
    assert manifest["identity_head"] == context["identity_head"] and manifest["freeze1_document"] == DOCUMENT
    assert manifest["identity_code_sha256"] == identity.code_sha256()
    assert manifest["verify_freeze1_receipt_sha256"] == sha256_file(context["receipt"])
    for fold in range(N_OUTER_FOLDS):
        receipts = manifest["folds"][str(fold)]["receipts"]
        assert {"fasta_receipt_complement.json", "search_complement/search_receipt.json",
                "edges_receipt_complement.json", "strata_receipt.json", "fasta_receipt_metallo_train.json",
                "components_receipt_inner.json", identity.INNER_ASSIGNMENT_RECEIPT,
                identity.INNER_CERTIFICATE} <= set(receipts)
        assert f"strata_fold{fold}.tsv" in manifest["folds"][str(fold)]["outputs"]
        assert manifest["folds"][str(fold)]["work"] == f"identity/fold{fold}"
    assert sum(name.startswith("r1_inner_exclusions_receipt_fold0") for name in manifest["folds"]["0"]["receipts"]) == 4
    assert manifest["cohort"]["outputs"] == {"components_cohort.tsv": sha256_file(
        context["world"].identity_root / "cohort" / "components_cohort.tsv")}
    counts = freeze.freeze1b_counts(manifest)
    # Per fold: three metallo and three complement chain receipts, inner components, assignment, certificate and
    # strata; fold 0 adds its four R1-inner receipts; the cohort components receipt once.
    assert counts == {"folds": 5, "receipts": 5 * 10 + 4 + 1, "outputs": counts["outputs"]}


def test_freeze1b_action_prints_counts_only(freeze1b_world, pinned, tmp_path, monkeypatch, capsys):
    context = freeze1b_world
    monkeypatch.setattr(freeze, "REPO_ROOT", context["repo"])
    out = tmp_path / "freeze1b.json"
    args = argparse.Namespace(freeze1=str(context["document"]), freeze1_sha256=context["F"],
                              identity_root=str(context["world"].identity_root), fold_work=[],
                              verify_freeze1=str(context["receipt"]), identity_head=context["identity_head"][:7],
                              out=str(out))
    assert freeze.action_manifest_freeze1b(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"manifest", "kind", "sha256", "folds", "receipts", "outputs"}
    assert printed["sha256"] == sha256_file(out)
    assert not [accession for accession in context["world"].stems if accession in json.dumps(printed)]


def _tampered_fold(context: dict, tmp_path: Path, fold: int, name: str, change) -> list[str]:
    work = tmp_path / f"fold{fold}_copy"
    shutil.copytree(context["world"].work(fold), work)
    receipt = json.loads((work / name).read_text())
    change(receipt)
    (work / name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return [f"{fold}={work}"]


def test_freeze1b_refuses_a_receipt_without_f(freeze1b_world, pinned, tmp_path):
    context = freeze1b_world
    fold_work = _tampered_fold(context, tmp_path, 2, "strata_receipt.json",
                               lambda receipt: receipt.update(freeze1_sha256=OTHER_F))
    text = _freeze1b_problems(context, fold_work=fold_work)
    assert "fold 2: strata_receipt.json does not record the Freeze-1 SHA-256" in text
    assert "--freeze1 does not hash to --freeze1-sha256" in _freeze1b_problems(context, freeze1_sha256=OTHER_F)


def test_freeze1b_refuses_another_identity_code_sha256(freeze1b_world, pinned, tmp_path):
    context = freeze1b_world
    fold_work = _tampered_fold(context, tmp_path, 3, identity.INNER_CERTIFICATE,
                               lambda receipt: receipt["code_sha256"].update({"src/ec_clean30/identity.py": "0" * 64}))
    text = _freeze1b_problems(context, fold_work=fold_work)
    assert (f"fold 3: {identity.INNER_CERTIFICATE} records another identity code SHA-256 than Freeze 1 pins" in text)
    assert f"fold 3: {identity.INNER_CERTIFICATE} differs from the copy Freeze 1 bound" in text


def test_freeze1b_refuses_a_changed_r_id_file(freeze1b_world, pinned, tmp_path):
    context = dict(freeze1b_world)
    repo = _copy_repo(context["repo"], tmp_path / "repo")
    _commit(repo, {"src/project_paths.py": (repo / "src/project_paths.py").read_text() + "\n# changed\n"})
    _git(repo, "update-ref", freeze.ORIGIN_MAIN_REF, "HEAD")
    context.update(repo=repo, document=repo / DOCUMENT)
    text = _freeze1b_problems(context)
    assert "the R-ID files ['src/project_paths.py'] changed between --identity-head and HEAD" in text
    assert "the R-ID file src/project_paths.py differs from its Freeze-1 pin" in text


def test_freeze1b_refuses_an_identity_head_missing_from_the_log_or_not_an_ancestor(freeze1b_world, pinned,
                                                                                   tmp_path):
    context = dict(freeze1b_world)
    head = _git(context["repo"], "rev-parse", "HEAD")
    text = _freeze1b_problems(context, identity_head=head)
    assert f"--identity-head is not named in {LOG} at HEAD" in text
    repo = _copy_repo(context["repo"], tmp_path / "repo")
    _git(repo, "checkout", "-q", "-b", "side")
    side = _commit(repo, {"docs/side.md": "side\n"})
    _git(repo, "checkout", "-q", "main")
    context.update(repo=repo, document=repo / DOCUMENT)
    text = _freeze1b_problems(context, identity_head=side)
    assert "--identity-head is not an ancestor of HEAD" in text
    assert "--identity-head names no commit" in _freeze1b_problems(context, identity_head="1234567")


def test_freeze1b_refuses_a_verify_receipt_that_is_not_pushed(freeze1b_world, pinned, tmp_path):
    context = freeze1b_world
    record = json.loads(context["receipt"].read_text())
    record.update(pushed=False)
    receipt = tmp_path / "verify_not_pushed.json"
    receipt.write_text(json.dumps(record))
    assert "not a passed verify-freeze1 receipt" in _freeze1b_problems(context, verify_receipt=receipt)
    repo = _copy_repo(context["repo"], tmp_path / "repo")
    _git(repo, "update-ref", "-d", freeze.ORIGIN_MAIN_REF)
    text = _freeze1b_problems({**context, "repo": repo, "document": repo / DOCUMENT})
    assert "verify-freeze1 now:" in text and "does not contain the commit" in text


def test_freeze1b_refuses_a_verify_receipt_of_another_document(freeze1b_world, pinned, tmp_path):
    context = freeze1b_world
    record = json.loads(context["receipt"].read_text())
    record.update(document="docs/other/freeze1.json")
    receipt = tmp_path / "verify_other_document.json"
    receipt.write_text(json.dumps(record))
    text = _freeze1b_problems(context, verify_receipt=receipt)
    assert "--verify-freeze1 names another document or logging commit than the check made now" in text


def test_freeze1b_refuses_a_test_named_fold_work_folder(freeze1b_world, pinned, tmp_path, monkeypatch):
    context = freeze1b_world
    work = tmp_path / "fold0_test"
    with pytest.raises(provenance.HeldOutInputError, match="--fold-work folder"):
        freeze.build_freeze1b(**freeze1b_options(context, fold_work=[f"0={work}"]))
    monkeypatch.setattr(freeze, "REPO_ROOT", context["repo"])
    args = argparse.Namespace(freeze1=str(context["document"]), freeze1_sha256=context["F"],
                              identity_root=str(context["world"].identity_root), fold_work=[f"0={work}"],
                              verify_freeze1=str(context["receipt"]), identity_head=context["identity_head"][:7],
                              out=str(tmp_path / "freeze1b.json"))
    with pytest.raises(provenance.HeldOutInputError, match="--fold-work"):
        freeze.action_manifest_freeze1b(args)
    assert not (tmp_path / "freeze1b.json").exists()


def test_freeze1b_refuses_a_missing_complement_step(freeze1b_world, pinned, tmp_path):
    context = freeze1b_world
    work = tmp_path / "fold4_copy"
    shutil.copytree(context["world"].work(4), work)
    (work / "edges_receipt_complement.json").unlink()
    text = _freeze1b_problems(context, fold_work=[f"4={work}"])
    assert "fold 4: edges_receipt_complement.json does not verify" in text
