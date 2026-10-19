"""Tests for v3 step F (pmm_v3_step_f.py, run_pmm_v3_step_f.py, pmm_v3_test_prep.py, pmm_v3_step_f_report.py;
rules docs/campaigns/pmm_ion_metal_v3/step_f_spec.json). Synthetic and training-side data only: no test data is read.

Every test here runs on CPU without fitting a model, except the ones marked synthetic fit, which run a tiny refit of
the selected cell on the synthetic campaign (then label-blind preparation, prediction and the report on a synthetic
test side) and are skipped unless DEEPMZYME_SYNTHETIC_FIT_TESTS=1.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in (
    str(_Path(__file__).resolve().parents[2]),
    str(_Path(__file__).resolve().parents[2] / "src"),
    str(_Path(__file__).resolve().parents[1]),
    str(_Path(__file__).resolve().parent),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from project_paths import RELOCATED_FILES, install_relocated_module_aliases, resolve_existing_path, resolve_repo_file
install_relocated_module_aliases()
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_step_c as c  # noqa: E402
import pmm_v3_step_d as d  # noqa: E402
import pmm_v3_step_e2 as e2  # noqa: E402
import pmm_v3_step_f as f  # noqa: E402
import pmm_v3_step_f_report as report  # noqa: E402
import pmm_v3_test_prep as prep  # noqa: E402
import run_pmm_v3_step_f as vm  # noqa: E402
from pmm_campaign_fixtures import _atom_line, write_pdb  # noqa: E402
from test_v3_campaign import campaign, policy, settle  # noqa: E402,F401  (fixture reuse)
from test_v3_step_c_launcher import FakeVM, session  # noqa: E402
from test_v3_step_c_launcher import vm as vm_snapshot  # noqa: E402

FUSION = "gvp_late_fusion__four_class__baseline"
REFIT = "stage6b_refit__gvp_late_fusion__four_class__baseline__e50__seed42"
RUNNER = d.local_runner_sha256()
SYNTHETIC_FIT = pytest.mark.skipif(os.environ.get("DEEPMZYME_SYNTHETIC_FIT_TESTS") != "1",
                                   reason="runs a tiny synthetic CPU fit; opt in with DEEPMZYME_SYNTHETIC_FIT_TESTS=1")


def sha(path: Path) -> str:
    return hashlib.sha256(resolve_existing_path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Specification and hygiene
# ---------------------------------------------------------------------------

def test_the_specification_binds_the_selected_50_epoch_cell_and_its_rules():
    spec = f.load_spec()
    assert spec["selection"]["cell"] == FUSION and spec["refit"]["run_name"] == REFIT
    assert (spec["selection"]["epochs"], spec["refit"]["epochs"], spec["refit"]["seed"]) == (50, 50, 42)
    assert spec["refit"]["checkpoint_rule"] == "terminal" and spec["refit"]["calibration"] is None
    assert spec["refit"]["added_flags"] == {"--val-fraction": "0.0", "--selection-metric": "train_loss"}
    assert spec["test_rows"]["source_rows"] == 1488 and spec["test_inputs"]["previously_opened_structure_files"] == 316
    assert spec["test_inputs"]["source_csv_sha256"] == "ec6427ad0b8a18261dbc1d6822bdcfe7194730c7da4a36c208957bcb00f6a0f2"
    assert [c_["id"] for c_ in spec["clean_subset"]["criteria"]] == ["C1_training_pdb_id", "C2_near_copy_chain",
                                                                     "C3_previously_opened_pdb_id"]
    assert spec["disclosures"][0].startswith("Labels: both_results_secondary")
    assert spec["report"]["pmm_comparison"]["published_test_figure_2b"]["mean"] == 67.85
    assert prep.SPEC_SHA256 == f.FROZEN_SPEC_SHA256 == sha(f.SPEC_JSON)  # the pinned specification
    assert not set(prep.SITE_ALLOWLIST) & set(prep.FORBIDDEN_COLUMNS) and spec["label_file"]["column"] not in \
        prep.SOURCE_ALLOWLIST


def test_the_launcher_and_the_report_never_import_torch():
    for module in ("pmm_v3_step_f", "pmm_v3_step_f_report"):
        code = f"import sys, pathlib; sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'src')!r}]; import project_paths, {module}; print('torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        assert out.strip() == "False", module


def test_no_frozen_runner_file_is_touched_and_only_the_report_names_the_label_column():
    tracked = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "--", *(str(resolve_existing_path(ROOT / f)) for f in v3.RUNNER_FILES)])
    assert tracked.returncode == 0
    assert not set(f.STEP_F_TOOL_FILES) & set(v3.RUNNER_FILES)
    for name in ("pmm_v3_step_f.py", "run_pmm_v3_step_f.py", "pmm_v3_test_prep.py"):
        for text in code_strings(ROOT / name):  # string constants outside docstrings
            assert text != "element" and not any(word in text for word in ("label_metal", "metaltype", "label_file")), \
                (name, text)
    assert 'LABELS["column"]' in resolve_existing_path(ROOT / "pmm_v3_step_f_report.py").read_text()


def code_strings(path: Path) -> list[str]:
    import ast

    tree = ast.parse(resolve_existing_path(path).read_text())
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body
                  and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings]


# ---------------------------------------------------------------------------
# F2: the refit command (synthetic campaign; no fit)
# ---------------------------------------------------------------------------

def test_the_refit_drops_only_fold_validation_and_weight_flags_and_binds_full_cohort_weights(campaign):  # noqa: F811
    from collections import Counter

    from training.config import config_to_payload, parse_args
    from training.source_cohort import read_cohort_csv

    train, esm_dir, paths, _, _ = campaign
    settle(paths)
    common = dict(epochs=50, selection_sha256="ab" * 32, python_bin=sys.executable, train_dir=train, esm_dir=esm_dir,
                  device="cpu", lane=0)
    diff = vm.refit_differences(paths, **common)
    assert diff["unexpected"] == [] and "epochs" not in diff["differing"] and "seed" not in diff["differing"]
    command = diff["command"]
    for flag in ("--n-folds", "--fold-index", "--fold-membership-csv", "--export-validation-predictions",
                 "--run-test-eval"):
        assert flag not in command
    config = config_to_payload(parse_args(command[3:]))
    assert (config["val_fraction"], config["epochs"], config["seed"], config["checkpoint_rule"], config["lr_schedule"],
            config["selection_metric"], config["run_name"]) == (0.0, 50, 42, "terminal", "cosine", "train_loss", REFIT)
    bindings = read_cohort_csv(paths.cohort)
    n = Counter(v3.v2.COMMON_FOUR[b.native_element] for b in bindings)
    expected = {key: len(bindings) / (4 * n[label]) for key, label in
                (("mn", "Mn"), ("cu", "Cu"), ("zn", "Zn"), ("class_viii", "Class VIII"))}
    assert diff["multipliers"] == {f"{k}_loss_multiplier": pytest.approx(v) for k, v in expected.items()}
    identity = diff["identity"]
    assert json.loads(command[-1]) == identity and identity["validation"] == "none" and identity["epochs"] == 50
    assert identity["selection_record_sha256"] == "ab" * 32 and identity["fold_set_id"] is None
    assert identity["step_f_spec_sha256"] == sha(f.SPEC_JSON) and identity["step_f_runner_sha256"] == sha(
        ROOT / "run_pmm_v3_step_f.py")
    with pytest.raises(ValueError, match="never retried"):  # an existing artifact blocks any run
        stray = paths.lane(1) / "runs" / REFIT
        stray.mkdir(parents=True)
        vm.run_refit(paths, lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable, device="cpu",
                     execution_policy=policy(paths.root.parent), epochs=2, selection_sha256="ab" * 32)


def test_the_command_line_refuses_another_epoch_count_or_a_missing_selection_binding(campaign):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    with pytest.raises(ValueError, match="bound schedule is 50 epochs"):
        vm.main(["--action", "plan-refit", "--campaign-dir", str(paths.root), "--train-dir", str(train),
                 "--epochs", "10"])
    with pytest.raises(ValueError, match="selection-sha256"):
        vm.main(["--action", "refit", "--campaign-dir", str(paths.root), "--train-dir", str(train),
                 "--unit", REFIT, "--device", "cpu"])


# ---------------------------------------------------------------------------
# A synthetic test side (every disposition) and its label-blind preparation
# ---------------------------------------------------------------------------

def link_line(left, right, operators=("1555", "2555")) -> str:
    chars = [" "] * 80

    def put(index, text):
        chars[index:index + len(text)] = list(text)

    put(0, "LINK  ")
    put(12, f"{left[0]:<4}"), put(17, f"{left[1]:>3}"), put(21, left[2]), put(22, f"{left[3]:>4}")
    put(42, f"{right[0]:<4}"), put(47, f"{right[1]:>3}"), put(51, right[2]), put(52, f"{right[3]:>4}")
    put(59, f"{operators[0]:>6}"), put(66, f"{operators[1]:>6}")
    return "".join(chars).rstrip()


def add_lines(path: Path, *, head=(), tail=()) -> None:
    lines = path.read_text().splitlines()
    end = lines.index("END")
    path.write_text("\n".join(list(head) + lines[:end] + list(tail) + lines[end:]) + "\n")


CODE = {"MN": "1", "FE": "2", "CO": "2", "NI": "2", "CU": "6", "ZN": "7", "CA": "3"}
# source row -> (pdbid, chain, resseq, element in the label columns, expected disposition, reason, alias of row)
ROWS = {
    1: ("1aba", "A", 201, "ZN", "scored", "", None), 2: ("1aba", "A", 202, "MN", "scored", "", None),
    3: ("7ttt", "A", 301, "CU", "scored_link_flagged", "", None),
    4: ("7ttt", "A", 301, "CU", "scored_link_flagged", "", 3), 5: ("7ttt", "A", 302, "FE", "scored", "", None),
    6: ("7uuu", "A", 401, "ZN", "unscorable", "target_atom_alternate_locations", None),
    7: ("7uuu", "A", 402, "NI", "scored", "", None),
    8: ("7vvv", "A", 501, "ZN", "unscorable", "ambiguous_target_residue", None),
    9: ("7vvv", "A", 502, "CO", "scored", "", None),
    10: ("7www", "A", 601, "ZN", "unscorable", "multiple_atoms_in_target_residue", None),
    11: ("7www", "A", 602, "MN", "scored", "", None),
    12: ("7www", "A", 603, "CA", "unscorable", "target_not_a_supported_metal_ion", None),
    13: ("7www", "A", 999, "ZN", "unscorable", "target_residue_absent_in_model_0", None),
    14: ("7www", "A", 604, "ZN", "unscorable", "no_context_residues", None),
    15: ("7xxx", "A", 701, "ZN", "unscorable", "structure_unverified", None),
    16: ("7yyy", "A", 801, "ZN", "unscorable", "structure_parse_error", None),
    17: ("7zzz", "A", 901, "MN", "unscorable", "manifest_source_identifier_mismatch", None),
    18: ("7zzz", "A", 902, "CU", "unscorable", "missing_reconstruction_row", None),
    19: ("7zzz", "A", 903, "ZN", "scored", "", None),
}
SWAP = {"ZN": "MN", "MN": "ZN", "CU": "FE", "FE": "CU", "CO": "NI", "NI": "CO"}


def build_test_side(root: Path, *, labels=None, swap: dict[str, str] | None = None) -> tuple[Path, Path]:
    """root/test with manifests and structures plus a matching source CSV; ``labels`` replaces the label columns'
    element per row (a permutation); ``swap`` changes the metal of every supported ion in the structures."""
    swap = swap or {}
    test = root / "test"
    (test / "structures").mkdir(parents=True)
    structures = {
        "1aba": [("A", 201, "", "ZN", (0.0, 0.0, 0.0)), ("A", 202, "", "MN", (3.2, 0.0, 0.0))],
        "7ttt": [("A", 301, "", "CU", (0.0, 0.0, 0.0)), ("A", 302, "", "FE", (30.0, 0.0, 0.0))],
        "7uuu": [("A", 401, "", "ZN", (0.0, 0.0, 0.0)), ("A", 402, "", "NI", (30.0, 0.0, 0.0))],
        "7vvv": [("A", 501, "", "ZN", (0.0, 0.0, 0.0)), ("A", 501, "A", "ZN", (2.0, 0.0, 0.0)),
                 ("A", 502, "", "CO", (30.0, 0.0, 0.0))],
        "7www": [("A", 601, "", "ZN", (0.0, 0.0, 0.0)), ("A", 602, "", "MN", (30.0, 0.0, 0.0))],
        "7xxx": [("A", 701, "", "ZN", (0.0, 0.0, 0.0))],
        "7zzz": [("A", 901, "", "MN", (0.0, 0.0, 0.0)), ("A", 902, "", "CU", (30.0, 0.0, 0.0)),
                 ("A", 903, "", "ZN", (60.0, 0.0, 0.0))],
    }
    names = {}
    for pdbid, ions in structures.items():
        name = f"{pdbid}__chain_A__EC_0.0.0.0.pdb"
        names[pdbid] = name
        path = test / "structures" / name
        write_pdb(path, ions=[(ch, r, i, swap.get(e, e), xyz) for ch, r, i, e, xyz in ions], chains=("A", "B"),
                  altloc_ion=pdbid == "7uuu")
        if pdbid == "7ttt":
            metal = swap.get("CU", "CU")
            add_lines(path, head=[link_line((metal, metal, "A", 301), ("NE2", "HIS", "A", 1))])
        if pdbid == "7www":
            zn = swap.get("ZN", "ZN")
            lines = path.read_text().splitlines()
            ion = next(i for i, line in enumerate(lines) if line.startswith("HETATM") and line[22:26] == " 601")
            lines.insert(ion + 1, _atom_line("HETATM", 9001, "O1", "", zn, "A", 601, "", (0.6, 0.0, 0.0), "O"))
            path.write_text("\n".join(lines) + "\n")  # a second atom in the ion's residue (contiguous)
            add_lines(path, tail=[_atom_line("HETATM", 9002, "CA", "", "CA", "A", 603, "", (31.0, 3.0, 0.0), "CA"),
                                  _atom_line("HETATM", 9003, zn, "", zn, "A", 604, "", (500.0, 500.0, 500.0), zn)])
    names["7yyy"] = "7yyy__chain_A__EC_0.0.0.0.pdb"
    (test / "structures" / names["7yyy"]).write_text(
        "ATOM      1  N   ALA A   1      xx.000   1.000   1.000  1.00 20.00           N\nEND\n")
    source = root / "classmodel_test_set"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pdbid", "residueid_ion", "metalid", "label_metal", "ched_count", "f1"])
        for row, (pdbid, _chain, _resseq, element, *_rest) in ROWS.items():
            label = (labels or {}).get(row, element)
            writer.writerow([pdbid, 5000 + row, row, CODE[label], 3, float(row)])
    source_sha = sha(source)
    with (test / "site_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["source_uid", "source_row", "source_side", "pdbid", "chain", "resseq", "chain_resi", "element",
                         "metaltype", "label_metal", "structure_name", "residueid_ion", "resolution_method"])
        for row, (pdbid, chain, resseq, element, _disp, reason, _alias) in ROWS.items():
            if reason == "missing_reconstruction_row":
                continue
            label = (labels or {}).get(row, element)
            identifier = 5000 + row + (1 if reason == "manifest_source_identifier_mismatch" else 0)
            writer.writerow([f"sha256:{source_sha}:row:{row}", row, "test", pdbid, chain, resseq, f"{chain}_{resseq}",
                             label, label, CODE[label], names[pdbid], identifier, "fixture"])
    with (test / "structure_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["structure_name", "relative_path", "sha256", "size_bytes"])
        for pdbid, name in sorted(names.items()):
            path = test / "structures" / name
            digest = "0" * 64 if pdbid == "7xxx" else sha(path)
            writer.writerow([name, f"structures/{name}", digest, path.stat().st_size])
    return test, source


def prepared(tmp_path, name="prep", **kwargs):
    test, source = build_test_side(tmp_path / name / "dataset", **kwargs)
    rows, extra = prep.prepare_rows(test, source, expected_source_sha256=sha(source))
    return test, source, rows, extra


def test_every_source_row_gets_its_disposition_aliases_and_link_flags_without_a_label(tmp_path):
    test, source, rows, extra = prepared(tmp_path)
    assert [r["source_row"] for r in rows] == list(ROWS)
    uid = {r["source_row"]: r["source_uid"] for r in rows}
    for r in rows:
        pdbid, chain, resseq, _element, disposition, reason, alias = ROWS[r["source_row"]]
        assert (r["disposition"], r["reason"]) == (disposition, reason), r["source_row"]
        assert r["pdbid"] == pdbid
        if disposition != "unscorable":
            assert r["alias_primary_source_uid"] == uid[alias or r["source_row"]]
            assert r["context_chains"] == "A;B" and r["n_context_residues"] > 0 and len(r["ion_key"]) == 64
            assert r["link_flagged"] is (disposition == "scored_link_flagged")
    summary = extra["summary"]
    assert summary["source_rows"] == 19 and summary["physical_ions"] == 8 and summary["row_subset"] is None
    assert summary["dispositions"] == {"scored": 7, "scored_link_flagged": 2, "unscorable": 10}
    # Stored outputs carry no element, residue name, atom name or readable ion identifier.
    files = prep.membership_files(rows, prep.esm_plan(rows, extra["sequences"]), previously_opened=set(),
                                  training_pdbids=set())
    text = b"".join(files.values()).decode()
    for word in ("ZN", "MN", "CU", "FE", "NI", "CO", "Zn", "Mn"):
        assert word not in text
    with pytest.raises(prep.PreparationError, match="not the pinned"):
        prep.prepare_rows(test, source, expected_source_sha256="0" * 64)


def test_the_preparation_is_invariant_to_the_labels_and_to_which_supported_metal_each_ion_is(tmp_path):
    _, source_a, rows_a, _ = prepared(tmp_path, "a")
    permuted = {row: ROWS[(row % 19) + 1][3] for row in ROWS}  # every label column changed
    _, source_b, rows_b, _ = prepared(tmp_path, "b", labels=permuted)
    swap_test, _, rows_c, _ = prepared(tmp_path, "c", swap=SWAP)

    def strip(rows, source):
        return [{k: str(v).replace(sha(source), "S") for k, v in r.items()} for r in rows]

    assert strip(rows_a, source_a) == strip(rows_b, source_b)  # labels differ, structures equal: identical rows
    keep = [k for k in prep.ROW_COLUMNS if k not in ("structure_sha256", "ion_key")]
    assert [{k: r[k] for k in keep} for r in rows_a] == [{k: r[k] for k in keep} for r in rows_c]


def test_the_only_rows_option_limits_readiness_checks_to_named_rows(tmp_path):
    test, source = build_test_side(tmp_path / "dataset")
    rows, extra = prep.prepare_rows(test, source, expected_source_sha256=None, only_rows={3, 4, 9})
    assert [r["source_row"] for r in rows] == [3, 4, 9] and extra["summary"]["row_subset"] == 3


def test_the_ion_key_hashes_stored_identifiers_only_and_never_the_atom_name_or_element(tmp_path):
    import inspect

    assert list(inspect.signature(prep.ion_key).parameters) == ["structure_sha256", "model_index", "chain", "resseq",
                                                                 "icode", "altloc"]
    _, _, rows, extra = prepared(tmp_path)
    for r in rows:
        if r["disposition"] != "unscorable":  # the key is a function of the stored columns: it adds no information
            identity = "|".join([r["structure_sha256"], "0", r["chain"], str(int(r["resseq"])), r["icode"], r["altloc"]])
            assert r["ion_key"] == hashlib.sha256(identity.encode()).hexdigest()
    # Same structure bytes except the metal: the key differs only through the structure's content hash.
    _, _, swapped, _ = prepared(tmp_path, "swapped", swap=SWAP)
    for a, b in zip(rows, swapped):
        if a["disposition"] != "unscorable":
            assert b["ion_key"] == prep.ion_key(b["structure_sha256"], 0, a["chain"], int(a["resseq"]), a["icode"],
                                                a["altloc"])
    assert extra["summary"]["aliases"] == 1 and extra["summary"]["physical_ions"] == 8


def test_row_accounting_refuses_lost_rows_inconsistent_aliases_and_keys():
    base = {"structure_sha256": "a" * 64, "chain": "A", "resseq": "7", "icode": "", "altloc": "", "coord_x": "1.000",
            "coord_y": "2.000", "coord_z": "3.000", "reason": "", "disposition": "scored"}
    key = prep.ion_key("a" * 64, 0, "A", 7, "", "")
    rows = [dict(base, source_row=1, source_uid="u1", ion_key=key, alias_primary_source_uid="u1"),
            dict(base, source_row=2, source_uid="u2", ion_key=key, alias_primary_source_uid="u1"),
            dict(base, source_row=3, source_uid="u3", ion_key="", alias_primary_source_uid="", disposition="unscorable",
                 reason="structure_unverified")]
    assert prep.check_row_accounting(rows, expected_rows=3) == {"rows": 3, "scored": 2, "physical_ions": 1, "aliases": 1}
    with pytest.raises(prep.PreparationError, match="prepared rows for 4 source rows"):
        prep.check_row_accounting(rows, expected_rows=4)
    moved = [rows[0], dict(rows[1], coord_x="9.000"), rows[2]]
    with pytest.raises(prep.PreparationError, match="hash of its stored identifiers|differs from its primary"):
        prep.check_row_accounting(moved, expected_rows=3)
    wrong_alias = [rows[0], dict(rows[1], alias_primary_source_uid="u2"), rows[2]]
    with pytest.raises(prep.PreparationError, match="first row of its physical ion"):
        prep.check_row_accounting(wrong_alias, expected_rows=3)
    forged = [dict(rows[0], ion_key="f" * 64, alias_primary_source_uid="u1")] + rows[1:]
    with pytest.raises(prep.PreparationError, match="hash of its stored identifiers"):
        prep.check_row_accounting(forged, expected_rows=3)
    unscorable_key = rows[:2] + [dict(rows[2], ion_key=key)]
    with pytest.raises(prep.PreparationError, match="unscorable row carries"):
        prep.check_row_accounting(unscorable_key, expected_rows=3)


# ---------------------------------------------------------------------------
# Clean subset (criteria C1-C3)
# ---------------------------------------------------------------------------

AMINO = "ACDEFGHIKLMNPQRSTVWY"


def mutate(sequence: str, fraction: float, rng: random.Random) -> str:
    chars = list(sequence)
    for index in rng.sample(range(len(chars)), int(len(chars) * fraction)):
        chars[index] = rng.choice([a for a in AMINO if a != chars[index]])
    return "".join(chars)


def test_near_copy_hits_equal_the_brute_force_a1_rule():
    from benchmarking.v3_folds import IDENTITY_THRESHOLD, MIN_CHAIN_LENGTH, near_copy_identity

    rng = random.Random(7)
    references = ["".join(rng.choice(AMINO) for _ in range(rng.randint(35, 80))) for _ in range(11)]
    references.append("".join(rng.choice(AMINO) for _ in range(25)))  # too short to be a near-copy
    unknown = "X" * 20 + references[6][20:]  # over 10% non-standard: below 90% identity even against itself
    references.append(unknown)
    queries = [mutate(references[0], 0.05, rng), mutate(references[1], 0.08, rng), mutate(references[2], 0.15, rng),
               references[3][5:], references[4] + "".join(rng.choice(AMINO) for _ in range(40)),
               "".join(rng.choice(AMINO) for _ in range(60)), references[5][:20], unknown, references[11]]
    brute = [len(q) >= MIN_CHAIN_LENGTH and any(len(r) >= MIN_CHAIN_LENGTH and (q == r or near_copy_identity(q, r)[0]
                                                                                 >= IDENTITY_THRESHOLD)
                                                 for r in references) for q in queries]
    assert prep.near_copy_hits(queries, references) == brute
    assert brute[:2] == [True, True] and brute[2] is False and brute[6] is False
    # The identical chain with many unknown residues matches (as A1 links identical sequences); its identity alone
    # would not. A 25-residue chain identical to a reference stays below the A1 length and never matches.
    assert near_copy_identity(unknown, unknown)[0] < IDENTITY_THRESHOLD and brute[7] is True
    assert len(references[11]) == 25 and brute[8] is False


def test_clean_subset_codes_follow_c1_to_c3_and_unverifiable_rows_are_excluded():
    rng = random.Random(3)
    training = "".join(rng.choice(AMINO) for _ in range(60))
    near, other = mutate(training, 0.05, rng), "".join(rng.choice(AMINO) for _ in range(60))
    sequences = {("s1", "A"): near, ("s2", "A"): other, ("s3", "A"): "ACDEFGHIK", ("s4", "A"): other}

    def row(n, pdbid, structure, chains):
        return {"source_uid": f"u{n}", "pdbid": pdbid, "structure_name": structure, "context_chains": chains}

    rows = [row(1, "1aaa", "s1", "A"), row(2, "2bbb", "s2", "A"), row(3, "3ccc", "s3", "A"), row(4, "4ddd", "s2", ""),
            row(5, "5eee", "s4", "A;Z"), row(6, "6fff", "s2", "A"), row(7, "7ggg", "s2", "A")]
    out = prep.apply_clean_subset(rows, sequences, training_pdbids={"6fff"}, training_sequences=[training],
                                  previously_opened={"7ggg", "1aaa"})
    codes = {r["source_uid"]: r["clean_exclusions"] for r in rows}
    assert codes == {"u1": "C2_near_copy_chain;C3_previously_opened_pdb_id", "u2": "",
                     "u3": "",  # a chain under 30 residues is never a near-copy under A1 (the disclosed limitation)
                     "u4": "C2_unverifiable", "u5": "C2_unverifiable",  # no context chain / one without a sequence
                     "u6": "C1_training_pdb_id", "u7": "C3_previously_opened_pdb_id"}
    assert [r["clean_subset"] for r in rows] == [False, True, True, False, False, False, False]
    assert out["clean_subset_rows"] == 2 and out["near_copy_sequences"] == 1
    assert out["rows_with_only_short_context_chains"] == 1 and out["identical_to_a_reference"] == 0
    # The short training peptide itself in a test pocket: still not excluded (no short-chain exclusion is added).
    short = [row(8, "8hhh", "s3", "A")]
    prep.apply_clean_subset(short, sequences, training_pdbids=set(), training_sequences=["ACDEFGHIK"],
                            previously_opened=set())
    assert short[0]["clean_exclusions"] == "" and short[0]["clean_subset"] is True


def test_an_identical_long_chain_with_unknown_residues_excludes_its_row():
    rng = random.Random(5)
    chain = "X" * 20 + "".join(rng.choice(AMINO) for _ in range(51))  # 4uer chain 9 shape: 51 of 71 standard
    rows = [{"source_uid": "u1", "pdbid": "1aaa", "structure_name": "s1", "context_chains": "A"}]
    out = prep.apply_clean_subset(rows, {("s1", "A"): chain}, training_pdbids=set(), training_sequences=[chain],
                                  previously_opened=set())
    assert rows[0]["clean_exclusions"] == "C2_near_copy_chain" and out["identical_to_a_reference"] == 1


def test_previously_opened_ids_come_from_file_names_and_need_the_ledger_count(tmp_path):
    folder = tmp_path / "opened" / "structures"
    folder.mkdir(parents=True)
    for name in ("1abc__chain_A.pdb", "1ABC__chain_B.pdb", "2xyz.cif", "notes.txt"):
        (folder / name).write_text("x")
    assert vm.previously_opened_ids(tmp_path / "opened", expected_files=3) == {"1abc", "2xyz"}
    with pytest.raises(ValueError, match="not 316"):
        vm.previously_opened_ids(tmp_path / "opened", expected_files=316)


# ---------------------------------------------------------------------------
# Test-side ESMC payloads (injected embedder)
# ---------------------------------------------------------------------------

def fake_embedder(dim: int):
    import torch

    def embed(sequence: str):
        seed = int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16)
        return torch.randn(len(sequence), dim, generator=torch.Generator().manual_seed(seed))

    return embed


def test_embeddings_follow_the_plan_once_per_sequence_in_the_training_format(tmp_path, monkeypatch):
    import torch

    from benchmarking import pmm_ion_features as features

    monkeypatch.setattr(features, "ESM_DIM", 8)
    test, _, rows, extra = prepared(tmp_path)
    plan = prep.esm_plan(rows, extra["sequences"])
    assert {(p["structure_name"].split("__")[0], p["chain"]) for p in plan} == {
        (s, ch) for s in ("1aba", "7ttt", "7uuu", "7vvv", "7www", "7zzz") for ch in "AB"}
    receipt = prep.generate_embeddings(plan, test, tmp_path / "esm", embed=fake_embedder(8))
    assert receipt["files"] == len(plan) and receipt["model"] == "injected-test-embedder"
    payload = torch.load(tmp_path / "esm" / receipt["inventory"][0]["file"], weights_only=False)
    assert payload["embeddings"].shape[1] == 8 and (tmp_path / "esm" / (receipt["inventory"][0]["file"] + ".json")).is_file()
    with pytest.raises(FileExistsError):
        prep.generate_embeddings(plan, test, tmp_path / "esm", embed=fake_embedder(8))
    bad = [dict(plan[0], sequence_sha256="0" * 64)]
    with pytest.raises(prep.PreparationError, match="differs from the plan"):
        prep.generate_embeddings(bad, test, tmp_path / "esm2", embed=fake_embedder(8))


# ---------------------------------------------------------------------------
# The report on hand-made frozen inputs (no model)
# ---------------------------------------------------------------------------

def write_record(path: Path, record: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return sha(path)


def record_freeze1(campaign: Path, checkpoint_sha: str, **changes) -> str:
    """A recorded Freeze 1 at its canonical path, binding the current code (what record-freeze1 writes, reduced to the
    fields the test-side gates read)."""
    record = {"kind": "step_f_freeze_1", "user_go_ahead_log_entry": "v3-901", "step_f_spec_sha256": sha(prep.SPEC_PATH),
              "checkpoint_sha256": checkpoint_sha, "tool_sha256": {name: sha(ROOT / name) for name in f.STEP_F_TOOL_FILES},
              "cohort_sha256": "d" * 64, "test_source_csv_sha256": "e" * 64, "previously_opened_files": 316}
    return write_record(prep.evidence_paths(campaign)["freeze1"], {**record, **changes})


def record_freeze2(campaign: Path, freeze1_sha: str, prepared_dir: Path, *, checkpoint_sha: str, inventory_sha: str,
                   source_sha: str, rows: int, **changes) -> str:
    record = {"kind": "step_f_freeze_2", "user_go_ahead_log_entry": "v3-902", "step_f_spec_sha256": sha(prep.SPEC_PATH),
              "freeze1_sha256": freeze1_sha, "prepared_files_sha256": f.prepared_hashes(prepared_dir),
              "checkpoint_sha256": checkpoint_sha, "test_esm_inventory_sha256": inventory_sha,
              "test_source_csv_sha256": source_sha, "test_source_rows": rows}
    return write_record(prep.evidence_paths(campaign)["freeze2"], {**record, **changes})


def frozen_inputs(tmp_path: Path, cases: list[tuple], *, clean=()) -> dict[str, Path]:
    """cases: (pdbid, label element or other code, disposition, link, alias_of_index or None, predicted class or None).
    Canonical Freeze 1, preparation, Freeze 2, prediction claim, predictions and receipt, as the pass writes them."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "labels.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pdbid", "residueid_ion", "label_metal"])
        for index, (pdbid, label, *_rest) in enumerate(cases, start=1):
            writer.writerow([pdbid, index, CODE.get(label, label)])
    source_sha = sha(source)
    campaign = tmp_path / "campaign"
    freeze1_sha = record_freeze1(campaign, "c" * 64)
    rows = []
    for index, (pdbid, _label, disposition, link, alias, _pred) in enumerate(cases, start=1):
        row = {column: "" for column in prep.ROW_COLUMNS}
        uid = f"sha256:{source_sha}:row:{index}"
        row.update(source_uid=uid, source_row=index, pdbid=pdbid, disposition=disposition, link_flagged=link,
                   clean_subset=index in clean, reason="" if disposition != "unscorable" else "structure_unverified",
                   context_chains="A" if disposition != "unscorable" else "")
        if disposition != "unscorable":
            row["alias_primary_source_uid"] = uid if alias is None else f"sha256:{source_sha}:row:{alias}"
        rows.append(row)
    prepared_dir = tmp_path / "prepared"
    prep.write_preparation(prepared_dir, prep.membership_files(rows, [], previously_opened=set(), training_pdbids=set()),
                           {"source_csv_sha256": source_sha, "source_rows": len(rows), "row_subset": None},
                           freeze1_sha256=freeze1_sha)
    freeze2_sha = record_freeze2(campaign, freeze1_sha, prepared_dir, checkpoint_sha="c" * 64, inventory_sha="e" * 64,
                                 source_sha=source_sha, rows=len(rows))
    predictions_dir = tmp_path / "predictions"
    _, claim_sha = prep.claim_prediction(campaign, {"user_go_ahead_log_entry": "v3-903",
                                                    "freeze1_sha256": freeze1_sha, "freeze2_sha256": freeze2_sha,
                                                    "checkpoint_sha256": "c" * 64,
                                                    "out_dir": str(predictions_dir.resolve())})
    predictions_dir.mkdir()
    path = predictions_dir / "test_predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(vm.PREDICTION_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row, case in zip(rows, cases):
            record = {key: row[key] for key in ("source_uid", "source_row", "pdbid", "disposition", "reason",
                                                "link_flagged", "alias_primary_source_uid", "clean_subset")}
            record.update(checkpoint_sha256="c" * 64, epochs=50, pred_common4="" if case[5] is None else case[5])
            for k, label in enumerate(vm.COMMON4):
                record[f"p_common4_{label.replace(' ', '_')}"] = "" if case[5] is None else f"{float(k == case[5]):.8f}"
            writer.writerow(record)
    (predictions_dir / "prediction_receipt.json").write_text(json.dumps({
        "predictions_sha256": sha(path), "freeze2_sha256": freeze2_sha, "prediction_claim_sha256": claim_sha,
        "labels_read": False, "checkpoint_sha256": "c" * 64}))
    return {"campaign_dir": campaign, "predictions_dir": predictions_dir, "prepared_dir": prepared_dir,
            "source_csv": source}


# Mn 0, Cu 1, Zn 2, Class VIII 3
CASES = [("1aaa", "MN", "scored", False, None, 0), ("1aaa", "MN", "scored", False, None, 2),
         ("2bbb", "CU", "scored", False, None, 1), ("2bbb", "CU", "unscorable", False, None, None),
         ("3ccc", "ZN", "scored_link_flagged", True, None, 2), ("3ccc", "ZN", "scored_link_flagged", True, 5, 2),
         ("4ddd", "FE", "scored", False, None, 3), ("4ddd", "NI", "scored", False, None, 0),
         ("5eee", "CA", "scored", False, None, 1)]


def test_report_values_are_hand_computable_with_unscorable_rows_as_errors(tmp_path):
    inputs = frozen_inputs(tmp_path, CASES, clean=(1, 2, 3, 7))
    out = report.build_report(**inputs)
    full = out["lines"]["full_test"]
    assert out["source_rows"] == 9 and out["scored_statement"] == "8 of 9 scored"
    assert out["rows_with_other_label_codes"] == {"count": 1, "rows": [{"source_row": 9, "label_code": "3"}],
                                                  "rule": report.LABELS["other_codes"]}
    assert full["support"] == {"Mn": 2, "Cu": 2, "Zn": 2, "Class VIII": 2}
    assert full["correct"] == {"Mn": 1, "Cu": 1, "Zn": 2, "Class VIII": 1}
    assert full["balanced_accuracy"] == pytest.approx((0.5 + 0.5 + 1.0 + 0.5) / 4)
    assert (full["rows"], full["scored"], full["unscorable_counted_as_errors"]) == (8, 7, 1)
    link_free = out["lines"]["link_free"]
    assert link_free["support"]["Zn"] == 0 and link_free["balanced_accuracy"] is None
    eligible = out["lines"]["training_eligible"]
    assert eligible["rows"] == 5 and eligible["support"]["Cu"] == 1  # scored, not LINK, primary
    clean = out["lines"]["clean_subset"]
    assert clean["rows"] == 4 and clean["recalls"]["Mn"] == 0.5 and clean["recalls"]["Zn"] is None
    comparison = out["pmm_comparison"]["values"]
    assert comparison["mean"] == {"deepmzyme_v3_percent": 62.5, "pmm_published_percent": 67.85, "relation": "lower than"}
    assert comparison["Zn"]["relation"] == "higher than"
    assert out["label"] == "both_results_secondary" and out["disclosures"] == f.load_spec()["disclosures"]
    boot = full["bootstrap"]
    assert boot["entries"] == 4 and boot["resamples"] == 10000 and boot["unit"].startswith("PDB entry")
    assert boot["balanced_accuracy"]["lower"] <= full["balanced_accuracy"] <= boot["balanced_accuracy"]["upper"]
    assert boot["balanced_accuracy"]["defined_resamples"] < 10000  # resamples missing a class are left out


def test_the_report_is_reproducible_byte_for_byte_and_refuses_changed_inputs(tmp_path):
    inputs = frozen_inputs(tmp_path, CASES, clean=(1, 2))
    first = report.write_report(report.build_report(**inputs), tmp_path / "r1", paths={})
    second = report.write_report(report.build_report(**inputs), tmp_path / "r2", paths={})
    assert (tmp_path / "r1" / "report.json").read_bytes() == (tmp_path / "r2" / "report.json").read_bytes()
    assert (tmp_path / "r1" / "report.md").read_bytes() == (tmp_path / "r2" / "report.md").read_bytes()
    assert first["report_json_sha256"] == second["report_json_sha256"]
    assert "not a matched comparison" in (tmp_path / "r1" / "report.md").read_text()
    with pytest.raises(FileExistsError):
        report.write_report(report.build_report(**inputs), tmp_path / "r1", paths={})
    path = inputs["predictions_dir"] / "test_predictions.csv"
    path.write_text(path.read_text().replace(",0.00000000,", ",0.00000001,", 1))
    with pytest.raises(ValueError, match="changed after the pass"):
        report.build_report(**inputs)


def test_the_report_refuses_a_freeze2_mismatch_or_a_different_label_file(tmp_path):
    inputs = frozen_inputs(tmp_path, CASES)
    (inputs["prepared_dir"] / "clean_exclusions.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="differs from Freeze 2"):
        report.build_report(**inputs)
    inputs = frozen_inputs(tmp_path / "b", CASES)
    inputs["source_csv"].write_text(inputs["source_csv"].read_text() + "9zzz,10,1\n")
    with pytest.raises(ValueError, match="not the pinned test source CSV"):
        report.build_report(**inputs)


def test_the_report_accepts_only_the_claimed_predictions_directory(tmp_path):
    import shutil

    inputs = frozen_inputs(tmp_path, CASES)
    copy = tmp_path / "copy"
    shutil.copytree(inputs["predictions_dir"], copy)
    with pytest.raises(ValueError, match="not the one the single prediction pass claimed"):
        report.build_report(**{**inputs, "predictions_dir": copy})
    claim_path = prep.evidence_paths(inputs["campaign_dir"])["prediction"]
    claim_path.write_text(json.dumps({k: v for k, v in json.loads(claim_path.read_text()).items()
                                      if k != "user_go_ahead_log_entry"}))
    with pytest.raises(ValueError, match="names no logged go-ahead"):
        report.build_report(**inputs)
    claim_path.unlink()
    with pytest.raises(ValueError, match="No prediction claim"):
        report.build_report(**inputs)


# ---------------------------------------------------------------------------
# Launcher: selection record, manifest, gates, launch script and freeze records
# ---------------------------------------------------------------------------

def fake_campaign(tmp_path: Path, monkeypatch, *, selection=FUSION, outcome="not_adopted") -> Path:
    campaign = tmp_path / "campaign"
    ext_path = campaign / e2.EXTENSION_COPY
    ext_path.parent.mkdir(parents=True)
    ext_path.write_text(json.dumps({"extension_id": d.EXTENSION_ID, "runner_sha256": RUNNER}))
    spec = f.load_spec()
    e2_path = campaign / spec["selection"]["e2_assessment"]
    e2_path.parent.mkdir(parents=True)
    e2_path.write_text(json.dumps({"kind": "E2", "held_out_test_accessed": False,
                                   "result": {"status": "complete", "final_selection": {"selected": selection}}}))
    s10_path = campaign / spec["selection"]["schedule_assessment"]
    s10_path.parent.mkdir(parents=True)
    s10_path.write_text(json.dumps({"held_out_test_accessed": False, "result": {"status": "complete", "schedule_decision": {
        "outcome": outcome, "adopted": outcome == "adopted", "epochs": 50 if outcome == "not_adopted" else 10,
        "selection": FUSION, "checkpoint_rule": "terminal", "meaning": "the 50-epoch selection stands (E_sel = 50)"}}}))
    test_spec = json.loads(f.SPEC_JSON.read_text())
    test_spec["selection"].update(e2_assessment_sha256=sha(e2_path), schedule_assessment_sha256=sha(s10_path))
    spec_path = tmp_path / "step_f_spec.json"
    spec_path.write_text(json.dumps(test_spec, indent=2, sort_keys=True))
    monkeypatch.setattr(f, "SPEC_JSON", spec_path)
    monkeypatch.setattr(f, "FROZEN_SPEC_SHA256", sha(spec_path))
    readiness = campaign / "audits" / "d_readiness_20261006T200000Z" / "readiness_report.json"
    readiness.parent.mkdir(parents=True)
    readiness.write_text(json.dumps({"ready": True, "runner_sha256": RUNNER, "checks": {}}))
    return campaign


def test_the_selection_record_binds_e2_and_the_not_adopted_schedule(tmp_path, monkeypatch):
    campaign = fake_campaign(tmp_path, monkeypatch)
    record = f.selection_record(campaign, log_entry="v3-099")
    assert record["selection"]["cell"] == FUSION and record["selection"]["epochs"] == 50
    assert record["schedule_assessment"]["outcome"] == "not_adopted" and record["refit"]["run_name"] == REFIT
    assert "untested, not disproven" in record["deferred_untested"] and record["held_out_access"] is False


@pytest.mark.parametrize("kwargs,message", [({"selection": "only_esm__four_class__baseline"}, "E2 final selection"),
                                            ({"outcome": "adopted"}, "not adopted with E_sel = 50")])
def test_the_selection_record_refuses_another_selection_or_schedule(tmp_path, monkeypatch, kwargs, message):
    campaign = fake_campaign(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(f.LauncherError, match=message):
        f.selection_record(campaign, log_entry="v3-099")


def gated(tmp_path, monkeypatch, *, authorize=True, ceiling=True):
    campaign = fake_campaign(tmp_path, monkeypatch)
    records = f.record_paths(campaign)
    f.write_json_once(records["selection"], f.selection_record(campaign, log_entry="v3-099"))
    manifest = f.write_json_once(records["manifest"], f.build_manifest(campaign, log_entry="v3-099"))
    if ceiling:
        f.write_json_once(records["ceiling"], f.ceiling_record(
            usd=55.0, spent_usd=40.65, spent_at="2026-10-12T11:19:00Z", spent_source="log v3-041", log_entry="v3-099"))
    if authorize:
        f.write_json_once(records["authorization"], f.authorization_record(
            manifest_path=records["manifest"], ceiling_path=records["ceiling"], log_entry="v3-100"))
    return campaign, manifest


def ready_state(manifest):
    return vm_snapshot(extension={"id": d.EXTENSION_ID, "runner_sha256": RUNNER, "sha256": manifest["extension"]["sha256"]})


def step_for(campaign, manifest, *, vm_files=None, spend=lambda _ceiling: {"within_ceiling": True}):
    bound = {name: item["sha256"] for name, item in manifest["vm_files"].items()}
    return f.build_step(campaign, vm_files=vm_files or (lambda: bound), runner=RUNNER, spend=spend)


def test_an_authorized_manifest_admits_only_the_refit_with_its_forecast_and_runner(tmp_path, monkeypatch):
    campaign, manifest = gated(tmp_path, monkeypatch)
    assert set(manifest["vm_files"]) == set(f.VM_FILES) and f.verify_manifest(manifest, campaign=campaign) == []
    step = step_for(campaign, manifest)
    planned = c.plan_units([(REFIT, 0)], ready_state(manifest), step=step)
    assert planned == [{"unit": REFIT, "lane": 0, "fit_seconds": 3300.0}]
    with pytest.raises(f.LauncherError, match="not a step F refit unit"):
        c.plan_units([(f"{FUSION}__fold1__seed42", 0)], ready_state(manifest), step=step)
    fake = FakeVM(ready_state(manifest))
    out = f.cmd_launch(fake, session(), [(REFIT, 0)], step=step, campaign=campaign, wait=False)
    script = fake.launched()[0]
    selection_sha = sha(f.record_paths(campaign)["selection"])
    assert out["batch"].startswith("f-") and f"cd {d.REMOTE_CODE}" in script
    assert (f"run_pmm_v3_step_f.py --action refit --campaign-dir {f.b.REMOTE_V3} --unit {REFIT} "
            f"--selection-sha256 {selection_sha} --epochs 50") in script
    assert "run_pmm_v3_campaign.py" not in script and "--test-dir" not in script


@pytest.mark.parametrize("change,message", [({"authorize": False}, "no recorded GPU-start authorization"),
                                            ({"ceiling": False, "authorize": False}, "no ceiling record")])
def test_no_refit_launch_without_the_recorded_authorization_or_ceiling(tmp_path, monkeypatch, change, message):
    campaign, manifest = gated(tmp_path, monkeypatch, **change)
    with pytest.raises(f.LauncherError, match=message):
        c.plan_units([(REFIT, 0)], ready_state(manifest), step=step_for(campaign, manifest))


def test_no_refit_launch_with_unbound_vm_files_above_the_ceiling_or_changed_tooling(tmp_path, monkeypatch):
    campaign, manifest = gated(tmp_path, monkeypatch)
    missing = lambda: {name: None for name in f.VM_FILES}  # noqa: E731
    with pytest.raises(f.LauncherError, match="VM copy of pmm_v3_test_prep.py"):
        c.plan_units([(REFIT, 0)], ready_state(manifest), step=step_for(campaign, manifest, vm_files=missing))
    above = lambda _c: {"within_ceiling": False, "committed_usd": 54.0, "reserve_usd": 3.0, "ceiling_usd": 55.0}  # noqa: E731
    with pytest.raises(f.LauncherError, match="exceed the recorded 55.00 USD ceiling"):
        c.plan_units([(REFIT, 0)], ready_state(manifest), step=step_for(campaign, manifest, spend=above))
    changed = dict(manifest, tooling_sha256={**manifest["tooling_sha256"], "pmm_v3_step_f_report.py": "0" * 64})
    assert any("tooling" in p for p in f.verify_manifest(changed, campaign=campaign))


def test_deploy_needs_the_authorization_and_never_overwrites_a_different_file(tmp_path, monkeypatch):
    campaign, manifest = gated(tmp_path, monkeypatch, authorize=False)

    class Remote:
        def __init__(self, existing):
            self.existing, self.put = existing, []
            self.transport = self

        def run(self, command):
            for name, path in f.REMOTE_FILES.items():
                if command.startswith(f"if [ -f {path} ]") and name in self.existing:
                    return type("R", (), {"stdout": f"{self.existing[name]}  {path}\n"})()
            return type("R", (), {"stdout": ""})()

    with pytest.raises(f.LauncherError, match="No recorded GPU-start authorization"):
        f.cmd_deploy(Remote({}), campaign)
    records = f.record_paths(campaign)
    f.write_json_once(records["authorization"], f.authorization_record(
        manifest_path=records["manifest"], ceiling_path=records["ceiling"], log_entry="v3-100"))
    with pytest.raises(f.LauncherError, match="never overwrite"):
        f.cmd_deploy(Remote({f.VM_RUNNER: "1" * 64}), campaign)


def test_freeze_templates_mark_what_is_unknown_and_records_keep_the_freeze_order(tmp_path, monkeypatch):
    campaign = fake_campaign(tmp_path, monkeypatch)
    template = f.freeze1_record(campaign, log_entry=None, template=True)
    assert template["checkpoint_sha256"] == f.LEDGER_PLACEHOLDER and template["test_side_read_before_this_record"] is False
    assert set(template["tool_sha256"]) == set(f.STEP_F_TOOL_FILES) and template["label"] == "both_results_secondary"
    assert template["test_source_csv_sha256"].startswith("ec6427ad") and template["previously_opened_files"] == 316
    assert "Freeze 1" in f.ledger_text(template) and "before any test-side read" in f.ledger_text(template)
    with pytest.raises(f.LauncherError, match="needs the verified refit"):
        f.freeze1_record(campaign, log_entry="v3-101")
    second = f.freeze2_record(campaign, log_entry=None, prepared_dir=None, test_esm_dir=None, template=True)
    assert second["prepared_files_sha256"] == f.LEDGER_PLACEHOLDER and second["labels_read"] is False
    assert "before any prediction or label read" in f.ledger_text(second)
    with pytest.raises(f.LauncherError, match="Freeze 1 is not recorded"):
        f.freeze2_record(campaign, log_entry="v3-102", prepared_dir=tmp_path, test_esm_dir=tmp_path)


def fake_refit(monkeypatch, runner_sha: str | None = None) -> None:
    """refit-receipt of a verified refit that ran this worktree's VM runner (or ``runner_sha``)."""
    receipt = {"checkpoint_sha256": "c" * 64, "run_config_sha256": "1" * 64, "run_metadata_sha256": "2" * 64,
               "status_sha256": "3" * 64, "reload_check": {"passed": True},
               "identity": {"cohort_sha256": "d" * 64,
                            "step_f_runner_sha256": runner_sha or sha(ROOT / f.VM_RUNNER)}}
    monkeypatch.setattr(f, "refit_receipt", lambda campaign=None: receipt)


def test_freeze_1_binds_the_refit_manifests_code_and_an_amendment_is_explicit(tmp_path, monkeypatch):
    campaign, _ = gated(tmp_path, monkeypatch)
    paths = f.record_paths(campaign)
    fake_refit(monkeypatch)
    record = f.freeze1_record(campaign, log_entry="v3-101")
    assert record["tools_differ_from_refit_manifest"] == [] and record["amends"] is None
    assert record["manifest_sha256"] == sha(paths["manifest"]) and record["test_esm"]["model_dtype"] == "bfloat16"
    assert record["test_side_read_before_this_record"] is False and "git_commit" in record
    with pytest.raises(f.LauncherError, match="go-ahead"):
        f.freeze1_record(campaign, log_entry="v3-dry-run")
    f.write_json_once(paths["freeze1"], record)
    first = sha(paths["freeze1"])
    # A fix after Freeze 1 (here: the runner differs from the refit's): only an amendment may bind it.
    fake_refit(monkeypatch, runner_sha="9" * 64)
    archived = paths["freeze1"].with_name("freeze1.superseded_20261012T170000Z.json")
    paths["freeze1"].rename(archived)
    with pytest.raises(f.LauncherError, match="record this one with --amends"):
        f.freeze1_record(campaign, log_entry="v3-103")
    with pytest.raises(f.LauncherError, match="No archived Freeze 1"):
        f.freeze1_record(campaign, log_entry="v3-103", amends="0" * 64, prior_test_side_read="none")
    with pytest.raises(f.LauncherError, match="prior-test-side-read"):
        f.freeze1_record(campaign, log_entry="v3-103", amends=first)
    amended = f.freeze1_record(campaign, log_entry="v3-103", amends=first, prior_test_side_read="yes")
    assert amended["amends"] == {"superseded_sha256": first, "archived_path": str(archived), "prior_test_side_read": "yes"}
    assert amended["tools_differ_from_refit_manifest"] == [f.VM_RUNNER] and amended["test_side_read_before_this_record"]
    assert "Amended Freeze 1" in f.ledger_text(amended) and "test-side read under it: yes" in f.ledger_text(amended)
    write_record(paths["freeze2"], {"kind": "step_f_freeze_2"})
    with pytest.raises(f.LauncherError, match="not amended after them"):
        f.freeze1_record(campaign, log_entry="v3-104", amends=first, prior_test_side_read="none")


def test_freeze_1_refuses_code_that_differs_from_the_refit_manifest(tmp_path, monkeypatch):
    campaign, _ = gated(tmp_path, monkeypatch)
    fake_refit(monkeypatch, runner_sha="9" * 64)
    with pytest.raises(f.LauncherError, match="differs from the refit manifest's"):
        f.freeze1_record(campaign, log_entry="v3-101")


def test_record_freeze_2_needs_the_recorded_freeze_1_and_outputs_made_under_it(tmp_path, monkeypatch):
    side = prepared_campaign(tmp_path, monkeypatch, "c" * 64)
    campaign, prepared_dir, esm = side["campaign"], side["prepared_dir"], side["esm"]
    f.record_paths(campaign)["freeze2"].unlink()  # record-freeze2 writes it here
    spec = f.load_spec()  # the synthetic source stands in for the pinned test source CSV
    spec["test_inputs"]["source_csv_sha256"], spec["test_rows"]["source_rows"] = sha(side["source"]), 19
    monkeypatch.setattr(f, "load_spec", lambda path=None: spec)
    with pytest.raises(f.LauncherError, match="CPU route"):  # the injected test embedder is not the CPU route
        f.freeze2_record(campaign, log_entry="v3-102", prepared_dir=prepared_dir, test_esm_dir=esm)
    receipt_path = esm / "test_esm_receipt.json"
    receipt_path.write_text(json.dumps({**json.loads(receipt_path.read_text()), "device": "cpu",
                                        "model_dtype": "bfloat16"}))
    record = f.freeze2_record(campaign, log_entry="v3-102", prepared_dir=prepared_dir, test_esm_dir=esm)
    assert record["freeze1_sha256"] == side["freeze1_sha"]
    assert set(record["prepared_files_sha256"]) == set(f.MEMBERSHIP_FILE_NAMES) | {"preparation_summary.json"}
    assert f.MEMBERSHIP_FILE_NAMES == prep.MEMBERSHIP_FILE_NAMES and f.PREDICTION_CLAIM_NAME == \
        prep.RECORD_NAMES["prediction"] and f.EVIDENCE_DIR == prep.EVIDENCE_DIR
    summary_path = prepared_dir / "preparation_summary.json"
    summary_path.write_text(json.dumps({**json.loads(summary_path.read_text()), "freeze1_sha256": "0" * 64}))
    with pytest.raises(f.LauncherError, match="under the recorded Freeze 1"):
        f.freeze2_record(campaign, log_entry="v3-102", prepared_dir=prepared_dir, test_esm_dir=esm)
    prep.claim_prediction(campaign, {"freeze2_sha256": "x"})
    with pytest.raises(f.LauncherError, match="Freeze 2 precedes it"):
        f.freeze2_record(campaign, log_entry="v3-102", prepared_dir=prepared_dir, test_esm_dir=esm)


def test_records_that_name_a_git_commit_need_the_bound_files_committed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                       check=True, capture_output=True)

    git("init", "-q")
    (repo / "a.py").write_text("1\n")
    git("add", "a.py")
    git("commit", "-q", "-m", "a")
    assert f.git_binding(repo, ("a.py",))["bound_files_committed"] is True
    (repo / "a.py").write_text("2\n")
    (repo / "b.py").write_text("x\n")
    assert f.git_binding(repo, ("a.py", "b.py"))["uncommitted"] == ["a.py", "b.py"]
    # The command line refuses to write such a record; nothing is written anywhere (the records point to tmp_path).
    monkeypatch.setattr(f, "CAMPAIGN", tmp_path / "campaign")
    monkeypatch.setattr(f, "write_json_once", lambda *a, **k: pytest.fail("a record was written"))
    monkeypatch.setattr(f, "git_binding", lambda root=f.ROOT, files=f.BOUND_FILES: {
        "commit": "0" * 40, "bound_files_committed": False, "uncommitted": ["pmm_v3_step_f.py"]})
    for command in (["bind-selection", "--log-entry", "v3-104"], ["build-manifest", "--log-entry", "v3-104"],
                    ["record-freeze1", "--log-entry", "v3-104"]):
        with pytest.raises(f.LauncherError, match="Commit the bound step F files first"):
            f.main(command)


def test_the_test_side_reads_only_the_recorded_freeze_1_with_a_go_ahead_and_the_bound_code(tmp_path):
    campaign = tmp_path / "campaign"
    tools = vm.test_side_tools()
    with pytest.raises(ValueError, match="Freeze 1 is not recorded"):
        prep.check_freeze1(campaign, tools=tools)
    elsewhere = tmp_path / "freeze1.json"  # a record outside the canonical path is never read
    write_record(elsewhere, {"kind": "step_f_freeze_1", "user_go_ahead_log_entry": "v3-901"})
    with pytest.raises(ValueError, match="Freeze 1 is not recorded"):
        prep.check_freeze1(campaign, tools=tools)
    record_freeze1(campaign, "c" * 64, user_go_ahead_log_entry=f.LEDGER_PLACEHOLDER)  # a template's go-ahead
    with pytest.raises(ValueError, match="logged go-ahead"):
        prep.check_freeze1(campaign, tools=tools)
    record_freeze1(campaign, "c" * 64, cohort_sha256=f.LEDGER_PLACEHOLDER)  # any other unfilled field
    with pytest.raises(ValueError, match="a template is not a record"):
        prep.check_freeze1(campaign, tools=tools)
    write_record(prep.evidence_paths(campaign)["freeze1"], {"record": {"kind": "step_f_freeze_1"}, "ledger_text": ""})
    with pytest.raises(ValueError, match="logged go-ahead"):  # the printed template wrapper
        prep.check_freeze1(campaign, tools=tools)
    record_freeze1(campaign, "c" * 64, tool_sha256={**{n: sha(ROOT / n) for n in f.STEP_F_TOOL_FILES},
                                                  "run_pmm_v3_step_f.py": "0" * 64})
    with pytest.raises(ValueError, match="amended Freeze 1"):  # code changed after Freeze 1
        prep.check_freeze1(campaign, tools=tools)
    record_freeze1(campaign, "c" * 64, step_f_spec_sha256="0" * 64)
    with pytest.raises(ValueError, match="another specification"):
        prep.check_freeze1(campaign, tools=tools)
    record_freeze1(campaign, "c" * 64)
    record, digest = prep.check_freeze1(campaign, tools=tools)
    assert record["checkpoint_sha256"] == "c" * 64 and digest == sha(prep.evidence_paths(campaign)["freeze1"])
    assert vm.require_preparation_window(campaign)[1] == digest
    write_record(prep.evidence_paths(campaign)["freeze2"], {"kind": "step_f_freeze_2"})
    with pytest.raises(ValueError, match="preparation is closed"):  # after Freeze 2 the memberships are final
        vm.require_preparation_window(campaign)
    with pytest.raises(ValueError, match="preparation is closed"):
        vm.main(["--action", "prepare-test", "--campaign-dir", str(campaign)])
    with pytest.raises(ValueError, match="needs --campaign-dir"):
        vm.main(["--action", "prepare-test"])


def test_the_test_esm_settings_are_machine_readable_for_the_real_model_branch():
    """The real ESMC branch (not reached by the injected test embedder) casts with these fields: each must name a
    torch dtype exactly (log v3-044: a descriptive storage_dtype crashed the CPU check at its first chain)."""
    import torch

    for key in ("model_dtype", "storage_dtype"):
        assert isinstance(getattr(torch, prep.TEST_ESM[key]), torch.dtype), key
    assert (prep.TEST_ESM["model"], prep.TEST_ESM["device"], prep.TEST_ESM["storage_dtype"]) == (
        "esmc_600m", "cpu", "float32")


def test_test_embeddings_run_on_the_cpu_route_only(tmp_path):
    campaign = tmp_path / "campaign"
    record_freeze1(campaign, "c" * 64)
    with pytest.raises(ValueError, match="workstation CPU only"):
        vm.main(["--action", "test-embeddings", "--campaign-dir", str(campaign), "--device", "cuda",
                 "--prepared-dir", str(tmp_path), "--test-dir", str(tmp_path), "--out-dir", str(tmp_path / "esm")])
    assert prep.TEST_ESM["device"] == "cpu" and prep.TEST_ESM["model_dtype"] == "bfloat16"


def prepared_campaign(tmp_path: Path, monkeypatch, checkpoint_sha: str) -> dict[str, Any]:
    """A synthetic test side prepared and embedded (injected embedder) under a recorded Freeze 1, with Freeze 2."""
    from benchmarking import pmm_ion_features as features

    monkeypatch.setattr(features, "ESM_DIM", 8)
    campaign = tmp_path / "campaign"
    freeze1_sha = record_freeze1(campaign, checkpoint_sha)
    test, source = build_test_side(tmp_path / "dataset")
    prepared_dir = tmp_path / "prepared"
    prep.prepare(test, source, prepared_dir, expected_source_sha256=sha(source), training_pdbids=set(),
                 training_sequences=lambda: [], previously_opened=set(), freeze1_sha256=freeze1_sha)
    with (prepared_dir / "test_esm_plan.csv").open(encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    receipt = prep.generate_embeddings(plan, test, tmp_path / "esm", embed=fake_embedder(8), freeze1_sha256=freeze1_sha)
    freeze2_sha = record_freeze2(campaign, freeze1_sha, prepared_dir, checkpoint_sha=checkpoint_sha,
                                 inventory_sha=receipt["inventory_sha256"], source_sha=sha(source), rows=19)
    return {"campaign": campaign, "test": test, "source": source, "prepared_dir": prepared_dir, "esm": tmp_path / "esm",
            "freeze1_sha": freeze1_sha, "freeze2_sha": freeze2_sha, "receipt": receipt}


def test_freeze_2_binds_every_membership_file_the_freeze_1_and_the_recomputed_inventory(tmp_path, monkeypatch):
    side = prepared_campaign(tmp_path, monkeypatch, "c" * 64)
    campaign, prepared_dir, esm = side["campaign"], side["prepared_dir"], side["esm"]
    record, digest = prep.check_freeze2(campaign, freeze1_sha256=side["freeze1_sha"], prepared_dir=prepared_dir,
                                        esm_dir=esm)
    assert digest == side["freeze2_sha"] and side["receipt"]["freeze1_sha256"] == side["freeze1_sha"]
    summary = json.loads((prepared_dir / "preparation_summary.json").read_text())
    assert summary["freeze1_sha256"] == side["freeze1_sha"] and summary["aliases"] == 1
    with pytest.raises(ValueError, match="not the one checked"):
        prep.check_freeze2(campaign, freeze1_sha256="0" * 64, prepared_dir=prepared_dir, esm_dir=esm)
    common = dict(checkpoint_sha="c" * 64, inventory_sha=side["receipt"]["inventory_sha256"],
                  source_sha=sha(side["source"]), rows=19)
    record_freeze2(campaign, "0" * 64, prepared_dir, **common)
    with pytest.raises(ValueError, match="another Freeze 1"):
        prep.check_freeze2(campaign, freeze1_sha256=side["freeze1_sha"], prepared_dir=prepared_dir, esm_dir=esm)
    record_freeze2(campaign, side["freeze1_sha"], prepared_dir, prepared_files_sha256={}, **common)
    with pytest.raises(ValueError, match="every prepared membership file"):  # an empty binding is not a binding
        prep.check_freeze2(campaign, freeze1_sha256=side["freeze1_sha"], prepared_dir=prepared_dir, esm_dir=esm)
    record_freeze2(campaign, side["freeze1_sha"], prepared_dir, **{**common, "inventory_sha": "e" * 64})
    with pytest.raises(ValueError, match="embedding inventory differs"):
        prep.check_freeze2(campaign, freeze1_sha256=side["freeze1_sha"], prepared_dir=prepared_dir, esm_dir=esm)
    record_freeze2(campaign, side["freeze1_sha"], prepared_dir, **common)
    receipt_path = esm / "test_esm_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["inventory"][0]["sha256"] = "0" * 64  # the list edited, its hash left: recomputation catches it
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="embedding inventory differs"):
        prep.check_freeze2(campaign, freeze1_sha256=side["freeze1_sha"], prepared_dir=prepared_dir, esm_dir=esm)


def test_a_second_prediction_pass_is_refused_campaign_wide_before_any_test_read(tmp_path, monkeypatch):
    checkpoint = tmp_path / "bound.pt"
    checkpoint.write_bytes(b"bound refit checkpoint")
    side = prepared_campaign(tmp_path, monkeypatch, sha(checkpoint))
    kwargs = dict(campaign_dir=side["campaign"], prepared_dir=side["prepared_dir"], checkpoint_path=checkpoint,
                  test_dir=side["test"], esm_dir=side["esm"], external_root=None, device="cpu",
                  source_csv=side["source"], log_entry="v3-903")
    with pytest.raises(ValueError, match="go-ahead for the single test pass"):
        vm.predict(out_dir=tmp_path / "p0", **{**kwargs, "log_entry": f.LEDGER_PLACEHOLDER})
    other = tmp_path / "other.pt"
    other.write_bytes(b"another checkpoint")
    with pytest.raises(ValueError, match="not the bound refit"):
        vm.predict(out_dir=tmp_path / "p0", **{**kwargs, "checkpoint_path": other})
    prep.claim_prediction(side["campaign"], {"freeze2_sha256": side["freeze2_sha"], "out_dir": str(tmp_path / "p1")})
    for out in ("p1", "p2"):  # the claimed directory and any other one
        with pytest.raises(ValueError, match="already claimed"):
            vm.predict(out_dir=tmp_path / out, **kwargs)
    assert not (tmp_path / "p2").exists()
    with pytest.raises(ValueError, match="already claimed"):
        prep.claim_prediction(side["campaign"], {"freeze2_sha256": side["freeze2_sha"]})
    with pytest.raises(ValueError, match="needs --source-csv"):
        vm.main(["--action", "predict", "--campaign-dir", str(side["campaign"])])


# ---------------------------------------------------------------------------
# Synthetic fit: refit, reload check, persistence, then the whole test-side path on synthetic data
# ---------------------------------------------------------------------------

def refit(campaign_tuple, tmp_path, selection_sha256="ab" * 32):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign_tuple
    settle(paths)
    result = vm.run_refit(paths, epochs=2, selection_sha256=selection_sha256, lane=0, train_dir=train, esm_dir=esm_dir,
                          python_bin=sys.executable, device="cpu", execution_policy=policy(tmp_path), load_workers=1)
    return result, paths.lane(0) / "runs" / REFIT


@SYNTHETIC_FIT
def test_the_refit_runs_validation_free_reload_checks_persists_and_never_reruns_synthetic_fit(campaign, tmp_path,  # noqa: F811
                                                                                              monkeypatch):
    train, esm_dir, paths, _, _ = campaign
    records = f.record_paths(tmp_path)  # the workstation records next to the durable copy (tmp_path/durable)
    records["selection"].parent.mkdir(parents=True)
    records["selection"].write_text(json.dumps({"kind": "step_f_selection", "synthetic": True}))
    records["manifest"].write_text(json.dumps({"vm_files": {f.VM_RUNNER: {"sha256": sha(ROOT / f.VM_RUNNER)}},
                                               "tooling_sha256": {n: sha(ROOT / n) for n in f.STEP_F_TOOL_FILES}}))
    result, run_dir = refit(campaign, tmp_path, selection_sha256=sha(records["selection"]))
    assert result["status"] == "completed", result
    assert result["selected_epoch"] == 2 and result["selected_checkpoint"] == "terminal_model_checkpoint.pt"
    check = result["refit_receipt"]["reload_check"]
    assert check["passed"] and check["abs_difference"] <= 0.001 and check["n_ions"] == 60
    assert not (run_dir / "val_predictions.csv").exists() and not (run_dir / "selected_checkpoint.json").exists()
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["test_report"] is None and metadata["config"]["val_fraction"] == 0.0
    weights = vm.full_cohort_weights(paths)["common_four_weights"]
    applied = metadata["metal_class_weights"]
    assert {k: applied[k] for k in ("Mn", "Cu", "Zn")} == pytest.approx({k: weights[k] for k in ("Mn", "Cu", "Zn")})
    status = v3.completed_units(paths)[REFIT]
    assert status["status"] == "completed" and status["step"] == "F2" and status["identity"]["validation"] == "none"
    durable = tmp_path / "durable" / "lane0"
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        copy = durable / "runs" / REFIT / path.relative_to(run_dir)
        assert copy.is_file() and copy.read_bytes() == path.read_bytes(), copy
    assert (durable / f"run_status_{REFIT}.json").read_bytes() == (paths.lane(0) / f"run_status_{REFIT}.json").read_bytes()
    again = vm.reload_check(run_dir, paths, status["identity"], epochs=2, train_dir=train, esm_dir=esm_dir,
                            external_root=paths.empty_external_features, device="cpu")
    assert again["reload_check"]["reevaluated_train_common4_ba"] == check["reevaluated_train_common4_ba"]
    with pytest.raises(ValueError, match="identity differs"):
        vm.verify_refit(run_dir, {**status["identity"], "epochs": 3}, epochs=2)
    # The workstation checks of the pulled refit and Freeze 1 (the bound refit is the 50-epoch one; this fit has 2).
    with pytest.raises(f.LauncherError, match="not the 50-epoch terminal validation-free fit"):
        f.refit_receipt(tmp_path)
    spec = f.load_spec()
    spec["refit"]["epochs"] = 2
    monkeypatch.setattr(f, "load_spec", lambda path=None: spec)
    receipt = f.refit_receipt(tmp_path)
    assert receipt["checkpoint_sha256"] == result["selected_checkpoint_sha256"] and receipt["reload_check"]["passed"]
    freeze1 = f.freeze1_record(tmp_path, log_entry="v3-101")
    assert freeze1["checkpoint_sha256"] == receipt["checkpoint_sha256"] and freeze1["cohort_sha256"] == \
        status["identity"]["cohort_sha256"] and freeze1["selection_record_sha256"] == sha(records["selection"])
    assert freeze1["user_go_ahead_log_entry"] == "v3-101" and "v3-101" in f.ledger_text(freeze1)
    assert freeze1["refit"]["runner_sha256"] == sha(ROOT / f.VM_RUNNER) and freeze1["tools_differ_from_refit_manifest"] == []
    (durable / "runs" / REFIT / "run_config.json").write_text("{}")
    with pytest.raises(f.LauncherError, match="changed after the refit"):
        f.refit_receipt(tmp_path)
    with pytest.raises(ValueError, match="never retried"):
        vm.run_refit(paths, epochs=2, selection_sha256="ab" * 32, lane=1, train_dir=train, esm_dir=esm_dir,
                     python_bin=sys.executable, device="cpu", execution_policy=policy(tmp_path), load_workers=1)


def synthetic_test_side(tmp_path, name, checkpoint, *, swap=None):
    """A synthetic test side under its own campaign root: recorded Freeze 1, label-blind preparation and injected
    embeddings made under it, recorded Freeze 2."""
    from benchmarking import pmm_ion_features as features

    assert features.ESM_DIM == 8  # the campaign fixture's ESMC width
    campaign = tmp_path / name / "campaign"
    freeze1_sha = record_freeze1(campaign, sha(checkpoint))
    test, source = build_test_side(tmp_path / name / "dataset", swap=swap)
    out = tmp_path / name / "prepared"
    prep.prepare(test, source, out, expected_source_sha256=sha(source), training_pdbids={"1aba"},
                 training_sequences=lambda: [], previously_opened={"7uuu"}, freeze1_sha256=freeze1_sha)
    with (out / "test_esm_plan.csv").open(encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    esm = tmp_path / name / "esm"
    receipt = prep.generate_embeddings(plan, test, esm, embed=fake_embedder(8), freeze1_sha256=freeze1_sha)
    record_freeze2(campaign, freeze1_sha, out, checkpoint_sha=sha(checkpoint), inventory_sha=receipt["inventory_sha256"],
                   source_sha=sha(source), rows=19)
    return test, source, out, esm, campaign


@SYNTHETIC_FIT
def test_label_free_prediction_covers_every_row_and_the_report_reads_labels_last_synthetic_fit(campaign, tmp_path):  # noqa: F811
    import torch

    result, run_dir = refit(campaign, tmp_path)
    assert result["status"] == "completed", result
    checkpoint = run_dir / "terminal_model_checkpoint.pt"
    test, source, prepared_dir, esm, campaign_t = synthetic_test_side(tmp_path, "t", checkpoint)
    summary = json.loads((prepared_dir / "preparation_summary.json").read_text())
    assert summary["clean_subset"]["exclusion_counts"]["C1_training_pdb_id"] == 2
    assert summary["clean_subset"]["exclusion_counts"]["C3_previously_opened_pdb_id"] == 2
    kwargs = dict(campaign_dir=campaign_t, prepared_dir=prepared_dir, checkpoint_path=checkpoint, test_dir=test,
                  esm_dir=esm, external_root=None, device="cpu", source_csv=source, log_entry="v3-903")
    receipt = vm.predict(out_dir=tmp_path / "t" / "predictions", **kwargs)
    assert receipt["rows"] == 19 and receipt["scored_rows"] == 9 and receipt["physical_ions_predicted"] == 8
    rows = list(csv.DictReader((tmp_path / "t" / "predictions" / "test_predictions.csv").open()))
    assert [int(r["source_row"]) for r in rows] == list(ROWS)  # every source row, in order
    by_row = {int(r["source_row"]): r for r in rows}
    for number, (*_rest, disposition, _reason, alias) in ROWS.items():
        assert (by_row[number]["pred_common4"] != "") is (disposition != "unscorable"), number
    assert {k: v for k, v in by_row[4].items() if k.startswith("p_")} == {
        k: v for k, v in by_row[3].items() if k.startswith("p_")}  # an alias carries its primary's prediction
    for r in rows:
        if r["pred_common4"]:
            probabilities = [float(r[f"p_common4_{label.replace(' ', '_')}"]) for label in vm.COMMON4]
            assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
            assert int(r["pred_common4"]) == max(range(4), key=probabilities.__getitem__)
    # The label-bearing files were unreadable during (and after) the pass.
    with pytest.raises(PermissionError):
        (test / "site_manifest.csv").read_text()
    with pytest.raises(PermissionError):
        source.read_text()
    # One shot per campaign: a second pass is refused into the same or any other directory; another checkpoint too.
    receipt_t = json.loads((tmp_path / "t" / "predictions" / "prediction_receipt.json").read_text())
    claim = json.loads(prep.evidence_paths(campaign_t)["prediction"].read_text())
    assert claim["out_dir"] == str((tmp_path / "t" / "predictions").resolve()) and claim["device"] == "cpu"
    assert claim["user_go_ahead_log_entry"] == "v3-903"
    assert receipt_t["prediction_claim_sha256"] == sha(prep.evidence_paths(campaign_t)["prediction"])
    for out in ("predictions", "p2"):
        with pytest.raises(ValueError, match="already claimed"):
            vm.predict(out_dir=tmp_path / "t" / out, **kwargs)
    assert not (tmp_path / "t" / "p2").exists()
    other = tmp_path / "other.pt"
    torch.save(torch.load(checkpoint, weights_only=False) | {"note": 1}, other)
    with pytest.raises(ValueError, match="not the bound refit"):
        vm.predict(out_dir=tmp_path / "t" / "p3", **{**kwargs, "checkpoint_path": other})
    # The ion's metal never enters the input: swapping every supported metal leaves every probability unchanged.
    test_b, source_b, prepared_b, esm_b, campaign_s = synthetic_test_side(tmp_path, "s", checkpoint, swap=SWAP)
    vm.predict(out_dir=tmp_path / "s" / "predictions", campaign_dir=campaign_s, prepared_dir=prepared_b,
               checkpoint_path=checkpoint, test_dir=test_b, esm_dir=esm_b, external_root=None, device="cpu",
               source_csv=source_b, log_entry="v3-903")
    swapped = list(csv.DictReader((tmp_path / "s" / "predictions" / "test_predictions.csv").open()))
    assert [{k: v for k, v in r.items() if k.startswith("p_") or k == "pred_common4"} for r in swapped] == \
        [{k: v for k, v in r.items() if k.startswith("p_") or k == "pred_common4"} for r in rows]
    # The report, the only label reader, runs on a fresh copy of the label file (the guard covers the original path).
    labels = tmp_path / "labels_copy.csv"
    labels.write_bytes(subprocess.run(["cat", str(source)], capture_output=True, check=True).stdout)
    out = report.build_report(campaign_dir=campaign_t, predictions_dir=tmp_path / "t" / "predictions",
                              prepared_dir=prepared_dir, source_csv=labels)
    assert out["source_rows"] == 19 and out["scored_statement"] == "9 of 19 scored"
    assert out["rows_with_other_label_codes"]["count"] == 1  # the calcium row's code
    assert out["lines"]["full_test"]["rows"] == 18 and out["lines"]["full_test"]["unscorable_counted_as_errors"] == 9
