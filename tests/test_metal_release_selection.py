"""Synthetic checks for nested OOF ensemble re-selection and the release CLI; no repository data."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from metal_release.__main__ import main
from metal_release.audit import COMMON4
from metal_release.inventory import BASELINE_PROBE
from metal_release.manifest import aggregate_probabilities, candidate_member_ids, load_manifest, sha256_file
from metal_release.selection import (collect_families, family_identity, fixed_recipe, nested_selection,
                                     select_late_fusion, select_trimodal)

LABELS = np.repeat(np.arange(4), 5)


def scored(wrong: set[int], wrong_probability: float = 0.55, labels: np.ndarray = LABELS) -> np.ndarray:
    """Two-class mass per row: the true class, or the next class with `wrong_probability` on listed rows."""
    probabilities = np.zeros((len(labels), 4))
    for row, label in enumerate(labels):
        mistake = wrong_probability if row in wrong else 0.1
        probabilities[row, label] = 1 - mistake
        probabilities[row, (label + 1) % 4] = mistake
    return probabilities


def test_family_identity_aliases_and_rejects_refits():
    assert family_identity("metal_completion__confirmation__P__five_class__fold2__seed43") == ("mc_P_five_class", 2, 43)
    assert family_identity("metal_completion__confirmation__B__four_class__fold1__seed44") == (
        "gvp_late_fusion__four_class__baseline", 1, 44)
    assert family_identity(BASELINE_PROBE) == ("gvp_late_fusion__four_class__baseline", 0, 42)
    assert family_identity("only_gvp__four_class__baseline__fold3__seed42") == ("only_gvp__four_class__baseline", 3, 42)
    assert family_identity("stage6b_refit__gvp_late_fusion__four_class__baseline__e50__seed42") is None


def test_late_fusion_ranks_by_selection_ba_and_keeps_best_top_n():
    families = {"a": scored({0, 5}), "b": scored({10, 15, 16}), "c": scored({0, 1, 2, 3, 4}, 0.95)}
    choice = select_late_fusion(families, LABELS, np.ones(len(LABELS), dtype=bool))
    assert [item["family"] for item in choice["ranking"]] == ["a", "b", "c"]
    assert [item["selection_ba"] for item in choice["ranking"]] == pytest.approx([0.9, 0.85, 0.75])
    # a+b repair each other's low-confidence errors; adding c breaks row 0 again.
    assert choice["top_n"] == 2 and choice["families"] == ["a", "b"] and choice["selection_ba"] == 1.0


def test_late_fusion_ties_keep_smaller_n_and_respect_max_n():
    families = {"a": scored(set()), "b": scored(set())}
    choice = select_late_fusion(families, LABELS, np.ones(len(LABELS), dtype=bool), max_n=1)
    assert choice["top_n"] == 1 and choice["families"] == ["a"] and len(choice["ranking"]) == 1
    assert select_late_fusion(families, LABELS, np.ones(len(LABELS), dtype=bool))["top_n"] == 1


def test_trimodal_grid_picks_smallest_sufficient_weight_and_skips_harmful_branch():
    late = scored({0}, 0.6)
    perfect = np.eye(4)[LABELS]
    harmful = np.eye(4)[(LABELS + 1) % 4]
    choice = select_trimodal(late, {"esm_a": perfect, "esm_b": perfect}, {"gvp": harmful}, LABELS,
                             np.ones(len(LABELS), dtype=bool))
    assert choice == {"esm_families": ["esm_a"], "esm_weight": 0.25, "gvp_families": [], "gvp_weight": 0.0,
                      "selection_ba": 1.0}
    unchanged = select_trimodal(scored(set()), {"esm": harmful}, {"gvp": harmful}, LABELS,
                                np.ones(len(LABELS), dtype=bool))
    assert unchanged["esm_families"] == [] and unchanged["gvp_families"] == []


def synthetic_pool(seed: int):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.repeat(np.arange(4), 6), 5)
    folds = np.repeat(np.arange(5), 24)
    families = {}
    for name, branch, signal in (("lf_a", "late_fusion", 1.5), ("lf_b", "late_fusion", 1.0),
                                 ("lf_c", "late_fusion", 0.5), ("esm", "only_esm", 1.0), ("gvp", "only_gvp", 0.8)):
        logits = rng.normal(size=(len(labels), 4)) + signal * np.eye(4)[labels]
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        families[name] = {"branch": branch, "probs": probabilities}
    return labels, folds, families


def test_outer_fold_labels_and_predictions_never_influence_its_own_choice():
    labels, folds, families = synthetic_pool(3)
    baseline = nested_selection(families, labels, folds, max_n=3)
    changed_labels = labels.copy()
    held_out = folds == 2
    changed_labels[held_out] = np.random.default_rng(9).permutation(labels[held_out])
    changed = {name: dict(item, probs=item["probs"].copy()) for name, item in families.items()}
    for item in changed.values():
        item["probs"][held_out] = np.random.default_rng(11).dirichlet(np.ones(4), size=int(held_out.sum()))
    perturbed = nested_selection(changed, changed_labels, folds, max_n=3)

    def choice(result):
        fold = result["per_fold"][2]
        return ({k: v for k, v in fold["late_fusion"].items() if k != "outer_fold_ba"},
                {k: v for k, v in fold["trimodal"].items() if k != "outer_fold_ba"})

    assert choice(baseline) == choice(perturbed)
    outer = [result["per_fold"][2]["late_fusion"]["outer_fold_ba"] for result in (baseline, perturbed)]
    assert outer[0] != outer[1]
    chosen = baseline["per_fold"][2]["late_fusion"]["families"]
    expected = np.mean([families[name]["probs"] for name in chosen], axis=0)[held_out]
    np.testing.assert_array_equal(baseline["predictions"]["late_fusion"][held_out], expected)
    assert np.isfinite(baseline["predictions"]["trimodal"]).all()


def write_run(directory, family, fold, seed, rows, probabilities, membership_sha):
    directory.mkdir(parents=True)
    prediction = directory / "val_predictions.csv"
    with prediction.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_uid", "fold", "y_common4", *(f"p_common4_{n.replace(' ', '_')}" for n in COMMON4)])
        for row, values in zip(rows, probabilities):
            writer.writerow([row["source_uid"], fold, row["label"], *values])
    receipt = {"checkpoint_rule": "terminal", "fit_status": "completed", "reconciliation_status": "match",
               "validation_predictions": {"path": prediction.name, "sha256": sha256_file(prediction)},
               "campaign_run_identity": {"family": family, "fold": fold, "model_seed": seed,
                                         "fold_membership_sha256": membership_sha}}
    (directory / "selected_checkpoint.json").write_text(json.dumps(receipt), encoding="utf-8")
    return {"validation_predictions": {"path": str(prediction), "sha256": sha256_file(prediction)},
            "selected_checkpoint": {"path": str(directory / "selected_checkpoint.json"),
                                    "sha256": sha256_file(directory / "selected_checkpoint.json")}}


@pytest.fixture
def development_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("nested_case")
    elements = ("MN", "CU", "ZN", "FE")
    rows = [{"source_uid": f"u{fold}-{label}-{rep}", "physical_ion_id": f"ion{fold}-{label}-{rep}",
             "group_id": f"group{fold}-{label}", "pdbid": f"pdb{fold}{label}", "native_element": elements[label],
             "fold": fold, "label": label} for fold in range(5) for label in range(4) for rep in range(3)]
    membership = root / "fold_membership.csv"
    with membership.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k in rows[0] if k != "label"])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "label"} for row in rows)
    membership_sha = sha256_file(membership)
    rng = np.random.default_rng(5)
    runs = root / "runs"
    members = {}

    def run(name, family, fold, seed, lane="lane0", signal=1.0):
        fold_rows = sorted((r for r in rows if r["fold"] == fold), key=lambda r: r["source_uid"])
        labels = np.array([r["label"] for r in fold_rows])
        logits = rng.normal(size=(len(fold_rows), 4)) + signal * np.eye(4)[labels]
        probabilities = np.round(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True), 8)
        probabilities[:, 3] = 1 - probabilities[:, :3].sum(axis=1)
        return write_run(runs / lane / f"{name}__fold{fold}__seed{seed}", family, fold, seed, fold_rows,
                         probabilities, membership_sha)

    for fold in range(5):
        for name, family, target in (("gvp_late_fusion__four_class__alpha", "gvp_late_fusion", "four_class"),
                                     ("only_esm__four_class__baseline", "only_esm", "four_class")):
            members[f"{name}__fold{fold}__seed42"] = {"target": target, "fold": fold, "seed": 42,
                                                      **run(name, family, fold, 42, lane="bound")}
        run("gvp_late_fusion__four_class__beta", "gvp_late_fusion", fold, 42)
        run("only_gvp__four_class__baseline", "only_gvp", fold, 42, signal=0.5)
        run("only_gvp__four_class__duplicate", "only_gvp", fold, 42)
        for seed in (43, 44):
            run("metal_completion__confirmation__P__four_class", "gvp_late_fusion", fold, seed)
    run("only_gvp__four_class__duplicate", "only_gvp", 1, 42, lane="lane1")
    run("gvp_late_fusion__four_class__screen", "gvp_late_fusion", 0, 42)
    alpha = [m for m in members if m.startswith("gvp_late_fusion")]
    esm = [m for m in members if m.startswith("only_esm")]
    manifest = {"schema": "deepmzyme-metal-release-v1", "reference": "ref", "n_folds": 5,
                "fold_membership": {"path": str(membership), "sha256": membership_sha}, "members": members,
                "candidates": {"lf": {"groups": [{"id": "alpha", "weight": 1.0, "members": alpha}]},
                               "ref": {"groups": [{"id": "alpha", "weight": 1.0, "members": alpha},
                                                  {"id": "esm", "weight": 0.25, "members": esm}],
                                       "blocks": [{"groups": ["alpha"], "weight": 1.0},
                                                  {"groups": ["esm"], "weight": 0.25}]}}}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, runs


def test_cli_nested_selection_reports_pool_exclusions_and_fold_choices(development_runs, tmp_path_factory, capsys):
    manifest, runs = development_runs
    output = tmp_path_factory.mktemp("nested_output") / "nested_selection.json"
    code = main(["nested-selection", "--manifest", str(manifest), "--run-root", str(runs),
                 "--late-fusion-candidate", "lf", "--expect-family", "gvp_late_fusion__four_class__beta",
                 "--expect-family", "gvp_late_fusion__four_class__missing", "--output", str(output),
                 "--bootstrap-samples", "50"])
    assert code == 0, capsys.readouterr().err
    report = json.loads(output.read_text())
    families = {item["family"]: item for item in report["families"]}
    assert set(families) == {"gvp_late_fusion__four_class__alpha", "gvp_late_fusion__four_class__beta",
                             "mc_P_four_class", "only_esm__four_class__baseline", "only_gvp__four_class__baseline"}
    assert families["mc_P_four_class"]["seeds"] == [43, 44]
    assert families["only_esm__four_class__baseline"]["branch"] == "only_esm"
    assert "conflicting copies" in report["excluded_families"]["only_gvp__four_class__duplicate"]
    assert "folds [0] only" in report["excluded_families"]["gvp_late_fusion__four_class__screen"]
    assert report["expected_families"] == {"gvp_late_fusion__four_class__beta": "included",
                                           "gvp_late_fusion__four_class__missing": "no run directory found"}
    assert [item["outer_fold"] for item in report["per_fold"]] == [0, 1, 2, 3, 4]
    assert report["summary"]["late_fusion"]["fixed_candidate"] == "lf"
    assert report["summary"]["trimodal"]["fixed_candidate"] == "ref"
    assert str(runs) not in output.read_text()
    assert main(["nested-selection", "--manifest", str(manifest), "--output", str(output)]) == 2


def test_unknown_late_fusion_candidate_is_a_readable_error(development_runs, tmp_path_factory, capsys):
    manifest, runs = development_runs
    output = tmp_path_factory.mktemp("nested_unknown") / "nested_selection.json"
    code = main(["nested-selection", "--manifest", str(manifest), "--run-root", str(runs),
                 "--late-fusion-candidate", "missing", "--output", str(output)])
    assert code == 2
    assert "Unknown candidate: missing" in capsys.readouterr().err
    assert not output.exists()


def test_fixed_recipe_matches_manifest_fold_aggregation(development_runs):
    manifest_path, runs = development_runs
    manifest = load_manifest(manifest_path)
    rows, families, _ = collect_families(manifest, [runs])
    folds = np.array([int(row["fold"]) for row in rows])
    for candidate in ("lf", "ref"):
        expected = np.empty((len(rows), 4))
        for fold in range(5):
            mask = folds == fold
            probabilities = {member: families[family_identity(member)[0]]["seeds"][42][mask]
                             for member in candidate_member_ids(manifest, candidate)}
            expected[mask] = aggregate_probabilities(manifest, candidate, probabilities, fold=fold)
        np.testing.assert_allclose(fixed_recipe(manifest, candidate, families), expected, rtol=0, atol=1e-12)


def test_catalog_command_exits_zero_and_never_overwrites(tmp_path_factory, capsys):
    output = tmp_path_factory.mktemp("catalog") / "catalog.json"
    assert main(["catalog", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["release_status"] == "blocked"
    assert main(["catalog", "--output", str(output)]) == 2
    assert "File exists" in capsys.readouterr().err
