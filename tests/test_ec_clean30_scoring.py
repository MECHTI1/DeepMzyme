"""Frozen EC class versus CLEAN scorer: claim, score and the pre-declared analysis (synthetic inputs only)."""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from typing import Callable, Optional, Sequence

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import run_clean_split30_comparator as runner  # noqa: E402
from ec_clean30 import scoring, stats  # noqa: E402
from ec_clean30.identity import COMPONENT_COLUMNS, IDENTITY_BIN_LABELS, STRATA_COLUMNS  # noqa: E402
from ec_clean30.scoring import (  # noqa: E402
    DECISION_NOT_SHOWN, DECISION_PASS, Prediction, ScoredRow, ScoringError, ScoringTable,
)

SCRIPT = ROOT / "scripts" / "score_ec_clean30_comparison.py"
FREEZE1 = "1" * 64
ALL_FOLDS = (0, 1, 2, 3, 4)
DM_SEEDS = (42, 43, 44, 45, 46)
R1_SEEDS = (42, 43, 44, 45, 46)
THREE_SEEDS = (42, 43, 44)
MODEL = {"model_architecture": "gvp", "fusion_mode": "late_fusion", "use_global_esm_branch": True,
         "global_esm_layer": "final", "ec_aux_levels": "", "ec_supcon_weight": 0.0, "campaign_family": "F4"}
RECIPE = {"family": "F4", "epochs": 30, "lever_configs": []}
EC_LABELS = ("1.1.1.1", "1.9.9.9", "2.1.1.1", "3.1.1.1", "4.1.1.1", "5.1.1.1", "6.1.1.1", "7.1.1.1")
ESMC_TAG = "esmc300m_final_mean"
ACKNOWLEDGEMENT = "the Example Research Laboratory"
TEST_DRAWS = 400
# Report wording rules: never "EC1" for the first-digit task, never "equivalent" for a non-significant result.
FORBIDDEN_REPORT_WORDS = ("EC1", "equivalent")
ENV = {**os.environ, "MKL_THREADING_LAYER": "GNU"}
# A subprocess applies the same in-process test switches as the fixture below.
SUBPROCESS_SETUP = (
    f"import sys; sys.path[:0] = [{str(ROOT / 'tests')!r}, {str(ROOT / 'scripts')!r}, {str(ROOT / 'src')!r}]; "
    f"import ec_clean30.scoring as _scoring; _scoring._TEST_SETTING_OVERRIDES.update(n_draws={TEST_DRAWS}); "
    "_scoring._TEST_ALLOW_FOLD_SUBSET = True"
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(autouse=True)
def synthetic_test_switches(monkeypatch):
    """Fewer bootstrap draws and two-fold campaigns, through the in-process switches only."""
    monkeypatch.setattr(scoring, "_TEST_SETTING_OVERRIDES", {"n_draws": TEST_DRAWS})
    monkeypatch.setattr(scoring, "_TEST_ALLOW_FOLD_SUBSET", True)


@pytest.fixture
def space(tmp_path_factory):
    return tmp_path_factory.mktemp("scoring")


# ---------------------------------------------------------------- synthetic campaign

@dataclass(frozen=True)
class Protein:
    fold: int
    accession: str
    token: str  # true EC class token, or "amb" for a protein with two EC classes
    kind: str  # "nh30", "high" (D3 hit) or "donor" (no D3 hit, shared donor)
    index: int

    @property
    def ec_numbers(self) -> str:
        if self.token == "amb":
            return "1.1.1.1;3.4.21.4"
        return f"{self.token}.1.1.{self.index + 1}"


def default_proteins(folds=(0, 1), per_class=12) -> list[Protein]:
    proteins = []
    for fold in folds:
        for token in "123456":
            for i in range(per_class):
                proteins.append(Protein(fold, f"F{fold}C{token}N{i:02d}", token, ("nh30", "high", "donor")[i % 3], i))
        proteins.append(Protein(fold, f"F{fold}C7N00", "7", "nh30", 0))
        proteins.append(Protein(fold, f"F{fold}AMB", "amb", "nh30", 0))
    return proteins


def wrong(token: str) -> str:
    return "1" if token in ("7", "amb") else str(int(token) % 6 + 1)


def mixed(protein: Protein) -> str:
    if protein.token == "amb":
        return "1"
    return protein.token if protein.index % 2 == 0 else wrong(protein.token)


def baseline_outcome(protein: Protein) -> str:
    if protein.token == "amb":
        return "1"
    return protein.token if protein.index % 3 == 0 else "no_hit"


def better_plan(system: str, seed: Optional[int], protein: Protein):
    if system == "baseline":
        return baseline_outcome(protein)
    if system == "dm":
        return "1" if protein.token == "amb" else protein.token
    return mixed(protein)


def identical_plan(system: str, seed: Optional[int], protein: Protein):
    return baseline_outcome(protein) if system == "baseline" else mixed(protein)


def dm_logits(outcome, seed: int) -> list[float]:
    if isinstance(outcome, (list, tuple)):
        return [float(value) for value in outcome]
    values = [-2.0 - 0.01 * j - 0.001 * (seed - 40) for j in range(7)]
    values[int(outcome) - 1] = 3.0 + 0.001 * seed
    return values


def clean_distances(outcome: str) -> list[float]:
    distances = [2.0 + 0.01 * j for j in range(len(EC_LABELS))]
    if outcome.startswith("tie:"):
        for token in outcome[4:].split(","):
            distances[EC_LABELS.index(f"{token}.1.1.1")] = 1.0
    elif outcome == "sametie:1":
        distances[EC_LABELS.index("1.1.1.1")] = distances[EC_LABELS.index("1.9.9.9")] = 1.0
    else:
        distances[EC_LABELS.index(f"{outcome}.1.1.1")] = 1.0
    return distances


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict], delimiter: str = ",") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def dm_checkpoint(fold: int, seed: int) -> str:
    return sha(f"checkpoint-{fold}-{seed}")


def clean_checkpoint(regime: str, embedding: str, fold: int, seed: int) -> str:
    return sha(f"{regime}-{embedding}-{fold}-{seed}")


def write_freeze2(root: Path, checkpoints: Sequence[str], edit: Optional[Callable[[dict], None]] = None,
                  ) -> scoring.DeclaredFile:
    """A Freeze-2 manifest in the test-pass format, pinning the live scorer code."""
    code = scoring.code_sha256()
    payload = {"schema_version": 1, "kind": "freeze2", "smoke": False, "checkpoint_sha256s": sorted(set(checkpoints)),
               "code_sha256": {path: digest for path, digest in code.items() if path.startswith("src/")},
               "additional_code_sha256": {path: digest for path, digest in code.items() if not path.startswith("src/")},
               "test_pass_output_root": str(root / "test_passes")}
    if edit is not None:
        edit(payload)
    path = write_json(root / "freeze2" / "freeze2_manifest.json", payload)
    return scoring.DeclaredFile(path, runner.sha256_file(path))


def freeze2_for(root: Path, checkpoints: Sequence[str]) -> scoring.Freeze2:
    return scoring.read_freeze2_manifest(write_freeze2(root, checkpoints))


def strata_row(protein: Protein, cluster: str) -> dict:
    flags = dict.fromkeys(("has_any_hit", "has_bin_hit", "has_d3", "has_d2", "has_d2_min_pairs", "has_d1",
                           "donor_shared_with_metallo_train"), "0")
    if protein.kind == "high":
        identity_bin = "[30,50]" if protein.index % 6 == 1 else ">50"
        flags.update(has_any_hit="1", has_bin_hit="1", has_d3="1", has_d2="1", has_d2_min_pairs="1")
    elif protein.kind == "donor":
        identity_bin = "(25,30)"
        flags.update(has_any_hit="1", has_bin_hit="1", donor_shared_with_metallo_train="1",
                     has_d2="1" if protein.index % 6 == 2 else "0")
    elif protein.index % 6 == 3:
        identity_bin = "(20,25]"
        flags.update(has_any_hit="1", has_bin_hit="1")
    else:
        identity_bin = "no_hit"
    d3 = flags["has_d3"] == "1"
    donor = flags["donor_shared_with_metallo_train"] == "1"
    return {"accession": protein.accession, "identity_bin": identity_bin,
            "max_bin_identity": "" if identity_bin == "no_hit" else "0.400000",
            "max_bin_target": "" if identity_bin == "no_hit" else "TRAIN1", **flags,
            "no_d3_hit": "0" if d3 else "1", "no_d3_hit_no_shared_donor": "1" if not d3 and not donor else "0",
            "nearest_target": "", "nearest_bits": "", "nearest_evalue": "", "cluster_id": cluster}


def cluster_ids(proteins: Sequence[Protein]) -> dict[str, str]:
    clusters = {protein.accession: f"K_{protein.accession}" for protein in proteins}
    for fold in {protein.fold for protein in proteins}:
        for accession in (f"F{fold}C1N00", f"F{fold}C1N03"):  # a within-fold pair
            if accession in clusters:
                clusters[accession] = f"K_F{fold}C1_pair"
    for accession in ("F0C2N00", "F1C2N00"):  # a cluster across folds
        if accession in clusters:
            clusters[accession] = "K_C2_cross"
    return clusters


@dataclass
class Campaign:
    root: Path
    manifest: Path
    label_table: Path
    report_dir: Path
    proteins: list
    raw: dict

    @property
    def claim_path(self) -> Path:
        return self.report_dir / scoring.CLAIM_NAME

    @property
    def scoring_ledger(self) -> Path:
        return self.root / "test_passes" / scoring.SCORING_LEDGER_NAME

    def rewrite_manifest(self, **changes) -> None:
        self.raw.update(changes)
        self.manifest.write_text(json.dumps(self.raw, indent=2, sort_keys=True), encoding="utf-8")

    def manifest_copy(self, name: str, **changes) -> Path:
        """Another manifest file with ``changes`` (the original stays as claimed)."""
        return write_json(self.root / name, {**self.raw, **changes})


def write_dm_pass(folder: Path, fold: int, seed: int, proteins: Sequence[Protein], plan: Callable,
                  freeze2: scoring.DeclaredFile, failures: Sequence[str] = (), absent: Sequence[str] = ()) -> str:
    pass_id = folder.name
    rows, inputs = [], []
    failures = sorted(set(failures) & {protein.accession for protein in proteins})
    for protein in sorted(proteins, key=lambda item: item.accession):
        if protein.accession in absent:
            continue
        inputs.append({"accession": protein.accession, "sha256": sha(protein.accession)})
        if protein.accession in failures:
            continue
        logits = dm_logits(plan("dm", seed, protein), seed)
        rows.append({"pass_id": pass_id, "accession": protein.accession, "n_pockets": "1",
                     "pred": str(int(np.argmax(logits))),
                     **{column: repr(value) for column, value in zip(scoring.C3_LOGIT_COLUMNS, logits)}})
    csv_path = write_csv(folder / scoring.C3_TEST_PROTEIN_PREDICTIONS_NAME, scoring.C3_TEST_PROTEIN_PREDICTION_COLUMNS,
                         rows)
    checkpoint = dm_checkpoint(fold, seed)
    output_root = json.loads(freeze2.path.read_text())["test_pass_output_root"]
    write_json(folder / scoring.C3_TEST_RECEIPT_NAME, {
        "schema_version": 1, "pass_id": pass_id, "label_free": True, "labels_read": False, "metrics_computed": False,
        "placeholder_suffix": "__EC_0.0.0.0",
        "freeze2_manifest": {"path": str(freeze2.path), "sha256": freeze2.sha256, "test_pass_output_root": output_root},
        "checkpoint_sha256": checkpoint, "model": MODEL, "seed": seed, "ec_class_tokens": list("1234567"),
        "aggregation": "mean_pocket_logits", "input_structures": inputs, "n_structures": len(inputs),
        "n_proteins_predicted": len(rows),
        "failures": [{"accession": accession, "reason": "no_selected_sites", "detail": ""} for accession in failures],
        "protein_predictions": {"path": csv_path.name, "sha256": runner.sha256_file(csv_path), "n_rows": len(rows)},
    })
    return checkpoint


def write_refit_receipt(path: Path, seed: int, checkpoint: str, counts: Optional[dict] = None, *, fold: int = 0,
                        identity: Optional[dict] = None) -> Path:
    unit_id = f"ecc30_f{fold}_S4_selected_refit_s{seed}"
    run_identity = {"campaign_id": scoring.CAMPAIGN_ID, "unit_id": unit_id, "step": "S4", "kind": "refit",
                    "fold": fold, "inner_fold": None, "seed": seed, "smoke": False, "recipe": RECIPE,
                    "freeze1_sha256": FREEZE1, **(identity or {})}
    return write_json(path, {
        "schema_version": 1, "mode": "probe", "run_id": unit_id, "campaign_run_identity": run_identity,
        "seed": seed, "checkpoint": {"sha256": checkpoint, "epoch": 30}, "model": MODEL,
        "ec_class_tokens": list("1234567"),
        "training_protein_class_counts": counts or {token: 50 for token in "1234567"},
    })


def clean_binding(freeze2: scoring.DeclaredFile) -> dict:
    """The test-export binding a CLEAN export of this Freeze 2 carries (contract C4 version 2)."""
    return {"freeze2_manifest_sha256": freeze2.sha256,
            "test_pass_output_root": json.loads(freeze2.path.read_text())["test_pass_output_root"],
            "runner_sha256": runner.sha256_file(scoring.COMPARATOR_PATH),
            "export_script_sha256": runner.EXPORT_SCRIPT_SHA256}


def write_clean_export(path: Path, *, fold: int, regime: str, seed: int, embedding: str, input_dim: int,
                       proteins: Sequence[Protein], outcome: Callable, binding: Optional[dict] = None) -> Path:
    ordered = sorted(proteins, key=lambda item: item.accession)
    distances = np.array([clean_distances(outcome(protein)) for protein in ordered], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_distance_export(path, distances, [protein.accession for protein in ordered], list(EC_LABELS), {
        "fold": fold, "regime": regime, "seed": seed, "input_dim": input_dim, "embedding": embedding, "mode": "test",
        "checkpoint_sha256": clean_checkpoint(regime, embedding, fold, seed),
        "training_table_sha256": sha(f"table-{regime}-{fold}"),
        "embedding_farm_sha256": sha(f"farm-{embedding}-{fold}"),
        "query_embeddings_sha256": sha(f"queries-{embedding}-{fold}"), **(binding or {})})
    return path


CLEAN_ARM_SPECS = (("r1", "r1", R1_SEEDS, runner.DEFAULT_EMBEDDING, 1280),
                   ("r2", "r2", THREE_SEEDS, runner.DEFAULT_EMBEDDING, 1280),
                   ("c_esmc", "r1", THREE_SEEDS, ESMC_TAG, 960))


def build_campaign(root: Path, plan: Callable = better_plan, *, folds=(0, 1), per_class=12, proteins=None,
                   dm_failures=(), dm_absent=(), clean_absent=(), with_b3=False,
                   transfers: Sequence[tuple[str, str]] = (),
                   tau: float = 0.0, clean_plan: Optional[Callable] = None,
                   freeze2_edit: Optional[Callable[[dict], None]] = None) -> Campaign:
    proteins = proteins or default_proteins(folds, per_class)
    by_fold = {fold: [protein for protein in proteins if protein.fold == fold] for fold in folds}
    clusters = cluster_ids(proteins)
    clean_plan = clean_plan or (lambda arm, seed, protein: plan(arm, seed, protein))

    checkpoints = [dm_checkpoint(fold, seed) for fold in folds for seed in DM_SEEDS]
    checkpoints += [clean_checkpoint(regime, embedding, fold, seed) for fold in folds
                    for _, regime, seeds, embedding, _ in CLEAN_ARM_SPECS for seed in seeds]
    freeze2 = write_freeze2(root, checkpoints, freeze2_edit)

    membership = write_csv(root / "phase7" / "test_membership.csv", ("fold", "accession"),
                           [{"fold": p.fold, "accession": p.accession} for p in proteins])
    labels = [{"accession": p.accession, "ec_numbers": p.ec_numbers} for p in proteins]
    labels.append({"accession": "TRAINONLY1", "ec_numbers": "2.7.1.1"})
    label_table = write_csv(root / "labels" / "native_labels.tsv", ("accession", "ec_numbers"), labels, delimiter="\t")
    transfer_entry = None
    if transfers:
        transfer_path = write_csv(root / "labels" / "ec_class_transfers.csv", scoring.TRANSFER_COLUMNS,
                                  [{"ec_number": source, "transferred_to": targets} for source, targets in transfers])
        transfer_entry = {"path": str(transfer_path), "sha256": runner.sha256_file(transfer_path)}

    cohort = root / "identity" / "cohort"
    component_rows = [{"accession": accession, "component_id": cluster,
                       "component_size": sum(value == cluster for value in clusters.values())}
                      for accession, cluster in sorted(clusters.items())]
    component_rows.append({"accession": "TRAINONLY1", "component_id": "K_TRAINONLY1", "component_size": 1})
    clusters_path = write_csv(cohort / "components_cohort.tsv", COMPONENT_COLUMNS, component_rows, delimiter="\t")
    clusters_sha = runner.sha256_file(clusters_path)
    write_json(cohort / "components_receipt_cohort.json", {
        "scope": "cohort", "freeze1_sha256": FREEZE1, "outputs": {"components_cohort.tsv": {"sha256": clusters_sha}}})
    strata = {}
    for fold in folds:
        path = write_csv(root / "identity" / f"fold{fold}" / f"strata_fold{fold}.tsv", STRATA_COLUMNS,
                         [strata_row(p, clusters[p.accession])
                          for p in sorted(by_fold[fold], key=lambda p: p.accession)],
                         delimiter="\t")
        write_json(path.with_name("strata_receipt.json"), {
            "schema_version": 1, "step": "strata", "fold": fold, "freeze1_sha256": FREEZE1,
            "identity_bins": list(IDENTITY_BIN_LABELS), "labels_read": False, "curated_test_membership_read": False,
            "inputs": {"clusters_sha256": clusters_sha}, "outputs": {path.name: {"sha256": runner.sha256_file(path)}}})
        strata[str(fold)] = str(path)

    deepmzyme, clean = {}, {"r1": {}, "r2": {}, "c_esmc": {}}
    for fold in folds:
        passes, refits = [], []
        for seed in DM_SEEDS:
            folder = root / "test_passes" / f"pass-f{fold}-s{seed}"
            checkpoint = write_dm_pass(folder, fold, seed, by_fold[fold], plan, freeze2, dm_failures, dm_absent)
            passes.append(str(folder))
            refits.append(str(write_refit_receipt(root / "dm_refits" / f"fold{fold}" / f"seed{seed}" /
                                                  scoring.C3_RUN_RECEIPT_NAME, seed, checkpoint, fold=fold)))
        deepmzyme[str(fold)] = {"test_passes": passes, "refit_receipts": refits, "logit_adjustment_tau": tau}
        for arm, regime, seeds, embedding, width in CLEAN_ARM_SPECS:
            queries = [p for p in by_fold[fold] if not (arm == "r1" and p.accession in clean_absent)]
            clean[arm][str(fold)] = [str(write_clean_export(
                root / "clean" / arm / f"fold{fold}" / f"{arm}_fold{fold}_seed{seed}.npz", fold=fold, regime=regime,
                seed=seed, embedding=embedding, input_dim=width, proteins=queries, binding=clean_binding(freeze2),
                outcome=lambda protein, arm=arm, seed=seed: clean_plan(arm, seed, protein))) for seed in seeds]

    def class_rows(system):
        rows = []
        for protein in proteins:
            outcome = plan(system, None, protein)
            predicted = outcome not in ("no_hit", "tie", "failure")
            rows.append({"fold": protein.fold, "accession": protein.accession,
                         "predicted_ec_class": outcome if predicted else "",
                         "status": "predicted" if predicted else outcome})
        return rows

    baseline = write_csv(root / "baseline" / "baseline_predictions.csv", scoring.CLASS_TABLE_COLUMNS,
                         class_rows("baseline"))
    raw = {
        "schema_version": 1, "kind": scoring.MANIFEST_KIND, "report_dir": str(root / "report"), "folds": list(folds),
        "test_membership": str(membership), "label_table": {"path": str(label_table),
                                                            "sha256": runner.sha256_file(label_table)},
        "ec_class_transfers": transfer_entry, "freeze2_manifest": {"path": str(freeze2.path), "sha256": freeze2.sha256},
        "strata": strata, "clusters": str(clusters_path), "deepmzyme": deepmzyme, "clean": clean,
        "baseline": str(baseline), "settings": {"acknowledgement_name": ACKNOWLEDGEMENT},
    }
    if with_b3:
        raw["b3"] = str(write_csv(root / "b3" / "b3_predictions.csv", scoring.CLASS_TABLE_COLUMNS, class_rows("b3")))
    manifest = write_json(root / "scoring_inputs.json", raw)
    return Campaign(root, manifest, label_table, root / "report", proteins, raw)


def claim_and_score(campaign: Campaign) -> dict:
    _, digest = scoring.claim_predictions(campaign.manifest)
    return scoring.score_comparison(campaign.claim_path, digest)


def settings_for_tests() -> dict:
    return scoring.resolve_settings({"acknowledgement_name": ACKNOWLEDGEMENT})


# ---------------------------------------------------------------- claim

def test_claim_hashes_every_input_and_records_no_label(space):
    campaign = build_campaign(space)
    claim, digest = scoring.claim_predictions(campaign.manifest)
    assert runner.sha256_file(campaign.claim_path) == digest
    assert claim["labels_read"] is False and claim["label_side_inputs"]["opened_by_claim"] is False
    assert claim["label_side_inputs"]["label_table"]["declared_sha256"] == runner.sha256_file(campaign.label_table)
    paths = {entry["path"] for entry in claim["inputs"]}
    assert str(campaign.label_table) not in paths
    expected = {campaign.raw["test_membership"], campaign.raw["clusters"], campaign.raw["baseline"],
                str(campaign.manifest), campaign.raw["freeze2_manifest"]["path"]}
    for fold in ("0", "1"):
        expected |= {campaign.raw["strata"][fold]}
        for arm in ("r1", "r2", "c_esmc"):
            for npz in campaign.raw["clean"][arm][fold]:
                expected |= {npz, str(runner.c4_receipt_path(Path(npz)))}
        for folder in campaign.raw["deepmzyme"][fold]["test_passes"]:
            expected |= {str(Path(folder) / name) for name in (scoring.C3_TEST_PROTEIN_PREDICTIONS_NAME,
                                                               scoring.C3_TEST_RECEIPT_NAME)}
        expected |= set(campaign.raw["deepmzyme"][fold]["refit_receipts"])
    assert expected <= paths
    assert all(entry["sha256"] == runner.sha256_file(Path(entry["path"])) for entry in claim["inputs"])
    assert set(claim["code_sha256"]) == set(scoring.SCORER_CODE_PATHS)
    assert claim["freeze2_manifest"]["sha256"] == campaign.raw["freeze2_manifest"]["sha256"]
    assert claim["scoring_ledger"] == str(campaign.scoring_ledger) and not campaign.scoring_ledger.exists()
    assert claim["system_inputs"]["r1_ensemble"] == claim["system_inputs"]["r1_seed_mean"]
    assert claim["coverage"]["r1_ensemble"]["0"] == {"n_membership": 74, "predicted": 74, "tie": 0, "no_hit": 0,
                                                     "failure": 0}
    assert claim["settings_frozen"] is False  # the test switch lowers the draw count


def test_claim_is_write_once(space):
    campaign = build_campaign(space)
    scoring.claim_predictions(campaign.manifest)
    with pytest.raises(ScoringError, match="already exists"):
        scoring.claim_predictions(campaign.manifest)


def test_claim_refuses_a_missing_input(space):
    campaign = build_campaign(space)
    runner.c4_receipt_path(Path(campaign.raw["clean"]["r1"]["1"][2])).unlink()
    with pytest.raises(ScoringError, match="missing"):
        scoring.claim_predictions(campaign.manifest)
    assert not campaign.claim_path.exists()


def test_claim_refuses_a_fold_subset_outside_the_tests(space, monkeypatch):
    campaign = build_campaign(space)
    monkeypatch.setattr(scoring, "_TEST_ALLOW_FOLD_SUBSET", False)
    with pytest.raises(ScoringError, match="covers all folds"):
        scoring.claim_predictions(campaign.manifest)


def test_claim_refuses_a_label_table_listed_as_a_prediction_input(space):
    campaign = build_campaign(space)
    campaign.rewrite_manifest(baseline=str(campaign.label_table))
    with pytest.raises(ScoringError, match="label-side input"):
        scoring.claim_predictions(campaign.manifest)


def test_claim_refuses_another_name_or_a_copy_of_the_label_table(space):
    campaign = build_campaign(space)
    link = space / "baseline" / "linked_predictions.csv"
    os.link(campaign.label_table, link)
    with scoring.label_read_guard([campaign.label_table]):
        with pytest.raises(PermissionError, match="label-side input"):
            link.read_bytes()  # the guard knows the file, not only its name
    campaign.rewrite_manifest(baseline=str(link))
    with pytest.raises(ScoringError, match="label-side input is listed"):
        scoring.claim_predictions(campaign.manifest)
    copy = shutil.copyfile(campaign.label_table, space / "baseline" / "copied_predictions.csv")
    campaign.rewrite_manifest(baseline=str(copy))
    with pytest.raises(ScoringError, match="content of a label-side input"):
        scoring.claim_predictions(campaign.manifest)
    assert not campaign.claim_path.exists()


def test_label_read_guard_refuses_reads_inside_the_block_only(space):
    label = space / "labels.tsv"
    label.write_text("accession\tec_numbers\n")
    with scoring.label_read_guard([label]):
        with pytest.raises(PermissionError):
            label.read_text()
        with pytest.raises(PermissionError):
            os.listdir(label)
    assert label.read_text().startswith("accession")


def test_claim_refuses_inputs_under_a_held_out_root(space, monkeypatch):
    campaign = build_campaign(space)
    held_out = Path(campaign.raw["baseline"]).parent
    monkeypatch.setattr(scoring, "held_out_roots", lambda: [held_out])
    with pytest.raises(ScoringError, match="held-out root"):
        scoring.claim_predictions(campaign.manifest)
    with scoring.label_read_guard([], [held_out]):
        with pytest.raises(PermissionError, match="held-out root"):
            Path(campaign.raw["baseline"]).read_text()
    assert scoring.held_out_roots()  # the comparator's roots are refused by default


def test_claim_stops_if_any_step_tries_to_read_the_labels(space, monkeypatch):
    campaign = build_campaign(space)
    original = scoring.load_label_free_inputs

    def peeking(manifest, freeze2):
        manifest.label_table.path.read_bytes()
        return original(manifest, freeze2)

    monkeypatch.setattr(scoring, "load_label_free_inputs", peeking)
    with pytest.raises(PermissionError, match="label-side input"):
        scoring.claim_predictions(campaign.manifest)
    assert not campaign.claim_path.exists()


def test_claim_never_opens_the_label_table_under_an_audit_hook(space):
    campaign = build_campaign(space, transfers=(("5.1.1.1", "1.14.13.8"),))
    log = space / "opened.json"
    code = textwrap.dedent(f"""
        import json, os, runpy, sys
        opened = []
        def hook(event, args):
            if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                target = os.fsdecode(args[0]) if isinstance(args[0], bytes) else os.fspath(args[0])
                opened.append(os.path.realpath(target))
        sys.addaudithook(hook)
        {SUBPROCESS_SETUP}
        sys.argv = ["score_ec_clean30_comparison.py", "claim", "--manifest", {str(campaign.manifest)!r}]
        status = 0
        try:
            runpy.run_path({str(SCRIPT)!r}, run_name="__main__")
        except SystemExit as exc:
            status = exc.code
        with open({str(log)!r}, "w") as handle:
            json.dump({{"status": status, "opened": opened}}, handle)
    """)
    subprocess.run([sys.executable, "-c", code], check=True, env=ENV, capture_output=True, text=True)
    record = json.loads(log.read_text())
    opened = set(record["opened"])
    assert record["status"] == 0
    assert os.path.realpath(campaign.label_table) not in opened
    assert os.path.realpath(campaign.raw["ec_class_transfers"]["path"]) not in opened
    # The hook saw the claim's real reads and write, so its silence on the label table is meaningful.
    assert os.path.realpath(campaign.raw["strata"]["0"]) in opened
    assert os.path.realpath(campaign.raw["clean"]["r1"]["0"][0]) in opened
    assert os.path.realpath(campaign.claim_path) in opened


@pytest.mark.parametrize("gap, message", [
    ("deepmzyme", "neither a prediction nor an explicit failure"),
    ("clean", "lacks 1 test protein"),
    ("baseline", "has no row for 1 test protein"),
])
def test_claim_refuses_a_test_protein_without_a_row(space, gap, message):
    # A missing row would otherwise become an error of that system (for CLEAN, a DeepMzyme win).
    absent = "F0C3N08"
    campaign = build_campaign(space, dm_absent=(absent,) if gap == "deepmzyme" else (),
                              clean_absent=(absent,) if gap == "clean" else ())
    if gap == "baseline":
        path = Path(campaign.raw["baseline"])
        path.write_text("".join(line for line in path.read_text().splitlines(keepends=True)
                                if f",{absent}," not in line))
    with pytest.raises(ScoringError, match=message):
        scoring.claim_predictions(campaign.manifest)
    assert not campaign.claim_path.exists()


def test_claim_binds_code_and_checkpoints_to_freeze2(space):
    def other_stats_pin(payload):
        payload["code_sha256"]["src/ec_clean30/stats.py"] = "0" * 64

    def unpinned_scorer(payload):
        payload["additional_code_sha256"].pop("scripts/score_ec_clean30_comparison.py")

    def unlisted_clean_checkpoint(payload):
        payload["checkpoint_sha256s"].remove(clean_checkpoint("r2", runner.DEFAULT_EMBEDDING, 1, 43))

    for edit, message in ((other_stats_pin, "differs from its Freeze-2 pin"), (unpinned_scorer, "does not pin"),
                          (unlisted_clean_checkpoint, "Freeze 2 does not list")):
        root = space / edit.__name__
        campaign = build_campaign(root, freeze2_edit=edit)
        with pytest.raises(ScoringError, match=message):
            scoring.claim_predictions(campaign.manifest)
    campaign = build_campaign(space / "declared")
    campaign.rewrite_manifest(freeze2_manifest={**campaign.raw["freeze2_manifest"], "sha256": "3" * 64})
    with pytest.raises(ScoringError, match="does not hash to its declared"):
        scoring.claim_predictions(campaign.manifest)


def test_refit_receipts_must_be_the_folds_s4_refits(space):
    proteins = [Protein(0, "F0C1N00", "1", "nh30", 0)]
    freeze2_file = write_freeze2(space, [dm_checkpoint(0, seed) for seed in DM_SEEDS])
    freeze2 = scoring.read_freeze2_manifest(freeze2_file)
    passes = []
    for seed in DM_SEEDS:
        write_dm_pass(space / "test_passes" / f"p{seed}", 0, seed, proteins, better_plan, freeze2_file)
        passes.append(space / "test_passes" / f"p{seed}")

    tags = iter(range(100))

    def load(changes: dict, seeds_changed=(46,)):
        folder = space / f"refits{next(tags)}"
        refits = [write_refit_receipt(folder / f"seed{seed}" / scoring.C3_RUN_RECEIPT_NAME, seed,
                                      dm_checkpoint(0, seed), identity=changes if seed in seeds_changed else None)
                  for seed in DM_SEEDS]
        return scoring.load_deepmzyme_fold(0, scoring.DeepMzymeFoldInputs(tuple(passes), tuple(refits), 0.0),
                                           ["F0C1N00"], settings_for_tests(), freeze2, FREEZE1)

    arms, details = load({})
    assert arms["deepmzyme_final"]["F0C1N00"].token == "1" and details["recipe"] == RECIPE
    for changes in ({"fold": 3}, {"kind": "inner"}, {"step": "S2"}, {"smoke": True}, {"freeze1_sha256": "9" * 64},
                    {"unit_id": "another_unit"}, {"campaign_id": "another_campaign"}, {"inner_fold": 1}):
        with pytest.raises(ScoringError, match="the run identity is not fold 0's S4 refit"):
            load(changes)
    with pytest.raises(ScoringError, match="differ in recipe"):
        load({"recipe": {**RECIPE, "epochs": 12}})


# ---------------------------------------------------------------- score

def test_score_refuses_an_input_changed_after_the_claim(space):
    campaign = build_campaign(space)
    _, digest = scoring.claim_predictions(campaign.manifest)
    csv_path = Path(campaign.raw["deepmzyme"]["0"]["test_passes"][1]) / scoring.C3_TEST_PROTEIN_PREDICTIONS_NAME
    csv_path.write_text(csv_path.read_text().replace("3.043", "3.044", 1))
    with pytest.raises(ScoringError, match="changed after the claim"):
        scoring.score_comparison(campaign.claim_path, digest)
    assert not (campaign.report_dir / scoring.REPORT_JSON_NAME).exists()
    assert not campaign.scoring_ledger.exists()  # refused before the scoring record and the label read


def test_score_refuses_a_wrong_or_edited_claim(space):
    campaign = build_campaign(space)
    _, digest = scoring.claim_predictions(campaign.manifest)
    with pytest.raises(ScoringError, match="does not hash"):
        scoring.score_comparison(campaign.claim_path, "0" * 64)
    edited = campaign.claim_path.read_text().replace('"labels_read": false', '"labels_read": true')
    campaign.claim_path.write_text(edited)
    with pytest.raises(ScoringError, match="does not hash"):
        scoring.score_comparison(campaign.claim_path, digest)


def test_score_refuses_a_label_table_that_does_not_match_its_declared_hash(space):
    campaign = build_campaign(space)
    _, digest = scoring.claim_predictions(campaign.manifest)
    with campaign.label_table.open("a") as handle:
        handle.write("EXTRA\t3.1.1.1\n")
    with pytest.raises(ScoringError, match="declared SHA-256"):
        scoring.score_comparison(campaign.claim_path, digest)


def test_score_refuses_changed_scorer_code(space, monkeypatch):
    campaign = build_campaign(space)
    _, digest = scoring.claim_predictions(campaign.manifest)
    live = scoring.code_sha256()
    monkeypatch.setattr(scoring, "code_sha256", lambda: {**live, "src/ec_clean30/stats.py": "0" * 64})
    with pytest.raises(ScoringError, match="Scorer code differs"):
        scoring.score_comparison(campaign.claim_path, digest)


def test_one_scoring_pass_per_freeze2(space, monkeypatch):
    campaign = build_campaign(space)
    _, first = scoring.claim_predictions(campaign.manifest)
    first_claim = campaign.claim_path
    # A second claim made before any score (another report folder) cannot be scored after the first.
    _, second = scoring.claim_predictions(campaign.manifest_copy("inputs_b.json", report_dir=str(space / "report_b")))
    original = scoring.analyse_comparison

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash after the label read")

    monkeypatch.setattr(scoring, "analyse_comparison", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        scoring.score_comparison(first_claim, first)
    record = json.loads(campaign.scoring_ledger.read_text())
    assert record["claim"]["sha256"] == first and record["label_table_sha256"] == runner.sha256_file(
        campaign.label_table)
    monkeypatch.setattr(scoring, "analyse_comparison", original)
    with pytest.raises(ScoringError, match="one scoring pass per Freeze 2"):
        scoring.score_comparison(space / "report_b" / scoring.CLAIM_NAME, second)
    report = scoring.score_comparison(first_claim, first)  # the crashed pass may finish under the same claim
    assert report["access_receipt"]["scoring_pass"]["resumed"] is True
    # Neither a new claim nor a new report folder reopens it.
    with pytest.raises(ScoringError, match="already scored once"):
        scoring.claim_predictions(campaign.manifest_copy("inputs_c.json", report_dir=str(space / "report_c")))


def test_clearly_better_system_passes_both_co_primaries(space):
    campaign = build_campaign(space, folds=ALL_FOLDS, per_class=6, dm_failures=("F0C3N04",), with_b3=True)
    report = claim_and_score(campaign)
    assert report["decisions"] == {"P1": DECISION_PASS["P1"], "P2": DECISION_PASS["P2"]}
    steps = report["co_primary"]["steps"]
    assert [step["alpha"] for step in steps] == [0.025, 0.05] and all(step["passed"] for step in steps)
    assert steps[0]["p_value"] <= steps[1]["p_value"]
    for step in steps:
        assert {item["name"] for item in step["test"]["components"]} == set(scoring.PRIMARY_COMPONENTS)
        assert all(item["lower_bound"] > 0 for item in step["test"]["components"])
    # An explicit failure is an error for DeepMzyme.
    coverage = report["coverage"]["deepmzyme_final"]["0"]
    assert coverage["failure"] == 1 and coverage["predicted"] == 37
    points = report["points"]["B1-all"]
    assert points["deepmzyme_final"]["n_examples"] == 185
    assert points["deepmzyme_final"]["accuracy"] == pytest.approx(184 / 185)
    assert points["deepmzyme_final"]["status_counts"]["failure"] == 1
    assert points["b3"]["n_examples"] == 185
    # Seed-mean systems keep each seed's confusion matrix.
    per_seed = points["r1_seed_mean"]["arms"]["r1_seed42"]["confusion_matrix_predictions_only"]
    assert np.asarray(per_seed).shape == (7, 7) and sum(map(sum, per_seed)) == 185
    assert report["access_receipt"]["label_table"]["open_events_during_score"] == 1
    assert report["access_receipt"]["scoring_pass"] == {"path": str(campaign.scoring_ledger), "resumed": False}
    assert report["all_five_folds"] is True and report["settings_frozen"] is False
    assert any("test override" in item for item in report["warnings"])
    claim = json.loads(campaign.claim_path.read_text())
    assert claim["held_out_roots_refused"] == [str(root) for root in scoring.held_out_roots()]
    for name in (scoring.REPORT_JSON_NAME, scoring.REPORT_MD_NAME):
        text = (campaign.report_dir / name).read_text(encoding="utf-8")
        assert DECISION_PASS["P1"] in text and DECISION_PASS["P2"] in text
    markdown = (campaign.report_dir / scoring.REPORT_MD_NAME).read_text(encoding="utf-8")
    for word in FORBIDDEN_REPORT_WORDS:
        assert word.lower() not in markdown.lower()
    assert f"used by {ACKNOWLEDGEMENT}" in markdown and "fair-esm 2.0.0" in markdown and "PYTHONHASHSEED" in markdown
    assert "No list of later nomenclature transfers" in markdown
    # One scoring pass: the reports are write-once.
    with pytest.raises(ScoringError, match="already exists"):
        scoring.score_comparison(campaign.claim_path, report["claim"]["sha256"])


def test_identical_systems_are_not_shown_to_outperform(space):
    campaign = build_campaign(space, identical_plan, folds=ALL_FOLDS, per_class=6)
    report = claim_and_score(campaign)
    assert report["decisions"] == {"P1": DECISION_NOT_SHOWN["P1"], "P2": DECISION_NOT_SHOWN["P2"]}
    first, second = report["co_primary"]["steps"]
    assert first["tested"] and not first["passed"] and not second["tested"]
    for item in first["test"]["components"]:
        assert item["estimate"] == 0.0 and item["lower_bound"] == 0.0 and item["p_value"] == 1.0
    text = json.dumps(report).lower() + (campaign.report_dir / scoring.REPORT_MD_NAME).read_text().lower()
    assert "not shown to outperform" in text and "equivalent" not in text


def test_a_fold_subset_issues_no_decision(space):
    report = claim_and_score(build_campaign(space))  # clearly better, but only folds 0 and 1
    expected = scoring.DECISION_NO_DECISION.format(folds=[0, 1])
    assert report["decisions"] == {"P1": expected, "P2": expected}
    assert report["co_primary"]["steps"][0]["passed"]  # the statistics alone would have passed
    assert report["all_five_folds"] is False and any("no decision" in item for item in report["warnings"])


def test_populations_bins_exclusions_and_baseline_errors(space):
    report = claim_and_score(build_campaign(space))
    populations = report["populations"]
    assert report["rows"]["n_membership_rows"] == 148 and report["rows"]["n_excluded_ambiguous_ec_class"] == 2
    assert populations["B1-all"]["n_rows"] == 146
    assert populations["B1-NH30"]["n_rows"] == 50  # no D3 hit and no shared donor
    assert populations["B1-NH30-seq"]["n_rows"] == 98  # no D3 hit
    assert populations["B1-D2"]["n_rows"] == 74
    assert populations["B1-all"]["n_clusters"] == 146 - 2 - 1  # two within-fold pairs, one cross-fold pair
    bins = {label: record["population"]["n_rows"] for label, record in report["descriptive"]["identity_bins"].items()}
    assert bins == {"no_hit": 26, "<=20": 0, "(20,25]": 24, "(25,30)": 48, "[30,50]": 24, ">50": 24}
    assert populations["B1-all"]["class_support"]["7"] == 2.0
    assert len(report["descriptive"]["class_7_outcomes"]) == 2
    assert set(report["descriptive"]["per_fold"]) == {"0", "1"}
    # The baseline's proteins without a hit are errors: 4 of 12 per class and fold are hits, plus both class 7.
    baseline = report["points"]["B1-all"]["baseline"]
    assert baseline["accuracy"] == pytest.approx(50 / 146)
    assert baseline["status_counts"]["no_hit"] == 96 and sum(baseline["abstentions_by_true_class"]) == 96
    assert baseline["per_class_recall"][0] == pytest.approx(1 / 3) and baseline["per_class_precision"][0] == 1.0


def test_projection_tie_rule_counts_ties_wrong(space):
    proteins = default_proteins((0,))[:6]  # class 1, indices 0..5

    def plan(arm, seed, protein):
        if arm == "r1" and seed == 42 and protein.index == 0:
            return "tie:1,4"  # the true class is among the tied classes: still wrong
        if arm == "r1" and seed == 42 and protein.index == 1:
            return "sametie:1"  # two class-1 centres tie: not a tie between classes
        return protein.token

    declared = write_freeze2(space, [clean_checkpoint("r1", runner.DEFAULT_EMBEDDING, 0, seed) for seed in R1_SEEDS])
    freeze2 = scoring.read_freeze2_manifest(declared)
    unbound = [write_clean_export(space / "unbound" / f"r1_seed{seed}.npz", fold=0, regime="r1", seed=seed,
                                  embedding=runner.DEFAULT_EMBEDDING, input_dim=1280, proteins=proteins,
                                  outcome=lambda protein, seed=seed: plan("r1", seed, protein)) for seed in R1_SEEDS]
    paths = [write_clean_export(space / f"r1_seed{seed}.npz", fold=0, regime="r1", seed=seed,
                                embedding=runner.DEFAULT_EMBEDDING, input_dim=1280, proteins=proteins,
                                binding=clean_binding(declared),
                                outcome=lambda protein, seed=seed: plan("r1", seed, protein)) for seed in R1_SEEDS]
    membership = [protein.accession for protein in proteins]
    with pytest.raises(ScoringError, match="carries no Freeze-2 binding"):
        scoring.load_clean_fold("r1", 0, unbound, membership, settings_for_tests(), freeze2)
    other_root = space / "other"
    wrong = [write_clean_export(other_root / f"r1_seed{seed}.npz", fold=0, regime="r1", seed=seed,
                                embedding=runner.DEFAULT_EMBEDDING, input_dim=1280, proteins=proteins,
                                binding={**clean_binding(declared), "test_pass_output_root": str(other_root)},
                                outcome=lambda protein, seed=seed: plan("r1", seed, protein)) for seed in R1_SEEDS]
    with pytest.raises(ScoringError, match="another Freeze 2"):
        scoring.load_clean_fold("r1", 0, wrong, membership, settings_for_tests(), freeze2)
    arms, details = scoring.load_clean_fold("r1", 0, paths, membership, settings_for_tests(), freeze2)
    assert arms["r1_seed42"]["F0C1N00"] == Prediction(None, "tie")
    assert arms["r1_seed42"]["F0C1N01"] == Prediction("1", "predicted")
    assert details["n_ties"]["r1_seed42"] == 1 and details["n_ties"]["r1_ensemble"] == 0
    rows = [ScoredRow(0, accession, 0, 1.0, f"K{accession}", "no_hit", True, True, True) for accession in membership]
    table = ScoringTable(tuple(rows), {name: tuple(arms[name][accession] for accession in membership)
                                       for name in arms}, {})
    arrays = scoring._Arrays(table)
    assert arrays.abstained["r1_seed42"][0]
    assert arrays.predicted["r1_seed42"][0] == scoring.ABSTENTION_COLUMN_FOR_CLASSES_1_6
    metrics = scoring._point_metrics_for_arm(arrays, "r1_seed42", np.ones(len(rows), dtype=bool))
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["abstentions_by_true_class"][0] == 1.0 and metrics["status_counts"]["tie"] == 1
    assert sum(map(sum, metrics["confusion_matrix_predictions_only"])) == 5
    # A seed that lacks a test protein of the fold is refused, not counted as a CLEAN error.
    with pytest.raises(ScoringError, match="lacks 1 test protein"):
        scoring.load_clean_fold("r1", 0, paths, membership + ["F0C2N00"], settings_for_tests(), freeze2)


def test_r1_ensemble_averages_distances_before_projection(space):
    def clean_plan(arm, seed, protein):
        if protein.token == "amb":
            return "1"
        return protein.token if seed in (42, 43, 44) else wrong(protein.token)

    report = claim_and_score(build_campaign(space, clean_plan=clean_plan))
    points = report["points"]["B1-all"]
    assert points["r1_ensemble"]["accuracy"] == pytest.approx(1.0)
    assert points["r1_seed_mean"]["accuracy"] == pytest.approx(0.6)
    assert points["r1_seed_mean"]["n_arms"] == 5 and points["r1_ensemble"]["n_arms"] == 1
    contrast = report["descriptive"]["contrasts"]["B1-all"]["r1_ensemble_minus_r1_seed_mean"]
    assert contrast["estimate"] == pytest.approx(0.4)


def test_logit_adjustment_uses_refit_priors_after_seed_averaging(space):
    proteins = [Protein(0, "F0C1N00", "1", "nh30", 0)]
    tilted = [1.0, 0.0, -5.0, -5.0, -5.0, -5.0, -5.0]  # class 1 ahead of class 2 by 1.0
    freeze2_file = write_freeze2(space, [dm_checkpoint(0, seed) for seed in DM_SEEDS])
    freeze2 = scoring.read_freeze2_manifest(freeze2_file)
    passes, refits = [], []
    counts = {"1": 1000, "2": 1, "3": 10, "4": 10, "5": 10, "6": 10, "7": 10}
    for seed in DM_SEEDS:
        folder = space / "test_passes" / f"pass{seed}"
        checkpoint = write_dm_pass(folder, 0, seed, proteins, lambda *args: tilted, freeze2_file)
        passes.append(folder)
        refits.append(write_refit_receipt(space / f"refit{seed}" / scoring.C3_RUN_RECEIPT_NAME, seed, checkpoint,
                                          counts))
    settings = settings_for_tests()
    for tau, expected in ((0.0, "1"), (1.0, "2")):
        inputs = scoring.DeepMzymeFoldInputs(tuple(passes), tuple(refits), tau)
        arms, details = scoring.load_deepmzyme_fold(0, inputs, ["F0C1N00"], settings, freeze2, FREEZE1)
        assert arms["deepmzyme_final"]["F0C1N00"].token == expected
        assert all(arms[f"deepmzyme_seed{seed}"]["F0C1N00"].token == expected for seed in DM_SEEDS)
    assert details["refit_priors"][0] == pytest.approx(1001 / 1058)


def test_deepmzyme_refuses_foreign_refits_unlisted_checkpoints_and_control_families(space):
    proteins = [Protein(0, "F0C1N00", "1", "nh30", 0)]
    freeze2_file = write_freeze2(space, [dm_checkpoint(0, seed) for seed in DM_SEEDS])
    freeze2 = scoring.read_freeze2_manifest(freeze2_file)
    passes, refits = [], []
    for seed in DM_SEEDS:
        checkpoint = write_dm_pass(space / "test_passes" / f"p{seed}", 0, seed, proteins, better_plan, freeze2_file)
        passes.append(space / "test_passes" / f"p{seed}")
        refits.append(write_refit_receipt(space / f"r{seed}" / scoring.C3_RUN_RECEIPT_NAME, seed,
                                          checkpoint if seed != 46 else sha("other")))
    settings = settings_for_tests()
    inputs = scoring.DeepMzymeFoldInputs(tuple(passes), tuple(refits), 0.0)
    with pytest.raises(ScoringError, match="no refit receipt"):
        scoring.load_deepmzyme_fold(0, inputs, ["F0C1N00"], settings, freeze2, FREEZE1)
    refits[-1] = write_refit_receipt(space / "r46b" / scoring.C3_RUN_RECEIPT_NAME, 46, dm_checkpoint(0, 46))
    inputs = scoring.DeepMzymeFoldInputs(tuple(passes), tuple(refits), 0.0)
    unlisted = replace(freeze2, checkpoint_sha256s=frozenset(dm_checkpoint(0, seed) for seed in DM_SEEDS[:4]))
    with pytest.raises(ScoringError, match="Freeze 2 does not list"):
        scoring.load_deepmzyme_fold(0, inputs, ["F0C1N00"], settings, unlisted, FREEZE1)
    with pytest.raises(ScoringError, match="another Freeze-2 manifest"):
        scoring.load_deepmzyme_fold(0, inputs, ["F0C1N00"], settings, replace(freeze2, sha256="4" * 64), FREEZE1)
    for path in [folder / scoring.C3_TEST_RECEIPT_NAME for folder in passes] + refits:
        payload = json.loads(path.read_text())
        payload["model"] = {**payload["model"], "campaign_family": "C-G"}
        write_json(path, payload)
    with pytest.raises(ScoringError, match="not a selectable candidate"):
        scoring.load_deepmzyme_fold(0, inputs, ["F0C1N00"], settings, freeze2, FREEZE1)
    with pytest.raises(ScoringError, match="must both equal"):
        scoring.load_deepmzyme_fold(0, scoring.DeepMzymeFoldInputs(tuple(passes[:4]), tuple(refits[:4]), 0.0),
                                    ["F0C1N00"], settings, freeze2, FREEZE1)


def test_rescore_without_ec_class_transfers(space):
    # 5.1.1.1 and 6.1.1.4 are carried by one protein in each fold; 2.7.7.99 by none; 3.1.1.1 stays in class 3.
    transfers = (("5.1.1.1", "1.14.13.8"), ("6.1.1.4", "3.6.1.2; 6.3.4.1"), ("2.7.7.99", "7.1.1.1"),
                 ("3.1.1.1", "3.1.1.2"))
    report = claim_and_score(build_campaign(space, transfers=transfers))
    rescore = report["descriptive"]["ec_class_transfer_rescore"]
    assert rescore["P1"]["population"]["n_rows"] == 142 and rescore["P2"]["population"]["n_rows"] == 46
    assert rescore["P1"]["test"]["available"] and rescore["P1"]["test"]["passed"]
    assert report["rows"]["n_ec_class_transfers_listed"] == 3
    assert report["rows"]["n_ec_class_transfer_rows_ignored"] == 1
    assert report["rows"]["n_b1_all_rows_with_ec_class_transfer"] == 4
    assert report["access_receipt"]["ec_class_transfers"]["sha256"]
    disclosure = next(item for item in report["disclosures"] if item["id"] == scoring.TRANSFER_DISCLOSURE_ID)
    assert disclosure["text"].startswith("The transfer list names 3 EC number(s)") and "4 B1-all row(s)" in disclosure["text"]


@pytest.mark.parametrize("row", [("5.1.1.1;5.1.1.2", "1.1.1.1"), ("Q12345", "1.1.1.1"), ("5.1.1.1", ""),
                                 ("5.1.1.1", "9.1.1.1")])
def test_transfer_list_refuses_rows_that_are_not_ec_numbers(row):
    data = ("ec_number,transferred_to\n" + ",".join(f'"{value}"' for value in row) + "\n").encode()
    with pytest.raises(ScoringError, match="ec_class_transfers.csv"):
        scoring.parse_transfer_list(data, "ec_class_transfers.csv")


def test_transfer_list_with_accession_columns_is_refused():
    with pytest.raises(ScoringError, match="exactly the columns"):
        scoring.parse_transfer_list(b"accession\nF0C5N00\n", "ec_class_transfers.csv")


# ---------------------------------------------------------------- analysis on synthetic tables

KIND_STRATA = {"nh30": ("no_hit", True, True, True), "donor": ("(25,30)", False, True, True),
               "high": ("[30,50]", False, False, False)}


def synthetic_table(specs, dm, r1, *, r1_ensemble=None) -> ScoringTable:
    """``specs``: (accession, true token, kind); ``dm``/``r1``: functions of (index, spec) -> token or None."""
    rows = []
    for index, (accession, token, kind) in enumerate(specs):
        identity_bin, nh30, nh30_seq, d2 = KIND_STRATA[kind]
        rows.append(ScoredRow(0, accession, int(token) - 1, 1.0, f"K{accession}", identity_bin, nh30, nh30_seq, d2))

    def arm(function):
        values = []
        for index, spec in enumerate(specs):
            token = function(index, spec)
            values.append(Prediction(token, "predicted") if token is not None else Prediction(None, "failure"))
        return tuple(values)

    arms = {"deepmzyme_final": arm(dm), "r1_ensemble": arm(r1_ensemble or r1), "baseline": arm(r1)}
    for seed in DM_SEEDS:
        arms[f"deepmzyme_seed{seed}"] = arms["deepmzyme_final"]
    for seed in R1_SEEDS:
        arms[f"r1_seed{seed}"] = arm(r1)
    for seed in THREE_SEEDS:
        arms[f"r2_seed{seed}"] = arm(r1)
    arms["r2_ensemble"] = arm(r1)
    settings = scoring.validate_settings(scoring.FROZEN_SETTINGS)
    return ScoringTable(tuple(rows), arms, scoring.system_definitions(sorted(arms), settings))


def specs_for(kinds_per_class: dict, classes="123456") -> list:
    specs = []
    for token in classes:
        for kind, count in kinds_per_class.items():
            specs += [(f"C{token}{kind}{i:02d}", token, kind) for i in range(count)]
    return specs


# The analysis is pure; it takes any settings (the score path cannot: it resolves the frozen ones).
ANALYSIS_SETTINGS = {**scoring.validate_settings(scoring.FROZEN_SETTINGS), "acknowledgement_name": ACKNOWLEDGEMENT,
                     "n_draws": TEST_DRAWS, "bootstrap_seed": 11}


def truth(index, spec):
    return spec[1]


def test_holm_tests_the_smaller_p_value_first():
    specs = specs_for({"nh30": 10, "high": 14})
    dm = lambda index, spec: spec[1] if spec[2] == "nh30" else wrong(spec[1])  # noqa: E731
    r1 = lambda index, spec: wrong(spec[1]) if spec[2] == "nh30" else spec[1]  # noqa: E731
    result = scoring.analyse_comparison(synthetic_table(specs, dm, r1), ANALYSIS_SETTINGS)
    first, second = result["co_primary"]["steps"]
    assert (first["name"], first["alpha"], first["passed"]) == ("P2", 0.025, True)
    assert (second["name"], second["alpha"], second["tested"], second["passed"]) == ("P1", 0.05, True, False)
    assert result["decisions"] == {"P1": DECISION_NOT_SHOWN["P1"], "P2": DECISION_PASS["P2"]}


def test_a_failed_first_co_primary_stops_the_second():
    specs = specs_for({"nh30": 10, "high": 10})
    dm = lambda index, spec: wrong(spec[1]) if index % 2 else spec[1]  # noqa: E731
    result = scoring.analyse_comparison(synthetic_table(specs, dm, truth), ANALYSIS_SETTINGS)
    first, second = result["co_primary"]["steps"]
    assert first["tested"] and not first["passed"] and not second["tested"] and not second["passed"]


def test_beating_r1_seed_mean_alone_does_not_pass():
    # Intersection-union: DeepMzyme (75% correct) beats the R1 seeds (50%) but not R1-ens (all correct).
    specs = specs_for({"nh30": 10, "high": 10})
    dm = lambda index, spec: spec[1] if index % 4 else wrong(spec[1])  # noqa: E731
    r1 = lambda index, spec: spec[1] if index % 2 else wrong(spec[1])  # noqa: E731
    result = scoring.analyse_comparison(synthetic_table(specs, dm, r1, r1_ensemble=truth), ANALYSIS_SETTINGS)
    assert result["decisions"] == {"P1": DECISION_NOT_SHOWN["P1"], "P2": DECISION_NOT_SHOWN["P2"]}
    first = result["co_primary"]["steps"][0]
    bounds = {item["name"]: item["lower_bound"] for item in first["test"]["components"]}
    assert bounds["deepmzyme_final_minus_r1_seed_mean"] > 0 > bounds["deepmzyme_final_minus_r1_ensemble"]
    assert first["p_value"] == max(item["p_value"] for item in first["test"]["components"]) and not first["passed"]


def test_template_check_reports_both_directions():
    specs = specs_for({"nh30": 10, "donor": 10, "high": 10})
    # P2 (no D3 hit, no shared donor) passes; B1-NH30-seq also holds donor-sharing proteins, where R1 wins.
    dm = lambda index, spec: spec[1] if spec[2] == "nh30" else wrong(spec[1])  # noqa: E731
    r1 = lambda index, spec: wrong(spec[1]) if spec[2] == "nh30" else spec[1]  # noqa: E731
    result = scoring.analyse_comparison(synthetic_table(specs, dm, r1), ANALYSIS_SETTINGS)
    check = result["template_check"]
    assert check["p2_passed"] and not check["test"]["passed"] and check["claims_differ"]
    assert check["template_dependent"] and not check["sequence_only_passes_without_p2"]
    assert result["decisions"]["P2"] == (f"{DECISION_PASS['P2']} (template-dependent: the sequence-only "
                                         "population B1-NH30-seq does not support it)")
    # The reverse: equal on B1-NH30, DeepMzyme better on the donor-sharing proteins of B1-NH30-seq.
    dm = lambda index, spec: wrong(spec[1]) if spec[2] == "donor" and index % 5 == 0 else spec[1]  # noqa: E731
    r1 = lambda index, spec: wrong(spec[1]) if spec[2] == "donor" else spec[1]  # noqa: E731
    result = scoring.analyse_comparison(synthetic_table(specs, dm, r1), ANALYSIS_SETTINGS)
    check = result["template_check"]
    assert not check["p2_passed"] and check["test"]["passed"] and check["claims_differ"]
    assert check["sequence_only_passes_without_p2"] and not check["template_dependent"]
    assert result["decisions"]["P2"] == DECISION_NOT_SHOWN["P2"]


def test_similarity_difference_in_differences_sign():
    specs = specs_for({"nh30": 10, "high": 10})
    half = lambda index, spec: spec[1] if index % 2 == 0 else wrong(spec[1])  # noqa: E731
    clean_needs_similarity = lambda index, spec: spec[1] if spec[2] == "high" else wrong(spec[1])  # noqa: E731
    result = scoring.analyse_comparison(synthetic_table(specs, half, clean_needs_similarity), ANALYSIS_SETTINGS)
    did = result["secondary_family"]["difference_in_differences"]
    assert did["estimate"] == pytest.approx(1.0)
    step = next(item for item in result["secondary_family"]["steps"] if item["name"] == scoring.DID_CONTRAST)
    assert step["available"] and step["p_value"] == 0.0
    reverse = scoring.analyse_comparison(synthetic_table(specs, clean_needs_similarity, half), ANALYSIS_SETTINGS)
    assert reverse["secondary_family"]["difference_in_differences"]["estimate"] == pytest.approx(-1.0)
    assert len(result["secondary_family"]["steps"]) == 7


def test_rare_class_removal_marks_a_dependent_claim():
    specs = specs_for({"nh30": 6, "high": 6})
    dm = lambda index, spec: spec[1]  # noqa: E731
    r1 = lambda index, spec: "1" if spec[1] == "5" else spec[1]  # noqa: E731
    result = scoring.analyse_comparison(synthetic_table(specs, dm, r1), ANALYSIS_SETTINGS)
    assert all(step["passed"] for step in result["co_primary"]["steps"])
    for name in ("P1", "P2"):
        record = result["rare_class_dependence"][name]
        assert record["dependent_on_ec_classes"] == ["5"]
        assert not record["removed"]["5"]["test"]["passed"] and record["removed"]["6"]["test"]["passed"]
        assert result["decisions"][name] == f"{DECISION_PASS[name]} (dependent on EC class 5)"


def test_seed_mean_rounding_does_not_separate_identical_systems():
    # Five identical arms average to values that differ from one arm in the last place, so the raw
    # engine gives identical systems p < 1; the scorer's snap restores p = 1 and a bound of exactly 0.
    rng = np.random.default_rng(11)
    n = int(rng.integers(30, 120))
    true = rng.integers(0, 6, n)
    true[:6] = np.arange(6)
    predicted = np.where(rng.random(n) < 0.6, true, rng.integers(0, 7, n))
    ids = [f"p{i}" for i in range(n)]
    table = stats.build_cluster_count_table(ids, true, {"one": predicted, **{f"r{k}": predicted for k in range(5)}},
                                            {pid: pid for pid in ids})
    raw = stats.paired_cluster_bootstrap(table, [stats.Contrast.difference("d", "one", "mean")],
                                         systems={"one": ["one"], "mean": [f"r{k}" for k in range(5)]},
                                         n_draws=200, seed=3, max_skip_fraction=1.0)
    assert np.count_nonzero(raw.draws("d")) > 0 and raw.p_value("d") < 1.0
    assert np.all(np.abs(raw.draws("d")) < scoring.CONTRAST_ZERO_TOLERANCE)
    snapped = scoring.snap_contrasts(raw)
    assert np.all(snapped.draws("d") == 0.0) and snapped.p_value("d") == 1.0
    assert snapped.lower_bound("d", 0.025) == 0.0 and not snapped.test("d", 0.025).passed


def test_failures_and_ties_are_errors_for_their_system():
    specs = specs_for({"nh30": 4, "high": 4})
    dm = lambda index, spec: None if index == 0 else spec[1]  # noqa: E731
    result = scoring.analyse_comparison(synthetic_table(specs, dm, truth), ANALYSIS_SETTINGS)
    points = result["points"]["B1-all"]
    assert points["deepmzyme_final"]["accuracy"] == pytest.approx(1 - 1 / len(specs))
    assert points["r1_seed_mean"]["accuracy"] == 1.0
    assert points["deepmzyme_final"]["abstentions_by_true_class"][0] == 1.0
    assert sum(map(sum, points["deepmzyme_final"]["confusion_matrix_predictions_only"])) == len(specs) - 1


def test_class_7_errors_add_no_false_class_1_prediction():
    # DeepMzyme fails on every class-7 protein and is right elsewhere: accuracy and seven-class BA drop,
    # but no class 1-6 precision may change, so macro-F1 over classes 1-6 stays 1 (points and bootstrap).
    specs = specs_for({"nh30": 6, "high": 6}, classes="1234567")
    dm = lambda index, spec: None if spec[1] == "7" else spec[1]  # noqa: E731
    result = scoring.analyse_comparison(synthetic_table(specs, dm, truth), ANALYSIS_SETTINGS)
    point = result["points"]["B1-all"]["deepmzyme_final"]
    assert point["accuracy"] == pytest.approx(72 / 84) and point["seven_class_balanced_accuracy"] < 1.0
    assert point["per_class_precision"][:6] == [1.0] * 6 and point["ec_classes_1_6_macro_f1"] == 1.0
    assert point["n_class_7_abstentions"] == 12.0
    bootstrap = result["bootstrap"][f"{stats.METRIC_EC_CLASSES_1_6_MACRO_F1} B1-all"]
    assert bootstrap["system_point"]["all"]["deepmzyme_final"] == 1.0
    accuracy = result["bootstrap"][f"{stats.METRIC_ACCURACY} B1-all"]
    assert accuracy["system_point"]["all"]["deepmzyme_final"] == pytest.approx(72 / 84)


def test_untestable_secondary_hypothesis_is_ordered_last_and_never_passes():
    specs = specs_for({"nh30": 10})  # no high-similarity protein: the difference in differences cannot run
    result = scoring.analyse_comparison(synthetic_table(specs, truth, lambda i, s: wrong(s[1]) if i % 2 else s[1]),
                                        ANALYSIS_SETTINGS)
    steps = result["secondary_family"]["steps"]
    did = next(step for step in steps if step["name"] == scoring.DID_CONTRAST)
    assert not did["available"] and not did["passed"] and did["p_value"] == 1.0 and "empty" in did["reason"]
    assert [step["alpha"] for step in steps] == pytest.approx([0.05 / (7 - rank) for rank in range(7)])


def test_decision_wording_follows_the_design():
    assert scoring.decision_wording("P1", True) == "Outperforms CLEAN on the official CLEAN split30 benchmark"
    assert scoring.decision_wording("P2", True) == (
        "Outperforms CLEAN on test proteins without a ≥30% identity hit to training")
    for name in ("P1", "P2"):
        assert scoring.decision_wording(name, False, ["5"], True).startswith("Not shown to outperform CLEAN on ")
        assert scoring.decision_wording(name, True, folds=(0, 1)) == (
            "No decision: only folds [0, 1] of the five official folds were scored")
    assert scoring.decision_wording("P2", True, ["5", "6"], True) == (
        f"{DECISION_PASS['P2']} (dependent on EC class 5 and EC class 6; template-dependent: the sequence-only "
        "population B1-NH30-seq does not support it)")
    texts = [scoring.decision_wording(name, passed, classes, template, folds) for name in ("P1", "P2")
             for passed in (True, False) for classes in ((), ("5",)) for template in (True, False)
             for folds in (ALL_FOLDS, (0,))]
    assert not any("equivalent" in text.lower() or "EC1" in text for text in texts)


def test_repeated_accession_is_weighted_to_one_and_kept_in_one_cluster():
    row = {"identity_bin": "no_hit", "no_d3_hit": "1", "no_d3_hit_no_shared_donor": "1", "has_d2": "0",
           "has_d2_min_pairs": "0", "cluster_id": "K1"}
    loaded = scoring.LoadedInputs(
        membership={0: ("A1",), 1: ("A1", "B1")},
        strata={0: {"A1": row}, 1: {"A1": row, "B1": {**row, "cluster_id": "K2"}}},
        clusters={"A1": "K1", "B1": "K2"},
        arms={"baseline": {(0, "A1"): Prediction("1", "predicted"), (1, "A1"): Prediction("1", "predicted"),
                           (1, "B1"): Prediction("2", "predicted")}},
        systems={}, details={})
    table, summary = scoring.build_scoring_table(loaded, {"A1": "1.1.1.1", "B1": "2.1.1.1"}, "has_d2")
    assert [row.weight for row in table.rows] == [0.5, 0.5, 1.0]
    assert summary["n_repeated_accessions"] == 1 and table.folds == (0, 1)


def test_settings_are_frozen_and_the_acknowledgement_is_required(space, monkeypatch):
    monkeypatch.setattr(scoring, "_TEST_SETTING_OVERRIDES", {})
    settings = scoring.resolve_settings({"acknowledgement_name": ACKNOWLEDGEMENT, "n_draws": 10_000,
                                         "r1_seeds": [46, 45, 44, 43, 42]})  # repeating a frozen value is allowed
    assert scoring.settings_are_frozen(settings) and settings["n_draws"] == stats.DEFAULT_N_DRAWS
    for changes, message in (({"n_draw": 10}, "Unknown scoring settings"), ({"r2_form": "best"}, "r2_form"),
                             ({"r2_form": "ensemble"}, "differs from the frozen value"),
                             ({"n_draws": 50}, "differs from the frozen value"),
                             ({"bootstrap_seed": 1}, "differs from the frozen value"),
                             ({"max_skip_fraction": 0.5}, "differs from the frozen value"),
                             ({"secondary_family_alpha": 0.1}, "differs from the frozen value"),
                             ({"d2_flag": "has_d2_min_pairs"}, "differs from the frozen value")):
        with pytest.raises(ScoringError, match=message):
            scoring.resolve_settings({"acknowledgement_name": ACKNOWLEDGEMENT, **changes})
    for name in (None, "", "[name]", "<institution>"):
        with pytest.raises(ScoringError, match="acknowledgement_name"):
            scoring.resolve_settings({} if name is None else {"acknowledgement_name": name})
    campaign = build_campaign(space)
    campaign.rewrite_manifest(extra_key=1)
    with pytest.raises(ScoringError, match="unknown"):
        scoring.load_manifest(campaign.manifest)


def test_analysis_is_deterministic_across_hash_seeds():
    code = textwrap.dedent(f"""
        {SUBPROCESS_SETUP}
        import test_ec_clean30_scoring as t
        specs = t.specs_for({{"nh30": 6, "donor": 4, "high": 6}}, classes="1234567")
        dm = lambda i, s: s[1] if i % 3 else t.wrong(s[1])
        r1 = lambda i, s: s[1] if i % 2 else t.wrong(s[1])
        result = _scoring.analyse_comparison(t.synthetic_table(specs, dm, r1), t.ANALYSIS_SETTINGS)
        sys.stdout.write(_scoring._json_text(result))
    """)
    outputs = [subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                              env={**ENV, "PYTHONHASHSEED": seed}).stdout for seed in ("0", "12345")]
    assert outputs[0] == outputs[1] and '"co_primary"' in outputs[0]


# ---------------------------------------------------------------- module contracts

def test_disclosures_are_fixed_and_publicly_worded():
    ids = [key for key, _ in scoring.DISCLOSURES]
    assert ids == ["language_model_pretraining", "structure_prediction_inputs", "seen_ec_number_test_closure",
                   "metallo_cohort_screening_gap", "computational_metal_sites",
                   "deepmzyme_trains_on_metalloenzymes_only", "tuning_asymmetry", "primary_metric_deviation",
                   "design_time_label_exposure", "prior_test_metadata_exposure", "clean_keyed_masks",
                   "clean_fair_esm_version", "clean_pythonhashseed"]
    summary = {"n_ec_class_transfers_listed": 0, "n_b1_all_rows_with_ec_class_transfer": 0}
    transfers = [scoring.transfer_disclosure(summary, supplied) for supplied in (False, True)]
    text = " ".join([item for _, item in scoring.DISCLOSURES] + transfers)
    assert "(accession, seed)" in text and "fair-esm 2.0.0" in text and "1.0.2" in text and "PYTHONHASHSEED" in text
    assert "AlphaFold" in text and "declared deviation" in text
    assert "no test label was used" not in text  # it would contradict the design-time label exposure
    for word in FORBIDDEN_REPORT_WORDS:
        assert word.lower() not in text.lower()


def test_copied_c3_constants_match_the_export_module():
    source = (ROOT / "src" / "training" / "ec_prediction_export.py").read_text()
    tree = ast.parse(source)
    names = ("LOGIT_COLUMNS", "TEST_PROTEIN_PREDICTION_COLUMNS", "TEST_PROTEIN_PREDICTIONS_NAME", "TEST_RECEIPT_NAME",
             "EC_PREDICTION_RECEIPT_NAME", "MODE_PROBE", "AGGREGATION", "FREEZE2_SCHEMA_VERSION")
    namespace = {"EC_CLASS_TOKENS": tuple("1234567")}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            expression = ast.get_source_segment(source, node.value)
            namespace[node.targets[0].id] = eval(expression, {}, namespace)  # noqa: S307
    assert namespace["LOGIT_COLUMNS"] == scoring.C3_LOGIT_COLUMNS
    assert namespace["TEST_PROTEIN_PREDICTION_COLUMNS"] == scoring.C3_TEST_PROTEIN_PREDICTION_COLUMNS
    assert namespace["TEST_PROTEIN_PREDICTIONS_NAME"] == scoring.C3_TEST_PROTEIN_PREDICTIONS_NAME
    assert namespace["TEST_RECEIPT_NAME"] == scoring.C3_TEST_RECEIPT_NAME
    assert namespace["EC_PREDICTION_RECEIPT_NAME"] == scoring.C3_RUN_RECEIPT_NAME
    assert namespace["MODE_PROBE"] == scoring.C3_MODE_PROBE and namespace["AGGREGATION"] == scoring.C3_AGGREGATION
    assert namespace["FREEZE2_SCHEMA_VERSION"] == scoring.C3_FREEZE2_SCHEMA_VERSION


def test_scorer_is_torch_free_and_pins_every_repository_module_it_loads():
    code = (
        f"import json, os, sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import ec_clean30.scoring; "
        f"root = os.path.realpath({str(ROOT)!r}); "
        "files = sorted(os.path.relpath(os.path.realpath(m.__file__), root) for m in list(sys.modules.values()) "
        "if getattr(m, '__file__', None) and os.path.realpath(m.__file__).startswith(root + os.sep)); "
        "print(json.dumps({'torch': 'torch' in sys.modules, 'files': files}))"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=ENV).stdout
    record = json.loads(output)
    assert record["torch"] is False
    assert set(record["files"]) == set(scoring.SCORER_CODE_PATHS) - {"scripts/score_ec_clean30_comparison.py"}


def run_script(*arguments: str) -> subprocess.CompletedProcess:
    """The command-line script under the in-process test switches (no flag reaches them)."""
    code = (f"{SUBPROCESS_SETUP}; import runpy; sys.argv = ['score_ec_clean30_comparison.py', *{list(arguments)!r}]; "
            f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=ENV)


def test_command_line_help_and_round_trip(space):
    help_text = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True,
                               env=ENV).stdout
    assert "claim" in help_text and "score" in help_text
    campaign = build_campaign(space, folds=ALL_FOLDS, per_class=6)
    claimed = run_script("claim", "--manifest", str(campaign.manifest))
    assert claimed.returncode == 0, claimed.stderr
    digest = json.loads(claimed.stdout)["claim_sha256"]
    scored = run_script("score", "--claim", str(campaign.claim_path), "--claim-sha256", digest)
    assert scored.returncode == 0, scored.stderr
    assert json.loads(scored.stdout)["decisions"]["P1"] == DECISION_PASS["P1"]
    again = run_script("score", "--claim", str(campaign.claim_path), "--claim-sha256", digest)
    assert again.returncode != 0 and "already exists" in again.stderr
    # Without the test switches the script refuses a manifest that changes a frozen setting.
    campaign.rewrite_manifest(report_dir=str(space / "report_b"),
                              settings={"acknowledgement_name": ACKNOWLEDGEMENT, "n_draws": 400})
    frozen = subprocess.run([sys.executable, str(SCRIPT), "claim", "--manifest", str(campaign.manifest)],
                            capture_output=True, text=True, env=ENV)
    assert frozen.returncode != 0 and "differs from the frozen value" in frozen.stderr
