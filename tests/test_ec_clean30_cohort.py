"""Training-side cohort audit of the EC class vs CLEAN30 campaign (synthetic data only).

The fixtures below write a five-fold synthetic data root in the layout the audit reads: each
protein sits in four of the five training files, structures are small PDB files with explicit
metal sites, ESMC payloads are random 960-wide tensors with the real payload and sidecar format,
and the external features are produced by the feature module itself (without PROPKA), so the
charge-proxy parity can be checked exactly. No real campaign file is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ec_clean30 import cohort  # noqa: E402
from ec_clean30.campaign import CAMPAIGN_ID  # noqa: E402
from ec_clean30.cohort import (  # noqa: E402
    COORDINATE_COLUMNS, CohortAuditError, D2_METALS_NAME, D2_RESIDUES_NAME, PROTEINS_NAME, QUARANTINE_NAME,
    RECEIPT_NAME, SITE_FAILURES_NAME, SITES_NAME, run_cohort_audit,
)
from ec_clean30.identity import DONOR_COLUMN  # noqa: E402
from ec_clean30.provenance import HeldOutInputError  # noqa: E402
from ec_clean30.views import SourceLayout, ViewError  # noqa: E402

PROVENANCE = {"note": "synthetic test provenance"}
N_FOLDS = 5
ESM_DIM = 960
TRAIN_COLUMNS = ("structure", "chain_resi", "metaltype", "ecnumber", "whether_catalytic", "uniprot_id",
                 "clean_identity", "clean_fold", DONOR_COLUMN, *COORDINATE_COLUMNS)
LABEL_COLUMNS = ("accession", "ec_numbers", "ec_class", "ec_class_status", "n_files", "sequence_sha256")
MAP_COLUMNS = ("accession", "sha256", "length", "n_files", "sequence")
AMBIGUOUS = "P9000"
SIDE_CHAIN = {"HIS": ("NE2", "N"), "ASP": ("OD1", "O"), "GLU": ("OE1", "O"), "CYS": ("SG", "S"),
              "LYS": ("NZ", "N"), "ARG": ("NH1", "N"), "SER": ("OG", "O")}
DEFAULT_RESIDUES = (("HIS", (0.0, 3.6, 0.0)), ("ASP", (3.6, 0.5, 0.0)), ("GLU", (-3.6, 0.5, 0.0)),
                    ("CYS", (0.5, -3.6, 0.0)), ("LYS", (0.0, 0.5, 6.0)), ("ALA", (0.5, 0.0, -6.0)),
                    ("ARG", (7.0, 7.0, 1.0)), ("SER", (25.0, 0.0, 0.0)))
DEFAULT_SEQUENCE = "HDECKARS"
FAR_RESSEQ = 8  # the SER 25 Å from the site: never a pocket residue


# ---------------------------------------------------------------------------
# Synthetic five-fold data root
# ---------------------------------------------------------------------------

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _out(base: Path) -> Path:
    """The only folder the audit writes: <campaign root>/cohort, the campaign root named after the campaign."""
    return base / CAMPAIGN_ID / "cohort"


def _pdb_line(record, serial, name, resname, chain, resseq, xyz, element):
    atom_name = f" {name:<3}" if len(name) < 4 and len(element) == 1 else f"{name:<4}"
    x, y, z = xyz
    return (f"{record:<6}{serial:>5} {atom_name} {resname:>3} {chain:1}{resseq:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          {element:>2}")


def _residue_atoms(resname, ca):
    atoms = [("N", "N", (ca[0] + 1.0, ca[1], ca[2] + 0.9)), ("CA", "C", ca),
             ("C", "C", (ca[0] - 1.0, ca[1], ca[2] + 0.9)), ("O", "O", (ca[0] - 1.2, ca[1], ca[2] + 2.1))]
    if resname in SIDE_CHAIN:
        name, element = SIDE_CHAIN[resname]
        norm = math.sqrt(sum(value * value for value in ca)) or 1.0
        atoms.append((name, element, tuple(value - 1.5 * value / norm for value in ca)))
    return atoms


def write_structure(path: Path, residues, metals) -> None:
    """``residues``: (resname, CA xyz[, chain]); ``metals``: (resseq, element, xyz[, atom name]) HETATM records of
    chain A (the atom name defaults to the element)."""
    lines, serial, numbers = [], 1, {}
    for item in residues:
        resname, ca = item[0], item[1]
        chain = item[2] if len(item) > 2 else "A"
        numbers[chain] = numbers.get(chain, 0) + 1
        for name, element, xyz in _residue_atoms(resname, ca):
            lines.append(_pdb_line("ATOM", serial, name, resname, chain, numbers[chain], xyz, element))
            serial += 1
    for metal in metals:
        resseq, element, xyz = metal[:3]
        name = metal[3] if len(metal) > 3 else element
        lines.append(_pdb_line("HETATM", serial, name, element, "A", resseq, xyz, element))
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _spec_defaults(spec: dict) -> dict:
    spec = dict(spec)
    spec.setdefault("residues", DEFAULT_RESIDUES)
    spec.setdefault("metals", [(900, "ZN", (0.0, 0.0, 0.0))])
    first = spec["metals"][0]
    spec.setdefault("sites", [(first[0], first[1], first[2], "1")])
    spec.setdefault("pka", "propka")
    return spec


def _write_esm(structure_path: Path, esm_dir: Path, stem: str, *, source_path: str) -> str:
    import torch
    from embed_helpers.esmc import extract_chain_sequences, parse_structure
    from training.esm_feature_loading import (
        build_embedding_payload, embedding_metadata_from_payload, residue_keys_for_structure_chain,
        write_embedding_metadata_sidecar,
    )

    structure = parse_structure(structure_path)
    sequence = extract_chain_sequences(structure)["A"]
    residue_ids, _with_ca = residue_keys_for_structure_chain(structure, "A")
    generator = torch.Generator().manual_seed(len(stem))
    payload = build_embedding_payload(
        torch.randn(len(residue_ids), ESM_DIM, generator=generator), residue_ids, structure_id=stem, chain_id="A",
        source_path=source_path,
        metadata={"esm_model_name": "esmc_300m", "esm_checkpoint_name": "esmc_300m", "embedding_dim": ESM_DIM,
                  "n_residues": len(residue_ids), "source_sequence_sha256": _sha(sequence.encode()),
                  "source_sequence": sequence})
    target = esm_dir / f"{stem}_chain_A_esmc.pt"
    torch.save(payload, target)
    write_embedding_metadata_sidecar(target, embedding_metadata_from_payload(payload))
    return sequence


def _write_external(structure_path: Path, external_root: Path, stem: str, *, pka: str) -> None:
    from feature_extraction.core import build_structure_feature_payload

    payload = build_structure_feature_payload(structure_path, include_propka=False)
    payload["source_path"] = f"/synthetic/structures/{stem}.pdb"
    payload["tooling"] = {**payload["tooling"], "pka": pka}
    target = external_root / stem / "residue_features.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, columns, rows, *, delimiter=",") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path.read_bytes()


def build_cohort(base: Path, proteins: dict[str, dict], *, test_folds: dict[str, int] | None = None,
                 label_overrides: dict[str, str] | None = None) -> dict:
    """Write the synthetic data root, the label table and the sequence map; returns their paths and hashes."""
    from training.ec_classes import classify_ec_class, normalize_ec_number_list

    data = base / "DeepMzyme_Data"
    staging = base / "staging"
    store = data / "structure_store" / "objects"
    esm_dir = data / "esm_embeddings"
    external = data / "updated_feature_extraction"
    for directory in (staging, store, esm_dir, external):
        directory.mkdir(parents=True, exist_ok=True)
    order = sorted(proteins)
    test_fold = {accession: (test_folds or {}).get(accession, index % N_FOLDS) for index, accession in enumerate(order)}
    manifest_rows = {fold: [] for fold in range(N_FOLDS)}
    csv_rows = {fold: [] for fold in range(N_FOLDS)}
    sequences, stems, specs = {}, {}, {}
    for accession in order:
        spec = specs[accession] = _spec_defaults(proteins[accession])
        stem = stems[accession] = f"{accession}__chain_A__EC_{spec['ec']}"
        source = staging / f"{stem}.pdb"
        write_structure(source, spec["residues"], spec["metals"])
        data_bytes = source.read_bytes()
        digest = _sha(data_bytes)
        target = store / digest[:2] / digest / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sequences[accession] = _write_esm(source, esm_dir, stem,
                                          source_path=spec.get("esm_source_path", f"/synthetic/structures/{stem}.pdb"))
        _write_external(source, external, stem, pka=spec["pka"])
        for fold in range(N_FOLDS):
            if fold == test_fold[accession]:
                continue
            train_dir = data / f"CLEAN_30_train_test_split_{fold}" / "train"
            manifest_rows[fold].append({"structure_name": source.name, "relative_path": os.path.relpath(target, train_dir),
                                        "sha256": digest, "size_bytes": len(data_bytes)})
            for resseq, metaltype, coord, catalytic in spec["sites"]:
                csv_rows[fold].append({
                    "structure": accession, "chain_resi": f"A_{resseq}", "metaltype": metaltype, "ecnumber": spec["ec"],
                    "whether_catalytic": catalytic, "uniprot_id": accession, "clean_identity": "30",
                    "clean_fold": str(fold), DONOR_COLUMN: "1abc",
                    **{column: f"{value:.3f}" for column, value in zip(COORDINATE_COLUMNS, coord)}})
    for fold in range(N_FOLDS):
        _write_csv(data / f"CLEAN_30_train_test_split_{fold}" / "train" / "structure_manifest.csv",
                   ("structure_name", "relative_path", "sha256", "size_bytes"), manifest_rows[fold])
        _write_csv(data / "CLEAN_30_main" / "folds" / f"CLEAN_30_train_test_split_{fold}_train.csv", TRAIN_COLUMNS,
                   csv_rows[fold])
    shutil.rmtree(staging)

    labels = base / "labels"
    map_rows, label_rows = [], []
    extras = {"Q0001": ("1.1.1.9", "MKV"), "Q0002": ("3.4.1.1", "MAAG")}
    entries = {accession: (specs[accession]["ec"], specs[accession].get("map_sequence", sequences[accession]))
               for accession in order}
    entries.update(extras)
    for accession in sorted(entries):
        ec, sequence = entries[accession]
        native = (label_overrides or {}).get(accession, normalize_ec_number_list(ec))
        result = classify_ec_class(native.split(";"))
        sequence_sha = _sha(sequence.encode())
        map_rows.append({"accession": accession, "sha256": sequence_sha, "length": len(sequence), "n_files": 4,
                         "sequence": sequence})
        label_rows.append({"accession": accession, "ec_numbers": native, "ec_class": result.token or "",
                           "ec_class_status": result.status, "n_files": 4, "sequence_sha256": sequence_sha})
    label_bytes = _write_csv(labels / "native_label_table.tsv", LABEL_COLUMNS, label_rows, delimiter="\t")
    map_bytes = _write_csv(labels / "sequence_map.tsv", MAP_COLUMNS, map_rows, delimiter="\t")
    return {"base": base, "data_root": data, "esm_dir": esm_dir, "external": external, "stems": stems,
            "specs": specs, "test_fold": test_fold, "sequences": sequences,
            "label_table": labels / "native_label_table.tsv", "label_sha": _sha(label_bytes),
            "sequence_map": labels / "sequence_map.tsv", "map_sha": _sha(map_bytes)}


def _copy_cohort(paths: dict, dest: Path) -> dict:
    """A copy of a built cohort (manifests hold relative paths, so the copy is self-contained)."""
    shutil.copytree(paths["base"], dest, symlinks=True)
    moved = {key: dest / Path(paths[key]).relative_to(paths["base"])
             for key in ("data_root", "esm_dir", "external", "label_table", "sequence_map")}
    return {**paths, **moved, "base": dest}


def main_proteins() -> dict[str, dict]:
    """Twelve single-class proteins (EC classes 1-3) and one protein with two EC classes.

    P1001 holds an unselected FE 6 Å from its selected ZN, P2001 an unselected ZN copy 1.5 Å away,
    P3001's map sequence differs at position 2 (D in the structure, N in the map), and P3002's
    payload names a test folder in its source path.
    """
    proteins = {}
    for class_index, ec_class in enumerate(("1", "2", "3")):
        for k in range(4):
            proteins[f"P{class_index + 1}00{k}"] = {"ec": f"{ec_class}.1.1.{k + 1}"}
    proteins["P1001"]["metals"] = [(900, "ZN", (0.0, 0.0, 0.0)), (901, "FE", (6.0, 0.0, 0.0))]
    proteins["P2001"]["metals"] = [(900, "ZN", (0.0, 0.0, 0.0)), (901, "ZN", (1.5, 0.0, 0.0))]
    proteins["P3001"]["map_sequence"] = "HNECKARS"
    proteins["P3002"]["esm_source_path"] = "/synthetic/CLEAN_30_train_test_split_0/test/P3002.pdb"
    proteins[AMBIGUOUS] = {"ec": "2.5.1.3,4.1.99.17"}
    return proteins


EXPECTED_DIFFERENCE = "P3001:N2D"


def _run(paths: dict, out: Path, **kwargs) -> dict:
    options = {"expected_sequence_differences": [EXPECTED_DIFFERENCE], "provenance": PROVENANCE, **kwargs}
    return run_cohort_audit(layout=SourceLayout.from_data_root(paths["data_root"]), out_dir=out,
                            label_table=paths["label_table"], label_table_sha256=paths["label_sha"],
                            sequence_map=paths["sequence_map"], sequence_map_sha256=paths["map_sha"], **options)


def _read_tsv(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _rows_by_accession(path: Path) -> dict[str, dict[str, str]]:
    return {row["accession"]: row for row in _read_tsv(path)}


def _rewrite_csv(path: Path, mutate, *, delimiter: str = ",") -> bytes:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields, rows = list(reader.fieldnames or []), list(reader)
    mutate(rows)
    return _write_csv(path, fields, rows, delimiter=delimiter)


def _expected_file(directory: Path, values) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "expected_label_eligible.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path, _sha(path.read_bytes())


@pytest.fixture(scope="module")
def main_cohort(tmp_path_factory):
    return build_cohort(tmp_path_factory.mktemp("cohort_source"), main_proteins())


@pytest.fixture(scope="module")
def main_audit(main_cohort, tmp_path_factory):
    out = _out(tmp_path_factory.mktemp("campaign"))
    summary = _run(main_cohort, out)
    return {"summary": summary, "out": out, "receipt": json.loads((out / RECEIPT_NAME).read_text())}


def _expected_eligible(paths: dict) -> list[int]:
    return [sum(1 for accession, fold in paths["test_fold"].items() if fold != k and accession != AMBIGUOUS)
            for k in range(N_FOLDS)]


# ---------------------------------------------------------------------------
# The trainer's own loader on a fold's training inputs
# ---------------------------------------------------------------------------

def _trainer_inputs(paths: dict, work: Path, accessions) -> Path:
    for name in ("structures", "esm", "external", "ring"):
        (work / name).mkdir(parents=True)
    for accession in accessions:
        stem = paths["stems"][accession]
        shutil.copyfile(next((paths["data_root"] / "structure_store").rglob(f"{stem}.pdb")),
                        work / "structures" / f"{stem}.pdb")
        for suffix in ("_chain_A_esmc.pt", "_chain_A_esmc.pt.json"):
            source = paths["esm_dir"] / f"{stem}{suffix}"
            if source.is_file():
                shutil.copyfile(source, work / "esm" / f"{stem}{suffix}")
        shutil.copytree(paths["external"] / stem, work / "external" / stem)
    return work


def _trainer_load(paths: dict, inputs: Path, *, fold: int = 0):
    from training.data import load_labeled_pockets_with_report_from_dir

    return load_labeled_pockets_with_report_from_dir(
        inputs / "structures",
        summary_csv=paths["data_root"] / "CLEAN_30_main" / "folds" / f"CLEAN_30_train_test_split_{fold}_train.csv",
        esm_dim=ESM_DIM, esm_embeddings_dir=inputs / "esm", require_esm_embeddings=True,
        ring_features_dir=inputs / "ring", external_features_root_dir=inputs / "external",
        external_feature_source="auto", require_external_features=True, required_targets=("ec",),
        ec_label_depth=1, ec_class_vocabulary="fixed_seven", site_extraction="selected_sites", load_workers=1)


def _trainer_outcomes(paths: dict, work: Path, *, fold: int = 0) -> tuple[dict[str, str], dict]:
    """Per accession of fold ``fold``'s training side: ``ok``, a skip reason or ``invalid_structure``."""
    train = sorted(accession for accession, test in paths["test_fold"].items() if test != fold)
    result = _trainer_load(paths, _trainer_inputs(paths, work, train), fold=fold)
    report = result.feature_report
    by_stem = {pocket.structure_id: "ok" for pocket in result.pockets}
    for row in report["skipped_pockets"]:
        by_stem.setdefault(row["structure_id"], row["reason"])
    for row in report["invalid_structures"]:
        by_stem[Path(row["structure_path"]).stem] = "invalid_structure"
    accessions = {stem: accession for accession, stem in paths["stems"].items()}
    assert set(by_stem) == {paths["stems"][accession] for accession in train}
    return {accessions[stem]: outcome for stem, outcome in by_stem.items()}, report


# ---------------------------------------------------------------------------
# A passing audit
# ---------------------------------------------------------------------------

def test_clean_cohort_passes_and_writes_every_output(main_cohort, main_audit):
    summary, out, receipt = main_audit["summary"], main_audit["out"], main_audit["receipt"]
    assert summary["status"] == "passed", summary["failed_checks"]
    assert receipt["status"] == "passed" and receipt["failed_checks"] == []
    assert summary["n_proteins"] == 13 and summary["n_selected_sites"] == 13
    assert summary["load_status_counts"]["ok"] == 13
    assert all(value is not False for value in summary["checks"].values())
    assert summary["checks"]["charge_proxy_parity_complete"] is True
    assert summary["checks_not_run"] == receipt["checks_not_run"] == ["label_eligible_as_expected"]
    for name in (PROTEINS_NAME, SITES_NAME, D2_METALS_NAME, D2_RESIDUES_NAME, RECEIPT_NAME,
                 *(f"cohort_counts_fold{k}.json" for k in range(N_FOLDS))):
        assert (out / name).is_file(), name
    for name in (QUARANTINE_NAME, SITE_FAILURES_NAME, *(f"cohort_exclusions_fold{k}.tsv" for k in range(N_FOLDS))):
        assert not (out / name).exists(), name
    assert not any(path.name.startswith(".") for path in (*out.iterdir(), *out.parent.iterdir()))
    for name, record in receipt["outputs"].items():
        assert _sha((out / name).read_bytes()) == record["sha256"]
    for fold, item in receipt["per_fold"].items():
        assert _sha((out / item["file"]).read_bytes()) == item["sha256"]

    proteins = _rows_by_accession(out / PROTEINS_NAME)
    assert set(proteins) == set(main_cohort["stems"])
    assert all(row["n_training_files"] == "4" for row in proteins.values())
    assert all(row["esm_status"] == "ok" and row["external_status"] == "ok" for row in proteins.values())
    assert proteins["P3001"]["sequence_map_match"] == "0" and proteins["P3001"]["sequence_map_difference"] == "N2D"
    assert proteins["P1000"]["structure_sha256"] == _sha(
        next((main_cohort["data_root"] / "structure_store").rglob(main_cohort["stems"]["P1000"] + ".pdb")).read_bytes())
    sites = _read_tsv(out / SITES_NAME, delimiter=",")
    assert list(sites[0]) == ["accession", "chain_resi", "metaltype"]
    assert {(row["accession"], row["chain_resi"], row["metaltype"]) for row in sites} == {
        (accession, "A_900", "ZN") for accession in main_cohort["stems"]}


def test_per_fold_counts_cover_training_proteins_only(main_cohort, main_audit):
    out, receipt = main_audit["out"], main_audit["receipt"]
    expected = _expected_eligible(main_cohort)
    for fold in range(N_FOLDS):
        counts = json.loads((out / f"cohort_counts_fold{fold}.json").read_text())
        train = [accession for accession, test in main_cohort["test_fold"].items() if test != fold]
        assert counts["fold"] == fold and counts["n_proteins"] == len(train)
        assert counts["n_label_eligible"] == expected[fold]
        assert counts["trainer_status_counts"]["ambiguous_ec_class"] == int(AMBIGUOUS in train)
        assert counts["trainer_status_counts"]["ok"] == expected[fold]
        assert counts["n_load_time_exclusions"] == 0 and counts["exclusions_file"] is None
        assert counts["d2"]["n_proteins"] == expected[fold]
        assert "expected_label_eligible" not in counts
        assert receipt["per_fold"][str(fold)]["no_load_time_exclusions"] is True


def test_outputs_hold_no_labels_structure_names_or_class_breakdowns(main_cohort, main_audit):
    out = main_audit["out"]
    ec_strings = {spec["ec"] for spec in main_cohort["specs"].values()} | {"2.5.1.3", "4.1.99.17"}
    for path in out.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "__EC_" not in text, path.name
        assert not any(ec in text for ec in ec_strings), path.name
    for name in (PROTEINS_NAME, D2_METALS_NAME, D2_RESIDUES_NAME):
        header = (out / name).read_text().splitlines()[0].split("\t")
        assert not any("ec" == column or column.startswith("ec_") or "fold" in column for column in header), name

    def walk(value, keys=()):
        if isinstance(value, dict):
            # A dict keyed by EC class tokens (fold-keyed dicts start at "0", EC classes at "1").
            assert not ({"1", "2", "3"} <= set(value) and "0" not in value), keys
            for key, item in value.items():
                assert not any(part in key for part in ("by_ec_class", "ec_class_counts", "proportion")), key
                walk(item, (*keys, key))
        elif isinstance(value, list):
            for item in value:
                walk(item, keys)

    walk(main_audit["receipt"])
    for fold in range(N_FOLDS):
        walk(json.loads((out / f"cohort_counts_fold{fold}.json").read_text()))
    assert set(main_audit["receipt"]["per_fold"]["0"]) == {
        "file", "sha256", "a1_option_b_criteria_met", "charge_proxy_parity_complete",
        "label_eligible_matches_expected", "no_load_time_exclusions"}


def test_two_runs_write_identical_tables(main_cohort, main_audit, tmp_path):
    again = _out(tmp_path)
    _run(main_cohort, again)
    for name in (PROTEINS_NAME, SITES_NAME, D2_METALS_NAME, D2_RESIDUES_NAME,
                 *(f"cohort_counts_fold{k}.json" for k in range(N_FOLDS))):
        assert (again / name).read_bytes() == (main_audit["out"] / name).read_bytes(), name


def test_label_free_site_table_reads_with_the_test_pass_reader(main_audit):
    from training.ec_prediction_export import read_test_site_table

    sites = read_test_site_table(main_audit["out"] / SITES_NAME)
    assert len(sites) == 13 and set(sites.values()) == {"ZN"}


# ---------------------------------------------------------------------------
# Expected per-fold counts: a private file, never the command line
# ---------------------------------------------------------------------------

def test_expected_label_eligible_counts_come_from_a_private_file(main_cohort, tmp_path):
    values = _expected_eligible(main_cohort)
    good_file, good_sha = _expected_file(tmp_path / "private_good", values)
    good = _run(main_cohort, _out(tmp_path / "good"), expected_label_eligible_file=good_file,
                expected_label_eligible_sha256=good_sha)
    assert good["status"] == "passed" and good["label_eligible_as_expected"] is True
    assert good["checks_not_run"] == []
    receipt = json.loads((_out(tmp_path / "good") / RECEIPT_NAME).read_text())
    assert receipt["inputs"]["expected_label_eligible_file"] == {"path": str(good_file), "sha256": good_sha}
    assert "expected_label_eligible" not in receipt["inputs"]
    assert json.dumps(values) not in json.dumps(receipt) and json.dumps(values) not in json.dumps(good)
    assert all(item["label_eligible_matches_expected"] is True for item in receipt["per_fold"].values())

    wrong_file, wrong_sha = _expected_file(tmp_path / "private_wrong", [value + 1 for value in values])
    bad = _run(main_cohort, _out(tmp_path / "bad"), expected_label_eligible_file=wrong_file,
               expected_label_eligible_sha256=wrong_sha)
    assert bad["status"] == "stopped" and bad["failed_checks"] == ["label_eligible_as_expected"]


@pytest.mark.parametrize("contents", [[1, 2], [1, 2, 3, 4, -5], [1, 2, 3, 4, True], {"0": 1}, "1,2,3,4,5"])
def test_a_malformed_expected_counts_file_is_refused_without_its_contents(main_cohort, tmp_path, contents):
    path, digest = _expected_file(tmp_path / "private", contents)
    with pytest.raises(CohortAuditError, match="JSON list") as caught:
        _run(main_cohort, _out(tmp_path), expected_label_eligible_file=path, expected_label_eligible_sha256=digest)
    assert json.dumps(contents) not in str(caught.value)
    assert not _out(tmp_path).exists()


def test_the_expected_counts_file_is_checked_and_kept_out_of_the_repository(main_cohort, tmp_path):
    path, digest = _expected_file(tmp_path / "private", _expected_eligible(main_cohort))
    with pytest.raises(CohortAuditError, match="SHA-256"):
        _run(main_cohort, _out(tmp_path), expected_label_eligible_file=path, expected_label_eligible_sha256="0" * 64)
    with pytest.raises(CohortAuditError, match="both or neither"):
        _run(main_cohort, _out(tmp_path), expected_label_eligible_file=path)
    with pytest.raises(CohortAuditError, match="both or neither"):
        _run(main_cohort, _out(tmp_path), expected_label_eligible_sha256=digest)
    inside = ROOT / "src" / "expected_label_eligible_refused.json"
    with pytest.raises(CohortAuditError, match="repository"):
        _run(main_cohort, _out(tmp_path), expected_label_eligible_file=inside, expected_label_eligible_sha256=digest)
    assert not inside.exists() and not _out(tmp_path).exists()


# ---------------------------------------------------------------------------
# D2 geometry and the charge proxy
# ---------------------------------------------------------------------------

def test_same_site_copy_and_unselected_metal_distances(main_audit):
    metals = {row["accession"]: row for row in _read_tsv(main_audit["out"] / D2_METALS_NAME)}
    assert set(metals) == {"P1001", "P2001"}
    copy, iron = metals["P2001"], metals["P1001"]
    assert copy["element"] == "ZN" and copy["same_site_copy"] == "1"
    assert float(copy["nearest_selected_distance"]) == pytest.approx(1.5, abs=1e-5)
    assert iron["element"] == "FE" and iron["same_site_copy"] == "0" and iron["chain_resi"] == "A_901"
    assert float(iron["nearest_selected_distance"]) == pytest.approx(6.0, abs=1e-5)
    assert iron["within_pocket_radius"] == "1"
    d2 = main_audit["receipt"]["totals"]["d2_over_loaded_proteins"]
    assert d2["n_unselected_metals"] == 2 and d2["n_same_site_copies"] == 1
    proteins = _rows_by_accession(main_audit["out"] / PROTEINS_NAME)
    assert proteins["P2001"]["metal_records_dropped"] == "1" and proteins["P2001"]["n_same_site_copies"] == "1"
    assert proteins["P2001"]["components_with_unselected_metals_count"] == "1"
    assert proteins["P1001"]["components_with_unselected_metals_count"] == "0"
    assert proteins["P1001"]["propka_sees_unselected_metals"] == "1"
    assert proteins["P1000"]["propka_sees_unselected_metals"] == "0"


def test_charge_proxy_share_equals_the_feature_module_with_and_without_unselected_metals(main_cohort, main_audit,
                                                                                         tmp_path):
    from feature_extraction.core import generate_feature_map_for_structure

    stem = main_cohort["stems"]["P1001"]
    full = next((main_cohort["data_root"] / "structure_store").rglob(f"{stem}.pdb"))
    selected_only = tmp_path / f"{stem}.pdb"
    selected_only.write_text("".join(line for line in full.read_text().splitlines(keepends=True)
                                     if not (line.startswith("HETATM") and line[17:20].strip() == "FE")))
    with_all, _meta = generate_feature_map_for_structure(full, include_propka=False)
    without, _meta = generate_feature_map_for_structure(selected_only, include_propka=False)
    rows = [row for row in _read_tsv(main_audit["out"] / D2_RESIDUES_NAME) if row["accession"] == "P1001"]
    assert rows and all(row["parity_ok"] == "1" for row in rows)
    contributing = 0
    for row in rows:
        key = (row["chain"], int(row["resseq"]), row["icode"])
        expected = (with_all[key]["custom_charge_distance_proxy"] - without[key]["custom_charge_distance_proxy"])
        assert float(row["unselected_contribution"]) == pytest.approx(expected, abs=1e-9)
        assert float(row["recomputed_proxy"]) == pytest.approx(with_all[key]["custom_charge_distance_proxy"], abs=1e-12)
        assert float(row["stored_proxy"]) == pytest.approx(with_all[key]["custom_charge_distance_proxy"], abs=1e-12)
        share = float(row["unselected_share"])
        if float(row["unselected_contribution"]) != 0.0:
            contributing += 1
            total = float(row["recomputed_proxy"])
            assert share == pytest.approx(min(1.0, abs(expected) / abs(total)) if total else 1.0)
        else:
            assert share == 0.0
    assert contributing > 0
    other = [row for row in _read_tsv(main_audit["out"] / D2_RESIDUES_NAME) if row["accession"] == "P1000"]
    assert other and all(float(row["unselected_contribution"]) == 0.0 for row in other)


def test_a1_numbers_per_fold_and_the_recommended_option(main_cohort, main_audit):
    receipt = main_audit["receipt"]
    for fold in range(N_FOLDS):
        counts = json.loads((main_audit["out"] / f"cohort_counts_fold{fold}.json").read_text())
        d2 = counts["d2"]
        train = {accession for accession, test in main_cohort["test_fold"].items() if test != fold}
        assert d2["n_same_site_copies"] == int("P2001" in train)
        parity = d2["charge_proxy_parity"]
        assert parity["n_equal"] == parity["n_compared"] == d2["n_pocket_residues"] and parity["complete"] is True
        assert counts["a1"]["charge_proxy_parity_complete"] is True
        # Only P1001 (FE) and P2001 (ZN copy) hold unselected metals.
        assert (d2["n_pocket_residues_with_unselected_contribution"] > 0) == bool({"P1001", "P2001"} & train)
        fraction = d2["fraction_with_unselected_contribution"]
        median = d2["median_share_among_them"] or 0.0
        assert counts["a1"]["option_b_criteria_met"] == (fraction < 0.10 and median < 0.10)
        assert receipt["per_fold"][str(fold)]["a1_option_b_criteria_met"] == counts["a1"]["option_b_criteria_met"]
    flags = [item["a1_option_b_criteria_met"] for item in receipt["per_fold"].values()]
    assert None not in flags
    assert receipt["a1_recommended_option"] == ("b" if all(flags) else "d")


def test_unselected_share_rule():
    assert cohort.unselected_share(0.0, 0.0) == 0.0
    assert cohort.unselected_share(0.5, 0.0) == 1.0
    assert cohort.unselected_share(-0.25, 1.0) == 0.25
    assert cohort.unselected_share(2.0, -1.0) == 1.0


def test_recommended_a1_option_rule():
    assert cohort.recommended_a1_option([True] * N_FOLDS) == "b"
    assert cohort.recommended_a1_option([True, False, True, True, True]) == "d"
    assert cohort.recommended_a1_option([True, None, True, True, True]) is None
    assert cohort.recommended_a1_option([False, None, True, True, True]) is None


def test_d2_fraction_counts_recomputed_residues_and_parity_gates_a1():
    def residue(contribution, share, parity):
        return {"unselected_contribution": contribution, "unselected_share": share, "parity_ok": parity}

    def record(residues):
        return {"residues": residues, "metals": [], "missing_feature_values": Counter(), "propka_flag": False}

    complete = cohort.d2_numbers([record([residue(0.0, 0.0, True), residue(0.5, 0.05, True)])])
    assert complete["fraction_with_unselected_contribution"] == 0.5
    assert complete["charge_proxy_parity"] == {"n_compared": 2, "n_equal": 2, "n_not_compared": 0, "complete": True}
    assert complete["a1_option_b_criteria_met"] is False
    small = cohort.d2_numbers([record([residue(0.0, 0.0, True)] * 19 + [residue(0.5, 0.05, True)])])
    assert small["a1_option_b_criteria_met"] is True
    # One residue not recomputed: the fraction counts the three recomputed ones, and A1 is undetermined.
    gap = cohort.d2_numbers([record([residue(None, None, None), residue(0.5, 0.05, None),
                                     residue(0.0, 0.0, True), residue(0.0, 0.0, True)])])
    assert gap["fraction_with_unselected_contribution"] == pytest.approx(1 / 3)
    assert gap["charge_proxy_parity"]["complete"] is False and gap["a1_option_b_criteria_met"] is None
    unequal = cohort.d2_numbers([record([residue(0.0, 0.0, True), residue(0.0, 0.0, False)])])
    assert unequal["charge_proxy_parity"]["complete"] is False and unequal["a1_option_b_criteria_met"] is None
    assert cohort.d2_numbers([])["charge_proxy_parity"]["complete"] is True


# ---------------------------------------------------------------------------
# Load statuses, mapped as the trainer maps them
# ---------------------------------------------------------------------------

def status_proteins() -> dict[str, dict]:
    proteins = {f"P{index}000": {"ec": f"{index}.1.1.1"} for index in (1, 2, 3)}
    proteins["P1001"] = {"ec": "1.2.1.1", "metals": [(900, "ZN", (0.0, 0.0, 0.0)), (901, "FE", (6.0, 0.0, 0.0))]}
    proteins[AMBIGUOUS] = {"ec": "2.5.1.3,4.1.99.17"}
    proteins["P4000"] = {"ec": "1.3.1.1", "sites": [(900, "ZN", (0.0, 0.0, 0.0), "0")]}
    proteins["P4001"] = {"ec": "1.4.1.1", "metals": [(900, "ZN", (100.0, 100.0, 100.0))]}
    proteins["P4002"] = {"ec": "1.5.1.1", "sites": [(950, "ZN", (0.0, 0.0, 0.0), "1")]}
    proteins["P4003"] = {"ec": "1.6.1.1", "residues": (*DEFAULT_RESIDUES, ("ALA", (0.0, -6.0, 3.0), "B"))}
    return proteins


STATUS_EXPECTED = {
    "P1000": ("ok", ""), "P1001": ("ok", ""), "P2000": ("ok", ""), "P3000": ("ok", ""), AMBIGUOUS: ("ok", ""),
    "P4000": ("no_selected_sites", ""), "P4001": ("selected_sites_without_pocket", ""),
    "P4002": ("invalid_structure", "unresolved_selected_site"),
    "P4003": ("invalid_structure", "esm_pocket_residue_missing"),
}


@pytest.fixture(scope="module")
def status_cohort(tmp_path_factory):
    folds = {accession: 4 for accession in status_proteins()}
    return build_cohort(tmp_path_factory.mktemp("status_source"), status_proteins(), test_folds=folds)


@pytest.fixture(scope="module")
def status_audit(status_cohort, tmp_path_factory):
    out = _out(tmp_path_factory.mktemp("campaign"))
    summary = _run(status_cohort, out, expected_sequence_differences=[])
    return {"summary": summary, "out": out, "receipt": json.loads((out / RECEIPT_NAME).read_text())}


def test_every_load_status_on_synthetic_structures(status_audit):
    summary, out = status_audit["summary"], status_audit["out"]
    proteins = _rows_by_accession(out / PROTEINS_NAME)
    assert {accession: (row["load_status"], row["load_reason"]) for accession, row in proteins.items()} == STATUS_EXPECTED
    assert summary["status"] == "stopped" and "all_structures_load" in summary["failed_checks"]
    assert "selected_sites_verified" in summary["failed_checks"]  # the unresolved site
    failures = _read_tsv(out / SITE_FAILURES_NAME)
    assert [(row["accession"], row["chain_resi"], row["reason"]) for row in failures] == [("P4002", "A_950", "unresolved")]
    counts = json.loads((out / "cohort_counts_fold0.json").read_text())
    assert counts["trainer_status_counts"] == {
        "ok": 4, "no_selected_sites": 1, "selected_sites_without_pocket": 1, "invalid_structure": 2,
        "fatal_load_error": 0, "structure_input_error": 0, "ambiguous_ec_class": 1}
    assert counts["n_load_time_exclusions"] == 4
    exclusions = _read_tsv(out / "cohort_exclusions_fold0.tsv")
    assert {row["accession"]: row["load_status"] for row in exclusions} == {
        "P4000": "no_selected_sites", "P4001": "selected_sites_without_pocket", "P4002": "invalid_structure",
        "P4003": "invalid_structure"}
    assert list(exclusions[0]) == ["accession", "chain", "load_status", "load_reason"]
    assert not (out / "cohort_exclusions_fold4.tsv").exists()  # fold 4 trains on none of them


def test_load_statuses_match_the_trainer_loader(status_cohort, status_audit, tmp_path, monkeypatch):
    """The trainer's own loader on fold 0's training inputs gives the same per-structure outcomes and totals."""
    from ec_clean30.cohort import trainer_status

    monkeypatch.delenv("DEEPMZYME_PARSE_CACHE_DIR", raising=False)
    trainer, report = _trainer_outcomes(status_cohort, tmp_path / "fold0_inputs")
    audit_records = _rows_by_accession(status_audit["out"] / PROTEINS_NAME)
    for accession, outcome in trainer.items():
        label = "ambiguous_ec_class" if accession == AMBIGUOUS else "ok"
        assert trainer_status(audit_records[accession]["load_status"], label) == outcome, accession
    counts = json.loads((status_audit["out"] / "cohort_counts_fold0.json").read_text())
    assert counts["site_extraction"] == report["site_extraction_summary"]


def test_unsupported_site_metal_is_a_fatal_load_outcome(tmp_path, monkeypatch):
    proteins = {"P1000": {"ec": "1.1.1.1"}, "P5000": {"ec": "1.1.1.2", "sites": [(900, "CA", (0.0, 0.0, 0.0), "1")]}}
    paths = build_cohort(tmp_path / "source", proteins, test_folds={"P1000": 4, "P5000": 4})
    out = _out(tmp_path)
    summary = _run(paths, out, expected_sequence_differences=[])
    row = _rows_by_accession(out / PROTEINS_NAME)["P5000"]
    assert (row["load_status"], row["load_reason"]) == ("fatal_load_error", "unsupported_metal_label")
    assert {"no_fatal_load_outcomes", "all_structures_load", "selected_sites_verified"} <= set(summary["failed_checks"])
    failure = _read_tsv(out / SITE_FAILURES_NAME)[0]
    assert (failure["accession"], failure["reason"]) == ("P5000", "element_mismatch")
    # The trainer stops on the same structure.
    monkeypatch.delenv("DEEPMZYME_PARSE_CACHE_DIR", raising=False)
    inputs = _trainer_inputs(paths, tmp_path / "inputs", sorted(paths["stems"]))
    with pytest.raises(ValueError):
        _trainer_load(paths, inputs)


# ---------------------------------------------------------------------------
# One corrupted input at a time
# ---------------------------------------------------------------------------

CASE_PROTEINS = {"P1000": {"ec": "1.1.1.1"}, "P1001": {"ec": "1.1.1.2"}, "P1002": {"ec": "1.1.1.3"}}
CASE_FOLDS = {"P1000": 4, "P1001": 4, "P1002": 0}  # fold 0 trains on P1000 and P1001, fold 4 on P1002 only
CASE = "P1001"


@pytest.fixture(scope="module")
def case_cohort(tmp_path_factory):
    return build_cohort(tmp_path_factory.mktemp("case_source"), CASE_PROTEINS, test_folds=CASE_FOLDS)


def _esm_path(paths: dict, suffix: str = "") -> Path:
    return paths["esm_dir"] / f"{paths['stems'][CASE]}_chain_A_esmc.pt{suffix}"


def _external_path(paths: dict) -> Path:
    return paths["external"] / paths["stems"][CASE] / "residue_features.json"


def _structure_path(paths: dict) -> Path:
    return next((paths["data_root"] / "structure_store").rglob(f"{paths['stems'][CASE]}.pdb"))


def _edit_json(path: Path, edit) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sidecar(**fields):
    def edit(paths):
        _edit_json(_esm_path(paths, ".json"), lambda sidecar: sidecar.update(fields))
        return paths
    return edit


def _payload(change, *, rewrite_sidecar: bool = False):
    def edit(paths):
        import torch
        from training.esm_feature_loading import embedding_metadata_from_payload, write_embedding_metadata_sidecar

        target = _esm_path(paths)
        payload = torch.load(target, weights_only=True)
        change(payload)
        torch.save(payload, target)
        if rewrite_sidecar:
            write_embedding_metadata_sidecar(target, embedding_metadata_from_payload(payload))
        return paths
    return edit


def _external(change):
    def edit(paths):
        _edit_json(_external_path(paths), change)
        return paths
    return edit


def _remove(locate):
    def edit(paths):
        locate(paths).unlink()
        return paths
    return edit


def _narrow_payload(payload):
    payload["embeddings"] = payload["embeddings"][:, :ESM_DIM - 1].contiguous()
    payload["metadata"]["embedding_dim"] = ESM_DIM - 1


def _repeat_last_residue(payload):
    import torch

    payload["residue_ids"] = [*payload["residue_ids"], payload["residue_ids"][-1]]
    payload["embeddings"] = torch.cat([payload["embeddings"], payload["embeddings"][-1:]], dim=0)


def _drop_last_row(payload):
    payload["embeddings"] = payload["embeddings"][:-1].contiguous()


def _far_residue(payload) -> dict:
    return next(entry for entry in payload["residues"] if entry["resseq"] == FAR_RESSEQ)


def _non_finite_far_value(payload):
    features = _far_residue(payload)["features"]
    features[next(iter(features))] = float("nan")


def _extra_residue_key(payload):
    payload["residues"].append({**_far_residue(payload), "resseq": 99})
    payload["n_residues"] += 1


def _legacy_feature(payload):
    payload["residues"][0]["features"]["fa_elec"] = 0.0


def _drop_from_map(paths):
    data = _rewrite_csv(paths["sequence_map"], lambda rows: rows.remove(next(row for row in rows
                                                                              if row["accession"] == CASE)),
                        delimiter="\t")
    return {**paths, "map_sha": _sha(data)}


def _change_structure_bytes(paths):
    path = _structure_path(paths)
    path.write_text(path.read_text(encoding="utf-8").replace("END\n", "REMARK 999 CHANGED\nEND\n"), encoding="utf-8")
    return paths


def _symlink_structure(paths):
    path = _structure_path(paths)
    copy = paths["base"] / "elsewhere" / path.name
    copy.parent.mkdir(parents=True)
    shutil.move(path, copy)
    path.symlink_to(copy)
    return paths


def _value(*keys):
    def read(receipt):
        value = receipt
        for key in keys:
            value = value[key]
        return value
    return read


STRUCTURE_INPUT_FAILED = ["all_structures_load", "esmc_inputs_valid", "external_inputs_valid",
                          "structure_sequence_equals_sidecar", "structures_match_manifest"]

INPUT_CASES = {
    # ESMC sidecar
    "sidecar_sequence_sha256": dict(
        edit=_sidecar(source_sequence_sha256="0" * 64),
        row={"load_status": "ok", "esm_status": "sidecar_invalid", "sidecar_sequence_match": "0"},
        failed=["esmc_inputs_valid", "structure_sequence_equals_sidecar"]),
    "sidecar_model": dict(
        edit=_sidecar(esm_model_name="esmc_600m"),
        row={"load_status": "ok", "esm_status": "sidecar_invalid", "sidecar_sequence_match": "1"},
        failed=["esmc_inputs_valid"]),
    "sidecar_width": dict(
        edit=_sidecar(embedding_dim=1152),
        row={"load_status": "ok", "esm_status": "sidecar_invalid", "sidecar_sequence_match": "1"},
        failed=["esmc_inputs_valid"]),
    "sidecar_missing": dict(
        edit=_remove(lambda paths: _esm_path(paths, ".json")),
        row={"load_status": "ok", "esm_status": "sidecar_missing", "sidecar_sequence_match": "0",
             "esm_sidecar_sha256": ""},
        failed=["esmc_inputs_valid", "structure_sequence_equals_sidecar"], trainer=True),
    # ESMC payload
    "payload_metadata_model": dict(
        edit=_payload(lambda payload: payload["metadata"].update(esm_model_name="esmc_600m")),
        row={"load_status": "ok", "esm_status": "payload_invalid", "sidecar_sequence_match": "1"},
        failed=["esmc_inputs_valid"], trainer=True),
    "payload_width": dict(
        edit=_payload(_narrow_payload, rewrite_sidecar=True),
        row={"load_status": "invalid_structure", "load_reason": "esm_width", "esm_status": "payload_invalid",
             "sidecar_sequence_match": "1"},
        failed=["all_structures_load", "esmc_inputs_valid"], trainer=True),
    "payload_duplicate_residue_id": dict(
        edit=_payload(_repeat_last_residue),
        row={"load_status": "ok", "esm_status": "payload_duplicate_residue_ids", "sidecar_sequence_match": "1"},
        failed=["esmc_inputs_valid"], trainer=True),
    "payload_rows_misaligned": dict(
        edit=_payload(_drop_last_row),
        row={"load_status": "invalid_structure", "load_reason": "esm_payload_misaligned",
             "esm_status": "payload_misaligned", "sidecar_sequence_match": "1"},
        failed=["all_structures_load", "esmc_inputs_valid"], trainer=True),
    "payload_non_finite": dict(
        edit=_payload(lambda payload: payload["embeddings"][0].__setitem__(0, float("nan"))),
        row={"load_status": "ok", "esm_status": "payload_invalid"},
        failed=["esmc_inputs_valid"], trainer=True),
    "payload_missing": dict(
        edit=_remove(_esm_path),
        row={"load_status": "invalid_structure", "load_reason": "esm_missing", "esm_status": "missing",
             "esm_sha256": "", "sidecar_sequence_match": "1"},
        failed=["all_structures_load", "esmc_inputs_valid"], trainer=True),
    # External residue features
    "external_schema_version": dict(
        edit=_external(lambda payload: payload.update(schema_version=2)),
        row={"load_status": "ok", "external_status": "invalid_schema"}, failed=["external_inputs_valid"]),
    "external_structure_id": dict(
        edit=_external(lambda payload: payload.update(structure_id="P1001__chain_A__EC_9.9.9.9")),
        row={"load_status": "ok", "external_status": "invalid_structure_id"}, failed=["external_inputs_valid"]),
    "external_n_residues": dict(
        edit=_external(lambda payload: payload.update(n_residues=payload["n_residues"] + 1)),
        row={"load_status": "ok", "external_status": "invalid_n_residues"}, failed=["external_inputs_valid"]),
    "external_non_finite": dict(
        edit=_external(_non_finite_far_value),
        row={"load_status": "ok", "external_status": "non_finite"}, failed=["external_inputs_valid"], trainer=True),
    "external_extra_residue_key": dict(
        edit=_external(_extra_residue_key),
        row={"load_status": "ok", "external_status": "residue_keys_differ"}, failed=["external_inputs_valid"]),
    "external_legacy_feature": dict(
        edit=_external(_legacy_feature),
        row={"load_status": "invalid_structure", "load_reason": "external_legacy_feature",
             "external_status": "legacy_feature"},
        failed=["all_structures_load", "external_inputs_valid"], trainer=True),
    "external_missing": dict(
        edit=_remove(_external_path),
        row={"load_status": "invalid_structure", "load_reason": "external_missing", "external_status": "missing"},
        failed=["all_structures_load", "external_inputs_valid"], trainer=True),
    "external_pka_not_propka": dict(
        edit=_external(lambda payload: payload["tooling"].update(pka="unavailable")),
        row={"load_status": "ok", "external_status": "ok", "pka_tooling": "unavailable",
             "propka_sees_unselected_metals": "0"},
        failed=[], receipt={_value("totals", "n_pka_not_propka"): 1}),
    # Sequences
    "missing_from_sequence_map": dict(
        edit=_drop_from_map,
        row={"load_status": "ok", "sequence_map_match": "", "sequence_map_difference": ""},
        failed=["sequence_map_differences_as_expected"],
        receipt={_value("totals", "sequences", "n_missing_from_map"): 1}),
    "x_residue": dict(
        spec={"ec": "1.1.1.2", "residues": (*DEFAULT_RESIDUES, ("UNK", (-25.0, 0.0, 0.0)))},
        row={"load_status": "ok", "n_x_residues": "1", "sequence_length": "9"}, failed=[],
        receipt={_value("totals", "sequences", "n_with_x_residues"): 1}),
    # Structures
    "structure_bytes_changed": dict(
        edit=_change_structure_bytes,
        row={"load_status": "structure_input_error", "load_reason": "structure_differs_from_manifest"},
        failed=STRUCTURE_INPUT_FAILED),
    "structure_symlink": dict(
        edit=_symlink_structure,
        row={"load_status": "structure_input_error", "load_reason": "structure_symlink_refused"},
        failed=STRUCTURE_INPUT_FAILED),
    # Selected sites
    "two_metal_records_at_one_site": dict(
        spec={"ec": "1.1.1.2", "metals": [(900, "ZN", (0.0, 0.0, 0.0), "ZN1"), (900, "ZN", (0.4, 0.0, 0.0), "ZN2")]},
        row={"load_status": "invalid_structure", "load_reason": "ambiguous_selected_site", "n_sites_verified": "0"},
        failed=["all_structures_load", "selected_sites_verified"], site_reason="ambiguous", trainer=True),
}


@pytest.mark.parametrize("case", sorted(INPUT_CASES))
def test_one_corrupted_input_gives_its_status_and_failed_checks(case, case_cohort, tmp_path, monkeypatch):
    spec = INPUT_CASES[case]
    if "spec" in spec:
        paths = build_cohort(tmp_path / "source", {**CASE_PROTEINS, CASE: spec["spec"]}, test_folds=CASE_FOLDS)
    else:
        paths = spec["edit"](_copy_cohort(case_cohort, tmp_path / "source"))
    out = _out(tmp_path / "audit")
    summary = _run(paths, out, expected_sequence_differences=[])
    rows = _rows_by_accession(out / PROTEINS_NAME)
    assert {column: rows[CASE][column] for column in spec["row"]} == spec["row"]
    for accession in sorted(set(CASE_PROTEINS) - {CASE}):
        assert (rows[accession]["load_status"], rows[accession]["esm_status"],
                rows[accession]["external_status"]) == ("ok", "ok", "ok"), accession
    assert summary["failed_checks"] == spec["failed"]
    assert summary["status"] == ("stopped" if spec["failed"] else "passed")
    receipt = json.loads((out / RECEIPT_NAME).read_text())
    for read, expected in spec.get("receipt", {}).items():
        assert read(receipt) == expected
    failures = _read_tsv(out / SITE_FAILURES_NAME) if (out / SITE_FAILURES_NAME).exists() else []
    assert [row["reason"] for row in failures] == ([spec["site_reason"]] if "site_reason" in spec else [])
    if spec.get("trainer"):
        monkeypatch.delenv("DEEPMZYME_PARSE_CACHE_DIR", raising=False)
        outcomes, _report = _trainer_outcomes(paths, tmp_path / "trainer")
        assert set(outcomes) == {"P1000", CASE}
        for accession, outcome in outcomes.items():
            assert cohort.trainer_status(rows[accession]["load_status"], "ok") == outcome, accession


def test_a_stored_charge_proxy_that_differs_stops_the_audit_and_leaves_a1_undetermined(case_cohort, tmp_path):
    paths = _copy_cohort(case_cohort, tmp_path / "source")
    out = _out(tmp_path / "audit")
    baseline = _run(case_cohort, _out(tmp_path / "baseline"), expected_sequence_differences=[])
    assert baseline["status"] == "passed" and baseline["a1_recommended_option"] in ("b", "d")
    pocket = [row for row in _read_tsv(_out(tmp_path / "baseline") / D2_RESIDUES_NAME) if row["accession"] == CASE]
    first = (pocket[0]["chain"], int(pocket[0]["resseq"]), pocket[0]["icode"])

    def perturb(payload):
        entry = next(item for item in payload["residues"]
                     if (item["chain_id"], item["resseq"], item["icode"].strip()) == first)
        entry["features"]["custom_charge_distance_proxy"] += 1e-4

    _external(perturb)(paths)
    summary = _run(paths, out, expected_sequence_differences=[])
    assert summary["status"] == "stopped" and summary["failed_checks"] == ["charge_proxy_parity_complete"]
    parity = summary["d2_over_loaded_proteins"]["charge_proxy_parity"]
    assert parity["n_compared"] == parity["n_equal"] + 1 and parity["complete"] is False
    residues = [row for row in _read_tsv(out / D2_RESIDUES_NAME) if row["accession"] == CASE]
    assert [row["parity_ok"] for row in residues].count("0") == 1
    receipt = json.loads((out / RECEIPT_NAME).read_text())
    for fold in range(N_FOLDS):
        counts = json.loads((out / f"cohort_counts_fold{fold}.json").read_text())
        trains_case = CASE_FOLDS[CASE] != fold
        assert counts["a1"]["charge_proxy_parity_complete"] is (not trains_case)
        assert (counts["a1"]["option_b_criteria_met"] is None) is trains_case
        assert receipt["per_fold"][str(fold)]["charge_proxy_parity_complete"] is (not trains_case)
    assert receipt["a1_recommended_option"] is None and summary["a1_recommended_option"] is None


# ---------------------------------------------------------------------------
# Selected-site checks (A3)
# ---------------------------------------------------------------------------

def test_site_tolerance_just_inside_and_outside_and_element_mismatch(tmp_path):
    proteins = {
        "P1000": {"ec": "1.1.1.1", "sites": [(900, "ZN", (0.009, 0.0, 0.0), "1")]},
        "P1001": {"ec": "1.1.1.2", "sites": [(900, "ZN", (0.0, 0.0, 0.011), "1")]},
        "P1002": {"ec": "1.1.1.3", "sites": [(900, "FE", (0.0, 0.0, 0.0), "1")]},
        "P1003": {"ec": "1.1.1.4"},
    }
    paths = build_cohort(tmp_path / "source", proteins)
    out = _out(tmp_path)
    summary = _run(paths, out, expected_sequence_differences=[])
    failures = {row["accession"]: row for row in _read_tsv(out / SITE_FAILURES_NAME)}
    assert set(failures) == {"P1001", "P1002"}
    assert failures["P1001"]["reason"] == "outside_tolerance"
    assert float(failures["P1001"]["distance"]) == pytest.approx(0.011, abs=1e-6)
    assert failures["P1002"]["reason"] == "element_mismatch"
    assert summary["status"] == "stopped" and summary["failed_checks"] == ["selected_sites_verified"]
    receipt = json.loads((out / RECEIPT_NAME).read_text())
    check = receipt["totals"]["selected_site_check"]
    assert check["n_failed"] == 2 and check["failures_by_reason"] == {"element_mismatch": 1, "outside_tolerance": 1}
    assert check["failed_distances"]["max"] == pytest.approx(0.011, abs=1e-6)
    proteins_table = _rows_by_accession(out / PROTEINS_NAME)
    assert proteins_table["P1000"]["n_sites_verified"] == "1"
    assert float(proteins_table["P1000"]["max_site_distance"]) == pytest.approx(0.009, abs=1e-6)
    # The trainer still loads the element-mismatch site (its label comes from the site table).
    assert proteins_table["P1002"]["load_status"] == "ok"


def test_the_tolerance_cannot_be_loosened(main_cohort, tmp_path):
    with pytest.raises(CohortAuditError, match="tolerance"):
        _run(main_cohort, _out(tmp_path / "loose"), coordinate_tolerance=0.02)
    tighter = _run(main_cohort, _out(tmp_path / "tight"), coordinate_tolerance=0.005)
    assert tighter["status"] == "passed"


# ---------------------------------------------------------------------------
# Labels (A4), sequences and the A13 count
# ---------------------------------------------------------------------------

def test_an_ec_content_difference_is_quarantined_and_stops_after_the_receipt(tmp_path):
    proteins = {"P1000": {"ec": "1.1.1.1"}, "P1001": {"ec": "1.1.1.2"}, AMBIGUOUS: {"ec": "2.5.1.3,4.1.99.17"}}
    paths = build_cohort(tmp_path / "source", proteins,
                         label_overrides={"P1001": "1.1.1.7", AMBIGUOUS: "4.1.99.17;2.5.1.3"})
    out = _out(tmp_path)
    summary = _run(paths, out, expected_sequence_differences=[])
    assert summary["status"] == "stopped" and summary["failed_checks"] == ["no_ec_content_differences"]
    assert summary["a4"] == {"equal": 1, "order_only": 1, "different": 1, "missing_from_label_table": 0}
    assert _read_tsv(out / QUARANTINE_NAME) == [{"accession": "P1001", "a4_outcome": "different"}]
    receipt = json.loads((out / RECEIPT_NAME).read_text())
    assert receipt["status"] == "stopped" and QUARANTINE_NAME in receipt["outputs"]


def test_compare_ec_numbers():
    assert cohort.compare_ec_numbers("1.1.1.1", "1.1.1.1") == "equal"
    assert cohort.compare_ec_numbers("2.5.1.3,4.1.99.17", "2.5.1.3;4.1.99.17") == "equal"
    assert cohort.compare_ec_numbers("2.5.1.3,4.1.99.17", "4.1.99.17;2.5.1.3") == "order_only"
    assert cohort.compare_ec_numbers("1.1.1.1", "1.1.1.2") == "different"
    assert cohort.compare_ec_numbers("1.1.1.1", None) == "missing_from_label_table"


def test_a_sequence_map_difference_must_be_declared(main_cohort, tmp_path):
    undeclared = _run(main_cohort, _out(tmp_path / "undeclared"), expected_sequence_differences=[])
    assert undeclared["status"] == "stopped"
    assert undeclared["failed_checks"] == ["sequence_map_differences_as_expected"]
    assert undeclared["n_sequence_map_differences"] == 1
    other = _run(main_cohort, _out(tmp_path / "other"), expected_sequence_differences=["P3001:N2E"])
    assert other["failed_checks"] == ["sequence_map_differences_as_expected"]
    with pytest.raises(CohortAuditError):
        _run(main_cohort, _out(tmp_path / "malformed"), expected_sequence_differences=["P3001"])


def test_sequence_difference_formats():
    assert cohort._sequence_difference("HDECK", "HNECK") == "N2D"
    assert cohort._sequence_difference("HDECX", "HNECK") == "N2D;K5X"
    assert cohort._sequence_difference("HDEC", "HDECK") == "length:4vs5"


def test_test_named_strings_are_counted_not_shown(main_audit):
    receipt = main_audit["receipt"]
    # P3002's payload and its sidecar both carry the test-named source path; the external files do not.
    assert receipt["totals"]["a13_test_named_string_count"] == {"payload_and_sidecar": 2, "external": 0, "total": 2}
    assert main_audit["summary"]["a13_test_named_string_count"] == 2
    assert "/test/" not in json.dumps(receipt)
    assert cohort.count_test_named_strings({"a": ["x/test/y", "fine"], "b": {"c": "CLEAN_30_main/metadata/z"}}) == 2
    assert cohort.count_test_named_strings({"path": "/data/CLEAN_30_train_test_split_0/train/x.pdb"}) == 0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_symlinked_input_is_never_read_and_stops_the_audit(tmp_path):
    paths = build_cohort(tmp_path / "source", {"P1000": {"ec": "1.1.1.1"}, "P1001": {"ec": "1.1.1.2"}})
    target = paths["esm_dir"] / f"{paths['stems']['P1001']}_chain_A_esmc.pt"
    decoy = tmp_path / "held_out" / "test" / "decoy.pt"
    decoy.parent.mkdir(parents=True)
    shutil.move(target, decoy)
    os.chmod(decoy, 0)  # reading it would raise PermissionError
    target.symlink_to(decoy)
    out = _out(tmp_path)
    try:
        summary = _run(paths, out, expected_sequence_differences=[])
    finally:
        os.chmod(decoy, 0o644)
    row = _rows_by_accession(out / PROTEINS_NAME)["P1001"]
    assert row["esm_status"] == "symlink_refused" and row["esm_sha256"] == ""
    assert (row["load_status"], row["load_reason"]) == ("invalid_structure", "esm_symlink_refused")
    assert "esmc_inputs_valid" in summary["failed_checks"]


def test_test_named_paths_are_refused(main_cohort, tmp_path):
    with pytest.raises(HeldOutInputError):
        _run(main_cohort, tmp_path / CAMPAIGN_ID / "cohort_test")
    renamed = tmp_path / "labels_test.tsv"
    shutil.copyfile(main_cohort["label_table"], renamed)
    with pytest.raises(HeldOutInputError):
        _run({**main_cohort, "label_table": renamed}, _out(tmp_path))
    assert not _out(tmp_path).exists()


def test_inputs_must_match_their_sha256(main_cohort, tmp_path):
    with pytest.raises(CohortAuditError, match="SHA-256"):
        _run({**main_cohort, "label_sha": "0" * 64}, _out(tmp_path))
    with pytest.raises(CohortAuditError, match="SHA-256"):
        _run({**main_cohort, "map_sha": "0" * 64}, _out(tmp_path))


def test_the_output_folder_is_written_once_and_outside_shared_roots(main_cohort, main_audit, tmp_path):
    with pytest.raises(CohortAuditError, match="not empty"):
        _run(main_cohort, main_audit["out"])
    with pytest.raises(CohortAuditError, match="shared"):
        _run(main_cohort, _out(main_cohort["data_root"] / "esm_embeddings"))
    inside = ROOT / "src" / CAMPAIGN_ID
    with pytest.raises(CohortAuditError, match="repository"):
        _run(main_cohort, inside / "cohort")
    assert not inside.exists()
    assert issubclass(CohortAuditError, ViewError)


def test_the_output_folder_must_be_the_campaign_cohort_folder(main_cohort, tmp_path):
    campaign = tmp_path / CAMPAIGN_ID
    for refused in (campaign / "views" / "fold0" / "cohort", campaign / "runs" / "fold1" / "cohort",
                    campaign / "smoke_views" / "g1" / "cohort", campaign / "overlay" / "cohort",
                    campaign / "cohort" / "cohort", campaign / "cohort_rerun", tmp_path / "elsewhere" / "cohort"):
        with pytest.raises(CohortAuditError, match="campaign root") as caught:
            _run(main_cohort, refused)
        assert CAMPAIGN_ID in str(caught.value), refused
        assert not refused.exists(), refused
    assert not campaign.exists()


def test_staging_sits_inside_the_cohort_folder_and_is_removed_on_failure(main_cohort, tmp_path, monkeypatch):
    out = _out(tmp_path)
    seen: list[Path] = []
    original = cohort.write_new_json

    def failing(path, payload):
        if Path(path).name == RECEIPT_NAME:
            seen.append(Path(path))
            raise OSError("synthetic write failure")
        return original(path, payload)

    monkeypatch.setattr(cohort, "write_new_json", failing)
    with pytest.raises(OSError, match="synthetic"):
        _run(main_cohort, out)
    assert seen and seen[0].parent.parent == out and seen[0].parent.name.startswith(cohort.STAGING_PREFIX)
    assert not out.exists() and list(out.parent.iterdir()) == []


def test_a_leftover_staging_folder_refuses_a_rerun(main_cohort, tmp_path):
    out = _out(tmp_path)
    (out / f"{cohort.STAGING_PREFIX}12345").mkdir(parents=True)
    with pytest.raises(CohortAuditError, match="not empty"):
        _run(main_cohort, out)


@pytest.mark.parametrize("kind, table, change", [
    ("site_rows", "CLEAN_30_main/folds/CLEAN_30_train_test_split_1_train.csv",
     lambda rows: rows[0].__setitem__("single_donor_coordinate_x", "0.500")),
    ("structure_sha256", "CLEAN_30_train_test_split_1/train/structure_manifest.csv",
     lambda rows: rows[1].__setitem__("sha256", "f" * 64)),
    ("structure_name", "CLEAN_30_train_test_split_1/train/structure_manifest.csv",
     lambda rows: rows[1].__setitem__("structure_name", rows[1]["structure_name"][:-len(".pdb")] + ".cif")),
])
def test_a_cross_file_conflict_is_refused_without_naming_the_protein(case_cohort, tmp_path, kind, table, change):
    paths = _copy_cohort(case_cohort, tmp_path / "source")
    _rewrite_csv(paths["data_root"] / table, change)
    with pytest.raises(CohortAuditError) as caught:
        _run(paths, _out(tmp_path), expected_sequence_differences=[])
    assert f"1 {kind} conflicts" in str(caught.value)
    assert not any(accession in str(caught.value) for accession in CASE_PROTEINS)
    assert not _out(tmp_path).exists()


def _test_named_relative_path(rows):
    rows[0]["relative_path"] = rows[0]["relative_path"].replace("structure_store", "test/../structure_store")


@pytest.mark.parametrize("table, change, refusal", [
    ("CLEAN_30_main/folds/CLEAN_30_train_test_split_2_train.csv",
     lambda rows: rows[0].__setitem__("ecnumber", "1.9.9.9"), CohortAuditError),
    ("CLEAN_30_train_test_split_3/train/structure_manifest.csv", _test_named_relative_path, HeldOutInputError),
    ("CLEAN_30_train_test_split_1/train/structure_manifest.csv",
     lambda rows: rows[0].__setitem__("structure_name", "P1000.pdb"), CohortAuditError),
    ("CLEAN_30_train_test_split_1/train/structure_manifest.csv",
     lambda rows: rows[0].__setitem__("structure_name", rows[0]["structure_name"] + ".txt"), CohortAuditError),
    ("CLEAN_30_train_test_split_0/train/structure_manifest.csv", lambda rows: rows.append(dict(rows[0])),
     CohortAuditError),
], ids=["ec_mismatch", "test_named_relative_path", "unparsable_name", "bad_suffix", "duplicate_row"])
def test_a_training_source_refusal_is_shown_without_its_message(case_cohort, tmp_path, table, change, refusal):
    """Refusals of the reused training-source reader can quote a structure of another fold's file."""
    paths = _copy_cohort(case_cohort, tmp_path / "source")
    _rewrite_csv(paths["data_root"] / table, change)
    with pytest.raises(refusal) as caught:
        _run(paths, _out(tmp_path), expected_sequence_differences=[])
    message = str(caught.value)
    assert type(caught.value) is refusal
    assert "withheld" in message
    assert not any(accession in message for accession in CASE_PROTEINS)
    assert "__EC_" not in message and ".pdb" not in message and "1.1.1." not in message
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert not _out(tmp_path).exists()


def test_missing_coordinate_columns_are_refused(tmp_path):
    paths = build_cohort(tmp_path / "source", {"P1000": {"ec": "1.1.1.1"}, "P1001": {"ec": "1.1.1.2"}})
    for fold in range(N_FOLDS):
        table = paths["data_root"] / "CLEAN_30_main" / "folds" / f"CLEAN_30_train_test_split_{fold}_train.csv"
        rows = _read_tsv(table, delimiter=",")
        columns = [column for column in TRAIN_COLUMNS if column != "single_donor_coordinate_z"]
        _write_csv(table, columns, [{column: row[column] for column in columns} for row in rows])
    with pytest.raises(CohortAuditError, match="single_donor_coordinate_z"):
        _run(paths, _out(tmp_path), expected_sequence_differences=[])


def test_a_reader_refusal_is_shown_without_its_message(main_cohort, tmp_path):
    bad = tmp_path / "labels.tsv"
    bad.write_text("accession\tec_numbers\nbad accession\t1.1.1.1\n", encoding="utf-8")
    with pytest.raises(CohortAuditError) as caught:
        _run({**main_cohort, "label_table": bad, "label_sha": _sha(bad.read_bytes())}, _out(tmp_path))
    assert "withheld" in str(caught.value) and "bad accession" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_an_unforeseen_per_protein_failure_is_reported_without_the_protein(main_cohort, tmp_path, monkeypatch):
    def broken(protein, ctx):
        raise OSError(f"cannot read {protein.stem}")

    monkeypatch.setattr(cohort, "audit_protein", broken)
    with pytest.raises(CohortAuditError) as caught:
        _run(main_cohort, _out(tmp_path))
    assert "protein 1 of 13" in str(caught.value) and "__EC_" not in str(caught.value)
    assert caught.value.__cause__ is None and not _out(tmp_path).exists()


# ---------------------------------------------------------------------------
# Runner action, constants and imports
# ---------------------------------------------------------------------------

def _action_args(paths: dict, out: Path, *extra: str) -> list[str]:
    return ["--out", str(out), "--label-table", str(paths["label_table"]), "--label-table-sha256", paths["label_sha"],
            "--sequence-map", str(paths["sequence_map"]), "--sequence-map-sha256", paths["map_sha"],
            "--data-root", str(paths["data_root"]), *extra]


def test_runner_action_prints_totals_and_flags_only(main_cohort, tmp_path, capsys):
    parser = cohort.add_cohort_audit_arguments(argparse.ArgumentParser())
    values = _expected_eligible(main_cohort)
    expected_file, expected_sha = _expected_file(tmp_path / "private", values)
    out = _out(tmp_path / "passed")
    args = parser.parse_args(_action_args(
        main_cohort, out, "--expected-label-eligible-file", str(expected_file),
        "--expected-label-eligible-sha256", expected_sha, "--expected-sequence-difference", EXPECTED_DIFFERENCE,
        "--smoke"))
    assert cohort.action_cohort_audit(args) == 0
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["status"] == "passed" and printed["label_eligible_as_expected"] is True
    for accession in main_cohort["stems"]:
        assert accession not in captured.out and accession not in captured.err
    assert "__EC_" not in captured.out and json.dumps(values) not in captured.out
    receipt = json.loads((out / RECEIPT_NAME).read_text())
    assert receipt["smoke"] is True and receipt["provenance"]["smoke"] is True
    assert "[cohort-audit] 13/13 proteins" in captured.err
    stopped = parser.parse_args(_action_args(main_cohort, _out(tmp_path / "stopped"), "--smoke"))
    assert cohort.action_cohort_audit(stopped) == 1
    capsys.readouterr()


def test_the_counts_cannot_be_given_on_the_command_line(main_cohort, tmp_path, capsys):
    parser = cohort.add_cohort_audit_arguments(argparse.ArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args(_action_args(main_cohort, _out(tmp_path), "--expected-label-eligible", "1,2,3,4,5"))
    capsys.readouterr()


def test_constants_match_their_sources():
    from embed_helpers.esmc import DEFAULT_ESMC_MODEL_NAME
    from feature_extraction.constants import METAL_CHARGE_PROXIES
    from training.esm_feature_loading import DEFAULT_ESMC_EMBED_DIM

    assert cohort.ESMC_MODEL_NAME == DEFAULT_ESMC_MODEL_NAME
    assert cohort.ESMC_EMBEDDING_DIM == DEFAULT_ESMC_EMBED_DIM
    assert cohort.DONOR_COLUMN == DONOR_COLUMN
    assert set(METAL_CHARGE_PROXIES) == {"MN", "FE", "CO", "NI", "CU", "ZN"}


def test_cohort_module_imports_without_torch():
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
            "import ec_clean30.cohort; print('torch' in sys.modules)")
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"


G1_SMOKE_VIEW_ENV = "DEEPMZYME_EC_CLEAN30_G1_SMOKE_VIEW"


@pytest.mark.skipif(not os.environ.get(G1_SMOKE_VIEW_ENV), reason=f"real-data check; set {G1_SMOKE_VIEW_ENV} to the "
                    "G1 smoke view folder (main checkout only, not in worktrees)")
def test_g1_smoke_structures_reproduce_the_trainer_site_extraction_summary():
    """Real data (opt-in): the 32 G1 smoke structures give the trainer's G1 site-extraction totals."""
    from training.site_filter import SiteKeyIndex, load_allowed_site_metal_labels

    view = Path(os.environ[G1_SMOKE_VIEW_ENV])
    labels = load_allowed_site_metal_labels(view / "site_summary_train.csv")
    layout = SourceLayout(view, view / "esm", view / "external")
    ctx = cohort._Context(layout=layout, site_labels=labels, site_index=SiteKeyIndex(labels),
                          tolerance=cohort.A3_COORDINATE_TOLERANCE, esm_model_name=cohort.ESMC_MODEL_NAME,
                          esm_dim=cohort.ESMC_EMBEDDING_DIM, map_sequences={}, native_ec={})
    with (view / "site_summary_train.csv").open(encoding="utf-8", newline="") as handle:
        rows_by_accession: dict[str, list] = {}
        for row in csv.DictReader(handle):
            rows_by_accession.setdefault(row["structure"].strip(), []).append(row)
    records = []
    for row in _read_tsv(view / "structures" / "structure_manifest.csv", delimiter=","):
        stem = Path(row["structure_name"]).stem
        accession = stem.split("__")[0]
        protein = cohort.UnionProtein(
            accession=accession, stem=stem, chain="A", structure_name=row["structure_name"],
            path=view / "structures" / row["relative_path"], sha256=row["sha256"], size_bytes=int(row["size_bytes"]),
            folds=(0,), site_rows=tuple(rows_by_accession[accession]), label_status="ok")
        records.append(cohort.audit_protein(protein, ctx))
    totals = cohort.site_extraction_totals(records)
    assert totals["n_structures"] == 32
    assert (totals["metal_records_total"], totals["metal_records_selected"], totals["metal_records_dropped"]) == (
        118, 42, 76)
    assert totals["structures_with_unselected_metals"] == 16
    assert totals["structures_skipped_no_selected_sites"] == totals["structures_skipped_selected_sites_without_pocket"] == 0
