"""Synthetic data root for the EC class vs CLEAN30 campaign runner tests (no real data).

``write_clean30_data_root`` mimics the training-side layout the runner reads:

- ``CLEAN_30_train_test_split_<k>/train/structure_manifest.csv`` with relative paths into a
  content-addressed ``structure_store/objects/<aa>/<sha>/<name>``;
- ``CLEAN_30_main/folds/CLEAN_30_train_test_split_<k>_train.csv`` (training site rows);
- ``esm_embeddings/<stem>_chain_A_esmc.pt`` (+ sidecars) and
  ``updated_feature_extraction/<stem>/residue_features.json``.

Decoys that a view build must never open or copy are added and made unreadable: a test-side
manifest and site table, the mixed ``CLEAN_30_main/structures/structure_manifest.csv``, and
shared-cache files of a test-side protein and of a name that extends a training stem.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_fixtures import default_ec_proteins, write_ec_dataset  # noqa: E402

TRAIN_COLUMNS = ("structure", "chain_resi", "metaltype", "ecnumber", "whether_catalytic", "uniprot_id",
                 "clean_identity", "clean_fold", "single_donor_selected_alphafill_pdb_id")
ESM_DIM = 960
AMBIGUOUS_ACCESSION = "P9000"
TEST_ACCESSION = "Q9999"


def campaign_proteins() -> dict[str, dict]:
    """12 single-class proteins (EC classes 1-3) plus one protein annotated with two EC classes."""
    proteins = default_ec_proteins(n_per_class=4, classes=("1", "2", "3"))
    proteins[AMBIGUOUS_ACCESSION] = {"ec": "2.5.1.3,4.1.99.17", "ions": [("A", 900, "", "ZN", [0.0, 40.0, 0.0])]}
    return proteins


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_external_features(structure: Path, out_dir: Path) -> Path:
    """A ``residue_features.json`` with deterministic values for every polymer residue of chain A."""
    from embed_helpers.esmc import parse_structure
    from training.esm_feature_loading import residue_keys_for_structure_chain

    residue_ids, _with_ca = residue_keys_for_structure_chain(parse_structure(structure), "A")
    residues = []
    for index, (chain, resseq, icode) in enumerate(residue_ids):
        residues.append({"chain_id": chain, "resseq": resseq, "icode": icode,
                         "features": {"biotite_residue_sasa": 0.1 + 0.05 * (index % 7),
                                      "custom_charge_distance_proxy": 0.2 * ((index % 3) - 1),
                                      "dpka_titr": 0.01 * (index % 5)}})
    target = out_dir / structure.stem / "residue_features.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"residues": residues}), encoding="utf-8")
    return target


def write_clean30_data_root(base: Path, *, fold: int = 0, proteins: dict[str, dict] | None = None,
                            lock_decoys: bool = True) -> dict:
    """Write the synthetic data root; returns paths, stems and the decoy files."""
    proteins = dict(proteins or campaign_proteins())
    staging = base / "staging"
    decoy_proteins = {TEST_ACCESSION: {"ec": "3.4.24.17", "ions": [("A", 900, "", "ZN", [5.0, 5.0, 5.0])]}}
    paths = write_ec_dataset(staging, {**proteins, **decoy_proteins}, esm_dim=ESM_DIM)
    data = base / "DeepMzyme_Data"
    train_dir = data / f"CLEAN_30_train_test_split_{fold}" / "train"
    test_dir = data / f"CLEAN_30_train_test_split_{fold}" / "test"
    folds_dir = data / "CLEAN_30_main" / "folds"
    store = data / "structure_store" / "objects"
    esm_dir = data / "esm_embeddings"
    external = data / "updated_feature_extraction"
    for directory in (train_dir, test_dir, folds_dir, esm_dir, external, data / "CLEAN_30_main" / "structures"):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows = {"train": [], "test": []}
    stems = {}
    for accession, stem in sorted(paths["stems"].items()):
        source = paths["structures"] / f"{stem}.pdb"
        digest = _sha256(source)
        target = store / digest[:2] / digest / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        side = "test" if accession in decoy_proteins else "train"
        side_dir = test_dir if side == "test" else train_dir
        manifest_rows[side].append({"structure_name": source.name,
                                    "relative_path": os.path.relpath(target, side_dir),
                                    "sha256": digest, "size_bytes": target.stat().st_size})
        for name in (f"{stem}_chain_A_esmc.pt", f"{stem}_chain_A_esmc.pt.json"):
            shutil.copyfile(paths["esm"] / name, esm_dir / name)
        write_external_features(source, external)
        if side == "train":
            stems[accession] = stem
    for side, side_dir in (("train", train_dir), ("test", test_dir)):
        with (side_dir / "structure_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["structure_name", "relative_path", "sha256", "size_bytes"])
            writer.writeheader()
            writer.writerows(manifest_rows[side])

    with paths["summary"].open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    split_rows = {"train": [], "test": []}
    for row in summary_rows:
        side = "test" if row["structure"] in decoy_proteins else "train"
        split_rows[side].append({**row, "uniprot_id": row["structure"], "clean_identity": "30",
                                 "clean_fold": str(fold), "single_donor_selected_alphafill_pdb_id": "1abc"})
    for side in ("train", "test"):
        with (folds_dir / f"CLEAN_30_train_test_split_{fold}_{side}.csv").open("w", encoding="utf-8",
                                                                               newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TRAIN_COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(split_rows[side])

    # Decoys: a mixed manifest, and shared-cache files whose names extend a training stem.
    mixed = data / "CLEAN_30_main" / "structures" / "structure_manifest.csv"
    mixed.write_text("structure_name,relative_path,sha256,size_bytes\n", encoding="utf-8")
    first_stem = sorted(stems.values())[0]
    colliding = esm_dir / f"{first_stem}.9_chain_A_esmc.pt"
    shutil.copyfile(esm_dir / f"{first_stem}_chain_A_esmc.pt", colliding)
    locked = [test_dir, mixed, folds_dir / f"CLEAN_30_train_test_split_{fold}_test.csv"]
    if lock_decoys:
        for path in locked:
            os.chmod(path, 0)
    shutil.rmtree(staging)
    return {"data_root": data, "train_dir": train_dir, "esm_dir": esm_dir, "external": external,
            "training_csv": folds_dir / f"CLEAN_30_train_test_split_{fold}_train.csv",
            "train_manifest": train_dir / "structure_manifest.csv", "stems": stems,
            "decoys": {"locked": locked, "colliding_esm": colliding,
                       "test_stem": paths["stems"][TEST_ACCESSION]}}


def unlock(paths: dict) -> None:
    """Make the locked decoys readable again (so pytest can clean the folder)."""
    for path in paths["decoys"]["locked"]:
        if os.path.exists(path):
            os.chmod(path, 0o755 if os.path.isdir(path) else 0o644)


def make_writable(root: Path) -> None:
    """Undo a view's read-only permissions (to tamper with it in a test)."""
    for current, dirs, files in os.walk(root):
        os.chmod(current, 0o755)
        for name in files:
            path = Path(current) / name
            if not path.is_symlink():
                os.chmod(path, 0o644)


def build_chain_means(view_root: Path) -> str:
    """Final-layer chain means of a view from its per-residue caches (G1.10 step 2); returns the manifest SHA-256."""
    from ec_clean30.views import ALLOWLIST_NAME, View
    from embed_helpers.esmc_chain_means import build_from_residue_cache

    view = View.load(view_root)
    summary = build_from_residue_cache(view.root / ALLOWLIST_NAME, view.structures, view.esm,
                                       view.chain_means_dir("final"), embedding_dim=ESM_DIM)
    return summary["manifest_sha256"]


def write_inner_membership(paths: dict, out_dir: Path, *, fold: int = 0, certified: bool = True,
                           n_inner_folds: int = 4) -> tuple[Path, str, Path]:
    """A 4-fold inner membership over the eligible training structures plus a synthetic certify-inner receipt.

    The receipt mirrors the fields of ``ec_clean30.identity.certify_inner`` that a view build checks
    (step, fold, n_inner_folds, certified, membership file and SHA-256). Returns (membership, SHA-256, receipt).
    """
    from ec_clean30.identity import INNER_CERTIFICATE, INNER_MEMBERSHIP_NAME
    from training.ec_membership import read_ec_fold_membership, write_ec_fold_membership

    eligible = [stem for stem in sorted(paths["stems"].values()) if not stem.startswith(AMBIGUOUS_ACCESSION)]
    rows = [{"structure_id": stem, "accession": stem.split("__")[0], "group_id": stem.split("__")[0],
             "fold": str(index % n_inner_folds)} for index, stem in enumerate(eligible)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    membership = out_dir / INNER_MEMBERSHIP_NAME.format(fold=fold)
    digest = write_ec_fold_membership(rows, membership)
    loaded = read_ec_fold_membership(membership, digest)
    receipt = {"schema_version": 1, "step": "certify_inner", "fold": fold, "n_inner_folds": n_inner_folds,
               "certified": certified, "failures": [] if certified else ["cross_fold_links"],
               "membership": {"file": membership.name, "sha256": digest, "identity_sha256": loaded.identity_sha256},
               "labels_read": True}
    certificate = out_dir / INNER_CERTIFICATE
    certificate.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return membership, digest, certificate


def build_smoke_view(paths: dict, out: Path, *, seed: int = 42, max_proteins: int = 8, chain_means: bool = True,
                     provenance: dict | None = None):
    """A synthetic smoke view of fold 0 (bundle ID ``g1-smoke-view``), optionally with its chain means."""
    from ec_clean30.views import SourceLayout, View, build_view

    build_view(fold=0, layout=SourceLayout.from_data_root(paths["data_root"]), out_dir=out, smoke=True,
               max_proteins=max_proteins, seed=seed, bundle_id="g1-smoke-view",
               provenance=provenance or {"note": "synthetic test provenance"})
    if chain_means:
        build_chain_means(out)
    return View.load(out)
