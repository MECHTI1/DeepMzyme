"""Synthetic EC-class training fixtures (no real data): EC-named structures, site summary and embeddings."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pmm_campaign_fixtures import write_fake_embeddings, write_pdb  # noqa: E402

OMITTED_EXTERNAL_FEATURES = ("biotite_residue_sasa", "custom_charge_distance_proxy", "dpka_titr")


def default_ec_proteins(n_per_class: int = 4, classes: tuple[str, ...] = ("1", "2", "3")) -> dict[str, dict]:
    """Accession -> spec with one zinc site each, cycling through the given EC classes."""
    proteins = {}
    for class_index, ec_class in enumerate(classes):
        for k in range(n_per_class):
            accession = f"P{class_index}{k:03d}"
            proteins[accession] = {
                "ec": f"{ec_class}.1.1.{k + 1}",
                "ions": [("A", 900, "", "ZN", [10.0 * k, 5.0 * class_index, 0.0])],
            }
    return proteins


def write_ec_dataset(root: Path, proteins: dict[str, dict], *, esm_dim: int = 8) -> dict[str, Path]:
    """Write ``root/structures``, a selected-site summary CSV and fake ESMC embeddings.

    Each spec is {"ec": "3.1.1.1" (or "2.5.1.3,4.1.99.17"), "ions": [(chain, resseq, icode, element, xyz)],
    optionally "summary_sites": [(chain, resseq, element)] when the summary should list fewer sites}.
    """
    structures = root / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    stems = {}
    for accession, spec in proteins.items():
        stem = f"{accession}__chain_A__EC_{spec['ec']}"
        write_pdb(structures / f"{stem}.pdb", ions=spec["ions"])
        stems[accession] = stem
        sites = spec.get("summary_sites") or [(chain, resseq, element) for chain, resseq, _i, element, _xyz in spec["ions"]]
        for chain, resseq, element in sites:
            summary_rows.append({"structure": accession, "chain_resi": f"{chain}_{resseq}", "metaltype": element,
                                 "ecnumber": spec["ec"], "whether_catalytic": "1"})
    summary = root / "site_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    esm = root / "esm"
    write_fake_embeddings(root, esm, dim=esm_dim, model_name="esmc_300m")
    external = root / "empty_external"
    external.mkdir(exist_ok=True)
    return {"root": root, "structures": structures, "summary": summary, "esm": esm, "external": external,
            "stems": stems}


def tiny_ec_argv(paths: dict[str, Path], run_dir: Path, *extra: str, esm_dim: int = 8) -> list[str]:
    """Training arguments for a tiny CPU EC-class fit on the fixture (extend with membership or options)."""
    return [
        "--task", "ec", "--ec-label-depth", "1", "--structure-dir", str(paths["structures"]),
        "--summary-csv", str(paths["summary"]), "--esm-embeddings-dir", str(paths["esm"]),
        "--esm-dim", str(esm_dim), "--no-prepare-missing-esm-embeddings",
        "--external-features-root-dir", str(paths["external"]), "--allow-missing-external-features",
        "--omit-node-features", ",".join(OMITTED_EXTERNAL_FEATURES), "--shell-role-source", "geometry",
        "--no-prepare-missing-ring-edges", "--model-architecture", "only_esm", "--hidden-s", "16",
        "--hidden-v", "4", "--edge-hidden", "8", "--gvp-layers", "2", "--esm-fusion-dim", "8",
        "--learning-rate", "1e-3", "--batch-size", "4", "--epochs", "2", "--seed", "42",
        "--runs-dir", str(run_dir), "--load-workers", "1", "--device", "cpu", *extra,
    ]
