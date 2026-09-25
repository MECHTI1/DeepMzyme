"""Synthetic training-side fixtures for the PMM ion campaign tests (no real data)."""

from __future__ import annotations

import csv
import hashlib
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.esm_feature_loading import build_embedding_payload, residue_keys_for_structure_chain  # noqa: E402

def _atom_line(record, serial, name, altloc, resname, chain, resseq, icode, xyz, element, occupancy=1.0):
    padded = f" {name:<3}" if len(name) < 4 and len(element) == 1 else f"{name:<4}"
    x, y, z = xyz
    return (f"{record:<6}{serial:>5} {padded}{altloc:1}{resname:>3} {chain:1}{resseq:>4}{icode:1}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{20.0:>6.2f}          {element:>2}")


def _residue_atoms(resname: str, ca, towards):
    """Backbone plus one donor atom placed 1.5 Å from CA towards ``towards``."""
    direction = [t - c for t, c in zip(towards, ca)]
    norm = math.sqrt(sum(d * d for d in direction)) or 1.0
    unit = [d / norm for d in direction]
    atoms = [("N", [ca[0] - 1.2, ca[1] + 0.8, ca[2]], "N"), ("CA", list(ca), "C"),
             ("C", [ca[0] + 1.2, ca[1] + 0.8, ca[2]], "C"), ("O", [ca[0] + 1.3, ca[1] + 2.0, ca[2]], "O"),
             ("CB", [c + 0.8 * u for c, u in zip(ca, unit)], "C")]
    donor = {"HIS": ("NE2", "N"), "CYS": ("SG", "S"), "GLU": ("OE1", "O"), "ASP": ("OD1", "O")}.get(resname)
    if donor is not None:
        atoms.append((donor[0], None, donor[1]))
    return atoms


def write_pdb(path: Path, *, ions, chains=("A",), n_models: int = 1, altloc_ion: bool = False) -> None:
    """Write a small PDB with residues ringed around each ion and first-shell donors.

    ``ions`` is a list of ``(chain, resseq, icode, element, xyz)``.
    """
    lines = []
    for model in range(n_models):
        if n_models > 1:
            lines.append(f"MODEL     {model + 1:>4}")
        serial = 1
        for chain_index, chain in enumerate(chains):
            resseq = 1
            for ion_index, (_c, _r, _i, _e, xyz) in enumerate(ions):
                for k in range(6):
                    angle = 2 * math.pi * k / 6 + 0.3 * chain_index
                    radius = 4.2 if chain_index == 0 else 7.5
                    ca = [xyz[0] + radius * math.cos(angle), xyz[1] + radius * math.sin(angle),
                          xyz[2] + (1.5 if chain_index else 0.0)]
                    resname = ("HIS", "CYS", "GLU", "ALA", "ASP", "GLY")[k] if chain_index == 0 else "ALA"
                    for name, coord, element in _residue_atoms(resname, ca, xyz):
                        if coord is None:
                            # First-shell donor at 2.1 Å from the ion for k < 2 (chain 0 only).
                            vec = [c - x for c, x in zip(ca, xyz)]
                            norm = math.sqrt(sum(v * v for v in vec))
                            dist = 2.1 if (k < 2 and chain_index == 0) else 3.6
                            coord = [x + dist * v / norm for x, v in zip(xyz, vec)]
                        lines.append(_atom_line("ATOM", serial, name, "", resname, chain, resseq, "", coord, element))
                        serial += 1
                    resseq += 1
        for chain, resseq_ion, icode, element, xyz in ions:
            if altloc_ion and (chain, resseq_ion) == (ions[0][0], ions[0][1]):
                lines.append(_atom_line("HETATM", serial, element, "A", element, chain, resseq_ion, icode, xyz, element, 0.6))
                serial += 1
                shifted = [xyz[0] + 0.8, xyz[1], xyz[2]]
                lines.append(_atom_line("HETATM", serial, element, "B", element, chain, resseq_ion, icode, shifted, element, 0.4))
            else:
                lines.append(_atom_line("HETATM", serial, element, "", element, chain, resseq_ion, icode, xyz, element))
            serial += 1
        if n_models > 1:
            lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


LABEL_CODE = {"MN": "1", "CU": "6", "ZN": "7", "FE": "2", "CO": "2", "NI": "2"}


def build_train_dir(root: Path, entries: dict[str, dict]) -> tuple[Path, Path, str]:
    """Create ``root/train`` with manifests and a matching fake PMM source CSV.

    ``entries`` maps pdbid -> {"ions": [...], "chains": (...), "n_models": int, "altloc_ion": bool,
    "rows": [(chain, resseq, manifest_element, label_code), ...]}. Returns (train_dir, source_csv, source_sha).
    """
    train = root / "train"
    (train / "structures").mkdir(parents=True)
    planned = []
    for pdbid, spec in entries.items():
        for chain, resseq, element, code in spec["rows"]:
            planned.append((pdbid, chain, resseq, element, code))
    source = root / "classmodel_train_set"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdbid", "residueid_ion", "metalid", "label_metal", "ched_count",
                                                    "f1", "f2"])
        writer.writeheader()
        for index, (pdbid, _chain, _resseq, _element, code) in enumerate(planned, start=1):
            writer.writerow({"pdbid": pdbid, "residueid_ion": 1000 + index, "metalid": index, "label_metal": code,
                             "ched_count": 3, "f1": float(index % 7), "f2": 1.0})
    source_sha = sha256(source)
    site_rows, structure_rows, chain_rows, summary_rows = [], [], [], []
    source_row = 0
    for pdbid, spec in entries.items():
        name = f"{pdbid}__chain_{spec['chains'][0]}__EC_0.0.0.0.pdb"
        path = train / "structures" / name
        write_pdb(path, ions=spec["ions"], chains=spec.get("chains", ("A",)), n_models=spec.get("n_models", 1),
                  altloc_ion=spec.get("altloc_ion", False))
        structure_rows.append({"structure_name": name, "relative_path": f"structures/{name}", "sha256": sha256(path),
                               "size_bytes": path.stat().st_size})
        chain_rows.append({"structure_name": name, "pdbid": pdbid, "chain": spec["chains"][0]})
        for chain, resseq, element, code in spec["rows"]:
            source_row += 1
            site_rows.append({"source_uid": f"sha256:{source_sha}:row:{source_row}", "source_row": source_row,
                              "source_side": "train", "pdbid": pdbid, "chain": chain, "resseq": resseq,
                              "chain_resi": f"{chain}_{resseq}", "element": element, "metaltype": element,
                              "label_metal": code, "structure_name": name, "residueid_ion": 1000 + source_row,
                              "resolution_method": "fixture"})
            summary_rows.append({"structure": pdbid, "chain_resi": f"{chain}_{resseq}", "metaltype": element,
                                 "ecnumber": "0.0.0.0", "whether_catalytic": "1"})
    for filename, rows in (("site_manifest.csv", site_rows), ("structure_manifest.csv", structure_rows),
                           ("structure_chain_manifest.csv", chain_rows),
                           ("final_data_summarazing_table.csv", summary_rows)):
        with (train / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return train, source, source_sha


def write_fake_embeddings(train_dir: Path, out_dir: Path, *, dim: int, skip_chain: str | None = None,
                          model_name: str = "esmc_600m") -> None:
    from embed_helpers.esmc import extract_chain_sequences, parse_structure
    from training.esm_feature_loading import embedding_metadata_from_payload, write_embedding_metadata_sidecar

    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    for path in sorted((train_dir / "structures").glob("*.pdb")):
        structure = parse_structure(path)
        sequences = extract_chain_sequences(structure)
        for chain, sequence in sequences.items():
            if chain == skip_chain:
                continue
            residue_ids, _ = residue_keys_for_structure_chain(structure, chain)
            payload = build_embedding_payload(
                torch.randn(len(residue_ids), dim, generator=generator), residue_ids, structure_id=path.stem,
                chain_id=chain, source_path=str(path),
                metadata={"esm_model_name": model_name, "embedding_dim": dim,
                          "source_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()},
            )
            target = out_dir / f"{path.stem}_chain_{chain}_esmc.pt"
            torch.save(payload, target)
            write_embedding_metadata_sidecar(target, embedding_metadata_from_payload(payload))


ELEMENT_CYCLE = ("ZN", "FE", "MN", "CU", "CO", "NI")


def standard_entries(n_pdb: int = 12) -> dict[str, dict]:
    """PDB entries with two ions each (a close sibling pair) cycling through all six elements."""
    entries = {}
    for index in range(n_pdb):
        pdbid = f"{index + 1}ab{chr(ord('a') + index % 26)}"
        first = ELEMENT_CYCLE[(2 * index) % 6]
        second = ELEMENT_CYCLE[(2 * index + 1) % 6]
        ions = [("A", 201, "", first, (0.0, 0.0, 0.0)), ("A", 202, "", second, (3.2, 0.0, 0.0))]
        entries[pdbid] = {"ions": ions, "chains": ("A", "B"),
                          "rows": [("A", 201, first, LABEL_CODE[first]), ("A", 202, second, LABEL_CODE[second])]}
    return entries
