"""Create a train-only PROPKA overlay without modifying the published caches."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_structures import EXTERNAL_FEATURE_DPKA_TITR
from feature_extraction.propka_support import run_propka_for_structure
from graph.structure_parsing import parse_structure_file
from structure_store import resolve_structure_files
from training.runtime_preparation import updated_external_feature_path_candidates

DATASET = "train_and_test_sets_structures_non_overlapped_pinmymetal"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def residue_numbering_plan(parsed):
    """Keep original identifiers unless PROPKA's labels would lose insertions."""
    residues = [(chain.id.strip() or "_", residue)
                for chain in next(parsed.get_models()) for residue in chain
                if residue.id[0] == " "]
    if not any(residue.id[2].strip() for _, residue in residues):
        return None
    per_chain = {}
    mapping = []
    for chain, residue in residues:
        temporary_resseq = per_chain.get(chain, 0) + 1
        per_chain[chain] = temporary_resseq
        if temporary_resseq > 9999:
            raise ValueError("Temporary PROPKA numbering exceeds the PDB residue-number field")
        mapping.append(dict(chain_id=chain, resseq=int(residue.id[1]),
                            icode=residue.id[2].strip(), resname=residue.resname,
                            temporary_resseq=temporary_resseq))
    return mapping


def numbering_metadata(mapping):
    if mapping is None:
        return {"residue_numbering_mode": "original_residue_numbers"}
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(residue_numbering_mode="temporary_unique_residue_numbers_for_insertions",
                residue_numbering_map_sha256=digest, residue_numbering_map=mapping)


def run_propka_with_numbering(structure, mapping):
    if mapping is None:
        return run_propka_for_structure(structure, ph=7.0)
    if structure.suffix.lower() != ".pdb":
        raise ValueError("Insertion-code PROPKA repair currently requires a PDB structure")
    by_residue = {(row["chain_id"], row["resseq"], row["icode"]): row["temporary_resseq"]
                  for row in mapping}
    lines = []
    for line in structure.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM  ", "TER   ")):
            key = (line[21:22].strip() or "_", int(line[22:26]), line[26:27].strip())
            if key not in by_residue:
                raise ValueError(f"Unmapped residue in temporary PROPKA input: {key}")
            # Change only residue numbering/insertion columns. Atom names,
            # coordinates, occupancies, alternate locations and metals survive.
            line = line[:22] + f"{by_residue[key]:4d} " + line[27:]
        lines.append(line)
    with tempfile.TemporaryDirectory(prefix="metal_propka_unique_residues_") as directory:
        temporary_structure = Path(directory) / structure.name
        temporary_structure.write_text("".join(lines))
        return run_propka_for_structure(temporary_structure, ph=7.0)


def needs_numbering_refresh(path):
    parsed = parse_structure_file(str(path))
    return any(residue.id[0] == " " and (
        residue.id[1] >= 1000 or residue.id[1] <= -100 or residue.id[2].strip()
    ) for residue in parsed.get_residues())


def repair_one(structure, source, output, refresh=False):
    started = time.monotonic()
    target = output / structure.stem / "residue_features.json"
    source_sha, structure_sha = sha(source), sha(structure)
    parsed = parse_structure_file(str(structure))
    mapping = residue_numbering_plan(parsed)
    numbering = numbering_metadata(mapping)
    if target.is_file() and not refresh:
        existing = json.loads(target.read_text())
        repair = existing.get("pka_repair", {})
        if (repair.get("source_features_sha256") == source_sha
                and repair.get("structure_sha256") == structure_sha
                and existing.get("tooling", {}).get("pka") == "propka"
                and repair.get("ph") == 7.0
                and repair.get("method") == "existing_geometry_features_plus_propka"
                and repair.get("updated_residues", 0) > 0
                and (mapping is None or all(repair.get(key) == value for key, value in numbering.items()))
                and existing.get("residues")
                and sum(row["features"].get(EXTERNAL_FEATURE_DPKA_TITR + "_missing") == 0
                        for row in existing["residues"]) >= repair["updated_residues"]
                and all(math.isfinite(float(v)) for row in existing.get("residues", [])
                        for v in row["features"].values())):
            return dict(structure=structure.name, sha256=sha(target), path=str(target), reused=True,
                        source_features_sha256=source_sha, structure_sha256=structure_sha,
                        updated_residues=repair["updated_residues"])
        raise ValueError(f"Incompatible existing overlay: {target}")
    payload = json.loads(source.read_text())
    residue_names = {
        (chain.id.strip() or "_", residue.id[1], residue.id[2].strip()): residue.resname
        for chain in next(parsed.get_models()) for residue in chain if residue.id[0] == " "
    }
    result = run_propka_with_numbering(structure, mapping)
    if not result.residues:
        raise ValueError(f"No PROPKA residue results for {structure.name}")
    updated = 0
    mapped_keys = ({(row["chain_id"], row["resseq"], row["icode"], row["resname"]):
                    (row["chain_id"], row["temporary_resseq"], row["resname"])
                    for row in mapping} if mapping is not None else None)
    for row in payload["residues"]:
        residue_key = (str(row["chain_id"]).strip() or "_", int(row["resseq"]), str(row["icode"]).strip())
        resname = residue_names.get(residue_key, "")
        key = (mapped_keys.get((*residue_key, resname)) if mapped_keys is not None else
               (residue_key[0], residue_key[1], resname))
        if key in result.residues:
            value = float(result.residues[key].dpka_titr)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite pKa feature for {structure.name}: {key}")
            row["features"][EXTERNAL_FEATURE_DPKA_TITR] = value
            row["features"][EXTERNAL_FEATURE_DPKA_TITR + "_missing"] = 0.0
            updated += 1
    if not updated:
        raise ValueError(f"PROPKA keys did not match any cached residues for {structure.name}")
    payload["tooling"]["pka"] = "propka"
    payload["pka_repair"] = dict(
        source_features_sha256=source_sha, structure_sha256=structure_sha,
        ph=7.0, method="existing_geometry_features_plus_propka", updated_residues=updated,
        propka_version=version("propka"),
        propka_parser_revision="compact_residue_tokens_v1",
        original_warnings=payload.get("warnings", []),
        **numbering,
    )
    payload["warnings"] = result.warnings
    save(target, payload)
    return dict(structure=structure.name, sha256=sha(target), path=str(target),
                source_features_sha256=source_sha, structure_sha256=structure_sha,
                updated_residues=updated, seconds=time.monotonic() - started, reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh-wide-residue-numbers", action="store_true",
                        help="Recompute compact PROPKA residue-number and insertion-code repairs")
    args = parser.parse_args()
    if args.jobs < 1 or args.limit < 0:
        parser.error("jobs must be positive and limit nonnegative")
    train = args.data_root / DATASET / "train"
    structures = resolve_structure_files(train, recursive_legacy_scan=False)
    selected = structures[:args.limit] if args.limit else structures
    sources = [next(p for p in updated_external_feature_path_candidates(
        structure, structure_root=train,
        external_features_root_dir=args.data_root / "updated_feature_extraction") if p.is_file())
        for structure in selected]
    output = args.output_root.resolve()
    if output == (args.data_root / "updated_feature_extraction").resolve():
        raise ValueError("Use a separate overlay directory; original caches are immutable")
    report = dict(dataset=DATASET, split="train", test_evaluation=False,
                  total_structures=len(structures), selected_structures=len(selected),
                  propka_version=version("propka"),
                  refresh_wide_residue_numbers=args.refresh_wide_residue_numbers,
                  refresh_includes_insertion_codes=True,
                  started_at=datetime.now(timezone.utc).isoformat(), files=[], failures=[])
    report_path = output / "feature_overlay_manifest.json"
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        def needs_refresh(path):
            if not args.refresh_wide_residue_numbers:
                return False
            return needs_numbering_refresh(path)
        pending = {pool.submit(repair_one, p, source, output, needs_refresh(p)): p
                   for p, source in zip(selected, sources)}
        for index, future in enumerate(as_completed(pending), 1):
            path = pending[future]
            try:
                report["files"].append(future.result())
            except Exception as exc:
                report["failures"].append(dict(structure=path.name, error=str(exc)))
                print(f"FAILED {path.name}: {exc}", flush=True)
            report["status"] = "partial"
            save(report_path, report)
            if index % 25 == 0 or index == len(selected):
                print(f"PROPKA overlay {index}/{len(selected)}; failures={len(report['failures'])}", flush=True)
    report["files"].sort(key=lambda row: row["structure"])
    report["status"] = "complete" if not report["failures"] and len(selected) == len(structures) else "partial"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    save(report_path, report)
    print(report_path, flush=True)
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
