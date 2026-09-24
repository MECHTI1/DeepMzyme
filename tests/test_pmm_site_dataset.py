from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src import build_pmm_site_dataset as pmm


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def ion_line(serial: int, element: str, resseq: int, x: float, y: float, z: float) -> str:
    return (f"HETATM{serial:5d} {element:>2s}   {element:>3s} A{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{30.0:6.2f}          {element:>2s}  \n")


class PmmSiteDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source_file(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        write_csv(path, rows, ["pdbid", "residueid_ion", "metalid", "label_metal", "ched_count", "feature"])
        return path

    def test_dropna_counts_and_distinct_source_row_uids(self) -> None:
        path = self.source_file("train.csv", [
            {"pdbid": "1a0e", "residueid_ion": "886", "metalid": "4638", "label_metal": "2", "ched_count": "4", "feature": "3"},
            {"pdbid": "1a0e", "residueid_ion": "887", "metalid": "4637", "label_metal": "2", "ched_count": "2", "feature": ""},
        ])
        with patch.dict(pmm.SOURCE_HASHES, {"train": pmm.sha256(path)}):
            rows, stats = pmm.load_source(path, "train")
        self.assertEqual(stats["raw_rows"], 2)
        self.assertEqual(stats["effective_dropna_rows"], 1)
        self.assertEqual(stats["released_script_missing_columns"], ["source"])
        self.assertNotEqual(rows[0]["source_uid"], rows[1]["source_uid"])
        self.assertEqual([r["effective_dropna"] for r in rows], [True, False])

    def test_crosswalk_preserves_same_site_ions_and_source_sides(self) -> None:
        first_structure = self.root / "1a0e.pdb"
        second_structure = self.root / "4d8f.pdb"
        first_structure.write_text(ion_line(1, "CO", 491, 0, 0, 0) + ion_line(2, "CO", 492, 3, 0, 0))
        second_structure.write_text(ion_line(3, "FE", 401, 20, 0, 0) + ion_line(4, "MN", 402, 23, 0, 0))
        specs = [
            ("train", "1a0e", "887", "4637", "2", "CO", "491", 0),
            ("test", "1a0e", "886", "4638", "2", "CO", "492", 3),
            ("train", "4d8f", "1316", "3761", "1", "MN", "402", 23),
            ("test", "4d8f", "1317", "3762", "2", "FE", "401", 20),
        ]
        rows = []
        evidence = {}
        for index, (side, pdbid, residueid, metalid, label, element, resseq, x) in enumerate(specs):
            uid = f"uid:{index}"
            source = {"source_uid": uid, "pdbid": pdbid, "residueid_ion": residueid,
                      "metalid": metalid, "label_metal": label, "four_class_target": pmm.LABELS[label],
                      "effective_dropna": True, "original_side": side}
            rows.append(source)
            structure = first_structure if pdbid == "1a0e" else second_structure
            evidence[uid] = [{"coordinate_path": structure.name, "coordinate_sha256": pmm.sha256(structure),
                              "model": "1", "chain": "A", "resseq": resseq, "icode": "", "altloc": "",
                              "element": element, "evidence_kind": "pmm_neighborhood_residue_record",
                              "evidence_reference": "fixture", "evidence_sha256": "fixture", "reviewer": "fixture",
                              "structure_accession": pdbid, "structure_version": "fixture-v1",
                              "label_chain": "A", "symmetry_operator": "identity", "context_complete": "yes"}]
        mapped = pmm.crosswalk(rows, evidence, self.root)
        pmm.assign_physical_groups(mapped)
        self.assertEqual([r["mapping_status"] for r in mapped], ["exact"] * 4)
        self.assertEqual(len({r["source_uid"] for r in mapped}), 4)
        self.assertEqual(mapped[0]["physical_site_group"], mapped[1]["physical_site_group"])
        self.assertEqual(mapped[2]["physical_site_group"], mapped[3]["physical_site_group"])
        self.assertNotEqual(mapped[0]["physical_site_group"], mapped[2]["physical_site_group"])
        self.assertEqual([r["original_side"] for r in mapped], ["train", "test", "train", "test"])

    def test_checksum_mismatch_fails_and_graph_omits_label_and_ion_symbol(self) -> None:
        structure = self.root / "site.pdb"
        structure.write_text(
            "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C  \n"
            + "ATOM      2  CA  GLY A   2       4.000   0.000   0.000  1.00 20.00           C  \n"
            + ion_line(3, "CO", 491, 0, 0, 0)
        )
        source = {"pdbid": "site", "four_class_target": "CLASS_VIII"}
        evidence = {"coordinate_path": "site.pdb", "coordinate_sha256": "bad", "model": "1",
                    "chain": "A", "resseq": "491", "icode": "", "altloc": "", "element": "CO",
                    "structure_accession": "site", "structure_version": "fixture-v1",
                    "label_chain": "A", "symmetry_operator": "identity", "context_complete": "yes"}
        self.assertEqual(pmm.verify_match(source, evidence, self.root)[:2],
                         ("incomplete", "coordinate_checksum_missing_or_mismatch"))
        anchor = {"x": 0.0, "y": 0.0, "z": 0.0}
        original = pmm.pocket_graph(structure, anchor)
        structure.write_text(structure.read_text().replace(" CO ", " FE ").replace("          CO", "          FE"))
        changed = pmm.pocket_graph(structure, anchor)
        self.assertEqual(original, changed)
        self.assertNotIn("CO", json.dumps(original))
        self.assertNotIn("CLASS_VIII", json.dumps(original))
        self.assertEqual(len(original["nodes"]), 2)

    def test_non_ec_and_negative_catalytic_rows_are_not_filtered(self) -> None:
        path = self.root / "non_ec.csv"
        write_csv(path, [
            {"pdbid": "abcd", "residueid_ion": "1", "metalid": "2", "label_metal": "7",
             "ched_count": "0", "feature": "0", "whether_catalytic": "0"},
        ], ["pdbid", "residueid_ion", "metalid", "label_metal", "ched_count", "feature", "whether_catalytic"])
        with patch.dict(pmm.SOURCE_HASHES, {"train": pmm.sha256(path)}):
            rows, stats = pmm.load_source(path, "train")
        self.assertEqual(stats["effective_dropna_rows"], 1)
        self.assertEqual(rows[0]["four_class_target"], "ZN")

    def test_audit_keeps_all_rows_without_inventing_site_matches(self) -> None:
        train = self.source_file("train.csv", [
            {"pdbid": "1a0e", "residueid_ion": "887", "metalid": "4637", "label_metal": "2",
             "ched_count": "0", "feature": "1"},
        ])
        test = self.source_file("test.csv", [
            {"pdbid": "1a0e", "residueid_ion": "886", "metalid": "4638", "label_metal": "2",
             "ched_count": "0", "feature": "1"},
        ])
        output = self.root / "audit"
        args = Namespace(source_train=train, source_test=test, evidence_csv=None,
                         structure_root=None, output_root=output, emit_graphs=False)
        with patch.dict(pmm.SOURCE_HASHES, {"train": pmm.sha256(train), "test": pmm.sha256(test)}):
            coverage = pmm.build(args)
        self.assertEqual(coverage["mapping_status"], {"incomplete": 2})
        with (output / "site_crosswalk.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["original_side"] for row in rows], ["train", "test"])
        self.assertNotEqual(rows[0]["source_uid"], rows[1]["source_uid"])
        for side in ("train", "test"):
            with (output / side / "site_manifest.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
        metadata = json.loads((output / "split_metadata.json").read_text())
        self.assertEqual(metadata["overlap"]["source_shared_pdbids"], 1)
        self.assertEqual(metadata["overlap"]["source_shared_identifier_triplets"], 0)
        self.assertFalse(metadata["training_loader_certified"])

    def test_reviewed_two_ion_rows_emit_distinct_label_blind_graphs(self) -> None:
        train = self.source_file("train.csv", [
            {"pdbid": "1a0e", "residueid_ion": "887", "metalid": "4637", "label_metal": "2", "ched_count": "0", "feature": "1"},
            {"pdbid": "1a0e", "residueid_ion": "886", "metalid": "4638", "label_metal": "2", "ched_count": "0", "feature": "1"},
        ])
        test = self.source_file("test.csv", [])
        structure = self.root / "1a0e.pdb"
        structure.write_text(
            "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C  \n"
            + "ATOM      2  CA  GLY A   2       4.000   0.000   0.000  1.00 20.00           C  \n"
            + ion_line(3, "CO", 491, 0, 0, 0) + ion_line(4, "CO", 492, 3, 0, 0)
        )
        source_sha = pmm.sha256(train)
        evidence_rows = []
        for row_number, (residueid, metalid, resseq) in enumerate((("887", "4637", "491"), ("886", "4638", "492")), start=1):
            evidence_rows.append({
                "source_uid": f"sha256:{source_sha}:row:{row_number}", "pdbid": "1a0e",
                "residueid_ion": residueid, "metalid": metalid, "coordinate_path": structure.name,
                "model": "1", "chain": "A", "resseq": resseq, "icode": "", "altloc": "", "element": "CO",
                "structure_accession": "1a0e", "structure_version": "fixture-v1", "label_chain": "A",
                "symmetry_operator": "identity", "context_complete": "yes", "coordinate_sha256": pmm.sha256(structure),
                "evidence_kind": "pmm_neighborhood_residue_record", "evidence_reference": "reviewed-fixture",
                "evidence_sha256": "reviewed-fixture-sha", "reviewer": "unit-test",
            })
        evidence_file = self.root / "evidence.csv"
        write_csv(evidence_file, evidence_rows, list(pmm.EVIDENCE_COLUMNS))
        output = self.root / "built"
        args = Namespace(source_train=train, source_test=test, evidence_csv=evidence_file,
                         structure_root=self.root, output_root=output, emit_graphs=True)
        with patch.dict(pmm.SOURCE_HASHES, {"train": source_sha, "test": pmm.sha256(test)}):
            coverage = pmm.build(args)
        self.assertEqual(coverage["exact_rows"], 2)
        with (output / "train" / "site_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["physical_site_group"], rows[1]["physical_site_group"])
        self.assertNotEqual(rows[0]["graph_path"], rows[1]["graph_path"])
        for row in rows:
            graph = json.loads((output / row["graph_path"]).read_text())
            self.assertEqual(graph["source_uid"], row["source_uid"])
            self.assertNotIn("CO", json.dumps(graph))
            self.assertNotIn("CLASS_VIII", json.dumps(graph))


if __name__ == "__main__":
    unittest.main()
