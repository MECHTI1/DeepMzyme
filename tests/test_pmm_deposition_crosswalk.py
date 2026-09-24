from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src import audit_pmm_deposition_crosswalk as audit


def ion_line(serial: int, element: str, chain: str, resseq: int, x: float) -> str:
    return (f"HETATM{serial:5d} {element:>2s}   {element:>3s} {chain}{resseq:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{30.0:6.2f}          {element:>2s}  \n")


class DepositionCrosswalkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, pdbid: str, label: str = "1") -> dict:
        return {"source_uid": f"test:{pdbid}", "original_side": "test", "source_row_number": 7,
                "pdbid": pdbid, "residueid_ion": "42", "metalid": "900",
                "label_metal": label, "four_class_target": audit.LABELS[label], "effective_dropna": True}

    def test_deposition_tables_keep_internal_id_and_coordinate_separate(self) -> None:
        types_path = self.root / "type.csv"
        types_path.write_text("pdbid,residueid_ion,resi_type,resname_ion,which_metal\n"
                              "4d8f,42,H1ED2,_MN,25\n", encoding="utf-8")
        coords_path = self.root / "coords.txt"
        coords_path.write_text("pdbid\tchainid\tresseq\tresname_ion\texp_metal_coord\n"
                               "4d8f\tA\t402\tMN\t3.641,0,0\n", encoding="utf-8")
        types = audit.read_type_records(types_path)
        coords = audit.read_coordinate_records([coords_path])
        structure = self.root / "4d8f_rcsb_current.pdb"
        structure.write_text(ion_line(1, "MN", "A", 402, 3.641)
                             + ion_line(2, "MN", "B", 401, 27.129))
        reviewed = audit.review_row(self.source("4d8f"), types, coords, self.root)
        self.assertEqual(reviewed["mapping_status"], "ambiguous")
        self.assertEqual(reviewed["mapping_reason"], "multiple_structural_ions_without_identifier_link")
        self.assertEqual(reviewed["original_side"], "test")
        self.assertEqual(reviewed["coordinate_candidate_count"], 1)
        self.assertFalse(reviewed.get("resseq"))

    def test_unique_element_and_exact_deposited_coordinate_required(self) -> None:
        source = self.source("abcd", "7")
        types = {("abcd", "42"): [{"element": "ZN", "line": 2}]}
        coords = {("abcd", "ZN"): [{"chain": "A", "resseq": "100",
                                    "coordinate": (1.0, 0.0, 0.0), "references": ["coords:2"]}]}
        structure = self.root / "abcd_rcsb_current.pdb"
        structure.write_text(ion_line(1, "ZN", "A", 100, 1.0))
        reviewed = audit.review_row(source, types, coords, self.root)
        self.assertEqual(reviewed["mapping_status"], "exact")
        self.assertEqual((reviewed["element"], reviewed["chain"], reviewed["resseq"]), ("ZN", "A", "100"))
        structure.write_text(ion_line(1, "ZN", "A", 100, 2.0))
        self.assertEqual(audit.review_row(source, types, coords, self.root)["mapping_status"], "unmatched")
        structure.write_text(ion_line(1, "ZN", "A", 100, 1.0) + ion_line(2, "ZN", "B", 100, 2.0))
        self.assertEqual(audit.review_row(source, types, coords, self.root)["mapping_status"], "ambiguous")

    def test_multiple_deposited_sites_do_not_assign_either_source_side(self) -> None:
        types = {("1a0e", "886"): [{"element": "CO", "line": 4}],
                 ("1a0e", "887"): [{"element": "CO", "line": 5}]}
        coords = {("1a0e", "CO"): [
            {"chain": "A", "resseq": "491", "coordinate": (1.0, 0.0, 0.0), "references": ["EDH:2"]},
            {"chain": "A", "resseq": "492", "coordinate": (3.0, 0.0, 0.0), "references": ["EDH:3"]},
        ]}
        rows = []
        for side, residueid, metalid in (("train", "887", "4637"), ("test", "886", "4638")):
            source = self.source("1a0e", "2")
            source.update(original_side=side, residueid_ion=residueid, metalid=metalid)
            rows.append(audit.review_row(source, types, coords, self.root))
        self.assertEqual([row["mapping_status"] for row in rows], ["ambiguous", "ambiguous"])
        self.assertEqual([row["original_side"] for row in rows], ["train", "test"])
        self.assertEqual([row["metalid"] for row in rows], ["4637", "4638"])
        self.assertTrue(all(not row.get("resseq") for row in rows))

    def test_conflict_and_missing_evidence_do_not_map(self) -> None:
        source = self.source("abcd", "7")
        self.assertEqual(audit.review_row(source, {}, {}, self.root)["mapping_status"], "incomplete")
        types = {("abcd", "42"): [{"element": "FE", "line": 2}]}
        self.assertEqual(audit.review_row(source, types, {}, self.root)["mapping_reason"],
                         "deposited_element_conflicts_with_source_class")


if __name__ == "__main__":
    unittest.main()
