"""Regression coverage for PROPKA fixed-width residue labels and determinants."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_extraction.propka_support import parse_propka_output_text


def primary(label, sidechain="XXX   0 X", backbone="XXX   0 X", coulomb="XXX   0 X"):
    return (f"{label}   3.66    40 %    1.48  392   0.28    0"
            f"   -1.04 {sidechain}   -0.04 {backbone}   -0.13 {coulomb}")


def continuation(label, value, partner="XXX   0 X"):
    return f"{label}                                            0.00 XXX   0 X    0.00 XXX   0 X    {value} {partner}"


def table(*rows):
    return "RESIDUE    pKa    BURIED\n---------\n" + "\n".join(rows) + "\nSUMMARY OF THIS PREDICTION\n"


def test_real_2a21_four_digit_primary_and_continuations():
    text = table(
        "ASP1045 A   3.66    40 %    1.48  392   0.28    0   -1.04 ARG1056 A   -0.04 LYS1046 A   -0.13 LYS1046 A",
        "ASP1045 A                                            0.00 XXX   0 X    0.00 XXX   0 X    0.21 ASP1081 A",
        "ASP1045 A                                            0.00 XXX   0 X    0.00 XXX   0 X   -0.44 HIS1083 A",
        "ASP1045 A                                            0.00 XXX   0 X    0.00 XXX   0 X   -0.46 ARG1056 A",
    )
    parsed = parse_propka_output_text(text)
    assert set(parsed) == {("A", 1045, "ASP")}
    assert parsed[("A", 1045, "ASP")].dpka_titr == pytest.approx(-0.82)


@pytest.mark.parametrize("resseq", [12, 999, 1000, 9999, -1, -99, -100, -999])
def test_residue_number_width_does_not_change_coulombic_sum(resseq):
    label = f"GLU{resseq:4d} B"
    parsed = parse_propka_output_text(table(primary(label), continuation(label, "0.23")))
    assert set(parsed) == {("B", resseq, "GLU")}
    assert parsed[("B", resseq, "GLU")].dpka_titr == pytest.approx(0.10)


def test_small_residue_with_compact_partner_labels_keeps_correct_column():
    # Normalizing the first label alone still reads the wrong primary column.
    label = "ASP  12 A"
    parsed = parse_propka_output_text(table(
        primary(label, sidechain="ARG1056 A", backbone="LYS-100 A", coulomb="ZN   ZN A"),
        continuation(label, "0.21", partner="ASP1081 A"),
    ))
    assert parsed[("A", 12, "ASP")].dpka_titr == pytest.approx(0.08)


def test_compact_terminal_labels_are_excluded_and_orphan_continuations_ignored():
    parsed = parse_propka_output_text(table(
        primary("N+ 1002 A"), primary("C- -100 A"),
        continuation("ASP1004 A", "2.0"),
        primary("GLU  12 A"),
    ))
    assert set(parsed) == {("A", 12, "GLU")}


def test_coupled_pka_marker_and_separate_chains_preserve_legacy_behavior():
    parsed = parse_propka_output_text(table(
        primary("ASP  12 A").replace("3.66", "3.66*"),
        primary("ASP  12 B"),
    ))
    assert set(parsed) == {("A", 12, "ASP"), ("B", 12, "ASP")}
    assert all(value.dpka_titr == pytest.approx(-0.13) for value in parsed.values())
