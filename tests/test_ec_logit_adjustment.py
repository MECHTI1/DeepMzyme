"""Post-hoc logit adjustment by training class priors (synthetic; no dataset access)."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ec_clean30.logit_adjustment import TAUS, adjust, predict_ec_class, protein_class_priors, validate_tau


def test_tau_grid_is_frozen():
    assert TAUS == (-1.0, -0.5, 0.0, 0.5, 1.0)


def test_add_one_smoothing_counts_each_protein_once():
    priors = protein_class_priors([0, 0, 0, 1, 2, 2])
    assert priors.shape == (7,)
    assert priors.sum() == pytest.approx(1.0)
    assert priors[0] == pytest.approx(4 / 13)
    assert priors[6] == pytest.approx(1 / 13)
    pockets = protein_class_priors([0, 0, 0, 0, 1, 2, 2, 2], protein_ids=["a", "a", "b", "c", "d", "e", "f", "f"])
    assert np.allclose(pockets, priors)


def test_a_protein_with_two_classes_is_refused():
    with pytest.raises(ValueError, match="more than one EC class"):
        protein_class_priors([0, 1], protein_ids=["a", "a"])


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (([0, 1],), {"n_classes": 6}, "fixed 7-class"),
        (([],), {}, "non-empty"),
        (([0, 7],), {}, "0..6"),
        (([0.5, 1.0],), {}, "integers"),
        (([0, 1],), {"protein_ids": ["a"]}, "length"),
    ],
)
def test_prior_inputs_are_validated(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        protein_class_priors(*args, **kwargs)


def test_tau_zero_is_the_identity_and_tau_one_subtracts_log_priors():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 7))
    priors = protein_class_priors([0, 1, 1, 2, 3, 3, 3, 4, 5])
    assert np.array_equal(adjust(logits, priors, 0.0), logits)
    assert np.allclose(adjust(logits, priors, 1.0), logits - np.log(priors))
    assert np.allclose(adjust(logits, priors, -0.5), logits + 0.5 * np.log(priors))


def test_per_row_priors_broadcast_row_by_row():
    logits = np.zeros((2, 7))
    first, second = protein_class_priors([0, 0, 1]), protein_class_priors([5, 5, 5, 6])
    adjusted = adjust(logits, np.stack([first, second]), 1.0)
    assert np.allclose(adjusted[0], -np.log(first))
    assert np.allclose(adjusted[1], -np.log(second))


def test_predictions_ignore_row_constants_and_row_order():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(20, 7))
    priors = protein_class_priors(rng.integers(0, 7, size=50))
    base = predict_ec_class(adjust(logits, priors, 0.5))
    shifted = predict_ec_class(adjust(logits + rng.normal(size=(20, 1)) * 10, priors, 0.5))
    order = rng.permutation(20)
    assert np.array_equal(base, shifted)
    assert np.array_equal(predict_ec_class(adjust(logits[order], priors, 0.5)), base[order])


def test_positive_tau_flips_a_prediction_toward_a_rare_class():
    priors = protein_class_priors([2] * 40 + [4] * 2)
    logits = np.full(7, -5.0)
    logits[2], logits[4] = 1.0, 0.5
    assert predict_ec_class(logits) == 2
    assert predict_ec_class(adjust(logits, priors, 0.5)) == 4
    assert predict_ec_class(adjust(logits, priors, -1.0)) == 2


def test_exact_ties_go_to_the_lowest_class_index():
    logits = np.array([[0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0]])
    assert predict_ec_class(logits).tolist() == [1]


@pytest.mark.parametrize("tau", [0.25, 2.0, -0.75])
def test_tau_outside_the_frozen_grid_is_refused(tau):
    with pytest.raises(ValueError, match="frozen values"):
        validate_tau(tau)
    with pytest.raises(ValueError, match="frozen values"):
        adjust(np.zeros((1, 7)), protein_class_priors([0]), tau)


@pytest.mark.parametrize(
    ("logits", "priors", "message"),
    [
        (np.zeros((2, 6)), np.full(7, 1 / 7), "logits must have shape"),
        (np.zeros((2, 7)), np.full(6, 1 / 6), "priors must have shape"),
        (np.zeros((2, 7)), np.r_[0.0, np.full(6, 1 / 6)], "finite and positive"),
        (np.zeros((2, 7)), np.full(7, 0.2), "sum to 1"),
        (np.full((1, 7), np.nan), np.full(7, 1 / 7), "finite"),
    ],
)
def test_adjustment_inputs_are_validated(logits, priors, message):
    with pytest.raises(ValueError, match=message):
        adjust(logits, priors, 1.0)


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (np.full((1, 7), np.nan), "finite"),
        (np.array([[0.0, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]]), "finite"),
        (np.zeros((2, 3, 7)), "must have shape"),
        (np.zeros((2, 6)), "must have shape"),
    ],
)
def test_prediction_inputs_are_validated(logits, message):
    with pytest.raises(ValueError, match=message):
        predict_ec_class(logits)


def test_module_does_not_import_torch():
    src = Path(__file__).resolve().parents[1] / "src"
    code = f"import sys; sys.path.insert(0, {str(src)!r}); import ec_clean30.logit_adjustment; print('torch' in sys.modules)"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"
