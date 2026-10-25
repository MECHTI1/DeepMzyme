"""Fixed-denominator EC-class metrics (synthetic; no dataset access)."""
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.ec_class_metrics import ec_class_metrics, ec_class_metrics_from_logits


def test_all_seven_classes_present_matches_sklearn():
    rng = np.random.default_rng(0)
    true = np.repeat(np.arange(7), 5)
    pred = np.where(rng.random(true.size) < 0.7, true, rng.integers(0, 7, true.size))
    metrics = ec_class_metrics(true, pred)
    assert metrics["seven_class_balanced_accuracy"] == pytest.approx(balanced_accuracy_score(true, pred))
    assert metrics["seven_class_macro_f1"] == pytest.approx(
        f1_score(true, pred, labels=list(range(7)), average="macro", zero_division=0)
    )
    primary = true < 6
    recalls = [np.mean(pred[true == c] == c) for c in range(6)]
    assert metrics["ec_classes_1_6_balanced_accuracy"] == pytest.approx(np.mean(recalls))
    assert metrics["ec_classes_1_6_n_present"] == 6 and metrics["seven_class_full_support"]
    assert metrics["accuracy"] == pytest.approx(np.mean(true == pred))
    assert int(np.sum(metrics["confusion_matrix"])) == true.size and primary.any()


def test_wrong_class7_prediction_counts_as_error_but_class7_is_not_averaged():
    true = np.array([0, 0, 1, 1, 6])
    pred = np.array([0, 6, 1, 1, 6])
    metrics = ec_class_metrics(true, pred)
    assert metrics["ec_classes_1_6_balanced_accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["ec_classes_1_6_n_present"] == 2
    assert metrics["ec_classes_1_6_label"] == "2-class BA"
    assert metrics["seven_class_balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert metrics["seven_class_n_present"] == 3 and not metrics["seven_class_full_support"]


def test_absent_class_is_counted_not_averaged():
    true = np.array([0, 1, 2, 3, 5])
    pred = np.array([0, 1, 2, 4, 5])
    metrics = ec_class_metrics(true, pred)
    assert metrics["per_class_support"][4] == 0
    assert metrics["per_class_recall"][4] == 0.0 and metrics["per_class_precision"][4] == 0.0
    assert metrics["ec_classes_1_6_n_present"] == 5
    assert metrics["ec_classes_1_6_balanced_accuracy"] == pytest.approx(4 / 5)


def test_weights_scale_counts():
    true = np.array([0, 0, 1])
    pred = np.array([0, 1, 1])
    weighted = ec_class_metrics(true, pred, weights=[3.0, 1.0, 1.0])
    assert weighted["per_class_recall"][0] == pytest.approx(0.75)
    assert weighted["total_weight"] == pytest.approx(5.0)
    duplicated = ec_class_metrics([0, 0, 0, 0, 1], [0, 0, 0, 1, 1])
    assert weighted["ec_classes_1_6_balanced_accuracy"] == pytest.approx(
        duplicated["ec_classes_1_6_balanced_accuracy"]
    )


def test_from_logits_uses_argmax():
    logits = np.zeros((3, 7))
    logits[0, 2] = 1.0
    logits[1, 6] = 1.0
    logits[2, 0] = 1.0
    metrics = ec_class_metrics_from_logits(logits, [2, 1, 0])
    assert metrics["accuracy"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("true", "pred", "kwargs", "message"),
    [
        ([], [], {}, "at least one"),
        ([0, 1], [0], {}, "differ in length"),
        ([0, 7], [0, 1], {}, "outside"),
        ([0, -1], [0, 1], {}, "outside"),
        ([0, 1], [0, 1], {"weights": [1.0]}, "weights must match"),
        ([0, 1], [0, 1], {"weights": [0.0, 0.0]}, "not all zero"),
        ([[0, 1]], [[0, 1]], {}, "one-dimensional"),
    ],
)
def test_invalid_inputs_raise(true, pred, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ec_class_metrics(true, pred, **kwargs)


def test_logits_shape_is_checked():
    with pytest.raises(ValueError, match="shape"):
        ec_class_metrics_from_logits(np.zeros((2, 6)), [0, 1])
    with pytest.raises(ValueError, match="finite"):
        ec_class_metrics_from_logits(np.full((1, 7), np.nan), [0])
