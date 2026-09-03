import numpy as np
import pytest

from keplerbench.metrics import angular_error, normalized_transfer_auc


def test_angular_error_known_quarter_turn():
    truth = np.array([[1.0, 0.0, 0.0]])
    prediction = np.array([[0.0, 1.0, 0.0]])
    assert angular_error(truth, prediction)[0] == pytest.approx(np.pi / 2)


def test_transfer_auc_constant_curve():
    assert normalized_transfer_auc(np.array([1, 2, 8]), np.array([0.4, 0.4, 0.4])) == pytest.approx(0.4)
