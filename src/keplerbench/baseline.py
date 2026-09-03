"""A deliberately non-physical harmonic baseline for prediction."""

from __future__ import annotations

import numpy as np


def _features(times: np.ndarray, order: int, period: float) -> np.ndarray:
    columns = [np.ones_like(times), times]
    for harmonic in range(1, order + 1):
        phase = 2.0 * np.pi * harmonic * times / period
        columns.extend([np.sin(phase), np.cos(phase)])
    return np.column_stack(columns)


def harmonic_predict(train_times: np.ndarray, train_directions: np.ndarray, test_times: np.ndarray, order: int = 8, ridge: float = 1e-6) -> np.ndarray:
    if len(train_times) < 2:
        raise ValueError("at least two training observations are required")
    period = float(np.ptp(train_times))
    if period <= 0:
        raise ValueError("training times must span a nonzero interval")
    design = _features(train_times, order, period)
    weights = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ train_directions)
    prediction = _features(test_times, order, period) @ weights
    norms = np.linalg.norm(prediction, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise FloatingPointError("baseline produced a zero direction")
    return prediction / norms
