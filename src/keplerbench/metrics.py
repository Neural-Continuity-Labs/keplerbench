"""Architecture-independent benchmark metrics."""

from __future__ import annotations

import numpy as np


def angular_error(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Great-circle angular error in radians for corresponding vectors."""

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("truth and prediction must have matching shape (n, 3)")
    truth_norm = np.linalg.norm(truth, axis=1, keepdims=True)
    prediction_norm = np.linalg.norm(prediction, axis=1, keepdims=True)
    if np.any(truth_norm == 0) or np.any(prediction_norm == 0):
        raise ValueError("directions must be nonzero")
    cosine = np.sum((truth / truth_norm) * (prediction / prediction_norm), axis=1)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def summarize_angles(errors: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors, dtype=float)
    return {
        "mean_radians": float(np.mean(errors)),
        "median_radians": float(np.median(errors)),
        "p95_radians": float(np.quantile(errors, 0.95)),
        "mean_arcminutes": float(np.mean(errors) * 180.0 / np.pi * 60.0),
    }


def normalized_transfer_auc(evidence_counts: np.ndarray, errors: np.ndarray) -> float:
    """Area under an error-vs-log-evidence curve, normalized to [0, 1] width."""

    counts = np.asarray(evidence_counts, dtype=float)
    values = np.asarray(errors, dtype=float)
    if counts.ndim != 1 or values.shape != counts.shape or len(counts) < 2:
        raise ValueError("evidence_counts and errors must be equal 1D arrays of length >= 2")
    if np.any(counts <= 0) or np.any(np.diff(counts) <= 0):
        raise ValueError("evidence_counts must be positive and strictly increasing")
    x = np.log(counts)
    widths = np.diff(x)
    area = np.sum(widths * (values[:-1] + values[1:]) / 2.0)
    return float(area / (x[-1] - x[0]))


def data_blind_verdict(entry_score: float | None, data_blind_score: float | None, noise_floor: float, unit: str) -> dict[str, object]:
    """Apply the protocol's data-blind rule to a pair of scores in the same unit.

    Both scores are reported. The entry earns credit only when the data-blind
    run scores worse than the entry by more than the sub-benchmark's noise
    floor; a difference inside the floor means the score was carried by prior
    knowledge rather than extracted from the evidence.
    """

    if entry_score is None or data_blind_score is None:
        return {"entry": entry_score, "data_blind": data_blind_score, "noise_floor": noise_floor, "unit": unit, "credit": None,
                "note": "a score is missing, so the rule cannot be applied"}
    difference = float(data_blind_score - entry_score)
    return {"entry": float(entry_score), "data_blind": float(data_blind_score), "difference": difference,
            "noise_floor": float(noise_floor), "unit": unit, "credit": bool(difference > noise_floor)}
