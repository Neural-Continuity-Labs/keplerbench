"""The global-law transfer harness.

The protocol's second capability test: a discovery procedure produces shared
structure from source systems, the shared object is frozen, and the learner
receives k observations from a held-out target system. Adaptation may estimate
target-specific quantities but must not update the frozen shared object. The
harness enforces the budget and the split; it cannot inspect an opaque
submission, so the freeze itself is part of the submission contract and the
accompanying declaration. For each k the harness measures angular prediction
error on the target's held-out observations, and the complete error-versus-k
curve is the primary result, with the normalized area under the curve against
log k as the summary.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from .io import times_and_directions
from .metrics import angular_error, normalized_transfer_auc, summarize_angles

Rows = Sequence[dict[str, object]]

# An adapter receives the k exposed observations of one target system and
# returns a predictor mapping times to unit planet directions of shape (n, 3).
Adapter = Callable[[Rows], Callable[[np.ndarray], np.ndarray]]


class TransferSubmission(Protocol):
    """What a global-law entry provides to the harness."""

    def discover(self, source_worlds: Sequence[Rows]) -> None:
        """Build the shared object from the source systems, then freeze it."""

    def adapt(self, exposed: Rows) -> Callable[[np.ndarray], np.ndarray]:
        """Estimate system-specific quantities from the exposed observations.

        Must not update anything produced by discover. Returns a predictor
        from times to unit planet directions, shape (n, 3).
        """


def _exposed_and_heldout(rows: Rows, k: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if k > len(train):
        raise ValueError(f"k={k} exceeds the {len(train)} training observations")
    train.sort(key=lambda row: float(row["time"]))
    return train[:k], test


def evaluate_transfer(
    adapt: Adapter,
    target_worlds: Sequence[Rows],
    k_values: Sequence[int],
) -> dict[str, object]:
    """Score one frozen submission on held-out target systems.

    The k exposed observations are the chronologically first k of the target's
    training split, so adaptation faces the same temporal extrapolation as the
    prediction test. Scores are the mean great-circle angular error on the
    target's test split, reported per world and per k, with the mean curve
    across worlds and its normalized area against log k as summaries.
    """

    k_values = sorted(int(k) for k in k_values)
    if len(k_values) < 2:
        raise ValueError("at least two evidence budgets are required for a curve")
    per_world: list[dict[str, object]] = []
    for rows in target_worlds:
        world_id = str(rows[0]["world_id"])
        errors: list[float] = []
        detail: dict[str, dict[str, float]] = {}
        for k in k_values:
            exposed, heldout = _exposed_and_heldout(rows, k)
            times, truth = times_and_directions(heldout)
            prediction = adapt(exposed)(times)
            summary = summarize_angles(angular_error(truth, prediction))
            errors.append(summary["mean_radians"])
            detail[str(k)] = summary
        per_world.append({
            "world_id": world_id,
            "mean_radians_by_k": dict(zip(map(str, k_values), errors)),
            "auc_radians": normalized_transfer_auc(np.array(k_values, dtype=float), np.array(errors)),
            "detail": detail,
        })
    curves = np.array([[world["mean_radians_by_k"][str(k)] for k in k_values] for world in per_world])
    mean_curve = curves.mean(axis=0)
    return {
        "k_values": k_values,
        "worlds": per_world,
        "mean_curve_radians": mean_curve.tolist(),
        "mean_curve_arcminutes": (mean_curve * 180.0 / np.pi * 60.0).tolist(),
        "median_curve_arcminutes": (np.median(curves, axis=0) * 180.0 / np.pi * 60.0).tolist(),
        "auc_of_mean_curve_radians": normalized_transfer_auc(np.array(k_values, dtype=float), mean_curve),
        "median_world_auc_radians": float(np.median([world["auc_radians"] for world in per_world])),
    }
