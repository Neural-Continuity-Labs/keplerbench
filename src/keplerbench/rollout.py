"""The local dynamical-law rollout harness.

The protocol's third capability test: a submission provides an encoder that
constructs a fixed-dimensional state from an initialization window, a
transition rule mapping state and time increment to the next state, and a
decoder from state to public observations. After initialization the harness
removes the observation history and rolls the same transition rule forward,
scoring long-horizon angular error and the time to first exceed an error
threshold. The harness also checks window consistency: initializing the same
physical instant from different observation windows should lead to the same
predicted future, which a stored whole-trajectory model inside the state does
not generally satisfy.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .io import times_and_directions
from .metrics import angular_error

Rows = Sequence[dict[str, object]]


class RolloutSubmission(Protocol):
    """What a local-law entry provides to the harness."""

    state_dimension: int

    def encode(self, window: Rows) -> tuple[np.ndarray, object]:
        """Construct (state, context) from an initialization window.

        The state must be a vector of length state_dimension valid at the time
        of the window's last observation. The context holds system-specific
        quantities that the transition rule may read but not rewrite.
        """

    def step(self, state: np.ndarray, dt: float, context: object) -> np.ndarray:
        """Advance the state by dt using the one frozen transition rule."""

    def decode(self, state: np.ndarray, context: object) -> np.ndarray:
        """Return the unit planet direction (3,) observed from the state."""


def _window_and_future(rows: Rows, window_end: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ordered = sorted(rows, key=lambda row: float(row["time"]))
    window = [row for row in ordered if float(row["time"]) <= window_end]
    future = [row for row in ordered if float(row["time"]) > window_end]
    if len(window) < 2 or not future:
        raise ValueError("the window and the future must both be non-empty")
    return window, future


def _rollout_errors(submission: RolloutSubmission, window: Rows, future: Rows) -> tuple[np.ndarray, np.ndarray]:
    state, context = submission.encode(window)
    state = np.asarray(state, dtype=float)
    if state.shape != (submission.state_dimension,):
        raise ValueError(f"encoder returned shape {state.shape}, declared {submission.state_dimension}")
    times, truth = times_and_directions(future)
    current = float(window[-1]["time"])
    predictions = np.empty((len(times), 3))
    for index, time in enumerate(times):
        state = submission.step(state, float(time) - current, context)
        current = float(time)
        predictions[index] = submission.decode(state, context)
    return times, angular_error(truth, predictions)


def evaluate_rollout(
    submission: RolloutSubmission,
    rows: Rows,
    window_end: float,
    horizon_edges: Sequence[float],
    threshold_radians: float,
) -> dict[str, object]:
    """Roll one submission forward and score it against the withheld future.

    Horizons are measured from the end of the initialization window, and
    horizon_edges are the bucket boundaries. The failure time is the first
    horizon at which the angular error exceeds the threshold.
    """

    window, future = _window_and_future(rows, window_end)
    times, errors = _rollout_errors(submission, window, future)
    horizons = times - float(window[-1]["time"])
    edges = [0.0] + sorted(float(edge) for edge in horizon_edges)
    buckets: dict[str, dict[str, float]] = {}
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (horizons > low) & (horizons <= high)
        if not mask.any():
            continue
        bucket = errors[mask]
        buckets[f"{low:g}-{high:g}"] = {
            "count": int(mask.sum()),
            "median_arcminutes": float(np.median(bucket) * 180.0 / np.pi * 60.0),
            "mean_arcminutes": float(np.mean(bucket) * 180.0 / np.pi * 60.0),
        }
    exceed = np.nonzero(errors > threshold_radians)[0]
    return {
        "world_id": str(rows[0]["world_id"]),
        "state_dimension": submission.state_dimension,
        "window_end": window_end,
        "rollout_span": float(horizons[-1]),
        "buckets": buckets,
        "failure_horizon": float(horizons[exceed[0]]) if exceed.size else None,
        "threshold_arcminutes": threshold_radians * 180.0 / np.pi * 60.0,
        "final_error_arcminutes": float(errors[-1] * 180.0 / np.pi * 60.0),
    }


def window_consistency(
    submission: RolloutSubmission,
    rows: Rows,
    windows: Sequence[tuple[float, float]],
    future_count: int = 50,
) -> dict[str, object]:
    """Initialize from different windows ending at the same instant and compare.

    All windows must share their end time. Each initialization is rolled to the
    same future observation times and the divergence is the angular difference
    between the predicted directions of each pair of initializations. A local
    law in a near-Markovian state keeps the divergence near the noise level; a
    trajectory model stored in the state diverges with the window it was fitted
    on.
    """

    ends = {float(end) for _, end in windows}
    if len(ends) != 1:
        raise ValueError("all windows must end at the same instant")
    end = ends.pop()
    ordered = sorted(rows, key=lambda row: float(row["time"]))
    future = [row for row in ordered if float(row["time"]) > end][:future_count]
    if not future:
        raise ValueError("no future observations beyond the shared window end")
    times = np.array([float(row["time"]) for row in future])
    predictions = []
    for start, _ in windows:
        window = [row for row in ordered if start <= float(row["time"]) <= end]
        if len(window) < 2:
            raise ValueError(f"window starting at {start} holds fewer than two observations")
        state, context = submission.encode(window)
        state = np.asarray(state, dtype=float)
        current = end
        predicted = np.empty((len(times), 3))
        for index, time in enumerate(times):
            state = submission.step(state, float(time) - current, context)
            current = float(time)
            predicted[index] = submission.decode(state, context)
        predictions.append(predicted)
    divergences = []
    for first in range(len(predictions)):
        for second in range(first + 1, len(predictions)):
            divergences.append(angular_error(predictions[first], predictions[second]))
    stacked = np.concatenate(divergences)
    return {
        "world_id": str(rows[0]["world_id"]),
        "windows": [[float(start), float(end_)] for start, end_ in windows],
        "future_count": len(times),
        "median_divergence_arcminutes": float(np.median(stacked) * 180.0 / np.pi * 60.0),
        "max_divergence_arcminutes": float(np.max(stacked) * 180.0 / np.pi * 60.0),
    }
