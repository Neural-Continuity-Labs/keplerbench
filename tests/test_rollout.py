import numpy as np
import pytest

from keplerbench.rollout import evaluate_rollout, window_consistency
from keplerbench.synthetic import WorldConfig, generate_observations


class _FrozenSubmission:
    """Holds the direction seen at the end of the window; a pure control."""

    state_dimension = 3

    def encode(self, window):
        last = window[-1]
        return np.array([float(last[f"planet_dir_{axis}"]) for axis in "xyz"]), None

    def step(self, state, dt, context):
        return state

    def decode(self, state, context):
        return state / np.linalg.norm(state)


def test_evaluate_rollout_scores_buckets_and_failure():
    rows = generate_observations(WorldConfig(sample_count=120, duration=20.0), seed=3)
    report = evaluate_rollout(_FrozenSubmission(), rows, window_end=5.0,
                              horizon_edges=[5.0, 15.0], threshold_radians=np.deg2rad(1.0))
    assert report["state_dimension"] == 3
    assert report["buckets"]
    assert report["failure_horizon"] is not None


def test_declared_state_dimension_is_enforced():
    class Lying(_FrozenSubmission):
        state_dimension = 5

    rows = generate_observations(WorldConfig(sample_count=60), seed=3)
    with pytest.raises(ValueError):
        evaluate_rollout(Lying(), rows, window_end=5.0, horizon_edges=[5.0], threshold_radians=0.1)


def test_window_consistency_requires_shared_end_and_reports_divergence():
    rows = generate_observations(WorldConfig(sample_count=120, duration=20.0), seed=3)
    report = window_consistency(_FrozenSubmission(), rows, [(0.0, 5.0), (2.5, 5.0)], future_count=20)
    assert report["future_count"] == 20
    assert report["max_divergence_arcminutes"] >= 0.0
    with pytest.raises(ValueError):
        window_consistency(_FrozenSubmission(), rows, [(0.0, 5.0), (2.5, 6.0)])
