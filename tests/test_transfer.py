import numpy as np
import pytest

from keplerbench.synthetic import WorldConfig, generate_observations
from keplerbench.transfer import evaluate_transfer, _exposed_and_heldout


def _constant_adapter(exposed):
    directions = np.array([[float(row[f"planet_dir_{axis}"]) for axis in "xyz"] for row in exposed])
    mean = directions.mean(axis=0)
    mean = mean / np.linalg.norm(mean)

    def predict(times):
        return np.tile(mean, (len(times), 1))

    return predict


def test_exposed_are_the_chronologically_first_k():
    rows = generate_observations(WorldConfig(sample_count=50), seed=1)
    exposed, heldout = _exposed_and_heldout(rows, 10)
    assert len(exposed) == 10
    assert max(float(row["time"]) for row in exposed) < min(float(row["time"]) for row in heldout)
    assert all(row["split"] == "test" for row in heldout)


def test_k_beyond_training_split_is_refused():
    rows = generate_observations(WorldConfig(sample_count=50), seed=1)
    with pytest.raises(ValueError):
        _exposed_and_heldout(rows, 1000)


def test_evaluate_transfer_reports_curve_and_auc():
    worlds = [generate_observations(WorldConfig(world_id=f"w{i}", sample_count=60), seed=i) for i in range(2)]
    result = evaluate_transfer(_constant_adapter, worlds, [4, 8, 16])
    assert result["k_values"] == [4, 8, 16]
    assert len(result["worlds"]) == 2
    assert len(result["mean_curve_arcminutes"]) == 3
    assert np.isfinite(result["auc_of_mean_curve_radians"])
