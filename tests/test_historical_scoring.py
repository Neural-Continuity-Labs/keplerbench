import csv
import json
from pathlib import Path

import numpy as np

from keplerbench.cli import main
from keplerbench.historical.scoring import bootstrap_median_ci, read_predictions, read_truth, score_declinations


def _table(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    path = tmp_path / "mars.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch_index", "jd", "year", "declination_deg", "n_repeats", "split"])
        writer.writeheader()
        index = 0
        for year, count, split in ((1590, 30, "train"), (1594, 5, "heldout"), (1595, 40, "heldout")):
            for _ in range(count):
                writer.writerow({"epoch_index": index, "jd": 2299000.5 + index, "year": year, "declination_deg": float(rng.uniform(-25, 25)), "n_repeats": 1, "split": split})
                index += 1
    return path


def _predictions(truth: Path, path: Path, offset_arcmin: float, uncertainty_deg: float | None = None) -> Path:
    rows = list(csv.DictReader(truth.open()))
    with path.open("w", newline="") as handle:
        fields = ["epoch_index", "declination_deg"] + (["uncertainty_deg"] if uncertainty_deg is not None else [])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {"epoch_index": row["epoch_index"], "declination_deg": float(row["declination_deg"]) + offset_arcmin / 60.0}
            if uncertainty_deg is not None:
                record["uncertainty_deg"] = uncertainty_deg
            writer.writerow(record)
    return path


def test_bootstrap_matches_the_loop_form_used_for_the_aifeynman_row():
    """The pilot's vectorized bootstrap and the AI Feynman scorer's loop draw the same indices."""
    rng = np.random.default_rng(9)
    values = np.abs(rng.normal(size=50))
    loop = np.random.default_rng(23)
    boots = [np.median(loop.choice(values, len(values), replace=True)) for _ in range(2000)]
    expected = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    assert bootstrap_median_ci(values, resamples=2000, seed=23) == expected


def test_score_declinations_reports_median_campaigns_and_reporting_rule(tmp_path: Path):
    truth = _table(tmp_path)
    predictions = _predictions(truth, tmp_path / "p.csv", offset_arcmin=3.0, uncertainty_deg=0.1)
    result = score_declinations(read_truth(truth), read_predictions(predictions), "heldout", resamples=200, seed=1)
    assert result["n"] == 45
    assert abs(result["median_abs_arcmin"] - 3.0) < 1e-9
    low, high = result["median_abs_ci95"]
    assert low <= 3.0 + 1e-9 and high >= 3.0 - 1e-9, "a constant residual gives a degenerate interval at the median"
    assert result["by_campaign"]["1595"]["n"] == 40 and "median_abs_arcmin" in result["by_campaign"]["1595"]
    small = result["by_campaign"]["1594"]
    assert small["n"] == 5 and "median_abs_arcmin" not in small and len(small["abs_residuals_sorted_arcmin"]) == 5
    assert result["stated_uncertainty"]["coverage_2sigma"] == 1.0


def test_evaluate_historical_command_applies_the_data_blind_rule(tmp_path: Path, capsys):
    truth = _table(tmp_path)
    entry = _predictions(truth, tmp_path / "entry.csv", offset_arcmin=1.0)
    same = _predictions(truth, tmp_path / "same.csv", offset_arcmin=1.0)
    worse = _predictions(truth, tmp_path / "worse.csv", offset_arcmin=40.0)
    main(["evaluate-historical", "--truth", str(truth), "--predictions", str(entry), "--data-blind-predictions", str(same), "--resamples", "100"])
    assert json.loads(capsys.readouterr().out)["data_blind_control"]["credit"] is False
    main(["evaluate-historical", "--truth", str(truth), "--predictions", str(entry), "--data-blind-predictions", str(worse), "--resamples", "100"])
    out = json.loads(capsys.readouterr().out)
    assert out["data_blind_control"]["credit"] is True
    assert out["data_blind_control"]["data_blind_run"]["median_abs_arcmin"] > 39.0
