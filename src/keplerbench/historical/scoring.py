"""Scorer for the historical Mars sub-benchmark.

Predictions are declinations in degrees at the epochs of one split of the
rebuilt table, matched on `epoch_index`. The statistic is the median absolute
residual in arcmin with a percentile bootstrap interval over epochs, the
procedure of the paper's pilot: the generator is seeded inside the call, the
resampled medians are taken over `resamples` draws with replacement, and the
2.5 and 97.5 percentiles are reported. A median is quoted only on at least
`min_n` epochs; below that the sorted absolute residuals are given instead,
which is how the paper reports the smaller observing campaigns.

An optional `uncertainty_deg` column carries the entry's stated predictive
uncertainty, and the scorer reports the fraction of residuals inside two of
them.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from ..metrics import data_blind_verdict

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260829
MIN_N_FOR_MEDIAN = 10
GROSS_ARCMIN = 60.0
NOISE_FLOOR_ARCMIN = 2.0


def bootstrap_median_ci(values: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> list[float]:
    """Percentile bootstrap 95 percent interval of the median, seeded inside the call."""

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(resamples, len(values)))
    medians = np.median(values[index], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return [float(low), float(high)]


def residual_statistics(residual_arcmin: np.ndarray, resamples: int, seed: int, min_n: int = MIN_N_FOR_MEDIAN) -> dict[str, object]:
    residual = np.asarray(residual_arcmin, dtype=float)
    absolute = np.abs(residual)
    out: dict[str, object] = {"n": int(residual.size)}
    if residual.size == 0:
        return out
    out.update({
        "p90_arcmin": float(np.percentile(absolute, 90)),
        "rms_arcmin": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_arcmin": float(np.max(absolute)),
        "n_gross": int(np.sum(absolute > GROSS_ARCMIN)),
        "median_signed_arcmin": float(np.median(residual)),
    })
    if residual.size >= min_n:
        out["median_abs_arcmin"] = float(np.median(absolute))
        out["median_abs_ci95"] = bootstrap_median_ci(absolute, resamples, seed)
    else:
        out["abs_residuals_sorted_arcmin"] = [float(x) for x in np.sort(absolute)]
        out["note"] = f"n = {residual.size} is below the reporting minimum of {min_n}; sorted absolute residuals are given instead of a median"
    return out


def read_truth(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"epoch_index", "jd", "year", "declination_deg", "split"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("truth table must have epoch_index, jd, year, declination_deg and split columns")
    return rows


def read_predictions(path: Path) -> dict[int, tuple[float, float | None]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"epoch_index", "declination_deg"} <= set(rows[0]):
        raise ValueError("prediction file must have epoch_index and declination_deg columns")
    out: dict[int, tuple[float, float | None]] = {}
    for row in rows:
        uncertainty = row.get("uncertainty_deg")
        out[int(row["epoch_index"])] = (float(row["declination_deg"]), float(uncertainty) if uncertainty not in (None, "") else None)
    return out


def score_declinations(truth_rows: list[dict[str, str]], predictions: dict[int, tuple[float, float | None]], split: str = "heldout",
                       resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, object]:
    rows = [row for row in truth_rows if row["split"] == split]
    if not rows:
        raise ValueError(f"no epochs with split {split!r}")
    missing = [row["epoch_index"] for row in rows if int(row["epoch_index"]) not in predictions]
    if missing:
        raise ValueError(f"predictions missing for {len(missing)} epochs of split {split!r}, first {missing[:5]}")
    truth = np.array([float(row["declination_deg"]) for row in rows])
    predicted = np.array([predictions[int(row["epoch_index"])][0] for row in rows])
    years = np.array([int(row["year"]) for row in rows])
    residual = (predicted - truth) * 60.0
    result: dict[str, object] = {
        "split": split, "statistic": "median absolute residual in arcmin",
        "bootstrap": {"resamples": resamples, "seed": seed, "kind": "percentile bootstrap of the median over epochs"},
        **residual_statistics(residual, resamples, seed),
        "by_campaign": {str(year): residual_statistics(residual[years == year], resamples, seed) for year in sorted(set(years.tolist()))},
    }
    stated = [predictions[int(row["epoch_index"])][1] for row in rows]
    if all(value is not None for value in stated):
        sigma = np.array(stated, dtype=float) * 60.0
        result["stated_uncertainty"] = {
            "median_arcmin": float(np.median(sigma)),
            "coverage_2sigma": float(np.mean(np.abs(residual) <= 2.0 * sigma)),
        }
    return result


def evaluate_historical(truth_path: Path, predictions_path: Path, split: str = "heldout", resamples: int = BOOTSTRAP_RESAMPLES,
                        seed: int = BOOTSTRAP_SEED, data_blind_predictions_path: Path | None = None,
                        noise_floor_arcmin: float = NOISE_FLOOR_ARCMIN) -> dict[str, object]:
    truth_rows = read_truth(truth_path)
    result = score_declinations(truth_rows, read_predictions(predictions_path), split, resamples, seed)
    if data_blind_predictions_path is not None:
        blind = score_declinations(truth_rows, read_predictions(data_blind_predictions_path), split, resamples, seed)
        result["data_blind_control"] = data_blind_verdict(result.get("median_abs_arcmin"), blind.get("median_abs_arcmin"), noise_floor_arcmin, "arcmin")
        result["data_blind_control"]["data_blind_run"] = blind
    return result
