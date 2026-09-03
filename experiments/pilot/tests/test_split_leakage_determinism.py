"""Split definition, leakage, determinism, and the horizon-bucket rules.
"""

import os

import numpy as np
import pytest

import constants as S


# ------------------------------------------------------------------ split
def test_F1_split_jd_is_exact(ephem, built):
    assert float(ephem.julian_calendar_to_jd(S.SPLIT_YEAR, 1, 1)) == S.SPLIT_JD
    assert built["split_jd"] == S.SPLIT_JD


def test_F2_no_epoch_lands_on_the_wrong_side(split):
    assert np.all(split["jd_tr"] < S.SPLIT_JD)
    assert np.all(split["jd_te"] >= S.SPLIT_JD)
    assert split["train_end"] == S.TRAIN_END
    assert split["jd_ref"] == pytest.approx(S.JD_REF, abs=1e-5)


def test_epoch_counts(built, split):
    assert built["meta"]["n_epochs"] == S.N_EPOCHS
    assert built["meta"]["n_dropped_spread"] == S.N_DROPPED_SPREAD
    assert int(split["tr_all"].sum()) == S.N_TRAIN_NO_DELETION
    assert int(split["te_all"].sum()) == S.N_TEST_NO_DELETION


def test_horizon_is_measured_from_the_end_of_training(split):
    h = split["horizon"]
    assert h.min() > 0.0
    assert 2200 < h.max() < 2350
    assert h.min() < 10.0, "the split cuts an observing campaign; say so"


def test_horizon_buckets_are_synodic_and_populated(split):
    """The superseded calendar bins produce n = 1, 1, 3, 81. A median
    absolute error on n = 1 is not a statistic."""
    h = split["horizon"]
    counts = [int(((h >= lo) & (h < hi)).sum())
              for lo, hi in S.HORIZON_BUCKETS_SYNODIC]
    assert counts[0] >= S.MIN_N_FOR_MEDIAN, counts
    assert counts[1] >= S.MIN_N_FOR_MEDIAN, counts
    assert sum(counts) == len(h), f"{counts} does not cover all {len(h)} epochs"
    assert 0 < counts[2] < S.MIN_N_FOR_MEDIAN, (
        f"bucket H3 holds {counts[2]} epochs. The reporting rule records n = 7 "
        "there and forbids a headline median below n = 10, so this bucket is populated "
        "but unquotable; no median is enforced by "
        "test_no_median_is_quoted_on_a_bucket_below_the_minimum_n")


def test_old_bins_are_unusable(split, results):
    """Documentation of why the bins changed: the superseded bins put one
    observation in a bucket, and a median on one observation is not a
    statistic. The counts stay on the record; the medians do not."""
    h = split["horizon"]
    old = [(0, 30), (30, 90), (90, 365)]
    counts = [int(((h >= lo) & (h < hi)).sum()) for lo, hi in old]
    assert min(counts) < S.MIN_N_FOR_MEDIAN
    kept = results["data"]["horizon_counts_superseded_bins"]
    assert kept["0-30d"] == counts[0] and kept["30-90d"] == counts[1], kept
    assert min(kept.values()) < S.MIN_N_FOR_MEDIAN, (
        "the superseded bins must stay on the record with their counts, "
        "because those counts are why the reporting rule replaced them")


def test_test_window_is_three_campaigns(ephem, split):
    """The paper must not describe the test period as a uniform 1594 to 1600
    span. It is 1594 n=5, 1595 n=54, 1596 n=20, 1600 n=7, with nothing deleted."""
    years = []
    for x in split["jd_te"]:
        y = 1580
        while float(ephem.julian_calendar_to_jd(y + 1, 1, 1)) <= x:
            y += 1
        years.append(y)
    u, c = np.unique(years, return_counts=True)
    got = dict(zip(u.tolist(), c.tolist()))
    assert got == {1594: 5, 1595: 54, 1596: 20, 1600: 7}, got


# ---------------------------------------------------------------- leakage
def test_F3_frequency_selection_touches_training_data_only(models, split,
                                                           monkeypatch):
    """Instrument greedy_order and fit_harmonic: record the largest jd that any
    selection call ever sees. It must stay below split_jd."""
    seen = []
    real_greedy = models.greedy_order
    real_fit = models.fit_harmonic

    def spy_greedy(t_tr, y_tr, lattice, n_max):
        seen.append(float(np.max(t_tr)))
        return real_greedy(t_tr, y_tr, lattice, n_max)

    def spy_fit(t_tr, y_tr, freqs):
        seen.append(float(np.max(t_tr)))
        return real_fit(t_tr, y_tr, freqs)

    monkeypatch.setattr(models, "greedy_order", spy_greedy)
    monkeypatch.setattr(models, "fit_harmonic", spy_fit)

    lattice = models.frequency_lattice()
    t_tr = split["jd_tr"] - split["jd_ref"]
    order = models.greedy_order(t_tr, split["dec_tr"], lattice, 4)
    models.fit_harmonic(t_tr, split["dec_tr"], [f for f, _, _, _ in order])
    assert seen, "the spy never fired"
    limit = S.SPLIT_JD - split["jd_ref"]
    assert max(seen) < limit, (
        f"a selection call saw t = {max(seen)}, past the split at {limit}")


def test_F4_preprocessing_does_not_delete_test_epochs(built, split):
    """The smoothness flag must never remove a test-period observation using
    test-period neighbours. The headline analysis deletes nothing."""
    flag = built["outlier"]
    n_test_deleted = int((flag & split["te_all"]).sum())
    assert n_test_deleted == 0, (
        f"{n_test_deleted} test epoch(s) deleted by preprocessing; either drop "
        "the deletion or compute the flag on the training window only")


def test_cross_validation_cuts_stay_inside_training(split):
    """The CV protocol in run_pilot builds cuts from t_tr only. Reproduce the
    arithmetic and check the last validation window still ends inside train."""
    t_tr = split["jd_tr"] - split["jd_ref"]
    lo, hi = t_tr.min(), t_tr.max()
    for q in (0.50, 0.60, 0.70, 0.80, 0.90):
        c = lo + q * (hi - lo)
        end = c + 0.20 * (hi - lo)
        assert c < hi
        assert min(end, hi) + split["jd_ref"] <= S.SPLIT_JD


# ------------------------------------------------------------ determinism
def test_F5_data_build_is_deterministic():
    import pilot_data as data
    if not os.path.exists(S.CSV):
        pytest.skip("epoch table not rebuilt")
    a = data.build(split_year=S.SPLIT_YEAR)
    b = data.build(split_year=S.SPLIT_YEAR)
    for k in ("jd", "dec", "nrep", "outlier", "keep"):
        assert np.array_equal(a[k], b[k]), k
    assert a["split_jd"] == b["split_jd"]


@pytest.mark.slow
def test_F6_kepler_fit_is_reproducible_for_a_fixed_seed(models, split):
    a = models.fit_kepler(split["jd_tr"], split["dec_tr"], split["jd_ref"],
                          n_restarts=4, seed=7)
    b = models.fit_kepler(split["jd_tr"], split["dec_tr"], split["jd_ref"],
                          n_restarts=4, seed=7)
    assert np.array_equal(a["theta"], b["theta"])
    assert a["rms"] == b["rms"]


def test_gp_fit_is_reproducible(models, split):
    t_tr = split["jd_tr"] - split["jd_ref"]
    a = models.fit_gp(t_tr, split["dec_tr"])
    b = models.fit_gp(t_tr, split["dec_tr"])
    assert np.array_equal(a["p"], b["p"])
    assert a["nll"] == b["nll"]


def test_frequency_lattice_is_deterministic_and_well_formed(models):
    a = models.frequency_lattice()
    b = models.frequency_lattice()
    assert a == b
    freqs = [f for f, _, _, _ in a]
    assert freqs == sorted(set(freqs), key=freqs.index)
    assert all(f > 0 for f in freqs)
    assert all(1.0 / f >= 25.0 - 1e-9 for f in freqs)
    periods = sorted(1.0 / f for f, _, _, _ in a[:2])
    assert periods[0] == pytest.approx(models.P_SID, abs=1e-6)
    assert periods[1] == pytest.approx(models.P_SYN, abs=1e-6)


def test_F8_seeds_and_provenance_are_recorded(results):
    """Without a provenance block the run is not reproducible by anyone else,
    and the paper cannot state what produced its numbers."""
    assert "provenance" in results, "results.json has no provenance block"
    p = results["provenance"]
    for k in ("input_sha256", "transcription_sha256", "numpy", "scipy", "seeds", "utc_timestamp"):
        assert k in p, f"provenance is missing {k}"
    assert p["transcription_sha256"] == S.XLS_SHA256
    assert p["seeds"]["bootstrap"] == S.BOOTSTRAP_SEED
    assert p["seeds"]["bootstrap_resamples"] == S.BOOTSTRAP_N
