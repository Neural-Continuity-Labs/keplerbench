"""Ephemeris validation against unfitted modern elements.

This is the only check in the pilot that is independent of every fitted model.
If it fails, nothing downstream means anything, so the thresholds are stated
explicitly rather than left implicit in a printed sanity line.

Threshold I require: median absolute error at most 5.0 arcmin over all 354
epochs, with at least 85 percent of epochs inside 10 arcmin. Justification:
Tycho's own repeat scatter is 2 to 4 arcmin, the JPL secular series extrapolated
400 years back is good to roughly 1 to 2 arcmin on Mars, and the record carries
about 5 percent gross transcription errors, so a correct pipeline must land near
3 to 4 arcmin. Anything above 5 arcmin means a frame, unit or calendar defect.
"""

import numpy as np
import pytest

import constants as S
from conftest import medae, transit_altitude_deg


def test_C1_median_absolute_error_over_all_epochs(modern_resid):
    m = medae(modern_resid)
    assert m <= S.EPHEM_MEDAE_ALL_MAX, f"median absolute error {m:.2f} arcmin"
    assert m > 0.5, "suspiciously perfect; check the ephemeris is not fitted"


def test_C2_fraction_within_ten_arcmin(modern_resid):
    f = float(np.mean(np.abs(modern_resid) < 10.0))
    assert f >= S.EPHEM_FRAC_WITHIN_10_MIN, f"only {f:.3f} within 10 arcmin"


def test_C3_median_on_the_non_gross_subset(modern_resid, split):
    good = np.abs(modern_resid) < S.GROSS_ARCMIN
    for lab, m in (("train", split["tr_all"]), ("test", split["te_all"])):
        sel = good & m
        v = medae(modern_resid[sel])
        assert v <= S.EPHEM_MEDAE_CLEAN_MAX, f"{lab}: {v:.2f} arcmin"


def test_gross_error_census(modern_resid, split):
    """The census the paper must publish: 18 of 354 epochs exceed 60 arcmin,
    15 of them in the training window, 9 of them outright sign flips."""
    big = np.abs(modern_resid) > S.GROSS_ARCMIN
    assert 14 <= int(big.sum()) <= 22, f"{int(big.sum())} gross epochs"
    assert int((big & split["tr_all"]).sum()) >= 10
    assert float(np.max(np.abs(modern_resid))) > 500.0, (
        "the largest transcription error should be over 500 arcmin")


def test_sign_flip_subpopulation(ephem, split, modern, modern_resid):
    """A sign flip is an epoch where -dec_obs matches the prediction far better
    than +dec_obs does. There must be several, and they must be the extreme
    tail, because that is what forces a robust loss."""
    el = modern
    p = ephem.predicted_declination(split["jd"], el["a"], el["e"], el["inc"],
                                    el["node"], el["argp"], el["M0"],
                                    split["jd_ref"])
    flip = np.abs(-split["dec"] - p) < 0.2 * np.abs(split["dec"] - p)
    assert int(flip.sum()) >= 5, f"only {int(flip.sum())} sign flips found"
    assert np.all(np.abs(modern_resid[flip]) > S.GROSS_ARCMIN)


@pytest.mark.parametrize("shift", [10.0, -10.0])
def test_C4_ten_day_shift(ephem, split, modern, shift):
    el = modern
    p = ephem.predicted_declination(split["jd"] + shift, el["a"], el["e"],
                                    el["inc"], el["node"], el["argp"],
                                    el["M0"], split["jd_ref"])
    assert medae((split["dec"] - p) / ephem.ARCMIN) >= S.SHIFT_10D_MEDAE_MIN


def test_C6_light_time_matters_and_is_small(ephem, split, modern):
    el = modern
    kw = dict(a=el["a"], e=el["e"], inc=el["inc"], node=el["node"],
              argp=el["argp"], M0=el["M0"], jd_ref=split["jd_ref"])
    on = ephem.predicted_declination(split["jd"], light_time=True, **kw)
    off = ephem.predicted_declination(split["jd"], light_time=False, **kw)
    d = np.max(np.abs(on - off)) / ephem.ARCMIN
    lo, hi = S.LIGHT_TIME_EFFECT_ARCMIN
    assert lo < d < hi, f"light-time effect {d:.3f} arcmin, expected {lo} to {hi}"
    assert medae((split["dec"] - on) / ephem.ARCMIN) <= \
        medae((split["dec"] - off) / ephem.ARCMIN) + 0.05


def test_prediction_is_a_pure_function(ephem, split, modern):
    el = modern
    a = ephem.predicted_declination(split["jd"], el["a"], el["e"], el["inc"],
                                    el["node"], el["argp"], el["M0"],
                                    split["jd_ref"])
    b = ephem.predicted_declination(split["jd"], el["a"], el["e"], el["inc"],
                                    el["node"], el["argp"], el["M0"],
                                    split["jd_ref"])
    assert np.array_equal(a, b)


def test_ephemeris_is_sensitive_to_each_element(ephem, split, modern):
    """A parameter that does not move the prediction is not identifiable and
    must not be reported with a standard error. Perturb each element by a small
    amount and require a measurable change."""
    el = modern
    base = ephem.predicted_declination(split["jd"], el["a"], el["e"], el["inc"],
                                       el["node"], el["argp"], el["M0"],
                                       split["jd_ref"]) / ephem.ARCMIN
    bumps = dict(a=1e-3, e=1e-3, inc=np.radians(0.1), node=np.radians(1.0),
                 argp=np.radians(1.0), M0=np.radians(0.1))
    for k, db in bumps.items():
        p = dict(el)
        p[k] = p[k] + db
        alt = ephem.predicted_declination(split["jd"], p["a"], p["e"], p["inc"],
                                          p["node"], p["argp"], p["M0"],
                                          split["jd_ref"]) / ephem.ARCMIN
        rms = float(np.sqrt(np.mean((alt - base) ** 2)))
        assert rms > 0.05, (
            f"element {k} moves the prediction by only {rms:.4f} arcmin for a "
            f"bump of {db}; it is not identifiable from this data")


# --------------------------------------------------- refraction confound (s.6)
def test_residual_depends_on_transit_altitude(modern_resid, split):
    """This must be reported: Tycho's refraction correction leaves an
    altitude-dependent bias, so part of any few-arcminute train-to-test
    difference is instrumental."""
    from scipy import stats
    good = np.abs(modern_resid) < S.GROSS_ARCMIN
    alt = transit_altitude_deg(split["dec"])
    rho, p = stats.spearmanr(alt[good], modern_resid[good])
    assert p < S.REFRACTION_SPEARMAN_P_MAX, (
        f"altitude dependence not detected (rho={rho:.3f}, p={p:.2g}); if this "
        "ever stops holding the refraction caveat can be dropped")
    assert rho < 0, "sign: under-corrected refraction pushes low objects north"
    low = good & (alt < 15)
    high = good & (alt >= 40)
    assert float(np.median(modern_resid[low])) - \
        float(np.median(modern_resid[high])) > 3.0, (
        "the low-altitude bias should exceed 3 arcmin")


def test_train_and_test_sample_different_altitudes(modern_resid, split):
    """The instrumental bias differs between the two windows, so a
    small held-out degradation is not automatically model failure."""
    good = np.abs(modern_resid) < S.GROSS_ARCMIN
    mtr = float(np.median(modern_resid[good & split["tr_all"]]))
    mte = float(np.median(modern_resid[good & split["te_all"]]))
    assert abs(mtr - mte) > 1.0, (
        f"train median bias {mtr:.2f}, test {mte:.2f}; the caveat rests on this "
        "gap being real")
