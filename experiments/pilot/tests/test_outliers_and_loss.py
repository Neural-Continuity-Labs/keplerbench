"""Outlier handling and the choice of loss.

The single most consequential fact in this pilot: 99 percent of the training sum
of squares comes from 5 percent of the rows. Ordinary least squares therefore
does not measure how well a model describes Tycho's observations; it measures how
far a model bends towards fourteen transcription errors.
"""

import os
import re

import numpy as np
import pytest

import constants as S
from conftest import medae


def test_ols_objective_is_dominated_by_gross_errors(modern_resid, split):
    """The number the paper must print next to every L2 figure."""
    r = modern_resid[split["tr"]]
    big = np.abs(r) > S.GROSS_ARCMIN
    share = float(np.sum(r[big] ** 2) / np.sum(r ** 2))
    assert share > 0.90, f"gross rows carry only {share:.1%} of the SSE"
    frac_rows = float(big.mean())
    assert frac_rows < 0.10, f"{frac_rows:.1%} of rows"


def test_rms_and_median_disagree_by_an_order_of_magnitude(modern_resid, split):
    """When these two disagree this much, RMS is a count of bad rows and must
    not be the headline statistic, nor the y axis of the figure."""
    r = modern_resid[split["tr"]]
    rms = float(np.sqrt(np.mean(r ** 2)))
    med = medae(r)
    assert rms / med > 10.0, f"rms {rms:.1f}, median {med:.1f}"


def test_smoothness_detector_precision_and_recall(built, modern_resid):
    """Score the detector against the reference criterion: residual
    against unfitted modern elements above 60 arcmin."""
    flag = built["outlier"]
    truth = np.abs(modern_resid) > S.GROSS_ARCMIN
    tp = int((flag & truth).sum())
    precision = tp / max(int(flag.sum()), 1)
    recall = tp / max(int(truth.sum()), 1)
    assert precision >= S.DETECTOR_MIN_PRECISION, (
        f"precision {precision:.3f}: the detector deletes "
        f"{int(flag.sum()) - tp} good epochs")
    assert recall >= S.DETECTOR_MIN_RECALL, (
        f"recall {recall:.3f}: the detector leaves {int(truth.sum()) - tp} gross "
        "errors in the data it hands to the fitters")


def test_detector_deletes_good_observations(built, modern_resid):
    """Name any good epoch the detector removes."""
    flag = built["outlier"]
    victims = np.abs(modern_resid[flag])
    n_clean = int(np.sum(victims < 10.0))
    assert n_clean == 0, (
        f"{n_clean} deleted epochs agree with the modern ephemeris to better "
        f"than 10 arcmin; their residuals are "
        f"{np.round(np.sort(victims)[:6], 1).tolist()} arcmin")


def test_repeat_rows_within_an_epoch_do_not_remove_sign_errors(built):
    """The 1593 pair at JD 2303244.79 and .83 is one night, one hour apart, with
    opposite signs. Grouping on rounded JD keeps them separate, which is
    correct, and means grouping cannot be relied on to clean the record."""
    jd, dec = built["jd"], built["dec"]
    i = np.argmin(np.abs(jd - 2303244.7917))
    j = np.argmin(np.abs(jd - 2303244.8333))
    assert i != j
    assert np.degrees(dec[i]) * np.degrees(dec[j]) < 0
    assert abs(abs(np.degrees(dec[i])) - abs(np.degrees(dec[j]))) < 0.05


@pytest.mark.slow
def test_huber_beats_ols_on_the_real_record_out_of_sample(models, ephem, split,
                                                          results):
    """The two shipped fits differ only in the loss. Confirm the
    ordering from the stored elements rather than refitting."""
    def resid(entry, jd, dec):
        e = entry["elements"]
        th = [e["a_AU"], e["e"], np.radians(e["inc_deg"]),
              np.radians(e["node_deg"]), np.radians(e["argp_deg"]),
              np.radians(e["M0_deg"])]
        p = ephem.predicted_declination(jd, *th, split["jd_ref"])
        return (dec - p) / ephem.ARCMIN

    ols = results["models"]["kepler6"]
    hub = results["robust_huber"]["models"]["kepler6"]
    m_ols = medae(resid(ols, split["jd_te"], split["dec_te"]))
    m_hub = medae(resid(hub, split["jd_te"], split["dec_te"]))
    assert m_hub < m_ols / 3.0, (
        f"OLS held-out median {m_ols:.1f} arcmin, Huber {m_hub:.1f}; the "
        "headline must use the robust fit")
    assert m_hub <= S.KEPLER_ROBUST_OUT_MEDAE_MAX


def test_unfitted_modern_elements_beat_the_ols_fit_out_of_sample(results,
                                                                 modern_resid,
                                                                 split):
    """The clearest single symptom. A model with SIX free parameters, fitted to
    the training window, predicts the held-out window five times worse than the
    same functional form with ZERO parameters fitted."""
    m_modern = medae(modern_resid[split["te"]])
    m_ols = results["models"]["kepler6"]["out_of_sample"]["medae"]
    assert m_modern < m_ols / 3.0, (
        f"unfitted {m_modern:.2f} arcmin versus fitted {m_ols:.2f} arcmin")


def test_every_family_is_fitted_under_both_losses(results):
    """Every family must carry a robust-loss entry, or the headline comparison
    is not matched across families."""
    rob = results.get("robust_huber", {}).get("models", {})
    for k in ("kepler6", "gp_quasiperiodic", "control_mean"):
        assert k in rob, f"{k} has no robust-loss entry; the comparison is unmatched"


def test_robust_script_does_not_claim_an_ordering_it_does_not_do(pilot_dir):
    """If run_robust.py says the greedy order is recomputed under the robust
    loss, the robust ordering must exist."""
    src = open(os.path.join(pilot_dir, "run_robust.py")).read()
    claims = "recomputed under the robust loss" in src
    has_robust_order = "irls_greedy" in src or "greedy_order_robust" in src
    assert (not claims) or has_robust_order, (
        "run_robust.py claims a robust frequency ordering it does not implement")


def test_huber_scale_is_above_tycho_repeat_scatter(pilot_dir):
    """f_scale must sit above the instrument's own scatter (2 to 4 arcmin) or
    the robust fit starts discarding honest observations."""
    import run_robust
    assert 4.0 <= run_robust.F_SCALE <= 8.0, run_robust.F_SCALE


def test_huber_weight_function(pilot_dir):
    import run_robust
    w = run_robust.huber_w(np.array([0.0, 1.0, 5.0, 10.0, 1000.0]))
    assert w[0] == 1.0 and w[1] == 1.0 and w[2] == 1.0
    assert w[3] == pytest.approx(0.5)
    assert w[4] == pytest.approx(0.005)
    assert np.all(np.diff(w[2:]) < 0)


def test_the_ols_fit_is_at_the_l2_optimum_not_a_search_failure(ephem, split,
                                                               results):
    """An optimizer failure is fixed by more restarts, a loss failure is not,
    so the distinction matters.

    Proof: evaluate the L2 objective at the unfitted modern elements, which are
    the physically correct answer, and at the fitted optimum. The fitted point
    has the LOWER training RMS. The search therefore succeeded; the objective it
    was handed is what disagrees with the sky."""
    el = ephem.mars_modern_elements(split["jd_ref"])
    p_mod = ephem.predicted_declination(split["jd_tr"], el["a"], el["e"],
                                        el["inc"], el["node"], el["argp"],
                                        el["M0"], split["jd_ref"])
    rms_modern = float(np.sqrt(np.mean(((split["dec_tr"] - p_mod)
                                        / ephem.ARCMIN) ** 2)))
    rms_fit = results["models"]["kepler6"]["in_sample"]["rms"]
    assert rms_fit <= rms_modern, (
        f"fitted L2 training RMS {rms_fit:.3f} is worse than the RMS at the "
        f"true elements {rms_modern:.3f}; only then would 'not at the right "
        "optimum' be the correct diagnosis")
    assert rms_modern - rms_fit < 2.0, (
        "the six free parameters buy less than 2 arcmin of training RMS over "
        "the zero-parameter modern elements, because the objective is almost "
        "entirely the transcription errors")
