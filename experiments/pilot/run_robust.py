"""The robust arm of the comparison: Huber loss, f_scale 5 arcmin.

Ordinary least squares on this series is not a measure of fit. On the training
window 99.0 percent of the sum of squares comes from 15 of 268 epochs whose
declination is transcribed with a gross error, up to 19 degrees out. Those rows
are identical for every model, but they enter the sum of squares
quadratically, so an L2 comparison largely measures which family bends
furthest towards bad rows. The Huber scale is set at 5 arcmin, just above
Tycho's own repeat scatter of 2 to 4 arcmin, so honest observations are
untouched and gross errors enter linearly.

This module holds the robust machinery. Every family in the pilot is fitted
under BOTH losses by the same code path, and the greedy frequency ordering of
the harmonic family is recomputed under the robust loss by
models.greedy_order_robust, so the claim and the implementation agree.
run_pilot.py is the single entry point and writes the whole of results.json,
including the robust_huber block; running this file just calls it.

Nothing here is tuned on the test period.
"""

import numpy as np

import ephem as E
import models

F_SCALE = 5.0


def huber_w(r, d=F_SCALE):
    """IRLS weight for a residual r in arcmin under the Huber loss."""
    a = np.abs(r)
    return np.where(a <= d, 1.0, d / np.maximum(a, 1e-12))


def irls_harmonic(t, y, freqs, f_scale=F_SCALE):
    """Huber fit of the harmonic design by iteratively reweighted least squares."""
    coef, _cond = models.fit_harmonic_robust(t, y, freqs, f_scale=f_scale)
    return coef


def robust_mean(y, f_scale=F_SCALE, n_iter=50):
    """Huber location estimate, the robust counterpart of the mean control."""
    mu = float(np.median(y))
    for _ in range(n_iter):
        w = huber_w((y - mu) / E.ARCMIN, f_scale)
        mu_new = float(np.sum(w * y) / np.sum(w))
        if abs(mu_new - mu) < 1e-15:
            mu = mu_new
            break
        mu = mu_new
    return mu


def main():
    import run_pilot
    run_pilot.main()


if __name__ == "__main__":
    main()
