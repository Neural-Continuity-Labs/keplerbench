"""The three model families compared in the pilot.

1. Epicycle / harmonic interpolator. Truncated Fourier series on the lattice
   of frequencies n1 f_sid + n2 f_syn built from Mars's sidereal (686.98 d)
   and synodic (779.94 d) periods. Their sum is the annual frequency, so the
   lattice is the Ptolemaic deferent plus epicycle family and its harmonics.
   Linear in its coefficients, so it is fitted by ordinary least squares.
   Frequencies are ordered by GREEDY FORWARD SELECTION on the TRAINING data
   only, which gives the strongest K term member of the family rather than an
   arbitrarily ordered one.

2. Keplerian two body geometric model. Mars on an ellipse about the Sun,
   Kepler's equation solved by Newton iteration, projected to geocentric
   declination for an observer whose heliocentric position comes from the
   Meeus solar theory. Earth's orbit is FIXED, not fitted. Six free
   parameters (a, e, i, node, argument of perihelion, mean anomaly at epoch);
   the mean motion follows Kepler's third law from a. A seven parameter
   variant frees the mean motion so that Kepler's third law can be tested
   rather than assumed.

3. Quasi periodic Gaussian process on the declination series, a second
   interpolator with no orbital geometry in it.

Every family is fitted under BOTH losses by the same
fitter, on the same training window: ordinary least squares (reported as an
ablation) and Huber with f_scale 5 arcmin (the headline). The harmonic family
uses iteratively reweighted least squares and re-runs its greedy frequency
selection under the robust weights; the Keplerian family passes the loss
straight to scipy least_squares; the Gaussian process is robustified by
iterating its per-point noise on leave-one-out residuals.

Every residual is in arcminutes.
"""

import numpy as np
from scipy.optimize import least_squares, minimize

import ephem as E

P_SID = 686.9800
P_SYN = 779.9361
F_SCALE_DEFAULT = 5.0


def wrap_angle(x):
    """Wrap an angle or array of angles to (-pi, pi]."""
    return -((-np.asarray(x, dtype=float) + np.pi) % (2 * np.pi) - np.pi)


def huber_weights(r, f_scale=F_SCALE_DEFAULT):
    """IRLS weights for the Huber loss: 1 inside f_scale, f_scale/|r| outside."""
    a = np.abs(np.asarray(r, dtype=float))
    return np.where(a <= f_scale, 1.0, f_scale / np.maximum(a, 1e-12))


def huber_cost(r, f_scale=F_SCALE_DEFAULT):
    """The Huber objective itself, in arcmin squared, for model selection."""
    a = np.abs(np.asarray(r, dtype=float))
    return float(np.sum(np.where(a <= f_scale, a ** 2,
                                 2 * f_scale * a - f_scale ** 2)))


def frequency_lattice(max_order=8, min_period=25.0):
    f1, f2 = 1.0 / P_SID, 1.0 / P_SYN
    cand = []
    for n1 in range(-max_order, max_order + 1):
        for n2 in range(-max_order, max_order + 1):
            if n1 == 0 and n2 == 0:
                continue
            if abs(n1) + abs(n2) > max_order:
                continue
            f = n1 * f1 + n2 * f2
            if f <= 0 or f > 1.0 / min_period:
                continue
            cand.append((abs(n1) + abs(n2), f, n1, n2))
    cand.sort()
    out = []
    for order, f, n1, n2 in cand:
        if all(abs(f - g[0]) > 1e-7 for g in out):
            out.append((f, n1, n2, order))
    return out


def design(t, freqs):
    """[N, 1 + 2*len(freqs)] harmonic design matrix; t in days."""
    cols = [np.ones_like(t)]
    for f in freqs:
        w = 2 * np.pi * f * t
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, -1)


def greedy_order(t_tr, y_tr, lattice, n_max):
    """Order lattice frequencies by greedy training-SSE reduction."""
    chosen, pool = [], list(range(len(lattice)))
    X = np.ones((len(t_tr), 1))
    resid = y_tr - y_tr.mean()
    for _ in range(n_max):
        best, best_sse = None, np.inf
        for j in pool:
            f = lattice[j][0]
            w = 2 * np.pi * f * t_tr
            Xn = np.concatenate([X, np.cos(w)[:, None], np.sin(w)[:, None]], 1)
            try:
                c, *_ = np.linalg.lstsq(Xn, y_tr, rcond=None)
            except np.linalg.LinAlgError:
                continue
            sse = float(np.sum((y_tr - Xn @ c) ** 2))
            if sse < best_sse:
                best_sse, best, best_X = sse, j, Xn
        if best is None:
            break
        chosen.append(best)
        pool.remove(best)
        X = best_X
    return [lattice[j] for j in chosen]


def greedy_order_robust(t_tr, y_tr, lattice, n_max, f_scale=F_SCALE_DEFAULT):
    """Greedy forward selection under the Huber objective.

    Same protocol as greedy_order, but every candidate model is fitted by IRLS
    with Huber weights and scored by the Huber cost rather than the sum of
    squares, so the robust arm of the comparison selects its frequencies under
    the loss it is finally judged by.
    """
    chosen, pool = [], list(range(len(lattice)))
    X = np.ones((len(t_tr), 1))
    for _ in range(n_max):
        best, best_cost, best_X = None, np.inf, None
        for j in pool:
            f = lattice[j][0]
            w = 2 * np.pi * f * t_tr
            Xn = np.concatenate([X, np.cos(w)[:, None], np.sin(w)[:, None]], 1)
            try:
                coef = _irls_lstsq(Xn, y_tr, f_scale)
            except np.linalg.LinAlgError:
                continue
            c = huber_cost((y_tr - Xn @ coef) / E.ARCMIN, f_scale)
            if c < best_cost:
                best_cost, best, best_X = c, j, Xn
        if best is None:
            break
        chosen.append(best)
        pool.remove(best)
        X = best_X
    return [lattice[j] for j in chosen]


def _irls_lstsq(X, y, f_scale, n_iter=25):
    """Huber IRLS for a linear model; y and the residuals are in radians."""
    w = np.ones(len(y))
    coef = np.zeros(X.shape[1])
    for _ in range(n_iter):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        w_new = huber_weights((y - X @ coef) / E.ARCMIN, f_scale)
        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w = w_new
    return coef


def fit_harmonic_robust(t_tr, y_tr, freqs, f_scale=F_SCALE_DEFAULT):
    """Huber IRLS fit of the harmonic design; same design matrix as fit_harmonic."""
    X = design(t_tr, freqs)
    coef = _irls_lstsq(X, y_tr, f_scale)
    s = np.linalg.svd(X, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    return coef, cond


def fit_harmonic(t_tr, y_tr, freqs):
    X = design(t_tr, freqs)
    coef, *_ = np.linalg.lstsq(X, y_tr, rcond=None)
    s = np.linalg.svd(X, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    return coef, cond


def predict_harmonic(t, freqs, coef):
    return design(t, freqs) @ coef


# --------------------------------------------------------------------------
# Keplerian model
# --------------------------------------------------------------------------
KEP_NAMES = ["a", "e", "inc", "node", "argp", "M0"]
KEP_LO = np.array([1.05, 0.0, 0.0, -8.0, -8.0, -8.0])
KEP_HI = np.array([2.60, 0.60, 0.60, 8.0, 8.0, 8.0])


def kepler_resid(theta, jd, dec_obs, jd_ref, free_n=False):
    a, e, inc, node, argp, M0 = theta[:6]
    n = theta[6] if free_n else None
    pred = E.predicted_declination(jd, a, e, inc, node, argp, M0, jd_ref, n=n)
    return (dec_obs - pred) / E.ARCMIN


def _kepler_jacobian(theta, jd, dec_obs, jd_ref, free_n):
    """Central-difference Jacobian of the arcmin residuals at theta."""
    theta = np.asarray(theta, float)
    cols = []
    for k in range(len(theta)):
        h = 1e-7 * max(abs(theta[k]), 1.0)
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        rp = kepler_resid(tp, jd, dec_obs, jd_ref, free_n)
        rm = kepler_resid(tm, jd, dec_obs, jd_ref, free_n)
        cols.append((rp - rm) / (2 * h))
    return np.stack(cols, -1)


def kepler_covariance(theta, jd, dec_obs, jd_ref, free_n=False,
                      loss="linear", f_scale=F_SCALE_DEFAULT):
    """Parameter covariance and correlation under the loss actually used.

    For the robust fit the Gauss-Newton normal matrix is weighted by the same
    Huber weights the fit used, so the two losses are reported on the same
    footing instead of one of them borrowing the other's error model.
    """
    J = _kepler_jacobian(theta, jd, dec_obs, jd_ref, free_n)
    r = kepler_resid(theta, jd, dec_obs, jd_ref, free_n)
    w = np.ones_like(r) if loss == "linear" else huber_weights(r, f_scale)
    A = J.T @ (w[:, None] * J)
    dof = max(len(jd) - len(theta), 1)
    sigma2 = float(np.sum(w * r ** 2) / dof)
    sv = np.linalg.svd(J * np.sqrt(w)[:, None], compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    try:
        C = np.linalg.inv(A) * sigma2
        sd = np.sqrt(np.clip(np.diag(C), 0, None))
        corr = C / np.outer(np.where(sd > 0, sd, 1.0), np.where(sd > 0, sd, 1.0))
    except np.linalg.LinAlgError:
        sd = np.full(len(theta), np.nan)
        corr = np.full((len(theta), len(theta)), np.nan)
    return dict(perr=sd, corr=corr, cond=cond, sigma2=sigma2)


def fit_kepler(jd, dec_obs, jd_ref, n_restarts=40, seed=0, free_n=False,
               include_modern_start=True, loss="linear",
               f_scale=F_SCALE_DEFAULT):
    """Fit the Keplerian model under either loss with many restarts.

    loss="linear" is ordinary least squares, loss="huber" is the robust
    headline fit. Everything else (bounds, restart schedule, seed, tolerances)
    is identical between the two, so the only difference between the L2 and the
    Huber numbers in results.json is the loss function.

    Returns the best fit plus the full restart record.
    """
    if loss not in ("linear", "huber"):
        raise ValueError(f"unsupported loss {loss!r}")
    rng = np.random.default_rng(seed)
    starts = []
    if include_modern_start:
        el = E.mars_modern_elements(jd_ref)
        starts.append([el["a"], el["e"], el["inc"], el["node"], el["argp"], el["M0"]])
    for _ in range(n_restarts):
        starts.append([rng.uniform(1.2, 2.0), rng.uniform(0.0, 0.30),
                       rng.uniform(0.0, 0.15), rng.uniform(-np.pi, np.pi),
                       rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)])
    lo, hi = KEP_LO.copy(), KEP_HI.copy()
    if free_n:
        n_kep = np.sqrt(E.GM_SUN / 1.5237 ** 3)
        lo = np.append(lo, n_kep * 0.7)
        hi = np.append(hi, n_kep * 1.4)

    records = []
    for s in starts:
        s = list(s)
        if free_n:
            s.append(float(np.sqrt(E.GM_SUN / max(s[0], 1e-3) ** 3)))
        s = np.clip(np.array(s, float), lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(kepler_resid, s, bounds=(lo, hi), method="trf",
                              loss=loss, f_scale=f_scale,
                              args=(jd, dec_obs, jd_ref, free_n),
                              xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
        except Exception:
            continue
        if not np.all(np.isfinite(r.x)):
            continue
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        records.append(dict(x=r.x.copy(), rms=rms, cost=float(r.cost)))
    if not records:
        raise RuntimeError("no Kepler fit converged")
    records.sort(key=lambda d: (d["cost"], d["rms"]))
    best = records[0]
    cov = kepler_covariance(best["x"], jd, dec_obs, jd_ref, free_n, loss, f_scale)
    return dict(theta=best["x"], rms=best["rms"], cost=best["cost"],
                cond=cov["cond"], perr=cov["perr"], corr=cov["corr"],
                records=records, free_n=free_n, jd_ref=jd_ref,
                loss=loss, f_scale=(f_scale if loss == "huber" else None))


ANGLE_INDICES = (3, 4, 5)


def restart_summary(fit, rel_tol=0.01):
    """Restart record with every angular spread wrapped to (-pi, pi].

    The parameter bounds run to +-8 rad, wider than one revolution, so raw
    spreads of node, argp and M0 across restarts are whole-revolution
    relabellings of one solution rather than distinct fits. Spreads are
    therefore taken relative to the best fit and wrapped before the range is
    measured.
    """
    recs = fit["records"]
    cost = np.array([r["cost"] for r in recs])
    rms = np.array([r["rms"] for r in recs])
    best_cost = float(cost.min())
    at_global = cost <= best_cost * (1.0 + rel_tol) + 1e-12
    X = np.array([r["x"] for r in recs])[at_global]
    ref = np.asarray(fit["theta"], float)
    spread = []
    for i in range(X.shape[1]):
        d = X[:, i] - ref[i]
        if i in ANGLE_INDICES:
            d = wrap_angle(d)
        spread.append(float(np.ptp(d)))
    return dict(
        n=int(len(recs)), n_at_global=int(at_global.sum()),
        best_rms=float(rms.min()), worst_rms=float(rms.max()),
        best_cost=best_cost, worst_cost=float(cost.max()),
        distinct_optima_rms=[float(x) for x in np.unique(np.round(rms, 1))][:12],
        spread_within_global_arcmin=float(np.ptp(rms[at_global]))
        if at_global.sum() else 0.0,
        param_spread_within_global=spread,
        angle_wrapping_note=("node, argp and M0 spreads are wrapped to "
                             "(-pi, pi] relative to the best fit; the raw "
                             "bounds span more than one revolution"))


def predict_kepler(jd, fit):
    th = fit["theta"]
    n = th[6] if fit["free_n"] else None
    return E.predicted_declination(jd, th[0], th[1], th[2], th[3], th[4], th[5],
                                   fit["jd_ref"], n=n)


# --------------------------------------------------------------------------
# Quasi periodic Gaussian process
# --------------------------------------------------------------------------
def _gp_K(t1, t2, p):
    la1, lp1, lL1, la2, lp2, lL2 = p[:6]
    d = t1[:, None] - t2[None, :]
    k = np.exp(2 * la1) * np.exp(
        -2 * np.sin(np.pi * d / P_SID) ** 2 / np.exp(2 * lp1)
        - d ** 2 / (2 * np.exp(2 * lL1)))
    k += np.exp(2 * la2) * np.exp(
        -2 * np.sin(np.pi * d / P_SYN) ** 2 / np.exp(2 * lp2)
        - d ** 2 / (2 * np.exp(2 * lL2)))
    return k


def _gp_solve(t_tr, yc, p, weights):
    """Cholesky factor and alpha for the noise model sigma^2 / w_i."""
    s2 = np.exp(2 * p[6]) + 1e-10
    K = _gp_K(t_tr, t_tr, p) + np.diag(s2 / weights)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
    return L, alpha, s2


def fit_gp(t_tr, y_tr, weights=None, n_starts=8, seed=20260829,
           initial_p=None):
    """Type II maximum likelihood for the quasi periodic kernel.

    weights are per-observation precisions: the noise variance at point i is
    sigma^2 / w_i, so w = 1 everywhere is the ordinary L2 Gaussian process and
    downweighted points are the robust variant of fit_gp_robust.
    """
    mu = float(np.mean(y_tr))
    yc = y_tr - mu
    sd = float(np.std(yc)) + 1e-9
    w = np.ones(len(y_tr)) if weights is None else np.asarray(weights, float)

    def nll(p):
        try:
            L, al, _ = _gp_solve(t_tr, yc, p, w)
        except np.linalg.LinAlgError:
            return 1e12
        return float(0.5 * yc @ al + np.sum(np.log(np.diag(L))))

    p0 = np.array([np.log(sd), np.log(1.0), np.log(3000.0),
                   np.log(sd), np.log(1.0), np.log(3000.0),
                   np.log(max(sd * 0.02, 1e-4))])
    if initial_p is not None:
        p0 = np.asarray(initial_p, float).copy()
    rng = np.random.default_rng(seed)
    starts = [p0]
    # Perturb amplitudes, periodic length scales, time envelopes and nugget on
    # their natural log scales. This is the same seeded restart principle used
    # by fit_kepler, applied to the non-convex marginal likelihood.
    scales = np.array([0.8, 0.8, 1.5, 0.8, 0.8, 1.5, 1.0])
    for _ in range(max(int(n_starts), 1) - 1):
        starts.append(p0 + rng.normal(size=7) * scales)
    best, best_v = p0, np.inf
    for start in starts:
        candidate, candidate_v = start, nll(start)
        for method in ("L-BFGS-B", "Powell"):
            try:
                opts = dict(maxiter=600) if method == "Powell" else None
                r = minimize(nll, candidate, method=method, options=opts)
                if np.isfinite(r.fun) and r.fun < candidate_v:
                    candidate, candidate_v = r.x, float(r.fun)
            except Exception:
                pass
        if candidate_v < best_v:
            best, best_v = candidate, candidate_v
    L, alpha, s2 = _gp_solve(t_tr, yc, best, w)
    return dict(p=best, t=t_tr, alpha=alpha, mu=mu, nll=best_v, weights=w,
                sigma2=s2, L=L, n_starts=int(max(int(n_starts), 1)),
                seed=int(seed))


def gp_effective_dof(fit):
    """trace(K (K + S)^-1), the number of parameters the GP actually spends."""
    K = _gp_K(fit["t"], fit["t"], fit["p"])
    Ky = K + np.diag(fit["sigma2"] / fit["weights"])
    return float(np.trace(K @ np.linalg.inv(Ky)))


def gp_loo_residuals(fit, y_tr):
    """Leave-one-out residuals in the units of y (Rasmussen and Williams 5.12).

    r_i = alpha_i / [Ky^-1]_ii is exactly y_i minus the predictive mean of a
    model refitted without point i, so outliers are visible even though the
    in-sample residual of an interpolating GP is not.
    """
    K = _gp_K(fit["t"], fit["t"], fit["p"])
    Ky = K + np.diag(fit["sigma2"] / fit["weights"])
    Kinv = np.linalg.inv(Ky)
    return fit["alpha"] / np.diag(Kinv)


def fit_gp_robust(t_tr, y_tr, f_scale_arcmin=F_SCALE_DEFAULT, n_iter=6,
                  n_starts=8, seed=20260829):
    """Robust Gaussian process: IRLS on the nugget, driven by LOO residuals.

    A Gaussian process interpolates its training points, so its in-sample
    residual cannot identify a gross error. The leave-one-out residual can, and
    it has a closed form. Each round downweights point i by the Huber weight of
    its LOO residual, which inflates that point's noise to sigma^2 / w_i and is
    the Gaussian process analogue of the Huber fit used for the other families.
    """
    w = np.ones(len(y_tr))
    fit = None
    for iteration in range(n_iter):
        fit = fit_gp(t_tr, y_tr, weights=w,
                     n_starts=n_starts if iteration == 0 else 1, seed=seed,
                     initial_p=None if fit is None else fit["p"])
        r = gp_loo_residuals(fit, y_tr) / E.ARCMIN
        w_new = huber_weights(r, f_scale_arcmin)
        if np.max(np.abs(w_new - w)) < 1e-6:
            w = w_new
            break
        w = w_new
    fit = fit_gp(t_tr, y_tr, weights=w, n_starts=1, seed=seed,
                 initial_p=fit["p"])
    fit["n_starts_initial"] = int(n_starts)
    fit["f_scale_arcmin"] = f_scale_arcmin
    fit["n_downweighted"] = int(np.sum(w < 0.999))
    return fit


def predict_gp(t, fit):
    return fit["mu"] + _gp_K(t, fit["t"], fit["p"]) @ fit["alpha"]
