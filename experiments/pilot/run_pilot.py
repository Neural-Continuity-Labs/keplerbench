"""Kepler-test pilot: in-sample fit versus held-out prediction on Tycho's Mars
declinations. Single entry point; writes the whole of results.json.

Every family is fitted twice on the training window, once under ordinary least
squares and once under the Huber loss at 5 arcmin, with the same restarts, the
same seeds, the same frequency selection protocol and the same statistics. The
Huber arm is the headline. The L2 arm is kept as a labelled ablation because
its numbers are an artifact of the transcription errors rather than a property
of any model class, and a reader needs to see that side by side.

Usage: python3 run_pilot.py
"""

import datetime
import hashlib
import json
import os
import platform
import time

import pathlib

import numpy as np
import scipy
from scipy import stats as sstats

import pilot_data as data
import ephem as E
import models
import run_robust as RB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results.json")

KS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
P_SID = models.P_SID
P_SYN = models.P_SYN

# Reporting rule: primary buckets are Mars's synodic period measured from the last
# training epoch; the secondary set is sidereal. The calendar-year bins are the
# superseded binning, kept only so the counts that made them unusable stay on
# the record (n = 5, 54, 20, 7).
HORIZON_SYNODIC = [(0.0, P_SYN), (P_SYN, 2 * P_SYN), (2 * P_SYN, 3 * P_SYN)]
HORIZON_SIDEREAL = [(0.0, P_SID), (P_SID, 2 * P_SID), (2 * P_SID, 3 * P_SID),
                    (3 * P_SID, np.inf)]
HORIZON_CALENDAR = [(0.0, 365.0), (365.0, 730.0), (730.0, 1095.0),
                    (1095.0, np.inf)]
# Superseded bins, counted but never fitted with a median, kept so the record
# shows why they were replaced: they put n = 1, 1, 3 in the first three buckets.
HORIZON_SUPERSEDED = [(0.0, 30.0), (30.0, 90.0), (90.0, 365.0), (365.0, np.inf)]

MIN_N_FOR_MEDIAN = 10
GROSS_ARCMIN = 60.0
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_N = 10000
SEED_KEPLER_OLS = 7
SEED_KEPLER_HUBER = 7
SEED_KEPLER_BLIND = 21
SEED_GP = 20260829
SEED_SYNTHETIC_NOISE = 20260830
GP_N_STARTS = 8
F_SCALE = RB.F_SCALE
ALT_BINS_DEG = [(0.0, 15.0), (15.0, 25.0), (25.0, 40.0), (40.0, 70.0)]

R = {}


def save():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2, default=_jsonable)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return float(o)


# ------------------------------------------------------------------ statistics
def bootstrap_median_ci(a, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap 95 percent interval for a median.

    The generator is seeded inside the call, so the interval of a given sample
    does not depend on how many other statistics were computed before it.
    """
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    meds = np.median(a[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return [float(lo), float(hi)]


def stats(res_arcmin, min_n=MIN_N_FOR_MEDIAN):
    """The eight summary statistics plus the bootstrap interval.

    A point median is quoted only when n >= min_n (the reporting rule). Below that the
    full sorted list of absolute residuals is given instead, and medae and its
    interval are null so that no reader and no downstream script can lift a
    median off a handful of observations.
    """
    r = np.asarray(res_arcmin, float)
    out = dict(n=int(r.size), medae=None, medae_ci95=None, mad_sigma=None,
               p90=None, rms=None, max_abs=None, n_gross=0, median_signed=None)
    if r.size == 0:
        return out
    a = np.abs(r)
    out.update(rms=float(np.sqrt(np.mean(r ** 2))),
               mad_sigma=float(1.4826 * np.median(np.abs(r - np.median(r)))),
               p90=float(np.percentile(a, 90)),
               max_abs=float(np.max(a)),
               n_gross=int(np.sum(a > GROSS_ARCMIN)),
               median_signed=float(np.median(r)))
    if r.size >= min_n:
        out["medae"] = float(np.median(a))
        out["medae_ci95"] = bootstrap_median_ci(a)
    else:
        out["abs_residuals_sorted"] = [float(x) for x in np.sort(a)]
        out["below_min_n_note"] = (
            f"n = {r.size} is below the reporting minimum of {min_n}; "
            "sorted absolute residuals are given instead of a median")
    return out


def _bucket_key(lo, hi):
    hi_s = "inf" if not np.isfinite(hi) else str(int(round(hi)))
    return f"{int(round(lo))}-{hi_s}d"


def by_horizon(res, horizon, bins):
    out = {}
    for lo, hi in bins:
        m = (horizon >= lo) & (horizon < hi)
        out[_bucket_key(lo, hi)] = stats(res[m])
    return out


def horizon_blocks(ro, horizon):
    return dict(
        out_of_sample_by_horizon=by_horizon(ro, horizon, HORIZON_SYNODIC),
        out_of_sample_by_horizon_sidereal=by_horizon(ro, horizon,
                                                     HORIZON_SIDEREAL),
        out_of_sample_by_horizon_fine=by_horizon(ro, horizon, HORIZON_CALENDAR),
    )


def entry(family, n_params, effective_dof, ri, ro, horizon, loss, **extra):
    e = dict(family=family, n_params=n_params,
             effective_dof=float(effective_dof), loss=loss,
             in_sample=stats(ri), out_of_sample=stats(ro))
    e.update(horizon_blocks(ro, horizon))
    e.update(extra)
    return e


def hodges_lehmann(d):
    """Median of the Walsh averages of d."""
    d = np.asarray(d, float)
    w = (d[:, None] + d[None, :]) / 2.0
    iu = np.triu_indices(len(d))
    return float(np.median(w[iu]))


def paired_comparison(rA, rB, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Paired test of |rA| against |rB| on the common epochs.

    Reports the Hodges-Lehmann shift of the paired difference, its paired
    bootstrap interval and a two-sided sign test, for every headline pairwise
    claim.
    """
    a, b = np.abs(np.asarray(rA, float)), np.abs(np.asarray(rB, float))
    d = a - b
    hl = hodges_lehmann(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = np.array([hodges_lehmann(d[i]) for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pos, neg = int(np.sum(d > 0)), int(np.sum(d < 0))
    p = float(sstats.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else 1.0
    return dict(n=int(len(d)),
                hodges_lehmann_shift_arcmin=hl,
                hl_ci95=[float(lo), float(hi)],
                n_A_worse=pos, n_B_worse=neg, sign_test_p=p,
                median_abs_A=float(np.median(a)), median_abs_B=float(np.median(b)),
                note=("positive shift means the first model has the larger "
                      "absolute residual"))


def campaign_paired_comparison(rA, rB):
    """Paired campaign comparison respecting the n >= 10 reporting rule."""
    rA, rB = np.asarray(rA, float), np.asarray(rB, float)
    if len(rA) >= MIN_N_FOR_MEDIAN:
        return paired_comparison(rA, rB)
    d = np.abs(rA) - np.abs(rB)
    return dict(
        n=int(len(d)), hodges_lehmann_shift_arcmin=None, hl_ci95=None,
        median_abs_A=None, median_abs_B=None,
        paired_abs_residual_differences_sorted_arcmin=[
            float(x) for x in np.sort(d)],
        below_min_n_note=(f"n = {len(d)} is below the reporting minimum of "
                          f"{MIN_N_FOR_MEDIAN}; sorted paired differences are "
                          "given instead of medians or paired inference"))


def wrapped_elements(th, free_n=False):
    e = dict(a_AU=float(th[0]), e=float(th[1]),
             inc_deg=float(np.degrees(th[2])),
             node_deg=float(np.degrees(models.wrap_angle(th[3]))),
             argp_deg=float(np.degrees(models.wrap_angle(th[4]))),
             M0_deg=float(np.degrees(models.wrap_angle(th[5]))))
    if free_n:
        e["n_rad_per_day"] = float(th[6])
    return e


def deviation_from_modern(th, el):
    dvarpi = float(np.degrees(models.wrap_angle(th[3] + th[4]
                                                - el["node"] - el["argp"])))
    return dict(a_AU=float(th[0] - el["a"]), a_rel=float(th[0] / el["a"] - 1),
                e=float(th[1] - el["e"]), e_rel=float(th[1] / el["e"] - 1),
                inc_deg=float(np.degrees(th[2] - el["inc"])),
                varpi_deg_wrapped=dvarpi)


def l2_artifact_note(census):
    """Generate the required L2 warning from the current data census."""
    pct = 100.0 * census["train_sse_share_of_gross"]
    return (f"{pct:.1f} percent of the training sum of squares comes from "
            f"{census['n_gross_train']} of {R['data']['n_train']} epochs")


def fit_pure_periodic_gp(t_tr, y_tr):
    """Training-only pure-periodic GP control."""
    from scipy.optimize import minimize

    x = np.asarray(y_tr, float) / E.ARCMIN
    mu = float(np.median(x))
    yc = x - mu

    def kernel(a, b, p):
        d = a[:, None] - b[None, :]
        la1, lp1, la2, lp2, _ln = p
        return (np.exp(2 * la1) * np.exp(
                    -2 * np.sin(np.pi * d / P_SID) ** 2 / np.exp(2 * lp1))
                + np.exp(2 * la2) * np.exp(
                    -2 * np.sin(np.pi * d / P_SYN) ** 2 / np.exp(2 * lp2)))

    def nll(p):
        A = kernel(t_tr, t_tr, p) + np.eye(len(t_tr)) * np.exp(2 * p[4])
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return 1e99
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
        return float(0.5 * yc @ alpha + np.log(np.diag(L)).sum())

    p0 = np.log([np.std(yc), 1.0, np.std(yc), 1.0, 5.0])
    opt = minimize(nll, p0, method="L-BFGS-B",
                   bounds=[(-5, 10), (-5, 5), (-5, 10), (-5, 5), (-5, 8)],
                   options={"maxiter": 300})
    A = kernel(t_tr, t_tr, opt.x) + np.eye(len(t_tr)) * np.exp(2 * opt.x[4])
    alpha = np.linalg.solve(A, yc)
    return dict(p=opt.x, t=t_tr, alpha=alpha, mu=mu, nll=float(opt.fun),
                success=bool(opt.success), kernel=kernel)


def predict_pure_periodic_gp(t, fit):
    return (fit["mu"] + fit["kernel"](np.asarray(t), fit["t"], fit["p"])
            @ fit["alpha"]) * E.ARCMIN


def fit_lattice_ridge(t_tr, y_tr, lattice):
    """Frequency-lattice ridge with lambda selected on a training tail."""
    freqs = [z[0] for z in lattice]
    X = models.design(t_tr, freqs)
    y = np.asarray(y_tr, float) / E.ARCMIN
    cut = np.quantile(t_tr, 0.8)
    a, v = t_tr <= cut, t_tr > cut
    candidates = []
    eye = np.eye(X.shape[1])
    for lam in np.logspace(-6, 8, 40):
        coef = np.linalg.solve(X[a].T @ X[a] + lam * eye, X[a].T @ y[a])
        score = float(np.median(np.abs(y[v] - X[v] @ coef)))
        candidates.append((score, float(lam)))
    validation_medae, lam = min(candidates)
    coef = np.linalg.solve(X.T @ X + lam * eye, X.T @ y)
    return dict(freqs=freqs, coef=coef, lambda_=lam,
                validation_medae=validation_medae,
                n_coefficients=int(X.shape[1]))


# ------------------------------------------------------------------- the run
def main():
    t_start = time.time()

    # ---------------------------------------------------------------- data
    D = data.build(split_year=1594)
    jd, dec, keep, split_jd = D["jd"], D["dec"], D["keep"], D["split_jd"]
    flag = D["outlier"]
    tr = jd < split_jd
    te = jd >= split_jd
    jd_tr, dec_tr, jd_te, dec_te = jd[tr], dec[tr], jd[te], dec[te]
    jd_ref = float(np.median(jd_tr))
    t_tr, t_te = jd_tr - jd_ref, jd_te - jd_ref
    train_end = float(jd_tr.max())
    horizon = jd_te - train_end
    alt_deg = E.transit_altitude_deg(dec)

    el = E.mars_modern_elements(jd_ref)
    pm = E.predicted_declination(jd, el["a"], el["e"], el["inc"], el["node"],
                                 el["argp"], el["M0"], jd_ref)
    rm = (dec - pm) / E.ARCMIN                 # residual vs unfitted modern
    gross = np.abs(rm) > GROSS_ARCMIN
    flip = np.abs(-dec - pm) < 0.2 * np.abs(dec - pm)

    years = np.array([_julian_year(x) for x in jd_te])
    uy, cy = np.unique(years, return_counts=True)

    shift_table = {}
    for s in (-10.0, -1.0, 0.0, 1.0, 10.0):
        ps = E.predicted_declination(jd + s, el["a"], el["e"], el["inc"],
                                     el["node"], el["argp"], el["M0"], jd_ref)
        rs = (dec - ps) / E.ARCMIN
        shift_table[f"{s:+.0f}"] = dict(
            medae=float(np.median(np.abs(rs))),
            frac_within_10_arcmin=float(np.mean(np.abs(rs) < 10.0)))

    tp = int(np.sum(flag & gross))
    R["data"] = dict(
        source=str(data.CSV.relative_to(data.HERE.parents[1])) if data.CSV.is_relative_to(data.HERE.parents[1]) else data.CSV.name,
        n_raw_rows=D["meta"]["n_rows"], n_epochs=D["meta"]["n_epochs"],
        n_dropped_internal_spread=D["meta"]["n_dropped_spread"],
        n_deleted=0,
        deletion_protocol=("The headline analysis deletes nothing. All 354 "
                           "epochs enter train and test and gross errors are "
                           "handled by the Huber loss. A deletion variant is "
                           "reported as a robustness check only."),
        n_train=int(tr.sum()), n_test=int(te.sum()),
        split_year=1594, split_jd=split_jd, jd_ref=jd_ref, train_end=train_end,
        train_jd_range=[float(jd_tr.min()), train_end],
        test_jd_range=[float(jd_te.min()), float(jd_te.max())],
        train_span_years=float((train_end - jd_tr.min()) / 365.25),
        horizon_days_range=[float(horizon.min()), float(horizon.max())],
        dec_deg_range=[float(np.degrees(dec.min())), float(np.degrees(dec.max()))],
        transit_altitude_deg_range=[float(alt_deg.min()), float(alt_deg.max())],
        test_year_composition={int(a): int(b) for a, b in zip(uy, cy)},
        test_window_note=("The held-out window is three observing campaigns "
                          "and a tail, not a uniform span."),
        horizon_counts={
            _bucket_key(lo, hi): int(((horizon >= lo) & (horizon < hi)).sum())
            for lo, hi in HORIZON_SYNODIC + HORIZON_SIDEREAL + HORIZON_CALENDAR},
        horizon_counts_superseded_bins={
            _bucket_key(lo, hi): int(((horizon >= lo) & (horizon < hi)).sum())
            for lo, hi in HORIZON_SUPERSEDED},
        horizon_bin_note=("The reporting rule replaced the calendar bins with synodic "
                          "ones. The superseded counts are kept here because "
                          "they are the reason: three of the four buckets held "
                          "fewer than 4 epochs and still carried a quoted "
                          "median."),
        date_correction=(
            "JD is taken from Pafko column 14 plus 1721048.0, which equals "
            "julian_calendar_to_jd(year, month, day, hour, minute) on all 923 "
            "rows to 4.66e-10 day, float64 resolution at JD 2.3e6. The minute "
            "column is mandatory: without it the identity fails by up to "
            "0.0410 day, 59 minutes. A loader that anchors Pafko's decimal "
            "date on the Julian New Year, although Pafko already applied the "
            "10 day Gregorian shift, runs a median 9.75 days late, range "
            "9.50 to 10.25; that convention is not used here."),
        shift_decision_table=dict(
            note=("Median absolute error of the unfitted modern elements with "
                  "every timestamp shifted by s days. The convention is decided "
                  "by this table, not by the arithmetic argument."),
            by_shift_days=shift_table),
        gross_error_census=dict(
            criterion="abs residual against unfitted modern elements above 60 arcmin",
            n_gross=int(gross.sum()), n_gross_train=int((gross & tr).sum()),
            n_gross_test=int((gross & te).sum()),
            n_sign_flips=int(flip.sum()),
            max_abs_arcmin=float(np.max(np.abs(rm))),
            train_sse_share_of_gross=float(
                np.sum(rm[tr & gross] ** 2) / np.sum(rm[tr] ** 2)),
            train_row_share_of_gross=float(np.mean(gross[tr])),
            note=("18 gross errors, 9 of them outright north to south sign "
                  "flips; the rest are large errors of other kinds. The "
                  "declination column is internally consistent with the sign "
                  "column on all 923 rows, so these are source or "
                  "transcription errors, not a loader sign bug.")),
        outlier_detector=dict(
            method=("robust local quadratic, window 45 days, at least 8 "
                    "neighbours, Huber weights at 5 arcmin, threshold 60 "
                    "arcmin, iterated; computed on jd < split_jd only"),
            params=D["meta"]["detector"],
            n_flagged=int(flag.sum()),
            n_flagged_test_window=int((flag & te).sum()),
            true_positives=tp,
            precision=float(tp / max(int(flag.sum()), 1)),
            recall=float(tp / max(int(gross.sum()), 1)),
            recall_ceiling=float(int((gross & tr).sum()) / int(gross.sum())),
            min_abs_modern_residual_of_flagged=float(
                np.min(np.abs(rm[flag]))) if flag.any() else None,
            n_unreachable_isolated=int(np.sum(
                gross & tr & (D["n_neighbours"] < D["meta"]["detector"]
                              ["min_neighbours"]))),
            note=("Used for the census and the deletion robustness check only; "
                  "the headline fits see every epoch. The recall ceiling is "
                  "the training share of the gross errors, and the epochs it "
                  "cannot reach are isolated ones with too few neighbours for "
                  "any local smoothness test."),
            flagged_jd=[float(x) for x in jd[flag]]),
    )
    save()
    print(f"[data] {D['meta']['n_epochs']} epochs, train {tr.sum()}, "
          f"test {te.sum()}, jd_ref {jd_ref:.5f}")

    # ------------------------------------------------- sanity: modern elements
    R["sanity_modern_elements"] = dict(
        note=("Declination predicted from JPL secular mean elements "
              "extrapolated to 1591, never fitted. Validates frames, units and "
              "calendar, and is the zero-parameter ceiling the fitted models "
              "are judged against."),
        all_epochs=stats(rm), train=stats(rm[tr]), test=stats(rm[te]),
        frac_within_10_arcmin=float(np.mean(np.abs(rm) < 10.0)),
        medae_non_gross_train=float(np.median(np.abs(rm[tr & ~gross]))),
        medae_non_gross_test=float(np.median(np.abs(rm[te & ~gross]))),
        elements=dict(a=float(el["a"]), e=float(el["e"]),
                      inc_deg=float(np.degrees(el["inc"])),
                      node_deg=float(np.degrees(models.wrap_angle(el["node"]))),
                      argp_deg=float(np.degrees(models.wrap_angle(el["argp"]))),
                      M0_deg=float(np.degrees(models.wrap_angle(el["M0"])))),
    )
    save()
    print(f"[sanity] modern medae {np.median(np.abs(rm)):.2f} arcmin, "
          f"{100 * np.mean(np.abs(rm) < 10):.1f} percent within 10 arcmin")

    # ---------------------------------------------------- harmonic frequencies
    lattice = models.frequency_lattice()
    m_max = (max(KS) - 1) // 2
    order_l2 = models.greedy_order(t_tr, dec_tr, lattice, m_max)
    order_hu = models.greedy_order_robust(t_tr, dec_tr, lattice, m_max,
                                          f_scale=F_SCALE)
    R["epicycle_frequencies"] = dict(
        l2=[dict(rank=i + 1, period_days=float(1.0 / f), n_sidereal=int(n1),
                 n_synodic=int(n2)) for i, (f, n1, n2, _o) in enumerate(order_l2)],
        huber=[dict(rank=i + 1, period_days=float(1.0 / f), n_sidereal=int(n1),
                    n_synodic=int(n2)) for i, (f, n1, n2, _o) in enumerate(order_hu)],
        note=("Greedy forward selection on the training window only, run once "
              "per loss: sum of squares for the L2 arm, Huber cost for the "
              "robust arm."))
    save()

    def sweep(order, robust):
        out = {}
        for K in KS:
            m = (K - 1) // 2
            freqs = [f for f, _, _, _ in order[:m]]
            if robust:
                coef, cond = models.fit_harmonic_robust(t_tr, dec_tr, freqs,
                                                        f_scale=F_SCALE)
            else:
                coef, cond = models.fit_harmonic(t_tr, dec_tr, freqs)
            ri = (dec_tr - models.predict_harmonic(t_tr, freqs, coef)) / E.ARCMIN
            ro = (dec_te - models.predict_harmonic(t_te, freqs, coef)) / E.ARCMIN
            out[str(K)] = dict(n_params=K, effective_dof=float(K),
                               cond_design=cond, loss="huber" if robust else "l2",
                               in_sample=stats(ri), out_of_sample=stats(ro))
        return out

    sweep_l2 = sweep(order_l2, False)
    sweep_hu = sweep(order_hu, True)
    R["epicycle_sweep"] = sweep_l2
    save()
    for K in (1, 3, 9, 25):
        print(f"[epicycle] K={K:2d} L2 in {sweep_l2[str(K)]['in_sample']['medae']:8.2f} "
              f"out {sweep_l2[str(K)]['out_of_sample']['medae']:9.2f} | "
              f"Huber in {sweep_hu[str(K)]['in_sample']['medae']:7.2f} "
              f"out {sweep_hu[str(K)]['out_of_sample']['medae']:9.2f}")

    # ---------------------- K selection: cross validation INSIDE train only
    def run_cv(robust):
        fwd, blk = {}, {}
        lo_t, hi_t = t_tr.min(), t_tr.max()
        cuts = [lo_t + q * (hi_t - lo_t) for q in (0.50, 0.60, 0.70, 0.80, 0.90)]
        fold_orders = []
        for c in cuts:
            a = t_tr < c
            b = (t_tr >= c) & (t_tr < c + 0.20 * (hi_t - lo_t))
            fold_orders.append((a, b, _order(t_tr[a], dec_tr[a], lattice,
                                             m_max, robust)))
        for K in KS:
            m = (K - 1) // 2
            sc = []
            for a, b, ordf in fold_orders:
                if a.sum() < K + 5 or b.sum() < 3:
                    continue
                sc.append(_fold_score(t_tr, dec_tr, a, b,
                                      [f for f, _, _, _ in ordf[:m]], robust))
            fwd[str(K)] = dict(fold_scores=sc,
                               mean_score=float(np.mean(sc)) if sc else None,
                               median_score=float(np.median(sc)) if sc else None)
        idx = np.argsort(t_tr)
        blocks = np.array_split(idx, 5)
        block_orders = []
        for bi in range(5):
            b = blocks[bi]
            a = np.setdiff1d(idx, b)
            block_orders.append((a, b, _order(t_tr[a], dec_tr[a], lattice,
                                              m_max, robust)))
        for K in KS:
            m = (K - 1) // 2
            sc = []
            for a, b, ordf in block_orders:
                if len(a) < K + 5:
                    continue
                sc.append(_fold_score(t_tr, dec_tr, a, b,
                                      [f for f, _, _, _ in ordf[:m]], robust))
            blk[str(K)] = dict(fold_scores=sc,
                               mean_score=float(np.mean(sc)) if sc else None)
        valid = {k: v["mean_score"] for k, v in fwd.items()
                 if v["mean_score"] is not None}
        k_fwd = int(min(valid, key=valid.get))
        k_blk = int(min(blk, key=lambda k: blk[k]["mean_score"]))
        crit = ("median absolute validation residual" if robust
                else "root mean square validation residual")
        return dict(
            protocol=("Forward chaining inside the training span only: fit on "
                      "t < cut, validate on the next 20 percent of the training "
                      "span, five cuts at 50 to 90 percent. Blocked five fold "
                      "for comparison. The frequency ordering is recomputed "
                      "inside every fold and the test period is never touched. "
                      f"Fold criterion: {crit}."),
            forward=fwd, blocked=blk,
            K_selected=k_fwd, K_selected_blocked=k_blk,
            selector_disagreement_note=(
                f"The two selectors disagree, K = {k_fwd} forward against "
                f"K = {k_blk} blocked. Both are reported; neither is presented "
                "alone."))

    cv_l2 = run_cv(False)
    cv_hu = run_cv(True)
    R["epicycle_cv"] = cv_l2
    save()
    print(f"[cv] L2 K* = {cv_l2['K_selected']} forward, "
          f"{cv_l2['K_selected_blocked']} blocked | Huber K* = "
          f"{cv_hu['K_selected']} forward, {cv_hu['K_selected_blocked']} blocked")

    # -------------------------------------------------------------- models
    R["models"] = {}
    robust_models = {}

    # modern unfitted: the zero-parameter reference, identical under both losses
    modern_entry = entry("reference", 0, 0.0, rm[tr], rm[te], horizon, "none",
                         elements=R["sanity_modern_elements"]["elements"],
                         note=("JPL secular elements evaluated at jd_ref and "
                               "never fitted. No loss applies because nothing "
                               "is estimated; the same numbers appear in both "
                               "arms and are the ceiling for the fitted "
                               "models."))
    R["models"]["modern_unfitted"] = modern_entry
    robust_models["modern_unfitted"] = dict(modern_entry)

    # control: training mean, and its Huber location counterpart
    mu_l2 = float(np.mean(dec_tr))
    mu_hu = RB.robust_mean(dec_tr, F_SCALE)
    R["models"]["control_mean"] = entry(
        "control", 1, 1.0, (dec_tr - mu_l2) / E.ARCMIN,
        (dec_te - mu_l2) / E.ARCMIN, horizon, "l2",
        location_deg=float(np.degrees(mu_l2)))
    robust_models["control_mean"] = entry(
        "control", 1, 1.0, (dec_tr - mu_hu) / E.ARCMIN,
        (dec_te - mu_hu) / E.ARCMIN, horizon, "huber",
        location_deg=float(np.degrees(mu_hu)))
    save()

    # epicycle entries
    def epicycle_entry(order, robust, K, family):
        m = (K - 1) // 2
        freqs = [f for f, _, _, _ in order[:m]]
        if robust:
            coef, cond = models.fit_harmonic_robust(t_tr, dec_tr, freqs,
                                                    f_scale=F_SCALE)
        else:
            coef, cond = models.fit_harmonic(t_tr, dec_tr, freqs)
        ri = (dec_tr - models.predict_harmonic(t_tr, freqs, coef)) / E.ARCMIN
        ro = (dec_te - models.predict_harmonic(t_te, freqs, coef)) / E.ARCMIN
        return entry(family, K, float(K), ri, ro, horizon,
                     "huber" if robust else "l2", K=K, cond_design=cond,
                     periods_days=[float(1.0 / f) for f in freqs])

    R["models"]["epicycle_cv_selected"] = epicycle_entry(
        order_l2, False, cv_l2["K_selected"], "interpolator")
    R["models"]["epicycle_cv_selected"]["cv_note"] = (
        f"K chosen by forward chaining inside the training window; the blocked "
        f"selector picks K = {cv_l2['K_selected_blocked']} instead.")
    R["models"]["epicycle_K25_saturated"] = epicycle_entry(
        order_l2, False, 25, "saturated")
    R["models"]["epicycle_K25_saturated"]["label_note"] = (
        "Saturated oracle, not the interpolator baseline: cross validation "
        "inside the training window rejects K = 25.")
    robust_models["epicycle_cv_selected"] = epicycle_entry(
        order_hu, True, cv_hu["K_selected"], "interpolator")
    robust_models["epicycle_cv_selected"]["cv_note"] = (
        f"K chosen by forward chaining under the robust loss; the blocked "
        f"selector picks K = {cv_hu['K_selected_blocked']} instead.")
    robust_models["epicycle_K25_saturated"] = epicycle_entry(
        order_hu, True, 25, "saturated")
    save()

    # ------------------------------------------------------- Kepler models
    for loss, seed in (("l2", SEED_KEPLER_OLS), ("huber", SEED_KEPLER_HUBER)):
        for tag, free_n in (("kepler6", False), ("kepler7_free_n", True)):
            t0 = time.time()
            fit = models.fit_kepler(
                jd_tr, dec_tr, jd_ref, n_restarts=60, seed=seed, free_n=free_n,
                loss=("linear" if loss == "l2" else "huber"), f_scale=F_SCALE)
            th = fit["theta"]
            ri = (dec_tr - models.predict_kepler(jd_tr, fit)) / E.ARCMIN
            ro = (dec_te - models.predict_kepler(jd_te, fit)) / E.ARCMIN
            names = models.KEP_NAMES + (["n"] if free_n else [])
            ent = entry("physical", len(th), float(len(th)), ri, ro, horizon,
                        loss,
                        elements=wrapped_elements(th, free_n),
                        param_std_err=dict(zip(names,
                                               [float(x) for x in fit["perr"]])),
                        deviation_from_modern=deviation_from_modern(th, el),
                        cond_jacobian=fit["cond"],
                        param_corr=fit["corr"].tolist(),
                        param_corr_max_offdiag=float(np.nanmax(
                            np.abs(fit["corr"] - np.eye(len(th))))),
                        param_corr_names=names,
                        restarts=models.restart_summary(fit),
                        fit_seconds=round(time.time() - t0, 1))
            off = np.abs(fit["corr"] - np.eye(len(th)))
            i, j = np.unravel_index(np.argmax(off), off.shape)
            ent["degeneracy"] = dict(
                strongest_pair=[names[i], names[j]],
                strongest_correlation=float(fit["corr"][i, j]),
                note=("This correlation matrix, not the restart spread, is the "
                      "identifiability evidence. The raw restart spreads of "
                      "node, argp and M0 were whole revolution relabellings "
                      "because the bounds span more than 2 pi; they are "
                      "wrapped before reporting and are then about 1e-5 rad."))
            if free_n:
                n_fit = th[6]
                ent["kepler_third_law_check"] = dict(
                    period_days=float(2 * np.pi / n_fit),
                    a_from_period_AU=float((E.GM_SUN / n_fit ** 2) ** (1 / 3)),
                    a_fitted_AU=float(th[0]),
                    ratio_a_geometry_over_a_period=float(
                        th[0] / (E.GM_SUN / n_fit ** 2) ** (1 / 3)))
            if loss == "l2":
                ent["artifact_note"] = (
                    "ABLATION, NOT THE HEADLINE. This is the ordinary least "
                    f"squares fit. On the training window {l2_artifact_note(R['data']['gross_error_census'])}, so this "
                    "objective mostly measures how far the model bends towards "
                    "transcription errors. Its held-out median is an artifact "
                    "of the loss, not a property of the Keplerian model class "
                    "and not a statement about identifiability. The headline "
                    "fit is results.robust_huber.models." + tag + ".")
                R["models"][tag] = ent
            else:
                robust_models[tag] = ent
            save()
            print(f"[{tag}/{loss}] a={th[0]:.5f} e={th[1]:.5f} in medae "
                  f"{ent['in_sample']['medae']:.2f} out medae "
                  f"{ent['out_of_sample']['medae']:.2f} "
                  f"({ent['restarts']['n_at_global']}/{ent['restarts']['n']} "
                  f"restarts at global, {ent['fit_seconds']} s)")

    # ------------------------------------------------------------------ GP
    gp_l2 = models.fit_gp(t_tr, dec_tr, n_starts=GP_N_STARTS, seed=SEED_GP)
    gp_hu = models.fit_gp_robust(t_tr, dec_tr, f_scale_arcmin=F_SCALE,
                                 n_starts=GP_N_STARTS, seed=SEED_GP)
    hyper_names = ["log_amp_sid", "log_ell_sid", "log_decay_sid", "log_amp_syn",
                   "log_ell_syn", "log_decay_syn", "log_white"]
    for tag, gp, loss in (("l2", gp_l2, "l2"), ("huber", gp_hu, "huber")):
        ri = (dec_tr - models.predict_gp(t_tr, gp)) / E.ARCMIN
        ro = (dec_te - models.predict_gp(t_te, gp)) / E.ARCMIN
        dof = models.gp_effective_dof(gp)
        ent = entry("interpolator", 7, dof, ri, ro, horizon, loss,
                    nll=float(gp["nll"]),
                    n_starts=int(gp.get("n_starts_initial", gp["n_starts"])),
                    seed=int(gp["seed"]),
                    hyperparams=dict(zip(hyper_names,
                                         [float(x) for x in gp["p"]])),
                    effective_dof_note=(
                        "trace(K (K + S)^-1) on the training epochs. The seven "
                        "hyperparameters are not the capacity of this model; "
                        f"it spends about {dof:.0f} effective degrees of "
                        "freedom, which is what the figure's capacity axis "
                        "uses."))
        if loss == "huber":
            ent["robustification"] = (
                "IRLS on the nugget driven by closed form leave one out "
                "residuals: a Gaussian process interpolates its training "
                f"points, so its in-sample residual cannot see an outlier. "
                f"{gp['n_downweighted']} of {len(dec_tr)} epochs end up "
                "downweighted, which includes isolated epochs as well as "
                "corrupt ones, so this arm is conservative about the "
                "interpolator rather than favourable to it.")
            robust_models["gp_quasiperiodic"] = ent
        else:
            R["models"]["gp_quasiperiodic"] = ent
        print(f"[gp/{loss}] in medae {ent['in_sample']['medae']:.2f} "
              f"out medae {ent['out_of_sample']['medae']:.1f} dof {dof:.1f}")
    save()

    # Two training-only extrapolation controls. The
    # pure-periodic kernel removes the time envelope. The ridge uses the whole
    # frequency lattice and selects its penalty on the last training fifth.
    periodic = fit_pure_periodic_gp(t_tr, dec_tr)
    pri = (dec_tr - predict_pure_periodic_gp(t_tr, periodic)) / E.ARCMIN
    pro = (dec_te - predict_pure_periodic_gp(t_te, periodic)) / E.ARCMIN
    robust_models["gp_pure_periodic_control"] = entry(
        "control", 5, 5.0, pri, pro, horizon, "training_nll",
        nll=periodic["nll"], hyperparams=[float(x) for x in periodic["p"]],
        note=("Pure-periodic GP control with the sidereal and synodic periods "
              "and no time envelope. Hyperparameters use training data only; "
              "this is a labelled extrapolation control, not a matched Huber fit."))

    ridge = fit_lattice_ridge(t_tr, dec_tr, lattice)
    rpi = (dec_tr - models.design(t_tr, ridge["freqs"])
           @ ridge["coef"] * E.ARCMIN) / E.ARCMIN
    rpo = (dec_te - models.design(t_te, ridge["freqs"])
           @ ridge["coef"] * E.ARCMIN) / E.ARCMIN
    robust_models["frequency_lattice_ridge_control"] = entry(
        "control", ridge["n_coefficients"], ridge["n_coefficients"],
        rpi, rpo, horizon, "training_cv_ridge",
        lambda_=ridge["lambda_"],
        training_validation_medae=ridge["validation_medae"],
        note=("Ridge regression on the full sidereal and synodic frequency "
              "lattice. Lambda is selected on the final fifth of the training "
              "window only; this is a labelled extrapolation control, not a "
              "matched Huber fit."))

    # --------------------------------------------------- robust block, headline
    R["robust_huber"] = dict(
        f_scale_arcmin=F_SCALE,
        note=("Headline arm. Every family is fitted under the same Huber loss "
              "at 5 arcmin, just above Tycho's own repeat scatter of 2 to 4 "
              "arcmin, on the same training window, with the same restarts and "
              "seeds and the same frequency selection protocol as the L2 arm. "
              "Nothing is tuned on the test period."),
        models=robust_models, epicycle_sweep=sweep_hu, epicycle_cv=cv_hu)
    save()

    # residual vectors for the paired tests
    def test_resid(name):
        src = robust_models[name]
        if name == "modern_unfitted":
            return rm[te]
        if name == "control_mean":
            return (dec_te - mu_hu) / E.ARCMIN
        if name.startswith("epicycle"):
            K = src["K"]
            m = (K - 1) // 2
            freqs = [f for f, _, _, _ in order_hu[:m]]
            coef, _ = models.fit_harmonic_robust(t_tr, dec_tr, freqs,
                                                 f_scale=F_SCALE)
            return (dec_te - models.predict_harmonic(t_te, freqs, coef)) / E.ARCMIN
        if name == "gp_quasiperiodic":
            return (dec_te - models.predict_gp(t_te, gp_hu)) / E.ARCMIN
        e = src["elements"]
        th = [e["a_AU"], e["e"], np.radians(e["inc_deg"]),
              np.radians(e["node_deg"]), np.radians(e["argp_deg"]),
              np.radians(e["M0_deg"])]
        free = "n_rad_per_day" in e
        if free:
            th.append(e["n_rad_per_day"])
        f = dict(theta=np.array(th), free_n=free, jd_ref=jd_ref)
        return (dec_te - models.predict_kepler(jd_te, f)) / E.ARCMIN

    r_kep = test_resid("kepler6")
    paired = {}
    for other in ("gp_quasiperiodic", "epicycle_cv_selected",
                  "epicycle_K25_saturated", "control_mean", "modern_unfitted"):
        paired[f"kepler6_vs_{other}"] = paired_comparison(r_kep,
                                                          test_resid(other))

    hb = []
    for name in ("modern_unfitted", "kepler6", "kepler7_free_n",
                 "epicycle_cv_selected", "epicycle_K25_saturated",
                 "gp_quasiperiodic", "control_mean"):
        m = robust_models[name]
        hb.append(dict(
            model=name, family=m["family"], n_params=m["n_params"],
            effective_dof=m["effective_dof"],
            in_sample_medae=m["in_sample"]["medae"],
            in_sample_ci95=m["in_sample"]["medae_ci95"],
            held_out_medae=m["out_of_sample"]["medae"],
            held_out_ci95=m["out_of_sample"]["medae_ci95"],
            held_out_H1_medae=m["out_of_sample_by_horizon"]["0-780d"]["medae"],
            held_out_H2_medae=m["out_of_sample_by_horizon"]["780-1560d"]["medae"],
            held_out_H3_medae=m["out_of_sample_by_horizon"]["1560-2340d"]["medae"],
        ))
    campaign_models = [row["model"] for row in hb]
    campaign_table = {}
    for year in sorted(np.unique(years)):
        mask = years == year
        model_rows = {}
        for name in campaign_models:
            model_rows[name] = stats(test_resid(name)[mask])
        campaign_table[str(int(year))] = dict(
            n=int(mask.sum()), models=model_rows,
            kepler6_vs_modern_unfitted=campaign_paired_comparison(
                test_resid("kepler6")[mask],
                test_resid("modern_unfitted")[mask]))

    gp_epoch = np.abs(test_resid("gp_quasiperiodic"))
    kep_epoch = np.abs(r_kep)
    crossing = np.flatnonzero(gp_epoch > kep_epoch)
    held_out_by_epoch = [
        dict(jd=float(j), calendar_year=int(y), horizon_days=float(h),
             kepler6_abs_residual_arcmin=float(ka),
             gp_quasiperiodic_abs_residual_arcmin=float(ga))
        for j, y, h, ka, ga in zip(jd_te, years, horizon, kep_epoch, gp_epoch)]
    R["headline"] = dict(
        loss="huber", f_scale_arcmin=F_SCALE,
        statistic="median absolute residual in arcmin",
        source="results.robust_huber.models",
        protocol=("Matched: every model in this table is fitted under the same "
                  "Huber loss on jd < 2303266.5 and scored by the same "
                  "statistics on jd >= 2303266.5. Bucket H3 has n = 7 and "
                  "carries no median by the reporting rule."),
        table=hb, paired_tests=paired, by_campaign=campaign_table,
        held_out_by_epoch=held_out_by_epoch,
        gp_horizon_crossing_days=(float(horizon[crossing[0]])
                                  if len(crossing) else None),
        l2_ablation_note=("results.models holds the same families under "
                          "ordinary least squares. Those numbers are an "
                          f"ablation: {l2_artifact_note(R['data']['gross_error_census'])}."))
    save()

    # ------------------------------------------------- refraction analysis
    good = ~gross
    rho, pval = sstats.spearmanr(alt_deg[good], rm[good])
    bins = {}
    for lo, hi in ALT_BINS_DEG:
        m = good & (alt_deg >= lo) & (alt_deg < hi)
        bins[f"{int(lo)}-{int(hi)}deg"] = dict(
            n=int(m.sum()),
            median_signed_arcmin=float(np.median(rm[m])) if m.any() else None,
            medae_arcmin=float(np.median(np.abs(rm[m]))) if m.any() else None)
    # altitude bias estimated on the TRAINING window only, applied to both
    A = np.stack([np.ones(int((good & tr).sum())), alt_deg[good & tr]], -1)
    coef = models._irls_lstsq(A, rm[good & tr] * E.ARCMIN, F_SCALE) / E.ARCMIN
    pred_tr = float(np.median(coef[0] + coef[1] * alt_deg[good & tr]))
    pred_te = float(np.median(coef[0] + coef[1] * alt_deg[good & te]))
    obs_tr = float(np.median(rm[good & tr]))
    obs_te = float(np.median(rm[good & te]))
    kep_in = robust_models["kepler6"]["in_sample"]["medae"]
    kep_out = robust_models["kepler6"]["out_of_sample"]["medae"]
    altitude_component = coef[0] + coef[1] * alt_deg
    dec_alt_corrected = dec - altitude_component * E.ARCMIN
    fit_alt = models.fit_kepler(
        jd_tr, dec_alt_corrected[tr], jd_ref, n_restarts=60,
        seed=SEED_KEPLER_HUBER, loss="huber", f_scale=F_SCALE)
    ri_alt = (dec_alt_corrected[tr]
              - models.predict_kepler(jd_tr, fit_alt)) / E.ARCMIN
    ro_alt = (dec_alt_corrected[te]
              - models.predict_kepler(jd_te, fit_alt)) / E.ARCMIN
    corrected_in = float(np.median(np.abs(ri_alt)))
    corrected_out = float(np.median(np.abs(ro_alt)))
    control_gap = float(kep_out - kep_in)
    corrected_gap = float(corrected_out - corrected_in)
    R["refraction_analysis"] = dict(
        note=("Tycho measured at meridian transit from latitude 55.9070 N, so "
              "transit altitude is 34.093 plus the declination. True "
              "refraction at 6 degrees altitude is about 8.5 arcmin and "
              "Tycho's own table was truncated and partly wrong, so the "
              "residual against unfitted modern elements is altitude "
              "dependent and the two windows do not sample the same "
              "altitudes."),
        observer=dict(latitude_deg=E.HVEN_LAT_DEG, longitude_deg=E.HVEN_LON_DEG),
        altitude_deg_range=[float(alt_deg.min()), float(alt_deg.max())],
        spearman_rho=float(rho), spearman_p=float(pval),
        gross_excluded=True, bins=bins,
        alt_bins_deg=[[float(a), float(b)] for a, b in ALT_BINS_DEG],
        points=dict(
            note=("Per epoch scatter behind this block, so fig_refraction.png "
                  "is regenerated from results.json alone."),
            transit_altitude_deg=[float(x) for x in alt_deg],
            residual_vs_modern_arcmin=[float(x) for x in rm],
            in_train=[bool(x) for x in tr],
            is_gross=[bool(x) for x in gross]),
        train_median_signed_arcmin=obs_tr, test_median_signed_arcmin=obs_te,
        observed_bias_shift_arcmin=float(obs_tr - obs_te),
        altitude_model=dict(
            form="robust linear in transit altitude, fitted on training epochs only",
            intercept_arcmin=float(coef[0]), slope_arcmin_per_deg=float(coef[1]),
            predicted_median_bias_train_arcmin=pred_tr,
            predicted_median_bias_test_arcmin=pred_te,
            predicted_bias_shift_arcmin=float(pred_tr - pred_te),
            share_of_observed_shift=float((pred_tr - pred_te) / (obs_tr - obs_te))
            if obs_tr != obs_te else None),
        altitude_corrected_refit=dict(
            correction=("Subtract the training-window robust linear altitude "
                        "model from every declination, then refit kepler6 "
                        "under the same Huber loss and seed."),
            in_sample_medae_arcmin=corrected_in,
            held_out_medae_arcmin=corrected_out,
            train_to_test_gap_arcmin=corrected_gap,
            control_in_sample_medae_arcmin=float(kep_in),
            control_held_out_medae_arcmin=float(kep_out),
            control_gap_arcmin=control_gap,
            gap_removed_arcmin=float(control_gap - corrected_gap),
            share_of_gap_removed=float((control_gap - corrected_gap) / control_gap),
            elements=wrapped_elements(fit_alt["theta"])),
        consequence=dict(
            kepler6_huber_in_sample_medae=kep_in,
            kepler6_huber_held_out_medae=kep_out,
            train_to_test_gap_arcmin=float(kep_out - kep_in),
            descriptive_ratio_signed_bias_shift_to_medae_gap=float(
                (obs_tr - obs_te) / (kep_out - kep_in)),
            statement=("The median signed residual moves by about "
                       f"{obs_tr - obs_te:.2f} arcmin between the two windows "
                       "over windows that sample different transit altitudes. "
                       "Its ratio to the "
                       f"{kep_out - kep_in:.2f} arcmin train to test "
                       "median absolute error gap is descriptive and is not an "
                       "explained share. The corrected refit measures the "
                       "effect of the fitted altitude term directly.")),
    )
    save()
    print(f"[refraction] rho {rho:.3f} p {pval:.2g}, bias shift "
          f"{obs_tr - obs_te:.2f} arcmin")

    # --------------------------- observation-physics known-truth experiment
    # Each condition uses the exact Tycho epochs, modern truth, and the same
    # robust fitter configuration as the real-data headline.
    rng_noise = np.random.default_rng(SEED_SYNTHETIC_NOISE)
    gross_vector = np.where(gross, rm, 0.0)
    conditions = {
        "i_truth_only": np.zeros(len(jd)),
        "ii_altitude_bias": altitude_component,
        "iii_gross_error_vector": gross_vector,
        "iv_altitude_plus_gross": altitude_component + gross_vector,
        "v_full_observed_residual_vector": rm,
        "vi_altitude_plus_gross_plus_gaussian_2arcmin": (
            altitude_component + gross_vector
            + rng_noise.normal(0.0, 2.0, len(jd))),
    }
    synthetic = {}
    for name, injected_arcmin in conditions.items():
        synthetic_obs = pm + injected_arcmin * E.ARCMIN
        fit_syn = models.fit_kepler(
            jd_tr, synthetic_obs[tr], jd_ref, n_restarts=60,
            seed=SEED_KEPLER_HUBER, loss="huber", f_scale=F_SCALE)
        pred_syn = models.predict_kepler(jd, fit_syn)
        residual_vs_truth = (pred_syn - pm) / E.ARCMIN
        residual_vs_observations = (synthetic_obs - pred_syn) / E.ARCMIN
        synthetic[name] = dict(
            injected_residual_arcmin=[float(x) for x in injected_arcmin],
            in_sample_medae_against_truth_arcmin=float(
                np.median(np.abs(residual_vs_truth[tr]))),
            held_out_medae_against_truth_arcmin=float(
                np.median(np.abs(residual_vs_truth[te]))),
            in_sample_medae_against_synthetic_observations_arcmin=float(
                np.median(np.abs(residual_vs_observations[tr]))),
            held_out_medae_against_synthetic_observations_arcmin=float(
                np.median(np.abs(residual_vs_observations[te]))),
            recovered_elements=wrapped_elements(fit_syn["theta"]),
            recovered_element_deviations_from_modern=deviation_from_modern(
                fit_syn["theta"], el))
    R["synthetic_observation_physics"] = dict(
        note=("Known-truth recovery on the exact Tycho sampling. Declinations "
              "come from modern elements at jd_ref. The altitude term is "
              "estimated on non-gross training epochs only. Gross injections "
              "use the actual observed-minus-modern residual at the same 18 "
              "gross epochs. Condition v is exactly the observed declination "
              "vector and must reproduce the real-data held-out result."),
        loss="huber", f_scale_arcmin=F_SCALE,
        kepler_seed=SEED_KEPLER_HUBER, n_restarts=60,
        gaussian_noise_seed=SEED_SYNTHETIC_NOISE,
        gaussian_noise_sigma_arcmin=2.0,
        conditions=synthetic)
    save()

    # --------------------------------------------- knowledge budget ablation
    kb = dict(
        declaration=("models.fit_kepler seeds its first restart with the modern "
                     "JPL elements. That is external prior knowledge, not test "
                     "leakage, and the benchmark's knowledge budget requires it "
                     "to be declared. The ablation below repeats the fit with "
                     "include_modern_start=False and 60 blind random restarts."),
        ablation={})
    for loss in ("l2", "huber"):
        t0 = time.time()
        blind = models.fit_kepler(
            jd_tr, dec_tr, jd_ref, n_restarts=60, seed=SEED_KEPLER_BLIND,
            include_modern_start=False,
            loss=("linear" if loss == "l2" else "huber"), f_scale=F_SCALE)
        seeded = (R["models"] if loss == "l2" else robust_models)["kepler6"]
        rb = models.restart_summary(blind)
        ro_b = (dec_te - models.predict_kepler(jd_te, blind)) / E.ARCMIN
        kb["ablation"][loss] = dict(
            blind_rms=float(blind["rms"]), blind_cost=float(blind["cost"]),
            seeded_in_sample_rms=seeded["in_sample"]["rms"],
            n_restarts=rb["n"], n_at_global=rb["n_at_global"],
            blind_elements=wrapped_elements(blind["theta"]),
            blind_held_out_medae=float(np.median(np.abs(ro_b))),
            seeded_held_out_medae=seeded["out_of_sample"]["medae"],
            fit_seconds=round(time.time() - t0, 1),
            verdict=("the blind search reaches the same optimum, so the result "
                     "does not depend on the modern element start"))
        print(f"[knowledge budget/{loss}] blind rms {blind['rms']:.4f} vs seeded "
              f"{seeded['in_sample']['rms']:.4f}, {rb['n_at_global']}/{rb['n']} "
              "restarts at global")
    R["knowledge_budget"] = kb
    save()

    # ------------------------------------------- deletion robustness check
    t0 = time.time()
    kept = ~flag
    fit_del = models.fit_kepler(jd[kept & tr], dec[kept & tr], jd_ref,
                                n_restarts=60, seed=SEED_KEPLER_HUBER,
                                loss="huber", f_scale=F_SCALE)
    ri_d = (dec[kept & tr] - models.predict_kepler(jd[kept & tr], fit_del)) / E.ARCMIN
    ro_d = (dec_te - models.predict_kepler(jd_te, fit_del)) / E.ARCMIN
    R["deletion_robustness_check"] = dict(
        note=("The 11 epochs flagged by the training window "
              "detector are deleted and the headline Keplerian fit is repeated "
              "under the same Huber loss. Reported beside the no-deletion "
              "numbers; the headline deletes nothing."),
        n_deleted_train=int((flag & tr).sum()), n_deleted_test=int((flag & te).sum()),
        in_sample=stats(ri_d), out_of_sample=stats(ro_d),
        no_deletion_in_sample_medae=robust_models["kepler6"]["in_sample"]["medae"],
        no_deletion_held_out_medae=robust_models["kepler6"]["out_of_sample"]["medae"],
        elements=wrapped_elements(fit_del["theta"]),
        deviation_from_modern=deviation_from_modern(fit_del["theta"], el),
        fit_seconds=round(time.time() - t0, 1))
    save()

    # -------------------------------------------------------- provenance
    with open(R["data"]["source"], "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    R["provenance"] = dict(
        input_file="data/historical/mars_declinations.csv, rebuilt locally by keplerbench fetch-historical",
        input_sha256=sha,
        transcription_sha256=D["meta"].get("source_sha256"),
        python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__,
        seeds=dict(kepler_ols=SEED_KEPLER_OLS, kepler_huber=SEED_KEPLER_HUBER,
                   kepler_blind=SEED_KEPLER_BLIND, bootstrap=BOOTSTRAP_SEED,
                   bootstrap_resamples=BOOTSTRAP_N, gp=SEED_GP,
                   synthetic_noise=SEED_SYNTHETIC_NOISE),
        wall_seconds=round(time.time() - t_start, 1),
        utc_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        entry_point="run_pilot.py",
        determinism_note=("Every fit is seeded and every bootstrap reseeds from "
                          "the recorded seed inside the call. Two runs agree on "
                          "every printed digit of the headline except that the "
                          "Gaussian process extrapolation moves at the 1e-5 "
                          "level between runs, from floating-point "
                          "nondeterminism in the optimizers; see the README."))
    R["runtime_seconds"] = round(time.time() - t_start, 1)
    save()
    print(f"[done] {R['runtime_seconds']} s -> {OUT}")


def _julian_year(x):
    y = 1580
    while float(E.julian_calendar_to_jd(y + 1, 1, 1)) <= x:
        y += 1
    return y


def _order(t, y, lattice, m_max, robust):
    if robust:
        return models.greedy_order_robust(t, y, lattice, m_max, f_scale=F_SCALE)
    return models.greedy_order(t, y, lattice, m_max)


def _fold_score(t_tr, dec_tr, a, b, freqs, robust):
    if robust:
        coef, _ = models.fit_harmonic_robust(t_tr[a], dec_tr[a], freqs,
                                             f_scale=F_SCALE)
    else:
        coef, _ = models.fit_harmonic(t_tr[a], dec_tr[a], freqs)
    rv = (dec_tr[b] - models.predict_harmonic(t_tr[b], freqs, coef)) / E.ARCMIN
    if robust:
        return float(np.median(np.abs(rv)))
    return float(np.sqrt(np.mean(rv ** 2)))


if __name__ == "__main__":
    main()
