"""Reduced Tycho Mars declination series for the pilot.

The epoch table is rebuilt locally by `keplerbench fetch-historical`, which
downloads Pafko's transcription and applies the reduction documented in
docs/historical.md: the Julian day is Pafko's day count (column 14) plus
1721048.0, repeat readings at one timestamp become their median, and timestamps
whose readings spread by more than one degree are dropped. This module reads
that table and adds the training-window gross-error detector used for the
published census and the deletion robustness check.

DATE CONVENTION. Column 14 satisfies, to 4.66e-10 day on every row,

    col14 + 1721048.0 == julian_calendar_to_jd(year, month, day, hour, minute)

with the raw Julian-calendar date columns, minute column included. Pafko's
decimal-date column is not used: it derives from a day column that already
carries the 10 day Gregorian shift, so anchoring it on the Julian New Year
applies the shift twice and runs about 10 days late. The convention is decided
by an external, never-fitted reference: the median absolute error of unfitted
modern elements against these declinations with every timestamp shifted by
s days is

    s = -10 : 53.65 arcmin      s = -1 : 8.62 arcmin
    s =   0 :  3.65 arcmin      s = +1 : 4.79 arcmin
    s = +10 : 51.46 arcmin

and s = 0 is the argmin by more than a factor of 13. run_pilot.py recomputes
this table into results.json["data"]["shift_decision_table"].

OUTLIER PROTOCOL. The headline analysis DELETES NOTHING: all 354 epochs enter
train and test, and gross errors are handled in the loss (Huber, f_scale 5
arcmin), not by deletion. A detector is still computed, as a published census
and as a robustness check only, and it is computed on the TRAINING WINDOW
ONLY so that no test epoch is ever judged by its test-period neighbours.

The detector is a robust local quadratic (window 45 days, at least 8
neighbours, IRLS Huber weights at 5 arcmin, threshold 60 arcmin, iterated).
Scored against the reference criterion "residual against unfitted modern
elements above 60 arcmin" it reaches precision 0.909 and recall 0.556. Its
recall ceiling is 0.833, because 3 of the 18 gross errors lie in the test
window and are out of scope by construction; of the 15 in the training window
5 are isolated epochs with fewer than 8 neighbours inside 45 days, which no
local smoothness test can judge. That ceiling is the reason the headline
analysis handles gross errors in the loss instead of by deletion.
"""

import csv
import json
import pathlib

import numpy as np

import ephem as E

HERE = pathlib.Path(__file__).resolve().parent
CSV = HERE.parents[1] / "data" / "historical" / "mars_declinations.csv"
PAFKO_JD_OFFSET = 1721048.0

DETECTOR = dict(window_days=45.0, degree=2, min_neighbours=8,
                threshold_arcmin=60.0, f_scale_arcmin=5.0, n_iter=6)


def load(path=CSV):
    """Read the rebuilt epoch table: Julian day, declination (rad), repeats.

    The sidecar written beside the table by `keplerbench fetch-historical`
    supplies the reduction counts and the transcription checksum; without it
    those fields are None.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `keplerbench fetch-historical` first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    jd = np.array([float(r["jd"]) for r in rows])
    dec = np.array([float(r["declination_deg"]) for r in rows]) * E.DEG
    nrep = np.array([int(r["n_repeats"]) for r in rows])
    order = np.argsort(jd, kind="stable")
    jd, dec, nrep = jd[order], dec[order], nrep[order]
    meta = dict(n_rows=None, n_epochs=len(jd), n_dropped_spread=None,
                source_sha256=None, table=str(path))
    sidecar = path.with_name(path.stem + ".meta.json")
    if sidecar.exists():
        m = json.loads(sidecar.read_text())
        meta.update(n_rows=m.get("n_raw_rows"),
                    n_dropped_spread=m.get("n_dropped_spread"),
                    source_sha256=m.get("source_sha256"))
    return jd, dec, nrep, meta


def _irls_poly(V, y, f_scale, n_iter=12):
    """Huber-weighted polynomial fit by iteratively reweighted least squares."""
    w = np.ones(len(y))
    coef = np.zeros(V.shape[1])
    for _ in range(n_iter):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(V * sw[:, None], y * sw, rcond=None)
        a = np.abs(y - V @ coef)
        w_new = np.where(a <= f_scale, 1.0, f_scale / np.maximum(a, 1e-12))
        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w = w_new
    return coef


def local_robust_outliers(jd, dec, candidates, window_days=45.0, degree=2,
                          min_neighbours=8, threshold_arcmin=60.0,
                          f_scale_arcmin=5.0, n_iter=6):
    """Model free gross-error detector, computed on `candidates` only.

    For every candidate epoch a robust polynomial is fitted to its neighbours
    inside `window_days`, with the epoch itself and every already flagged
    epoch excluded, and the epoch is flagged when it sits further than
    `threshold_arcmin` from that local prediction. The Huber scale is fixed at
    Tycho's own repeat scatter instead of a local median absolute deviation,
    which is what let the previous detector be swamped by the consecutive runs
    of corrupted rows and flag their honest neighbours instead.

    Epochs outside `candidates` are never flagged and never enter a
    neighbourhood, so the flag on the training window is independent of every
    test-period observation.

    Returns (flag, resid_arcmin, n_neighbours).
    """
    n = len(jd)
    candidates = np.asarray(candidates, bool)
    flag = np.zeros(n, bool)
    resid = np.full(n, np.nan)
    nnb = np.zeros(n, int)
    for _ in range(n_iter):
        new = flag.copy()
        for i in np.where(candidates)[0]:
            sel = candidates & (np.abs(jd - jd[i]) <= window_days) & (~flag)
            sel[i] = False
            nnb[i] = int(sel.sum())
            if nnb[i] < min_neighbours:
                continue
            t = (jd[sel] - jd[i]) / window_days
            if np.ptp(t) < 0.15:
                continue
            deg = min(degree, nnb[i] - 2)
            if deg < 1:
                continue
            V = np.vander(t, deg + 1)
            coef = _irls_poly(V, dec[sel] / E.ARCMIN, f_scale_arcmin)
            resid[i] = dec[i] / E.ARCMIN - coef[-1]     # prediction at t = 0
            new[i] = abs(resid[i]) > threshold_arcmin
        if (new == flag).all():
            flag = new
            break
        flag = new
    return flag, resid, nnb


def build(split_year=1594):
    """All 354 epochs, nothing deleted.

    `keep` is all True by design. `outlier` is the diagnostic
    flag of the training-window detector, published as a census and used only
    by the explicit deletion robustness check, never by the headline fits.
    """
    jd, dec, nrep, meta = load()
    split_jd = float(E.julian_calendar_to_jd(split_year, 1, 1))
    train = jd < split_jd
    flag, sres, nnb = local_robust_outliers(
        jd, dec, train,
        window_days=DETECTOR["window_days"], degree=DETECTOR["degree"],
        min_neighbours=DETECTOR["min_neighbours"],
        threshold_arcmin=DETECTOR["threshold_arcmin"],
        f_scale_arcmin=DETECTOR["f_scale_arcmin"], n_iter=DETECTOR["n_iter"])
    meta.update(split_jd=split_jd, split_year=split_year,
                detector=dict(DETECTOR),
                n_flagged=int(flag.sum()),
                flagged_jd=[float(x) for x in jd[flag]],
                n_deleted=0,
                deletion_note=("headline analysis deletes nothing; the flag is "
                               "a census and a robustness check, computed on "
                               "jd < split_jd only"))
    keep = np.ones(len(jd), bool)
    return dict(jd=jd, dec=dec, nrep=nrep, outlier=flag, smooth_resid=sres,
                n_neighbours=nnb, keep=keep, split_jd=split_jd, meta=meta)
