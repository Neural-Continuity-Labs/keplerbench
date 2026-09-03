"""Shared fixtures for the pilot test suite.

Nothing in this directory imports from run_pilot.py or run_robust.py at module
scope, so a broken run script cannot take the suite down with it.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PILOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PILOT not in sys.path:
    sys.path.insert(0, PILOT)

import constants as S  # noqa: E402  (after sys.path fix)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than about 10 seconds")


@pytest.fixture(scope="session")
def pilot_dir():
    return PILOT


@pytest.fixture(scope="session")
def ephem():
    import ephem
    return ephem


@pytest.fixture(scope="session")
def models():
    import models
    return models


@pytest.fixture(scope="session")
def raw(request):
    """The Pafko sheet, rows 6 onward, as numeric arrays (local cache only)."""
    if not os.path.exists(S.XLS):
        pytest.skip("Pafko spreadsheet not cached; run `keplerbench fetch-historical`")
    df = pd.read_excel(S.XLS, header=None)
    d = df.iloc[6:].reset_index(drop=True)
    num = lambda c: pd.to_numeric(d[c], errors="coerce").values  # noqa: E731
    return dict(
        shape=df.shape, n=len(d),
        year=num(S.COL_YEAR), month=num(S.COL_MONTH), day=num(S.COL_DAY),
        day_adj=num(S.COL_DAY_ADJ),
        hour=np.nan_to_num(num(S.COL_HOUR)), minute=np.nan_to_num(num(S.COL_MIN)),
        days_since_1ad=num(S.COL_DAYS_SINCE_1AD),
        decimal_date=num(S.COL_DECIMAL_DATE),
        sign=num(S.COL_SIGN), deg=num(S.COL_DEG),
        arcmin=np.nan_to_num(num(S.COL_ARCMIN)),
        arcsec=np.nan_to_num(num(S.COL_ARCSEC)),
        dec=num(S.COL_DEC),
        quote=d[4].astype(str).values,
    )


@pytest.fixture(scope="session")
def built():
    """pilot_data.build(1594) on the locally rebuilt epoch table."""
    import pilot_data as data
    if not os.path.exists(S.CSV):
        pytest.skip("epoch table not rebuilt; run `keplerbench fetch-historical`")
    return data.build(split_year=S.SPLIT_YEAR)


@pytest.fixture(scope="session")
def split(built):
    """Masks and arrays for the split, with and without outlier deletion."""
    jd, dec, keep = built["jd"], built["dec"], built["keep"]
    sj = built["split_jd"]
    tr_all, te_all = jd < sj, jd >= sj
    tr, te = keep & tr_all, keep & te_all
    jd_ref = float(np.median(jd[tr]))
    return dict(jd=jd, dec=dec, keep=keep, split_jd=sj,
                tr_all=tr_all, te_all=te_all, tr=tr, te=te,
                jd_tr=jd[tr], dec_tr=dec[tr], jd_te=jd[te], dec_te=dec[te],
                jd_ref=jd_ref, train_end=float(jd[tr].max()),
                horizon=jd[te] - float(jd[tr].max()))


@pytest.fixture(scope="session")
def modern(ephem, split):
    return ephem.mars_modern_elements(split["jd_ref"])


@pytest.fixture(scope="session")
def modern_resid(ephem, split, modern):
    """Residual in arcmin of every epoch against the unfitted modern elements."""
    el, jd, dec = modern, split["jd"], split["dec"]
    p = ephem.predicted_declination(jd, el["a"], el["e"], el["inc"], el["node"],
                                    el["argp"], el["M0"], split["jd_ref"])
    return (dec - p) / ephem.ARCMIN


@pytest.fixture(scope="session")
def results(pilot_dir):
    import json
    """The full results file after a run, else the committed summary."""
    for name in ("results.json", "results_summary.json"):
        path = os.path.join(pilot_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    pytest.skip("no results file; run run_pilot.py")


# ------------------------------------------------------------------ helpers
def wrap(x):
    """Wrap an angle or array of angles to (-pi, pi]."""
    return (np.asarray(x, float) + np.pi) % (2 * np.pi) - np.pi


def medae(r):
    return float(np.median(np.abs(np.asarray(r, float))))


def transit_altitude_deg(dec_rad):
    """Meridian transit altitude at Hven for a body of declination dec."""
    return 90.0 - S.HVEN_LAT_DEG + np.degrees(dec_rad)
