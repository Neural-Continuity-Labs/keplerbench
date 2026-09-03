"""Calendar and time.

The Julian-to-JD conversion is checked against six values that come from outside
this codebase (the Gregorian reform date, the J2000 epoch, the JD epoch itself and
three worked examples from Meeus, Astronomical Algorithms, chapter 7). The 10-day
question is then decided, not assumed, by asking which time convention agrees with
an independent unfitted ephemeris.
"""

import hashlib
import os
import re

import numpy as np
import pytest

import constants as S
from conftest import medae

# (year, month, day, hour), expected JD, provenance of the expected value
CALENDAR_REFERENCES = [
    ((1582, 10, 5, 0.0), 2299160.5,
     "Julian 1582 Oct 5.0 is Gregorian 1582 Oct 15.0, the reform boundary"),
    ((1999, 12, 19, 12.0), 2451545.0,
     "J2000.0 = Gregorian 2000 Jan 1.5 = Julian 1999 Dec 19.5"),
    ((-4712, 1, 1, 12.0), 0.0,
     "the Julian Day epoch, 4713 BC Jan 1.5 Julian"),
    ((333, 1, 27, 12.0), 1842713.0,
     "Meeus, Astronomical Algorithms, example 7.b"),
    ((837, 4, 10, 7.2), 2026871.8,
     "Meeus example 7.a, 837 Apr 10.3 Julian"),
    ((-1000, 7, 12, 12.0), 1356001.0,
     "Meeus chapter 7 table, -1000 July 12.5 Julian"),
]


@pytest.mark.parametrize("args,expected,why", CALENDAR_REFERENCES,
                         ids=[r[2][:28] for r in CALENDAR_REFERENCES])
def test_julian_calendar_to_jd_against_independent_references(ephem, args,
                                                              expected, why):
    """B1. Six references, none of them produced by this codebase."""
    got = float(ephem.julian_calendar_to_jd(*args))
    assert abs(got - expected) < 1e-9, f"{why}: got {got}, expected {expected}"


def test_julian_leap_rule(ephem):
    """The Julian calendar makes every fourth year a leap year, including 1500,
    which the Gregorian calendar does not. Getting this wrong shifts Tycho's
    dates by whole days."""
    y1500 = float(ephem.julian_calendar_to_jd(1500, 3, 1)
                  - ephem.julian_calendar_to_jd(1500, 2, 28))
    assert y1500 == 2.0, "1500 must be a leap year in the Julian calendar"
    y1600 = float(ephem.julian_calendar_to_jd(1601, 1, 1)
                  - ephem.julian_calendar_to_jd(1600, 1, 1))
    assert y1600 == 366.0
    y1595 = float(ephem.julian_calendar_to_jd(1596, 1, 1)
                  - ephem.julian_calendar_to_jd(1595, 1, 1))
    assert y1595 == 365.0


def test_jd_half_day_convention(ephem):
    """JD rolls over at noon, so a civil midnight lands on a .5 boundary."""
    jd = float(ephem.julian_calendar_to_jd(1594, 1, 1))
    assert jd % 1.0 == 0.5
    assert jd == S.SPLIT_JD


def test_hour_and_minute_are_additive(ephem):
    base = float(ephem.julian_calendar_to_jd(1590, 6, 15))
    # 1e-9 day is float64 resolution at JD 2.3e6, not a tolerance choice
    assert abs(float(ephem.julian_calendar_to_jd(1590, 6, 15, 6.0)) - base
               - 0.25) < 1e-9
    assert abs(float(ephem.julian_calendar_to_jd(1590, 6, 15, 0.0, 30.0)) - base
               - 30.0 / 1440.0) < 1e-9


def test_month_rollover(ephem):
    """January and February are folded into the previous year by the algorithm;
    the fold must not leak a day."""
    for y in (1583, 1588, 1592, 1600):
        for m in range(1, 13):
            a = float(ephem.julian_calendar_to_jd(y, m, 1))
            b = float(ephem.julian_calendar_to_jd(y if m < 12 else y + 1,
                                                  m + 1 if m < 12 else 1, 1))
            assert b - a in (28.0, 29.0, 30.0, 31.0), (y, m, b - a)


# --------------------------------------------------------------- the source
def test_source_file_identity(raw):
    """The whole pilot is one spreadsheet. Pin it."""
    assert raw["shape"] == S.XLS_SHAPE
    assert raw["n"] == S.N_RAW_ROWS
    h = hashlib.sha256(open(S.XLS, "rb").read()).hexdigest()
    assert h == S.XLS_SHA256, "input file changed; every number below is stale"


def test_time_invariant_I1_requires_minutes(ephem, raw):
    """A1. jd = col14 + 1721048 must equal the Julian-calendar conversion of
    (year, month, day, hour, MINUTE) on all 923 rows."""
    jd = raw["days_since_1ad"] + S.PAFKO_JD_OFFSET
    ref = ephem.julian_calendar_to_jd(raw["year"], raw["month"], raw["day"],
                                      raw["hour"], raw["minute"])
    err = np.max(np.abs(jd - ref))
    assert err < S.TIME_IDENTITY_TOL_DAY, f"max identity error {err} day"


def test_hour_only_identity_is_false(ephem, raw):
    """A5, first half. An identity that drops the minute column is false on
    this file, by up to 59 minutes."""
    jd = raw["days_since_1ad"] + S.PAFKO_JD_OFFSET
    hour_only = ephem.julian_calendar_to_jd(raw["year"], raw["month"],
                                            raw["day"]) + raw["hour"] / 24.0
    err = float(np.max(np.abs(jd - hour_only)))
    assert err > S.TIME_IDENTITY_TOL_DAY, (
        "if this ever passes, the minute column became empty and the docstring "
        "claim would be harmless; check the input file")
    assert err > 0.04, f"expected about 0.041 day, got {err}"


def test_loader_docstring_states_the_verified_identity(pilot_dir):
    """A5. The docstring must not claim an identity the data refutes."""
    src = open(os.path.join(pilot_dir, "pilot_data.py")).read()
    m = re.search(r"col14 \+ 1721048\.0 ==[^\n]*", src)
    assert m, "the identity claim vanished from pilot_data.py; re-verify it"
    claim = m.group(0)
    assert "minute" in claim or "hour/24 + minute" in claim, (
        "pilot_data.py claims '" + claim.strip() + "' but the identity only holds "
        "when the minute column (13) is included; measured error without it "
        "is 0.041 day")


def test_declination_invariant_I3(raw):
    """A2. Column 20 is signed and consistent with the sign column, so the gross
    outliers are source errors, not a loader sign bug."""
    assert set(np.unique(raw["sign"])) == {-1.0, 1.0}
    rebuilt = raw["sign"] * (np.abs(raw["deg"]) + raw["arcmin"] / 60.0
                             + raw["arcsec"] / 3600.0)
    err = float(np.max(np.abs(rebuilt - raw["dec"])))
    assert err < S.DEC_IDENTITY_TOL_DEG, f"max {err} deg"
    assert int(np.sum(raw["sign"] < 0)) == int(np.sum(raw["dec"] < 0)) == 322


def test_day_adj_column_is_the_ten_day_trap(raw):
    """Pafko's column 11 is column 10 plus exactly 10 on every row. This is the
    single fact behind the whole 10-day question, so pin it."""
    diff = np.unique(raw["day_adj"] - raw["day"])
    assert diff.tolist() == [10.0]


# ------------------------------------------------- the 10-day question (I2)
def test_upstream_convention_is_about_ten_days_late(ephem, raw):
    """A4, first half. Reproduce the upstream loader's time construction and
    measure its offset against the verified one."""
    correct = raw["days_since_1ad"] + S.PAFKO_JD_OFFSET
    yr = np.floor(raw["decimal_date"]).astype(int)
    frac = raw["decimal_date"] - yr
    jan1 = ephem.julian_calendar_to_jd(yr, np.ones_like(yr), np.ones_like(yr))
    upstream = jan1 + frac * 365.25
    off = upstream - correct
    lo, hi = S.UPSTREAM_SHIFT_RANGE_DAY
    assert lo < float(np.mean(off)) < hi, f"mean offset {np.mean(off)}"
    assert lo <= float(np.min(off)) and float(np.max(off)) <= hi
    assert float(np.std(off)) > 0.1, (
        "the spread is the lost time of day; if it vanished the upstream "
        "loader changed")


@pytest.mark.parametrize("shift,floor", [(10.0, S.SHIFT_10D_MEDAE_MIN),
                                         (-10.0, S.SHIFT_10D_MEDAE_MIN),
                                         (1.0, S.SHIFT_1D_MEDAE_MIN)])
def test_wrong_epoch_is_rejected_by_an_external_ephemeris(ephem, split, modern,
                                                          shift, floor):
    """A4, C4, C5. The convention is DECIDED, not assumed: shifting every
    timestamp degrades agreement with unfitted modern elements."""
    el, jd, dec = modern, split["jd"], split["dec"]
    p = ephem.predicted_declination(jd + shift, el["a"], el["e"], el["inc"],
                                    el["node"], el["argp"], el["M0"],
                                    split["jd_ref"])
    m = medae((dec - p) / ephem.ARCMIN)
    assert m >= floor, (
        f"a {shift:+g} day error only costs {m:.2f} arcmin; the test cannot "
        "distinguish the calendar conventions and the threshold is too loose")


def test_zero_shift_is_the_argmin(ephem, split, modern):
    """A4. Of the five candidate offsets, the pilot's convention is best."""
    el, jd, dec = modern, split["jd"], split["dec"]
    scores = {}
    for s in (-10.0, -1.0, 0.0, 1.0, 10.0):
        p = ephem.predicted_declination(jd + s, el["a"], el["e"], el["inc"],
                                        el["node"], el["argp"], el["M0"],
                                        split["jd_ref"])
        scores[s] = medae((dec - p) / ephem.ARCMIN)
    best = min(scores, key=scores.get)
    assert best == 0.0, f"best offset was {best:+g} day, scores {scores}"
    assert scores[10.0] / scores[0.0] > 5.0, (
        f"decision margin only {scores[10.0] / scores[0.0]:.1f}x")
