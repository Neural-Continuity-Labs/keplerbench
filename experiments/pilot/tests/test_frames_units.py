"""Frames, units and sign conventions.

Everything here is a closed-form or independently known quantity. No model is
fitted anywhere in this file.
"""

import os
import re

import numpy as np
import pytest

import constants as S
from conftest import transit_altitude_deg


# ------------------------------------------------------------------- units
def test_angle_constants(ephem):
    assert ephem.DEG == pytest.approx(np.pi / 180.0, rel=0, abs=0)
    assert ephem.ARCMIN == pytest.approx(np.pi / 180.0 / 60.0)
    assert ephem.ARCSEC == pytest.approx(np.pi / 180.0 / 3600.0)
    assert ephem.ARCMIN / ephem.ARCSEC == pytest.approx(60.0)


def test_gm_sun_is_gaussian_constant_squared(ephem):
    """GM in AU^3/day^2 must be k^2 with k = 0.01720209895, otherwise Kepler's
    third law is silently rescaled and 'a' comes out wrong."""
    assert ephem.GM_SUN == pytest.approx(0.01720209895 ** 2, rel=1e-15)
    n_earth = np.sqrt(ephem.GM_SUN / 1.0 ** 3)
    assert 2 * np.pi / n_earth == pytest.approx(365.2569, abs=0.01)
    n_mars = np.sqrt(ephem.GM_SUN / 1.52371034 ** 3)
    assert 2 * np.pi / n_mars == pytest.approx(686.98, abs=0.5)


def test_light_speed_constant(ephem):
    """AU per day; 499.005 s per AU."""
    assert ephem.C_AU_PER_DAY == pytest.approx(86400.0 / 499.004784, rel=1e-6)


# -------------------------------------------------------------- obliquity
def test_obliquity_at_j2000(ephem):
    """B2. IAU value 23 deg 26 arcmin 21.448 arcsec."""
    _, eps = ephem.sun_longitude_and_obliquity(np.array([2451545.0]))
    assert float(np.degrees(eps[0])) == pytest.approx(S.EPS_J2000_DEG, abs=1e-6)
    assert ephem.EPS_J2000 == pytest.approx(np.radians(S.EPS_J2000_DEG))


def test_obliquity_over_the_tycho_span_matches_iau1980(ephem):
    """B2. Independent reference series: eps0 = 23d26'21.448" - 46.8150"T
    - 0.00059"T^2 + 0.001813"T^3."""
    jd = ephem.julian_calendar_to_jd(np.array([1582, 1590, 1600]), 1, 1)
    T = (np.asarray(jd, float) - 2451545.0) / 36525.0
    ref = (23.0 + 26 / 60.0 + 21.448 / 3600.0
           + (-46.8150 * T - 0.00059 * T ** 2 + 0.001813 * T ** 3) / 3600.0)
    _, eps = ephem.sun_longitude_and_obliquity(jd)
    err_arcsec = np.abs(np.degrees(eps) - ref) * 3600.0
    assert float(np.max(err_arcsec)) < S.OBLIQUITY_AGREEMENT_ARCSEC
    assert 23.48 < float(np.degrees(eps[1])) < 23.50, "1590 obliquity"


# ------------------------------------------------------------- precession
def test_precession_matrix_is_orthonormal(ephem):
    T = np.array([-4.2, -4.0, 0.0, 0.5])
    P = ephem.precession_matrix(T)
    for i in range(len(T)):
        assert np.allclose(P[i].T @ P[i], np.eye(3), atol=S.ORTHONORMAL_TOL)
        assert float(np.linalg.det(P[i])) == pytest.approx(1.0, abs=1e-12)


def test_precession_is_identity_at_j2000(ephem):
    P = ephem.precession_matrix(np.array([0.0]))
    assert np.allclose(P[0], np.eye(3), atol=S.ORTHONORMAL_TOL)


def test_precession_magnitude_over_four_centuries(ephem):
    """General precession is about 50.29 arcsec per year, so 1590 to 2000 is
    about 5.7 degrees. A sign flip or a factor of 100 shows up here."""
    T = np.array([-4.1])
    P = ephem.precession_matrix(T)[0]
    ang = np.degrees(np.arccos(np.clip((np.trace(P) - 1) / 2.0, -1, 1)))
    assert 5.0 < ang < 6.5, f"rotation angle {ang} deg"


def test_precession_direction(ephem):
    """General precession is retrograde: the equinox drifts west at about 50.29
    arcsec per year, so a direction whose J2000 longitude is zero has NEGATIVE
    longitude when referred to the equinox of 1590. P maps J2000 to date, so
    P @ x_J2000 must land at RA about -5.7 degrees. A sign error in zeta or z
    flips this and moves every predicted declination by degrees."""
    T = np.array([-4.1])
    P = ephem.precession_matrix(T)[0]
    v2000 = np.array([1.0, 0.0, 0.0])          # equinox of J2000
    ra = np.degrees(np.arctan2(*(P @ v2000)[[1, 0]]))
    assert ra < 0.0, "sign of zeta/z reversed"
    assert -6.5 < ra < -4.5, f"RA of the J2000 equinox in 1590 is {ra} deg"
    ra_back = np.degrees(np.arctan2(*(P.T @ v2000)[[1, 0]]))
    assert ra_back == pytest.approx(-ra, rel=1e-3)


# ------------------------------------------------- ecliptic to equatorial
def test_rotx_maps_ecliptic_to_equatorial_with_the_right_sign(ephem):
    """B4. Ecliptic longitude 90 deg, latitude 0, is the summer solstice point;
    its declination must be +eps, not -eps."""
    eps = ephem.EPS_J2000
    v = np.array([[0.0, 1.0, 0.0]])
    out = ephem._rotx(v, eps)
    dec = np.degrees(np.arcsin(out[0, 2] / np.linalg.norm(out[0])))
    assert dec == pytest.approx(S.EPS_J2000_DEG, abs=1e-9)
    v2 = np.array([[0.0, -1.0, 0.0]])
    out2 = ephem._rotx(v2, eps)
    assert np.degrees(np.arcsin(out2[0, 2])) == pytest.approx(-S.EPS_J2000_DEG,
                                                              abs=1e-9)


def test_rotx_preserves_norm_and_x(ephem):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 3))
    out = ephem._rotx(v, 0.4)
    assert np.allclose(np.linalg.norm(v, axis=-1), np.linalg.norm(out, axis=-1))
    assert np.allclose(v[:, 0], out[:, 0])


def test_degrees_not_radians_confusion_in_elements(ephem):
    """mars_modern_elements must return radians for the angles and a bare number
    for a and e. A degrees/radians slip is a factor of 57."""
    el = ephem.mars_modern_elements(2301512.58333)
    assert 1.5 < el["a"] < 1.55
    assert 0.08 < el["e"] < 0.11
    assert 0.02 < el["inc"] < 0.04, "inclination must be radians (about 1.85 deg)"
    assert abs(el["node"]) <= 2 * np.pi and abs(el["argp"]) <= 2 * np.pi
    assert abs(el["M0"]) <= 2 * np.pi


# ------------------------------------------------------- solar theory, Earth
def test_solar_longitude_at_a_known_equinox(ephem):
    """B6. The March 2000 equinox was 2000-03-20 07:35 UT = JD 2451623.8160.
    Geometric solar longitude there is 0 to within the low-precision series."""
    lam, _ = ephem.sun_longitude_and_obliquity(np.array([2451623.8160]))
    d = (np.degrees(lam[0]) + 180.0) % 360.0 - 180.0
    assert abs(d) < S.SOLAR_LONGITUDE_EQUINOX_TOL_DEG, f"lambda = {d} deg"


def test_solar_longitude_advances_about_one_degree_per_day(ephem):
    jd = 2301512.0 + np.arange(0, 366, 1.0)
    lam, _ = ephem.sun_longitude_and_obliquity(jd)
    d = np.diff(np.unwrap(lam))
    assert 0.95 < np.degrees(d).min() and np.degrees(d).max() < 1.05
    assert np.degrees(np.unwrap(lam)[-1] - lam[0]) == pytest.approx(360.0, abs=1.5)


def test_earth_sun_distance_range_and_perihelion(ephem):
    """Earth's radius vector runs 0.9833 to 1.0167 AU with perihelion in early
    January."""
    jd = 2451544.5 + np.arange(0, 366, 1.0)
    r = ephem.sun_radius_vector(jd)
    assert 0.9825 < r.min() < 0.9840
    assert 1.0160 < r.max() < 1.0175
    peri_doy = int(np.argmin(r))
    assert peri_doy < 10 or peri_doy > 360, f"perihelion at day {peri_doy}"


def test_earth_heliocentric_is_opposite_the_sun(ephem):
    """The Earth seen from the Sun is 180 degrees from the Sun seen from Earth.
    A missing minus sign here moves everything by half an orbit."""
    jd = np.array([2301512.0])
    lam, eps = ephem.sun_longitude_and_obliquity(jd)
    e = ephem.earth_heliocentric_eq_j2000(jd)[0]
    # rotate back to ecliptic of date and compare longitudes
    P = ephem.precession_matrix((jd - 2451545.0) / 36525.0)[0]
    e_date = P @ e
    e_ecl = ephem._rotx(e_date[None, :], -float(eps[0]))[0]
    lon = np.arctan2(e_ecl[1], e_ecl[0])
    d = np.degrees((lon - lam[0] - np.pi + np.pi) % (2 * np.pi) - np.pi)
    assert abs(d) < 1e-6, f"Earth longitude is {d} deg from Sun longitude + 180"
    assert 0.98 < np.linalg.norm(e) < 1.02


# ----------------------------------------------------- Kepler and two body
@pytest.mark.parametrize("e", [0.0, 0.0934, 0.3, 0.6, 0.9])
def test_kepler_equation_solver(ephem, e):
    """B7."""
    M = np.linspace(-3 * np.pi, 3 * np.pi, 401)
    E = ephem.solve_kepler(M, e)
    Mw = (M + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(E - e * np.sin(E) - Mw)) < S.KEPLER_SOLVER_TOL
    if e == 0.0:
        assert np.allclose(E, Mw, atol=1e-14)


def test_two_body_invariants(ephem):
    """B8. Sample one Mars orbit and check |r| bounds, the orbital period, and
    conservation of the areal rate."""
    a, e = 1.52371, 0.09339
    n = np.sqrt(ephem.GM_SUN / a ** 3)
    jd = 2451545.0 + np.linspace(0, 2 * np.pi / n, 2001)
    r = ephem.elements_to_eq_j2000(jd, a, e, 0.032, 0.865, 5.0, 0.0, 2451545.0)
    d = np.linalg.norm(r, axis=-1)
    assert d.min() == pytest.approx(a * (1 - e), rel=1e-6)
    assert d.max() == pytest.approx(a * (1 + e), rel=1e-6)
    # angular momentum from finite differences
    v = np.gradient(r, jd, axis=0)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)[5:-5]
    # tolerances set by the second-order finite difference, not by the model
    assert (hn.max() - hn.min()) / hn.mean() < 1e-5, "areal rate not constant"
    hexp = np.sqrt(ephem.GM_SUN * a * (1 - e ** 2))
    assert hn.mean() == pytest.approx(hexp, rel=1e-5)


def test_orbit_closes_after_one_period(ephem):
    a, e = 1.52371, 0.09339
    n = np.sqrt(ephem.GM_SUN / a ** 3)
    P = 2 * np.pi / n
    jd = np.array([2451545.0, 2451545.0 + P, 2451545.0 + 5 * P])
    r = ephem.elements_to_eq_j2000(jd, a, e, 0.032, 0.865, 5.0, 0.3, 2451545.0)
    assert np.max(np.abs(r[1] - r[0])) < 1e-10
    assert np.max(np.abs(r[2] - r[0])) < 1e-9


def test_inclination_sets_the_out_of_plane_excursion(ephem):
    """Degrees-for-radians in the inclination would show up as a 57x error in
    ecliptic latitude."""
    a, e, inc = 1.52371, 0.0, np.radians(1.85)
    jd = 2451545.0 + np.linspace(0, 687, 400)
    r = ephem.elements_to_eq_j2000(jd, a, e, inc, 0.0, 0.0, 0.0, 2451545.0)
    ecl = ephem._rotx(r, -ephem.EPS_J2000)
    lat = np.degrees(np.arcsin(ecl[:, 2] / np.linalg.norm(ecl, axis=-1)))
    assert np.max(np.abs(lat)) == pytest.approx(1.85, abs=0.01)


# ----------------------------------------------- declination sign conventions
def test_declination_sign_north_positive_from_the_source(ephem, raw):
    """B5. Tycho writes B. for borealis (north) and A. for australis (south).
    The first row is 'Declinatio 23 7 B' and must come out at +23.117 degrees."""
    i = 0
    assert raw["year"][i] == 1582 and raw["month"][i] == 11 and raw["day"][i] == 12
    assert raw["dec"][i] == pytest.approx(23.0 + 7.0 / 60.0, abs=1e-6)
    assert raw["sign"][i] == 1.0
    south = np.where(raw["sign"] < 0)[0]
    assert len(south) > 0 and np.all(raw["dec"][south] < 0)


def test_predicted_declination_sign_and_range(ephem, split):
    """A body north of the celestial equator gets a positive declination, and
    Mars over 1582 to 1600 stays inside +-31 degrees."""
    el = ephem.mars_modern_elements(split["jd_ref"])
    p = ephem.predicted_declination(split["jd"], el["a"], el["e"], el["inc"],
                                    el["node"], el["argp"], el["M0"],
                                    split["jd_ref"])
    assert np.all(np.abs(p) < np.radians(31.0))
    assert np.degrees(p).min() < -20 and np.degrees(p).max() > 20
    obs = np.degrees(split["dec"])
    assert S.DEC_RANGE_DEG[0] <= obs.min() and obs.max() <= S.DEC_RANGE_DEG[1]


def test_declination_agrees_in_sign_with_the_observations(ephem, split,
                                                          modern_resid):
    """A global sign flip would put every residual near twice the declination."""
    frac_ok = float(np.mean(np.abs(modern_resid) < S.GROSS_ARCMIN))
    assert frac_ok > 0.9, f"only {frac_ok:.2%} of epochs agree in sign and size"


# ------------------------------------------------- observer latitude, parallax
def test_diurnal_parallax_from_hven_is_below_the_stated_bound(ephem, split):
    """B9. The model is geocentric. Bound the neglected topocentric term so the
    paper can state it: Earth radius over the geocentric distance, times a
    geometry factor no larger than 1."""
    el = ephem.mars_modern_elements(split["jd_ref"])
    jd = split["jd"]
    earth = ephem.earth_heliocentric_eq_j2000(jd)
    mars = ephem.elements_to_eq_j2000(jd, el["a"], el["e"], el["inc"],
                                      el["node"], el["argp"], el["M0"],
                                      split["jd_ref"])
    dist = np.linalg.norm(mars - earth, axis=-1)
    par_arcmin = np.degrees(S.EARTH_RADIUS_AU / dist.min()) * 60.0
    assert dist.min() > 0.35, "closest approach unphysically small"
    assert par_arcmin <= S.MAX_PARALLAX_ARCMIN, (
        f"maximum diurnal parallax {par_arcmin:.3f} arcmin exceeds the bound "
        "stated in the ephem.py docstring")


def test_ephem_documents_the_neglected_parallax(pilot_dir):
    """B9. The docstring lists nutation, aberration and refraction but not the
    topocentric term, and never gives the observer's latitude."""
    src = open(os.path.join(pilot_dir, "ephem.py")).read()
    head = src[:src.index("DEG =")]
    assert re.search(r"parallax", head, re.I), (
        "ephem.py must name the neglected diurnal parallax and its magnitude")
    assert re.search(r"55\.9|Hven|Uraniborg", head, re.I), (
        "ephem.py must state the observer, latitude 55.9070 N on Hven")


def test_transit_altitude_helper(split):
    alt = transit_altitude_deg(split["dec"])
    assert alt.min() == pytest.approx(90 - S.HVEN_LAT_DEG
                                      + np.degrees(split["dec"]).min())
    assert 5.0 < alt.min() < 8.0 and 58.0 < alt.max() < 63.0
