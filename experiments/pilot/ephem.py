"""Data loading and geometry for the Kepler-test pilot.

The calendar and solar-theory functions duplicate the ones in the package's
historical module so that this directory runs on its own.

Frames. Everything is done in the equatorial mean frame of J2000, which is
inertial over the 18 year data span. Earth comes from the Meeus low precision
solar theory (geometric solar longitude and radius vector, referred to the
mean equinox of date), rotated to J2000 with the IAU 1976 precession matrix.
Mars comes from six orbital elements referred to the mean ecliptic and equinox
of J2000. The predicted declination is referred to the mean equator of date,
which is what Tycho measured.

Observer. Tycho measured at meridian transit from Uraniborg on Hven, latitude
55.9070 N, longitude 12.6962 E. This module is GEOCENTRIC: the observer is
placed at the centre of the Earth, so the diurnal (topocentric) parallax is
neglected. Bound on that term over this data span: Mars comes no closer than
0.3793 AU, and one Earth radius is 4.2635e-5 AU, so the parallax never exceeds
0.386 arcmin. That is below Tycho's own 2 to 4 arcmin scatter but not below
the 2.6 arcmin robust residual of the fitted Keplerian model, so it belongs in
the error budget. The meridian transit altitude, 90 - 55.9070 + dec, is
supplied by transit_altitude_deg and runs from 6.1 to 61.4 degrees over this
record.

Neglected terms, with magnitudes in declination:

    nutation                     about 9 arcsec  (0.15 arcmin)
    annual aberration            about 20 arcsec (0.33 arcmin)
    diurnal parallax from Hven   at most 0.386 arcmin
    refraction residual left by Tycho's own table   up to about 6 arcmin at
        low transit altitude; this one is NOT below his scatter and it is
        altitude dependent, so it is analysed rather than assumed away (see
        the refraction_analysis block of results.json)
"""

import numpy as np

DEG = np.pi / 180.0
ARCMIN = DEG / 60.0
ARCSEC = DEG / 3600.0
DAY_NONDIM = 2.0 * np.pi / 365.25
GM_SUN = 0.01720209895 ** 2      # AU^3 / day^2
C_AU_PER_DAY = 173.1446326846693
EPS_J2000 = 23.4392911 * DEG
HVEN_LAT_DEG = 55.9070
HVEN_LON_DEG = 12.6962
EARTH_RADIUS_AU = 4.2635e-5


# --------------------------------------------------------------------------
# Calendar and solar theory
# --------------------------------------------------------------------------
def julian_calendar_to_jd(year, month, day, hour=0.0, minute=0.0):
    """JD for a JULIAN-calendar date (Tycho's Denmark kept the Julian
    calendar; no Gregorian correction applies)."""
    y = np.asarray(year, dtype=float)
    m = np.asarray(month, dtype=float)
    d = np.asarray(day, dtype=float) + np.asarray(hour, dtype=float) / 24.0 \
        + np.asarray(minute, dtype=float) / 1440.0
    y2 = np.where(m <= 2, y - 1, y)
    m2 = np.where(m <= 2, m + 12, m)
    jd = np.floor(365.25 * (y2 + 4716)) + np.floor(30.6001 * (m2 + 1)) + d - 1524.5
    return jd


def sun_longitude_and_obliquity(jd):
    """Geometric true solar longitude (rad) and mean obliquity (rad),
    Meeus low-precision series."""
    T = (np.asarray(jd, dtype=float) - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    Mr = M * DEG
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(Mr)
        + (0.019993 - 0.000101 * T) * np.sin(2 * Mr)
        + 0.000289 * np.sin(3 * Mr)
    )
    lam = np.mod(L0 + C, 360.0) * DEG
    eps = (23.439291 - 0.0130042 * T - 1.64e-7 * T**2) * DEG
    return lam, eps


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def sun_radius_vector(jd):
    """Earth-Sun distance in AU (Meeus 25.5)."""
    T = (np.asarray(jd, dtype=float) - 2451545.0) / 36525.0
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T**2
    M = (357.52911 + 35999.05029 * T - 0.0001537 * T**2) * DEG
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(M)
        + (0.019993 - 0.000101 * T) * np.sin(2 * M)
        + 0.000289 * np.sin(3 * M)
    ) * DEG
    v = M + C
    return 1.000001018 * (1.0 - e**2) / (1.0 + e * np.cos(v))


def precession_matrix(T):
    """IAU 1976 equatorial precession from J2000 to the mean equinox of date.

    Returns P with shape [..., 3, 3] such that r_date = P @ r_J2000.
    """
    T = np.asarray(T, dtype=float)
    zeta = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * ARCSEC
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * ARCSEC
    theta = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) * ARCSEC
    cz, sz = np.cos(zeta), np.sin(zeta)
    cZ, sZ = np.cos(z), np.sin(z)
    ct, st = np.cos(theta), np.sin(theta)
    P = np.empty(T.shape + (3, 3))
    P[..., 0, 0] = cz * ct * cZ - sz * sZ
    P[..., 0, 1] = -sz * ct * cZ - cz * sZ
    P[..., 0, 2] = -st * cZ
    P[..., 1, 0] = cz * ct * sZ + sz * cZ
    P[..., 1, 1] = -sz * ct * sZ + cz * cZ
    P[..., 1, 2] = -st * sZ
    P[..., 2, 0] = cz * st
    P[..., 2, 1] = -sz * st
    P[..., 2, 2] = ct
    return P


def _rotx(v, ang):
    """Rotate vectors [..., 3] about the x axis by ang (rad)."""
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = c * v[..., 1] - s * v[..., 2]
    out[..., 2] = s * v[..., 1] + c * v[..., 2]
    return out


def earth_heliocentric_eq_j2000(jd):
    """Earth heliocentric position [N, 3] in the equatorial J2000 frame."""
    jd = np.asarray(jd, dtype=float)
    T = (jd - 2451545.0) / 36525.0
    lam, eps = sun_longitude_and_obliquity(jd)
    R = sun_radius_vector(jd)
    # Earth as seen from the Sun is opposite the Sun as seen from Earth.
    ecl_date = np.stack([-R * np.cos(lam), -R * np.sin(lam), np.zeros_like(R)], -1)
    eq_date = _rotx(ecl_date, eps)
    P = precession_matrix(T)                       # r_date = P r_2000
    return np.einsum("nji,nj->ni", P, eq_date)     # r_2000 = P^T r_date


def solve_kepler(M, e, tol=1e-13, max_iter=60):
    M = np.asarray(M, dtype=float)
    Mw = np.mod(M + np.pi, 2 * np.pi) - np.pi
    E = Mw + e * np.sin(Mw)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - Mw
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def elements_to_eq_j2000(jd, a, e, inc, node, argp, M0, jd_ref, n=None):
    """Heliocentric position [N, 3] in equatorial J2000 from elements referred
    to the mean ecliptic and equinox of J2000. Mean motion follows Kepler's
    third law unless n (rad/day) is supplied explicitly."""
    jd = np.asarray(jd, dtype=float)
    if n is None:
        n = np.sqrt(GM_SUN / a**3)
    M = M0 + n * (jd - jd_ref)
    E = solve_kepler(M, e)
    xv = a * (np.cos(E) - e)
    yv = a * np.sqrt(np.maximum(1.0 - e**2, 1e-12)) * np.sin(E)
    cw, sw = np.cos(argp), np.sin(argp)
    ci, si = np.cos(inc), np.sin(inc)
    cO, sO = np.cos(node), np.sin(node)
    # R3(node) R1(inc) R3(argp) applied to (xv, yv, 0)
    x1 = cw * xv - sw * yv
    y1 = sw * xv + cw * yv
    x2, y2, z2 = x1, ci * y1, si * y1
    xe = cO * x2 - sO * y2
    ye = sO * x2 + cO * y2
    ze = z2
    ecl = np.stack([xe, ye, ze], -1)
    return _rotx(ecl, EPS_J2000)


def predicted_declination(jd, a, e, inc, node, argp, M0, jd_ref, n=None,
                          light_time=True):
    """Geocentric Mars declination (rad), referred to the mean equator of date."""
    jd = np.asarray(jd, dtype=float)
    earth = earth_heliocentric_eq_j2000(jd)
    mars = elements_to_eq_j2000(jd, a, e, inc, node, argp, M0, jd_ref, n)
    geo = mars - earth
    if light_time:
        for _ in range(2):
            delta = np.linalg.norm(geo, axis=-1)
            mars = elements_to_eq_j2000(jd - delta / C_AU_PER_DAY, a, e, inc,
                                        node, argp, M0, jd_ref, n)
            geo = mars - earth
    T = (jd - 2451545.0) / 36525.0
    P = precession_matrix(T)
    geo_date = np.einsum("nij,nj->ni", P, geo)
    r = np.linalg.norm(geo_date, axis=-1)
    return np.arcsin(geo_date[..., 2] / r)


# JPL "approximate positions of the major planets" mean elements for Mars,
# J2000 ecliptic, value at J2000 and rate per Julian century.
MARS_JPL = dict(a=(1.52371034, 0.00001847), e=(0.09339410, 0.00007882),
                inc=(1.84969142, -0.00813131), L=(-4.55343205, 19140.30268499),
                varpi=(-23.94362959, 0.44441088), node=(49.55953891, -0.29257343))


def mars_modern_elements(jd_ref):
    """Mars elements at jd_ref from the JPL secular series, in the form this
    module wants. Used only as a sanity reference, never fitted."""
    T = (jd_ref - 2451545.0) / 36525.0
    v = {k: c0 + c1 * T for k, (c0, c1) in MARS_JPL.items()}
    a = v["a"]
    e = v["e"]
    inc = v["inc"] * DEG
    node = v["node"] * DEG
    argp = (v["varpi"] - v["node"]) * DEG
    M0 = np.mod(v["L"] - v["varpi"] + 180.0, 360.0) * DEG - np.pi
    return dict(a=a, e=e, inc=inc, node=node, argp=argp, M0=M0)


def transit_altitude_deg(dec_rad):
    """Meridian transit altitude at Hven for a body of declination dec (rad)."""
    return 90.0 - HVEN_LAT_DEG + np.degrees(np.asarray(dec_rad, dtype=float))
