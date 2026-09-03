"""Every acceptance threshold of the pilot test suite, in one place.

Nothing in this file is derived from a model fit; the thresholds are set from
independent references (Meeus, JPL, Tycho's documented instrumental scatter)
or are deliberately loose bounds around measured values. The published Table 2
values that the results contract pins are in test_table2_contract.py.
"""

import pathlib

# ---------------------------------------------------------------- data source
# Pafko's spreadsheet is never committed; `keplerbench fetch-historical` caches
# it here, and the tests that read it skip when it is absent.
XLS = str(pathlib.Path.home() / ".cache" / "keplerbench" / "tycho_mars_pafko.xls")
CSV = str(pathlib.Path(__file__).resolve().parents[3] / "data" / "historical" / "mars_declinations.csv")
XLS_SHA256 = "0e1c95d3e61514a5d4120a56988801b13c31be32cb69ec51603b23fab6b2e845"
XLS_SHAPE = (929, 21)
N_RAW_ROWS = 923
PAFKO_JD_OFFSET = 1721048.0

# columns (0-indexed)
COL_YEAR, COL_MONTH, COL_DAY = 8, 9, 10
COL_DAY_ADJ = 11
COL_HOUR, COL_MIN = 12, 13
COL_DAYS_SINCE_1AD = 14
COL_DECIMAL_DATE = 15
COL_SIGN, COL_DEG, COL_ARCMIN, COL_ARCSEC = 16, 17, 18, 19
COL_DEC = 20

# ------------------------------------------------------------- I1, I2, I3
TIME_IDENTITY_TOL_DAY = 1e-6          # A1
DEC_IDENTITY_TOL_DEG = 1e-9           # A2
UPSTREAM_SHIFT_RANGE_DAY = (9.4, 10.3)  # measured 9.50 to 10.25

N_EPOCHS = 354                        # A3
N_DROPPED_SPREAD = 4
DEC_RANGE_DEG = (-27.975, 27.300)

# ------------------------------------------------------------------- split
SPLIT_YEAR = 1594
SPLIT_JD = 2303266.5                  # F1
N_TRAIN_NO_DELETION = 268
N_TEST_NO_DELETION = 86
JD_REF = 2301527.770835              # median of the 268 training epochs
TRAIN_END = 2303264.75

P_SID = 686.9800
P_SYN = 779.9361
HORIZON_BUCKETS_SYNODIC = [(0.0, P_SYN), (P_SYN, 2 * P_SYN), (2 * P_SYN, 3 * P_SYN)]
MIN_N_FOR_MEDIAN = 10                 # reporting rule

# --------------------------------------------------------- frames and units
HVEN_LAT_DEG = 55.9070
HVEN_LON_DEG = 12.6962
EARTH_RADIUS_AU = 4.2635e-5
MAX_PARALLAX_ARCMIN = 0.45            # B9, measured max 0.386 at 0.379 AU
EPS_J2000_DEG = 23.4392911
OBLIQUITY_AGREEMENT_ARCSEC = 1.0      # B2
ORTHONORMAL_TOL = 1e-12               # B3
KEPLER_SOLVER_TOL = 1e-12             # B7
TWO_BODY_INVARIANT_TOL = 1e-10        # B8
SOLAR_LONGITUDE_EQUINOX_TOL_DEG = 0.02  # B6

# ------------------------------------------------ C: ephemeris validation
EPHEM_MEDAE_ALL_MAX = 5.0             # C1, arcmin, measured 3.69
EPHEM_FRAC_WITHIN_10_MIN = 0.85       # C2, measured 0.881
EPHEM_MEDAE_CLEAN_MAX = 4.0           # C3, measured 3.32 / 3.44
GROSS_ARCMIN = 60.0                   # definition of a gross error
SHIFT_10D_MEDAE_MIN = 30.0            # C4, measured 51.5 / 53.7
SHIFT_1D_MEDAE_MIN = 4.5              # C5, measured 4.79
LIGHT_TIME_EFFECT_ARCMIN = (0.05, 5.0)  # C6

# ------------------------------------------------- D: synthetic recovery
TYCHO_SIGMA_ARCMIN = 2.0
CONTAMINATION_N = 14                  # of 268, near the real record's 15 of 268
NOISELESS_ELEMENT_TOL = 1e-6          # D1
NOISELESS_ANGLE_TOL_RAD = 1e-5        # D2
NOISELESS_HOLDOUT_ARCMIN = 0.01       # D3
NOISY_A_REL_TOL = 1e-3                # D4, D7
NOISY_HOLDOUT_ARCMIN = 3.0            # D5, D8
OLS_CONTAMINATED_HOLDOUT_MIN = 50.0   # D6, the artifact; measured 311
THIRD_LAW_TOL = 1e-4                  # D9

# ------------------------------------------------------ E: real-data results
F_SCALE_ARCMIN = 5.0
KEPLER_ROBUST_IN_MEDAE_MAX = 4.0      # E1, measured 2.71
KEPLER_ROBUST_OUT_MEDAE_MAX = 20.0    # E2, measured 12.62
KEPLER_ROBUST_H1_MEDAE_MAX = 15.0     # E3
KEPLER_ROBUST_OUT_MAXABS_MAX = 90.0   # E4, measured 49.89
KEPLER_A_REL_TOL = 3e-3               # E5
KEPLER_E_TOL = 5e-3                   # E6
KEPLER_INC_TOL_DEG = 0.05             # E7
KEPLER_VARPI_TOL_DEG = 2.0            # E8
GP_OUT_MEDAE_MIN = 300.0              # E9
GP_IN_MEDAE_MAX = 2.0                 # E10
EPI25_OUT_MEDAE_MIN = 100.0           # E11
EPI25_IN_MEDAE_MAX = 5.0              # E12
CONTROL_OUT_MEDAE_MIN = 800.0         # E13
RESTART_SPREAD_WRAPPED_MAX_RAD = 1e-3  # E16

# --------------------------------------------------- outlier detector quality
DETECTOR_MIN_PRECISION = 0.50
DETECTOR_MIN_RECALL = 0.50

# ------------------------------------------------------------- refraction
ALT_BINS_DEG = [(0, 15), (15, 25), (25, 40), (40, 70)]
REFRACTION_SPEARMAN_P_MAX = 0.01      # measured 2.5e-4

# --------------------------------------------------------------- provenance
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_N = 10000
DETERMINISM_RTOL = 1e-10

REQUIRED_RESULTS_BLOCKS = [
    "data", "models", "epicycle_sweep", "epicycle_cv",
    "robust_huber", "refraction_analysis", "provenance", "headline",
]
REQUIRED_MODEL_KEYS = [
    "modern_unfitted", "control_mean", "epicycle_cv_selected",
    "epicycle_K25_saturated", "kepler6", "kepler7_free_n", "gp_quasiperiodic",
]
REQUIRED_STAT_FIELDS = [
    "n", "medae", "mad_sigma", "p90", "rms", "max_abs", "n_gross",
    "median_signed", "medae_ci95",
]
REQUIRED_FIGURES = ["fig_kepler_test.png", "fig_kepler_test.pdf",
                    "fig_refraction.png"]
