"""The contract results.json and the figure scripts must satisfy.
"""

import os
import re

import numpy as np
import pytest

import constants as S
from conftest import wrap


# ------------------------------------------------------------- structure
def test_results_json_parses_and_has_the_shipped_blocks(results):
    for k in ("data", "models", "epicycle_sweep", "epicycle_cv", "robust_huber"):
        assert k in results, k


def test_results_json_has_every_required_block(results):
    missing = [k for k in S.REQUIRED_RESULTS_BLOCKS if k not in results]
    assert not missing, f"missing blocks: {missing}"


def test_model_table_is_complete(results):
    missing = [k for k in S.REQUIRED_MODEL_KEYS if k not in results["models"]]
    assert not missing, (
        f"missing model entries: {missing}; modern_unfitted in particular is the "
        "zero-parameter ceiling the fitted models are judged against")


def test_no_duplicate_model_entries(results):
    """No two model entries may carry the same numbers under different
    family labels."""
    seen = {}
    for k, v in results["models"].items():
        if "in_sample" not in v:
            continue
        sig = (round(v["in_sample"].get("rms", -1), 9),
               round(v["out_of_sample"].get("rms", -1), 9))
        if sig in seen:
            fam = (v.get("family"), results["models"][seen[sig]].get("family"))
            pytest.fail(f"'{k}' duplicates '{seen[sig]}' with families {fam}")
        seen[sig] = k


def test_every_stat_block_carries_the_required_fields(results):
    bad = []
    for name, m in results["models"].items():
        for split_key in ("in_sample", "out_of_sample"):
            blk = m.get(split_key)
            if not isinstance(blk, dict):
                continue
            for f in S.REQUIRED_STAT_FIELDS:
                if f not in blk:
                    bad.append(f"{name}.{split_key}.{f}")
    assert not bad, f"{len(bad)} missing statistic fields, first: {bad[:6]}"


HORIZON_KEYS = ("by_horizon", "out_of_sample_by_horizon",
                "out_of_sample_by_horizon_fine")


def _median_field(blk):
    for f in ("medae", "medae_arcmin"):
        if f in blk:
            return blk[f]
    return None


def test_no_median_is_quoted_on_a_bucket_below_the_minimum_n(results):
    offenders = []
    for name, m in results["models"].items():
        for key in HORIZON_KEYS:
            for bucket, blk in (m.get(key) or {}).items():
                if blk.get("n", 0) < S.MIN_N_FOR_MEDIAN and \
                        _median_field(blk) is not None:
                    offenders.append(f"{name}.{key}.{bucket} n={blk['n']}")
    assert not offenders, (
        f"{len(offenders)} medians quoted on fewer than {S.MIN_N_FOR_MEDIAN} "
        f"observations, first: {offenders[:5]}")


def test_horizon_buckets_are_the_synodic_ones(results):
    keys = set()
    for key in HORIZON_KEYS:
        keys |= set(results["models"]["kepler6"].get(key) or {})
    assert any("780" in k for k in keys), (
        f"horizon keys are {sorted(keys)}; the reporting rule requires synodic-period "
        "buckets keyed to 779.9361 days, not calendar years. The shipped bins "
        "put n = 5 in the first bucket and n = 7 in the last.")


# ------------------------------------------------------- headline numbers
def test_E9_gp_collapses_out_of_sample(results):
    gp = results["models"]["gp_quasiperiodic"]
    assert gp["out_of_sample"]["medae"] >= S.GP_OUT_MEDAE_MIN
    assert gp["in_sample"]["medae"] <= S.GP_IN_MEDAE_MAX


def test_E10_gp_fits_below_the_instrument_noise(results):
    """The point of the paper: the interpolator fits the training window better
    than Tycho could measure it."""
    assert results["models"]["gp_quasiperiodic"]["in_sample"]["medae"] < \
        S.TYCHO_SIGMA_ARCMIN


def test_E11_E12_saturated_epicycle(results):
    e = results["models"]["epicycle_K25_saturated"]
    assert e["out_of_sample"]["medae"] >= S.EPI25_OUT_MEDAE_MIN
    rob = results["robust_huber"]["epicycle_sweep"]["25"]
    assert rob["in_sample"]["medae"] <= S.EPI25_IN_MEDAE_MAX


def test_E13_control_is_the_floor(results):
    c = results["models"]["control_mean"]
    assert c["out_of_sample"]["medae"] >= S.CONTROL_OUT_MEDAE_MIN
    assert c["n_params"] == 1


def test_E1_to_E4_robust_kepler_holds_out(results):
    k = results["robust_huber"]["models"]["kepler6"]
    assert k["in_sample"]["medae"] <= S.KEPLER_ROBUST_IN_MEDAE_MAX
    assert k["out_of_sample"]["medae"] <= S.KEPLER_ROBUST_OUT_MEDAE_MAX
    assert k["out_of_sample"]["max_abs"] <= S.KEPLER_ROBUST_OUT_MAXABS_MAX
    near = k["out_of_sample_by_horizon"]["0-780d"]
    assert near["n"] >= S.MIN_N_FOR_MEDIAN, near["n"]
    assert _median_field(near) <= S.KEPLER_ROBUST_H1_MEDAE_MAX


def test_E5_to_E8_robust_kepler_recovers_the_orbit(results, ephem, split):
    k = results["robust_huber"]["models"]["kepler6"]["elements"]
    el = ephem.mars_modern_elements(split["jd_ref"])
    assert abs(k["a_AU"] / el["a"] - 1.0) < S.KEPLER_A_REL_TOL
    assert abs(k["e"] - el["e"]) < S.KEPLER_E_TOL
    assert abs(k["inc_deg"] - np.degrees(el["inc"])) < S.KEPLER_INC_TOL_DEG
    varpi = np.radians(k["node_deg"] + k["argp_deg"])
    d = abs(np.degrees(wrap(varpi - el["node"] - el["argp"])))
    assert d < S.KEPLER_VARPI_TOL_DEG, f"longitude of perihelion off by {d:.2f} deg"


def test_the_ols_kepler_entry_is_labelled_an_artifact(results):
    """The L2 numbers may stay in the file as an ablation, but they must not
    sit there unlabelled where a reader will lift them as the physical model's
    performance."""
    k = results["models"]["kepler6"]
    assert k["out_of_sample"]["medae"] > 40.0, (
        "if this changed, re-derive the diagnosis")
    assert "artifact_note" in k or "loss" in k, (
        "results.models.kepler6 holds the L2 fit with no marker that it is the "
        "outlier-dominated ablation")


def test_headline_block_names_the_robust_loss(results):
    h = results.get("headline")
    assert h is not None, "results.json has no headline block"
    assert h.get("loss") == "huber", h


# ------------------------------------------------------ degeneracy evidence
def test_E16_restart_spread_is_reported_with_wrapped_angles(results):
    """The parameter bounds are [-8, 8] rad, so equivalent solutions differ by
    whole revolutions and raw angular spreads would be wrapping artifacts."""
    for tag in ("kepler6", "kepler7_free_n"):
        spread = results["models"][tag]["restarts"]["param_spread_within_global"]
        for i, name in enumerate(["a", "e", "inc", "node", "argp", "M0"]):
            if name in ("node", "argp", "M0"):
                assert spread[i] < S.RESTART_SPREAD_WRAPPED_MAX_RAD, (
                    f"{tag}.{name} spread {spread[i]:.4f} rad "
                    f"({spread[i] / np.pi:.2f} pi) is an unwrapped-angle "
                    "artifact, not degeneracy evidence")


def test_angle_bounds_exceed_one_revolution(models):
    """Why the wrapping artifact exists. Not a bug on its own, but it makes any
    raw angular spread meaningless."""
    for i in (3, 4, 5):
        assert models.KEP_HI[i] - models.KEP_LO[i] > 2 * np.pi


def test_restarts_do_converge_to_one_basin(results):
    """The genuine degeneracy evidence: within the global basin the fit RMS is
    identical to 1e-6 arcmin, so the reported optimum is a real optimum."""
    for tag in ("kepler6", "kepler7_free_n"):
        r = results["models"][tag]["restarts"]
        assert r["n_at_global"] >= 5, r
        assert r["spread_within_global_arcmin"] < 1e-3, r


def test_parameter_correlations_are_reported_and_high(results):
    """The correlation matrix is the identifiability evidence, and the paper
    names the strongest pair."""
    c = np.array(results["models"]["kepler6"]["param_corr"])
    assert c.shape == (6, 6)
    off = np.abs(c - np.eye(6))
    assert off.max() > 0.9, "expected a near-degenerate pair"
    i, j = np.unravel_index(np.argmax(off), off.shape)
    assert {i, j} == {0, 5}, (
        f"strongest correlation is between parameters {i} and {j}; the paper "
        "must name the pair it reports")


# ------------------------------------------------------- capacity accounting
def test_E21_gp_effective_degrees_of_freedom_are_reported(results):
    gp = results["models"]["gp_quasiperiodic"]
    assert "effective_dof" in gp, (
        "the GP has seven hyperparameters; its capacity is the effective degrees "
        "of freedom, trace(K (K + sigma^2 I)^-1), which the table reports")


def test_gp_effective_dof_is_large(models, split):
    """Compute it here so the reported number is checkable."""
    t_tr = split["jd_tr"] - split["jd_ref"]
    gp = models.fit_gp(t_tr, split["dec_tr"])
    K = models._gp_K(t_tr, t_tr, gp["p"])
    s2 = np.exp(2 * gp["p"][6]) + 1e-10
    dof = float(np.trace(K @ np.linalg.inv(K + s2 * np.eye(len(t_tr)))))
    assert dof > 50.0, f"effective dof {dof:.1f}"
    assert dof < len(t_tr)


def test_parameter_counts_match_the_spec(results):
    expect = {"control_mean": 1, "epicycle_K25_saturated": 25, "kepler6": 6,
              "kepler7_free_n": 7}
    for k, n in expect.items():
        assert results["models"][k]["n_params"] == n, k
    assert results["models"]["epicycle_cv_selected"]["n_params"] == \
        results["epicycle_cv"]["K_selected"]


def test_cv_selectors_disagree_and_both_are_reported(results):
    """The two selectors disagree, and neither is presented alone."""
    cv = results["epicycle_cv"]
    assert cv["K_selected"] != cv["K_selected_blocked"]
    assert cv["K_selected"] in (1, 3, 5, 7, 9)
    assert results["models"]["epicycle_K25_saturated"].get("family") != \
        "interpolator" or True  # labelling checked below


def test_saturated_epicycle_is_labelled_saturated_not_the_baseline(results):
    fam = results["models"]["epicycle_K25_saturated"].get("family")
    assert fam in ("oracle", "saturated"), (
        f"K = 25 is filed as family '{fam}' although cross-validation inside the "
        "training window selects K = 3 or 9; presenting it as the interpolator "
        "baseline overstates the interpolator")


# ------------------------------------------------------- refraction block
def test_refraction_analysis_block_present(results):
    r = results.get("refraction_analysis")
    assert r is not None, "results.json has no refraction_analysis block"
    assert "spearman_rho" in r and "bins" in r


# ---------------------------------------------------------------- outputs
@pytest.mark.parametrize("fname", S.REQUIRED_FIGURES)
def test_required_figures_exist(pilot_dir, fname):
    """Figures are not committed; they are regenerated by make_figure.py."""
    if not any(os.path.exists(os.path.join(pilot_dir, f)) for f in S.REQUIRED_FIGURES):
        pytest.skip("figures not generated; run make_figure.py")
    assert os.path.exists(os.path.join(pilot_dir, fname)), fname


def _figure_scripts(pilot_dir):
    return sorted(f for f in os.listdir(pilot_dir)
                  if f.endswith(".py") and ("figure" in f or f.startswith("fig")))


def test_at_least_one_figure_script_is_shipped(pilot_dir):
    assert _figure_scripts(pilot_dir), "no figure script found"


def test_figure_plots_medians_not_rms(pilot_dir):
    """With 99 percent of the SSE in 5 percent of the rows, an RMS
    axis plots where the transcription errors landed. Every shipped figure
    script must use medians."""
    bad = []
    for fn in _figure_scripts(pilot_dir):
        src = open(os.path.join(pilot_dir, fn)).read()
        axis = re.findall(r'set_ylabel\("([^"]+)"', src)
        if "medae" not in src or any("RMS" in a for a in axis):
            bad.append(f"{fn} (y axes {axis})")
    assert not bad, f"figure scripts on an RMS axis: {bad}"


def test_figure_uses_the_robust_results(pilot_dir):
    """A figure script that never reads the robust fit would plot the
    outlier-dominated L2 ablation."""
    bad = [fn for fn in _figure_scripts(pilot_dir)
           if "robust_huber" not in open(os.path.join(pilot_dir, fn)).read()]
    assert not bad, f"figure scripts that never read the robust fit: {bad}"


def test_no_em_or_en_dashes_in_generated_text(pilot_dir):
    """House rule: no em or en dashes in generated text."""
    bad = []
    for fn in os.listdir(pilot_dir):
        if not fn.endswith((".py", ".md", ".json")):
            continue
        if fn.startswith("."):
            continue
        txt = open(os.path.join(pilot_dir, fn), encoding="utf-8",
                   errors="ignore").read()
        for ch, name in (("—", "em dash"), ("–", "en dash")):
            if ch in txt:
                bad.append(f"{fn}: {name}")
    assert not bad, bad
