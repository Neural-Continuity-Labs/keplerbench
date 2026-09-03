"""Table 2 of the paper is a test: every printed number must be reproduced.

Values are the ones printed in the submitted paper (two decimals). The results
fixture is results.json after a run, or the committed results_summary.json.
"""

import json
import os

import pytest

TABLE2 = {
    # key in results.robust_huber.models: (dof, in-sample, in CI, held-out, out CI)
    "modern_unfitted": (0, 3.95, (3.17, 4.53), 3.48, (2.74, 3.75)),
    "kepler6": (6, 2.66, (2.33, 3.07), 13.31, (12.12, 17.40)),
    "epicycle_K25_saturated": (25, 4.02, (3.36, 4.79), 731.19, (382.42, 924.58)),
    "gp_quasiperiodic": (122, 0.72, (0.59, 0.89), 1173.95, (1150.28, 1185.88)),
    "frequency_lattice_ridge_control": (145, 4.80, (3.97, 5.52), 1266.51, (1223.75, 1279.94)),
    "control_mean": (1, 714.09, (621.93, 761.32), 966.14, (959.85, 990.74)),
}
AIFEYNMAN_ROW = (580.87, (532.25, 601.79), 2694.14, (2690.12, 2698.09))
TOL_ABS = 0.0051   # half a unit in the last printed decimal, plus float slack
TOL_REL = 2e-5     # the Gaussian process extrapolation moves at this level between runs on
                   # inputs that differ in the last bit; see the README on reproduction precision


def _close(a, b):
    return abs(a - b) < max(TOL_ABS, TOL_REL * abs(b))


@pytest.mark.parametrize("key", sorted(TABLE2))
def test_table2_row(results, key):
    dof, in_med, in_ci, out_med, out_ci = TABLE2[key]
    m = results["robust_huber"]["models"][key]
    assert round(m["effective_dof"]) == dof, (key, m["effective_dof"])
    assert _close(m["in_sample"]["medae"], in_med), (key, m["in_sample"]["medae"])
    assert _close(m["out_of_sample"]["medae"], out_med), (key, m["out_of_sample"]["medae"])
    assert all(_close(a, b) for a, b in zip(m["in_sample"]["medae_ci95"], in_ci)), (key, m["in_sample"]["medae_ci95"])
    assert all(_close(a, b) for a, b in zip(m["out_of_sample"]["medae_ci95"], out_ci)), (key, m["out_of_sample"]["medae_ci95"])


def test_aifeynman_row(pilot_dir):
    path = os.path.join(os.path.dirname(pilot_dir), "aifeynman", "runs", "phases_20s", "scored_phases.json")
    if not os.path.exists(path):
        pytest.skip("AI Feynman scoring not present")
    selected = json.load(open(path))["selected"]
    in_med, in_ci, out_med, out_ci = AIFEYNMAN_ROW
    assert _close(selected["train"], in_med) and _close(selected["heldout"], out_med)
    assert all(_close(a, b) for a, b in zip(selected["train_ci95"], in_ci))
    assert all(_close(a, b) for a, b in zip(selected["heldout_ci95"], out_ci))


def test_facts_quoted_in_the_text(results):
    d = results["data"]
    assert d["n_epochs"] == 354 and d["n_train"] == 268 and d["n_test"] == 86
    assert {int(k): v for k, v in d["test_year_composition"].items()} == {1594: 5, 1595: 54, 1596: 20, 1600: 7}
    assert d["gross_error_census"]["n_gross"] == 18
    ra = results["refraction_analysis"]
    assert round(ra["spearman_rho"], 2) == -0.19
    k6 = results["robust_huber"]["models"]["kepler6"]
    assert set(k6["degeneracy"]["strongest_pair"]) == {"a", "M0"} and round(k6["degeneracy"]["strongest_correlation"], 2) == 0.98
    assert results["runtime_seconds"] > 0
