#!/usr/bin/env python3
"""Chronological-cutoff sensitivity of the historical Mars pilot (Table 2).

A reviewer asked whether the historical ranking persists across campaigns and
reasonable earlier training cutoffs. This script refits the six Table 2 rows
that this directory produces at the training cutoffs 1592, 1593, 1594 and 1595
(epochs with jd < 1 January of the split year train, the rest are held out),
under the same Huber objective at 5 arcmin and the same selection procedures
as run_pilot.py:

  oracle      modern mean elements at jd_ref (median training JD), never fitted
  kepler6     six elements, Huber least squares, 60 seeded starts plus the
              modern start (seed 7)
  epicycles   saturated K = 25, frequencies ordered by greedy forward selection
              under the Huber cost on the training window, Huber IRLS fit
  GP          quasi-periodic kernel, 8 seeded starts (seed 20260829), IRLS on
              the nugget driven by closed-form leave-one-out residuals
  ridge       full 145-coefficient lattice, penalty chosen on the last fifth
              of the training window
  constant    Huber location of the training declinations

Every fitter and every statistic is imported from run_pilot.py, models.py and
run_robust.py; nothing is copied. The AI Feynman row of Table 2 comes from the
separate pipeline in ../aifeynman/ and is not refitted here.

It also renders the per-campaign held-out results at the default split from
the committed results_summary.json, and checks that the 1594 refit reproduces
the printed Table 2 numbers.

Outputs: cutoff_sensitivity.json and cutoff_sensitivity.md beside this file.
results.json and results_summary.json are read only, never written.

Usage:
  python run_cutoff_sensitivity.py              fit all cutoffs, write both files
  python run_cutoff_sensitivity.py --jobs 1     same, one cutoff at a time
  python run_cutoff_sensitivity.py --render-only
                                                rebuild the md from the json
"""

import argparse
import datetime
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
try:
    import keplerbench  # noqa: F401  (not needed by the pilot modules)
except ImportError:
    sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import scipy

import ephem as E
import models
import pilot_data as data
import run_pilot as RP
import run_robust as RB

OUT_JSON = os.path.join(HERE, "cutoff_sensitivity.json")
OUT_MD = os.path.join(HERE, "cutoff_sensitivity.md")
SUMMARY = os.path.join(HERE, "results_summary.json")

SPLIT_YEARS = [1592, 1593, 1594, 1595]
DEFAULT_SPLIT = 1594
F_SCALE = RP.F_SCALE
K_SATURATED = 25

# (key, Table 2 label); the keys match results.json robust_huber.models
MODELS = [
    ("modern_unfitted", "Oracle (true orbit, no fitting)"),
    ("kepler6", "Ellipse fitted to Tycho (Keplerian)"),
    ("epicycle_K25_saturated", "Ptolemaic epicycles (25 terms)"),
    ("gp_quasiperiodic", "Flexible curve fit (Gaussian process)"),
    ("frequency_lattice_ridge_control", "Sine-wave fit (ridge)"),
    ("control_mean", "Constant (training mean)"),
]
LABEL = dict(MODELS)
FLEXIBLE = ["epicycle_K25_saturated", "gp_quasiperiodic",
            "frequency_lattice_ridge_control"]

# Table 2 of the paper at split 1594, (in sample, held out) in arcmin.
TABLE2 = {
    "modern_unfitted": (3.95, 3.48),
    "kepler6": (2.66, 13.31),
    "epicycle_K25_saturated": (4.02, 731.19),
    "gp_quasiperiodic": (0.72, 1173.95),
    "frequency_lattice_ridge_control": (4.80, 1266.51),
    "control_mean": (714.09, 966.14),
}
# Same tolerance as the Table 2 contract test: half a unit in the last printed
# decimal plus 2e-5 relative, which covers the documented Gaussian-process
# wobble from floating-point nondeterminism in the optimizers.
TOL_ABS = 0.005
TOL_REL = 2e-5

CAMPAIGNS = ["1594", "1595", "1596", "1600"]


# ------------------------------------------------------------------ fitting
def _summ(res_arcmin):
    """run_pilot.stats, trimmed to the fields this report uses."""
    s = RP.stats(res_arcmin)
    keep = ("n", "medae", "medae_ci95", "abs_residuals_sorted")
    return {k: s[k] for k in keep if k in s}


def fit_split(split_year):
    t0 = time.time()
    D = data.build(split_year=split_year)
    jd, dec, split_jd = D["jd"], D["dec"], D["split_jd"]
    tr, te = jd < split_jd, jd >= split_jd
    jd_tr, dec_tr, jd_te, dec_te = jd[tr], dec[tr], jd[te], dec[te]
    jd_ref = float(np.median(jd_tr))
    t_tr, t_te = jd_tr - jd_ref, jd_te - jd_ref
    years_te = np.array([RP._julian_year(x) for x in jd_te])
    uy, cy = np.unique(years_te, return_counts=True)

    def fit_oracle():
        el = E.mars_modern_elements(jd_ref)
        pm = E.predicted_declination(jd, el["a"], el["e"], el["inc"],
                                     el["node"], el["argp"], el["M0"], jd_ref)
        rm = (dec - pm) / E.ARCMIN
        return rm[tr], rm[te], {}

    def fit_kepler6():
        fit = models.fit_kepler(jd_tr, dec_tr, jd_ref, n_restarts=60,
                                seed=RP.SEED_KEPLER_HUBER, free_n=False,
                                loss="huber", f_scale=F_SCALE)
        ri = (dec_tr - models.predict_kepler(jd_tr, fit)) / E.ARCMIN
        ro = (dec_te - models.predict_kepler(jd_te, fit)) / E.ARCMIN
        rs = models.restart_summary(fit)
        return ri, ro, dict(elements=RP.wrapped_elements(fit["theta"]),
                            restarts_at_global=f"{rs['n_at_global']}/{rs['n']}")

    def fit_epicycles():
        lattice = models.frequency_lattice()
        m = (K_SATURATED - 1) // 2
        order = models.greedy_order_robust(t_tr, dec_tr, lattice, m,
                                           f_scale=F_SCALE)
        freqs = [f for f, _, _, _ in order[:m]]
        coef, cond = models.fit_harmonic_robust(t_tr, dec_tr, freqs,
                                                f_scale=F_SCALE)
        ri = (dec_tr - models.predict_harmonic(t_tr, freqs, coef)) / E.ARCMIN
        ro = (dec_te - models.predict_harmonic(t_te, freqs, coef)) / E.ARCMIN
        return ri, ro, dict(K=K_SATURATED, cond_design=cond,
                            periods_days=[float(1.0 / f) for f in freqs])

    def fit_gp():
        gp = models.fit_gp_robust(t_tr, dec_tr, f_scale_arcmin=F_SCALE,
                                  n_starts=RP.GP_N_STARTS, seed=RP.SEED_GP)
        ri = (dec_tr - models.predict_gp(t_tr, gp)) / E.ARCMIN
        ro = (dec_te - models.predict_gp(t_te, gp)) / E.ARCMIN
        return ri, ro, dict(nll=float(gp["nll"]),
                            effective_dof=models.gp_effective_dof(gp),
                            n_downweighted=int(gp["n_downweighted"]))

    def fit_ridge():
        ridge = RP.fit_lattice_ridge(t_tr, dec_tr, models.frequency_lattice())
        ri = dec_tr / E.ARCMIN - models.design(t_tr, ridge["freqs"]) @ ridge["coef"]
        ro = dec_te / E.ARCMIN - models.design(t_te, ridge["freqs"]) @ ridge["coef"]
        return ri, ro, dict(lambda_=ridge["lambda_"],
                            n_coefficients=ridge["n_coefficients"],
                            training_validation_medae=ridge["validation_medae"])

    def fit_constant():
        mu = RB.robust_mean(dec_tr, F_SCALE)
        return ((dec_tr - mu) / E.ARCMIN, (dec_te - mu) / E.ARCMIN,
                dict(location_deg=float(np.degrees(mu))))

    fitters = dict(modern_unfitted=fit_oracle, kepler6=fit_kepler6,
                   epicycle_K25_saturated=fit_epicycles,
                   gp_quasiperiodic=fit_gp,
                   frequency_lattice_ridge_control=fit_ridge,
                   control_mean=fit_constant)

    out_models = {}
    for key, _label in MODELS:
        t1 = time.time()
        try:
            ri, ro, extra = fitters[key]()
            by_campaign = {str(int(y)): _summ(ro[years_te == y])
                           for y in np.unique(years_te)}
            out_models[key] = dict(status="ok", in_sample=_summ(ri),
                                   held_out=_summ(ro),
                                   held_out_by_campaign=by_campaign,
                                   fit_seconds=round(time.time() - t1, 1),
                                   **extra)
        except Exception as exc:  # record the failure, never a number
            out_models[key] = dict(status="failed",
                                   error=f"{type(exc).__name__}: {exc}",
                                   traceback=traceback.format_exc(),
                                   fit_seconds=round(time.time() - t1, 1))
        print(f"[{split_year}] {key}: {out_models[key]['status']} "
              f"({out_models[key]['fit_seconds']} s)", flush=True)

    return dict(split_year=split_year, split_jd=split_jd, jd_ref=jd_ref,
                n_train=int(tr.sum()), n_held_out=int(te.sum()),
                held_out_campaigns={str(int(a)): int(b) for a, b in zip(uy, cy)},
                models=out_models, wall_seconds=round(time.time() - t0, 1))


# ------------------------------------------------------------ verification
def reproduction_check(split_result):
    lines, rows = [], {}
    for key, label in MODELS:
        exp_in, exp_out = TABLE2[key]
        m = split_result["models"][key]
        if m["status"] != "ok":
            rows[key] = dict(verdict="FAIL", reason="fit failed")
            lines.append(f"FAIL {label}: fit failed ({m['error']})")
            continue
        got_in, got_out = m["in_sample"]["medae"], m["held_out"]["medae"]
        ok_in = abs(got_in - exp_in) <= TOL_ABS + TOL_REL * abs(exp_in)
        ok_out = abs(got_out - exp_out) <= TOL_ABS + TOL_REL * abs(exp_out)
        verdict = "PASS" if ok_in and ok_out else "FAIL"
        rows[key] = dict(verdict=verdict, expected=[exp_in, exp_out],
                         got=[got_in, got_out])
        lines.append(f"{verdict} {label}: in sample {got_in:.2f} (Table 2 "
                     f"{exp_in:.2f}), held out {got_out:.2f} (Table 2 "
                     f"{exp_out:.2f})")
    return rows, lines


# ---------------------------------------------------------------- rendering
def _cell(s):
    if s is None:
        return "failed"
    if s.get("medae") is not None:
        lo, hi = s["medae_ci95"]
        return f"{s['medae']:.2f} [{lo:.2f}, {hi:.2f}]"
    if s.get("n", 0) == 0:
        return "no epochs"
    return ", ".join(f"{x:.2f}" for x in s["abs_residuals_sorted"])


def _held(split_result, key):
    m = split_result["models"][key]
    return m["held_out"]["medae"] if m["status"] == "ok" else None


def ordering_checks(splits):
    out = {}
    for s in splits:
        y = str(s["split_year"])
        h = {k: _held(s, k) for k, _ in MODELS}
        if any(v is None for v in h.values()):
            out[y] = dict(complete=False)
            continue
        m = s["models"]
        flex_min_key = min(FLEXIBLE, key=lambda k: h[k])
        kep_hi = m["kepler6"]["held_out"]["medae_ci95"][1]
        flex_lo = min(m[k]["held_out"]["medae_ci95"][0] for k in FLEXIBLE)
        others = [k for k, _ in MODELS if k != "modern_unfitted"]
        out[y] = dict(
            complete=True,
            held_out_rank=sorted(h, key=h.get),
            oracle_best_held_out=all(h["modern_unfitted"] < h[k] for k in others),
            ellipse_beats_every_flexible_fit=all(
                h["kepler6"] < h[k] for k in FLEXIBLE),
            ellipse_beats_constant=h["kepler6"] < h["control_mean"],
            best_flexible=flex_min_key,
            best_flexible_over_ellipse_ratio=h[flex_min_key] / h["kepler6"],
            intervals_separate=bool(kep_hi < flex_lo),
            gp_best_in_sample=all(
                m["gp_quasiperiodic"]["in_sample"]["medae"]
                < m[k]["in_sample"]["medae"]
                for k, _ in MODELS if k != "gp_quasiperiodic"))
    return out


def persistence_statement(splits, checks):
    done = [s for s in splits if checks[str(s["split_year"])]["complete"]]
    years = [s["split_year"] for s in done]
    failed = [s["split_year"] for s in splits
              if not checks[str(s["split_year"])]["complete"]]
    held = {s["split_year"]: s for s in splits}
    sent = []
    if not done:
        return "No cutoff completed every fit, so persistence cannot be assessed."
    all_oracle = all(checks[str(y)]["oracle_best_held_out"] for y in years)
    all_ellipse = all(checks[str(y)]["ellipse_beats_every_flexible_fit"]
                      for y in years)
    ratios = [checks[str(y)]["best_flexible_over_ellipse_ratio"] for y in years]
    sep = all(checks[str(y)]["intervals_separate"] for y in years)
    kep = [held[y]["models"]["kepler6"]["held_out"]["medae"] for y in years]
    orc = [held[y]["models"]["modern_unfitted"]["held_out"]["medae"]
           for y in years]
    yr = f"{years[0]} to {years[-1]}" if len(years) > 1 else str(years[0])
    if all_oracle and all_ellipse:
        sent.append(
            f"The ordering persists at every cutoff from {yr}: the unfitted "
            f"oracle has the lowest held-out median ({min(orc):.2f} to "
            f"{max(orc):.2f} arcmin), and the fitted ellipse "
            f"({min(kep):.2f} to {max(kep):.2f} arcmin) beats every flexible "
            f"fit held out by a factor of at least {min(ratios):.0f}.")
    else:
        bad_o = [y for y in years if not checks[str(y)]["oracle_best_held_out"]]
        bad_e = [y for y in years
                 if not checks[str(y)]["ellipse_beats_every_flexible_fit"]]
        parts = []
        if bad_o:
            parts.append("the oracle is not the best held-out row at "
                         + ", ".join(map(str, bad_o)))
        if bad_e:
            parts.append("the ellipse does not beat every flexible fit held "
                         "out at " + ", ".join(map(str, bad_e)))
        sent.append("The ordering does not persist at every cutoff: "
                    + "; ".join(parts) + ".")
    gp_in = all(checks[str(y)]["gp_best_in_sample"] for y in years)
    near = []
    for y in years:
        ins = {k: held[y]["models"][k]["in_sample"]["medae"] for k, _ in MODELS}
        g = ins.pop("gp_quasiperiodic")
        k2 = min(ins, key=ins.get)
        if ins[k2] - g < 0.05:
            near.append(f"{g:.2f} against {ins[k2]:.2f} for the "
                        f"{'ellipse' if k2 == 'kepler6' else LABEL[k2]} at {y}")
    near_s = f" (only marginally: {'; '.join(near)})" if near else ""
    if sep and gp_in:
        sent.append("At every cutoff the ellipse's held-out interval lies "
                    "entirely below every flexible fit's interval, while the "
                    f"Gaussian process remains the best in-sample fit{near_s}, "
                    "so the in-sample and held-out rankings disagree "
                    "throughout.")
    elif sep:
        sent.append("At every cutoff the ellipse's held-out interval lies "
                    "entirely below every flexible fit's interval.")
    elif gp_in:
        sent.append("The Gaussian process remains the best in-sample fit at "
                    "every cutoff, so the in-sample and held-out rankings "
                    "disagree throughout.")
    y_lo = years[int(np.argmin(kep))]
    y_hi = years[int(np.argmax(kep))]
    flex_orders = {tuple(k for k in checks[str(y)]["held_out_rank"]
                         if k in FLEXIBLE) for y in years}
    tail = (", and the order among the flexible fits changes from cutoff to "
            "cutoff, so the two headline relations are robust but the full "
            "ranking is not") if len(flex_orders) > 1 else ""
    sent.append(f"The ellipse's own held-out median is not stable across "
                f"cutoffs, lowest at {y_lo} and highest at {y_hi}{tail}.")
    if failed:
        sent.append("The cutoffs " + ", ".join(map(str, failed))
                    + " have at least one failed fit and are not assessed.")
    n94 = held[DEFAULT_SPLIT]["held_out_campaigns"].get("1594", 0) \
        if DEFAULT_SPLIT in held else None
    n00 = sorted({s["held_out_campaigns"].get("1600", 0) for s in splits})
    n00_s = " or ".join(map(str, n00))
    caveat = ("The caveat is that the four held-out windows are nested rather "
              "than independent (each later cutoff's window is a subset of "
              "the earlier ones, and all of them contain the 1595, 1596 and "
              "1600 epochs)")
    if n94 is not None:
        caveat += (f", the 1594 campaign has only {n94} epochs, so the 1594 "
                   f"and 1595 cutoffs differ by those {n94} training epochs "
                   "alone")
    caveat += (f", and the 1600 tail has {n00_s} epochs at every cutoff, "
               f"below the {RP.MIN_N_FOR_MEDIAN}-epoch minimum for a quoted "
               "median.")
    sent.append(caveat)
    return " ".join(sent)


def render(result):
    splits = result["splits"]
    L = []
    L.append("# Chronological-cutoff sensitivity of the historical Mars pilot")
    L.append("")
    L.append("Generated by `run_cutoff_sensitivity.py`; the numbers are in "
             "`cutoff_sensitivity.json`. Median absolute residual in arcmin "
             "with percentile-bootstrap 95 percent intervals over epochs "
             "(10,000 resamples, seed 20260829). Every row is fitted under the "
             "Huber loss at 5 arcmin with the selection procedures of "
             "run_pilot.py, on the epochs before 1 January of the split year, "
             "and scored on the epochs from that date on. The AI Feynman row of "
             "Table 2 is produced by the separate pipeline in `../aifeynman/` "
             "and is not refitted here.")
    L.append("")
    L.append("## Table A. Fit and held-out error by training cutoff")
    L.append("")
    L.append("| Split year | n train | n held out | Model | In sample (arcmin) "
             "| Held out (arcmin) |")
    L.append("|---:|---:|---:|---|---:|---:|")
    for s in splits:
        for key, label in MODELS:
            m = s["models"][key]
            if m["status"] == "ok":
                ci, co = _cell(m["in_sample"]), _cell(m["held_out"])
            else:
                ci = co = f"failed ({m['error'].splitlines()[0]})"
            L.append(f"| {s['split_year']} | {s['n_train']} | "
                     f"{s['n_held_out']} | {label} | {ci} | {co} |")
    L.append("")
    L.append("Held-out epochs by campaign at each cutoff:")
    L.append("")
    camp_years = sorted({c for s in splits for c in s["held_out_campaigns"]})
    L.append("| Split year | " + " | ".join(camp_years) + " |")
    L.append("|---:|" + "---:|" * len(camp_years))
    for s in splits:
        L.append(f"| {s['split_year']} | " + " | ".join(
            str(s["held_out_campaigns"].get(c, 0)) for c in camp_years) + " |")
    L.append("")
    L.append("## Reproduction of Table 2 at the default split (1594)")
    L.append("")
    L.append(f"Tolerance: {TOL_ABS} arcmin plus {TOL_REL:g} relative, as in "
             "the Table 2 contract test.")
    L.append("")
    for line in result["reproduction_check"]["lines"]:
        L.append(f"- {line}")
    L.append("")
    L.append("## Table B. Held-out error by campaign at the default split (1594)")
    L.append("")
    L.append("From the committed `results_summary.json` "
             "(`headline.by_campaign`). A median with its interval is quoted "
             "on at least 10 epochs; below that the sorted absolute residuals "
             "are listed. The ridge control is not in that block of the "
             "committed file, so it has no row here.")
    L.append("")
    camp = result["by_campaign_default_split"]
    hdr = [f"{c} (n = {camp[c]['n']})" for c in CAMPAIGNS if c in camp]
    L.append("| Model | " + " | ".join(hdr) + " |")
    L.append("|---|" + "---:|" * len(hdr))
    for key, label in MODELS:
        if not all(key in camp[c]["models"] for c in CAMPAIGNS if c in camp):
            continue
        L.append(f"| {label} | " + " | ".join(
            _cell(camp[c]["models"][key]) for c in CAMPAIGNS if c in camp)
            + " |")
    L.append("")
    L.append("## Does the ordering persist?")
    L.append("")
    L.append(result["persistence_statement"])
    L.append("")
    return "\n".join(L)


def campaign_block_from_summary():
    with open(SUMMARY) as f:
        summ = json.load(f)
    bc = summ["headline"]["by_campaign"]
    out = {}
    for c in CAMPAIGNS:
        if c not in bc:
            continue
        out[c] = dict(n=bc[c]["n"], models={
            k: {f: v[f] for f in ("n", "medae", "medae_ci95",
                                  "abs_residuals_sorted") if f in v}
            for k, v in bc[c]["models"].items()})
    return out


def finalize(splits, provenance):
    splits = sorted(splits, key=lambda s: s["split_year"])
    default = next(s for s in splits if s["split_year"] == DEFAULT_SPLIT)
    rows, lines = reproduction_check(default)
    for line in lines:
        print(line)
    checks = ordering_checks(splits)
    result = dict(
        description=("Chronological-cutoff sensitivity of the historical Mars "
                     "pilot: the six Table 2 rows produced in this directory, "
                     "refitted at four training cutoffs under the Huber loss "
                     "at 5 arcmin. AI Feynman is produced by the external "
                     "pipeline in ../aifeynman/ and is skipped."),
        statistic="median absolute residual in arcmin",
        bootstrap=dict(resamples=RP.BOOTSTRAP_N, seed=RP.BOOTSTRAP_SEED,
                       kind="percentile, over epochs"),
        min_n_for_median=RP.MIN_N_FOR_MEDIAN,
        f_scale_arcmin=F_SCALE,
        split_years=SPLIT_YEARS,
        splits=splits,
        reproduction_check=dict(split_year=DEFAULT_SPLIT, tol_abs=TOL_ABS,
                                tol_rel=TOL_REL, table2=TABLE2, rows=rows,
                                lines=lines),
        ordering_checks=checks,
        by_campaign_default_split=campaign_block_from_summary(),
        by_campaign_source="results_summary.json headline.by_campaign",
        provenance=provenance)
    result["persistence_statement"] = persistence_statement(splits, checks)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=len(SPLIT_YEARS))
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()

    if args.render_only:
        with open(OUT_JSON) as f:
            old = json.load(f)
        result = finalize(old["splits"], old["provenance"])
    else:
        t0 = time.time()
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                splits = list(ex.map(fit_split, SPLIT_YEARS))
        else:
            splits = [fit_split(y) for y in SPLIT_YEARS]
        provenance = dict(
            python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, entry_point="run_cutoff_sensitivity.py",
            jobs=args.jobs, wall_seconds=round(time.time() - t0, 1),
            utc_timestamp=datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"))
        result = finalize(splits, provenance)

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, default=RP._jsonable)
    with open(OUT_MD, "w") as f:
        f.write(render(result))
    print(f"wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
