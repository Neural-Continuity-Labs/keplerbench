#!/usr/bin/env python3
"""Print Table 2 of the paper from the pilot results and the AI Feynman scoring.

usage: make_table.py [--results results.json] [--aifeynman ../aifeynman/runs/phases_20s/scored_phases.json] [--write]

Prints the seven rows of Table 2 in Markdown and LaTeX, the campaign table, the
horizon table, and the facts quoted in the paper's text. With --write it also
writes TABLE2.md and results_summary.json, the results file without the three
per-epoch arrays (the refraction scatter, the held-out residuals by epoch and
the injected residual vectors), which could reconstruct declination values
derived from Pafko's transcription.
"""
import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = [
    ("Oracle (true orbit, no fitting)", "reference", "modern_unfitted"),
    ("Ellipse fitted to Tycho (Keplerian)", "physical law", "kepler6"),
    ("Ptolemaic epicycles (25 terms)", "flexible fit", "epicycle_K25_saturated"),
    ("Flexible curve fit (Gaussian process)", "flexible fit", "gp_quasiperiodic"),
    ("Sine-wave fit (ridge)", "control", "frequency_lattice_ridge_control"),
    ("Symbolic regression (AI Feynman)", "control", None),
    ("Constant (training mean)", "control", "control_mean"),
]
STRIP = [("refraction_analysis", "points"), ("headline", "held_out_by_epoch")]


def cell(value, ci):
    return f"{value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def table_rows(results, aif):
    models = results["robust_huber"]["models"]
    rows = []
    for label, kind, key in ROWS:
        if key is None:
            selected = aif["selected"] if aif else None
            if selected is None:
                rows.append((label, kind, "--", "not scored", "not scored"))
                continue
            rows.append((label, kind, "--", cell(selected["train"], selected["train_ci95"]), cell(selected["heldout"], selected["heldout_ci95"])))
            continue
        m = models[key]
        dof = m["effective_dof"]
        dof_s = f"{dof:.0f}" if abs(dof - round(dof)) < 0.5 else f"{dof:.1f}"
        rows.append((label, kind, dof_s, cell(m["in_sample"]["medae"], m["in_sample"]["medae_ci95"]), cell(m["out_of_sample"]["medae"], m["out_of_sample"]["medae_ci95"])))
    return rows


def markdown(results, aif):
    lines = ["| Model | Type | Eff. dof | In sample (arcmin) | Held out (arcmin) |", "|---|---|---:|---:|---:|"]
    lines += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in table_rows(results, aif)]
    return "\n".join(lines)


def latex(results, aif):
    return "\n".join(f"{a} & {b} & {c} & {d} & {e} \\\\" for a, b, c, d, e in table_rows(results, aif))


def campaign_table(results):
    camp = results["headline"]["by_campaign"]
    lines = ["| Campaign | n | Keplerian | Oracle (modern elements) |", "|---|---:|---:|---:|"]
    for year in sorted(camp):
        block = camp[year]
        def fmt(stats):
            if stats.get("medae") is not None:
                return cell(stats["medae"], stats["medae_ci95"])
            return ", ".join(f"{x:.2f}" for x in stats["abs_residuals_sorted"])
        lines.append(f"| {year} | {block['n']} | {fmt(block['models']['kepler6'])} | {fmt(block['models']['modern_unfitted'])} |")
    return "\n".join(lines)


def horizon_table(results):
    lines = ["| Horizon (days after training) | Keplerian | Gaussian process |", "|---|---:|---:|"]
    models = results["robust_huber"]["models"]
    for bucket in ("0-780d", "780-1560d", "1560-2340d"):
        cells = []
        for key in ("kepler6", "gp_quasiperiodic"):
            stats = models[key]["out_of_sample_by_horizon"][bucket]
            cells.append(cell(stats["medae"], stats["medae_ci95"]) if stats.get("medae") is not None else ", ".join(f"{x:.2f}" for x in stats["abs_residuals_sorted"]))
        lines.append(f"| {bucket} (n = {stats['n']}) | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def facts(results):
    d, ra = results["data"], results["refraction_analysis"]
    k6 = results["robust_huber"]["models"]["kepler6"]
    gp = results["robust_huber"]["models"]["gp_quasiperiodic"]
    cv = results["robust_huber"]["epicycle_cv"]
    kb = results["knowledge_budget"]["ablation"]["huber"]
    return [
        f"epochs {d['n_epochs']} (train {d['n_train']}, held out {d['n_test']}); held-out campaigns {d['test_year_composition']}",
        f"gross errors above 60 arcmin: {d['gross_error_census']['n_gross']} ({d['gross_error_census']['n_gross_train']} in training, {d['gross_error_census']['n_sign_flips']} sign flips)",
        f"transit altitude {ra['altitude_deg_range'][0]:.1f} to {ra['altitude_deg_range'][1]:.1f} deg; Spearman rho of residual against altitude {ra['spearman_rho']:.3f} (p = {ra['spearman_p']:.1e})",
        f"signed median residual shift between windows {ra['observed_bias_shift_arcmin']:.2f} arcmin; altitude-corrected Keplerian refit held out {ra['altitude_corrected_refit']['held_out_medae_arcmin']:.2f} arcmin",
        f"Keplerian degeneracy: {k6['degeneracy']['strongest_pair']} correlate at {k6['degeneracy']['strongest_correlation']:.3f}",
        f"Keplerian restarts at the global optimum {k6['restarts']['n_at_global']}/{k6['restarts']['n']}; blind restarts (no modern start) reach held out {kb['blind_held_out_medae']:.2f} arcmin",
        f"Gaussian process effective dof {gp['effective_dof']:.1f}, {gp['n_starts']} starts, negative log likelihood {gp['nll']:.2f}",
        f"epicycle K selected inside training: forward chaining {cv['K_selected']}, blocked folds {cv['K_selected_blocked']}; the table reports the saturated K = 25",
        f"runtime {results['runtime_seconds']} s on one CPU; python {results['provenance']['python']}, numpy {results['provenance']['numpy']}, scipy {results['provenance']['scipy']}",
    ]


def summary_without_per_epoch_arrays(results):
    out = copy.deepcopy(results)
    for block, key in STRIP:
        out.get(block, {}).pop(key, None)
    for condition in out.get("synthetic_observation_physics", {}).get("conditions", {}).values():
        condition.pop("injected_residual_arcmin", None)
    if "data" in out and "source" in out["data"]:
        out["data"]["source"] = "data/historical/mars_declinations.csv"
    # The extreme declinations and the altitudes derived from them are observed values; round them.
    for block, key in (("data", "dec_deg_range"), ("data", "transit_altitude_deg_range"), ("refraction_analysis", "altitude_deg_range")):
        if key in out.get(block, {}):
            out[block][key] = [round(float(x), 1) for x in out[block][key]]
    out["summary_note"] = ("results.json without its per-epoch arrays (refraction scatter, held-out residuals by epoch, "
                           "injected residual vectors), which are omitted because they could reconstruct declination values "
                           "derived from Pafko's transcription; rerun run_pilot.py for the full file")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=HERE / "results.json")
    ap.add_argument("--aifeynman", type=Path, default=HERE.parent / "aifeynman" / "runs" / "phases_20s" / "scored_phases.json")
    ap.add_argument("--write", action="store_true", help="write TABLE2.md and results_summary.json beside this script")
    args = ap.parse_args()
    results = json.loads(args.results.read_text())
    aif = json.loads(args.aifeynman.read_text()) if args.aifeynman.exists() else None

    md = ["# Table 2 of the paper, regenerated", "",
          "Median absolute residual on Tycho's Mars declinations in arcmin, with epoch-bootstrap 95 percent intervals.",
          "The six fitted and reference rows come from `results.json` (10,000 resamples, seed 20260829); the AI Feynman row",
          "comes from `../aifeynman/runs/phases_20s/scored_phases.json` (2,000 resamples, seed 23).", "",
          markdown(results, aif), "", "## Held-out campaigns", "", campaign_table(results), "",
          "## Held-out horizon (Keplerian against the Gaussian process)", "", horizon_table(results), "",
          "## Facts quoted in the paper", ""] + [f"- {f}" for f in facts(results)] + [""]
    text = "\n".join(md)
    print(text)
    print("LaTeX rows:\n" + latex(results, aif))
    if args.write:
        (HERE / "TABLE2.md").write_text(text)
        (HERE / "results_summary.json").write_text(json.dumps(summary_without_per_epoch_arrays(results), indent=2) + "\n")
        print("wrote TABLE2.md and results_summary.json")


if __name__ == "__main__":
    main()
