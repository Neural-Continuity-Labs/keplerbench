"""Figures for the Kepler-test pilot, regenerated from results.json alone.

Writes fig_kepler_test.png, fig_kepler_test.pdf, fig_refraction.png and
fig_horizon.png at 200 dpi in a muted palette. No result number is hard coded here: everything is read
out of results.json, including the bucket edges and the counts. The only
constants in this file are Tycho's documented repeat scatter, which sets the
lower edge of the shaded band, and the plotting style.

The y axis is the MEDIAN absolute residual, never the root mean square. On
this record 99 percent of the sum of squares comes from 5 percent of the rows,
so an RMS axis is a picture of where the transcription errors landed rather
than of how well a model describes the sky.

Usage: python3 make_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.json")
DPI = 200
TYCHO_SCATTER_ARCMIN = 2.0     # documented repeat scatter of the instrument

MUTED = dict(kepler6="#4c72b0", gp_quasiperiodic="#c44e52",
             epicycle_K25_saturated="#dd8452",
             epicycle_cv_selected="#937860", control_mean="#8c8c8c",
             modern_unfitted="#55a868")
LABEL = dict(kepler6="Keplerian, 6 parameters",
             gp_quasiperiodic="Gaussian process",
             epicycle_K25_saturated="epicycle, K = 25, saturated",
             epicycle_cv_selected="epicycle, K chosen by CV",
             control_mean="mean control",
             modern_unfitted="modern elements, nothing fitted")
SERIES = ["control_mean", "gp_quasiperiodic", "epicycle_cv_selected",
          "epicycle_K25_saturated", "kepler6"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})


def load():
    with open(RESULTS) as f:
        return json.load(f)


def bucket_label(key):
    """Turn a results.json bucket key such as 0-780d into an axis label."""
    lo, hi = key[:-1].split("-")
    return f"{lo} to {hi} d"


def panel_generalization(ax, R):
    rob = R["robust_huber"]["models"]
    buckets = list(rob["kepler6"]["out_of_sample_by_horizon"].keys())
    xs = np.arange(1 + len(buckets))
    labels = ["in sample"] + [bucket_label(b) for b in buckets]
    counts = [rob["kepler6"]["in_sample"]["n"]] + [
        rob["kepler6"]["out_of_sample_by_horizon"][b]["n"] for b in buckets]

    offsets = np.linspace(-0.22, 0.22, len(SERIES))
    for k, off in zip(SERIES, offsets):
        m = rob[k]
        blocks = [m["in_sample"]] + [m["out_of_sample_by_horizon"][b]
                                     for b in buckets]
        px, py, elo, ehi = [], [], [], []
        for x, blk in zip(xs, blocks):
            if blk.get("medae") is None:
                pts = blk.get("abs_residuals_sorted") or []
                if pts:
                    ax.plot(np.full(len(pts), x + off), pts, marker=".",
                            linestyle="none", markersize=2.5,
                            color=MUTED[k], alpha=0.65)
                continue
            px.append(x + off)
            py.append(blk["medae"])
            lo, hi = blk["medae_ci95"]
            elo.append(max(blk["medae"] - lo, 0.0))
            ehi.append(max(hi - blk["medae"], 0.0))
        ax.errorbar(px, py, yerr=[elo, ehi], marker="o", markersize=3.5,
                    linewidth=1.2, capsize=2, color=MUTED[k], label=LABEL[k])

    floor = rob["modern_unfitted"]["out_of_sample"]["medae"]
    ax.axhline(floor, linestyle="--", linewidth=1.0,
               color=MUTED["modern_unfitted"], label=LABEL["modern_unfitted"])
    band = ax.axhspan(TYCHO_SCATTER_ARCMIN, floor, color="#55a868", alpha=0.10,
                      lw=0, label="Tycho repeat scatter to ephemeris floor")

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, xs[-1] + 0.95)
    ax.set_ylabel("Median absolute residual (arcmin)")
    ax.set_title("A. Fit in sample, prediction out of sample, matched Huber loss")
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y * 0.55, hi_y * 30)
    for x, n in zip(xs, counts):
        note = f"n = {n}" if n >= 10 else f"n = {n}, below 10, points shown"
        ax.annotate(note, (x, 0.015), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6, color="#555555")
    ax.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.6,
              columnspacing=1.2)


def panel_capacity(ax, R):
    rob = R["robust_huber"]
    sweep = rob["epicycle_sweep"]
    ks = sorted(int(k) for k in sweep)
    dof = [sweep[str(k)]["effective_dof"] for k in ks]
    ins = [sweep[str(k)]["in_sample"]["medae"] for k in ks]
    out = [sweep[str(k)]["out_of_sample"]["medae"] for k in ks]
    c = MUTED["epicycle_K25_saturated"]
    ax.plot(dof, ins, "-o", color=c, markersize=3, linewidth=1.2,
            label="epicycle, in sample")
    ax.plot(dof, out, "--o", color=c, markersize=3, linewidth=1.2,
            markerfacecolor="white", label="epicycle, held out")

    for key, marker in (("kepler6", "s"), ("gp_quasiperiodic", "^"),
                        ("control_mean", "D")):
        m = rob["models"][key]
        ax.plot([m["effective_dof"]], [m["in_sample"]["medae"]], marker=marker,
                color=MUTED[key], markersize=6, linestyle="none",
                label=LABEL[key] + ", in sample")
        ax.plot([m["effective_dof"]], [m["out_of_sample"]["medae"]],
                marker=marker, color=MUTED[key], markersize=6,
                markerfacecolor="white", linestyle="none",
                label=LABEL[key] + ", held out")
        ax.plot([m["effective_dof"], m["effective_dof"]],
                [m["in_sample"]["medae"], m["out_of_sample"]["medae"]],
                color=MUTED[key], linewidth=0.8, alpha=0.5)

    floor = rob["models"]["modern_unfitted"]["out_of_sample"]["medae"]
    ax.axhline(floor, linestyle="--", linewidth=1.0,
               color=MUTED["modern_unfitted"], label=LABEL["modern_unfitted"])

    cv = rob["epicycle_cv"]
    k_star, k_blk = cv["K_selected"], cv["K_selected_blocked"]
    ax.axvline(k_star, color="#8172b3", linewidth=1.0, alpha=0.8)
    ax.annotate(f"CV selects K = {k_star} by forward chaining,\n"
                f"K = {k_blk} by blocked folds",
                (k_star, 0.97), xycoords=("data", "axes fraction"),
                fontsize=6.5, color="#6a5a97", ha="left", va="top",
                xytext=(4, 0), textcoords="offset points")

    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("log")
    ax.set_xlabel("Effective degrees of freedom")
    ax.set_ylabel("Median absolute residual (arcmin)")
    ax.set_title("B. Capacity against generalization, solid in sample, "
                 "dashed held out")
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y * 0.35, hi_y * 2.5)
    ax.legend(loc="lower left", frameon=False, fontsize=6, ncol=2,
              handletextpad=0.5, columnspacing=1.0)


def figure_main(R):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    panel_generalization(axes[0], R)
    panel_capacity(axes[1], R)
    fig.suptitle("The Kepler test on Tycho Brahe's Mars declinations, "
                 "trained before 1594 and tested on 1594 to 1600", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_kepler_test.{ext}"), dpi=DPI)
    plt.close(fig)


def figure_refraction(R):
    ref = R["refraction_analysis"]
    P = ref["points"]
    alt = np.array(P["transit_altitude_deg"])
    res = np.array(P["residual_vs_modern_arcmin"])
    trn = np.array(P["in_train"], bool)
    gross = np.array(P["is_gross"], bool)
    good = ~gross

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.plot(alt[good & trn], res[good & trn], "o", markersize=3.2,
            color="#4c72b0", alpha=0.7, label="training epochs, before 1594")
    ax.plot(alt[good & ~trn], res[good & ~trn], "^", markersize=3.6,
            color="#c44e52", alpha=0.8, label="held out epochs, 1594 to 1600")

    edges = ref["alt_bins_deg"]
    data, positions, widths = [], [], []
    for (lo, hi), key in zip(edges, ref["bins"]):
        m = good & (alt >= lo) & (alt < hi)
        if m.sum() == 0:
            continue
        data.append(res[m])
        positions.append(0.5 * (lo + hi))
        widths.append(0.55 * (hi - lo))
    bp = ax.boxplot(data, positions=positions, widths=widths, manage_ticks=False,
                    showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#d9d2c5", alpha=0.45, edgecolor="#7a7367",
                linewidth=0.8)
    for part in ("whiskers", "caps", "medians"):
        for ln in bp[part]:
            ln.set(color="#7a7367", linewidth=0.9)

    for (lo, hi), (key, blk) in zip(edges, ref["bins"].items()):
        ax.annotate(f"n = {blk['n']}\nmedian {blk['median_signed_arcmin']:+.2f}",
                    (0.5 * (lo + hi), 0.03), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6, color="#555555")

    ax.set_xlim(0, 70)
    ax.set_ylim(-25, 35)
    ax.set_xlabel("Meridian transit altitude at Hven (degrees)")
    ax.set_ylabel("Residual against unfitted modern elements (arcmin)")
    ax.set_title("Altitude dependent bias, Spearman rho = "
                 f"{ref['spearman_rho']:.3f}, p = {ref['spearman_p']:.1e}\n"
                 f"median bias {ref['train_median_signed_arcmin']:+.2f} arcmin "
                 f"in training, {ref['test_median_signed_arcmin']:+.2f} arcmin "
                 "held out", fontsize=9)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_refraction.png"), dpi=DPI)
    plt.close(fig)


def figure_horizon(R):
    """Per-epoch growth of held-out residuals with extrapolation horizon."""
    rows = R["headline"]["held_out_by_epoch"]
    horizon = np.array([x["horizon_days"] for x in rows], float)
    kep = np.array([x["kepler6_abs_residual_arcmin"] for x in rows], float)
    gp = np.array([x["gp_quasiperiodic_abs_residual_arcmin"] for x in rows], float)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(horizon, gp, "o-", color=MUTED["gp_quasiperiodic"],
            markersize=3.0, linewidth=1.0, alpha=0.85,
            label="quasi-periodic Gaussian process")
    ax.plot(horizon, kep, "s-", color=MUTED["kepler6"],
            markersize=2.8, linewidth=1.0, alpha=0.85,
            label="Keplerian, 6 parameters")
    crossing = R["headline"]["gp_horizon_crossing_days"]
    if crossing is not None:
        ax.axvline(crossing, color="#777777", linestyle=":", linewidth=0.9)
        ax.annotate(f"first crossing at {crossing:g} days",
                    (crossing, 0.04), xycoords=("data", "axes fraction"),
                    xytext=(5, 0), textcoords="offset points", fontsize=7,
                    rotation=90, va="bottom", color="#555555")
    ax.set_yscale("log")
    ax.set_xlabel("Held-out horizon from the last training epoch (days)")
    ax.set_ylabel("Absolute residual (arcmin)")
    ax.set_title("Gaussian process error grows with extrapolation horizon")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_horizon.png"), dpi=DPI)
    plt.close(fig)


def main():
    R = load()
    figure_main(R)
    figure_refraction(R)
    figure_horizon(R)
    print("[figures] fig_kepler_test.png, fig_kepler_test.pdf, "
          "fig_refraction.png and fig_horizon.png written at "
          f"{DPI} dpi from results.json")


if __name__ == "__main__":
    main()
