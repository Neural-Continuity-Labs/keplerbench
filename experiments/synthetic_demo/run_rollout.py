"""The end-to-end synthetic run of the local dynamical-law test.

The question a reviewer asked of the protocol: a state containing trajectory
coefficients and a clock can be evolved recurrently and reproduce a
precomputed trajectory, so what in the evaluation separates it from a local
dynamical law? This run measures the answer on held-out worlds, comparing

- the dynamical submission, whose 8-number state is the Cartesian state of
  both bodies and whose transition rule is one leapfrog integrator of the
  frozen central force, with
- the trajectory-plus-clock control, whose 35-number state stores the
  coefficients of a curve fitted to the planet's apparent angle over the
  initialization window next to a clock that its transition rule increments.

Both satisfy the interface: fixed-dimensional state, one reusable transition
rule, no history after initialization. The separation comes from the
evaluation: long-horizon rollout error, the time to first exceed an error
threshold, and the window-consistency check, under which initializing the same
instant from different windows must lead to the same predicted future.

Writes rollout_results.json, ROLLOUT.md and fig_rollout.pdf next to this
script. Run from the repository root: python experiments/synthetic_demo/run_rollout.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from keplerbench.rollout import evaluate_rollout, window_consistency, _window_and_future, _rollout_errors  # noqa: E402
from keplerbench.synthetic import generate_observations, sample_world  # noqa: E402

from learners import DynamicalSubmission, TrajectoryClockSubmission  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = [300, 301, 302]
WINDOW_END = 20.0
HORIZON_EDGES = [5.0, 10.0, 20.0, 40.0, 60.0]
THRESHOLD_RADIANS = np.deg2rad(1.0)  # 60 arcmin
CONSISTENCY_WINDOWS = [(0.0, WINDOW_END), (10.0, WINDOW_END)]

ARCMIN = 180.0 / np.pi * 60.0


def _world(seed: int):
    # Overrides, all recorded in the configuration: a long window so rollouts
    # span several target periods, matching sampling density, about one arcmin
    # of angular noise, and high generator integration accuracy.
    config = sample_world(
        "inverse-square", seed=seed, duration=80.0, sample_count=1600,
        train_fraction=0.5, angular_noise_std=3.0e-4, integration_substeps=120,
    )
    return config, generate_observations(config, seed=seed)


def main() -> None:
    started = time.time()
    submissions = {
        "dynamical": DynamicalSubmission(p=2.0),
        "trajectory_clock": TrajectoryClockSubmission(order=16),
    }
    results: dict[str, object] = {"worlds": [], "window_end": WINDOW_END,
                                  "threshold_arcminutes": THRESHOLD_RADIANS * ARCMIN}
    curves: dict[str, dict[str, object]] = {name: {} for name in submissions}
    for seed in SEEDS:
        config, rows = _world(seed)
        period = 2.0 * np.pi * np.sqrt(config.target_radius ** 3 / config.gravitational_parameter)
        world_entry: dict[str, object] = {
            "world_id": config.world_id,
            "target_period": period,
            "noise_floor_arcmin": config.angular_noise_std * ARCMIN,
            "submissions": {},
        }
        for name, submission in submissions.items():
            print(f"{config.world_id}: {name}", flush=True)
            report = evaluate_rollout(submission, rows, WINDOW_END, HORIZON_EDGES, THRESHOLD_RADIANS)
            consistency = window_consistency(submission, rows, CONSISTENCY_WINDOWS, future_count=80)
            report["window_consistency"] = consistency
            world_entry["submissions"][name] = report
            window, future = _window_and_future(rows, WINDOW_END)
            times, errors = _rollout_errors(submission, window, future)
            curves[name][config.world_id] = {
                "horizons": (times - WINDOW_END).tolist(),
                "errors_arcmin": (errors * ARCMIN).tolist(),
            }
        results["worlds"].append(world_entry)
    results["runtime_seconds"] = round(time.time() - started, 1)
    (HERE / "rollout_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_markdown(results)
    try:
        _write_figure(curves, results)
    except ImportError:
        print("matplotlib missing; figure skipped", flush=True)
    print(f"done in {results['runtime_seconds']} s", flush=True)


def _write_markdown(results: dict[str, object]) -> None:
    lines = ["# Local dynamical law, end to end", ""]
    lines.append(f"Initialization window up to t = {results['window_end']:g}; failure threshold "
                 f"{results['threshold_arcminutes']:.0f} arcmin; divergence from two initialization "
                 f"windows ending at the same instant.")
    lines.append("")
    lines.append("| World | Submission | State dim | Median error by horizon bucket (arcmin) | Failure horizon | Window divergence, median / max (arcmin) |")
    lines.append("|---|---|---|---|---|---|")
    for world in results["worlds"]:
        for name, report in world["submissions"].items():
            buckets = ", ".join(f"{key}: {value['median_arcminutes']:.1f}" for key, value in report["buckets"].items())
            failure = f"{report['failure_horizon']:.1f}" if report["failure_horizon"] is not None else "never"
            consistency = report["window_consistency"]
            lines.append(
                f"| {world['world_id']} (T = {world['target_period']:.1f}) | {name} | {report['state_dimension']} "
                f"| {buckets} | {failure} | {consistency['median_divergence_arcminutes']:.1f} / "
                f"{consistency['max_divergence_arcminutes']:.1f} |")
    (HERE / "ROLLOUT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(curves: dict[str, dict[str, object]], results: dict[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.4, 3.1))
    colors = {"dynamical": "tab:green", "trajectory_clock": "tab:red"}
    labels = {"dynamical": "dynamical state (8 numbers)", "trajectory_clock": "trajectory + clock (35 numbers)"}
    styles = ["-", "--", ":"]
    for name, by_world in curves.items():
        for style, (world_id, curve) in zip(styles, sorted(by_world.items())):
            axis.plot(curve["horizons"], np.maximum(curve["errors_arcmin"], 1e-2), style, color=colors[name],
                      alpha=0.85, label=labels[name] if style == "-" else None)
    axis.axhline(results["threshold_arcminutes"], color="gray", lw=0.8, ls="-.")
    axis.text(0.4, results["threshold_arcminutes"] * 1.3, "failure threshold", fontsize=7, color="gray")
    axis.set_xlabel("rollout horizon past the window (time units)")
    axis.set_ylabel("angular error (arcmin)")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(HERE / "fig_rollout.pdf", bbox_inches="tight")
    figure.savefig(HERE / "fig_rollout.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
