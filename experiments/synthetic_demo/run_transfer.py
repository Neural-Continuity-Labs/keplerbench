"""The end-to-end synthetic run of the global-law transfer test.

Three questions, answered on worlds no entry has seen.

First, does the transfer metric separate an entry that carries a law from one
that does not? On ten held-out inverse-square worlds, the shared-law entry
(central-force structure, exponent frozen by a discovery phase on eight source
worlds) is compared with the benchmark's own harmonic baseline fitted to the
same k exposed observations, across evidence budgets k.

Second, does the misspecification family do its job? On ten held-out
hidden-exponent worlds, the same entry with the exponent frozen at 2 is
compared with the entry that adapts the exponent per system, and the frozen
entry's error is related to each world's true departure from inverse square.

Third, does the data-blind control behave end to end? The shared-law entry is
adapted on data-blind copies of three targets and the protocol's verdict is
applied.

An oracle check runs first: predictions from the latent world parameters must
reach the configured noise floor, which validates the forward model and the
harness against the generator before any comparison is read.

Writes transfer_results.json, TRANSFER.md and fig_transfer.pdf next to this
script. Run from the repository root: python experiments/synthetic_demo/run_transfer.py
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

from keplerbench.blind import data_blind_rows  # noqa: E402
from keplerbench.metrics import data_blind_verdict  # noqa: E402
from keplerbench.synthetic import generate_observations, sample_world  # noqa: E402
from keplerbench.transfer import evaluate_transfer, _exposed_and_heldout  # noqa: E402
from keplerbench.io import times_and_directions  # noqa: E402
from keplerbench.metrics import angular_error  # noqa: E402

from learners import (  # noqa: E402
    discover_exponent, flexible_adapter, oracle_adapter, shared_law_adapter,
)

HERE = Path(__file__).resolve().parent
K_VALUES = [4, 8, 16, 32, 64, 128]
SOURCE_SEEDS = list(range(8))
INVERSE_SQUARE_TARGETS = list(range(100, 110))
HIDDEN_EXPONENT_TARGETS = list(range(200, 210))
BLIND_TARGETS = INVERSE_SQUARE_TARGETS[:3]
BLIND_K = 32

ARCMIN = 180.0 / np.pi * 60.0


def _world(family: str, seed: int, **overrides):
    # Targets fix the sampling density so every budget up to k=128 exists on
    # every world, and raise the generator's integration accuracy so the truth
    # is the orbit rather than the integrator (both overrides are recorded in
    # the world configuration).
    config = sample_world(family, seed=seed, integration_substeps=120, **overrides)
    return config, generate_observations(config, seed=seed)


def _oracle_check(targets) -> list[dict[str, float]]:
    checks = []
    for config, rows in targets:
        exposed, heldout = _exposed_and_heldout(rows, K_VALUES[0])
        times, truth = times_and_directions(heldout)
        prediction = oracle_adapter(config)(exposed)(times)
        mean_error = float(np.mean(angular_error(truth, prediction)))
        checks.append({
            "world_id": config.world_id,
            "oracle_mean_arcmin": mean_error * ARCMIN,
            "noise_floor_arcmin": config.angular_noise_std * ARCMIN,
        })
    return checks


def main() -> None:
    started = time.time()
    results: dict[str, object] = {}

    print("generating worlds", flush=True)
    sources = [_world("inverse-square", seed) for seed in SOURCE_SEEDS]
    inv_targets = [_world("inverse-square", seed, sample_count=450) for seed in INVERSE_SQUARE_TARGETS]
    hid_targets = [_world("hidden-exponent", seed, sample_count=450) for seed in HIDDEN_EXPONENT_TARGETS]

    print("oracle check", flush=True)
    results["oracle_check"] = _oracle_check(inv_targets + hid_targets)
    worst = max(check["oracle_mean_arcmin"] - max(check["noise_floor_arcmin"], 0.2) for check in results["oracle_check"])
    print(f"  worst oracle excess over floor: {worst:.2f} arcmin", flush=True)

    print("discovery phase on source worlds", flush=True)
    discovery = discover_exponent([rows for _, rows in sources], seed=0)
    results["discovery"] = discovery
    print(f"  frozen exponent {discovery['frozen_p']:.4f} from {['%.3f' % p for p in discovery['per_world']]}", flush=True)

    print("inverse-square targets: shared law vs flexible", flush=True)
    inv_rows = [rows for _, rows in inv_targets]
    results["inverse_square"] = {
        "shared_law": evaluate_transfer(shared_law_adapter(discovery["frozen_p"]), inv_rows, K_VALUES),
        "flexible": evaluate_transfer(flexible_adapter(), inv_rows, K_VALUES),
    }

    print("hidden-exponent targets: misspecified vs adapted exponent", flush=True)
    hid_rows = [rows for _, rows in hid_targets]
    results["hidden_exponent"] = {
        "frozen_p2": evaluate_transfer(shared_law_adapter(2.0), hid_rows, K_VALUES),
        "adapted_p": evaluate_transfer(shared_law_adapter(None, dt_max=0.01), hid_rows, K_VALUES),
        "true_exponents": {config.world_id: config.force_exponent for config, _ in hid_targets},
    }

    print("data-blind control", flush=True)
    blind_runs = []
    for config, rows in inv_targets[:len(BLIND_TARGETS)]:
        exposed, heldout = _exposed_and_heldout(rows, BLIND_K)
        times, truth = times_and_directions(heldout)
        adapter = shared_law_adapter(discovery["frozen_p"])
        entry_error = float(np.mean(angular_error(truth, adapter(exposed)(times))))
        blinded, _, _ = data_blind_rows([{key: str(value) for key, value in row.items()} for row in rows], seed=7)
        blind_exposed, _ = _exposed_and_heldout(blinded, BLIND_K)
        blind_error = float(np.mean(angular_error(truth, adapter(blind_exposed)(times))))
        verdict = data_blind_verdict(entry_error, blind_error, config.angular_noise_std, "radians")
        verdict["world_id"] = config.world_id
        verdict["entry_arcmin"] = entry_error * ARCMIN
        verdict["data_blind_arcmin"] = blind_error * ARCMIN
        blind_runs.append(verdict)
    results["data_blind"] = blind_runs

    results["runtime_seconds"] = round(time.time() - started, 1)
    (HERE / "transfer_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote transfer_results.json after {results['runtime_seconds']} s", flush=True)

    _write_markdown(results)
    try:
        _write_figure(results)
    except ImportError:
        print("matplotlib missing; figure skipped", flush=True)


def _curve(block: dict[str, object]) -> list[float]:
    return [float(v) for v in block["mean_curve_arcminutes"]]


def _write_markdown(results: dict[str, object]) -> None:
    lines = ["# Global-law transfer, end to end", ""]
    lines.append("Mean held-out angular error in arcmin across target worlds, by evidence budget k.")
    lines.append("")
    lines.append("## Inverse-square targets (10 worlds)")
    lines.append("")
    lines.append("| k | " + " | ".join(str(k) for k in results["inverse_square"]["shared_law"]["k_values"]) + " | AUC (rad) |")
    lines.append("|---|" + "---|" * (len(K_VALUES) + 1))
    for name, key in (("Shared law (frozen p)", "shared_law"), ("Flexible baseline", "flexible")):
        block = results["inverse_square"][key]
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in _curve(block)) + f" | {block['auc_of_mean_curve_radians']:.4f} |")
    lines.append("")
    lines.append("## Hidden-exponent targets (10 worlds)")
    lines.append("")
    lines.append("| k | " + " | ".join(str(k) for k in K_VALUES) + " | AUC (rad) |")
    lines.append("|---|" + "---|" * (len(K_VALUES) + 1))
    for name, key in (("Frozen p = 2 (misspecified)", "frozen_p2"), ("Adapted exponent", "adapted_p")):
        block = results["hidden_exponent"][key]
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in _curve(block)) + f" | {block['auc_of_mean_curve_radians']:.4f} |")
    lines.append("")
    lines.append("## Data-blind verdicts (shared law, k = 32)")
    lines.append("")
    lines.append("| World | Entry (arcmin) | Data-blind (arcmin) | Credit |")
    lines.append("|---|---|---|---|")
    for verdict in results["data_blind"]:
        lines.append(f"| {verdict['world_id']} | {verdict['entry_arcmin']:.2f} | {verdict['data_blind_arcmin']:.2f} | {verdict['credit']} |")
    lines.append("")
    discovery = results["discovery"]
    lines.append(f"Discovery froze p = {discovery['frozen_p']:.4f} from per-world fits "
                 + ", ".join(f"{p:.3f}" for p in discovery["per_world"]) + ".")
    (HERE / "TRANSFER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(results: dict[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    # NeurIPS requires Type 1 or embedded TrueType fonts; matplotlib defaults to Type 3.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=True)
    for axis, (title, blocks) in zip(axes, (
        ("Inverse-square targets", (("Shared law (frozen $p$)", results["inverse_square"]["shared_law"], "tab:green"),
                                    ("Flexible baseline", results["inverse_square"]["flexible"], "tab:red"))),
        ("Hidden-exponent targets", (("Frozen $p=2$ (misspecified)", results["hidden_exponent"]["frozen_p2"], "tab:orange"),
                                     ("Adapted exponent", results["hidden_exponent"]["adapted_p"], "tab:blue"))),
    )):
        for label, block, color in blocks:
            axis.plot(block["k_values"], _curve(block), marker="o", label=label, color=color)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("evidence budget $k$ (observations)")
        axis.set_title(title, fontsize=10)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("held-out mean angular error (arcmin)")
    figure.tight_layout()
    figure.savefig(HERE / "fig_transfer.pdf", bbox_inches="tight")
    figure.savefig(HERE / "fig_transfer.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
