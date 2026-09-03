#!/usr/bin/env python3
"""Score every expression on an AI Feynman Pareto frontier on the pilot's splits.

usage: score_solutions.py SOLUTION_FILE --inputs {time,phases}
                          [--epochs inputs/pilot_epochs.csv] [--resamples 2000] [--seed 23]

Metric: median absolute residual in arcmin on the training epochs and on the
held-out epochs, each with a percentile bootstrap 95 percent interval over
epochs. The defaults, 2,000 resamples and seed 23, are the procedure behind
the AI Feynman row of the paper's Table 2. The pilot's other rows use 10,000
resamples and seed 20260829, which `--resamples 10000 --seed 20260829`
reproduces here.

Selection rule: the reported expression is the one with the lowest training
error among expressions that evaluate to finite values on both splits.

Complexity: `complexity_bits` is column 4 of AI Feynman's solution file. When
the aifeynman package is importable, `complexity_bits_recomputed` is
aifeynman's own get_expr_complexity applied to the expression string.

Writes scored_<inputs>.json beside the solution file.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

HERE = Path(__file__).resolve().parent
INPUT_COLUMNS = {"time": ["t_years"], "phases": ["phase_sid", "phase_syn"]}


def bootstrap_median_arcmin(residual_deg, resamples, seed):
    rng = np.random.default_rng(seed)
    res = np.abs(residual_deg)
    boots = [np.median(rng.choice(res, len(res), replace=True)) * 60 for _ in range(resamples)]
    return [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("solution_file", type=Path)
    ap.add_argument("--inputs", choices=sorted(INPUT_COLUMNS), required=True)
    ap.add_argument("--epochs", type=Path, default=HERE / "inputs" / "pilot_epochs.csv")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    table = pd.read_csv(args.epochs)
    train, heldout = table[table.split == "train"], table[table.split == "heldout"]
    columns = INPUT_COLUMNS[args.inputs]
    symbols = sp.symbols(" ".join(f"x{i}" for i in range(len(columns))))
    if len(columns) == 1:
        symbols = (symbols,)
    try:
        from aifeynman.S_get_expr_complexity import get_expr_complexity
    except Exception:  # aifeynman not importable here
        get_expr_complexity = None

    rows = []
    for line in args.solution_file.read_text().strip().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6:
            continue
        expr = parts[5]
        record = {"expr": expr, "complexity_bits": float(parts[3])}
        if get_expr_complexity is not None:
            try:
                record["complexity_bits_recomputed"] = float(get_expr_complexity(expr))
            except Exception as error:
                record["complexity_error"] = str(error)[:80]
        try:
            f = sp.lambdify(symbols, sp.sympify(expr), "numpy")
            for name, part in (("train", train), ("heldout", heldout)):
                y = part.declination_deg.values
                pred = np.broadcast_to(np.asarray(f(*[part[c].values for c in columns]), dtype=float), y.shape)
                residual = pred - y
                record[name] = float(np.median(np.abs(residual)) * 60)
                record[f"{name}_ci95"] = bootstrap_median_arcmin(residual, args.resamples, args.seed)
        except Exception as error:
            record["error"] = str(error)[:80]
        rows.append(record)

    valid = [r for r in rows if all(math.isfinite(r.get(k, float("nan"))) for k in ("train", "heldout"))]
    selected = min(valid, key=lambda r: r["train"]) if valid else None
    rows.sort(key=lambda r: r.get("train", float("inf")) if math.isfinite(r.get("train", float("nan"))) else float("inf"))
    out = {
        "solution_file": args.solution_file.name, "inputs": columns,
        "metric": "median absolute residual in arcmin; percentile bootstrap of the median over epochs",
        "resamples": args.resamples, "seed": args.seed,
        "selection_rule": "lowest training error among expressions finite on both splits",
        "selected": selected, "frontier": rows,
    }
    target = args.solution_file.parent / f"scored_{args.inputs}.json"
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(rows)} expressions scored -> {target}")
    for r in rows:
        train_s = f"{r['train']:8.2f} {r['train_ci95']}" if "train" in r else "   invalid"
        held_s = f"{r['heldout']:8.2f} {r['heldout_ci95']}" if "heldout" in r else ""
        print(f"  bits={r['complexity_bits']:6.1f}  train={train_s}  heldout={held_s}  {r['expr'][:70]}")
    if selected:
        print(f"selected: {selected['expr']}  train {selected['train']:.2f} {selected['train_ci95']}  heldout {selected['heldout']:.2f} {selected['heldout_ci95']}")


if __name__ == "__main__":
    main()
