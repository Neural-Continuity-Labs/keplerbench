#!/usr/bin/env python3
"""Run AI Feynman on one prepared input file and record its Pareto frontier.

usage: run_aif.py INPUT TAG [--bf-time S] [--nn-epochs N] [--ops FILE] [--polyfit-deg D]

The input file's last column is the target and the earlier columns are the
inputs, in AI Feynman's convention. AI Feynman writes its working files under
runs/TAG/results/. This script copies the final frontier to runs/TAG/ and
writes runs/TAG/report.json with the settings, the runtime, and the frontier
with named columns. Scoring on the held-out epochs is done by
score_solutions.py, so there is a single scoring path.

Solution-file columns, from AI Feynman's S_run_aifeynman.py: the test error of
the expression on AI Feynman's own internal split, the mean description length
of the residuals in bits, that length summed over the data points, the
complexity of the expression in bits, the fit error, and the expression.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLUMNS = ("test_error", "log_error_bits", "pooled_log_error_bits",
           "complexity_bits", "fit_error", "expr")


def parse_solution_file(path):
    rows = []
    for line in Path(path).read_text().strip().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6:
            continue
        rows.append(dict(zip(COLUMNS, [*map(float, parts[:5]), parts[5]])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", type=Path, help="prepared input file, e.g. inputs/pilot_phases_to_dec.txt")
    ap.add_argument("tag", help="name of the run directory under runs/")
    ap.add_argument("--bf-time", type=int, default=60, help="brute-force try time in seconds (BF_try_time)")
    ap.add_argument("--nn-epochs", type=int, default=500, help="neural network epochs (NN_epochs)")
    ap.add_argument("--ops", default="14ops.txt", help="operation set file shipped with AI Feynman")
    ap.add_argument("--polyfit-deg", type=int, default=3, help="polynomial fit degree")
    args = ap.parse_args()

    inp = args.input.resolve()
    work = HERE / "runs" / args.tag
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)

    import aifeynman  # the brute-force binaries must be on PATH: run inside the AI Feynman venv
    t0 = time.time()
    aifeynman.run_aifeynman(str(inp.parent) + "/", inp.name, args.bf_time, args.ops,
                            polyfit_deg=args.polyfit_deg, NN_epochs=args.nn_epochs)
    runtime = time.time() - t0

    solution = work / "results" / f"solution_{inp.name}"
    final = work / f"solution_{inp.name}"
    shutil.copyfile(solution, final)
    report = {
        "input": inp.name, "tag": args.tag,
        "bf_try_time": args.bf_time, "nn_epochs": args.nn_epochs,
        "ops_file": args.ops, "polyfit_deg": args.polyfit_deg,
        "aifeynman_version": getattr(aifeynman, "__version__", "2.0.7"),
        "runtime_s": round(runtime, 1),
        "solution_file": final.name,
        "frontier": parse_solution_file(final),
    }
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "frontier"}, indent=2))
    for row in report["frontier"]:
        print(f"  complexity {row['complexity_bits']:6.1f} bits  {row['expr'][:90]}")


if __name__ == "__main__":
    main()
