"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json

import numpy as np

from .baseline import harmonic_predict
from .io import read_rows, times_and_directions
from .metrics import angular_error, summarize_angles
from .synthetic import WorldConfig, generate_observations, write_dataset


def _prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"time", "planet_dir_x", "planet_dir_y", "planet_dir_z"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("prediction file must contain time and planet_dir_x/y/z")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="keplerbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a synthetic observer-centred world")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--samples", type=int, default=401)
    generate.add_argument("--noise", type=float, default=0.0, help="angular noise standard deviation in radians")
    generate.add_argument("--force-exponent", type=float, default=2.0)

    baseline = subparsers.add_parser("baseline", help="fit the level-1 harmonic baseline")
    baseline.add_argument("--data", type=Path, required=True)
    baseline.add_argument("--predictions", type=Path, required=True)
    baseline.add_argument("--order", type=int, default=8)

    evaluate = subparsers.add_parser("evaluate-prediction", help="score test-set planet directions")
    evaluate.add_argument("--truth", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "generate":
        config = WorldConfig(sample_count=args.samples, angular_noise_std=args.noise, force_exponent=args.force_exponent)
        rows = generate_observations(config, seed=args.seed)
        write_dataset(rows, args.output, config, args.seed)
        print(json.dumps({"rows": len(rows), "output": str(args.output)}))
    elif args.command == "baseline":
        rows = read_rows(args.data)
        train = [row for row in rows if row["split"] == "train"]
        test = [row for row in rows if row["split"] == "test"]
        train_times, train_directions = times_and_directions(train)
        test_times, _ = times_and_directions(test)
        prediction = harmonic_predict(train_times, train_directions, test_times, order=args.order)
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "planet_dir_x", "planet_dir_y", "planet_dir_z"])
            writer.writerows(np.column_stack([test_times, prediction]))
        print(json.dumps({"rows": len(test), "output": str(args.predictions)}))
    else:
        truth_rows = [row for row in read_rows(args.truth) if row["split"] == "test"]
        prediction_rows = _prediction_rows(args.predictions)
        truth_times, truth = times_and_directions(truth_rows)
        prediction_times, prediction = times_and_directions(prediction_rows)
        if truth_times.shape != prediction_times.shape or not np.allclose(truth_times, prediction_times):
            raise ValueError("prediction times must exactly match truth test times")
        print(json.dumps(summarize_angles(angular_error(truth, prediction)), indent=2))


if __name__ == "__main__":
    main()
