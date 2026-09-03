"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json

import numpy as np

from .baseline import harmonic_predict
from .io import read_rows, times_and_directions
from .metrics import angular_error, data_blind_verdict, summarize_angles
from .synthetic import FAMILIES, WorldConfig, generate_observations, sample_world, write_dataset


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
    generate.add_argument("--samples", type=int, default=None, help="number of observation times (default 401; a family draws it unless given)")
    generate.add_argument("--noise", type=float, default=None, help="angular noise standard deviation in radians (default 0; a family draws it unless given)")
    generate.add_argument("--force-exponent", type=float, default=2.0, help="p in F(r) proportional to r^-p; 2 is inverse-square")
    generate.add_argument("--mu", type=float, default=1.0, help="gravitational parameter of the central body")
    generate.add_argument("--target-radius", type=float, default=1.7, help="target orbit scale, periapsis distance divided by (1 - e)")
    generate.add_argument("--eccentricity", type=float, default=0.2, help="target orbital eccentricity in [0, 1)")
    generate.add_argument("--observer-radius", type=float, default=1.0, help="radius of the observer's circular orbit")
    generate.add_argument("--target-phase", type=float, default=0.4, help="rotation of the target's initial state, radians")
    generate.add_argument("--observer-phase", type=float, default=0.0, help="rotation of the observer's initial state, radians")
    generate.add_argument("--duration", type=float, default=20.0, help="length of the observation window in time units")
    generate.add_argument("--train-fraction", type=float, default=0.7, help="fraction of the window used for training")
    generate.add_argument("--world-id", type=str, default=None, help="identifier written to every row; defaults to synthetic-000, or <family>-<seed> for a family draw")
    generate.add_argument("--family", type=str, default=None, choices=sorted(FAMILIES), help="draw the world parameters from a named family using --seed; explicit orbit parameters are then ignored, and --samples and --noise act as overrides")

    baseline = subparsers.add_parser("baseline", help="fit the level-1 harmonic baseline")
    baseline.add_argument("--data", type=Path, required=True)
    baseline.add_argument("--predictions", type=Path, required=True)
    baseline.add_argument("--order", type=int, default=8)

    evaluate = subparsers.add_parser("evaluate-prediction", help="score test-set planet directions")
    evaluate.add_argument("--truth", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--data-blind-predictions", type=Path, default=None, help="predictions of the same system run on the data-blind copy of the dataset; enables the data-blind rule")
    evaluate.add_argument("--noise-floor", type=float, default=None, help="noise floor in radians for the data-blind rule (default: the world's angular_noise_std from its metadata sidecar, else 0)")

    blind = subparsers.add_parser("data-blind", help="write the data-blind copy of a dataset: times, identifiers and splits kept, observed quantities permuted within each split")
    blind.add_argument("--input", type=Path, required=True)
    blind.add_argument("--output", type=Path, required=True)
    blind.add_argument("--seed", type=int, default=0)

    hist_eval = subparsers.add_parser("evaluate-historical", help="score declination predictions on one split of the rebuilt historical table")
    hist_eval.add_argument("--truth", type=Path, default=Path("data/historical/mars_declinations.csv"))
    hist_eval.add_argument("--predictions", type=Path, required=True, help="CSV with epoch_index, declination_deg and optionally uncertainty_deg")
    hist_eval.add_argument("--split", default="heldout", choices=["train", "heldout"])
    hist_eval.add_argument("--resamples", type=int, default=10000, help="bootstrap resamples (the pilot uses 10000)")
    hist_eval.add_argument("--seed", type=int, default=20260829, help="bootstrap seed (the pilot uses 20260829)")
    hist_eval.add_argument("--data-blind-predictions", type=Path, default=None, help="predictions of the same system run on the data-blind copy of the table")
    hist_eval.add_argument("--noise-floor-arcmin", type=float, default=2.0, help="noise floor of Tycho's record for the data-blind rule")

    historical = subparsers.add_parser("fetch-historical", help="download Pafko's transcription of Tycho's Mars declinations and rebuild the pilot epoch table locally")
    historical.add_argument("--output", type=Path, default=Path("data/historical/mars_declinations.csv"))
    historical.add_argument("--cache", type=Path, default=None, help="where the downloaded spreadsheet is kept (default ~/.cache/keplerbench)")
    historical.add_argument("--xls", type=Path, default=None, help="use an already downloaded spreadsheet instead of fetching")

    args = parser.parse_args(argv)
    if args.command == "generate":
        if args.family is not None:
            overrides = {name: value for name, value in (("sample_count", args.samples), ("angular_noise_std", args.noise)) if value is not None}
            config = sample_world(args.family, seed=args.seed, world_id=args.world_id, **overrides)
        else:
            config = WorldConfig(
                world_id=args.world_id or "synthetic-000", sample_count=args.samples if args.samples is not None else 401,
                angular_noise_std=args.noise if args.noise is not None else 0.0,
                force_exponent=args.force_exponent, gravitational_parameter=args.mu,
                target_radius=args.target_radius, target_eccentricity=args.eccentricity,
                observer_radius=args.observer_radius, target_phase=args.target_phase,
                observer_phase=args.observer_phase, duration=args.duration, train_fraction=args.train_fraction,
            )
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
    elif args.command == "fetch-historical":
        from . import __version__
        from .historical import fetch_pafko, reduce_pafko_with_info, summary, write_epochs
        from .historical.tycho import DEFAULT_CACHE, PAFKO_URL, SPLIT_YEAR, julian_calendar_to_jd, sha256
        xls = args.xls or fetch_pafko(args.cache or DEFAULT_CACHE)
        rows, info = reduce_pafko_with_info(xls)
        write_epochs(rows, args.output)
        report = {
            **summary(rows), **info,
            "split_jd": float(julian_calendar_to_jd(SPLIT_YEAR, 1, 1)),
            "source_url": PAFKO_URL, "source": str(xls), "source_sha256": sha256(xls),
            "output": str(args.output), "output_sha256": sha256(args.output),
            "reduction": ("jd = column 14 + 1721048.0; nightly median of repeat readings; timestamps whose "
                          "readings spread by more than one degree dropped; train before Julian-calendar "
                          f"1 January {SPLIT_YEAR}, held out from then on"),
            "keplerbench_version": __version__,
        }
        # The sidecar carries the reduction counts and checksums, never declinations.
        meta = args.output.with_name(args.output.stem + ".meta.json")
        meta.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "data-blind":
        from .blind import write_data_blind
        print(json.dumps(write_data_blind(args.input, args.output, args.seed), indent=2))
    elif args.command == "evaluate-historical":
        from .historical import evaluate_historical
        print(json.dumps(evaluate_historical(args.truth, args.predictions, args.split, args.resamples, args.seed,
                                             args.data_blind_predictions, args.noise_floor_arcmin), indent=2))
    else:
        truth_rows = [row for row in read_rows(args.truth) if row["split"] == "test"]
        truth_times, truth = times_and_directions(truth_rows)

        def score(path: Path) -> dict[str, float]:
            prediction_times, prediction = times_and_directions(_prediction_rows(path))
            if truth_times.shape != prediction_times.shape or not np.allclose(truth_times, prediction_times):
                raise ValueError("prediction times must exactly match truth test times")
            return summarize_angles(angular_error(truth, prediction))

        result: dict[str, object] = dict(score(args.predictions))
        if args.data_blind_predictions is not None:
            floor = args.noise_floor
            if floor is None:
                sidecar = args.truth.with_suffix(args.truth.suffix + ".metadata.json")
                floor = json.loads(sidecar.read_text(encoding="utf-8"))["latent_world_parameters"]["angular_noise_std"] if sidecar.exists() else 0.0
            blind = score(args.data_blind_predictions)
            result["data_blind_control"] = data_blind_verdict(result["mean_radians"], blind["mean_radians"], floor, "radians")
            result["data_blind_control"]["data_blind_run"] = blind
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
