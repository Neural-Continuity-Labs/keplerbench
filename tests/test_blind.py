import csv
import json
from pathlib import Path

import numpy as np

from keplerbench.blind import SYNTHETIC_COLUMNS, data_blind_rows, write_data_blind
from keplerbench.cli import main
from keplerbench.io import read_rows, times_and_directions
from keplerbench.metrics import angular_error, data_blind_verdict


def _world(tmp_path: Path, seed: int = 4) -> Path:
    dataset = tmp_path / "world.csv"
    main(["generate", "--output", str(dataset), "--samples", "121", "--seed", str(seed), "--noise", "0.0003"])
    return dataset


def test_data_blind_keeps_shape_and_permutes_within_split(tmp_path: Path):
    dataset = _world(tmp_path)
    rows = read_rows(dataset)
    blind, kind, columns = data_blind_rows(rows, seed=1)
    assert kind == "synthetic" and columns == SYNTHETIC_COLUMNS
    assert [r["time"] for r in blind] == [r["time"] for r in rows]
    assert [r["split"] for r in blind] == [r["split"] for r in rows]
    for label in ("train", "test"):
        original = sorted(tuple(r[c] for c in columns) for r in rows if r["split"] == label)
        permuted = sorted(tuple(r[c] for c in columns) for r in blind if r["split"] == label)
        assert original == permuted, "the null copy must be a permutation within the split"
    # rows stay internally consistent: each blind row is one original row's directions
    originals = {tuple(r[c] for c in columns) for r in rows}
    assert all(tuple(r[c] for c in columns) in originals for r in blind)
    assert blind != rows
    assert data_blind_rows(rows, seed=1)[0] == blind, "seeded permutation is deterministic"


def test_data_blind_command_writes_sidecar(tmp_path: Path):
    dataset = _world(tmp_path)
    output = tmp_path / "world_blind.csv"
    report = write_data_blind(dataset, output, seed=7)
    assert report["kind"] == "synthetic" and report["seed"] == 7
    sidecar = json.loads(output.with_suffix(".csv.metadata.json").read_text())
    assert sidecar["data_blind"] is True and sidecar["permuted_columns"] == list(SYNTHETIC_COLUMNS)
    _, directions = times_and_directions(read_rows(output))
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_historical_table_is_detected(tmp_path: Path):
    path = tmp_path / "mars.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch_index", "jd", "year", "declination_deg", "n_repeats", "split"])
        writer.writeheader()
        for i in range(12):
            writer.writerow({"epoch_index": i, "jd": 2299000.5 + i, "year": 1582, "declination_deg": float(i), "n_repeats": 1, "split": "train" if i < 8 else "heldout"})
    blind, kind, columns = data_blind_rows(list(csv.DictReader(path.open())), seed=3)
    assert kind == "historical" and columns == ("declination_deg",)
    assert sorted(float(r["declination_deg"]) for r in blind[:8]) == list(map(float, range(8)))
    assert sorted(float(r["declination_deg"]) for r in blind[8:]) == [8.0, 9.0, 10.0, 11.0]


def test_data_blind_rule_denies_credit_to_a_data_ignoring_system_and_grants_it_to_a_data_using_one(tmp_path: Path):
    dataset = _world(tmp_path)
    rows = read_rows(dataset)
    blind, _, _ = data_blind_rows(rows, seed=2)

    def last_training_direction(data):
        train = [r for r in data if r["split"] == "train"]
        test = [r for r in data if r["split"] == "test"]
        _, train_dirs = times_and_directions(train)
        return np.repeat(train_dirs[-1:], len(test), axis=0)

    def fixed_direction(data):
        test = [r for r in data if r["split"] == "test"]
        return np.repeat(np.array([[1.0, 0.0, 0.0]]), len(test), axis=0)

    _, truth = times_and_directions([r for r in rows if r["split"] == "test"])
    floor = 0.0003
    for system, expected in ((last_training_direction, True), (fixed_direction, False)):
        entry = float(np.mean(angular_error(truth, system(rows))))
        null = float(np.mean(angular_error(truth, system(blind))))
        verdict = data_blind_verdict(entry, null, floor, "radians")
        assert verdict["credit"] is expected, (system.__name__, verdict)


def test_evaluate_prediction_reports_both_scores(tmp_path: Path, capsys):
    dataset = _world(tmp_path)
    blind = tmp_path / "blind.csv"
    main(["data-blind", "--input", str(dataset), "--output", str(blind), "--seed", "5"])
    entry, null = tmp_path / "p.csv", tmp_path / "pb.csv"
    main(["baseline", "--data", str(dataset), "--predictions", str(entry), "--order", "3"])
    main(["baseline", "--data", str(blind), "--predictions", str(null), "--order", "3"])
    capsys.readouterr()
    main(["evaluate-prediction", "--truth", str(dataset), "--predictions", str(entry), "--data-blind-predictions", str(null)])
    out = json.loads(capsys.readouterr().out)
    control = out["data_blind_control"]
    assert control["noise_floor"] == 0.0003, "floor defaults to the world's angular noise"
    assert set(control) >= {"entry", "data_blind", "difference", "credit"}
    assert control["entry"] == out["mean_radians"]
