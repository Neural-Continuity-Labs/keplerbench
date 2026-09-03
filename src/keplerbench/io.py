"""CSV input helpers with strict public-contract validation."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

REQUIRED_COLUMNS = (
    "world_id", "system_id", "observer_id", "target_id", "time",
    "planet_dir_x", "planet_dir_y", "planet_dir_z",
    "star_dir_x", "star_dir_y", "star_dir_z", "split",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def times_and_directions(rows: list[dict[str, str]], prefix: str = "planet_dir") -> tuple[np.ndarray, np.ndarray]:
    times = np.array([float(row["time"]) for row in rows])
    directions = np.array([[float(row[f"{prefix}_{axis}"]) for axis in "xyz"] for row in rows])
    norms = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(directions)) or np.any(np.abs(norms - 1.0) > 1e-5):
        raise ValueError(f"{prefix} values must be finite unit vectors")
    return times, directions
