"""Data-blind control: null inputs with the shape of a dataset.

The protocol scores every entry twice, once on the evidence and once on a
data-blind copy in which the times, identifiers and split labels are kept and
the observed quantities are replaced by a draw from the null distribution. The
null distribution is a permutation of the observed quantities across the epochs
of each split. It keeps the marginal distribution of the observations and
destroys their relation to time, so a system whose score is carried by prior
knowledge scores the same on both inputs, while a system that extracts
structure from the evidence does not.

For a synthetic world the six direction components of a row move together, so
every row of the null copy is still an internally consistent observation. For
the historical table the declination moves.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

SYNTHETIC_COLUMNS = ("planet_dir_x", "planet_dir_y", "planet_dir_z", "star_dir_x", "star_dir_y", "star_dir_z")
HISTORICAL_COLUMNS = ("declination_deg",)
NULL_DESCRIPTION = "permutation of the observed quantities across the epochs of each split"


def permuted_rows(rows: list[dict[str, str]], columns: tuple[str, ...], seed: int, split_column: str = "split") -> list[dict[str, str]]:
    """Return copies of the rows with `columns` permuted across rows within each split."""

    rng = np.random.default_rng(seed)
    out = [dict(row) for row in rows]
    for label in sorted({row[split_column] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row[split_column] == label]
        order = rng.permutation(len(indices))
        for position, target in enumerate(indices):
            source = rows[indices[order[position]]]
            for column in columns:
                out[target][column] = source[column]
    return out


def data_blind_rows(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], str, tuple[str, ...]]:
    """Detect the dataset kind from its columns and return the null copy, the kind and the permuted columns."""

    if not rows:
        raise ValueError("dataset is empty")
    present = set(rows[0])
    if set(SYNTHETIC_COLUMNS) <= present:
        columns, kind = SYNTHETIC_COLUMNS, "synthetic"
    elif set(HISTORICAL_COLUMNS) <= present:
        columns, kind = HISTORICAL_COLUMNS, "historical"
    else:
        raise ValueError("dataset has neither the synthetic direction columns nor a declination column")
    if "split" not in present:
        raise ValueError("dataset needs a split column")
    return permuted_rows(rows, columns, seed), kind, columns


def write_data_blind(input_path: Path, output_path: Path, seed: int) -> dict[str, object]:
    """Write the null copy of a dataset and a sidecar that records the null and the seed."""

    input_path, output_path = Path(input_path), Path(output_path)
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    out, kind, columns = data_blind_rows(rows, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)
    report = {
        "data_blind": True, "kind": kind, "source": str(input_path), "output": str(output_path),
        "null": NULL_DESCRIPTION, "permuted_columns": list(columns), "seed": seed, "rows": len(out),
    }
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
