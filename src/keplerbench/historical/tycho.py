"""Historical Mars sub-benchmark: Tycho Brahe's Mars declinations.

The observations are Tycho's, printed in Dreyer's edition of the Opera Omnia
(vols. X to XIII, Copenhagen 1923 to 1926) and public domain. The
machine-readable transcription used here is Wayne Pafko's spreadsheet
(https://www.pafko.com/tycho/observe.html, 2000), which carries a copyright
notice and no license. This module therefore does not ship the transcription.
It downloads the spreadsheet from the source site on the user's machine and
reproduces the reduction and the chronological split used by the pilot in the
paper, so that the epoch table is rebuilt locally rather than redistributed.

Reduction (identical to the pilot):

* rows with a finite day count (column 14, "days since 1 AD") and a finite
  declination (column 20, degrees) are kept;
* the Julian day number is column 14 plus 1721048.0, which reproduces the
  Julian-calendar date and time of day exactly (Pafko's decimal-date column
  applies the Gregorian shift twice and is not used);
* repeat readings at one timestamp become their median, and timestamps whose
  readings spread by more than one degree are dropped;
* the split is chronological, training before Julian-calendar 1 January 1594
  and held out from then on.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import urllib.request

import numpy as np

PAFKO_URL = "https://www.pafko.com/tycho/mars.xls"
PAFKO_JD_OFFSET = 1721048.0
SPLIT_YEAR = 1594
DEFAULT_CACHE = Path.home() / ".cache" / "keplerbench"
FIELDS = ("epoch_index", "jd", "year", "declination_deg", "n_repeats", "split")


def julian_calendar_to_jd(year, month, day, hour=0.0, minute=0.0):
    """Julian day number of a Julian-calendar date (Meeus, chapter 7).

    Tycho's Denmark kept the Julian calendar, so no Gregorian correction applies.
    """

    y = np.asarray(year, dtype=float)
    m = np.asarray(month, dtype=float)
    d = np.asarray(day, dtype=float) + np.asarray(hour, dtype=float) / 24.0 + np.asarray(minute, dtype=float) / 1440.0
    y2 = np.where(m <= 2, y - 1, y)
    m2 = np.where(m <= 2, m + 12, m)
    return np.floor(365.25 * (y2 + 4716)) + np.floor(30.6001 * (m2 + 1)) + d - 1524.5


def julian_calendar_year(jd: float) -> int:
    """Julian-calendar year of a Julian day number (Meeus, chapter 7, Julian branch)."""

    z = int(jd + 0.5)
    b = z + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    month = e - 1 if e < 14 else e - 13
    return c - 4716 if month > 2 else c - 4715


def fetch_pafko(cache_dir: Path = DEFAULT_CACHE, url: str = PAFKO_URL) -> Path:
    """Download Pafko's spreadsheet into the cache directory if it is not there yet."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "tycho_mars_pafko.xls"
    if not target.exists():
        with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as handle:
            handle.write(response.read())
    return target


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reduce_pafko(xls_path: Path) -> list[dict[str, object]]:
    """Reduce the transcription to nightly epochs with the pilot's split."""

    return reduce_pafko_with_info(xls_path)[0]


def reduce_pafko_with_info(xls_path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Reduce the transcription and report the reduction counts.

    Returns the epoch rows and a dictionary with the number of finite source
    rows, the number of timestamps dropped for an internal spread above one
    degree, and the number of epochs kept.
    """

    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover
        raise ImportError("the historical sub-benchmark needs pandas and xlrd: pip install 'keplerbench[historical]'") from error
    frame = pd.read_excel(xls_path, header=None)
    data = frame.iloc[6:].reset_index(drop=True)
    jd_raw = pd.to_numeric(data[14], errors="coerce").to_numpy(dtype=float) + PAFKO_JD_OFFSET
    dec_raw = pd.to_numeric(data[20], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(jd_raw) & np.isfinite(dec_raw)
    jd_raw, dec_raw = jd_raw[ok], dec_raw[ok]

    groups: dict[float, list[int]] = defaultdict(list)
    for index, key in enumerate(np.round(jd_raw, 5)):
        groups[float(key)].append(index)
    split_jd = float(julian_calendar_to_jd(SPLIT_YEAR, 1, 1))
    rows: list[dict[str, object]] = []
    dropped = 0
    for key in sorted(groups):
        values = dec_raw[np.array(groups[key])]
        if np.ptp(values) > 1.0:
            dropped += 1
            continue
        rows.append({
            "epoch_index": len(rows),
            "jd": key,
            "year": julian_calendar_year(key),
            "declination_deg": float(np.median(values)),
            "n_repeats": len(values),
            "split": "train" if key < split_jd else "heldout",
        })
    info = {"n_raw_rows": int(ok.sum()), "n_dropped_spread": dropped, "n_epochs": len(rows)}
    return rows, info


def write_epochs(rows: list[dict[str, object]], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_split(rows: list[dict[str, object]], output: Path) -> None:
    """Write the split without declinations: epoch index, Julian day, year, label."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch_index", "jd", "year", "split"))
        writer.writeheader()
        writer.writerows({k: row[k] for k in ("epoch_index", "jd", "year", "split")} for row in rows)


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    n_train = sum(row["split"] == "train" for row in rows)
    return {"epochs": len(rows), "train": n_train, "heldout": len(rows) - n_train, "split_year": SPLIT_YEAR}
