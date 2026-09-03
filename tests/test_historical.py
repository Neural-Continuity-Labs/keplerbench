import csv
from pathlib import Path

import pytest

from keplerbench.historical.tycho import SPLIT_YEAR, julian_calendar_to_jd, julian_calendar_year, summary, write_epochs, write_split


def test_julian_calendar_round_trip():
    jd = float(julian_calendar_to_jd(1594, 1, 1))
    assert jd == 2303266.5
    assert julian_calendar_year(jd) == 1594
    assert julian_calendar_year(jd - 1.0) == 1593


def test_split_file_carries_no_declinations(tmp_path: Path):
    rows = [
        {"epoch_index": 0, "jd": 2299198.5, "year": 1582, "declination_deg": 23.1, "n_repeats": 1, "split": "train"},
        {"epoch_index": 1, "jd": 2303300.5, "year": 1594, "declination_deg": -5.0, "n_repeats": 2, "split": "heldout"},
    ]
    write_epochs(rows, tmp_path / "epochs.csv")
    write_split(rows, tmp_path / "split.csv")
    split = list(csv.DictReader((tmp_path / "split.csv").open()))
    assert [r["split"] for r in split] == ["train", "heldout"]
    assert "declination_deg" not in split[0]
    assert summary(rows) == {"epochs": 2, "train": 1, "heldout": 1, "split_year": SPLIT_YEAR}


def test_committed_split_matches_pilot_counts():
    path = Path(__file__).resolve().parents[1] / "data" / "historical" / "mars_split.csv"
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 354
    assert sum(r["split"] == "train" for r in rows) == 268
    assert sum(r["split"] == "heldout" for r in rows) == 86
    assert all(float(r["jd"]) < 2303266.5 for r in rows if r["split"] == "train")


def test_reduction_reproduces_split_when_source_available():
    xls = Path.home() / ".cache" / "keplerbench" / "tycho_mars_pafko.xls"
    if not xls.exists():
        pytest.skip("Pafko spreadsheet not fetched; run `keplerbench fetch-historical`")
    pytest.importorskip("pandas")
    from keplerbench.historical import reduce_pafko
    rows = reduce_pafko(xls)
    path = Path(__file__).resolve().parents[1] / "data" / "historical" / "mars_split.csv"
    committed = list(csv.DictReader(path.open()))
    assert [(r["jd"], r["split"]) for r in committed] == [(f"{row['jd']}", row["split"]) for row in rows]
