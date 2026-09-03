"""No committed file may carry declination values derived from Pafko's transcription."""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "historical" / "mars_declinations.csv"


def _tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / line for line in out.splitlines() if line]


def test_rebuilt_table_and_sidecar_are_not_tracked():
    tracked = {path.relative_to(ROOT).as_posix() for path in _tracked_files()}
    assert "data/historical/mars_declinations.csv" not in tracked
    assert "data/historical/mars_declinations.meta.json" not in tracked
    assert (ROOT / "data" / "historical" / "mars_split.csv").exists(), "the committed split file is missing"


def test_no_tracked_file_contains_declination_values():
    if not TABLE.exists():
        pytest.skip("epoch table not rebuilt; run `keplerbench fetch-historical`")
    rows = TABLE.read_text().splitlines()[1:]
    values = [row.split(",")[3] for row in rows]
    probes = {value for value in values[::37]} | {f"{float(value):.4f}" for value in values[::53]}
    probes = {probe.rstrip("0") for probe in probes if len(probe.rstrip("0")) >= 6}
    offenders = []
    for path in _tracked_files():
        if path.suffix in {".xls", ".png", ".pdf", ".jpg"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [probe for probe in probes if probe in text]
        if hits:
            offenders.append((path.relative_to(ROOT).as_posix(), hits[:3]))
    assert not offenders, offenders


def test_results_summary_has_no_per_epoch_arrays():
    summary = ROOT / "experiments" / "pilot" / "results_summary.json"
    if not summary.exists():
        pytest.skip("results summary not written")
    text = summary.read_text()
    for key in ('"points"', '"held_out_by_epoch"', '"injected_residual_arcmin"'):
        assert key not in text, key
    data = json.loads(text)
    assert "summary_note" in data
    long_arrays = re.findall(r"\[(?:\s*-?\d+\.\d+,){80,}", text)
    assert not long_arrays, "a per-epoch numeric array survived in the summary"
