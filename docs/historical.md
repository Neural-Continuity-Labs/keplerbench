# Historical sub-benchmarks

## What is available now: Tycho's Mars declinations

The historical Mars sub-benchmark currently consists of Tycho Brahe's Mars
declinations, 923 readings between 1582 and 1600 printed in Dreyer's edition of
the *Opera Omnia* (vols. X to XIII, Copenhagen 1923 to 1926). The observations
are public domain. The machine-readable transcription is Wayne Pafko's
spreadsheet (https://www.pafko.com/tycho/observe.html, 2000), which carries the
notice "Copyright 2000, Wayne Pafko" and no license. The repository therefore
does not redistribute the spreadsheet or any table of declination values derived
from it. Instead,

```bash
pip install 'keplerbench[historical]'
keplerbench fetch-historical
```

downloads the spreadsheet from the source site into `~/.cache/keplerbench/`,
reports its SHA-256 (the copy used for the paper hashes to
`0e1c95d3e61514a5d4120a56988801b13c31be32cb69ec51603b23fab6b2e845`), and rebuilds
`data/historical/mars_declinations.csv` locally with the reduction and split
used in the paper's pilot, together with a sidecar
`mars_declinations.meta.json` that records the source checksum, the reduction
counts and the split, and no declinations. The committed file
`data/historical/mars_split.csv` lists the 354 epochs by Julian day with their
split label and no declinations, so the split is fixed and checkable without
the source file.

### Reduction

- Rows with a finite day count (column 14, "days since 1 AD") and a finite
  declination (column 20, degrees) are kept.
- The Julian day is column 14 plus 1721048.0. This reproduces the
  Julian-calendar date and time of day exactly; Pafko's decimal-date column
  applies the Gregorian shift twice and runs about ten days late, so it is not
  used. The convention was checked against unfitted modern orbital elements,
  whose residual is minimized at zero shift by more than a factor of thirteen.
- Repeat readings at one timestamp become their median, and timestamps whose
  readings spread by more than one degree are dropped (four groups).
- Result: 354 nightly epochs. Training is everything before Julian-calendar
  1 January 1594 (268 epochs); the 86 epochs of the 1594, 1595, 1596 and 1600
  campaigns (5, 54, 20 and 7 epochs) are held out.

### What the record contains

The declination is one coordinate of the apparent direction from Earth. The
transcription has no right ascension (two rows carry one inside the quoted
text), no Sun direction, and no reference-star distances, so this sub-benchmark
supports the prediction capability under temporal extrapolation and nothing
beyond it. Tycho's repeat readings scatter by about 0.5 arcmin within a night,
and the record's noise floor is taken as 2 arcmin.

### Known structure in the residuals

Against unfitted modern orbital elements the record has a median residual of
about 3.5 arcmin, an altitude-dependent bias consistent with refraction
(Spearman rho of -0.19 between residual and transit altitude), 18 gross errors
above 60 arcmin, and a campaign-to-campaign offset that neither the altitude
term nor the gross errors explain. A six-parameter Keplerian orbit fitted to the
training years absorbs part of this structure and trails the unfitted modern
elements on the held-out years (13.31 against 3.48 arcmin); with declination
alone, the orbit scale and the phase are also nearly interchangeable (their
fitted values correlate at 0.98). These facts are the reason the paper fixes a
noise floor and asks entries to state their uncertainty. Each of them is
computed by the pilot and documented, with the key of the results file that
holds it, in [`../experiments/pilot/README.md`](../experiments/pilot/README.md).

### Scoring an entry

An entry predicts the declination at the held-out epochs and writes a CSV with
`epoch_index`, `declination_deg` and, optionally, `uncertainty_deg`. Then

```bash
keplerbench evaluate-historical --predictions predictions.csv
keplerbench data-blind --input data/historical/mars_declinations.csv --output data/historical/mars_blind.csv --seed 0
keplerbench evaluate-historical --predictions predictions.csv --data-blind-predictions predictions_blind.csv
```

report the median absolute residual with its bootstrap interval, the campaign
breakdown, the coverage of the stated uncertainty, and, with the predictions
of the same system run on the data-blind copy, both scores and the verdict of
the data-blind rule at the 2 arcmin floor.

## Provenance plan for the full historical release

There is no single table that can be identified without qualification as "the
data Kepler used." The full historical release will therefore distinguish

- direct observations;
- quantities available in contemporary tables or sources;
- values derived during Kepler's investigation; and
- modern reductions introduced by the benchmark maintainers.

Every released record will carry a source identifier, page or table reference
where possible, transcription status, units, calendar convention, uncertainty
notes, and a transformation log. Original and normalized forms will be retained
side by side. Modern heliocentric states and orbital elements will not appear in
the learner-facing table. Records will be published only once each has a
traceable source and a documented rights status; the Mars declinations above are
the first instance of that rule.
