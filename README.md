# KeplerBench

KeplerBench is a benchmark for scientific discovery from observer-centred
astronomical measurements. A learner receives times and apparent directions to
a planet and a star. It is not given heliocentric positions, velocities, orbital
elements, or the representation in which the governing law is simple. The
benchmark scores three levels of capability, prediction, global law and local
law, on three sub-benchmarks defined by their evidence: Tycho's Mars record, the
historical record of the other planets, and synthetic worlds.

This repository accompanies an anonymous workshop submission. `src/` holds the
benchmark: the observation contract, the synthetic generator with its world
families, the historical fetch-and-reduce route, the scorers and the data-blind
control. `experiments/` holds the evidence the paper reports: the pilot on
Tycho's Mars declinations and the AI Feynman controls.

## Where each statement of the paper lives

| The paper says | Delivered by |
|---|---|
| Code, data and documentation are in the repository (footnote 1) | this repository; the map below |
| The data-blind control is enforced in software (Section 2) | `keplerbench data-blind`, and the `--data-blind-predictions` option of both scorers; rule in [`docs/protocol.md`](docs/protocol.md) |
| Tycho's Mars declinations, 354 nightly epochs, 268 before 1594 for training and 86 held out (Section 4) | `keplerbench fetch-historical` rebuilds the table from Pafko's transcription; the split is committed as [`data/historical/mars_split.csv`](data/historical/mars_split.csv); reduction and rights in [`docs/historical.md`](docs/historical.md) |
| Seven predictors, their objectives, the AI Feynman settings, the selected expression and the bootstrap (Table 2, footnote 2) | [`experiments/pilot/`](experiments/pilot/README.md) and [`experiments/aifeynman/`](experiments/aifeynman/README.md); the regenerated table in [`experiments/pilot/TABLE2.md`](experiments/pilot/TABLE2.md) |
| The fit absorbs structure in Tycho's residuals that the release documents (Section 4) | the residual analysis in [`experiments/pilot/README.md`](experiments/pilot/README.md), each fact with its key in the results file |
| A noise floor per sub-benchmark, 2 arcmin for Tycho, and stated uncertainty (Section 4) | [`docs/protocol.md`](docs/protocol.md); `evaluate-historical` reads an `uncertainty_deg` column and reports its coverage |
| The first synthetic family varies masses, initial conditions, geometry, observer, sampling and noise under inverse-square dynamics, and harder families hide the force exponent (Section 4) | `keplerbench.synthetic.FAMILIES`, ranges tabulated in [`docs/synthetic.md`](docs/synthetic.md) |
| Data, split files and fitting code; model-selection details (checklist 4 and 6) | [`experiments/pilot/`](experiments/pilot/README.md) |
| Epoch-bootstrap intervals for every number (checklist 7) | `experiments/pilot/make_table.py` regenerates them; `experiments/pilot/tests/test_table2_contract.py` pins them |
| The complete pilot runs in 612 seconds on one CPU, the AI Feynman control in 12 minutes (checklist 8) | the provenance block of the pilot results and `experiments/aifeynman/runs/phases_20s/report.json` |
| Pafko's transcription is credited, not redistributed, and AI Feynman is used under MIT (checklist 12) | the license section below and [`docs/historical.md`](docs/historical.md) |

## Quick start

KeplerBench requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,historical,pilot]'

# synthetic sub-benchmark: an explicit world, or a world drawn from a family
keplerbench generate --output data/example_world.csv --seed 7
keplerbench generate --output data/world_a.csv --family inverse-square --seed 3
keplerbench baseline --data data/example_world.csv --predictions predictions.csv
keplerbench evaluate-prediction --truth data/example_world.csv --predictions predictions.csv

# the data-blind control: run the same system on the null copy, then score both
keplerbench data-blind --input data/example_world.csv --output data/example_blind.csv --seed 0
keplerbench baseline --data data/example_blind.csv --predictions predictions_blind.csv
keplerbench evaluate-prediction --truth data/example_world.csv --predictions predictions.csv --data-blind-predictions predictions_blind.csv

# historical Mars sub-benchmark: fetch the source and rebuild the epoch table locally
keplerbench fetch-historical
keplerbench evaluate-historical --predictions my_declinations.csv

pytest
```

The harmonic baseline addresses only the first capability and is built to fail
outside its training window: its period is the training span, so its held-out
error on the default world is several thousand arcminutes. It is a floor, not a
reference.

## Three capabilities

1. **Prediction**: predict held-out observations of the same system.
2. **Global law**: freeze the learned shared structure and transfer it to an
   unseen system from `k` new observations; the error-versus-`k` curve is the
   result and its area against log `k` the summary.
3. **Local dynamical law**: construct a fixed-dimensional state and a reusable
   transition rule that supports long rollouts under new conditions.

The form of a submitted model is unrestricted. Symbolic, neural, programmatic
and hybrid systems are scored through their empirical scope, not by whether
they reproduce a preferred equation. The contracts, the data-blind control and
the noise-floor rule are in [`docs/protocol.md`](docs/protocol.md), which also
states what the software automates today.

## Three sub-benchmarks

- **Historical Mars**: Tycho Brahe's Mars declinations, 354 nightly epochs from
  1582 to 1600, split chronologically at 1594. Rebuilt locally by
  `keplerbench fetch-historical` and scored by `keplerbench evaluate-historical`;
  see [`docs/historical.md`](docs/historical.md).
- **Historical multi-planet**: planned; it will add the evidence across the
  known planets needed to test transfer and cross-system regularities.
- **Synthetic worlds**: central-force systems observed from a moving observer,
  exposing the same observation contract, drawn from named families by seed;
  see [`docs/synthetic.md`](docs/synthetic.md).

## The pilot

The paper's pilot compares seven predictors on the historical Mars
sub-benchmark: an oracle (modern orbits of Mars and Earth, no fitting), a
Keplerian ellipse fitted to the training years, Ptolemaic epicycles, a
quasi-periodic Gaussian process, a sine-wave (frequency-lattice ridge) fit,
symbolic regression with AI Feynman, and the training-mean constant. The
models that fit the training years best predict the later years worst.
Everything behind Table 2 is in [`experiments/pilot/`](experiments/pilot/README.md):
the fitting code, the model-selection details, the bootstrap, the residual
analysis and an acceptance suite that pins every printed number. The AI Feynman
runs, their raw frontiers and their scoring are in
[`experiments/aifeynman/`](experiments/aifeynman/README.md).

## Reproducing the paper's numbers

```bash
python -m pip install -e '.[dev,historical,pilot]'
keplerbench fetch-historical                 # source into ~/.cache/keplerbench, table and sidecar rebuilt
python experiments/pilot/run_pilot.py        # every fit, about ten minutes on one CPU
python experiments/pilot/make_table.py       # Table 2, the campaign and horizon tables, the quoted facts
pytest                                       # the Table 2 contract runs once the results exist
```

The AI Feynman row needs its own environment; the recipe and the commands are
in [`experiments/aifeynman/README.md`](experiments/aifeynman/README.md).

## Observation contract

Each row of a synthetic world contains

```text
world_id, system_id, observer_id, target_id, time,
planet_dir_x, planet_dir_y, planet_dir_z,
star_dir_x, star_dir_y, star_dir_z, split
```

Direction triples are unit vectors in a fixed celestial frame. The public input
omits latent Cartesian states and physical parameters, which are written to a
sidecar metadata file for auditability only. The machine-readable schema is
[`data/schemas/observation.schema.json`](data/schemas/observation.schema.json).
The historical epoch table has the columns `epoch_index, jd, year,
declination_deg, n_repeats, split`, and a prediction file for it has
`epoch_index, declination_deg` and optionally `uncertainty_deg`.

## Repository layout

```text
src/keplerbench/            package and command-line interface
src/keplerbench/historical  fetch-and-reduce route and the historical scorer
src/keplerbench/blind.py    data-blind copies of a dataset
data/historical/            committed split of the Mars epochs (no declinations)
data/schemas/               machine-readable public data contract
docs/protocol.md            three capabilities, data-blind control, noise floor, what is automated
docs/historical.md          historical Mars: source, rights, reduction, split, provenance plan
docs/synthetic.md           synthetic worlds: generator, families with their ranges
experiments/pilot/          Table 2: fitting code, results summary, acceptance suite
experiments/aifeynman/      the AI Feynman controls: scripts, raw frontiers, scoring
examples/                   example submission metadata
tests/                      package tests
```

## Status

Version 0.2.0 automates the prediction test on both released sub-benchmarks,
the data-blind control, the noise-floor rule, the synthetic generator with two
families, and the historical fetch-and-reduce route. The transfer harness, the
rollout validator, evaluator-private world suites and the multi-planet
historical release are the next steps, as the paper's conclusion says.

## Reproducibility

Generated datasets record their seed and world parameters in a sidecar
metadata file, and the generator is deterministic for a fixed seed. The
historical epoch table is rebuilt from its source with a reported checksum.
Every pilot fit is seeded and the results file records versions, seeds and wall
time. The package makes no network call except the explicit
`fetch-historical` download.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Author details
are withheld during double-blind review.

## License

The code and documentation are released under the Apache License 2.0. AI
Feynman is used under its MIT license. Tycho's observations are public domain;
Pafko's transcription of them carries a copyright notice without a license and
is fetched from its source rather than redistributed, and no committed file
carries declination values derived from it.
