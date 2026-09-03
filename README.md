# KeplerBench

KeplerBench is a benchmark for scientific discovery from observer-centred
astronomical measurements. A learner receives times and apparent directions to
a planet and star. It is not given heliocentric positions, velocities, orbital
elements, or the representation in which the governing law is simple.

This repository accompanies an anonymous workshop submission. The benchmark is
under active development; the current release fixes the data contract,
implements the synthetic evidence setting, and provides a predictive baseline
and evaluation utilities. The provenance-tracked historical release is planned
but is not yet included.

## What is evaluated

KeplerBench separates three capabilities:

1. **Prediction** — predict held-out observations from the same system.
2. **Global-law discovery** — transfer learned structure to unseen physical
   systems with little new evidence.
3. **Local dynamical-law discovery** — construct a fixed-dimensional state and
   a reusable time-local rule that supports long rollouts under new conditions.

The internal form of a submitted model is unrestricted. Symbolic, neural,
programmatic, and hybrid systems are evaluated through their empirical scope,
not by whether they reproduce a preferred equation.

## Quick start

KeplerBench requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

keplerbench generate --output data/example_world.csv --seed 7
keplerbench baseline --data data/example_world.csv --predictions predictions.csv
keplerbench evaluate-prediction --truth data/example_world.csv --predictions predictions.csv
pytest
```

The baseline deliberately addresses only the first capability. It fits apparent
directions with harmonic features and should not be interpreted as a discovered
physical law.

## Observation contract

Each row contains

```text
world_id, system_id, observer_id, target_id, time,
planet_dir_x, planet_dir_y, planet_dir_z,
star_dir_x, star_dir_y, star_dir_z, split
```

Direction triples are unit vectors in a fixed celestial frame. The public input
omits latent Cartesian states and physical parameters. A machine-readable schema
is available at [`data/schemas/observation.schema.json`](data/schemas/observation.schema.json).

## Evidence settings

- **Historical Mars** will reconstruct the evidence relevant to Kepler's first
  two laws, with source-level provenance.
- **Historical multi-planet** will add the evidence needed to test transfer and
  cross-system regularities.
- **Synthetic worlds** expose the same observer-centred interface while varying
  initial conditions, observer motion, system parameters, and central-force
  laws.

Only the synthetic setting is implemented in this initial repository. It is
designed to prevent a submission from receiving privileged state variables
through the benchmark interface.

## Repository layout

```text
src/keplerbench/       Python package and command-line interface
data/schemas/          Machine-readable public data contracts
docs/protocol.md       Evaluation protocol and submission contract
docs/historical.md     Historical-data provenance plan
examples/              Example submission metadata
tests/                 Unit and end-to-end tests
```

## Reproducibility and status

Generated datasets record their seed and world parameters in a sidecar metadata
file. The package contains no hidden network calls, and the example generator is
deterministic for a fixed seed.

This is a specification-first preview, not a claim that all benchmark tracks or
reference baselines are complete. See [`ROADMAP.md`](ROADMAP.md) for the release
boundary and remaining work.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Author details
are intentionally withheld during double-blind review.

## License

The code and documentation are released under the Apache License 2.0. Historical
source material will retain its own provenance and rights metadata and may be
distributed under different terms.
