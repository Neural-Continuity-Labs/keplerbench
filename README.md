# KeplerBench

Artificial intelligence systems are increasingly expected to make scientific discoveries, and general relativity has been proposed as the test of whether they can take a conceptual leap of the highest order. KeplerBench argues for an earlier and simpler test that current systems have not passed. Kepler replaced two thousand years of circular astronomy with an ellipse and the Sun at one focus, added the equal-area law and the relation between orbital period and orbital scale, and gave Newton the results from which the inverse-square law follows. He worked from Tycho Brahe's sparse and noisy records of where the planets appeared in the sky, and the reasoning that took him from those records to the laws fills *Astronomia Nova*. KeplerBench gives a learner observations of that kind and asks whether it can propose a law as general as Kepler's, in whatever form, without being told the variables in which the law is simple. Simulated star systems with different masses, orbits and force laws then test whether the discovery procedure generalizes beyond the Solar System.

This repository is the software and data release that accompanies the paper, which is under anonymous review. It contains the benchmark, namely the observation contract, the synthetic generator with its world families, the route that rebuilds Tycho's Mars record locally, the scorers and the data-blind control. It also contains the evidence the paper reports, namely the pilot on Tycho's Mars declinations and the AI Feynman controls. The sections below follow the paper. We state the observational problem, the three capabilities the benchmark scores and the three sub-benchmarks it scores them on, the controls on prior knowledge and noise, and the pilot and what it showed. We then explain how to use the software, how to reproduce every number in the paper, and where each statement of the paper is delivered.

## Kepler's observational problem

Modern planetary mechanics begins with heliocentric position, velocity, orbital radius and orbital elements, none of which Kepler had. Tycho observed the planets from a moving Earth and recorded principally their apparent positions on the celestial sphere at particular times. KeplerBench therefore represents an observation as a time, the apparent direction of the planet, the apparent direction of the Sun or of a reference star, and metadata such as the instrument used and an error estimate. Nothing else reaches the learner. Heliocentric positions and velocities, the Sun-planet distance, the eccentricity, the semi-major axis, the orbital phase, and any explicit assumption that a simple heliocentric state exists are withheld, because supplying them would already perform part of the discovery. The apparent motion is complicated, with retrograde loops as the standard example, so a flexible model may predict it accurately without recovering the simpler structure beneath it. The intermediate representation is left open, and heliocentric coordinates, an equivalent coordinate system, or an opaque neural state are all admissible, judged only by the generality and the dynamics they support.

A benchmark built on a famous historical case cannot certify what a pretrained system already knows. Systems that claim historical rediscovery must therefore document and control their prior knowledge, and the benchmark enforces one such control in software rather than by self-report. Every scored entry is also run on a data-blind copy of its input, described below, and earns no credit when that run matches its score.

## Three capabilities

| Capability | Minimal test | Evaluation restriction |
|---|---|---|
| Prediction | Predict held-out observations of the same system. | Standard held-out evaluation. |
| Global law | Freeze the shared structure and transfer it to an unseen system from `k` new observations. | The shared law is not relearned on the target. |
| Local law | Infer a fixed-dimensional state, remove the history, and apply one transition rule repeatedly under unseen conditions. | No history and no separate trajectory predictor after initialization. |

The first capability is prediction of unseen observations of the same system, and Ptolemaic astronomy is its historical reference, since a model may predict planetary appearances well without describing the motion in Keplerian or Newtonian terms. A Fourier expansion, an epicyclic construction or a high-capacity sequence model can pass this test, and passing it establishes predictive adequacy only. The basic observable metric is the great-circle angle between the predicted and the observed direction.

The second capability is a global law, whose structure transfers across trajectories or systems. A discovery procedure first produces shared structure from one or more systems, and that structure is then frozen. Given `k` observations from a new system, the learner may infer system-specific quantities but may not relearn the shared structure, and the prediction error on the new system after adaptation from `k` observations is reported as a function of `k`. The complete curve is the primary result and its area against log `k` is the summary, averaged over held-out systems and seeds, so that generality and sample efficiency are measured together. An unconstrained fit may eventually learn a new planet, and a law should constrain it from much less evidence.

The third capability is a local dynamical law, which evolves a present state through a reusable rule. After the state is initialized from an observation window, the history is removed and the same rule must evolve the state repeatedly. Autonomous rollouts under new initial conditions and systems are scored by the angular error at fixed horizons and by the time to first exceed an error threshold, and initializing the same physical instant from different windows should lead to the same predicted future. For a single body, an ellipse with the area law already implies an inverse-square central force, so the universality of a local law across bodies is testable only where several systems are available.

## Three sub-benchmarks

The capabilities are one dimension of the benchmark and the evidence is another. Each sub-benchmark fixes which observations a learner receives and which are held out, and a capability can be tested on any sub-benchmark that supplies the evidence it needs.

The **historical Mars** sub-benchmark reconstructs, as closely as the record permits, the evidence available during Kepler's Mars investigation. Its current release is Tycho Brahe's Mars declinations, 923 readings between 1582 and 1600 as transcribed by Pafko from Dreyer's edition of the observation books and reduced to 354 nightly epochs. The 268 epochs before 1594 form the training years, and the 86 epochs of the 1594, 1595, 1596 and 1600 campaigns are held out, so held-out prediction is temporal extrapolation. The record carries declination only, one coordinate of the apparent direction, and the sub-benchmark therefore supports the prediction capability under extrapolation and supplies the discovery evidence for laws comparable to Kepler's first two. The transcription carries a copyright notice without a license, so the repository does not redistribute it. The command `keplerbench fetch-historical` downloads it from its source and rebuilds the epoch table locally with the paper's reduction, and the split is committed without declinations so that it can be checked without the source file. The reduction, the rights status, the known structure in the residuals and the provenance plan for the full historical release are in [`docs/historical.md`](docs/historical.md).

The **historical multi-planet** sub-benchmark is planned. It will add evidence across the known planets, so that a relation inferred from Mars can be tested on other planets and a cross-system relation such as the third law can be inferred from a subset of planets and tested on a held-out one.

The **synthetic worlds** sub-benchmark exists because historical data are scientifically meaningful but vulnerable to memorization and provide only one Solar System. It generates new star systems while preserving the same observer-centered observation format. Each world is a planar central-force system in which an observer and a target body move independently about a central mass. Each observation is a time with the unit directions from the observer to the target and to the central body, with optional angular noise. A world is drawn from a named family by a seed. The first family obeys inverse-square dynamics and varies the central mass, the initial conditions, the orbital geometry, the observer's orbit, the sampling and the noise. A second family also hides the exponent of the force law, so a candidate model built on Solar System familiarity is misspecified relative to it, and a learner that recovers the hidden force has detected the departure from observations alone. The generator, the observation contract and the parameter ranges of both families are in [`docs/synthetic.md`](docs/synthetic.md).

## Controls on prior knowledge and noise

The data-blind control is applied to every scored entry. The evaluator runs the same system a second time on a copy of the dataset in which the times, identifiers and split labels are kept and the observed quantities are permuted across the epochs of each split. The permutation keeps the marginal distribution of the observations and destroys their relation to time, so a system whose score is carried by prior knowledge scores the same on both inputs, whereas a system that extracts structure from the evidence does not. Both runs are scored against the true held-out observations and both scores are reported, and the entry earns credit only when the data-blind run is worse than the entry by more than the sub-benchmark's noise floor.

The noise floor is fixed per sub-benchmark from its measurement process, 2 arcmin for Tycho's record and the configured angular noise for a synthetic world. Entries state a predictive uncertainty for every prediction, the historical scorer reports the fraction of residuals inside two stated uncertainties, and residuals below the floor are not ranked against one another. The full protocol, including the contracts for the global-law and local-law tests and a statement of what the software automates today, is in [`docs/protocol.md`](docs/protocol.md).

## The pilot on Tycho's Mars record

The pilot asks a narrower question than discovery, namely whether fitting Tycho's record well says anything about predicting it in later years. We fitted seven predictors to the training years and scored them on the held-out campaigns. The oracle uses the modern orbits of Mars and Earth without fitting, so its residual is Tycho's measurement error and the best any model can reach. The fitted ellipse is Kepler's form of the law with six parameters fitted to the training years, and it assumes a heliocentric orbit and the modern orbit of Earth, so it is a physics-supplied reference rather than a discovery method. The remaining rows carry no physics. They are Ptolemaic epicycles, a flexible curve fit by a Gaussian process, a sine-wave fit by ridge regression, symbolic regression with AI Feynman given the same sidereal and synodic periods as the epicycles and the sine-wave fit, and a constant. Every fitted row uses the Huber loss with a scale of 5 arcmin on the training epochs, and every number carries an epoch-bootstrap 95 percent interval.

| Model | Type | Eff. dof | In sample (arcmin) | Held out (arcmin) |
|---|---|---:|---:|---:|
| Oracle (true orbit, no fitting) | reference | 0 | 3.95 [3.17, 4.53] | 3.48 [2.74, 3.75] |
| Ellipse fitted to Tycho (Keplerian) | physical law | 6 | 2.66 [2.33, 3.07] | 13.31 [12.12, 17.40] |
| Ptolemaic epicycles (25 terms) | flexible fit | 25 | 4.02 [3.36, 4.79] | 731.19 [382.42, 924.58] |
| Flexible curve fit (Gaussian process) | flexible fit | 122 | 0.72 [0.59, 0.89] | 1173.95 [1150.28, 1185.88] |
| Sine-wave fit (ridge) | control | 145 | 4.80 [3.97, 5.52] | 1266.51 [1223.75, 1279.94] |
| Symbolic regression (AI Feynman) | control | -- | 580.87 [532.25, 601.79] | 2694.14 [2690.12, 2698.09] |
| Constant (training mean) | control | 1 | 714.09 [621.93, 761.32] | 966.14 [959.85, 990.74] |

The models that fit the training years best predict the later years worst. The curve fit reaches 0.72 arcmin in sample and misses the later campaigns by three orders of magnitude, the epicycles and the sine-wave fit miss by hundreds of arcmin, and symbolic regression fits the training years worse than the constant and extrapolates worse still. The fitted ellipse degrades from 2.66 to 13.31 arcmin and is the only fitted model that still tracks the planet. In-sample fit is therefore not evidence of law discovery, and the pilot does not itself establish that a law has been discovered, since the ellipse was supplied. The ten-arcmin gap between the fitted ellipse and the oracle measures what remains beyond writing down an ellipse. The fit absorbs structure in Tycho's residuals, an altitude-dependent bias consistent with refraction, gross transcription errors and campaign-to-campaign offsets, and declination alone leaves the orbit scale and the phase nearly interchangeable. These facts are the reason the benchmark fixes a noise floor and asks entries to state their uncertainty. The fitting code, the model-selection details of every row, the residual analysis and an acceptance suite that pins every printed number are in [`experiments/pilot/`](experiments/pilot/README.md), and the AI Feynman runs with their raw frontiers and scoring are in [`experiments/aifeynman/`](experiments/aifeynman/README.md).

## Using the software

KeplerBench requires Python 3.10 or newer. The package depends on NumPy alone, and the optional extras add pandas and xlrd for reading Pafko's spreadsheet, SciPy for the pilot, Matplotlib for its figures, and pytest for the tests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,historical,pilot]'
```

The synthetic sub-benchmark is driven by four commands. `generate` writes a world as a CSV file of public observations together with a sidecar of latent parameters, and `baseline` fits the harmonic predictor of the first capability. `evaluate-prediction` scores predicted test-set directions by great-circle angular error, and `data-blind` writes the permuted copy of a dataset for the control.

```bash
keplerbench generate --output data/example_world.csv --seed 7
keplerbench generate --output data/world_a.csv --family inverse-square --seed 3
keplerbench generate --output data/world_b.csv --family hidden-exponent --seed 3 --noise 0.0006
keplerbench baseline --data data/example_world.csv --predictions predictions.csv
keplerbench data-blind --input data/example_world.csv --output data/example_blind.csv --seed 0
keplerbench baseline --data data/example_blind.csv --predictions predictions_blind.csv
keplerbench evaluate-prediction --truth data/example_world.csv --predictions predictions.csv --data-blind-predictions predictions_blind.csv
```

The harmonic baseline addresses only the first capability and is built to fail outside its training window. Its period is the training span, so its held-out error on the default world is several thousand arcminutes, and it is a floor rather than a reference.

The historical sub-benchmark is driven by two commands. `fetch-historical` downloads the spreadsheet into a cache directory, reports its checksum, and rebuilds the epoch table with a sidecar of reduction counts, and `evaluate-historical` scores a prediction file on one split.

```bash
keplerbench fetch-historical
keplerbench evaluate-historical --predictions my_declinations.csv
keplerbench data-blind --input data/historical/mars_declinations.csv --output data/historical/mars_blind.csv --seed 0
keplerbench evaluate-historical --predictions my_declinations.csv --data-blind-predictions my_declinations_blind.csv
```

An entry on the historical sub-benchmark submits its predicted declinations at the held-out epochs as a CSV file with the columns `epoch_index`, `declination_deg` and, optionally, `uncertainty_deg`, together with the predictions of the same system run on the data-blind copy of the table. The scorer reports the median absolute residual in arcmin with its bootstrap interval, the breakdown by campaign, the coverage of the stated uncertainty, and the verdict of the data-blind rule. An entry on the synthetic sub-benchmark submits predicted directions at the test times as a CSV file with the columns `time`, `planet_dir_x`, `planet_dir_y` and `planet_dir_z`, together with its data-blind counterpart. The metadata an entry declares, its capability, its sub-benchmark, its pretraining, its external data and its prior knowledge, follow [`examples/submission.json`](examples/submission.json).

## Reproducing the paper

```bash
python -m pip install -e '.[dev,historical,pilot]'
keplerbench fetch-historical
python experiments/pilot/run_pilot.py
python experiments/pilot/make_table.py
pytest
```

The first command installs the package with the extras the pilot needs, and the second rebuilds the epoch table the pilot reads. The pilot then runs every fit in about ten minutes on one CPU and writes `results.json`. From that file, `make_table.py` prints Table 2 of the paper, the campaign and horizon tables and the facts quoted in the text, and writes `TABLE2.md` together with a results summary without per-epoch arrays. The test suite includes a contract that asserts every printed number of Table 2 against the results. A rerun reproduces every printed digit except on the Gaussian-process row, where the held-out figures move by a hundredth of an arcmin between runs because of floating-point nondeterminism in the optimizers, as documented in [`experiments/pilot/README.md`](experiments/pilot/README.md). The AI Feynman row needs its own environment because AI Feynman builds against `numpy.distutils`, and the recipe, the runs and their scoring are in [`experiments/aifeynman/README.md`](experiments/aifeynman/README.md).

## Where each statement of the paper lives

| The paper says | Delivered by |
|---|---|
| Code, data and documentation are in the repository (footnote 1) | this repository, mapped by this section |
| The data-blind control is enforced in software (Section 2) | `keplerbench data-blind`, the `--data-blind-predictions` option of both scorers, and the rule in [`docs/protocol.md`](docs/protocol.md) |
| Tycho's Mars declinations, 354 nightly epochs, 268 before 1594 for training and 86 held out (Section 4) | `keplerbench fetch-historical` rebuilds the table from Pafko's transcription, the split is committed as [`data/historical/mars_split.csv`](data/historical/mars_split.csv), and the reduction and rights are in [`docs/historical.md`](docs/historical.md) |
| Seven predictors, their objectives, the AI Feynman settings, the selected expression and the bootstrap (Table 2, footnote 2) | [`experiments/pilot/`](experiments/pilot/README.md) and [`experiments/aifeynman/`](experiments/aifeynman/README.md), with the regenerated table in [`experiments/pilot/TABLE2.md`](experiments/pilot/TABLE2.md) |
| The fit absorbs structure in Tycho's residuals that the release documents (Section 4) | the residual analysis in [`experiments/pilot/README.md`](experiments/pilot/README.md), each fact with its key in the results file |
| A noise floor per sub-benchmark, 2 arcmin for Tycho, and stated uncertainty (Section 4) | [`docs/protocol.md`](docs/protocol.md), and the `uncertainty_deg` column read by `evaluate-historical` |
| The first synthetic family varies masses, initial conditions, geometry, observer, sampling and noise under inverse-square dynamics, and harder families hide the force exponent (Section 4) | `keplerbench.synthetic.FAMILIES`, with the ranges tabulated in [`docs/synthetic.md`](docs/synthetic.md) |
| Data, split files and fitting code, with model-selection details (checklist items 4 and 6) | [`experiments/pilot/`](experiments/pilot/README.md) |
| Epoch-bootstrap intervals for every number (checklist item 7) | `experiments/pilot/make_table.py` regenerates them and `experiments/pilot/tests/test_table2_contract.py` pins them |
| The complete pilot runs in 612 seconds on one CPU and the AI Feynman control in 12 minutes (checklist item 8) | the provenance block of the pilot results and `experiments/aifeynman/runs/phases_20s/report.json` |
| Pafko's transcription is credited and not redistributed, and AI Feynman is used under its MIT license (checklist item 12) | the license section below and [`docs/historical.md`](docs/historical.md) |

## Repository layout

```text
src/keplerbench/            package and command-line interface
src/keplerbench/historical  fetch-and-reduce route and the historical scorer
src/keplerbench/blind.py    data-blind copies of a dataset
data/historical/            committed split of the Mars epochs, without declinations
data/schemas/               machine-readable public data contract
docs/protocol.md            three capabilities, data-blind control, noise floor, what is automated
docs/historical.md          historical Mars: source, rights, reduction, split, provenance plan
docs/synthetic.md           synthetic worlds: generator, families with their ranges
experiments/pilot/          Table 2: fitting code, results summary, acceptance suite
experiments/aifeynman/      the AI Feynman controls: scripts, raw frontiers, scoring
examples/                   example submission metadata
tests/                      package tests
```

## Status and roadmap

Version 0.2.0 automates the prediction test on both released sub-benchmarks, the data-blind control, the noise-floor rule, the synthetic generator with two families, and the historical fetch-and-reduce route. The global-law and local-law contracts are specified and frozen for early system development. The transfer harness, the rollout validator, evaluator-private world suites and the multi-planet historical release are the next steps, as the paper's conclusion says. Beyond them, a curated pre-Kepler corpus could support experiments with controlled priors, and an active-observation sub-benchmark could test systems that choose which measurement to make next.

## Reproducibility and provenance

Generated datasets record their seed and their world parameters in a sidecar metadata file, and the generator is deterministic for a fixed seed. The historical epoch table is rebuilt from its source with a reported checksum, and its sidecar records the reduction counts. Every pilot fit is seeded, and the results file records the library versions, the seeds and the wall time of the run. The package makes no network call except the explicit `fetch-historical` download, and continuous integration runs the test suite on every push without fetching the source file.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff), with author details withheld during double-blind review. The code and documentation are released under the Apache License 2.0. AI Feynman is used under its MIT license. Tycho's observations are public domain, whereas Pafko's transcription of them carries a copyright notice without a license, so it is fetched from its source rather than redistributed, and no committed file carries declination values derived from it.
