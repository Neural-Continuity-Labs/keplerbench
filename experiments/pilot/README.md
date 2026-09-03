# The historical Mars pilot (Table 2 of the paper)

The pilot asks a narrower question than discovery: whether fitting Tycho's
Mars declinations well says anything about predicting them in later years.
Seven predictors are fitted to the 268 epochs before 1594 and scored on the 86
epochs of the 1594, 1595, 1596 and 1600 campaigns. This directory holds the
code that produced every number in Table 2 and in the pilot paragraphs of the
paper, the acceptance tests that pin those numbers, and the details of how each
model was fitted and selected.

## Reproducing

```bash
pip install -e '.[dev,historical,pilot]'
keplerbench fetch-historical                 # rebuilds data/historical/mars_declinations.csv and its sidecar
python experiments/pilot/run_pilot.py        # every fit, about ten minutes on one CPU, writes results.json
python experiments/pilot/make_table.py       # Table 2, the campaign and horizon tables, the quoted facts
python experiments/pilot/make_figure.py      # optional, needs the figures extra
pytest experiments/pilot/tests               # acceptance suite, including the Table 2 contract
```

`results.json` is not committed because three of its blocks are per-epoch
arrays from which declination values derived from Pafko's transcription could
be reconstructed. `results_summary.json`, the same file without those arrays,
and `TABLE2.md`, the regenerated table, are committed so the numbers can be
read without running anything. The AI Feynman row lives in
[`../aifeynman/`](../aifeynman/README.md).

## The seven rows

| Row in Table 2 | Key in `results.json` | What is fitted | Objective |
|---|---|---|---|
| Oracle (true orbit, no fitting) | `robust_huber.models.modern_unfitted` | nothing; JPL mean elements of Mars at the reference epoch, Earth from the Meeus solar theory | none |
| Ellipse fitted to Tycho (Keplerian) | `robust_huber.models.kepler6` | six orbital elements of Mars; Earth's orbit fixed | Huber loss, scale 5 arcmin, training epochs |
| Ptolemaic epicycles (25 terms) | `robust_huber.models.epicycle_K25_saturated` | 25 coefficients of a truncated Fourier series on the sidereal and synodic frequency lattice | Huber loss, scale 5 arcmin, training epochs |
| Flexible curve fit (Gaussian process) | `robust_huber.models.gp_quasiperiodic` | seven kernel hyperparameters of a quasi-periodic Gaussian process | marginal likelihood on the training epochs, robustified as described below |
| Sine-wave fit (ridge) | `robust_huber.models.frequency_lattice_ridge_control` | 145 coefficients on the full frequency lattice | ridge regression, penalty chosen on the last fifth of the training window |
| Symbolic regression (AI Feynman) | `../aifeynman/runs/phases_20s/scored_phases.json` | an expression in the sidereal and synodic phases | AI Feynman's own Pareto criterion; lowest training error reported |
| Constant (training mean) | `robust_huber.models.control_mean` | one location | Huber location estimate of the training declinations |

The rows the paper omits are in the same file: the seven-parameter Keplerian
fit with a free mean motion (`kepler7_free_n`), the epicycle model at the
cross-validated order (`epicycle_cv_selected`), a pure-periodic Gaussian
process without a time envelope (`gp_pure_periodic_control`), and, under
`models`, every family refitted by ordinary least squares. The least-squares
arm is an ablation, not a result, because 99 percent of the training sum of
squares comes from the 15 training epochs with gross transcription errors.

## Model selection and fitting details

**Data.** The table rebuilt by `keplerbench fetch-historical`, read by
`pilot_data.py`. The reference epoch `jd_ref` is the median training Julian
day. Time enters every model in days from `jd_ref`. The choice of Pafko's day
count as the time convention is decided by a table of the unfitted oracle's
residual under shifts of -10, -1, 0, +1 and +10 days
(`data.shift_decision_table`); zero shift wins by a factor above thirteen.

**Loss.** Every fitted row in Table 2 uses the Huber loss with a scale of 5
arcmin, just above Tycho's repeat scatter of 2 to 4 arcmin, so honest
observations enter quadratically and gross errors linearly. Nothing is deleted
for the headline. Linear models are fitted by iteratively reweighted least
squares, the Keplerian model passes the loss to `scipy.optimize.least_squares`,
and the Gaussian process is robustified as described below.

**Oracle.** JPL approximate mean elements of Mars (values at J2000 and rates
per Julian century) evaluated at `jd_ref`, Earth from the Meeus low-precision
solar theory, both referred to the equatorial J2000 frame and precessed to the
equator of date, light time included, geocentric (the neglected diurnal
parallax from Hven is below 0.39 arcmin). Never fitted.

**Keplerian ellipse.** Elements a, e, i, node, argument of perihelion and mean
anomaly at `jd_ref`, with the mean motion from Kepler's third law; the
seven-parameter variant frees the mean motion. Bounds are wide (angles over
more than a revolution, so reported angular spreads are wrapped). The fit runs
`scipy.optimize.least_squares` with the trust-region reflective method from 61
starts: the modern elements plus 60 seeded random starts (seed 7). The modern
start is declared as prior knowledge; a blind run of 60 random starts without
it (seed 21, `knowledge_budget.ablation.huber`) reaches the same optimum, so
the result does not depend on it. The standard errors and the correlation
matrix come from the Huber-weighted Gauss-Newton normal matrix, and the
strongest correlation, between a and the mean anomaly at 0.983, is the
identifiability evidence quoted in the paper.

**Epicycles.** The frequency lattice is every combination n1 f_sid + n2 f_syn
with |n1| + |n2| at most 8, a positive frequency and a period of at least 25
days, duplicates removed, from the sidereal period 686.9800 d and the synodic
period 779.9361 d. Their sum is the annual frequency, so the lattice is the
deferent-plus-epicycle family and its harmonics. Frequencies are ordered by
greedy forward selection on the training window under the Huber cost, and a
model with K terms uses the constant plus the first (K - 1) / 2 frequencies. K
is chosen inside the training span only, by forward chaining (fit before a
cut, validate on the next 20 percent of the training span, five cuts from 50
to 90 percent) and by blocked five-fold cross-validation, with the median
absolute validation residual as the criterion. The two selectors disagree,
K = 3 forward against K = 19 blocked (`robust_huber.epicycle_cv`). Table 2
reports the saturated K = 25 because it is the member of the family that fits
the training years best, which is the question the pilot asks; the
cross-validated K = 3 gives 193.19 arcmin in sample and 622.18 held out, and
K = 19 gives 7.42 and 655.30, so the held-out failure does not depend on the
selector.

**Gaussian process.** Kernel: the sum of two quasi-periodic components, one at
the sidereal and one at the synodic period, each an amplitude times a periodic
exponential-sine-squared factor times a squared-exponential time envelope,
plus white noise, seven hyperparameters in all (`hyperparams` in the entry).
Type II maximum likelihood with eight seeded starts (seed 20260829), each
optimized by L-BFGS-B and then Powell, the best marginal likelihood kept
(negative log likelihood -1263.64). Robustification: because an interpolating
process cannot see an outlier in its own residual, the closed-form
leave-one-out residuals drive an IRLS loop on the per-point noise with Huber
weights at 5 arcmin. Capacity is reported as the effective degrees of freedom
trace(K (K + S)^-1) on the training epochs, 122.0.

**Ridge.** Design matrix of the full lattice, 145 coefficients. The penalty is
chosen from 40 log-spaced values between 1e-6 and 1e8 by the median absolute
residual on the last fifth of the training window, then the model is refitted
on all training epochs. This and the pure-periodic process are training-only
extrapolation controls rather than matched Huber fits, and are labelled so in
the results file.

**Constant.** The Huber location of the training declinations; its
least-squares twin is the plain mean.

## Statistics

The statistic is the median absolute residual in arcmin. Intervals are
percentile bootstraps of the median over epochs within each split, 10,000
resamples, seed 20260829, with the generator seeded inside each call so an
interval does not depend on what was computed before it. A median is quoted
only on at least ten epochs; below that the sorted absolute residuals are
listed, which is why the 1594 and 1600 campaigns and the third horizon bucket
appear as lists. Held-out horizons are measured from the last training epoch
in synodic periods (0 to 780, 780 to 1560, 1560 to 2340 days; n = 69, 10, 7).
Paired comparisons report the Hodges-Lehmann shift of the paired absolute
residuals with a paired bootstrap interval and a two-sided sign test
(`headline.paired_tests`). Fitting variability is not in the intervals: every
row is one fitted run.

The AI Feynman row was scored with the same statistic but with 2,000 resamples
and seed 23; under this file's procedure its intervals would be [532.14,
602.41] in sample and [2690.04, 2698.09] held out. The synodic period given to
AI Feynman, 779.94 d, differs from the lattice value in the fourth decimal.

## Structure in Tycho's residuals

These are the facts the paper refers to as documented by the release, with the
key that holds each of them.

- Gross errors: 18 of 354 epochs differ from the oracle by more than 60
  arcmin, 15 of them in training and 9 of them sign flips
  (`data.gross_error_census`). A training-window smoothness detector flags
  11 epochs with precision 0.909 and recall 0.556 (`data.outlier_detector`);
  it is a census and a robustness check, never applied to the headline.
  Deleting the flagged epochs moves the Keplerian fit from 2.66 and 13.31
  to 2.57 and 16.10 arcmin (`deletion_robustness_check`).
- Altitude: transit altitudes run from 6.1 to 61.4 degrees, and the residual
  against the oracle has Spearman rho -0.189 with altitude, p = 4.8e-4
  (`refraction_analysis`). The median signed residual shifts by 3.30 arcmin
  between the two windows; a robust linear altitude term fitted on the
  training window reproduces 0.52 arcmin of that shift, and correcting every
  declination with it and refitting changes the held-out Keplerian median
  from 13.31 to 12.38 arcmin.
- Campaigns: the held-out window is three campaigns and a tail (5, 54, 20, 7
  epochs), and the aggregate Keplerian comparison is driven by the 1595
  campaign, 17.40 arcmin against the oracle's 3.56 (`headline.by_campaign`).
- Known-truth injection: declinations generated from the modern orbit on
  Tycho's own epochs are recovered exactly; adding the fitted altitude term,
  the observed gross-error vector and 2 arcmin Gaussian noise leaves the
  held-out error against truth below 2 arcmin, and only injecting the full
  observed residual vector reproduces the real 13.31 arcmin
  (`synthetic_observation_physics`). The degradation therefore lies in an
  unmodelled component of the residual pattern, not in identifiability.
- Degeneracy: with declination alone, a and the mean anomaly at epoch
  correlate at 0.983 in the fitted covariance (`robust_huber.models.kepler6.degeneracy`).

## Runtime, environment and reproduction precision

The paper's numbers were produced in 612 s on one CPU with Python 3.14.7,
NumPy 2.4.2 and SciPy 1.17.1; the provenance block of the results file records
the versions, seeds and wall time of any run.

Every fit is seeded, and a rerun reproduces every printed digit of Table 2
except on the Gaussian-process row, whose held-out figures come out as
1173.94 [1150.27, 1185.87] against the printed 1173.95 [1150.28, 1185.88].
The unmodified private pipeline rerun on the same machine with the same
library versions gives 1173.94 [1150.26, 1185.85], so the wobble is not in
this port: it is floating-point nondeterminism in the optimizers (the
marginal-likelihood search and the trust-region fits both go through LAPACK)
acting on an extrapolated quantity that sits three orders of magnitude above
the noise floor, where a relative change of 1e-5 is a hundredth of an arcmin.
Every other row, every interval, and every quoted fact reproduce to the
printed precision. The contract test therefore allows 2e-5 relative on top of
half a unit in the last printed decimal. The suite in `tests/` pins the
calendar, the ephemeris, the frames and units, the split, the outlier census,
the synthetic known-truth recovery of the fitter, and every number of Table 2.
Tests that read Pafko's spreadsheet use the local copy fetched by
`keplerbench fetch-historical` and skip without it.

## Files

| File | What |
|---|---|
| `pilot_data.py` | loader for the rebuilt table plus the training-window gross-error detector |
| `ephem.py` | calendar, solar theory, precession, Kepler's equation, the modern elements |
| `models.py` | the model families and their fitters |
| `run_robust.py` | the Huber machinery |
| `run_pilot.py` | the single entry point; writes `results.json` |
| `make_table.py` | Table 2 and the other tables from the results; writes `TABLE2.md` and `results_summary.json` |
| `make_figure.py` | figures from `results.json` |
| `tests/` | the acceptance suite; thresholds in `tests/constants.py` |
