# The historical Mars pilot (Table 2 of the paper)

The pilot asks a narrower question than discovery, namely whether fitting Tycho's Mars declinations well says anything about predicting them in later years. Seven predictors are fitted to the 268 epochs before 1594 and scored on the 86 epochs of the 1594, 1595, 1596 and 1600 campaigns. This directory holds the code that produced every number in Table 2 and in the pilot paragraphs of the paper, the acceptance tests that pin those numbers, and the details of how each model was fitted and selected. The AI Feynman row is produced in [`../aifeynman/`](../aifeynman/README.md) and is documented there.

## Reproducing

```bash
pip install -e '.[dev,historical,pilot]'
keplerbench fetch-historical
python experiments/pilot/run_pilot.py
python experiments/pilot/make_table.py
python experiments/pilot/make_figure.py
pytest experiments/pilot/tests
```

The first two commands install the package and rebuild `data/historical/mars_declinations.csv` with its sidecar. The pilot then runs every fit in about ten minutes on one CPU and writes `results.json`. From that file, `make_table.py` prints Table 2 with the campaign and horizon tables and the facts quoted in the paper, and `make_figure.py` draws the figures when the `figures` extra is installed. The acceptance suite then checks the run, including the Table 2 contract. `results.json` is not committed because three of its blocks are per-epoch arrays from which declination values derived from Pafko's transcription could be reconstructed. `results_summary.json`, the same file without those arrays, and `TABLE2.md`, the regenerated table, are committed so that the numbers can be read without running anything.

## The seven rows

| Row in Table 2 | Key in `results.json` | What is fitted | Objective |
|---|---|---|---|
| Oracle (true orbit, no fitting) | `robust_huber.models.modern_unfitted` | nothing, the modern mean elements of Mars at the reference epoch with Earth from the Meeus solar theory | none |
| Ellipse fitted to Tycho (Keplerian) | `robust_huber.models.kepler6` | six orbital elements of Mars, with Earth's orbit fixed | Huber loss, scale 5 arcmin, training epochs |
| Ptolemaic epicycles (25 terms) | `robust_huber.models.epicycle_K25_saturated` | 25 coefficients of a truncated Fourier series on the sidereal and synodic frequency lattice | Huber loss, scale 5 arcmin, training epochs |
| Flexible curve fit (Gaussian process) | `robust_huber.models.gp_quasiperiodic` | seven kernel hyperparameters of a quasi-periodic Gaussian process | marginal likelihood on the training epochs, robustified as described below |
| Sine-wave fit (ridge) | `robust_huber.models.frequency_lattice_ridge_control` | 145 coefficients on the full frequency lattice | ridge regression with the penalty chosen on the last fifth of the training window |
| Symbolic regression (AI Feynman) | `../aifeynman/runs/phases_20s/scored_phases.json` | an expression in the sidereal and synodic phases | AI Feynman's own Pareto criterion, with the lowest training error reported |
| Constant (training mean) | `robust_huber.models.control_mean` | one location | Huber location estimate of the training declinations |

The rows the paper omits are in the same file. They are the seven-parameter Keplerian fit with a free mean motion (`kepler7_free_n`), the epicycle model at the cross-validated order (`epicycle_cv_selected`), a pure-periodic Gaussian process without a time envelope (`gp_pure_periodic_control`), and, under `models`, every family refitted by ordinary least squares. The least-squares arm is an ablation rather than a result, because 99 percent of the training sum of squares comes from the 15 training epochs with gross transcription errors, so that objective mostly measures how far a model bends towards those epochs.

## How each model is fitted and selected

### Data and time convention

The pilot reads the table rebuilt by `keplerbench fetch-historical` through `pilot_data.py`. The reference epoch `jd_ref` is the median training Julian day, and time enters every model in days from it. Pafko's day count is taken as the time convention, and that choice is decided rather than assumed. The residual of the unfitted oracle is recomputed with every timestamp shifted by -10, -1, 0, +1 and +10 days (`data.shift_decision_table`), and the unshifted convention wins by more than a factor of thirteen.

### Loss

Every fitted row in Table 2 uses the Huber loss with a scale of 5 arcmin, just above Tycho's repeat scatter of 2 to 4 arcmin, so honest observations enter quadratically and gross errors linearly. Nothing is deleted for the headline. Linear models are fitted by iteratively reweighted least squares (IRLS), the Keplerian model passes the loss to `scipy.optimize.least_squares`, and the Gaussian process is robustified as described in its own paragraph.

### Oracle

The oracle evaluates the approximate mean elements of Mars published by the Jet Propulsion Laboratory, given as values at J2000 with rates per Julian century, at `jd_ref`. Earth comes from the low-precision solar theory of Meeus. Both are referred to the equatorial J2000 frame and precessed to the equator of date, light time is included, and the model is geocentric, since the neglected diurnal parallax from Hven stays below 0.39 arcmin. The oracle is never fitted.

### Keplerian ellipse

The elements are the semi-major axis, the eccentricity, the inclination, the longitude of the node, the argument of perihelion and the mean anomaly at `jd_ref`, with the mean motion following Kepler's third law. A seven-parameter variant frees the mean motion. The bounds are wide, and because the angular bounds span more than a revolution, angular spreads across restarts are wrapped before they are reported. The fit runs `scipy.optimize.least_squares` with the trust-region reflective method from 61 starts, namely the modern elements and 60 seeded random starts (seed 7). The modern start is declared as prior knowledge, and a blind run of 60 random starts without it (seed 21, `knowledge_budget.ablation.huber`) reaches the same optimum, so the result does not depend on it. The standard errors and the correlation matrix come from the Huber-weighted Gauss-Newton normal matrix, and the strongest correlation, between the semi-major axis and the mean anomaly at 0.983, is the identifiability evidence quoted in the paper.

### Epicycles

The frequency lattice contains every combination of the sidereal and synodic frequencies, n1 f_sid + n2 f_syn, with |n1| + |n2| at most 8, a positive frequency and a period of at least 25 days, with duplicates removed. The sidereal period is 686.9800 days and the synodic period 779.9361 days, and because their sum is the annual frequency, the lattice is the deferent-plus-epicycle family and its harmonics. Frequencies are ordered by greedy forward selection on the training window under the Huber cost, and a model with K terms uses the constant plus the first (K - 1) / 2 frequencies. The order K is chosen inside the training span only, by two selectors with the median absolute validation residual as the criterion. Forward chaining fits before a cut and validates on the next 20 percent of the training span, at five cuts from 50 to 90 percent, and blocked five-fold cross-validation is run for comparison. The two selectors disagree, K = 3 by forward chaining against K = 19 by blocked folds (`robust_huber.epicycle_cv`). Table 2 reports the saturated K = 25 because it is the member of the family that fits the training years best, which is the question the pilot asks. The cross-validated K = 3 gives 193.19 arcmin in sample and 622.18 held out, and K = 19 gives 7.42 and 655.30, so the held-out failure does not depend on the selector.

### Gaussian process

The kernel is the sum of two quasi-periodic components, one at the sidereal and one at the synodic period. Each component is an amplitude times a periodic exponential-sine-squared factor times a squared-exponential time envelope, and white noise is added, which gives seven hyperparameters (`hyperparams` in the entry). They are fitted by type II maximum likelihood from eight seeded starts (seed 20260829), each optimized by L-BFGS-B and then by Powell's method, and the best marginal likelihood is kept, with a negative log likelihood of -1263.64. An interpolating process cannot see an outlier in its own residual, so the fit is robustified by the closed-form leave-one-out residuals, which drive an IRLS loop on the per-point noise with Huber weights at 5 arcmin. Capacity is reported as the effective degrees of freedom trace(K (K + S)^-1) on the training epochs, 122.0.

### Ridge

The design matrix is the full lattice with 145 coefficients. The penalty is chosen from 40 log-spaced values between 1e-6 and 1e8 by the median absolute residual on the last fifth of the training window, and the model is then refitted on all training epochs. This row and the pure-periodic process are training-only extrapolation controls rather than matched Huber fits, and they are labelled as such in the results file.

### Constant

The constant is the Huber location of the training declinations, and its least-squares twin is the plain mean.

## Statistics

The statistic is the median absolute residual in arcmin. Intervals are percentile bootstraps of the median over epochs within each split, 10,000 resamples with seed 20260829, with the generator seeded inside each call so that an interval does not depend on what was computed before it. A median is quoted only on at least ten epochs, and below that the sorted absolute residuals are listed, which is why the 1594 and 1600 campaigns and the third horizon bucket appear as lists. Held-out horizons are measured from the last training epoch in synodic periods, 0 to 780, 780 to 1560 and 1560 to 2340 days, with 69, 10 and 7 epochs. Paired comparisons report the Hodges-Lehmann shift of the paired absolute residuals with a paired bootstrap interval and a two-sided sign test (`headline.paired_tests`). Fitting variability is not in the intervals, because every row is one fitted run.

The AI Feynman row was scored with the same statistic but with 2,000 resamples and seed 23, and under this file's procedure its intervals would be [532.14, 602.41] in sample and [2690.04, 2698.09] held out. The synodic period given to AI Feynman, 779.94 days, differs from the lattice value in the fourth decimal.

## Structure in Tycho's residuals

The paper refers to structure in Tycho's residuals that the release documents. The facts below are the ones it means, each with the key of the results file that holds it.

Gross errors are the first. Eighteen of the 354 epochs differ from the oracle by more than 60 arcmin, 15 of them in training and 9 of them sign flips (`data.gross_error_census`). A training-window smoothness detector flags 11 epochs with a precision of 0.909 and a recall of 0.556 (`data.outlier_detector`), and it is used as a census and a robustness check but never applied to the headline. Deleting the flagged epochs moves the Keplerian fit from 2.66 and 13.31 to 2.57 and 16.10 arcmin (`deletion_robustness_check`).

Altitude is the second. Transit altitudes run from 6.1 to 61.4 degrees, and the residual against the oracle has a Spearman correlation of -0.189 with altitude, with p = 4.8e-4 (`refraction_analysis`). The median signed residual shifts by 3.30 arcmin between the two windows, and a robust linear altitude term fitted on the training window reproduces 0.52 arcmin of that shift. Correcting every declination with the term and refitting changes the held-out Keplerian median from 13.31 to 12.38 arcmin.

Campaigns are the third. The held-out window is three campaigns and a tail with 5, 54, 20 and 7 epochs, and the aggregate Keplerian comparison is driven by the 1595 campaign, 17.40 arcmin against the oracle's 3.56 (`headline.by_campaign`).

A known-truth injection test separates these effects from identifiability. Declinations generated from the modern orbit on Tycho's own epochs are recovered exactly. Adding the fitted altitude term, the observed gross-error vector and Gaussian noise of 2 arcmin leaves the held-out error against the truth below 2 arcmin, and only injecting the full observed residual vector reproduces the real 13.31 arcmin (`synthetic_observation_physics`). The degradation therefore lies in an unmodelled component of the residual pattern rather than in identifiability.

Degeneracy is the last. With declination alone, the semi-major axis and the mean anomaly at epoch correlate at 0.983 in the fitted covariance (`robust_huber.models.kepler6.degeneracy`), which is the sense in which the paper says that declination alone leaves the orbital period weakly determined.

## Runtime, environment and reproduction precision

The paper's numbers were produced in 612 s on one CPU with Python 3.14.7, NumPy 2.4.2 and SciPy 1.17.1, and the provenance block of the results file records the versions, the seeds and the wall time of any run.

Every fit is seeded, and a rerun reproduces every printed digit of Table 2 except on the Gaussian-process row, whose held-out figures come out as 1173.94 [1150.27, 1185.87] against the printed 1173.95 [1150.28, 1185.88]. The unmodified private pipeline rerun on the same machine with the same library versions gives 1173.94 [1150.26, 1185.85], so the wobble is not introduced by this port. It is floating-point nondeterminism in the optimizers, since the marginal-likelihood search and the trust-region fits both go through LAPACK. The affected quantity is an extrapolation that sits three orders of magnitude above the noise floor, where a relative change of 1e-5 is a hundredth of an arcmin. Every other row, every interval and every quoted fact reproduce to the printed precision, and the contract test therefore allows 2e-5 relative on top of half a unit in the last printed decimal.

The suite in `tests/` pins the calendar, the ephemeris, the frames and units, the split, the outlier census, the synthetic known-truth recovery of the fitter, the results contract, and every number of Table 2. Tests that read Pafko's spreadsheet use the local copy fetched by `keplerbench fetch-historical` and skip without it.

## Files

| File | What |
|---|---|
| `pilot_data.py` | loader for the rebuilt table, with the training-window gross-error detector |
| `ephem.py` | calendar, solar theory, precession, Kepler's equation, the modern elements |
| `models.py` | the model families and their fitters |
| `run_robust.py` | the Huber machinery |
| `run_pilot.py` | the single entry point, which writes `results.json` |
| `make_table.py` | Table 2 and the other tables from the results, which writes `TABLE2.md` and `results_summary.json` |
| `make_figure.py` | figures from `results.json` |
| `tests/` | the acceptance suite, with thresholds in `tests/constants.py` |
