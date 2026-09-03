# Table 2 of the paper, regenerated

Median absolute residual on Tycho's Mars declinations in arcmin, with epoch-bootstrap 95 percent intervals.
The six fitted and reference rows come from `results.json` (10,000 resamples, seed 20260829); the AI Feynman row
comes from `../aifeynman/runs/phases_20s/scored_phases.json` (2,000 resamples, seed 23).

| Model | Type | Eff. dof | In sample (arcmin) | Held out (arcmin) |
|---|---|---:|---:|---:|
| Oracle (true orbit, no fitting) | reference | 0 | 3.95 [3.17, 4.53] | 3.48 [2.74, 3.75] |
| Ellipse fitted to Tycho (Keplerian) | physical law | 6 | 2.66 [2.33, 3.07] | 13.31 [12.12, 17.40] |
| Ptolemaic epicycles (25 terms) | flexible fit | 25 | 4.02 [3.36, 4.79] | 731.19 [382.42, 924.58] |
| Flexible curve fit (Gaussian process) | flexible fit | 122 | 0.72 [0.59, 0.89] | 1173.94 [1150.27, 1185.87] |
| Sine-wave fit (ridge) | control | 145 | 4.80 [3.97, 5.52] | 1266.51 [1223.75, 1279.94] |
| Symbolic regression (AI Feynman) | control | -- | 580.87 [532.25, 601.79] | 2694.14 [2690.12, 2698.09] |
| Constant (training mean) | control | 1 | 714.09 [621.93, 761.32] | 966.14 [959.85, 990.74] |

## Held-out campaigns

| Campaign | n | Keplerian | Oracle (modern elements) |
|---|---:|---:|---:|
| 1594 | 5 | 1.14, 7.80, 7.95, 9.65, 12.12 | 5.09, 6.91, 7.58, 8.39, 15.96 |
| 1595 | 54 | 17.40 [13.75, 18.63] | 3.56 [2.98, 4.00] |
| 1596 | 20 | 3.57 [2.08, 5.70] | 2.43 [1.99, 2.59] |
| 1600 | 7 | 30.83, 42.28, 44.48, 46.83, 47.86, 47.96, 48.37 | 20.57, 32.75, 38.76, 38.91, 64.40, 67.32, 69.38 |

## Held-out horizon (Keplerian against the Gaussian process)

| Horizon (days after training) | Keplerian | Gaussian process |
|---|---:|---:|
| 0-780d (n = 69) | 13.60 [12.48, 17.65] | 1150.27 [999.80, 1173.68] |
| 780-1560d (n = 10) | 1.60 [0.65, 2.98] | 1501.64 [1438.86, 1552.94] |
| 1560-2340d (n = 7) | 30.83, 42.28, 44.48, 46.83, 47.86, 47.96, 48.37 | 1399.15, 1413.82, 1428.60, 1529.93, 1534.10, 1551.49, 1560.45 |

## Facts quoted in the paper

- epochs 354 (train 268, held out 86); held-out campaigns {'1594': 5, '1595': 54, '1596': 20, '1600': 7}
- gross errors above 60 arcmin: 18 (15 in training, 9 sign flips)
- transit altitude 6.1 to 61.4 deg; Spearman rho of residual against altitude -0.189 (p = 4.8e-04)
- signed median residual shift between windows 3.30 arcmin; altitude-corrected Keplerian refit held out 12.38 arcmin
- Keplerian degeneracy: ['M0', 'a'] correlate at 0.983
- Keplerian restarts at the global optimum 8/61; blind restarts (no modern start) reach held out 13.31 arcmin
- Gaussian process effective dof 122.0, 8 starts, negative log likelihood -1263.64
- epicycle K selected inside training: forward chaining 3, blocked folds 19; the table reports the saturated K = 25
- runtime 667.0 s on one CPU; python 3.14.6, numpy 2.4.2, scipy 1.17.1
