# AI Feynman controls (the symbolic-regression row of Table 2)

These are the symbolic-regression controls of the paper, run with AI Feynman
2.0.7 (MIT license) on the training split of Tycho's Mars declinations. They
are controls for the question of whether in-sample fit says anything about
later years, not a discovery method. The objectives of the other rows, the
bootstrap procedure of the pilot, and the residual analysis are in
[`../pilot/README.md`](../pilot/README.md).

## Runs

Inputs are built from the locally rebuilt epoch table by
`prepare_pilot_inputs.py`. Time is in Julian years since the first training
epoch; the phases are 2 pi t / P for Mars's sidereal period (686.980 d) and
synodic period (779.94 d), unwrapped. The phase run is the one in Table 2,
because it gives AI Feynman the same two periods that the epicycle and ridge
rows receive.

| Run | Inputs | Operations | BF_try_time | polyfit_deg | NN_epochs | Runtime | In sample (arcmin) | Held out (arcmin) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `phases_20s` (Table 2) | sidereal phase, synodic phase | 14ops.txt | 20 s | 3 | 200 | 12 min | 580.87 [532.25, 601.79] | 2694.14 [2690.12, 2698.09] |
| `time_300s` | time | 19ops.txt | 300 s | 4 | 4000 | 90 min | 675.44 [656.80, 715.91] | 1065.01 [1060.74, 1066.97] |
| `time_120s` | time | 14ops.txt | 120 s | 3 | 1000 | 36 min | 685.05 [646.29, 712.91] | 1038.82 [1016.92, 1081.32] |

Selected expressions, the lowest training error among the expressions of the
frontier that evaluate to finite values on both splits:

- `phases_20s`: `log(666*exp(pi - (x1 - 1)))`, a constant in disguise, 28.9
  bits. The next expression, `tan(-0.062 + sin(x0))`, gives 623.25 in sample
  and 1102.34 held out.
- `time_300s`: `asin(0.719103162249*sin(exp(sqrt(x0)/atan(1))))`, 73.0 bits.
- `time_120s`: `asin(0.967182888539*cos(x0*exp(exp(cos(cos(x0))))))`, 58.1 bits.

The constant control has 714.09 in sample and 966.14 held out. With the
periods supplied, AI Feynman fits the training years slightly better than the
constant and extrapolates worse; with time only it does neither. The phase run
had a much smaller search budget than the time-only runs because of the
submission deadline, and rerunning it at the larger budget is on the list for
the next version. All numbers are single runs.

## Scoring

`score_solutions.py` is the single scoring path. For every expression on the
frontier it reports the median absolute residual in arcmin on the training
and the held-out epochs, each with a percentile bootstrap 95 percent interval
over epochs. The defaults, 2,000 resamples and seed 23, are the procedure
behind the Table 2 row; the pilot's other rows use 10,000 resamples and seed
20260829, which `--resamples 10000 --seed 20260829` reproduces. Complexity is
column 4 of AI Feynman's solution file, and `complexity_bits_recomputed` is
AI Feynman's own `get_expr_complexity` applied to the expression string; the
two agree on every selected expression.

Solution-file columns, from AI Feynman's `S_run_aifeynman.py`: the test error
on AI Feynman's internal split, the mean description length of the residuals
in bits, that length summed over the data points, the complexity of the
expression in bits, the fit error, and the expression.

## Reproducing

AI Feynman needs `numpy.distutils`, so it has its own environment.

```bash
uv venv --python 3.11 .venv-aif && source .venv-aif/bin/activate
uv pip install "numpy<2" "scipy<1.14" pandas sympy matplotlib seaborn scikit-learn torch sortedcontainers "setuptools<65" wheel
SETUPTOOLS_USE_DISTUTILS=stdlib NPY_DISTUTILS_APPEND_FLAGS=1 uv pip install --no-deps --no-build-isolation aifeynman==2.0.7
# gfortran must be on PATH, and so must AI Feynman's brute-force binaries, which is why the venv stays activated

keplerbench fetch-historical                                   # rebuilds data/historical/mars_declinations.csv
python experiments/aifeynman/prepare_pilot_inputs.py
python experiments/aifeynman/run_aif.py experiments/aifeynman/inputs/pilot_phases_to_dec.txt phases_20s --bf-time 20 --nn-epochs 200 --ops 14ops.txt --polyfit-deg 3
python experiments/aifeynman/score_solutions.py experiments/aifeynman/runs/phases_20s/solution_pilot_phases_to_dec.txt --inputs phases
```

The neural-network stage of AI Feynman is not seeded, so a rerun produces a
frontier of its own; the committed frontiers are the ones behind the paper.

## Files

`runs/<name>/` holds AI Feynman's final Pareto frontier (`solution_*.txt`,
raw), the settings and runtime with the frontier's named columns
(`report.json`), and the scored frontier with the selected expression
(`scored_*.json`). The input files are not committed because they contain
declination values derived from Pafko's transcription; the two commands above
rebuild them.
