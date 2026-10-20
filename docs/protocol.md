# Evaluation protocol

## Common principles

The learner receives public observations only. Generator metadata, latent
Cartesian states, orbital elements, and force parameters are evaluator-private.
A submission must declare all training data, pretrained components, and prior
scientific knowledge used by its discovery procedure.

Each result reports the sub-benchmark, capability test, random seeds, and
sample budget. Scores from unrestricted pretrained systems remain useful, but
they do not establish uncontaminated historical rediscovery.

## Data-blind control

Prior knowledge is controlled in software rather than by self-report. For every
scored entry on a sub-benchmark, the evaluator also runs the same system on a
data-blind input of the same shape: the times, identifiers and split labels
are kept and the observed quantities are replaced by a draw from the
sub-benchmark's null distribution. The null distribution is a permutation of
the observed quantities across the epochs of each split, which keeps their
marginal distribution and destroys their relation to time; `keplerbench
data-blind` writes it for a synthetic world (the six direction components of a
row move together) and for the historical table (the declination moves). Both
runs are scored against the true held-out observations and both scores are
reported. The entry earns credit on the sub-benchmark only when the data-blind
run scores worse than the entry by more than the sub-benchmark's noise floor;
a difference inside the floor means the score was carried by prior knowledge
rather than extracted from the evidence. `evaluate-prediction` and
`evaluate-historical` apply the rule when given the data-blind predictions.

## Noise floor and stated uncertainty

Each sub-benchmark fixes a noise floor from its measurement process, 2 arcmin
for Tycho's record and the configured angular noise for a synthetic world,
read from the world's sidecar. Entries state a predictive uncertainty for
every prediction, and residuals below the floor are not ranked against one
another. The historical scorer accepts the stated uncertainty as a column and
reports the fraction of residuals inside two of them.

## 1. Prediction

The evaluator withholds a time block from one observer-centred sequence. The
submission predicts apparent target and star directions at the withheld times.
Primary metric: mean great-circle angular error. Median and 95th-percentile
angular errors are reported as diagnostics.

On the historical Mars sub-benchmark, where the record carries declination
only, the submission predicts the declination at the held-out epochs and the
statistic is the median absolute residual in arcmin with a percentile
bootstrap interval over epochs (10,000 resamples, seed 20260829, the pilot's
procedure), reported per campaign, with a median quoted only on at least ten
epochs. `keplerbench evaluate-historical` computes it.

Passing this test establishes predictive adequacy only.

## 2. Global-law discovery

The evaluator trains or runs the discovery procedure on designated source
systems, freezes the shared learned object, and exposes `k` observations from a
held-out target system. Adaptation may estimate target-specific quantities but
must not update the frozen shared object.

For each evidence budget `k`, the evaluator measures angular prediction error on
the remainder of the target sequence. The complete `error(k)` curve is primary;
normalized area under the error-versus-log-evidence curve is a compact summary.
Multiple unseen systems and seeds are required.

## 3. Local dynamical-law discovery

A submission provides:

- an encoder that constructs a fixed-dimensional state from an initialization
  window;
- a transition rule that maps the current state and time increment to the next
  state; and
- a decoder from state to public observations.

After initialization, the evaluator removes the observation history. The same
transition rule is rolled out repeatedly under held-out initial conditions,
physical parameters, observers, and systems. Primary measures are long-horizon
angular rollout error and failure time at fixed error thresholds. Evaluator-only
checks compare the generated trajectories with known global invariants or
regularities. Reversible worlds may include backward rollout.

## What is automated now

The software automates the prediction test on both released sub-benchmarks,
the data-blind control, the noise-floor rule, the synthetic generator with its
two families, and the historical fetch-and-reduce route. Since version 0.3.0
it also automates the global-law transfer harness (`keplerbench.transfer`,
with the error-versus-k curve and its normalized area) and the local-law
rollout harness with the window-consistency check (`keplerbench.rollout`).
Reference entries exercising both, with the data-blind control applied end to
end, are in `experiments/synthetic_demo/`. The evaluator-private world suites
are the next release.

Two operational points the harnesses fix. A global-law entry declares the
parameter count and meaning of its adapted per-system quantities, reported
beside its score: an adapter large enough to absorb the law is not excluded,
it is priced by its error-versus-k curve at small k. A local-law state may
contain a clock, and a state that stores a fitted trajectory next to one
satisfies the interface; it is bounded by prediction-from-window and separated
by long-horizon rollouts and the window-consistency check, as the reference
runs demonstrate.
