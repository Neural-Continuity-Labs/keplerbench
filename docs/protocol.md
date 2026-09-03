# Evaluation protocol

## Common principles

The learner receives public observations only. Generator metadata, latent
Cartesian states, orbital elements, and force parameters are evaluator-private.
A submission must declare all training data, pretrained components, and prior
scientific knowledge used by its discovery procedure.

Each result reports the evidence setting, capability test, random seeds, and
evidence budget. Scores from unrestricted pretrained systems remain useful, but
they do not establish uncontaminated historical rediscovery.

## 1. Prediction

The evaluator withholds a time block from one observer-centred sequence. The
submission predicts apparent target and star directions at the withheld times.
Primary metric: mean great-circle angular error. Median and 95th-percentile
angular errors are reported as diagnostics.

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

## Submission boundary

The initial software release fully automates only the prediction metric. The
global- and local-law contracts above are frozen for discussion and early system
development; official hidden-world suites and validators remain roadmap items.
