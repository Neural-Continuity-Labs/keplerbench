# The synthetic tests, end to end (Appendix D of the paper)

This directory runs the global-law transfer test and the local dynamical-law
rollout test to completion on the released synthetic families, with the
data-blind control applied, so that every score the benchmark defines is
demonstrated by at least one entry. The entries are reference constructions,
not discovery systems: the shared-law entry adopts the generator's
central-force structure as its frozen law, in the same physics-supplied sense
as the fitted ellipse of the historical pilot. What the runs establish is that
the harnesses execute the protocol end to end and that their scores separate
an entry that carries a reusable law from flexible adaptation, from a
misspecified familiar law, and from a stored trajectory.

## Reproducing

```bash
python experiments/synthetic_demo/run_transfer.py
python experiments/synthetic_demo/run_rollout.py
python experiments/synthetic_demo/run_exponent_recovery.py
```

Each script regenerates its worlds from seeds, runs in minutes on one CPU, and
writes its results (`transfer_results.json`, `rollout_results.json`,
`exponent_recovery.json`), a markdown summary (`TRANSFER.md`, `ROLLOUT.md`),
and the paper's figures. The committed results are the ones the paper quotes.

## The entries

`learners.py` holds them all.

- **Shared law, frozen exponent.** Discovery fits eight source worlds with the
  exponent free and freezes the median (p = 2.0001 on the inverse-square
  sources). Adaptation estimates six declared per-system quantities from the k
  exposed observations: effective mu, orbit scale ratio, target eccentricity
  and phase, observer eccentricity and phase, with the observer's orbital
  scale anchored at 1 (directions are invariant under a common rescaling of
  lengths by lambda with mu rescaled by lambda^(p+1), so only anchored
  quantities are identifiable).
- **Flexible baseline.** The benchmark's own level-1 harmonic baseline fitted
  to the k exposed observations alone, order growing with k. It shares nothing
  across systems.
- **Adapted exponent.** The same central-force structure with the exponent as
  a seventh conceptual (ninth actual) adapted quantity. It uses the complete
  initial-condition chart (radius, angle, radial and tangential speed per
  body) because the generator's periapsis-speed parameterization is only
  scale-covariant at p = 2, so the (a, e, phase) chart cannot represent a
  rescaled world with any other exponent.
- **Oracle.** Replays the generator's own construction from the latent world
  parameters, unscaled. Never an entry: it validates the harness and the
  integrator against each world's noise floor before any comparison is read.
- **Dynamical rollout submission.** An 8-number state (Cartesian positions and
  velocities of both bodies), context (mu, p), one leapfrog transition rule.
- **Trajectory-plus-clock control.** A 35-number state holding the
  coefficients of a trend-plus-harmonics curve fitted to the planet's apparent
  angle over the window, next to a clock its transition rule increments. It
  satisfies the letter of the local-law interface; the evaluation separates it
  through long horizons and the window-consistency check, since its rollout is
  bounded by prediction-from-window.

## What the committed results show

- Inverse-square targets (10 unseen worlds): the frozen law walks from 4041 to
  3.5 arcmin mean held-out error as k grows from 4 to 128, coming within 1.0 to
  2.4 times the oracle's residual on eight of ten worlds at k = 128 (the other
  two sit at 4.0 and 13.2 times), while the flexible baseline stays near 4000
  arcmin at every budget (32 times the law's error at k = 32, 1312 times at
  k = 128). Normalized AUC 0.42 against 1.18 rad.
- Hidden-exponent targets (10 unseen worlds): frozen p = 2 plateaus at 928
  arcmin at k = 128, and its per-world error grows with the world's true
  departure from inverse square (Spearman 0.67, p = 0.03). The
  adapted-exponent entry is worse at k <= 16 (the capacity price of three
  extra adapted degrees of freedom), overtakes at k = 32, reaches 13.3 arcmin
  mean at k = 128, and fits the hidden exponent to a median absolute error of
  0.0005 (worst 0.026, on a world whose exponent is 1.995).
- Data-blind control (shared law, k = 32, three targets): entry errors 143.3,
  3.4 and 60.9 arcmin against data-blind errors of 7453 to 9204 arcmin;
  credit granted on all three.
- Rollout (three long worlds, 60 time units past the window, 3.1 to 4.8
  target periods): the dynamical state stays at 0.5 to 1.5 arcmin in every
  horizon bucket, never crosses the 60-arcmin threshold, and moves by at most
  0.6 arcmin under re-initialization from a different window; the control
  crosses the threshold at the first observation after its window on two
  worlds and the third on the other (0.05 to 0.15 time units) and moves by 538
  to 9974 arcmin median under re-initialization.

## Conventions

Target worlds fix the sampling density at 450 so every budget up to k = 128
exists on every world, and raise the generator's integration substeps to 120
so the scored truth is the orbit rather than the integrator; both overrides
are recorded in each world's configuration sidecar. Angular noise is as drawn
by the family, up to about 2 arcmin. All randomness is seeded and stated in
the scripts.
