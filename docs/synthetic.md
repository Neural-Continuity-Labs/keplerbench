# Synthetic worlds

The synthetic sub-benchmark generates new star systems that keep the
observer-centred observation format of the historical record. A world is a
planar central-force system: a target body and an observer body move
independently about a central mass, the observer on a circular orbit and the
target on an orbit of chosen scale and eccentricity, integrated by velocity
Verlet with fixed substeps. Each observation is a time, the unit direction from
the observer to the target, and the unit direction from the observer to the
central body (the "star"), with optional angular noise on both directions. The
learner sees directions only; the latent Cartesian states and the physical
parameters are written to a sidecar metadata file for auditability.

## Observation contract

Each row contains

```text
world_id, system_id, observer_id, target_id, time,
planet_dir_x, planet_dir_y, planet_dir_z,
star_dir_x, star_dir_y, star_dir_z, split
```

with unit vectors in a fixed celestial frame and `split` in train, validation
or test. The first `train_fraction` of the window is training and the rest is
test, so held-out prediction is temporal extrapolation, as it is for Tycho's
record. The schema is `data/schemas/observation.schema.json`.

## Families

A family is a set of parameter ranges. A world is drawn by sampling every
listed parameter uniformly with the seed (integer fields inclusively), so a
family and a seed identify a world exactly. Everything not listed keeps the
generator default.

| Parameter | inverse-square | hidden-exponent | Meaning |
|---|---|---|---|
| `gravitational_parameter` | 0.5 to 2.0 | 0.5 to 2.0 | central mass times the gravitational constant |
| `force_exponent` | 2 (fixed) | 1.5 to 2.5 | p in F(r) proportional to r^-p |
| `target_radius` | 1.3 to 3.0 | 1.3 to 3.0 | target orbit scale, periapsis over (1 - e) |
| `target_eccentricity` | 0.0 to 0.6 | 0.0 to 0.4 | target eccentricity |
| `observer_radius` | 0.6 to 1.2 | 0.6 to 1.2 | radius of the observer's circular orbit |
| `target_phase`, `observer_phase` | 0 to 2 pi | 0 to 2 pi | rotation of each initial state, the initial conditions |
| `duration` | 10 to 40 | 10 to 40 | length of the observation window, time units |
| `sample_count` | 120 to 600 | 120 to 600 | number of observation times |
| `angular_noise_std` | 0 to 6e-4 rad | 0 to 6e-4 rad | angular noise on both directions, up to about 2 arcmin |

The first family obeys inverse-square dynamics and varies the central mass,
the initial conditions, the orbital geometry, the observer trajectory, the
sampling and the noise, which is the family the paper releases with the
benchmark. The second family also hides the force exponent, the first of the
harder families the paper announces. The ranges are in
`keplerbench.synthetic.FAMILIES`.

```bash
keplerbench generate --output data/world_a.csv --family inverse-square --seed 3
keplerbench generate --output data/world_b.csv --family hidden-exponent --seed 3 --noise 0.0006   # noise overridden
keplerbench generate --output data/world_c.csv --seed 7 --eccentricity 0.35 --duration 30          # explicit world
```

`--samples` and `--noise` override the family's draw when given; the orbit
parameters of a family draw cannot be overridden from the command line. The
sidecar `<output>.metadata.json` records the generator version, the seed and
every latent parameter, and must not be supplied to learners.

## Noise floor and data-blind copy

The noise floor of a synthetic world is its `angular_noise_std`, which the
scorer reads from the sidecar. `keplerbench data-blind --input world.csv
--output world_blind.csv --seed 0` writes the null copy used by the data-blind
control: the same times, identifiers and splits, with each row's six direction
components moved together to another epoch of the same split.
