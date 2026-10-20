"""Reference learners for the synthetic transfer and rollout demonstration.

Two kinds of entries are exercised against the harnesses in
`keplerbench.transfer` and `keplerbench.rollout`.

The central-force learner is a physics-supplied reference, in the same sense
as the fitted ellipse of the historical pilot: its shared structure is the
planar central-force family of the generator (two bodies orbiting a central
mass, observed from one of them), adopted as the discovered law rather than
discovered by any algorithm. Its per-system quantities are an effective
gravitational parameter, the target orbit's scale, eccentricity and phase, and
the observer orbit's eccentricity and phase. Directions are invariant under a
common rescaling of all lengths by lambda with mu rescaled by lambda^(p+1), so
the observer's orbital scale is anchored at 1 and every length is estimated
relative to it. The force exponent p is either frozen by a discovery phase, or
frozen at a misspecified value, or adapted per system, depending on the entry.

The flexible learner is the benchmark's own level-1 harmonic baseline fitted
to the exposed observations of each target alone. It shares nothing across
systems, which is exactly the contrast the global-law test is meant to expose.

The rollout entries are a dynamical submission, whose state is the Cartesian
state of both bodies and whose transition rule is one leapfrog integrator, and
a trajectory-plus-clock control, whose state stores the coefficients of a
curve fitted to the initialization window together with a clock. The control
is the construction a reviewer may worry passes a rollout evaluation without
holding any local law; the demonstration measures where the harness separates
the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from keplerbench.baseline import harmonic_predict
from keplerbench.synthetic import _initial_state, _integrate

Rows = Sequence[dict[str, object]]

_TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Forward models
# ---------------------------------------------------------------------------

def _kepler_positions(times: np.ndarray, a: float, e: float, phase: float, mu: float) -> np.ndarray:
    """Analytic positions on a closed inverse-square orbit, t=0 at periapsis.

    Matches the generator's initial-state convention: semi-major axis a,
    periapsis rotated by phase, counterclockwise motion.
    """

    n = np.sqrt(mu / a**3)
    M = n * times
    E = M + e * np.sin(M)
    for _ in range(12):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0))
    r = a * (1.0 - e * np.cos(E))
    c, s = np.cos(phase), np.sin(phase)
    x, y = r * np.cos(nu), r * np.sin(nu)
    return np.column_stack([c * x - s * y, s * x + c * y])


def _integrated_positions(times: np.ndarray, a: float, e: float, phase: float, mu: float, p: float, dt_max: float) -> np.ndarray:
    """Positions for a general force exponent, by leapfrog on a dense grid."""

    t_end = float(times.max())
    grid = np.unique(np.concatenate([np.arange(0.0, t_end + dt_max, dt_max), times, [0.0]]))
    position, velocity = _initial_state(a, e, phase, mu)
    trajectory = _integrate(position, velocity, grid, 1, mu, p)
    return trajectory[np.searchsorted(grid, times)]


def _positions(times: np.ndarray, a: float, e: float, phase: float, mu: float, p: float, dt_max: float) -> np.ndarray:
    if p == 2.0:
        return _kepler_positions(times, a, e, phase, mu)
    return _integrated_positions(times, a, e, phase, mu, p, dt_max)


def _directions(target: np.ndarray, observer: np.ndarray) -> np.ndarray:
    relative = target - observer
    padded = np.column_stack([relative, np.zeros(len(relative))])
    return padded / np.linalg.norm(padded, axis=1, keepdims=True)


def _star_directions(observer: np.ndarray) -> np.ndarray:
    padded = np.column_stack([-observer, np.zeros(len(observer))])
    return padded / np.linalg.norm(padded, axis=1, keepdims=True)


def _planar_angle_residuals(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Signed planar angle between unit direction pairs, smooth at zero."""

    cross = predicted[:, 0] * observed[:, 1] - predicted[:, 1] * observed[:, 0]
    dot = predicted[:, 0] * observed[:, 0] + predicted[:, 1] * observed[:, 1]
    return np.arctan2(cross, dot)


# ---------------------------------------------------------------------------
# Central-force fitting
# ---------------------------------------------------------------------------

@dataclass
class CentralForceFit:
    """A fitted per-system parameter vector with the frozen structure."""

    mu: float
    target_a: float
    target_e: float
    target_phase: float
    observer_e: float
    observer_phase: float
    p: float
    dt_max: float
    residual_rms_radians: float

    def predict_planet(self, times: np.ndarray) -> np.ndarray:
        target = _positions(times, self.target_a, self.target_e, self.target_phase, self.mu, self.p, self.dt_max)
        observer = _positions(times, 1.0, self.observer_e, self.observer_phase, self.mu, self.p, self.dt_max)
        return _directions(target, observer)

    def cartesian_state(self, time: float, dt_max: float | None = None) -> np.ndarray:
        """Positions and velocities of target and observer at one instant."""

        dt = dt_max or self.dt_max
        eps = 1e-4
        times = np.array([max(time - eps, 0.0), time + eps])
        target = _positions(times, self.target_a, self.target_e, self.target_phase, self.mu, self.p, dt)
        observer = _positions(times, 1.0, self.observer_e, self.observer_phase, self.mu, self.p, dt)
        target_v = (target[1] - target[0]) / (times[1] - times[0])
        observer_v = (observer[1] - observer[0]) / (times[1] - times[0])
        target_r = _positions(np.array([time]), self.target_a, self.target_e, self.target_phase, self.mu, self.p, dt)[0]
        observer_r = _positions(np.array([time]), 1.0, self.observer_e, self.observer_phase, self.mu, self.p, dt)[0]
        return np.concatenate([target_r, target_v, observer_r, observer_v])


def _rows_arrays(rows: Rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.array([float(row["time"]) for row in rows])
    planet = np.array([[float(row[f"planet_dir_{axis}"]) for axis in "xy"] for row in rows])
    star = np.array([[float(row[f"star_dir_{axis}"]) for axis in "xy"] for row in rows])
    order = np.argsort(times)
    return times[order], planet[order], star[order]


def _observer_initial_guess(times: np.ndarray, star: np.ndarray) -> tuple[float, float]:
    """Observer phase and effective mu from the star directions alone.

    The star direction is minus the observer's position direction, so the
    observer's angular history is read directly off the exposed observations.
    With the observer scale anchored at 1 and a near-circular orbit, the mean
    angular rate omega gives mu = omega^2 for any force exponent.
    """

    theta = np.unwrap(np.arctan2(-star[:, 1], -star[:, 0]))
    if len(times) >= 3:
        slope, intercept = np.polyfit(times, theta, 1)
    else:
        slope = (theta[-1] - theta[0]) / max(times[-1] - times[0], 1e-9)
        intercept = theta[0] - slope * times[0]
    return float(intercept % _TWO_PI), float(max(slope, 1e-3) ** 2)


def fit_central_force(
    rows: Rows,
    p: float | None = 2.0,
    seed: int = 0,
    starts: int = 3,
    dt_max: float = 0.01,
) -> CentralForceFit:
    """Fit the per-system quantities of the central-force structure.

    p=2.0 freezes the inverse-square exponent (the analytic fast path), any
    other float freezes that value, and None adapts the exponent as a seventh
    per-system parameter. The observer's phase and the effective mu are
    initialized from the star directions, and the target parameters from a
    coarse grid, after which the best few starts are polished by least squares
    on the signed planar angle residuals of both observed directions.
    """

    if p is None:
        return fit_state_chart(rows, seed=seed, dt_max=dt_max)
    times, planet, star = _rows_arrays(rows)
    observer_phase0, mu0 = _observer_initial_guess(times, star)
    p_fixed = float(p)

    def residuals(vector: np.ndarray) -> np.ndarray:
        mu, a, e_t, phase_t, e_o, phase_o = vector[:6]
        target = _positions(times, a, e_t, phase_t, mu, p_fixed, dt_max)
        observer = _positions(times, 1.0, e_o, phase_o, mu, p_fixed, dt_max)
        return np.concatenate([
            _planar_angle_residuals(_directions(target, observer)[:, :2], planet),
            _planar_angle_residuals(_star_directions(observer)[:, :2], star),
        ])

    rng = np.random.default_rng(seed)
    lower = [1e-3, 0.3, 0.0, -_TWO_PI, 0.0, -_TWO_PI]
    upper = [100.0, 12.0, 0.9, 2 * _TWO_PI, 0.9, 2 * _TWO_PI]
    grid = [
        (a, e, phase)
        for a in (1.2, 1.6, 2.2, 3.0, 4.2)
        for e in (0.05, 0.25, 0.45)
        for phase in np.linspace(0.0, _TWO_PI, 8, endpoint=False)
    ]
    scored = []
    for a, e, phase in grid:
        vector = np.array([mu0, a, e, phase, 0.0, observer_phase0])
        try:
            cost = float(np.sum(residuals(vector) ** 2))
        except FloatingPointError:
            continue
        scored.append((cost, vector))
    scored.sort(key=lambda item: item[0])
    candidates = [vector for _, vector in scored[:starts]]
    best = None
    for start in candidates:
        jitter = start + rng.normal(0.0, 1e-3, size=start.shape)
        jitter = np.clip(jitter, lower, upper)
        try:
            solution = least_squares(residuals, jitter, bounds=(lower, upper), method="trf", diff_step=1e-4)
        except (FloatingPointError, ValueError):
            continue
        if best is None or solution.cost < best.cost:
            best = solution
    if best is None:
        raise RuntimeError("no start converged")
    mu, a, e_t, phase_t, e_o, phase_o = best.x[:6]
    return CentralForceFit(
        mu=float(mu), target_a=float(a), target_e=float(e_t), target_phase=float(phase_t % _TWO_PI),
        observer_e=float(e_o), observer_phase=float(phase_o % _TWO_PI),
        p=p_fixed, dt_max=dt_max,
        residual_rms_radians=float(np.sqrt(np.mean(best.fun ** 2))),
    )


# ---------------------------------------------------------------------------
# The free-exponent chart
# ---------------------------------------------------------------------------
#
# The (a, e, phase) chart above reaches every orbit only when the exponent is
# 2: the generator's periapsis-speed formula is not covariant under the length
# rescaling that the anchored observer absorbs, so for p away from 2 the chart
# cannot represent a rescaled world exactly. The free-exponent entry therefore
# parameterizes each body by its initial conditions directly: radius, angle,
# radial speed and tangential speed at t = 0, with the observer's initial
# radius anchored at 1. That chart is complete for planar central-force motion
# and quotients the same scale equivalence.

@dataclass
class StateChartFit:
    """A fitted initial-condition chart with a free force exponent."""

    mu: float
    p: float
    target_init: tuple[float, float, float, float]
    observer_init: tuple[float, float, float]
    dt_max: float
    residual_rms_radians: float

    def _states(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r0t, th0t, vr0t, vt0t = self.target_init
        th0o, vr0o, vt0o = self.observer_init
        grid = np.unique(np.concatenate([np.arange(0.0, times.max() + self.dt_max, self.dt_max), times, [0.0]]))
        indices = np.searchsorted(grid, times)
        target = _integrate(*_chart_state(r0t, th0t, vr0t, vt0t), grid, 1, self.mu, self.p)[indices]
        observer = _integrate(*_chart_state(1.0, th0o, vr0o, vt0o), grid, 1, self.mu, self.p)[indices]
        return target, observer

    def predict_planet(self, times: np.ndarray) -> np.ndarray:
        target, observer = self._states(np.asarray(times, dtype=float))
        return _directions(target, observer)


def _chart_state(radius: float, angle: float, v_radial: float, v_tangential: float) -> tuple[np.ndarray, np.ndarray]:
    radial = np.array([np.cos(angle), np.sin(angle)])
    tangential = np.array([-np.sin(angle), np.cos(angle)])
    return radius * radial, v_radial * radial + v_tangential * tangential


def _to_chart(position: np.ndarray, velocity: np.ndarray) -> tuple[float, float, float, float]:
    radius = float(np.linalg.norm(position))
    radial = position / radius
    tangential = np.array([-radial[1], radial[0]])
    return radius, float(np.arctan2(position[1], position[0])), float(velocity @ radial), float(velocity @ tangential)


def fit_state_chart(rows: Rows, seed: int = 0, dt_max: float = 0.01) -> StateChartFit:
    """Fit mu, the exponent and both bodies' initial conditions.

    Warm-started from the frozen inverse-square fit: its state at t = 0 is
    rescaled so the observer's initial radius is exactly 1, converted to the
    chart, and polished with the exponent free from three bracketing starts.
    """

    times, planet, star = _rows_arrays(rows)
    warm = fit_central_force(rows, p=2.0, seed=seed, starts=4, dt_max=dt_max)
    state = warm.cartesian_state(0.0, dt_max=1e-3)
    scale = 1.0 / np.linalg.norm(state[4:6])
    target_chart = _to_chart(state[0:2] * scale, state[2:4] * scale)
    observer_chart = _to_chart(state[4:6] * scale, state[6:8] * scale)

    grid = np.unique(np.concatenate([np.arange(0.0, times.max() + dt_max, dt_max), times, [0.0]]))
    indices = np.searchsorted(grid, times)

    def residuals(vector: np.ndarray) -> np.ndarray:
        mu, p, r0t, th0t, vr0t, vt0t, th0o, vr0o, vt0o = vector
        target = _integrate(*_chart_state(r0t, th0t, vr0t, vt0t), grid, 1, mu, p)[indices]
        observer = _integrate(*_chart_state(1.0, th0o, vr0o, vt0o), grid, 1, mu, p)[indices]
        return np.concatenate([
            _planar_angle_residuals(_directions(target, observer)[:, :2], planet),
            _planar_angle_residuals(_star_directions(observer)[:, :2], star),
        ])

    base = np.array([warm.mu * scale ** 3, 2.0, *target_chart, observer_chart[1], observer_chart[2], observer_chart[3]])
    lower = [1e-3, 1.2, 0.2, -10.0, -10.0, -10.0, -10.0, -3.0, -5.0]
    upper = [100.0, 3.0, 15.0, 10.0, 10.0, 10.0, 10.0, 3.0, 5.0]
    best = None
    for exponent in (2.0, 1.7, 2.35):
        start = base.copy()
        start[1] = exponent
        start = np.clip(start, lower, upper)
        try:
            solution = least_squares(residuals, start, bounds=(lower, upper), method="trf", diff_step=1e-4)
        except (FloatingPointError, ValueError):
            continue
        if best is None or solution.cost < best.cost:
            best = solution
    if best is None:
        raise RuntimeError("no start converged")
    mu, p, r0t, th0t, vr0t, vt0t, th0o, vr0o, vt0o = best.x
    return StateChartFit(
        mu=float(mu), p=float(p), target_init=(float(r0t), float(th0t), float(vr0t), float(vt0t)),
        observer_init=(float(th0o), float(vr0o), float(vt0o)), dt_max=dt_max,
        residual_rms_radians=float(np.sqrt(np.mean(best.fun ** 2))),
    )


def discover_exponent(source_worlds: Sequence[Rows], seed: int = 0, dt_max: float = 0.02) -> dict[str, object]:
    """The discovery phase of the shared-law entry.

    Fits every source world's training split with the exponent free and
    freezes the median exponent as the shared structure. The per-world
    exponents are returned for the record.
    """

    exponents = []
    for index, rows in enumerate(source_worlds):
        train = [row for row in rows if row["split"] == "train"]
        fit = fit_central_force(train, p=None, seed=seed + index, starts=3, dt_max=dt_max)
        exponents.append(fit.p)
    return {"per_world": exponents, "frozen_p": float(np.median(exponents))}


# ---------------------------------------------------------------------------
# Transfer adapters
# ---------------------------------------------------------------------------

def shared_law_adapter(p: float | None, seed: int = 0, dt_max: float = 0.01):
    """An adapter whose frozen structure is the central-force family.

    With a float p the exponent is part of the frozen shared object; with None
    it is adapted per system.
    """

    def adapt(exposed: Rows):
        fit = fit_central_force(exposed, p=p, seed=seed, dt_max=dt_max)

        def predict(times: np.ndarray) -> np.ndarray:
            return fit.predict_planet(np.asarray(times, dtype=float))

        return predict

    return adapt


def flexible_adapter(max_order: int = 8):
    """The level-1 harmonic baseline fitted to the exposed observations only."""

    def adapt(exposed: Rows):
        times = np.array([float(row["time"]) for row in exposed])
        directions = np.array([[float(row[f"planet_dir_{axis}"]) for axis in "xyz"] for row in exposed])
        order = int(np.clip((len(exposed) - 2) // 4, 1, max_order))
        fallback = directions.mean(axis=0)
        fallback = fallback / np.linalg.norm(fallback)

        def predict(query: np.ndarray) -> np.ndarray:
            with np.errstate(all="ignore"):
                prediction = harmonic_predict(times, directions, np.asarray(query, dtype=float), order=order)
            # An overflowing extrapolation still gets a defined answer: the
            # mean training direction, the entry's own constant baseline.
            bad = ~np.isfinite(prediction).all(axis=1)
            if bad.any():
                prediction[bad] = fallback
            return prediction

        return predict

    return adapt


def oracle_adapter(config, dt_max: float = 0.002):
    """Predicts by replaying the generator's own construction, unscaled.

    Never a benchmark entry: it reads the evaluator-private configuration to
    verify that the harness and the integrator reproduce the generator to the
    noise floor before any comparison is read.
    """

    def adapt(exposed: Rows):
        def predict(times: np.ndarray) -> np.ndarray:
            times = np.asarray(times, dtype=float)
            grid = np.unique(np.concatenate([np.arange(0.0, times.max() + dt_max, dt_max), times, [0.0]]))
            mu, p = config.gravitational_parameter, config.force_exponent
            target0 = _initial_state(config.target_radius, config.target_eccentricity, config.target_phase, mu)
            observer0 = _initial_state(config.observer_radius, 0.0, config.observer_phase, mu)
            indices = np.searchsorted(grid, times)
            target = _integrate(*target0, grid, 1, mu, p)[indices]
            observer = _integrate(*observer0, grid, 1, mu, p)[indices]
            return _directions(target, observer)

        return predict

    return adapt


# ---------------------------------------------------------------------------
# Rollout submissions
# ---------------------------------------------------------------------------

class DynamicalSubmission:
    """State: Cartesian positions and velocities of both bodies, 8 numbers.

    The encoder fits the frozen central-force structure to the window and
    reads off the state at the window's last instant. The context holds the
    per-system effective mu and the frozen exponent. The transition rule is a
    leapfrog integrator of the central force, the same rule for every system,
    horizon and body.
    """

    state_dimension = 8

    def __init__(self, p: float = 2.0, seed: int = 0, dt_max: float = 0.005):
        self.p = p
        self.seed = seed
        self.dt_max = dt_max

    def encode(self, window: Rows):
        fit = fit_central_force(window, p=self.p, seed=self.seed, dt_max=self.dt_max)
        state = fit.cartesian_state(float(sorted(float(row["time"]) for row in window)[-1]), dt_max=1e-3)
        return state, (fit.mu, fit.p)

    def step(self, state: np.ndarray, dt: float, context) -> np.ndarray:
        mu, p = context
        steps = max(1, int(np.ceil(abs(dt) / self.dt_max)))
        h = dt / steps
        target_r, target_v = state[0:2].copy(), state[2:4].copy()
        observer_r, observer_v = state[4:6].copy(), state[6:8].copy()
        for _ in range(steps):
            for r, v in ((target_r, target_v), (observer_r, observer_v)):
                radius = np.linalg.norm(r)
                a0 = -mu * r / radius ** (p + 1.0)
                r += v * h + 0.5 * a0 * h * h
                radius = np.linalg.norm(r)
                a1 = -mu * r / radius ** (p + 1.0)
                v += 0.5 * (a0 + a1) * h
        return np.concatenate([target_r, target_v, observer_r, observer_v])

    def decode(self, state: np.ndarray, context) -> np.ndarray:
        relative = state[0:2] - state[4:6]
        direction = np.array([relative[0], relative[1], 0.0])
        return direction / np.linalg.norm(direction)


class TrajectoryClockSubmission:
    """The control a local-law evaluation must not mistake for a law.

    The encoder fits a trend-plus-harmonics curve to the unwrapped planar
    angle of the planet direction over the window and stores its coefficients
    in the state next to a clock. The transition rule increments the clock,
    and the decoder evaluates the stored curve. The construction satisfies the
    letter of the interface (fixed-dimensional state, one reusable transition
    rule, no history after initialization), so whatever separates it from a
    dynamical law must come from the evaluation itself. The angle chart is the
    strongest variant found for this family: component-wise fits ring at the
    window edge and score worse everywhere.
    """

    def __init__(self, order: int = 16):
        self.order = order
        # clock + offset + trend + harmonic coefficients of the angle curve.
        self.state_dimension = 1 + 2 + 2 * order

    def _features(self, times: np.ndarray, period: float) -> np.ndarray:
        columns = [np.ones_like(times), times]
        for harmonic in range(1, self.order + 1):
            phase = _TWO_PI * harmonic * times / period
            columns.extend([np.sin(phase), np.cos(phase)])
        return np.column_stack(columns)

    def encode(self, window: Rows):
        times = np.array([float(row["time"]) for row in window])
        angles = np.unwrap(np.arctan2([float(row["planet_dir_y"]) for row in window],
                                      [float(row["planet_dir_x"]) for row in window]))
        period = float(np.ptp(times))
        coefficients, *_ = np.linalg.lstsq(self._features(times, period), angles, rcond=None)
        state = np.concatenate([[times.max()], coefficients])
        return state, period

    def step(self, state: np.ndarray, dt: float, context) -> np.ndarray:
        advanced = state.copy()
        advanced[0] += dt
        return advanced

    def decode(self, state: np.ndarray, context) -> np.ndarray:
        period = context
        angle = float(self._features(np.array([state[0]]), period) @ state[1:])
        return np.array([np.cos(angle), np.sin(angle), 0.0])
