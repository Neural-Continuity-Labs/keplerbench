"""Synthetic observer-centred worlds for KeplerBench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np


@dataclass(frozen=True)
class WorldConfig:
    """Parameters for one planar central-force world.

    The public observations expose directions only. These parameters are saved
    separately for generator auditability and must not be supplied to learners.
    """

    world_id: str = "synthetic-000"
    system_id: str = "system-000"
    observer_id: str = "observer-000"
    target_id: str = "planet-000"
    gravitational_parameter: float = 1.0
    force_exponent: float = 2.0
    target_radius: float = 1.7
    target_eccentricity: float = 0.2
    observer_radius: float = 1.0
    target_phase: float = 0.4
    observer_phase: float = 0.0
    duration: float = 20.0
    sample_count: int = 401
    integration_substeps: int = 12
    angular_noise_std: float = 0.0
    train_fraction: float = 0.7

    def validate(self) -> None:
        if self.gravitational_parameter <= 0:
            raise ValueError("gravitational_parameter must be positive")
        if self.force_exponent <= 0:
            raise ValueError("force_exponent must be positive")
        if self.target_radius <= 0 or self.observer_radius <= 0:
            raise ValueError("orbital radii must be positive")
        if not 0 <= self.target_eccentricity < 1:
            raise ValueError("target_eccentricity must lie in [0, 1)")
        if self.duration <= 0 or self.sample_count < 3:
            raise ValueError("duration must be positive and sample_count >= 3")
        if self.integration_substeps < 1:
            raise ValueError("integration_substeps must be at least 1")
        if self.angular_noise_std < 0:
            raise ValueError("angular_noise_std cannot be negative")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must lie in (0, 1)")


def _acceleration(position: np.ndarray, mu: float, exponent: float) -> np.ndarray:
    radius = np.linalg.norm(position)
    if radius == 0:
        raise FloatingPointError("body reached the central singularity")
    return -mu * position / radius ** (exponent + 1.0)


def _initial_state(radius: float, eccentricity: float, phase: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    # Start at periapsis, then rotate the full state. The speed is exact for an
    # inverse-square world and remains a documented initial condition otherwise.
    periapsis = radius * (1.0 - eccentricity)
    speed = np.sqrt(mu * (1.0 + eccentricity) / (radius * (1.0 - eccentricity)))
    c, s = np.cos(phase), np.sin(phase)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ np.array([periapsis, 0.0]), rotation @ np.array([0.0, speed])


def _integrate(position: np.ndarray, velocity: np.ndarray, times: np.ndarray, substeps: int, mu: float, exponent: float) -> np.ndarray:
    trajectory = np.empty((len(times), 2), dtype=float)
    trajectory[0] = position
    for index in range(1, len(times)):
        step = (times[index] - times[index - 1]) / substeps
        for _ in range(substeps):
            a0 = _acceleration(position, mu, exponent)
            position = position + velocity * step + 0.5 * a0 * step * step
            a1 = _acceleration(position, mu, exponent)
            velocity = velocity + 0.5 * (a0 + a1) * step
        trajectory[index] = position
    return trajectory


def _unit3(vectors: np.ndarray) -> np.ndarray:
    padded = np.column_stack([vectors, np.zeros(len(vectors))])
    return padded / np.linalg.norm(padded, axis=1, keepdims=True)


def _add_angular_noise(vectors: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std == 0:
        return vectors
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) + rng.normal(0.0, std, len(vectors))
    return np.column_stack([np.cos(angles), np.sin(angles), np.zeros(len(vectors))])


def generate_observations(config: WorldConfig, seed: int = 0) -> list[dict[str, object]]:
    """Generate public observer-centred observations for one world."""

    config.validate()
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, config.duration, config.sample_count)
    target_state = _initial_state(config.target_radius, config.target_eccentricity, config.target_phase, config.gravitational_parameter)
    observer_state = _initial_state(config.observer_radius, 0.0, config.observer_phase, config.gravitational_parameter)
    target = _integrate(*target_state, times, config.integration_substeps, config.gravitational_parameter, config.force_exponent)
    observer = _integrate(*observer_state, times, config.integration_substeps, config.gravitational_parameter, config.force_exponent)
    planet_direction = _add_angular_noise(_unit3(target - observer), config.angular_noise_std, rng)
    star_direction = _add_angular_noise(_unit3(-observer), config.angular_noise_std, rng)
    train_end = int(round(config.train_fraction * len(times)))
    rows: list[dict[str, object]] = []
    for i, time in enumerate(times):
        rows.append({
            "world_id": config.world_id,
            "system_id": config.system_id,
            "observer_id": config.observer_id,
            "target_id": config.target_id,
            "time": float(time),
            "planet_dir_x": float(planet_direction[i, 0]),
            "planet_dir_y": float(planet_direction[i, 1]),
            "planet_dir_z": float(planet_direction[i, 2]),
            "star_dir_x": float(star_direction[i, 0]),
            "star_dir_y": float(star_direction[i, 1]),
            "star_dir_z": float(star_direction[i, 2]),
            "split": "train" if i < train_end else "test",
        })
    return rows


def write_dataset(rows: list[dict[str, object]], output: Path, config: WorldConfig, seed: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"generator": "keplerbench", "generator_version": "0.1.0", "seed": seed, "latent_world_parameters": asdict(config), "warning": "Do not expose latent_world_parameters to benchmark learners."}
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
