from keplerbench.synthetic import WorldConfig, generate_observations


def test_generator_is_deterministic_and_public():
    config = WorldConfig(sample_count=25, duration=2.0)
    first = generate_observations(config, seed=4)
    second = generate_observations(config, seed=4)
    assert first == second
    assert len(first) == 25
    forbidden = {"position", "velocity", "eccentricity", "force_exponent", "gravitational_parameter"}
    assert forbidden.isdisjoint(first[0])
    assert {row["split"] for row in first} == {"train", "test"}
