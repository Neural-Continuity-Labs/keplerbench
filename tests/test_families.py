from keplerbench.synthetic import FAMILIES, generate_observations, sample_world


def test_family_draw_is_deterministic_and_within_ranges():
    for family, ranges in FAMILIES.items():
        first = sample_world(family, seed=11)
        second = sample_world(family, seed=11)
        assert first == second
        assert first.world_id == f"{family}-0011"
        for name, (low, high) in ranges.items():
            assert low <= getattr(first, name) <= high
        assert sample_world(family, seed=12) != first


def test_family_world_generates_public_rows():
    config = sample_world("hidden-exponent", seed=5, sample_count=31)
    rows = generate_observations(config, seed=5)
    assert len(rows) == 31
    assert rows[0]["world_id"] == "hidden-exponent-0005"
    assert "force_exponent" not in rows[0]


def test_family_draws_sampling_and_noise():
    for family in FAMILIES:
        config = sample_world(family, seed=3)
        low, high = FAMILIES[family]["sample_count"]
        assert isinstance(config.sample_count, int) and low <= config.sample_count <= high
        low, high = FAMILIES[family]["angular_noise_std"]
        assert low <= config.angular_noise_std <= high
        assert sample_world(family, seed=3, angular_noise_std=0.0).angular_noise_std == 0.0
