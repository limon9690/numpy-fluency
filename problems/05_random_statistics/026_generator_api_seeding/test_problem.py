import numpy as np
import pytest


def _global_state_snapshot():
    state = np.random.get_state()
    return state[1].copy(), state[2]


# ---------------------------------------------------------------- noisy_signal

def test_noisy_signal_matches_recipe(sol):
    out = sol.noisy_signal(100, 0.1, 7)
    rng = np.random.default_rng(7)
    expected = np.sin(np.linspace(0, 2 * np.pi, 100)) + rng.normal(0, 0.1, size=100)
    assert out.shape == (100,) and out.dtype == np.float64
    np.testing.assert_array_equal(out, expected, err_msg="follow the recipe: default_rng(seed), then rng.normal once")


def test_noisy_signal_reproducible_and_seed_dependent(sol):
    a = sol.noisy_signal(50, 0.5, 1)
    np.random.seed(123)  # the legacy global seed must not matter
    b = sol.noisy_signal(50, 0.5, 1)
    c = sol.noisy_signal(50, 0.5, 2)
    np.testing.assert_array_equal(a, b, err_msg="same seed must give the same signal")
    assert not np.array_equal(a, c), "different seeds should give different noise"


def test_noisy_signal_zero_noise(sol):
    np.testing.assert_allclose(sol.noisy_signal(5, 0.0, 0), np.sin(np.linspace(0, 2 * np.pi, 5)))


# ---------------------------------------------------------------- dropout_mask

def test_dropout_mask_recipe(sol):
    out = sol.dropout_mask((20, 30), 0.3, 5)
    assert out.dtype == np.bool_ and out.shape == (20, 30)
    np.testing.assert_array_equal(out, np.random.default_rng(5).random((20, 30)) >= 0.3)
    assert 0.6 < out.mean() < 0.8, "about 70% of entries should be kept for p=0.3"


def test_dropout_mask_leaves_global_state_alone(sol):
    np.random.seed(0)
    before = _global_state_snapshot()
    sol.dropout_mask((10,), 0.5, 1)
    after = _global_state_snapshot()
    assert np.array_equal(before[0], after[0]) and before[1] == after[1], (
        "dropout_mask changed the legacy global random state; use a Generator"
    )


# ---------------------------------------------------------------- spawn_streams

def test_spawn_streams_recipe(sol):
    streams = sol.spawn_streams(42, 3)
    assert isinstance(streams, list) and len(streams) == 3
    assert all(isinstance(g, np.random.Generator) for g in streams), "every stream should be a Generator"
    expected = [np.random.default_rng(s).random(4) for s in np.random.SeedSequence(42).spawn(3)]
    for k, (g, e) in enumerate(zip(streams, expected)):
        np.testing.assert_array_equal(g.random(4), e, err_msg=f"stream {k} doesn't match SeedSequence(seed).spawn")


def test_spawn_streams_independent_and_reproducible(sol):
    first = [g.random() for g in sol.spawn_streams(0, 5)]
    again = [g.random() for g in sol.spawn_streams(0, 5)]
    assert first == again, "the same seed must give the same streams"
    assert len(set(first)) == 5, "streams should differ from each other"


# ---------------------------------------------------------------- bootstrap_means

def test_bootstrap_means_recipe(sol, no_loops):
    x = np.arange(10.0)
    out = sol.bootstrap_means(x, 200, np.random.default_rng(3))
    rng = np.random.default_rng(3)
    expected = x[rng.integers(0, 10, size=(200, 10))].mean(axis=1)
    assert out.shape == (200,) and out.dtype == np.float64
    np.testing.assert_allclose(out, expected, err_msg="draw all indices in one rng.integers call")
    no_loops(sol.bootstrap_means)


def test_bootstrap_means_uses_given_rng(sol):
    rng = np.random.default_rng(0)
    a = sol.bootstrap_means(np.arange(5.0), 3, rng)
    b = sol.bootstrap_means(np.arange(5.0), 3, rng)
    assert not np.array_equal(a, b), "calling twice with the same Generator should continue its stream"
    assert abs(sol.bootstrap_means(np.arange(5.0), 2000, np.random.default_rng(1)).mean() - 2.0) < 0.05
