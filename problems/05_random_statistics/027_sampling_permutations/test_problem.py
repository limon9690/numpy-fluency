import numpy as np
import pytest


# ---------------------------------------------------------------- split_indices

def test_split_indices_recipe(sol, no_loops):
    train, test = sol.split_indices(10, 0.3, np.random.default_rng(0))
    perm = np.random.default_rng(0).permutation(10)
    np.testing.assert_array_equal(test, perm[:3])
    np.testing.assert_array_equal(train, perm[3:])
    no_loops(sol.split_indices)


def test_split_indices_partition(sol):
    train, test = sol.split_indices(101, 0.2, np.random.default_rng(1))
    assert len(test) == 20 and len(train) == 81
    np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(101),
                                  err_msg="train and test together must contain every index exactly once")


# ---------------------------------------------------------------- shuffle_together

def test_shuffle_together_alignment(sol, no_loops):
    x = np.arange(20).reshape(10, 2)
    y = np.arange(10) * 10
    xb, yb = x.copy(), y.copy()
    xs, ys = sol.shuffle_together(x, y, np.random.default_rng(0))
    np.testing.assert_array_equal(xs[:, 0] * 5, ys, err_msg="x and y rows must stay paired")
    assert not np.array_equal(ys, y), "the arrays should actually be shuffled"
    np.testing.assert_array_equal(x, xb, err_msg="inputs must not be modified (rng.shuffle works in place)")
    np.testing.assert_array_equal(y, yb, err_msg="inputs must not be modified")
    no_loops(sol.shuffle_together)


def test_shuffle_together_recipe(sol):
    x = np.arange(8.0)
    y = -np.arange(8)
    xs, ys = sol.shuffle_together(x, y, np.random.default_rng(3))
    perm = np.random.default_rng(3).permutation(8)
    np.testing.assert_array_equal(xs, x[perm])
    np.testing.assert_array_equal(ys, y[perm])


# ---------------------------------------------------------------- sample_distinct

def test_sample_distinct(sol):
    pop = np.arange(100, 200)
    out = sol.sample_distinct(pop, 30, np.random.default_rng(0))
    assert out.shape == (30,)
    assert len(set(out.tolist())) == 30, "all samples must be different (no replacement)"
    assert np.isin(out, pop).all()
    np.testing.assert_array_equal(out, np.random.default_rng(0).choice(pop, size=30, replace=False))


def test_sample_distinct_too_many_raises(sol):
    with pytest.raises(ValueError):
        sol.sample_distinct(np.arange(3), 4, np.random.default_rng(0))


# ---------------------------------------------------------------- weighted_sample

def test_weighted_sample_frequencies(sol, no_loops):
    classes = np.array([10, 20, 30])
    weights = np.array([6, 0, 2])  # raw counts, not probabilities
    out = sol.weighted_sample(classes, weights, 4000, np.random.default_rng(0))
    assert out.shape == (4000,)
    assert not (out == 20).any(), "a class with weight 0 must never be drawn"
    assert abs((out == 10).mean() - 0.75) < 0.03, "class 10 should be drawn about 75% of the time"
    no_loops(sol.weighted_sample)


def test_weighted_sample_recipe(sol):
    w = np.array([1.0, 2.0, 3.0, 4.0])
    out = sol.weighted_sample(np.arange(4), w, 50, np.random.default_rng(9))
    np.testing.assert_array_equal(out, np.random.default_rng(9).choice(np.arange(4), size=50, p=w / w.sum()))
