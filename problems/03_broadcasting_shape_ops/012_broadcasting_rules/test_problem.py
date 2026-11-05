import numpy as np
import pytest

SHAPE_CASES = [
    ((8, 1, 6, 1), (7, 1, 5)),
    ((5, 4), (1,)),
    ((5, 4), (4,)),
    ((15, 3, 5), (15, 1, 5)),
    ((15, 3, 5), (3, 1)),
    ((), (2, 3)),
    ((1, 1), (1,)),
    ((0, 3), (1, 3)),
    ((2, 1), (1, 0)),
]


@pytest.mark.parametrize("s1, s2", SHAPE_CASES)
def test_broadcast_shape_compatible(sol, ban, s1, s2):
    expected = np.broadcast_shapes(s1, s2)
    ban(np, "broadcast_shapes")
    ban(np, "broadcast")
    out = sol.broadcast_shape(s1, s2)
    assert tuple(out) == expected, f"broadcast_shape({s1}, {s2}): expected {expected}, got {out}"
    out = sol.broadcast_shape(s2, s1)
    assert tuple(out) == expected, f"the order of the arguments must not matter ({s2}, {s1})"


@pytest.mark.parametrize("s1, s2", [((3, 4), (3,)), ((2, 3), (3, 2)), ((4, 1, 3), (2, 2)), ((0,), (2,))])
def test_broadcast_shape_incompatible(sol, ban, s1, s2):
    ban(np, "broadcast_shapes")
    ban(np, "broadcast")
    with pytest.raises(ValueError):
        sol.broadcast_shape(s1, s2)


def test_broadcast_shape_returns_tuple(sol):
    assert isinstance(sol.broadcast_shape((2, 1), (3,)), tuple), "return a tuple"


def _no_ufunc_outer(sol, name):
    """np.add.outer etc. can't be monkeypatched (ufunc methods), so check the source instead."""
    import inspect
    src = inspect.getsource(getattr(sol, name))
    for banned in ("add.outer", "subtract.outer", "np.outer", "multiply.outer"):
        assert banned not in src, f"`{banned}` is banned in {name}; use broadcasting"


# ---------------------------------------------------------------- outer_sum

def test_outer_sum_example(sol, ban, no_loops):
    ban(np, "outer")
    out = sol.outer_sum(np.array([0, 10, 20]), np.array([1, 2]))
    np.testing.assert_array_equal(out, [[1, 2], [11, 12], [21, 22]])
    no_loops(sol.outer_sum)
    _no_ufunc_outer(sol, "outer_sum")


def test_outer_sum_shapes_and_dtype(sol):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5).astype(np.float32)
    b = rng.standard_normal(7).astype(np.float32)
    out = sol.outer_sum(a, b)
    assert out.shape == (5, 7), f"expected (5, 7), got {out.shape}"
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.add.outer(a, b))


# ---------------------------------------------------------------- pairwise_diff

def test_pairwise_diff(sol, no_loops):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 3))
    out = sol.pairwise_diff(x)
    assert out.shape == (6, 6, 3), f"expected (n, n, d) = (6, 6, 3), got {out.shape}"
    for i in range(6):
        for j in range(6):
            np.testing.assert_allclose(out[i, j], x[i] - x[j], err_msg=f"out[{i}, {j}] should be x[{i}] - x[{j}]")
    no_loops(sol.pairwise_diff)
    _no_ufunc_outer(sol, "pairwise_diff")


def test_pairwise_diff_example(sol):
    out = sol.pairwise_diff(np.array([[0, 0], [1, 2]]))
    np.testing.assert_array_equal(out[1, 0], [1, 2])
    np.testing.assert_array_equal(out[0, 1], [-1, -2])
    np.testing.assert_array_equal(out[0, 0], [0, 0])


# ---------------------------------------------------------------- pairwise_sq_dist

def test_pairwise_sq_dist_example(sol, no_loops):
    out = sol.pairwise_sq_dist(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert out.shape == (1, 2)
    assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
    np.testing.assert_allclose(out, [[25.0, 1.0]])
    no_loops(sol.pairwise_sq_dist)


def test_pairwise_sq_dist_random(sol):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, 4))
    y = rng.standard_normal((5, 4))
    out = sol.pairwise_sq_dist(x, y)
    assert out.shape == (7, 5), f"expected (n, m) = (7, 5), got {out.shape}"
    expected = np.array([[np.sum((xi - yj) ** 2) for yj in y] for xi in x])
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert np.all(out >= 0), "squared distances can't be negative"


def test_pairwise_sq_dist_int_input(sol):
    out = sol.pairwise_sq_dist(np.array([[0, 0], [1, 1]]), np.array([[2, 2]]))
    assert out.dtype == np.float64, f"int input should still give float64, got {out.dtype}"
    np.testing.assert_allclose(out, [[8.0], [2.0]])
