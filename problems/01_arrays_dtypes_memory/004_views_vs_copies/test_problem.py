import numpy as np
import pytest


# ---------------------------------------------------------------- memory_relation

def test_memory_relation_same(sol):
    a = np.arange(6)
    assert sol.memory_relation(a, a) == "same"
    assert sol.memory_relation(a, np.asarray(a)) == "same", "np.asarray returns the same object"


def test_memory_relation_shared(sol):
    a = np.arange(12).reshape(3, 4)
    for b, what in [(a[1:], "a[1:]"), (a.T, "a.T"), (a[:, 2], "a[:, 2]"),
                    (a.reshape(4, 3), "a.reshape(4, 3)"), (a.view(), "a.view()")]:
        assert sol.memory_relation(a, b) == "shared", f"{what} is a view of a, expected 'shared'"
    assert sol.memory_relation(a[0], a[0, 1:3]) == "shared", "two views that overlap are 'shared'"


def test_memory_relation_independent(sol):
    a = np.arange(12).reshape(3, 4)
    for b, what in [(a.copy(), "a.copy()"), (a[[0, 1]], "a[[0, 1]] (fancy)"),
                    (a[a > 3], "a[a > 3] (mask)"), (a.astype(np.float64), "a.astype(float)"),
                    (np.arange(12), "an unrelated array")]:
        assert sol.memory_relation(a, b) == "independent", f"{what} is a copy, expected 'independent'"


def test_memory_relation_interleaved(sol):
    a = np.arange(10)
    assert sol.memory_relation(a[::2], a[1::2]) == "independent", (
        "a[::2] and a[1::2] never touch the same element; use np.shares_memory, not may_share_memory"
    )


# ---------------------------------------------------------------- first_column_copy

def test_first_column_copy_values(sol):
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    col = sol.first_column_copy(a)
    assert col.shape == (2,), f"expected a 1-D array of shape (2,), got {col.shape}"
    assert col.dtype == np.int16, f"dtype should stay int16, got {col.dtype}"
    np.testing.assert_array_equal(col, [1, 4])


def test_first_column_copy_is_independent(sol):
    a = np.arange(12.0).reshape(4, 3)
    col = sol.first_column_copy(a)
    assert not np.shares_memory(col, a), "the column must not share memory with `a` (a[:, 0] is a view)"
    col[:] = -1
    np.testing.assert_array_equal(a[:, 0], [0, 3, 6, 9], err_msg="modifying the result changed `a`")


# ---------------------------------------------------------------- standardize_inplace

def test_standardize_inplace_modifies_caller(sol):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    original_id = id(x)
    ret = sol.standardize_inplace(x)
    assert ret is None, "standardize_inplace should return None (it works in place)"
    assert id(x) == original_id
    expected = (np.array([1.0, 2.0, 3.0, 4.0]) - 2.5) / np.std([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(x, expected, err_msg="the caller's array was not standardized in place")


def test_standardize_inplace_on_view(sol):
    rng = np.random.default_rng(0)
    big = rng.normal(5, 3, size=(50, 3))
    before = big.copy()
    sol.standardize_inplace(big[:, 1])
    col = big[:, 1]
    assert abs(col.mean()) < 1e-12 and abs(col.std() - 1) < 1e-12, "the viewed column should now have mean 0, std 1"
    np.testing.assert_array_equal(big[:, [0, 2]], before[:, [0, 2]], err_msg="other columns must not change")


def test_standardize_inplace_float32_2d(sol):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    sol.standardize_inplace(x)
    assert x.dtype == np.float32
    expected = (np.arange(12.0) - 5.5) / np.arange(12.0).std()
    np.testing.assert_allclose(x.ravel(), expected, rtol=1e-5)


def test_standardize_inplace_constant(sol):
    x = np.full(5, 3.0)
    sol.standardize_inplace(x)
    np.testing.assert_array_equal(x, np.zeros(5), err_msg="constant input should become all zeros (no NaN)")


def test_standardize_inplace_int_raises(sol):
    with pytest.raises(TypeError):
        sol.standardize_inplace(np.array([1, 2, 3]))


# ---------------------------------------------------------------- zero_small_in_rows

def test_zero_small_in_rows_example(sol):
    a = np.array([[-1, 5], [-2, 6], [-3, 7]])
    ret = sol.zero_small_in_rows(a, [0, 2], 0)
    assert ret is None, "zero_small_in_rows should return None (it works in place)"
    np.testing.assert_array_equal(a, [[0, 5], [-2, 6], [0, 7]],
                                  err_msg="`a` was not changed; chained indexing like a[rows][mask] = 0 writes to a copy")


def test_zero_small_in_rows_random(sol):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 5))
    expected = a.copy()
    rows = [4, 1, 5]
    for r in rows:
        expected[r][expected[r] < 0.3] = 0
    sol.zero_small_in_rows(a, rows, 0.3)
    np.testing.assert_array_equal(a, expected)
