import json

import numpy as np
import pytest


def _assert_builtin(value, path="result"):
    """Walk a converted structure and fail if any NumPy type is left inside."""
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_builtin(v, f"{path}[{k!r}]")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _assert_builtin(v, f"{path}[{i}]")
    else:
        assert type(value) in (int, float, bool, str, type(None), complex), (
            f"{path} is {type(value).__name__}, expected a built-in Python type"
        )


# ---------------------------------------------------------------- to_jsonable

def test_to_jsonable_scalars_and_arrays(sol):
    obj = {
        "n": np.int64(3),
        "f": np.float32(1.5),
        "b": np.bool_(True),
        "v": np.array([1.5, 2.0], dtype=np.float32),
        "m": np.arange(6, dtype=np.uint8).reshape(2, 3),
    }
    out = sol.to_jsonable(obj)
    _assert_builtin(out)
    assert out == {"n": 3, "f": 1.5, "b": True, "v": [1.5, 2.0], "m": [[0, 1, 2], [3, 4, 5]]}
    json.dumps(out)  # must not raise


def test_to_jsonable_nested_and_tuples(sol):
    obj = {"outer": [{"x": np.float64(0.25)}, (np.int8(-1), "label", None)], "keep": 7}
    out = sol.to_jsonable(obj)
    _assert_builtin(out)
    assert out == {"outer": [{"x": 0.25}, [-1, "label", None]], "keep": 7}, (
        "tuples should become lists and nested values should be converted"
    )


def test_to_jsonable_top_level_array_and_input_untouched(sol):
    a = np.array([[True, False]])
    out = sol.to_jsonable(a)
    assert out == [[True, False]]
    _assert_builtin(out)
    assert isinstance(a, np.ndarray) and a.dtype == np.bool_, "the input must not be changed"
    out = sol.to_jsonable(np.array(5.0))
    assert out == 5.0 and type(out) is float, "a 0-d array should become a plain Python number"


# ---------------------------------------------------------------- as_python_number

@pytest.mark.parametrize(
    "x, expected, kind",
    [
        (np.float32(2.5), 2.5, float),
        (np.int64(7), 7, int),
        (np.bool_(False), False, bool),
        (np.array(3.0), 3.0, float),
        (np.array([[7]], dtype=np.int16), 7, int),
        (np.array([1.25]), 1.25, float),
        (4, 4, int),
        (0.5, 0.5, float),
    ],
)
def test_as_python_number(sol, x, expected, kind):
    out = sol.as_python_number(x)
    assert type(out) is kind, f"expected a Python {kind.__name__}, got {type(out).__name__}"
    assert out == expected


def test_as_python_number_wrong_size(sol):
    with pytest.raises(ValueError):
        sol.as_python_number(np.array([1, 2]))
    with pytest.raises(ValueError):
        sol.as_python_number(np.zeros((0,)))


# ---------------------------------------------------------------- array_to_json / array_from_json

ROUND_TRIP_CASES = [
    np.array([[1, 2], [3, 4]], dtype=np.uint8),
    np.random.default_rng(0).standard_normal((3, 4)),
    np.random.default_rng(1).standard_normal(5).astype(np.float32),
    np.array([True, False, True]),
    np.array([-5, 0, 2**40], dtype=np.int64),
    np.array(3.25),
    np.zeros((0, 3), dtype=np.int16),
    np.arange(24, dtype=np.int32).reshape(2, 3, 4),
]


@pytest.mark.parametrize("a", ROUND_TRIP_CASES, ids=lambda a: f"{a.dtype}{a.shape}")
def test_json_round_trip(sol, a):
    s = sol.array_to_json(a)
    assert isinstance(s, str), "array_to_json should return a string"
    json.loads(s)  # must be valid JSON
    back = sol.array_from_json(s)
    assert isinstance(back, np.ndarray), "array_from_json should return an ndarray"
    assert back.dtype == a.dtype, f"dtype changed: {a.dtype} -> {back.dtype}"
    assert back.shape == a.shape, f"shape changed: {a.shape} -> {back.shape}"
    np.testing.assert_array_equal(back, a, err_msg="values changed in the round trip")


def test_json_float_values_exact(sol):
    a = np.array([0.1, 1 / 3, 1e-300, -2.5e10])
    back = sol.array_from_json(sol.array_to_json(a))
    assert np.array_equal(back, a), "float64 values must round-trip exactly, not approximately"
