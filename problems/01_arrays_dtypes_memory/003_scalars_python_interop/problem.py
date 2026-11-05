"""003 · NumPy scalars and Python interop."""

import numpy as np


def to_jsonable(obj):
    """Recursively convert NumPy values inside `obj` into built-in Python types.

    - dict           -> dict with converted values (keys unchanged)
    - list or tuple  -> list with converted items
    - np.ndarray     -> nested Python lists of Python numbers (or bools)
    - NumPy scalar   -> the matching Python int / float / bool
    - anything else  -> returned unchanged

    After conversion, json.dumps(result) must work.

    Example:
        to_jsonable({"n": np.int64(3), "v": np.array([1.5, 2.0], np.float32), "t": (np.bool_(True),)})
        -> {"n": 3, "v": [1.5, 2.0], "t": [True]}
    """
    raise NotImplementedError


def as_python_number(x):
    """Return `x` as a built-in Python scalar (int, float, bool or complex).

    `x` may be a Python number, a NumPy scalar, a 0-d array, or any array with exactly one
    element (e.g. shape (1,) or (1, 1)). Raise ValueError if it has more or fewer than one element.

    Example:
        as_python_number(np.float32(2.5))       -> 2.5      (type float)
        as_python_number(np.array([[7]]))       -> 7        (type int)
        as_python_number(np.array([1, 2]))      -> ValueError
    """
    raise NotImplementedError


def array_to_json(a: np.ndarray) -> str:
    """Encode an array as a JSON string that keeps its dtype, shape and values.

    Any format is fine as long as json.loads can read it and array_from_json can invert it.
    Must work for bool, integer and float dtypes, 0-d arrays and empty arrays.

    Example:
        array_to_json(np.array([[1, 2], [3, 4]], np.uint8))
        -> '{"dtype": "uint8", "shape": [2, 2], "data": [1, 2, 3, 4]}'   (one possible format)
    """
    raise NotImplementedError


def array_from_json(s: str) -> np.ndarray:
    """Decode a string made by array_to_json back into the identical array.

    The result has the same dtype, the same shape and exactly the same values as the original.

    Example:
        array_from_json(array_to_json(a)) -> equal to `a`, same dtype and shape
    """
    raise NotImplementedError
