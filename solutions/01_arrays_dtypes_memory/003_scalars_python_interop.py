"""003 · NumPy scalars and Python interop (reference solution)."""

import json

import numpy as np


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # recursive, and every leaf becomes a Python scalar
    if isinstance(obj, np.generic):  # base class of all NumPy scalars
        return obj.item()
    return obj


def as_python_number(x):
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"expected exactly one element, got {arr.size}")
    return arr.item()  # works for any shape with one element, unlike float()/int()


def array_to_json(a: np.ndarray) -> str:
    # ravel() so 0-d arrays give [x] and empty arrays give []; the shape restores the rest.
    return json.dumps({"dtype": str(a.dtype), "shape": list(a.shape), "data": a.ravel().tolist()})


def array_from_json(s: str) -> np.ndarray:
    d = json.loads(s)
    return np.array(d["data"], dtype=d["dtype"]).reshape(d["shape"])
