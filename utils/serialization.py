from __future__ import annotations

import json
from typing import Any

import numpy as np


def array_to_blob(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.float32).tobytes()


def blob_to_array(blob: bytes, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.frombuffer(blob, dtype=np.float32).copy()
    if shape is not None:
        array = array.reshape(shape)
    return array


def json_dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=True)


def json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default

