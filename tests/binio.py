"""Tiny raw array container shared by the Rust dumper and the cv2 reference."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_CODES = {np.dtype(np.uint8): 0, np.dtype(np.float32): 1, np.dtype(np.uint16): 2}
_DTYPES = {v: k for k, v in _CODES.items()}


def write(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    c = arr.shape[2] if arr.ndim == 3 else 1
    with path.open("wb") as f:
        f.write(struct.pack("<IIIB", h, w, c, _CODES[arr.dtype]))
        f.write(arr.tobytes())


def read(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        h, w, c, code = struct.unpack("<IIIB", f.read(13))
        data = np.frombuffer(f.read(), dtype=_DTYPES[code])
    return data.reshape(h, w, c) if c > 1 else data.reshape(h, w)
