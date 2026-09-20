"""Command line entry point.

Argument parsing and the run itself live in the Rust core, so the `focusweave`
console script and the standalone binary accept exactly the same flags.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from focusweave import _core


def load_image(path: Path) -> np.ndarray:
    """Read an image file as an RGB ndarray at its native bit depth.

    8-bit sources come back as uint8 and 16-bit sources as uint16.
    """
    return _core.read_image(str(path))


def save_image(img: np.ndarray, path: Path, quality: int = 95) -> None:
    """Save a uint8 or uint16 RGB ndarray to path.

    JPEG and WebP do not support 16-bit depth; uint16 images are reduced to
    uint8 before being written to those formats. All other formats retain full
    16-bit depth when the array is uint16.
    """
    _core.write_image(np.ascontiguousarray(img), str(path), quality)


def main() -> None:
    raise SystemExit(_core.cli_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
