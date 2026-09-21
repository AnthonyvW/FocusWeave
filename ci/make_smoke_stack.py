"""Write a tiny focus stack for CI smoke tests, using only numpy and Pillow.

    python ci/make_smoke_stack.py OUTPUT_DIR [FRAMES]

Deliberately avoids OpenCV: these frames are what proves a downloaded binary or
an installed wheel carries its own copy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

W, H = 320, 240


def main() -> int:
    out = Path(sys.argv[1])
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    scene = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)

    for i in range(frames):
        # Each frame is sharp in one horizontal band and blurred elsewhere, so
        # a correct stack is sharp everywhere and a broken one is not.
        focus = (i + 0.5) / frames * H
        blurred = np.asarray(Image.fromarray(scene).filter(ImageFilter.GaussianBlur(3)))
        rows = np.arange(H)[:, None, None]
        weight = np.clip(1.0 - np.abs(rows - focus) / (H / frames), 0.0, 1.0)
        frame = (scene * weight + blurred * (1.0 - weight)).astype(np.uint8)
        Image.fromarray(frame).save(out / f"frame_{i:02d}.png")

    print(f"wrote {frames} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
