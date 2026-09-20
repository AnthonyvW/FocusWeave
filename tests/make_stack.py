"""Generate a synthetic focus stack: one scene, a sweeping focal plane."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def build(out: Path, n: int = 8, w: int = 360, h: int = 260, depth: int = 8,
          jitter: float = 1.4, seed: int = 4242) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # High-frequency texture, so sharpness differences are unambiguous.
    noise = rng.integers(0, 256, size=(h, w, 3)).astype(np.float32)
    rings = 128 + 90 * np.sin(np.hypot(xx - w / 2, yy - h / 2) / 4.0)
    scene = np.clip(0.55 * noise + 0.45 * rings[:, :, None], 0, 255).astype(np.float32)
    for _ in range(18):
        cx, cy = rng.integers(20, w - 20), rng.integers(20, h - 20)
        colour = rng.integers(0, 256, size=3).astype(np.float32)
        cv2.circle(scene, (int(cx), int(cy)), int(rng.integers(6, 22)), colour.tolist(), -1)

    # A depth ramp across the frame: each frame brings one band into focus.
    depth_map = (xx / w + 0.25 * yy / h) / 1.25

    for i in range(n):
        focal = (i + 0.5) / n
        blur = np.abs(depth_map - focal) * 9.0
        frame = np.zeros_like(scene)
        # Composite a handful of blur levels and pick per pixel.
        levels = [0.0, 0.8, 1.6, 2.6, 4.0]
        stack = [scene if s == 0 else cv2.GaussianBlur(scene, (0, 0), s) for s in levels]
        idx = np.digitize(blur, levels) - 1
        idx = np.clip(idx, 0, len(levels) - 1)
        for k in range(len(levels)):
            m = idx == k
            frame[m] = stack[k][m]
        shift = rng.normal(0, jitter, 2)
        m = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
        frame = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT)
        if depth == 16:
            arr = np.clip(frame * 257.0, 0, 65535).astype(np.uint16)
            ext = "tiff"
        else:
            arr = np.clip(frame, 0, 255).astype(np.uint8)
            ext = "png"
        cv2.imwrite(str(out / f"frame_{i:02d}.{ext}"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    target = Path(sys.argv[1])
    build(target, depth=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    print(f"wrote {len(list(target.iterdir()))} frames to {target}")
