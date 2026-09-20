"""Cross-check the Rust primitives against the OpenCV routines they replace."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import binio

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "target" / "primcheck"
WORK.mkdir(parents=True, exist_ok=True)

K1D = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
K1D_X2 = K1D * 2

rng = np.random.default_rng(20260920)
h, w = 97, 131
base = rng.integers(0, 256, size=(h, w, 3)).astype(np.uint8)
# Blend in smooth structure so gradient-based operators see real signal.
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
smooth = (
    127 + 100 * np.sin(xx / 7.0) * np.cos(yy / 11.0)
)[:, :, None] * np.array([1.0, 0.85, 0.6], dtype=np.float32)
rgb = np.clip(0.75 * smooth + 0.25 * base, 0, 255).astype(np.uint8)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

binio.write(WORK / "in_rgb.bin", rgb)
binio.write(WORK / "in_gray.bin", gray)

subprocess.run(
    ["cargo", "run", "-q", "--release", "-p", "focusweave-core",
     "--example", "dump_primitives", "--", str(WORK)],
    cwd=ROOT, check=True,
)

grayf = gray.astype(np.float32)
failures: list[str] = []


def check(name: str, expected: np.ndarray, tol: float, rel_to: float | None = None) -> None:
    got = binio.read(WORK / f"out_{name}.bin")
    expected = np.asarray(expected)
    if got.shape != expected.shape:
        failures.append(f"{name}: shape {got.shape} != {expected.shape}")
        return
    diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    scale = rel_to if rel_to is not None else 1.0
    worst = float(diff.max()) / scale
    mean = float(diff.mean()) / scale
    status = "ok " if worst <= tol else "FAIL"
    if worst > tol:
        failures.append(f"{name}: max diff {worst:.6g} > {tol:g}")
    print(f"  {status} {name:22s} max={worst:.6g} mean={mean:.6g}")


print("\nfilters")
check("sep_reflect", cv2.sepFilter2D(grayf, cv2.CV_32F, K1D, K1D, borderType=cv2.BORDER_REFLECT), 1e-3)
check("sep_rgb", cv2.sepFilter2D(rgb.astype(np.float32), cv2.CV_32F, K1D_X2, K1D_X2,
                                 borderType=cv2.BORDER_REFLECT), 1e-2)
check("box3", cv2.boxFilter(grayf, cv2.CV_32F, (3, 3), borderType=cv2.BORDER_REFLECT), 1e-3)
check("box8", cv2.boxFilter(grayf, cv2.CV_32F, (8, 8), borderType=cv2.BORDER_REFLECT), 1e-3)
check("sqrbox3", cv2.sqrBoxFilter(grayf, cv2.CV_32F, (3, 3), normalize=True,
                                  borderType=cv2.BORDER_REFLECT), 1e-1)
check("gauss31", cv2.GaussianBlur(grayf, (31, 31), 0), 1e-3)
check("gauss15", cv2.GaussianBlur(grayf, (15, 15), 0), 1e-3)
check("gauss3", cv2.GaussianBlur(grayf, (3, 3), 0), 1e-3)
check("gauss5", cv2.GaussianBlur(grayf, (5, 5), 0), 1e-3)
check("sobel_x", cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=5), 1e-2)
check("sobel_y", cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=5), 1e-2)
check("laplacian", cv2.Laplacian(grayf, cv2.CV_32F, ksize=3), 1e-2)

print("\nmorphology and colour")
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = np.where(gray > 128, 255, 0).astype(np.uint8)
check("dilate", cv2.dilate(mask, se), 0)
# OpenCV 5 resolves exact .5 ties in its SIMD path differently to the
# integer formula; a 1 LSB spread on ties is expected.
check("gray_from_rgb", gray, 1)
check("lab_l", cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[:, :, 0], 1)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
check("clahe", clahe.apply(gray), 1)

print("\nresize")
check("resize_small", cv2.resize(rgb, (41, 29), interpolation=cv2.INTER_AREA), 1)
check("resize_f32", cv2.resize(grayf, (53, 37), interpolation=cv2.INTER_AREA), 1e-3)

print("\nwarp")
m = np.array([[1.004, -0.013, 7.35], [0.011, 0.997, -4.2]], dtype=np.float32)
check("warp_cubic", cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REFLECT), 3)
t = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, -3.0]], dtype=np.float32)
check("warp_translate", cv2.warpAffine(rgb, t, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT), 0)

print()
if failures:
    print(f"{len(failures)} mismatch(es):")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("all primitives match the OpenCV reference")
