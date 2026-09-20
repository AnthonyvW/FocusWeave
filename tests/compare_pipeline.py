"""End-to-end comparison of the Rust pipeline against the Python reference.

Tolerances are in output LSB. Runs without alignment isolate the fusion maths
and agree to within a couple of LSB; runs with alignment inherit a sub-pixel
disagreement in the masked ECC solve, which shows up as a few LSB of
resampling difference on high-frequency synthetic texture.

`--reference 0` is deliberately absent: it chains seven warps in one direction
and lands within 0.05 px of the pipeline's `min_shift` gate, so the two
implementations legitimately disagree about whether three frames are warped at
all. tests/compare_warps.py covers registration accuracy directly instead.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "reference"))

from focusweave import FocusStackConfig, run  # noqa: E402
from focusweave.main import save_image  # noqa: E402
import make_stack  # noqa: E402

WORK = ROOT / "target" / "pipecheck"
BIN = ROOT / "target" / "release" / "focusweave"

CASES: list[tuple[str, int, dict, list[str], float, float]] = [
    # name, depth, python kwargs, cli flags, max LSB, mean LSB
    ("no-align",        8,  dict(no_align=True),                   ["--no-align"],            3, 0.02),
    ("default",         8,  dict(),                                [],                       16, 0.60),
    ("keep-size",       8,  dict(keep_size=True),                  ["--keep-size"],          16, 0.60),
    # Cropping tightens the canvas to the intersection of all extents, so a
    # sub-pixel disagreement in the extents shifts the whole frame slightly.
    ("crop",            8,  dict(crop=True),                       ["--crop"],               24, 1.50),
    ("no-fill",         8,  dict(no_fill=True),                    ["--no-fill"],            16, 0.60),
    ("global-align",    8,  dict(global_align=True),               ["--global-align"],       16, 0.60),
    ("sharpness-1.5",   8,  dict(sharpness=1.5),                   ["--sharpness", "1.5"],   16, 0.60),
    ("levels-3",        8,  dict(levels=3),                        ["--levels", "3"],        16, 0.60),
    ("workers-1",       8,  dict(workers=1),                       ["--workers", "1"],       16, 0.60),
    ("no-rotation",     8,  dict(no_rotation=True),                ["--no-rotation"],        16, 0.60),
    ("no-scale-shear",  8,  dict(no_scale=True, no_shear=True),    ["--no-scale", "--no-shear"], 16, 0.60),
    ("cull",            8,  dict(cull=0.35),                       ["--cull", "0.35"],       16, 0.60),
    ("slab",            8,  dict(slab=(4, 2)),                     ["--slab", "4", "2"],     16, 0.60),
    ("slab-recursive",  8,  dict(slab=(3, 1), recursive_slab=True), ["--slab", "3", "1", "--recursive-slab"], 16, 0.60),
    ("16bit-no-align", 16,  dict(no_align=True),                   ["--no-align"],          768, 5.0),
    ("16bit-default",  16,  dict(),                                [],                     4096, 160.0),
]


def prepare(depth: int) -> Path:
    folder = WORK / f"src{depth}"
    if not folder.exists():
        make_stack.build(folder, depth=depth)
    return folder


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        print(f"missing {BIN}; run: cargo build --release -p focusweave-cli")
        return 1

    failures: list[str] = []
    for name, depth, kwargs, flags, tol_max, tol_mean in CASES:
        folder = prepare(depth)
        ext = "png" if depth == 8 else "tiff"
        py_out = WORK / f"py_{name}.{ext}"
        rs_out = WORK / f"rs_{name}.{ext}"

        result = run(FocusStackConfig(images=folder, **kwargs))
        save_image(result.image, py_out, 95)
        subprocess.run([str(BIN), str(folder), "--output", str(rs_out), *flags],
                       check=True, capture_output=True)

        a = cv2.imread(str(py_out), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(rs_out), cv2.IMREAD_UNCHANGED)
        if a.shape != b.shape:
            failures.append(f"{name}: shape {b.shape} != {a.shape}")
            print(f"  FAIL {name:16s} shape {b.shape} != {a.shape}")
            continue
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
        worst, mean = int(d.max()), float(d.mean())
        ok = worst <= tol_max and mean <= tol_mean
        print(f"  {'ok  ' if ok else 'FAIL'} {name:16s} {a.shape[1]}x{a.shape[0]}  max={worst:<6} mean={mean:.4f}")
        if not ok:
            failures.append(f"{name}: max={worst} mean={mean:.4f}")

    print()
    if failures:
        for f in failures:
            print(f"  {f}")
        return 1
    print("pipeline output matches the Python reference within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
