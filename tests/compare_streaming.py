"""Compare the Rust streaming stacker against the reference implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import make_stack  # noqa: E402

MAX_LSB = 20
MEAN_LSB = 0.60


def main() -> int:
    folder = ROOT / "target" / "streamcheck" / "src8"
    if not folder.exists():
        make_stack.build(folder, depth=8)

    import focusweave as rust
    images = [rust.load_image(p) for p in sorted(folder.iterdir())]
    height, width = images[0].shape[:2]

    previews: list[tuple[int, tuple[int, ...]]] = []
    streamer = rust.StreamingFocusStacker(
        reference_size=(width, height),
        on_preview=lambda preview, count: previews.append((count, preview.shape)),
        preview_scale=0.25,
    )
    for image in images:
        streamer.add_image(image)
    got = streamer.finish().image

    for module in [m for m in sys.modules if m.startswith("focusweave")]:
        del sys.modules[module]
    sys.path.insert(0, str(ROOT / "tests" / "reference"))
    from focusweave.streaming_stack import StreamingFocusStacker as Reference

    reference = Reference(reference_size=(width, height))
    for image in images:
        reference.add_image(image)
    expected = reference.finish().image

    print(f"  previews emitted: {[c for c, _ in previews]}")
    print(f"  preview size:     {previews[-1][1]}")
    if got.shape != expected.shape:
        print(f"  FAIL shape {got.shape} != {expected.shape}")
        return 1
    diff = np.abs(got.astype(np.int64) - expected.astype(np.int64))
    worst, mean = int(diff.max()), float(diff.mean())
    print(f"  output {got.shape}  max={worst}  mean={mean:.4f}")
    if worst > MAX_LSB or mean > MEAN_LSB:
        print(f"  FAIL max={worst} (limit {MAX_LSB}) mean={mean:.4f} (limit {MEAN_LSB})")
        return 1
    print("\nstreaming output matches the reference within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
