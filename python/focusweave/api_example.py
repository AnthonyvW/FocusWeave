"""Worked examples of the focusweave library API.

Run directly to stack a folder:

    python -m focusweave.api_example path/to/images/ --output result.tiff

The streaming example shows how to feed frames in as they are captured:

    python -m focusweave.api_example path/to/images/ --streaming
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from focusweave import (
    IMAGE_EXTENSIONS,
    FocusStackConfig,
    Interrupted,
    RunResult,
    StreamingFocusStacker,
    load_image,
    run,
    save_image,
)


def load_folder(folder: Path) -> list[np.ndarray]:
    """Load every image in folder as an RGB ndarray at its native bit depth.

    Mixed-depth folders are not supported; all images should share a depth.
    """
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in '{folder}', found {len(paths)}.")
    return [load_image(p) for p in paths]


def stack_folder(folder: Path, output: Path, workers: int = 3) -> RunResult:
    """Stack a folder of images, reporting progress as it goes."""

    def on_progress(fraction: float, stage: str, message: str) -> None:
        if message:
            print(f"  [{stage:>9}] {fraction * 100:5.1f}%  {message}")

    cfg = FocusStackConfig(images=folder, workers=workers)
    started = time.perf_counter()
    result = run(cfg, progress=on_progress)
    print(f"Stacked in {time.perf_counter() - started:.2f}s")

    save_image(result.image, output)
    print(f"Saved: {output}")
    return result


def stack_arrays(folder: Path, output: Path) -> RunResult:
    """Stack images that are already in memory rather than on disk."""
    images = load_folder(folder)
    print(f"Loaded {len(images)} frames of {images[0].shape} ({images[0].dtype})")
    result = run(FocusStackConfig(images=images, workers=4))
    save_image(result.image, output)
    print(f"Saved: {output}")
    return result


def stack_streaming(folder: Path, output: Path) -> RunResult:
    """Feed frames in one at a time, as a capture loop would."""
    images = load_folder(folder)
    height, width = images[0].shape[:2]

    def on_preview(preview: np.ndarray, count: int) -> None:
        print(f"  preview after {count} frame(s): {preview.shape}")

    stacker = StreamingFocusStacker(
        reference_size=(width, height),
        on_preview=on_preview,
        preview_scale=0.25,
    )
    for image in images:
        stacker.add_image(image)

    result = stacker.finish()
    save_image(result.image, output)
    print(f"Saved: {output}")
    return result


def stack_cancellable(folder: Path, deadline_seconds: float) -> RunResult | None:
    """Stop a long run from another thread, or on a deadline."""
    started = time.perf_counter()
    cfg = FocusStackConfig(
        images=folder,
        interrupt=lambda: time.perf_counter() - started > deadline_seconds,
    )
    try:
        return run(cfg)
    except Interrupted:
        print(f"Stack cancelled after {deadline_seconds}s.")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder containing input images.")
    parser.add_argument("--output", type=Path, default=None, help="Output file path.")
    parser.add_argument("--arrays", action="store_true", help="Pass pre-loaded arrays instead of a folder.")
    parser.add_argument("--streaming", action="store_true", help="Use the incremental streaming stacker.")
    parser.add_argument("--workers", type=int, default=3, help="Parallel stacking workers.")
    args = parser.parse_args()

    output = args.output or args.folder / "stacked.jpg"
    if args.streaming:
        stack_streaming(args.folder, output)
    elif args.arrays:
        stack_arrays(args.folder, output)
    else:
        stack_folder(args.folder, output, args.workers)


if __name__ == "__main__":
    main()
