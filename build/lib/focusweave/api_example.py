from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from focusweave.focus_stack import IMAGE_EXTENSIONS, FocusStackConfig, RunResult, run


def load_folder(folder: Path) -> list[np.ndarray]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in '{folder}', found {len(paths)}.")
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def save_image(img: np.ndarray, path: Path, quality: int = 95) -> None:
    fmt = path.suffix.lower().lstrip(".")
    fmt = "jpeg" if fmt in ("jpg", "jpeg") else fmt.upper()
    save_kwargs: dict = {"quality": quality} if fmt == "jpeg" else {}
    Image.fromarray(img).save(path, fmt, **save_kwargs)


def stack(
    folder: Path,
    output: Path | None = None,
    workers: int = 3,
    quality: int = 95,
    slab: tuple[int, int] | None = None,
    output_steps: bool = False,
    only_slab: bool = False,
    recursive_slab: bool = False,
    slab_format: str | None = None,
) -> RunResult:
    out_path = output if output is not None else folder / "stacked.jpg"

    emit_steps = output_steps or only_slab
    steps_dir = out_path.parent / "focusweave_slabs" if emit_steps else None
    final_ext = slab_format.lstrip(".") if slab_format else "tiff"

    def _on_slab(label: str, array: np.ndarray) -> None:
        assert steps_dir is not None
        steps_dir.mkdir(parents=True, exist_ok=True)
        slab_file = steps_dir / f"{label}.{final_ext}"
        t = time.perf_counter()
        save_image(array, slab_file, quality)
        print(f"    Saved: {slab_file} ({time.perf_counter() - t:.2f}s)")

    print(f"Loading images from {folder}...")
    images = load_folder(folder)
    print(f"Loaded {len(images)} images.")

    cfg = FocusStackConfig(
        images=images,
        workers=workers,
        slab=slab,
        only_slab=only_slab,
        recursive_slab=recursive_slab,
        on_slab=_on_slab if emit_steps else None,
    )

    t_start = time.perf_counter()
    result = run(cfg)
    if result.slabs is not None:
        if steps_dir is not None:
            print(f"Slabs saved to: {steps_dir}")
        print(f"Done ({time.perf_counter() - t_start:.2f}s total)")
        return result

    t_save = time.perf_counter()
    save_image(result.image, out_path, quality)  # type: ignore[arg-type]
    print(f"Saved: {out_path} ({time.perf_counter() - t_save:.2f}s)")
    print(f"Done ({time.perf_counter() - t_start:.2f}s total)")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Focus stack example using pre-loaded arrays."
    )
    parser.add_argument("folder", type=Path, help="Folder containing input images.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Output file path (default: stacked.jpg inside the input folder). "
            "Format is inferred from the extension; .jpg and .jpeg use the --quality setting."
        ),
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG output quality 1-95 (default: 95).",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help=(
            "Number of parallel workers for stacking (default: 3). "
            "Higher values are faster but increase peak RAM by ~100 MiB per additional worker. "
            "Set to 0 to use all CPU cores."
        ),
    )
    parser.add_argument(
        "--slab", type=int, nargs=2, metavar=("SIZE", "OVERLAP"), default=None,
        help=(
            "Enable slabbing: split the aligned image set into overlapping sub-stacks, "
            "stack each independently, then fuse the slab results. "
            "SIZE is the number of images per sub-stack; OVERLAP is how many images "
            "adjacent slabs share. Example: --slab 20 5"
        ),
    )
    parser.add_argument(
        "--output-steps", action="store_true",
        help=(
            "Save each intermediate slab result into a 'focusweave_slabs' folder "
            "inside the output directory. Requires --slab."
        ),
    )
    parser.add_argument(
        "--only-slab", action="store_true",
        help=(
            "Stop after producing slabs; do not fuse them into a final image. "
            "Requires --slab. Implies --output-steps."
        ),
    )
    parser.add_argument(
        "--recursive-slab", action="store_true",
        help=(
            "Enable recursive slabbing: if the layer-1 slab results still outnumber "
            "the slab SIZE, apply slabbing again to those results as layer 2, and so "
            "on, until the count fits in a single stack pass. Without this flag the "
            "layer-1 results are fused in one final stack regardless of count. "
            "Ignored when --only-slab is used."
        ),
    )
    parser.add_argument(
        "--slab-format", type=str, default=None, metavar="EXT",
        help=(
            "File format for images saved to focusweave_slabs/ (e.g. tiff, png, jpg). "
            "Defaults to tiff when not specified. Requires --output-steps or --only-slab."
        ),
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Error: '{args.folder}' is not a directory.")
        sys.exit(1)

    stack(
        folder=args.folder,
        output=args.output,
        workers=args.workers,
        quality=args.quality,
        slab=tuple(args.slab) if args.slab is not None else None,  # type: ignore[arg-type]
        output_steps=args.output_steps,
        only_slab=args.only_slab,
        recursive_slab=args.recursive_slab,
        slab_format=args.slab_format,
    )