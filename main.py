from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from focus_stack import FocusStackConfig, Interrupted, run


def save_image(img: np.ndarray, path: Path, quality: int) -> None:
    fmt = path.suffix.lower().lstrip(".")
    fmt = "jpeg" if fmt in ("jpg", "jpeg") else fmt.upper()
    save_kwargs: dict = {"quality": quality} if fmt == "jpeg" else {}
    Image.fromarray(img).save(path, fmt, **save_kwargs)


def _progress(fraction: float, message: str) -> None:
    if message:
        print(f"  {fraction * 100:5.1f}%  {message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focus stack a folder of images using Laplacian pyramid fusion."
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
        "--no-align", action="store_true",
        help="Skip ECC alignment (use when images are already registered).",
    )
    parser.add_argument(
        "--keep-size", action="store_true",
        help="Keep the output image the same size as the input images. Warps are applied in-place.",
    )
    parser.add_argument(
        "--crop", action="store_true",
        help=(
            "Crop the output to the intersection of all transformed image extents — the largest "
            "rectangle covered by every image. Removes all border regions but shrinks "
            "the output relative to the default expanded canvas."
        ),
    )
    parser.add_argument(
        "--no-fill", action="store_true",
        help=(
            "Fill border regions outside each image's coverage with black instead of "
            "reflecting edge pixels. Pairs naturally with --crop to trim the black borders away."
        ),
    )
    parser.add_argument(
        "--reference", type=int, default=-1,
        help=(
            "Index of the image to use as the alignment reference (default: middle image). "
            "All other images are aligned so that this image receives an identity warp. "
            "Images before the reference are chained backward; images after are chained forward."
        ),
    )
    parser.add_argument(
        "--global-align", action="store_true",
        help=(
            "Align every image directly to the reference instead of chaining through neighbours. "
            "More robust when images are not ordered by similarity, but more sensitive to large "
            "displacements between non-adjacent frames."
        ),
    )
    parser.add_argument(
        "--no-rotation", action="store_true",
        help=(
            "Suppress pure rotation correction during alignment. "
            "Scale and shear are still corrected unless also disabled. "
            "Useful when camera tilt is not a factor."
        ),
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help=(
            "Suppress scale correction during alignment — both uniform zoom and non-uniform "
            "per-axis scaling. Rotation and shear are still corrected unless also disabled."
        ),
    )
    parser.add_argument(
        "--no-shear", action="store_true",
        help=(
            "Suppress shear correction during alignment. "
            "Rotation and scale are still corrected unless also disabled."
        ),
    )
    parser.add_argument(
        "--no-translation", action="store_true",
        help=(
            "Suppress translation correction during alignment — correct rotation/scale/shear only. "
            "Useful when images are already spatially registered but may differ in orientation or zoom."
        ),
    )
    parser.add_argument(
        "--full-res", action="store_true",
        help=(
            "Run the fine ECC alignment pass at the original image resolution instead of the "
            "default 2048px cap. More accurate for large or subtle displacements at the cost "
            "of significantly increased alignment time."
        ),
    )
    parser.add_argument(
        "--min-shift", type=float, default=5.0,
        help="Minimum shift magnitude in pixels before alignment is applied (default: 5.0).",
    )
    parser.add_argument(
        "--levels", type=int, default=0,
        help="Pyramid levels (0 = auto-detect from image size).",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG output quality 1-95 (default: 95).",
    )
    parser.add_argument(
        "--sharpness", type=float, default=4.0,
        help=(
            "Weight sharpness exponent (default: 4.0). "
            "Higher values favour the sharpest image more aggressively at each pixel, "
            "approaching a hard winner-take-all selection. "
            "Lower values blend more smoothly across images. "
            "Useful range is roughly 1.0 (soft) to 8.0 (near-hard)."
        ),
    )
    parser.add_argument(
        "--dark-threshold", type=float, default=30.0,
        help=(
            "Luminance threshold (0-255) below which chroma is suppressed toward neutral (default: 30.0). "
            "Pixels with L below this value have their a/b channels lerped toward 128 (achromatic), "
            "preventing color drift in dark/black regions caused by floating point reconstruction error. "
            "Raise if color casts remain in shadows; lower if legitimate dark colors are being desaturated."
        ),
    )
    parser.add_argument(
        "--cull", type=float, nargs="?", const=0.6, default=None,
        metavar="THRESHOLD",
        help=(
            "Remove wholly out-of-focus images before stacking. "
            "Each frame is scored by its Tenengrad response; frames below THRESHOLD x peak score "
            "are dropped. At least the two sharpest frames are always retained. "
            "THRESHOLD defaults to 0.6 when --cull is given without a value; "
            "raise toward 1.0 to cull more aggressively, lower toward 0.0 to keep almost everything."
        ),
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

    out_path = args.output if args.output is not None else args.folder / "stacked.jpg"

    output_steps = args.output_steps or args.only_slab
    steps_dir = out_path.parent / "focusweave_slabs" if output_steps else None
    final_ext = args.slab_format.lstrip(".") if args.slab_format else "tiff"

    def _on_slab(label: str, array: np.ndarray) -> None:
        assert steps_dir is not None
        steps_dir.mkdir(parents=True, exist_ok=True)
        slab_file = steps_dir / f"{label}.{final_ext}"
        t = time.perf_counter()
        save_image(array, slab_file, args.quality)
        print(f"    Saved: {slab_file} ({time.perf_counter() - t:.2f}s)")

    cfg = FocusStackConfig(
        images=args.folder,
        no_align=args.no_align,
        keep_size=args.keep_size,
        crop=args.crop,
        no_fill=args.no_fill,
        reference=args.reference,
        cull=args.cull,
        global_align=args.global_align,
        no_rotation=args.no_rotation,
        no_scale=args.no_scale,
        no_shear=args.no_shear,
        no_translation=args.no_translation,
        full_res=args.full_res,
        min_shift=args.min_shift,
        levels=args.levels,
        sharpness=args.sharpness,
        dark_threshold=args.dark_threshold,
        workers=args.workers,
        slab=tuple(args.slab) if args.slab is not None else None,  # type: ignore[arg-type]
        only_slab=args.only_slab,
        recursive_slab=args.recursive_slab,
        on_slab=_on_slab if output_steps else None,
    )

    try:
        result = run(cfg, progress=_progress)
    except Interrupted:
        print("\nInterrupted.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(result.stats.format_report())

    if result.slabs is not None:
        if steps_dir is not None:
            print(f"Slabs saved to: {steps_dir}")
        return

    t_save = time.perf_counter()
    save_image(result.image, out_path, args.quality)  # type: ignore[arg-type]
    print(f"Saved: {out_path} ({time.perf_counter() - t_save:.2f}s)")


if __name__ == "__main__":
    main()