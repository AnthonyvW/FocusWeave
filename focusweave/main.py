from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from focusweave.focus_stack import (
    IMAGE_EXTENSIONS,
    FocusStackConfig,
    Interrupted,
    _load_and_warp,
    align_images,
    compute_canvas,
    resolve_images,
    run,
)


def _get_version() -> str:
    try:
        return version("focusweave")
    except PackageNotFoundError:
        pass
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    try:
        with pyproject.open("rb") as f:
            import tomllib
            return tomllib.load(f)["project"]["version"]
    except Exception:
        return "unknown"


def save_image(img: np.ndarray, path: Path, quality: int) -> None:
    fmt = path.suffix.lower().lstrip(".")
    fmt = "jpeg" if fmt in ("jpg", "jpeg") else fmt.upper()
    save_kwargs: dict = {"quality": quality} if fmt == "jpeg" else {}
    Image.fromarray(img).save(path, fmt, **save_kwargs)


def _progress(fraction: float, stage: str, message: str) -> None:
    if message:
        print(f"  {fraction * 100:5.1f}%  {message}")


_MP4_FPS: int = 24


def _make_animation(
    frames_bgr: list[np.ndarray],
    out_path: Path,
    duration_s: float,
    fmt: str = "webp",
    scale: float = 1.0,
    gif_colors: int = 256,
) -> None:
    """Encode BGR frames into an animated WebP or GIF.

    Frame duration is distributed evenly across all frames. All frames are
    conformed to the most common (h, w) so the encoder never sees mixed sizes.
    """
    frame_duration_ms = int(duration_s * 1000 / len(frames_bgr))

    size_counts: Counter[tuple[int, int]] = Counter(
        (f.shape[0], f.shape[1]) for f in frames_bgr
    )
    canonical_h, canonical_w = size_counts.most_common(1)[0][0]
    if scale != 1.0:
        canonical_w = max(1, int(canonical_w * scale))
        canonical_h = max(1, int(canonical_h * scale))

    pil_frames: list[Image.Image] = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target_w = max(1, int(rgb.shape[1] * scale)) if scale != 1.0 else rgb.shape[1]
        target_h = max(1, int(rgb.shape[0] * scale)) if scale != 1.0 else rgb.shape[0]
        if target_w != canonical_w or target_h != canonical_h:
            target_w, target_h = canonical_w, canonical_h
        if target_w != rgb.shape[1] or target_h != rgb.shape[0]:
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        pil_frame = Image.fromarray(rgb)
        if fmt == "gif":
            pil_frame = pil_frame.quantize(
                colors=gif_colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
        pil_frames.append(pil_frame)

    save_kwargs: dict = {
        "save_all": True,
        "append_images": pil_frames[1:],
        "duration": frame_duration_ms,
        "loop": 0,
    }
    if fmt == "webp":
        save_kwargs.update({"lossless": False, "quality": 80, "method": 4})
    elif fmt == "gif":
        save_kwargs["optimize"] = True

    pil_frames[0].save(str(out_path), **save_kwargs)


def _make_video(
    frames_bgr: list[np.ndarray],
    out_path: Path,
    duration_s: float,
    fmt: str = "mp4",
    scale: float = 1.0,
    crf: int = 28,
) -> None:
    """Encode BGR frames into an MP4 (H.264) or WebM (VP9) video via imageio-ffmpeg.

    Encodes at a fixed 24fps, repeating each source frame for its proportional
    share of the total duration so inter-frame compression reduces repeated frames
    to near-zero bytes.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio and imageio-ffmpeg are required for video output. "
            "Install with: pip install imageio imageio-ffmpeg"
        )

    size_counts: Counter[tuple[int, int]] = Counter(
        (f.shape[0], f.shape[1]) for f in frames_bgr
    )
    canonical_h, canonical_w = size_counts.most_common(1)[0][0]
    if scale != 1.0:
        canonical_w = max(2, int(canonical_w * scale))
        canonical_h = max(2, int(canonical_h * scale))
    canonical_w += canonical_w % 2
    canonical_h += canonical_h % 2

    total_video_frames = round(duration_s * _MP4_FPS)
    n = len(frames_bgr)
    repeat_counts = [
        round((i + 1) * total_video_frames / n) - round(i * total_video_frames / n)
        for i in range(n)
    ]

    if fmt == "webm":
        codec = "libvpx-vp9"
        pixelformat = "yuv420p"
        extra_params = ["-b:v", "0", "-crf", str(crf)]
    else:
        codec = "libx264"
        pixelformat = "yuv420p"
        extra_params = ["-crf", str(crf), "-preset", "slow"]

    with imageio.get_writer(
        str(out_path),
        format="ffmpeg",
        mode="I",
        fps=_MP4_FPS,
        codec=codec,
        pixelformat=pixelformat,
        output_params=extra_params,
    ) as writer:
        for bgr, repeat in zip(frames_bgr, repeat_counts):
            if bgr.shape[1] != canonical_w or bgr.shape[0] != canonical_h:
                bgr = cv2.resize(bgr, (canonical_w, canonical_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for _ in range(repeat):
                writer.append_data(rgb)


def _build_media_post(
    src_images: list[Path | np.ndarray],
    warps: list[np.ndarray],
    canvas_size: tuple[int, int],
    no_fill: bool,
    stacked: np.ndarray | None,
    out_path: Path,
    fmt: str,
    duration_s: float,
    scale: float,
    gif_colors: int,
    crf: int,
) -> None:
    """Warp each source frame onto the canvas and encode as an animation.

    When stacked is provided (combined mode) it is placed to the right of each
    animation frame before encoding. The stacked panel is resized to match the
    animation frame height so the composite is always rectangular.

    Frames are produced as uint8 RGB arrays then converted to BGR before being
    passed to the encoders, which expect BGR input (matching the focus_overlay
    convention).
    """
    border_mode = cv2.BORDER_CONSTANT if no_fill else cv2.BORDER_REFLECT

    stacked_bgr: np.ndarray | None = None
    if stacked is not None:
        stacked_bgr = cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR)

    frames_bgr: list[np.ndarray] = []
    n = len(src_images)
    for i, (src, warp) in enumerate(zip(src_images, warps)):
        print(f"  Rendering frame {i + 1}/{n}...")
        frame_f32 = _load_and_warp(src, warp, canvas_size, border_mode)
        frame_u8 = np.clip(frame_f32, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)

        if stacked_bgr is not None:
            panel_h = frame_bgr.shape[0]
            panel_w = int(round(stacked_bgr.shape[1] * panel_h / stacked_bgr.shape[0]))
            panel = cv2.resize(stacked_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            frame_bgr = np.concatenate([frame_bgr, panel], axis=1)

        frames_bgr.append(frame_bgr)

    if not frames_bgr:
        print("  No frames produced; skipping animation.")
        return

    if fmt in ("mp4", "webm"):
        _make_video(frames_bgr, out_path, duration_s, fmt=fmt, scale=scale, crf=crf)
    else:
        _make_animation(frames_bgr, out_path, duration_s, fmt=fmt, scale=scale, gif_colors=gif_colors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focus stack a folder of images using Laplacian pyramid fusion."
    )
    parser.add_argument("folder", type=Path, nargs="?", default=None, help="Folder containing input images.")
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
    parser.add_argument(
        "--media-post", choices=["separate", "combined"], default=None,
        metavar="MODE",
        help=(
            "Export an animation alongside the standard stacked output. "
            "The animation shows each aligned source frame in sequence. "
            "'separate' outputs the animation alone. "
            "'combined' places each animation frame to the left of the focus-stacked result. "
            "Output format is controlled by --media-format (default: webp)."
        ),
    )
    parser.add_argument(
        "--media-output", type=Path, default=None,
        help="File path for the media post animation (default: media_post.<ext> inside the input folder).",
    )
    parser.add_argument(
        "--media-format", default="webp", choices=["webp", "gif", "mp4", "webm"],
        help=(
            "Animation format: webp (default, smaller, full colour), "
            "gif (256-colour palette, widest compatibility), "
            "mp4 (H.264, requires imageio-ffmpeg), "
            "webm (VP9, recommended for Google Slides, requires imageio-ffmpeg)."
        ),
    )
    parser.add_argument(
        "--media-duration", type=float, default=5.0,
        metavar="SECONDS",
        help="Total animation duration in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--media-scale", type=float, default=1.0,
        metavar="FACTOR",
        help=(
            "Scale factor applied to animation frames before encoding (default: 1.0). "
            "0.5 halves each dimension, reducing file size by roughly 4x."
        ),
    )
    parser.add_argument(
        "--media-gif-colors", type=int, default=256,
        metavar="N",
        help="Number of palette colours for GIF output (2-256, default: 256).",
    )
    parser.add_argument(
        "--media-crf", type=int, default=28,
        metavar="CRF",
        help=(
            "Constant rate factor for MP4/WebM output (default: 28). "
            "H.264 range 0-51, VP9 range 0-63. Lower values give higher quality and larger files."
        ),
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show the focusweave version number and exit.",
    )
    parser.add_argument(
        "--formats", action="store_true",
        help="List all image formats supported by the current Pillow installation and exit.",
    )
    args = parser.parse_args()

    if args.version:
        print(f"focusweave {_get_version()}")
        return

    if args.formats:
        exts = sorted(IMAGE_EXTENSIONS)
        print("Supported image extensions (via Pillow):")
        print("  " + "  ".join(exts))
        return

    if args.folder is None:
        parser.error("the following arguments are required: folder")

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
        t_start = time.perf_counter()
        result = run(cfg, progress=_progress)
    except Interrupted:
        print("\nInterrupted.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if result.slabs is not None:
        if steps_dir is not None:
            print(f"Slabs saved to: {steps_dir}")
        print(f"Done ({time.perf_counter() - t_start:.2f}s total)")
        return

    t_save = time.perf_counter()
    save_image(result.image, out_path, args.quality)  # type: ignore[arg-type]
    print(f"Saved: {out_path} ({time.perf_counter() - t_save:.2f}s)")

    if args.media_post is not None:
        fmt = args.media_format
        media_out = args.media_output if args.media_output is not None else args.folder / f"media_post.{fmt}"

        print(f"Building media post animation ({args.media_post}, {fmt.upper()})...")
        t_media = time.perf_counter()

        src_images, reference_size = resolve_images(args.folder)
        n_images = len(src_images)
        reference = n_images // 2 if args.reference < 0 else args.reference

        identity = np.eye(2, 3, dtype=np.float32)
        if not args.no_align:
            warps = align_images(
                src_images,
                reference_size,
                reference_idx=reference,
                global_align=args.global_align,
                no_rotation=args.no_rotation,
                no_scale=args.no_scale,
                no_shear=args.no_shear,
                no_translation=args.no_translation,
                full_res=args.full_res,
                min_shift=args.min_shift,
                progress=_progress,
            )
        else:
            warps = [identity.copy() for _ in src_images]

        canvas_size, adjusted_warps = compute_canvas(
            warps, reference_size, keep_size=args.keep_size, crop=args.crop
        )

        stacked_for_media: np.ndarray | None = (
            result.image if args.media_post == "combined" else None  # type: ignore[assignment]
        )

        _build_media_post(
            src_images=src_images,
            warps=adjusted_warps,
            canvas_size=canvas_size,
            no_fill=args.no_fill,
            stacked=stacked_for_media,
            out_path=media_out,
            fmt=fmt,
            duration_s=args.media_duration,
            scale=args.media_scale,
            gif_colors=args.media_gif_colors,
            crf=args.media_crf,
        )
        print(f"Saved: {media_out} ({time.perf_counter() - t_media:.2f}s)")

    print(f"Done ({time.perf_counter() - t_start:.2f}s total)")


if __name__ == "__main__":
    main()