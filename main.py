"""
Focus stacking CLI.

Usage:
    python cli.py image1.tif image2.tif image3.tif ... -o stacked.tif

    # Or pass a directory
    python cli.py ./frames/ -o stacked.tif

    # Control Gaussian smoothing radius (larger = smoother region boundaries)
    python cli.py ./frames/ -o stacked.tif --sigma 4

    # Suppress background halos from in-focus subject edges bleeding into the background
    python cli.py ./frames/ -o stacked.tif --depth-radius 4 --bg-percentile 20

    # Save a false-colour depth map showing which frame was chosen per pixel
    python cli.py ./frames/ -o stacked.tif --depth-map depth.png

    # Discard frames with no in-focus content before stacking
    python cli.py ./frames/ -o stacked.tif --cull

    # Cull more aggressively (keep only frames within 20% of the sharpest)
    python cli.py ./frames/ -o stacked.tif --cull --cull-threshold 0.2

    # Save per-frame focus score maps for debugging
    python cli.py ./frames/ -o stacked.tif --debug-scores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from focus_stack import (
    collect_paths,
    load_images,
    cull_unfocused_images,
    align_images,
    fill_alignment_gaps,
    crop_to_valid_union,
    compute_focus_maps,
    select_best_pixels,
    save_depth_map,
    save_bg_mask,
    save_score_maps,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Focus stack microscope images using Tenengrad hard selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="IMAGE_OR_DIR",
        help="Input image files or a single directory containing them.",
    )
    p.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file path (e.g. stacked.tif or stacked.png).",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        metavar="PIXELS",
        help="Gaussian smoothing radius for focus maps.  "
             "Larger values give smoother region boundaries (default: 5.0).",
    )
    p.add_argument(
        "--sobel-ksize",
        type=int,
        default=5,
        choices=[1, 3, 5, 7],
        metavar="K",
        dest="sobel_ksize",
        help="Sobel kernel size for Tenengrad (1, 3, 5, or 7; default: 5).  "
             "Larger kernels are less sensitive to single-pixel shot noise.",
    )
    p.add_argument(
        "--score-power",
        type=float,
        default=2.0,
        metavar="P",
        dest="score_power",
        help=(
            "Exponent applied to the raw Tenengrad score before smoothing (default: 2.0).  "
            "Higher values widen the gap between sharp-edge spikes and the broad, "
            "moderate-magnitude plateaus produced by diffraction halos inside "
            "out-of-focus cells.  Increase to 3-4 if halo contamination persists; "
            "reduce toward 1.0 for subjects with very low overall contrast."
        ),
    )
    p.add_argument(
        "--fill-quiet",
        action="store_true",
        dest="fill_quiet",
        help=(
            "Fill featureless low-detail pixels by propagating the frame assignment "
            "from their nearest sharp neighbours, rather than using argmax.  "
            "Fixes the failure where a smooth region that is clean when in focus "
            "gets assigned to a noisy out-of-focus frame (noise lifts the score "
            "above the clean in-focus score).  The quiet pixel has no reliable "
            "signal of its own, so it inherits the depth of the surrounding "
            "in-focus detail.  The boundary between quiet and sharp pixels is "
            "determined automatically by Otsu's method on the score distribution."
        ),
    )
    p.add_argument(
        "--fill-quiet-search-radius",
        type=int,
        default=32,
        metavar="R",
        dest="fill_quiet_search_radius",
        help=(
            "Gaussian propagation radius in pixels for --fill-quiet (default: 32).  "
            "Increase if large featureless regions are not being fully filled from "
            "their sharp neighbours; decrease to keep the depth assignment more local."
        ),
    )
    p.add_argument(
        "--no-align",
        action="store_true",
        dest="no_align",
        help="Skip alignment (use when images are already registered).",
    )
    p.add_argument(
        "--warp",
        default="translation",
        choices=["translation", "euclidean", "affine", "homography"],
        help="ECC warp model (default: translation).  "
             "Use homography for strong perspective distortion.",
    )
    p.add_argument(
        "--crop",
        action="store_true",
        help="Trim the output to the bounding box where every frame contributed "
             "real pixel data (intersection of valid regions).  Without this flag "
             "the output is cropped only to the union of all valid regions, "
             "with gap pixels filled from nearest neighbour frames.",
    )
    p.add_argument(
        "--depth-radius",
        type=int,
        default=None,
        metavar="R",
        dest="depth_radius",
        help=(
            "Restrict pixel selection to a window of [peak-R, peak+R] frames "
            "around each pixel's unconstrained best-focus frame (default: disabled).  "
            "For flat subjects ordered by depth, this prevents distant frames from "
            "stealing pixels due to diffraction halos or noise.  "
            "Typical values: 2-4 for thin flat specimens; larger for deep subjects."
        ),
    )
    p.add_argument(
        "--smooth-source",
        type=int,
        default=None,
        metavar="R",
        dest="smooth_source",
        help=(
            "Apply two passes of median filtering (radius R) to the source-frame "
            "map after selection (default: disabled).  Removes isolated outlier frame "
            "assignments caused by noise spikes or diffraction halos.  "
            "Recommended starting value: 5."
        ),
    )
    p.add_argument(
        "--blend-low-confidence",
        type=float,
        default=None,
        metavar="T",
        dest="blend_low_confidence",
        help=(
            "Blend frames by focus-score weight for pixels whose winner confidence "
            "falls below T (default: disabled, hard selection everywhere).  "
            "Confidence = (best - second_best) / best; near 0 in featureless regions "
            "where hard selection would introduce noise.  Recommended starting value: 0.3."
        ),
    )
    p.add_argument(
        "--bg-percentile",
        nargs="?",
        const="otsu",
        default=None,
        metavar="P",
        dest="bg_percentile",
        help=(
            "Enable background halo suppression.  Background pixels (those whose "
            "peak focus score across the stack is below a threshold) are assigned "
            "to a halo-safe frame rather than the argmax frame.  "
            "The threshold is set by Otsu's method on the log-compressed score "
            "distribution.  Requires --depth-radius to be set.  Use --bg-mask to "
            "save the mask and verify the classification.  "
            "The optional value P, when given, acts as a percentile lower bound: the "
            "threshold is raised to at least the Pth percentile of the peak-score "
            "distribution, forcing at least P%% of pixels into the background class. "
            "Use this when Otsu's split sits too high and hot-spot regions are still "
            "appearing white in the mask.  Start with P=30 and raise until the "
            "problem regions turn white."
        ),
    )
    p.add_argument(
        "--bg-search-radius",
        type=int,
        default=64,
        metavar="R",
        dest="bg_search_radius",
        help=(
            "Gaussian propagation radius (in pixels) used to estimate the local "
            "subject focus depth at each background pixel when --bg-percentile is "
            "active (default: 64).  The subject peak-frame map is blurred with a "
            "Gaussian of this sigma so that background pixels inherit the focus depth "
            "of their nearest subject neighbours.  Increase if your background has "
            "large regions far from any subject edge; decrease if the subject has "
            "fine structure and you want the depth estimate to stay local."
        ),
    )
    p.add_argument(
        "--bg-mask",
        metavar="FILE",
        dest="bg_mask_path",
        help=(
            "Save the subject/background classification mask to this path (PNG).  "
            "White pixels are classified as background; black pixels as subject.  "
            "Use this to diagnose --bg-percentile: hot-spot regions that are still "
            "causing problems should appear white (correctly classified as background). "
            "If they appear black they are being treated as subject — raise "
            "--bg-percentile until they turn white."
        ),
    )
    p.add_argument(
        "--bg-blend-radius",
        type=int,
        default=None,
        metavar="R",
        dest="bg_blend_radius",
        help=(
            "Feather the subject/background boundary by this many pixels (default: "
            "disabled, hard edge).  A Gaussian of sigma R is applied to the subject "
            "mask to produce a soft alpha: 1 inside the subject (uses the normal "
            "focus-selected frame) fading to 0 deep in the background (uses the "
            "halo-safe frame from --bg-percentile).  Without this, the hard switch at "
            "the mask boundary can produce a visible seam where the frame index jumps "
            "abruptly.  Start with R=20; increase if a step is still visible, decrease "
            "if the transition bleeds too far into the background.  Requires "
            "--bg-percentile to have any effect."
        ),
    )
    p.add_argument(
        "--depth-map",
        metavar="FILE",
        help="Save a greyscale source-frame depth map to this path (black = first frame, white = last frame).",
    )
    p.add_argument(
        "--debug-scores",
        action="store_true",
        help="Save per-frame Tenengrad score maps to a 'focus_scores/' subdirectory "
             "next to the output file.",
    )
    p.add_argument(
        "--cull",
        action="store_true",
        help=(
            "Before stacking, discard any image that has no sufficiently sharp region.  "
            "Each frame is scored by the mean Tenengrad response in its top 5%% of "
            "pixels; frames whose score falls below --cull-threshold × (peak score) "
            "are dropped.  At least two frames are always kept.  "
            "Use --debug-scores to inspect per-frame scores before committing."
        ),
    )
    p.add_argument(
        "--cull-threshold",
        type=float,
        default=0.19,
        metavar="T",
        dest="cull_threshold",
        help=(
            "Fraction of the sharpest frame's focus score below which a frame is "
            "culled when --cull is active (default: 0.05 = 5%%).  "
            "Raise toward 1.0 to discard more frames; lower toward 0.0 to keep "
            "almost everything.  A value of 0.1-0.2 is a good starting point for "
            "stacks where a few frames are wholly out of focus."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Collect and load -------------------------------------------------------
    image_paths = collect_paths(args.inputs)
    images = load_images(image_paths)

    # --- Cull wholly out-of-focus frames ----------------------------------------
    if args.cull:
        images, image_paths = cull_unfocused_images(
            images,
            image_paths,
            threshold=args.cull_threshold,
            ksize=args.sobel_ksize,
        )

    # Validate consistent dimensions
    ref_shape = images[0].shape[:2]
    for i, img in enumerate(images[1:], start=1):
        if img.shape[:2] != ref_shape:
            raise ValueError(
                f"Image {image_paths[i]} has shape {img.shape[:2]} but reference is "
                f"{ref_shape}.  All images must have identical dimensions."
            )

    # --- Align ------------------------------------------------------------------
    if args.no_align:
        print("Skipping alignment (--no-align).")
        aligned = images
        valid_union: np.ndarray | None = None
        valid_intersection: np.ndarray | None = None
    else:
        warp_mode = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean":   cv2.MOTION_EUCLIDEAN,
            "affine":      cv2.MOTION_AFFINE,
            "homography":  cv2.MOTION_HOMOGRAPHY,
        }[args.warp]
        aligned, valid_masks = align_images(images, warp_mode=warp_mode)
        print("Filling alignment gap pixels from nearest neighbour frames...")
        aligned = fill_alignment_gaps(aligned, valid_masks)
        aligned, valid_union, valid_intersection = crop_to_valid_union(
            aligned, valid_masks,
        )

    # --- Focus maps -------------------------------------------------------------
    focus_maps = compute_focus_maps(
        aligned,
        sigma=args.sigma,
        ksize=args.sobel_ksize,
        power=args.score_power,
        valid_mask=valid_intersection if not args.no_align else None,
    )

    # --- Composite --------------------------------------------------------------
    # --bg-percentile with no value -> const="otsu" (pure Otsu, no percentile floor)
    # --bg-percentile 30            -> 30.0          (Otsu + percentile lower bound)
    # flag absent                   -> None           (feature disabled)
    bg_percentile_arg = args.bg_percentile
    bg_enabled = bg_percentile_arg is not None
    bg_percentile_value: float | None = (
        None if (not bg_enabled or bg_percentile_arg == "otsu")
        else float(bg_percentile_arg)
    )

    result, source_map, bg_mask = select_best_pixels(
        aligned, focus_maps,
        depth_radius=args.depth_radius,
        smooth_radius=args.smooth_source,
        blend_low_confidence=args.blend_low_confidence,
        bg_threshold_percentile=bg_percentile_value if bg_enabled else None,
        bg_halo_suppression=bg_enabled,
        bg_search_radius=args.bg_search_radius,
        bg_blend_radius=args.bg_blend_radius if bg_enabled else None,
        fill_quiet=args.fill_quiet,
        fill_quiet_search_radius=args.fill_quiet_search_radius,
    )

    # --- Crop to valid intersection (--crop) ------------------------------------
    # Trims the union-cropped result further to the bounding box of the
    # intersection -- the largest rectangle where *every* frame contributed real
    # pixel data.  Both masks are already in union-cropped canvas coordinates.
    if args.crop and not args.no_align and valid_intersection is not None:
        _irows = np.any(valid_intersection, axis=1)
        _icols = np.any(valid_intersection, axis=0)
        if _irows.any() and _icols.any():
            iy0 = int(np.argmax(_irows))
            iy1 = int(len(_irows) - 1 - np.argmax(_irows[::-1]))
            ix0 = int(np.argmax(_icols))
            ix1 = int(len(_icols) - 1 - np.argmax(_icols[::-1]))
            result     = result    [iy0:iy1 + 1, ix0:ix1 + 1]
            source_map = source_map[iy0:iy1 + 1, ix0:ix1 + 1]
            if bg_mask is not None:
                bg_mask = bg_mask[iy0:iy1 + 1, ix0:ix1 + 1]
            crop_h, crop_w = result.shape[:2]
            print(f"Cropped output to valid intersection: {crop_w}x{crop_h}")

    # --- Save outputs -----------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer lossless compression when writing TIFF
    params: list[int] = []
    if out_path.suffix.lower() in {".tif", ".tiff"}:
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 5]  # LZW
    elif out_path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

    cv2.imwrite(str(out_path), result, params)
    print(f"\nStacked image saved -> {out_path}")

    n = len(images)
    for idx in range(n):
        pct = (source_map == idx).sum() / source_map.size * 100
        print(f"  Frame {idx + 1:3d}  ({image_paths[idx].name}):  "
              f"{pct:5.1f}% of pixels selected")

    if args.depth_map:
        save_depth_map(source_map, n, Path(args.depth_map))

    if args.bg_mask_path:
        if bg_mask is not None:
            save_bg_mask(bg_mask, Path(args.bg_mask_path))
        else:
            print("[warn] --bg-mask specified but --bg-percentile was not set; no mask to save.")

    if args.debug_scores:
        scores_dir = out_path.parent / "focus_scores"
        save_score_maps(focus_maps, scores_dir, image_paths)


if __name__ == "__main__":
    main()