from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d, uniform_filter


KERNEL_1D = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


class _Checkpoint:
    __slots__ = ("name", "elapsed", "current_mib", "peak_mib")

    def __init__(self, name: str, elapsed: float, current_mib: float, peak_mib: float) -> None:
        self.name = name
        self.elapsed = elapsed
        self.current_mib = current_mib
        self.peak_mib = peak_mib


def _snap(name: str, step_start: float, checkpoints: list[_Checkpoint]) -> float:
    """Record elapsed time and current tracemalloc stats, return now for the next timer."""
    now = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    checkpoints.append(_Checkpoint(name, now - step_start, current / 2**20, peak / 2**20))
    tracemalloc.reset_peak()
    return now


def _print_report(checkpoints: list[_Checkpoint], total_elapsed: float) -> None:
    name_w = max(len(c.name) for c in checkpoints)
    header = f"  {'Step':<{name_w}}  {'Time':>8}  {'Current MiB':>12}  {'Peak MiB':>10}"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for c in checkpoints:
        print(f"  {c.name:<{name_w}}  {c.elapsed:>7.2f}s  {c.current_mib:>11.1f}  {c.peak_mib:>9.1f}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'Total':<{name_w}}  {total_elapsed:>7.2f}s")

def load_images(folder: Path) -> tuple[list[Path], tuple[int, int]]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        print(f"Error: need at least 2 images in '{folder}', found {len(paths)}.")
        sys.exit(1)

    img0 = Image.open(paths[0]).convert("RGB")
    reference_size: tuple[int, int] = img0.size
    n = len(paths)
    print(f"  Found {n} images, reference size: {reference_size[0]}x{reference_size[1]}")
    return paths, reference_size


def _to_gray_cv(image: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB (H,W,3) to uint8 grayscale using OpenCV."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _ecc_align(ref_gray: np.ndarray, src_gray: np.ndarray,
               max_resolution: int, rough: bool) -> np.ndarray:
    """Single cv2 ECC alignment pass.

    Downscales both images so the longer edge is at most max_resolution,
    runs findTransformECC with MOTION_AFFINE, then rescales the translation
    component back to full resolution and returns the 2x3 affine matrix.
    """
    h, w = ref_gray.shape
    resolution = max(h, w)
    if resolution > max_resolution:
        scale = max_resolution / resolution
        ref_small = cv2.resize(ref_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        src_small = cv2.resize(src_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        ref_small = ref_gray
        src_small = src_gray

    warp = np.eye(2, 3, dtype=np.float32)
    if rough:
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, 0.01)
        gauss_levels = 1
    else:
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 0.001)
        gauss_levels = 3

    cv2.findTransformECC(src_small.astype(np.float32),
                         ref_small.astype(np.float32),
                         warp, cv2.MOTION_AFFINE, criteria,
                         None, gauss_levels)

    warp[0, 2] /= scale
    warp[1, 2] /= scale
    return warp


def _chain_affines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine matrices: apply b then a."""
    a3 = np.vstack([a, [0, 0, 1]])
    b3 = np.vstack([b, [0, 0, 1]])
    return (a3 @ b3)[:2]


def _invert_affine(warp: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix."""
    m3 = np.vstack([warp, [0.0, 0.0, 1.0]])
    return np.linalg.inv(m3)[:2]


def _run_ecc(
    ref_gray: np.ndarray,
    src_gray: np.ndarray,
    full_res: bool = False,
) -> tuple[np.ndarray, bool]:
    """Run the coarse-to-fine ECC cascade. Returns (warp, converged).

    The fine pass resolution is adaptive: 2048px by default, or 4096px if the
    image's long edge exceeds 4096px (i.e. 2048 would be less than half the
    image size). With full_res=True it runs at the original image size instead.
    """
    identity = np.eye(2, 3, dtype=np.float32)
    warp = identity.copy()
    if full_res:
        fine_res = 2 ** 31
    else:
        long_edge = max(ref_gray.shape)
        fine_res = 4096 if long_edge > 4096 else 2048
    for max_res, rough in [(256, True), (fine_res, False)]:
        try:
            warp = _ecc_align(ref_gray, src_gray, max_res, rough)
        except cv2.error:
            return identity.copy(), False
    return warp, True


def _report_warp(label: str, warp: np.ndarray, no_rotation: bool = False) -> None:
    angle = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
    tx, ty = warp[0, 2], warp[1, 2]
    print(f"  {label}: rotation {angle:+.2f}°  shift ({ty:+.1f}, {tx:+.1f}) px")


def _constrain_warp(
    warp: np.ndarray,
    no_rotation: bool,
    no_scale: bool,
    no_shear: bool,
    no_translation: bool,
) -> np.ndarray:
    """Suppress selected degrees of freedom from a 2x3 affine warp.

    The linear part M is decomposed via polar decomposition M = R @ S, where R
    is a pure rotation (det = +1) and S is a symmetric matrix encoding scale and
    shear. S is further factored via SVD as Vt.T @ diag(sv) @ Vt.

    Each flag suppresses one component before recomposing:

      no_rotation — drop R  (result is S alone: Vt.T @ diag(sv) @ Vt)
      no_scale    — normalize sv to geometric mean 1 (removes zoom, keeps R and shear)
      no_shear    — drop Vt from S (result is R @ diag(sv): rotation + axis-aligned scale)
      no_translation — zero warp[:, 2]

    Any combination is valid. All four flags together collapse the warp to identity.
    """
    if not (no_rotation or no_scale or no_shear or no_translation):
        return warp
    result = warp.copy()
    if no_rotation or no_scale or no_shear:
        M = warp[:, :2].astype(np.float64)
        U, sv, Vt = np.linalg.svd(M)
        # SVD may produce det(U)*det(Vt) = -1, encoding a reflection. Fix by
        # flipping the last column of U and last row of Vt (and negating that
        # singular value) so that R = U @ Vt has det = +1.
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            U[:, -1] *= -1
            sv[-1] *= -1
        R = U @ Vt  # pure rotation, det = +1
        if no_scale:
            geomean = np.sqrt(abs(sv[0] * sv[1]))
            sv = sv / (geomean if geomean > 1e-10 else 1.0)
        if no_rotation and no_shear:
            result[:, :2] = np.diag(sv)
        elif no_rotation:
            result[:, :2] = Vt.T * sv @ Vt  # S = Vt.T @ diag(sv) @ Vt
        elif no_shear:
            result[:, :2] = R @ np.diag(sv)
        else:
            result[:, :2] = (U * sv) @ Vt  # full recompose with modified sv
    if no_translation:
        result[:, 2] = 0.0
    return result.astype(np.float32)


def align_images(
    src_paths: list[Path],
    reference_size: tuple[int, int],
    reference_idx: int = 0,
    global_align: bool = False,
    no_rotation: bool = False,
    no_scale: bool = False,
    no_shear: bool = False,
    no_translation: bool = False,
    full_res: bool = False,
    min_shift: float = 5.0,
) -> list[np.ndarray]:
    """Compute affine warps for all images relative to reference_idx.

    Two strategies are supported:

    Neighbour-chained (default): ECC is run on consecutive raw grayscale pairs.
    Warps are composed mathematically so interpolation error never accumulates.
    Images after reference_idx chain forward; images before it chain backward
    by inverting each neighbour warp.

    Global (global_align=True): every image is aligned directly to the reference
    grayscale. No chaining — each warp is the raw ECC result. More robust when
    images are not ordered by similarity, but more sensitive to large displacements.

    With full_res=True the fine ECC pass runs at the original image resolution
    instead of 2048px, improving accuracy at the cost of speed.

    no_rotation, no_scale, no_shear, and no_translation constrain each per-step
    ECC warp before it is chained or used, so the constraints compose correctly
    through the chain. See _constrain_warp for decomposition details.

    In both modes the reference image always receives an identity warp, and
    images whose cumulative transform is negligible (below min_shift with no
    meaningful rotation/scale) are assigned identity.
    """
    ref_w, ref_h = reference_size
    identity = np.eye(2, 3, dtype=np.float32)
    n = len(src_paths)

    def _load_raw_gray(path: Path) -> np.ndarray:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        if img.shape[1] != ref_w or img.shape[0] != ref_h:
            img = cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
        return _to_gray_cv(img)

    def _is_negligible(warp: np.ndarray) -> bool:
        translation = np.linalg.norm(warp[:, 2])
        linear_is_identity = np.allclose(warp[:, :2], np.eye(2), atol=1e-3)
        return translation < min_shift and linear_is_identity

    def _constrain(warp: np.ndarray) -> np.ndarray:
        return _constrain_warp(warp, no_rotation, no_scale, no_shear, no_translation)

    def _run(ref: np.ndarray, src: np.ndarray) -> tuple[np.ndarray, bool]:
        return _run_ecc(ref, src, full_res)

    def _report(label: str, warp: np.ndarray) -> None:
        _report_warp(label, warp, no_rotation)

    ref_gray = _load_raw_gray(src_paths[reference_idx])
    warps: list[np.ndarray] = [identity.copy()] * n
    warps[reference_idx] = identity.copy()

    if global_align:
        for i in range(n):
            if i == reference_idx:
                continue
            src_gray = _load_raw_gray(src_paths[i])
            warp, converged = _run(ref_gray, src_gray)
            warp = _constrain(warp)
            label = f"Image {i + 1}"
            if not converged:
                print(f"  {label}: ECC did not converge — using identity")
                warps[i] = identity.copy()
            elif _is_negligible(warp):
                print(f"  {label}: transform negligible — skipped")
                warps[i] = identity.copy()
            else:
                _report(label, warp)
                warps[i] = warp
        return warps

    # Neighbour-chained: forward pass (reference_idx -> end)
    cumulative = identity.copy()
    prev_gray = ref_gray
    for i in range(reference_idx + 1, n):
        src_gray = _load_raw_gray(src_paths[i])
        warp, converged = _run(prev_gray, src_gray)
        warp = _constrain(warp)
        prev_gray = src_gray
        label = f"Image {i + 1}"
        if not converged:
            print(f"  {label}: ECC did not converge — using previous transform")
            warps[i] = cumulative.copy()
            continue
        cumulative = _constrain(_chain_affines(cumulative, warp))
        if no_rotation:
            cumulative[:, :2] = np.eye(2)
        if _is_negligible(cumulative):
            print(f"  {label}: transform negligible — skipped")
            warps[i] = identity.copy()
        else:
            _report(label, cumulative)
            warps[i] = cumulative.copy()

    # Neighbour-chained: backward pass (reference_idx -> start), warps inverted
    cumulative = identity.copy()
    prev_gray = ref_gray
    for i in range(reference_idx - 1, -1, -1):
        src_gray = _load_raw_gray(src_paths[i])
        warp, converged = _run(prev_gray, src_gray)
        warp = _constrain(warp)
        prev_gray = src_gray
        label = f"Image {i + 1}"
        if not converged:
            print(f"  {label}: ECC did not converge — using previous transform")
            warps[i] = cumulative.copy()
            continue
        cumulative = _constrain(_chain_affines(cumulative, _invert_affine(warp)))
        if no_rotation:
            cumulative[:, :2] = np.eye(2)
        if _is_negligible(cumulative):
            print(f"  {label}: transform negligible — skipped")
            warps[i] = identity.copy()
        else:
            _report(label, cumulative)
            warps[i] = cumulative.copy()

    return warps



def _smooth(image: np.ndarray) -> np.ndarray:
    """Apply separable 5-tap Gaussian smoothing."""
    return convolve1d(convolve1d(image, KERNEL_1D, axis=0), KERNEL_1D, axis=1)


def reduce(image: np.ndarray) -> np.ndarray:
    return _smooth(image)[::2, ::2]


def expand(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    upsampled = np.zeros((h * 2, w * 2), dtype=np.float32)
    upsampled[::2, ::2] = image
    expanded = convolve1d(convolve1d(upsampled, KERNEL_1D * 2, axis=0), KERNEL_1D * 2, axis=1)
    return expanded[: target_shape[0], : target_shape[1]]



def region_energy(lp_level: np.ndarray, window: int = 3) -> np.ndarray:
    return uniform_filter(lp_level ** 2, size=window)


def region_deviation(image: np.ndarray, window: int = 3) -> np.ndarray:
    mean = uniform_filter(image, size=window)
    mean_sq = uniform_filter(image ** 2, size=window)
    return np.sqrt(np.maximum(mean_sq - mean ** 2, 0))


def region_entropy(image: np.ndarray, window: int = 8) -> np.ndarray:
    normed = (image - image.min()) / (image.max() - image.min() + 1e-10)
    eps = 1e-10
    ent = -normed * np.log2(normed + eps) - (1 - normed) * np.log2(1 - normed + eps)
    return uniform_filter(ent, size=window)




def _image_to_lab(img: np.ndarray) -> np.ndarray:
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2Lab).astype(np.float32)


def _lab_lap_pyramid(lab: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build full 3-channel Laplacian pyramid bands for a Lab image.

    Returns levels+1 entries of shape (H_l, W_l, 3). Builds one level at a
    time, keeping only the current and next Gaussian in memory simultaneously
    so the full Gaussian stack is never allocated at once.
    The luminance Laplacian bands are the [:,:,0] slice of each entry.
    """
    bands: list[np.ndarray] = []
    current = lab
    for _ in range(levels):
        nxt = np.empty((
            (current.shape[0] + 1) // 2,
            (current.shape[1] + 1) // 2,
            3,
        ), dtype=np.float32)
        for c in range(3):
            nxt[:, :, c] = reduce(current[:, :, c])
        exp = np.empty_like(current)
        for c in range(3):
            exp[:, :, c] = expand(nxt[:, :, c], current.shape[:2])
        bands.append(current - exp)
        current = nxt
    bands.append(current)
    return bands


def _load_and_warp(
    path: Path | np.ndarray,
    warp: np.ndarray,
    size: tuple[int, int],
    border_mode: int = cv2.BORDER_REFLECT,
) -> np.ndarray:
    """Load an image (or use an already-loaded array), resize if needed, and apply a warp.

    size is (w, h) as expected by cv2.warpAffine.
    border_mode controls how regions outside the source image are filled;
    cv2.BORDER_REFLECT (default) mirrors edge pixels, cv2.BORDER_CONSTANT fills with black.
    Returns a float32 RGB array.
    """
    if isinstance(path, np.ndarray):
        img = path.astype(np.float32)
    else:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    if img.shape[1] != size[0] or img.shape[0] != size[1]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if not np.array_equal(warp, np.eye(2, 3, dtype=np.float32)):
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.warpAffine(u8, warp, size,
                             flags=cv2.INTER_CUBIC,
                             borderMode=border_mode).astype(np.float32)
    return img



def compute_canvas(
    warps: list[np.ndarray],
    src_size: tuple[int, int],
    keep_size: bool = False,
    crop: bool = False,
) -> tuple[tuple[int, int], list[np.ndarray]]:
    """Compute the output canvas size and adjusted warps for all images.

    Default: the canvas expands to the full extent of all transformed image
    corners. Every pixel has data from at least one image; border regions not
    covered by all images use reflected fill from the nearest image edge.

    With crop=True the canvas is further tightened to the intersection of all
    transformed image extents — the largest rectangle covered by every image.
    This removes all reflected-fill borders but shrinks the output.

    With keep_size=True the canvas stays at src_size and warps are unchanged.

    Returns (canvas_size, adjusted_warps) where canvas_size is (w, h) and each
    adjusted warp includes a translation that maps the bounding box origin to (0,0).
    """
    if keep_size:
        return src_size, warps

    w, h = src_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    # Transform each image's corners through its warp to find coverage in canvas space
    all_corners: list[np.ndarray] = []
    for warp in warps:
        M = warp.astype(np.float64)
        transformed = (M[:, :2] @ corners.T + M[:, 2:]).T
        all_corners.append(transformed)

    all_pts = np.concatenate(all_corners, axis=0)
    canvas_min = all_pts.min(axis=0)  # (x_min, y_min) — full extent
    canvas_max = all_pts.max(axis=0)  # (x_max, y_max) — full extent

    if crop:
        # Shift corners into canvas space first, then find the intersection
        per_img_mins = np.array([c.min(axis=0) for c in all_corners]) - canvas_min
        per_img_maxs = np.array([c.max(axis=0) for c in all_corners]) - canvas_min
        crop_min = per_img_mins.max(axis=0)  # tightest left/top edge
        crop_max = per_img_maxs.min(axis=0)  # tightest right/bottom edge
        if np.all(crop_max > crop_min):
            # Express crop bounds back in original canvas space for the shift calculation
            canvas_max = canvas_min + crop_max
            canvas_min = canvas_min + crop_min

    # Shift all warps so that canvas_min maps to (0, 0)
    tx, ty = -canvas_min
    shift = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    adjusted = []
    for warp in warps:
        adjusted.append(_chain_affines(shift, warp.astype(np.float32)))

    canvas_w = max(1, int(np.ceil(canvas_max[0] - canvas_min[0])))
    canvas_h = max(1, int(np.ceil(canvas_max[1] - canvas_min[1])))
    return (canvas_w, canvas_h), adjusted


def stack_images(
    src_paths: list[Path | np.ndarray],
    warps: list[np.ndarray],
    levels: int,
    sharpness: float,
    dark_threshold: float,
    canvas_size: tuple[int, int] | None = None,
    no_fill: bool = False,
    workers: int = 3,
) -> np.ndarray:
    """Fuse a stack of images using single-pass unnormalized Laplacian pyramid fusion.

    Accumulates energy-weighted Lab bands and energy sums in one disk read per
    image, then divides at the end. Mathematically identical to normalized fusion:
      Σ(e_k / Σe_k · x_k) = Σ(e_k · x_k) / Σe_k

    canvas_size overrides the output dimensions (w, h). If None, the size of the
    first image is used (equivalent to --keep-size behaviour).

    no_fill uses BORDER_CONSTANT (black) instead of BORDER_REFLECT for regions
    outside each image's coverage after warping.

    workers controls how many images are processed concurrently. Peak RAM scales
    with workers × ~100 MiB per image plus the fixed fused_lp accumulator.
    Default of 3 workers balances speed and memory for most systems.
    """

    if isinstance(src_paths[0], np.ndarray):
        h, w = src_paths[0].shape[:2]
    else:
        img0 = np.array(Image.open(src_paths[0]).convert("RGB"), dtype=np.float32)
        h, w = img0.shape[:2]
        del img0
    cv2_size = canvas_size if canvas_size is not None else (w, h)
    border_mode = cv2.BORDER_CONSTANT if no_fill else cv2.BORDER_REFLECT

    n_workers = min(len(src_paths), workers if workers > 0 else (os.cpu_count() or 4))

    def _process_image(args: tuple[Path | np.ndarray, np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        path, warp = args
        lab = _image_to_lab(_load_and_warp(path, warp, cv2_size, border_mode))
        lap = _lab_lap_pyramid(lab, levels)
        del lab
        energies: list[np.ndarray] = []
        for i in range(levels):
            energies.append(region_energy(lap[i][:, :, 0]) ** sharpness)
        lv = lap[-1][:, :, 0]
        energies.append(((region_deviation(lv) + region_entropy(lv)) * 0.5) ** sharpness)
        return energies, lap

    t = time.perf_counter()
    print(f"  Fusing ({n_workers} workers)...")
    energy_sums: list[np.ndarray | None] = [None] * (levels + 1)
    fused_lp: list[np.ndarray | None] = [None] * (levels + 1)
    weighted_buf: np.ndarray | None = None

    items = list(zip(src_paths, warps))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending: dict = {}
        submit_idx = 0

        while submit_idx < len(items) and len(pending) < n_workers:
            f = pool.submit(_process_image, items[submit_idx])
            pending[f] = submit_idx
            submit_idx += 1

        while pending:
            done = next(as_completed(pending))
            pending.pop(done)

            if submit_idx < len(items):
                f = pool.submit(_process_image, items[submit_idx])
                pending[f] = submit_idx
                submit_idx += 1

            energies, lap = done.result()
            for i in range(levels + 1):
                e = energies[i]
                band = lap[i]
                lap[i] = None  # type: ignore[call-overload]
                energy_sums[i] = e if energy_sums[i] is None else energy_sums[i] + e
                if weighted_buf is None or weighted_buf.shape != band.shape:
                    weighted_buf = np.empty_like(band)
                np.multiply(band, e[:, :, np.newaxis], out=weighted_buf)
                if fused_lp[i] is None:
                    fused_lp[i] = weighted_buf.copy()
                else:
                    fused_lp[i] += weighted_buf  # type: ignore[operator]

    for i in range(levels + 1):
        fused_lp[i] /= energy_sums[i][:, :, np.newaxis] + 1e-10  # type: ignore[operator,index]

    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    print("  Reconstructing...")
    image = fused_lp[-1].copy()  # type: ignore[union-attr]
    for band in reversed(fused_lp[:-1]):
        cur_shape = band.shape[:2]  # type: ignore[union-attr]
        exp = np.stack([expand(image[:, :, c], cur_shape) for c in range(3)], axis=-1)
        image = exp + band  # type: ignore[operator]
    fused_lab = image
    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    print("  Suppressing chroma in dark regions...")
    fused_lab = _suppress_dark_chroma(fused_lab, dark_threshold)
    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    print("  Converting back to RGB...")
    result = cv2.cvtColor(np.clip(fused_lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2RGB)
    print(f"    done ({_elapsed(t)})")

    return result


def _suppress_dark_chroma(fused_lab: np.ndarray, threshold: float) -> np.ndarray:
    """Lerp a/b channels toward neutral (128) in dark regions.

    In Lab (OpenCV uint8 encoding) a neutral/achromatic pixel has a=128, b=128.
    Floating point drift during pyramid reconstruction can push dark pixels away
    from neutral, producing visible color casts in areas that should be black.
    The mask is 0 where L=0 and ramps linearly to 1 at L=threshold, clamped
    to 1 above that, so only genuinely dark pixels are affected.
    """
    l = fused_lab[:, :, 0]
    mask = np.clip(l / (threshold + 1e-10), 0.0, 1.0)
    result = fused_lab.copy()
    result[:, :, 1] = 128.0 + (fused_lab[:, :, 1] - 128.0) * mask
    result[:, :, 2] = 128.0 + (fused_lab[:, :, 2] - 128.0) * mask
    return result


def compute_levels(shape: tuple[int, int], max_levels: int = 6) -> int:
    min_dim = min(shape[0], shape[1])
    levels = 0
    size = min_dim
    while size > 16 and levels < max_levels:
        size //= 2
        levels += 1
    return levels


def _save_image(img: np.ndarray, path: Path, quality: int) -> None:
    fmt = path.suffix.lower().lstrip(".")
    fmt = "jpeg" if fmt in ("jpg", "jpeg") else fmt.upper()
    save_kwargs: dict = {"quality": quality} if fmt == "jpeg" else {}
    Image.fromarray(img).save(path, fmt, **save_kwargs)


def _compute_slabs(n: int, slab_size: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for a list of n items."""
    step = max(1, slab_size - overlap)
    slabs: list[tuple[int, int]] = []
    for s in range(0, n, step):
        end = min(s + slab_size, n)
        slabs.append((s, end))
        if end == n:
            break
    return slabs


def slab_images(
    src_paths: list[Path],
    adjusted_warps: list[np.ndarray],
    slab_size: int,
    overlap: int,
    levels: int,
    sharpness: float,
    dark_threshold: float,
    canvas_size: tuple[int, int],
    no_fill: bool,
    workers: int,
    output_steps: bool,
    steps_dir: Path | None,
    quality: int,
    only_slab: bool,
    recursive: bool,
    final_extension: str,
) -> list[Path] | np.ndarray:
    """Stack images using slabbing, optionally recursive.

    Layer 1 splits src_paths into overlapping sub-stacks and stacks each one.
    With recursive=True, if the layer-1 results still outnumber slab_size the
    same split is applied to those results as layer 2, and so on, until the
    remaining count fits within a single stack pass.
    With recursive=False, the layer-1 results are fused in one final stack
    regardless of how many there are.

    With only_slab=True recursion is ignored: layer 1 slabs are saved (if
    output_steps) and the function returns immediately so the user can touch
    up the intermediates before a manual final stack.

    With output_steps=True each layer's slab results are saved to steps_dir
    using the naming scheme slab_<layer>_<NNN>.<ext>.
    """
    identity = np.eye(2, 3, dtype=np.float32)
    final_ext = final_extension.lstrip(".")

    # current_items holds either the original source Paths (layer 1) or
    # stacked result arrays from the previous layer (layer 2+).
    current_items: list[Path | np.ndarray] = list(src_paths)
    current_warps: list[np.ndarray] = list(adjusted_warps)
    current_canvas: tuple[int, int] = canvas_size

    layer = 1
    while True:
        n = len(current_items)
        slabs = _compute_slabs(n, slab_size, overlap)

        indent = "  " * layer
        print(f"{indent}Layer {layer}: {len(slabs)} slabs of up to {slab_size} from {n} images, {overlap} overlap")

        slab_paths: list[Path] = []
        slab_arrays: list[np.ndarray] = []

        for idx, (start, end) in enumerate(slabs):
            label = f"slab_{layer}_{idx + 1:03d}"
            print(f"{indent}  Stacking {label} (images {start + 1}-{end})...")
            result = stack_images(
                current_items[start:end], current_warps[start:end],
                levels, sharpness, dark_threshold, current_canvas, no_fill, workers,
            )

            if output_steps and steps_dir is not None:
                slab_file = steps_dir / f"{label}.{final_ext}"
                _save_image(result, slab_file, quality)
                print(f"{indent}    Saved: {slab_file}")
                slab_paths.append(slab_file)

            slab_arrays.append(result)

        # --only-slab: stop after the first layer regardless of how many slabs
        # were produced, so the user can inspect and manually edit them.
        if only_slab:
            if not output_steps or steps_dir is None:
                print("Warning: --only-slab used without --output-steps; slab images were not saved to disk.")
            return slab_paths

        # If this layer produced only one slab, that result is the final image.
        if len(slab_arrays) == 1:
            return slab_arrays[0]

        # If all slabs fit in one more pass, or recursion is disabled,
        # fuse the slab arrays directly without writing anything to disk.
        if len(slab_arrays) <= slab_size or not recursive:
            print(f"{indent}  Final stack: merging {len(slab_arrays)} slab results...")
            final_warps = [identity.copy() for _ in slab_arrays]
            return stack_images(
                slab_arrays, final_warps, levels, sharpness, dark_threshold,
                current_canvas, no_fill, workers,
            )

        # recursive=True and still too many results: loop into the next layer
        # passing the arrays directly — no disk writes needed.
        current_items = list(slab_arrays)
        current_warps = [identity.copy() for _ in slab_arrays]
        current_canvas = canvas_size
        layer += 1


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
            "Index of the image to use as the alignment reference (default: 0, i.e. the first image). "
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
            "Useful range is roughly 1.0 (soft) to 16.0 (near-hard)."
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

    checkpoints: list[_Checkpoint] = []
    t_total = time.perf_counter()
    tracemalloc.start()
    t = t_total

    print(f"Loading images from '{args.folder}'...")
    src_paths, reference_size = load_images(args.folder)
    t = _snap("Load", t, checkpoints)

    n_images = len(src_paths)
    reference = args.reference if args.reference >= 0 else n_images // 2
    if not (0 <= reference < n_images):
        print(f"Error: --reference {args.reference} is out of range (0\u2013{n_images - 1}).")
        sys.exit(1)

    ref_w, ref_h = reference_size
    levels = args.levels if args.levels > 0 else compute_levels((ref_h, ref_w))
    print(f"Image size: {ref_w}x{ref_h}  |  Images: {n_images}  |  Pyramid levels: {levels}  |  Sharpness: {args.sharpness}  |  Dark threshold: {args.dark_threshold}")

    identity = np.eye(2, 3, dtype=np.float32)
    if not args.no_align:
        strategy = "global" if args.global_align else "neighbour-chained"
        print(f"Aligning (ECC affine, {strategy}, reference image {reference + 1})...")
        warps = align_images(
            src_paths,
            reference_size,
            reference_idx=reference,
            global_align=args.global_align,
            no_rotation=args.no_rotation,
            no_scale=args.no_scale,
            no_shear=args.no_shear,
            no_translation=args.no_translation,
            full_res=args.full_res,
            min_shift=args.min_shift,
        )
        t = _snap("Align", t, checkpoints)
    else:
        print("Skipping alignment.")
        warps = [identity.copy() for _ in src_paths]

    use_slabs = args.slab is not None
    only_slab = args.only_slab
    output_steps = args.output_steps or only_slab

    out_path = args.output if args.output is not None else args.folder / "stacked.jpg"

    out_extension = out_path.suffix.lower().lstrip(".")

    canvas_size, adjusted_warps = compute_canvas(warps, reference_size, keep_size=args.keep_size, crop=args.crop)

    if use_slabs:
        slab_size, overlap = args.slab
        if slab_size < 2:
            print("Error: slab SIZE must be at least 2.")
            sys.exit(1)
        if overlap < 0 or overlap >= slab_size:
            print(f"Error: slab OVERLAP must be >= 0 and < SIZE ({slab_size}).")
            sys.exit(1)

        steps_dir: Path | None = None
        if output_steps:
            steps_dir = out_path.parent / "focusweave_slabs"
            steps_dir.mkdir(parents=True, exist_ok=True)

        print("Slabbing...")
        slab_result = slab_images(
            src_paths=src_paths,
            adjusted_warps=adjusted_warps,
            slab_size=slab_size,
            overlap=overlap,
            levels=levels,
            sharpness=args.sharpness,
            dark_threshold=args.dark_threshold,
            canvas_size=canvas_size,
            no_fill=args.no_fill,
            workers=args.workers,
            output_steps=output_steps,
            steps_dir=steps_dir,
            quality=args.quality,
            only_slab=only_slab,
            recursive=args.recursive_slab and not only_slab,
            final_extension=args.slab_format.lstrip(".") if args.slab_format else "tiff",
        )
        t = _snap("Slab", t, checkpoints)

        if only_slab:
            tracemalloc.stop()
            if steps_dir is not None:
                print(f"Slabs saved to: {steps_dir}")
            _print_report(checkpoints, time.perf_counter() - t_total)
            return

        result = slab_result  # type: ignore[assignment]
    else:
        print("Stacking...")
        result = stack_images(src_paths, adjusted_warps, levels, args.sharpness, args.dark_threshold, canvas_size, args.no_fill, args.workers)
        t = _snap("Stack", t, checkpoints)

    _save_image(result, out_path, args.quality)  # type: ignore[arg-type]
    t = _snap("Save", t, checkpoints)

    tracemalloc.stop()
    print(f"Saved: {out_path}")
    _print_report(checkpoints, time.perf_counter() - t_total)



if __name__ == "__main__":
    main()