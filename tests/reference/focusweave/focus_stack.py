from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


_K1D = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
_K1D_X2 = _K1D * 2

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})


Stage = Literal["loading", "culling", "aligning", "stacking", "slabbing", "complete"]

ProgressCallback = Callable[[float, Stage, str], None]
SlabCallback = Callable[[str, np.ndarray], None]
InterruptCallback = Callable[[], bool]


class Interrupted(Exception):
    """Raised when an interrupt callback signals that the run should stop."""


@dataclass
class CullEntry:
    path: Path | np.ndarray
    score: float
    kept: bool


@dataclass
class CullResult:
    entries: list[CullEntry]
    cutoff: float
    n_culled: int

    @property
    def kept(self) -> list[Path | np.ndarray]:
        return [e.path for e in self.entries if e.kept]


@dataclass
class RunResult:
    image: np.ndarray | None
    slabs: list[np.ndarray] | None


def _cv2_image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image file using cv2. Raises ValueError on failure."""
    probe = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if probe is None:
        raise ValueError(f"cv2 could not read image: '{path}'")
    h, w = probe.shape[:2]
    return w, h


def load_images(folder: Path) -> tuple[list[Path], tuple[int, int]]:
    """Discover image paths in a folder and return them with the reference size.

    Paths are sorted alphabetically. Raises ValueError if fewer than 2 images
    are found. The reference size is read from the first image.
    """
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in '{folder}', found {len(paths)}.")

    return paths, _cv2_image_size(paths[0])


def resolve_images(
    images: Path | list[Path] | list[np.ndarray],
) -> tuple[list[Path] | list[np.ndarray], tuple[int, int]]:
    """Resolve the images argument into a uniform (items, reference_size) pair.

    Accepts:
      - a Path to a folder: discovers image files, reads size from the first.
      - a list of Paths: uses them as-is, reads size from the first file.
      - a list of ndarrays: uses them as-is, reads size from the first array.

    Raises ValueError if fewer than 2 images are provided.
    """
    if isinstance(images, Path):
        return load_images(images)

    if len(images) < 2:
        raise ValueError(f"Need at least 2 images, got {len(images)}.")

    if isinstance(images[0], np.ndarray):
        imgs: list[np.ndarray] = images  # type: ignore[assignment]
        h, w = imgs[0].shape[:2]
        return imgs, (w, h)

    paths: list[Path] = images  # type: ignore[assignment]
    return paths, _cv2_image_size(paths[0])


def _tenengrad_score_map(
    path: Path | np.ndarray,
    reference_size: tuple[int, int],
    ksize: int = 5,
    max_resolution: int = 1024,
) -> tuple[np.ndarray, float]:
    """Compute the Tenengrad score map for a single image.

    Accepts a file path or a pre-loaded uint8 RGB ndarray. Loads and downscales
    the image to max_resolution on the long edge, applies CLAHE normalisation,
    then returns (score_map, scale) where score_map is the raw float32
    (Gx² + Gy²) array at the downscaled resolution and scale is the factor
    applied (1.0 if no downscaling was needed). The caller is responsible for
    deriving scalar summaries and saving debug output from the returned map.
    """
    img = _load_raw_u8(path, reference_size)

    long_edge = max(img.shape[:2])
    scale = 1.0
    if long_edge > max_resolution:
        scale = max_resolution / long_edge
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=ksize)
    return gx ** 2 + gy ** 2, scale


def _score_map_to_scalar(score_map: np.ndarray) -> float:
    """Summarise a 2D score map as the high-to-low spatial frequency energy ratio.

    The score map is normalised to [0, 1] then split into a low-frequency
    component (Gaussian blur with a large kernel) and a high-frequency residual
    (original minus blurred). The metric is the ratio of HF energy to LF energy
    computed over the non-background subject pixels (norm > 0.01).

    Why this works:
    - Genuine fine texture in the source image produces compact, dot-like bright
      spots in the score map: high HF energy relative to LF.
    - Thick halo edges and diffuse glows produce large smooth blobs in the score
      map: high LF energy, low HF/LF ratio.
    - Uniformly blurry images have low overall energy in both bands: low ratio.

    The ratio is scale-invariant (division by LF energy) so frames with different
    overall brightness levels are compared fairly.
    """
    norm = score_map / (float(score_map.max()) + 1e-8)
    lf = cv2.GaussianBlur(norm, (31, 31), 0)
    hf = norm - lf
    subj = norm > 0.01
    lf_energy = float((lf[subj] ** 2).mean())
    hf_energy = float((hf[subj] ** 2).mean())
    return hf_energy / lf_energy if lf_energy > 0 else 0.0


def _compute_all_score_maps(
    images: list[Path | np.ndarray],
    reference_size: tuple[int, int],
    ksize: int = 5,
    max_resolution: int = 1024,
    progress: ProgressCallback | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Load and compute Tenengrad score maps and scalar scores for all images.

    Returns (score_maps, scores).

    progress is called as progress(fraction, stage, message) after each image.
    """
    maps: list[np.ndarray] = []
    scores: list[float] = []
    n = len(images)
    for i, img in enumerate(images):
        score_map, _ = _tenengrad_score_map(img, reference_size, ksize=ksize,
                                            max_resolution=max_resolution)
        score = _score_map_to_scalar(score_map)
        maps.append(score_map)
        scores.append(score)
        if progress is not None:
            label = img.name if isinstance(img, Path) else f"image_{i}"
            progress((i + 1) / n, "culling", f"Scored {label}  ({score:.4f})")
    return maps, scores


def cull_unfocused_images(
    images: list[Path | np.ndarray],
    reference_size: tuple[int, int],
    threshold: float = 0.05,
    progress: ProgressCallback | None = None,
) -> CullResult:
    """Remove images whose focus score falls below threshold.

    Each image is scored by the HF/LF ratio of its Tenengrad score map
    (see _score_map_to_scalar). A frame is culled when its raw score is below
    threshold.

    At least the two sharpest frames are always retained so the stack can
    proceed even when threshold is set very high.

    progress is called as progress(fraction, stage, message) after each image.

    Raises ValueError if fewer than 2 images survive (guards against degenerate
    inputs where the safety floor itself cannot produce 2 valid frames).
    """
    score_maps, scores = _compute_all_score_maps(images, reference_size,
                                                 progress=progress)

    peak = max(scores)
    if peak == 0.0:
        entries = [CullEntry(path=img, score=s, kept=True) for img, s in zip(images, scores)]
        return CullResult(entries=entries, cutoff=threshold, n_culled=0)

    keep_flags = [s >= threshold for s in scores]

    if sum(keep_flags) < 2:
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in ranked[:2]:
            keep_flags[idx] = True

    entries = [
        CullEntry(path=img, score=s, kept=k)
        for img, s, k in zip(images, scores, keep_flags)
    ]
    n_culled = sum(1 for e in entries if not e.kept)
    result = CullResult(entries=entries, cutoff=threshold, n_culled=n_culled)

    if len(result.kept) < 2:
        raise ValueError(
            "Fewer than 2 images survived culling. "
            "Lower --cull-threshold or disable --cull."
        )
    return result


def _to_gray_cv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a uint8 grayscale image to boost local contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _phase_correlation_translation(
    ref_gray: np.ndarray,
    src_gray: np.ndarray,
    max_resolution: int = 512,
) -> tuple[float, float]:
    """Estimate translation via phase correlation in the frequency domain.

    Downscales both images to max_resolution on the long edge, computes the
    normalised cross-power spectrum, and returns the peak shift (tx, ty) scaled
    back to full resolution. This is robust on low-contrast and textureless
    regions where gradient-based methods struggle because it operates globally
    across all frequencies rather than chasing local intensity gradients.
    """
    h, w = ref_gray.shape
    scale = 1.0
    if max(h, w) > max_resolution:
        scale = max_resolution / max(h, w)
        ref_s = cv2.resize(ref_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        src_s = cv2.resize(src_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        ref_s, src_s = ref_gray, src_gray

    ref_f = ref_s.astype(np.float32)
    src_f = src_s.astype(np.float32)

    shift, _ = cv2.phaseCorrelate(src_f, ref_f)
    return shift[0] / scale, shift[1] / scale


def _focus_mask(gray: np.ndarray, percentile: float = 30.0) -> np.ndarray:
    """Return a uint8 mask of the sharpest pixels in a grayscale image.

    Computes a Laplacian variance map (local sharpness), then thresholds at
    the given percentile so only the sharper fraction of pixels are kept.
    The mask is dilated slightly so ECC has continuous regions to work with
    rather than isolated islands.

    This is the key to reliable macro alignment: ECC is only asked to match
    pixels that are actually sharp and informative in both frames. Blurry
    out-of-focus regions — which dominate macro frames and contain misleading
    intensity patterns — are excluded from the cost function entirely.
    """
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    sharpness = cv2.GaussianBlur(lap ** 2, (15, 15), 0)
    threshold = float(np.percentile(sharpness, percentile))
    mask = (sharpness >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(mask, kernel)


def _prepare_for_ecc(gray: np.ndarray, max_resolution: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Downscale, CLAHE-equalise, and compute the focus mask for one image.

    Returns (clahe_small, mask, scale) where scale is the downscale factor
    applied (1.0 if no downscaling was needed). Separating this from _ecc_align
    allows callers to cache the result per image per resolution, avoiding
    redundant recomputation across the two ECC passes and across the forward/
    backward neighbour chains where each image appears as both ref and src.
    """
    h, w = gray.shape
    if max(h, w) > max_resolution:
        scale = max_resolution / max(h, w)
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = gray
    equalised = _apply_clahe(small)
    mask = _focus_mask(equalised)
    return equalised, mask, scale


def _ecc_align(
    ref_prepared: tuple[np.ndarray, np.ndarray, float],
    src_prepared: tuple[np.ndarray, np.ndarray, float],
    rough: bool,
    init_warp: np.ndarray | None = None,
    relaxed: bool = False,
) -> np.ndarray:
    """Single cv2 ECC alignment pass using pre-prepared image data.

    Accepts pre-computed (clahe_small, mask, scale) tuples from _prepare_for_ecc
    so CLAHE and focus mask computation are never repeated across passes or chains.

    Combines the ref and src masks so ECC only fits pixels that are sharp in
    both frames. Falls back to the union if the intersection is too sparse,
    and to no mask if the union is also too sparse.

    relaxed=True uses looser termination criteria and more Gaussian smoothing
    levels, useful as a fallback for difficult image pairs.
    """
    ref_small, ref_mask, ref_scale = ref_prepared
    src_small, src_mask, src_scale = src_prepared
    scale = ref_scale

    combined_mask = cv2.bitwise_and(ref_mask, src_mask)
    if cv2.countNonZero(combined_mask) < 100:
        combined_mask = cv2.bitwise_or(ref_mask, src_mask)
    input_mask = combined_mask if cv2.countNonZero(combined_mask) >= 100 else None

    warp = init_warp.copy() if init_warp is not None else np.eye(2, 3, dtype=np.float32)
    if scale != 1.0:
        warp[0, 2] *= scale
        warp[1, 2] *= scale

    if rough:
        if relaxed:
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 0.05)
            gauss_levels = 3
        else:
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, 0.01)
            gauss_levels = 1
    else:
        if relaxed:
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.005)
            gauss_levels = 5
        else:
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 0.001)
            gauss_levels = 3

    cv2.findTransformECC(src_small.astype(np.float32),
                         ref_small.astype(np.float32),
                         warp, cv2.MOTION_AFFINE, criteria,
                         input_mask, gauss_levels)

    warp[0, 2] /= scale
    warp[1, 2] /= scale
    return warp


def _chain_affines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine matrices: apply b then a."""
    a3 = np.vstack([a, [0, 0, 1]])
    b3 = np.vstack([b, [0, 0, 1]])
    return (a3 @ b3)[:2]


def _validate_warp(
    warp: np.ndarray,
    seed_warp: np.ndarray,
    image_long_edge: int,
    translation_tolerance: float = 0.15,
    affine_distortion_limit: float = 0.02,
) -> bool:
    """Return True if warp is plausible for a focus stack frame.

    Two independent checks:

    Translation agreement: the ECC translation must not disagree with the
    phase-correlation seed by more than translation_tolerance * image_long_edge.
    A large disagreement means ECC wandered to a false minimum — the
    phase-correlation estimate is a far more reliable anchor on low-texture
    subjects.

    Affine distortion: the linear part of the warp (rotation, scale, shear
    combined) should be close to identity for a focus stack where the camera
    hasn't moved. The Frobenius distance of the 2x2 linear block from I is
    bounded by affine_distortion_limit. A warp that passes ECC convergence
    but returns a large affine distortion is almost certainly a false minimum.
    """
    ecc_tx, ecc_ty = warp[0, 2], warp[1, 2]
    seed_tx, seed_ty = seed_warp[0, 2], seed_warp[1, 2]
    translation_disagreement = np.hypot(ecc_tx - seed_tx, ecc_ty - seed_ty)
    if translation_disagreement > translation_tolerance * image_long_edge:
        return False

    linear = warp[:, :2].astype(np.float64)
    affine_distortion = float(np.linalg.norm(linear - np.eye(2), ord="fro"))
    if affine_distortion > affine_distortion_limit:
        return False

    return True


def _run_ecc(
    ref_gray: np.ndarray,
    src_gray: np.ndarray,
    full_res: bool = False,
    ref_prepared_cache: dict[int, tuple[np.ndarray, np.ndarray, float]] | None = None,
    src_prepared_cache: dict[int, tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> tuple[np.ndarray, bool]:
    """Run ECC alignment seeded by phase correlation.

    Phase correlation provides a reliable translation seed, so a separate
    coarse ECC pass at 256px is not needed. A single fine pass at 1024px
    (or full resolution when full_res=True) gives sub-pixel accuracy with
    roughly 4× less work than running at 2048px.

    Returns (warp, converged).
    """
    identity = np.eye(2, 3, dtype=np.float32)
    if full_res:
        fine_res = 2 ** 31
    else:
        long_edge = max(ref_gray.shape)
        fine_res = min(1024, long_edge)

    image_long_edge = max(ref_gray.shape)
    tx, ty = _phase_correlation_translation(ref_gray, src_gray)
    seed_warp = identity.copy()
    seed_warp[0, 2] = tx
    seed_warp[1, 2] = ty

    if ref_prepared_cache is not None and fine_res in ref_prepared_cache:
        ref_prep = ref_prepared_cache[fine_res]
    else:
        ref_prep = _prepare_for_ecc(ref_gray, fine_res)
        if ref_prepared_cache is not None:
            ref_prepared_cache[fine_res] = ref_prep

    if src_prepared_cache is not None and fine_res in src_prepared_cache:
        src_prep = src_prepared_cache[fine_res]
    else:
        src_prep = _prepare_for_ecc(src_gray, fine_res)
        if src_prepared_cache is not None:
            src_prepared_cache[fine_res] = src_prep

    warp = seed_warp.copy()
    try:
        warp = _ecc_align(ref_prep, src_prep, rough=False, init_warp=warp)
    except cv2.error:
        try:
            warp = _ecc_align(ref_prep, src_prep, rough=False,
                              init_warp=seed_warp, relaxed=True)
        except cv2.error:
            return identity.copy(), False

    if not _validate_warp(warp, seed_warp, image_long_edge):
        return identity.copy(), False

    return warp, True


def _warp_message(label: str, warp: np.ndarray) -> str:
    angle = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
    tx, ty = warp[0, 2], warp[1, 2]
    return f"{label}: rotation {angle:+.2f}°  shift ({ty:+.1f}, {tx:+.1f}) px"


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
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            U[:, -1] *= -1
            sv[-1] *= -1
        R = U @ Vt
        if no_scale:
            geomean = np.sqrt(abs(sv[0] * sv[1]))
            sv = sv / (geomean if geomean > 1e-10 else 1.0)
        if no_rotation and no_shear:
            result[:, :2] = np.diag(sv)
        elif no_rotation:
            result[:, :2] = Vt.T * sv @ Vt
        elif no_shear:
            result[:, :2] = R @ np.diag(sv)
        else:
            result[:, :2] = (U * sv) @ Vt
    if no_translation:
        result[:, 2] = 0.0
    return result.astype(np.float32)


def align_images(
    images: list[Path | np.ndarray],
    reference_size: tuple[int, int],
    reference_idx: int = 0,
    global_align: bool = False,
    no_rotation: bool = False,
    no_scale: bool = False,
    no_shear: bool = False,
    no_translation: bool = False,
    full_res: bool = False,
    min_shift: float = 5.0,
    workers: int = 0,
    progress: ProgressCallback | None = None,
    interrupt: InterruptCallback | None = None,
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

    progress is called as progress(fraction, stage, message) after each image is aligned,
    where fraction is in [0, 1] relative to the total number of non-reference images.
    The message describes the alignment result for that image.

    interrupt is called after each image; if it returns True, Interrupted is raised.
    """
    ref_w, ref_h = reference_size
    identity = np.eye(2, 3, dtype=np.float32)
    n = len(images)
    n_to_align = n - 1

    def _load_raw_gray(img: Path | np.ndarray) -> np.ndarray:
        return _to_gray_cv(_load_raw_u8(img, (ref_w, ref_h)))

    def _is_negligible(warp: np.ndarray) -> bool:
        translation = np.linalg.norm(warp[:, 2])
        linear_is_identity = np.allclose(warp[:, :2], np.eye(2), atol=1e-3)
        return translation < min_shift and linear_is_identity

    def _constrain(warp: np.ndarray) -> np.ndarray:
        return _constrain_warp(warp, no_rotation, no_scale, no_shear, no_translation)

    def _notify(done: int, message: str) -> None:
        if progress is not None:
            progress(done / max(n_to_align, 1), "aligning", message)
        if interrupt is not None and interrupt():
            raise Interrupted

    ref_gray = _load_raw_gray(images[reference_idx])
    warps: list[np.ndarray] = [identity.copy()] * n
    warps[reference_idx] = identity.copy()
    aligned = 0

    # Pre-load all grayscale images and pre-prepare CLAHE+mask data in parallel.
    # The ECC chain is serial (each step depends on the previous warp), but loading
    # and _prepare_for_ecc are pure I/O + compute with no inter-image dependencies.
    # Pre-filling the cache here removes both from the serial critical path.
    if full_res:
        fine_res = 2 ** 31
    else:
        long_edge = max(ref_gray.shape)
        fine_res = min(1024, long_edge)

    grays: list[np.ndarray] = [ref_gray if i == reference_idx else None  # type: ignore[list-item]
                                for i in range(n)]
    prepared: list[dict[int, tuple[np.ndarray, np.ndarray, float]]] = [{} for _ in range(n)]

    def _preload_and_prepare(i: int) -> None:
        if grays[i] is None:
            grays[i] = _load_raw_gray(images[i])
        prepared[i][fine_res] = _prepare_for_ecc(grays[i], fine_res)

    n_prep_workers = min(n, workers if workers > 0 else (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n_prep_workers) as pool:
        list(pool.map(_preload_and_prepare, range(n)))

    def _run(ref_idx: int, ref: np.ndarray, src_idx: int, src: np.ndarray) -> tuple[np.ndarray, bool]:
        return _run_ecc(ref, src, full_res,
                        ref_prepared_cache=prepared[ref_idx],
                        src_prepared_cache=prepared[src_idx])

    if global_align:
        for i in range(n):
            if i == reference_idx:
                continue
            src_gray = grays[i]
            warp, converged = _run(reference_idx, ref_gray, i, src_gray)
            warp = _constrain(warp)
            label = f"Image {i + 1}"
            if not converged:
                warps[i] = identity.copy()
                msg = f"{label}: ECC did not converge — using identity"
            elif _is_negligible(warp):
                warps[i] = identity.copy()
                msg = f"{label}: transform negligible — skipped"
            else:
                warps[i] = warp
                msg = _warp_message(label, warp)
            aligned += 1
            _notify(aligned, msg)
        return warps

    # Neighbour-chained: forward pass (reference_idx -> end)
    cumulative = identity.copy()
    prev_gray = ref_gray
    prev_idx = reference_idx
    for i in range(reference_idx + 1, n):
        src_gray = grays[i]
        warp, converged = _run(prev_idx, prev_gray, i, src_gray)
        warp = _constrain(warp)
        prev_gray = src_gray
        prev_idx = i
        label = f"Image {i + 1}"
        if not converged:
            warps[i] = cumulative.copy()
            msg = f"{label}: ECC did not converge — using previous transform"
        else:
            cumulative = _constrain(_chain_affines(cumulative, warp))
            if no_rotation:
                cumulative[:, :2] = np.eye(2)
            if _is_negligible(cumulative):
                warps[i] = identity.copy()
                msg = f"{label}: transform negligible — skipped"
            else:
                warps[i] = cumulative.copy()
                msg = _warp_message(label, cumulative)
        aligned += 1
        _notify(aligned, msg)

    # Neighbour-chained: backward pass (reference_idx -> start)
    cumulative = identity.copy()
    prev_gray = ref_gray
    prev_idx = reference_idx
    for i in range(reference_idx - 1, -1, -1):
        src_gray = grays[i]
        warp, converged = _run(prev_idx, prev_gray, i, src_gray)
        warp = _constrain(warp)
        prev_gray = src_gray
        prev_idx = i
        label = f"Image {i + 1}"
        if not converged:
            warps[i] = cumulative.copy()
            msg = f"{label}: ECC did not converge — using previous transform"
        else:
            cumulative = _constrain(_chain_affines(cumulative, warp))
            if no_rotation:
                cumulative[:, :2] = np.eye(2)
            if _is_negligible(cumulative):
                warps[i] = identity.copy()
                msg = f"{label}: transform negligible — skipped"
            else:
                warps[i] = cumulative.copy()
                msg = _warp_message(label, cumulative)
        aligned += 1
        _notify(aligned, msg)

    return warps


def reduce(image: np.ndarray) -> np.ndarray:
    smoothed = cv2.sepFilter2D(image, cv2.CV_32F, _K1D, _K1D,
                               borderType=cv2.BORDER_REFLECT)
    return smoothed[::2, ::2]


def expand(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    upsampled = np.zeros((h * 2, w * 2), dtype=np.float32)
    upsampled[::2, ::2] = image
    expanded = cv2.sepFilter2D(upsampled, cv2.CV_32F, _K1D_X2, _K1D_X2,
                               borderType=cv2.BORDER_REFLECT)
    return expanded[: target_shape[0], : target_shape[1]]


def region_energy(lp_level: np.ndarray, window: int = 3) -> np.ndarray:
    return cv2.sqrBoxFilter(lp_level, cv2.CV_32F, (window, window), normalize=True,
                            borderType=cv2.BORDER_REFLECT)


def region_deviation(image: np.ndarray, window: int = 3) -> np.ndarray:
    mean = cv2.boxFilter(image, cv2.CV_32F, (window, window),
                         borderType=cv2.BORDER_REFLECT)
    sq_mean = cv2.sqrBoxFilter(image, cv2.CV_32F, (window, window), normalize=True,
                               borderType=cv2.BORDER_REFLECT)
    return cv2.sqrt(np.maximum(sq_mean - mean * mean, 0.0))


def region_entropy(image: np.ndarray, window: int = 8) -> np.ndarray:
    normed = (image - image.min()) / (image.max() - image.min() + 1e-10)
    eps = 1e-10
    ent = (-normed * np.log2(normed + eps) - (1 - normed) * np.log2(1 - normed + eps)).astype(np.float32)
    return cv2.boxFilter(ent, cv2.CV_32F, (window, window),
                         borderType=cv2.BORDER_REFLECT)


def _source_depth(src: Path | np.ndarray) -> int:
    """Return 8 or 16 based on the bit depth of the first source image or array.

    For file paths, reads the file header only (IMREAD_UNCHANGED on a 1x1 crop
    is not possible, so we read the full file but only inspect dtype). For
    ndarrays, inspects dtype directly. Anything that is not uint16 is treated
    as 8-bit.
    """
    if isinstance(src, np.ndarray):
        return 16 if src.dtype == np.uint16 else 8
    raw = cv2.imread(str(src), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise ValueError(f"cv2 could not read image: '{src}'")
    return 16 if raw.dtype == np.uint16 else 8


def _image_to_lab(img: np.ndarray, max_val: float) -> np.ndarray:
    """Convert a float32 RGB image to Lab float32.

    max_val is the white-point of img (255.0 for 8-bit sources, 65535.0 for
    16-bit). cv2 requires uint8 [0, 255] input so the image is scaled to that
    range before conversion. The resulting Lab values use cv2's uint8 encoding
    (L in [0, 255], a/b in [0, 255] with neutral at 128).
    """
    u8 = np.clip(img * (255.0 / max_val), 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2Lab).astype(np.float32)


def _lab_lap_pyramid(lab: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build full 3-channel Laplacian pyramid bands for a Lab image.

    Returns levels+1 entries of shape (H_l, W_l, 3). Uses cv2.sepFilter2D
    for vectorized separable convolution across all three channels at once,
    which is significantly faster than per-channel scipy convolve1d.
    """
    bands: list[np.ndarray] = []
    current = lab
    for _ in range(levels):
        h, w = current.shape[:2]
        nh, nw = (h + 1) // 2, (w + 1) // 2
        smoothed = cv2.sepFilter2D(current, cv2.CV_32F, _K1D, _K1D,
                                   borderType=cv2.BORDER_REFLECT)
        nxt = smoothed[::2, ::2, :].copy()

        up = np.zeros((nh * 2, nw * 2, 3), dtype=np.float32)
        up[::2, ::2, :] = nxt
        exp = cv2.sepFilter2D(up, cv2.CV_32F, _K1D_X2, _K1D_X2,
                              borderType=cv2.BORDER_REFLECT)
        current -= exp[:h, :w, :]
        bands.append(current)
        current = nxt
    bands.append(current)
    return bands


def _is_pure_translation(warp: np.ndarray, tol: float = 1e-4) -> bool:
    return bool(np.allclose(warp[:, :2], np.eye(2), atol=tol))


def _load_raw_u8(
    path: Path | np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Load an image as uint8 RGB, resizing to size if needed.

    Used only for alignment and sharpness scoring, where 8-bit precision is
    sufficient. 16-bit arrays and files are scaled down to uint8.
    Raises ValueError if cv2 cannot read the file.
    """
    if isinstance(path, np.ndarray):
        if path.dtype == np.uint8:
            img: np.ndarray = path
        elif path.dtype == np.uint16:
            img = (path >> 8).astype(np.uint8)
        else:
            img = np.clip(path / 257.0, 0, 255).astype(np.uint8)
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cv2 could not read image: '{path}'")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img.shape[1] != size[0] or img.shape[0] != size[1]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _load_raw(
    path: Path | np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Load an image as float32 RGB in its native range, resizing if needed.

    16-bit files/arrays are returned in [0, 65535]; 8-bit in [0, 255]. The
    range is not normalised so callers that need to know the scale should
    check the source dtype beforehand via _source_depth.
    Raises ValueError if cv2 cannot read the file.
    """
    if isinstance(path, np.ndarray):
        img = path.astype(np.float32)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise ValueError(f"cv2 could not read image: '{path}'")
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        elif raw.shape[2] == 4:
            raw = raw[:, :, :3]
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        img = raw.astype(np.float32)
    if img.shape[1] != size[0] or img.shape[0] != size[1]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _load_and_warp(
    path: Path | np.ndarray,
    warp: np.ndarray,
    size: tuple[int, int],
    border_mode: int = cv2.BORDER_REFLECT,
) -> np.ndarray:
    """Load an image, resize if needed, and apply a warp. Returns float32 RGB.

    size is (w, h) as expected by cv2.warpAffine.
    """
    img = _load_raw_u8(path, size)
    if not np.array_equal(warp, np.eye(2, 3, dtype=np.float32)):
        flags = cv2.INTER_LINEAR if _is_pure_translation(warp) else cv2.INTER_CUBIC
        img = cv2.warpAffine(img, warp, size, flags=flags, borderMode=border_mode)
    return img.astype(np.float32)


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

    all_corners: list[np.ndarray] = []
    for warp in warps:
        M = warp.astype(np.float64)
        transformed = (M[:, :2] @ corners.T + M[:, 2:]).T
        all_corners.append(transformed)

    all_pts = np.concatenate(all_corners, axis=0)
    canvas_min = all_pts.min(axis=0)
    canvas_max = all_pts.max(axis=0)

    if crop:
        per_img_mins = np.array([c.min(axis=0) for c in all_corners]) - canvas_min
        per_img_maxs = np.array([c.max(axis=0) for c in all_corners]) - canvas_min
        crop_min = per_img_mins.max(axis=0)
        crop_max = per_img_maxs.min(axis=0)
        if np.all(crop_max > crop_min):
            canvas_max = canvas_min + crop_max
            canvas_min = canvas_min + crop_min

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
    canvas_size: tuple[int, int] | None = None,
    no_fill: bool = False,
    workers: int = 3,
    progress: ProgressCallback | None = None,
    interrupt: InterruptCallback | None = None,
) -> np.ndarray:
    """Fuse a stack of images using single-pass unnormalized Laplacian pyramid fusion.

    Accumulates energy-weighted RGB bands and energy sums in one disk read per
    image, then divides at the end. Focus weights are derived from the Lab
    Laplacian pyramid so colour does not influence sharpness scoring, but the
    blended pixel values come from the original RGB data preserving full depth.

    Mathematically identical to normalized fusion:
      Σ(e_k / Σe_k · x_k) = Σ(e_k · x_k) / Σe_k

    The output dtype matches the source depth: uint16 for 16-bit sources,
    uint8 for 8-bit sources. Depth is detected from the first image/array.

    canvas_size overrides the output dimensions (w, h). If None, the size of the
    first image is used (equivalent to --keep-size behaviour).

    no_fill uses BORDER_CONSTANT (black) instead of BORDER_REFLECT for regions
    outside each image's coverage after warping.

    workers controls how many images are processed concurrently. Peak RAM scales
    with workers × ~100 MiB per image plus the fixed fused_lp accumulator.
    Default of 3 workers balances speed and memory for most systems.

    progress is called as progress(fraction, stage, message) after each image is fused
    and at each subsequent stage. fraction runs from 0 to 1 across the full
    stack_images call: ~0–0.7 fusing, 0.8 reconstructing, 0.9 colour correction,
    1.0 complete.

    interrupt is called after each image is fused; if it returns True, Interrupted is raised.
    """
    depth = _source_depth(src_paths[0])
    max_val = 65535.0 if depth == 16 else 255.0
    out_dtype = np.uint16 if depth == 16 else np.uint8

    if isinstance(src_paths[0], np.ndarray):
        h, w = src_paths[0].shape[:2]
    else:
        probe = cv2.imread(str(src_paths[0]), cv2.IMREAD_UNCHANGED)
        if probe is None:
            raise ValueError(f"cv2 could not read image: '{src_paths[0]}'")
        h, w = probe.shape[:2]
        del probe
    cv2_size = canvas_size if canvas_size is not None else (w, h)
    border_mode = cv2.BORDER_CONSTANT if no_fill else cv2.BORDER_REFLECT

    n = len(src_paths)
    n_workers = min(n, workers if workers > 0 else (os.cpu_count() or 4))

    identity_warp = np.eye(2, 3, dtype=np.float32)

    # (energy_sums, weighted_band_sums, unweighted_band_sums, image_count)
    PartialAccum = tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int]

    if depth == 16:
        warp_dtype: type = np.uint16

        def _pixel_lap_pyramid(img: np.ndarray) -> list[np.ndarray]:
            """Laplacian pyramid over float32 RGB in [0, 65535]."""
            bands: list[np.ndarray] = []
            current = img
            for _ in range(levels):
                h_c, w_c = current.shape[:2]
                nh, nw = (h_c + 1) // 2, (w_c + 1) // 2
                smoothed = cv2.sepFilter2D(current, cv2.CV_32F, _K1D, _K1D,
                                           borderType=cv2.BORDER_REFLECT)
                nxt = smoothed[::2, ::2, :].copy()
                up = np.zeros((nh * 2, nw * 2, 3), dtype=np.float32)
                up[::2, ::2, :] = nxt
                exp = cv2.sepFilter2D(up, cv2.CV_32F, _K1D_X2, _K1D_X2,
                                      borderType=cv2.BORDER_REFLECT)
                current = current - exp[:h_c, :w_c, :]
                bands.append(current)
                current = nxt
            bands.append(current)
            return bands

        def _process_batch(batch: list[tuple[Path | np.ndarray, np.ndarray]]) -> PartialAccum:
            """16-bit: blend RGB pyramid bands; derive focus weights from Lab."""
            local_energy_sums: list[np.ndarray | None] = [None] * (levels + 1)
            local_fused: list[np.ndarray | None] = [None] * (levels + 1)
            local_unweighted: list[np.ndarray | None] = [None] * (levels + 1)
            local_wb: np.ndarray | None = None
            local_count = 0

            for path, warp in batch:
                img = _load_raw(path, cv2_size)

                if not np.array_equal(warp, identity_warp):
                    clamped = np.clip(img, 0, max_val).astype(warp_dtype)
                    flags = cv2.INTER_LINEAR if _is_pure_translation(warp) else cv2.INTER_CUBIC
                    img = cv2.warpAffine(clamped, warp, cv2_size, flags=flags,
                                         borderMode=border_mode).astype(np.float32)

                lab = _image_to_lab(img, max_val)
                lab_lap = _lab_lap_pyramid(lab, levels)
                del lab
                pixel_lap = _pixel_lap_pyramid(img)
                del img

                energies: list[np.ndarray] = []
                for i in range(levels):
                    energies.append(region_energy(lab_lap[i][:, :, 0]) ** sharpness)
                lv = lab_lap[-1][:, :, 0]
                energies.append(((region_deviation(lv) + region_entropy(lv)) * 0.5) ** sharpness)

                for i in range(levels + 1):
                    e = energies[i]
                    band = pixel_lap[i]
                    pixel_lap[i] = None  # type: ignore[call-overload]
                    local_energy_sums[i] = e if local_energy_sums[i] is None else local_energy_sums[i] + e
                    local_unweighted[i] = band.copy() if local_unweighted[i] is None else local_unweighted[i] + band
                    if local_wb is None or local_wb.shape != band.shape:
                        local_wb = np.empty_like(band)
                    np.multiply(band, e[:, :, np.newaxis], out=local_wb)
                    if local_fused[i] is None:
                        local_fused[i] = local_wb.copy()
                    else:
                        local_fused[i] += local_wb  # type: ignore[operator]

                local_count += 1

            return local_energy_sums, local_fused, local_unweighted, local_count  # type: ignore[return-value]

        def _reconstruct(fused: list[np.ndarray]) -> np.ndarray:
            image = fused[-1].copy()
            for band in reversed(fused[:-1]):
                cur_shape = band.shape[:2]
                h_i, w_i = image.shape[:2]
                up = np.zeros((h_i * 2, w_i * 2, 3), dtype=np.float32)
                up[::2, ::2, :] = image
                exp = cv2.sepFilter2D(up, cv2.CV_32F, _K1D_X2, _K1D_X2,
                                      borderType=cv2.BORDER_REFLECT)
                image = exp[: cur_shape[0], : cur_shape[1], :] + band  # type: ignore[operator]
            return image

    else:
        def _pixel_lap_pyramid_u8(img: np.ndarray) -> list[np.ndarray]:
            """Laplacian pyramid over float32 RGB in [0, 255]."""
            bands: list[np.ndarray] = []
            current = img
            for _ in range(levels):
                h_c, w_c = current.shape[:2]
                nh, nw = (h_c + 1) // 2, (w_c + 1) // 2
                smoothed = cv2.sepFilter2D(current, cv2.CV_32F, _K1D, _K1D,
                                           borderType=cv2.BORDER_REFLECT)
                nxt = smoothed[::2, ::2, :].copy()
                up = np.zeros((nh * 2, nw * 2, 3), dtype=np.float32)
                up[::2, ::2, :] = nxt
                exp = cv2.sepFilter2D(up, cv2.CV_32F, _K1D_X2, _K1D_X2,
                                      borderType=cv2.BORDER_REFLECT)
                current = current - exp[:h_c, :w_c, :]
                bands.append(current)
                current = nxt
            bands.append(current)
            return bands

        def _process_batch(batch: list[tuple[Path | np.ndarray, np.ndarray]]) -> PartialAccum:  # type: ignore[misc]
            """8-bit: blend RGB pyramid bands; derive focus weights from Lab."""
            local_energy_sums: list[np.ndarray | None] = [None] * (levels + 1)
            local_fused: list[np.ndarray | None] = [None] * (levels + 1)
            local_unweighted: list[np.ndarray | None] = [None] * (levels + 1)
            local_wb: np.ndarray | None = None
            local_count = 0

            for path, warp in batch:
                img = _load_raw_u8(path, cv2_size)

                if not np.array_equal(warp, identity_warp):
                    flags = cv2.INTER_LINEAR if _is_pure_translation(warp) else cv2.INTER_CUBIC
                    img = cv2.warpAffine(img, warp, cv2_size, flags=flags,
                                         borderMode=border_mode)

                img_f = img.astype(np.float32)
                del img
                lab = cv2.cvtColor(img_f.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2Lab).astype(np.float32)
                lab_lap = _lab_lap_pyramid(lab, levels)
                del lab
                pixel_lap = _pixel_lap_pyramid_u8(img_f)
                del img_f

                energies: list[np.ndarray] = []
                for i in range(levels):
                    energies.append(region_energy(lab_lap[i][:, :, 0]) ** sharpness)
                lv = lab_lap[-1][:, :, 0]
                energies.append(((region_deviation(lv) + region_entropy(lv)) * 0.5) ** sharpness)

                for i in range(levels + 1):
                    e = energies[i]
                    band = pixel_lap[i]
                    pixel_lap[i] = None  # type: ignore[call-overload]
                    local_energy_sums[i] = e if local_energy_sums[i] is None else local_energy_sums[i] + e
                    local_unweighted[i] = band.copy() if local_unweighted[i] is None else local_unweighted[i] + band
                    if local_wb is None or local_wb.shape != band.shape:
                        local_wb = np.empty_like(band)
                    np.multiply(band, e[:, :, np.newaxis], out=local_wb)
                    if local_fused[i] is None:
                        local_fused[i] = local_wb.copy()
                    else:
                        local_fused[i] += local_wb  # type: ignore[operator]

                local_count += 1

            return local_energy_sums, local_fused, local_unweighted, local_count  # type: ignore[return-value]

        def _reconstruct(fused: list[np.ndarray]) -> np.ndarray:  # type: ignore[misc]
            image = fused[-1].copy()
            for band in reversed(fused[:-1]):
                cur_shape = band.shape[:2]
                h_i, w_i = image.shape[:2]
                up = np.zeros((h_i * 2, w_i * 2, 3), dtype=np.float32)
                up[::2, ::2, :] = image
                exp = cv2.sepFilter2D(up, cv2.CV_32F, _K1D_X2, _K1D_X2,
                                      borderType=cv2.BORDER_REFLECT)
                image = exp[: cur_shape[0], : cur_shape[1], :] + band  # type: ignore[operator]
            return image

    if progress is not None:
        progress(0.0, "stacking", f"Fusing ({n_workers} workers)...")
    energy_sums: list[np.ndarray | None] = [None] * (levels + 1)
    fused: list[np.ndarray | None] = [None] * (levels + 1)
    unweighted: list[np.ndarray | None] = [None] * (levels + 1)
    total_count = 0

    items = list(zip(src_paths, warps))

    # Distribute images across workers as evenly as possible so each worker
    # accumulates its own partial fused/energy_sums. The main thread then
    # merges n_workers partial results instead of accumulating all n images serially.
    chunk_size = max(1, (len(items) + n_workers - 1) // n_workers)
    batches = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    done_count = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            partial_energy, partial_fused, partial_unweighted, partial_count = future.result()

            for i in range(levels + 1):
                energy_sums[i] = (partial_energy[i] if energy_sums[i] is None
                                  else energy_sums[i] + partial_energy[i])  # type: ignore[operator]
                fused[i] = (partial_fused[i] if fused[i] is None
                            else fused[i] + partial_fused[i])  # type: ignore[operator]
                unweighted[i] = (partial_unweighted[i] if unweighted[i] is None
                                 else unweighted[i] + partial_unweighted[i])  # type: ignore[operator]

            total_count += partial_count
            done_count += len(futures[future])
            if progress is not None:
                progress(done_count / n * 0.7, "stacking", f"Fused image {done_count}/{n}")
            if interrupt is not None and interrupt():
                raise Interrupted

    for i in range(levels + 1):
        e = energy_sums[i]  # type: ignore[index]
        avg = unweighted[i] / total_count  # type: ignore[operator,index]
        # Blend between unweighted average and energy-weighted result using a
        # confidence weight derived from total energy. At featureless regions
        # (flat white/black backgrounds) all images contribute near-zero sharpness
        # energy, making the weighted numerator and denominator both near-zero but
        # independently, so their ratio is unreliable noise. At weakly-textured
        # regions the sharpness exponent (default 4.0) compresses small energy
        # differences toward zero, making the weighting noise-dominated there too.
        #
        # The confidence alpha ramps from 0 (pure unweighted average) to 1 (pure
        # energy-weighted) as total energy crosses a transition band. The midpoint
        # 1e-8 corresponds to Laplacian magnitude ~0.1 on a 0-255 scale — just
        # below any visually meaningful texture — and the width spans 4 decades so
        # the transition is smooth and never produces a hard boundary artifact.
        alpha = np.clip((np.log10(e + 1e-40) + 20.0) / 12.0, 0.0, 1.0)[:, :, np.newaxis]  # type: ignore[operator]
        weighted = fused[i] / (e[:, :, np.newaxis] + 1e-40)  # type: ignore[operator,index]
        fused[i] = alpha * weighted + (1.0 - alpha) * avg  # type: ignore[index]

    if progress is not None:
        progress(0.8, "stacking", "Reconstructing pyramid...")
    image = _reconstruct(fused)  # type: ignore[arg-type]
    result = np.clip(image, 0, max_val).astype(out_dtype)

    if progress is not None:
        progress(1.0, "stacking", "Stacking complete")

    return result


def compute_levels(shape: tuple[int, int], max_levels: int = 6) -> int:
    min_dim = min(shape[0], shape[1])
    levels = 0
    size = min_dim
    while size > 16 and levels < max_levels:
        size //= 2
        levels += 1
    return levels


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
    src_paths: list[Path | np.ndarray],
    adjusted_warps: list[np.ndarray],
    slab_size: int,
    overlap: int,
    levels: int,
    sharpness: float,
    canvas_size: tuple[int, int],
    no_fill: bool,
    workers: int,
    only_slab: bool,
    recursive: bool,
    on_slab: SlabCallback | None = None,
    progress: ProgressCallback | None = None,
    interrupt: InterruptCallback | None = None,
) -> list[np.ndarray] | np.ndarray:
    """Stack images using slabbing, optionally recursive.

    Layer 1 splits src_paths into overlapping sub-stacks and stacks each one.
    With recursive=True, if the layer-1 results still outnumber slab_size the
    same split is applied to those results as layer 2, and so on, until the
    remaining count fits within a single stack pass.
    With recursive=False, the layer-1 results are fused in one final stack
    regardless of how many there are.

    With only_slab=True recursion stops after layer 1 and the slab arrays are
    returned directly without further fusing.

    on_slab is called as on_slab(label, array) for each completed slab when the
    caller wants to persist intermediate results. The label follows the pattern
    slab_<layer>_<NNN>. Saving is entirely the caller's responsibility.

    progress is called as progress(fraction, stage, message) after each completed slab,
    where fraction advances by 1/total_slabs per slab.

    interrupt is called after each completed slab; if it returns True, Interrupted is raised.
    """
    identity = np.eye(2, 3, dtype=np.float32)

    current_items: list[Path | np.ndarray] = list(src_paths)
    current_warps: list[np.ndarray] = list(adjusted_warps)
    current_canvas: tuple[int, int] = canvas_size

    layer = 1
    while True:
        n = len(current_items)
        slabs = _compute_slabs(n, slab_size, overlap)
        total_slabs = len(slabs)

        if progress is not None:
            progress(0.0, "slabbing", f"Layer {layer}: {total_slabs} slabs from {n} images")

        slab_arrays: list[np.ndarray] = []

        for idx, (start, end) in enumerate(slabs):
            label = f"slab_{layer}_{idx + 1:03d}"
            if progress is not None:
                progress(idx / total_slabs, "slabbing", f"Stacking {label} (images {start + 1}–{end})")
            result = stack_images(
                current_items[start:end], current_warps[start:end],
                levels, sharpness, current_canvas, no_fill, workers,
            )

            if on_slab is not None:
                on_slab(label, result)

            slab_arrays.append(result)

            if interrupt is not None and interrupt():
                raise Interrupted

        if progress is not None:
            progress(1.0, "slabbing", f"Layer {layer} complete")

        if only_slab:
            return slab_arrays

        if len(slab_arrays) == 1:
            return slab_arrays[0]

        if len(slab_arrays) <= slab_size or not recursive:
            final_warps = [identity.copy() for _ in slab_arrays]
            return stack_images(
                slab_arrays, final_warps, levels, sharpness,
                current_canvas, no_fill, workers, progress, interrupt,
            )

        current_items = list(slab_arrays)
        current_warps = [identity.copy() for _ in slab_arrays]
        current_canvas = canvas_size
        layer += 1


@dataclass
class FocusStackConfig:
    images: Path | list[Path] | list[np.ndarray]
    no_align: bool = False
    keep_size: bool = False
    crop: bool = False
    no_fill: bool = False
    reference: int = -1
    cull: float | None = None
    global_align: bool = False
    no_rotation: bool = False
    no_scale: bool = False
    no_shear: bool = False
    no_translation: bool = False
    full_res: bool = False
    min_shift: float = 5.0
    levels: int = 0
    sharpness: float = 4.0
    workers: int = 3
    slab: tuple[int, int] | None = None
    only_slab: bool = False
    recursive_slab: bool = False
    on_slab: SlabCallback | None = None
    interrupt: InterruptCallback | None = None


def run(
    cfg: FocusStackConfig,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Run the full focus stacking pipeline and return the result.

    Returns a RunResult with:
      - image: uint8 or uint16 RGB ndarray matching the source bit depth, or None when only_slab is True.
      - slabs: list of intermediate slab arrays when only_slab is True, else None.

    progress is called throughout as progress(fraction, stage, message) where fraction
    is in [0, 1] across the whole run and stage is a Stage literal indicating which
    part of the pipeline is active. The stages and their approximate weight:
      loading ~5%   alignment ~25%   stacking ~65%

    Raises Interrupted if cfg.interrupt returns True at any checkpoint.
    Raises ValueError for invalid configuration (bad reference index, slab params).
    """
    def _stage(fraction: float, stage: Stage, message: str) -> None:
        if progress is not None:
            progress(fraction, stage, message)

    _stage(0.0, "loading", "Loading images...")
    src_images, reference_size = resolve_images(cfg.images)
    n_images = len(src_images)

    if cfg.cull is not None:
        _stage(0.02, "culling", "Culling unfocused images...")

        def _cull_progress(fraction: float, stage: Stage, message: str) -> None:
            if progress is not None:
                progress(0.02 + fraction * 0.03, stage, message)

        cull_result = cull_unfocused_images(
            src_images,
            reference_size,
            threshold=cfg.cull,
            progress=_cull_progress,
        )
        for i, entry in enumerate(cull_result.entries):
            status = "keep" if entry.kept else "CULL"
            label = entry.path.name if isinstance(entry.path, Path) else f"image_{i}"
            _stage(0.02, "culling", f"  [{status}] {label}  (score={entry.score:.4g}, cutoff={cull_result.cutoff:.4g})")
        _stage(0.05, "culling", f"Culled {cull_result.n_culled}/{len(cull_result.entries)} image(s); {len(cull_result.kept)} frame(s) remaining.")
        src_images = cull_result.kept
        n_images = len(src_images)

    if cfg.reference >= 0:
        reference = cfg.reference
        if not (0 <= reference < n_images):
            raise ValueError(f"--reference {cfg.reference} is out of range (0\u2013{n_images - 1}).")
    else:
        reference = n_images // 2

    ref_w, ref_h = reference_size
    levels = cfg.levels if cfg.levels > 0 else compute_levels((ref_h, ref_w))

    identity = np.eye(2, 3, dtype=np.float32)
    if not cfg.no_align:
        strategy = "global" if cfg.global_align else "neighbour-chained"
        _stage(0.08, "aligning", f"Aligning ({strategy}, reference image {reference + 1})...")

        def _align_progress(fraction: float, stage: Stage, message: str) -> None:
            if progress is not None:
                progress(0.08 + fraction * 0.22, stage, message)

        warps = align_images(
            src_images,
            reference_size,
            reference_idx=reference,
            global_align=cfg.global_align,
            no_rotation=cfg.no_rotation,
            no_scale=cfg.no_scale,
            no_shear=cfg.no_shear,
            no_translation=cfg.no_translation,
            full_res=cfg.full_res,
            min_shift=cfg.min_shift,
            workers=cfg.workers,
            progress=_align_progress,
            interrupt=cfg.interrupt,
        )
    else:
        _stage(0.08, "aligning", "Skipping alignment.")
        warps = [identity.copy() for _ in src_images]

    use_slabs = cfg.slab is not None
    only_slab = cfg.only_slab

    canvas_size, adjusted_warps = compute_canvas(warps, reference_size, keep_size=cfg.keep_size, crop=cfg.crop)

    def _stack_progress(fraction: float, stage: Stage, message: str) -> None:
        if progress is not None:
            progress(0.30 + fraction * 0.65, stage, message)

    if use_slabs:
        slab_size, overlap = cfg.slab  # type: ignore[misc]
        if slab_size < 2:
            raise ValueError("Slab SIZE must be at least 2.")
        if overlap < 0 or overlap >= slab_size:
            raise ValueError(f"Slab OVERLAP must be >= 0 and < SIZE ({slab_size}).")

        _stage(0.30, "slabbing", "Slabbing...")
        slab_result = slab_images(
            src_paths=src_images,
            adjusted_warps=adjusted_warps,
            slab_size=slab_size,
            overlap=overlap,
            levels=levels,
            sharpness=cfg.sharpness,
            canvas_size=canvas_size,
            no_fill=cfg.no_fill,
            workers=cfg.workers,
            only_slab=only_slab,
            recursive=cfg.recursive_slab and not only_slab,
            on_slab=cfg.on_slab,
            progress=_stack_progress,
            interrupt=cfg.interrupt,
        )
        _stage(1.0, "complete", "Complete")

        if only_slab:
            return RunResult(image=None, slabs=slab_result)  # type: ignore[arg-type]

        return RunResult(image=slab_result, slabs=None)  # type: ignore[arg-type]

    _stage(0.30, "stacking", "Stacking...")
    result = stack_images(
        src_images, adjusted_warps, levels, cfg.sharpness,
        canvas_size, cfg.no_fill, cfg.workers, _stack_progress, cfg.interrupt,
    )

    _stage(1.0, "complete", "Complete")
    return RunResult(image=result, slabs=None)