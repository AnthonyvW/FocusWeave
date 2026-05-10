from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from focusweave.focus_stack import (
    ProgressCallback,
    RunResult,
    _chain_affines,
    _constrain_warp,
    _load_raw_u8,
    _prepare_for_ecc,
    _run_ecc,
    _score_map_to_scalar,
    _tenengrad_score_map,
    _to_gray_cv,
    compute_canvas,
    compute_levels,
    slab_images,
    stack_images,
)


class StreamingFocusStacker:
    """Accept images one at a time and perform culling and alignment eagerly.

    Images are assumed to arrive in acquisition order (front-to-back or
    back-to-front focus sweep). Each call to add_image immediately scores the
    image for culling and computes the pairwise ECC warp against its predecessor,
    so those costs are paid concurrently with image capture rather than all at
    once before stacking.

    Call finish() once all images have been added to execute the final stack.

    Example::

        stacker = StreamingFocusStacker(reference_size=(w, h), cull_threshold=0.6)
        for img in camera_feed():
            stacker.add_image(img)
        result = stacker.finish()

    Thread safety: add_image is not thread-safe. Call it from a single thread
    (typically your acquisition loop). finish() must only be called after the
    last add_image call has returned.
    """

    def __init__(
        self,
        reference_size: tuple[int, int],
        cull_threshold: float | None = None,
        no_rotation: bool = False,
        no_scale: bool = False,
        no_shear: bool = False,
        no_translation: bool = False,
        full_res: bool = False,
        min_shift: float = 5.0,
        levels: int = 0,
        sharpness: float = 4.0,
        dark_threshold: float = 30.0,
        no_fill: bool = False,
        workers: int = 3,
        slab: tuple[int, int] | None = None,
        only_slab: bool = False,
        recursive_slab: bool = False,
        on_slab: Callable[[str, np.ndarray], None] | None = None,
    ) -> None:
        self._reference_size = reference_size
        self._cull_threshold = cull_threshold
        self._no_rotation = no_rotation
        self._no_scale = no_scale
        self._no_shear = no_shear
        self._no_translation = no_translation
        self._full_res = full_res
        self._min_shift = min_shift
        self._levels = levels
        self._sharpness = sharpness
        self._dark_threshold = dark_threshold
        self._no_fill = no_fill
        self._workers = workers
        self._slab = slab
        self._only_slab = only_slab
        self._recursive_slab = recursive_slab
        self._on_slab = on_slab

        ref_w, ref_h = reference_size
        if full_res:
            self._fine_res = 2 ** 31
        else:
            self._fine_res = min(1024, max(ref_w, ref_h))

        self._images: list[np.ndarray] = []
        self._grays: list[np.ndarray] = []
        self._prepared: list[dict[int, tuple[np.ndarray, np.ndarray, float]]] = []
        self._pairwise_warps: list[tuple[np.ndarray, bool]] = []
        self._cull_scores: list[float] = []

        self._prev_gray: np.ndarray | None = None
        self._prev_prepared: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    def add_image(self, image: np.ndarray) -> None:
        """Add the next image in the acquisition sequence.

        Immediately scores the image for culling and aligns it to the previous
        image. Both operations complete before this method returns, so the
        acquisition loop can overlap them with camera I/O on a separate thread.

        image must be a uint8 or uint16 RGB ndarray matching the reference_size
        passed to the constructor.
        """
        ref_w, ref_h = self._reference_size
        if image.shape[1] != ref_w or image.shape[0] != ref_h:
            raise ValueError(
                f"Image size {image.shape[1]}x{image.shape[0]} does not match "
                f"reference size {ref_w}x{ref_h}."
            )

        score_map, _ = _tenengrad_score_map(image, self._reference_size)
        self._cull_scores.append(_score_map_to_scalar(score_map))

        gray = _to_gray_cv(_load_raw_u8(image, self._reference_size))

        cur_prepared: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        cur_prepared[self._fine_res] = _prepare_for_ecc(gray, self._fine_res)

        if self._prev_gray is not None:
            warp, converged = _run_ecc(
                self._prev_gray, gray, self._full_res,
                ref_prepared_cache=self._prev_prepared,
                src_prepared_cache=cur_prepared,
            )
            self._pairwise_warps.append((_constrain_warp(
                warp,
                self._no_rotation, self._no_scale,
                self._no_shear, self._no_translation,
            ), converged))

        self._images.append(image)
        self._grays.append(gray)
        self._prepared.append(cur_prepared)
        self._prev_gray = gray
        self._prev_prepared = cur_prepared

    def finish(
        self,
        keep_size: bool = False,
        crop: bool = False,
        progress: ProgressCallback | None = None,
    ) -> RunResult:
        """Finalize alignment, apply culling, and run the focus stack.

        Resolves the pairwise warp chain into per-image warps relative to the
        middle image (chosen as reference), applies culling if a threshold was
        configured, then calls stack_images or slab_images.

        progress is called as progress(fraction, stage, message) throughout
        stacking, with the same semantics as focus_stack.run().

        Returns a RunResult with the same semantics as focus_stack.run().
        """
        images = self._images
        n = len(images)
        if n < 2:
            raise ValueError(f"Need at least 2 images, got {n}.")

        kept_mask = self._apply_cull(n)

        kept_indices = [i for i, k in enumerate(kept_mask) if k]
        ref_idx = kept_indices[len(kept_indices) // 2]

        identity = np.eye(2, 3, dtype=np.float32)
        abs_warps = self._resolve_chain(n, ref_idx, identity)

        src_images = [images[i] for i in kept_indices]
        src_warps = [abs_warps[i] for i in kept_indices]

        ref_w, ref_h = self._reference_size
        levels = self._levels if self._levels > 0 else compute_levels((ref_h, ref_w))
        canvas_size, adjusted_warps = compute_canvas(
            src_warps, self._reference_size, keep_size=keep_size, crop=crop,
        )

        if self._slab is not None:
            slab_size, overlap = self._slab
            if slab_size < 2:
                raise ValueError("Slab SIZE must be at least 2.")
            if overlap < 0 or overlap >= slab_size:
                raise ValueError(f"Slab OVERLAP must be >= 0 and < SIZE ({slab_size}).")
            slab_result = slab_images(
                src_paths=src_images,
                adjusted_warps=adjusted_warps,
                slab_size=slab_size,
                overlap=overlap,
                levels=levels,
                sharpness=self._sharpness,
                dark_threshold=self._dark_threshold,
                canvas_size=canvas_size,
                no_fill=self._no_fill,
                workers=self._workers,
                only_slab=self._only_slab,
                recursive=self._recursive_slab and not self._only_slab,
                on_slab=self._on_slab,
                progress=progress,
            )
            if self._only_slab:
                return RunResult(image=None, slabs=slab_result)
            return RunResult(image=slab_result, slabs=None)

        result = stack_images(
            src_images, adjusted_warps, levels, self._sharpness,
            self._dark_threshold, canvas_size, self._no_fill, self._workers,
            progress,
        )
        return RunResult(image=result, slabs=None)

    def _apply_cull(self, n: int) -> list[bool]:
        if self._cull_threshold is None:
            return [True] * n
        scores = self._cull_scores
        if max(scores) == 0.0:
            return [True] * n
        kept = [s >= self._cull_threshold for s in scores]
        if sum(kept) < 2:
            ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
            kept = [False] * n
            for idx in ranked[:2]:
                kept[idx] = True
        return kept

    def _resolve_chain(
        self,
        n: int,
        ref_idx: int,
        identity: np.ndarray,
    ) -> list[np.ndarray]:
        abs_warps: list[np.ndarray] = [identity.copy() for _ in range(n)]

        cumulative = identity.copy()
        for i in range(ref_idx + 1, n):
            pw, converged = self._pairwise_warps[i - 1]
            if not converged:
                abs_warps[i] = cumulative.copy()
                continue
            cumulative = _chain_affines(cumulative, pw)
            if self._no_rotation:
                cumulative[:, :2] = np.eye(2)
            abs_warps[i] = self._maybe_identity(cumulative, identity)

        cumulative = identity.copy()
        for i in range(ref_idx - 1, -1, -1):
            # Compute the backward warp directly (gray_{i+1} as ref, gray_i as src),
            # matching align_images' backward pass. Inverting the forward warp is not
            # equivalent because ECC is not symmetric — the focus masks and iterative
            # solver produce different results depending on which image is ref vs src.
            ref_g = self._grays[i + 1]
            src_g = self._grays[i]
            warp, converged = _run_ecc(
                ref_g, src_g, self._full_res,
                ref_prepared_cache=self._prepared[i + 1],
                src_prepared_cache=self._prepared[i],
            )
            warp = _constrain_warp(warp, self._no_rotation, self._no_scale,
                                   self._no_shear, self._no_translation)
            if not converged:
                abs_warps[i] = cumulative.copy()
                continue
            cumulative = _chain_affines(cumulative, warp)
            if self._no_rotation:
                cumulative[:, :2] = np.eye(2)
            abs_warps[i] = self._maybe_identity(cumulative, identity)

        return abs_warps

    def _maybe_identity(self, warp: np.ndarray, identity: np.ndarray) -> np.ndarray:
        translation = float(np.linalg.norm(warp[:, 2]))
        linear_ok = np.allclose(warp[:, :2], np.eye(2), atol=1e-3)
        return identity.copy() if translation < self._min_shift and linear_ok else warp