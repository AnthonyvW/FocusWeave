from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from focus_stack import (
    ProgressCallback,
    RunResult,
    _K1D,
    _K1D_X2,
    _chain_affines,
    _constrain_warp,
    _image_to_lab,
    _is_pure_translation,
    _lab_lap_pyramid,
    _load_raw,
    _load_raw_u8,
    _prepare_for_ecc,
    _run_ecc,
    _score_map_to_scalar,
    _source_depth,
    _suppress_dark_chroma,
    _suppress_dark_chroma_rgb,
    _tenengrad_score_map,
    _to_gray_cv,
    compute_canvas,
    compute_levels,
    region_deviation,
    region_energy,
    region_entropy,
    slab_images,
    stack_images,
)

PreviewCallback = Callable[[np.ndarray, int], None]


class StreamingFocusStacker:
    """Accept images one at a time and perform culling and alignment eagerly.

    Images are assumed to arrive in acquisition order (front-to-back or
    back-to-front focus sweep). Each call to add_image immediately scores the
    image for culling and computes the pairwise ECC warp against its predecessor,
    so those costs are paid incrementally with image capture rather than all at
    once before stacking.

    If on_preview is provided, a downscaled focus-stacked preview is emitted
    after each add_image call, computed by incrementally accumulating Laplacian
    pyramid bands and reconstructing at preview_scale resolution. This produces
    a visually correct partial stack of all images seen so far — sharp regions
    fill in progressively as depth coverage grows. The preview is uint8 RGB.

    Call finish() once all images have been added to execute the final stack.

    Example::

        def show(preview: np.ndarray, count: int) -> None:
            cv2.imshow("preview", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        stacker = StreamingFocusStacker(
            reference_size=(w, h),
            cull_threshold=0.6,
            on_preview=show,
            preview_scale=0.25,
        )
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
        reference: int = -1,
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
        on_preview: PreviewCallback | None = None,
        preview_scale: float = 0.25,
    ) -> None:
        self._reference_size = reference_size
        self._reference = reference
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
        self._on_preview = on_preview
        self._preview_scale = max(0.05, min(1.0, preview_scale))

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

        # Incremental preview accumulator state. Warps here are relative to the
        # first image (index 0), accumulated forward as each image arrives.
        # They are not the final chain-relative-to-middle warps used by finish(),
        # but they produce a visually correct partial stack for preview purposes.
        self._preview_levels: int = 0
        self._preview_size: tuple[int, int] = (0, 0)
        self._preview_depth: int = 8
        self._preview_energy_sums: list[np.ndarray] | None = None
        self._preview_fused: list[np.ndarray] | None = None
        self._preview_cumulative_warp: np.ndarray = np.eye(2, 3, dtype=np.float32)



    def add_image(self, image: np.ndarray) -> None:
        """Add the next image in the acquisition sequence.

        Immediately scores the image for culling and aligns it to the previous
        image. Both operations complete before this method returns, so the
        acquisition loop can overlap them with camera I/O on a separate thread.

        If on_preview was configured, a reconstructed partial stack is computed
        and delivered before this method returns.

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

        pairwise_warp: np.ndarray | None = None
        if self._prev_gray is not None:
            warp, converged = _run_ecc(
                self._prev_gray, gray, self._full_res,
                ref_prepared_cache=self._prev_prepared,
                src_prepared_cache=cur_prepared,
            )
            pw = _constrain_warp(
                warp,
                self._no_rotation, self._no_scale,
                self._no_shear, self._no_translation,
            )
            self._pairwise_warps.append((pw, converged))
            if converged:
                pairwise_warp = pw

        self._images.append(image)
        self._grays.append(gray)
        self._prepared.append(cur_prepared)
        self._prev_gray = gray
        self._prev_prepared = cur_prepared

        if self._on_preview is not None:
            count = len(self._images)
            self._run_preview(image, pairwise_warp, count)

    def get_preview(self) -> np.ndarray | None:
        """Return a uint8 RGB preview of the current partial stack, or None if
        fewer than one image has been added or preview support is disabled.

        Requires on_preview to have been set (otherwise the accumulator is not
        maintained).
        """
        if self._on_preview is None:
            return None
        return self._reconstruct_preview()

    def flush_preview(self) -> None:
        """No-op kept for API compatibility.

        Previously drained a background preview executor; preview work now runs
        synchronously inside add_image, so there is nothing to flush.
        """

    def _run_preview(
        self,
        image: np.ndarray,
        pairwise_warp: np.ndarray | None,
        count: int,
    ) -> None:
        """Accumulate one image and fire the preview callback."""
        self._update_preview_accum(image, pairwise_warp)
        preview = self._reconstruct_preview()
        if preview is not None:
            self._on_preview(preview, count)  # type: ignore[misc]

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

        self.flush_preview()

        kept_mask = self._apply_cull(n)

        kept_indices = [i for i, k in enumerate(kept_mask) if k]
        if self._reference >= 0:
            if self._reference >= n:
                raise ValueError(
                    f"reference {self._reference} is out of range (0–{n - 1})."
                )
            ref_idx = self._reference
        else:
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

    # ------------------------------------------------------------------
    # Preview accumulator
    # ------------------------------------------------------------------

    def _update_preview_accum(
        self,
        image: np.ndarray,
        pairwise_warp: np.ndarray | None,
    ) -> None:
        """Add one image to the incremental preview accumulator.

        On the first call, initialises the accumulator dimensions and pyramid
        level count from the image. Subsequent calls use the same geometry.

        pairwise_warp is the constrained ECC warp from the previous image to
        this one (None for the first image or when ECC did not converge). Warps
        are chained forward from image 0 so the accumulator always has a
        consistent coordinate frame.
        """
        ref_w, ref_h = self._reference_size
        pw = int(round(ref_w * self._preview_scale))
        ph = int(round(ref_h * self._preview_scale))
        pw = max(pw, 2)
        ph = max(ph, 2)
        preview_size = (pw, ph)

        depth = _source_depth(image)
        max_val = 65535.0 if depth == 16 else 255.0

        if self._preview_energy_sums is None:
            self._preview_size = preview_size
            self._preview_depth = depth
            levels = self._levels if self._levels > 0 else compute_levels((ph, pw))
            self._preview_levels = levels
            self._preview_energy_sums = [None] * (levels + 1)  # type: ignore[list-item]
            self._preview_fused = [None] * (levels + 1)  # type: ignore[list-item]
            self._preview_cumulative_warp = np.eye(2, 3, dtype=np.float32)
        else:
            levels = self._preview_levels

        # Advance the cumulative warp by the pairwise step (if any).
        if pairwise_warp is not None:
            self._preview_cumulative_warp = _chain_affines(
                self._preview_cumulative_warp, pairwise_warp
            )

        # Scale the translation part of the warp to preview resolution.
        warp = self._preview_cumulative_warp.copy()
        warp[0, 2] *= self._preview_scale
        warp[1, 2] *= self._preview_scale

        border_mode = cv2.BORDER_CONSTANT if self._no_fill else cv2.BORDER_REFLECT
        identity_warp = np.eye(2, 3, dtype=np.float32)

        if depth == 16:
            img = _load_raw(image, preview_size)
            if not np.array_equal(warp, identity_warp):
                clamped = np.clip(img, 0, max_val).astype(np.uint16)
                flags = cv2.INTER_LINEAR if _is_pure_translation(warp) else cv2.INTER_CUBIC
                img = cv2.warpAffine(clamped, warp, preview_size, flags=flags,
                                     borderMode=border_mode).astype(np.float32)
            lab = _image_to_lab(img, max_val)
            lab_lap = _lab_lap_pyramid(lab, levels)

            energies: list[np.ndarray] = []
            for i in range(levels):
                energies.append(region_energy(lab_lap[i][:, :, 0]) ** self._sharpness)
            lv = lab_lap[-1][:, :, 0]
            energies.append(
                ((region_deviation(lv) + region_entropy(lv)) * 0.5) ** self._sharpness
            )

            pixel_lap = self._pixel_lap_pyramid_16(img, levels)

            for i in range(levels + 1):
                e = energies[i]
                band = pixel_lap[i]
                if self._preview_energy_sums[i] is None:
                    self._preview_energy_sums[i] = e
                    self._preview_fused[i] = band * e[:, :, np.newaxis]
                else:
                    self._preview_energy_sums[i] = self._preview_energy_sums[i] + e
                    self._preview_fused[i] = self._preview_fused[i] + band * e[:, :, np.newaxis]
        else:
            img_u8 = _load_raw_u8(image, preview_size)
            if not np.array_equal(warp, identity_warp):
                flags = cv2.INTER_LINEAR if _is_pure_translation(warp) else cv2.INTER_CUBIC
                img_u8 = cv2.warpAffine(img_u8, warp, preview_size, flags=flags,
                                        borderMode=border_mode)
            lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2Lab).astype(np.float32)
            lab_lap = _lab_lap_pyramid(lab, levels)

            energies = []
            for i in range(levels):
                energies.append(region_energy(lab_lap[i][:, :, 0]) ** self._sharpness)
            lv = lab_lap[-1][:, :, 0]
            energies.append(
                ((region_deviation(lv) + region_entropy(lv)) * 0.5) ** self._sharpness
            )

            for i in range(levels + 1):
                e = energies[i]
                band = lab_lap[i]
                if self._preview_energy_sums[i] is None:
                    self._preview_energy_sums[i] = e
                    self._preview_fused[i] = band * e[:, :, np.newaxis]
                else:
                    self._preview_energy_sums[i] = self._preview_energy_sums[i] + e
                    self._preview_fused[i] = self._preview_fused[i] + band * e[:, :, np.newaxis]

    def _reconstruct_preview(self) -> np.ndarray | None:
        """Reconstruct a uint8 RGB preview from the current accumulator state."""
        if self._preview_energy_sums is None or self._preview_energy_sums[0] is None:
            return None

        levels = self._preview_levels
        fused = [
            self._preview_fused[i] / (self._preview_energy_sums[i][:, :, np.newaxis] + 1e-10)
            for i in range(levels + 1)
        ]

        if self._preview_depth == 16:
            image = fused[-1].copy()
            for band in reversed(fused[:-1]):
                cur_shape = band.shape[:2]
                h_i, w_i = image.shape[:2]
                up = np.zeros((h_i * 2, w_i * 2, 3), dtype=np.float32)
                up[::2, ::2, :] = image
                exp = cv2.sepFilter2D(up, cv2.CV_32F, _K1D_X2, _K1D_X2,
                                      borderType=cv2.BORDER_REFLECT)
                image = exp[: cur_shape[0], : cur_shape[1], :] + band
            image = _suppress_dark_chroma_rgb(image, self._dark_threshold, 65535.0)
            return np.clip(image / 257.0, 0, 255).astype(np.uint8)
        else:
            image = fused[-1].copy()
            for band in reversed(fused[:-1]):
                cur_shape = band.shape[:2]
                h_i, w_i = image.shape[:2]
                up = np.zeros((h_i * 2, w_i * 2, 3), dtype=np.float32)
                up[::2, ::2, :] = image
                exp = cv2.sepFilter2D(up, cv2.CV_32F, _K1D_X2, _K1D_X2,
                                      borderType=cv2.BORDER_REFLECT)
                image = exp[: cur_shape[0], : cur_shape[1], :] + band
            fused_lab = _suppress_dark_chroma(image, self._dark_threshold)
            return cv2.cvtColor(
                np.clip(fused_lab, 0, 255).astype(np.uint8),
                cv2.COLOR_Lab2RGB,
            )

    @staticmethod
    def _pixel_lap_pyramid_16(img: np.ndarray, levels: int) -> list[np.ndarray]:
        """Laplacian pyramid over float32 RGB in [0, 65535] range."""
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

    # ------------------------------------------------------------------
    # Internal helpers (unchanged from original)
    # ------------------------------------------------------------------

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