"""Incremental focus stacking for frames that arrive one at a time."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from focusweave import _core
from focusweave.focus_stack import (
    InterruptCallback,
    ProgressCallback,
    RunResult,
    SlabCallback,
)

PreviewCallback = Callable[[np.ndarray, int], None]


class StreamingFocusStacker:
    """Accept images one at a time and perform culling and alignment eagerly.

    Images are assumed to arrive in acquisition order (front-to-back or
    back-to-front focus sweep). Each call to add_image immediately scores the
    image for culling and computes the pairwise warp against its predecessor,
    so those costs are paid incrementally with image capture rather than all at
    once before stacking.

    If on_preview is provided, a downscaled focus-stacked preview is emitted
    after each add_image call. Sharp regions fill in progressively as depth
    coverage grows. The preview is uint8 RGB.

    Call finish() once all images have been added to execute the final stack.

    Example::

        stacker = StreamingFocusStacker(
            reference_size=(w, h),
            cull_threshold=0.6,
            on_preview=lambda preview, n: show(preview),
            preview_scale=0.25,
        )
        for img in camera_feed():
            stacker.add_image(img)
        result = stacker.finish()

    Thread safety: add_image is not thread-safe. Call it from a single thread,
    typically your acquisition loop. finish() must only be called after the
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
        no_fill: bool = False,
        workers: int = 0,
        slab: tuple[int, int] | None = None,
        only_slab: bool = False,
        recursive_slab: bool = False,
        on_slab: SlabCallback | None = None,
        on_preview: PreviewCallback | None = None,
        preview_scale: float = 0.25,
    ) -> None:
        self._on_slab = on_slab
        self._on_preview = on_preview
        self._inner = _core.StreamingFocusStacker(
            tuple(reference_size), reference, cull_threshold, no_rotation,
            no_scale, no_shear, no_translation, full_res, min_shift, levels,
            sharpness, no_fill, workers, slab, only_slab, recursive_slab,
            preview_scale if on_preview is not None else None,
        )

    def add_image(self, image: np.ndarray) -> None:
        """Add the next image in the acquisition sequence.

        Scores the image for culling and aligns it to its predecessor before
        returning, so an acquisition loop can overlap those costs with camera
        I/O on a separate thread.
        """
        preview = self._inner.add_image(image)
        if preview is not None and self._on_preview is not None:
            self._on_preview(preview, len(self._inner))

    def get_preview(self) -> np.ndarray | None:
        """Return a uint8 RGB preview of the current partial stack.

        Returns None when no image has been added yet or previews are disabled.
        """
        return self._inner.get_preview()

    def flush_preview(self) -> None:
        """No-op kept for API compatibility.

        Preview work runs synchronously inside add_image, so there is nothing
        to drain.
        """

    def finish(
        self,
        keep_size: bool = False,
        crop: bool = False,
        progress: ProgressCallback | None = None,
        interrupt: InterruptCallback | None = None,
    ) -> RunResult:
        """Finalize alignment, apply culling, and run the focus stack.

        Resolves the pairwise warp chain into per-image warps relative to the
        middle kept image, applies culling if a threshold was configured, then
        stacks. Returns a RunResult with the same semantics as run().
        """
        image, slabs = self._inner.finish(keep_size, crop, progress, interrupt, self._on_slab)
        return RunResult(image=image, slabs=slabs)

    def __len__(self) -> int:
        return len(self._inner)
