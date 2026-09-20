"""Focus stacking via Laplacian pyramid fusion.

The implementation lives in the `focusweave._core` extension module; this
module provides the dataclasses and keyword signatures the package has always
exposed, so callers written against the pure-Python version keep working.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from focusweave import _core

IMAGE_EXTENSIONS = frozenset(_core.IMAGE_EXTENSIONS)

Stage = Literal["loading", "culling", "aligning", "stacking", "slabbing", "complete"]

ProgressCallback = Callable[[float, Stage, str], None]
SlabCallback = Callable[[str, np.ndarray], None]
InterruptCallback = Callable[[], bool]

#: Raised when an interrupt callback signals that the run should stop.
Interrupted = _core.Interrupted

ImageSource = Path | np.ndarray


@dataclass
class CullEntry:
    path: ImageSource
    score: float
    kept: bool


@dataclass
class CullResult:
    entries: list[CullEntry]
    cutoff: float
    n_culled: int

    @property
    def kept(self) -> list[ImageSource]:
        return [e.path for e in self.entries if e.kept]


@dataclass
class RunResult:
    image: np.ndarray | None
    slabs: list[np.ndarray] | None


def _as_core_images(images: ImageSource | list[ImageSource]) -> object:
    """Normalise the images argument into something the extension accepts."""
    if isinstance(images, (str, Path)):
        return str(images)
    return [str(i) if isinstance(i, (str, Path)) else i for i in images]


def load_images(folder: Path) -> tuple[list[Path], tuple[int, int]]:
    """Discover image paths in a folder and return them with the reference size.

    Paths are sorted alphabetically. Raises ValueError if fewer than 2 images
    are found. The reference size is read from the first image.
    """
    paths = [Path(p) for p in _core.list_image_files(str(folder))]
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in '{folder}', found {len(paths)}.")
    return paths, _core.probe_size(str(paths[0]))


def resolve_images(
    images: Path | list[Path] | list[np.ndarray],
) -> tuple[list[Path] | list[np.ndarray], tuple[int, int]]:
    """Resolve the images argument into a uniform (items, reference_size) pair.

    Accepts a folder path, a list of paths, or a list of pre-loaded ndarrays.
    Raises ValueError if fewer than 2 images are provided.
    """
    if isinstance(images, (str, Path)):
        return load_images(Path(images))
    if len(images) < 2:
        raise ValueError(f"Need at least 2 images, got {len(images)}.")
    first = images[0]
    if isinstance(first, np.ndarray):
        h, w = first.shape[:2]
        return list(images), (w, h)
    paths = [Path(p) for p in images]
    return paths, _core.probe_size(str(paths[0]))


def cull_unfocused_images(
    images: list[ImageSource],
    reference_size: tuple[int, int],
    threshold: float = 0.05,
    progress: ProgressCallback | None = None,
) -> CullResult:
    """Remove images whose focus score falls below threshold.

    Each image is scored by the high- to low-frequency energy ratio of its
    Tenengrad response map. The two sharpest frames are always retained so the
    stack can proceed even at an aggressive threshold.
    """
    scores, kept, cutoff, n_culled = _core.cull_scores(
        _as_core_images(images), reference_size, threshold, progress, None,
    )
    entries = [CullEntry(path=img, score=s, kept=k) for img, s, k in zip(images, scores, kept)]
    return CullResult(entries=entries, cutoff=cutoff, n_culled=n_culled)


def align_images(
    images: list[ImageSource],
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

    Neighbour-chained by default: ECC runs on consecutive pairs and the warps
    are composed mathematically, so interpolation error never accumulates.
    With global_align every image is aligned directly to the reference.
    """
    return _core.align_images_py(
        _as_core_images(images), reference_size, reference_idx, global_align,
        no_rotation, no_scale, no_shear, no_translation, full_res, min_shift,
        workers, progress, interrupt,
    )


def reduce(image: np.ndarray) -> np.ndarray:
    """Smooth and decimate a single-channel float32 image by two."""
    return _core.reduce(np.ascontiguousarray(image, dtype=np.float32))


def expand(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Upsample a single-channel float32 image by two and crop to target_shape."""
    return _core.expand(np.ascontiguousarray(image, dtype=np.float32), tuple(target_shape))


def region_energy(lp_level: np.ndarray, window: int = 3) -> np.ndarray:
    return _core.region_energy(np.ascontiguousarray(lp_level, dtype=np.float32), window)


def region_deviation(image: np.ndarray, window: int = 3) -> np.ndarray:
    return _core.region_deviation(np.ascontiguousarray(image, dtype=np.float32), window)


def region_entropy(image: np.ndarray, window: int = 8) -> np.ndarray:
    return _core.region_entropy(np.ascontiguousarray(image, dtype=np.float32), window)


def compute_canvas(
    warps: list[np.ndarray],
    src_size: tuple[int, int],
    keep_size: bool = False,
    crop: bool = False,
) -> tuple[tuple[int, int], list[np.ndarray]]:
    """Compute the output canvas size and the warps adjusted to land on it.

    By default the canvas expands to the full extent of all transformed
    corners. With crop it tightens to the intersection of all extents; with
    keep_size it stays at src_size and the warps are returned unchanged.
    """
    prepared = [np.ascontiguousarray(w, dtype=np.float32) for w in warps]
    return _core.compute_canvas_py(prepared, tuple(src_size), keep_size, crop)


def compute_levels(shape: tuple[int, int], max_levels: int = 6) -> int:
    return _core.compute_levels(tuple(shape), max_levels)


def stack_images(
    src_paths: list[ImageSource],
    warps: list[np.ndarray],
    levels: int,
    sharpness: float,
    canvas_size: tuple[int, int] | None = None,
    no_fill: bool = False,
    workers: int = 0,
    progress: ProgressCallback | None = None,
    interrupt: InterruptCallback | None = None,
) -> np.ndarray:
    """Fuse a stack of images using Laplacian pyramid fusion.

    Focus weights come from the Lab lightness pyramid so colour does not
    influence sharpness scoring, while the blended values come from the
    original RGB data. The output dtype matches the source bit depth.
    """
    prepared = [np.ascontiguousarray(w, dtype=np.float32) for w in warps]
    return _core.stack_images_py(
        _as_core_images(src_paths), prepared, levels, sharpness,
        tuple(canvas_size) if canvas_size is not None else None,
        no_fill, workers, progress, interrupt,
    )


def slab_images(
    src_paths: list[ImageSource],
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
    """Stack images in overlapping sub-stacks, then fuse the results.

    Splitting the set reduces how many frames compete in any one fusion pass.
    With recursive the layer's results are slabbed again until they fit a
    single pass; with only_slab the layer-1 arrays are returned directly.
    """
    prepared = [np.ascontiguousarray(w, dtype=np.float32) for w in adjusted_warps]
    return _core.slab_images_py(
        _as_core_images(src_paths), prepared, slab_size, overlap, levels, sharpness,
        tuple(canvas_size), no_fill, workers, only_slab, recursive,
        on_slab, progress, interrupt,
    )


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
    #: Frames fused concurrently; 0 picks one per core, capped to fit in memory.
    workers: int = 0
    slab: tuple[int, int] | None = None
    only_slab: bool = False
    recursive_slab: bool = False
    on_slab: SlabCallback | None = field(default=None, repr=False)
    interrupt: InterruptCallback | None = field(default=None, repr=False)


_CONFIG_KEYS = (
    "no_align", "keep_size", "crop", "no_fill", "reference", "cull",
    "global_align", "no_rotation", "no_scale", "no_shear", "no_translation",
    "full_res", "min_shift", "levels", "sharpness", "workers", "slab",
    "only_slab", "recursive_slab",
)


def run(cfg: FocusStackConfig, progress: ProgressCallback | None = None) -> RunResult:
    """Run the full focus stacking pipeline and return the result.

    progress is called as progress(fraction, stage, message) with fraction in
    [0, 1] across the whole run. Raises Interrupted if cfg.interrupt returns
    True at any checkpoint, and ValueError for invalid configuration.
    """
    payload: dict[str, object] = {"images": _as_core_images(cfg.images)}
    for key in _CONFIG_KEYS:
        payload[key] = getattr(cfg, key)
    image, slabs = _core.run_py(payload, progress, cfg.interrupt, cfg.on_slab)
    return RunResult(image=image, slabs=slabs)
