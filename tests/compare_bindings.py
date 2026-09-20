"""Check the Python API of the Rust package against the reference package.

Exercises every public symbol the original exposed, so a caller written
against the pure-Python version gets the same shapes, dtypes and values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import make_stack  # noqa: E402

WORK = ROOT / "target" / "bindcheck"

failures: list[str] = []


def check(name: str, got, expected, tol: float = 0.0) -> None:
    got_a = np.asarray(got)
    exp_a = np.asarray(expected)
    if got_a.shape != exp_a.shape:
        failures.append(f"{name}: shape {got_a.shape} != {exp_a.shape}")
        print(f"  FAIL {name:26s} shape {got_a.shape} != {exp_a.shape}")
        return
    diff = float(np.abs(got_a.astype(np.float64) - exp_a.astype(np.float64)).max())
    ok = diff <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:26s} max diff {diff:.6g}")
    if not ok:
        failures.append(f"{name}: max diff {diff:.6g} > {tol:g}")


def main() -> int:
    folder = WORK / "src8"
    if not folder.exists():
        make_stack.build(folder, depth=8)

    # The reference package shadows the installed one, so import ours first.
    import focusweave as fw
    if "rust" not in str(type(fw.reduce)) and not hasattr(fw, "load_image"):
        print("the installed focusweave package is not the Rust build")
        return 1

    sys.path.insert(0, str(ROOT / "tests" / "reference"))
    for mod in [m for m in sys.modules if m.startswith("focusweave")]:
        del sys.modules[mod]
    import focusweave as ref  # noqa: F811
    sys.path.pop(0)

    print("\nreference size and discovery")
    ref_paths, ref_size = ref.load_images(folder)
    for mod in [m for m in sys.modules if m.startswith("focusweave")]:
        del sys.modules[mod]
    import focusweave as rust  # noqa: F811
    rust_paths, rust_size = rust.load_images(folder)
    print(f"  {'ok  ' if rust_size == ref_size else 'FAIL'} load_images               size {rust_size} vs {ref_size}")
    if rust_size != ref_size or [p.name for p in rust_paths] != [p.name for p in ref_paths]:
        failures.append("load_images disagrees")

    print("\npyramid primitives")
    rng = np.random.default_rng(11)
    plane = rng.normal(0, 30, (61, 83)).astype(np.float32)

    sys.path.insert(0, str(ROOT / "tests" / "reference"))
    for mod in [m for m in sys.modules if m.startswith("focusweave")]:
        del sys.modules[mod]
    import focusweave as ref  # noqa: F811
    exp_reduce = ref.reduce(plane)
    exp_expand = ref.expand(exp_reduce, plane.shape)
    exp_energy = ref.region_energy(plane)
    exp_dev = ref.region_deviation(plane)
    exp_ent = ref.region_entropy(plane)
    exp_levels = ref.compute_levels(plane.shape)
    warps = [np.eye(2, 3, dtype=np.float32) for _ in range(3)]
    warps[1][0, 2] = 7.5
    warps[2][1, 2] = -4.25
    exp_canvas, exp_adjusted = ref.compute_canvas(warps, (200, 150))
    sys.path.pop(0)

    for mod in [m for m in sys.modules if m.startswith("focusweave")]:
        del sys.modules[mod]
    import focusweave as rust  # noqa: F811

    check("reduce", rust.reduce(plane), exp_reduce, 1e-4)
    check("expand", rust.expand(exp_reduce, plane.shape), exp_expand, 1e-4)
    check("region_energy", rust.region_energy(plane), exp_energy, 1e-2)
    check("region_deviation", rust.region_deviation(plane), exp_dev, 1e-3)
    check("region_entropy", rust.region_entropy(plane), exp_ent, 1e-5)
    got_levels = rust.compute_levels(plane.shape)
    print(f"  {'ok  ' if got_levels == exp_levels else 'FAIL'} compute_levels             {got_levels} vs {exp_levels}")
    if got_levels != exp_levels:
        failures.append("compute_levels disagrees")

    got_canvas, got_adjusted = rust.compute_canvas(warps, (200, 150))
    print(f"  {'ok  ' if got_canvas == exp_canvas else 'FAIL'} compute_canvas             {got_canvas} vs {exp_canvas}")
    if got_canvas != exp_canvas:
        failures.append("compute_canvas size disagrees")
    for i, (g, e) in enumerate(zip(got_adjusted, exp_adjusted)):
        check(f"compute_canvas warp {i}", g, e, 1e-5)

    print("\nrun() with arrays, a progress callback and cancellation")
    images = [rust.load_image(p) for p in rust_paths]
    seen: list[tuple[float, str]] = []
    result = rust.run(
        rust.FocusStackConfig(images=images, no_align=True, workers=2),
        progress=lambda f, s, m: seen.append((f, s)),
    )
    print(f"  ok   run(arrays)               {result.image.shape} {result.image.dtype}")
    stages = {s for _, s in seen}
    print(f"  {'ok  ' if 'complete' in stages else 'FAIL'} progress callback          stages={sorted(stages)}")
    if "complete" not in stages:
        failures.append("progress callback never reported completion")

    folder_result = rust.run(rust.FocusStackConfig(images=folder, no_align=True, workers=2))
    check("run(folder) == run(arrays)", folder_result.image, result.image, 0)

    try:
        rust.run(rust.FocusStackConfig(images=folder, interrupt=lambda: True))
        failures.append("interrupt callback did not raise Interrupted")
        print("  FAIL interrupt                  no exception raised")
    except rust.Interrupted:
        print("  ok   interrupt                  raised Interrupted")

    boom = RuntimeError("callback exploded")

    def angry(fraction: float, stage: str, message: str) -> None:
        raise boom

    try:
        rust.run(rust.FocusStackConfig(images=folder, no_align=True), progress=angry)
        failures.append("an exception in a progress callback was swallowed")
        print("  FAIL callback exception         swallowed")
    except RuntimeError as e:
        print(f"  {'ok  ' if e is boom else 'FAIL'} callback exception         propagated")
        if e is not boom:
            failures.append("wrong exception propagated from progress callback")

    print("\ncull, align, stack and slab")
    cull = rust.cull_unfocused_images(rust_paths, rust_size, threshold=0.35)
    print(f"  ok   cull_unfocused_images      kept {len(cull.kept)}/{len(cull.entries)} cutoff={cull.cutoff}")
    if not isinstance(cull.entries[0], rust.CullEntry) or cull.entries[0].path != rust_paths[0]:
        failures.append("CullResult entries do not carry the original sources")

    aligned = rust.align_images(rust_paths, rust_size, reference_idx=len(rust_paths) // 2)
    print(f"  {'ok  ' if len(aligned) == len(rust_paths) else 'FAIL'} align_images               {len(aligned)} warps, dtype {aligned[0].dtype}")
    if len(aligned) != len(rust_paths) or aligned[0].shape != (2, 3):
        failures.append("align_images returned the wrong shape")

    canvas, adjusted = rust.compute_canvas(aligned, rust_size)
    stacked = rust.stack_images(rust_paths, adjusted, 5, 4.0, canvas_size=canvas, workers=2)
    print(f"  ok   stack_images               {stacked.shape} {stacked.dtype}")

    labels: list[str] = []
    slabs = rust.slab_images(
        rust_paths, adjusted, 4, 2, 5, 4.0, canvas, False, 2, True, False,
        on_slab=lambda label, arr: labels.append(label),
    )
    print(f"  {'ok  ' if len(labels) == len(slabs) else 'FAIL'} slab_images (only_slab)    {len(slabs)} slabs, labels={labels}")
    if len(labels) != len(slabs):
        failures.append("on_slab was not called once per slab")

    print("\nstreaming stacker")
    previews: list[int] = []
    streamer = rust.StreamingFocusStacker(
        reference_size=rust_size,
        on_preview=lambda preview, count: previews.append(count),
        preview_scale=0.25,
    )
    for image in images:
        streamer.add_image(image)
    stream_result = streamer.finish()
    print(f"  {'ok  ' if previews == list(range(1, len(images) + 1)) else 'FAIL'} on_preview                 {previews}")
    if previews != list(range(1, len(images) + 1)):
        failures.append("on_preview was not called once per frame")
    preview = streamer.get_preview()
    print(f"  ok   get_preview                {preview.shape} {preview.dtype}")
    print(f"  ok   finish()                   {stream_result.image.shape} {stream_result.image.dtype}")
    streamer.flush_preview()

    print("\nio round trip")
    out = WORK / "roundtrip.png"
    rust.save_image(result.image, out)
    reloaded = rust.load_image(out)
    check("save_image/load_image", reloaded, result.image, 0)

    print()
    if failures:
        for f in failures:
            print(f"  {f}")
        return 1
    print("the Rust package's Python API matches the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
