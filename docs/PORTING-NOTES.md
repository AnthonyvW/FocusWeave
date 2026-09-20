Porting notes
=============

Findings from porting the pure-Python/OpenCV implementation to Rust. The
original is preserved under `tests/reference/` and the comparison harness in
`tests/` checks the port against it.

Layout
------

    crates/focusweave-core/   the algorithm and every primitive it needs
    crates/focusweave-cli/    a thin wrapper around core::cli
    crates/focusweave-py/     PyO3 bindings, built as focusweave._core
    python/focusweave/        the Python package: dataclasses and signatures
    tests/reference/          the original implementation, kept for comparison
    tests/                    the comparison harness

The CLI argument parser lives in the core crate so the native binary and the
`focusweave` console script cannot drift apart.

No OpenCV
---------

The port implements the OpenCV routines the pipeline used rather than linking
against them: `sepFilter2D`, `filter2D`, `boxFilter`, `sqrBoxFilter`,
`GaussianBlur`, `Sobel`, `Laplacian`, `dilate`, `resize` with `INTER_AREA`,
`warpAffine`, `cvtColor` for gray and Lab, `createCLAHE`, `phaseCorrelate` and
`findTransformECC`. Image decoding and encoding go through the `image` crate.

That keeps the build free of system dependencies and the binary at about 3 MB,
at the cost of scalar inner loops where OpenCV has SIMD.

Both backends
-------------

The `opencv` crate, which binds the C++ library from Rust, is available behind
the `opencv-backend` feature, so the two can be measured against each other
from one tree. Every primitive is a thin dispatcher in front of a `_native`
implementation; the feature swaps which one is called, and conversion in both
directions borrows rather than copies, so the numbers reflect the kernels
rather than marshalling.

On 4 cores, 10 frames at 2000x1400, best of three:

| case                       | own kernels | OpenCV backend | Python + OpenCV |
| -------------------------- | ----------- | -------------- | --------------- |
| full run                   | 3.68 s      | 2.74 s         | 2.96 s          |
| fusion only                | 1.61 s      | 1.32 s         | 1.63 s          |
| full run, all cores        | 2.99 s      | 2.37 s         | —               |
| peak memory                | 1096 MiB    | 1168 MiB       | 1106 MiB        |
| binary                     | 3.5 MB      | 2.7 MB + 16 MB of shared libraries | — |

Linking OpenCV is worth about 25% over the default build and about 8% over the
Python it replaces. That second number is the interesting one: in that
configuration most of a run is OpenCV either way, and what Rust adds on top —
no GIL, no marshalling — is worth less than the kernels themselves.

The cost is the thing the rewrite was for. The OpenCV build needs headers,
libraries and libclang on every platform and links seven shared libraries
instead of standing alone. For the Python side it is worse: a self-contained
wheel would have to bundle those libraries, so `pip install focusweave` pulls
in OpenCV again, as a private second copy alongside whatever `cv2` the caller
already has.

The default build stays the pure one. The feature is kept because it is the
only honest way to answer "how much is this costing us", and because it is a
reasonable choice for anyone who wants bit-identical parity and does not care
about distribution.

Where the port is not bit-exact
-------------------------------

**Rounding ties.** `RGB2GRAY`, `RGB2Lab` and CLAHE disagree with OpenCV 5 by
one unit in the last place on roughly one pixel in a thousand, always where
the exact value sits on a `.5` boundary. OpenCV 5 resolves those in its SIMD
path differently from its own integer formula, so this is not reproducible
from the documented behaviour and does not affect anything downstream — these
feed sharpness scoring and alignment masks, not output pixels.

**Warping precision.** `warp_affine` evaluates source coordinates in full
floating point. Historically OpenCV quantised them to 1/32 of a pixel via
fixed-point coordinate maps; OpenCV 5 no longer does, and the two agree to
about 0.001 of a grey level on float images.

**Masked ECC.** Without a mask, the port's ECC solver agrees with
`cv2.findTransformECC` to about 1e-7. With one — which is what the pipeline
uses, since it masks to the sharp pixels — the two settle on slightly
different fixed points, within 0.07 px of translation and 2.3e-4 in the linear
block on a real stack. The difference comes from the one-pixel band around the
mask boundary, where the warped gradients are non-zero but the nearest-warped
mask is zero. OpenCV 5's handling of that band is not reproducible from its
documented behaviour; several plausible variants were tested against it and
none matched. Both answers sit equally close to the known ground truth on a
synthetic pair with an exact answer.

This matters in one visible way. The pipeline snaps a cumulative warp to the
identity when its translation is below `min_shift` (5 px by default) and its
linear part is within 1e-3 of the identity. A chain that lands within a hair
of either threshold can fall on opposite sides in the two implementations, and
then a frame is either warped or copied. On the synthetic 8-frame set this
happens with `--reference 0`, which chains seven warps in one direction and
arrives at a translation norm of 5.045 in one implementation and just under
5.0 in the other. `tests/compare_warps.py` checks registration accuracy
directly, before that gate, for exactly this reason.

Defects found in the original
-----------------------------

**The preview divided by the wrong count.** `StreamingFocusStacker`'s preview
accumulator computed

    n_images = sum(1 for e in self._preview_energy_sums if e is not None)

which counts pyramid *levels*, not images — it is `levels + 1` from the first
frame onward. The unweighted average that blends in where no frame carries
sharpness was therefore divided by 6 rather than by the number of frames seen,
making flat regions of the preview progressively too bright. The port tracks
the real frame count. This affects only the on-screen preview, never a stacked
output.

**Sources are stretched to the canvas before being warped.** `stack_images`
loads each frame with `_load_raw(path, cv2_size)`, where `cv2_size` is the
*output canvas*, and that loader resizes anything whose size differs. When the
canvas has been expanded to cover all transformed corners — the default — each
source is therefore scaled up to the canvas and only then warped, rather than
being warped onto a larger canvas at native size.

Every frame gets the same stretch, so the frames stay registered relative to
each other and the stack does not fall apart. The costs are a slight overall
zoom, warp translations effectively applied in stretched coordinates (an error
of about 1% of each shift for a canvas 1% larger), and the sharpness lost to
an extra resampling pass in a tool whose entire purpose is sharpness.

**This is replicated as-is**, so the port's output matches the original. It is
worth fixing — load at native size and warp straight onto the canvas — but
that is a behaviour change, not a port, so it is left as a decision to make
rather than made silently. `--keep-size` avoids it entirely, since the canvas
then equals the source size and no resize happens.

**A docstring that describes the wrong result.** `_constrain_warp` claims "All
four flags together collapse the warp to identity." They do not: with
`no_rotation`, `no_shear` and `no_scale` the result is `diag(sv)` with the
singular values normalised to geometric mean one, which is an axis-aligned
scale of unit determinant and equals the identity only when the two singular
values already agreed. The code is right and the port matches it across all
sixteen flag combinations; only the comment was wrong.

Deliberate additions
--------------------

`focusweave.load_image` and `focusweave.save_image` are new. The original
package documented `cv2` for file I/O in its examples; with OpenCV gone, the
package supplies its own.

Performance work
----------------

Three changes account for most of the speed of the Rust build, and are worth
knowing about before optimising further:

- **The ECC normal equations are accumulated in one pass.** The reference
  materialises the Jacobian as six full-resolution planes and then takes 39
  dot products over them. Since each plane is a warped gradient times a
  coordinate weight, every entry of the Hessian and of the projections can be
  summed in a single traversal. This cut `run_ecc` from 3.1 s to 2.1 s on the
  benchmark set before any parallelism.
- **Separable filtering splits margins from the interior, and accumulates one
  tap at a time over contiguous slices.** Columns whose taps all land inside
  the image skip border-index lookups entirely. Within the interior, tap `j`
  of output element `i` lives at `base[i + j * c]` for any channel count, so
  each tap is a contiguous slice of the source and the accumulation becomes a
  multiply-add the autovectoriser can widen. Writing it as a per-element
  gather instead — the obvious formulation — leaves it scalar. The column pass
  gathers its contributing rows first and sums them in one traversal rather
  than adding each tap into the destination separately, which would read and
  rewrite the whole row once per tap.
- **Dilation uses per-row prefix sums.** The structuring element's rows are
  contiguous runs, so "is any pixel in this window set" is a constant-time
  query rather than a scan of the neighbourhood.

- **The separable filter's two passes are fused.** Writing the horizontally
  filtered image out in full and reading it back adds two trips through main
  memory per filter, and the kernels are memory bound long before they are
  compute bound — coarse parallelism over frames stopped scaling past two
  workers because of it. Each band of output rows now keeps a ring of just
  `ky.len()` filtered rows, small enough to stay in cache, and the
  intermediate never exists in full. This also made the rayon work items
  whole bands rather than single rows, which matters much more the more cores
  the machine has.
- **The fusion workers are rayon tasks, not OS threads.** The filters they
  call parallelise internally, and rayon composes nested parallelism from
  inside its own pool; injecting it from foreign threads instead turns every
  inner `parallel_for` into a cross-thread handshake. `in_place_scope` is what
  allows this while the non-`Send` progress hooks stay on the calling thread.
- **Large scratch buffers go through an allocator that caches them.** The
  pipeline allocates and frees multi-megabyte buffers on every pyramid level.
  glibc services those with `mmap` and returns each one to the kernel
  immediately, so the next is re-faulted page by page on first write. Swapping
  the binary and the extension module to mimalloc took half a second off a
  full run — more than the SIMD work did.

Two findings worth carrying forward. Memory traffic and scheduling shape
mattered far more than instruction set — and both only showed up at scale. A
ten-frame benchmark on four cores hid them entirely; twenty-five frames on
twenty threads made stacking *slower* than the same work on four. Loop shape
came next: reshaping the separable filter so each tap is a contiguous
slice took the 5-tap RGB case from 107 ms to 47 ms per core, while enabling
AVX2 on top of that was worth only a few percent, because the multi-pass form
it replaced was bandwidth bound rather than compute bound. And the remaining
gap to OpenCV is concentrated in `warp_affine`, whose per-pixel gather is the
part that will not vectorise without explicit intrinsics; that is the one
kernel where closing the distance means writing SIMD by hand.

One pitfall worth recording: replacing a division by a reciprocal multiply in
the fusion blend produced black patches. The denominator there can be
denormal, and its reciprocal overflows to infinity where the division stays
finite; `0 * inf` is NaN, and NaN casts to zero. The division is deliberate.
