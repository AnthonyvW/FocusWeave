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

The alternative considered was the `opencv` crate, which binds the C++ library
from Rust. It was rejected because it undoes what the rewrite is for: it needs
OpenCV headers and libs plus libclang at build time on every platform, and it
links dynamically, so the single-file binary becomes a binary plus a set of
shared libraries. For the Python side it is worse — a self-contained wheel
would have to bundle those libraries, so `pip install focusweave` pulls in
OpenCV again, only now as a private second copy alongside whatever `cv2` the
caller already has. At that point the Rust layer is replacing orchestration
code that was never the bottleneck. It remains a reasonable choice for anyone
who wants bit-identical parity with OpenCV and does not care about
distribution; see RUNNING.md for where the remaining performance gap sits.

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

Those two changes took fusion from slower than the OpenCV build to faster than
it. What remains is `warp_affine`, which the ECC solver calls four times per
iteration and which is still several times slower per core than OpenCV's. That
one needs real SIMD: OpenCV dispatches to AVX2 at runtime and keeps a whole
kernel in vector registers, and neither `-C target-cpu=native` nor reshaping
the loops gets close on its own.

One pitfall worth recording: replacing a division by a reciprocal multiply in
the fusion blend produced black patches. The denominator there can be
denormal, and its reciprocal overflows to infinity where the division stays
finite; `0 * inf` is NaN, and NaN casts to zero. The division is deliberate.
