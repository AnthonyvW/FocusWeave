Porting notes
=============

Findings from porting the pure-Python/OpenCV implementation to Rust. The
original is preserved under `tests/reference/` and the comparison harness in
`tests/` checks the port against it.

Layout
------

    crates/focusweave-core/   the algorithm, and cv.rs, its OpenCV bindings
    crates/focusweave-cli/    a thin wrapper around core::cli
    crates/focusweave-py/     PyO3 bindings, built as focusweave._core
    python/focusweave/        the Python package: dataclasses and signatures
    tests/reference/          the original implementation, kept for comparison
    tests/                    the comparison harness

The CLI argument parser lives in the core crate so the native binary and the
`focusweave` console script cannot drift apart.

Which parts are OpenCV's
------------------------

The pipeline's image processing calls OpenCV through the `opencv` crate:
`sepFilter2D`, `filter2D`, `boxFilter`, `sqrBoxFilter`, `GaussianBlur`,
`Sobel`, `Laplacian`, `dilate`, `resize` with `INTER_AREA`, `warpAffine`,
`cvtColor` for gray and Lab, and `createCLAHE`. Conversion in both directions
borrows rather than copies — a `Mat` here is a plain row-major buffer, and
`cv::Mat::new_rows_cols_with_data` wraps it in place — so nothing is marshalled
across the boundary.

Only `core` and `imgproc` are linked. Registration stays on this crate's own
ECC solver and phase correlation, because they measure the same speed as
OpenCV's: on 25 frames the alignment stage takes about 2 s either way. Taking
OpenCV's would mean linking `opencv_video` for one function,
`findTransformECC`, and that drags in dnn, calib3d, features2d and flann —
6.7 MB of the 16 MB it would otherwise cost, for nothing. Trimmed to core and
imgproc the dependency is two libraries and 8.2 MB.

Image decoding and encoding go through the `image` crate rather than
`imgcodecs`, which would pull in libjpeg, libpng, libtiff and libwebp on top.

The releases link an OpenCV built by `ci/build_opencv.py` rather than a
packaged one, because every distribution's build carries a different set of
things this project never calls, and each of them cost a day. Ubuntu's
`libopencv_core` links LAPACK, BLAS, gfortran, GL and X11 — thirteen libraries
beyond glibc. Homebrew's links OpenBLAS, which reaches libgcc through
`@rpath`, which no relocation tool could resolve. Chocolatey ships only the
monolithic `opencv_world`, carrying dnn, calib3d and the rest. Built with
`BUILD_LIST=core,imgproc` and every optional dependency off, the result needs
nothing but the C and C++ runtimes, which is what makes an artifact
relocatable at all. It measures the same speed: 6.27 s against 6.25 s on the
benchmark set, with identical output, so IPP and TBB were not buying
anything here.

This was not the first design. The port originally implemented every one of
those routines in Rust and linked nothing, which made for a self-contained
3.5 MB binary — and ran the benchmark set in 10.0 s, exactly level with the
Python it replaced and 1.6x slower than linking OpenCV. Those kernels were
deleted; the history has them. What they cost in maintenance was a second
implementation of every primitive, with its own border handling, its own
rounding, and its own SIMD, to stay within a few LSB of the library the
reference was calling anyway. What they bought was portability. The exchange
rate was not good, and the performance section below records why: the gap was
not the instruction set.

Where the port is not bit-exact
-------------------------------

Filtering, resampling, warping and colour conversion are the same OpenCV calls
the reference makes, so against the same OpenCV they are exact, and stacked
output with `--no-align` lands within 1 of 255. Comparing across OpenCV major
versions — a pip `opencv-python` on one side, the system library on the other —
adds a little drift of its own: cubic `warpAffine` by up to 2 of 255 and wide
`GaussianBlur` by about 5e-5, from changes to OpenCV's fixed-point
interpolation tables. Neither is the port's doing. One thing is.

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
package documented `cv2` for file I/O in its examples. The Rust package links
OpenCV but does not re-export it, and numpy remains its only Python
dependency, so it supplies its own.

Performance work
----------------

What the port is faster at is everything around the kernels, since the kernels
are the same ones the reference called. Four changes account for most of it:

- **The ECC normal equations are accumulated in one pass.** The reference
  materialises the Jacobian as six full-resolution planes and then takes 39
  dot products over them. Since each plane is a warped gradient times a
  coordinate weight, every entry of the Hessian and of the projections can be
  summed in a single traversal. This cut `run_ecc` from 3.1 s to 2.1 s on the
  benchmark set before any parallelism.
- **Frames are fused concurrently, and the worker count is sized to the
  machine.** Much of fusing a frame is per-pixel work that no individual
  kernel parallelises, so coarse parallelism over frames is what scales. The
  fixed default of three inherited from the Python implementation left a
  twenty-thread machine mostly idle; the default is now one worker per core,
  capped by measured free memory, since each costs about 110 MB per megapixel
  of output.
- **The fusion workers are rayon tasks, not OS threads.** OpenCV parallelises
  internally too, and injecting work into a thread pool from foreign threads
  turns every inner parallel region into a cross-thread handshake.
  `in_place_scope` is what allows running the workers on rayon's own pool
  while the non-`Send` progress hooks stay on the calling thread.
- **Large scratch buffers go through an allocator that caches them.** The
  pipeline allocates and frees multi-megabyte buffers on every pyramid level.
  glibc services those with `mmap` and returns each one to the kernel
  immediately, so the next is re-faulted page by page on first write. Swapping
  the binary and the extension module to mimalloc took half a second off a
  full run.

What the deleted Rust kernels taught
------------------------------------

Kept because the conclusions outlived the code, and because they are the
reason the OpenCV dependency is worth its size.

Memory traffic and scheduling shape mattered far more than instruction set,
and both only showed up at scale. A ten-frame benchmark on four cores hid them
entirely; twenty-five frames on twenty threads made stacking *slower* than the
same work on four, because the separable filter materialised its horizontally
filtered intermediate in full and the kernels were memory bound long before
they were compute bound. Fusing the two passes behind a small ring of rows
fixed it.

Loop shape came next. Reshaping the separable filter so that tap `j` of output
element `i` reads `base[i + j * c]` — a contiguous slice per tap, rather than a
per-element gather — took the 5-tap RGB case from 107 ms to 47 ms per core.
Enabling AVX2 on top of that was worth a few percent, because the multi-pass
form it replaced was bandwidth bound rather than compute bound.

And the gap that would not close was `warpAffine`, whose per-pixel gather does
not vectorise without hand-written intrinsics: 231 ms against OpenCV's 63 ms
on cubic RGB. That was the kernel that decided it. Matching OpenCV meant
writing its SIMD again, one primitive at a time, to arrive at the same
arithmetic.

One pitfall worth recording from that work, since the fusion blend it concerns
is still here: replacing a division by a reciprocal multiply produced black
patches. The denominator there can be denormal, and its reciprocal overflows
to infinity where the division stays finite; `0 * inf` is NaN, and NaN casts to
zero. The division is deliberate.
