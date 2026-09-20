Running the Rust FocusWeave
===========================

FocusWeave is now a Rust workspace with Python bindings.

**There are two builds, and the difference matters.** The default one uses the
image-processing kernels implemented in this repository and needs nothing but
a Rust toolchain. The `opencv-backend` one calls the OpenCV C++ library
instead and is the faster of the two — about 25% on a full run. If you are
timing FocusWeave against the old Python implementation, build that one, or
you are comparing this project's hand-written kernels against OpenCV's
hand-written SIMD and the Python will look good. [Pick a build](#2-pick-a-build)
has both commands and what each needs installed.

Contents:

1. [Get a toolchain](#1-get-a-toolchain)
2. [Pick a build](#2-pick-a-build)
3. [Run the CLI](#3-run-the-cli)
4. [Build and use the Python package](#4-build-and-use-the-python-package)
5. [Make yourself a test stack](#5-make-yourself-a-test-stack)
6. [Check it against the old implementation](#6-check-it-against-the-old-implementation)
7. [What to look at first](#7-what-to-look-at-first)
8. [Troubleshooting](#8-troubleshooting)


1. Get a toolchain
------------------

Rust 1.82 or newer:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

On Windows, install [rustup](https://rustup.rs) and the MSVC build tools.

Check it:

    cargo --version

That is all the default build needs. The OpenCV build needs a little more,
below.


2. Pick a build
---------------

Both builds accept the same flags and produce the same picture. They differ
only in whose image-processing kernels run underneath.

| | default | `--features opencv-backend` |
| --- | --- | --- |
| command | `cargo build --release -p focusweave-cli` | `cargo build --release -p focusweave-cli --features opencv-backend` |
| needs | a Rust toolchain | OpenCV 4 development files, plus libclang |
| speed | baseline | ~25% faster overall, ~3x on some kernels |
| result | self-contained 3.5 MB binary | 2.7 MB binary plus ~16 MB of OpenCV shared libraries |
| output | within a few LSB of the original | identical to the original without alignment, within a few LSB with it |

**Build the OpenCV one if speed is what you care about.** The numbers are in
[What to look at first](#7-what-to-look-at-first); the short version is that
the default build's kernels are scalar Rust and OpenCV's are hand-written
AVX2, and no amount of threading closes that on its own.

Both write to the same path, `target/release/focusweave`, so copy each aside
if you want to compare them:

    cargo build --release -p focusweave-cli
    cp target/release/focusweave target/focusweave-native

    cargo build --release -p focusweave-cli --features opencv-backend
    cp target/release/focusweave target/focusweave-opencv

### Prerequisites for the OpenCV build

The `opencv` crate compiles against OpenCV's headers and links its libraries,
and it uses libclang to generate the bindings. Both have to be installed
before `cargo build` will work.

**Linux (Debian, Ubuntu):**

    sudo apt-get install libopencv-dev libclang-dev

That is all — the crate finds everything through `pkg-config`.

**Windows:** install OpenCV and LLVM, then point the crate at them. With
[Chocolatey](https://chocolatey.org):

    choco install llvm opencv

Chocolatey puts OpenCV in `C:\tools\opencv`. In the shell you build from
(PowerShell here; adjust the version number to match what was installed, and
`vc16` to your toolset):

    $env:OPENCV_INCLUDE_PATHS = "C:\tools\opencv\build\include"
    $env:OPENCV_LINK_PATHS    = "C:\tools\opencv\build\x64\vc16\lib"
    $env:OPENCV_LINK_LIBS     = "opencv_world4130"
    $env:LIBCLANG_PATH        = "C:\Program Files\LLVM\bin"
    cargo build --release -p focusweave-cli --features opencv-backend

The resulting `focusweave.exe` needs OpenCV's DLLs at run time, so either add
`C:\tools\opencv\build\x64\vc16\bin` to `PATH` or copy `opencv_world*.dll`
next to the executable. vcpkg works too (`vcpkg install llvm opencv4`, with
`VCPKG_ROOT` set), and the crate then discovers everything by itself.

**macOS:**

    brew install opencv llvm

These Windows and macOS steps follow the `opencv` crate's own setup, which is
the authoritative reference if something does not line up —
[its README](https://github.com/twistedfall/opencv-rust#getting-opencv) lists
every environment variable it reads. They have not been verified on those
platforms from this repository; CI covers the Linux path only.


3. Run the CLI
--------------

The binary lands at `target/release/focusweave` (`focusweave.exe` on Windows).

    ./target/release/focusweave --help
    ./target/release/focusweave path/to/images/

By default it writes `stacked.jpg` into the input folder. Every flag from the
old Python CLI is accepted, with identical names and defaults:

    ./target/release/focusweave path/to/images/ --output result.tiff --workers 0
    ./target/release/focusweave path/to/images/ --cull --crop
    ./target/release/focusweave path/to/images/ --slab 20 5 --output-steps

Three flags to know about while testing:

- `--workers 0` uses every core. The default is still 3, inherited from the
  Python implementation where it was a memory trade-off. On anything with more
  than four cores it is leaving performance on the table — worth 15% here and
  more on a bigger machine.
- `--no-align` skips registration. Useful for isolating the fusion stage when
  comparing output against the old implementation.
- `--timings` prints where the time went, and which kernels the binary was
  built with. Start here if a run is slower than you expect:

      focusweave path/to/images/ --output out.png --timings

      Timings
        build          native kernels
        threads        4 available
        loading          0.00s   0.0%
        aligning         4.07s  31.0%
        stacking         9.01s  68.6%
        total           13.13s


4. Build and use the Python package
-----------------------------------

The package is built with [maturin](https://maturin.rs). In a virtualenv:

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install maturin

Then either install it:

    maturin develop --release

or build a wheel to install elsewhere:

    maturin build --release --out dist
    pip install --find-links dist focusweave

The backend choice from [Pick a build](#2-pick-a-build) applies here too, and
is worth the same on this side — on the benchmark set the default wheel takes
3.8 s and the OpenCV-backed one 2.3 s:

    maturin develop --release --features opencv-backend

Building a *wheel* that way also needs `patchelf` on Linux
(`pip install patchelf`), because maturin has to bundle the shared libraries
into it. That is the distribution cost made concrete: the wheel comes out at
about 13.5 MB instead of 1.5 MB, carrying OpenCV plus libprotobuf, libtbb and
the rest of its dependency chain inside it — a second private copy of OpenCV
alongside whatever `cv2` the environment already has.

The public API is unchanged, so anything written against the old package keeps
working:

```python
from pathlib import Path
from focusweave import FocusStackConfig, run

result = run(FocusStackConfig(images=Path("path/to/images/")))
# result.image is a uint8 (or uint16) RGB numpy array
```

Progress, cancellation and slab callbacks work as before. Exceptions raised
inside a callback propagate out of `run`:

```python
from focusweave import FocusStackConfig, Interrupted, run

cancelled = False

def on_progress(fraction: float, stage: str, message: str) -> None:
    print(f"[{stage}] {fraction * 100:.1f}%  {message}")

try:
    result = run(
        FocusStackConfig(images=Path("images/"), interrupt=lambda: cancelled),
        progress=on_progress,
    )
except Interrupted:
    print("cancelled")
```

Frames already in memory work too, and so does the streaming stacker:

```python
from focusweave import StreamingFocusStacker, load_image

frames = [load_image(p) for p in sorted(folder.iterdir())]
height, width = frames[0].shape[:2]

stacker = StreamingFocusStacker(
    reference_size=(width, height),
    on_preview=lambda preview, count: print(f"preview {count}: {preview.shape}"),
    preview_scale=0.25,
)
for frame in frames:
    stacker.add_image(frame)
result = stacker.finish()
```

`focusweave.load_image` and `focusweave.save_image` are new. The old package
leaned on `cv2` for file I/O in its examples; since OpenCV is no longer a
dependency, the package provides its own. `python -m focusweave.api_example
path/to/images/ --streaming` runs a worked example.

The `focusweave` console script is installed with the wheel and is the same
CLI as the native binary — both call the same Rust argument parser.


5. Make yourself a test stack
-----------------------------

If you do not have a focus stack to hand, `tests/make_stack.py` generates a
synthetic one: a textured scene with a focal plane that sweeps across the
frame, plus a little inter-frame jitter for the aligner to find.

    pip install numpy opencv-python-headless    # generator only
    python tests/make_stack.py /tmp/stack8 8    # 8-bit PNG frames
    python tests/make_stack.py /tmp/stack16 16  # 16-bit TIFF frames

    ./target/release/focusweave /tmp/stack8 --output /tmp/out.png

The result should be sharp edge to edge, where each input frame is sharp only
in one band.


6. Check it against the old implementation
------------------------------------------

The original pure-Python/OpenCV code is preserved verbatim under
`tests/reference/`, and the scripts in `tests/` compare the two directly.

    pip install -r tests/requirements.txt
    cargo build --release -p focusweave-cli
    maturin develop --release
    python tests/run_all.py

Stages can be run individually — `python tests/run_all.py primitives warps` —
and each script also runs standalone. What they cover:

| stage          | what it compares                                             |
| -------------- | ------------------------------------------------------------ |
| `primitives`   | every filter, colour and resampling routine vs OpenCV         |
| `registration` | `phaseCorrelate` and `findTransformECC` vs OpenCV             |
| `warps`        | per-pair alignment on a real stack                            |
| `pipeline`     | end-to-end output across 16 flag combinations                 |
| `bindings`     | the whole Python API surface, callbacks included              |
| `streaming`    | the incremental stacker and its previews                      |

Expected results, which the scripts assert:

- Filters, resampling and warping are exact or differ by one unit in the last
  place on rounding ties.
- Phase correlation agrees to about 3e-6 px.
- Unmasked ECC agrees to about 1e-7; masked ECC, which is what the pipeline
  actually uses, agrees to within 0.07 px of translation.
- With `--no-align`, stacked output differs by at most 2 of 255. With
  alignment, by at most 16 of 255 on synthetic high-frequency texture, mean
  0.3 — that is the sub-pixel registration difference showing up as resampling
  noise, not a change in what the algorithm does.

`cargo test` covers the parts that stand alone from image data: border modes,
the 2x2 SVD, warp constraints, pyramid round-tripping, canvas layout, slab
index arithmetic.

To check the OpenCV build instead, point the pipeline stage at that binary:

    FOCUSWEAVE_BIN=$PWD/target/focusweave-opencv python tests/compare_pipeline.py

It comes out bit-exact against the reference on `--no-align`, which is the
cleanest confirmation that the two backends differ only in their kernels and
that the fusion arithmetic is shared.


7. What to look at first
------------------------

Some things worth poking at, in rough order of how likely they are to matter
to you:

**Does it produce the picture you expect.** Run it on a real stack you already
have a known-good result for and compare. This is the check that matters; the
synthetic tests only prove the port is faithful, not that you like the output.

**Speed.** There are two builds. The default uses the kernels in this
repository; `--features opencv-backend` routes every primitive to the OpenCV
C++ library instead. On this machine (4 cores, 10 frames at 2000x1400, best of
three):

| case                    | own kernels | OpenCV backend | Python + OpenCV |
| ----------------------- | ----------- | -------------- | --------------- |
| full run                | 3.68 s      | 2.74 s         | 2.96 s          |
| fusion only (`--no-align`) | 1.61 s   | 1.32 s         | 1.63 s          |
| full run, `--workers 0` | 2.99 s      | 2.37 s         | —               |
| peak memory             | 1096 MiB    | 1168 MiB       | 1106 MiB        |

Reproduce with `python tests/bench_all.py`, after building both binaries as
that script's docstring describes.

The headline: linking OpenCV buys about 25% over the default build, but only
about 8% over the Python it replaces. Most of a run is OpenCV either way in
that configuration, and what Rust adds on top — no GIL, no marshalling — is
worth less than the kernels themselves. The lever is the kernels, not the
language.

Per core, the kernels compare like this (`RAYON_NUM_THREADS=1 cargo run
--release -p focusweave-core --example bench_primitives`, and again with
`--features opencv-backend`):

| kernel                        | own kernels | OpenCV |
| ----------------------------- | ----------- | ------ |
| `sepFilter2D` 5-tap RGB f32   | 47 ms       | 23 ms  |
| `sepFilter2D` 5-tap gray f32  | 16 ms       | 3 ms   |
| `sqrBoxFilter` 3x3 gray       | 20 ms       | 7 ms   |
| `GaussianBlur` 15 gray        | 22 ms       | 6 ms   |
| `warpAffine` cubic RGB u8     | 231 ms      | 63 ms  |
| `warpAffine` linear gray f32  | 56 ms       | 17 ms  |
| `resize` INTER_AREA           | 42 ms       | 15 ms  |
| RGB to Lab                    | 7 ms        | 17 ms  |

Lab is the one the default build wins, because it only computes the channel
the fusion weights actually use rather than all three. Everything else is the
SIMD gap: OpenCV dispatches hand-written AVX2 at runtime, while these loops
are what the autovectoriser manages on its own. `warpAffine` is the worst of
them and the one that still costs the default build a full run, because the
ECC solver calls it four times per iteration; its per-pixel gather is the part
that does not vectorise without explicit intrinsics.

Two things that mattered more than expected while getting here. Reshaping the
separable filter so each tap is a contiguous slice took the 5-tap RGB case
from 107 ms to 47 ms — loop shape, not instruction set. And switching the
binary to an allocator that caches large blocks took a further 0.5 s off a
full run, because the pipeline allocates and frees multi-megabyte scratch
buffers on every pyramid level and glibc hands each one back to the kernel to
be re-faulted.

**Alignment on a hard stack.** Macro stacks with lots of out-of-focus area are
where the ECC differences would show. Run with `--no-align` and without, on
the same set, and compare.

**16-bit input.** `--output result.tiff` with 16-bit TIFF sources keeps full
depth end to end.

**The Python API**, if you use FocusWeave as a library. Everything that used
to be importable still is, from the same module paths.


8. Troubleshooting
------------------

**It is slower than I expected.** Run it again with `--timings`. That prints
which kernels the binary was built with, how many threads it can see, and how
the time splits between alignment and stacking, which is enough to say where
it is going. Three things to check first:

- **Which build.** The default uses this project's own kernels and is the
  slower of the two; `--features opencv-backend` is the fast one. `--timings`
  says `native kernels` or `opencv kernels` outright.
- **`--workers 0`.** The default of 3 is inherited from the Python
  implementation. If `--timings` reports more than four threads available, the
  default is under-using the machine.
- **How big the set is.** Expect roughly linear scaling in total pixels; 25
  frames at 2592x1944 take about 13 s with the default build and 8 s with the
  OpenCV one on a four-core machine.

**The OpenCV build cannot find OpenCV.** The `opencv` crate reports what it
looked for and where. On Linux it wants `pkg-config --modversion opencv4` to
succeed; on Windows and macOS it usually needs `OPENCV_INCLUDE_PATHS`,
`OPENCV_LINK_PATHS`, `OPENCV_LINK_LIBS` and `LIBCLANG_PATH` set as in
[Prerequisites](#prerequisites-for-the-opencv-build). `cargo build -vv` shows
the crate's own diagnostics.

**The OpenCV build compiles but will not start.** It links OpenCV
dynamically, so the libraries have to be findable at run time: `PATH` on
Windows, `LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS. A package
manager install normally puts them somewhere already searched.

**`cargo build` fails to fetch crates.** The build needs network access the
first time. After that, `cargo build --offline` works.

**`maturin develop` says it cannot find a virtualenv.** Activate one first, or
use `maturin build` and `pip install` the wheel.

**`import focusweave` picks up the wrong package.** The reference
implementation under `tests/reference/` is also called `focusweave`. It is
only importable when that directory is on `sys.path`, which the comparison
scripts do deliberately. Do not add it to `PYTHONPATH` for normal use.

**The comparison scripts fail on `import cv2`.** They need
`pip install -r tests/requirements.txt`; the `focusweave` package itself needs
only numpy.

**WebP output fails.** WebP encoding is lossless-only here, and 16-bit images
are reduced to 8-bit for it, as for JPEG. Prefer PNG or TIFF for 16-bit
output.

**A stack takes much more memory than expected.** Peak memory scales with
`--workers`. Drop to `--workers 1` to roughly halve it at the cost of
throughput.
