Running the Rust FocusWeave
===========================

FocusWeave is a Rust workspace with Python bindings. The image-processing
kernels come from OpenCV's C++ library, so **OpenCV 4 and libclang have to be
installed before anything will build.** [Install the prerequisites](#1-install-the-prerequisites)
has the commands for each platform; everything after that is a normal Rust
build.

Contents:

1. [Install the prerequisites](#1-install-the-prerequisites)
2. [Build](#2-build)
3. [Run the CLI](#3-run-the-cli)
4. [Build and use the Python package](#4-build-and-use-the-python-package)
5. [Make yourself a test stack](#5-make-yourself-a-test-stack)
6. [Stack many sets at once](#6-stack-many-sets-at-once)
7. [What to look at first](#7-what-to-look-at-first)
8. [Troubleshooting](#8-troubleshooting)


1. Install the prerequisites
----------------------------

Rust 1.82 or newer:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

On Windows, install [rustup](https://rustup.rs) and the MSVC build tools.

Then OpenCV. The `opencv` crate compiles against OpenCV's headers, links its
libraries, and uses libclang to generate the bindings, so both OpenCV and LLVM
have to be present before `cargo build` will work.

Only `core` and `imgproc` are needed. Registration stays on this project's own
ECC solver, which measures the same speed as OpenCV's; taking OpenCV's would
mean linking `opencv_video`, and that drags in dnn, calib3d, features2d and
flann for one function.

A distribution OpenCV is the quick way to get building, and is what the
per-platform commands below install. It is not what the releases are built
against: `ci/build_opencv.py` compiles core and imgproc with everything
optional turned off, and that build depends on nothing but the C and C++
runtimes. A packaged one drags in a numerics and display stack FocusWeave
never calls — on Ubuntu, LAPACK, BLAS, gfortran, GL and X11, thirteen
libraries and 12.6 MB of them. If you are producing something to hand to
someone else, build it:

    python ci/build_opencv.py --version 4.10.0 --prefix ~/opencv-min \
                              --env-file /dev/stdout

That prints the four environment variables to export, and takes a few minutes.
It needs cmake and a C++ compiler, and nothing else.

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
    $env:OPENCV_LINK_LIBS     = "opencv_world4130"   # or "opencv_core4130,opencv_imgproc4130"
    $env:LIBCLANG_PATH        = "C:\Program Files\LLVM\bin"
    $env:PATH = "C:\tools\opencv\build\x64\vc16\bin;$env:PATH"

`LIBCLANG_PATH` is enough on its own: the crate is built with its
`clang-runtime` feature, so the build script loads `libclang.dll` through that
variable rather than having it resolved by the Windows loader at process start.
Without that feature, `LIBCLANG_PATH` alone gives an opaque
`exit code: 0xc0000135` from `build-script-build` — `STATUS_DLL_NOT_FOUND` —
and the LLVM `bin` directory has to be on `PATH` as well.

The OpenCV `bin` directory does belong on `PATH`, because the resulting
`focusweave.exe` loads those DLLs at run time; copying them next to the
executable works as well. Chocolatey ships the monolithic `opencv_world` build,
which carries every module whether or not it is used; a modular OpenCV build
lets you ship just `opencv_core` and `opencv_imgproc`. vcpkg works too
(`vcpkg install llvm opencv4`, with `VCPKG_ROOT` set), and the crate then
discovers everything by itself.

**macOS:**

    brew install opencv llvm
    export LIBCLANG_PATH="$(brew --prefix llvm)/lib"

Homebrew keeps `llvm` keg-only, so `libclang.dylib` is not on the default
search path and `LIBCLANG_PATH` has to name it.

These Windows and macOS steps follow the `opencv` crate's own setup, which is
the authoritative reference if something does not line up —
[its README](https://github.com/twistedfall/opencv-rust#getting-opencv) lists
every environment variable it reads.


2. Build
--------

    cargo build --release -p focusweave-cli

The binary lands at `target/release/focusweave` (`focusweave.exe` on Windows).
It is about 3.8 MB and links OpenCV's `core` and `imgproc` dynamically, another
8.2 MB on Ubuntu, so it runs on the machine that built it but not on one
without OpenCV installed.

The release archives solve that by carrying the libraries beside the
executable, which is also what `.github/workflows/build.yml` does if you want
to produce a portable copy yourself:

    python ci/bundle_elf.py target/release/focusweave FocusWeave-linux \
        --search ~/opencv-min/lib

That copies every non-system library the binary loads into `FocusWeave-linux/`,
strips them, and rewrites `RPATH` to `$ORIGIN` so the loader looks next to the
executable first. Against the minimal OpenCV that is two libraries and 13.8 MB;
against Ubuntu's it is fifteen and 27 MB. A library it cannot resolve is a hard
error rather than a quietly incomplete archive — pass `--search` for anything
outside the system paths. On macOS and Windows the equivalent is copying the
two `libopencv_*` files into the same folder, which works because the minimal
build has no further dependencies of its own.


3. Run the CLI
--------------

    ./target/release/focusweave --help
    ./target/release/focusweave path/to/images/

By default it writes `stacked.jpg` into the input folder. Every flag from the
old Python CLI is accepted, with identical names and defaults:

    ./target/release/focusweave path/to/images/ --output result.tiff --workers 0
    ./target/release/focusweave path/to/images/ --cull --crop
    ./target/release/focusweave path/to/images/ --slab 20 5 --output-steps

Three flags to know about while testing:

- `--workers N` sets how many frames are fused at once. The default is
  automatic: one per core, capped so the workers' buffers fit in free memory,
  since each costs roughly 110 MB per megapixel of output. Coarse parallelism
  over frames is what scales here — much of fusing a frame is per-pixel work no
  individual kernel parallelises — so this is the knob that matters most. The
  line `Fusing (N workers)` reports what was chosen.
- `--no-align` skips registration. Useful for isolating the fusion stage when
  comparing output between runs.
- `--timings` prints where the time went. Start here if a run is slower than
  you expect:

      focusweave path/to/images/ --output out.png --timings

      Timings
        threads        4 available
        loading          0.00s   0.0%
        aligning         1.93s  30.9%
        stacking         4.27s  68.4%
        total            6.25s


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

Building a *wheel* also needs `patchelf` on Linux (`pip install patchelf`),
because maturin bundles the shared libraries into it. On Ubuntu 24.04 that
wheel comes out at **10.0 MB compressed, 26 MB unpacked**:

| part                                    | unpacked |
| --------------------------------------- | -------- |
| `libopencv_imgproc` + `libopencv_core`   | 8.8 MB   |
| `_core.abi3.so` (FocusWeave itself)      | 4.1 MB   |
| LAPACK, BLAS, libgfortran                | 11.3 MB  |
| TBB, X11, GL stubs                       | 1.5 MB   |

The bottom two rows are worth knowing about: nothing in FocusWeave calls
LAPACK, BLAS or GL, but Ubuntu's `libopencv_core` is linked against them, so
`auditwheel` pulls them in. An OpenCV built with `-DBUILD_LIST=core,imgproc
-DWITH_LAPACK=OFF -DWITH_OPENGL=OFF` would cut the wheel to roughly a third of
this. That is the trade being made: a wheel that carries a private copy of
OpenCV alongside whatever `cv2` the environment already has, in exchange for
the speed in [What to look at first](#7-what-to-look-at-first).

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
leaned on `cv2` for file I/O in its examples; the Rust package links OpenCV but
does not re-export it, and numpy is still its only Python dependency, so it
provides its own. `python -m focusweave.api_example path/to/images/ --streaming`
runs a worked example.

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


6. Stack many sets at once
--------------------------

`--batch` takes a folder of folders and stacks each subfolder as its own set:

    shoot/
      beetle/        frame_01.tiff  frame_02.tiff  ...
      moss/          IMG_4410.jpg   IMG_4411.jpg   ...
      pollen/        ...

    ./target/release/focusweave --batch shoot/
    ./target/release/focusweave --batch shoot/ --output results/ --crop

Each result is named after its subfolder — `beetle`, `moss`, `pollen` — and
written into `shoot/` itself, or into the folder `--output` names, which is
created if it does not exist.

`--batch-format` picks the format. The default, `inherit`, uses the most common
extension among each set's own images, so a set of 16-bit TIFFs comes out as a
16-bit TIFF and a set of JPEGs as a JPEG — in the example above, `beetle.tiff`
and `moss.jpg`. A folder that mixes formats gets whichever is most common, with
a tie going to the one whose file sorts first by name. Give an extension
instead to write every set the same way:

    ./target/release/focusweave --batch shoot/ --batch-format tiff
 With `--batch`, `--output` is
always a folder; a name ending in an image extension is rejected rather than
quietly turned into one. Every other flag applies to each set, and
`--output-steps` puts each set's slabs in `focusweave_slabs/<set>/`.

Subfolders with no images are skipped with a note, and hidden folders are
ignored. A set that fails — too few images, a file that will not decode — is
reported and the batch carries on; the summary at the end names the failures,
and the exit code is 1 if there were any.

Re-running the same command is safe: the output folder and
`focusweave_slabs/` are never treated as sets, even when they sit inside the
batch folder. What is not detected is a *different* earlier output folder —
a `results/` from a previous run with another `--output` is just a subfolder
full of images, and gets stacked like one.

`cargo test` covers the parts that stand alone from image data: the 2x2 SVD,
warp constraints, pyramid round-tripping, canvas layout, slab index arithmetic,
DFT sizing. The comparison against the original Python implementation was
retired once the two stopped being meant to match; the last commit carrying it
is `638c5f0`, and `docs/PORTING-NOTES.md` records what it found.


7. What to look at first
------------------------

Some things worth poking at, in rough order of how likely they are to matter
to you:

**Does it produce the picture you expect.** Run it on a real stack you already
have a known-good result for and compare. This is the check that matters; the
synthetic tests only prove the port is faithful, not that you like the output.

**Speed.** On 4 cores, 25 frames at 2592x1944, best of three:

| build             | total   |
| ----------------- | ------- |
| Rust + OpenCV     | 6.3 s   |
| Python + OpenCV   | 9.6 s   |

The margin is larger on a machine with more cores, because the Python
implementation's fusion loop is per-frame NumPy and the Rust one fuses several
frames at once on rayon: on 20 threads the same set takes 6.0 s against 9.8 s.

That margin is smaller than a rewrite might promise, and the reason is worth
stating plainly: the original was already calling OpenCV for the expensive
parts, so the kernels never changed. What the port bought is the work *around*
the kernels — threading that composes, one pass over the ECC normal equations
instead of six materialised Jacobian planes, and no Python object churn per
pyramid level — plus a single binary with no interpreter.

An earlier revision of this port implemented the kernels in Rust instead of
linking OpenCV. It ran the full set in 10.0 s on the same machine — that is,
level with the Python it replaced, and 1.6x slower than this. Those kernels
were deleted; `git log` has them if the history is interesting. The lesson they
taught is in [PORTING-NOTES.md](docs/PORTING-NOTES.md): memory traffic and
scheduling shape mattered far more than the instruction set, and hand-written
AVX2 is a hard thing to beat from an autovectoriser.

**Alignment on a hard stack.** Macro stacks with lots of out-of-focus area are
where the ECC differences would show. Run with `--no-align` and without, on
the same set, and compare.

**16-bit input.** `--output result.tiff` with 16-bit TIFF sources keeps full
depth end to end.

**The Python API**, if you use FocusWeave as a library. Everything that used
to be importable still is, from the same module paths.


8. Troubleshooting
------------------

**`build-script-build` exits with `0xc0000135` on Windows, or dies with
SIGABRT and `Library not loaded: @rpath/libclang.dylib` on macOS.** Both are
the `opencv` crate's build script failing to find libclang. Set
`LIBCLANG_PATH` to the directory holding `libclang.dll` or `libclang.dylib`,
as in [Install the prerequisites](#1-install-the-prerequisites), and make sure
it is set in the shell you actually build from — a virtualenv activated from a
different shell carries a different environment than the one you built the CLI
in. If you are on a checkout that predates the `clang-runtime` feature, the
LLVM `bin` directory also has to be on `PATH`.

**The build cannot find OpenCV.** The `opencv` crate reports what it looked for
and where. On Linux it wants `pkg-config --modversion opencv4` to succeed; on
Windows and macOS it usually needs `OPENCV_INCLUDE_PATHS`, `OPENCV_LINK_PATHS`,
`OPENCV_LINK_LIBS` and `LIBCLANG_PATH` set. `cargo build -vv` shows the crate's
own diagnostics.

**It compiles but will not start.** OpenCV is linked dynamically, so the
libraries have to be findable at run time: `PATH` on Windows,
`LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS. A package manager
install normally puts them somewhere already searched.

**It is slower than I expected.** Run it again with `--timings`. That prints
how many threads it can see and how the time splits between alignment and
stacking, which is enough to say where it is going. Expect roughly linear
scaling in total pixels; 25 frames at 2592x1944 take about 6 s on a four-core
machine. If `--timings` reports plenty of threads and
`Fusing (N workers)` reports few, the worker count was capped by free memory
rather than by cores.

**`cargo build` fails to fetch crates.** The build needs network access the
first time. After that, `cargo build --offline` works.

**`maturin develop` says it cannot find a virtualenv.** Activate one first, or
use `maturin build` and `pip install` the wheel.

**Output differs between two machines.** Check `focusweave --opencv-version`
on both. It reports the OpenCV library actually loaded at run time, which for a
build linked against a system OpenCV is whatever that system has, not what the
build was compiled against.

**Re-running a folder stacks the previous result into itself.** The default
output, `stacked.jpg`, is written inside the input folder, and the next run of
that folder sees it as one more frame. This is how the original behaved too.
Pass `--output` somewhere else, or delete it before re-running.

**WebP output fails.** WebP encoding is lossless-only here, and 16-bit images
are reduced to 8-bit for it, as for JPEG. Prefer PNG or TIFF for 16-bit
output.

**A stack takes much more memory than expected.** Peak memory scales with
`--workers`. Drop to `--workers 1` to roughly halve it at the cost of
throughput.
