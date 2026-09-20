Running the Rust FocusWeave
===========================

FocusWeave is now a Rust workspace with Python bindings. Everything the
pipeline needs — image decoding, filtering, CLAHE, phase correlation, the ECC
solver, pyramid fusion — is implemented in this repository, so there is no
OpenCV to install and nothing to link against. You need a Rust toolchain, and
Python only if you want the bindings.

Contents:

1. [Get a toolchain](#1-get-a-toolchain)
2. [Build and run the CLI](#2-build-and-run-the-cli)
3. [Build and use the Python package](#3-build-and-use-the-python-package)
4. [Make yourself a test stack](#4-make-yourself-a-test-stack)
5. [Check it against the old implementation](#5-check-it-against-the-old-implementation)
6. [What to look at first](#6-what-to-look-at-first)
7. [Troubleshooting](#7-troubleshooting)


1. Get a toolchain
------------------

Rust 1.82 or newer:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

On Windows, install [rustup](https://rustup.rs) and the MSVC build tools.

Check it:

    cargo --version

That is the only hard requirement. The rest of this guide is optional
depending on what you want to try.


2. Build and run the CLI
------------------------

    cargo build --release -p focusweave-cli

The binary lands at `target/release/focusweave` (`focusweave.exe` on Windows).
It is self-contained — about 3 MB, no shared libraries beyond libc — so you can
copy it anywhere.

    ./target/release/focusweave --help
    ./target/release/focusweave path/to/images/

By default it writes `stacked.jpg` into the input folder. Every flag from the
old Python CLI is accepted, with identical names and defaults:

    ./target/release/focusweave path/to/images/ --output result.tiff --workers 0
    ./target/release/focusweave path/to/images/ --cull --crop
    ./target/release/focusweave path/to/images/ --slab 20 5 --output-steps

Two flags to know about while testing:

- `--workers 0` uses every core. The default is still 3, matching the old
  behaviour and its memory profile.
- `--no-align` skips registration. Useful for isolating the fusion stage when
  comparing output against the old implementation.


3. Build and use the Python package
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


4. Make yourself a test stack
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


5. Check it against the old implementation
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


6. What to look at first
------------------------

Some things worth poking at, in rough order of how likely they are to matter
to you:

**Does it produce the picture you expect.** Run it on a real stack you already
have a known-good result for and compare. This is the check that matters; the
synthetic tests only prove the port is faithful, not that you like the output.

**Speed.** On this machine (4 cores, 10 frames at 2000x1400):

|                       | Rust    | Python + OpenCV |
| --------------------- | ------- | --------------- |
| fusion only (`--no-align`) | 1.9 s | 2.2 s         |
| full run              | 4.4 s   | 3.2 s           |
| full run, `--workers 0` | 3.9 s |                 |
| peak memory           | 926 MiB | 1085 MiB        |

Fusion is now faster than the OpenCV build. The full run is still about 1.3x
behind, and all of that sits in alignment — specifically in `warp_affine`,
which the ECC solver calls four times per iteration. Per core the kernels
compare like this against OpenCV's SIMD:

| kernel                        | Rust    | OpenCV  |
| ----------------------------- | ------- | ------- |
| `sepFilter2D` 5-tap RGB f32   | 44 ms   | 12 ms   |
| `sepFilter2D` 5-tap gray f32  | 15 ms   | 2 ms    |
| `GaussianBlur` 15 gray        | 23 ms   | 6 ms    |
| `warpAffine` cubic RGB u8     | 225 ms  | 52 ms   |
| `warpAffine` linear gray f32  | 52 ms   | 7 ms    |
| RGB to Lab (L only)           | 7 ms    | 14 ms   |

Two things to read from that. The Lab conversion is faster because the port
only computes the channel the fusion weights actually use, which is an
algorithmic win rather than a micro-optimised one. Everything else is the
SIMD gap: OpenCV dispatches to AVX2 at runtime and accumulates a whole kernel
in vector registers, while these loops are what the autovectoriser manages on
its own. Closing it means hand-written SIMD in two functions — `sep_filter`
and the warp inner loop — with runtime feature detection so the binary stays
portable. `cargo run --release -p focusweave-core --example bench_primitives`
reproduces the table, and `RAYON_NUM_THREADS=1` gives the per-core figures.

**Alignment on a hard stack.** Macro stacks with lots of out-of-focus area are
where the ECC differences would show. Run with `--no-align` and without, on
the same set, and compare.

**16-bit input.** `--output result.tiff` with 16-bit TIFF sources keeps full
depth end to end.

**The Python API**, if you use FocusWeave as a library. Everything that used
to be importable still is, from the same module paths.


7. Troubleshooting
------------------

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
