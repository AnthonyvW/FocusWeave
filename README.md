FocusWeave
==========

Focus stacking via Laplacian pyramid fusion. Takes a set of images captured at
different focus distances and combines them into a single image where the entire
subject is sharp.

Written in Rust with Python bindings. Every image-processing routine the
pipeline needs is implemented in this repository, so there is no OpenCV to
install and nothing to link against — the command line binary is a single
self-contained file of about 3 MB. An `opencv-backend` feature links the C++
library instead, for anyone who would rather have the last 25% of speed than
the standalone binary; see [RUNNING.md](RUNNING.md) for the comparison.

Download
--------
Pre-built executables for Windows, macOS and Linux are available on the
[releases tab](https://github.com/AnthonyvW/FocusWeave/releases), alongside
Python wheels. Download the binary for your platform.

Basic usage
-----------
Point focusweave at a folder of images and it will produce `stacked.jpg` inside
that folder:

    focusweave path/to/images/

Input images can be JPG, PNG, TIFF, or WebP. Output format is inferred
from the extension:

    focusweave path/to/images/ --output result.tiff

For the full list of options:

    focusweave --help

Command-line options
--------------------

**Output options**

    --output PATH           Output file path (default: stacked.jpg in the input folder).
    --quality N             JPEG output quality 1–95 (default: 95).

    Alignment options
    --no-align              Skip alignment entirely. Use when images are already registered.
    --reference N           Index of the image to align all others to (default: middle image).
    --global-align          Align every image directly to the reference instead of
                            chaining through neighbours. More robust when images are not
                            ordered by similarity.
    --full-res              Run the fine alignment pass at full resolution instead of the
                            default 1024px cap. More accurate but significantly slower.
    --min-shift PIXELS      Minimum shift in pixels before alignment is applied (default: 5.0).
    --no-rotation           Suppress rotation correction during alignment.
    --no-scale              Suppress scale correction during alignment.
    --no-shear              Suppress shear correction during alignment.
    --no-translation        Suppress translation correction during alignment.

    Canvas options
    --keep-size             Keep the output the same size as the inputs; warps are applied
                            in-place rather than expanding the canvas.
    --crop                  Crop the output to the intersection of all image extents —
                            removes all border regions but produces a smaller result.
    --no-fill               Fill border regions with black instead of reflecting edge pixels.
                            Pairs naturally with --crop to trim the borders away.

    Stacking options
    --levels N              Laplacian pyramid levels (default: auto from image size).
    --sharpness EXPONENT    Weight sharpness exponent (default: 4.0). Higher values favour
                            the sharpest image more aggressively at each pixel, approaching
                            a hard winner-take-all selection. Useful range is roughly
                            1.0 (soft blend) to 8.0 (near-hard selection).
    --workers N             Number of parallel stacking workers (default: 3). Higher values
                            are faster but increase peak RAM by ~100 MiB per additional
                            worker. Set to 0 to use all CPU cores.

    Culling options
    --cull [THRESHOLD]      Remove wholly out-of-focus images before stacking. Each frame
                            is scored by the high- to low-frequency energy ratio of its
                            Tenengrad response; frames scoring below THRESHOLD are dropped.
                            The threshold is absolute, not relative to the sharpest frame.
                            THRESHOLD defaults to 0.6 when --cull is given without a value.
                            At least the two sharpest frames are always retained.

**Slabbing options**


Slabbing splits a large image set into overlapping sub-stacks, stacks each one independently, then fuses the results. This can improve quality by reducing the number of images competing in each fusion pass.

    --slab SIZE OVERLAP     Enable slabbing. SIZE is images per sub-stack; OVERLAP is how
                            many images adjacent slabs share. Example: --slab 20 5
    --output-steps          Save each intermediate slab result to a focusweave_slabs/
                            folder inside the output directory. Requires --slab.
    --only-slab             Stop after producing slabs; skip the final fusion. Implies
                            --output-steps. Requires --slab.
    --recursive-slab        If the layer-1 slab results still outnumber SIZE, apply
                            slabbing again as layer 2, and so on, until the count fits in
                            a single stack pass.
    --slab-format EXT       File format for slab output images (e.g. tiff, png, jpg).
                            Defaults to tiff. Requires --output-steps or --only-slab.

Memory usage
------------
With default settings (3 workers), expect around 200 MiB per megapixel of input
image resolution. To halve memory usage at the cost of roughly double the
processing time, set workers to 1:

    focusweave path/to/images/ --workers 1

Set `--workers 0` to use every core.

Python API
----------
focusweave can be installed as a dependency in your own project and used
directly as a library without going through the CLI. Add it to your
`pyproject.toml`:

```toml
[project]
dependencies = [
    "focusweave @ git+https://github.com/AnthonyvW/FocusWeave.git",
]
```

Or install it into your environment directly:

    pip install "focusweave @ git+https://github.com/AnthonyvW/FocusWeave.git"

All public symbols are importable from the top-level `focusweave` package.
Only numpy is required at runtime. The main entry point is `FocusStackConfig`
and `run`:

```python
from pathlib import Path
from focusweave import FocusStackConfig, run

cfg = FocusStackConfig(images=Path("path/to/images/"))
result = run(cfg)

# result.image is a uint8 RGB numpy array
```

Images can be supplied as a folder path, a list of `Path` objects, or a list of
pre-loaded `numpy` arrays:

```python
import numpy as np
from focusweave import FocusStackConfig, run

images: list[np.ndarray] = [...]  # pre-loaded uint8 RGB arrays
cfg = FocusStackConfig(images=images, workers=4)
result = run(cfg)
```

A progress callback can be passed to `run` to receive stage-by-stage updates:

```python
from focusweave import FocusStackConfig, run

def on_progress(fraction: float, stage: str, message: str) -> None:
    print(f"[{stage}] {fraction * 100:.1f}%  {message}")

cfg = FocusStackConfig(images=Path("path/to/images/"))
result = run(cfg, progress=on_progress)
```

Long-running stacks can be cancelled by supplying an interrupt callback in the
config. If it returns `True` at any checkpoint, `Interrupted` is raised:

```python
from pathlib import Path
from focusweave import FocusStackConfig, Interrupted, run

cancelled = False

cfg = FocusStackConfig(
    images=Path("path/to/images/"),
    interrupt=lambda: cancelled,
)

try:
    result = run(cfg)
except Interrupted:
    print("Stack cancelled.")
```

Images can be read and written without pulling in another imaging library:

```python
from focusweave import load_image, save_image

frame = load_image(Path("frame_00.tiff"))   # uint8 or uint16 RGB
save_image(result.image, Path("stacked.tiff"))
```

See `python/focusweave/api_example.py` for a more complete example, or run it:

    python -m focusweave.api_example path/to/images/ --streaming

Building from source
--------------------
A Rust toolchain (1.82 or newer) is required; Python 3.10 or newer if you want
the bindings.

There are two builds. The default needs nothing but Rust and produces a
self-contained binary. The `opencv-backend` one calls the OpenCV C++ library
for its image processing and is about 25% faster, at the cost of needing
OpenCV 4 development files and libclang to build and linking OpenCV's shared
libraries at run time. **If you are benchmarking, build that one** —
[RUNNING.md](RUNNING.md) has the per-platform prerequisites and the numbers.

Command line binary:

    cargo build --release -p focusweave-cli                            # default
    cargo build --release -p focusweave-cli --features opencv-backend  # faster
    ./target/release/focusweave --help

Python package, via [maturin](https://maturin.rs):

    pip install maturin
    maturin develop --release                            # default
    maturin develop --release --features opencv-backend  # faster

Once installed, the `focusweave` command is available on your PATH and is the
same CLI as the native binary.

[RUNNING.md](RUNNING.md) walks through building, testing and benchmarking in
more detail. [docs/PORTING-NOTES.md](docs/PORTING-NOTES.md) records how the
Rust implementation relates to the original Python one, including where the two
deliberately differ.

Testing
-------
The original pure-Python/OpenCV implementation is preserved under
`tests/reference/`, and the port is validated against it:

    pip install -r tests/requirements.txt
    cargo build --release -p focusweave-cli
    maturin develop --release
    python tests/run_all.py

`cargo test` covers the parts that stand alone from image data.

Algorithms
----------
The focus stacking algorithm is based on Laplacian pyramid fusion as described in:

> Wang, W., & Chang, F. (2011). A Multi-focus Image Fusion Method Based on
> Laplacian Pyramid. *Journal of Computers*.

Image alignment uses a custom coarse-to-fine pipeline built on an enhanced
correlation coefficient solver (Evangelidis & Psarakis, 2008), implemented
here. It seeds ECC with a phase-correlation translation estimate,
applies CLAHE normalisation and a focus-aware pixel mask to concentrate the
optimisation on sharp, informative regions, then validates the result against the
seed to reject false minima. Warps are composed mathematically through a
neighbour chain so interpolation error does not accumulate across the stack.