"""Time the Rust build against the original Python implementation on one image set.

    python tests/bench_all.py [image-folder] [repeats]

Builds nothing; build the CLI first:

    cargo build --release -p focusweave-cli
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "target" / "bench"
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = ROOT / "target" / "bench-out.png"

CASES = [("full run", []), ("fusion only", ["--no-align"]), ("one worker", ["--workers", "1"])]


def time_binary(binary: Path, flags: list[str]) -> float:
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        subprocess.run([str(binary), str(FOLDER), "--output", str(OUT), *flags],
                       check=True, capture_output=True)
        best = min(best, time.perf_counter() - started)
    return best


def time_python(flags: list[str]) -> float:
    sys.path.insert(0, str(ROOT / "tests" / "reference"))
    for module in [m for m in sys.modules if m.startswith("focusweave")]:
        del sys.modules[module]
    from focusweave import FocusStackConfig, run

    kwargs: dict[str, object] = {}
    if "--no-align" in flags:
        kwargs["no_align"] = True
    if "--workers" in flags:
        kwargs["workers"] = int(flags[flags.index("--workers") + 1])
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        run(FocusStackConfig(images=FOLDER, **kwargs))
        best = min(best, time.perf_counter() - started)
    sys.path.pop(0)
    return best


def main() -> int:
    frames = sorted(p for p in FOLDER.iterdir() if p.suffix.lower() in {".png", ".tiff", ".jpg"})
    if not frames:
        print(f"no images in {FOLDER}")
        return 1
    print(f"{len(frames)} frames from {FOLDER}, best of {REPEATS}\n")

    binary = Path(sys.environ.get("FOCUSWEAVE_BIN", ROOT / "target" / "release" / "focusweave"))
    if not binary.exists():
        print(f"missing build: {binary}")
        return 1

    names = ["rust", "python (opencv)"]
    header = f"{'case':<14}" + "".join(f"{name:>22}" for name in names)
    print(header)
    print("-" * len(header))
    for label, flags in CASES:
        cells = [f"{time_binary(binary, flags):.2f} s", f"{time_python(flags):.2f} s"]
        print(f"{label:<14}" + "".join(f"{c:>22}" for c in cells))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
