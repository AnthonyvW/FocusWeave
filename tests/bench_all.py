"""Time the three implementations end to end on the same image set.

    python tests/bench_all.py [image-folder] [repeats]

Builds nothing; run the two CLI builds first:

    cargo build --release -p focusweave-cli && cp target/release/focusweave target/focusweave-native
    cargo build --release -p focusweave-cli --features opencv-backend && cp target/release/focusweave target/focusweave-opencv
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

CASES = [("full run", []), ("fusion only", ["--no-align"]), ("all cores", ["--workers", "0"])]


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

    kwargs: dict[str, object] = {"workers": 3}
    if "--no-align" in flags:
        kwargs["no_align"] = True
    if "--workers" in flags:
        kwargs["workers"] = 0
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
    import struct
    print(f"{len(frames)} frames from {FOLDER}, best of {REPEATS}\n")

    binaries = {
        "rust (own kernels)": ROOT / "target" / "focusweave-native",
        "rust (opencv)": ROOT / "target" / "focusweave-opencv",
    }
    missing = [n for n, p in binaries.items() if not p.exists()]
    if missing:
        print(f"missing builds: {', '.join(missing)}")
        return 1

    rows: list[tuple[str, list[str]]] = []
    header = f"{'case':<14}" + "".join(f"{name:>22}" for name in [*binaries, "python (opencv)"])
    print(header)
    print("-" * len(header))
    for label, flags in CASES:
        cells = []
        for path in binaries.values():
            cells.append(f"{time_binary(path, flags):.2f} s")
        cells.append(f"{time_python(flags):.2f} s" if flags != ["--workers", "0"] else "-")
        print(f"{label:<14}" + "".join(f"{c:>22}" for c in cells))
        rows.append((label, cells))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
