"""Run every comparison the port is validated against.

    pip install -r tests/requirements.txt
    cargo build --release -p focusweave-cli
    maturin develop --release        # for the binding checks
    python tests/run_all.py

Each stage is a standalone script and can be run on its own.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGES = [
    ("primitives", "compare_primitives.py", "imgproc routines vs OpenCV"),
    ("registration", "compare_registration.py", "phase correlation and ECC vs OpenCV"),
    ("warps", "compare_warps.py", "per-pair alignment on a real stack"),
    ("pipeline", "compare_pipeline.py", "end-to-end CLI output across the flag matrix"),
    ("bindings", "compare_bindings.py", "the Python API surface"),
    ("streaming", "compare_streaming.py", "the incremental stacker"),
]


def main() -> int:
    selected = sys.argv[1:]
    failed: list[str] = []
    for name, script, description in STAGES:
        if selected and name not in selected:
            continue
        print(f"\n{'=' * 72}\n{name}: {description}\n{'=' * 72}")
        result = subprocess.run([sys.executable, str(ROOT / "tests" / script)], cwd=ROOT)
        if result.returncode != 0:
            failed.append(name)

    print(f"\n{'=' * 72}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("all comparison stages passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
