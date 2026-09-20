"""Compare the registration the Rust port computes against the Python reference.

Pixel-level comparison of a full run is confounded by the pipeline's
`min_shift` gate: a cumulative warp whose translation lands within a hair of
the threshold is snapped to the identity, so a sub-pixel disagreement between
two correct solvers can flip whole frames between "warped" and "copied". This
checks the quantity that actually reflects registration accuracy — the warps
themselves — before that gate is applied.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "reference"))

from focusweave.focus_stack import (  # noqa: E402
    _load_raw_u8, _prepare_for_ecc, _run_ecc, _to_gray_cv, resolve_images,
)
import make_stack  # noqa: E402

TRANSLATION_TOL = 0.15  # pixels
LINEAR_TOL = 1e-3

WORK = ROOT / "target" / "warpcheck"


def main() -> int:
    folder = WORK / "src8"
    if not folder.exists():
        make_stack.build(folder, depth=8)

    dump = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "focusweave-core",
         "--example", "dump_pairs", "--", str(folder)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout

    rust = {}
    for line in dump.splitlines():
        parts = line.split()
        rust[int(parts[0])] = (int(parts[1]), np.array([float(v) for v in parts[2:]]).reshape(2, 3))

    sources, size = resolve_images(folder)
    grays = [_to_gray_cv(_load_raw_u8(s, size)) for s in sources]
    fine = min(1024, max(grays[0].shape))
    prepared = [{fine: _prepare_for_ecc(g, fine)} for g in grays]

    failures: list[str] = []
    print(f"{'pair':6s} {'py translation':>21s} {'rust translation':>21s} {'d trans':>9s} {'d linear':>10s}")
    for i in range(1, len(grays)):
        expected, converged = _run_ecc(grays[i - 1], grays[i], False, prepared[i - 1], prepared[i])
        rust_converged, got = rust[i]
        d_trans = float(np.abs(got[:, 2] - expected[:, 2]).max())
        d_lin = float(np.abs(got[:, :2] - expected[:, :2]).max())
        ok = (d_trans <= TRANSLATION_TOL and d_lin <= LINEAR_TOL
              and rust_converged == int(converged))
        print(f"{i-1}->{i}  ({expected[0,2]:+9.4f},{expected[1,2]:+9.4f}) "
              f"({got[0,2]:+9.4f},{got[1,2]:+9.4f}) {d_trans:9.4f} {d_lin:10.2e}"
              f"{'' if ok else '   FAIL'}")
        if not ok:
            failures.append(
                f"pair {i-1}->{i}: d_trans={d_trans:.4f} d_linear={d_lin:.3g} "
                f"converged py={int(converged)} rust={rust_converged}"
            )

    print()
    if failures:
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"all pairwise warps agree within {TRANSLATION_TOL} px and {LINEAR_TOL:g} linear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
