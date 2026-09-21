"""Copy an ELF binary and the shared libraries it needs into one directory.

    python ci/bundle_elf.py target/release/focusweave upload/FocusWeave-linux

The result runs on a machine with no OpenCV installed: every non-system
dependency is copied in beside the binary and RPATH is rewritten to $ORIGIN, so
the loader looks next to the executable first. Needs patchelf.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Libraries every glibc system already provides. Copying these in is not just
# redundant, it is actively harmful: a bundled libc that disagrees with the
# host's loader will not load at all.
SYSTEM = {
    "linux-vdso.so.1", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1",
    "libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "librt.so.1",
    "libgcc_s.so.1", "libstdc++.so.6", "libresolv.so.2", "libutil.so.1",
}


def needed(path: Path) -> dict[str, Path]:
    out = subprocess.run(["ldd", str(path)], capture_output=True, text=True, check=True).stdout
    found: dict[str, Path] = {}
    for line in out.splitlines():
        line = line.strip()
        if "=>" not in line:
            continue
        name, _, rest = line.partition(" => ")
        name = name.strip()
        resolved = rest.split(" (")[0].strip()
        if name in SYSTEM or not resolved or resolved == "not found":
            continue
        found[name] = Path(resolved)
    return found


def main() -> int:
    binary = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / binary.name
    shutil.copy2(binary, target)

    pending = [target]
    copied: set[str] = set()
    while pending:
        current = pending.pop()
        for name, source in needed(current).items():
            if name in copied:
                continue
            copied.add(name)
            local = dest / name
            shutil.copy2(source, local)
            local.chmod(0o755)
            pending.append(local)

    for path in [target, *(dest / name for name in copied)]:
        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(path)], check=True)

    total = sum(p.stat().st_size for p in dest.iterdir())
    print(f"bundled {len(copied)} libraries, {total / 1e6:.1f} MB total")
    for name in sorted(copied):
        print(f"  {(dest / name).stat().st_size / 1e6:6.2f} MB  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
