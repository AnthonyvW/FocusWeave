"""Copy an ELF binary and the shared libraries it needs into one directory.

    python ci/bundle_elf.py target/release/focusweave FocusWeave-linux \
                            --search /path/to/opencv/lib

The result runs on a machine with none of those libraries installed: every
non-system dependency is copied in beside the binary and RPATH is rewritten to
$ORIGIN, so the loader looks next to the executable first. Needs patchelf.
"""
from __future__ import annotations

import argparse
import os
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


def needed(path: Path, env: dict[str, str]) -> tuple[dict[str, Path], list[str]]:
    out = subprocess.run(
        ["ldd", str(path)], capture_output=True, text=True, check=True, env=env
    ).stdout
    found: dict[str, Path] = {}
    missing: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if "=>" not in line:
            continue
        name, _, rest = line.partition(" => ")
        name = name.strip()
        if name in SYSTEM:
            continue
        resolved = rest.split(" (")[0].strip()
        # An unresolved entry means the bundle would be missing a library. It
        # has to stop the build: the alternative is an archive that looks fine
        # here and dies on the loader for whoever downloads it.
        if not resolved or resolved == "not found":
            missing.append(name)
            continue
        found[name] = Path(resolved)
    return found, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("dest")
    parser.add_argument("--search", action="append", default=[],
                        help="extra directory to resolve libraries from")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.search:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*args.search, existing]).strip(os.pathsep)

    binary = Path(args.binary)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / binary.name
    shutil.copy2(binary, target)

    pending = [target]
    copied: set[str] = set()
    unresolved: set[str] = set()
    while pending:
        current = pending.pop()
        found, missing = needed(current, env)
        unresolved.update(missing)
        for name, source in found.items():
            if name in copied:
                continue
            copied.add(name)
            local = dest / name
            shutil.copy2(source, local)
            local.chmod(0o755)
            pending.append(local)

    if unresolved:
        print(f"unresolved libraries: {', '.join(sorted(unresolved))}", file=sys.stderr)
        print("pass --search with the directory holding them", file=sys.stderr)
        return 1

    for path in [target, *(dest / name for name in copied)]:
        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(path)], check=True)
        subprocess.run(["strip", "--strip-unneeded", str(path)], check=False)

    total = sum(p.stat().st_size for p in dest.iterdir() if not p.is_symlink())
    print(f"bundled {len(copied)} libraries, {total / 1e6:.1f} MB total")
    for name in sorted(copied):
        print(f"  {(dest / name).stat().st_size / 1e6:6.2f} MB  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
