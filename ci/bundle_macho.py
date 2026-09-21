"""Copy a Mach-O binary and the dylibs it needs into one directory.

    python ci/bundle_macho.py target/release/focusweave FocusWeave-macos \
                              --search ~/opencv-min/lib

The OpenCV built by ci/build_opencv.py installs with @rpath install names and
the binary is linked with an @executable_path rpath, so placing the dylibs
beside it is the whole of bundling -- nothing needs its load commands
rewritten, and the code signature stays valid.

Each library installs as one real file and two symlinks to it. Only the name
the binary actually references is copied, dereferenced: upload-artifact and
most zip tools follow symlinks, so shipping all three would store three full
copies of every library.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Everything under these prefixes ships with macOS, and several of them cannot
# legally or safely be copied out of it.
SYSTEM_PREFIXES = ("/usr/lib/", "/System/")


def needed(path: Path) -> list[str]:
    out = subprocess.run(
        ["otool", "-L", str(path)], capture_output=True, text=True, check=True
    ).stdout
    names = []
    for line in out.splitlines()[1:]:
        install_name = line.strip().split(" (compatibility")[0].strip()
        if not install_name or install_name.startswith(SYSTEM_PREFIXES):
            continue
        # A dylib's own id is the first entry and is not a dependency.
        if Path(install_name).name == path.name:
            continue
        names.append(install_name)
    return names


def resolve(install_name: str, search: list[Path], dest: Path) -> Path | None:
    name = Path(install_name).name
    if (dest / name).exists():
        return dest / name
    if install_name.startswith(("@rpath/", "@loader_path/", "@executable_path/")):
        for directory in search:
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None
    direct = Path(install_name)
    return direct if direct.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("dest")
    parser.add_argument("--search", action="append", default=[])
    args = parser.parse_args()

    search = [Path(s).expanduser() for s in args.search]
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
        for install_name in needed(current):
            name = Path(install_name).name
            if name in copied:
                continue
            source = resolve(install_name, search, dest)
            if source is None:
                unresolved.add(install_name)
                continue
            copied.add(name)
            local = dest / name
            # follow_symlinks resolves the link so one real file is stored
            # under the name the binary asks for.
            shutil.copy2(source, local, follow_symlinks=True)
            local.chmod(0o755)
            pending.append(local)

    if unresolved:
        print(f"unresolved libraries: {', '.join(sorted(unresolved))}", file=sys.stderr)
        print("pass --search with the directory holding them", file=sys.stderr)
        return 1

    total = sum(p.stat().st_size for p in dest.iterdir())
    print(f"bundled {len(copied)} libraries, {total / 1e6:.1f} MB total")
    for name in sorted(copied):
        print(f"  {(dest / name).stat().st_size / 1e6:6.2f} MB  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
