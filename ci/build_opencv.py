"""Build a minimal OpenCV and install it to a prefix.

    python ci/build_opencv.py --version 4.10.0 --prefix ~/opencv-min \
                              --env-file "$GITHUB_ENV"

Only core and imgproc are built, and everything those two can optionally link
is turned off. That matters more than it sounds: a distribution OpenCV links
LAPACK, which pulls in OpenBLAS, gfortran and libgcc, and none of it is ever
called from here. Left in, it is 11 MB of a Linux wheel, it is what makes a
macOS bundle unrelocatable, and on Windows the only prebuilt option is the
monolithic opencv_world carrying every module there is.

Writing one build for all three platforms also puts them on the same OpenCV
version, so a signature change cannot break one and not the others.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

SOURCE_URL = "https://github.com/opencv/opencv/archive/refs/tags/{version}.tar.gz"

MODULES = ["core", "imgproc"]

FLAGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    f"-DBUILD_LIST={','.join(MODULES)}",
    "-DBUILD_SHARED_LIBS=ON",
    # Install the same way everywhere. Left alone, Windows installs into
    # <prefix>/x64/vc17/ and the three platforms need three sets of paths.
    "-DOPENCV_BIN_INSTALL_PATH=bin",
    "-DOPENCV_LIB_INSTALL_PATH=lib",
    "-DOPENCV_3P_LIB_INSTALL_PATH=lib",
    "-DOPENCV_INCLUDE_INSTALL_PATH=include",
    "-DOPENCV_CONFIG_INSTALL_PATH=cmake",
    "-DOPENCV_GENERATE_PKGCONFIG=ON",
    "-DOPENCV_GENERATE_SETUPVARS=OFF",
    # macOS only, and what makes the dylibs relocatable: their install name
    # becomes @rpath/... instead of this prefix, so copying them next to a
    # binary that carries an @executable_path rpath is the whole of bundling.
    "-DCMAKE_INSTALL_NAME_DIR=@rpath",
    # The numerics stack this whole script exists to avoid.
    "-DWITH_LAPACK=OFF",
    "-DWITH_EIGEN=OFF",
    # Display, acceleration and codec backends, none of which core or imgproc
    # need and all of which become bundled libraries.
    "-DWITH_OPENGL=OFF",
    "-DWITH_OPENCL=OFF",
    "-DWITH_IPP=OFF",
    "-DWITH_TBB=OFF",
    "-DWITH_ITT=OFF",
    "-DWITH_PROTOBUF=OFF",
    "-DWITH_FFMPEG=OFF",
    "-DWITH_GSTREAMER=OFF",
    "-DWITH_GTK=OFF",
    "-DWITH_QT=OFF",
    "-DWITH_WIN32UI=OFF",
    "-DWITH_AVFOUNDATION=OFF",
    "-DWITH_V4L=OFF",
    "-DWITH_1394=OFF",
    "-DWITH_ADE=OFF",
    "-DWITH_VTK=OFF",
    # zlib is the one thing core genuinely needs; building the bundled copy in
    # removes the last external dependency.
    "-DBUILD_ZLIB=ON",
    "-DWITH_JPEG=OFF",
    "-DWITH_PNG=OFF",
    "-DWITH_TIFF=OFF",
    "-DWITH_WEBP=OFF",
    "-DWITH_OPENJPEG=OFF",
    "-DWITH_JASPER=OFF",
    "-DWITH_OPENEXR=OFF",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_DOCS=OFF",
    "-DBUILD_opencv_apps=OFF",
    "-DBUILD_JAVA=OFF",
    "-DBUILD_opencv_python2=OFF",
    "-DBUILD_opencv_python3=OFF",
]


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(c) for c in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def fetch_source(version: str, work: Path) -> Path:
    source = work / f"opencv-{version}"
    if (source / "CMakeLists.txt").exists():
        print(f"reusing {source}")
        return source
    archive = work / f"opencv-{version}.tar.gz"
    if not archive.exists():
        url = SOURCE_URL.format(version=version)
        print(f"downloading {url}", flush=True)
        # Written to a temporary name first so an interrupted download is not
        # mistaken for a complete one on the next run.
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(url) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.rename(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    return source


def link_libraries(prefix: Path) -> list[str]:
    """Names for OPENCV_LINK_LIBS, as the installed files actually spell them.

    Windows versions its import libraries (opencv_core4100.lib) and the other
    two do not, so this is discovered rather than constructed.
    """
    lib_dir = prefix / "lib"
    names = []
    for module in MODULES:
        matches = sorted(lib_dir.glob(f"*opencv_{module}*"))
        if not matches:
            raise SystemExit(f"no library for opencv_{module} in {lib_dir}")
        stem = matches[0].name
        for prefix_text in ("lib",):
            stem = stem.removeprefix(prefix_text)
        stem = stem.split(".")[0]
        names.append(stem)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--work", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args()

    prefix = Path(args.prefix).expanduser().resolve()
    work = Path(args.work).expanduser().resolve() if args.work else prefix.parent / "opencv-build"
    work.mkdir(parents=True, exist_ok=True)

    marker = prefix / ".focusweave-opencv-version"
    if marker.exists() and marker.read_text().strip() == args.version:
        print(f"OpenCV {args.version} already installed at {prefix}")
    else:
        source = fetch_source(args.version, work)
        build = work / "build"
        build.mkdir(exist_ok=True)
        configure = ["cmake", "-S", str(source), "-B", str(build),
                     f"-DCMAKE_INSTALL_PREFIX={prefix}", *FLAGS]
        if shutil.which("ninja") and platform.system() != "Windows":
            configure += ["-G", "Ninja"]
        run(configure)
        run(["cmake", "--build", str(build), "--config", "Release",
             "--parallel", str(args.jobs)])
        # --strip is worth about 2.5 MB of the two libraries, and every
        # artifact carries them. MSVC has no strip tool and cmake skips it.
        run(["cmake", "--install", str(build), "--config", "Release", "--strip"])
        marker.write_text(args.version)

    include = prefix / "include"
    # 4.x installs headers under include/opencv4; 5.x drops the suffix.
    if (include / "opencv4" / "opencv2").is_dir():
        include = include / "opencv4"

    runtime = prefix / ("bin" if platform.system() == "Windows" else "lib")
    settings = {
        "OPENCV_INCLUDE_PATHS": str(include),
        "OPENCV_LINK_PATHS": str(prefix / "lib"),
        "OPENCV_LINK_LIBS": ",".join(link_libraries(prefix)),
        "FOCUSWEAVE_OPENCV_LIB_DIR": str(runtime),
    }
    for key, value in settings.items():
        print(f"{key}={value}")
    if args.env_file:
        with open(args.env_file, "a", encoding="utf-8") as handle:
            for key, value in settings.items():
                handle.write(f"{key}={value}\n")

    # Not counting symlinks, of which each library installs two.
    total = sum(
        p.stat().st_size for p in runtime.rglob("*") if p.is_file() and not p.is_symlink()
    )
    print(f"runtime libraries: {total / 1e6:.1f} MB in {runtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
