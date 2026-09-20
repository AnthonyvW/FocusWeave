Reference implementation
========================

This is the original pure-Python/OpenCV implementation of FocusWeave, kept
verbatim so the Rust port can be checked against it. It is not packaged or
installed; the comparison scripts in `tests/` put this directory on
`sys.path` ahead of the installed package and import it directly.

Running the comparisons needs the reference's own dependencies:

    pip install numpy opencv-python-headless

Nothing in `python/focusweave/` or `crates/` depends on this directory.
