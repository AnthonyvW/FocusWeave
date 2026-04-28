# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["focusweave/main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "cv2",
        "numpy",
        "PIL",
        "PIL.Image",
        "PIL.ImageFile",
        "scipy.ndimage",
        "scipy.ndimage._filters",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FocusWeave",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
