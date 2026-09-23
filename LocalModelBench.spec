# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path

hiddenimports = []
hiddenimports += collect_submodules('local_model_bench')


a = Analysis(
    ['src\\local_model_bench\\main.py'],
    pathex=['src'],
    binaries=[],
    datas=[('assets\\app.ico', 'assets')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# A third-party ICU on PATH (for example Poppler's ICU 78) exports versioned
# symbols, while Qt6Core imports the Windows ICU API's unversioned symbols.
# Bundling that DLL makes PySide6.QtCore fail before the app window opens.
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.casefold() != 'icuuc.dll']
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LocalModelBench',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='assets\\version_info.txt',
    icon=['assets\\app.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LocalModelBench',
)
