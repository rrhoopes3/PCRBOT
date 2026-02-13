# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('bearish.mp4', '.'),
        ('bullish.mp4', '.'),
        ('neutral.mp4', '.'),
        ('bearish.m4a', '.'),
        ('bullish.m4a', '.'),
        ('neutral.m4a', '.'),
        ('app_icon.ico', '.'),
    ],
    hiddenimports=[
        'PyQt6.QtMultimedia',
        'PyQt6.QtMultimediaWidgets',
        'PyQt6.sip',
        'pandas',
        'numpy',
        'plotly',
        'plotly.express',
        'plotly.graph_objects',
        'requests',
        'urllib3',
        'charset_normalizer',
        'certifi',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MultiPCRDashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Change to True temporarily for debugging
    icon='app_icon.ico',
)

# One-folder build (easier to distribute, less DLL issues)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MultiPCRDashboard',
)
