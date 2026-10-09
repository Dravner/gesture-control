# Standalone native macOS bundle; built-in resources are strictly allowlisted.
from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

ROOT = Path(SPECPATH).parent
exec((ROOT / 'gesture_system' / 'app_paths.py').read_text(), paths := {})
models = [(str(ROOT / 'models' / name), 'models') for name in paths['BUILTIN_MODELS']]
# Tasks C bindings locate their shared library and bundled package assets at runtime.
datas = models + collect_data_files('mediapipe') + copy_metadata('mediapipe')
datas += [(str(ROOT / 'packaging' / 'MODEL_PROVENANCE.md'), 'licenses')]
binaries = collect_dynamic_libs('mediapipe')
hiddenimports = ['Vision', 'Quartz', 'objc', 'Foundation', 'CoreML', 'Quartz.CoreVideo',
                 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
                 'gesture_system.temporal', 'src.realtime_inference']
a = Analysis([str(ROOT / 'desktop_launcher.py')], pathex=[str(ROOT)],
    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[], runtime_hooks=[], hooksconfig={'matplotlib': {'backends': ['Agg']}},
    excludes=['torchvision', 'tensorflow', 'IPython', 'notebook', 'jupyter',
              'pytest', 'pandas', 'sklearn', 'PyQt5', 'PyQt6', 'PySide2'],
    noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='GestureControl',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, argv_emulation=False, target_arch='arm64',
    codesign_identity=os.environ.get('GESTURE_CODESIGN_IDENTITY'), entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='GestureControl')
app = BUNDLE(coll, name='Жестовое управление.app',
    icon=str(ROOT / 'packaging' / 'GestureControl.icns'),
    bundle_identifier='org.dravner.gesturecontrol',
    info_plist={'CFBundleDisplayName': 'Жестовое управление',
        'CFBundleName': 'Жестовое управление', 'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1', 'NSHighResolutionCapable': True,
        'NSCameraUsageDescription': 'Камера используется для распознавания жестов только после нажатия «Запустить».',
        'LSMinimumSystemVersion': '15.0', 'NSPrincipalClass': 'NSApplication'})
