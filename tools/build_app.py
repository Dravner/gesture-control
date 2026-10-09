"""Reproducible macOS arm64 app build from the project's installed environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports' / 'desktop-package-20261009'


def create_icon():
    from PIL import Image, ImageDraw
    image = Image.new('RGBA', (1024, 1024))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 20, 1004, 1004), radius=220, fill='#102b45')
    # Original geometric hand, drawn locally; no external photographs/assets.
    draw.rounded_rectangle((338, 410, 724, 816), radius=140, fill='#6fe3c4')
    for x, top in ((350, 260), (448, 184), (546, 214), (644, 286)):
        draw.rounded_rectangle((x, top, x+80, 602), radius=40, fill='#6fe3c4')
    draw.rounded_rectangle((246, 478, 416, 690), radius=70, fill='#6fe3c4')
    draw.arc((184, 150, 846, 878), 170, 252, fill='#ffffff', width=22)
    image.save(ROOT / 'packaging' / 'GestureControl.icns', format='ICNS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('Build requires native Apple Silicon macOS; create a separate build for Intel.')
    import PyInstaller
    REPORT.mkdir(parents=True, exist_ok=True)
    create_icon()
    sys.path.insert(0, str(ROOT))
    from gesture_system.app_paths import BUILTIN_MODELS
    manifest = {'python':sys.version, 'pyinstaller':PyInstaller.__version__,
        'architecture':platform.machine(), 'build_macos':platform.mac_ver()[0],
        'bundle_id':'org.dravner.gesturecontrol', 'minimum_macos':'15.0', 'models':{}}
    for name in BUILTIN_MODELS:
        path = ROOT / 'models' / name
        manifest['models'][name] = {'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (REPORT / 'build-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    with (REPORT / 'environment.txt').open('w') as output:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=output, check=True)
    if args.prepare_only:
        return 0
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
        '--distpath', str(ROOT / 'dist'), '--workpath', str(ROOT / 'build' / 'desktop'),
        str(ROOT / 'packaging' / 'GestureControl.spec')]
    build_environment = os.environ.copy()
    build_environment['PYINSTALLER_CONFIG_DIR'] = str(ROOT / 'build' / 'pyinstaller-cache')
    build_environment['MPLCONFIGDIR'] = str(ROOT / 'build' / 'matplotlib-cache')
    with (REPORT / 'build.log').open('w') as output:
        subprocess.run(command, cwd=ROOT, env=build_environment, stdout=output, stderr=subprocess.STDOUT, check=True)
    bundle = ROOT / 'dist' / 'Жестовое управление.app'
    with (REPORT / 'codesign.txt').open('w') as output:
        subprocess.run(['codesign', '--verify', '--deep', '--strict', '--verbose=1', str(bundle)],
            stdout=output, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['codesign', '-d', '--verbose=4', str(bundle)], stdout=output, stderr=subprocess.STDOUT, check=True)
    print(bundle)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
