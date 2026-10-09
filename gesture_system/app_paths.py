"""Read-only bundled resources and persistent per-user desktop application files."""
from pathlib import Path
import shutil
import sys

APP_DIRECTORY = 'GestureControl'
BUILTIN_MODELS = (
    'hand_landmarker.task', 'mlp_hagrid_6classes.pth',
    'mlp_hagrid_2d_no_gesture.pth', 'mlp_hagrid_subsample.pth',
    'streaming_joint.pth', 'streaming_selected.pth',
    'streaming_two_stage.pth', 'streaming_window.pth', 'selection.json',
)


def resource_root():
    """PyInstaller exposes bundled resources beneath _MEIPASS."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def application_data_root(home=None):
    home = Path.home() if home is None else Path(home)
    return home / 'Library' / 'Application Support' / APP_DIRECTORY


def bootstrap_user_root(resources=None, destination=None):
    """Install only missing built-in models; never import project user data.

    Exclusive creation also avoids replacing a model another app process or the
    user has just installed. Existing profiles, settings and models are preserved.
    """
    resources = resource_root() if resources is None else Path(resources)
    destination = application_data_root() if destination is None else Path(destination)
    model_dir = destination / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    (destination / 'data').mkdir(parents=True, exist_ok=True)
    for name in BUILTIN_MODELS:
        source = resources / 'models' / name
        target = model_dir / name
        if not source.is_file():
            raise FileNotFoundError(f'Missing bundled model: {source}')
        try:
            output = target.open('xb')
        except FileExistsError:
            continue
        try:
            with source.open('rb') as original, output:
                shutil.copyfileobj(original, output)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    return destination
