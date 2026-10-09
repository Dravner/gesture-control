from pathlib import Path
import sys
import pytest
from gesture_system import app_paths


def resources(tmp_path):
    root = tmp_path / 'resources'
    (root / 'models').mkdir(parents=True)
    for name in app_paths.BUILTIN_MODELS:
        (root / 'models' / name).write_bytes(b'builtin')
    (root / 'data' / 'gesture_profiles').mkdir(parents=True)
    (root / 'data' / 'gesture_profiles' / 'private.json').write_text('private')
    (root / 'models' / 'unlisted.bin').write_bytes(b'private')
    return root


def test_bootstrap_preserves_user_files_and_excludes_project_data(tmp_path):
    source = resources(tmp_path)
    destination = tmp_path / 'user'
    (destination / 'models').mkdir(parents=True)
    existing = destination / 'models' / app_paths.BUILTIN_MODELS[0]
    existing.write_bytes(b'user model')
    (destination / 'data').mkdir()
    settings = destination / 'data' / 'ui_settings.json'
    settings.write_text('custom settings')
    app_paths.bootstrap_user_root(source, destination)
    assert existing.read_bytes() == b'user model'
    assert settings.read_text() == 'custom settings'
    assert not (destination / 'data' / 'gesture_profiles').exists()
    assert not (destination / 'models' / 'unlisted.bin').exists()
    assert len(list((destination / 'models').iterdir())) == len(app_paths.BUILTIN_MODELS)
    app_paths.bootstrap_user_root(source, destination)
    assert existing.read_bytes() == b'user model'


def test_paths_use_frozen_resources_and_explicit_home(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
    assert app_paths.resource_root() == tmp_path
    assert app_paths.application_data_root(tmp_path) == tmp_path / 'Library' / 'Application Support' / 'GestureControl'


def test_missing_resource_reports_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='Missing bundled model'):
        app_paths.bootstrap_user_root(tmp_path / 'missing', tmp_path / 'user')
