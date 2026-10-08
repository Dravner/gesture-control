import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication
from gesture_system.workspace_pointer import WorkspaceMapper,WorkspaceBounds
from gesture_system.workspace_ui import WorkspaceEditor

@pytest.fixture
def app():return QApplication.instance() or QApplication([])


def test_editor_preserves_relative_flag_and_recenter_is_only_absolute(tmp_path,app):
    context={'source_kind':'camera','frame_size':[640,480]}
    mapper=WorkspaceMapper(context,WorkspaceBounds(span=(.35,.35)),relative=True)
    editor=WorkspaceEditor(tmp_path,mapper,lambda:context)
    assert editor.relative_checkbox.isChecked() and not editor.recenter_button.isEnabled()
    editor.width_slider.setValue(40);assert editor.mapper.relative
    editor.relative_checkbox.setChecked(False)
    assert not editor.mapper.relative and editor.recenter_button.isEnabled()
    editor.relative_checkbox.setChecked(True);editor.save_configuration()
    assert WorkspaceMapper.load(tmp_path/'data'/'workspace_config.json').relative
