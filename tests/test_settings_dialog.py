import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication


def test_dialog_saves_tuning_and_bindings_but_does_not_enable_input(tmp_path):
    from gesture_system.settings_ui import SettingsDialog
    from gesture_system.interaction_settings import InteractionSettings
    from gesture_system.profiles import GestureLibrary
    app=QApplication.instance() or QApplication([]);seen=[];library=GestureLibrary(tmp_path)
    d=SettingsDialog(InteractionSettings(),library.list_gestures(),lambda settings,bindings:seen.append((settings,bindings)))
    d.controls['pointer_sensitivity'].setValue(1.6)
    d.mapping_keys['swipe_left'].setText('command, [')
    d.save_button.click()
    assert seen[0][0].pointer_sensitivity==1.6
    assert seen[0][1]['swipe_left']['keys']==['command','[']
    assert not hasattr(d,'enable_input')
    d.close()


def test_invalid_shortcut_does_not_apply_partial_settings(tmp_path):
    from gesture_system.settings_ui import SettingsDialog
    from gesture_system.interaction_settings import InteractionSettings
    from gesture_system.profiles import GestureLibrary
    app=QApplication.instance() or QApplication([]);seen=[]
    d=SettingsDialog(InteractionSettings(),GestureLibrary(tmp_path).list_gestures(),lambda *args:seen.append(args))
    d.mapping_keys['swipe_left'].setText('command, delete_everything')
    d.save_button.click()
    assert not seen and d.error.text()
    d.close()


def test_atomic_mapping_validation_rejects_all_changes_on_bad_binding(tmp_path):
    from gesture_system.profiles import GestureLibrary
    import pytest
    library=GestureLibrary(tmp_path);before=library.list_gestures()
    with pytest.raises(ValueError):library.update_mappings({'point':{'action':'none','keys':[]},'swipe_left':{'action':'hotkey','keys':['invalid']}})
    assert library.list_gestures()==before


def test_main_window_persists_tuning_and_custom_master_remains_accessible(tmp_path):
    from gesture_system.gui import MainWindow,read_preferences
    from gesture_system.engine import Engine
    from gesture_system.profiles import GestureLibrary
    from gesture_system.interaction_settings import InteractionSettings
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'data'/'gesture_profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[])
    window.open_settings();d=window.settings_dialog
    d.controls['pointer_sensitivity'].setValue(1.6);d.save_button.click()
    assert read_preferences(tmp_path)['interaction']['pointer_sensitivity']==1.6
    assert window.engine.stable_controller.pointer_acquire==.1
    assert not window.enable_input.isChecked()
    window.open_settings();window.settings_dialog.open_custom()
    assert window.profile_selector.currentData()=='advanced' and window.record_group.isVisibleTo(window)
    window.close()


def test_missing_experimental_checkpoint_disarms_and_reports_error(tmp_path,monkeypatch):
    from gesture_system.gui import MainWindow
    from gesture_system.engine import Engine
    from gesture_system.profiles import GestureLibrary
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'data'/'gesture_profiles')
    def fail(self,root):raise OSError('missing checkpoint')
    monkeypatch.setattr(Engine,'load_temporal_predictor',fail)
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[])
    window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.experimental_neural.setChecked(True)
    assert not window.engine.experimental_neural and not window.experimental_neural.isChecked()
    assert 'missing checkpoint' in window.log.toPlainText()
    window.close()


def test_tuning_save_preserves_existing_advanced_binding(tmp_path):
    from gesture_system.settings_ui import SettingsDialog
    from gesture_system.interaction_settings import InteractionSettings
    from gesture_system.profiles import GestureLibrary
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path);seen=[]
    library.update_mapping('builtin:point','hotkey',['command','a'])
    library.update_mapping('builtin:pinch','drag_start')
    d=SettingsDialog(InteractionSettings(),library.list_gestures(),lambda settings,bindings:seen.append(bindings))
    d.controls['pointer_sensitivity'].setValue(1.4);d.save_button.click()
    assert seen[0]['point']=={'action':'hotkey','keys':['command','a']}
    assert seen[0]['pinch']['action']=='drag_start'
    d.close()
