import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from gesture_system.types import FrameFeatures
from gesture_system.pointing import FingerRay,CameraIntrinsics,PointingCalibration

@pytest.fixture
def app():return QApplication.instance() or QApplication([])

def feature(stamp,uv):
    W,H=.3442447,.2225239;origin=np.array([0,.1,.5]);direction=np.array([W*(.5-uv[0]),H*uv[1],0])-origin
    f=FrameFeatures(stamp,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99,np.arange(63).reshape(21,3)*.001,
                    image_points=np.ones((21,2))*.5,frame_size=(640,480),geometry_label='point',geometry_confidence=.98)
    f.ray=FingerRay(origin,direction,timestamp=stamp);return f

def context():return {'camera_index':0,'source_kind':'camera','frame_size':[640,480],
                       'intrinsics':CameraIntrinsics.from_fov(640,480).to_dict(),'display':{'id':1,'geometry':[0,0,1000,700]}}

def fill(dialog,uv,now=10.):
    for stamp in np.linspace(now-.49,now,7):dialog.observe(feature(float(stamp),uv))

def test_cancel_and_bad_fit_preserve_existing_calibration(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    screen=DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239),2.)
    path=tmp_path/'data'/'pointing_calibration.json';path.parent.mkdir();path.write_text('original')
    dialog=RayTargetDialog(tmp_path,screen,context(),clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    dialog.reject();assert path.read_text()=='original'
    dialog=RayTargetDialog(tmp_path,screen,context(),clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    for _ in range(9):fill(dialog,(.5,.5));dialog.capture_target()
    assert path.read_text()=='original' and not dialog.completed
    assert dialog.message.text()
    dialog.reject()

def test_calibration_and_heldout_targets_are_distinct_and_never_refit(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    screen=DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239),2.)
    dialog=RayTargetDialog(tmp_path,screen,context(),clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    calibration_targets=list(dialog.targets)
    for uv in calibration_targets:fill(dialog,uv);dialog.capture_target()
    assert dialog.completed
    dialog.commit_calibration()
    path=tmp_path/'data'/'pointing_calibration.json';before=path.read_bytes();model=PointingCalibration.load(path)
    trial=RayTargetDialog(tmp_path,screen,context(),validation_model=model,clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    assert len(trial.targets)==5 and not any(tuple(p) in calibration_targets for p in trial.targets)
    for uv in list(trial.targets):fill(trial,uv);trial.capture_target()
    assert trial.completed and path.read_bytes()==before
    reports=list((tmp_path/'data'/'pointing_trials').glob('*.json'));assert len(reports)==1
    result=json.loads(reports[0].read_text());assert result['kind']=='independent_target_validation'
    assert result['errors']['rmse_pixels']<1
    assert result['samples'][0]['raw'][0]['image_points']

def test_stale_and_insufficient_rays_reject_capture(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),
                           clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    for stamp in [8.,8.1,8.2,8.3,8.4,8.5]:dialog.observe(feature(stamp,(.5,.5)))
    dialog.capture_target();assert dialog.index==0
    dialog.reject()


def make_window(root):
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.pointing_ui import DisplayInfo
    class Actions:
        executed=[]
        def available(self):return True
        def current_position(self):return np.array([.5,.5])
        def execute(self,event):self.executed.append(event)
        def release(self):pass
    library=GestureLibrary(root/'profiles')
    return MainWindow(root,library,Engine(library,load_models=False),actions=Actions(),
                      screen_provider=lambda:[DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239),2.)])


def test_stable_default_requires_calibration_and_reload_preserves_it(tmp_path,app):
    window=make_window(tmp_path)
    assert window.profile_selector.currentData()=='stable' and window.engine.stable_profile
    window.enable_input.setChecked(True);assert not window.enable_input.isChecked()
    window._running=True;window._last_feature=feature(__import__('time').monotonic(),(.5,.5))
    calibration_context=window.pointing_context(window._last_feature)
    targets=[(x,y) for y in [.12,.5,.88] for x in [.12,.5,.88]]
    rays=[feature(1,target).ray for target in targets]
    model=PointingCalibration.fit(rays,np.asarray(targets),calibration_context)
    window.install_calibration(model)
    window.reload_engine()
    assert window.engine.pointing_calibration is model and window.engine.stable_profile
    window.threshold.setValue(.99);assert window.engine.conf_threshold==.75
    window.close()


def test_context_mismatch_and_wizard_block_commands(tmp_path,app):
    from gesture_system.types import ControlEvent
    window=make_window(tmp_path);window._running=True
    window._last_feature=feature(__import__('time').monotonic(),(.5,.5))
    targets=[(x,y) for y in [.12,.5,.88] for x in [.12,.5,.88]]
    model=PointingCalibration.fit([feature(1,t).ray for t in targets],np.asarray(targets),window.pointing_context(window._last_feature))
    window.install_calibration(model)
    window._last_feature.frame_size=(1280,720)
    window.enable_input.setChecked(True);assert not window.enable_input.isChecked()
    window._last_feature.frame_size=(640,480)
    window.open_pointing_calibration()
    assert window.pointing_dialog is not None
    window.enable_input.setChecked(True);window.dispatch_events([ControlEvent('pinch','click',1,{})])
    assert window.actions.executed==[] and not window.enable_input.isChecked()
    window.pointing_dialog.reject();window.close()


def test_rendered_target_uses_global_widget_offset_not_nominal_grid(tmp_path,app):
    from gesture_system.pointing_ui import DisplayInfo,RayTargetDialog
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context())
    dialog.setGeometry(100,50,800,500)
    target=dialog.target_screen_coordinates()
    local=dialog.target_local_point();global_point=dialog.mapToGlobal(local)
    assert np.allclose(target,[global_point.x()/999,global_point.y()/699])
    assert not np.allclose(target,dialog.targets[0])
    dialog.reject()


def test_gesture_surface_uses_qt_command_shortcuts_and_keeps_receipts_separate(tmp_path,app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from gesture_system.pointing_ui import GestureReceiptDialog,DisplayInfo
    dialog=GestureReceiptDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239),2.))
    dialog.show();app.processEvents()
    assert dialog.text.focusPolicy()==Qt.NoFocus
    QTest.keyClick(dialog,Qt.Key_Right,Qt.ControlModifier);app.processEvents()
    assert dialog.page==1 and dialog.receipts[-1]['shortcut']=='command+right'
    QTest.keyClick(dialog,Qt.Key_Left,Qt.ControlModifier);app.processEvents()
    assert dialog.page==0
    assert dialog.commands==[] and not dialog.enabled.isChecked()
    dialog.reject()
    data=json.loads(next((tmp_path/'data'/'pointing_trials').glob('gesture-receipts*.json')).read_text())
    assert data['engine_commands']==[] and data['automatic_success_claim'] is False


def test_session_preserves_raw_image_correspondences_and_geometry(tmp_path):
    from gesture_system.gui import SessionRecorder
    f=feature(1,(.5,.5));session=SessionRecorder(tmp_path,{'kind':'camera'},'two_stage',[])
    session.add_frame(f,'point',.98,'pointer',False,'two_stage')
    session.add_frame(FrameFeatures.absent(2),'no_hand',0,'recovery',False,'two_stage')
    path=session.finish('stop');data=np.load(path/'frames.npz',allow_pickle=False)
    assert data['image_points'].shape==(2,42) and data['image_present'].tolist()==[True,False]
    assert data['frame_size'].tolist()==[[640,480],[0,0]]
    assert data['geometry_labels'].tolist()==['point','']


def test_cancel_after_successful_fit_preserves_saved_model(tmp_path,app):
    from gesture_system.pointing_ui import DisplayInfo,RayTargetDialog
    path=tmp_path/'data'/'pointing_calibration.json';path.parent.mkdir();path.write_text('original')
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    for uv in list(dialog.targets):fill(dialog,uv);dialog.capture_target()
    assert dialog.completed and path.read_text()=='original'
    dialog.reject();assert path.read_text()=='original'


def test_custom_trial_restores_stable_profile_and_never_enables_input(tmp_path,app):
    window=make_window(tmp_path)
    ident=window.library.add('Своя поза','static',np.ones((8,48)),'click')
    window.refresh_library()
    row=next(i for i,g in enumerate(window._entries) if g['id']==ident)
    window.gestures.setCurrentRow(row);window._running=True
    window.begin_trial()
    assert not window.engine.stable_profile and window.engine.control_mode=='personal'
    assert not window.engine.experimental_neural and not window.enable_input.isChecked()
    window.cancel_recording()
    assert window.engine.stable_profile and not window.enable_input.isChecked()
    window.close()


def test_scene_mouse_wheel_receipts_and_deactivation_are_safe(tmp_path,app):
    from PySide6.QtCore import Qt,QEvent,QPoint,QPointF
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QPushButton
    from PySide6.QtTest import QTest
    from gesture_system.pointing_ui import GestureReceiptDialog,DisplayInfo
    dialog=GestureReceiptDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)))
    dialog.show();app.processEvents()
    QTest.mouseClick(dialog.findChild(QPushButton,'click-target-1'),Qt.LeftButton)
    wheel=QWheelEvent(QPointF(5,5),QPointF(25,25),QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
    QApplication.sendEvent(dialog.text.viewport(),wheel)
    assert {'mouse_press','wheel'}<=set(r['type'] for r in dialog.receipts)
    assert not dialog.commands
    dialog.enabled.setChecked(True)
    QApplication.sendEvent(dialog,QEvent(QEvent.WindowDeactivate))
    assert not dialog.enabled.isChecked()
    dialog.reject()


def test_wizard_requires_fresh_present_world_and_image_observations(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True
    f=feature(time.monotonic(),(.5,.5));f.world_points=None;window._last_feature=f
    window.open_pointing_calibration();assert window.pointing_dialog is None
    window._last_feature=FrameFeatures.absent(time.monotonic())
    window.open_pointing_calibration();assert window.pointing_dialog is None
    window.close()


def test_receipt_save_failure_still_disables_input_and_closes(tmp_path,app):
    from PySide6.QtWidgets import QDialog
    from gesture_system.pointing_ui import GestureReceiptDialog,DisplayInfo
    blocked=tmp_path/'data'/'pointing_trials';blocked.parent.mkdir();blocked.write_text('blocked')
    dialog=GestureReceiptDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)))
    closed=[];dialog.closed.connect(lambda:closed.append(True));dialog.enabled.setChecked(True)
    dialog.show();dialog.done(QDialog.Rejected)
    assert not dialog.enabled.isChecked() and closed==[True] and not dialog.isVisible()
    assert 'сохран' in dialog.status.text().lower()


def test_unknown_context_invalidates_wizard_buffer_and_capture(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;window._last_feature=feature(time.monotonic(),(.5,.5))
    window.open_pointing_calibration();dialog=window.pointing_dialog;dialog.ray_estimator=lambda f,k:f.ray
    def unavailable():raise RuntimeError('display unavailable')
    window.screen_provider=unavailable
    now=time.monotonic();dialog.clock=lambda:now
    for stamp in np.linspace(now-.49,now,7):window.process_observation(feature(float(stamp),(.12,.12)),float(stamp))
    dialog.capture_target()
    assert dialog.index==0 and not dialog.buffer and not dialog.capture_button.isEnabled()
    dialog.reject();window.close()


def test_failed_fit_space_is_terminal_and_records_numeric_attempt(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    path=tmp_path/'data'/'pointing_calibration.json';path.parent.mkdir();path.write_text('original')
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    for _ in range(9):fill(dialog,(.5,.5));dialog.capture_target()
    fill(dialog,(.5,.5));dialog.capture_target()
    assert dialog.index==9 and path.read_text()=='original' and not dialog.shortcut.isEnabled()
    data=json.loads(next((tmp_path/'reports'/'pointing-v3'/'attempts').glob('*.json')).read_text())
    assert data['outcome']=='fit_rejected' and data['error'] and len(data['samples'])==9
    assert data['samples'][0]['raw'][0]['image_points'] and data['context']==context()
    dialog.reject()


def test_cancel_attempt_has_raw_trace_without_replacing_old_model(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    path=tmp_path/'data'/'pointing_calibration.json';path.parent.mkdir();path.write_text('original')
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    fill(dialog,(.12,.12));dialog.capture_target();dialog.reject()
    data=json.loads(next((tmp_path/'reports'/'pointing-v3'/'attempts').glob('*.json')).read_text())
    assert data['outcome']=='cancelled' and len(data['samples'])==1 and path.read_text()=='original'


def test_cancel_trace_storage_failure_does_not_prevent_dialog_cleanup(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    blocked=tmp_path/'reports';blocked.write_text('blocked')
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context())
    errors=[];finished=[];dialog.report_failed.connect(errors.append);dialog.finished.connect(finished.append)
    dialog.show();dialog.reject()
    assert errors and finished==[0] and not dialog.isVisible()


def test_absent_frame_with_unknown_display_invalidates_previously_valid_rays(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;window._last_feature=feature(time.monotonic(),(.5,.5))
    window.open_pointing_calibration();dialog=window.pointing_dialog
    dialog.clock=lambda:10.;dialog.ray_estimator=lambda f,k:f.ray
    for stamp in np.linspace(9.51,9.99,7):window.process_observation(feature(float(stamp),(.12,.12)),float(stamp))
    assert len(dialog.buffer)==7
    def unavailable():raise RuntimeError('display unavailable')
    window.screen_provider=unavailable
    window.process_observation(FrameFeatures.absent(10.),10.)
    dialog.capture_target()
    assert dialog.index==0 and dialog.invalidated and not dialog.buffer and not dialog.capture_button.isEnabled()
    dialog.reject();window.close()


def test_known_context_absence_remains_invalid_coverage_observation(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;window._last_feature=feature(time.monotonic(),(.5,.5))
    window.open_pointing_calibration();dialog=window.pointing_dialog
    dialog.clock=lambda:10.;dialog.ray_estimator=lambda f,k:f.ray
    for stamp in np.linspace(9.51,9.95,7):window.process_observation(feature(float(stamp),(.12,.12)),float(stamp))
    window.process_observation(FrameFeatures.absent(10.),10.)
    assert not dialog.invalidated and len(dialog.buffer)==8 and dialog.buffer[-1][1] is None
    dialog.capture_target();assert dialog.index==1
    dialog.reject();window.close()


def test_calibration_uses_valid_index_ray_despite_uncertain_other_fingers(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),
                           clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    for stamp in np.linspace(9.51,10,7):
        f=feature(float(stamp),(.12,.12));f.geometry_label='uncertain';f.geometry_confidence=0
        dialog.observe(f)
    dialog.capture_target()
    assert dialog.index==1
    dialog.reject()


def test_calibration_displays_actual_pose_rejection_before_capture(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),clock=lambda:10)
    f=feature(10,(.12,.12));f.world_points=np.zeros((21,3));f.geometry_label='uncertain'
    dialog.observe(f)
    assert 'вырожденная геометрия кисти' in dialog.status.text()
    dialog.capture_target()
    assert dialog.index==0 and 'вырожденная геометрия кисти' in dialog.message.text()
    dialog.reject()


def test_space_starts_countdown_then_collects_without_another_key(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    clock=[10.]
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),
                           clock=lambda:clock[0],ray_estimator=lambda f,k:f.ray)
    dialog.begin_collection()
    assert dialog.index==0 and dialog.collecting
    for t in np.linspace(10,13.6,100):
        clock[0]=float(t);dialog.observe(feature(float(t),(.12,.12)))
    assert dialog.index==1 and not dialog.collecting
    assert dialog.samples[0]['quality']['valid_frames']>=12
    dialog.reject()


def test_failed_fit_allows_retaking_one_target_preserving_other_eight(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    for _ in range(9):fill(dialog,(.5,.5));dialog.capture_target()
    before=[s.copy() for s in dialog.samples]
    assert not dialog.invalidated and dialog.retry_selector.isVisibleTo(dialog)
    dialog.select_retry_target(4)
    fill(dialog,(.5,.5));dialog.capture_target()
    assert len(dialog.samples)==9
    assert all(dialog.samples[i]==before[i] for i in range(9) if i!=4)
    dialog.reject()


def test_timed_capture_rechecks_display_when_no_new_frame_arrives(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;window._last_feature=feature(time.monotonic(),(.5,.5))
    window.open_pointing_calibration();dialog=window.pointing_dialog
    clock=[10.];dialog.clock=lambda:clock[0];dialog.ray_estimator=lambda f,k:f.ray
    dialog.begin_collection()
    for t in np.linspace(11.2,13.35,45):
        clock[0]=float(t);window.process_observation(feature(float(t),(.12,.12)),float(t))
    def unavailable():raise RuntimeError('display unavailable')
    window.screen_provider=unavailable;clock[0]=13.41;dialog.collection_tick()
    assert dialog.index==0 and dialog.invalidated
    dialog.reject();window.close()


def test_lower_targets_are_inside_comfortable_area_and_can_move_before_capture(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),clock=lambda:10)
    assert max(v for _,v in dialog.targets)<=.72
    dialog.index=7;old=dialog.target_screen_coordinates().copy();dialog.move_target(0,-.06)
    assert dialog.target_screen_coordinates()[1]<old[1]
    dialog.begin_collection();position=dialog.target_screen_coordinates().copy();dialog.move_target(0,-.06)
    assert np.allclose(dialog.target_screen_coordinates(),position)
    dialog.reject()


def test_resume_is_explicit_and_preserves_seven_captured_targets(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    samples=[]
    for uv in [(x,y) for y in [.12,.5,.88] for x in [.12,.5,.88]][:7]:
        f=feature(1,uv);samples.append({'target':list(uv),'target_local_requested':list(uv),
            'origin':f.ray.origin.tolist(),'direction':f.ray.direction.tolist(),'quality':{},'raw':[]})
    attempt={'kind':'pointing_calibration_attempt','context':context(),'samples':samples,'source':'example.json'}
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),resume_attempt=attempt)
    assert dialog.index==0 and not dialog.samples
    dialog.resume_previous()
    assert dialog.index==7 and dialog.samples==samples and len(dialog.rays)==7
    assert dialog.targets[7][1]<=.72 and not dialog.collecting
    dialog.reject()


def test_resume_mismatched_camera_context_is_rejected(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    attempt={'kind':'pointing_calibration_attempt','context':{'wrong':True},'samples':[{}]*7}
    dialog=RayTargetDialog(tmp_path,DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239)),context(),resume_attempt=attempt)
    dialog.resume_previous();assert dialog.index==0 and not dialog.samples
    dialog.reject()


def test_calibration_has_live_camera_preview(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;f=feature(time.monotonic(),(.5,.5));window._last_feature=f
    window.open_pointing_calibration();dialog=window.pointing_dialog
    window.consume_frame(np.zeros((480,640,3),dtype=np.uint8),f,f.timestamp,annotated=True)
    assert dialog.camera_preview.pixmap() is not None and not dialog.camera_preview.pixmap().isNull()
    dialog.reject();window.close()


def test_independent_validation_targets_cannot_be_moved(tmp_path,app):
    from gesture_system.pointing_ui import RayTargetDialog,DisplayInfo
    screen=DisplayInfo(1,'Test',(0,0,1000,700),(.3442447,.2225239))
    training=RayTargetDialog(tmp_path,screen,context(),clock=lambda:10,ray_estimator=lambda f,k:f.ray)
    for uv in list(training.targets):fill(training,uv);training.capture_target()
    assert training.completed
    trial=RayTargetDialog(tmp_path,screen,context(),validation_model=training.model)
    before=list(trial.targets);trial.index=2
    for _ in range(10):trial.move_target(0,-.06)
    assert trial.targets==before and not trial.lift_button.isEnabled() and not trial.lift_shortcut.isEnabled()
    trial.reject();training.reject()


def test_explicit_workspace_mode_is_usable_without_world_or_ray_fit(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True
    f=feature(time.monotonic(),(.5,.5));f.world_points=None;window._last_feature=f
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'))
    window.process_observation(f,f.timestamp)
    ok,reason=window.pointing_ready()
    assert ok,reason
    assert window.pointing_calibration is None
    assert window.engine.pointing_calibration.kind=='camera_workspace_2d'
    assert window.engine.stable_controller.prefer_image_geometry
    assert not window.enable_input.isChecked() and not window.target_test_button.isEnabled()
    window.close()


def test_workspace_editor_disables_os_and_saves_separate_configuration(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;window._last_feature=feature(time.monotonic(),(.5,.5))
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'))
    window.open_pointing_calibration();editor=window.workspace_dialog
    assert editor is not None and window.pointing_dialog is None
    window.enable_input.setChecked(True);assert not window.enable_input.isChecked()
    editor.width_slider.setValue(40);editor.save_configuration()
    assert (tmp_path/'data'/'workspace_config.json').exists()
    assert not (tmp_path/'data'/'pointing_calibration.json').exists()
    assert window.workspace_dialog is None and window.workspace_mapper.bounds.span[0]==.4
    window.reload_engine();assert window.engine.pointing_calibration is window.workspace_mapper
    window.close()


def test_workspace_mode_context_failure_stops_input(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;f=feature(time.monotonic(),(.5,.5));window._last_feature=f
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'))
    window.enable_input.setChecked(True);assert window.enable_input.isChecked()
    def missing():raise RuntimeError('missing display')
    window.screen_provider=missing
    window.process_observation(f,f.timestamp)
    assert not window.enable_input.isChecked() and window.engine.pointing_calibration is None
    window.close()


def test_workspace_editor_save_checks_current_camera_context(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;window._last_feature=feature(time.monotonic(),(.5,.5))
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'));window.open_pointing_calibration()
    editor=window.workspace_dialog
    window.source.setCurrentIndex(1);editor.save_configuration()
    assert not (tmp_path/'data'/'workspace_config.json').exists()
    assert not window.enable_input.isChecked()
    editor.reject();window.close()


def test_workspace_brief_absence_keeps_confirmed_context_and_input_armed(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;f=feature(time.monotonic(),(.5,.5));window._last_feature=f
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'));mapper=window.workspace_mapper
    window.enable_input.setChecked(True);assert window.enable_input.isChecked()
    window.process_observation(FrameFeatures.absent(f.timestamp+.01),f.timestamp+.01,telemetry={'replayed_absence':True})
    assert window.workspace_mapper is mapper and window.enable_input.isChecked()
    assert window.actions.executed==[]
    window.close()


def test_pointer_mode_persists_without_auto_enabling_input(tmp_path,app):
    window=make_window(tmp_path)
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'));window.close()
    second=make_window(tmp_path)
    assert second.pointer_mode.currentData()=='workspace' and not second.enable_input.isChecked()
    second.close()


def test_ray_wizard_locks_mode_selection_until_closed(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;window._last_feature=feature(time.monotonic(),(.5,.5))
    window.open_pointing_calibration()
    assert window.pointing_dialog is not None and not window.pointer_mode.isEnabled()
    window.pointing_dialog.reject();assert window.pointer_mode.isEnabled()
    window.close()


def test_workspace_video_and_context_failures_remain_in_raw_trace(tmp_path,app):
    import time
    from gesture_system.gui import SessionRecorder
    for video in [True,False]:
        window=make_window(tmp_path);window._running=True;f=feature(time.monotonic(),(.5,.5));window._last_feature=f
        window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'))
        window.session=SessionRecorder(tmp_path/'sessions',{'kind':'video' if video else 'camera'},'two_stage',[])
        if video:window.video_path='example.mp4'
        else:
            def missing():raise RuntimeError('missing display')
            window.screen_provider=missing
        window.process_observation(f,f.timestamp)
        assert len(window.session.frames)==1 and not window.enable_input.isChecked()
        assert window.session.frames[0]['task']['mode']=='workspace_context_error'
        window.close()


def test_effective_geometry_source_distinguishes_advanced_profile(tmp_path,app):
    window=make_window(tmp_path)
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'))
    assert window.effective_geometry_source()=='image_aspect_corrected'
    window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    assert window.effective_geometry_source()=='legacy_or_personal'
    window.close()


def test_saved_frame_trace_keeps_pointer_mode_geometry_source_and_error(tmp_path):
    from gesture_system.gui import SessionRecorder
    session=SessionRecorder(tmp_path,{'kind':'video'},'two_stage',[])
    session.add_frame(feature(1,(.5,.5)),'workspace_unavailable',0,'workspace_context_error',False,'two_stage',
                      task={'mode':'workspace_context_error','pointer_mode':'workspace','controller_geometry_source':'image_aspect_corrected','error':'camera source unavailable'})
    path=session.finish('stop');data=np.load(path/'frames.npz',allow_pickle=False)
    assert data['task_pointer_mode'].tolist()==['workspace']
    assert data['task_controller_geometry_source'].tolist()==['image_aspect_corrected']
    assert data['task_error'].tolist()==['camera source unavailable']


def test_workspace_recenter_really_centers_hand_near_image_edge(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;f=feature(time.monotonic(),(.5,.5));f.image_points[:,0]=.2;window._last_feature=f
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'));window.open_pointing_calibration();editor=window.workspace_dialog
    editor.relative_checkbox.setChecked(False)
    editor.observe(f);editor.recenter()
    assert np.allclose(editor.mapper.map_feature(f),[.5,.5])
    assert editor.mapper.bounds.span[0]<=.4
    editor.reject();window.close()


def test_relative_workspace_readiness_is_pure_and_os_callback_lifecycle(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;window._last_feature=feature(time.monotonic(),(.5,.5))
    window.actions.current_position=lambda:np.array([.2,.7])
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'))
    mapper=window.workspace_mapper
    assert mapper.relative and mapper.bounds.span==(.35,.35)
    assert window.pointing_ready()[0] and mapper._reference_hand is None
    c=window.engine.stable_controller;mapper.begin_pointer(window._last_feature);c.state='pointer'
    assert c.position_provider is None
    window.enable_input.setChecked(True)
    assert c.state=='recovery' and c.position_provider is not None
    assert np.allclose(c.position_provider(),[.2,.7])
    window.enable_input.setChecked(False);assert c.position_provider is None
    window.enable_input.setChecked(True);window.release_control();assert c.position_provider is None
    window.enable_input.setChecked(True);window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('ray'))
    assert c.position_provider is None and not window.enable_input.isChecked()
    window.close()


def test_workspace_receipt_window_opens_without_hand_but_cannot_enable(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;window._last_feature=feature(time.monotonic(),(.5,.5))
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'))
    window._last_feature=FrameFeatures.absent(time.monotonic());window._last_feature.frame_size=(640,480)
    window.open_gesture_receipts();dialog=window.receipt_dialog
    assert dialog is not None and not dialog.enabled.isChecked()
    dialog.enabled.setChecked(True)
    assert not dialog.enabled.isChecked() and not window.enable_input.isChecked()
    dialog.reject();window.close()


def test_existing_v1_workspace_remains_absolute_after_reload_and_ensure(tmp_path,app):
    import time
    window=make_window(tmp_path);window._running=True;f=feature(time.monotonic(),(.5,.5));window._last_feature=f
    known=window.workspace_context(f);window.close()
    path=tmp_path/'data'/'workspace_config.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'version':1,'kind':'camera_workspace_2d','center':[.5,.5],'span':[.5,.5],'context':known}))
    window=make_window(tmp_path);window._running=True;window._last_feature=f
    window.pointer_mode.setCurrentIndex(window.pointer_mode.findData('workspace'))
    assert not window.workspace_mapper.relative and window.workspace_mapper.bounds.span==(.5,.5)
    window.close()
