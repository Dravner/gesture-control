"""UI contracts: no camera or Accessibility access is needed."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from types import SimpleNamespace
import numpy as np

def test_stable_diagnostics_show_actual_pinch_and_uncertain_finger(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app=QApplication.instance() or QApplication([])
    library=GestureLibrary(tmp_path)
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[])
    window.engine.stable_controller.geometry=SimpleNamespace(pinch=.23,finger_states=('open','folded','uncertain','folded'))
    window.update_personal_diagnostics()
    text=window.diagnostic_label.text()
    assert 'Щипок 0.23' in text and 'безымянный: неясно' in text
    window.close()

def test_receipt_scene_shows_live_gesture_feedback(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.pointing_ui import GestureReceiptDialog,DisplayInfo
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path)
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[])
    dialog=GestureReceiptDialog(tmp_path,DisplayInfo(1,'test',(0,0,640,480),(.3,.2)));window.receipt_dialog=dialog
    window.engine.stable_controller.geometry=SimpleNamespace(pinch=.23,finger_states=('open','folded','uncertain','folded'))
    window.update_personal_diagnostics()
    assert 'Щипок 0.23' in dialog.tracking_status.text()
    assert 'ожидание спокойного наведения' in dialog.tracking_status.text()
    window.receipt_dialog=None;dialog.reject();window.close()

def test_native_backend_change_stops_input_selects_2d_and_persists(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow,read_preferences
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    import gesture_system.tracking as tracking
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(tracking,'apple_vision_available',lambda:True)
    class Actions:
        def release(self):pass
        def available(self):return True
        def execute(self,event):raise AssertionError('No OS input during backend change')
    class Runtime:
        def __init__(self):self.stops=0
        def stop(self,**kwargs):self.stops+=1;return True
    runtime=Runtime();library=GestureLibrary(tmp_path)
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),actions=Actions(),screen_provider=lambda:[])
    window.live_runtime=runtime;window._running=True
    window.tracking_backend.setCurrentIndex(window.tracking_backend.findData('apple_vision'))
    assert runtime.stops and not window._running and not window.enable_input.isChecked()
    assert window.live_runtime is None and window.pointer_mode.currentData()=='workspace'
    assert read_preferences(tmp_path)['tracker_backend']=='apple_vision'
    window.close()

def test_selected_backend_factory_and_context_are_used_for_live_camera(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.pointing_ui import DisplayInfo
    from gesture_system.types import FrameFeatures
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    import gesture_system.tracking as tracking
    app=QApplication.instance() or QApplication([]);seen=[];sentinel=object()
    monkeypatch.setattr(tracking,'apple_vision_available',lambda:True)
    monkeypatch.setattr(tracking,'make_tracker',lambda root,backend:seen.append(backend) or sentinel)
    class Runtime:
        def __init__(self,*args,**kwargs):self.factory=kwargs['tracker_factory']
        def start(self):assert self.factory() is sentinel;return 1
        def stop(self,**kwargs):return True
    library=GestureLibrary(tmp_path)
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),runtime_factory=Runtime,screen_provider=lambda:[DisplayInfo(1,'test',(0,0,640,480),(.3,.2))])
    window.tracking_backend.setCurrentIndex(window.tracking_backend.findData('apple_vision'));window.start_stream()
    assert seen==['apple_vision']
    f=FrameFeatures.absent(1);f.frame_size=(640,480)
    assert window.workspace_context(f)['tracker_backend']=='apple_vision'
    window.stop_stream();window.close()

def test_explicit_startup_backend_overrides_preferences_without_starting_input(tmp_path,monkeypatch):
    import json
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    import gesture_system.tracking as tracking
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(tracking,'apple_vision_available',lambda:True)
    path=tmp_path/'data/ui_settings.json';path.parent.mkdir();original=json.dumps({'tracker_backend':'mediapipe','pointer_mode':'ray'});path.write_text(original)
    library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[],tracker_backend='apple_vision')
    assert window.tracking_backend.currentData()=='apple_vision' and window.pointer_mode.currentData()=='workspace'
    assert not window._running and not window.enable_input.isChecked()
    assert path.read_text()==original
    window.close()


def test_select_custom_profile_prefills_enrollment_without_renaming(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app = QApplication.instance() or QApplication([])
    library = GestureLibrary(tmp_path)
    target = library.add('сall', 'static', np.ones((32, 48)), 'none')
    window = MainWindow(tmp_path, library, Engine(library, load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    row = next(i for i, entry in enumerate(window._entries) if entry['id'] == target)
    window.gestures.setCurrentRow(row)
    assert window.record_name.text() == 'сall'
    assert window.record_kind.currentData() == 'static'
    window.close()


def test_recording_excludes_absent_frames_and_completes_automatically():
    from gesture_system.gui import RecordingSession
    session = RecordingSession('dynamic', started=10, countdown=3)
    feature = SimpleNamespace(present=True, vector=lambda: np.ones(48))
    assert not session.feed(feature, 12)
    assert len(session.samples) == 0
    assert not session.feed(feature, 13)
    session.feed(SimpleNamespace(present=False), 14)
    assert len(session.samples) == 1
    assert session.feed(feature, 16)
    assert np.asarray(session.samples).shape == (2, 48)


def test_video_and_recording_never_execute_os_events(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app = QApplication.instance() or QApplication([])
    class Actions:
        executed = []
        releases = 0
        def available(self): return True
        def execute(self, event): self.executed.append(event)
        def release(self): self.releases += 1
    actions = Actions()
    library = GestureLibrary(tmp_path)
    window = MainWindow(tmp_path, library, Engine(library), actions=actions,screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    event = SimpleNamespace(gesture='pinch', action='click', timestamp=1, payload={})
    window.dispatch_events([event])
    assert actions.executed == []
    window.video_path = 'clip.mp4'
    window.enable_input.setChecked(True)
    window.dispatch_events([event])
    assert actions.executed == []
    window.video_path = None
    window.recording = SimpleNamespace()
    window.dispatch_events([event])
    assert actions.executed == []
    window.recording = None
    window.enable_input.setChecked(True)
    window.dispatch_events([event])
    assert actions.executed == [event]
    window.pause_stream()
    assert not window.enable_input.isChecked()
    assert actions.releases >= 1
    window.close()
    app.processEvents()


def test_frame_failure_releases_capture_and_cancels_recording(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow, RecordingSession
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app = QApplication.instance() or QApplication([])
    class Capture:
        released = False
        def read(self): return False, None
        def release(self): self.released = True
    library = GestureLibrary(tmp_path)
    window = MainWindow(tmp_path, library, Engine(library),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    capture = Capture()
    window.capture = capture
    window.recording = RecordingSession('static', 1)
    window.tick()
    assert capture.released
    assert window.capture is None
    assert window.recording is None
    assert not window.timer.isActive()
    assert 'источник' in window.log.toPlainText()
    window.close()
    app.processEvents()


def test_recording_persists_custom_name_and_mapping(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow, RecordingSession
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app = QApplication.instance() or QApplication([])
    library = GestureLibrary(tmp_path)
    window = MainWindow(tmp_path, library, Engine(library),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window._running = True
    window.record_name.setText('Мой жест сохранения')
    window.mapping.setCurrentIndex(window.mapping.findData('hotkey'))
    window.keys.setText('command, shift, s')
    window.begin_recording()
    assert window.recording is not None
    assert not window.enable_input.isEnabled()
    feature = SimpleNamespace(present=True, vector=lambda: np.ones(48))
    started = window.recording.started
    for index in range(8):
        window.recording.feed(feature, started + 3 + index * .1)
    window.finish_recording()
    entry = next(e for e in GestureLibrary(tmp_path).list_gestures() if e['name'] == 'Мой жест сохранения')
    assert entry['action'] == 'hotkey'
    assert entry['keys'] == ['command', 'shift', 's']
    assert not window.enable_input.isChecked()
    window.close()
    app.processEvents()


def test_additional_example_extends_selected_gesture_without_renaming(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app = QApplication.instance() or QApplication([])
    library = GestureLibrary(tmp_path)
    ident = library.add('Мой жест', 'static', np.ones((8, 48)), 'click')
    window = MainWindow(tmp_path, library, Engine(library),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window._running = True
    row = next(i for i, entry in enumerate(window._entries) if entry['id'] == ident)
    window.gestures.setCurrentRow(row)
    assert hasattr(window, 'begin_example')
    window.begin_example()
    feature = SimpleNamespace(present=True, vector=lambda: np.ones(48) * .5)
    for i in range(8):
        window.recording.feed(feature, window.recording.started + 3 + i * .1)
    window.finish_recording()
    entry = library.get(ident)
    assert entry['name'] == 'Мой жест'
    assert len(entry['templates']) == 2
    assert entry['action'] == 'click'
    window.close()
    app.processEvents()


def test_missing_hand_releases_any_os_hold(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import FrameFeatures
    app = QApplication.instance() or QApplication([])
    class Actions:
        releases = 0
        def release(self): self.releases += 1
        def available(self): return True
        def execute(self, event): pass
    class Capture:
        def read(self): return True, np.zeros((40, 40, 3), dtype=np.uint8)
        def release(self): pass
    class Tracker:
        def process(self, frame, now): return FrameFeatures.absent(now)
        def annotate(self, frame, feature): return frame
        def close(self): pass
    actions = Actions()
    library = GestureLibrary(tmp_path)
    window = MainWindow(tmp_path, library, Engine(library), tracker=Tracker(), actions=actions,screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.capture = Capture()
    before = actions.releases
    window.tick()
    assert actions.releases > before
    window.close()
    app.processEvents()


def test_pause_gesture_disables_os_input_immediately(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app = QApplication.instance() or QApplication([])
    class Actions:
        def available(self): return True
        def execute(self, event): pass
        def release(self): pass
    library = GestureLibrary(tmp_path)
    window = MainWindow(tmp_path, library, Engine(library), actions=Actions(),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.enable_input.setChecked(True)
    window.dispatch_events([SimpleNamespace(gesture='fist', action='pause', timestamp=1, payload={})])
    assert not window.enable_input.isChecked()
    window.close()
    app.processEvents()


def test_session_persistence_keeps_missing_frames_and_event_status(tmp_path):
    import json
    from gesture_system.gui import SessionRecorder
    from gesture_system.types import FrameFeatures, ControlEvent
    session = SessionRecorder(tmp_path, source={'kind': 'camera', 'index': 0}, method='two_stage', profiles=[])
    feature = FrameFeatures(10, True, np.ones(42), (.2,.3), (.4,.5), .6,.2,'point',.9)
    session.add_frame(feature, prediction='point', confidence=.91, phase='active', recording=False, method='two_stage')
    session.add_frame(FrameFeatures.absent(11), prediction='no_hand', confidence=0, phase='idle', recording=True, method='two_stage')
    session.add_event(ControlEvent('point','move',10,{'x':.6,'y':.5}), executed=False)
    path = session.finish('pause')
    data = np.load(path / 'frames.npz', allow_pickle=False)
    assert data['vectors'].shape == (2,48)
    assert data['present'].tolist() == [True,False]
    assert data['timestamps'].tolist() == [10,11]
    assert data['recording'].tolist() == [False,True]
    assert data['predictions'].tolist() == ['point','no_hand']
    metadata = json.loads((path / 'session.json').read_text())
    assert metadata['events'][0]['action'] == 'move'
    assert metadata['events'][0]['executed'] is False
    assert metadata['source'] == {'kind':'camera','index':0}
    assert metadata['hardware']['machine']
    assert metadata['end_reason'] == 'pause'
    assert sorted(p.name for p in path.iterdir()) == ['frames.npz','session.json']


def test_log_throttles_motion_without_dropping_os_execution(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import ControlEvent
    app = QApplication.instance() or QApplication([])
    class Actions:
        executed=[]
        def available(self): return True
        def execute(self,event): self.executed.append(event)
        def release(self): pass
    library = GestureLibrary(tmp_path)
    actions = Actions()
    window = MainWindow(tmp_path,library,Engine(library),actions=actions,screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.enable_input.setChecked(True)
    events=[ControlEvent('point','move',float(t),{'x':.5,'y':.4}) for t in np.arange(0,1,.025)]
    window.dispatch_events(events)
    assert len(actions.executed) == 40
    assert 4 <= window.log.toPlainText().count('→ move') <= 5
    window.dispatch_events([ControlEvent('pinch','click',.99,{})])
    assert '→ click' in window.log.toPlainText()
    window.close()
    app.processEvents()


def test_gui_uses_selected_method_and_invalid_selection_falls_back(tmp_path):
    import json
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app = QApplication.instance() or QApplication([])
    (tmp_path/'models').mkdir()
    (tmp_path/'models'/'selection.json').write_text(json.dumps({'method':'window','basis':'validation only'}))
    library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    assert window.method.currentData() == 'window'
    assert window.engine.method == 'window'
    window.close()
    (tmp_path/'models'/'selection.json').write_text(json.dumps({'method':'unknown'}))
    # This fallback case has no explicit saved user choice.
    (tmp_path/'data'/'ui_settings.json').unlink()
    window=MainWindow(tmp_path,library,Engine(library),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    assert window.method.currentData() == 'two_stage'
    window.close()
    app.processEvents()


def test_stream_pause_saves_actual_frame_and_command_trace(tmp_path):
    import json
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import FrameFeatures
    app=QApplication.instance() or QApplication([])
    from gesture_system.live_runtime import FramePacket,FeatureSnapshot
    import time
    class Runtime:
        error=None
        calls=0
        def start(self):return 1
        def take_latest(self):
            self.calls+=1
            now=time.monotonic()
            feature=FrameFeatures(now,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99) if self.calls==1 else FrameFeatures.absent(now)
            return FramePacket(1,self.calls,now,now,np.zeros((40,40,3),np.uint8),FeatureSnapshot.copy(feature),{})
        def stop(self,**kwargs):return True
    library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),runtime_factory=lambda *args,**kwargs:Runtime(),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.start_stream()
    window.tick()
    window.tick()
    window.pause_stream()
    sessions=list((tmp_path/'data'/'sessions').iterdir())
    assert len(sessions)==1
    data=np.load(sessions[0]/'frames.npz',allow_pickle=False)
    assert data['present'].tolist()==[True,False]
    assert data['labels'].tolist()==['point','no_hand']
    metadata=json.loads((sessions[0]/'session.json').read_text())
    assert any(event['action']=='move' and not event['executed'] for event in metadata['events'])
    assert metadata['end_reason']=='pause'
    window.close()
    assert len(list((tmp_path/'data'/'sessions').iterdir()))==1
    app.processEvents()


def test_gui_preferences_survive_restart(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app=QApplication.instance() or QApplication([])
    library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.threshold.setValue(.85)
    window.hold_time.setValue(.35)
    window.method.setCurrentIndex(window.method.findData('window'))
    window.close()
    restarted=MainWindow(tmp_path,library,Engine(library),screen_provider=lambda:[]);restarted.profile_selector.setCurrentIndex(restarted.profile_selector.findData('advanced'))
    assert restarted.engine.conf_threshold==.85
    assert restarted.engine.hold_seconds==.35
    assert restarted.engine.method=='window'
    assert not restarted.enable_input.isChecked()
    restarted.close()
    app.processEvents()

def test_observation_pause_candidate_does_not_erase_custom_motion_buffer(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import ControlEvent
    app=QApplication.instance() or QApplication([])
    library=GestureLibrary(tmp_path);engine=Engine(library,load_models=False)
    window=MainWindow(tmp_path,library,engine,screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.engine._custom_history.append((0.,np.zeros(48)))
    window.dispatch_events([ControlEvent('fist','pause',.2,{})])
    assert len(window.engine._custom_history)==1
    window.close();app.processEvents()

def test_custom_tolerance_persisted_independently_of_neural_threshold(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app=QApplication.instance() or QApplication([])
    library=GestureLibrary(tmp_path);key=library.add('phone','static',np.zeros((8,48)),'none')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    row=next(i for i,g in enumerate(window._entries) if g['id']==key)
    window.gestures.setCurrentRow(row)
    window.tolerance.setValue(.12);window.save_mapping()
    assert np.isclose(library.get(key)['threshold'],.24)
    assert np.isclose(window.engine.conf_threshold,.75)
    assert 'phone' in window.gestures.item(row).text()
    window.close();app.processEvents()


def test_recording_world_alignment_keeps_holes_without_reindexing():
    from gesture_system.gui import RecordingSession
    from gesture_system.types import FrameFeatures
    session=RecordingSession('static',0,countdown=0)
    for i,world in enumerate([np.ones((21,3)),None,np.ones((21,3))*2]):
        feature=FrameFeatures(float(i)/10,True,np.ones(42),(.5,.5),(.5,.4),1,.2,'point',.99,world)
        session.feed(feature,float(i)/10)
    assert len(session.samples)==len(session.world_samples)==3
    assert session.world_samples[1] is None
    assert session.world_array() is None
    assert session.quality()['world_coverage']==2/3


def test_sessions_preserve_world_presence_and_fixed_width(tmp_path):
    from gesture_system.gui import SessionRecorder
    from gesture_system.types import FrameFeatures
    session=SessionRecorder(tmp_path,source={'kind':'camera'},method='two_stage',profiles=[])
    feature=FrameFeatures(1,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99,np.arange(63).reshape(21,3))
    session.add_frame(feature,'point',.99,'active',False,'two_stage')
    session.add_frame(FrameFeatures.absent(2),'no_hand',0,'idle',False,'two_stage')
    path=session.finish('stop');data=np.load(path/'frames.npz',allow_pickle=False)
    assert data['world_points'].shape==(2,63)
    assert data['world_present'].tolist()==[True,False]
    assert data['world_points'][0,-1]==62
    assert np.isnan(data['world_points'][1]).all()


def test_guided_enrollment_adds_three_examples_and_keeps_mapping(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import FrameFeatures
    app=QApplication.instance() or QApplication([])
    library=GestureLibrary(tmp_path/'profiles');window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window._running=True;window.record_name.setText('Новый жест');window.mapping.setCurrentIndex(window.mapping.findData('click'))
    window.begin_guided_enrollment()
    for example in range(3):
        assert not window.enable_input.isEnabled()
        start=window.recording.started+window.recording.countdown
        for i in range(8):
            feature=FrameFeatures(start+i*.1,True,np.ones(42),(.5,.5),(.5,.4),1,.2,'point',.99,np.ones((21,3)))
            window.recording.feed(feature,feature.timestamp)
        window.finish_recording()
    entries=[e for e in library.list_gestures() if e['name']=='Новый жест']
    assert len(entries)==1 and len(entries[0]['templates'])==3
    assert len(entries[0]['world_templates'])==3
    assert entries[0]['action']=='click'
    assert window.protocol is None and not window.enable_input.isChecked()
    window.close();app.processEvents()


def test_trial_records_annotations_and_detections_without_training(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow,SessionRecorder
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import FrameFeatures,ControlEvent
    import json
    app=QApplication.instance() or QApplication([])
    library=GestureLibrary(tmp_path/'profiles');ident=library.add('Проверяемый','static',np.ones((8,48)),'click')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'));window._running=True
    window.session=SessionRecorder(tmp_path/'sessions',source={'kind':'camera'},method='two_stage',profiles=library.list_gestures())
    window.gestures.setCurrentRow(next(i for i,e in enumerate(window._entries) if e['id']==ident))
    before=library.path.read_bytes();window.begin_trial()
    assert window.engine.control_mode=='personal'
    for example in range(3):
        start=window.recording.started+window.recording.countdown
        for i in range(8):
            feature=FrameFeatures(start+i*.1,True,np.ones(42),(.5,.5),(.5,.4),1,.2,'point',.99)
            window.recording.feed(feature,feature.timestamp)
        window.protocol_events([ControlEvent(ident,'click',start+.4,{})])
        window.protocol.last_time=start+window.recording.duration+window.protocol.grace
        window.finish_recording()
    assert library.path.read_bytes()==before
    assert window.protocol is None and not window.enable_input.isChecked()
    path=window.session.finish('stop');metadata=json.loads((path/'session.json').read_text())
    trials=[t for t in metadata['tasks'] if t['mode']=='trial']
    assert len(trials)==3 and all(t['expected_id']==ident and t['matched_count']==1 for t in trials)
    assert all(t['countdown_start']<t['start']<=t['end'] for t in trials)
    window.session=None;window.close();app.processEvents()


def test_control_modes_and_experimental_neural_are_explicit_and_persist(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    monkeypatch.setattr(Engine,'load_temporal_predictor',lambda self,root:setattr(self,'_predictor',object()))
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    assert not window.experimental_neural.isChecked()
    window.control_mode.setCurrentIndex(window.control_mode.findData('pointer'))
    window.experimental_neural.setChecked(True)
    assert window.engine.control_mode=='pointer' and window.engine.experimental_neural
    window.close()
    reopened=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);reopened.profile_selector.setCurrentIndex(reopened.profile_selector.findData('advanced'))
    assert reopened.control_mode.currentData()=='pointer' and reopened.experimental_neural.isChecked()
    assert not reopened.enable_input.isChecked()
    reopened.close();app.processEvents()


def test_diagnostics_show_rejection_distance_and_progress(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.engine.personal_phase='rejected'
    window.engine.personal_result={'gesture_id':None,'distance':.12,'radius':.07,'reason':'неоднозначный жест','progress':.5}
    window.update_personal_diagnostics()
    text=window.diagnostic_label.text()
    assert '0.120' in text and '0.070' in text and '50%' in text and 'неоднозначный жест' in text
    window.close();app.processEvents()


def test_cancelling_trial_restores_mode_and_keeps_os_input_off(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    monkeypatch.setattr(Engine,'load_temporal_predictor',lambda self,root:setattr(self,'_predictor',object()))
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    ident=library.add('Тест','static',np.ones((8,48)),'click')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'));window._running=True
    window.control_mode.setCurrentIndex(window.control_mode.findData('pointer'))
    window.experimental_neural.setChecked(True)
    window.gestures.setCurrentRow(next(i for i,e in enumerate(window._entries) if e['id']==ident))
    window.begin_trial();assert window.engine.control_mode=='personal' and not window.engine.experimental_neural
    window.enable_input.setChecked(True);assert not window.enable_input.isChecked()
    window.cancel_recording()
    assert window.engine.control_mode=='pointer' and window.engine.experimental_neural
    assert window.protocol is None and window.recording is None and not window.enable_input.isChecked()
    window.close();app.processEvents()


def test_guided_training_rejects_absent_hand_without_creating_template(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import FrameFeatures
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'));window._running=True
    window.record_name.setText('Не сохранять');window.begin_guided_enrollment()
    start=window.recording.started+3
    for i in range(8):window.recording.feed(FrameFeatures.absent(start+i*.1),start+i*.1)
    window.finish_recording()
    assert not any(e['name']=='Не сохранять' for e in library.list_gestures())
    assert 'отклонён' in window.record_status.text() and not window.enable_input.isChecked()
    assert window.protocol is None
    window.close();app.processEvents()


def test_trial_counts_delayed_completion_during_grace_but_ignores_countdown(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow,SessionRecorder
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import FrameFeatures,ControlEvent
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    samples=np.ones((8,48));samples[:,42]=np.linspace(.2,.7,8)
    ident=library.add('Движение','dynamic',samples,'click')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'));window._running=True
    window.session=SessionRecorder(tmp_path/'sessions',source={'kind':'camera'},method='two_stage',profiles=library.list_gestures())
    window.example_count.setValue(1);window.gestures.setCurrentRow(next(i for i,e in enumerate(window._entries) if e['id']==ident))
    before=library.path.read_bytes();window.begin_trial();start=window.recording.started+3
    window.protocol_events([ControlEvent(ident,'click',start-.1,{})])
    feature=FrameFeatures(start,True,np.ones(42),(.5,.5),(.5,.4),1,.2,'point',.99)
    window.recording.feed(feature,start)
    window.protocol_events([ControlEvent(ident,'click',start-.1,{}),ControlEvent(ident,'click',start+3+.3,{})])
    window.protocol.last_time=start+3+.3;window.finish_recording()
    assert window.protocol is not None
    window.protocol.last_time=start+3+.8;window.finish_recording()
    task=window.session.metadata['tasks'][0]
    assert task['matched_count']==1 and task['exercise_end']==start+3
    assert task['grace_seconds']==.8 and task['evaluation_end']==start+3+.8
    assert library.path.read_bytes()==before and not window.enable_input.isChecked()
    window.close();app.processEvents()


def test_custom_static_orientation_option_persists_without_enabling_input(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    ident=library.add('Направление','static',np.ones((8,48)),'click')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.gestures.setCurrentRow(next(i for i,e in enumerate(window._entries) if e['id']==ident))
    assert window.orientation_sensitive.isEnabled() and not window.orientation_sensitive.isChecked()
    window.orientation_sensitive.setChecked(True);window.save_mapping()
    assert GestureLibrary(tmp_path/'profiles').get(ident)['orientation_sensitive'] is True
    assert not window.enable_input.isChecked()
    window.gestures.setCurrentRow(0);assert not window.orientation_sensitive.isEnabled()
    window.close();app.processEvents()


def test_replayed_absence_boundary_event_is_saved_and_counted_in_trial_grace(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow,SessionRecorder
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import FrameFeatures,ControlEvent
    from gesture_system.live_runtime import FramePacket,FeatureSnapshot
    import time
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    samples=np.ones((8,48));samples[:,42]=np.linspace(.2,.7,8)
    ident=library.add('Ожидаемый','dynamic',samples,'click');engine=Engine(library,load_models=False)
    class Runtime:
        error=None
        packet=None
        def start(self):return 1
        def stop(self,**kwargs):return True
        def take_latest(self):return self.packet
    runtime=Runtime();window=MainWindow(tmp_path,library,engine,runtime_factory=lambda *args,**kwargs:runtime,screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.start_stream();window.example_count.setValue(1)
    window.gestures.setCurrentRow(next(i for i,e in enumerate(window._entries) if e['id']==ident));window.begin_trial()
    present_stamp=time.monotonic();absence_stamp=present_stamp-.04;exercise_start=absence_stamp-3.3
    window.recording.started=exercise_start-3
    window.recording.active_start=exercise_start;window.recording.active_end=exercise_start+3
    window.protocol.exercise_started=True;window._minimum_capture_time=exercise_start
    engine._clock=absence_stamp-.06
    engine.process=lambda feature:[ControlEvent(ident,'click',feature.timestamp,{})] if not feature.present else []
    feature=FrameFeatures(present_stamp,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99)
    runtime.packet=FramePacket(1,2,present_stamp,present_stamp,np.zeros((8,8,3),np.uint8),FeatureSnapshot.copy(feature),
                               {'missing_hand_frames':1,'last_absent_time':absence_stamp})
    window.tick()
    assert len(window.protocol.events)==1 and window.protocol.events[0]['timestamp']==absence_stamp
    assert window.session.events[0]['timestamp']==absence_stamp and not window.session.events[0]['executed']
    assert window.session.frames[0]['telemetry']['replayed_absence']
    window.cancel_recording();window.close();app.processEvents()
