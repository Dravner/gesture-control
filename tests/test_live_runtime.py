"""Bounded camera pipeline tests use artificial sources only."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading
import time
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from gesture_system.types import FrameFeatures


def until(predicate, timeout=2):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        time.sleep(.003)
    raise AssertionError('worker condition timed out')


class Capture:
    def __init__(self):self.sequence=0;self.owner=None;self.released=False
    def isOpened(self):self.owner=threading.get_ident();return True
    def read(self):
        assert threading.get_ident()==self.owner
        time.sleep(.003)
        self.sequence+=1
        return True,np.full((8,8,3),self.sequence%255,np.uint8)
    def release(self):
        assert threading.get_ident()==self.owner
        self.released=True


class Tracker:
    def __init__(self, delay=.03):self.delay=delay;self.owner=threading.get_ident();self.closed=False;self.observed=[]
    def process(self,image,stamp):
        assert threading.get_ident()==self.owner
        self.observed.append(int(image[0,0,0]));time.sleep(self.delay)
        return FrameFeatures(stamp,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99)
    def annotate(self,image,feature):return image
    def close(self):
        assert threading.get_ident()==self.owner
        self.closed=True

def test_short_failed_camera_reads_recover_without_losing_worker(tmp_path):
    from gesture_system.live_runtime import LiveRuntime
    class Intermittent(Capture):
        def read(self):
            self.sequence+=1
            if self.sequence<=2:return False,None
            time.sleep(.003);return True,np.ones((8,8,3),np.uint8)
    capture=Intermittent();runtime=LiveRuntime(tmp_path,capture_factory=lambda source:capture,tracker_factory=lambda:Tracker(.001))
    runtime.start()
    try:
        packet=until(runtime.take_latest)
        assert packet.feature.present and runtime.error is None
        assert packet.telemetry['capture_read_retries']==2
    finally:assert runtime.stop(wait=True,timeout=2)


def test_slow_inference_skips_frames_and_packets_are_immutable(tmp_path):
    from gesture_system.live_runtime import LiveRuntime
    capture=Capture();trackers=[]
    def factory():
        tracker=Tracker();trackers.append(tracker);return tracker
    runtime=LiveRuntime(tmp_path,capture_factory=lambda source:capture,tracker_factory=factory)
    runtime.start()
    first=until(runtime.take_latest)
    time.sleep(.14)
    latest=until(runtime.take_latest)
    assert latest.sequence>first.sequence+10
    assert len(trackers[0].observed)<capture.sequence/3
    assert runtime.take_latest() is None
    assert latest.result_time>=latest.capture_time
    assert latest.feature.timestamp==latest.capture_time
    assert latest.telemetry['dropped_capture_frames']>0
    with pytest.raises(ValueError):latest.image[0,0,0]=1
    with pytest.raises(ValueError):latest.feature.pose[0]=1
    with pytest.raises((FrozenInstanceError,AttributeError)):latest.feature.label='changed'
    with pytest.raises(TypeError):latest.telemetry['dropped_capture_frames']=0
    runtime.stop()
    assert runtime.wait(2)
    assert capture.released and trackers[0].closed


def test_stop_discards_inflight_result_and_restart_has_new_generation(tmp_path):
    from gesture_system.live_runtime import LiveRuntime
    entered=threading.Event();resume=threading.Event();captures=[]
    class Blocking(Tracker):
        def process(self,image,stamp):
            entered.set();resume.wait(2)
            return super().process(image,stamp)
    def capture_factory(source):
        capture=Capture();captures.append(capture);return capture
    runtime=LiveRuntime(tmp_path,capture_factory=capture_factory,tracker_factory=lambda:Blocking(0))
    first=runtime.start();assert entered.wait(2)
    started=time.monotonic();runtime.stop(wait=False)
    assert time.monotonic()-started<.1
    resume.set();assert runtime.wait(2)
    assert runtime.take_latest() is None
    second=runtime.start()
    packet=until(runtime.take_latest)
    assert second>first and packet.generation==second
    runtime.stop();assert runtime.wait(2)
    assert all(c.released for c in captures)


@pytest.mark.parametrize('failure',['capture','tracker'])
def test_worker_errors_surface_and_release_owned_resources(tmp_path,failure):
    from gesture_system.live_runtime import LiveRuntime
    capture=Capture();trackers=[]
    if failure=='capture':
        capture.read=lambda: (_ for _ in ()).throw(RuntimeError('camera disconnected'))
    class Failing(Tracker):
        def process(self,image,stamp):raise RuntimeError('tracker failed')
    def factory():
        tracker=Failing(0) if failure=='tracker' else Tracker(0)
        trackers.append(tracker);return tracker
    runtime=LiveRuntime(tmp_path,capture_factory=lambda source:capture,tracker_factory=factory)
    runtime.start()
    assert failure in until(lambda:runtime.error)
    assert runtime.wait(2)
    assert capture.released
    assert all(t.closed for t in trackers)
    assert runtime.take_latest() is None


def test_gui_remains_responsive_and_executes_only_on_main_thread(tmp_path):
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app=QApplication.instance() or QApplication([])
    main=threading.get_ident();calls=[];ticks=[]
    class Actions:
        def available(self):return True
        def execute(self,event):calls.append(threading.get_ident())
        def release(self):assert threading.get_ident()==main
    library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),actions=Actions(),
                      capture_factory=lambda source:Capture(),tracker_factory=lambda:Tracker(.12),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    pulse=QTimer();pulse.setInterval(5);pulse.timeout.connect(lambda:ticks.append(time.monotonic()))
    pulse.start();window.start_stream();window.enable_input.setChecked(True)
    deadline=time.monotonic()+.42
    while time.monotonic()<deadline:
        app.processEvents();time.sleep(.001)
    pulse.stop()
    assert len(ticks)>=30
    assert calls and set(calls)=={main}
    assert 'Обработка' in window.runtime_label.text()
    window.pause_stream()
    assert not window.enable_input.isChecked()
    assert window.live_runtime.wait(2)
    assert window.preview.pixmap().isNull()
    window.close();app.processEvents()


def test_gui_discards_previous_generation_and_pre_reset_capture(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.live_runtime import FramePacket,FeatureSnapshot
    app=QApplication.instance() or QApplication([])
    from types import SimpleNamespace
    class Runtime:
        error=None
        packets=[]
        def start(self):return 5
        def take_latest(self):return self.packets.pop(0) if self.packets else None
        def stop(self,**kwargs):return True
    class Actions:
        executed=[]
        def available(self):return True
        def execute(self,event):self.executed.append(event)
        def release(self):pass
    runtime=Runtime();actions=Actions();library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),actions=actions,
                      runtime_factory=lambda *args,**kwargs:runtime,screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.start_stream();window.enable_input.setChecked(True)
    now=time.monotonic()
    feature=FeatureSnapshot.copy(FrameFeatures(now,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99))
    runtime.packets=[FramePacket(4,1,now,now,np.zeros((8,8,3),np.uint8),feature,{})]
    window.recording=SimpleNamespace(samples=[])
    window.tick()
    assert actions.executed==[] and window.recording.samples==[]
    window.recording=None
    window.release_control();window.enable_input.setChecked(True)
    runtime.packets=[FramePacket(5,2,now,now,np.zeros((8,8,3),np.uint8),feature,{})]
    window.tick()
    assert actions.executed==[] and window.session.frames==[]
    window.pause_stream();window.close();app.processEvents()


def test_gui_worker_error_stops_and_releases_input(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app=QApplication.instance() or QApplication([])
    class Runtime:
        error='capture: disconnected'
        stopped=False
        def start(self):return 1
        def take_latest(self):raise AssertionError('must stop before packet consumption')
        def stop(self,**kwargs):self.stopped=True;return True
    class Actions:
        released=0
        def available(self):return True
        def execute(self,event):raise AssertionError('must not execute after source failure')
        def release(self):self.released+=1
    runtime=Runtime();actions=Actions();library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),actions=actions,
                      runtime_factory=lambda *args,**kwargs:runtime,screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.start_stream();window.enable_input.setChecked(True);window.tick()
    assert runtime.stopped and actions.released>0
    assert not window.timer.isActive() and not window.enable_input.isChecked()
    assert 'disconnected' in window.log.toPlainText()
    assert window.preview.pixmap().isNull()
    window.close();app.processEvents()


def test_overwritten_missing_hand_packet_still_releases_drag(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.live_runtime import FramePacket,FeatureSnapshot
    app=QApplication.instance() or QApplication([])
    class Runtime:
        error=None
        def start(self):return 1
        def stop(self,**kwargs):return True
        def take_latest(self):
            now=time.monotonic()
            feature=FeatureSnapshot.copy(FrameFeatures(now,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99))
            return FramePacket(1,3,now,now,np.zeros((8,8,3),np.uint8),feature,{'missing_hand_frames':1,'last_absent_time':now-.1})
    class Actions:
        released=0
        def available(self):return True
        def execute(self,event):pass
        def release(self):self.released+=1
    actions=Actions();library=GestureLibrary(tmp_path/'profiles')
    window=MainWindow(tmp_path,library,Engine(library,load_models=False),actions=actions,
                      runtime_factory=lambda *args,**kwargs:Runtime(),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.start_stream();window.enable_input.setChecked(True);window.engine._drag=True
    before=actions.released;window.tick()
    assert actions.released>before and not window.engine._drag
    window.pause_stream();window.close();app.processEvents()


def test_native_camera_aspect_is_preserved_and_geometry_is_reported(tmp_path):
    from gesture_system.live_runtime import LiveRuntime
    class Native(Capture):
        configured=[]
        def set(self,key,value):self.configured.append((key,value))
        def read(self):
            super().read()
            return True,np.zeros((18,32,3),np.uint8)
    capture=Native()
    runtime=LiveRuntime(tmp_path,capture_factory=lambda source:capture,tracker_factory=lambda:Tracker(0))
    runtime.start();packet=until(runtime.take_latest)
    assert capture.configured==[]
    assert packet.telemetry['width']==32 and packet.telemetry['height']==18
    assert packet.telemetry['capture_fps']>=0
    runtime.stop();assert runtime.wait(2)


def test_session_persists_actual_capture_geometry_and_latency(tmp_path):
    import json
    from gesture_system.gui import SessionRecorder
    session=SessionRecorder(tmp_path,source={'kind':'camera'},method='two_stage',profiles=[])
    session.add_frame(FrameFeatures.absent(1),prediction='no_hand',confidence=0,phase='idle',recording=False,
                      method='two_stage',telemetry={'width':1280,'height':720,'capture_fps':29.8,
                                                    'inference_ms':22.,'capture_to_display_ms':31.})
    path=session.finish('stop');data=np.load(path/'frames.npz',allow_pickle=False)
    assert data['capture_width'].tolist()==[1280]
    assert data['capture_height'].tolist()==[720]
    assert np.isclose(data['capture_fps'][0],29.8)
    assert data['capture_to_display_ms'].tolist()==[31.]
    metadata=json.loads((path/'session.json').read_text())
    assert metadata['source']['capture_formats']==[{'width':1280,'height':720,'aspect_ratio':1280/720}]


def test_world_points_survive_packet_snapshot_without_mutable_aliases():
    from gesture_system.live_runtime import FeatureSnapshot
    points=np.arange(63,dtype=np.float32).reshape(21,3)
    feature=FrameFeatures(1,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99,points)
    snapshot=FeatureSnapshot.copy(feature)
    points[0,0]=999
    assert snapshot.world_points.shape==(21,3)
    assert snapshot.world_points[0,0]==0
    with pytest.raises(ValueError):snapshot.world_points[0,0]=1
    assert snapshot.vector().shape==(48,)


@pytest.mark.parametrize('gap,long_gap',[(.04,False),(.22,False),(.44,True)])
def test_replayed_absence_preserves_short_episode_and_segments_long_gap(tmp_path,gap,long_gap):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.live_runtime import FramePacket,FeatureSnapshot
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    base=time.monotonic()-(gap+.15)
    def feature(t,x):return FrameFeatures(t,True,np.zeros(42),(x,.5),(x,.4),1,.2,'point',.99)
    last_seen=base+.06;absence=last_seen+gap;current=absence+.03
    class Runtime:
        error=None
        def start(self):return 1
        def stop(self,**kwargs):return True
        def take_latest(self):return FramePacket(1,4,current,current,np.zeros((8,8,3),np.uint8),FeatureSnapshot.copy(feature(current,.3)),
                                             {'missing_hand_frames':1,'last_absent_time':absence})
    class Actions:
        released=0
        def available(self):return True
        def execute(self,event):pass
        def release(self):self.released+=1
    actions=Actions();engine=Engine(library,load_models=False)
    window=MainWindow(tmp_path,library,engine,actions=actions,runtime_factory=lambda *args,**kwargs:Runtime(),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.start_stream();window._minimum_capture_time=base-.01
    engine.process(feature(base,.2));engine.process(feature(last_seen,.25));engine._drag=True
    before=actions.released;window.tick()
    assert actions.released>before and not engine._drag
    assert len(engine.personal.rows)==(1 if long_gap else 3)
    assert any(not row['present'] and row['timestamp']==absence for row in window.session.frames)
    window.pause_stream();window.close();app.processEvents()


def test_runtime_reports_actual_last_processed_absence_timestamp(tmp_path):
    from gesture_system.live_runtime import LiveRuntime
    absent=[]
    class OnceAbsent(Tracker):
        def process(self,image,stamp):
            if not absent:
                absent.append(stamp)
                return FrameFeatures.absent(stamp)
            return super().process(image,stamp)
    runtime=LiveRuntime(tmp_path,capture_factory=lambda source:Capture(),tracker_factory=lambda:OnceAbsent(.003))
    runtime.start()
    def present_packet():
        packet=runtime.take_latest()
        return packet if packet and packet.feature.present else None
    packet=until(present_packet)
    assert packet.telemetry['missing_hand_frames']==1
    assert packet.telemetry['last_absent_time']==absent[0]<packet.capture_time
    runtime.stop();assert runtime.wait(2)


def test_stale_packet_cannot_dispatch_replayed_absence_command(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    from gesture_system.types import ControlEvent
    from gesture_system.live_runtime import FramePacket,FeatureSnapshot
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    stamp=time.monotonic()-.7
    class Runtime:
        error=None
        def start(self):return 1
        def stop(self,**kwargs):return True
        def take_latest(self):
            feature=FrameFeatures(stamp,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99)
            return FramePacket(1,2,stamp,stamp,np.zeros((8,8,3),np.uint8),FeatureSnapshot.copy(feature),
                               {'missing_hand_frames':1,'last_absent_time':stamp-.02})
    class Actions:
        executed=[]
        def available(self):return True
        def execute(self,event):self.executed.append(event)
        def release(self):pass
    actions=Actions();engine=Engine(library,load_models=False)
    window=MainWindow(tmp_path,library,engine,actions=actions,runtime_factory=lambda *args,**kwargs:Runtime(),screen_provider=lambda:[]);window.profile_selector.setCurrentIndex(window.profile_selector.findData('advanced'))
    window.start_stream();window._minimum_capture_time=stamp-.1;window.enable_input.setChecked(True)
    engine.process=lambda feature:[ControlEvent('thumbs_up','click',feature.timestamp,{})]
    window.tick()
    assert actions.executed==[]
    assert window.session.frames[-1]['phase']=='stale'
    window.pause_stream();window.close();app.processEvents()
