import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

def test_camera_activity_balances_one_token_and_is_safe_to_repeat():
    from gesture_system.camera_activity import CameraActivity
    class Process:
        def __init__(self):self.begins=[];self.ends=[]
        def beginActivityWithOptions_reason_(self,options,reason):
            self.begins.append((options,reason));return object()
        def endActivity_(self,token):self.ends.append(token)
    process=Process();activity=CameraActivity(platform='darwin',provider=lambda:(process,123))
    assert activity.start() and activity.start()
    assert len(process.begins)==1
    activity.stop();activity.stop();assert len(process.ends)==1
    assert activity.start();activity.stop();assert len(process.ends)==2

def test_camera_activity_is_noop_outside_macos():
    from gesture_system.camera_activity import CameraActivity
    activity=CameraActivity(platform='linux',provider=lambda:(_ for _ in ()).throw(AssertionError('must not load Foundation')))
    assert not activity.start();activity.stop()

def test_gui_releases_activity_on_pause_and_failed_start(tmp_path):
    from PySide6.QtWidgets import QApplication
    from gesture_system.gui import MainWindow
    from gesture_system.profiles import GestureLibrary
    from gesture_system.engine import Engine
    app=QApplication.instance() or QApplication([])
    class Activity:
        active=False
        def start(self):self.active=True;return True
        def stop(self):self.active=False
    class Runtime:
        fail=False
        def start(self):
            if self.fail:raise RuntimeError('camera start failed')
            return 1
        def stop(self,**kwargs):return True
    activity=Activity();runtime=Runtime();library=GestureLibrary(tmp_path/'profiles')
    w=MainWindow(tmp_path,library,Engine(library,load_models=False),runtime_factory=lambda *args,**kwargs:runtime,screen_provider=lambda:[],camera_activity=activity)
    w.start_stream();assert activity.active
    w.pause_stream();assert not activity.active
    runtime.fail=True;w.start_stream();assert not activity.active and not w._running
    w.close()
