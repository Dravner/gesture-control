import os,time
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from gesture_system.gui import MainWindow
from gesture_system.pointing_ui import GestureReceiptDialog,DisplayInfo
from gesture_system.profiles import GestureLibrary
from gesture_system.engine import Engine
from gesture_system.types import FrameFeatures

class Actions:
    def available(self):return True
    def release(self):pass
    def execute(self,event):raise AssertionError('No input should execute in a stalled stream')

class Runtime:
    error=None
    def take_latest(self):return None
    def stop(self,**kwargs):return True

def window(tmp_path):
    app=QApplication.instance() or QApplication([]);library=GestureLibrary(tmp_path/'profiles')
    result=MainWindow(tmp_path,library,Engine(library,load_models=False),actions=Actions(),screen_provider=lambda:[DisplayInfo(1,'test',(0,0,1000,700),(.3,.2))])
    result.pointer_mode.setCurrentIndex(result.pointer_mode.findData('workspace'))
    result._running=True;result.live_runtime=Runtime();result._last_packet_wall=time.monotonic()-2
    f=FrameFeatures.absent(time.monotonic());f.frame_size=(640,480);result._last_feature=f
    return result

def test_receipt_is_parented_window_with_native_close_not_automatic_fullscreen(tmp_path):
    w=window(tmp_path);w.open_gesture_receipts();dialog=w.receipt_dialog
    assert dialog is not None and dialog.parent()==w
    assert not dialog.windowFlags() & Qt.FramelessWindowHint
    assert not dialog.isFullScreen()
    w.close()

def test_missing_packets_disarm_and_report_stall_inside_receipt(tmp_path):
    w=window(tmp_path);w.profile_selector.setCurrentIndex(w.profile_selector.findData('advanced'))
    dialog=GestureReceiptDialog(tmp_path,DisplayInfo(1,'test',(0,0,1000,700),(.3,.2)));w.receipt_dialog=dialog
    w.enable_input.setChecked(True);dialog.enabled.setChecked(True)
    w.tick()
    assert not w.enable_input.isChecked() and not dialog.enabled.isChecked()
    assert 'кадр' in dialog.tracking_status.text().lower()
    assert 'выключен' in dialog.tracking_status.text().lower()
    w.close()

def test_worker_error_is_visible_and_disables_receipt_input(tmp_path):
    w=window(tmp_path);dialog=GestureReceiptDialog(tmp_path,DisplayInfo(1,'test',(0,0,1000,700),(.3,.2)));w.receipt_dialog=dialog
    dialog.enabled.setChecked(True);w.live_runtime.error='capture: camera failed'
    w.tick()
    assert not w._running and not dialog.enabled.isChecked() and not dialog.enabled.isEnabled()
    assert 'camera failed' in dialog.tracking_status.text()
    w.close()
