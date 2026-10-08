import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt,QPoint,QPointF
from PySide6.QtGui import QWheelEvent
from gesture_system.pointing_ui import GestureReceiptDialog,DisplayInfo

def test_pixel_wheel_moves_content_exactly_and_reports_actual_distance(tmp_path):
    app=QApplication.instance() or QApplication([])
    d=GestureReceiptDialog(tmp_path,DisplayInfo(1,'test',(0,0,1200,950),(.3,.2)),navigation_pose='three')
    d.show();app.processEvents();bar=d.text.verticalScrollBar();assert bar.maximum()>200
    for pixels in [27,5]:
        event=QWheelEvent(QPointF(50,50),QPointF(d.text.mapToGlobal(QPoint(50,50))),QPoint(0,-pixels),QPoint(0,0),Qt.NoButton,Qt.NoModifier,Qt.ScrollUpdate,False)
        app.sendEvent(d.text.viewport(),event);app.processEvents()
    assert bar.value()==32
    assert '32' in d.scroll_feedback.text()
    assert any(r.get('pixel_delta')==[0,-27] for r in d.receipts)
    d.close()

def test_first_left_swipe_changes_visible_card_and_never_resets_scroll(tmp_path):
    app=QApplication.instance() or QApplication([])
    d=GestureReceiptDialog(tmp_path,DisplayInfo(1,'test',(0,0,1200,950),(.3,.2)),navigation_pose='three')
    d.show();app.processEvents();d.text.verticalScrollBar().setValue(100);label=d.page_label.text()
    d.navigate(-1,'command+left');assert d.page_label.text()!=label
    assert d.text.verticalScrollBar().value()==100
    d.navigate(1,'command+right');assert d.page_label.text()==label
    d.close()
