"""Synthetic result-to-GUI delivery benchmark. No camera/OS input/accuracy claim."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json,time,threading,sys
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject,Signal,QTimer,Qt
from PySide6.QtWidgets import QApplication
app=QApplication([])
class Bench(QObject):
    ready=Signal()
    def __init__(self,queued,n=180):
        super().__init__();self.packet=None;self.lock=threading.Lock();self.delays=[];self.n=n;self.stop=threading.Event();self.timer=QTimer();self.timer.setInterval(30);self.timer.timeout.connect(self.consume);self.ready.connect(self.consume,Qt.ConnectionType.QueuedConnection);self.queued=queued
    def produce(self):
        # Incommensurate period avoids favorable alignment with 30ms polling.
        while not self.stop.wait(.017):
            with self.lock:self.packet=time.perf_counter()
            if self.queued:self.ready.emit()
    def consume(self):
        with self.lock:stamp=self.packet;self.packet=None
        if stamp is not None:self.delays.append((time.perf_counter()-stamp)*1000)
        if len(self.delays)>=self.n:self.stop.set();app.quit()
    def run(self):
        worker=threading.Thread(target=self.produce);worker.start()
        if not self.queued:self.timer.start()
        app.exec();self.stop.set();worker.join();self.timer.stop()
        return {'samples':len(self.delays),'median_ms':float(np.median(self.delays)),'p95_ms':float(np.percentile(self.delays,95)),'max_ms':max(self.delays)}
result={'kind':'synthetic Qt handoff, not end-to-end camera latency','poll30':Bench(False).run(),'queued':Bench(True).run()}
Path(__file__).with_name('handoff.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
