"""Measure the actual latest-frame camera pipeline; never enable OS input."""
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gesture_system.live_runtime import LiveRuntime

def main():
    torch.set_num_threads(4);runtime=LiveRuntime(ROOT);values=[];started=time.monotonic()
    try:
        runtime.start()
        while time.monotonic()-started<12:
            if runtime.error:raise RuntimeError(runtime.error)
            packet=runtime.take_latest()
            if packet is not None:
                metrics=dict(packet.telemetry)
                metrics.update(timestamp=packet.capture_time,age_ms=(time.monotonic()-packet.capture_time)*1000,present=packet.feature.present,world_present=packet.feature.world_points is not None)
                values.append(metrics)
            else:time.sleep(.005)
    finally:
        runtime.stop();runtime.wait(3)
    if not values:raise RuntimeError('No actual camera packets')
    measurements=values[20:]
    summary=dict(scope='Actual camera latest-frame capture+MediaPipe+MLP+annotate pipeline, no GUI/Engine/OS, no hand accuracy claim; excludes first20 result packets from timing',packets=len(values),present=sum(v['present'] for v in values),world_present=sum(v['world_present'] for v in values),dimensions=[values[-1].get('width'),values[-1].get('height')],camera_fps=values[-1].get('capture_fps'),processed_fps=(len(values)-1)/(values[-1]['timestamp']-values[0]['timestamp']),timing={key:dict(p50=float(np.median([v[key] for v in measurements])),p95=float(np.percentile([v[key] for v in measurements],95))) for key in ['age_ms','compute_ms']},final_telemetry=values[-1],stopped=runtime.wait(0))
    report=ROOT/'reports/realtime-v2';report.mkdir(parents=True,exist_ok=True)
    (report/'live-camera-runtime.json').write_text(json.dumps(summary,indent=2));(report/'live-camera-samples.json').write_text(json.dumps(values,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
