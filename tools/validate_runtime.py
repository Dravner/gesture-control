"""Actual public RGB replay through the deployed pipeline, without OS input."""
import json,sys,time,tempfile
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gesture_system.vision import HandTracker
from gesture_system.engine import Engine
from gesture_system.profiles import GestureLibrary

def main():
    torch.set_num_threads(4)
    selection=json.loads((ROOT/'reports/streaming_research/selection.json').read_text())
    config=json.loads((ROOT/'reports/streaming_research/config.json').read_text())
    video=ROOT/'research/data/videos'/(config['clips']['test'][0]+'.avi')
    tracker=HandTracker(ROOT);cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    report=ROOT/'reports/runtime-validation';report.mkdir(parents=True,exist_ok=True)
    samples=[];events=[];present=0;screenshot=False;frames=0
    try:
        with tempfile.TemporaryDirectory() as directory:
            engine=Engine(GestureLibrary(directory),selection['method'])
            if engine._predictor is None:raise RuntimeError('Trained model not loaded')
            while frames<1200:
                ok,img=cap.read()
                if not ok:break
                stamp=frames/fps
                start=time.perf_counter();feature=tracker.process(img,stamp)
                after_frontend=time.perf_counter();output=engine.process(feature)
                after_engine=time.perf_counter();present+=int(feature.present)
                samples.append(dict(frame=frames,present=feature.present,frontend_ms=(after_frontend-start)*1000,controller_ms=(after_engine-after_frontend)*1000,total_ms=(after_engine-start)*1000))
                events.extend(dict(gesture=e.gesture,action=e.action,timestamp=e.timestamp,payload=e.payload) for e in output)
                if feature.present and not screenshot and frames>=30:
                    shown=tracker.annotate(img.copy(),feature)
                    cv2.putText(shown,f'Public IPN RGB replay | {selection["method"]} | input OFF',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                    cv2.putText(shown,f'{engine.last_label}: {engine.confidence:.2f}',(10,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                    cv2.imwrite(str(report/'public-replay.png'),shown);screenshot=True
                frames+=1
            engine.reset()
    finally:cap.release();tracker.close()
    measured=samples[30:] # exclude initialization warm-up, retain raw samples
    summary=dict(video=str(video.relative_to(ROOT)),method=selection['method'],frames=frames,present_frames=present,video_fps=fps,video_seconds=frames/fps,scope='1200 consecutive decoded frames of the first held-out public clip; actual MP+MLP+Engine, CPU, no GUI/render/capture/display/OS latency; no personal templates; no real input',thresholds=dict(confidence=.75,active=.7,confirm=3,neutral_rearm_s=.3,cooldown_s=.5),timing={key:dict(p50_ms=float(np.median([s[key] for s in measured])),p95_ms=float(np.percentile([s[key] for s in measured],95))) for key in ['frontend_ms','controller_ms','total_ms']},events_by_action={action:sum(e['action']==action for e in events) for action in sorted(set(e['action'] for e in events))})
    (report/'summary.json').write_text(json.dumps(summary,indent=2))
    (report/'samples.json').write_text(json.dumps(samples));(report/'events.json').write_text(json.dumps(events))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
