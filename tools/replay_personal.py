"""Replay numerical personal recordings for debugging; never execute OS input."""
import argparse,json,sys,time,tempfile
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gesture_system.types import FrameFeatures
from gesture_system.engine import Engine
from gesture_system.profiles import GestureLibrary

def main(session,output):
    torch.set_num_threads(4)
    session=Path(session);d=np.load(session/'frames.npz',allow_pickle=False)
    metadata=json.loads((session/'session.json').read_text());events=[];names=[];timings=[]
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'gestures.json'
        path.write_text(json.dumps(dict(version=1,gestures=metadata['profiles'])))
        library=GestureLibrary(directory);engine=Engine(library,metadata['method'])
        engine.conf_threshold=metadata['settings']['confidence'];engine.hold_seconds=metadata['settings']['hold_seconds']
        for i,v in enumerate(d['vectors']):
            f=FrameFeatures(float(d['timestamps'][i]),bool(d['present'][i]),v[:42],tuple(v[42:44]),tuple(v[44:46]),float(v[46]),float(v[47]),str(d['labels'][i]),float(d['raw_confidence'][i]))
            start=time.perf_counter();out=engine.process(f);timings.append((time.perf_counter()-start)*1000)
            names.append(engine.last_label)
            for e in out:
                try:name=library.get(e.gesture if len(e.gesture)==32 or ':' in e.gesture else 'builtin:'+e.gesture)['name']
                except StopIteration:name=e.gesture
                events.append(dict(gesture=e.gesture,name=name,action=e.action,elapsed_s=e.timestamp-float(d['timestamps'][0])))
    summary=dict(session=str(session),scope='Numerical replay with current implementation; reported 3 call +3 switch failures used to diagnose and modify code, therefore this replay is development evidence, not independent accuracy after fixing',custom_events=[e for e in events if len(e['gesture'])==32],recognition_states={name:names.count(name) for name in set(names)},controller_timing=dict(p50_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95))),all_events=events)
    Path(output).write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['all_events','recognition_states']},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('--output',default=str(ROOT/'reports/personal-debug-replay.json'));a=p.parse_args();main(a.session,a.output)
