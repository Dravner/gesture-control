"""Public image reflection probe; not annotated live-hand gesture accuracy."""
from pathlib import Path
import sys,json,collections,time
import numpy as np,cv2
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from gesture_system.apple_vision import VisionHandTracker
from gesture_system.pose_stability import ImagePoseLatch
videos=sorted((root/'research/data/videos').glob('*.avi'));chosen=[];people=set()
for path in videos:
    person=path.name.split('_')[0]
    if person not in people:chosen.append(path);people.add(person)
    if len(chosen)==5:break
results=[]
for path in chosen:
    cap=cv2.VideoCapture(str(path));start=max(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//2-20);cap.set(cv2.CAP_PROP_POS_FRAMES,start);frames=[]
    for _ in range(40):
        ok,frame=cap.read()
        if not ok:break
        frames.append(frame)
    cap.release()
    for mirrored in [False,True]:
        tracker=VisionHandTracker(root);latch=ImagePoseLatch();counts=collections.Counter();sides=collections.Counter();timings=[];critical=[];old_accepted=0
        for i,frame in enumerate(frames):
            if mirrored:frame=cv2.flip(frame,1)
            begin=time.perf_counter();f=tracker.process(frame,i/30);timings.append((time.perf_counter()-begin)*1000)
            counts['processed']+=1;counts['present']+=bool(f.present);sides[f.handedness or 'unknown']+=1
            q=f.joint_confidence
            if q is not None and q.shape==(21,):
                old_accepted+=bool(np.all(q>=.4));critical.append(float(np.min(q[[0,5,8,9]])))
            counts[latch.update(f).label]+=1
        tracker.close();results.append({'video':path.name,'mirror':mirrored,'counts':dict(counts),'handedness':dict(sides),'old_all_21_confidence_at_least_04':old_accepted,'route_ms_p50':float(np.percentile(timings[5:],50))})
        print(path.name,mirrored,dict(counts),flush=True)
Path(__file__).with_name('image-mirror-results.json').write_text(json.dumps({'scope':'Five public IPN Hand video snippets; 40 contiguous midpoint frames per video and orientation. Actual RGB images mirrored, not just coordinates. Pose intentions not annotated. Chirality predictions are not hand ground truth.','results':results},indent=2))
