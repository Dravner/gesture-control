"""Apple Vision vs existing MediaPipe extraction on a fixed public video subset."""
from pathlib import Path
import sys,json,time,csv,hashlib,platform
import numpy as np
import cv2,Vision,Quartz as q,objc
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gesture_system.vision import HandTracker

JOINTS=['Wrist','ThumbCMC','ThumbMP','ThumbIP','ThumbTip']+[finger+joint for finger in ['Index','Middle','Ring','Little'] for joint in ['MCP','PIP','DIP','Tip']]

def vision_frame(frame,request):
    start=time.perf_counter();rgba=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA);h,w=rgba.shape[:2]
    blob=rgba.tobytes();provider=q.CGDataProviderCreateWithData(None,blob,len(blob),None)
    image=q.CGImageCreate(w,h,8,32,w*4,q.CGColorSpaceCreateDeviceRGB(),q.kCGImageAlphaPremultipliedLast,provider,None,False,q.kCGRenderingIntentDefault)
    handler=Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image,{})
    conversion_ms=(time.perf_counter()-start)*1000;begin=time.perf_counter()
    ok,error=handler.performRequests_error_([request],None)
    if not ok:raise RuntimeError(str(error))
    native_ms=(time.perf_counter()-begin)*1000;observations=list(request.results() or [])
    points=None;confidence=[]
    if observations:
        points=[]
        for name in JOINTS:
            point,error=observations[0].recognizedPointForJointName_error_(getattr(Vision,'VNHumanHandPoseObservationJointName'+name),None)
            if point is None:points.append([np.nan,np.nan]);confidence.append(0.)
            else:
                location=point.location();points.append([float(location.x),1-float(location.y)]);confidence.append(float(point.confidence()))
        points=np.array(points)
    return {'present':bool(observations),'route_ms':(time.perf_counter()-start)*1000,'native_request_ms':native_ms,'conversion_ms':conversion_ms,'all21_confident_ge04':bool(confidence) and min(confidence)>=.4,'minimum_joint_confidence':min(confidence) if confidence else None},points

def main():
    out=Path(__file__).resolve().parent/'vision-probe';out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'research/data/manifest.json').read_text());selected=[];persons=set()
    for item in manifest:
        if item['person'] in persons or not (ROOT/'research/data/videos'/item['video']).exists():continue
        persons.add(item['person']);selected.append(item)
        if len(selected)==5:break
    request=Vision.VNDetectHumanHandPoseRequest.alloc().init();request.setMaximumHandCount_(1)
    tracker=HandTracker(ROOT);tracker.model=None # Isolate landmark extraction; no MLP commands.
    rows=[];point_rows=[];index=0
    try:
        for item in selected:
            cap=cv2.VideoCapture(str(ROOT/'research/data/videos'/item['video']));first=item['source_frames']//2;cap.set(cv2.CAP_PROP_POS_FRAMES,first)
            try:
                for offset in range(30):
                    ok,frame=cap.read()
                    if not ok:raise RuntimeError('Public video ended early')
                    results={}
                    for method in (['vision','mediapipe'] if index%2==0 else ['mediapipe','vision']):
                        if method=='vision':
                            with objc.autorelease_pool():metrics,points=vision_frame(frame,request)
                        else:
                            start=time.perf_counter();feature=tracker.process(frame,(index+1)/30)
                            metrics={'present':bool(feature.present),'route_ms':(time.perf_counter()-start)*1000}
                            points=feature.image_points if feature.present else None
                        metrics.update(method=method,video=item['video'],person=item['person'],frame=first+offset,warmup=offset<5,width=frame.shape[1],height=frame.shape[0]);rows.append(metrics)
                        if points is not None:point_rows.append({'method':method,'video':item['video'],'frame':first+offset,'points':points.tolist()})
                    index+=1
            finally:cap.release()
    finally:tracker.close()
    summary={}
    for method in ['mediapipe','vision']:
        measured=[r for r in rows if r['method']==method and not r['warmup']]
        ms=np.array([r['route_ms'] for r in measured]);summary[method]={'measured_frames':len(measured),'hand_observations':sum(r['present'] for r in measured),'route_ms_p50':float(np.median(ms)),'route_ms_p95':float(np.percentile(ms,95))}
        if method=='vision':summary[method]['all21_confident_ge04']=sum(r['all21_confident_ge04'] for r in measured)
    document={'scope':'No camera or OS input. Existing five public IPN Hand videos, first available video of each distinct person,30 consecutive frames at midpoint;5 warmup per clip. Alternating backend order. Route includes pixel conversion + model + point extraction; video decode excluded. MediaPipe includes features extraction, existing MLP disabled. Vision uses default native request device selection; no claim of a specific accelerator.','limitations':'No independent joint/hand-presence ground truth; hand counts and Apple per-point confidence are observations, not recall/accuracy. Confidence scores are not comparable across APIs. Repeated small clips do not prove continuous live robustness. Different temporal APIs: MediaPipe VIDEO, Vision image requests. Native 3D hand world points unavailable from this Apple API.','platform':platform.platform(),'vision_revision':int(request.revision()),'mediapipe_model_sha256':hashlib.sha256((ROOT/'models/hand_landmarker.task').read_bytes()).hexdigest(),'inputs':[{'video':x['video'],'person':x['person'],'sha256_from_existing_manifest':x['sha256'],'first_frame':x['source_frames']//2} for x in selected],'summary':summary,'measurements':rows}
    (out/'results.json').write_text(json.dumps(document,indent=2)+'\n');(out/'points.json').write_text(json.dumps(point_rows,allow_nan=False)+'\n')
    columns=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'timings.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
