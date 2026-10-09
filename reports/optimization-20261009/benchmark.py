"""Bounded public-video benchmark, no camera/window/OS events. Safe product NSData route."""
from pathlib import Path
import sys,json,csv,time,hashlib,platform,importlib.metadata,resource
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gesture_system.apple_vision import VisionHandTracker
from gesture_system.pose_stability import ImagePoseLatch
OUT=Path(__file__).resolve().parent

def main():
 inventory=json.loads((OUT/'input-inventory.json').read_text());rows=[];details=[]
 for item in inventory['selected']:
  cap=cv2.VideoCapture(str(ROOT/'research/data/videos'/item['video']));start=item['frames']//2-20;cap.set(cv2.CAP_PROP_POS_FRAMES,start);frames=[]
  for i in range(40):
   ok,frame=cap.read()
   if not ok:raise RuntimeError('clip ended')
   frames.append(frame)
  cap.release()
  for mirror in (False,True):
   for cohort,caps in [('native640',[640,480,320]),('upscale1920',[1920,1280,960,640])]:
    trackers={size:VisionHandTracker(ROOT) for size in caps};latches={size:ImagePoseLatch() for size in caps};obs={}
    try:
     for i,frame in enumerate(frames):
      image=cv2.flip(frame,1) if mirror else frame
      if cohort=='upscale1920':image=cv2.resize(image,(1920,1440),interpolation=cv2.INTER_CUBIC)
      for size in (caps if i%2==0 else caps[::-1]):
       begin=time.perf_counter();scale=min(1.,size/max(image.shape[:2]));prepared=image if scale==1 else cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)),interpolation=cv2.INTER_AREA)
       resize_ms=(time.perf_counter()-begin)*1000
       f=trackers[size].process(prepared,i/30);pose=latches[size].update(f)
       r={'video':item['video'],'frame':start+i,'mirror':mirror,'cohort':cohort,'max_side':size,'width':prepared.shape[1],'height':prepared.shape[0],'warmup':i<5,'resize_ms':resize_ms,'total_ms':(time.perf_counter()-begin)*1000,'accepted':bool(f.present),'pose':pose.label,**trackers[size].detector.last_metrics};rows.append(r)
       obs[(i,size)]=(f,pose)
      base,bpose=obs[(i,caps[0])]
      for size in caps:
       f,pose=obs[(i,size)];r=rows[-len(caps):][caps.index(size) if i%2==0 else len(caps)-1-caps.index(size)]
       r['baseline_accepted']=bool(base.present);r['acceptance_agreement']=bool(base.present)==bool(f.present)
       r['baseline_pose']=bpose.label;r['confident_pose_pair']=base.present and f.present and bpose.confidence>=.9 and pose.confidence>=.9;r['pose_agreement']=bpose.label==pose.label
       if base.present and f.present:
        ids=np.isfinite(base.image_points).all(1)&np.isfinite(f.image_points).all(1)&(base.joint_confidence>=.25)&(f.joint_confidence>=.25)
        displacement=np.linalg.norm((base.image_points-f.image_points)*[640,480],axis=1)
        r['joint_pair_count']=int(ids.sum());r['joint_displacement_source_px_mean']=float(np.mean(displacement[ids])) if ids.any() else None
        # Actual stable workspace controller uses raw wrist/middle-MCP midpoint.
        r['pointer_displacement_source_px']=float(np.linalg.norm(((base.image_points[[0,9]].mean(0)-f.image_points[[0,9]].mean(0))*[640,480])))
       details.append({'video':item['video'],'frame':start+i,'mirror':mirror,'cohort':cohort,'max_side':size,'points':f.image_points.tolist() if f.present else None,'confidence':f.joint_confidence.tolist() if f.joint_confidence is not None else None})
    finally:
     for tracker in trackers.values():tracker.close()
    print(item['video'],mirror,cohort,flush=True)
 summary=[]
 for cohort,caps in [('native640',[640,480,320]),('upscale1920',[1920,1280,960,640])]:
  for size in caps:
   selected=[r for r in rows if r['cohort']==cohort and r['max_side']==size and not r['warmup']];s={'cohort':cohort,'max_side':size,'n':len(selected),'accepted':sum(r['accepted'] for r in selected),'baseline_accepted':sum(r['baseline_accepted'] for r in selected),'acceptance_agreement_rate':np.mean([r['acceptance_agreement'] for r in selected])};pairs=[r for r in selected if r['confident_pose_pair']];s['confident_pose_pairs']=len(pairs);s['confident_pose_agreement_rate']=np.mean([r['pose_agreement'] for r in pairs]) if pairs else None
   for field in ('route_ms','native_request_ms','conversion_ms','resize_ms','total_ms','joint_displacement_source_px_mean','pointer_displacement_source_px'):
    values=[r[field] for r in selected if r.get(field) is not None]
    for percentile in (50,95):s[field+'_p'+str(percentile)]=float(np.percentile(values,percentile)) if values else None
   summary.append(s)
 versions={name:importlib.metadata.version(name) for name in ('numpy','opencv-python','pyobjc-core','pyobjc-framework-Vision')}
 document={'platform':platform.platform(),'python':sys.version,'versions':versions,'route_source_sha256':hashlib.sha256((ROOT/'gesture_system/apple_vision.py').read_bytes()).hexdigest(),'vision_revision':int(VisionHandTracker(ROOT).detector.request.revision()),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'summary':summary,'measurements':rows}
 (OUT/'results.json').write_text(json.dumps(document,indent=2)+'\n');(OUT/'landmarks.json').write_text(json.dumps(details)+'\n')
 columns=sorted(set().union(*(r.keys() for r in rows)))
 with (OUT/'timings.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
