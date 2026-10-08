"""Bounded native Vision test without camera, GUI, or OS input."""
import sys,time,json,resource,subprocess,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gesture_system.apple_vision import NativeVisionHands
tracker=NativeVisionHands();frame=np.zeros((1080,1920,3),np.uint8);rows=[];timings=[]
for i in range(401):
    begin=time.perf_counter();tracker.detect(frame);timings.append((time.perf_counter()-begin)*1000)
    if i%20==0:
        rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())]).decode().strip())/1024
        row={'frame':i,'rss_mib':rss,'peak_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024};rows.append(row);print(row,flush=True)
tracker.close()
result={'scope':'Synthetic black 1920x1080 BGR frames; real Apple Vision, no hand, camera or GUI. RSS includes runtime caches.','samples':rows,'route_ms_p50':float(np.percentile(timings[20:],50)),'route_ms_p95':float(np.percentile(timings[20:],95)),'frames':401}
Path(__file__).with_name('vision-memory-fixed.json').write_text(json.dumps(result,indent=2))
