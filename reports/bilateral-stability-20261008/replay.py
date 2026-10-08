"""Same saved image coordinates; no camera, OS actions or intention ground truth."""
import sys,json,tempfile,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gesture_system.vision import features_from_points
from gesture_system.types import FrameFeatures
from gesture_system.profiles import GestureLibrary
from gesture_system.stable_control import StableController
from gesture_system.workspace_pointer import WorkspaceMapper
root=Path(__file__).resolve().parents[2];a=np.load(root/'data/sessions/20261008T183159-a8604adb/frames.npz');summaries={};traces={}
for mode,mirror in [('before',False),('after',False),('after_mirror',True)]:
    c=StableController(GestureLibrary(Path(tempfile.mkdtemp())),WorkspaceMapper({'source_kind':'camera','frame_size':[1920,1080]},relative=True));c.prefer_image_geometry=True;c.scroll_dead_zone=.006
    c.calibration.palm_anchor='midpoint' if mode!='before' else 'median';c.robust_geometry=mode!='before';c.smooth_scroll=mode!='before';c.navigation_pose='victory' if mode!='before' else 'palm';c.scroll_pixels_per_unit=1980
    labels=[];phases=[];events=[]
    for t,present,raw in zip(a['timestamps'],a['present'],a['image_points']):
        if not present:f=FrameFeatures.absent(float(t))
        else:
            points=raw.reshape(21,2).copy()
            if mirror:points[:,0]=1-points[:,0]
            f=features_from_points(points,'Left' if mirror else 'Right',float(t),'',0);f.image_points=points
        f.frame_size=(1920,1080);events+=c.process(f);labels.append(c.last_label);phases.append(c.phase)
    summaries[mode]={'labels':dict(collections.Counter(labels)),'phases':dict(collections.Counter(phases)),'label_transitions':sum(x!=y for x,y in zip(labels,labels[1:])),'phase_transitions':sum(x!=y for x,y in zip(phases,phases[1:])),'actions':dict(collections.Counter(e.action for e in events))}
    traces[mode]=(labels,phases)
summaries['mirror_phase_mismatches']=sum(x!=y for x,y in zip(traces['after'][1],traces['after_mirror'][1]))
summaries['limits']='Only saved accepted image coordinates can be replayed. Old rejected hand observations have no image points; new confidence gate cannot recover them. More cursor frames/fewer changes are not accuracy; no intention labels. Horizontal mirrored motion reverses navigation direction by design.'
Path(__file__).with_name('replay-results.json').write_text(json.dumps(summaries,indent=2));print(json.dumps(summaries,indent=2))
