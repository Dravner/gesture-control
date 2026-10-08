"""Synthetic timing/occlusion probe of the command controller, not vision accuracy."""
from pathlib import Path
import sys,json,csv,tempfile,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gesture_system.stable_control import StableController
from gesture_system.types import FrameFeatures
from gesture_system.profiles import GestureLibrary

class Mapper:
    def map_feature(self,f):return np.array(f.pointer)
    def camera_wrist(self,f):return np.array([*f.wrist,0.])

def main():
    out=Path(__file__).resolve().parent/'swipe-probe';out.mkdir(exist_ok=True)
    summaries=[];trace=[]
    with tempfile.TemporaryDirectory() as temporary:
        library=GestureLibrary(temporary)
        for fps in [15,30,60]:
            for gap in [0.,.08,.20]:
                for sign in [-1,1]:
                    for blur in [False,True]:
                        name=f'{fps}fps-gap{gap:.2f}-dir{sign}-uncertain{blur}'
                        c=StableController(library,Mapper());events=[];present_times=[]
                        for t in np.arange(0,1.6,1/fps):
                            t=float(t);x=.5+sign*.3*np.clip((t-.4)/.45,0,1)
                            label='palm' if t<.4 or not blur else 'uncertain'
                            if t>=1.1:label='uncertain'
                            missing=gap>0 and .52<=t<.52+gap-1e-9
                            if missing:f=FrameFeatures.absent(t)
                            else:
                                f=FrameFeatures(t,True,np.zeros(42),(float(x),.5),(.5,.5),1.,.2,'palm',.99,geometry_label=label,geometry_confidence=.99)
                                present_times.append(t)
                            current=c.process(f);events+=current
                            trace.append([name,t,not missing,label,float(x),c.state,','.join(e.gesture for e in current)])
                        swipe=[e for e in events if e.gesture.startswith('swipe')]
                        expected='swipe_left' if sign>0 else 'swipe_right'
                        summaries.append({'scenario':name,'fps':fps,'nominal_gap_seconds':gap,'maximum_observed_present_gap_seconds':float(np.diff(present_times).max()),'uncertain_during_motion':blur,'direction':expected,'swipe_count':len(swipe),'swipe_labels':[e.gesture for e in swipe],'first_event_seconds':swipe[0].timestamp if swipe else None,'non_swipe_command_count':sum(not e.gesture.startswith('swipe') for e in events)})
    document={'scope':'Synthetic observations explicitly supply pose labels; HandTracker/MediaPipe and OS/GUI bypassed. This measures controller timing invariance and gap policy, not recognition accuracy or live latency. No article edits.','protocol':'Open palm at rest until0.4s,0.3frame horizontal displacement over0.45s, optional absence begins0.52s, optional uncertain pose during motion; neutral after1.1s;15/30/60Hz. Temporary default gesture bindings.','source_sha256':hashlib.sha256((ROOT/'gesture_system/stable_control.py').read_bytes()).hexdigest(),'scenarios':summaries}
    (out/'results.json').write_text(json.dumps(document,indent=2)+'\n')
    with (out/'timeline.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['scenario','timestamp','present','pose_label','raw_x','phase','emitted_gesture']);writer.writerows(trace)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    matrix=np.array([[sum(s['swipe_count'] for s in summaries if s['fps']==fps and s['nominal_gap_seconds']==gap) for gap in [0.,.08,.20]] for fps in [15,30,60]])
    ax.imshow(matrix,vmin=0,vmax=4,cmap='Blues')
    for y in range(3):
        for x in range(3):ax.text(x,y,f'{matrix[y,x]} / 4',ha='center',va='center',color='white' if matrix[y,x]>2 else 'black',fontsize=14)
    ax.set_xticks(range(3),['0','80','200']);ax.set_yticks(range(3),['15','30','60'])
    ax.set_xlabel('Заданный пропуск наблюдений, мс');ax.set_ylabel('Частота входных отсчётов, Гц')
    ax.set_title('Команды свайпа: два направления × две политики позы')
    fig.text(.5,.015,'Синтетический контроллер; распознавание камеры и ОС не измеряются.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'swipe-counts.png',dpi=160);plt.close(fig)
    print(json.dumps({'scenarios':len(summaries),'commands_by_fps_gap':matrix.tolist(),'non_swipe_commands':sum(s['non_swipe_command_count'] for s in summaries)},indent=2))
if __name__=='__main__':main()
