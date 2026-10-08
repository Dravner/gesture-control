"""Extract complete members of a growing official archive, then MediaPipe features."""
import argparse,csv,hashlib,json,tarfile,time,sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.protocol import LABELS,frame_labels
from gesture_system.vision import features_from_points

def extract_complete(archive,out):
    out.mkdir(parents=True,exist_ok=True); count=0
    try:
        with tarfile.open(archive,mode='r|gz') as tar:
            members=0
            for item in tar:
                name=Path(item.name).name
                if item.isfile() and name.endswith(('.mp4','.avi')):
                    members+=1
                    if members>8:break
                    dest=out/name
                    if dest.exists(): continue
                    tmp=dest.with_suffix('.tmp')
                    try:
                        with tar.extractfile(item) as src,open(tmp,'wb') as dst:
                            while chunk:=src.read(1024*1024): dst.write(chunk)
                        if tmp.stat().st_size==item.size: tmp.rename(dest); count+=1;print('Extracted',name,flush=True)
                    except (EOFError,tarfile.ReadError): tmp.unlink(missing_ok=True);break
    except (EOFError,tarfile.ReadError): pass
    # A subset marker proves eight complete tar members; partial archives stay labelled partial.
    stem=Path(archive).name.split('.tgz')[0]
    if 'members' in locals() and members>=8:
        try:
            names=[]
            with tarfile.open(archive,mode='r|gz') as tar:
                for m in tar:
                    if m.isfile() and m.name.endswith(('.avi','.mp4')):
                        dest=out/Path(m.name).name
                        if dest.exists() and dest.stat().st_size==m.size:names.append(dest.name)
                        if len(names)==8:break
            if len(names)==8:
                marker=Path(archive).parent/(stem+'.subset.json')
                if not marker.exists():marker.write_text(json.dumps(dict(archive=stem,complete_members=names,downloaded_compressed_bytes=Path(archive).stat().st_size,selection='First eight complete RGB video members, fixed before recognition; official archive intentionally partial.'),indent=2))
        except (EOFError,tarfile.ReadError):pass
    return count

def annotations():
    rows=csv.DictReader(open(ROOT/'research/data/annotations/annotations/Annot_List.txt'))
    events={}
    for r in rows: events.setdefault(r['video'],[]).append((int(r['id'])-1,int(r['t_start']),int(r['t_end'])))
    return events

def features(video,stride=3):
    import mediapipe as mp
    events=annotations()[video.stem]
    options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/hand_landmarker.task')),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=1,min_hand_detection_confidence=.4,min_hand_presence_confidence=.4,min_tracking_confidence=.4)
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or not n: raise ValueError(f'Unreadable {video}')
    vectors=[];present=[];frames=[];latency=[]; count=0
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ok,frame=cap.read()
            if not ok: break
            count+=1
            if (count-1)%stride: continue
            t=(count-1)/fps
            start=time.perf_counter()
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),round(t*1000))
            found=bool(result.hand_landmarks);v=np.zeros(48,np.float32)
            if found:
                xy=np.array([(p.x,p.y) for p in result.hand_landmarks[0]],dtype=np.float32)
                handedness=result.handedness[0][0].category_name
                v=features_from_points(xy,handedness,t,'none',0.).vector()
            latency.append((time.perf_counter()-start)*1000)
            vectors.append(v);present.append(found);frames.append(count)
    cap.release()
    frames=np.array(frames);x=np.array(vectors)
    labels=frame_labels(frames,events,unknown=-1)
    out=ROOT/'research/data/features';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(video.stem+'.npz'),x=x,present=np.array(present),frames=frames,t=(frames-1)/fps,y=labels,latency_ms=np.array(latency),fps=fps,stride=stride)
    return dict(feature_version=2,video=video.name,person='_'.join(video.stem.split('_')[:2]),fps=fps,source_frames=n,sampled_frames=len(frames),missing_frames=int(sum(~np.array(present))),sha256=hashlib.sha256(video.read_bytes()).hexdigest(),events=events,duration_s=n/fps)

def run(watch=False):
    data=ROOT/'research/data'; out=data/'videos';manifest=data/'manifest.json'
    entries=json.loads(manifest.read_text()) if manifest.exists() else []
    entries=[e for e in entries if e.get('feature_version')==2]
    while True:
        archives=list((data/'archives').glob('*.tgz'))+list((data/'archives').glob('*.part'))
        for archive in archives: extract_complete(archive,out)
        known={e['video'] for e in entries}
        for video in sorted(out.glob('*')):
            if video.suffix in ('.avi','.mp4') and video.name not in known:
                e=features(video);entries.append(e);manifest.write_text(json.dumps(entries,indent=2));print('FEATURES',e['video'],e['person'],e['sampled_frames'],flush=True)
        if not watch:break
        if len(list((data/'archives').glob('*.subset.json')))==5 and len(entries)==40:break
        time.sleep(45)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args();run(args.watch)
