"""Shared 48-dimensional representation, keeping pose and global trajectory separate."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from .types import FrameFeatures

HAND_CONNECTIONS=((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                  (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                  (13,17),(0,17),(17,18),(18,19),(19,20))

def guard_discrete_label(label,world):
    from .personal import anatomy
    reaches=anatomy(world)[::3]
    if label=='fist' and np.any(reaches[1:]>.65):return 'no_gesture'
    if label in {'thumbs_up','thumbs_down'} and (reaches[0]<.75 or np.any(reaches[1:]>.65)):return 'no_gesture'
    return label

def features_from_points(points,handedness,t,label,confidence):
    pts=np.asarray(points,dtype=np.float32).reshape(21,2)
    scale=max(float(np.linalg.norm(pts[9]-pts[0])),1e-6)
    pose=(pts-pts[0])/scale
    if handedness.lower()=='left':pose[:,0]*=-1
    pinch=float(np.linalg.norm(pts[4]-pts[8])/scale)
    return FrameFeatures(t,True,pose.reshape(42),tuple(map(float,pts[0])),tuple(map(float,pts[8])),pinch,scale,label,float(confidence),handedness=handedness)


@dataclass(frozen=True)
class GeometryObservation:
    label: str
    confidence: float
    pinch: float
    source: str
    finger_states: tuple = ()


def classify_geometry(feature,prefer_image=False):
    """Three-valued anatomical reach: open, folded, or deadband; no MLP input."""
    if not feature.present:return GeometryObservation('no_hand',0.,np.inf,'absent')
    world=getattr(feature,'world_points',None)
    if prefer_image:
        try:
            points=np.asarray(getattr(feature,'image_points',None),dtype=float)
            size=np.asarray(getattr(feature,'frame_size',None),dtype=float)
        except (ValueError,TypeError):return GeometryObservation('uncertain',0.,np.inf,'invalid_image')
        if points.shape!=(21,2) or not np.isfinite(points).all() or np.any(points<-.2) or np.any(points>1.2) or size.shape!=(2,) or not np.isfinite(size).all() or np.any(size<=0):
            return GeometryObservation('uncertain',0.,np.inf,'invalid_image')
        points=points.copy();points[:,0]*=size[0]/size[1]
        source='image_aspect_corrected';confidence=.90
    elif world is not None:
        points=np.asarray(world,dtype=float);source='world';confidence=.98
        if points.shape!=(21,3) or not np.isfinite(points).all():return GeometryObservation('uncertain',0.,np.inf,'invalid_world')
    else:
        points=np.asarray(feature.pose,dtype=float).reshape(21,2).copy()
        size=getattr(feature,'frame_size',None)
        if size is not None and len(size)==2 and min(size)>0:
            points[:,0]*=size[0]/size[1];source='xy_aspect_corrected';confidence=.90
        else:source='xy_aspect_unknown';confidence=.78
    scale=float(np.linalg.norm(points[9]-points[0]))
    if not np.isfinite(points).all() or scale<1e-6:return GeometryObservation('uncertain',0.,np.inf,'invalid_image' if prefer_image else source)
    states=[]
    for base in [5,9,13,17]:
        joints=points[base:base+4];chain=float(np.linalg.norm(np.diff(joints,axis=0),axis=1).sum())
        reach=float(np.linalg.norm(joints[-1]-joints[0]))/max(chain,1e-9)
        states.append('open' if reach>=.88 else 'folded' if reach<=.62 else 'uncertain')
    pinch=float(np.linalg.norm(points[4]-points[8])/scale)
    # In camera-workspace mode, a half-curled ring finger does not have to
    # cross the strict folded threshold. Middle and pinky still disambiguate
    # pointing/scroll; an extended ring remains rejected. Keep ray mode intact.
    ring_folded=states[2]=='folded' or (prefer_image and states[2]=='uncertain')
    closed_others=states[1]=='folded' and ring_folded and states[3]=='folded'
    if closed_others and pinch<.42:return GeometryObservation('pinch',confidence,pinch,source,tuple(states))
    if states[0]=='open' and closed_others:label='point'
    elif states[0:2]==['open','open'] and ring_folded and states[3]=='folded':label='victory'
    elif states==['open']*4:label='palm'
    else:label='uncertain';confidence=0.
    return GeometryObservation(label,confidence,pinch,source,tuple(states))

class HandTracker:
    def __init__(self,root):
        import mediapipe as mp
        from src.realtime_inference import load_checkpoint
        self.root=Path(root);self.mp=mp
        options=mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.root/'models'/'hand_landmarker.task')),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=1,
            min_hand_detection_confidence=.4,min_hand_presence_confidence=.4,min_tracking_confidence=.4)
        self.landmarker=mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.model=self.classes=self.mean=self.std=None
        model_path=self.root/'models'/'mlp_hagrid_6classes.pth'
        if model_path.exists():self.model,self.classes,self.mean,self.std=load_checkpoint(model_path)
        self.points=None;self._ts=-1

    def process(self,frame,t):
        import cv2
        from src.realtime_inference import predict
        ts=max(self._ts+1,int(t*1000));self._ts=ts
        img=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        result=self.landmarker.detect_for_video(img,ts)
        if not result.hand_landmarks:
            self.points=None
            absent=FrameFeatures.absent(t);absent.frame_size=(frame.shape[1],frame.shape[0]);return absent
        self.points=np.array([[p.x,p.y] for p in result.hand_landmarks[0]],dtype=np.float32)
        handedness=result.handedness[0][0].category_name
        label,confidence=('no_gesture',0.)
        if self.model is not None:label,confidence=predict(self.model,self.points.reshape(-1),handedness,self.mean,self.std,self.classes)
        f=features_from_points(self.points,handedness,t,label,confidence)
        f.image_points=self.points.copy();f.frame_size=(frame.shape[1],frame.shape[0])
        if result.hand_world_landmarks:
            world=np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[0]],dtype=np.float32)
            if np.isfinite(world).all():f.world_points=world
        if f.world_points is not None:
            f.label=guard_discrete_label(f.label,f.world_points)
            if f.label=='no_gesture':f.confidence=0.
        distances=np.linalg.norm(self.points-self.points[0],axis=1)
        extended=[distances[tip]>distances[pip]*1.18 for tip,pip in [(8,6),(12,10),(16,14),(20,18)]]
        if f.world_points is not None:
            from .personal import anatomy
            reaches=anatomy(f.world_points)[::3]
            extended=list(reaches[1:]>.78)
        if extended[0] and not any(extended[1:]):f.label='point';f.confidence=.98
        elif extended[0] and extended[1] and not any(extended[2:]):f.label='victory';f.confidence=max(confidence,.95)
        # Pinch is an analog control state; do not discard it because static MLP has no pinch class.
        control_pinch=f.pinch if f.world_points is None else float(np.linalg.norm(f.world_points[4]-f.world_points[8])/max(np.linalg.norm(f.world_points[9]-f.world_points[0]),.005))
        geometry=classify_geometry(f)
        f.geometry_label=geometry.label;f.geometry_confidence=geometry.confidence;f.geometry_pinch=geometry.pinch
        return f

    def annotate(self,frame,f):
        import cv2
        if self.points is not None:
            h,w=frame.shape[:2];valid=np.isfinite(self.points).all(axis=1)
            pixels=np.zeros((21,2),int);pixels[valid]=(self.points[valid]*np.array([w,h])).astype(int)
            for a,b in HAND_CONNECTIONS:
                if valid[a] and valid[b]:cv2.line(frame,tuple(pixels[a]),tuple(pixels[b]),(100,220,120),2)
            for p in pixels[valid]:cv2.circle(frame,tuple(p),3,(70,180,255),-1)
        return frame

    def close(self):
        self.landmarker.close()
