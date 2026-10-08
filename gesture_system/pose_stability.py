"""Hand-neutral image posture with separate engage/hold evidence per finger.

No coordinate extrapolation. Low confidence is unknown; only a recently known
finger state may bridge it briefly. This is separate from raw tracker output.
"""
import numpy as np
from .vision import GeometryObservation

class ImagePoseLatch:
    def __init__(self):self.reset()
    def reset(self):
        self.states=['uncertain']*4;self.seen=[-float('inf')]*4;self.clock=-float('inf')
    def update(self,f):
        t=float(f.timestamp)
        if t-self.clock>.3:self.reset()
        self.clock=t
        if not f.present:return GeometryObservation('no_hand',0.,np.inf,'absent')
        p=np.asarray(getattr(f,'image_points',None),dtype=float)
        if p.shape!=(21,2):
            return GeometryObservation('uncertain',0.,np.inf,'invalid_image')
        usable=np.isfinite(p).all(axis=1) & ((p>=-.2)&(p<=1.2)).all(axis=1)
        p=p.copy();size=getattr(f,'frame_size',None)
        if size:p[:,0]*=size[0]/size[1]
        scale=np.linalg.norm(p[9]-p[0])
        if scale<1e-6:return GeometryObservation('uncertain',0.,np.inf,'invalid_image')
        confidence=getattr(f,'joint_confidence',None)
        confidence=np.ones(21) if confidence is None else np.asarray(confidence)
        if confidence.shape!=(21,) or not np.isfinite(confidence).all():
            return GeometryObservation('uncertain',0.,np.inf,'invalid_image')
        if not np.all(usable | (confidence<.25)) or not np.all(usable[[0,5,8,9]]):
            return GeometryObservation('uncertain',0.,np.inf,'invalid_image')
        for i,base in enumerate([5,9,13,17]):
            ids=[base,base+1,base+2,base+3]
            if np.min(confidence[ids])<.25 or not np.all(usable[ids]):
                # Hidden curled tips need not satisfy the *extended* finger
                # confidence floor: visible knuckle/PIP plus a short tip extent
                # provide independent closed evidence. Never use this to open.
                short=np.all(usable[[base,base+3]]) and np.linalg.norm(p[base+3]-p[base])/scale<=.25
                if min(confidence[base],confidence[base+1])>=.25 and confidence[base+3]>=.10 and short:
                    self.states[i]='folded';self.seen[i]=t
                elif t-self.seen[i]>.18:self.states[i]='uncertain'
                continue
            joints=p[base:base+4];chain=np.linalg.norm(np.diff(joints,axis=0),axis=1).sum()
            extent=np.linalg.norm(joints[-1]-joints[0])/scale
            reach=np.linalg.norm(joints[-1]-joints[0])/max(chain,1e-9)
            radial=np.linalg.norm(joints[-1]-p[0])/max(np.linalg.norm(joints[1]-p[0]),1e-9)
            opened=reach>=.86 and extent>=.45 and radial>=1.08
            folded=reach<=.65 or extent<=.30 or radial<=1.02
            if opened:self.states[i]='open'
            elif folded:self.states[i]='folded'
            elif self.states[i]=='open' and not (reach>=.76 and extent>=.35 and radial>=1.03):self.states[i]='uncertain'
            # A measured deadband retains the finger's last state, unlike a
            # missing joint, whose hold is explicitly bounded above.
            self.seen[i]=t
        states=tuple(self.states)
        pinch=float(np.linalg.norm(p[4]-p[8])/scale) if min(confidence[4],confidence[8])>=.4 and np.all(usable[[4,8]]) else np.inf
        if states[1:] == ('folded',)*3 and pinch<.42:label='pinch'
        elif states==('open','folded','folded','folded'):label='point'
        elif states==('open','open','folded','folded'):label='victory'
        elif states==('open','open','open','folded'):label='navigation'
        elif states==('open',)*4:label='palm'
        else:label='uncertain'
        return GeometryObservation(label,.9 if label!='uncertain' else 0.,pinch,'image_temporal',states)
