"""Personal shape recognition and causal motion episodes, independent of MLP labels.

Static distances use finger articulation, not screen orientation. Dynamic matching
preserves articulation order, palm rotation and wrist direction. Stored legacy
examples remain readable; no test recordings are added to enrollment implicitly.
"""
from dataclasses import dataclass
import numpy as np
from .profiles import _resample

CHAINS=((1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20))

def anatomy(points):
    p=np.asarray(points,dtype=float).reshape(21,-1)
    values=[]
    for chain in CHAINS:
        q=p[list(chain)];bones=np.diff(q,axis=0);length=np.maximum(np.linalg.norm(bones,axis=1),1e-8)
        values.extend([np.linalg.norm(q[-1]-q[0])/length.sum(),
                       np.dot(bones[0],bones[1])/(length[0]*length[1]),
                       np.dot(bones[1],bones[2])/(length[1]*length[2])])
    return np.clip(np.asarray(values,dtype=np.float32),-1,1)

def shape_descriptor(points):
    p=np.asarray(points,dtype=float).reshape(21,-1)
    tips=p[[4,8,12,16,20]]
    finger_length=np.median([np.linalg.norm(np.diff(p[list(c)],axis=0),axis=1).sum() for c in CHAINS])
    distances=np.linalg.norm(tips[:,None]-tips[None,:],axis=2)[np.triu_indices(5,1)]
    return np.r_[anatomy(p),np.clip(distances/max(finger_length*2,1e-8),0,1)].astype(np.float32)

def shape_sequence(rows):return np.stack([shape_descriptor(row[:42].reshape(21,2)) for row in rows])

def orientation(points):
    p=np.asarray(points).reshape(21,-1);axis=p[9]-p[0];axis/=max(np.linalg.norm(axis),1e-8)
    if p.shape[1]==2:return axis
    normal=np.cross(p[5]-p[17],axis);normal/=max(np.linalg.norm(normal),1e-8)
    return np.r_[axis,normal]

def dynamic_embedding(rows,n=48,world_rows=None,trajectory_weight=.3):
    x=_resample(rows,n);pose=x[:,:42].reshape(n,21,2)
    # One global alignment preserves rotations inside the sequence.
    phi=np.arctan2(pose[:,9,1],pose[:,9,0]);initial=float(phi[0])
    rotation=np.array([[np.cos(initial),-np.sin(initial)],[np.sin(initial),np.cos(initial)]])
    aligned=pose@rotation
    orient=np.column_stack([np.cos(phi-initial),np.sin(phi-initial)])
    scale=max(float(np.median(x[:,47])),.05)
    wrist=(x[:,42:44]-x[0,42:44])/scale
    if world_rows is not None:
        world=_resample(np.asarray(world_rows).reshape(len(world_rows),63),n).reshape(n,21,3)
        world=world-world[:,0:1]
        world/=np.maximum(np.linalg.norm(world[:,9],axis=1),.005)[:,None,None]
        y=world[0,9];y=y/max(np.linalg.norm(y),1e-8)
        axis=world[0,5]-world[0,17];axis-=np.dot(axis,y)*y;axis/=max(np.linalg.norm(axis),1e-8)
        z=np.cross(axis,y);basis=np.column_stack([axis,y,z])
        canonical=world@basis
        normals=np.cross(world[:,5]-world[:,17],world[:,9]);normals/=np.maximum(np.linalg.norm(normals,axis=1),1e-8)[:,None]
        return np.column_stack([canonical.reshape(n,63)*.3,np.stack([shape_descriptor(p) for p in world])*.5,normals@basis*.5,wrist*trajectory_weight])
    return np.column_stack([aligned.reshape(n,42)*.3,shape_sequence(x)*.5,orient*.5,wrist*trajectory_weight])

def motion_amount(rows,world_rows=None,trajectory_weight=.3):
    # Fixed resampling bounds noise accumulation independently of capture FPS.
    e=dynamic_embedding(rows,32,world_rows,trajectory_weight)
    return float(np.linalg.norm(np.diff(e,axis=0),axis=1).sum())

def rotation_travel(rows,world_rows=None):
    """Observed orientation travel, not a claim about physical wrist angle."""
    if world_rows is not None:
        p=_resample(np.asarray(world_rows).reshape(-1,63),64).reshape(64,21,3)
        directions=np.cross(p[:,5]-p[:,17],p[:,9]-p[:,0])
    else:
        p=_resample(rows,64)[:,:42].reshape(64,21,2);directions=p[:,9]-p[:,0]
    directions/=np.maximum(np.linalg.norm(directions,axis=1),1e-8)[:,None]
    smooth=[directions[0]]
    for v in directions[1:]:
        next_value=.5*v+.5*smooth[-1];smooth.append(next_value/max(np.linalg.norm(next_value),1e-8))
    smooth=np.asarray(smooth)
    return float(np.arccos(np.clip(np.sum(smooth[1:]*smooth[:-1],axis=1),-1,1)).sum()),float(np.arccos(np.clip(np.dot(smooth[0],smooth[-1]),-1,1)))

def dtw(a,b):
    local=np.sqrt(np.mean((a[:,None]-b[None,:])**2,axis=2))
    n,m=local.shape;cost=np.full((n+1,m+1),np.inf);cost[0,0]=0
    for i in range(1,n+1):
        for j in range(max(1,i-10),min(m,i+10)+1):
            cost[i,j]=float(local[i-1,j-1])+min(cost[i-1,j-1],cost[i-1,j],cost[i,j-1])
    return float(cost[n,m]/max(n,m))

def calibrated_radius(gesture):
    """Positive enrollment dispersion only; never consult trials or live matches."""
    examples=[np.asarray(t,dtype=np.float32) for t in gesture['templates']]
    worlds=[np.asarray(w,dtype=np.float32).reshape(-1,21,3) if w is not None else None for w in gesture.get('world_templates',[None]*len(examples))]
    if gesture['kind']=='static':
        shapes=[np.stack([shape_descriptor(p) for p in w]) if w is not None else shape_sequence(x) for x,w in zip(examples,worlds)]
        noise=max(float(np.percentile(np.sqrt(np.mean((s-np.median(s,axis=0))**2,axis=1)),95)) for s in shapes)
        return float(np.clip(2*noise+.035,.07,.18))
    if len(examples)<2:return .10
    groups=[(x,w) for x,w in zip(examples,worlds) if w is not None] if any(w is not None for w in worlds) else list(zip(examples,worlds))
    if len(groups)<2:return .10
    signatures=[dynamic_embedding(x,world_rows=w,trajectory_weight=.04) for x,w in groups]
    nearest=[min(dtw(a,b) for j,b in enumerate(signatures) if i!=j) for i,a in enumerate(signatures)]
    return float(np.clip(np.percentile(nearest,90)*1.5+.025,.08,.18))

@dataclass(frozen=True)
class Match:
    identifier:str
    confidence:float
    distance:float
    radius:float
    kind:str

@dataclass(frozen=True)
class Decision:
    match:Match|None=None
    phase:str='idle'
    reserved:bool=False
    distance:float|None=None
    threshold:float|None=None
    progress:float=0.

class PersonalRecognizer:
    def __init__(self,library):
        self.library=library;self.models=[]
        for g in library.list_gestures():
            if g['kind'] not in {'static','dynamic'} or not g.get('enabled',True):continue
            templates=[np.asarray(t,dtype=np.float32) for t in g['templates']]
            shapes=[shape_sequence(t) for t in templates]
            prototype=[np.median(s,axis=0) for s in shapes]
            world=[np.asarray(w,dtype=np.float32).reshape(-1,21,3) if w is not None else None for w in g.get('world_templates',[None]*len(templates))]
            world_shapes=[np.stack([shape_descriptor(p) for p in w]) if w is not None else None for w in world]
            world_prototypes=[np.median(s,axis=0) for s in world_shapes if s is not None]
            if g['kind']=='static' and all(np.linalg.norm(t[0,18:20]-t[0,:2])<.01 for t in templates):continue
            dispersion=max(float(np.percentile(np.sqrt(np.mean((s-p)**2,axis=1)),95)) for s,p in zip(shapes,prototype))
            radius=float(g.get('personal_radius',max(.07,min(.18,dispersion*2+.035)))) if g['kind']=='static' else float(g.get('personal_radius',.10))
            self.models.append(dict(g=g,templates=templates,shapes=shapes,prototypes=prototype,radius=radius,motions=[motion_amount(t) for t in templates],world=world,world_shapes=world_shapes,world_prototypes=world_prototypes))
        self.last_distance=None;self.last_radius=None;self.reset()
        self.last_reason='ожидание'

    def reset(self):
        self.rows=[];self.world_rows=[];self.times=[];self.prev=None;self.last_seen=None
        self.last_motion=None;self.moving=False;self.last_check=-np.inf;self.phase='idle'
        self.reserved=False

    def static_match(self,pose,world_points=None):
        q=shape_descriptor(np.asarray(pose).reshape(21,2));candidates=[]
        self.last_distance=None;self.last_radius=None
        for m in self.models:
            if m['g']['kind']!='static':continue
            use_world=world_points is not None and m['world_prototypes']
            query=shape_descriptor(world_points) if use_world else q
            prototypes=m['world_prototypes'] if use_world else m['prototypes']
            distance=min(float(np.sqrt(np.mean((query-p)**2))) for p in prototypes)
            if m['g'].get('orientation_sensitive',False):
                query_orientation=orientation(world_points if use_world else np.asarray(pose).reshape(21,2))
                enrolled=m['world'] if use_world else [t[:,:42].reshape(-1,21,2) for t in m['templates']]
                angles=[np.mean([orientation(p) for p in points],axis=0) for points in enrolled if points is not None]
                distance=max(distance,min(float(np.sqrt(np.mean((query_orientation-a)**2)))*.25 for a in angles))
            candidates.append((distance/m['radius'],distance,m))
        if not candidates:self.last_reason='нет подходящего статического примера';return None
        candidates.sort(key=lambda v:v[0]);ratio,d,m=candidates[0]
        self.last_distance=d;self.last_radius=m['radius']
        if ratio>1:self.last_reason='форма вне допуска';return None
        if len(candidates)>1 and candidates[1][0]-ratio<.15:self.last_reason='неоднозначная форма';return None
        self.last_reason='форма совпала'
        return Match(m['g']['id'],max(.5,1-.5*ratio),d,m['radius'],'static')

    def classify_dynamic(self,rows,world_rows=None):
        rows=np.asarray(rows,dtype=np.float32)
        self.last_partial=False;self.partial_progress=0.
        self.last_distance=None;self.last_radius=None
        if len(rows)<8:self.last_reason='мало кадров движения';return None
        q=dynamic_embedding(rows);motion=motion_amount(rows);candidates=[]
        for m in self.models:
            if m['g']['kind']!='dynamic':continue
            for template,amount,world in zip(m['templates'],m['motions'],m['world']):
                if world_rows is not None and any(w is not None for w in m['world']) and world is None:continue
                use_world=world_rows is not None and world is not None
                extent=float(np.max(np.ptp(template[:,42:44],axis=0))/max(float(np.median(template[:,47])),.05))
                articulation=motion_amount(template,world if use_world else None,0.)
                weight=.4 if extent>.8 or (extent>.15 and articulation<.3) else .04
                tmotion=motion_amount(template,world if use_world else None,weight)
                qmotion=motion_amount(rows,world_rows if use_world else None,weight)
                if tmotion<.15:continue
                fraction=qmotion/tmotion
                if .18<=fraction<.72:
                    stride=max(1,len(rows)//8)
                    qs=np.stack([shape_descriptor(p) for p in world_rows[::stride]]) if use_world else shape_sequence(rows[::stride])
                    bank=np.stack([shape_descriptor(p) for p in world]) if use_world else shape_sequence(template)
                    shape_distance=float(np.sqrt(np.mean((qs[:,None]-bank[None,:])**2,axis=2)).min(axis=1).mean())
                    if shape_distance<.12:self.last_partial=True;self.partial_progress=max(self.partial_progress,fraction)
                if not .72<=fraction<=1.65:continue
                enrolled_travel,enrolled_end=rotation_travel(template,world if use_world else None)
                observed_travel,observed_end=rotation_travel(rows,world_rows if use_world else None)
                if enrolled_travel>4.7 and (not .80<=observed_travel/enrolled_travel<=1.25 or abs(observed_end-enrolled_end)>.6):continue
                d=dtw(dynamic_embedding(rows,world_rows=world_rows if use_world else None,trajectory_weight=weight),dynamic_embedding(template,world_rows=world if use_world else None,trajectory_weight=weight));candidates.append((d/m['radius'],d,m))
        if not candidates:self.last_reason='движение неполное или отличается от примера';return None
        candidates.sort(key=lambda v:v[0]);ratio,d,m=candidates[0]
        self.last_distance=d;self.last_radius=m['radius']
        other=[c for c in candidates[1:] if c[2]['g']['id']!=m['g']['id']]
        if ratio>1:self.last_reason='порядок движения вне допуска';return None
        if other and other[0][0]-ratio<.15:self.last_reason='неоднозначное движение';return None
        self.last_reason='полное движение совпало'
        return Match(m['g']['id'],max(.5,1-.5*ratio),d,m['radius'],'dynamic')

    def _finish(self,t=None,retain_partial=False):
        rows=np.asarray(self.rows)
        world=np.asarray(self.world_rows) if self.world_rows and all(w is not None for w in self.world_rows) else None
        if not self.moving:self.last_distance=None;self.last_radius=None;self.last_reason='нет законченного движения'
        match=self.classify_dynamic(rows,world) if self.moving and len(rows)>=8 else None
        if retain_partial and not match and getattr(self,'last_partial',False) and t-self.last_motion<1.:
            return Decision(None,'partial',self.reserved,self.last_distance,self.last_radius,self.partial_progress)
        distance=self.last_distance;radius=self.last_radius
        self.reset()
        return Decision(match,'recognized' if match else 'rejected',False,distance,radius,1. if match else 0.)

    def update(self,feature):
        t=float(feature.timestamp)
        if not feature.present:
            if self.last_seen is not None and t-self.last_seen>=.4-1e-9:return self._finish()
            return Decision(phase='tracking_gap' if self.rows else 'idle',reserved=self.reserved)
        v=feature.vector()
        if self.last_seen is not None and t-self.last_seen>.5:self.reset()
        self.last_seen=t
        # 20Hz wall-clock samples: no accumulation proportional to webcam FPS.
        if self.times and t-self.times[-1]<.05-1e-6:return Decision(phase=self.phase,reserved=self.reserved)
        if self.prev is not None:
            dt=max(t-self.times[-1],.01)
            alpha=1-np.exp(-dt/.06);v[:46]=self.prev[:46]+alpha*(v[:46]-self.prev[:46])
            step=max(float(np.sqrt(np.mean((v[:42]-self.prev[:42])**2))),float(np.linalg.norm(v[42:44]-self.prev[42:44])/max(v[47],.05)))/dt
            world=getattr(feature,'world_points',None)
            if world is not None and self.world_rows and self.world_rows[-1] is not None:
                a=world-world[0];b=self.world_rows[-1]-self.world_rows[-1][0]
                scale=max(float(np.linalg.norm(a[9])),.005)
                step=max(step,float(np.sqrt(np.mean((a-b)**2)))/scale/dt)
            if step>.18:self.last_motion=t;self.moving=True
        self.prev=v.copy();self.rows.append(v.copy());self.times.append(t)
        self.world_rows.append(np.array(feature.world_points,copy=True) if getattr(feature,'world_points',None) is not None else None)
        if len(self.rows)>240:self.rows.pop(0);self.times.pop(0);self.world_rows.pop(0)
        if self.moving and self.last_motion is not None and t-self.last_motion>=.3:
            return self._finish(t,retain_partial=True)
        if not self.moving and self.times and t-self.times[0]>1.:
            self.rows=self.rows[-5:];self.times=self.times[-5:];self.world_rows=self.world_rows[-5:]
        q=shape_descriptor(v[:42].reshape(21,2));compatible=False
        for m in self.models:
            world=getattr(feature,'world_points',None);banks=[s for s in m['world_shapes'] if s is not None] if world is not None else []
            query=shape_descriptor(world) if banks else q
            if m['g']['kind']=='dynamic' and min(float(np.sqrt(np.mean((s-query)**2,axis=1)).min()) for s in (banks or m['shapes']))<.12:
                compatible=True;break
        reserved=self.moving and compatible
        self.reserved=reserved
        self.phase='motion' if reserved else 'observing'
        return Decision(phase=self.phase,reserved=reserved)
