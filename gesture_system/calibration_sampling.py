"""Robust central ray for a timed target capture; scatter is not accuracy."""
import numpy as np
from .pointing import FingerRay

def _unit(vector):
    length=float(np.linalg.norm(vector))
    if length<1e-8:raise ValueError('Направления противоречат друг другу; повторите цель.')
    return vector/length

def _angle(a,b):return float(np.rad2deg(np.arccos(np.clip(np.dot(a,b),-1,1))))

def aggregate_rays(entries,now,min_frames=12,min_span=1.):
    recent=[e for e in entries if now-2.3<=e[0]<=now+.01]
    valid=[e for e in recent if e[1] is not None]
    if not recent or len(valid)<min_frames or len(valid)/len(recent)<.6 or now-valid[-1][0]>.25 or valid[-1][0]-valid[0][0]<min_span:
        reason=next((e[3] for e in reversed(recent) if e[3]),None)
        raise ValueError(reason or 'Мало наблюдений кисти. Держите её в кадре и повторите сбор.')
    directions=np.array([e[1].direction for e in valid]);origins=np.array([e[1].origin for e in valid])
    center=_unit(np.median(directions,axis=0))
    angles=np.rad2deg(np.arccos(np.clip(directions@center,-1,1)))
    median=float(np.median(angles));mad=float(np.median(np.abs(angles-median)))
    threshold=min(12.,max(4.,median+3*1.4826*mad))
    mask=angles<=threshold
    inliers=[e for e,keep in zip(valid,mask) if keep]
    if len(inliers)<min_frames or mask.mean()<.7 or inliers[-1][0]-inliers[0][0]<min_span or now-inliers[-1][0]>.25:raise ValueError('Слишком большой разброс направления или нет свежего устойчивого луча. Повторите сбор без движения между целями.')
    dirs=directions[mask];points=origins[mask];direction=_unit(np.median(dirs,axis=0));origin=np.median(points,axis=0)
    central_angles=np.rad2deg(np.arccos(np.clip(dirs@direction,-1,1)))
    middle=(inliers[0][0]+inliers[-1][0])/2
    early=[e[1] for e in inliers if e[0]<=middle];late=[e[1] for e in inliers if e[0]>middle]
    if not early or not late:raise ValueError('Недостаточно длительный сбор; повторите цель.')
    drift=_angle(_unit(np.median([r.direction for r in early],axis=0)),_unit(np.median([r.direction for r in late],axis=0)))
    movement=float(np.linalg.norm(np.median([r.origin for r in early],axis=0)-np.median([r.origin for r in late],axis=0)))
    if np.percentile(central_angles,80)>8 or drift>6 or movement>.04 or np.percentile(np.linalg.norm(points-origin,axis=1),90)>.06:
        raise ValueError(f'Направление заметно меняется (дрейф {drift:.1f}°). Удобно обоприте локоть и повторите сбор.')
    quality={'aggregation':'robust_median','valid_frames':len(inliers),'candidate_frames':len(valid),'total_frames':len(recent),
             'coverage':len(valid)/len(recent),'excluded_outliers':len(valid)-len(inliers),'span_seconds':inliers[-1][0]-inliers[0][0],
             'raw_angular_spread_max_degrees':float(angles.max()),'central_angular_p80_degrees':float(np.percentile(central_angles,80)),
             'angular_drift_degrees':drift,'origin_drift_m':movement,'angular_spread_max_degrees':float(central_angles.max())}
    return FingerRay(origin,direction,quality,timestamp=inliers[-1][0]),inliers
