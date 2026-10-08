import numpy as np
import pytest
from gesture_system.pointing import FingerRay

def observations(sweep=False):
    rng=np.random.default_rng(42);out=[]
    for i,t in enumerate(np.linspace(0,2.2,45)):
        angle=(i/44*20 if sweep else rng.normal(0,2))
        if i in [6,25]:angle=30
        a=np.deg2rad(angle);ray=FingerRay([0,.1,.4],[np.sin(a),0,-np.cos(a)],timestamp=t)
        out.append((t,ray,{},None))
    return out

def test_robust_collection_accepts_natural_jitter_and_removes_spikes():
    from gesture_system.calibration_sampling import aggregate_rays
    ray,entries=aggregate_rays(observations(),now=2.2,min_frames=12,min_span=1.)
    assert len(entries)>=40 and len(entries)<45
    assert abs(ray.direction[0])<.04
    assert ray.quality['aggregation']=='robust_median'
    assert ray.quality['raw_angular_spread_max_degrees']>20

def test_deliberate_sweep_is_not_silently_averaged():
    from gesture_system.calibration_sampling import aggregate_rays
    with pytest.raises(ValueError,match='[Нн]аправление|разброс'):
        aggregate_rays(observations(True),now=2.2,min_frames=12,min_span=1.)

def test_missing_and_stale_frames_do_not_finish_target():
    from gesture_system.calibration_sampling import aggregate_rays
    entries=observations();entries=[(t,None,r,'нет кисти') if i%2 else (t,v,r,e) for i,(t,v,r,e) in enumerate(entries)]
    with pytest.raises(ValueError):aggregate_rays(entries,now=2.2,min_frames=12,min_span=1.)
    with pytest.raises(ValueError):aggregate_rays(observations(),now=4,min_frames=12,min_span=1.)

def test_fresh_outliers_cannot_make_old_retained_rays_fresh():
    from gesture_system.calibration_sampling import aggregate_rays
    entries=[]
    for t in np.linspace(0,1.8,40):entries.append((float(t),FingerRay([0,.1,.4],[0,0,-1],timestamp=t),{},None))
    for t in [2.,2.1,2.2]:
        a=np.deg2rad(30);entries.append((t,FingerRay([0,.1,.4],[np.sin(a),0,-np.cos(a)],timestamp=t),{},None))
    with pytest.raises(ValueError):aggregate_rays(entries,now=2.2,min_frames=12,min_span=1.)
