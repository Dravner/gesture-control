import json
import numpy as np
import pytest

def targets():return np.array([(x,y) for y in [.1,.5,.9] for x in [.1,.5,.9]])

def rays_for(target,origin=None):
    from gesture_system.pointing import FingerRay
    W,H=.3442447,.2225239
    return [FingerRay(np.array([.02,.12,.5]) if origin is None else np.asarray(origin),
                      np.array([W*(.5-x),H*y,0])-(np.array([.02,.12,.5]) if origin is None else np.asarray(origin))) for x,y in target]

def test_metric_fit_persists_and_independent_origin_changes(tmp_path):
    from gesture_system.pointing import PointingCalibration, validation_errors
    q=targets();model=PointingCalibration.fit(rays_for(q),q,{'camera':0,'frame_size':[640,480]})
    assert model.quality['training_rmse_normalized']<1e-4
    check=np.array([[.2,.25],[.72,.8]])
    error=validation_errors(model,rays_for(check,[-.04,.07,.65]),check,(1728,1117))
    assert error['rmse_pixels']<1
    path=tmp_path/'fit.json';model.save(path);loaded=PointingCalibration.load(path)
    assert np.allclose(loaded.apply(rays_for(check)[0]),check[0],atol=1e-4)
    assert loaded.matches_context({'camera':0,'frame_size':[640,480]})
    assert not loaded.matches_context({'camera':1,'frame_size':[640,480]})
    original=path.read_bytes()
    with pytest.raises(ValueError):PointingCalibration.fit(rays_for(q),np.zeros((9,2)),{}).save(path)
    assert path.read_bytes()==original

def test_parallel_backward_and_gross_outside_are_rejected():
    from gesture_system.pointing import PointingCalibration,FingerRay
    model=PointingCalibration.fit(rays_for(targets()),targets(),{})
    for direction in [[1,0,0],[0,0,1],[1,0,-.01]]:
        with pytest.raises(ValueError):model.apply(FingerRay([0,.1,.5],direction))

def test_bad_intrinsics_and_bad_fit_fail():
    from gesture_system.pointing import CameraIntrinsics,PointingCalibration
    with pytest.raises(ValueError):CameraIntrinsics.from_fov(640,480,180)
    with pytest.raises(ValueError):CameraIntrinsics(np.zeros((3,3)),640,480)
    bad=targets().copy();bad[::2]=bad[::-2]
    with pytest.raises(ValueError):PointingCalibration.fit(rays_for(targets()),bad,{})

def test_pnp_ray_recovers_camera_pose_not_fingertip_xy():
    import cv2
    from gesture_system.pointing import CameraIntrinsics,estimate_finger_ray
    from gesture_system.types import FrameFeatures
    K=CameraIntrinsics.from_fov(640,480)
    world=np.random.default_rng(4).normal(0,.03,(21,3));world[5]=[.01,.02,.06];world[6]=[.01,.02,.04];world[7]=[.01,.02,0];world[8]=[.01,.02,-.04]
    rotation=np.array([.1,-.2,.07]);translation=np.array([.02,.01,.5]);R=cv2.Rodrigues(rotation)[0]
    image=cv2.projectPoints(world,rotation,translation,K.matrix,None)[0].reshape(21,2)/[640,480]
    feature=FrameFeatures(1,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'point',.99,world,
                          image_points=image,frame_size=(640,480))
    ray=estimate_finger_ray(feature,K)
    assert ray is not None
    assert np.allclose(ray.origin,R@world[8]+translation,atol=1e-4)
    assert np.allclose(ray.direction,R@np.array([0,0,-1]),atol=1e-4)
    assert ray.quality['reprojection_rmse_px']<.1
    # A different finger axis with the same origin changes the screen intersection.
    from gesture_system.pointing import PointingCalibration,FingerRay
    model=PointingCalibration.fit(rays_for(targets()),targets(),{})
    a=FingerRay([0,.1,.5],[0,0,-1]);b=FingerRay([0,.1,.5],[.08,0,-1])
    assert np.linalg.norm(model.apply(a)-model.apply(b))>.1

def test_timestamp_filter_reduces_jitter_and_has_finite_lag():
    from gesture_system.pointing import TimeAwarePointerFilter
    f=TimeAwarePointerFilter();rng=np.random.default_rng(2);raw=[];smooth=[]
    for t in np.arange(0,2,.02):
        p=np.array([.5,.5])+rng.normal(0,.005,2);raw.append(p);smooth.append(f.update(p,float(t)))
    assert np.std(np.asarray(smooth)[10:,0])<np.std(np.asarray(raw)[10:,0])*.8
    before=np.asarray(smooth)[-1];after=f.update([.8,.5],2.01)
    assert before[0]<after[0]<.8
