import numpy as np
import pytest

def test_native_conversion_releases_python_pixel_buffer(monkeypatch):
    """Real CoreGraphics bridge; no camera, GPU request or RSS timing assumption."""
    import sys,gc
    if sys.platform!='darwin':pytest.skip('CoreGraphics ownership regression')
    import Quartz,objc,cv2
    from types import SimpleNamespace
    from gesture_system.apple_vision import NativeVisionHands
    buffers=[]
    class Pixels(np.ndarray):
        def tobytes(self,*args,**kwargs):
            result=super().tobytes(*args,**kwargs);buffers.append(result);return result
    conversion=cv2.cvtColor
    monkeypatch.setattr(cv2,'cvtColor',lambda *args:conversion(*args).view(Pixels))
    class Handler:
        @classmethod
        def alloc(cls):return cls()
        def initWithCGImage_options_(self,image,options):self.image=image;return self
        def performRequests_error_(self,requests,error):return True,None
    native=NativeVisionHands.__new__(NativeVisionHands)
    native.quartz=Quartz;native.objc=objc
    native.vision=SimpleNamespace(VNImageRequestHandler=Handler)
    native.request=SimpleNamespace(results=lambda:[])
    for _ in range(3):native.detect(np.zeros((32,32,3),np.uint8))
    gc.collect()
    # One reference in buffers, one temporary getrefcount argument. No retained
    # bridge export of the full RGBA frame may survive the synchronous request.
    assert [sys.getrefcount(blob) for blob in buffers]==[2,2,2]

class Detector:
    def __init__(self):
        self.points=np.full((21,2),.5);self.points[0]=[.5,.68];self.confidence=np.full(21,.9);self.closed=False
        for base,x in [(5,.44),(9,.5),(13,.56),(17,.61)]:
            self.points[base]=[x,.55];self.points[base+1]=[x,.52];self.points[base+2]=[x,.49]
            self.points[base+3]=[x,.46 if base==5 else .545]
        self.points[4]=[.32,.6]
    def detect(self,frame):return self.points,self.confidence,'Right'
    def close(self):self.closed=True

@pytest.mark.parametrize('hand',['Left','Right'])
def test_low_confidence_curled_fingers_do_not_discard_visible_pointer(tmp_path,hand):
    from gesture_system.apple_vision import VisionHandTracker
    detector=Detector();detector.confidence[15:17]=.12;detector.confidence[19:21]=.15
    detector.detect=lambda frame:(detector.points,detector.confidence,hand)
    tracker=VisionHandTracker(tmp_path,detector=detector)
    f=tracker.process(np.zeros((480,640,3),np.uint8),1.)
    assert f.present and f.handedness==hand
    assert np.array_equal(f.joint_confidence,detector.confidence)
    from gesture_system.live_runtime import FeatureSnapshot
    snapshot=FeatureSnapshot.copy(f)
    detector.confidence[:]=0
    assert snapshot.joint_confidence[8]==.9 and snapshot.handedness==hand
    assert not snapshot.joint_confidence.flags.writeable

def test_vision_adapter_outputs_raw_image_geometry_without_inventing_depth(tmp_path):
    from gesture_system.apple_vision import VisionHandTracker
    detector=Detector();tracker=VisionHandTracker(tmp_path,detector=detector)
    f=tracker.process(np.zeros((480,640,3),np.uint8),1.)
    assert f.present and f.frame_size==(640,480) and f.world_points is None
    assert np.allclose(f.image_points,detector.points) and np.isfinite(f.vector()).all()
    assert f.geometry_label=='point'
    detector.points[0]=[.8,.9]
    assert not np.allclose(f.image_points,detector.points) # Snapshot independent of backend storage.
    tracker.close();assert detector.closed

@pytest.mark.parametrize('bad',['low_confidence','nan','shape','outside'])
def test_vision_rejects_incomplete_or_unreliable_joints(tmp_path,bad):
    from gesture_system.apple_vision import VisionHandTracker
    detector=Detector()
    if bad=='low_confidence':detector.confidence[8]=.1
    elif bad=='nan':detector.points[8,0]=np.nan
    elif bad=='shape':detector.points=detector.points[:20]
    else:detector.points[8,0]=2.
    tracker=VisionHandTracker(tmp_path,detector=detector)
    f=tracker.process(np.zeros((480,640,3),np.uint8),1.)
    assert not f.present and f.frame_size==(640,480) and tracker.points is None
    tracker.close()

def test_backend_selection_uses_requested_factory_and_rejects_unknown(tmp_path):
    import gesture_system.apple_vision as module
    from gesture_system.tracking import make_tracker
    with pytest.MonkeyPatch.context() as patch:
        sentinel=object();patch.setattr(module,'VisionHandTracker',lambda root:sentinel)
        assert make_tracker(tmp_path,'apple_vision') is sentinel
    with pytest.raises(ValueError):make_tracker(tmp_path,'not_a_backend')

def test_native_preview_does_not_require_legacy_neural_demo_import(tmp_path,monkeypatch):
    import sys
    from gesture_system.apple_vision import VisionHandTracker
    tracker=VisionHandTracker(tmp_path,detector=Detector())
    frame=np.zeros((480,640,3),np.uint8);f=tracker.process(frame,1.)
    monkeypatch.setitem(sys.modules,'src.realtime_inference',None)
    result=tracker.annotate(frame,f)
    assert result.shape==frame.shape and np.any(result>0)
    tracker.close()

def test_missing_occluded_tip_keeps_raw_quality_without_fabricating_joint(tmp_path):
    from gesture_system.apple_vision import VisionHandTracker
    detector=Detector();detector.points[16]=[np.nan,np.nan];detector.confidence[16]=0
    f=VisionHandTracker(tmp_path,detector=detector).process(np.zeros((480,640,3),np.uint8),1.)
    assert f.present and np.isnan(f.image_points[16]).all() and f.joint_confidence[16]==0
    assert np.isfinite(f.vector()).all()
    from gesture_system.workspace_pointer import WorkspaceMapper
    mapper=WorkspaceMapper({'source_kind':'camera','frame_size':[640,480]});mapper.palm_anchor='midpoint'
    assert np.allclose(mapper.hand_point(f),(f.image_points[0]+f.image_points[9])/2)
    tracker=VisionHandTracker(tmp_path,detector=detector)
    tracker.process(np.zeros((480,640,3),np.uint8),1.)
    assert tracker.annotate(np.zeros((480,640,3),np.uint8),f).shape==(480,640,3)
