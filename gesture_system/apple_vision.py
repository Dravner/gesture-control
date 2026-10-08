"""Optional native macOS 2D hand tracker; no camera capture or OS input here."""
from pathlib import Path
import sys
import time
import numpy as np
from .types import FrameFeatures
from .vision import HandTracker,features_from_points,classify_geometry

JOINTS=('Wrist','ThumbCMC','ThumbMP','ThumbIP','ThumbTip')+tuple(finger+joint for finger in ('Index','Middle','Ring','Little') for joint in ('MCP','PIP','DIP','Tip'))

class NativeVisionHands:
    def __init__(self):
        if sys.platform!='darwin':raise RuntimeError('Apple Vision доступен только в macOS.')
        try:
            import Vision,Quartz,objc
        except ImportError as exc:raise RuntimeError('Для Apple Vision установите pyobjc-framework-Vision той же версии, что pyobjc-core.') from exc
        self.vision=Vision;self.quartz=Quartz;self.objc=objc
        self.request=Vision.VNDetectHumanHandPoseRequest.alloc().init();self.request.setMaximumHandCount_(1)
        self.last_metrics={}
    def detect(self,frame):
        if self.request is None:raise RuntimeError('Apple Vision tracker closed')
        import cv2
        from Foundation import NSData
        v,q=self.vision,self.quartz;start=time.perf_counter()
        with self.objc.autorelease_pool():
            rgba=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA);h,w=rgba.shape[:2]
            # Copy into native-owned storage. Exporting Python bytes directly to
            # CGDataProvider leaves an extra buffer reference on this PyObjC /
            # Python build, leaking a complete RGBA frame on every request.
            blob=rgba.tobytes();data=NSData.dataWithBytes_length_(blob,len(blob))
            provider=q.CGDataProviderCreateWithCFData(data)
            image=q.CGImageCreate(w,h,8,32,w*4,q.CGColorSpaceCreateDeviceRGB(),q.kCGImageAlphaPremultipliedLast,provider,None,False,q.kCGRenderingIntentDefault)
            handler=v.VNImageRequestHandler.alloc().initWithCGImage_options_(image,{})
            conversion_ms=(time.perf_counter()-start)*1000;begin=time.perf_counter()
            ok,error=handler.performRequests_error_([self.request],None)
            if not ok:raise RuntimeError(f'Apple Vision: {error}')
            native_ms=(time.perf_counter()-begin)*1000;results=list(self.request.results() or [])
            points=None;confidence=[];handedness='Right'
            if results:
                observation=results[0];points=[]
                for name in JOINTS:
                    point,error=observation.recognizedPointForJointName_error_(getattr(v,'VNHumanHandPoseObservationJointName'+name),None)
                    if point is None:points.append([np.nan,np.nan]);confidence.append(0.)
                    else:
                        location=point.location();points.append([float(location.x),1-float(location.y)])
                        confidence.append(float(point.confidence()))
                points=np.asarray(points,dtype=float)
                if hasattr(observation,'chirality') and observation.chirality()==v.VNChiralityLeft:handedness='Left'
            self.last_metrics={'route_ms':(time.perf_counter()-start)*1000,'conversion_ms':conversion_ms,'native_request_ms':native_ms,'present':bool(results)}
            return points,np.asarray(confidence,dtype=float),handedness
    def close(self):self.request=None

class VisionHandTracker:
    backend='apple_vision'
    def __init__(self,root,detector=None,minimum_joint_confidence=.25):
        if not np.isfinite(minimum_joint_confidence) or not 0<=minimum_joint_confidence<=1:raise ValueError('Invalid joint confidence threshold')
        self.root=Path(root);self.detector=detector if detector is not None else NativeVisionHands()
        self.minimum_joint_confidence=minimum_joint_confidence;self.points=None
    def process(self,frame,t):
        if not isinstance(frame,np.ndarray) or frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8 or min(frame.shape[:2])<=0:
            raise ValueError('Vision expects a nonempty uint8 BGR frame')
        if not np.isfinite(t):raise ValueError('Invalid timestamp')
        points,confidence,handedness=self.detector.detect(frame)
        points=np.asarray(points,dtype=float);confidence=np.asarray(confidence,dtype=float)
        valid=points.shape==(21,2) and confidence.shape==(21,) and np.isfinite(confidence).all()
        valid=valid and np.all((confidence>=0)&(confidence<=1))
        usable=np.isfinite(points).all(axis=1) & ((points>=-.2)&(points<=1.2)).all(axis=1) if points.shape==(21,2) else np.zeros(21,bool)
        valid=valid and np.all(usable | (confidence<self.minimum_joint_confidence))
        # Folded fingers often occlude each other. Their confidence belongs to
        # pose checks, not a minimum-of-21 whole-hand presence veto.
        valid=valid and np.all(confidence[[0,5,8,9]]>=self.minimum_joint_confidence) and np.all(usable[[0,5,8,9]])
        if not valid:
            self.points=None;f=FrameFeatures.absent(t);f.frame_size=(frame.shape[1],frame.shape[0])
            f.handedness=handedness
            if confidence.shape==(21,):f.joint_confidence=confidence.copy()
            return f
        self.points=points.astype(np.float32,copy=True)
        # Legacy vector requires finite storage. Unknown entries use zero input coordinates
        # before normalization in that compatibility vector; raw image points and quality retain
        # their missingness and the working controller never uses the zeros.
        packed=self.points.copy();packed[~usable]=0
        f=features_from_points(packed,handedness,t,'no_gesture',0.)
        f.image_points=self.points.copy();f.frame_size=(frame.shape[1],frame.shape[0])
        f.joint_confidence=confidence.copy()
        geometry=classify_geometry(f,prefer_image=True)
        f.geometry_label=geometry.label;f.geometry_confidence=geometry.confidence;f.geometry_pinch=geometry.pinch
        f.label=geometry.label;f.confidence=geometry.confidence
        return f
    annotate=HandTracker.annotate
    def close(self):self.points=None;self.detector.close()
