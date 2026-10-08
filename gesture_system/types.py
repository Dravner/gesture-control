from dataclasses import dataclass, field
import numpy as np

@dataclass
class FrameFeatures:
    timestamp: float
    present: bool
    pose: np.ndarray
    wrist: tuple[float,float]
    pointer: tuple[float,float]
    pinch: float
    scale: float
    label: str
    confidence: float
    world_points: np.ndarray|None = None
    image_points: np.ndarray|None = None
    frame_size: tuple[int,int]|None = None
    geometry_label: str|None = None
    geometry_confidence: float = 0.
    geometry_pinch: float|None = None
    joint_confidence: np.ndarray|None = None
    handedness: str|None = None

    def vector(self):
        return np.concatenate((np.asarray(self.pose,dtype=np.float32),self.wrist,self.pointer,[self.pinch,self.scale])).astype(np.float32)

    @classmethod
    def absent(cls,t):
        return cls(t,False,np.zeros(42,dtype=np.float32),(0.,0.),(0.,0.),1.,0.,'no_hand',0.)

@dataclass
class ControlEvent:
    gesture: str
    action: str
    timestamp: float
    payload: dict=field(default_factory=dict)
