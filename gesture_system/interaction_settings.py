"""Validated user-facing tuning, independent of Qt and camera models."""
from dataclasses import dataclass,asdict,fields
import math

BOUNDS={'pointer_sensitivity':(.3,3.),'smoothing':(0,100),'scroll_gain':(.5,5.),
        'pose_confirm_ms':(30,160),'pointer_acquire_ms':(60,300),'swipe_distance':(.04,.20),
        'pinch_press':(.16,.32),'drag_hold_ms':(250,1000)}

@dataclass(frozen=True)
class InteractionSettings:
    pointer_sensitivity:float=1.
    smoothing:float=70.
    scroll_gain:float=2.2
    pose_confirm_ms:float=60.
    pointer_acquire_ms:float=100.
    swipe_distance:float=.08
    pinch_press:float=.24
    drag_hold_ms:float=450.
    allow_drag:bool=True
    def __post_init__(self):
        for name,(low,high) in BOUNDS.items():
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Invalid interaction setting: '+name)
        if not isinstance(self.allow_drag,bool):raise ValueError('Invalid drag setting')
    def to_dict(self):return asdict(self)
    @classmethod
    def from_dict(cls,raw):
        if not isinstance(raw,dict):return cls()
        defaults=cls().to_dict();values={}
        for field in fields(cls):
            name=field.name;value=raw.get(name,defaults[name])
            try:cls(**{name:value});values[name]=value
            except (ValueError,TypeError):values[name]=defaults[name]
        return cls(**values)
    def apply(self,controller,mapper=None):
        controller.filter.min_cutoff=.8+(100-self.smoothing)/30
        controller.pose_confirm=self.pose_confirm_ms/1000
        controller.pointer_acquire=self.pointer_acquire_ms/1000
        controller.swipe_distance=self.swipe_distance
        controller.pinch_press=self.pinch_press;controller.pinch_release=round(self.pinch_press+.10,6)
        controller.drag_hold=self.drag_hold_ms/1000;controller.allow_drag=self.allow_drag
        if mapper is not None:mapper.sensitivity=self.pointer_sensitivity
