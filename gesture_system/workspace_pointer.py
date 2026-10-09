"""Explicit 2D camera-workspace pointer; this is never a physical finger ray.

Only raw image landmarks are used. World-coordinate rotations and PnP camera
models have no role here. The caller must separately authorize the optional mode.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import numpy as np


def _finite_vector(value,size,name):
    try:array=np.asarray(value,dtype=float)
    except (TypeError,ValueError) as exc:raise ValueError(f'Invalid {name}') from exc
    if array.shape!=(size,) or not np.isfinite(array).all():raise ValueError(f'Invalid {name}')
    return array


@dataclass(frozen=True)
class WorkspaceBounds:
    center: tuple = (.5,.5)
    span: tuple = (.5,.5)

    def __post_init__(self):
        center=_finite_vector(self.center,2,'workspace center')
        span=_finite_vector(self.span,2,'workspace span')
        if np.any(span<.15) or np.any(span>.9):raise ValueError('Workspace span must be between .15 and .9')
        if np.any(center-span/2<0) or np.any(center+span/2>1):raise ValueError('Workspace must lie inside the normalized image')
        object.__setattr__(self,'center',tuple(float(x) for x in center))
        object.__setattr__(self,'span',tuple(float(x) for x in span))


class WorkspaceMapper:
    kind='camera_workspace_2d'
    version=2
    palm_indices=(0,5,9,13,17)

    def __init__(self,context,bounds=None,relative=False):
        if not isinstance(context,dict):raise ValueError('Workspace requires camera context')
        size=_finite_vector(context.get('frame_size'),2,'frame context')
        if np.any(size<=0) or np.any(size!=np.floor(size)):raise ValueError('Frame dimensions must be positive integers')
        if context.get('source_kind')!='camera':raise ValueError('Workspace requires camera source')
        try:json.dumps(context,allow_nan=False)
        except (TypeError,ValueError) as exc:raise ValueError('Context must be finite JSON data') from exc
        if not isinstance(relative,bool):raise ValueError('Relative mode must be boolean')
        self.relative=relative;self.palm_anchor='median';self.sensitivity=1.;self.reset_pointer()
        self.context=deepcopy(context)
        self.bounds=WorkspaceBounds() if bounds is None else bounds
        if not isinstance(self.bounds,WorkspaceBounds):raise ValueError('Expected WorkspaceBounds')

    def matches_context(self,context):
        return self.context==context

    def _palm_center(self,feature):
        if not getattr(feature,'present',False):raise ValueError('No tracked hand')
        size=_finite_vector(getattr(feature,'frame_size',None),2,'frame context')
        if not np.array_equal(size,np.asarray(self.context['frame_size'])):raise ValueError('Camera frame context changed')
        try:points=np.asarray(getattr(feature,'image_points',None),dtype=float)
        except (ValueError,TypeError) as exc:raise ValueError('Invalid image landmarks') from exc
        if points.shape!=(21,2):raise ValueError('Expected raw image landmarks (21,2)')
        if self.palm_anchor=='midpoint':
            anchors=points[[0,9]]
            if not np.isfinite(anchors).all() or np.any(anchors<-.2) or np.any(anchors>1.2):raise ValueError('Invalid palm anchor')
            return anchors.mean(axis=0)
        if not np.isfinite(points).all():raise ValueError('Expected finite raw image landmarks (21,2)')
        # Small off-image landmark excursions are possible; gross invalid coordinates reject.
        if np.any(points<-.2) or np.any(points>1.2):raise ValueError('Image landmarks are grossly outside normalized coordinates')
        return np.median(points[list(self.palm_indices)],axis=0)

    def hand_point(self,feature):
        """Raw normalized palm median, with the same frame/image safety checks."""
        return self._palm_center(feature)

    def reset_pointer(self):
        self._reference_hand=None;self._reference_position=None

    def acquisition_point(self,feature):
        return self.hand_point(feature)/self.bounds.span

    def begin_pointer(self,feature,position=None):
        hand=self.hand_point(feature)
        anchor=np.array([.5,.5]) if position is None else _finite_vector(position,2,'screen anchor')
        if np.any(anchor<0) or np.any(anchor>1):raise ValueError('Screen anchor must be normalized')
        self._reference_hand=hand.copy();self._reference_position=anchor.copy()
        return anchor.copy()

    def map_feature(self,feature):
        center=self.hand_point(feature)
        if self.relative:
            if self._reference_hand is None:raise ValueError('Relative pointer requires quiet acquisition')
            delta=(center-self._reference_hand)/self.bounds.span*self.sensitivity
            target=self._reference_position+[-delta[0],delta[1]]
            clamped=np.clip(target,0,1)
            if np.any(target!=clamped):
                self._reference_hand=center.copy();self._reference_position=clamped.copy()
            return clamped
        raw=(center-np.asarray(self.bounds.center))/self.bounds.span*self.sensitivity+.5
        return np.clip(np.array([1-raw[0],raw[1]]),0,1)

    def camera_wrist(self,feature):
        # StableController's historical name; here the anchor is a 2D palm center.
        center=self.hand_point(feature)
        return np.array([center[0],center[1],0.])

    def translate_anchor(self,anchor,reference_wrist,current_wrist):
        anchor=_finite_vector(anchor,2,'screen anchor')
        reference=_finite_vector(reference_wrist,3,'reference palm')
        current=_finite_vector(current_wrist,3,'current palm')
        delta=(current[:2]-reference[:2])/self.bounds.span*self.sensitivity
        # Leave final clamping to controller/filter, preserving relative drag displacement.
        return anchor+np.array([-delta[0],delta[1]])

    def save(self,path):
        path=Path(path)
        data={'version':self.version,'kind':self.kind,'center':list(self.bounds.center),
              'span':list(self.bounds.span),'context':self.context,'relative':self.relative}
        encoded=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
                temporary=Path(stream.name);stream.write(encoded)
            temporary.replace(path)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text())
        if not isinstance(data,dict) or data.get('version') not in (1,2) or data.get('kind')!=cls.kind:
            raise ValueError('Unsupported workspace configuration')
        try:return cls(data['context'],WorkspaceBounds(data['center'],data['span']),relative=False if data['version']==1 else data['relative'])
        except KeyError as exc:raise ValueError('Incomplete workspace configuration') from exc
