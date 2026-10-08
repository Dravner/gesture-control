"""Estimated index-axis ray and fixed-size metric display plane, not fingertip XY mapping."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


def _readonly(value):
    value=np.asarray(value,dtype=np.float64)
    return np.frombuffer(value.tobytes(),dtype=np.float64).reshape(value.shape)


@dataclass(frozen=True)
class FingerRay:
    origin: np.ndarray
    direction: np.ndarray
    quality: dict = field(default_factory=dict)
    wrist: np.ndarray | None = None
    timestamp: float | None = None
    def __post_init__(self):
        origin=np.asarray(self.origin,dtype=float);direction=np.asarray(self.direction,dtype=float)
        if origin.shape!=(3,) or direction.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-9:
            raise ValueError('Некорректный луч пальца')
        object.__setattr__(self,'origin',_readonly(origin));object.__setattr__(self,'direction',_readonly(direction/np.linalg.norm(direction)))
        if self.wrist is not None:
            if np.asarray(self.wrist).shape!=(3,) or not np.isfinite(self.wrist).all():raise ValueError('Некорректное положение запястья')
            object.__setattr__(self,'wrist',_readonly(self.wrist))


@dataclass
class CameraIntrinsics:
    matrix: np.ndarray
    width: int
    height: int
    assumption: str = 'provided_intrinsics'
    horizontal_fov_degrees: float | None = None
    def __post_init__(self):
        self.matrix=np.asarray(self.matrix,dtype=float).copy()
        if self.width<=0 or self.height<=0 or self.matrix.shape!=(3,3) or not np.isfinite(self.matrix).all() or self.matrix[0,0]<=0 or self.matrix[1,1]<=0 or not np.allclose(self.matrix[2],[0,0,1]):raise ValueError('Некорректная матрица камеры')
    @classmethod
    def from_fov(cls,width,height,horizontal_fov_degrees=60):
        if not np.isfinite(horizontal_fov_degrees) or not 10<horizontal_fov_degrees<150:raise ValueError('Недопустимый предполагаемый FOV')
        fx=width/(2*np.tan(np.deg2rad(horizontal_fov_degrees)/2))
        return cls([[fx,0,width/2],[0,fx,height/2],[0,0,1]],width,height,'approximate_horizontal_fov',float(horizontal_fov_degrees))
    def to_dict(self):return dict(matrix=self.matrix.tolist(),width=self.width,height=self.height,assumption=self.assumption,horizontal_fov_degrees=self.horizontal_fov_degrees)
    @classmethod
    def from_dict(cls,data):return cls(**data)


def estimate_hand_pose(feature,intrinsics):
    """SQPnP + iterative refinement, with explicit normalized reprojection/depth gates."""
    import cv2
    diagnostics={'valid':False,'reason':'нет image/world координат'}
    world=getattr(feature,'world_points',None);image=getattr(feature,'image_points',None)
    if not feature.present or world is None or image is None:return None,diagnostics
    world=np.asarray(world,dtype=float);image=np.asarray(image,dtype=float)
    if world.shape!=(21,3) or image.shape!=(21,2) or not np.isfinite(world).all() or not np.isfinite(image).all():
        diagnostics['reason']='некорректные image/world координаты';return None,diagnostics
    if getattr(feature,'frame_size',None) is not None and tuple(feature.frame_size)!=(intrinsics.width,intrinsics.height):
        diagnostics['reason']='формат кадра не совпадает с intrinsics';return None,diagnostics
    spread=np.linalg.svd(world-world.mean(axis=0),compute_uv=False)
    if spread[1]<.002 or np.linalg.norm(world[9]-world[0])<.005:
        diagnostics['reason']='вырожденная геометрия кисти';return None,diagnostics
    pixels=image*[intrinsics.width,intrinsics.height]
    try:
        ok,rvec,tvec=cv2.solvePnP(world,pixels,intrinsics.matrix,None,flags=cv2.SOLVEPNP_SQPNP)
        if not ok:raise ValueError('SQPnP не нашёл позу')
        rvec,tvec=cv2.solvePnPRefineLM(world,pixels,intrinsics.matrix,None,rvec,tvec)
        R=cv2.Rodrigues(rvec)[0];translation=tvec.reshape(3);camera=(R@world.T).T+translation
        projected=cv2.projectPoints(world,rvec,tvec,intrinsics.matrix,None)[0].reshape(21,2)
        error=np.linalg.norm(projected-pixels,axis=1);rmse=float(np.sqrt(np.mean(error**2)));maximum=float(error.max())
        diagnostics.update(reprojection_rmse_px=rmse,reprojection_max_px=maximum,
                           reprojection_rmse_normalized=rmse/np.hypot(intrinsics.width,intrinsics.height),
                           intrinsics_assumption=intrinsics.assumption)
        if not np.isfinite(camera).all() or camera[:,2].min()<.03 or camera[:,2].max()>3:raise ValueError('недопустимая глубина кисти')
        if rmse>.02*np.hypot(intrinsics.width,intrinsics.height) or maximum>.06*np.hypot(intrinsics.width,intrinsics.height):raise ValueError('большая ошибка репроекции')
        diagnostics.update(valid=True,reason='estimated hand pose')
        return (R,translation),diagnostics
    except (cv2.error,ValueError) as exc:
        diagnostics['reason']=str(exc);return None,diagnostics


def estimate_finger_ray(feature,intrinsics,return_diagnostics=False):
    pose,diagnostics=estimate_hand_pose(feature,intrinsics)
    ray=None
    if pose is not None:
        world=np.asarray(feature.world_points);joints=world[[5,6,7,8]]
        center=joints.mean(axis=0);_,_,vt=np.linalg.svd(joints-center,full_matrices=False);axis=vt[0]
        if np.dot(axis,joints[-1]-joints[0])<0:axis=-axis
        residual=float(np.sqrt(np.mean(np.linalg.norm(np.cross(joints-center,axis),axis=1)**2)))
        length=float(np.linalg.norm(joints[-1]-joints[0]));chain=float(np.linalg.norm(np.diff(joints,axis=0),axis=1).sum())
        if length<.025 or length/max(chain,1e-9)<.88 or residual/max(length,1e-9)>.06:
            diagnostics.update(valid=False,reason='указательный палец согнут или ось ненадёжна')
        else:
            R,t=pose;diagnostics.update(axis_method='PCA index joints 5/6/7/8, sign MCP-to-tip',axis_line_residual_m=residual)
            ray=FingerRay(R@world[8]+t,R@axis,dict(diagnostics),R@world[0]+t,float(feature.timestamp))
    return (ray,diagnostics) if return_diagnostics else ray


class PointingCalibration:
    SCREEN_SIZE=(.3442447,.2225239)
    BASE_ROTATION=np.diag([-1.,1.,-1.])
    def __init__(self,C,R,screen_size_m,context,quality=None,samples=None):
        self.C=np.asarray(C,dtype=float);self.R=np.asarray(R,dtype=float);self.screen_size_m=tuple(screen_size_m)
        if self.C.shape!=(3,) or self.R.shape!=(3,3) or not np.isfinite(self.C).all() or not np.isfinite(self.R).all() or not np.allclose(self.R.T@self.R,np.eye(3),atol=1e-6) or abs(np.linalg.det(self.R)-1)>1e-6 or len(self.screen_size_m)!=2 or not np.isfinite(self.screen_size_m).all() or min(self.screen_size_m)<=0:raise ValueError('Некорректная плоскость экрана')
        self.context=json.loads(json.dumps(context));self.quality=quality or {};self.samples=samples or []
        self.intrinsics=CameraIntrinsics.from_dict(self.context['intrinsics']) if 'intrinsics' in self.context else None
    @classmethod
    def fit(cls,rays,target,context,screen_size_m=SCREEN_SIZE):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        rays=list(rays);target=np.asarray(target,dtype=float);W,H=screen_size_m
        if len(rays)<6 or target.shape!=(len(rays),2) or not np.isfinite(target).all() or np.any(target<0) or np.any(target>1):raise ValueError('Нужны минимум 6 корректных экранных целей')
        if np.linalg.matrix_rank(target-target.mean(axis=0),tol=.01)<2 or np.ptp(target,axis=0).min()<.4:raise ValueError('Цели не покрывают плоскость экрана')
        origins=np.array([r.origin for r in rays]);directions=np.array([r.direction for r in rays]);xy=target*[W,H]
        if np.linalg.matrix_rank(directions-directions.mean(axis=0),tol=.005)<2:raise ValueError('Направления лучей вырождены')
        center=np.array([W/2,0.,0.]);bounds=np.array([.35,.35,.35,.08,.08,.08])
        def unpack(p):return center+p[3:],Rotation.from_rotvec(p[:3]).as_matrix()@cls.BASE_ROTATION
        def residual(p):
            C,R=unpack(p);points=C+xy[:,0,None]*R[:,0]+xy[:,1,None]*R[:,1]
            return (np.cross(points-origins,directions)/np.hypot(W,H)).ravel()
        fit=least_squares(residual,np.zeros(6),bounds=(-bounds,bounds),loss='soft_l1',f_scale=.03,max_nfev=800)
        if not fit.success or np.any(np.abs(fit.x)>=bounds*.98) or np.linalg.matrix_rank(fit.jac,tol=1e-5)<6:raise ValueError('Плоскость не определена надёжно или достигнуты границы fit')
        C,R=unpack(fit.x);model=cls(C,R,screen_size_m,context)
        predicted=np.array([model.apply(r) for r in rays]);errors=np.linalg.norm(predicted-target,axis=1)
        points=C+xy[:,0,None]*R[:,0]+xy[:,1,None]*R[:,1];vectors=points-origins
        angles=np.rad2deg(np.arccos(np.clip(np.sum(vectors/np.linalg.norm(vectors,axis=1)[:,None]*directions,axis=1),-1,1)))
        rmse=float(np.sqrt(np.mean(errors**2)))
        if rmse>.08 or errors.max()>.18 or np.sqrt(np.mean(angles**2))>4:raise ValueError('Ошибка калибровки слишком велика')
        model.quality=dict(training_rmse_normalized=rmse,training_max_normalized=float(errors.max()),
                           training_angular_rmse_degrees=float(np.sqrt(np.mean(angles**2))),jacobian_condition=float(np.linalg.cond(fit.jac)),
                           training_points=len(rays),independent_validation=False)
        model.samples=[dict(origin=r.origin.tolist(),direction=r.direction.tolist(),target=q.tolist(),quality=r.quality,timestamp=r.timestamp) for r,q in zip(rays,target)]
        return model
    def apply(self,ray):
        if not isinstance(ray,FingerRay):raise ValueError('Нужен camera-relative FingerRay; XY не является лучом')
        normal=self.R[:,2];denominator=float(np.dot(normal,ray.direction))
        if abs(denominator)<.05:raise ValueError('Луч почти параллелен экрану')
        distance=float(np.dot(normal,self.C-ray.origin)/denominator)
        if distance<=0 or distance>4:raise ValueError('Пересечение позади пальца или слишком далеко')
        point=ray.origin+distance*ray.direction-self.C
        result=np.array([np.dot(point,self.R[:,0])/self.screen_size_m[0],np.dot(point,self.R[:,1])/self.screen_size_m[1]])
        if not np.isfinite(result).all() or np.any(result<-.2) or np.any(result>1.2):raise ValueError('Луч далеко за пределами экрана')
        return result
    def map_feature(self,feature):
        if self.intrinsics is None:raise ValueError('В калибровке отсутствуют intrinsics')
        ray,reason=estimate_finger_ray(feature,self.intrinsics,True)
        if ray is None:raise ValueError(reason['reason'])
        return self.apply(ray)
    def camera_wrist(self,feature):
        if self.intrinsics is None:raise ValueError('В калибровке отсутствуют intrinsics')
        pose,reason=estimate_hand_pose(feature,self.intrinsics)
        if pose is None:raise ValueError(reason['reason'])
        R,t=pose;return R@np.asarray(feature.world_points)[0]+t
    def translate_anchor(self,anchor,reference_wrist,current_wrist):
        delta=self.R.T@(np.asarray(current_wrist)-reference_wrist)
        return np.asarray(anchor)+delta[:2]/self.screen_size_m
    def matches_context(self,current):return self.context==current
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
        data=dict(version=1,kind='metric_finger_ray_plane',C=self.C.tolist(),R=self.R.tolist(),screen_size_m=list(self.screen_size_m),context=self.context,quality=self.quality,samples=self.samples)
        temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));temporary.replace(path)
    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text())
        if data.get('version')!=1 or data.get('kind')!='metric_finger_ray_plane':raise ValueError('Неподдерживаемая калибровка')
        return cls(**{k:data[k] for k in ['C','R','screen_size_m','context','quality','samples']})


def validation_errors(model,rays,target,screen_size=None):
    predicted=np.array([model.apply(ray) for ray in rays]);target=np.asarray(target,dtype=float)
    if predicted.shape!=target.shape or not np.isfinite(target).all():raise ValueError('Некорректные проверочные цели')
    delta=predicted-target;errors=np.linalg.norm(delta,axis=1)
    result=dict(errors_normalized=errors.tolist(),rmse_normalized=float(np.sqrt(np.mean(errors**2))),p95_normalized=float(np.percentile(errors,95)),max_normalized=float(errors.max()),independent_validation=True)
    if screen_size is not None:
        pixels=np.linalg.norm(delta*np.asarray(screen_size),axis=1);result.update(errors_pixels=pixels.tolist(),rmse_pixels=float(np.sqrt(np.mean(pixels**2))),p95_pixels=float(np.percentile(pixels,95)),max_pixels=float(pixels.max()),logical_screen_size=list(screen_size))
    return result


class TimeAwarePointerFilter:
    def __init__(self,min_cutoff=1.8,beta=.25,derivative_cutoff=1.):
        self.min_cutoff=min_cutoff;self.beta=beta;self.derivative_cutoff=derivative_cutoff;self.reset()
    def reset(self,value=None,timestamp=None):
        self.value=None if value is None else np.asarray(value,dtype=float).copy();self.raw=None if value is None else self.value.copy();self.timestamp=timestamp;self.derivative=np.zeros(2)
    @staticmethod
    def alpha(cutoff,dt):return 1/(1+1/(2*np.pi*cutoff*dt))
    def update(self,point,t):
        point=np.asarray(point,dtype=float)
        if point.shape!=(2,) or not np.isfinite(point).all() or not np.isfinite(t):raise ValueError('Некорректная позиция/время фильтра')
        if self.value is not None and self.timestamp is not None and t<=self.timestamp:return self.value.copy()
        if self.value is None or self.timestamp is None or t-self.timestamp>.5:
            self.reset(point,float(t));return self.value.copy()
        dt=t-self.timestamp;velocity=(point-self.raw)/dt;alpha=self.alpha(self.derivative_cutoff,dt)
        self.derivative+=alpha*(velocity-self.derivative);cutoff=self.min_cutoff+self.beta*np.linalg.norm(self.derivative)
        self.value+=self.alpha(cutoff,dt)*(point-self.value);self.raw=point.copy();self.timestamp=float(t)
        return self.value.copy()
