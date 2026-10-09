"""Five-class time-based controller. Geometry states own commands; no OS side effects."""
import numpy as np
from collections import deque
from .types import ControlEvent
from .pointing import TimeAwarePointerFilter


class StableController:
    def __init__(self,library,calibration=None):
        self.library=library;self.calibration=calibration;self.conf_threshold=.75;self.prefer_image_geometry=False;self.position_provider=None
        from .pose_stability import ImagePoseLatch
        self.robust_geometry=False;self.pose_latch=ImagePoseLatch()
        self.entry_dwell=.16;self.rearm_dwell=.22
        self.pose_confirm=.06;self.pointer_acquire=.10;self.swipe_distance=.08
        self.swipe_gap_grace=.15;self.swipe_rest_window=.12;self.swipe_max_speed=4.
        self.scroll_dead_zone=0.
        self.smooth_scroll=False;self.scroll_pixels_per_unit=1800.;self.navigation_pose='palm'
        self.pinch_candidate=.42;self.pinch_cancel=.48;self.pinch_press=.24;self.pinch_release=.34
        self.press_dwell=.06;self.release_dwell=.05;self.drag_hold=.45;self.allow_drag=True
        self.filter=TimeAwarePointerFilter(beta=4.);self.acquire_filter=TimeAwarePointerFilter();self._clock=-1.;self.position=None;self._binding=None;self._mapper=None
        self.state='recovery';self.phase='recovery';self.last_label='no_hand';self.confidence=0.;self.geometry=None;self.reason='нужен спокойный указательный палец'
        self._clear_episode()
    def _clear_episode(self):
        if getattr(self.calibration,'relative',False):self.calibration.reset_pointer()
        self.acquire_filter.reset();self._point_since=None;self._point_origin=None;self._uncertain_since=None;self._exit_since=None
        self._entry_since=None;self._scroll_y=None;self._scroll_fraction=0.;self._swipe_origin=None;self._swipe_start=None;self._neutral_since=None
        self._swipe_point=None;self._swipe_last_seen=None
        self._swipe_trace=deque()
        self._scroll_smooth_y=None;self._scroll_time=None;self._scroll_origin=None;self._scroll_axis=None
        self._navigation_since=None;self._navigation_origin=None
        self._last_visible_time=None;self._tracking_gap=False;self._pose_candidate=None;self._pose_since=None
        self.pose_latch.reset()
        self.anchor=None;self._reference_wrist=None;self._press_since=None;self._pressed=False;self._release_since=None
        self._pinch_unknown_since=None
    def reset(self,timestamp=None):
        t=max(self._clock,0.) if timestamp is None else timestamp
        events=[ControlEvent('pinch','drag_end',t,self._position_payload())] if self.state=='drag' else []
        self.state='recovery';self.phase='recovery';self.reason='нужен спокойный указательный палец';self._clear_episode()
        self.filter.reset(self.position,None);return events
    def _position_payload(self,position=None):
        p=self.position if position is None else position
        return {} if p is None else {'x':float(p[0]),'y':float(p[1])}
    def _profile(self,label):
        try:return self.library.get('builtin:'+label)
        except StopIteration:return {'enabled':False,'action':'none'}
    def _event(self,label,t,payload=None,forbidden=()):
        g=self._profile(label);action=g['action']
        if not g.get('enabled',True) or action=='none' or action in forbidden:return []
        data=dict(payload or {})
        if action=='hotkey':data={'keys':list(g.get('keys',[]))}
        return [ControlEvent(label,action,t,data)]
    def _map(self,f):
        if self.calibration is None:raise ValueError('нужна калибровка луча на экран')
        p=np.asarray(self.calibration.map_feature(f),dtype=float)
        if p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):raise ValueError('луч находится вне экрана')
        return p
    def _motion_point(self,f):
        # Palm points survive finger articulation better than a fingertip. Keep
        # legacy normalized wrist coordinates for the experimental 3D branch.
        hand_point=getattr(self.calibration,'hand_point',None)
        p=np.asarray(hand_point(f) if self.prefer_image_geometry and callable(hand_point) else f.wrist,dtype=float)
        if p.shape!=(2,) or not np.isfinite(p).all():raise ValueError('invalid motion coordinates')
        return p
    def _start_swipe(self,f,t):
        try:p=self._motion_point(f)
        except ValueError:self._recover();return
        self.state='swipe_pending';self._entry_since=t;self._swipe_origin=p.copy()
        self._swipe_point=p.copy();self._swipe_last_seen=t;self._neutral_since=None
        self._swipe_trace=deque([(t,p.copy())])
        self.reason='удерживайте открытую ладонь спокойно перед свайпом'
    def _process_swipe(self,f,t,label):
        active_pose='navigation' if self.navigation_pose=='three' else 'palm'
        if self.state=='swipe_latched':
            self._swipe_last_seen=t
            if label==active_pose:self._neutral_since=None
            else:
                if self._neutral_since is None:self._neutral_since=t
                if t-self._neutral_since>=.3:self._recover()
            return []
        if self.state=='swipe_pending' and label!=active_pose:self._recover();return []
        if self.state=='swipe' and label in {'pinch','victory'}:self._recover();return []
        try:p=self._motion_point(f)
        except ValueError:self._recover();return []
        dt=t-self._swipe_last_seen
        if dt>self.swipe_gap_grace+1e-9:
            self._recover();self.reason='свайп отменён: слишком долгая потеря руки';return []
        step=float(np.linalg.norm(p-self._swipe_point))
        if step>self.swipe_max_speed*dt+.035:
            self._recover();self.reason='свайп отменён: скачок координат';return []
        self._swipe_point=p.copy();self._swipe_last_seen=t
        self._swipe_trace.append((t,p.copy()))
        while len(self._swipe_trace)>2 and t-self._swipe_trace[1][0]>=self.swipe_rest_window:
            self._swipe_trace.popleft()
        if self.state=='swipe_pending':
            modern=self.navigation_pose=='three'
            if not modern and np.linalg.norm(p-self._swipe_origin)>.025:self._entry_since=t;self._swipe_origin=p.copy()
            if t-self._entry_since>=(self.pose_confirm if modern else self.entry_dwell):
                self.state='swipe'
                if not modern:self._swipe_origin=p.copy()
                self._swipe_start=self._entry_since if modern else t
                self.reason='три пальца: проведите кистью влево или вправо' if modern else 'свайп готов: проведите ладонью влево или вправо'
            return []
        # Require the pose only to arm. Uncertain finger estimates under motion
        # do not discard a valid trajectory. Slow drift refreshes the origin.
        rest_t,rest_point=self._swipe_trace[0]
        if label==active_pose and t-rest_t>=.06 and np.linalg.norm(p-rest_point)<.012:
            self._swipe_origin=p.copy();self._swipe_start=t;return []
        dx,dy=p-self._swipe_origin;duration=t-self._swipe_start
        minimum,maximum,travel=(.08,2.,self.swipe_distance) if self.navigation_pose=='three' else (.12,1.2,.20)
        if minimum<=duration<=maximum and abs(dx)>=travel and abs(dx)>=2*abs(dy):
            events=self._event('swipe_left' if dx>0 else 'swipe_right',t,self._position_payload(),forbidden=('move','scroll','drag_start','drag_end'))
            self.state='swipe_latched';self._neutral_since=None;self.reason='свайп завершён: расслабьте руку перед следующим'
            return events
        if duration>maximum:self.state='swipe_latched';self.reason='неполный или слишком медленный свайп'
        return []
    def _pinch(self,f):
        if getattr(f,'geometry_pinch',None) is not None:return float(f.geometry_pinch)
        if getattr(f,'world_points',None) is not None:
            p=np.asarray(f.world_points,dtype=float)
            if p.shape!=(21,3) or not np.isfinite(p).all():return np.inf
            scale=np.linalg.norm(p[9]-p[0])
            return float(np.linalg.norm(p[4]-p[8])/scale) if scale>.005 else np.inf
        return float(f.pinch)
    def _scroll_motion(self,f,t,y):
        p=self._motion_point(f)
        if self._scroll_origin is None:self._scroll_origin=p.copy()
        dx,dy=p-self._scroll_origin
        if self.navigation_pose=='victory':
            if self._scroll_axis=='latched':return []
            if self._scroll_axis is None and max(abs(dx),abs(dy))>=.012:
                if abs(dx)>=1.6*abs(dy):self._scroll_axis='horizontal'
                elif abs(dy)>=1.6*abs(dx):self._scroll_axis='vertical'
            if self._scroll_axis=='horizontal':
                self.reason='два пальца: горизонтальный переход, курсор зафиксирован'
                if abs(dx)>=.08:
                    self._scroll_axis='latched';self.reason='переход выполнен: расслабьте пальцы перед следующим'
                    return self._event('swipe_left' if dx>0 else 'swipe_right',t,forbidden=('move','scroll','drag_start','drag_end'))
                return []
            if self._scroll_axis is None:return []
        dt=max(1e-3,t-(self._scroll_time if self._scroll_time is not None else t-.033))
        self._scroll_time=t
        if self._scroll_smooth_y is None:self._scroll_smooth_y=y
        if abs(y-self._scroll_smooth_y)>.12:
            self._scroll_smooth_y=y;self._scroll_y=y;self._scroll_fraction=0.;return []
        self._scroll_smooth_y+=(1-np.exp(-dt/.055))*(y-self._scroll_smooth_y)
        delta=self._scroll_y-self._scroll_smooth_y
        dead=max(.0015,self.scroll_dead_zone*.35)
        if abs(delta)<=dead:return []
        accepted=self._scroll_smooth_y+np.sign(delta)*dead
        pixels=(self._scroll_y-accepted)*self.scroll_pixels_per_unit
        self._scroll_y=accepted
        pixels=float(np.clip(pixels,-1800*dt,1800*dt))
        return self._event('victory',t,{'dy':pixels,'unit':'pixel'},forbidden=('move','drag_start','drag_end','click','right_click','hotkey'))
    def _begin_pinch(self,f,t):
        self.anchor=None if self.position is None else self.position.copy();self.state='pinch_candidate';self._press_since=None;self._pressed=False;self._release_since=None
        try:self._reference_wrist=self.calibration.camera_wrist(f)
        except (ValueError,AttributeError):self._reference_wrist=None
        self.reason='цель зафиксирована до сгибания пальца'
    def _recover(self):
        self.state='recovery';self._clear_episode();self.filter.reset(self.position,None)
    def process(self,f):
        t=float(f.timestamp)
        if not np.isfinite(t) or t<=self._clock:return []
        self.geometry=None
        events=[]
        if self._clock>=0 and t-self._clock>.5:events+=self.reset(t)
        self._clock=t
        bindings=tuple((key,self._profile(key).get('enabled',True),self._profile(key)['action'],tuple(self._profile(key).get('keys',[]))) for key in ['point','pinch','victory','swipe_left','swipe_right'])
        if self._binding is not None and bindings!=self._binding:events+=self.reset(t)
        if self._mapper is not None and self.calibration is not self._mapper:events+=self.reset(t)
        self._binding=bindings;self._mapper=self.calibration
        if self.robust_geometry and not f.present and self.state in {'pointer','scroll_pending','scroll'} and self._last_visible_time is not None and t-self._last_visible_time<=.20:
            self._navigation_since=None;self._navigation_origin=None
            self._tracking_gap=True;self.confidence=0.;self.phase=self.state
            self.reason='краткая потеря кисти: режим сохранён, команды приостановлены'
            return events
        if not f.present and self.state in {'swipe','swipe_latched'} and self._swipe_last_seen is not None:
            grace=self.swipe_gap_grace if self.state=='swipe' else .3
            if t-self._swipe_last_seen<=grace+1e-9:
                self.last_label='no_hand';self.confidence=0.;self.phase=self.state
                self.reason='краткая потеря руки в свайпе: ожидаю продолжение'
                return events
        world=getattr(f,'world_points',None)
        invalid_world=not self.prefer_image_geometry and world is not None and (np.asarray(world).shape!=(21,3) or not np.isfinite(world).all())
        geometry=None
        if self.prefer_image_geometry:
            from .vision import classify_geometry
            geometry=self.pose_latch.update(f) if self.robust_geometry else classify_geometry(f,prefer_image=True)
        self.geometry=geometry
        invalid_image=geometry is not None and geometry.source=='invalid_image'
        if not f.present or not np.isfinite(f.vector()).all() or invalid_world or invalid_image:
            events+=self.reset(t);self.last_label='no_hand';self.confidence=0.;self.phase=self.state;return events
        label=geometry.label if geometry is not None else getattr(f,'geometry_label',None)
        confidence=geometry.confidence if geometry is not None else float(getattr(f,'geometry_confidence',0.))
        if confidence<self.conf_threshold or label is None:label='uncertain'
        self.last_label=label;self.confidence=confidence
        pinch=geometry.pinch if geometry is not None else self._pinch(f)
        if self.robust_geometry and self.state in {'pinch_candidate','pinch_held','drag'} and not np.isfinite(pinch):
            # Unknown thumb/index distance is not an observed pinch release.
            # Wait briefly without clicking; a held OS button releases at once.
            if self.state=='drag':events.append(ControlEvent('pinch','drag_end',t,self._position_payload()));self._recover()
            else:
                if self._pinch_unknown_since is None:self._pinch_unknown_since=t
                if t-self._pinch_unknown_since>=.20:self._recover()
            self.phase=self.state;self.reason='щипок частично скрыт: клик не подтверждён';return events
        self._pinch_unknown_since=None
        if self.robust_geometry:
            self._last_visible_time=t
            if self._tracking_gap:
                self._tracking_gap=False
                if self.state=='pointer' and getattr(self.calibration,'relative',False):
                    self.calibration.begin_pointer(f,self.position);self.filter.reset(self.position,t)
                if self.state in {'scroll_pending','scroll'}:
                    self._scroll_y=float(self._motion_point(f)[1]);self._scroll_fraction=0.
                    self._scroll_smooth_y=self._scroll_y;self._scroll_time=t;self._scroll_origin=self._motion_point(f)
            # Freeze immediately for a candidate; only a stable alternative
            # takes ownership. Pinch freezes on the first closing observation.
            if self.state=='pointer' and label in {'victory','palm','navigation'}:
                if self._pose_candidate!=label:self._pose_candidate=label;self._pose_since=t
                if t-self._pose_since<self.pose_confirm-1e-9:
                    self.last_label='point';self.phase=self.state;return events
            else:self._pose_candidate=None;self._pose_since=None
        if self.state in {'scroll_pending','scroll'} and label=='navigation' and self.navigation_pose=='three':
            if self._navigation_since is None:
                self._navigation_since=t;self._navigation_origin=self._motion_point(f).copy()
            if t-self._navigation_since>=self.pose_confirm-1e-9:
                since,origin=self._navigation_since,self._navigation_origin.copy()
                self._start_swipe(f,t);self._entry_since=since;self._swipe_origin=origin
                events+=self._process_swipe(f,t,label)
            self.phase=self.state;self.reason='три пальца: переключение на свайп, прокрутка остановлена';return events
        self._navigation_since=None;self._navigation_origin=None
        if self.calibration is None:
            events+=self.reset(t);self.phase='calibration_required';self.reason='нужна калибровка camera-relative луча';return events
        if self.state in {'scroll_pending','scroll'} or label=='victory':
            try:scroll_y=float(self._motion_point(f)[1])
            except ValueError:
                events+=self.reset(t);self.phase=self.state;self.reason='не удалось оценить перемещение кисти';return events
        if self.state in {'pinch_candidate','pinch_held','drag'}:
            if label in {'victory','palm','navigation'}:
                if self.state=='drag':events.append(ControlEvent('pinch','drag_end',t,self._position_payload()))
                self._recover()
            elif pinch>=(self.pinch_release if self._pressed else self.pinch_cancel):
                if self._release_since is None:self._release_since=t
                if t-self._release_since>=self.release_dwell-1e-9:
                    if self.state=='drag':events.append(ControlEvent('pinch','drag_end',t,self._position_payload()))
                    elif self._pressed and self.anchor is not None:events+=self._event('pinch',t,self._position_payload(self.anchor),forbidden=('move','scroll','drag_start','drag_end'))
                    self._recover()
            else:
                self._release_since=None
                if pinch<=self.pinch_press:
                    if self._press_since is None:self._press_since=t
                    if t-self._press_since>=self.press_dwell-1e-9:self._pressed=True;self.state='pinch_held' if self.state!='drag' else 'drag'
                elif not self._pressed:self._press_since=None
                g=self._profile('pinch')
                if self._pressed and self.allow_drag and g.get('enabled',True) and g['action'] in {'click','drag_start'} and self._press_since is not None and t-self._press_since>=self.drag_hold:
                    if self.state!='drag' and self.anchor is not None and self._reference_wrist is not None:
                        events.append(ControlEvent('pinch','drag_start',t,self._position_payload(self.anchor)));self.state='drag';self.filter.reset(self.anchor,t)
                    if self.state=='drag':
                        try:
                            wrist=self.calibration.camera_wrist(f);target=self.calibration.translate_anchor(self.anchor,self._reference_wrist,wrist)
                            if not np.isfinite(target).all():raise ValueError('invalid drag translation')
                            self.position=np.clip(self.filter.update(target,t),0,1);events.append(ControlEvent('point','move',t,self._position_payload()))
                        except (ValueError,AttributeError):
                            events.append(ControlEvent('pinch','drag_end',t,self._position_payload()));self._recover();self.reason='не удалось оценить перенос запястья'
        elif self.state in {'scroll_pending','scroll'}:
            if label=='victory':
                if self._exit_since is not None:
                    self._exit_since=None;self._scroll_y=scroll_y;self._scroll_fraction=0.
                    self._scroll_smooth_y=scroll_y;self._scroll_time=t;self._scroll_origin=self._motion_point(f)
                    if self.state=='scroll_pending':self._entry_since=t
                    self.phase=self.state
                    return events
                self._exit_since=None
                if self.state=='scroll_pending':
                    if t-self._entry_since>=(self.pose_confirm if self.robust_geometry else self.entry_dwell)-1e-9:
                        self.state='scroll';self._scroll_y=scroll_y;self.reason='курсор зафиксирован: двигайте кисть вверх или вниз'
                        self._scroll_smooth_y=scroll_y;self._scroll_time=t;self._scroll_origin=self._motion_point(f);self._scroll_axis=None
                else:
                    if self.smooth_scroll:
                        events+=self._scroll_motion(f,t,scroll_y);self.phase=self.state;return events
                    delta=self._scroll_y-scroll_y
                    if abs(delta)>=.12:
                        self._scroll_y=scroll_y;self._scroll_fraction=0.
                    elif abs(delta)>self.scroll_dead_zone:
                        # A movable dead zone: slow increments accumulate until
                        # they leave the band; rest noise cannot rock the wheel
                        # back and forth across an integer-step boundary.
                        accepted=scroll_y+np.sign(delta)*self.scroll_dead_zone
                        self._scroll_fraction+=(self._scroll_y-accepted)*100
                        self._scroll_y=accepted
                    if self._profile('victory')['action']=='scroll':
                        steps=int(np.clip(np.trunc(self._scroll_fraction),-12,12))
                        if steps:events+=self._event('victory',t,{'dy':steps});self._scroll_fraction-=steps
                    elif self._neutral_since is None:
                        events+=self._event('victory',t,self._position_payload(),forbidden=('move','drag_start','drag_end'));self._neutral_since=t
            else:
                self._scroll_y=scroll_y
                if self._exit_since is None:self._exit_since=t
                if t-self._exit_since>=self.rearm_dwell:self._recover()
        elif self.state in {'swipe_pending','swipe','swipe_latched'}:
            events+=self._process_swipe(f,t,label)
        elif self.state=='pointer':
            if label in {'point','pinch'} and pinch<self.pinch_candidate and self.position is not None:
                self._begin_pinch(f,t)
                if pinch<=self.pinch_press:self._press_since=t
            elif label=='victory':
                self.state='scroll' if self.robust_geometry else 'scroll_pending';self._entry_since=t;self._scroll_y=scroll_y;self._scroll_fraction=0.;self._neutral_since=None
                self._scroll_smooth_y=scroll_y;self._scroll_time=t;self._scroll_origin=self._motion_point(f);self._scroll_axis=None
            elif (label=='palm' and self.navigation_pose=='palm') or (label=='navigation' and self.navigation_pose=='three'):
                self._start_swipe(f,t)
            elif label=='point':
                self._uncertain_since=None
                try:
                    self.position=self.filter.update(self._map(f),t);events+=self._event('point',t,self._position_payload(),forbidden=('scroll','drag_start','drag_end','click','right_click','hotkey'))
                except ValueError as exc:self._recover();self.reason=str(exc)
            else:
                if self._uncertain_since is None:self._uncertain_since=t
                if self.robust_geometry:self.last_label='point';self.reason='поза кратко неясна: курсор зафиксирован'
                if t-self._uncertain_since>=(.3 if self.robust_geometry else .2):self._recover()
        else:
            if (label=='palm' and self.navigation_pose=='palm') or (label=='navigation' and self.navigation_pose=='three'):self._start_swipe(f,t)
            elif label=='victory':self.state='scroll_pending';self._entry_since=t;self._scroll_y=scroll_y;self._scroll_fraction=0.;self._neutral_since=None
            elif label=='point' and pinch>=self.pinch_candidate:
                try:
                    relative=getattr(self.calibration,'relative',False)
                    raw=self.calibration.acquisition_point(f) if relative else self._map(f)
                    raw=np.asarray(raw,dtype=float)
                    if raw.shape!=(2,) or not np.isfinite(raw).all():raise ValueError('Invalid acquisition point')
                    target=self.acquire_filter.update(raw,t)
                    moving_acquisition=self.robust_geometry and relative
                    # Relative control anchors to the existing cursor. A stable
                    # pose can acquire during motion; absolute ray mode still
                    # needs a quiet target. Reject implausible tracking jumps.
                    threshold=.35 if moving_acquisition else .025
                    if self._point_origin is None or np.linalg.norm(target-self._point_origin)>threshold:self._point_since=t
                    if moving_acquisition or self._point_origin is None:self._point_origin=target.copy()
                    elif np.linalg.norm(target-self._point_origin)>.025:self._point_origin=target.copy()
                    if t-self._point_since>=(self.pointer_acquire if moving_acquisition else self.rearm_dwell)-1e-9:
                        if relative:
                            try:anchor=self.position_provider() if self.position_provider is not None else self.position
                            except Exception as exc:raise ValueError('Cursor position unavailable') from exc
                            # None is valid only for first acquisition without a provider.
                            if self.position_provider is not None and anchor is None:raise ValueError('Cursor position unavailable')
                            target=self.calibration.begin_pointer(f,anchor)
                            self.position=target.copy();self.filter.reset(target,t)
                        else:
                            self.filter.reset(self.position,t);self.position=self.filter.update(target,t)
                        self.state='pointer';self.reason='двигайте кисть; для возврата руки расслабьте позу' if relative else 'наведение активно'
                        events+=self._event('point',t,self._position_payload(),forbidden=('scroll','drag_start','drag_end'))
                except ValueError as exc:self._point_since=None;self._point_origin=None;self.reason=str(exc)
            else:self._point_since=None;self._point_origin=None
        self.phase=self.state
        return events
