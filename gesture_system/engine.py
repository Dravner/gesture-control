"""Time-based causal controller. Recognition never posts OS events itself."""
from collections import deque
import numpy as np
from .types import ControlEvent

class Engine:
    def __init__(self,library,method='two_stage',predictor=None,load_models=True,experimental_neural=None):
        if method not in {'window','two_stage','joint'}:raise ValueError('Неизвестный метод')
        self.library=library;self.method=method
        from .stable_control import StableController
        self.stable_profile=False;self.pointing_calibration=None
        self.stable_controller=StableController(library)
        from .personal import PersonalRecognizer
        self.personal=PersonalRecognizer(library)
        self.personal_phase='idle';self.personal_result={}
        self.control_mode='combined'
        self.experimental_neural=bool(predictor is not None) if experimental_neural is None else bool(experimental_neural)
        self.hold_seconds=.18;self.conf_threshold=.75
        self.last_label='no_hand';self.confidence=0.;self.phase='idle'
        self._clock=-1.;self._candidate=None;self._since=0.;self._fired=False
        self._static_latched=None;self._static_neutral=None
        self._history=deque(maxlen=12);self._segment=[];self._start=None;self._swiped=False
        self._pinch_since=None;self._drag=False;self._pointer=None;self._prev=None
        self._last_hand_pointer=None
        self._scroll_source=None;self._scroll_fraction=0.
        self._custom_history=deque(maxlen=240);self._custom_check_t=-1.;self._custom_recent=None
        self._predictor=predictor
        self._neural_t=-1.;self._neural_candidate=None;self._neural_count=0;self._neural_latched=None
        self._neural_neutral=None;self._neural_last_event=-1.
        if predictor is None and load_models:
            from pathlib import Path
            path=Path(__file__).resolve().parents[1]/'models'/f'streaming_{method}.pth'
            if path.exists():
                from .temporal import TemporalPredictor
                self._predictor=TemporalPredictor(path)

    def load_temporal_predictor(self,root):
        from pathlib import Path
        from .temporal import TemporalPredictor
        import torch
        torch.set_num_threads(4)
        self._predictor=TemporalPredictor(Path(root)/'models'/f'streaming_{self.method}.pth')

    def reset(self,preserve_custom=False,preserve_neural=False,preserve_static=False,preserve_personal=False):
        events=self.stable_controller.reset()
        events += [ControlEvent('pinch','drag_end',max(self._clock,0.),{})] if self._drag else []
        self._drag=False;self._pinch_since=None;self._history.clear();self._candidate=None;self._fired=False
        self._segment=[];self._start=None;self._swiped=False;self._prev=None;self._pointer=None
        self._scroll_source=None;self._scroll_fraction=0.
        self.phase='idle';self.last_label='no_hand';self.confidence=0
        if not preserve_static:self._static_latched=None;self._static_neutral=None
        if not preserve_personal:self.personal.reset();self.personal_phase='idle';self.personal_result={};self._last_hand_pointer=None
        if not preserve_custom:
            self._custom_history.clear();self._custom_recent=None
        if not preserve_neural:
            self._neural_t=-1.;self._neural_candidate=None;self._neural_count=0;self._neural_latched=None
            self._neural_neutral=None;self._neural_last_event=-1.
        if not preserve_neural and self._predictor and hasattr(self._predictor,'reset'):self._predictor.reset()
        return events

    def _event(self,identifier,t):
        try:g=self.library.get(identifier if ':' in identifier or len(identifier)==32 else 'builtin:'+identifier)
        except StopIteration:return []
        if not g.get('enabled',True):return []
        if self.control_mode=='pointer' and g['action']=='hotkey':return []
        payload={'keys':g['keys']} if g['action']=='hotkey' else {}
        if g['action']=='scroll':payload={'dy':-6 if identifier=='scroll_down' else 6}
        if identifier in {'doubleclick','double_rightclick'}:payload['count']=2
        preceding=[]
        if self._drag and g['action'] in {'click','right_click'}:
            preceding=[ControlEvent('pinch','drag_end',t,{})];self._drag=False
        if g['action']=='drag_start':self._drag=True
        elif g['action']=='drag_end':self._drag=False
        position=self._pointer if self._pointer is not None else self._last_hand_pointer
        if g['action']=='move' and position is not None:payload={'x':float(position[0]),'y':float(position[1])}
        return preceding+[ControlEvent(identifier,g['action'],t,payload)]

    def _enabled_action(self,label,action):
        try:g=self.library.get('builtin:'+label)
        except StopIteration:return False
        return g.get('enabled',True) and g['action']==action

    def _static_observation(self,label,t):
        if label==self._static_latched:
            self._static_neutral=None
        else:
            if self._static_neutral is None:self._static_neutral=t
            if t-self._static_neutral>=.3-1e-9:self._static_latched=None

    def _match_custom(self,t,force=False):
        if len(self._custom_history)<8:return []
        if not force and (t-self._custom_history[0][0]<.6 or t-self._custom_check_t<.2):return []
        self._custom_check_t=t;best=None;seen=set()
        for horizon in ([3.2] if force else [.8,1.4,2.2,3.2]):
            rows=[v for stamp,v in self._custom_history if stamp>=t-horizon]
            if len(rows)<8 or len(rows) in seen:continue
            seen.add(len(rows));match=self.library.recognize_dynamic(np.asarray(rows))
            if match and (best is None or match[1]>best[1]):best=match
        if best and best[1]>=.5:
            self._custom_recent=(t,best[0],best[1]);self._custom_history.clear()
            return self._event(best[0],t)
        return []

    def process(self,f):
        if self.stable_profile:
            self.stable_controller.calibration=self.pointing_calibration
            self.stable_controller.conf_threshold=self.conf_threshold
            events=self.stable_controller.process(f)
            self.last_label=self.stable_controller.last_label;self.phase=self.stable_controller.phase;self.confidence=self.stable_controller.confidence
            self._clock=max(self._clock,self.stable_controller._clock)
            return events
        t=float(f.timestamp)
        if not np.isfinite(t) or t<=self._clock:return []
        out=[]
        if self._clock>=0 and t-self._clock>.5:out+=self.reset()
        self._clock=t
        quality=getattr(f,'joint_confidence',None)
        if f.present and quality is not None and (np.asarray(quality).shape!=(21,) or not np.isfinite(quality).all() or np.min(quality)<.25):
            out+=self.reset();self.last_label='Неполная кисть';self.personal_phase='invalid'
            self.personal_result={'reason':'недостаточно уверенных суставов для полного шаблона'}
            return out
        world=getattr(f,'world_points',None)
        if f.present and (not np.isfinite(f.vector()).all() or (world is not None and (np.asarray(world).shape!=(21,3) or not np.isfinite(world).all()))):
            out+=self.reset();self.last_label='Некорректные координаты';self.personal_phase='invalid';self.personal_result={'reason':'некорректные координаты кисти'}
            return out
        if f.present:self._last_hand_pointer=np.clip([1-f.pointer[0],f.pointer[1]],0.,1.)
        decision=self.personal.update(f)
        self.personal_phase=decision.phase
        self.personal_result=dict(gesture_id=decision.match.identifier if decision.match else None,distance=decision.distance,radius=decision.threshold,reason=decision.phase,progress=decision.progress)
        if decision.match and self.control_mode!='pointer':
            self._scroll_source=None;self._scroll_fraction=0.;self._prev=None
            if not f.present and self._drag:out.append(ControlEvent('pinch','drag_end',t,{}));self._drag=False
            if not f.present and self.library.get(decision.match.identifier)['action']=='drag_start':
                out.append(ControlEvent(decision.match.identifier,'none',t,{'reason':'Удержание не начато: рука вне кадра'}))
            else:out+=self._event(decision.match.identifier,t)
            self._custom_recent=(t,decision.match.identifier,decision.match.confidence)
            self.last_label=self.library.get(decision.match.identifier)['name'];self.confidence=decision.match.confidence;self.phase='recognized'
            self.personal_result['reason']='полное движение совпало'
            if f.present:
                ending=self.personal.static_match(f.pose,getattr(f,'world_points',None))
                if ending:self._static_latched=ending.identifier;self._candidate=ending.identifier;self._fired=True;self._static_neutral=None
            return out
        if not f.present or not np.isfinite(f.vector()).all():
            self._static_observation(None,t)
            if self._predictor:
                if self._neural_t<0 or t-self._neural_t>=1/self._predictor.sample_fps-1e-6:
                    self._neural_t=t;self._predictor.update(np.zeros(48,dtype=np.float32),False)
                self._neural_candidate=None;self._neural_count=0
                if self._neural_neutral is None:self._neural_neutral=t
                if t-self._neural_neutral>=.3-1e-9:self._neural_latched=None
            out+=self.reset(preserve_static=True,preserve_neural=True,preserve_personal=True,preserve_custom=True)
            if decision.match:self.last_label=self.library.get(decision.match.identifier)['name'];self.confidence=decision.match.confidence;self.phase='recognized'
            return out
        label=f.label;confidence=float(f.confidence)
        static=self.personal.static_match(f.pose,getattr(f,'world_points',None)) if self.control_mode!='pointer' else None
        custom=(static.identifier,static.confidence) if static else None
        if custom:
            label,confidence=custom
            self.personal_phase='static';self.personal_result.update(gesture_id=label,distance=static.distance,radius=static.radius,reason='форма совпала')
        elif self.personal.last_distance is not None:
            self.personal_result.update(distance=self.personal.last_distance,radius=self.personal.last_radius,reason=self.personal.last_reason)
        if (decision.reserved and self.control_mode!='pointer') or (self.control_mode=='personal' and not custom):
            self._scroll_source=None;self._scroll_fraction=0.;self._prev=None
            if self._drag:out.append(ControlEvent('pinch','drag_end',t,{}));self._drag=False
            self._pinch_since=None;self._static_observation(None,t)
            self._candidate=None;self._fired=False
            self.last_label='Динамический жест…' if decision.reserved else 'Ожидание своего жеста';self.confidence=0.;self.phase=decision.phase
            if self._custom_recent and t-self._custom_recent[0]<.8:
                self.last_label=self.library.get(self._custom_recent[1])['name'];self.confidence=self._custom_recent[2];self.phase='recognized'
            return out
        if self._predictor and self.experimental_neural and self.control_mode=='combined' and (self._neural_t<0 or t-self._neural_t>=1/self._predictor.sample_fps-1e-6):
            self._neural_t=t
            predicted,score,active=self._predictor.update(f.vector(),True)
            thresholds=self._predictor.thresholds
            accepted=score>=max(self.conf_threshold,thresholds.get('confidence',.3)) and active>=max(.7,thresholds.get('active',.5)) and predicted!='none'
            if not accepted:
                self._neural_candidate=None;self._neural_count=0
                if self._neural_neutral is None:self._neural_neutral=t
                if t-self._neural_neutral>=.3:self._neural_latched=None
            else:
                self._neural_neutral=None
                self._neural_count=self._neural_count+1 if predicted==self._neural_candidate else 1
                self._neural_candidate=predicted
                if predicted in {'point','victory'} and not custom:
                    label=predicted;confidence=max(float(f.confidence),score)
                elif self._neural_count>=max(3,thresholds.get('confirm',2)) and self._neural_latched is None and t-self._neural_last_event>=.5 and not custom:
                    if predicted.startswith('swipe_'):out+=self._event(predicted,t)
                    elif predicted=='click' and f.pinch>=.25 and self._pinch_since is None and not self._drag:out+=self._event('pinch',t)
                    elif predicted=='rightclick':out+=self._event('thumbs_down',t)
                    elif predicted in {'scroll_up','scroll_down','open_twice','zoom_in','zoom_out','double_rightclick'}:out+=self._event(predicted,t)
                    elif predicted=='doubleclick' and f.pinch>=.25 and self._pinch_since is None and not self._drag:out+=self._event(predicted,t)
                    self._neural_latched=predicted
                    self._neural_last_event=t
        self._history.append((t,label,confidence))
        if self.method=='window':
            recent=[x[1] for x in self._history if t-x[0]<.25]
            if len(recent)>=3:label=max(set(recent),key=recent.count)
            confidence=sum(x[2] for x in self._history)/len(self._history)
        elif self.method=='joint' and not self._predictor:
            # Explicit fallback only; GUI reports no trained checkpoint, not a neural claim.
            recent=[x for x in self._history if t-x[0]<.2]
            confidence=sum(x[2] for x in recent if x[1]==label)/max(len(recent),1)
        self.last_label=self.library.get(label)['name'] if custom and label==custom[0] else label;self.confidence=confidence
        if confidence<(.5 if custom else self.conf_threshold) or label in {'no_gesture','uncertain','no_hand'}:
            self._static_observation(None,t)
            out+=self.reset(preserve_custom=True,preserve_neural=True,preserve_static=True,preserve_personal=True)
            if self._custom_recent and t-self._custom_recent[0]<.5:
                self.last_label=self.library.get(self._custom_recent[1])['name'];self.confidence=self._custom_recent[2];self.phase='custom'
            return out
        self.phase='active'
        self._static_observation(label,t)
        if label!=self._candidate:
            self._candidate=label;self._since=t;self._fired=False
        stable=t-self._since>=self.hold_seconds
        # Separate coordinate mapping from pose normalization. Mirror matches preview.
        pointer=np.clip([1-f.pointer[0],f.pointer[1]],0.,1.)
        custom_action=self.library.get(custom[0])['action'] if custom else None
        pinch=f.pinch
        world=getattr(f,'world_points',None)
        if world is not None:
            pinch=float(np.linalg.norm(world[4]-world[8])/max(np.linalg.norm(world[9]-world[0]),.005))
        pinch_profile=self.library.get('builtin:pinch')
        pinch_enabled=pinch_profile.get('enabled',True) and pinch_profile['action']!='none' and not custom and self.control_mode!='personal'
        pinch_drag=pinch_enabled and pinch_profile['action'] in {'click','drag_start'}
        if custom_action=='move' or (label=='point' and self._enabled_action('point','move')) or (pinch<.25 and pinch_drag) or self._drag:
            if self._pointer is None:self._pointer=pointer
            else:
                delta=float(np.linalg.norm(pointer-self._pointer));alpha=min(.85,.18+delta*4)
                self._pointer=alpha*pointer+(1-alpha)*self._pointer
            out.append(ControlEvent(custom[0] if custom_action=='move' else 'point','move',t,{'x':float(self._pointer[0]),'y':float(self._pointer[1])}))
        if pinch<.25 and pinch_enabled:
            # Tracker labels a pinch as point. One physical episode owns one command.
            if label=='point':self._fired=True
            if self._pinch_since is None:self._pinch_since=t
            if pinch_drag and not self._drag and t-self._pinch_since>=.45:
                self._drag=True;out.append(ControlEvent('pinch','drag_start',t,{}))
        elif self._pinch_since is not None:
            duration=t-self._pinch_since
            if self._drag:out.append(ControlEvent('pinch','drag_end',t,{}));self._drag=False
            elif duration>=.08:out+=self._event('pinch',t)
            self._pinch_since=None
        scroll_source=custom[0] if custom_action=='scroll' else 'victory' if label=='victory' and self._enabled_action('victory','scroll') and self.control_mode!='personal' else None
        if scroll_source and self._scroll_source==scroll_source and self._prev is not None:
            self._scroll_fraction+=(self._prev.wrist[1]-f.wrist[1])*100
            steps=int(np.clip(np.trunc(self._scroll_fraction),-12,12))
            if steps:
                out.append(ControlEvent(scroll_source,'scroll',t,{'dy':steps}));self._scroll_fraction-=steps
        else:self._scroll_fraction=0.
        self._scroll_source=scroll_source
        analog=custom_action in {'move','scroll'} or (label=='point' and self._enabled_action('point','move')) or (label=='victory' and self._enabled_action('victory','scroll')) or label=='pinch'
        if stable and not self._fired and label!=self._static_latched and not analog and (custom or pinch>=.25):
            out+=self._event(label,t);self._fired=True;self._static_latched=label;self._static_neutral=None
        # Two-stage segment: start with a confident hand, end on pause/loss; 1.4s cap.
        if self._start is None:
            self._start=f;self._segment=[];self._swiped=False
        self._segment.append(f.vector())
        dx=-(f.wrist[0]-self._start.wrist[0]);dy=f.wrist[1]-self._start.wrist[1]
        elapsed=t-self._start.timestamp
        if not self.experimental_neural and self.control_mode=='combined' and label=='palm' and not self._swiped and .12<=elapsed<=1.4 and max(abs(dx),abs(dy))>.23:
            direction=('right' if dx>0 else 'left') if abs(dx)>abs(dy)*1.3 else ('down' if dy>0 else 'up')
            out+=self._event('swipe_'+direction,t);self._swiped=True
        if elapsed>=1.4:
            self._start=f;self._segment=[];self._swiped=False
        self._prev=f
        if self._custom_recent and t-self._custom_recent[0]<.5:
            self.last_label=self.library.get(self._custom_recent[1])['name'];self.confidence=self._custom_recent[2];self.phase='custom'
        return out
