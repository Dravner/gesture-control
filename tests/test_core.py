import numpy as np
import pytest
from gesture_system.types import FrameFeatures
from gesture_system.profiles import GestureLibrary, sequence_distance,sequence_embedding
from gesture_system.engine import Engine

@pytest.fixture(autouse=True)
def deterministic_unit_controller(monkeypatch):
    # These tests exercise controller logic, independently of downloaded weights.
    original=Engine.__init__
    def init(self,*args,**kwargs):
        kwargs['load_models']=False;original(self,*args,**kwargs)
    monkeypatch.setattr(Engine,'__init__',init)

def frame(t,label='palm',x=.5,pinch=1.):
    return FrameFeatures(t,True,np.zeros(42,dtype=np.float32),(x,.5),(x,.4),pinch,.2,label,.99)

def test_single_event_per_hold_and_rearm(tmp_path):
    lib=GestureLibrary(tmp_path)
    engine=Engine(lib)
    events=[]
    for t in np.arange(0,1,.05): events+=engine.process(frame(float(t),'thumbs_up'))
    assert sum(e.action=='hotkey' for e in events)==1
    for t in np.arange(1,1.5,.05): engine.process(FrameFeatures.absent(float(t)))
    second=[]
    for t in np.arange(1.5,2,.05): second+=engine.process(frame(float(t),'thumbs_up'))
    assert sum(e.action=='hotkey' for e in second)==1

def test_drag_released_when_hand_lost(tmp_path):
    e=Engine(GestureLibrary(tmp_path))
    result=[]
    for t in np.arange(0,.8,.05):result+=e.process(frame(float(t),'point',pinch=.1))
    assert any(x.action=='drag_start' for x in result)
    assert any(x.action=='drag_end' for x in e.process(FrameFeatures.absent(.9)))

def test_non_monotonic_clock_cannot_execute(tmp_path):
    e=Engine(GestureLibrary(tmp_path));e.process(frame(1,'thumbs_up'))
    assert e.process(frame(.5,'thumbs_up'))==[]

def test_custom_static_persist_and_unknown_rejected(tmp_path):
    lib=GestureLibrary(tmp_path);a=np.zeros((8,48));a[:,0]=.3
    key=lib.add('Мой жест','static',a,'hotkey',['command','a'])
    new=GestureLibrary(tmp_path)
    assert new.recognize_static(a[0,:42])[0]==key
    assert new.recognize_static(np.ones(42)*5) is None
    new.delete(key);assert not any(g['id']==key for g in new.list_gestures())

def test_dynamic_direction_is_not_normalized_away():
    a=np.zeros((25,48));a[:,42]=np.linspace(.1,.7,25);a[:,47]=.2
    b=a.copy();b[:,42]=np.linspace(.7,.1,25)
    shifted=a.copy();shifted[:,42]+=.1
    assert sequence_distance(a,shifted)<.001
    assert sequence_distance(a,b)>.1

def test_profile_rejects_nan_and_bad_hotkey(tmp_path):
    lib=GestureLibrary(tmp_path)
    with pytest.raises(ValueError):lib.add('bad','dynamic',np.full((10,48),np.nan),'click')
    with pytest.raises(ValueError):lib.add('bad','static',np.zeros((10,48)),'hotkey',['shell'])

def test_swipe_once_and_cursor_clamped(tmp_path):
    e=Engine(GestureLibrary(tmp_path));events=[]
    for t,x in zip(np.arange(0,1,.05),np.linspace(.2,.8,20)):events+=e.process(frame(float(t),'palm',float(x)))
    assert sum(x.gesture.startswith('swipe') for x in events)==1
    out=e.process(frame(2,'point',x=2))
    assert all(0<=x.payload['x']<=1 and 0<=x.payload['y']<=1 for x in out if x.action=='move')

def test_gap_releases_drag(tmp_path):
    e=Engine(GestureLibrary(tmp_path))
    for t in np.arange(0,.8,.05):e.process(frame(float(t),'point',pinch=.1))
    assert any(x.action=='drag_end' for x in e.process(frame(2,'point')))

def test_disabled_builtin_stops_analog_actions(tmp_path):
    lib=GestureLibrary(tmp_path);lib.update_mapping('builtin:point','none')
    lib.update_mapping('builtin:victory','none')
    e=Engine(lib);assert not any(x.action=='move' for x in e.process(frame(0,'point')))
    e.process(frame(.1,'victory'));assert not any(x.action=='scroll' for x in e.process(frame(.2,'victory')))

def test_dynamic_matches_when_hand_leaves(tmp_path):
    lib=GestureLibrary(tmp_path)
    frames=[frame(float(t),'palm',float(x)) for t,x in zip(np.arange(0,.6,.03),np.linspace(.3,.45,20))]
    key=lib.add('Движение','dynamic',np.stack([f.vector() for f in frames]),'click')
    e=Engine(lib)
    for f in frames:e.process(f)
    assert not any(x.gesture==key for x in e.process(FrameFeatures.absent(.7)))
    assert any(x.gesture==key for x in e.process(FrameFeatures.absent(1.)))

class FakeTemporal:
    sample_fps=10.;thresholds={'confidence':.25,'active':.5,'confirm':2}
    def __init__(self):self.calls=0
    def update(self,v,present):self.calls+=1;return ('swipe_left',.9,.9)
    def reset(self):pass

def test_neural_integration_cadence_and_single_event(tmp_path):
    predictor=FakeTemporal();e=Engine(GestureLibrary(tmp_path),'joint',predictor=predictor)
    events=[]
    for t in np.arange(0,1,.025):events+=e.process(frame(float(t),'palm'))
    assert 8<=predictor.calls<=10
    assert sum(x.gesture=='swipe_left' for x in events)==1

def test_custom_dynamic_does_not_require_known_static_pose(tmp_path):
    lib=GestureLibrary(tmp_path)
    fs=[frame(float(t),'uncertain',float(x)) for t,x in zip(np.arange(0,1,.05),np.linspace(.2,.6,20))]
    for f in fs:f.confidence=.1
    key=lib.add('Новый динамический','dynamic',np.stack([f.vector() for f in fs]),'click')
    e=Engine(lib);out=[]
    for f in fs:out+=e.process(f)
    out+=e.process(FrameFeatures.absent(1.1))
    out+=e.process(FrameFeatures.absent(1.2))
    out+=e.process(FrameFeatures.absent(1.4))
    assert any(x.gesture==key for x in out)

def test_static_hold_does_not_match_dynamic_template(tmp_path):
    lib=GestureLibrary(tmp_path)
    seq=np.zeros((25,48));seq[:,42]=np.linspace(.2,.7,25);seq[:,47]=.2
    lib.add('Свайп','dynamic',seq,'click')
    held=np.tile(seq[12],(25,1))
    assert lib.recognize_dynamic(held) is None

def test_stationary_fingers_do_not_trigger_rotation(tmp_path):
    lib=GestureLibrary(tmp_path)
    a=np.zeros((32,48));a[:,:42]=np.sin(np.linspace(0,4*np.pi,32))[:,None]*.7;a[:,47]=.2
    lib.add('Два поворота','dynamic',a,'click')
    assert lib.recognize_dynamic(np.tile(a[16],(32,1))) is None

@pytest.mark.parametrize('label',['point','victory','fist'])
def test_remapped_hold_emits_one_discrete_command(tmp_path,label):
    lib=GestureLibrary(tmp_path);lib.update_mapping('builtin:'+label,'right_click')
    e=Engine(lib);out=[]
    for t in np.arange(0,1,.05):out+=e.process(frame(float(t),label))
    assert sum(x.action=='right_click' for x in out)==1

def test_remapped_pinch_uses_selected_action(tmp_path):
    lib=GestureLibrary(tmp_path);lib.update_mapping('builtin:pinch','right_click')
    e=Engine(lib);out=[]
    for t in [0.,.1,.2]:out+=e.process(frame(t,'point',pinch=.1))
    out+=e.process(frame(.3,'point'))
    assert sum(x.action=='right_click' for x in out)==1
    assert not any(x.action=='drag_start' for x in out)

def test_neural_and_pinch_do_not_duplicate_click(tmp_path):
    class ClickPredictor(FakeTemporal):
        def update(self,v,present):return ('click',.9,.9)
    e=Engine(GestureLibrary(tmp_path),predictor=ClickPredictor());out=[]
    for t,p in [(0.,.1),(.1,.1),(.2,1.)]:out+=e.process(frame(t,'point',pinch=p))
    assert sum(x.action=='click' for x in out)==1

def test_weak_custom_match_preserves_valid_pointer(tmp_path):
    lib=GestureLibrary(tmp_path);lib.add('custom','static',np.zeros((8,48)),'none')
    f=frame(0,'point');f.pose[:]=.1
    assert any(x.action=='move' for x in Engine(lib).process(f))

def test_empty_custom_templates_rejected_on_load(tmp_path):
    import json
    lib=GestureLibrary(tmp_path);key=lib.add('custom','static',np.zeros((8,48)),'none')
    data=json.loads(lib.path.read_text())
    next(g for g in data['gestures'] if g['id']==key)['templates']=[]
    lib.path.write_text(json.dumps(data))
    with pytest.raises(ValueError):GestureLibrary(tmp_path)

def test_neural_command_requires_user_confidence_threshold(tmp_path):
    class Weak(FakeTemporal):
        def update(self,v,present):return ('swipe_left',.4,.9)
    e=Engine(GestureLibrary(tmp_path),predictor=Weak())
    assert not any(x.gesture=='swipe_left' for t in np.arange(0,1,.1) for x in e.process(frame(float(t))))

def test_neural_rearm_requires_sustained_neutral(tmp_path):
    class Interrupted(FakeTemporal):
        def update(self,v,present):
            self.calls+=1
            return ('none',0.,0.) if self.calls==4 else ('swipe_left',.9,.9)
    e=Engine(GestureLibrary(tmp_path),predictor=Interrupted());out=[]
    for t in np.arange(0,1,.1):out+=e.process(frame(float(t)))
    assert sum(x.gesture=='swipe_left' for x in out)==1

def test_pinch_episode_does_not_also_fire_remapped_point(tmp_path):
    lib=GestureLibrary(tmp_path);lib.update_mapping('builtin:point','click')
    e=Engine(lib);out=[]
    for t,p in [(0.,.1),(.1,.1),(.2,1.),(.3,1.)]:out+=e.process(frame(t,'point',pinch=p))
    assert sum(x.action=='click' for x in out)==1

def test_brief_classifier_dropout_does_not_repeat_static_command(tmp_path):
    e=Engine(GestureLibrary(tmp_path));out=[]
    for t in np.arange(0,.6,.05):out+=e.process(frame(float(t),'thumbs_up'))
    out+=e.process(frame(.6,'uncertain'))
    for t in np.arange(.65,1.2,.05):out+=e.process(frame(float(t),'thumbs_up'))
    assert sum(x.action=='hotkey' for x in out)==1

def test_sustained_neutral_rearms_static_command(tmp_path):
    e=Engine(GestureLibrary(tmp_path));out=[]
    for t in np.arange(0,.6,.05):out+=e.process(frame(float(t),'thumbs_up'))
    for t in np.arange(.6,1.1,.05):out+=e.process(FrameFeatures.absent(float(t)))
    for t in np.arange(1.1,1.6,.05):out+=e.process(frame(float(t),'thumbs_up'))
    assert sum(x.action=='hotkey' for x in out)==2

def test_dtw_precomputed_costs_match_reference_recurrence():
    rng=np.random.default_rng(29);x=rng.normal(size=(19,48));y=rng.normal(size=(41,48))
    x[:,47]=.2;y[:,47]=.2;a=sequence_embedding(x);b=sequence_embedding(y)
    cost=np.full((33,33),np.inf);cost[0,0]=0
    for i in range(1,33):
        for j in range(max(1,i-8),min(32,i+8)+1):
            local=float(np.sqrt(np.mean((a[i-1]-b[j-1])**2)))
            cost[i,j]=local+min(cost[i-1,j],cost[i,j-1],cost[i-1,j-1])
    assert np.isclose(sequence_distance(x,y),cost[-1,-1]/32,rtol=1e-6)

def test_brief_hand_loss_does_not_repeat_neural_command(tmp_path):
    e=Engine(GestureLibrary(tmp_path),predictor=FakeTemporal());out=[]
    for t in [0.,.1,.2]:out+=e.process(frame(t))
    out+=e.process(FrameFeatures.absent(.3))
    for t in [.4,.5,.6]:out+=e.process(frame(t))
    assert sum(x.gesture=='swipe_left' for x in out)==1

def test_custom_distance_acceptance_is_independent_of_neural_probability(tmp_path):
    pose=np.zeros((21,2),dtype=np.float32)
    for base,x in zip([1,5,9,13,17],[-.7,-.3,0,.3,.6]):
        pose[base:base+4]=[[x,-.3],[x,-.6],[x,-.9],[x,-1.2]]
    sample=frame(0).vector();sample[:42]=pose.reshape(-1)
    library=GestureLibrary(tmp_path);key=library.add('phone','static',np.tile(sample,(8,1)),'none')
    e=Engine(library);out=[]
    e.conf_threshold=.999
    for t in [0.,.1,.2]:
        f=frame(t,'thumbs_up');f.pose=pose.reshape(-1).copy();f.pose[16]+=.005;out+=e.process(f)
    assert any(x.gesture==key for x in out)
    assert e.last_label=='phone'

@pytest.mark.parametrize('fps',[16,30,60])
def test_slow_scroll_accumulates_fractional_motion_independently_of_fps(tmp_path,fps):
    e=Engine(GestureLibrary(tmp_path));e.control_mode='pointer';out=[]
    for i in range(fps+1):
        f=frame(i/fps,'victory');f.wrist=(.5,.5+.06*i/fps);out+=e.process(f)
    wheel=[x for x in out if x.action=='scroll']
    assert wheel and all(isinstance(x.payload['dy'],int) for x in wheel)
    assert 5<=-sum(x.payload['dy'] for x in wheel)<=6

@pytest.mark.parametrize('completed',[False, True])
def test_suppressed_dynamic_motion_does_not_scroll_after_settling(tmp_path,completed):
    from gesture_system.personal import Decision, Match
    e=Engine(GestureLibrary(tmp_path));out=[]
    e.personal.update=lambda f: Decision(None,'idle',False)
    for i in range(5):out+=e.process(frame(i*.05,'victory'))
    e._scroll_fraction=.7
    e.personal.update=lambda f: Decision(Match('builtin:palm',1.,0.,.1,'dynamic') if completed else None,'recognized' if completed else 'motion',True)
    f=frame(.25,'victory');f.wrist=(.5,.7);out+=e.process(f)
    e.personal.update=lambda f: Decision(None,'idle',False)
    for i in range(6,10):
        f=frame(i*.05,'victory');f.wrist=(.5,.7);out+=e.process(f)
    assert not [event for event in out if event.action=='scroll']
