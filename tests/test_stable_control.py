import numpy as np
from gesture_system.types import FrameFeatures
from gesture_system.profiles import GestureLibrary

class Mapper:
    def map_feature(self,f):return np.array(f.pointer)
    def camera_wrist(self,f):return np.array([f.wrist[0],f.wrist[1],.5])
    def translate_anchor(self,anchor,reference,current):return np.array(anchor)+(np.array(current)-reference)[:2]

def f(t,label='point',pointer=(.3,.4),wrist=(.5,.5),pinch=1.):
    return FrameFeatures(float(t),True,np.zeros(42),wrist,pointer,pinch,.2,'thumbs_up',.99,
                         geometry_label=label,geometry_confidence=.99)

def controller(tmp_path):
    from gesture_system.stable_control import StableController
    return StableController(GestureLibrary(tmp_path),Mapper())

def point(c,start=0,position=(.3,.4)):
    result=[]
    for t in np.arange(start,start+.4,.02):result+=c.process(f(t,pointer=position))
    return result

def test_pinch_freezes_pre_curl_target_and_emits_once(tmp_path):
    c=controller(tmp_path);assert any(e.action=='move' for e in point(c))
    result=[]
    for t,p in [(.42,.39),(.45,.22),(.52,.20),(.59,.27),(.63,.20),(.70,.38),(.78,.38)]:
        result+=c.process(f(t,'pinch',pointer=(.8,.8),pinch=p))
    assert not any(e.action=='move' for e in result)
    clicks=[e for e in result if e.action=='click'];assert len(clicks)==1
    assert clicks[0].payload['x']==.3 and clicks[0].payload['y']==.4
    assert not any(e.action=='click' for e in point(c,.8))

def test_candidate_without_confirmed_press_never_clicks(tmp_path):
    c=controller(tmp_path);point(c)
    events=[]
    for t,p in [(.42,.39),(.5,.29),(.58,.40),(.7,.6),(.78,.6)]:events+=c.process(f(t,'pinch',pinch=p))
    assert not any(e.action in {'click','drag_start'} for e in events)

def test_scroll_dead_zone_suppresses_rest_jitter_without_discarding_slow_motion(tmp_path):
    c=controller(tmp_path);c.scroll_dead_zone=.006;point(c);events=[]
    for t in np.arange(.42,.7,.02):events+=c.process(f(float(t),'victory',wrist=(.5,.5)))
    # Rest just past a wheel quantization boundary after deliberate movement.
    for t in np.arange(.7,.9,.02):c.process(f(float(t),'victory',wrist=(.5,.484)))
    events=[]
    for i,t in enumerate(np.arange(.9,1.3,.02)):
        events+=c.process(f(float(t),'victory',wrist=(.5,.484+(.0055 if i%2 else -.0055))))
    assert not any(e.action in {'scroll','move'} for e in events)
    events=[]
    for t,y in zip(np.arange(1.3,1.9,.02),np.linspace(.484,.44,30)):
        events+=c.process(f(float(t),'victory',wrist=(.5,float(y))))
    assert any(e.action=='scroll' for e in events)
    assert not any(e.action=='move' for e in events)

def test_scroll_retains_fraction_and_no_point_leak_until_quiet_rearm(tmp_path):
    c=controller(tmp_path);point(c);events=[]
    for t in np.arange(.42,.9,.02):events+=c.process(f(t,'victory',pointer=(.9,.9),wrist=(.5,.5-(t-.42)*.08)))
    assert any(e.action=='scroll' for e in events) and not any(e.action=='move' for e in events)
    assert not any(e.action=='move' for t in [.92,.97] for e in c.process(f(t,'point',pointer=(.8,.8))))
    for t in [1.01,1.08]:assert not any(e.action=='move' for e in c.process(f(t,'victory')))
    recovery=[]
    for t in np.arange(1.1,1.6,.02):recovery+=c.process(f(t,'point',pointer=(.8,.8)))
    moves=[e for e in recovery if e.action=='move'];assert moves
    assert moves[0].payload['x']<.8

def test_partial_swipe_noise_and_one_event_until_neutral(tmp_path):
    c=controller(tmp_path);events=[]
    for t in np.arange(0,.24,.02):events+=c.process(f(t,'palm'))
    for t,x in zip(np.arange(.24,.64,.02),np.linspace(.5,.62,20)):events+=c.process(f(t,'palm',wrist=(x,.5)))
    assert not any(e.gesture.startswith('swipe') for e in events)
    for t,x in zip(np.arange(.64,1.04,.02),np.linspace(.62,.9,20)):events+=c.process(f(t,'palm',wrist=(x,.5)))
    for t in np.arange(1.04,1.6,.02):events+=c.process(f(t,'palm',wrist=(.9,.5)))
    swipes=[e for e in events if e.gesture.startswith('swipe')];assert len(swipes)==1
    assert not any(e.action=='move' for e in events)

def test_drag_uses_wrist_not_curl_and_hand_loss_releases_without_click(tmp_path):
    c=controller(tmp_path);point(c);events=[]
    for t in np.arange(.42,1.1,.02):events+=c.process(f(t,'pinch',pointer=(.95,.95),pinch=.15))
    assert sum(e.action=='drag_start' for e in events)==1
    moved=c.process(f(1.12,'pinch',pointer=(.1,.1),pinch=.15,wrist=(.6,.5)))
    assert any(e.action=='move' and .3<e.payload['x']<.5 for e in moved)
    lost=c.process(FrameFeatures.absent(1.14))
    assert sum(e.action=='drag_end' for e in lost)==1 and not any(e.action=='click' for e in lost)

def test_mapping_disabled_and_missing_calibration_block_output(tmp_path):
    from gesture_system.stable_control import StableController
    library=GestureLibrary(tmp_path);library.update_mapping('builtin:point','none')
    c=StableController(library,Mapper());assert not point(c)
    c=StableController(library,None);assert not point(c)
    assert c.phase=='calibration_required'

def test_engine_stable_profile_ignores_custom_neural_and_legacy_label(tmp_path):
    from gesture_system.engine import Engine
    library=GestureLibrary(tmp_path);library.add('Конкурент','static',np.zeros((8,48)),'click')
    class Predictor:
        sample_fps=10;thresholds={}
        def update(self,*args):raise AssertionError('stable must not run neural')
        def reset(self):pass
    engine=Engine(library,predictor=Predictor(),load_models=False);engine.stable_profile=True;engine.pointing_calibration=Mapper()
    events=[]
    for t in np.arange(0,.5,.02):events+=engine.process(f(t))
    assert any(e.action=='move' for e in events)
    assert not any(e.action in {'click','hotkey'} for e in events)


def test_fast_same_frame_pinch_does_not_update_ray_before_anchor(tmp_path):
    c=controller(tmp_path);point(c)
    class CurlMapper(Mapper):
        def map_feature(self,feature):
            if feature.pinch<.42:raise AssertionError('must not map curling finger')
            return super().map_feature(feature)
    c.calibration=CurlMapper();point(c,.4)
    events=c.process(f(.82,'pinch',pointer=(.95,.05),pinch=.15))
    assert c.anchor.tolist()==[.3,.4] and not any(e.action=='move' for e in events)


def test_both_swipe_directions_rearm_only_after_neutral(tmp_path):
    c=controller(tmp_path);events=[]
    def swipe(start,first,last):
        result=[]
        for t in np.arange(start,start+.24,.02):result+=c.process(f(t,'palm',wrist=(first,.5)))
        for t,x in zip(np.arange(start+.24,start+.74,.02),np.linspace(first,last,25)):result+=c.process(f(t,'palm',wrist=(float(x),.5)))
        return result
    events+=swipe(0,.5,.85)
    for t in np.arange(.74,1.1,.02):events+=c.process(f(t,'uncertain'))
    events+=swipe(1.12,.85,.5)
    names=[e.gesture for e in events if e.gesture.startswith('swipe')]
    assert names==['swipe_left','swipe_right']


def test_scroll_units_are_consistent_across_frame_rates(tmp_path):
    counts=[]
    for fps in [15,30,60]:
        c=controller(tmp_path/str(fps));events=[]
        for t in np.arange(0,1.2,1/fps):events+=c.process(f(t,'victory',wrist=(.5,.6-.08*t)))
        counts.append(sum(e.payload['dy'] for e in events if e.action=='scroll'))
    assert max(counts)-min(counts)<=1 and min(counts)>=7


def test_mapping_change_mid_drag_releases_before_recovery(tmp_path):
    c=controller(tmp_path);point(c)
    for t in np.arange(.42,1.1,.02):c.process(f(t,'pinch',pinch=.15))
    assert c.state=='drag'
    c.library.update_mapping('builtin:pinch','none')
    events=c.process(f(1.12,'pinch',pinch=.15))
    assert sum(e.action=='drag_end' for e in events)==1
    assert not any(e.action in {'click','drag_start'} for e in events)


def test_noisy_stationary_ray_can_reacquire_without_four_second_lockout(tmp_path):
    c=controller(tmp_path);rng=np.random.default_rng(43);events=[]
    for t in np.arange(0,4,1/30):
        target=tuple(np.array([.5,.5])+rng.normal(0,.02,2));events+=c.process(f(t,pointer=target))
    moves=[e for e in events if e.action=='move']
    assert moves and moves[0].timestamp<1.


def test_invalid_world_tracking_during_press_never_becomes_release_click(tmp_path):
    c=controller(tmp_path);point(c)
    for t in [.42,.50,.58]:c.process(f(t,'pinch',pinch=.15))
    broken=f(.65,'pinch',pinch=.5);broken.world_points=np.full((21,3),np.nan)
    events=c.process(broken)
    assert not any(e.action=='click' for e in events)
    assert c.state=='recovery'


def test_slow_closing_pinch_candidate_survives_until_confirmed_press(tmp_path):
    c=controller(tmp_path);point(c);events=[]
    for t,p in [(.42,.41),(.44,.39),(.48,.37),(.50,.35)]:
        events+=c.process(f(t,'pinch',pointer=(.9,.9),pinch=p))
        assert c.state=='pinch_candidate'
    for t,p in [(.60,.22),(.68,.20),(.92,.50),(.98,.50)]:
        events+=c.process(f(t,'pinch',pointer=(.9,.9),pinch=p))
    clicks=[e for e in events if e.action=='click']
    assert len(clicks)==1 and clicks[0].payload=={'x':.3,'y':.4}
    assert not any(e.action=='move' for e in events)


def test_unpressed_candidate_cancel_requires_open_hysteresis(tmp_path):
    c=controller(tmp_path);point(c)
    for t,p in [(.42,.41),(.46,.44),(.50,.43)]:
        assert not c.process(f(t,'pinch',pinch=p))
        assert c.state=='pinch_candidate'
    events=[]
    for t in [.55,.62]:events+=c.process(f(t,'point',pinch=.7))
    assert c.state=='recovery' and not any(e.action in {'click','drag_start'} for e in events)


def test_scroll_resumption_sets_first_valid_v_baseline_without_cross_gap_wheel(tmp_path):
    c=controller(tmp_path);events=[]
    for t in np.arange(0,.30,.02):events+=c.process(f(t,'victory',wrist=(.5,.50)))
    assert c.state=='scroll' and not any(e.action=='scroll' for e in events)
    assert not c.process(f(.32,'point',wrist=(.5,.60)))
    resumed=c.process(f(.34,'victory',wrist=(.5,.51)))
    assert not any(e.action in {'scroll','move'} for e in resumed)
    next_valid=c.process(f(.36,'victory',wrist=(.5,.49)))
    assert sum(e.payload['dy'] for e in next_valid if e.action=='scroll')==2


def test_active_pointer_tracks_fast_ramp_without_recovery_filter_lag(tmp_path):
    # Synthetic input only: a known 0.8 screen-width/s ray ramp after quiet acquisition.
    c=controller(tmp_path);point(c);errors=[]
    for t in np.arange(.42,1.08,1/30):
        target=.3+.8*(t-.42)
        events=c.process(f(float(t),'point',pointer=(target,.4)))
        moves=[e for e in events if e.action=='move']
        assert moves and c.state=='pointer'
        if t>=.7:errors.append(abs(target-moves[-1].payload['x']))
    assert np.mean(errors)<.035  # <44ms ramp-equivalent lag; not measured camera/OS latency.
    assert c.acquire_filter.beta==.25


def relative_controller(tmp_path):
    from gesture_system.workspace_pointer import WorkspaceMapper,WorkspaceBounds
    from gesture_system.stable_control import StableController
    mapper=WorkspaceMapper({'source_kind':'camera','frame_size':[640,480]},WorkspaceBounds(span=(.35,.35)),relative=True)
    return StableController(GestureLibrary(tmp_path),mapper)


def relative_feature(t,xy=(.9,.8),label='point',pinch=1):
    frame=f(t,label,pinch=pinch);frame.image_points=np.tile(xy,(21,1));frame.frame_size=(640,480)
    return frame


def acquire_relative(c,start=0,xy=(.9,.8)):
    events=[]
    for t in np.arange(start,start+.4,.02):events+=c.process(relative_feature(float(t),xy))
    return events


def test_relative_acquisition_offcenter_and_reacquisition_never_jump(tmp_path):
    c=relative_controller(tmp_path);events=acquire_relative(c)
    assert events and all(np.allclose([e.payload['x'],e.payload['y']],[.5,.5]) for e in events)
    c.process(FrameFeatures.absent(.42));events=acquire_relative(c,.44,xy=(.1,.2))
    assert events and all(np.allclose([e.payload['x'],e.payload['y']],[.5,.5]) for e in events)


def test_relative_acquisition_uses_current_cursor_provider_and_failed_provider_has_no_moves(tmp_path):
    c=relative_controller(tmp_path);c.position_provider=lambda:np.array([.2,.7])
    events=acquire_relative(c)
    assert events and np.allclose(c.position,[.2,.7])
    c.process(FrameFeatures.absent(.42));c.position_provider=lambda:[.8,.1]
    events=acquire_relative(c,.44,xy=(.1,.2));assert events and np.allclose(c.position,[.8,.1])
    for result in [[np.nan,0],[2,0],None]:
        c.reset();c.position_provider=lambda:result
        assert not acquire_relative(c,c._clock+.02) and c.state=='recovery'
    c.position_provider=lambda:(_ for _ in ()).throw(RuntimeError('cursor unavailable'))
    assert not acquire_relative(c,c._clock+.02) and c.state=='recovery'


def test_relative_same_frame_pinch_freezes_anchor_and_scroll_never_moves(tmp_path):
    c=relative_controller(tmp_path);acquire_relative(c)
    assert not c.process(relative_feature(.42,(.1,.1),'pinch',.2))
    c.process(relative_feature(.5,(.1,.1),'pinch',.2))
    events=c.process(relative_feature(.6,(.1,.1),'point',1))+c.process(relative_feature(.68,(.1,.1),'point',1))
    click=[e for e in events if e.action=='click'];assert len(click)==1 and click[0].payload=={'x':.5,'y':.5}
    acquire_relative(c,.7,xy=(.1,.1));events=[]
    for t in np.arange(1.12,1.6,.02):events+=c.process(relative_feature(float(t),(.9,.9),'victory'))
    assert not any(e.action=='move' for e in events)


def test_relative_neutral_rearm_retains_last_output_after_hand_reposition(tmp_path):
    c=relative_controller(tmp_path);acquire_relative(c)
    for t in [.42,.46,.5]:c.process(relative_feature(t,(.865,.8)))
    anchor=c.position.copy();assert anchor[0]>.5
    c.process(relative_feature(.52,label='uncertain'))
    c.process(relative_feature(.74,label='uncertain'))
    assert c.state=='recovery'
    events=acquire_relative(c,.76,xy=(.1,.2))
    assert events and all(np.allclose([e.payload['x'],e.payload['y']],anchor) for e in events)
