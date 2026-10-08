import numpy as np
import pytest
from test_image_geometry import feature

def test_finger_hold_survives_low_confidence_curled_joints_without_arming_other_pose():
    from gesture_system.pose_stability import ImagePoseLatch
    latch=ImagePoseLatch();f=feature(0);f.joint_confidence=np.full(21,.9)
    assert latch.update(f).label=='point'
    for t in [.03,.06,.09,.12]:
        f=feature(t);f.joint_confidence=np.full(21,.9);f.joint_confidence[15:17]=.1;f.joint_confidence[19:21]=.1
        f.image_points[16]=f.image_points[13]+[0,-.09]
        assert latch.update(f).label=='point'
    f.timestamp=.4;assert latch.update(f).label=='uncertain'

@pytest.mark.parametrize('mirrored',[False,True])
def test_projection_of_folded_fingers_does_not_read_as_open_chain(mirrored):
    from gesture_system.pose_stability import ImagePoseLatch
    f=feature(0)
    # A folded finger may look like a straight short segment in 2D.
    for base in [9,13,17]:
        for j in range(1,4):f.image_points[base+j]=f.image_points[base]+[0,-.003*j]
    if mirrored:f.image_points[:,0]=1-f.image_points[:,0]
    assert ImagePoseLatch().update(f).label=='point'

def test_uncertain_middle_does_not_start_scroll_without_extension_evidence():
    from gesture_system.pose_stability import ImagePoseLatch
    latch=ImagePoseLatch();assert latch.update(feature(0)).label=='point'
    for t in [.03,.06,.09]:assert latch.update(feature(t,middle='ambiguous')).label=='point'
    assert latch.update(feature(.12,middle='open')).label=='victory'

@pytest.mark.parametrize('mirrored',[False,True])
def test_long_palm_stay_rejects_point_and_pinch(mirrored):
    from gesture_system.pose_stability import ImagePoseLatch
    f=feature(0,middle='open')
    for base in [13,17]:f.image_points[base+3]=f.image_points[base]+[0,-.09]
    if mirrored:f.image_points[:,0]=1-f.image_points[:,0]
    assert ImagePoseLatch().update(f).label=='palm'

def modern_controller(tmp_path):
    from gesture_system.stable_control import StableController
    from gesture_system.profiles import GestureLibrary
    from gesture_system.workspace_pointer import WorkspaceMapper
    c=StableController(GestureLibrary(tmp_path),WorkspaceMapper({'source_kind':'camera','frame_size':[640,480]},relative=True))
    c.prefer_image_geometry=True;c.robust_geometry=True
    for t in np.arange(0,.4,.02):c.process(feature(float(t)))
    assert c.state=='pointer'
    return c

def test_pointer_gap_holds_mode_without_commands_and_rebases_on_return(tmp_path):
    from gesture_system.types import FrameFeatures
    c=modern_controller(tmp_path);position=c.position.copy()
    for t in [.42,.45,.48]:assert c.process(FrameFeatures.absent(t))==[] and c.state=='pointer'
    f=feature(.5);f.image_points[:,0]+=.1
    events=c.process(f)
    assert c.state=='pointer' and np.allclose(c.position,position)
    assert not any(e.action in {'click','scroll','hotkey'} for e in events)
    for t in [.55,.7,.9]:c.process(FrameFeatures.absent(t))
    assert c.state=='recovery'

def test_single_spurious_scroll_pose_does_not_take_cursor_mode(tmp_path):
    c=modern_controller(tmp_path)
    c.process(feature(.42,middle='open'));assert c.state=='pointer'
    c.process(feature(.45));assert c.state=='pointer'
    for t in [.5,.54,.58]:c.process(feature(t,middle='open'))
    assert c.state=='scroll_pending'

@pytest.mark.parametrize('fps',[15,30,60])
def test_pixel_scroll_same_travel_at_different_frame_rates(tmp_path,fps):
    c=modern_controller(tmp_path);c.smooth_scroll=True;c.scroll_pixels_per_unit=1200.
    events=[]
    for t in np.arange(.42,.8,1/fps):c.process(feature(float(t),middle='open'))
    for t in np.arange(.8,1.8,1/fps):
        f=feature(float(t),middle='open');f.image_points[:,1]-=.06*min(t-.8,.6)/.6
        events+=c.process(f)
    scroll=[e for e in events if e.action=='scroll']
    assert scroll and all(e.payload.get('unit')=='pixel' for e in scroll)
    assert 65<=sum(e.payload['dy'] for e in scroll)<=73
    assert not any(e.action=='move' for e in events)

def test_horizontal_two_finger_stroke_emits_once_and_no_vertical_scroll(tmp_path):
    c=modern_controller(tmp_path);c.navigation_pose='victory';c.smooth_scroll=True
    for t in np.arange(.42,.8,.02):c.process(feature(float(t),middle='open'))
    events=[]
    for t,x in zip(np.arange(.8,1.4,.02),np.linspace(0,.13,30)):
        f=feature(float(t),middle='open');f.image_points[:,0]+=x;events+=c.process(f)
    assert [e.gesture for e in events if e.action=='hotkey']==['swipe_left']
    assert not any(e.action in {'move','scroll'} for e in events)

def test_short_curled_low_confidence_tips_can_start_pointer_with_visible_knuckles():
    from gesture_system.pose_stability import ImagePoseLatch
    f=feature(0);f.joint_confidence=np.full(21,.9)
    f.joint_confidence[[15,16,19,20]]=.15
    result=ImagePoseLatch().update(f)
    assert result.label=='point'

def test_session_keeps_hand_and_joint_quality_for_independent_analysis(tmp_path):
    from gesture_system.gui import SessionRecorder
    f=feature(1);f.handedness='Right';f.joint_confidence=np.linspace(.1,.9,21)
    recorder=SessionRecorder(tmp_path,{'kind':'camera'},'test',[])
    recorder.add_frame(f,'point',.9,'pointer',False,'test');recorder.finish('test')
    a=np.load(recorder.path/'frames.npz')
    assert a['handedness'][0]=='Right' and np.allclose(a['joint_confidence'][0],f.joint_confidence)

def test_pixel_scroll_steady_hand_does_not_alternate_wheel_direction(tmp_path):
    c=modern_controller(tmp_path);c.smooth_scroll=True;c.scroll_dead_zone=.006
    for t in np.arange(.42,.8,.02):c.process(feature(float(t),middle='open'))
    for t,y in zip(np.arange(.8,1.3,.02),np.linspace(0,-.04,25)):
        f=feature(float(t),middle='open');f.image_points[:,1]+=y;c.process(f)
    events=[]
    for i,t in enumerate(np.arange(1.3,2.3,.02)):
        f=feature(float(t),middle='open');f.image_points[:,1]+=-.04+(.0055 if i%2 else -.0055);events+=c.process(f)
    signs=[np.sign(e.payload['dy']) for e in events if e.action=='scroll']
    assert not (1 in signs and -1 in signs)

def test_aspect_correction_does_not_reject_hand_on_right_side_of_wide_frame():
    from gesture_system.pose_stability import ImagePoseLatch
    f=feature(0);f.frame_size=(1920,1080);f.image_points[:,0]+=.25
    assert ImagePoseLatch().update(f).label=='point'

def test_advanced_templates_never_match_zero_packed_unknown_joints(tmp_path):
    from gesture_system.engine import Engine
    from gesture_system.profiles import GestureLibrary
    e=Engine(GestureLibrary(tmp_path),load_models=False)
    f=feature(1);f.world_points=None;f.joint_confidence=np.ones(21);f.joint_confidence[16]=0
    e.process(f)
    assert e.personal_phase=='invalid' and e.personal_result['reason']=='недостаточно уверенных суставов для полного шаблона'

def test_unreadable_thumb_during_confirmed_pinch_is_not_a_release_click(tmp_path):
    c=modern_controller(tmp_path)
    for t in [.42,.50]:c.process(feature(t,pinch=True))
    assert c.state=='pinch_held'
    events=[]
    for t in [.53,.59,.65,.77]:
        f=feature(t,pinch=True);f.joint_confidence=np.ones(21);f.joint_confidence[4]=0
        events+=c.process(f)
    assert not any(e.action=='click' for e in events)
    assert c.state=='recovery'
