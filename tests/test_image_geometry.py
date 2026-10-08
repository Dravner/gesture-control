import numpy as np
import pytest
from gesture_system.types import FrameFeatures
from gesture_system.vision import classify_geometry


def image_points(middle='folded',pinch=False):
    p=np.full((21,2),.5);p[0]=[.5,.68];p[4]=[.32,.6]
    for base,x in [(5,.44),(9,.50),(13,.56),(17,.61)]:
        p[base]=[x,.55]
        state='open' if base==5 else middle if base==9 else 'folded'
        p[base+1]=p[base]+[0,-.03];p[base+2]=p[base]+[0,-.06]
        if state=='open':p[base+3]=p[base]+[0,-.09]
        elif state=='folded':p[base+3]=p[base]+[0,-.005]
        else:p[base+3]=p[base]+[.03,-.055]  # reach in open/folded deadband
    if pinch:p[4]=p[8]+[.005,0]
    return p


def feature(stamp=0,pinch=False,middle='folded'):
    return FrameFeatures(stamp,True,np.zeros(42),(.5,.68),(.8,.2),1,.2,'palm',.99,
        world_points=np.full((21,3),np.nan),image_points=image_points(middle,pinch),frame_size=(640,480),
        geometry_label='palm',geometry_confidence=.99,geometry_pinch=1.)

@pytest.mark.parametrize('middle,pinch,expected',[('folded',False,'point'),('open',False,'victory'),('folded',True,'pinch')])
def test_half_curled_ring_does_not_block_unambiguous_two_finger_control(middle,pinch,expected):
    f=feature(pinch=pinch,middle=middle)
    f.image_points[16]=f.image_points[13]+[.03,-.055]
    result=classify_geometry(f,prefer_image=True)
    assert result.label==expected
    assert result.finger_states[2]=='uncertain'

def test_extended_ring_and_ambiguous_middle_do_not_become_pointer():
    f=feature();f.image_points[16]=f.image_points[13]+[0,-.09]
    assert classify_geometry(f,prefer_image=True).label=='uncertain'
    f=feature(middle='uncertain')
    assert classify_geometry(f,prefer_image=True).label=='uncertain'

def test_controller_caches_actual_image_diagnostics_and_clears_on_absence(tmp_path):
    from gesture_system.stable_control import StableController
    from gesture_system.profiles import GestureLibrary
    from gesture_system.workspace_pointer import WorkspaceMapper
    c=StableController(GestureLibrary(tmp_path),WorkspaceMapper({'source_kind':'camera','frame_size':[640,480]}));c.prefer_image_geometry=True
    c.process(feature(1.))
    assert c.geometry.finger_states==('open','folded','folded','folded')
    c.process(FrameFeatures.absent(1.1))
    assert c.geometry.label=='no_hand' and not c.geometry.finger_states

def test_camera_scroll_uses_palm_motion_instead_of_stale_vector_wrist(tmp_path):
    from gesture_system.stable_control import StableController
    from gesture_system.profiles import GestureLibrary
    from gesture_system.workspace_pointer import WorkspaceMapper
    c=StableController(GestureLibrary(tmp_path),WorkspaceMapper({'source_kind':'camera','frame_size':[640,480]}));c.prefer_image_geometry=True
    events=[]
    for t in np.arange(0,.4,.02):events+=c.process(feature(float(t),middle='open'))
    events=[]
    for t,y in zip(np.arange(.4,1.,.02),np.linspace(0,-.05,30)):
        f=feature(float(t),middle='open');f.image_points[:,1]+=y
        events+=c.process(f)
    assert any(e.action=='scroll' for e in events)
    assert not any(e.action=='move' for e in events)


@pytest.mark.parametrize('world',[None,np.full((21,3),np.nan),np.ones((3,3)),np.arange(63).reshape(21,3)])
def test_image_geometry_ignores_world_and_stale_prediction(world):
    f=feature();f.world_points=world
    result=classify_geometry(f,prefer_image=True)
    assert result.label=='point' and result.confidence==.90
    assert f.geometry_label=='palm'  # no immutable-feature mutation


def test_image_geometry_mirror_rotation_and_deadband():
    f=feature();p=f.image_points.copy();p[:,0]=1-p[:,0];f.image_points=p
    assert classify_geometry(f,prefer_image=True).label=='point'
    xy=(p-[.5,.5])*[640/480,1];angle=.7;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    f.image_points=(xy@R.T)/[640/480,1]+[.5,.5]
    assert classify_geometry(f,prefer_image=True).label=='point'
    f=feature(middle='ambiguous');assert classify_geometry(f,prefer_image=True).label=='uncertain'
    f=feature(middle='open');assert classify_geometry(f,prefer_image=True).label=='victory'


@pytest.mark.parametrize('bad',[None,np.zeros(42),np.full((21,2),np.nan),np.full((21,2),2)])
def test_invalid_image_safely_aborts_confirmed_pinch_without_click(tmp_path,bad):
    from gesture_system.stable_control import StableController
    from gesture_system.profiles import GestureLibrary
    from gesture_system.workspace_pointer import WorkspaceMapper
    c=StableController(GestureLibrary(tmp_path),WorkspaceMapper({'source_kind':'camera','frame_size':[640,480]}));c.prefer_image_geometry=True
    events=[]
    for t in np.arange(0,.4,.02):events+=c.process(feature(float(t)))
    assert any(e.action=='move' for e in events)
    for t in [.42,.5]:c.process(feature(t,pinch=True))
    assert c.state=='pinch_held'
    f=feature(.6);f.image_points=bad
    events=c.process(f)+c.process(feature(.68))
    assert c.state=='recovery' and not any(e.action=='click' for e in events)


def test_default_ray_geometry_and_invalid_world_guard_are_unchanged(tmp_path):
    from gesture_system.stable_control import StableController
    from gesture_system.profiles import GestureLibrary
    c=StableController(GestureLibrary(tmp_path))
    assert not c.prefer_image_geometry
    f=feature();assert classify_geometry(f).source=='invalid_world'
    assert c.process(f)==[] and c.state=='recovery'


def test_invalid_image_during_drag_emits_release_without_click(tmp_path):
    from gesture_system.stable_control import StableController
    from gesture_system.profiles import GestureLibrary
    from gesture_system.workspace_pointer import WorkspaceMapper
    c=StableController(GestureLibrary(tmp_path),WorkspaceMapper({'source_kind':'camera','frame_size':[640,480]}));c.prefer_image_geometry=True
    for t in np.arange(0,.4,.02):c.process(feature(float(t)))
    events=[]
    for t in np.arange(.42,1.1,.02):events+=c.process(feature(float(t),pinch=True))
    assert any(e.action=='drag_start' for e in events) and c.state=='drag'
    f=feature(1.12);f.frame_size=None
    events=c.process(f)
    assert [e.action for e in events]==['drag_end'] and c.state=='recovery'


def test_raw_readonly_image_is_not_mutated_and_bad_aspect_rejected():
    f=feature();original=f.image_points.copy();f.image_points.setflags(write=False)
    assert classify_geometry(f,prefer_image=True).label=='point'
    assert np.array_equal(f.image_points,original)
    f.frame_size=(640,0)
    assert classify_geometry(f,prefer_image=True).source=='invalid_image'
