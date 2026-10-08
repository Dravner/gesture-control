import json
import numpy as np
import pytest
from gesture_system.types import FrameFeatures
from gesture_system.workspace_pointer import WorkspaceBounds,WorkspaceMapper


def context():return {'source_kind':'camera','camera_index':0,'frame_size':[640,480],'display':{'id':1,'geometry':[0,0,1000,700]}}

def feature(x=.5,y=.5):
    return FrameFeatures(1.,True,np.zeros(42),(.1,.1),(.9,.9),1.,.2,'point',.99,
                         world_points=np.ones((21,3)),image_points=np.tile([x,y],(21,1)),frame_size=(640,480))


def test_mirrored_workspace_edges_clamp_and_center():
    mapper=WorkspaceMapper(context())
    for xy,expected in [((.25,.25),(1,0)),((.75,.75),(0,1)),((.5,.5),(.5,.5)),((.05,.95),(1,1))]:
        assert np.allclose(mapper.map_feature(feature(*xy)),expected)
    points=feature();points.image_points[0]=[.95,.95]
    assert np.allclose(mapper.map_feature(points),(.5,.5))  # median rejects one palm outlier
    points.pointer=(0,0);assert np.allclose(mapper.map_feature(points),(.5,.5))


def test_palm_orientation_world_transforms_do_not_change_image_mapping():
    mapper=WorkspaceMapper(context());f=feature(.35,.65);expected=mapper.map_feature(f)
    f.world_points=np.arange(63).reshape(21,3)*-.01
    assert np.allclose(mapper.map_feature(f),expected)
    f.world_points=None;assert np.allclose(mapper.map_feature(f),expected)


def test_drag_delta_is_palm_relative_mirrored_and_does_not_clamp_anchor():
    mapper=WorkspaceMapper(context());reference=mapper.camera_wrist(feature())
    current=mapper.camera_wrist(feature(.55,.45))
    assert np.allclose(reference,[.5,.5,0])
    assert np.allclose(mapper.translate_anchor([.4,.6],reference,current),[.3,.5])
    assert np.allclose(mapper.translate_anchor([0,1],reference,current),[-.1,.9])


@pytest.mark.parametrize('change',[
    lambda f:setattr(f,'present',False),lambda f:setattr(f,'image_points',None),
    lambda f:setattr(f,'image_points',np.zeros(42)),lambda f:setattr(f,'image_points',np.full((21,2),np.nan)),
    lambda f:setattr(f,'image_points',np.full((21,2),2)),lambda f:setattr(f,'frame_size',(1280,720)),
    lambda f:setattr(f,'frame_size',None),
])
def test_invalid_feature_cannot_map_or_start_drag(change):
    mapper=WorkspaceMapper(context());f=feature();change(f)
    for method in [mapper.map_feature,mapper.camera_wrist]:
        with pytest.raises(ValueError):method(f)


@pytest.mark.parametrize('center,span',[( (.5,.5),(.14,.5)),((.5,.5),(.91,.5)),((.1,.5),(.5,.5)),((np.nan,.5),(.5,.5)),((.5,),(.5,.5))])
def test_workspace_bounds_validate(center,span):
    with pytest.raises(ValueError):WorkspaceBounds(center=center,span=span)


def test_context_deepcopy_and_versioned_atomic_serialization(tmp_path):
    original=context();mapper=WorkspaceMapper(original,WorkspaceBounds((.45,.55),(.6,.7)))
    original['display']['id']=88
    assert mapper.matches_context(context()) and not mapper.matches_context(original)
    path=tmp_path/'workspace_config.json';mapper.save(path);data=json.loads(path.read_text())
    assert data['kind']=='camera_workspace_2d' and data['version']==2 and not path.with_suffix('.json.tmp').exists()
    loaded=WorkspaceMapper.load(path)
    assert loaded.matches_context(context()) and np.allclose(loaded.map_feature(feature(.3,.7)),mapper.map_feature(feature(.3,.7)))
    data['kind']='metric_finger_ray_plane';path.write_text(json.dumps(data))
    with pytest.raises(ValueError):WorkspaceMapper.load(path)


def test_context_and_drag_inputs_require_known_finite_shapes():
    with pytest.raises(ValueError):WorkspaceMapper({})
    mapper=WorkspaceMapper(context())
    for reference in [np.zeros(2),[np.nan,0,0]]:
        with pytest.raises(ValueError):mapper.translate_anchor([.5,.5],reference,[.5,.5,0])


def test_public_hand_point_is_raw_image_median_and_never_requires_world():
    mapper=WorkspaceMapper(context());f=feature(.3,.6);f.world_points=np.full((21,3),np.nan)
    assert np.allclose(mapper.hand_point(f),[.3,.6])
    assert np.allclose(mapper.map_feature(f),[.9,.7])
    f.frame_size=0
    with pytest.raises(ValueError):mapper.hand_point(f)


def test_stable_controller_freezes_workspace_target_before_same_frame_pinch(tmp_path):
    from gesture_system.profiles import GestureLibrary
    from gesture_system.stable_control import StableController
    controller=StableController(GestureLibrary(tmp_path),WorkspaceMapper(context()))
    def observation(stamp,pinch,x=.5):
        f=feature(x,.5);f.timestamp=stamp;f.geometry_label='pinch' if pinch<.42 else 'point'
        f.geometry_confidence=.98;f.geometry_pinch=pinch;return f
    for stamp in np.arange(0,.4,.02):controller.process(observation(float(stamp),1))
    assert np.allclose(controller.position,[.5,.5])
    candidate=controller.process(observation(.42,.2,.9))
    assert not any(event.action=='move' for event in candidate)
    events=[]
    for stamp,pinch in [(.50,.2),(.60,.2),(.68,1),(.74,1)]:events+=controller.process(observation(stamp,pinch,.9))
    clicks=[event for event in events if event.action=='click']
    assert len(clicks)==1 and np.allclose([clicks[0].payload['x'],clicks[0].payload['y']],[.5,.5])


def test_relative_workspace_off_center_anchor_and_mirrored_gain():
    mapper=WorkspaceMapper(context(),WorkspaceBounds(span=(.35,.35)),relative=True)
    f=feature(.9,.8)
    assert np.allclose(mapper.acquisition_point(f),[.9/.35,.8/.35])
    with pytest.raises(ValueError):mapper.map_feature(f)
    mapper.begin_pointer(f)
    assert np.allclose(mapper.map_feature(f),[.5,.5])
    assert np.allclose(mapper.map_feature(feature(.865,.835)),[.6,.6])


def test_relative_edge_rebases_so_reverse_is_immediate():
    mapper=WorkspaceMapper(context(),relative=True);mapper.begin_pointer(feature(.5,.5))
    assert np.allclose(mapper.map_feature(feature(.95,.95)),[0,1])
    assert np.allclose(mapper.map_feature(feature(.90,.90)),[.1,.9])


def test_relative_configuration_v2_does_not_serialize_live_references_and_loads_v1(tmp_path):
    path=tmp_path/'workspace_config.json';mapper=WorkspaceMapper(context(),relative=True)
    mapper.begin_pointer(feature(.9,.8),[.3,.7]);mapper.save(path)
    data=json.loads(path.read_text());assert data['version']==2 and data['relative'] is True
    loaded=WorkspaceMapper.load(path);assert loaded.relative
    with pytest.raises(ValueError):loaded.map_feature(feature(.9,.8))
    data['version']=1;data.pop('relative');path.write_text(json.dumps(data))
    loaded=WorkspaceMapper.load(path);assert not loaded.relative
    assert np.allclose(loaded.map_feature(feature(.5,.5)),[.5,.5])
