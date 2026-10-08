import numpy as np
from gesture_system.types import FrameFeatures
from gesture_system.profiles import GestureLibrary

def hand():
    p=np.zeros((21,2))
    for base,x in zip([1,5,9,13,17],[-.7,-.3,0,.3,.6]):
        p[base:base+4]=np.array([[x,-.3],[x,-.6],[x,-.9],[x,-1.2]])
    return p

def vector(p):
    return np.r_[p.reshape(-1),.5,.5,.5,.4,1.,.2].astype(np.float32)

def test_anatomy_invariant_to_rotation_scale_and_mirror():
    from gesture_system.personal import anatomy
    p=hand();theta=.8;r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    assert np.allclose(anatomy(p),anatomy(p@r*1.7+2),atol=1e-5)
    assert np.allclose(anatomy(p),anatomy(p*np.array([-1,1])),atol=1e-5)

def test_static_shape_accepts_rotation_but_rejects_folded_fingers(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    lib=GestureLibrary(tmp_path);p=hand();key=lib.add('open','static',np.tile(vector(p),(30,1)),'none')
    r=PersonalRecognizer(lib);theta=.7;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    assert r.static_match((p@rot).reshape(-1)).identifier==key
    q=p.copy();q[7]=[-.3,-.2];q[8]=[-.3,-.1]
    assert r.static_match(q.reshape(-1)) is None

def rotation(turns=2,n=61):
    p=hand();rows=[]
    for theta in np.linspace(0,turns*2*np.pi,n):
        rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        rows.append(vector(p@rot))
    return np.array(rows)

def test_dynamic_order_full_cycles_speed_and_stationary(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    lib=GestureLibrary(tmp_path);key=lib.add('turn','dynamic',rotation(),'none');r=PersonalRecognizer(lib)
    assert r.classify_dynamic(rotation(n=101)).identifier==key
    assert r.classify_dynamic(rotation(n=31)).identifier==key
    assert r.classify_dynamic(rotation(turns=1)) is None
    assert r.classify_dynamic(np.tile(rotation()[0],(61,1))) is None

def test_live_episode_emits_once_and_rearms_after_hand_leaves(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    lib=GestureLibrary(tmp_path);key=lib.add('turn','dynamic',rotation(),'none');r=PersonalRecognizer(lib);out=[]
    for repetition in range(2):
        for i,v in enumerate(rotation()):
            f=FrameFeatures(repetition*5+i*.05,True,v[:42],tuple(v[42:44]),tuple(v[44:46]),v[46],v[47],'fist',.99)
            result=r.update(f)
            if result.match:out.append(result.match.identifier)
        for stamp in [3.2,3.4,3.6]:
            result=r.update(FrameFeatures.absent(repetition*5+stamp))
            if result.match:out.append(result.match.identifier)
    assert out==[key,key]

def test_world_static_remains_valid_with_foreshortened_image(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    p=hand();world=np.column_stack([p,np.zeros(21)])*.05
    lib=GestureLibrary(tmp_path);key=lib.add('open3D','static',np.tile(vector(p),(30,1)),'none',world_samples=np.tile(world,(30,1,1)))
    r=PersonalRecognizer(lib);theta=1.3;rot=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    viewed=world@rot
    assert r.static_match((viewed[:,:2]*20).reshape(-1),viewed).identifier==key

def test_world_wrist_roll_preserves_temporal_repetitions(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    p=hand();rows=np.tile(vector(p),(61,1));base=np.column_stack([p,np.zeros(21)])*.05
    def roll(turns,n):
        values=[]
        for theta in np.linspace(0,turns*2*np.pi,n):
            rot=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
            values.append(base@rot)
        return np.array(values)
    lib=GestureLibrary(tmp_path);key=lib.add('roll3D','dynamic',rows,'none',world_samples=roll(2,61));r=PersonalRecognizer(lib)
    assert r.classify_dynamic(np.tile(vector(p),(101,1)),roll(2,101)).identifier==key
    assert r.classify_dynamic(rows,roll(1,61)) is None

def test_unknown_motion_does_not_block_normal_controls(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    lib=GestureLibrary(tmp_path);lib.add('turn','dynamic',rotation(),'none');r=PersonalRecognizer(lib)
    p=hand();p[7]=[-.3,-.2];p[8]=[-.3,-.1]
    v=vector(p)
    assert not r.update(FrameFeatures(0,True,v[:42],(.2,.5),(.2,.4),1.,.2,'point',.99)).reserved
    assert not r.update(FrameFeatures(.05,True,v[:42],(.3,.5),(.3,.4),1.,.2,'point',.99)).reserved
    assert not r.update(FrameFeatures(.075,True,v[:42],(.35,.5),(.35,.4),1.,.2,'point',.99)).reserved

def test_orientation_sensitive_custom_poses_do_not_alias(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    lib=GestureLibrary(tmp_path);p=hand()
    first=lib.add('up','static',np.tile(vector(p),(20,1)),'none')
    second=lib.add('down','static',np.tile(vector(-p),(20,1)),'none')
    lib.update_mapping(first,'none',orientation_sensitive=True)
    lib.update_mapping(second,'none',orientation_sensitive=True)
    r=PersonalRecognizer(lib)
    assert r.static_match(p.reshape(-1)).identifier==first
    assert r.static_match((-p).reshape(-1)).identifier==second

def test_pause_between_rotation_phases_does_not_split_full_gesture(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    lib=GestureLibrary(tmp_path);key=lib.add('twice','dynamic',rotation(),'none');r=PersonalRecognizer(lib);out=[]
    one=rotation(turns=1,n=31);rows=np.concatenate([one,np.tile(one[-1],(9,1)),one])
    for i,v in enumerate(rows):
        f=FrameFeatures(i*.05,True,v[:42],tuple(v[42:44]),tuple(v[44:46]),v[46],v[47],'fist',.99)
        result=r.update(f)
        if result.match:out.append(result.match.identifier)
    result=r.update(FrameFeatures.absent(len(rows)*.05+.4))
    if result.match:out.append(result.match.identifier)
    assert out==[key]

def test_spread_fingers_differ_even_when_joint_bends_are_equal():
    from gesture_system.personal import anatomy,shape_descriptor
    p=hand();q=p.copy()
    for base,angle in [(5,-.3),(9,.3)]:
        origin=p[base].copy();rotation_matrix=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        q[base:base+4]=(p[base:base+4]-origin)@rotation_matrix+origin
    assert np.allclose(anatomy(p),anatomy(q),atol=1e-5)
    assert np.linalg.norm(shape_descriptor(p)-shape_descriptor(q))>.03

def test_short_occlusion_between_two_phases_preserves_episode(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    lib=GestureLibrary(tmp_path);key=lib.add('twice','dynamic',rotation(),'none');r=PersonalRecognizer(lib);out=[]
    one=rotation(turns=1,n=31)
    for offset in [0.,1.8]:
        for i,v in enumerate(one):
            result=r.update(FrameFeatures(offset+i*.05,True,v[:42],tuple(v[42:44]),tuple(v[44:46]),v[46],v[47],'fist',.99))
            if result.match:out.append(result.match.identifier)
        if offset==0:
            for stamp in [1.55,1.65,1.75]:
                result=r.update(FrameFeatures.absent(stamp))
                if result.match:out.append(result.match.identifier)
    result=r.update(FrameFeatures.absent(3.75))
    if result.match:out.append(result.match.identifier)
    assert out==[key]

def test_personal_static_cursor_mapping_is_continuous(tmp_path):
    from gesture_system.engine import Engine
    p=hand();lib=GestureLibrary(tmp_path);key=lib.add('mycursor','static',np.tile(vector(p),(20,1)),'move')
    engine=Engine(lib,load_models=False);engine.control_mode='personal';events=[]
    for i in range(10):
        events+=engine.process(FrameFeatures(i*.05,True,p.reshape(-1),(.5,.5),(.2+i*.03,.4),1.,.2,'thumbs_up',.99))
    moves=[e for e in events if e.action=='move']
    assert len(moves)==10
    assert all(e.gesture==key and 'x' in e.payload for e in moves)
    assert moves[-1].payload['x']<moves[0].payload['x']

def test_dynamic_completion_on_missing_hand_cannot_start_drag(tmp_path):
    from gesture_system.engine import Engine
    lib=GestureLibrary(tmp_path);key=lib.add('turn','dynamic',rotation(),'drag_start');e=Engine(lib,load_models=False);e.control_mode='personal'
    for i,v in enumerate(rotation()):e.process(FrameFeatures(i*.05,True,v[:42],tuple(v[42:44]),tuple(v[44:46]),v[46],v[47],'fist',.99))
    out=e.process(FrameFeatures.absent(3.4))
    assert not any(event.action=='drag_start' for event in out)
    assert not e._drag

def test_static_command_cannot_fire_inside_reserved_dynamic_episode(tmp_path):
    from gesture_system.engine import Engine
    lib=GestureLibrary(tmp_path);p=hand();lib.add('open','static',np.tile(vector(p),(20,1)),'hotkey',['space']);lib.add('turn','dynamic',rotation(),'none')
    e=Engine(lib,load_models=False);out=[]
    for i,v in enumerate(rotation()):out+=e.process(FrameFeatures(i*.05,True,v[:42],tuple(v[42:44]),tuple(v[44:46]),v[46],v[47],'fist',.99))
    assert not any(event.action=='hotkey' for event in out)

def test_custom_pinch_pose_mapping_works_in_personal_mode(tmp_path):
    from gesture_system.engine import Engine
    p=hand();p[4]=p[8]+[.001,0];lib=GestureLibrary(tmp_path);lib.add('my_pinch','static',np.tile(vector(p),(20,1)),'hotkey',['space'])
    e=Engine(lib,load_models=False);e.control_mode='personal';out=[]
    for i in range(20):out+=e.process(FrameFeatures(i*.05,True,p.reshape(-1),(.5,.5),(.5,.4),.03,.2,'point',.99))
    assert sum(event.action=='hotkey' for event in out)==1

def test_partial_world_match_uses_world_shape_bank(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    base=np.column_stack([hand(),np.zeros(21)])*.05
    def roll(turns):
        return np.array([base@np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]) for a in np.linspace(0,turns*2*np.pi,61)])
    world=roll(2);rows=np.array([vector(p[:,:2]*20) for p in world]);lib=GestureLibrary(tmp_path);lib.add('roll','dynamic',rows,'none',world_samples=world)
    r=PersonalRecognizer(lib);angle=1.3;view=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    query=roll(1)@view;qrows=np.array([vector(p[:,:2]*20) for p in query])
    assert r.classify_dynamic(qrows,query) is None
    assert r.last_partial

def test_pointer_only_mode_cannot_dispatch_hotkeys(tmp_path):
    from gesture_system.engine import Engine
    lib=GestureLibrary(tmp_path);e=Engine(lib,load_models=False);e.control_mode='pointer';out=[]
    for i in range(20):out+=e.process(FrameFeatures(i*.05,True,hand().reshape(-1),(.5,.5),(.5,.4),1.,.2,'thumbs_up',.99))
    assert not any(event.action=='hotkey' for event in out)

def test_invalid_world_features_cannot_dispatch_or_pollute_history(tmp_path):
    from gesture_system.engine import Engine
    lib=GestureLibrary(tmp_path);e=Engine(lib,load_models=False)
    assert e.process(FrameFeatures(1.,True,hand().reshape(-1),(.5,.5),(.5,.4),1.,.2,'thumbs_up',.99,np.full((21,3),np.nan)))==[]
    assert not e.personal.rows

def test_personal_radius_calibration_uses_enrollment_and_respects_manual_override(tmp_path):
    lib=GestureLibrary(tmp_path);p=hand();rows=np.tile(vector(p),(40,1))
    rows[:,16]+=.4*np.sin(np.linspace(0,2*np.pi,40))
    key=lib.add('variable','static',rows,'none')
    assert lib.get(key)['personal_radius_auto']
    assert lib.get(key)['personal_radius']>.07
    lib.update_mapping(key,'none',threshold=.10)
    lib.add_example(key,rows)
    assert np.isclose(lib.get(key)['personal_radius'],.05)
    assert not lib.get(key)['personal_radius_auto']

def test_repeated_rotation_count_cannot_be_warped_into_different_count(tmp_path):
    from gesture_system.personal import PersonalRecognizer
    lib=GestureLibrary(tmp_path);lib.add('two_turns','dynamic',rotation(),'none');r=PersonalRecognizer(lib)
    assert r.classify_dynamic(rotation(1.5)) is None
    assert r.classify_dynamic(rotation(3)) is None
