import numpy as np
from gesture_system.vision import features_from_points
from gesture_system.actions import screen_position

def test_pose_geometry_preserves_global_path():
    pts=np.zeros((21,2));pts[:,1]=np.linspace(.8,.3,21);pts[:,0]=.5
    a=features_from_points(pts,'Right',1,'palm',.9)
    b=features_from_points(pts+np.array([.2,0]),'Right',2,'palm',.9)
    assert np.allclose(a.pose,b.pose)
    assert np.isclose(b.wrist[0]-a.wrist[0],.2)
    assert a.vector().shape==(48,)

def test_pointer_coordinates_bounded():
    assert screen_position(2,-1,1920,1080)==(1919.,0.)
    assert screen_position(.5,.5,1920,1080)==(959.5,539.5)

def test_missing_ax_api_is_handled_without_crash():
    from gesture_system.actions import MacActions
    obj=MacActions();obj.q=object()
    assert isinstance(obj.available(),bool)

class FakeQuartz:
    kCGMouseButtonLeft=0;kCGMouseButtonRight=1;kCGHIDEventTap=0
    kCGEventLeftMouseDown='ld';kCGEventLeftMouseUp='lu'
    kCGEventRightMouseDown='rd';kCGEventRightMouseUp='ru';kCGMouseEventClickState=1
    def __init__(self):self.posted=[]
    def CGEventCreate(self,_):return None
    def CGEventGetLocation(self,_):
        from types import SimpleNamespace
        return SimpleNamespace(x=100,y=200)
    def CGEventCreateMouseEvent(self,_,kind,position,button):return {'kind':kind,'button':button}
    def CGEventSetIntegerValueField(self,e,field,value):e['count']=value
    def CGEventPost(self,_,e):self.posted.append(e)

def adapter():
    from gesture_system.actions import MacActions
    obj=MacActions();obj.q=FakeQuartz();obj.available=lambda:True
    return obj

def test_right_click_during_drag_releases_left_button():
    from gesture_system.types import ControlEvent
    a=adapter()
    for action in ['drag_start','right_click']:a.execute(ControlEvent('test',action,0,{}))
    a.release()
    assert [e['kind'] for e in a.q.posted]==['ld','lu','rd','ru']

def test_doubleclick_posts_two_pairs_with_click_state():
    from gesture_system.types import ControlEvent
    a=adapter();a.execute(ControlEvent('test','click',0,{'count':2}))
    assert [e['kind'] for e in a.q.posted]==['ld','lu','ld','lu']
    assert [e['count'] for e in a.q.posted]==[1,1,2,2]

def test_phone_pose_cannot_become_builtin_thumb_hotkey():
    from gesture_system.vision import guard_discrete_label
    p=np.zeros((21,3),dtype=float)
    for base,x in zip([1,5,9,13,17],[-.7,-.3,0,.3,.6]):
        p[base:base+4]=[[x,-.3,0],[x,-.6,0],[x,-.9,0],[x,-1.2,0]]
    for base in [5,9,13]:p[base+2,1]=-.45;p[base+3,1]=-.35
    assert guard_discrete_label('thumbs_up',p)=='no_gesture'
    p[19,1]=-.45;p[20,1]=-.35
    assert guard_discrete_label('thumbs_up',p)=='thumbs_up'

def test_extended_fingers_cannot_become_builtin_fist_pause():
    from gesture_system.vision import guard_discrete_label
    p=np.zeros((21,3),dtype=float)
    for base,x in zip([1,5,9,13,17],[-.7,-.3,0,.3,.6]):p[base:base+4]=[[x,-.3,0],[x,-.6,0],[x,-.9,0],[x,-1.2,0]]
    assert guard_discrete_label('fist',p)=='no_gesture'


def geometry_points(index=1.,middle=.2,ring=.2,pinky=.2):
    p=np.zeros((21,3));p[0]=[0,0,0];p[9]=[0,-.045,0]
    p[1:5]=[[-.02,-.015,0],[-.04,-.02,0],[-.06,-.025,0],[-.08,-.03,0]]
    for base,x,y,reach in [(5,-.02,-.04,index),(9,0,-.045,middle),(13,.02,-.04,ring),(17,.037,-.03,pinky)]:
        p[base]=[x,y,0]
        # Three equal bones: selected endpoint/chain reach, no threshold-derived labels.
        angle=np.arccos(np.clip((9*reach*reach-5)/4,-1,1))
        length=.07/3
        p[base+1]=p[base]+[0,-length,0]
        p[base+2]=p[base+1]+[np.sin(angle)*length,-np.cos(angle)*length,0]
        p[base+3]=p[base+2]+[0,-length,0]
        if reach<.65:p[base+3]=p[base]+[0,-.005,0]
    return p

def test_geometry_has_deadband_instead_of_false_victory_to_point():
    from gesture_system.vision import classify_geometry
    from gesture_system.types import FrameFeatures
    def observation(middle):
        f=FrameFeatures(1,True,np.zeros(42),(.5,.5),(.5,.4),1,.2,'thumbs_up',.99,geometry_points(middle=middle))
        return classify_geometry(f)
    assert observation(1.).label=='victory'
    assert observation(.77).label=='uncertain'
    assert observation(.2).label=='point'

def test_geometry_ignores_static_mlp_and_reports_world_pinch_separately():
    from gesture_system.vision import classify_geometry
    from gesture_system.types import FrameFeatures
    p=geometry_points();p[4]=p[8]+[.006,0,0]
    f=FrameFeatures(1,True,np.zeros(42),(.5,.5),(.5,.4),4,.2,'victory',.99,p)
    result=classify_geometry(f)
    assert result.label=='pinch' and result.pinch<.24


def test_tracker_populates_geometry_and_raw_correspondences_without_camera():
    from types import SimpleNamespace
    from gesture_system.vision import HandTracker,features_from_points
    world=geometry_points(middle=.77);xy=world[:,:2]*2+[.5,.6]
    landmark=lambda p:SimpleNamespace(x=float(p[0]),y=float(p[1]),z=float(p[2]) if len(p)>2 else 0)
    result=SimpleNamespace(hand_landmarks=[[landmark(p) for p in xy]],handedness=[[SimpleNamespace(category_name='Right')]],
                           hand_world_landmarks=[[landmark(p) for p in world]])
    tracker=HandTracker.__new__(HandTracker);tracker.model=None;tracker._ts=-1
    tracker.mp=SimpleNamespace(Image=lambda **kwargs:None,ImageFormat=SimpleNamespace(SRGB=0))
    tracker.landmarker=SimpleNamespace(detect_for_video=lambda image,stamp:result)
    f=tracker.process(np.zeros((240,320,3),np.uint8),1.)
    assert f.geometry_label=='uncertain'
    assert f.frame_size==(320,240) and np.allclose(f.image_points,xy)
    baseline=features_from_points(xy,'Right',1,'no_gesture',0)
    assert np.allclose(f.vector(),baseline.vector())


def test_selected_display_origin_and_anchor_ignore_live_cursor():
    from types import SimpleNamespace
    from gesture_system.actions import MacActions
    from gesture_system.types import ControlEvent
    events=[]
    q=SimpleNamespace(kCGEventMouseMoved=1,kCGEventLeftMouseDragged=2,kCGMouseButtonLeft=0,kCGMouseButtonRight=1,
       kCGEventRightMouseDown=3,kCGEventLeftMouseDown=4,kCGEventRightMouseUp=5,kCGEventLeftMouseUp=6,
       kCGMouseEventClickState=7,kCGHIDEventTap=8,CGMainDisplayID=lambda:1,
       CGDisplayBounds=lambda identifier:SimpleNamespace(origin=SimpleNamespace(x=-1000,y=50),size=SimpleNamespace(width=1000,height=700)),
       CGEventCreate=lambda source:object(),CGEventGetLocation=lambda event:SimpleNamespace(x=99,y=99),
       CGEventCreateMouseEvent=lambda source,typ,pos,button:{'type':typ,'pos':pos},
       CGEventSetIntegerValueField=lambda *args:None,CGEventPost=lambda tap,event:events.append(event))
    actions=MacActions();actions.q=q;actions.available=lambda:True
    actions.set_display(2)
    actions.execute(ControlEvent('point','move',1,{'x':.5,'y':.5}))
    assert events[-1]['pos']==(-500.5,399.5)
    actions.execute(ControlEvent('pinch','click',2,{'x':.25,'y':.75}))
    assert events[-2]['pos']==(-750.25,574.25) and events[-1]['pos']==(-750.25,574.25)
    actions.execute(ControlEvent('legacy','click',3,{}))
    assert events[-1]['pos']==(99,99)


def test_current_position_reads_selected_logical_display_without_posting():
    from types import SimpleNamespace
    from gesture_system.actions import MacActions
    posted=[]
    q=SimpleNamespace(CGDisplayBounds=lambda identifier:SimpleNamespace(origin=SimpleNamespace(x=-1000,y=50),size=SimpleNamespace(width=1000,height=700)),
        CGMainDisplayID=lambda:1,CGEventCreate=lambda source:None,
        CGEventGetLocation=lambda event:SimpleNamespace(x=-500.5,y=399.5),CGEventPost=lambda *args:posted.append(args))
    actions=MacActions(display_id=2);actions.q=q
    assert np.allclose(actions.current_position(),[.5,.5]) and posted==[]

def test_pixel_scroll_is_continuous_and_preserves_fractional_travel():
    from gesture_system.types import ControlEvent
    a=adapter();q=a.q;q.kCGScrollEventUnitPixel='pixel';q.kCGScrollEventUnitLine='line';q.kCGScrollWheelEventIsContinuous=99
    q.CGEventCreateScrollWheelEvent=lambda source,unit,axes,dy:{'unit':unit,'dy':dy}
    for i in range(10):a.execute(ControlEvent('victory','scroll',i,{'dy':.4,'unit':'pixel'}))
    assert sum(e['dy'] for e in q.posted)==4
    assert all(e['unit']=='pixel' and e['count']==1 for e in q.posted)
