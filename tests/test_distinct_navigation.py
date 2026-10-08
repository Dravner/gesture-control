import numpy as np
from test_pose_stability import modern_controller
from test_image_geometry import feature


def three(t):
    f=feature(float(t),middle='open');f.image_points[16]=f.image_points[13]+[0,-.09];return f

def test_three_fingers_have_distinct_navigation_pose():
    from gesture_system.pose_stability import ImagePoseLatch
    assert ImagePoseLatch().update(three(0)).label=='navigation'

def test_diagonal_two_finger_scroll_never_navigates(tmp_path):
    c=modern_controller(tmp_path);c.navigation_pose='three';c.smooth_scroll=True;events=[]
    for t in np.arange(.42,1.4,.02):
        f=feature(float(t),middle='open');f.image_points+=np.array([.13*max(t-.7,0),-.08*max(t-.7,0)]);events+=c.process(f)
    assert any(e.action=='scroll' for e in events)
    assert not any(e.action in {'move','hotkey'} for e in events)

def test_three_finger_swipe_without_stationary_palm_arm_fires_once(tmp_path):
    c=modern_controller(tmp_path);c.navigation_pose='three';events=[]
    for t in np.arange(.42,1.4,.02):
        f=three(t);f.image_points[:,0]+=.2*(t-.42);events+=c.process(f)
    assert [e.gesture for e in events if e.action=='hotkey']==['swipe_left']
    assert not any(e.action in {'move','scroll'} for e in events)

def test_switching_scroll_to_three_finger_swipe_keeps_initial_motion(tmp_path):
    c=modern_controller(tmp_path);c.navigation_pose='three';c.smooth_scroll=True
    for t in np.arange(.42,.8,.02):c.process(feature(float(t),middle='open'))
    assert c.state=='scroll'
    events=[]
    for t in np.arange(.8,1.12,.02):
        f=three(t);f.image_points[:,0]+=.4*(t-.8);events+=c.process(f)
    assert [e.gesture for e in events if e.action=='hotkey']==['swipe_left']
    assert not any(e.action in {'move','scroll'} for e in events)
