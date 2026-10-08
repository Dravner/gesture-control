import numpy as np
import pytest
from gesture_system.types import FrameFeatures
from gesture_system.profiles import GestureLibrary
from gesture_system.stable_control import StableController

class Mapper:
    def map_feature(self,f):return np.array(f.pointer)
    def camera_wrist(self,f):return np.array([*f.wrist,0.])

def observation(t,x=.5,y=.5,label='palm'):
    return FrameFeatures(t,True,np.zeros(42),(x,y),(.5,.5),1.,.2,'palm',.99,geometry_label=label,geometry_confidence=.99)

def armed(tmp_path):
    c=StableController(GestureLibrary(tmp_path),Mapper())
    for t in np.arange(0,.24,.02):assert not c.process(observation(float(t)))
    assert c.state=='swipe'
    return c

def swipes(events):return [e for e in events if e.gesture.startswith('swipe')]

def test_armed_swipe_survives_uncertain_finger_pose(tmp_path):
    c=armed(tmp_path);events=[]
    for t,x in zip(np.arange(.24,.6,.02),np.linspace(.5,.76,18)):
        events+=c.process(observation(float(t),float(x),label='uncertain'))
    assert len(swipes(events))==1
    assert swipes(events)[0].gesture=='swipe_left'
    assert not any(e.action in {'move','click','scroll'} for e in events)

def test_brief_missing_frames_bridge_only_an_armed_swipe(tmp_path):
    c=armed(tmp_path);events=c.process(observation(.26,.54,label='uncertain'))
    events+=c.process(FrameFeatures.absent(.30));events+=c.process(FrameFeatures.absent(.34))
    for t,x in [(.38,.67),(.42,.74)]:events+=c.process(observation(t,x,label='uncertain'))
    assert len(swipes(events))==1
    # Loss immediately after firing must not turn the same movement into a new episode.
    assert not c.process(FrameFeatures.absent(.46))
    assert c.state=='swipe_latched'
    assert not swipes(c.process(observation(.50,.8)))

def test_long_gap_and_corrupt_present_frame_cancel_swipe(tmp_path):
    c=armed(tmp_path)
    c.process(FrameFeatures.absent(.40))
    assert c.state=='recovery'
    assert not swipes(c.process(observation(.44,.8,label='uncertain')))
    c=armed(tmp_path/'second');bad=observation(.26);bad.pose[0]=np.nan
    assert not c.process(bad)
    assert c.state=='recovery'

def test_slow_rest_drift_never_accumulates_a_swipe(tmp_path):
    c=armed(tmp_path);events=[]
    for t in np.arange(.24,5,.02):events+=c.process(observation(float(t),.5+float(t)*.06))
    assert not swipes(events)
    assert c.state=='swipe'

def test_small_stationary_jitter_keeps_swipe_ready(tmp_path):
    c=armed(tmp_path)
    for i,t in enumerate(np.arange(.24,2,.02)):
        assert not c.process(observation(float(t),.5+(.004 if i%2 else -.004)))
    assert c.state=='swipe'

@pytest.mark.parametrize('fps',[15,30,60])
def test_continuous_open_palm_motion_is_frame_rate_independent(tmp_path,fps):
    c=armed(tmp_path);events=[]
    for t in np.arange(.24,.8,1/fps):
        events+=c.process(observation(float(t),.5+.3*np.clip((t-.24)/.45,0,1)))
    assert len(swipes(events))==1

def test_unarmed_ambiguous_motion_and_teleport_never_swipe(tmp_path):
    c=StableController(GestureLibrary(tmp_path),Mapper());events=[]
    for t,x in zip(np.arange(0,.4,.02),np.linspace(.5,.85,20)):events+=c.process(observation(float(t),float(x),label='uncertain'))
    assert not swipes(events)
    c=armed(tmp_path/'jump')
    assert not swipes(c.process(observation(.24,.95,label='uncertain')))
    assert c.state=='recovery'

def test_explicit_scroll_or_pinch_cancels_armed_swipe(tmp_path):
    for label in ['victory','pinch']:
        c=armed(tmp_path/label);assert not c.process(observation(.26,.54,label=label))
        assert c.state=='recovery'
        assert not swipes(c.process(observation(.4,.8,label='uncertain')))
