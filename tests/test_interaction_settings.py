import pytest
import numpy as np


def test_settings_roundtrip_and_invalid_values_do_not_override_safe_defaults():
    from gesture_system.interaction_settings import InteractionSettings
    value=InteractionSettings(pointer_sensitivity=1.7,scroll_gain=3.)
    assert InteractionSettings.from_dict(value.to_dict())==value
    invalid={'pointer_sensitivity':float('nan'),'scroll_gain':100,'allow_drag':'yes','pose_confirm_ms':0}
    assert InteractionSettings.from_dict(invalid)==InteractionSettings()
    with pytest.raises(ValueError):InteractionSettings(pointer_sensitivity=float('inf'))


def test_relative_sensitivity_scales_hand_motion_and_drag_without_jump():
    from gesture_system.workspace_pointer import WorkspaceMapper
    from test_image_geometry import feature
    m=WorkspaceMapper({'source_kind':'camera','frame_size':[640,480]},relative=True)
    m.sensitivity=2.;m.begin_pointer(feature(0),[.5,.5])
    f=feature(1);f.image_points[:,0]+=.025
    assert np.allclose(m.map_feature(f),[.4,.5])
    assert np.allclose(m.translate_anchor([.5,.5],[.5,.5,0],[.525,.5,0]),[.4,.5])


def test_settings_apply_preserves_current_defaults_except_documented_faster_acquisition(tmp_path):
    from gesture_system.interaction_settings import InteractionSettings
    from test_stable_control import controller
    c=controller(tmp_path);InteractionSettings().apply(c)
    assert c.filter.min_cutoff==pytest.approx(1.8)
    assert c.pinch_press==.24 and c.pinch_release==.34 and c.drag_hold==.45
    assert c.pose_confirm==.06 and c.pointer_acquire==.1
