"""Standalone desktop entry point: no shell wrapper or external interpreter."""
import argparse
import json
import os
from pathlib import Path
import platform
import tempfile

from gesture_system.app_paths import bootstrap_user_root, resource_root, application_data_root


def diagnostics():
    """Camera-free native dependency/model probe; never writes to user home."""
    cache = tempfile.TemporaryDirectory(prefix='gesture-control-matplotlib-')
    os.environ['MPLCONFIGDIR'] = cache.name
    import cv2
    import numpy
    import torch
    import mediapipe
    import Vision, Quartz, objc, Foundation
    from PySide6 import QtCore
    from gesture_system.temporal import TemporalPredictor
    from src.realtime_inference import load_checkpoint
    with tempfile.TemporaryDirectory(prefix='gesture-control-probe-') as folder:
        root = bootstrap_user_root(destination=Path(folder))
        options = mediapipe.tasks.vision.HandLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(model_asset_path=str(root / 'models' / 'hand_landmarker.task')),
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO)
        with mediapipe.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            blank = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=numpy.zeros((160, 160, 3), dtype=numpy.uint8))
            result = landmarker.detect_for_video(blank, 0)
            assert not result.hand_landmarks
        model, _, _, _ = load_checkpoint(root / 'models' / 'mlp_hagrid_6classes.pth')
        with torch.inference_mode():
            assert torch.isfinite(model(torch.zeros((1, 42)))).all()
        for method in ('joint', 'two_stage', 'window'):
            predictor = TemporalPredictor(root / 'models' / f'streaming_{method}.pth')
            _, confidence, active = predictor.update(numpy.zeros(48), True)
            assert numpy.isfinite([confidence, active]).all()
        print(json.dumps({'status':'ok', 'architecture':platform.machine(),
            'macos':platform.mac_ver()[0], 'qt':QtCore.qVersion(), 'numpy':numpy.__version__,
            'opencv':cv2.__version__, 'torch':torch.__version__, 'mediapipe':mediapipe.__version__,
            'vision':bool(Vision.VNDetectHumanHandPoseRequest), 'resource_root':str(resource_root()),
            'user_data_root':str(application_data_root()), 'camera_opened':False,
            'os_events_posted':False}, ensure_ascii=False))
    cache.cleanup()
    return 0


def main():
    parser = argparse.ArgumentParser(description='Жестовое управление')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--tracker', choices=['apple_vision', 'mediapipe'])
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    if args.diagnostics:
        return diagnostics()
    root = bootstrap_user_root()
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'data' / 'cache' / 'matplotlib'))
    from gesture_system.gui import run, read_preferences
    saved_backend = read_preferences(root).get('tracker_backend')
    backend = args.tracker or saved_backend or ('apple_vision' if platform.system() == 'Darwin' else 'mediapipe')
    return run(root, args.video, args.screenshot, tracker_backend=backend)


if __name__ == '__main__':
    raise SystemExit(main())
