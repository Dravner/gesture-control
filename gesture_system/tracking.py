"""Backend choice is explicit; no silent model fallback."""
import sys
import importlib.util

def apple_vision_available():
    return sys.platform=='darwin' and importlib.util.find_spec('Vision') is not None

def make_tracker(root,backend='mediapipe'):
    if backend=='mediapipe':
        from .vision import HandTracker
        return HandTracker(root)
    if backend=='apple_vision':
        from .apple_vision import VisionHandTracker
        return VisionHandTracker(root)
    raise ValueError('Unknown hand tracking backend')
