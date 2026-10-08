"""Bounded latest-frame camera pipeline. No engine or OS-input work runs here."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
import threading
import time
import numpy as np


def _immutable_array(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class FeatureSnapshot:
    timestamp: float
    present: bool
    pose: np.ndarray
    wrist: tuple
    pointer: tuple
    pinch: float
    scale: float
    label: str
    confidence: float
    world_points: np.ndarray | None = None
    image_points: np.ndarray | None = None
    frame_size: tuple | None = None
    geometry_label: str | None = None
    geometry_confidence: float = 0.
    geometry_pinch: float | None = None
    joint_confidence: np.ndarray | None = None
    handedness: str | None = None

    @classmethod
    def copy(cls, feature):
        return cls(float(feature.timestamp), bool(feature.present), _immutable_array(feature.pose),
                   tuple(feature.wrist), tuple(feature.pointer), float(feature.pinch),
                   float(feature.scale), str(feature.label), float(feature.confidence),
                   _immutable_array(feature.world_points) if getattr(feature, 'world_points', None) is not None else None,
                   _immutable_array(feature.image_points) if getattr(feature, 'image_points', None) is not None else None,
                   tuple(feature.frame_size) if getattr(feature, 'frame_size', None) is not None else None,
                   getattr(feature, 'geometry_label', None), float(getattr(feature, 'geometry_confidence', 0.)),
                   getattr(feature, 'geometry_pinch', None),
                   _immutable_array(feature.joint_confidence) if getattr(feature,'joint_confidence',None) is not None else None,
                   getattr(feature,'handedness',None))

    def vector(self):
        return np.concatenate((self.pose, self.wrist, self.pointer, [self.pinch,self.scale])).astype(np.float32)


@dataclass(frozen=True)
class FramePacket:
    generation: int
    sequence: int
    capture_time: float
    result_time: float
    image: np.ndarray
    feature: FeatureSnapshot
    telemetry: object


@dataclass
class _Run:
    generation: int
    stop: threading.Event = field(default_factory=threading.Event)
    condition: threading.Condition = field(default_factory=threading.Condition)
    frame: object = None
    packet: object = None
    error: str | None = None
    captured: int = 0
    processed: int = 0
    dropped_capture: int = 0
    dropped_result: int = 0
    capture_read_retries: int = 0
    missing_hands: int = 0
    last_absent_time: float | None = None
    first_capture_time: float | None = None
    latest_capture_time: float = 0.
    threads: list = field(default_factory=list)


class LiveRuntime:
    def __init__(self, root, camera_index=0, capture_factory=None, tracker_factory=None,
                 resolution=None, clock=time.monotonic):
        self.root = Path(root)
        self.camera_index = camera_index
        self.capture_factory = capture_factory
        self.tracker_factory = tracker_factory
        self.resolution = resolution
        self.clock = clock
        self._state = None
        self._generation = 0

    @property
    def error(self):
        state = self._state
        if state is None:
            return None
        with state.condition:
            return state.error

    def start(self):
        if self._state and any(thread.is_alive() for thread in self._state.threads):
            raise RuntimeError('Предыдущие потоки камеры ещё завершаются; повторите запуск.')
        self._generation += 1
        state = _Run(self._generation)
        self._state = state
        state.threads = [threading.Thread(target=self._capture, args=(state,), daemon=True, name='gesture-camera'),
                         threading.Thread(target=self._inference, args=(state,), daemon=True, name='gesture-inference')]
        for thread in state.threads:
            thread.start()
        return state.generation

    def _fail(self, state, origin, exception):
        with state.condition:
            if state.error is None:
                state.error = f'{origin}: {exception}'
            state.stop.set()
            state.frame = state.packet = None
            state.condition.notify_all()

    def _capture(self, state):
        capture = None
        failed_reads=0
        try:
            if self.capture_factory:
                capture = self.capture_factory(self.camera_index)
            else:
                import cv2
                capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                raise RuntimeError('Не удалось открыть камеру. Проверьте доступ и выбранный источник.')
            if self.resolution and hasattr(capture, 'set'):
                import cv2
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not state.stop.is_set():
                ok, frame = capture.read()
                stamp = self.clock()
                if state.stop.is_set():
                    break
                if not ok or frame is None:
                    failed_reads+=1
                    with state.condition:state.capture_read_retries+=1
                    if failed_reads>=3:raise RuntimeError('Камера перестала возвращать кадры после трёх попыток.')
                    state.stop.wait(.05)
                    continue
                failed_reads=0
                image = np.array(frame, copy=True)
                with state.condition:
                    state.captured += 1
                    if state.first_capture_time is None:
                        state.first_capture_time = stamp
                    state.latest_capture_time = stamp
                    if state.frame is not None:
                        state.dropped_capture += 1
                    state.frame = (state.captured, stamp, image)
                    state.condition.notify_all()
        except Exception as exc:
            self._fail(state, 'capture', exc)
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    self._fail(state, 'capture release', exc)

    def _inference(self, state):
        tracker = None
        try:
            if state.stop.is_set():
                return
            if self.tracker_factory:
                tracker = self.tracker_factory()
            else:
                from .vision import HandTracker
                tracker = HandTracker(self.root)
            while not state.stop.is_set():
                with state.condition:
                    state.condition.wait_for(lambda: state.frame is not None or state.stop.is_set())
                    if state.stop.is_set():
                        break
                    sequence, stamp, frame = state.frame
                    state.frame = None
                start = self.clock()
                feature = FeatureSnapshot.copy(tracker.process(frame, stamp))
                annotated = tracker.annotate(frame, feature)
                image = _immutable_array(annotated)
                finished = self.clock()
                with state.condition:
                    if state.stop.is_set():
                        break
                    state.processed += 1
                    if not feature.present:
                        state.missing_hands += 1
                        state.last_absent_time = stamp
                    if state.packet is not None:
                        state.dropped_result += 1
                    metrics = MappingProxyType({
                        'captured_frames': state.captured, 'processed_frames': state.processed,
                        'width': int(frame.shape[1]), 'height': int(frame.shape[0]),
                        'capture_fps': (state.captured-1)/max(state.latest_capture_time-state.first_capture_time, 1e-6),
                        'dropped_capture_frames': state.dropped_capture,
                        'capture_read_retries':state.capture_read_retries,
                        'dropped_completed_packets': state.dropped_result,
                        'missing_hand_frames': state.missing_hands,
                        'last_absent_time': state.last_absent_time,
                        'compute_ms': (finished-start)*1000,
                        'age_ms': (finished-stamp)*1000,
                        'inference_ms': (finished-start)*1000,
                        'capture_to_result_ms': (finished-stamp)*1000,
                    })
                    state.packet = FramePacket(state.generation,sequence,stamp,finished,image,feature,metrics)
        except Exception as exc:
            self._fail(state, 'tracker', exc)
        finally:
            if tracker is not None:
                try:
                    tracker.close()
                except Exception as exc:
                    self._fail(state, 'tracker close', exc)

    def take_latest(self):
        state = self._state
        if state is None:
            return None
        with state.condition:
            if state.stop.is_set():
                return None
            packet = state.packet
            state.packet = None
            return packet

    def stop(self, wait=False, timeout=2.):
        state = self._state
        if state is None:
            return True
        with state.condition:
            state.stop.set()
            state.frame = state.packet = None
            state.condition.notify_all()
        return self.wait(timeout) if wait else not any(thread.is_alive() for thread in state.threads)

    def wait(self, timeout=2.):
        state = self._state
        if state is None:
            return True
        deadline = time.monotonic()+timeout
        for thread in state.threads:
            thread.join(max(0.,deadline-time.monotonic()))
        return not any(thread.is_alive() for thread in state.threads)
