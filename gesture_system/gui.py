"""Native desktop interface. File replay and template recording never inject input."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import os
import platform
import uuid
from datetime import datetime, timezone
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QCheckBox, QLineEdit, QListWidget, QPlainTextEdit, QProgressBar,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox, QFileDialog, QMessageBox,
    QSplitter, QDoubleSpinBox, QScrollArea, QSpinBox)


def valid_world(feature):
    value = getattr(feature, 'world_points', None)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.size != 63 or not np.isfinite(array).all():
        return None
    return array.reshape(21,3).copy()


@dataclass
class RecordingSession:
    kind: str
    started: float
    countdown: float = 3.0
    samples: list = field(default_factory=list)
    world_samples: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    seen: int = 0
    invalid: int = 0
    active_start: float | None = None
    active_end: float | None = None

    @property
    def duration(self):
        return 2.0 if self.kind == 'static' else 3.0

    def feed(self, feature, now):
        elapsed = now - self.started - self.countdown
        if elapsed < 0:
            return False
        self.seen += 1
        if self.active_start is None:
            self.active_start = float(now)
        self.active_end = float(now)
        if feature.present:
            vector = np.asarray(feature.vector(), dtype=float)
            if vector.shape == (48,) and np.isfinite(vector).all():
                self.samples.append(vector.copy())
                self.world_samples.append(valid_world(feature))
                self.timestamps.append(float(now))
            else:
                self.invalid += 1
        return elapsed >= self.duration

    def world_array(self):
        if self.world_samples and len(self.world_samples) == len(self.samples) and all(p is not None for p in self.world_samples):
            return np.asarray(self.world_samples)
        return None

    def quality(self):
        world = sum(p is not None for p in self.world_samples)
        motion = 0.
        if len(self.samples)>1:
            samples = np.asarray(self.samples)
            path = np.linalg.norm(np.ptp(samples[:,42:44], axis=0))
            pose = np.sqrt(np.mean(np.var(samples[:,:42], axis=0)))
            motion = float(max(path, pose))
        return {'frames': len(self.samples), 'coverage': len(self.samples)/max(self.seen,1),
                'world_coverage': world/max(len(self.samples),1), 'invalid_frames': self.invalid, 'motion': motion}


@dataclass
class GuidedProtocol:
    mode: str
    total: int
    target: str | None
    name: str
    kind: str
    action: str
    keys: list | None
    index: int = 0
    results: list = field(default_factory=list)
    events: list = field(default_factory=list)
    rejections: list = field(default_factory=list)
    distances: list = field(default_factory=list)
    restore_mode: str = 'combined'
    last_time: float | None = None
    exercise_started: bool = False

    @property
    def grace(self):
        return .8 if self.kind=='dynamic' else .3


class SessionRecorder:
    """Persist numerical observations and complete event traces, never image data."""
    def __init__(self, directory, source, method, profiles, settings=None):
        stamp = datetime.now(timezone.utc)
        self.path = Path(directory) / (stamp.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True, exist_ok=False)
        self.frames = []
        self.events = []
        self.metadata = {
            'version': 1, 'started_utc': stamp.isoformat(), 'source': source,
            'method': method, 'settings': settings or {}, 'profiles': profiles,
            'clock': 'monotonic seconds', 'stores_images': False,
            'hardware': {'machine': platform.machine(), 'processor': platform.processor(),
                         'cpu_count': os.cpu_count(), 'platform': platform.platform()},
        }

    def add_frame(self, feature, prediction, confidence, phase, recording, method, telemetry=None, task=None):
        if telemetry and 'width' in telemetry and 'height' in telemetry:
            width, height = int(telemetry['width']), int(telemetry['height'])
            geometry = {'width': width, 'height': height, 'aspect_ratio': width / max(height, 1)}
            formats = self.metadata['source'].setdefault('capture_formats', [])
            if geometry not in formats:
                formats.append(geometry)
            self.metadata['source']['capture_fps'] = float(telemetry.get('capture_fps', 0.))
        world = valid_world(feature)
        image=getattr(feature,'image_points',None)
        image=np.asarray(image,dtype=np.float32).reshape(42).copy() if image is not None and np.asarray(image).size==42 else None
        self.frames.append({
            'world': world, 'image':image,
            'handedness':getattr(feature,'handedness',None) or '',
            'joint_confidence':None if getattr(feature,'joint_confidence',None) is None else np.asarray(feature.joint_confidence,dtype=np.float32).copy(),
            'geometry_label':getattr(feature,'geometry_label',None) or '',
            'geometry_confidence':float(getattr(feature,'geometry_confidence',0.)),
            'frame_size':tuple(getattr(feature,'frame_size',None) or (0,0)),
            'task': dict(task or {}),
            'vector': np.asarray(feature.vector(), dtype=np.float32).copy(),
            'timestamp': float(feature.timestamp), 'present': bool(feature.present),
            'label': str(feature.label), 'raw_confidence': float(feature.confidence),
            'prediction': str(prediction), 'confidence': float(confidence),
            'phase': str(phase), 'recording': bool(recording), 'method': str(method),
            'telemetry': dict(telemetry or {}),
        })

    def add_event(self, event, executed=False, error=None):
        self.events.append({'timestamp': float(event.timestamp), 'gesture': event.gesture,
                            'action': event.action, 'payload': dict(event.payload),
                            'executed': bool(executed), 'error': error})

    def finish(self, reason):
        rows = self.frames
        arrays = {
            'handedness':np.asarray([row['handedness'] for row in rows],dtype=str),
            'joint_confidence':np.stack([row['joint_confidence'] if row['joint_confidence'] is not None else np.full(21,np.nan,np.float32) for row in rows]) if rows else np.empty((0,21),np.float32),
            'world_points': np.stack([row['world'].reshape(63) if row['world'] is not None else np.full(63,np.nan,np.float32) for row in rows]) if rows else np.empty((0,63),np.float32),
            'world_present': np.asarray([row['world'] is not None for row in rows], dtype=bool),
            'image_points': np.stack([row['image'] if row['image'] is not None else np.full(42,np.nan,np.float32) for row in rows]) if rows else np.empty((0,42),np.float32),
            'image_present': np.asarray([row['image'] is not None for row in rows],dtype=bool),
            'geometry_labels':np.asarray([row['geometry_label'] for row in rows],dtype=str),
            'geometry_confidence':np.asarray([row['geometry_confidence'] for row in rows],dtype=np.float32),
            'frame_size':np.asarray([row['frame_size'] for row in rows],dtype=np.int32).reshape(-1,2),
            'task_mode': np.asarray([row['task'].get('mode', 'idle') for row in rows], dtype=str),
            'task_expected': np.asarray([row['task'].get('expected_id') or '' for row in rows], dtype=str),
            'task_index': np.asarray([row['task'].get('index', 0) for row in rows], dtype=np.int16),
            'task_pointer_mode':np.asarray([row['task'].get('pointer_mode') or '' for row in rows],dtype=str),
            'task_controller_geometry_source':np.asarray([row['task'].get('controller_geometry_source') or '' for row in rows],dtype=str),
            'task_error':np.asarray([row['task'].get('error') or '' for row in rows],dtype=str),
            'vectors': np.stack([row['vector'] for row in rows]) if rows else np.empty((0,48),np.float32),
            'timestamps': np.asarray([row['timestamp'] for row in rows], dtype=np.float64),
            'present': np.asarray([row['present'] for row in rows], dtype=bool),
            'labels': np.asarray([row['label'] for row in rows], dtype=str),
            'raw_confidence': np.asarray([row['raw_confidence'] for row in rows], dtype=np.float32),
            'predictions': np.asarray([row['prediction'] for row in rows], dtype=str),
            'confidence': np.asarray([row['confidence'] for row in rows], dtype=np.float32),
            'phases': np.asarray([row['phase'] for row in rows], dtype=str),
            'recording': np.asarray([row['recording'] for row in rows], dtype=bool),
            'methods': np.asarray([row['method'] for row in rows], dtype=str),
            'capture_width': np.asarray([row['telemetry'].get('width', 0) for row in rows], dtype=np.int32),
            'capture_height': np.asarray([row['telemetry'].get('height', 0) for row in rows], dtype=np.int32),
            'capture_fps': np.asarray([row['telemetry'].get('capture_fps', float('nan')) for row in rows], dtype=np.float32),
            'missing_hand_frames': np.asarray([row['telemetry'].get('missing_hand_frames', 0) for row in rows], dtype=np.int64),
            'last_absent_time': np.asarray([row['telemetry'].get('last_absent_time', float('nan')) for row in rows], dtype=np.float64),
            'replayed_absence': np.asarray([row['telemetry'].get('replayed_absence', False) for row in rows], dtype=bool),
            'inference_ms': np.asarray([row['telemetry'].get('inference_ms', float('nan')) for row in rows], dtype=np.float32),
            'capture_to_result_ms': np.asarray([row['telemetry'].get('capture_to_result_ms', float('nan')) for row in rows], dtype=np.float32),
            'capture_to_display_ms': np.asarray([row['telemetry'].get('capture_to_display_ms', float('nan')) for row in rows], dtype=np.float32),
            'dropped_capture_frames': np.asarray([row['telemetry'].get('dropped_capture_frames', 0) for row in rows], dtype=np.int64),
            'dropped_completed_packets': np.asarray([row['telemetry'].get('dropped_completed_packets', 0) for row in rows], dtype=np.int64),
        }
        temporary = self.path / 'frames.tmp'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(self.path / 'frames.npz')
        document = dict(self.metadata, ended_utc=datetime.now(timezone.utc).isoformat(),
                        end_reason=reason, frame_count=len(rows), events=self.events)
        temporary = self.path / 'session.tmp'
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.path / 'session.json')
        return self.path


def selected_method(root):
    try:
        value = json.loads((Path(root) / 'models' / 'selection.json').read_text())['method']
        return value if value in {'window', 'two_stage', 'joint'} else 'two_stage'
    except (OSError, ValueError, KeyError, TypeError):
        return 'two_stage'


def read_preferences(root):
    try:
        settings = json.loads((Path(root) / 'data' / 'ui_settings.json').read_text())
        valid = {}
        for name, low, high in [('confidence', .1, 1.), ('hold_seconds', .05, 2.)]:
            value = float(settings.get(name, float('nan')))
            if np.isfinite(value) and low <= value <= high:
                valid[name] = value
        if settings.get('method') in {'window', 'two_stage', 'joint'}:
            valid['method'] = settings['method']
        if settings.get('profile') in {'stable','advanced'}:valid['profile']=settings['profile']
        if settings.get('pointer_mode') in {'workspace','ray'}:valid['pointer_mode']=settings['pointer_mode']
        if settings.get('tracker_backend') in {'mediapipe','apple_vision'}:valid['tracker_backend']=settings['tracker_backend']
        if isinstance(settings.get('fov_degrees'),(float,int)) and 20<=settings['fov_degrees']<=120:valid['fov_degrees']=float(settings['fov_degrees'])
        if settings.get('control_mode') in {'combined', 'personal', 'pointer'}:
            valid['control_mode'] = settings['control_mode']
        if isinstance(settings.get('experimental_neural'), bool):
            valid['experimental_neural'] = settings['experimental_neural']
        valid['interaction']=settings.get('interaction',{})
        return valid
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


class MainWindow(QMainWindow):
    packet_ready=Signal()
    def __init__(self, root, library, engine, tracker=None, actions=None, capture_factory=None, tracker_factory=None, runtime_factory=None, screen_provider=None,tracker_backend=None,camera_activity=None):
        super().__init__()
        self.root, self.library, self.engine = Path(root), library, engine
        from .camera_activity import CameraActivity
        self.camera_activity=CameraActivity() if camera_activity is None else camera_activity
        self._settings_ready = False
        self.screen_provider = screen_provider
        self._display = None
        self._actions_display = None
        self.pointing_dialog = None
        self.workspace_dialog = None
        self.workspace_mapper = None
        self.receipt_dialog = None
        self._last_feature = None
        self._protocol_restore_stable = None
        self.pointing_calibration = None
        self._calibration_error = None
        from .pointing import PointingCalibration
        calibration_path = Path(root)/'data'/'pointing_calibration.json'
        if calibration_path.exists():
            try:self.pointing_calibration=PointingCalibration.load(calibration_path)
            except (ValueError,OSError,KeyError,TypeError) as exc:self._calibration_error=str(exc)
        self._preferences = read_preferences(self.root)
        from .interaction_settings import InteractionSettings
        self.interaction_settings=InteractionSettings.from_dict(self._preferences.get('interaction'))
        self.settings_dialog=None
        if tracker_backend is not None:
            if tracker_backend not in {'mediapipe','apple_vision'}:raise ValueError('Unknown hand tracking backend')
            if tracker_backend=='apple_vision':
                from .tracking import apple_vision_available
                if not apple_vision_available():raise RuntimeError('Apple Vision недоступен: нужны macOS и pyobjc-framework-Vision.')
            self._preferences['tracker_backend']=tracker_backend
        self._configured_backend=self._preferences.get('tracker_backend','mediapipe')
        workspace_path=Path(root)/'data'/'workspace_config.json'
        if workspace_path.exists():
            try:
                from .workspace_pointer import WorkspaceMapper
                self.workspace_mapper=WorkspaceMapper.load(workspace_path)
            except (ValueError,OSError,KeyError,TypeError):pass
        self.engine.conf_threshold = self._preferences.get('confidence', self.engine.conf_threshold)
        self.engine.hold_seconds = self._preferences.get('hold_seconds', self.engine.hold_seconds)
        self.engine.control_mode = self._preferences.get('control_mode', 'combined')
        self.engine.experimental_neural = self._preferences.get('experimental_neural', False)
        self.tracker, self.actions = tracker, actions
        self._provided_tracker = tracker is not None
        self.capture_factory = capture_factory
        self.tracker_factory = tracker_factory
        self.runtime_factory = runtime_factory
        self.live_runtime = None
        self._live_generation = None
        self._live_uses_tracker = False
        self._minimum_capture_time = 0.
        self._last_packet_wall = 0.
        self._last_missing_hand_count = 0
        self._stream_stalled=False
        self.capture = None
        self.session = None
        self._motion_log_t = {}
        self.video_path = None
        self.recording = None
        self.protocol = None
        self._record_name = ''
        self._record_target = None
        self._last_frame_time = None
        self._running = False
        self._entries = []
        self.setWindowTitle('Жестовое управление · Stream Gesture')
        self.resize(1240, 1040)
        self.setMinimumSize(1050, 740)
        self.timer = QTimer(self)
        self.timer.setInterval(30 if self.runtime_factory is not None else 100)
        self.packet_ready.connect(self.tick,Qt.ConnectionType.QueuedConnection)
        self.timer.timeout.connect(self.tick)
        self._build_ui()
        self.refresh_library()
        method = self._preferences.get('method', selected_method(self.root))
        index = self.method.findData(method)
        self.method.blockSignals(True)
        self.method.setCurrentIndex(index)
        self.method.blockSignals(False)
        if self.engine.method != method:
            self.reload_engine()
        self.apply_settings()
        self._settings_ready = True
        if self._calibration_error:self.note('Сохранённая калибровка недоступна: '+self._calibration_error)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 20)
        heading = QLabel('Жестовое управление')
        heading.setObjectName('heading')
        subtitle = QLabel('Потоковое распознавание · камера и видеозаписи · персональные команды')
        subtitle.setObjectName('muted')
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        source_bar = QHBoxLayout()
        self.source = QComboBox()
        self.source.addItems(['Камера', 'Видеофайл'])
        self.source.currentIndexChanged.connect(self.source_changed)
        self.choose_video = QPushButton('Открыть видео…')
        self.choose_video.clicked.connect(self.browse_video)
        self.source_name = QLabel('Камера 0')
        self.source_name.setObjectName('muted')
        self.method = QComboBox()
        for title, key in [('Детектор → классификатор', 'two_stage'), ('Скользящее окно', 'window'), ('Совместная модель', 'joint')]:
            self.method.addItem(title, key)
        self.method.currentIndexChanged.connect(self.change_method)
        self.method.setToolTip('Этот метод управляет командами только при включённом флажке экспериментальной нейросети. Свои жесты используют персональные примеры.')
        self.start_button = QPushButton('▶  Запустить')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.start_stream)
        self.pause_button = QPushButton('Ⅱ  Пауза')
        self.pause_button.clicked.connect(self.pause_stream)
        stop_button = QPushButton('■  Стоп')
        stop_button.clicked.connect(self.stop_stream)
        settings_button=QPushButton('Настройки…');settings_button.clicked.connect(self.open_settings)
        for item in [settings_button,self.source, self.choose_video, self.source_name, self.method, self.start_button, self.pause_button, stop_button]:
            source_bar.addWidget(item)
        layout.addLayout(source_bar)
        pointing_bar = QHBoxLayout()
        self.profile_selector = QComboBox()
        self.profile_selector.addItem('Наведение на экран · пять жестов','stable')
        self.profile_selector.addItem('Свои жесты и экспериментальные методы','advanced')
        self.profile_selector.setCurrentIndex(self.profile_selector.findData(self._preferences.get('profile','stable')))
        self.profile_selector.currentIndexChanged.connect(self.apply_settings)
        self.calibrate_button=QPushButton('Калибровка · 9 целей');self.calibrate_button.clicked.connect(self.open_pointing_calibration)
        self.target_test_button=QPushButton('Проверить наведение · 5 целей');self.target_test_button.clicked.connect(self.open_target_validation)
        self.gesture_test_button=QPushButton('Проверка пяти жестов');self.gesture_test_button.clicked.connect(self.open_gesture_receipts)
        self.fov=QDoubleSpinBox();self.fov.setRange(20,120);self.fov.setValue(self._preferences.get('fov_degrees',60.));self.fov.setSuffix('°')
        self.fov.setToolTip('Предполагаемое горизонтальное поле зрения. Это не автоматически измеренная калибровка камеры.')
        self.fov.valueChanged.connect(self.apply_settings)
        self.fov_label=QLabel('Угол обзора камеры (приближённый)')
        for item in [self.profile_selector,self.calibrate_button,self.target_test_button,self.gesture_test_button,self.fov_label,self.fov]:pointing_bar.addWidget(item)
        layout.addLayout(pointing_bar)
        mode_bar=QHBoxLayout();mode_bar.addWidget(QLabel('Способ наведения'))
        self.pointer_mode=QComboBox();self.pointer_mode.addItem('Курсор движением кисти · 2D','workspace');self.pointer_mode.addItem('Экспериментальный луч пальца · 3D','ray')
        self.pointer_mode.setCurrentIndex(self.pointer_mode.findData(self._preferences.get('pointer_mode','ray')))
        self.pointer_mode.currentIndexChanged.connect(self.apply_settings);mode_bar.addWidget(self.pointer_mode,1);layout.addLayout(mode_bar)
        backend_bar=QHBoxLayout();backend_bar.addWidget(QLabel('Трекинг кисти'))
        self.tracking_backend=QComboBox();self.tracking_backend.addItem('MediaPipe · 2D / экспериментальный 3D','mediapipe');self.tracking_backend.addItem('Apple Vision · 2D · сравнение','apple_vision')
        from .tracking import apple_vision_available
        available=apple_vision_available();self.tracking_backend.model().item(1).setEnabled(available)
        if self._configured_backend=='apple_vision' and not available:self._configured_backend='mediapipe'
        self.tracking_backend.setCurrentIndex(self.tracking_backend.findData(self._configured_backend))
        self.tracking_backend.currentIndexChanged.connect(self.backend_changed);backend_bar.addWidget(self.tracking_backend,1);layout.addLayout(backend_bar)
        self.pointing_status=QLabel('Калибровка отсутствует. Направляйте прямой указательный палец К ЭКРАНУ, как лазер; остальные три согнуты.')
        self.pointing_status.setWordWrap(True);self.pointing_status.setObjectName('muted');layout.addWidget(self.pointing_status)
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 10, 0)
        self.preview = QLabel('Камера выключена\n\nВыберите источник и нажмите «Запустить»\n\nУправление системой включается отдельно')
        self.preview.setObjectName('preview')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(550, 345)
        left_layout.addWidget(self.preview, 1)
        status_row = QHBoxLayout()
        self.gesture_label = QLabel('Жест: —')
        self.phase_label = QLabel('Фаза: ожидание')
        self.fps_label = QLabel('FPS: —')
        for item in [self.gesture_label, self.phase_label, self.fps_label]:
            status_row.addWidget(item)
        left_layout.addLayout(status_row)
        self.runtime_label = QLabel('Камера ожидает запуска')
        self.runtime_label.setObjectName('muted')
        self.runtime_label.setWordWrap(True)
        left_layout.addWidget(self.runtime_label)
        self.diagnostic_label = QLabel('Свои жесты: ожидание наблюдений; независимая проверка ещё не выполнена')
        self.diagnostic_label.setObjectName('muted')
        self.diagnostic_label.setWordWrap(True)
        left_layout.addWidget(self.diagnostic_label)
        self.confidence = QProgressBar()
        self.confidence.setRange(0, 100)
        self.confidence.setFormat('Уверенность: %p%')
        self.confidence.setValue(0)
        left_layout.addWidget(self.confidence)
        self.enable_input = QCheckBox('Включить управление системой (OS input)')
        self.enable_input.setChecked(False)
        self.enable_input.toggled.connect(self.input_changed)
        left_layout.addWidget(self.enable_input)
        self.safety_label = QLabel('Режим наблюдения: команды отображаются в журнале')
        self.safety_label.setObjectName('muted')
        left_layout.addWidget(self.safety_label)
        right = QWidget()
        right.setMinimumWidth(360)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)
        group = QGroupBox('Библиотека жестов')
        self.library_group = group
        group.setMinimumHeight(305)
        gl = QVBoxLayout(group)
        self.gestures = QListWidget()
        self.gestures.setMinimumHeight(140)
        self.gestures.currentRowChanged.connect(self.select_gesture)
        gl.addWidget(self.gestures)
        form = QFormLayout()
        self.mapping = QComboBox()
        for title, action in [('Без команды', 'none'), ('Левый клик', 'click'), ('Правый клик', 'right_click'), ('Сочетание клавиш', 'hotkey'), ('Прокрутка', 'scroll'), ('Пауза удержания', 'pause'), ('Курсор', 'move'), ('Начать удержание', 'drag_start'), ('Отпустить удержание', 'drag_end')]:
            self.mapping.addItem(title, action)
        self.keys = QLineEdit()
        self.keys.setPlaceholderText('Например: command, shift, s')
        form.addRow('Команда', self.mapping)
        form.addRow('Клавиши', self.keys)
        self.tolerance=QDoubleSpinBox()
        self.tolerance.setDecimals(3);self.tolerance.setRange(.001,1.);self.tolerance.setSingleStep(.01)
        self.tolerance.setToolTip('Допуск для своего жеста: больше — свободнее, меньше — строже. Не меняет порог нейросети.')
        form.addRow('Допуск своего жеста',self.tolerance)
        self.orientation_sensitive = QCheckBox('Учитывать поворот кисти')
        self.orientation_sensitive.setToolTip('Включите для разных команд с похожей формой, но разным направлением кисти, например палец вверх и вниз.')
        form.addRow(self.orientation_sensitive)
        gl.addLayout(form)
        row = QHBoxLayout()
        save = QPushButton('Сохранить команду')
        save.clicked.connect(self.save_mapping)
        self.delete_button = QPushButton('Удалить')
        self.delete_button.clicked.connect(self.delete_gesture)
        row.addWidget(save)
        row.addWidget(self.delete_button)
        gl.addLayout(row)
        self.example_button = QPushButton('Добавить пример выбранного жеста')
        self.example_button.clicked.connect(self.begin_example)
        gl.addWidget(self.example_button)
        self.trial_button = QPushButton('Проверить выбранный жест · TRIAL')
        self.trial_button.clicked.connect(self.begin_trial)
        gl.addWidget(self.trial_button)
        right_layout.addWidget(group)
        record_group = QGroupBox('Записать свой жест')
        self.record_group = record_group
        form = QFormLayout(record_group)
        self.record_name = QLineEdit()
        self.record_name.setPlaceholderText('Название жеста')
        self.record_kind = QComboBox()
        self.record_kind.addItem('Статический · 2 секунды', 'static')
        self.record_kind.addItem('Динамический · 3 секунды', 'dynamic')
        self.record_button = QPushButton('●  Начать запись')
        self.record_button.clicked.connect(self.begin_recording)
        self.finish_button = QPushButton('Завершить сейчас')
        self.finish_button.setEnabled(False)
        self.finish_button.clicked.connect(self.finish_recording)
        self.record_status = QLabel('Отсчёт 3 с; затем запись только видимой руки')
        self.record_status.setWordWrap(True)
        form.addRow('Название', self.record_name)
        form.addRow('Тип', self.record_kind)
        form.addRow(self.record_button)
        self.example_count = QSpinBox()
        self.example_count.setRange(1, 5)
        self.example_count.setValue(3)
        self.guided_button = QPushButton('Обучить по отдельным примерам · TRAIN')
        self.guided_button.clicked.connect(self.begin_guided_enrollment)
        self.cancel_button = QPushButton('Отменить запись / проверку')
        self.cancel_button.clicked.connect(self.cancel_recording)
        self.cancel_button.setEnabled(False)
        form.addRow('Число примеров / повторов', self.example_count)
        form.addRow(self.guided_button)
        form.addRow(self.finish_button)
        form.addRow(self.cancel_button)
        form.addRow(self.record_status)
        right_layout.addWidget(record_group)
        guide = QLabel('Один указательный палец — лазер к экрану · Щипок — клик; остальные три пальца согнуты\nДва пальца — вертикальная прокрутка · Открытая ладонь: пауза, затем свайп влево / вправо')
        guide.setObjectName('muted')
        guide.setWordWrap(True)
        self.guide_label=guide
        left_layout.addWidget(guide)
        settings = QGroupBox('Подтверждение жеста')
        self.advanced_settings_group = settings
        settings_layout = QFormLayout(settings)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(.1, 1.)
        self.threshold.setSingleStep(.05)
        self.threshold.setValue(self.engine.conf_threshold)
        self.hold_time = QDoubleSpinBox()
        self.hold_time.setRange(.05, 2.)
        self.hold_time.setSingleStep(.05)
        self.hold_time.setSuffix(' с')
        self.hold_time.setValue(self.engine.hold_seconds)
        self.threshold.valueChanged.connect(self.apply_settings)
        self.hold_time.valueChanged.connect(self.apply_settings)
        settings_layout.addRow('Мин. уверенность', self.threshold)
        settings_layout.addRow('Время удержания', self.hold_time)
        self.control_mode = QComboBox()
        for title, value in [('Совместное управление', 'combined'), ('Только свои жесты', 'personal'), ('Курсор и встроенные жесты', 'pointer')]:
            self.control_mode.addItem(title, value)
        self.control_mode.setCurrentIndex(self.control_mode.findData(self.engine.control_mode))
        self.control_mode.currentIndexChanged.connect(self.apply_settings)
        self.experimental_neural = QCheckBox('Экспериментальная нейросеть управляет командами')
        self.experimental_neural.setChecked(self.engine.experimental_neural)
        self.experimental_neural.toggled.connect(self.apply_settings)
        settings_layout.addRow('Режим команд', self.control_mode)
        settings_layout.addRow(self.experimental_neural)
        right_layout.addWidget(settings)
        right_layout.addStretch()
        split.addWidget(left)
        scroll = QScrollArea()
        self.advanced_panel = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(right)
        scroll.setMinimumWidth(380)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)
        log_heading = QLabel('Журнал событий')
        layout.addWidget(log_heading)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        self.log.document().setMaximumBlockCount(250)
        layout.addWidget(self.log)
        self.setStyleSheet('''
            QWidget { background:#f3f5f8; color:#243248; font-size:13px; }
            QLabel#heading { font-size:28px; font-weight:700; }
            QLabel#muted { color:#66758a; }
            QLabel#preview { background:#182536; color:#c8d6e8; border-radius:12px; font-size:17px; }
            QGroupBox { border:1px solid #d6dde7; border-radius:10px; margin-top:12px; padding:12px 8px 8px; font-weight:600; }
            QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; }
            QPushButton { border:1px solid #ccd5e2; background:white; border-radius:6px; padding:8px 10px; }
            QPushButton:hover { background:#e7edf8; }
            QPushButton#primary { background:#2864d5; color:white; border-color:#2864d5; }
            QPushButton:disabled { color:#98a1af; }
            QLineEdit,QComboBox,QListWidget,QPlainTextEdit { border:1px solid #d6dde7; border-radius:6px; background:white; padding:5px; }
            QProgressBar { border:1px solid #d6dde7; border-radius:6px; background:white; text-align:center; }
            QProgressBar::chunk { background:#9bbcf3; border-radius:5px; }
        ''')
        self.note('Готово. Реальное управление выключено.')
        joint_index = self.method.findData('joint')
        if not (self.root / 'models' / 'streaming_joint.pth').exists():
            self.method.setItemText(joint_index, 'Совместная модель (не обучена)')

    def note(self, text):
        self.log.appendPlainText(f'{time.strftime("%H:%M:%S")}  {text}')

    def source_changed(self):
        self.stop_stream()
        if self.source.currentIndex() == 0:
            self.video_path = None
            self.source_name.setText('Камера 0')
        self.enable_input.setEnabled(self.source.currentIndex() == 0)

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Выберите видео', str(self.root), 'Видео (*.mp4 *.mov *.avi *.mkv);;Все файлы (*)')
        if path:
            self.set_video(path)

    def set_video(self, path):
        self.stop_stream()
        self.source.setCurrentIndex(1)
        self.video_path = str(path)
        self.source_name.setText(Path(path).name)
        self.enable_input.setChecked(False)
        self.enable_input.setEnabled(False)
        self.safety_label.setText('Видеофайл: только журнал, управление системой заблокировано')

    def input_changed(self, enabled):
        if enabled and (self.video_path or self.recording or self.protocol or self.pointing_dialog or self.workspace_dialog):
            self.enable_input.setChecked(False)
            return
        if enabled and self.engine.stable_profile:
            valid, reason = self.pointing_ready()
            if not valid:
                self.enable_input.setChecked(False);self.note(reason);return
        if enabled and (self.actions is None or not self.actions.available()):
            self.enable_input.setChecked(False)
            self.note('Системный ввод недоступен. Проверьте разрешение Accessibility для приложения запуска.')
            return
        controller=self.engine.stable_controller
        controller.position_provider=None
        if enabled and self.engine.stable_profile and self.pointer_mode.currentData()=='workspace' and self.workspace_mapper is not None and self.workspace_mapper.relative:
            provider=getattr(self.actions,'current_position',None)
            if not callable(provider):
                self.enable_input.setChecked(False);self.note('Не удалось определить текущую позицию курсора.');return
            self.engine.reset();self._minimum_capture_time=time.monotonic()
            controller.position_provider=provider
        if not enabled and self.actions:
            self.actions.release()
        if self.receipt_dialog and self.receipt_dialog.enabled.isChecked()!=bool(enabled):
            self.receipt_dialog.enabled.setChecked(bool(enabled))
        self.safety_label.setText('Управление системой включено' if enabled else 'Режим наблюдения: команды отображаются в журнале')

    def start_stream(self):
        if self._running:
            return
        try:
            if self.source.currentIndex() == 1 and not self.video_path:
                self.note('Сначала выберите видеофайл.')
                return
            if self.video_path:
                if self.capture is None:
                    import cv2
                    factory = self.capture_factory or cv2.VideoCapture
                    self.capture = factory(self.video_path)
                    if not self.capture.isOpened():
                        raise RuntimeError('Не удалось открыть видеофайл.')
                if self.tracker is None:
                    from .tracking import make_tracker
                    self.tracker = self.tracker_factory() if self.tracker_factory else make_tracker(self.root,self.tracking_backend.currentData())
            else:
                self.camera_activity.start()
                if self.live_runtime is None:
                    from .live_runtime import LiveRuntime
                    factory = self.runtime_factory or LiveRuntime
                    tracker_factory = self.tracker_factory
                    if tracker_factory is None and self._provided_tracker:
                        tracker_factory = lambda: self.tracker
                        self._live_uses_tracker = True
                    if tracker_factory is None:
                        from .tracking import make_tracker
                        backend=self.tracking_backend.currentData()
                        tracker_factory=lambda:make_tracker(self.root,backend)
                    self.live_runtime = factory(self.root, capture_factory=self.capture_factory,
                                                tracker_factory=tracker_factory,**({'notify_packet':self.packet_ready.emit} if self.runtime_factory is None else {}))
                self._live_generation = self.live_runtime.start()
                self._last_missing_hand_count = 0
            source = {'kind': 'video', 'path': self.video_path} if self.video_path else {'kind': 'camera', 'index': 0, 'pipeline': 'latest_frame_workers'}
            source['tracker_backend']=self.tracking_backend.currentData()
            self.session = SessionRecorder(self.root / 'data' / 'sessions', source=source,
                method=self.engine.method, profiles=self.library.list_gestures(),
                settings={'confidence': self.threshold.value(), 'hold_seconds': self.hold_time.value(),
                          'control_mode': self.control_mode.currentData(), 'experimental_neural': self.experimental_neural.isChecked(),
                          'interaction_profile':self.profile_selector.currentData(),
                          'pointing_context':self.pointing_calibration.context if self.pointing_calibration else None,
                          'pointer_mode':self.pointer_mode.currentData(),
                          'tracker_backend':self.tracking_backend.currentData(),
                          'camera_activity_active':self.camera_activity.active,
                          'controller_geometry_source':self.effective_geometry_source(),
                          'interaction':self.interaction_settings.to_dict(),
                          'raw_geometry_labels_source':self.tracking_backend.currentData()})
            self._motion_log_t.clear()
            self._minimum_capture_time = self._last_packet_wall = time.monotonic()
            self._stream_stalled=False
            self._running = True
            self._last_frame_time = None
            self.timer.setInterval(30 if self.video_path or self.runtime_factory is not None else 100)
            self.timer.start()
            self.runtime_label.setText('Видеофайл · безопасное воспроизведение' if self.video_path else 'Камера запускается в отдельном потоке…')
            self.start_button.setText('▶  Продолжить')
            self.note('Поток запущен' + (' · безопасное воспроизведение' if self.video_path else ' · камера в отдельном потоке'))
            self.note('Числовые измерения сеанса сохраняются при паузе/остановке; изображения не записываются.')
        except Exception as exc:
            self.stop_stream()
            self.note(str(exc))

    def release_control(self):
        self.engine.stable_controller.position_provider=None
        self._minimum_capture_time = time.monotonic()
        self.enable_input.setChecked(False)
        if self.receipt_dialog:self.receipt_dialog.enabled.setChecked(False)
        if self.actions:
            self.actions.release()
        self.engine.reset()

    def finish_session(self, reason):
        if self.session is None:
            return
        try:
            path = self.session.finish(reason)
            self.note(f'Сеанс сохранён: {path.name} · {len(self.session.frames)} кадров')
            self.session = None
        except (OSError, TypeError, ValueError) as exc:
            self.note(f'Не удалось сохранить сеанс: {exc}')

    def pause_stream(self, checked=False, reason='pause'):
        self.timer.stop()
        if self.pointing_dialog:self.pointing_dialog.reject()
        self._running = False
        if self.live_runtime:
            self.live_runtime.stop(wait=False)
        self.camera_activity.stop()
        self.release_control()
        if self.recording:
            self.cancel_recording()
        self.phase_label.setText('Фаза: пауза')
        self.preview.setText('Поток приостановлен\n\nДля продолжения нажмите «Запустить»')
        self.runtime_label.setText('Источник приостановлен · управление выключено')
        self.finish_session(reason)

    def stop_stream(self, checked=False, reason='stop'):
        self.pause_stream(reason=reason)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.phase_label.setText('Фаза: ожидание')
        self.preview.setText('Поток остановлен\n\nВыберите источник и нажмите «Запустить»')
        self.runtime_label.setText('Источник остановлен · управление выключено')

    def change_method(self):
        self.release_control()
        self.reload_engine()

    def backend_changed(self):
        if not self._settings_ready:return
        backend=self.tracking_backend.currentData()
        if self.pointing_dialog or self.workspace_dialog or self.receipt_dialog or self.recording or self.protocol:
            self.tracking_backend.blockSignals(True);self.tracking_backend.setCurrentIndex(self.tracking_backend.findData(self._configured_backend));self.tracking_backend.blockSignals(False)
            self.note('Сначала завершите настройку или проверку жестов.');return
        self.stop_stream(reason='tracker_change')
        if self.live_runtime and not self.live_runtime.stop(wait=True,timeout=2.):
            self.tracking_backend.blockSignals(True);self.tracking_backend.setCurrentIndex(self.tracking_backend.findData(self._configured_backend));self.tracking_backend.blockSignals(False)
            self.note('Предыдущий трекер ещё завершается. Повторите смену после остановки.');return
        self.live_runtime=None;self._live_uses_tracker=False
        if self.tracker is not None and not self._provided_tracker:self.tracker.close();self.tracker=None
        self._last_feature=None;self._configured_backend=backend;self.apply_settings()
        self.note('Трекер изменён. Нажмите «Запустить»; управление выключено.')

    def open_settings(self):
        if self.settings_dialog or self.pointing_dialog or self.workspace_dialog or self.receipt_dialog or self.recording or self.protocol:return
        self.release_control()
        from .settings_ui import SettingsDialog
        dialog=SettingsDialog(self.interaction_settings,self.library.list_gestures(),self.save_interaction_settings,self)
        self.settings_dialog=dialog
        dialog.custom_requested.connect(self.open_custom_settings)
        dialog.finished.connect(lambda _:setattr(self,'settings_dialog',None))
        dialog.open()
    def save_interaction_settings(self,settings,bindings):
        self.release_control();self.library.update_mappings(bindings)
        self.interaction_settings=settings;self.refresh_library();self.apply_settings()
        self.note('Настройки сохранены. Системный ввод включается отдельно.')
    def open_custom_settings(self):
        self.profile_selector.setCurrentIndex(self.profile_selector.findData('advanced'))
        self.record_name.setFocus()

    def reload_engine(self):
        from .engine import Engine
        self.engine = Engine(self.library, method=self.method.currentData(),load_models=False)
        self.apply_settings()

    def apply_settings(self):
        self.release_control()
        native=self.tracking_backend.currentData()=='apple_vision'
        self.pointer_mode.model().item(self.pointer_mode.findData('ray')).setEnabled(not native)
        if native and self.pointer_mode.currentData()!='workspace':
            self.pointer_mode.blockSignals(True);self.pointer_mode.setCurrentIndex(self.pointer_mode.findData('workspace'));self.pointer_mode.blockSignals(False)
        self.tracking_backend.setEnabled(not self._provided_tracker and self.tracker_factory is None and not self.pointing_dialog and not self.workspace_dialog and not self.receipt_dialog and not self.recording and not self.protocol)
        self.engine.conf_threshold = self.threshold.value()
        self.engine.hold_seconds = self.hold_time.value()
        stable = self.profile_selector.currentData()=='stable' and self._protocol_restore_stable is None
        self.engine.stable_profile = stable
        workspace=self.pointer_mode.currentData()=='workspace'
        if workspace and self._running and self._last_feature is not None:
            try:self.ensure_workspace(self._last_feature)
            except (ValueError,RuntimeError,ImportError,TypeError):self.workspace_mapper=None
        self.engine.pointing_calibration = self.workspace_mapper if workspace else self.pointing_calibration
        self.engine.stable_controller.prefer_image_geometry=workspace
        self.engine.stable_controller.robust_geometry=workspace
        self.engine.stable_controller.smooth_scroll=workspace
        self.engine.stable_controller.navigation_pose='three' if workspace else 'palm'
        self.engine.stable_controller.scroll_dead_zone=.006 if workspace else 0.
        self.pointer_mode.setEnabled(stable and not self.pointing_dialog and not self.workspace_dialog)
        self.calibrate_button.setText('Рабочая область · 2D' if workspace else 'Калибровка · 9 целей')
        self.target_test_button.setEnabled(stable and not workspace)
        self.calibrate_button.setEnabled(stable);self.gesture_test_button.setEnabled(stable)
        self.fov.setVisible(not workspace);self.fov_label.setVisible(not workspace)
        if workspace:
            self.pointing_status.setText('2D: палец вверх — двигайте кисть для перемещения курсора. Опустите пальцы или уберите руку, верните её удобно и снова поднимите палец: курсор сохранит место. Физический луч не используется.' if self.workspace_mapper is None or self.workspace_mapper.relative else 'Абсолютный 2D: положение кисти внутри рабочей области соответствует положению на экране. Физический луч не используется.')
            self.guide_label.setText('Палец вверх, кисть двигается — курсор · Щипок — клик / удержание для переноса\nДва пальца — прокрутка · Три пальца — свайп влево/вправо · Расслабьте пальцы между свайпами')
        else:
            self.pointing_status.setText('Экспериментальный 3D-луч: нужна калибровка физической плоскости. Её успешность и точность требуют независимой проверки.')
            self.guide_label.setText('Один указательный палец — экспериментальный луч к экрану · Щипок — клик\nДва пальца — прокрутка · Открытая ладонь: пауза, затем свайп влево / вправо')
        if not stable:
            self.pointing_status.setText('Расширенный профиль: личные жесты и исследовательские методы. Способ наведения основного профиля здесь не активен.')
            self.guide_label.setText('Запишите отдельные обучающие примеры своих жестов, затем проверьте новые повторы через TRIAL. Ввод в ОС включается отдельно.')
        if stable:self.engine.conf_threshold=.75
        advanced=not stable
        self.library_group.setVisible(advanced);self.record_group.setVisible(advanced);self.advanced_settings_group.setVisible(advanced)
        self.method.setEnabled(advanced);self.method.setVisible(advanced);self.confidence.setVisible(advanced)
        self.advanced_panel.setVisible(advanced)
        self.update_personal_diagnostics()
        self.engine.control_mode = 'personal' if self.protocol and self.protocol.mode=='trial' else self.control_mode.currentData()
        self.engine.experimental_neural = False if self.protocol or stable else self.experimental_neural.isChecked()
        if self.engine.experimental_neural and self.engine._predictor is None:
            try:self.engine.load_temporal_predictor(self.root)
            except (OSError,ValueError,RuntimeError,ImportError) as exc:
                self.engine.experimental_neural=False;self.experimental_neural.blockSignals(True);self.experimental_neural.setChecked(False);self.experimental_neural.blockSignals(False);self.note(f'Модель недоступна: {exc}')
        self.interaction_settings.apply(self.engine.stable_controller,self.workspace_mapper)
        if self._settings_ready:
            settings = {'method': self.engine.method, 'confidence': self.threshold.value(), 'hold_seconds': self.hold_time.value(),
                        'control_mode': self.control_mode.currentData(), 'experimental_neural': self.experimental_neural.isChecked(),
                        'profile':self.profile_selector.currentData(), 'fov_degrees':self.fov.value(),'pointer_mode':self.pointer_mode.currentData(),'tracker_backend':self.tracking_backend.currentData(),
                        'controller_geometry_source':self.effective_geometry_source(),'interaction':self.interaction_settings.to_dict()}
            path = self.root / 'data' / 'ui_settings.json'
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(settings, indent=2), encoding='utf-8')
                temporary.replace(path)
            except OSError as exc:
                self.note(f'Не удалось сохранить настройки: {exc}')
            if self.session:
                changes = self.session.metadata.setdefault('settings_changes', [])
                changes.append(dict(settings, timestamp=time.monotonic()))

    def resolve_display(self):
        from .pointing_ui import native_screens
        displays=(self.screen_provider or native_screens)()
        if len(displays)==1:self._display=displays[0]
        else:
            rectangle=self.screen().geometry()
            matches=[display for display in displays if tuple(display.geometry)==(rectangle.x(),rectangle.y(),rectangle.width(),rectangle.height())]
            if len(matches)!=1:raise ValueError('Выбранный экран неоднозначен. Управление заблокировано.')
            self._display=matches[0]
        if self._display.physical_size_m is None:raise ValueError('Физические размеры экрана недоступны; нельзя построить плоскость наведения.')
        return self._display

    def pointing_context(self,feature,missing_frame_size=None):
        from .pointing import CameraIntrinsics
        if not self._running or self.source.currentIndex()!=0 or self.video_path or feature is None:raise ValueError('Нужен работающий источник камеры.')
        frame_size=getattr(feature,'frame_size',None)
        if frame_size is None and not feature.present:frame_size=missing_frame_size
        if frame_size is None:raise ValueError('Нужен известный формат камеры.')
        screen=self.resolve_display();width,height=frame_size
        K=CameraIntrinsics.from_fov(width,height,self.fov.value())
        return {'camera_index':int(getattr(self.live_runtime,'camera_index',0)), 'source_kind':'camera',
                'frame_size':[width,height], 'intrinsics':K.to_dict(), 'display':screen.context(),
                'axis_definition':'PCA index joints 5/6/7/8, sign MCP-to-tip'}

    def pointing_ready(self):
        if self.pointing_dialog or self.workspace_dialog:return False,'Настройка наведения: системный ввод выключен.'
        if not self._running or self.video_path:return False,'Запустите камеру; видео не управляет системой.'
        if self.pointer_mode.currentData()=='workspace':
            feature=self._last_feature
            if feature is None or time.monotonic()-feature.timestamp>.5 or not feature.present:return False,'Нужна свежая отслеживаемая кисть.'
            try:
                context=self.workspace_context(feature)
                if self.workspace_mapper is None or not self.workspace_mapper.matches_context(context):return False,'Контекст рабочей области изменился.'
                self.workspace_mapper.hand_point(feature)
                if self.actions and hasattr(self.actions,'set_display') and self._actions_display!=self._display.display_id:
                    self.actions.set_display(self._display.display_id);self._actions_display=self._display.display_id
                return True,'2D-курсор по положению кисти; физический луч не используется.'
            except (ValueError,RuntimeError,ImportError,TypeError) as exc:return False,str(exc)
        if self.pointing_calibration is None:return False,'Сначала выполните калибровку наведения по девяти целям.'
        feature=self._last_feature
        if feature is None or time.monotonic()-feature.timestamp>.5 or not feature.present:return False,'Нужна свежая отслеживаемая рука.'
        if getattr(feature,'image_points',None) is None or getattr(feature,'world_points',None) is None:return False,'Для лазерного наведения нужны image/world координаты руки.'
        try:
            context=self.pointing_context(feature)
            if not self.pointing_calibration.matches_context(context):return False,'Изменились камера, формат, экран или FOV. Калибровку нужно повторить.'
            from .pointing import estimate_hand_pose
            pose,diagnostics=estimate_hand_pose(feature,self.pointing_calibration.intrinsics)
            if pose is None:return False,'Рука отслеживается ненадёжно: '+diagnostics['reason']
            if self.actions and hasattr(self.actions,'set_display') and self._actions_display!=self._display.display_id:
                self.actions.set_display(self._display.display_id);self._actions_display=self._display.display_id
            return True,'Калибровка согласована с текущей камерой и экраном. Точность проверяется отдельными целями.'
        except (ValueError,RuntimeError,ImportError,TypeError) as exc:return False,str(exc)

    def install_calibration(self,model):
        self.release_control();self.pointing_calibration=model;self.engine.pointing_calibration=model
        if self.session:self.session.metadata.setdefault('calibration_changes',[]).append({'timestamp':time.monotonic(),'context':model.context,'quality':model.quality})
        self.pointing_status.setText('Калибровка сохранена. Выполните отдельную проверку пяти целей; fit не доказывает живую точность.')

    def active_pointer_mapper(self):return self.workspace_mapper if self.pointer_mode.currentData()=='workspace' else self.pointing_calibration
    def effective_geometry_source(self):
        if not self.engine.stable_profile:return 'legacy_or_personal'
        return 'image_aspect_corrected' if self.pointer_mode.currentData()=='workspace' else 'world'
    def workspace_context(self,feature,missing_frame_size=None):
        if missing_frame_size is None and self.workspace_mapper is not None:missing_frame_size=self.workspace_mapper.context['frame_size']
        data=self.pointing_context(feature,missing_frame_size=missing_frame_size)
        context={k:data[k] for k in ['camera_index','source_kind','frame_size','display']}
        if self.tracking_backend.currentData()=='apple_vision':context['tracker_backend']='apple_vision'
        return context
    def ensure_workspace(self,feature,telemetry=None):
        from .workspace_pointer import WorkspaceMapper,WorkspaceBounds
        missing_size=None
        if telemetry and telemetry.get('width',0)>0 and telemetry.get('height',0)>0:missing_size=(int(telemetry['width']),int(telemetry['height']))
        context=self.workspace_context(feature,missing_frame_size=missing_size)
        if self.workspace_mapper is None or not self.workspace_mapper.matches_context(context):
            self.release_control();self.workspace_mapper=WorkspaceMapper(context,WorkspaceBounds(span=(.35,.35)),relative=True)
        self.engine.pointing_calibration=self.workspace_mapper
        self.workspace_mapper.palm_anchor='midpoint'
        self.engine.stable_controller.scroll_pixels_per_unit=self._display.geometry[3]*self.interaction_settings.scroll_gain
        self.interaction_settings.apply(self.engine.stable_controller,self.workspace_mapper)
    def open_workspace_editor(self):
        if self.workspace_dialog or self.pointing_dialog or self.receipt_dialog or self.recording or self.protocol:return
        try:
            self.ensure_workspace(self._last_feature);self.release_control()
            from .workspace_ui import WorkspaceEditor
            dialog=WorkspaceEditor(self.root,self.workspace_mapper,lambda:self.workspace_context(self._last_feature))
            self.workspace_dialog=dialog;dialog.saved.connect(self.install_workspace);dialog.finished.connect(self.workspace_editor_closed)
            self.profile_selector.setEnabled(False);self.pointer_mode.setEnabled(False);dialog.show()
        except (ValueError,RuntimeError,ImportError,TypeError) as exc:self.note(f'Для рабочей области запустите камеру: {exc}')
    def install_workspace(self,mapper):
        self.release_control();self.workspace_mapper=mapper;self.engine.pointing_calibration=mapper
        self.note('Рабочая область 2D сохранена. Ввод остаётся выключенным.')
    def workspace_editor_closed(self,result):
        self.workspace_dialog=None;self.profile_selector.setEnabled(True);self.apply_settings();self.release_control()
    def open_pointing_calibration(self):
        if self.pointer_mode.currentData()=='workspace':self.open_workspace_editor()
        else:self.open_pointing_dialog(False)
    def open_target_validation(self):self.open_pointing_dialog(True)
    def open_pointing_dialog(self,validation):
        if self.pointer_mode.currentData()=='workspace' or self.workspace_dialog or self.pointing_dialog or self.recording or self.protocol:return
        if not self._running or self.video_path or self._last_feature is None or not self._last_feature.present or time.monotonic()-self._last_feature.timestamp>.5 or getattr(self._last_feature,'world_points',None) is None or getattr(self._last_feature,'image_points',None) is None:
            self.note('Для калибровки нужна работающая камера и свежие наблюдения кисти.');return
        try:
            context=self.pointing_context(self._last_feature)
            if validation and (self.pointing_calibration is None or not self.pointing_calibration.matches_context(context)):
                raise ValueError('Нужна сохранённая калибровка текущей камеры/экрана перед независимой проверкой.')
            from .pointing_ui import RayTargetDialog,latest_resume_attempt
            self.release_control()
            dialog=RayTargetDialog(self.root,self._display,context,self.pointing_calibration if validation else None,
                                  context_provider=lambda:self.pointing_context(self._last_feature,missing_frame_size=context['frame_size']),
                                  resume_attempt=None if validation else latest_resume_attempt(self.root,context))
            self.pointing_dialog=dialog;dialog.calibrated.connect(self.install_calibration);dialog.report_failed.connect(self.note)
            dialog.finished.connect(self.pointing_dialog_closed);self.profile_selector.setEnabled(False);self.pointer_mode.setEnabled(False);self.fov.setEnabled(False)
            dialog.open_on_screen()
        except (ValueError,RuntimeError,ImportError) as exc:self.note(str(exc))
    def pointing_dialog_closed(self,result):
        self.pointing_dialog=None;self.profile_selector.setEnabled(True);self.fov.setEnabled(True);self.apply_settings();self.release_control()
        self.note('Калибровка / проверка целей завершена. Системный ввод остаётся выключенным.')

    def open_gesture_receipts(self):
        if self.receipt_dialog or self.pointing_dialog or self.workspace_dialog or self.recording or self.protocol:return
        if self.pointer_mode.currentData()=='workspace':
            try:self.ensure_workspace(self._last_feature)
            except (ValueError,RuntimeError,ImportError,TypeError) as exc:self.note(str(exc));return
        else:
            valid,reason=self.pointing_ready()
            if not valid:self.note(reason);return
        from .pointing_ui import GestureReceiptDialog
        self.release_control();dialog=GestureReceiptDialog(self.root,self._display,parent=self,navigation_pose=self.engine.stable_controller.navigation_pose);self.receipt_dialog=dialog
        def toggle(enabled):
            self.enable_input.setChecked(enabled)
            if enabled and not self.enable_input.isChecked():dialog.enabled.setChecked(False)
        def closed():
            self.receipt_dialog=None;self.release_control()
        dialog.input_requested.connect(toggle);dialog.closed.connect(closed);dialog.report_failed.connect(self.note);dialog.show()
        if self._display.qt_screen and dialog.windowHandle():dialog.windowHandle().setScreen(self._display.qt_screen)

    def show_stream_problem(self,message):
        self.phase_label.setText('Фаза: нет свежего кадра · ввод выключен')
        self.runtime_label.setText(message)
        self.preview.setText('Нет свежего кадра\n\n'+message)
        if self.receipt_dialog:
            self.receipt_dialog.enabled.setChecked(False);self.receipt_dialog.enabled.setEnabled(False)
            self.receipt_dialog.set_tracking_feedback(message+' Ввод выключен. Закройте проверку и перезапустите камеру, если поток не восстановится.')

    def update_personal_diagnostics(self):
        if self.engine.stable_profile:
            controller=self.engine.stable_controller
            phase={'recovery':'ожидание спокойного наведения','pointer':'наведение','pinch_candidate':'подготовка щипка','pinch_held':'щипок подтверждён','drag':'перенос','scroll_pending':'подготовка прокрутки','scroll':'прокрутка','swipe_pending':'подготовка свайпа','swipe':'движение ладони','swipe_latched':'свайп завершён','calibration_required':'нужна калибровка'}.get(controller.phase,controller.phase)
            details=''
            geometry=getattr(controller,'geometry',None)
            if geometry is not None and geometry.finger_states and np.isfinite(geometry.pinch):
                names=['указательный','средний','безымянный','мизинец'];states={'open':'прямой','folded':'согнут','uncertain':'неясно'}
                fingers=', '.join(f'{name}: {states.get(state,state)}' for name,state in zip(names,geometry.finger_states))
                details=f'\nЩипок {geometry.pinch:.2f} · нажать ≤{controller.pinch_press:.2f}, отпустить ≥{controller.pinch_release:.2f}\n{fingers}'
            self.diagnostic_label.setText(f'Пять жестов: {phase} · {controller.reason}'+details)
            if self.receipt_dialog:self.receipt_dialog.set_tracking_feedback(self.diagnostic_label.text())
            return
        result = getattr(self.engine, 'personal_result', {}) or {}
        phase = getattr(self.engine, 'personal_phase', 'idle')
        phase = {'idle': 'ожидание', 'static': 'поза', 'motion': 'движение', 'tracking_gap': 'потеря руки',
                 'recognized': 'распознан', 'rejected': 'отклонён'}.get(phase, phase)
        distance, radius = result.get('distance'), result.get('radius')
        numbers = f"расстояние {distance:.3f} / допуск {radius:.3f}" if distance is not None and radius is not None else 'расстояние: —'
        progress = float(result.get('progress') or 0.)
        reason = result.get('reason') or 'ожидание полного жеста'
        reason = {'idle': 'ожидание', 'motion': 'движение ещё не завершено', 'tracking_gap': 'ожидание после потери руки',
                  'recognized': 'жест подтверждён', 'rejected': 'жест отклонён'}.get(reason, reason)
        self.diagnostic_label.setText(f'Свои жесты: {phase} · {numbers} · прогресс {progress:.0%} · {reason}')

    def dispatch_events(self, events):
        for event in events:
            try:
                identifier=event.gesture if ':' in event.gesture or len(event.gesture)==32 else 'builtin:'+event.gesture
                display_name=self.library.get(identifier)['name']
            except StopIteration:display_name=event.gesture
            motion = event.action in {'move', 'scroll'}
            previous = self._motion_log_t.get(event.action, -float('inf'))
            if not motion or float(event.timestamp) - previous >= .2 - 1e-8:
                self.note(f'{display_name} → {event.action}' + (f' {event.payload}' if event.payload else ''))
                if motion:
                    self._motion_log_t[event.action] = float(event.timestamp)
            executed = False
            error = None
            if event.action == 'pause' and self.enable_input.isChecked():
                self.release_control()
            elif self.enable_input.isChecked() and not self.video_path and self.recording is None and self.protocol is None and self.pointing_dialog is None and self.workspace_dialog is None and self.actions:
                try:
                    valid=True
                    if self.engine.stable_profile:
                        valid,reason=self.pointing_ready()
                        if not valid:
                            error=reason
                            # Ordinary hand loss releases held buttons without requiring rearming.
                            if self._last_feature is not None and not self._last_feature.present:
                                self.actions.release()
                            else:self.release_control()
                            self.note(reason)
                    if valid:
                        self.actions.execute(event)
                        executed = event.action != 'none'
                except Exception as exc:
                    error = str(exc)
                    self.release_control()
                    self.note(f'Системный ввод остановлен: {exc}')
            if self.receipt_dialog:self.receipt_dialog.record_command(event,executed)
            if self.session:
                self.session.add_event(event, executed=executed, error=error)

    def tick(self):
        if self.live_runtime is not None and not self.video_path:
            if not self._running:
                return
            error = self.live_runtime.error
            if error:
                self.stop_stream(reason='camera_worker_error')
                self.show_stream_problem(f'Поток остановлен: {error}')
                self.note(f'Поток остановлен: {error}')
                return
            packet = self.live_runtime.take_latest()
            wall = time.monotonic()
            if packet is None:
                if wall - self._last_packet_wall > .5:
                    if not self._stream_stalled:
                        self._stream_stalled=True;self.release_control()
                        if self.session:self.session.metadata.setdefault('stream_stalls',[]).append({'timestamp':wall,'reason':'no_completed_packet','last_packet_wall':self._last_packet_wall})
                    self.show_stream_problem(f'Нет свежего кадра {wall-self._last_packet_wall:.1f} с.')
                return
            if packet.generation != self._live_generation or packet.capture_time < self._minimum_capture_time:
                return
            metrics = dict(packet.telemetry)
            metrics['capture_to_display_ms'] = (wall-packet.capture_time)*1000
            metrics['age_ms'] = metrics['capture_to_display_ms']
            self.runtime_label.setText(
                f'Обработка {metrics.get("inference_ms", 0):.0f} мс · кадр {metrics["capture_to_display_ms"]:.0f} мс · '
                f'{metrics.get("processed_frames", 0)}/{metrics.get("captured_frames", 0)} кадров · '
                f'пропущено {metrics.get("dropped_capture_frames", 0)}')
            if wall - packet.capture_time > .5:
                if not self._stream_stalled:self._stream_stalled=True;self.release_control()
                self.show_stream_problem('Устаревший кадр: управление заблокировано.')
                if self.session:
                    self.session.add_frame(packet.feature, prediction='stale', confidence=0., phase='stale',
                                           recording=self.recording is not None, method=self.engine.method, telemetry=metrics)
                return
            self._stream_stalled=False
            if self.receipt_dialog:self.receipt_dialog.enabled.setEnabled(True)
            missing_count = metrics.get('missing_hand_frames', 0)
            if missing_count > self._last_missing_hand_count and packet.feature.present:
                if self.actions:
                    self.actions.release()
                stamp = metrics.get('last_absent_time')
                if stamp is not None and np.isfinite(stamp) and getattr(self.engine, '_clock', -1.)<stamp<packet.capture_time:
                    from .types import FrameFeatures
                    self.process_observation(FrameFeatures.absent(float(stamp)), float(stamp),
                        telemetry={'replayed_absence': True, 'last_absent_time': float(stamp), 'missing_hand_frames': missing_count})
            self._last_missing_hand_count = missing_count
            self._last_packet_wall = wall
            try:
                self.consume_frame(packet.image, packet.feature, packet.capture_time, annotated=True, telemetry=metrics)
            except Exception as exc:
                self.stop_stream(reason='processing_error')
                self.note(f'Поток остановлен: {exc}')
            return
        if self.capture is None:
            return
        try:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.stop_stream(reason='end_of_video' if self.video_path else 'capture_failure')
                self.note('Конец видео или источник перестал возвращать кадры.')
                return
            now = time.monotonic()
            feature = self.tracker.process(frame, now)
            self.consume_frame(frame, feature, now)
        except Exception as exc:
            self.stop_stream(reason='processing_error')
            self.note(f'Поток остановлен: {exc}')

    def process_observation(self, feature, now, telemetry=None):
        """One real or timestamped missing observation, with no image-side effects."""
        self._last_feature=feature
        if self.workspace_dialog:
            self.workspace_dialog.observe(feature)
            if self.session:self.session.add_frame(feature,'workspace_setup',0,'workspace_setup',False,self.engine.method,telemetry=telemetry,task={'mode':'workspace_setup'})
            return
        if self.engine.stable_profile and self.pointer_mode.currentData()=='workspace':
            try:self.ensure_workspace(feature,telemetry=telemetry)
            except (ValueError,RuntimeError,ImportError,TypeError) as exc:
                self.workspace_mapper=None;self.engine.pointing_calibration=None;self.release_control();self.pointing_status.setText(f'Рабочая область недоступна: {exc}')
                if self.session:self.session.add_frame(feature,'workspace_unavailable',0,'workspace_context_error',False,self.engine.method,telemetry=telemetry,task={'mode':'workspace_context_error','pointer_mode':'workspace','controller_geometry_source':self.effective_geometry_source(),'error':str(exc)})
                return
        if self.pointing_dialog:
            try:
                # Missing-hand frames lack frame_size; reuse only its known geometry,
                # while actively rechecking source, display and FOV on every observation.
                known_size=self.pointing_dialog.context['frame_size']
                if telemetry and telemetry.get('width',0)>0 and telemetry.get('height',0)>0:
                    known_size=(int(telemetry['width']),int(telemetry['height']))
                if self.pointing_context(feature,missing_frame_size=known_size)!=self.pointing_dialog.context:
                    self.pointing_dialog.invalidate('Изменился экран, камера или формат. Отмените эту запись и начните заново.')
                else:self.pointing_dialog.observe(feature)
            except (ValueError,RuntimeError,ImportError,TypeError) as exc:
                self.pointing_dialog.invalidate(f'Контекст камеры / экрана недоступен: {exc}. Отмените и начните заново.')
            if self.session:self.session.add_frame(feature,'calibration',0,'calibration',False,self.engine.method,telemetry=telemetry,task={'mode':'pointing_validation' if self.pointing_dialog.validation else 'pointing_calibration'})
            return
        if self.engine.stable_profile and self.enable_input.isChecked():
            valid,reason=self.pointing_ready()
            if not valid and feature.present:
                self.release_control();self.pointing_status.setText(reason)
        was_recording = self.recording is not None and not (self.protocol and self.protocol.mode=='trial')
        events = []
        finish_capture = False
        if not feature.present and self.actions:
            self.actions.release()
        if self.recording:
            elapsed = now - self.recording.started - self.recording.countdown
            mode = self.protocol.mode.upper() if self.protocol else 'TRAIN'
            step = f' {self.protocol.index+1}/{self.protocol.total}' if self.protocol else ''
            self.record_status.setText(f'{mode}{step} · подготовка {max(0,-elapsed):.1f} с' if elapsed<0 else
                f'{mode}{step} · {min(elapsed,self.recording.duration):.1f}/{self.recording.duration:.0f} с · кадров {len(self.recording.samples)}')
            if self.protocol and self.protocol.mode=='trial':
                if elapsed>=0:
                    if not self.protocol.exercise_started:
                        self.engine.reset()
                        self.protocol.exercise_started = True
                    self.protocol.last_time = float(now)
                    if elapsed<=self.recording.duration:
                        self.recording.feed(feature, now)
                    else:
                        self.record_status.setText(f'TRIAL{step} · закончите движение, уберите руку; ожидание результата {max(0,self.recording.duration+self.protocol.grace-elapsed):.1f} с')
                    events = self.engine.process(feature)
                    self.protocol_events(events)
                    finish_capture = elapsed>=self.recording.duration+self.protocol.grace
            else:
                finish_capture = self.recording.feed(feature, now)
        else:
            events = self.engine.process(feature)
        if self.session:
            self.session.add_frame(feature, prediction=self.engine.last_label, confidence=self.engine.confidence,
                phase=self.engine.phase, recording=was_recording, method=self.engine.method, telemetry=telemetry, task=self.current_task(now))
        self.dispatch_events(events)
        self.update_personal_diagnostics()
        if finish_capture:
            self.finish_recording()
    def consume_frame(self, frame, feature, now, annotated=False, telemetry=None):
        """Run recognition, recording and OS actions exclusively on the GUI thread."""
        self.process_observation(feature, now, telemetry)
        self.gesture_label.setText(f'Жест: {self.engine.last_label or "—"}')
        self.phase_label.setText(f'Фаза: {self.engine.phase}')
        self.confidence.setValue(int(np.clip(self.engine.confidence, 0, 1) * 100))
        if self._last_frame_time is not None and now > self._last_frame_time:
            text = f'FPS: {1 / (now - self._last_frame_time):.1f}'
            if telemetry:
                text += f' · задержка {telemetry["capture_to_display_ms"]:.0f} мс'
            self.fps_label.setText(text)
        self._last_frame_time = now
        import cv2
        image_frame = frame if annotated else self.tracker.annotate(frame, feature)
        rgb = cv2.cvtColor(cv2.flip(image_frame, 1) if not self.video_path else image_frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.pointing_dialog:
            self.pointing_dialog.camera_preview.setPixmap(QPixmap.fromImage(image).scaled(self.pointing_dialog.camera_preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        if self.workspace_dialog:
            self.workspace_dialog.camera_preview.setPixmap(QPixmap.fromImage(image).scaled(self.workspace_dialog.camera_preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def current_task(self, now):
        if self.protocol is None or self.recording is None:
            return {'mode': 'train' if self.recording else 'idle','pointer_mode':self.pointer_mode.currentData(),'controller_geometry_source':self.effective_geometry_source(),'interaction':self.interaction_settings.to_dict()}
        elapsed = now-self.recording.started-self.recording.countdown
        mode = self.protocol.mode
        if elapsed<0:
            mode += '_countdown'
        elif mode=='trial' and elapsed>self.recording.duration:
            mode += '_grace'
        return {'mode': mode, 'expected_id': self.protocol.target, 'index': self.protocol.index+1,'pointer_mode':self.pointer_mode.currentData(),'controller_geometry_source':self.effective_geometry_source(),'interaction':self.interaction_settings.to_dict()}

    def begin_example(self):
        row = self.gestures.currentRow()
        if not 0 <= row < len(self._entries):
            return
        entry = self._entries[row]
        if entry['kind'] == 'builtin':
            return
        self.record_name.setText(entry['name'])
        self.record_kind.setCurrentIndex(self.record_kind.findData(entry['kind']))
        self.begin_recording(target=entry['id'])

    def _enter_custom_protocol(self):
        if self._protocol_restore_stable is None:self._protocol_restore_stable=self.engine.stable_profile
        self.engine.stable_profile=False;self.release_control()
    def _restore_custom_profile(self):
        self._protocol_restore_stable=None
        self.engine.stable_profile=self.profile_selector.currentData()=='stable';self.engine.pointing_calibration=self.active_pointer_mapper()
        if self.engine.stable_profile:self.engine.conf_threshold=.75;self.engine.experimental_neural=False

    def _record_controls(self, active):
        self.profile_selector.setEnabled(not active)
        self.pointer_mode.setEnabled(not active and self.profile_selector.currentData()=='stable')
        self.library_group.setEnabled(not active)
        self.control_mode.setEnabled(not active)
        self.method.setEnabled(not active)
        self.threshold.setEnabled(not active)
        self.hold_time.setEnabled(not active)
        self.experimental_neural.setEnabled(not active)
        self.record_button.setEnabled(not active)
        self.guided_button.setEnabled(not active)
        self.finish_button.setEnabled(active)
        self.cancel_button.setEnabled(active)
        self.record_name.setEnabled(not active)
        self.record_kind.setEnabled(not active)
        self.example_count.setEnabled(not active)
        self.enable_input.setEnabled(not active and not self.video_path)
        row = self.gestures.currentRow()
        custom = 0<=row<len(self._entries) and self._entries[row]['kind']!='builtin'
        self.example_button.setEnabled(not active and custom)
        self.trial_button.setEnabled(not active and custom)

    def begin_recording(self, checked=False, target=None):
        if self.recording or self.protocol:
            return
        if not self._running:
            self.note('Запустите камеру или видео перед записью жеста.')
            return
        name = self.record_name.text().strip()
        if not name:
            self.note('Введите название жеста.')
            return
        self._enter_custom_protocol()
        self._record_name = name
        self._record_target = target
        self.recording = RecordingSession(self.record_kind.currentData(), time.monotonic())
        self._record_controls(True)
        self.record_status.setText('TRAIN · подготовка: 3 секунды')
        self.note(f'Запись «{name}»: управление выключено.')

    def begin_guided_enrollment(self):
        if not self._running or self.recording or self.protocol:
            self.note('Для обучения запустите источник и завершите текущую запись.')
            return
        name = self.record_name.text().strip()
        if not name:
            self.note('Введите имя нового жеста.')
            return
        kind = self.record_kind.currentData()
        target = None
        row = self.gestures.currentRow()
        if 0 <= row < len(self._entries):
            entry = self._entries[row]
            if entry['kind'] == kind and entry['name'] == name:
                target = entry['id']
        self._enter_custom_protocol()
        self.protocol = GuidedProtocol('train', self.example_count.value(), target, name, kind,
                                      self.mapping.currentData(), self.parse_keys())
        self._start_protocol_step()

    def begin_trial(self):
        row = self.gestures.currentRow()
        if not self._running or self.recording or self.protocol or not 0 <= row < len(self._entries):
            self.note('Выберите свой жест и запустите источник перед независимой проверкой.')
            return
        entry = self._entries[row]
        if entry['kind'] == 'builtin':
            return
        self._enter_custom_protocol()
        self.protocol = GuidedProtocol('trial', self.example_count.value(), entry['id'], entry['name'],
                                      entry['kind'], entry['action'], entry.get('keys'),
                                      restore_mode=getattr(self.engine, 'control_mode', 'combined'))
        self.engine.control_mode = 'personal'
        self.engine.experimental_neural = False
        self._start_protocol_step()
        self.note('TRIAL: независимые повторы, шаблоны не изменяются, системный ввод заблокирован.')

    def _start_protocol_step(self):
        protocol = self.protocol
        self.release_control()
        protocol.events.clear();protocol.rejections.clear();protocol.distances.clear()
        protocol.last_time = None
        protocol.exercise_started = False
        self._record_name = protocol.name
        self._record_target = protocol.target
        self.recording = RecordingSession(protocol.kind, time.monotonic())
        self._record_controls(True)
        self.finish_button.setEnabled(protocol.mode!='trial')
        self.record_status.setText(f'{protocol.mode.upper()} · {protocol.index+1}/{protocol.total} · подготовка 3 с. Верните руку в исходное положение.')

    def protocol_events(self, events):
        protocol = self.protocol
        if protocol is None or protocol.mode != 'trial' or self.recording is None or self.recording.active_start is None:
            return
        evaluation_end = self.recording.started+self.recording.countdown+self.recording.duration+protocol.grace
        for event in events:
            if not self.recording.active_start<=event.timestamp<=evaluation_end:
                continue
            if event.action not in {'move','scroll','drag_start','drag_end'} or event.gesture == protocol.target:
                protocol.events.append({'gesture': event.gesture, 'action': event.action, 'timestamp': float(event.timestamp)})
        result = getattr(self.engine, 'personal_result', {}) or {}
        reason = result.get('reason')
        if reason and getattr(self.engine, 'personal_phase', '') == 'rejected':
            if not protocol.rejections or protocol.rejections[-1]['reason'] != reason:
                protocol.rejections.append({'timestamp': self.recording.active_end, 'reason': reason})
        distance = result.get('distance')
        if distance is not None and np.isfinite(distance):
            protocol.distances.append(float(distance))

    def _protocol_annotation(self, status):
        protocol, recording = self.protocol, self.recording
        if protocol is None or recording is None:
            return None
        matches = sum(event['gesture']==protocol.target for event in protocol.events)
        task = {'mode': protocol.mode, 'expected_id': protocol.target, 'name': protocol.name,
                'index': protocol.index+1, 'countdown_start': recording.started,
                'planned_start': recording.started+recording.countdown,
                'start': recording.active_start, 'exercise_end': recording.started+recording.countdown+recording.duration,
                'exercise_last_frame': recording.active_end, 'grace_seconds': protocol.grace if protocol.mode=='trial' else 0.,
                'evaluation_end': recording.started+recording.countdown+recording.duration+(protocol.grace if protocol.mode=='trial' else 0.),
                'end': protocol.last_time if protocol.mode=='trial' else recording.active_end, 'status': status,
                'matched_count': matches, 'detections': list(protocol.events),
                'rejections': list(protocol.rejections), 'quality': recording.quality(),
                'distance_min': min(protocol.distances) if protocol.distances else None}
        if self.session:
            self.session.metadata.setdefault('tasks', []).append(task)
        protocol.results.append(task)
        return task

    def cancel_recording(self, checked=False):
        if self.protocol:
            self._protocol_annotation('cancelled')
            if self.protocol.mode == 'trial':
                self.engine.control_mode = self.protocol.restore_mode
                self.engine.experimental_neural = self.experimental_neural.isChecked()
            self.protocol = None
        self.engine.experimental_neural = self.experimental_neural.isChecked()
        self.recording = None
        self._restore_custom_profile()
        self._record_controls(False)
        self.release_control()
        self.record_status.setText('Запись / проверка отменена. Уже сохранённые примеры сохранены.')

    def finish_recording(self):
        recording = self.recording
        if recording is None:
            return
        quality = recording.quality()
        protocol = self.protocol
        if protocol and protocol.mode=='trial':
            boundary = recording.started+recording.countdown+recording.duration+protocol.grace
            if protocol.last_time is None or protocol.last_time<boundary:
                self.record_status.setText('TRIAL: дождитесь конца движения и окна подтверждения результата.')
                return
        minimum = 5 if recording.kind == 'static' else 8
        reasons = []
        if len(recording.samples)<minimum:
            reasons.append('недостаточно кадров руки')
        if protocol and quality['coverage']<.6:
            reasons.append('рука видна менее 60% времени')
        if protocol and protocol.mode == 'train' and recording.kind == 'dynamic' and quality['motion']<.025:
            reasons.append('движение слишком мало; выполните жест целиком')
        if protocol and protocol.mode == 'train' and 0<quality['world_coverage']<1:
            reasons.append('неполная 3D геометрия; повторите с рукой в кадре')
        summary = f"рука {quality['coverage']:.0%}, 3D {quality['world_coverage']:.0%}, движение {quality['motion']:.3f}"
        if reasons and (protocol is None or protocol.mode == 'train'):
            self.note('Пример отклонён: '+ '; '.join(reasons))
            if protocol:
                self._protocol_annotation('rejected_quality')
                self.protocol = None
            self.cancel_recording()
            self.record_status.setText('Пример отклонён: '+ '; '.join(reasons)+' · '+summary)
            return
        try:
            if protocol is None or protocol.mode == 'train':
                target = protocol.target if protocol else self._record_target
                if target:
                    self.library.add_example(target, np.asarray(recording.samples), world_samples=recording.world_array())
                else:
                    target = self.library.add(self._record_name, recording.kind, np.asarray(recording.samples),
                        protocol.action if protocol else self.mapping.currentData(),
                        protocol.keys if protocol else self.parse_keys(), world_samples=recording.world_array())
                if protocol:
                    protocol.target = target
                if self.session:
                    self.session.metadata.setdefault('profile_changes', []).append(
                        {'timestamp': time.monotonic(), 'profiles': self.library.list_gestures()})
                self.refresh_library()
                self.gestures.setCurrentRow(next(i for i,e in enumerate(self._entries) if e['id']==target))
                self.reload_engine()
                self.note(f'Пример «{self._record_name}» сохранён · {summary}. Распознавание требует независимой проверки.')
            task = self._protocol_annotation('complete') if protocol else None
            if task:
                task['quality_reasons'] = reasons
            if protocol:
                protocol.index += 1
                if protocol.index < protocol.total:
                    self._start_protocol_step()
                    return
                if protocol.mode == 'trial':
                    counts = [item['matched_count'] for item in protocol.results]
                    message = f'TRIAL: подтверждённые совпадения по повторам {counts}. Шаблоны не изменены.'
                    bad = [item['index'] for item in protocol.results if item.get('quality_reasons')]
                    if bad:
                        message += f' Качество наблюдений недостаточно в повторах {bad}; их результат ограничен.'
                    self.engine.control_mode = protocol.restore_mode
                    self.engine.experimental_neural = self.experimental_neural.isChecked()
                else:
                    message = f'TRAIN: сохранено {protocol.total} отдельных примеров. Проверьте жест на новых повторах.'
                self.protocol = None
                self.engine.experimental_neural = self.experimental_neural.isChecked()
            else:
                message = f'Сохранено: {self._record_name} · {summary}. Проверка ещё не выполнена.'
            self.recording = None
            self._restore_custom_profile()
            self._record_controls(False)
            self.release_control()
            self.record_status.setText(message)
            self.note(message)
        except Exception as exc:
            self.cancel_recording()
            self.note(f'Не удалось завершить пример: {exc}')

    def parse_keys(self):
        return [key.strip().lower() for key in self.keys.text().replace('+', ',').split(',') if key.strip()] if self.mapping.currentData() == 'hotkey' else None

    def refresh_library(self):
        selected = self.gestures.currentRow()
        self.gestures.clear()
        self._entries = self.library.list_gestures()
        for entry in self._entries:
            kind = {'static': 'статический', 'dynamic': 'динамический', 'builtin': 'встроенный'}.get(entry['kind'], entry['kind'])
            self.gestures.addItem(f'{entry["name"]}  ·  {kind}')
        if self._entries:
            self.gestures.setCurrentRow(max(0, min(selected, len(self._entries) - 1)))

    def select_gesture(self, row):
        if 0 <= row < len(self._entries):
            entry = self._entries[row]
            if entry['kind'] != 'builtin':
                self.record_name.setText(entry['name'])
                self.record_kind.setCurrentIndex(self.record_kind.findData(entry['kind']))
            index = self.mapping.findData(entry.get('action', 'none'))
            self.mapping.setCurrentIndex(max(0, index))
            self.keys.setText(', '.join(entry.get('keys') or []))
            self.delete_button.setEnabled(entry['kind'] != 'builtin')
            self.example_button.setEnabled(entry['kind'] != 'builtin')
            self.tolerance.setEnabled(entry['kind']!='builtin')
            self.tolerance.setValue(entry.get('personal_radius', .07 if entry['kind']=='static' else .10))
            self.trial_button.setEnabled(entry['kind']!='builtin')
            self.orientation_sensitive.setEnabled(entry['kind']=='static')
            self.orientation_sensitive.setChecked(bool(entry.get('orientation_sensitive', False)))

    def save_mapping(self):
        row = self.gestures.currentRow()
        if not 0 <= row < len(self._entries):
            return
        self.release_control()
        try:
            self.library.update_mapping(self._entries[row]['id'], self.mapping.currentData(), self.parse_keys(),threshold=self.tolerance.value()*2,
                                        orientation_sensitive=self.orientation_sensitive.isChecked() if self._entries[row]['kind']=='static' else None)
            self.refresh_library()
            self.reload_engine()
            self.note('Команда сохранена.')
        except Exception as exc:
            self.note(f'Команда не сохранена: {exc}')

    def delete_gesture(self):
        row = self.gestures.currentRow()
        if not 0 <= row < len(self._entries):
            return
        entry = self._entries[row]
        if entry['kind'] == 'builtin':
            return
        if QMessageBox.question(self, 'Удалить жест', f'Удалить «{entry["name"]}» и его шаблон?') != QMessageBox.Yes:
            return
        self.release_control()
        try:
            self.library.delete(entry['id'])
            self.refresh_library()
            self.reload_engine()
            self.note(f'Жест «{entry["name"]}» удалён.')
        except Exception as exc:
            self.note(f'Удаление не выполнено: {exc}')

    def closeEvent(self, event):
        if self.workspace_dialog:self.workspace_dialog.reject()
        if self.receipt_dialog:self.receipt_dialog.reject()
        self.stop_stream(reason='close')
        if self.live_runtime:
            self.live_runtime.stop(wait=True, timeout=.25)
        if self.tracker and not self._live_uses_tracker:
            self.tracker.close()
        event.accept()


def run(root, video=None, screenshot=None,tracker_backend=None):
    import os
    if screenshot:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from .profiles import GestureLibrary
    from .engine import Engine
    from .actions import MacActions
    app = QApplication.instance() or QApplication([])
    app.setApplicationName('Stream Gesture')
    app.setFont(QFont('Helvetica Neue', 11))
    root = Path(root)
    library = GestureLibrary(root / 'data' / 'gesture_profiles')
    window = MainWindow(root, library, Engine(library, method=selected_method(root),load_models=False), actions=MacActions(),tracker_backend=tracker_backend)
    if video:
        window.set_video(video)
    window.show()
    if screenshot:
        path = Path(screenshot).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        window.note('Снимок интерфейса: камера не запущена, системный ввод выключен.')
        def save():
            ok = window.grab().save(str(path))
            window.close()
            app.exit(0 if ok else 1)
        QTimer.singleShot(200, save)
    elif video:
        QTimer.singleShot(0, window.start_stream)
    return app.exec()
