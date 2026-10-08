"""Camera/display calibration dialogs and a separate native-receipt test surface."""
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from datetime import datetime,timezone
import json
import time
import uuid
import numpy as np
from PySide6.QtCore import Qt,QPoint,QRect,QEvent,Signal,QTimer
from PySide6.QtGui import QPainter,QColor,QPen,QKeySequence,QShortcut
from PySide6.QtWidgets import QDialog,QLabel,QPushButton,QCheckBox,QTextEdit,QVBoxLayout,QHBoxLayout,QApplication,QWidget,QComboBox,QScrollArea
from .calibration_sampling import aggregate_rays
from .pointing import CameraIntrinsics,FingerRay,PointingCalibration,estimate_finger_ray,validation_errors


@dataclass
class DisplayInfo:
    display_id: int
    name: str
    geometry: tuple
    physical_size_m: tuple | None
    scale: float = 1.
    qt_screen: object = field(default=None,repr=False,compare=False)
    def context(self):
        return {'id':self.display_id,'name':self.name,'geometry':list(self.geometry),
                'physical_size_m':list(self.physical_size_m) if self.physical_size_m else None,'scale':self.scale}


def native_screens():
    """Read native IDs/bounds and require unambiguous Qt logical-coordinate matches."""
    import Quartz as q
    from PySide6.QtGui import QGuiApplication
    status,identifiers,count=q.CGGetActiveDisplayList(32,None,None)
    if status:raise RuntimeError('Не удалось определить активные дисплеи')
    result=[]
    for screen in QGuiApplication.screens():
        rectangle=screen.geometry();wanted=np.array([rectangle.x(),rectangle.y(),rectangle.width(),rectangle.height()],float)
        matches=[]
        for identifier in identifiers[:count]:
            bounds=q.CGDisplayBounds(identifier);geometry=(bounds.origin.x,bounds.origin.y,bounds.size.width,bounds.size.height)
            if np.allclose(wanted,geometry,atol=1):matches.append((identifier,geometry))
        if len(matches)!=1:continue
        identifier,geometry=matches[0];physical=q.CGDisplayScreenSize(identifier)
        size=(physical.width/1000,physical.height/1000) if min(physical.width,physical.height)>0 else None
        result.append(DisplayInfo(int(identifier),screen.name(),tuple(int(v) for v in geometry),size,float(screen.devicePixelRatio()),screen))
    if not result:raise RuntimeError('Qt и macOS не согласовали экран. Управление заблокировано.')
    return result


def json_save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));temporary.replace(path)


def timestamp_name(prefix):return prefix+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]+'.json'


def latest_resume_attempt(root,context):
    directory=Path(root)/'reports'/'pointing-v3'/'attempts'
    for path in sorted(directory.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        try:
            data=json.loads(path.read_text())
            if data.get('kind')=='pointing_calibration_attempt' and data.get('context')==context and 0<len(data.get('samples',[]))<9:
                data['source']=str(path);return data
        except (OSError,ValueError,TypeError):continue
    return None


class RayTargetDialog(QDialog):
    """User advances every target; calibration and held-out validation cannot mix."""
    calibrated=Signal(object)
    report_failed=Signal(str)
    def __init__(self,root,screen,context,validation_model=None,clock=time.monotonic,ray_estimator=estimate_finger_ray,context_provider=None,resume_attempt=None):
        super().__init__();self.root=Path(root);self.screen=screen;self.context=context
        self.model=validation_model;self.validation=validation_model is not None;self.clock=clock;self.ray_estimator=ray_estimator
        self.context_provider=context_provider
        self.intrinsics=CameraIntrinsics.from_dict(context['intrinsics'])
        self.targets=[(x,y) for y in [.18,.44,.70] for x in [.18,.5,.82]] if validation_model is None else [( .25,.25),(.75,.25),(.5,.65),(.25,.75),(.75,.75)]
        self.index=0;self.completed=False;self.invalidated=False;self._committed=False;self._reported_outcomes=set();self._last_error=None;self.buffer=deque(maxlen=300);self.samples=[];self.rays=[];self.actual_targets=[]
        self.collecting=False;self.collection_start=None;self.retry_mode=False;self.fit_attempts=0
        self.resume_attempt=resume_attempt if not self.validation and resume_attempt and resume_attempt.get('context')==context and 0<len(resume_attempt.get('samples',[]))<len(self.targets) else None
        self.resumed_from=None
        self.setWindowTitle('Калибровка наведения' if validation_model is None else 'Независимая проверка наведения')
        self.setWindowFlags(Qt.Dialog|Qt.FramelessWindowHint);self.setGeometry(QRect(*screen.geometry))
        self.setStyleSheet('QDialog{background:#142238;color:#eef4ff} QLabel{color:#eef4ff;font-size:17px} QPushButton{padding:12px;background:#eef4ff;color:#142238;border-radius:6px;font-size:15px}')
        self.message=QLabel(self);self.message.setWordWrap(True)
        self.status=QLabel('Ожидание наблюдений кисти…',self);self.status.setWordWrap(True)
        self.camera_preview=QLabel('Камера: держите кисть целиком в кадре',self);self.camera_preview.setWordWrap(True)
        self.camera_preview.setStyleSheet('background:#243750;color:#eef4ff;font-size:13px')
        self.capture_button=QPushButton('Начать сбор · Space',self);self.capture_button.clicked.connect(self.begin_collection)
        self.retry_selector=QComboBox(self);self.retry_selector.addItems([f'Повторить цель {i+1}' for i in range(len(self.targets))]);self.retry_selector.hide()
        self.retry_selector.activated.connect(self.select_retry_target)
        self.lift_button=QPushButton('Поднять цель · ↑',self);self.lift_button.clicked.connect(lambda:self.move_target(0,-.06))
        self.lift_shortcut=QShortcut(QKeySequence('Up'),self);self.lift_shortcut.activated.connect(lambda:self.move_target(0,-.06))
        self.lift_button.setVisible(not self.validation);self.lift_button.setEnabled(not self.validation);self.lift_shortcut.setEnabled(not self.validation)
        self.resume_button=QPushButton(f'Продолжить · {len(self.resume_attempt["samples"])} целей сохранено' if self.resume_attempt else 'Нет сохранённых целей',self)
        self.resume_button.clicked.connect(self.resume_previous);self.resume_button.setVisible(self.resume_attempt is not None)
        self.cancel_button=QPushButton('Отмена · Esc',self);self.cancel_button.clicked.connect(self.reject)
        self.shortcut=QShortcut(QKeySequence('Space'),self);self.shortcut.activated.connect(self.begin_collection)
        self.collection_timer=QTimer(self);self.collection_timer.setInterval(50);self.collection_timer.timeout.connect(self.collection_tick)
        self._prompt()
    def open_on_screen(self):
        self.show()
        if self.screen.qt_screen and self.windowHandle():self.windowHandle().setScreen(self.screen.qt_screen);self.showFullScreen()
    def _prompt(self):
        mode='Калибровка' if self.model is None else 'Независимая проверка'
        move_hint='' if self.validation else ' Если цель неудобна, поднимите её кнопкой ↑ до начала сбора.'
        self.message.setText(f'{mode}: цель {self.index+1}/{len(self.targets)}. Нажмите Space: 1,2 с подготовки, затем 2,2 с автоматического сбора. Направляйте прямой указательный палец к центру цели; обычное дрожание допустимо. Кисть должна быть видна в мини-камере.'+move_hint+f'\nСистемный ввод выключен. FOV {self.intrinsics.horizontal_fov_degrees or "задан"}° — предположение, не измерение камеры.')
        if self._last_error is not None:self.message.setText(self.message.text()+f'\nПредыдущая цель: ошибка {self._last_error:.1f} логических px.')
        self._place_controls();self.update()
    def _place_controls(self):
        self.message.setGeometry(35,15,max(200,self.width()-70),105)
        self.status.setGeometry(35,self.height()-130,max(200,self.width()-70),60)
        self.retry_selector.setGeometry(self.width()//2-135,self.height()-230,270,38)
        self.camera_preview.setGeometry(20,self.height()-300,240,135)
        self.lift_button.setGeometry(self.width()-245,self.height()-180,225,44)
        self.resume_button.setGeometry(self.width()//2-190,self.height()-280,380,44)
        self.capture_button.setGeometry(self.width()//2-135,self.height()-62,270,44)
        self.cancel_button.setGeometry(max(0,self.width()-165),self.height()-60,145,40)
    def resizeEvent(self,event):self._place_controls();super().resizeEvent(event)
    def move_target(self,dx,dy):
        if self.validation or self.collecting or self.completed or self.invalidated or self.index>=len(self.targets):return
        if not self.verify_context():return
        u,v=self.targets[self.index];self.targets[self.index]=(float(np.clip(u+dx,.12,.88)),float(np.clip(v+dy,.18,.78)))
        self.buffer.clear();self._prompt()
    def resume_previous(self):
        if not self.resume_attempt or self.index or self.samples or self.collecting or self.invalidated or self.completed:return
        if not self.verify_context():return
        try:
            samples=json.loads(json.dumps(self.resume_attempt['samples']));rays=[];targets=[]
            for sample in samples:
                target=np.asarray(sample['target'],dtype=float);local=np.asarray(sample['target_local_requested'],dtype=float)
                if target.shape!=(2,) or local.shape!=(2,) or not np.isfinite(target).all() or not np.isfinite(local).all() or np.any(target<0) or np.any(target>1) or np.any(local<0) or np.any(local>1):raise ValueError('Некорректная сохранённая цель')
                rays.append(FingerRay(sample['origin'],sample['direction'],sample.get('quality',{})));targets.append(target)
            self.samples=samples;self.rays=rays;self.actual_targets=targets
            for i,sample in enumerate(samples):self.targets[i]=tuple(sample['target_local_requested'])
            self.index=len(samples);self.resumed_from=self.resume_attempt.get('source');self.resume_button.hide();self.buffer.clear();self._prompt()
        except (KeyError,ValueError,TypeError) as exc:self.message.setText(f'Не удалось восстановить цели: {exc}. Начните новую серию.');self.resume_button.hide()
    def target_local_point(self):
        u,v=self.targets[min(self.index,len(self.targets)-1)]
        return QPoint(round(u*(self.width()-1)),round(v*(self.height()-1)))
    def target_screen_coordinates(self):
        point=self.mapToGlobal(self.target_local_point());x,y,w,h=self.screen.geometry
        return np.array([(point.x()-x)/(w-1),(point.y()-y)/(h-1)],dtype=float)
    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.Antialiasing)
        if not self.completed:
            point=self.target_local_point();painter.setPen(QPen(QColor('#ffffff'),3));painter.setBrush(QColor('#4793ff'));painter.drawEllipse(point,22,22)
            painter.setPen(QPen(QColor('#ffffff'),2));painter.drawLine(point.x()-35,point.y(),point.x()+35,point.y());painter.drawLine(point.x(),point.y()-35,point.x(),point.y()+35)
    def observe(self,feature):
        if self.invalidated:return
        reason=None;ray=None;diagnostics={}
        if not feature.present:reason='Кисть не обнаружена. Держите кисть целиком в кадре.'
        elif getattr(feature,'frame_size',None) is None or list(feature.frame_size)!=self.context['frame_size']:reason='Изменился формат камеры; начните калибровку заново.'
        else:
            try:
                if self.ray_estimator is estimate_finger_ray:
                    ray,diagnostics=self.ray_estimator(feature,self.intrinsics,return_diagnostics=True)
                else:
                    ray=self.ray_estimator(feature,self.intrinsics);diagnostics={}
                if ray is None:reason='Ось пальца не оценена: '+diagnostics.get('reason','держите кисть целиком в кадре и палец прямо')+'.'
                elif ray.direction[2]>=-.05:ray=None;reason='Направьте палец к дисплею, не вверх и не параллельно экрану.'
            except (ValueError,TypeError):reason='Ненадёжные координаты руки.'
        raw={'timestamp':float(feature.timestamp),'image_points':np.asarray(feature.image_points).tolist() if getattr(feature,'image_points',None) is not None else None,
             'world_points':np.asarray(feature.world_points).tolist() if getattr(feature,'world_points',None) is not None else None,
             'frame_size':list(feature.frame_size) if getattr(feature,'frame_size',None) else None,
             'geometry_label':getattr(feature,'geometry_label',None),'ray_rejection_reason':reason,'ray_diagnostics':diagnostics}
        if not self.collecting or float(feature.timestamp)>=self.collection_start:
            self.buffer.append((float(feature.timestamp),ray,raw,reason))
        recent=[entry for entry in self.buffer if self.clock()-.55<=entry[0]<=self.clock()+.01]
        valid=[entry for entry in recent if entry[1] is not None]
        spread_text=''
        if valid:
            directions=np.array([entry[1].direction for entry in valid]);mean=directions.mean(axis=0)
            if np.linalg.norm(mean)>1e-9:
                mean/=np.linalg.norm(mean)
                spread=float(np.rad2deg(np.arccos(np.clip(directions@mean,-1,1))).max())
                spread_text=f' Разброс: {spread:.1f}°.'
        if self.collecting:self.collection_tick()
        else:self.status.setText(reason or f'Луч оценён. Пригодных кадров: {len(valid)}/{len(recent)}.'+spread_text+' Нажмите Space для автоматического сбора.')
    def begin_collection(self):
        if self.completed or self.invalidated or self.collecting or self.index>=len(self.targets):return
        if not self.verify_context():return
        self.collecting=True;self.collection_start=self.clock()+1.2;self.buffer.clear()
        self.capture_button.setEnabled(False);self.retry_selector.setEnabled(False);self.lift_button.setEnabled(False);self.resume_button.setEnabled(False);self.collection_timer.start();self.collection_tick()
    def collection_tick(self):
        if not self.collecting:return
        now=self.clock()
        if now<self.collection_start:self.status.setText(f'Подготовка: {self.collection_start-now:.1f} с. Наведите палец на цель, больше клавиши нажимать не нужно.');return
        remaining=self.collection_start+2.2-now
        if remaining>0:
            valid=sum(e[1] is not None for e in self.buffer)
            reason=next((e[3] for e in reversed(self.buffer) if e[3]),'')
            self.status.setText(f'Сбор: осталось {remaining:.1f} с · пригодных кадров {valid}. '+(reason or 'Продолжайте указывать на центр цели.'));return
        self.collection_timer.stop();self.collecting=False;self.capture_button.setEnabled(True);self.retry_selector.setEnabled(True);self.lift_button.setEnabled(not self.validation);self.resume_button.setEnabled(True)
        self.capture_target(timed=True)
    def select_retry_target(self,index):
        if self.collecting or self.invalidated or self.completed or not self.retry_mode:return
        self.index=int(index);self.buffer.clear();self.capture_button.setEnabled(True);self.shortcut.setEnabled(True);self._prompt()
    def capture_target(self,timed=False):
        if self.completed or self.invalidated or self.index>=len(self.targets):return
        if not self.verify_context():return
        now=self.clock()
        try:ray,valid=aggregate_rays(list(self.buffer),now,min_frames=12 if timed else 5,min_span=1. if timed else .35)
        except ValueError as exc:
            self.message.setText(str(exc)+' Нажмите Space для повторного сбора этой цели.');return
        target=self.target_screen_coordinates()
        if np.any(target<0) or np.any(target>1):self.message.setText('Мишень оказалась вне выбранного экрана. Окно не согласовано с дисплеем.');return
        if self.model is not None:
            try:predicted=self.model.apply(ray)
            except ValueError as exc:self.message.setText(str(exc));return
            error=float(np.linalg.norm((predicted-target)*np.array(self.screen.geometry[2:])));self._last_error=error
            self.message.setText(f'Цель {self.index+1}: ошибка {error:.1f} логических px. Это проверка, не обновление калибровки.')
        sample={'target_local_requested':list(self.targets[self.index]),'target':target.tolist(),
                             'global_target':[self.mapToGlobal(self.target_local_point()).x(),self.mapToGlobal(self.target_local_point()).y()],
                             'origin':ray.origin.tolist(),'direction':ray.direction.tolist(),'quality':ray.quality,'raw':[entry[2] for entry in valid]}
        if self.retry_mode:
            self.rays[self.index]=ray;self.actual_targets[self.index]=target;self.samples[self.index]=sample
            self.index=len(self.targets)-1
        else:
            self.rays.append(ray);self.actual_targets.append(target);self.samples.append(sample)
        self.index+=1;self.buffer.clear()
        self.resume_button.hide()
        if self.index<len(self.targets):
            self._prompt();return
        try:
            if self.model is None:
                if self.screen.physical_size_m is None:raise ValueError('Физические размеры экрана не определены; калибровка невозможна.')
                model=PointingCalibration.fit(self.rays,np.asarray(self.actual_targets),self.context,self.screen.physical_size_m)
                model.samples=[dict(sample) for sample in self.samples]
                self.model=model
                result=f'Калибровка рассчитана. Нажмите «Сохранить», чтобы заменить прежнюю. Ошибка на обучающих целях: {model.quality["training_rmse_normalized"]:.3f}. Теперь выполните независимую проверку.'
            else:
                errors=validation_errors(self.model,self.rays,np.asarray(self.actual_targets),self.screen.geometry[2:])
                json_save(self.root/'data'/'pointing_trials'/timestamp_name('targets'),
                          {'kind':'independent_target_validation','context':self.context,'samples':self.samples,'errors':errors,
                           'calibration_quality':self.model.quality,'os_input':False})
                result=f'Независимая проверка: RMS {errors["rmse_pixels"]:.1f} px; p95 {errors["p95_pixels"]:.1f} px. Калибровка не изменена.'
            self.completed=True;self.retry_selector.hide();self.capture_button.setText('Завершить' if self.validation else 'Сохранить');self.capture_button.clicked.disconnect();self.capture_button.clicked.connect(self.accept if self.validation else self.commit_calibration)
            self.message.setText(result);self.update()
        except (ValueError,OSError,TypeError) as exc:
            self.message.setText(f'Калибровка отклонена: {exc}. Старый файл сохранён. Начните заново после проверки посадки и направления пальца.')
            self.fit_attempts+=1;self.save_attempt('fit_rejected' if self.fit_attempts==1 else f'fit_rejected_{self.fit_attempts}',str(exc))
            if isinstance(exc,ValueError) and not self.validation:
                self.retry_mode=True;self.retry_selector.show();self.shortcut.setEnabled(False)
                self.message.setText(f'Модель отклонила собранные цели: {exc}. Выберите внизу цель для повторного сбора. Остальные восемь сохранятся; прежняя калибровка не заменена.')
            else:self.invalidate(self.message.text())

    def invalidate(self,reason):
        self.invalidated=True;self.collecting=False;self.collection_timer.stop();self.buffer.clear();self.capture_button.setEnabled(False);self.shortcut.setEnabled(False);self.retry_selector.setEnabled(False)
        self.message.setText(reason)

    def verify_context(self):
        if self.context_provider is None:return True
        try:
            if self.context_provider()!=self.context:raise ValueError('Изменились камера, экран или формат.')
        except (ValueError,RuntimeError,ImportError,TypeError) as exc:
            self.invalidate(f'Контекст камеры / экрана недоступен: {exc}. Начните заново.');return False
        return True

    def save_attempt(self,outcome,error=None):
        if outcome in self._reported_outcomes:return
        self._reported_outcomes.add(outcome)
        try:
            json_save(self.root/'reports'/'pointing-v3'/'attempts'/timestamp_name('ray-attempt'),
                      {'kind':'pointing_validation_attempt' if self.validation else 'pointing_calibration_attempt',
                       'outcome':outcome,'error':error,'context':self.context,'samples':self.samples,'resumed_from':self.resumed_from,
                       'pending_observations':[entry[2] for entry in self.buffer], 'os_input':False})
        except (OSError,ValueError,TypeError) as exc:
            message=f'Не удалось сохранить числовой отчёт: {exc}'
            self.message.setText(self.message.text()+'\n'+message);self.report_failed.emit(message)

    def done(self,result):
        self.collection_timer.stop();self.collecting=False
        if result==QDialog.Rejected and not self._committed:self.save_attempt('cancelled')
        super().done(result)

    def commit_calibration(self):
        if self.validation or not self.completed or self.model is None:return
        if not self.verify_context():return
        try:self.model.save(self.root/'data'/'pointing_calibration.json')
        except OSError as exc:self.message.setText(f'Не удалось сохранить калибровку: {exc}');return
        self._committed=True;self.calibrated.emit(self.model);self.accept()


class ScrollPracticeArea(QScrollArea):
    """Explicit native pixel-wheel handling, independent of QTextEdit lines."""
    def __init__(self,parent=None):
        super().__init__(parent);self.setObjectName('scroll-area');self.setWidgetResizable(True)
        self.setMinimumHeight(220);self.setFocusPolicy(Qt.NoFocus)
        content=QWidget();layout=QVBoxLayout(content);layout.setSpacing(12)
        for i in range(1,21):
            card=QLabel(f'{i:02d}   ПРОКРУТКА\nДва пальца вверх/вниз. Курсор остаётся внутри этой области.')
            card.setMinimumHeight(100);card.setWordWrap(True)
            color='#e4f0fb' if i%2 else '#e9f4eb'
            card.setStyleSheet(f'background:{color};color:#243248;border-radius:10px;padding:16px;font-size:18px;')
            layout.addWidget(card)
        self.setWidget(content)
    def wheelEvent(self,event):
        if not event.pixelDelta().isNull():
            bar=self.verticalScrollBar();bar.setValue(bar.value()-event.pixelDelta().y());event.accept()
        else:super().wheelEvent(event)


class GestureReceiptDialog(QDialog):
    """Widget receipts, not inferred success. This dialog never injects input."""
    input_requested=Signal(bool)
    closed=Signal()
    report_failed=Signal(str)
    def __init__(self,root,screen,parent=None,navigation_pose='palm'):
        super().__init__(parent);self.root=Path(root);self.screen=screen;self.receipts=[];self.commands=[];self.page=0;self._saved=False
        self.setWindowTitle('Проверка пяти жестов');self.resize(min(screen.geometry[2],1150),min(screen.geometry[3],820))
        x,y,w,h=screen.geometry;dw=min(1150,max(400,w-80));dh=min(850,max(400,h-80))
        self.setGeometry(QRect(x+(w-dw)//2,y+(h-dh)//2,dw,dh))
        self.setWindowModality(Qt.WindowModal)
        layout=QVBoxLayout(self)
        motion='Два пальца: вверх/вниз — плавная прокрутка, влево/вправо — переход. Расслабьте пальцы между переходами.' if navigation_pose=='victory' else 'Открытая ладонь: короткая пауза, затем свайп влево / вправо.'
        if navigation_pose=='three':motion='Два пальца — только прокрутка. Три пальца (указательный, средний, безымянный) — свайп влево/вправо; мизинец согнут. Неподвижная пауза не нужна.'
        label=QLabel('Указательный палец — наведение на цель выбранным способом. В 2D двигайте кисть, держа ладонь к камере.\nЩипок — клик; остальные три пальца согнуты. Два пальца — прокрутка.\n'+motion);label.setWordWrap(True);layout.addWidget(label)
        self.enabled=QCheckBox('Включить жесты в этом окне');self.enabled.toggled.connect(self.input_requested);layout.addWidget(self.enabled)
        self.status=QLabel('Системный ввод выключен. Записи попаданий создаются только по событиям окна.');layout.addWidget(self.status)
        self.tracking_status=QLabel('Ожидание свежего наблюдения кисти…');self.tracking_status.setWordWrap(True);self.tracking_status.setFixedHeight(80);layout.addWidget(self.tracking_status)
        targets=QHBoxLayout()
        for i in range(3):
            button=QPushButton(f'Цель {i+1} · щипок');button.setObjectName(f'click-target-{i+1}');button.setMinimumHeight(80);targets.addWidget(button)
        layout.addLayout(targets)
        self.text=ScrollPracticeArea();layout.addWidget(self.text,1)
        self.scroll_feedback=QLabel('Прокрутка: 0 px · наведите курсор на цветную область');layout.addWidget(self.scroll_feedback)
        self.text.verticalScrollBar().valueChanged.connect(self.scroll_changed)
        self.page_label=QLabel();self.page_label.setAlignment(Qt.AlignCenter);self.page_label.setMinimumHeight(70);self.update_page();layout.addWidget(self.page_label)
        self.left=QShortcut(QKeySequence('Ctrl+Left'),self);self.left.activated.connect(lambda:self.navigate(-1,'command+left'))
        self.right=QShortcut(QKeySequence('Ctrl+Right'),self);self.right.activated.connect(lambda:self.navigate(1,'command+right'))
        close=QPushButton('Завершить · Esc');close.clicked.connect(self.reject);layout.addWidget(close)
        QApplication.instance().installEventFilter(self)
    def eventFilter(self,obj,event):
        if obj is self and event.type()==QEvent.WindowDeactivate:
            self.enabled.setChecked(False)
        if obj is self or (isinstance(obj,QWidget) and self.isAncestorOf(obj)):
            if event.type() in {QEvent.MouseButtonPress,QEvent.Wheel}:
                position=event.globalPosition();record={'type':'mouse_press' if event.type()==QEvent.MouseButtonPress else 'wheel',
                    'timestamp':time.monotonic(),'global_position':[position.x(),position.y()], 'target':obj.objectName() or type(obj).__name__,
                    'scene_page':self.page,'spontaneous':bool(event.spontaneous())}
                if event.type()==QEvent.Wheel:record['angle_delta']=[event.angleDelta().x(),event.angleDelta().y()];record['pixel_delta']=[event.pixelDelta().x(),event.pixelDelta().y()]
                self.receipts.append(record);self.status.setText(f'Событий окна: {len(self.receipts)}. Они ещё не сопоставлены с командами распознавания.')
        return super().eventFilter(obj,event)
    def navigate(self,delta,shortcut):
        self.page+=delta;self.update_page()
        self.receipts.append({'type':'shortcut','shortcut':shortcut,'timestamp':time.monotonic(),'scene_page':self.page})
    def update_page(self):
        index=(self.page+2)%5;colors=['#eadffc','#dff0fc','#def2e4','#fff0d4','#fce1e4']
        self.page_label.setText(f'←   Страница {index+1} из 5   →')
        self.page_label.setStyleSheet(f'background:{colors[index]};color:#243248;border-radius:10px;font-size:27px;font-weight:600;padding:10px;')
    def scroll_changed(self,value):
        self.scroll_feedback.setText(f'Прокрутка: {value} / {self.text.verticalScrollBar().maximum()} px · курсор над цветной областью')
        self.receipts.append({'type':'scroll_position','timestamp':time.monotonic(),'value':value,'maximum':self.text.verticalScrollBar().maximum()})
    def record_command(self,event,executed):
        self.commands.append({'gesture':event.gesture,'action':event.action,'capture_timestamp':event.timestamp,
                              'dispatch_timestamp':time.monotonic(),'payload':dict(event.payload),'executed':bool(executed)})
    def set_tracking_feedback(self,text):self.tracking_status.setText(text)
    def done(self,result):
        # Safety cleanup precedes storage; neither disk failure nor JSON errors may retain input.
        self.enabled.setChecked(False)
        QApplication.instance().removeEventFilter(self)
        self.closed.emit()
        try:
            if not self._saved:
                json_save(self.root/'data'/'pointing_trials'/timestamp_name('gesture-receipts'),
                          {'kind':'gesture_widget_receipts','display':self.screen.context(),'receipts':self.receipts,'engine_commands':self.commands,
                           'automatic_success_claim':False})
                self._saved=True
        except (OSError,ValueError,TypeError) as exc:
            message=f'Не удалось сохранить отчёт окна: {exc}. Управление выключено.'
            self.status.setText(message);self.report_failed.emit(message)
        finally:super().done(result)
    def closeEvent(self,event):self.reject();event.accept()
