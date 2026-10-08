"""Comfortable 2D camera workspace editor; no physical ray calibration or OS input."""
from pathlib import Path
import time
import numpy as np
from PySide6.QtCore import Qt,Signal,QRectF
from PySide6.QtGui import QPainter,QPen,QColor
from PySide6.QtWidgets import QDialog,QLabel,QSlider,QPushButton,QVBoxLayout,QHBoxLayout,QFormLayout,QCheckBox
from .workspace_pointer import WorkspaceMapper,WorkspaceBounds

class WorkspacePreview(QLabel):
    def __init__(self,bounds,parent=None):
        super().__init__(parent);self.bounds=bounds;self.hand=None
        self.setAlignment(Qt.AlignCenter);self.setMinimumSize(480,270)
        self.setText('Камера запускается…');self.setStyleSheet('background:#182b43;color:white')
    def paintEvent(self,event):
        super().paintEvent(event);pix=self.pixmap()
        if pix is None or pix.isNull():return
        w,h=pix.width()/pix.devicePixelRatio(),pix.height()/pix.devicePixelRatio();x=(self.width()-w)/2;y=(self.height()-h)/2
        cx,cy=self.bounds.center;sx,sy=self.bounds.span
        rect=QRectF(x+(1-cx-sx/2)*w,y+(cy-sy/2)*h,sx*w,sy*h)
        painter=QPainter(self);painter.setPen(QPen(QColor('#58e4aa'),3));painter.drawRect(rect)
        if self.hand is not None:
            hx,hy=self.hand;painter.setPen(QPen(QColor('#ffe079'),3));painter.drawEllipse(QRectF(x+(1-hx)*w-6,y+hy*h-6,12,12))

class WorkspaceEditor(QDialog):
    saved=Signal(object)
    def __init__(self,root,mapper,context_provider,clock=time.monotonic):
        super().__init__();self.root=Path(root);self.mapper=mapper;self.context_provider=context_provider;self.clock=clock;self.feature=None
        self.setWindowTitle('Рабочая область · курсор по кисти (2D)');self.resize(720,600)
        layout=QVBoxLayout(self)
        self.instructions=QLabel();self.instructions.setWordWrap(True);layout.addWidget(self.instructions)
        self.relative_checkbox=QCheckBox('Относительное наведение · без прыжка при возвращении руки');self.relative_checkbox.setChecked(mapper.relative);layout.addWidget(self.relative_checkbox)
        self.camera_preview=WorkspacePreview(mapper.bounds);layout.addWidget(self.camera_preview,1)
        self.width_slider=QSlider(Qt.Horizontal);self.width_slider.setRange(15,90);self.width_slider.setValue(round(mapper.bounds.span[0]*100))
        self.height_slider=QSlider(Qt.Horizontal);self.height_slider.setRange(15,90);self.height_slider.setValue(round(mapper.bounds.span[1]*100))
        form=QFormLayout();form.addRow('Ширина области, % кадра',self.width_slider);form.addRow('Высота области, % кадра',self.height_slider);layout.addLayout(form)
        self.width_slider.valueChanged.connect(self.update_bounds);self.height_slider.valueChanged.connect(self.update_bounds)
        self.relative_checkbox.toggled.connect(self.update_bounds)
        self.recenter_button=QPushButton('Сделать текущую кисть центром области');self.recenter_button.clicked.connect(self.recenter);layout.addWidget(self.recenter_button)
        self.status=QLabel('Ввод в ОС выключен. Настройка не требует девяти мишеней.');self.status.setWordWrap(True);layout.addWidget(self.status)
        buttons=QHBoxLayout();self.save_button=QPushButton('Сохранить настройки');self.save_button.clicked.connect(self.save_configuration);buttons.addWidget(self.save_button)
        cancel=QPushButton('Отмена');cancel.clicked.connect(self.reject);buttons.addWidget(cancel);layout.addLayout(buttons)
        self.update_instructions()
    def update_instructions(self):
        self.recenter_button.setEnabled(not self.mapper.relative)
        self.instructions.setText('Палец вверх — двигайте кисть для перемещения курсора. Опустите пальцы или уберите руку, верните её удобно и снова поднимите палец: курсор не прыгнет. Зелёная область показывает масштаб движения, а не края экрана. Меньшая область — выше чувствительность. Физический луч не используется.' if self.mapper.relative else 'Палец вверх, кисть внутри зелёной области — положение курсора на экране. Зелёная область соответствует всему экрану. Меньшая область делает курсор чувствительнее. Физический луч не используется.')
    def verify_context(self):
        try:
            if self.context_provider()!=self.mapper.context:raise ValueError('Изменились камера или экран.')
        except (ValueError,RuntimeError,TypeError,ImportError) as exc:
            self.status.setText(f'Контекст камеры/экрана недоступен: {exc}. Закройте настройку и откройте заново.');self.save_button.setEnabled(False);return False
        return True
    def update_bounds(self):
        span=np.array([self.width_slider.value(),self.height_slider.value()])/100
        center=np.clip(self.mapper.bounds.center,span/2,1-span/2)
        self.mapper=WorkspaceMapper(self.mapper.context,WorkspaceBounds(tuple(center),tuple(span)),relative=self.relative_checkbox.isChecked())
        self.camera_preview.bounds=self.mapper.bounds;self.camera_preview.update();self.update_instructions()
    def observe(self,feature):
        self.feature=feature
        if not self.verify_context():return
        try:self.camera_preview.hand=self.mapper.hand_point(feature);self.status.setText('Жёлтая точка — положение кисти. Зелёная область — масштаб движения. Ввод выключен.' if self.mapper.relative else 'Жёлтая точка — положение кисти. Зелёная область соответствует всему экрану. Ввод выключен.')
        except ValueError:self.camera_preview.hand=None;self.status.setText('Кисть не отслеживается. Держите её целиком в кадре, ладонью к камере.')
        self.camera_preview.update()
    def recenter(self):
        if self.mapper.relative:return
        if not self.verify_context():return
        try:
            if self.feature is None or self.clock()-self.feature.timestamp>.5:raise ValueError('Нужно свежее положение кисти.')
            center=self.mapper.hand_point(self.feature)
            span=np.minimum(self.mapper.bounds.span,2*np.minimum(center,1-center))
            span=np.floor(span*100+1e-9)/100
            if np.any(span<.15):raise ValueError('Кисть слишком близко к границе изображения. Переместите её немного к центру кадра.')
            self.mapper=WorkspaceMapper(self.mapper.context,WorkspaceBounds(tuple(center),tuple(span)),relative=self.relative_checkbox.isChecked())
            self.width_slider.blockSignals(True);self.height_slider.blockSignals(True)
            self.width_slider.setValue(round(span[0]*100));self.height_slider.setValue(round(span[1]*100))
            self.width_slider.blockSignals(False);self.height_slider.blockSignals(False)
            self.camera_preview.bounds=self.mapper.bounds;self.camera_preview.update();self.status.setText('Центр области обновлён. Нажмите «Сохранить настройки».')
        except ValueError as exc:self.status.setText(str(exc))
    def save_configuration(self):
        if not self.verify_context():return
        try:self.mapper.save(self.root/'data'/'workspace_config.json')
        except (OSError,ValueError,TypeError) as exc:self.status.setText(f'Не удалось сохранить настройки: {exc}');return
        self.saved.emit(self.mapper);self.accept()
