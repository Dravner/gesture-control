"""Native tuning panel; saving never enables OS input."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QWidget,QTabWidget,QDoubleSpinBox,QCheckBox,QComboBox,QLineEdit,QLabel,QPushButton,QHBoxLayout
from .interaction_settings import InteractionSettings,BOUNDS
from .profiles import KEYS

class SettingsDialog(QDialog):
    custom_requested=Signal()
    def __init__(self,settings,gestures,apply_callback,parent=None):
        super().__init__(parent);self.setWindowTitle('Настройки жестового управления');self.resize(640,560)
        self.apply_callback=apply_callback;self.controls={};self.mapping_actions={};self.mapping_keys={}
        layout=QVBoxLayout(self);tabs=QTabWidget();layout.addWidget(tabs)
        labels={'pointer_sensitivity':'Чувствительность курсора ×','smoothing':'Сглаживание (0 — быстро, 100 — плавно)', 'scroll_gain':'Чувствительность прокрутки ×','pose_confirm_ms':'Подтверждение позы, мс','pointer_acquire_ms':'Включение наведения, мс','swipe_distance':'Длина свайпа (доля ширины кадра)','pinch_press':'Порог щипка (доля размера ладони)','drag_hold_ms':'Удержание для переноса, мс'}
        for title,names in [('Курсор и прокрутка',['pointer_sensitivity','smoothing','scroll_gain']),('Распознавание',['pose_confirm_ms','pointer_acquire_ms','swipe_distance','pinch_press','drag_hold_ms'])]:
            panel=QWidget();form=QFormLayout(panel);tabs.addTab(panel,title)
            for name in names:
                spin=QDoubleSpinBox();spin.setRange(*BOUNDS[name]);spin.setDecimals(2 if BOUNDS[name][1]<10 else 0);spin.setSingleStep(.01 if name in {'swipe_distance','pinch_press'} else .1 if BOUNDS[name][1]<10 else 10);spin.setValue(getattr(settings,name));self.controls[name]=spin;form.addRow(labels[name],spin)
            if title=='Распознавание':
                self.allow_drag=QCheckBox('Перетаскивание при удержании щипка');self.allow_drag.setChecked(settings.allow_drag);form.addRow(self.allow_drag)
                note=QLabel('Меньшее время подтверждения ускоряет переключение, но повышает риск случайных жестов.\nРазмыкание щипка: порог закрытия + 0,10.');note.setWordWrap(True);form.addRow(note)
        panel=QWidget();form=QFormLayout(panel);tabs.addTab(panel,'Действия')
        action_labels={'move':'Курсор','scroll':'Прокрутка','click':'Левый клик','right_click':'Правый клик','hotkey':'Горячие клавиши','none':'Выключено'}
        choices={'point':['move','none'],'victory':['scroll','none'],'pinch':['click','right_click','hotkey','none'],'swipe_left':['hotkey','click','right_click','none'],'swipe_right':['hotkey','click','right_click','none']}
        for key,actions in choices.items():
            g=next(g for g in gestures if g['id']=='builtin:'+key);row=QWidget();bar=QHBoxLayout(row);bar.setContentsMargins(0,0,0,0);combo=QComboBox()
            for action in actions:combo.addItem(action_labels[action],action)
            if combo.findData(g['action'])<0:combo.addItem('Сохранённое действие: '+g['action'],g['action'])
            combo.setCurrentIndex(combo.findData(g['action']));edit=QLineEdit(', '.join(g.get('keys',[])));edit.setPlaceholderText('command, [');bar.addWidget(combo);bar.addWidget(edit);form.addRow(g['name'],row);self.mapping_actions[key]=combo;self.mapping_keys[key]=edit
        note=QLabel('Клавиши через запятую: command, shift, control, alt, буквы, цифры, стрелки, [ и ].\nДля пользовательских жестов откройте мастер и запишите отдельные обучающие и проверочные повторы.');note.setWordWrap(True);form.addRow(note)
        custom=QPushButton('Открыть мастер своих жестов');custom.clicked.connect(self.open_custom);form.addRow(custom)
        self.error=QLabel();self.error.setWordWrap(True);layout.addWidget(self.error)
        buttons=QHBoxLayout();reset=QPushButton('Сбросить параметры');reset.clicked.connect(self.reset_controls);buttons.addWidget(reset);buttons.addStretch();cancel=QPushButton('Отмена');cancel.clicked.connect(self.reject);buttons.addWidget(cancel);self.save_button=QPushButton('Сохранить');self.save_button.clicked.connect(self.save);buttons.addWidget(self.save_button);layout.addLayout(buttons)
    def reset_controls(self):
        defaults=InteractionSettings()
        for name,spin in self.controls.items():spin.setValue(getattr(defaults,name))
        self.allow_drag.setChecked(defaults.allow_drag)
    def open_custom(self):self.reject();self.custom_requested.emit()
    def save(self):
        try:
            settings=InteractionSettings(**{name:spin.value() for name,spin in self.controls.items()},allow_drag=self.allow_drag.isChecked());bindings={}
            for key,combo in self.mapping_actions.items():
                action=combo.currentData();keys=[k.strip().lower() for k in self.mapping_keys[key].text().split(',') if k.strip()]
                if any(k not in KEYS for k in keys):raise ValueError('Неподдерживаемая клавиша: '+key)
                if action=='hotkey' and not keys:raise ValueError('Укажите клавиши: '+key)
                bindings[key]={'action':action,'keys':keys if action=='hotkey' else []}
            self.apply_callback(settings,bindings)
        except (ValueError,OSError,StopIteration) as exc:self.error.setText(str(exc));return
        self.accept()
