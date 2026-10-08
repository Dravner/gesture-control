# Система жестового управления и исследование

Новый основной запуск: двойной клик по **Запустить систему.command** или `.venv/bin/python launch.py`. Окружение и модели подготовлены на этом MacBook.

Рабочая ветвь v4.3 использует RGB-камеру и относительное перемещение кисти. Указательный палец вверх, ладонь к камере: движение кисти двигает курсор. Уберите руку или расслабьте позу, верните её в удобное место и снова вытяните палец — курсор сохраняет свою позицию. Щипок фиксирует место клика, V-жест удерживает курсор и плавно прокручивает. Три поднятых пальца (указательный, средний, безымянный; мизинец согнут) дают свайп влево/вправо, без неподвижной паузы. Между свайпами расслабьте пальцы. Дополнительные датчики не нужны.

В «Способ наведения» выберите «Курсор движением кисти · 2D», затем в «Рабочая область · 2D» включите относительное перемещение и сохраните. Старые сохранённые абсолютные настройки автоматически не заменяются. Девять мишеней для этого режима не требуются. Физическое наведение осью пальца как лазером оставлено отдельным экспериментальным 3D-режимом: его точность на этой камере пока не подтверждена.

Подробная инструкция: [docs/run-guide.md](docs/run-guide.md). Системный ввод при запуске выключен. На видео и при записи шаблонов ввод отключён. Персональные «сall» и «switch», TRAIN/TRIAL и исследовательские модели сохранены в расширенном профиле; их живое качество не подтверждено.

В «Трекинг кисти» добавлен Apple Vision2D для macOS; MediaPipe сохранён. Выбор останавливает поток/ввод, глубина не выдумывается. Можно запустить явно: `.venv/bin/python launch.py --tracker apple_vision`. Аудит готовых решений, сравнения и графики: [reports/implementation-audit/README.md](reports/implementation-audit/README.md). Последний полный набор проверок:292 теста. Обновлённая работа обеими руками требует повторной живой проверки. Подробные изменения и измерения: [reports/bilateral-stability-20261008/README.md](reports/bilateral-stability-20261008/README.md).

Отдельные результаты новых экспериментов: [reports/pointing-v4/performance/README.md](reports/pointing-v4/performance/README.md). Статья временно не редактируется. Автоматические проверки и числовой replay проверяют программную логику, а не точность пяти жестов у пользователя.

Вся исходная демонстрация и её документация сохранены ниже. Её старые результаты HaGRID не являются оценкой новой потоковой системы.

# Gesture Control

Проект для распознавания жестов руки в реальном времени по изображению с веб-камеры. Видеопоток обрабатывается через MediaPipe Hand Landmarker, после чего координаты 21 ключевой точки кисти подаются в MLP-классификатор на PyTorch.

Основная демонстрация запускается из файла `src/realtime_inference.py` и распознает 6 классов:

- `fist` - кулак
- `no_gesture` - отсутствие целевого жеста
- `palm` - открытая ладонь
- `thumbs_down` - палец вниз
- `thumbs_up` - палец вверх
- `victory` - жест victory/peace

## Что делает проект

1. Открывает поток с камеры через OpenCV.
2. На каждом кадре находит кисть с помощью `hand_landmarker.task`.
3. Извлекает 21 точку кисти в формате `(x, y)`, всего 42 признака.
4. Нормализует координаты относительно запястья и масштаба кисти.
5. Приводит левую/правую руку к единому виду для более устойчивого распознавания.
6. Стандартизирует признаки через `mean` и `std`, сохраненные в чекпоинте модели.
7. Получает класс жеста с помощью MLP-модели.
8. Показывает результат, confidence и скелет кисти поверх изображения с камеры.

## Структура проекта

```text
.
|-- README.md
|-- data/
|   `-- landmarks/
|       |-- hagrid_6classes_landmarks.csv
|       |-- hagrid_2d_with_no_gesture.csv
|       |-- hagrid_subsample_landmarks.csv
|       `-- hagrid_with_no_gesture.csv
|-- external/
|   |-- hagrid/
|   |-- hagrid_ann_subsample/
|   `-- hagrid_6classes/
|-- models/
|   |-- hand_landmarker.task
|   |-- mlp_hagrid_6classes.pth
|   |-- mlp_hagrid_2d_no_gesture.pth
|   `-- mlp_hagrid_subsample.pth
|-- reports/
|   |-- classification_report_hagrid_6classes.txt
|   |-- classification_report_hagrid_2d_no_gesture.txt
|   |-- mlp_hagrid_subsample_classification_report.txt
|   |-- train_history_hagrid_6classes.csv
|   |-- train_history_hagrid_2d_no_gesture.csv
|   |-- confusion_matrix_hagrid_6classes.png
|   `-- confusion_matrix_hagrid_2d_no_gesture.png
|-- gesture_mlp_training.ipynb
`-- src/
    |-- capture_hand.py
    |-- convert_hagrid_subsample_to_csv.py
    |-- gesture_mlp_training.ipynb
    |-- realtime_inference.py
    `-- realtime_inference 2.py
```

## Основные файлы

`src/realtime_inference.py` - основной скрипт для демонстрации распознавания жестов с камеры. Использует модель `models/mlp_hagrid_6classes.pth`. Если она отсутствует, пытается загрузить fallback-модель `models/mlp_hagrid_2d_no_gesture.pth`.

`src/capture_hand.py` - простой скрипт для проверки камеры и MediaPipe. Он показывает скелет кисти и определение handedness без MLP-классификации жестов.

`src/convert_hagrid_subsample_to_csv.py` - конвертер JSON-разметки HaGRID из `external/hagrid_ann_subsample` в CSV с 2D landmarks. Результат сохраняется в `data/landmarks/hagrid_2d_with_no_gesture.csv`.

`gesture_mlp_training.ipynb` и `src/gesture_mlp_training.ipynb` - ноутбуки для подготовки признаков, обучения MLP, оценки качества и сохранения моделей/отчетов.

`models/hand_landmarker.task` - модель MediaPipe для поиска ключевых точек руки.

`models/mlp_hagrid_6classes.pth` - основная обученная PyTorch-модель для 6 классов.

## Требования

Проект проверялся в локальном виртуальном окружении `.venv`.

Минимальные зависимости для запуска распознавания:

```bash
pip install opencv-python mediapipe numpy torch
```

Дополнительные зависимости для обучения, анализа датасета и ноутбуков:

```bash
pip install pandas scikit-learn matplotlib jupyter
```

Если окружение создается заново:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install opencv-python mediapipe numpy torch pandas scikit-learn matplotlib jupyter
```

На macOS при первом запуске может появиться системный запрос на доступ к камере для Terminal/Python. Его нужно разрешить, иначе `cv2.VideoCapture(0)` не сможет открыть видеопоток.

## Быстрый запуск

Из корня проекта:

```bash
.venv/bin/python src/realtime_inference.py
```

Если появляются предупреждения про недоступный кэш Matplotlib/fontconfig, можно указать временную директорию для кэша:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache .venv/bin/python src/realtime_inference.py
```

После запуска откроется окно `Real-time Gesture Recognition`.

Управление:

- покажите жест перед камерой;
- смотрите текущий класс в поле `Gesture`;
- смотрите уверенность модели в поле `Confidence`;
- нажмите `ESC`, чтобы закрыть программу.

## Проверка MediaPipe без классификатора

Если нужно отдельно проверить, видит ли камера руку и строится ли скелет кисти:

```bash
.venv/bin/python src/capture_hand.py
```

Окно называется `Hand Capture`. Для выхода также используется `ESC`.

## Как устроено realtime-распознавание

В `src/realtime_inference.py` используются следующие настройки:

```python
CONF_THRESHOLD = 0.75
SHOW_SKELETON = True
MIRROR_CAMERA_PREVIEW = True
HAND_LANDMARKER_DELEGATE = "CPU"
DEBOUNCE_WINDOW = 5
DEBOUNCE_MIN_COUNT = 4
```

Смысл параметров:

- `CONF_THRESHOLD` - минимальная уверенность модели. Если confidence ниже порога, результат показывается как `uncertain`.
- `SHOW_SKELETON` - включает отрисовку точек и линий кисти.
- `MIRROR_CAMERA_PREVIEW` - зеркалит изображение, чтобы превью вело себя как обычная фронтальная камера.
- `HAND_LANDMARKER_DELEGATE` - backend для MediaPipe Tasks. Сейчас используется `CPU`.
- `DEBOUNCE_WINDOW` - размер окна последних предсказаний.
- `DEBOUNCE_MIN_COUNT` - сколько одинаковых предсказаний нужно внутри окна, чтобы жест считался стабильным.

Debounce нужен, чтобы подпись жеста не дергалась на каждом кадре из-за единичных неуверенных предсказаний.

## Признаки и модель

Основная модель: `models/mlp_hagrid_6classes.pth`.

Метаданные чекпоинта:

- `feature_mode`: `xy_only_normalized`
- входная размерность: `42`
- признаки: 21 точка кисти, каждая точка хранит `x` и `y`
- архитектура MLP: `256 -> 128 -> 64 -> 6`
- dropout: `0.3`
- классы: `fist`, `no_gesture`, `palm`, `thumbs_down`, `thumbs_up`, `victory`
- исходный CSV: `data/landmarks/hagrid_6classes_landmarks.csv`
- исходный датасет: `external/hagrid_6classes`

Перед подачей в модель координаты нормализуются:

1. Все точки сдвигаются так, чтобы запястье `landmark[0]` стало началом координат.
2. Для левой руки координата `x` отражается, чтобы левая и правая рука имели сопоставимое представление.
3. Масштаб вычисляется по расстоянию до `landmark[9]`; если оно слишком мало, используется `landmark[5]`.
4. Вектор признаков делится на масштаб кисти.
5. Признаки стандартизируются через `mean` и `std` из чекпоинта.

## Данные

Основные CSV-файлы с landmarks лежат в `data/landmarks`.

`hagrid_6classes_landmarks.csv` - основной набор для модели на 6 классов. В файле 137045 строк с учетом заголовка.

`hagrid_2d_with_no_gesture.csv` - компактный 2D-набор с классом `no_gesture`. В файле 623 строки с учетом заголовка.

`hagrid_subsample_landmarks.csv` - ранний subsample-набор. В файле 483 строки с учетом заголовка.

`hagrid_with_no_gesture.csv` - набор landmarks с добавленным классом отсутствия жеста. В файле 623 строки с учетом заголовка.

Формат основного CSV:

- служебные поля: `sample_id`, `image_id`, `label`, `source_label`, `hand_idx`, `handedness`, `handedness_score`, `image_width`, `image_height`, `source`, `image_path`;
- координаты landmarks: `x0`, `y0`, ..., `x20`, `y20`.

## Обучение модели

Основной сценарий обучения находится в ноутбуках:

```bash
jupyter notebook gesture_mlp_training.ipynb
```

или:

```bash
jupyter notebook src/gesture_mlp_training.ipynb
```

По метаданным сохраненной модели `mlp_hagrid_6classes.pth` использовались настройки:

- `random_state`: `42`
- `test_size`: `0.2`
- `batch_size`: `256`
- `epochs`: `30`
- `learning_rate`: `1e-05`
- `weight_decay`: `0.001`
- balanced class weights: включены
- `no_gesture_multiplier`: `1.5`
- `max_images_per_class`: `25000`

После обучения сохраняются:

- модель в `models/*.pth`;
- история обучения в `reports/train_history_*.csv`;
- classification report в `reports/*classification_report*.txt`;
- confusion matrix в `reports/confusion_matrix_*.png`.

## Результаты

Для основной модели `mlp_hagrid_6classes.pth`:

- test accuracy: `0.987194`
- test macro F1: `0.986987`
- размер test split: `27409` примеров

Качество по классам из `reports/classification_report_hagrid_6classes.txt`:

| Класс | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `fist` | 0.9941 | 0.9906 | 0.9923 | 4558 |
| `no_gesture` | 0.9621 | 0.9751 | 0.9685 | 4291 |
| `palm` | 0.9868 | 0.9975 | 0.9921 | 4866 |
| `thumbs_down` | 0.9962 | 0.9818 | 0.9889 | 4548 |
| `thumbs_up` | 0.9877 | 0.9877 | 0.9877 | 4642 |
| `victory` | 0.9955 | 0.9891 | 0.9923 | 4504 |

Результаты старых/альтернативных моделей:

- `mlp_hagrid_2d_no_gesture.pth`: accuracy `1.000000`, macro F1 `1.000000` на компактном test split из 125 примеров.
- `mlp_hagrid_subsample.pth`: accuracy `0.952000` на test split из 125 примеров.

## Конвертация HaGRID-разметки

Скрипт `src/convert_hagrid_subsample_to_csv.py` ожидает JSON-файлы в директории:

```text
external/hagrid_ann_subsample/
```

Основные исходные классы преобразуются так:

| Исходный класс HaGRID | Класс в проекте |
| --- | --- |
| `palm` | `palm` |
| `fist` | `fist` |
| `like` | `thumbs_up` |
| `dislike` | `thumbs_down` |
| `peace` | `victory` |

Строки с исходной меткой `no_gesture` добавляются отдельно из тех же JSON-файлов, пока не достигнут лимит `NO_GESTURE_LIMIT = 140`.

Запуск:

```bash
.venv/bin/python src/convert_hagrid_subsample_to_csv.py
```

Результат:

```text
data/landmarks/hagrid_2d_with_no_gesture.csv
```

## Типичные проблемы

### Не открывается камера

Проверьте, что камера не занята другим приложением. На macOS также проверьте разрешения:

```text
System Settings -> Privacy & Security -> Camera
```

Разрешение должно быть выдано приложению, из которого запускается Python.

### Ошибка `Не удалось открыть камеру`

Скрипт использует камеру с индексом `0`:

```python
cap = cv2.VideoCapture(0)
```

Если используется внешняя камера, попробуйте заменить индекс на `1` или `2`.

### Ошибка `Не найден файл модели`

Проверьте, что в проекте есть файлы:

```text
models/mlp_hagrid_6classes.pth
models/mlp_hagrid_2d_no_gesture.pth
models/hand_landmarker.task
```

`src/realtime_inference.py` вычисляет корень проекта относительно своего расположения, поэтому запускать его можно из корня проекта без правки абсолютных путей.

### Медленный первый запуск

При первом импорте библиотек может строиться кэш шрифтов Matplotlib/fontconfig. Если домашняя директория недоступна для записи, используйте:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache .venv/bin/python src/realtime_inference.py
```

### Жест часто становится `uncertain`

Возможные причины:

- confidence ниже `CONF_THRESHOLD = 0.75`;
- рука частично выходит из кадра;
- жест сильно отличается от примеров из HaGRID;
- плохое освещение;
- MediaPipe неправильно определяет landmarks.

Для диагностики сначала запустите `src/capture_hand.py` и проверьте, стабильно ли строится скелет кисти.

## Краткая команда для демонстрации

```bash
cd /Users/a1111/schcool/diplom/gesture-control
MPLCONFIGDIR=/private/tmp/matplotlib-cache .venv/bin/python src/realtime_inference.py
```

После запуска покажите один из поддерживаемых жестов перед камерой. Для завершения нажмите `ESC`.

Исправление зависаний от08.10.2026: устранено удержание RGBA-буферов Apple Vision, тестовое окно показывает потерю кадров и отключает ввод, автоматический fullscreen убран. [Диагностика и измерения памяти](reports/runtime-freeze-20261008/README.md). Проверка собственных тестов: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests` —264passed. Результат пользовательской проверки жестов фиксируется отдельно.

Обновление тестовой сцены и разделение поз: [reports/scroll-navigation-20261008/README.md](reports/scroll-navigation-20261008/README.md).

Последняя сборка с сохранением начала движения при переходеscroll→swipe: [reports/scroll-navigation-20261009/README.md](reports/scroll-navigation-20261009/README.md).
