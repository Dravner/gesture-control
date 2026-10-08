from collections import deque
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mlp_hagrid_6classes.pth"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "mlp_hagrid_2d_no_gesture.pth"
TASK_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

CONF_THRESHOLD = 0.75
SHOW_SKELETON = True
MIRROR_CAMERA_PREVIEW = True
HAND_LANDMARKER_DELEGATE = "CPU"
DEBOUNCE_WINDOW = 5
DEBOUNCE_MIN_COUNT = 4
OVERLAY_MARGIN = 10

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(128, 64), dropout=0.20):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def infer_hidden_dims_from_state_dict(state_dict):
    linear_layers = []
    for key, tensor in state_dict.items():
        if key.startswith("net.") and key.endswith(".weight") and tensor.ndim == 2:
            linear_layers.append((int(key.split(".")[1]), tensor.shape[0]))

    linear_layers.sort(key=lambda item: item[0])
    if len(linear_layers) <= 1:
        return []
    return [out_features for _, out_features in linear_layers[:-1]]

def resolve_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH
    if FALLBACK_MODEL_PATH.exists():
        print(f"New model is not found yet, using fallback model: {FALLBACK_MODEL_PATH}")
        return FALLBACK_MODEL_PATH
    raise FileNotFoundError(
        f"Не найден файл модели. Ожидались: {MODEL_PATH} или {FALLBACK_MODEL_PATH}"
    )

def load_checkpoint(model_path: Path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    label_classes = checkpoint["label_classes"]
    feature_mode = checkpoint.get("feature_mode", "xy_only_normalized")
    if feature_mode != "xy_only_normalized":
        raise ValueError(f"Неподдерживаемый feature_mode в checkpoint: {feature_mode}")

    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    if len(mean) != 42:
        raise ValueError(f"Ожидалось 42 признака 2D landmarks, получено: {len(mean)}")

    state_dict = checkpoint["model_state_dict"]
    model_config = checkpoint.get("model_config", {})
    hidden_dims = tuple(model_config.get("hidden_dims") or infer_hidden_dims_from_state_dict(state_dict))
    dropout = float(model_config.get("dropout", 0.20))

    model = MLP(
        input_dim=len(mean),
        num_classes=len(label_classes),
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, label_classes, mean, std

def invert_handedness(handedness):
    if not isinstance(handedness, str):
        return handedness

    lowered = handedness.lower()
    if lowered == "left":
        return "Right"
    if lowered == "right":
        return "Left"
    return handedness

def extract_features(result):
    if not result.hand_landmarks:
        return None, None, None, None

    hand_landmarks = result.hand_landmarks[0]

    features = []
    points_xy = []

    for lm in hand_landmarks:
        x = float(lm.x)
        y = float(lm.y)
        features.extend([x, y])
        points_xy.append((x, y))

    handedness = "unknown"
    if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
        handedness = result.handedness[0][0].category_name

    display_handedness = invert_handedness(handedness) if MIRROR_CAMERA_PREVIEW else handedness

    return np.array(features, dtype=np.float32), points_xy, handedness, display_handedness

def preprocess_landmarks_xy(flat_xy, handedness):
    pts = np.array(flat_xy, dtype=np.float32).reshape(21, 2)

    wrist = pts[0].copy()
    pts = pts - wrist

    if isinstance(handedness, str) and handedness.lower() == "left":
        pts[:, 0] = -pts[:, 0]

    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = np.linalg.norm(pts[5])
    if scale < 1e-6:
        scale = 1.0

    pts = pts / scale
    return pts.reshape(-1)

def predict(model, features, handedness, mean, std, label_classes):
    x = preprocess_landmarks_xy(features, handedness)
    x = (x - mean) / std
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = label_classes[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence

def draw_top_left_overlay(frame, gesture_label, confidence):
    label = gesture_label if gesture_label else "none"
    confidence_text = "n/a" if confidence is None else f"{confidence * 100:.1f}%"
    lines = [
        f"Gesture: {label}",
        f"Confidence: {confidence_text}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    line_height = 32
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_w = max(width for width, _ in sizes) + 24
    box_h = line_height * len(lines) + 18

    x0 = OVERLAY_MARGIN
    y0 = OVERLAY_MARGIN
    x1 = x0 + box_w
    y1 = y0 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 1)

    for idx, line in enumerate(lines):
        y = y0 + 31 + idx * line_height
        cv2.putText(frame, line, (x0 + 12, y), font, font_scale, (255, 255, 255), thickness)

def draw_hand(frame, points_xy):
    h, w, _ = frame.shape
    pts = [(int(x * w), int(y * h)) for x, y in points_xy]

    if SHOW_SKELETON:
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 0, 0), 2)

    return frame

def main():
    if not TASK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден файл hand landmarker: {TASK_MODEL_PATH}")

    model_path = resolve_model_path()
    model, label_classes, mean, std = load_checkpoint(model_path)
    print("Loaded model:", model_path)
    print("Loaded classes:", label_classes)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(TASK_MODEL_PATH),
            delegate=BaseOptions.Delegate[HAND_LANDMARKER_DELEGATE.upper()],
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    last_ts = 0
    pred_buffer = deque(maxlen=DEBOUNCE_WINDOW)
    stable_label = "none"

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if MIRROR_CAMERA_PREVIEW:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            now_ts = int(time.time() * 1000)
            timestamp_ms = max(now_ts, last_ts + 1)
            last_ts = timestamp_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            features, points_xy, handedness, display_handedness = extract_features(result)

            if features is None:
                pred_buffer.clear()
                stable_label = "none"
                draw_top_left_overlay(frame, "No hand", None)
            else:
                pred_label, confidence = predict(model, features, handedness, mean, std, label_classes)

                if confidence < CONF_THRESHOLD:
                    current_label = "uncertain"
                else:
                    current_label = pred_label

                pred_buffer.append(current_label)

                if len(pred_buffer) == DEBOUNCE_WINDOW:
                    counts = {}
                    for x in pred_buffer:
                        counts[x] = counts.get(x, 0) + 1

                    best_label = max(counts, key=counts.get)
                    if counts[best_label] >= DEBOUNCE_MIN_COUNT:
                        stable_label = best_label

                frame = draw_hand(frame, points_xy)
                draw_top_left_overlay(frame, stable_label, confidence)
                cv2.putText(frame, f"Hand: {display_handedness}", (10, 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.putText(frame, "ESC - exit", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Real-time Gesture Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
