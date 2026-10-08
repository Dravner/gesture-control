import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mlp_hagrid_subsample.pth"
TASK_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

CONF_THRESHOLD = 0.75
SHOW_SKELETON = True

# MediaPipe Tasks
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
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def load_checkpoint(model_path: Path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    label_classes = checkpoint["label_classes"]
    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)

    model = MLP(input_dim=len(mean), num_classes=len(label_classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, label_classes, mean, std

def extract_features(result):
    if not result.hand_landmarks:
        return None, None, None

    hand_landmarks = result.hand_landmarks[0]

    features = []
    points_xy = []

    for lm in hand_landmarks:
        x = float(lm.x)
        y = float(lm.y)
        z = float(getattr(lm, "z", 0.0))
        features.extend([x, y, z])
        points_xy.append((x, y))

    handedness = "unknown"
    if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
        handedness = result.handedness[0][0].category_name

    return np.array(features, dtype=np.float32), points_xy, handedness

def predict(model, features, mean, std, label_classes):
    x = (features - mean) / std
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = label_classes[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs

def draw_hand(frame, points_xy, handedness, pred_label=None, confidence=None):
    h, w, _ = frame.shape
    pts = [(int(x * w), int(y * h)) for x, y in points_xy]

    if SHOW_SKELETON:
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 0, 0), 2)

    text_y = 30
    cv2.putText(frame, f"Hand: {handedness}", (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    text_y += 35

    if pred_label is not None and confidence is not None:
        cv2.putText(frame, f"Pred: {pred_label}", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        text_y += 35
        cv2.putText(frame, f"Conf: {confidence:.3f}", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH}")
    if not TASK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден файл hand landmarker: {TASK_MODEL_PATH}")

    model, label_classes, mean, std = load_checkpoint(MODEL_PATH)
    print("Loaded classes:", label_classes)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK_MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    last_ts = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            now_ts = int(time.time() * 1000)
            timestamp_ms = max(now_ts, last_ts + 1)
            last_ts = timestamp_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            features, points_xy, handedness = extract_features(result)

            if features is None:
                cv2.putText(frame, "No hand", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                pred_label, confidence, _ = predict(model, features, mean, std, label_classes)

                if confidence < CONF_THRESHOLD:
                    pred_label = "uncertain"

                frame = draw_hand(frame, points_xy, handedness, pred_label, confidence)

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
