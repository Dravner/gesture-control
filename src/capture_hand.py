import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/hand_landmarker.task"
MIRROR_CAMERA_PREVIEW = True

def invert_handedness(handedness):
    lowered = handedness.lower()
    if lowered == "left":
        return "Right"
    if lowered == "right":
        return "Left"
    return handedness

def draw_hand(frame, result):
    if not result.hand_landmarks:
        return frame

    h, w, _ = frame.shape

    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Простейшая отрисовка связей
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]
        for a, b in connections:
            cv2.line(frame, points[a], points[b], (255, 0, 0), 2)

        if result.handedness and hand_idx < len(result.handedness):
            category = result.handedness[hand_idx][0]
            display_name = invert_handedness(category.category_name) if MIRROR_CAMERA_PREVIEW else category.category_name
            label = f"{display_name}: {category.score:.2f}"
            x0, y0 = points[0]
            cv2.putText(
                frame,
                label,
                (x0 + 10, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if MIRROR_CAMERA_PREVIEW:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame = draw_hand(frame, result)

            cv2.imshow("Hand Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
