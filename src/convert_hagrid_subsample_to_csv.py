import json
import csv
from pathlib import Path

INPUT_DIR = Path("external/hagrid_ann_subsample")
OUTPUT_CSV = Path("data/landmarks/hagrid_2d_with_no_gesture.csv")

POSITIVE_CLASS_MAP = {
    "palm": "palm",
    "fist": "fist",
    "like": "thumbs_up",
    "dislike": "thumbs_down",
    "peace": "victory",
}

NO_GESTURE_LIMIT = 140

def build_header():
    header = [
        "sample_id",
        "image_id",
        "label",
        "source_label",
        "subject_id",
        "session_id",
        "split",
        "handedness",
        "source",
        "image_path",
    ]
    for i in range(21):
        header += [f"x{i}", f"y{i}"]
    return header

def normalize_points(points):
    if not isinstance(points, list) or len(points) != 21:
        return None

    row = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return None
        x = float(pt[0])
        y = float(pt[1])
        row += [x, y]
    return row

def get_handedness(leading_hand, hand_idx):
    if isinstance(leading_hand, list) and hand_idx < len(leading_hand):
        return str(leading_hand[hand_idx])
    if isinstance(leading_hand, str):
        return leading_hand
    return "unknown"

def make_row(image_id, source_label, out_label, user_id, handedness, hand_idx, xy_row):
    sample_id = f"hagrid_{out_label}_{image_id}_{hand_idx}"
    row = [
        sample_id,
        image_id,
        out_label,
        source_label,
        user_id,
        "hagrid_subsample",
        "subsample",
        handedness,
        "HaGRID",
        "",
    ] + xy_row
    return row

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    no_gesture_count = 0

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(build_header())

        for source_label, out_label in POSITIVE_CLASS_MAP.items():
            json_path = INPUT_DIR / f"{source_label}.json"
            if not json_path.exists():
                print(f"SKIP: {json_path}")
                continue

            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            written_positive = 0
            written_no_gesture_from_this_file = 0

            for image_id, record in data.items():
                labels = record.get("labels", [])
                landmarks_all = record.get("landmarks", [])
                user_id = record.get("user_id", "unknown")
                leading_hand = record.get("leading_hand", "unknown")

                if not labels or not landmarks_all:
                    continue

                n = min(len(labels), len(landmarks_all))

                for hand_idx in range(n):
                    lbl = labels[hand_idx]
                    xy_row = normalize_points(landmarks_all[hand_idx])
                    if xy_row is None:
                        continue

                    handedness = get_handedness(leading_hand, hand_idx)

                    if lbl == source_label:
                        row = make_row(
                            image_id=image_id,
                            source_label=source_label,
                            out_label=out_label,
                            user_id=user_id,
                            handedness=handedness,
                            hand_idx=hand_idx,
                            xy_row=xy_row,
                        )
                        writer.writerow(row)
                        written_positive += 1
                        total_rows += 1

                    elif lbl == "no_gesture" and no_gesture_count < NO_GESTURE_LIMIT:
                        row = make_row(
                            image_id=image_id,
                            source_label=source_label,
                            out_label="no_gesture",
                            user_id=user_id,
                            handedness=handedness,
                            hand_idx=hand_idx,
                            xy_row=xy_row,
                        )
                        writer.writerow(row)
                        written_no_gesture_from_this_file += 1
                        no_gesture_count += 1
                        total_rows += 1

            print(f"{source_label}: positive={written_positive}, no_gesture_added={written_no_gesture_from_this_file}")

    print(f"no_gesture total: {no_gesture_count}")
    print(f"Готово. CSV: {OUTPUT_CSV}")
    print(f"Всего строк: {total_rows}")

if __name__ == "__main__":
    main()