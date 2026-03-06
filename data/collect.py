import cv2
import csv
import os

from vision.hand_tracker import get_hand_landmarks
from vision.feature_extractor import extract_features

# -----------------------
LABEL = "HELP"   # CHANGE THIS EACH TIME
SAMPLES = 300
# -----------------------

output_dir = "data/dataset"
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(output_dir, f"{LABEL}.csv")

cap = cv2.VideoCapture(0)

collected = 0

print(f"Collecting data for {LABEL}")
print("Press 's' to start collecting...")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    result = get_hand_landmarks(frame)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        features = extract_features(hand)

        if collected < SAMPLES:
            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(features)

            collected += 1

        cv2.putText(frame, f"{LABEL}: {collected}/{SAMPLES}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) == 27 or collected >= SAMPLES:
        break

cap.release()
cv2.destroyAllWindows()

print("Collection complete.")