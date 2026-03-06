import cv2
import os

DATASET_PATH = "custom_asl_dataset"
IMG_SIZE = 64
SAMPLES_PER_CLASS = 400

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

cap = cv2.VideoCapture(0)

for letter in letters:

    folder = os.path.join(DATASET_PATH, letter)
    os.makedirs(folder, exist_ok=True)

    print(f"\nCollecting images for: {letter}")
    print("Press SPACE to capture images")

    count = 0

    while count < SAMPLES_PER_CLASS:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # center crop box
        size = 300
        x1 = w//2 - size//2
        y1 = h//2 - size//2
        x2 = w//2 + size//2
        y2 = h//2 + size//2

        roi = frame[y1:y2, x1:x2]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"{letter} : {count}/{SAMPLES_PER_CLASS}",
                    (30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("ASL Dataset Collector", frame)

        key = cv2.waitKey(1)

        if key == 32:  # space key

            img = cv2.resize(roi,(IMG_SIZE,IMG_SIZE))

            path = os.path.join(folder,f"{count}.jpg")

            cv2.imwrite(path,img)

            count += 1

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()