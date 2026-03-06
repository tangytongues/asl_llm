import numpy as np
import cv2
from tensorflow.keras.models import load_model  # type: ignore[import]

IMG_SIZE = 64

# same label order used during training
LABELS = [
"A","B","C","D","E","F","G","H","I","J",
"K","L","M","N","O","P","Q","R","S","T",
"U","V","W","X","Y","Z","del","nothing","space"
]

model = load_model("models/custom_asl_model.h5")


def predict_letter(hand_image):
    """
    hand_image: cropped BGR image from camera
    returns: (letter, confidence)
    """

    img = cv2.resize(hand_image, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])

    letter = LABELS[idx]

    return letter, confidence