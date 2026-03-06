import numpy as np

def extract_features(hand_landmarks):
    coords = []

    wrist = hand_landmarks.landmark[0]

    for lm in hand_landmarks.landmark:
        coords.append(lm.x - wrist.x)
        coords.append(lm.y - wrist.y)
        coords.append(lm.z - wrist.z)

    coords = np.array(coords)

    max_val = np.max(np.abs(coords))
    if max_val != 0:
        coords = coords / max_val

    return coords