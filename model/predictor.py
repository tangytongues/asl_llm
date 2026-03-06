from joblib import load
import numpy as np

model = load("model/gesture_model.pkl")

# Relaxed thresholds
CONFIDENCE_THRESHOLD = 0.6
MARGIN_THRESHOLD = 0.05


def predict(features, return_conf: bool = False):
    """
    Predict gesture label for the given feature vector.
    If return_conf is True, also return the top class probability.
    """
    probs = model.predict_proba([features])[0]

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    # Get second highest probability
    sorted_probs = np.sort(probs)
    second_prob = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob

    if top_prob < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
        gesture = "UNKNOWN"
    else:
        gesture = model.classes_[top_idx]

    if return_conf:
        return gesture, top_prob
    return gesture