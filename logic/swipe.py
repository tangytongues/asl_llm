previous_x = None
SWIPE_THRESHOLD = 0.15

def detect_swipe(current_x):
    global previous_x

    if previous_x is None:
        previous_x = current_x
        return None

    dx = current_x - previous_x
    previous_x = current_x

    if dx > SWIPE_THRESHOLD:
        return "RIGHT"

    if dx < -SWIPE_THRESHOLD:
        return "LEFT"

    return None