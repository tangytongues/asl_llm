import cv2
import time
from collections import Counter

from vision.hand_tracker import get_hand_landmarks
from vision.feature_extractor import extract_features

from model.predictor import predict

from logic.swipe import detect_swipe

from ai.voice import speak
from ai.llm import ask_llm

from ui.hud import render_ui

from asl.alphabet_model import predict_letter
from asl.word_builder import WordBuilder


# ==============================
# SETTINGS
# ==============================

BUFFER_SIZE = 8
STABLE_RECENT_COUNT = 5

COOLDOWN = 3
CONFIRM_TIMEOUT = 5

SPEAKING_INDICATOR_DURATION = 0.5


# ==============================
# SYSTEM STATE
# ==============================

mode = "FAST"

prediction_buffer = []

state = "IDLE"
pending_action = None

last_trigger_time = 0
confirm_start_time = None

log_buffer = []

last_speak_end_time = 0

last_frame_time = time.time()
fps_value = 0.0


# Alphabet mode
builder = WordBuilder()


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

cv2.namedWindow("BOT1 - AI Assistant", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "BOT1 - AI Assistant",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


print("BOT1 system ready")
speak("System online. Awaiting gesture.")
last_speak_end_time = time.time()


# ==============================
# UTIL
# ==============================

def add_log(msg):
    log_buffer.append(msg)

    if len(log_buffer) > 5:
        log_buffer.pop(0)


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    result = get_hand_landmarks(frame)

    current_time = time.time()

    # ==============================
    # FPS
    # ==============================

    dt = current_time - last_frame_time
    last_frame_time = current_time

    if dt > 0:
        fps_value = 0.9 * fps_value + 0.1 * (1.0 / dt)


    gesture = "—"
    stable_gesture = "UNKNOWN"

    swipe_direction = None


    # ==============================
    # HAND DETECTED
    # ==============================

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        features = extract_features(hand)

        gesture = predict(features)

        prediction_buffer.append(gesture)

        if len(prediction_buffer) > BUFFER_SIZE:
            prediction_buffer.pop(0)

        # stabilization
        recent = prediction_buffer[-STABLE_RECENT_COUNT:]

        if len(set(recent)) == 1:
            stable_gesture = recent[0]


        # ==============================
        # MODE SWITCH
        # ==============================

        if stable_gesture == "MODE_SWITCH":

            if current_time - last_trigger_time > COOLDOWN:

                if mode == "FAST":
                    mode = "ALPHABET"
                    msg = "Alphabet mode activated"

                else:
                    mode = "FAST"
                    msg = "Fast gesture mode activated"

                speak(msg)
                add_log(msg)

                last_trigger_time = current_time


        # ==============================
        # MODE 1 : FAST COMMANDS
        # ==============================

        if mode == "FAST":

            if state == "IDLE":

                if stable_gesture != "UNKNOWN" and current_time - last_trigger_time > COOLDOWN:

                    if stable_gesture == "HELLO":

                        msg = "Hello. I am BOT1. How may I help you?"

                        speak(msg)

                        add_log(msg)

                        last_trigger_time = current_time
                        last_speak_end_time = time.time()


                    elif stable_gesture == "THANK_YOU":

                        msg = "You are welcome."

                        speak(msg)

                        add_log(msg)

                        last_trigger_time = current_time
                        last_speak_end_time = time.time()


                    elif stable_gesture == "HELP":

                        msg = ask_llm("User asks for help")

                        speak(msg)

                        add_log(msg)

                        last_trigger_time = current_time
                        last_speak_end_time = time.time()


                    elif stable_gesture in ["LIGHT_ON", "LIGHT_OFF"]:

                        pending_action = stable_gesture

                        state = "CONFIRM"

                        confirm_start_time = current_time

                        msg = f"Confirm {pending_action.replace('_',' ')}?"

                        speak(msg)

                        add_log(msg)

                        last_trigger_time = current_time


            elif state == "CONFIRM":

                wrist_x = hand.landmark[0].x

                swipe_direction = detect_swipe(wrist_x)


                if swipe_direction == "RIGHT":

                    msg = f"{pending_action} executed"

                    speak(msg)

                    add_log(msg)

                    state = "IDLE"
                    pending_action = None

                    last_trigger_time = current_time


                elif swipe_direction == "LEFT":

                    msg = "Action cancelled"

                    speak(msg)

                    add_log(msg)

                    state = "IDLE"
                    pending_action = None

                    last_trigger_time = current_time


                elif current_time - confirm_start_time > CONFIRM_TIMEOUT:

                    msg = "Confirmation timed out"

                    speak(msg)

                    add_log(msg)

                    state = "IDLE"
                    pending_action = None


        # ==============================
        # MODE 2 : ALPHABET
        # ==============================

        if mode == "ALPHABET":

            h, w, _ = frame.shape

            x_min = int(min(l.x for l in hand.landmark) * w)
            x_max = int(max(l.x for l in hand.landmark) * w)

            y_min = int(min(l.y for l in hand.landmark) * h)
            y_max = int(max(l.y for l in hand.landmark) * h)

            x_min = max(0, x_min - 20)
            y_min = max(0, y_min - 20)

            x_max = min(w, x_max + 20)
            y_max = min(h, y_max + 20)

            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size != 0:

                letter, conf = predict_letter(hand_crop)

                word, added = builder.update(letter)

                if added:
                    add_log("Typed: " + word)

                gesture = letter


    # ==============================
    # SPEAKING INDICATOR
    # ==============================

    speaking = (current_time - last_speak_end_time) < SPEAKING_INDICATOR_DURATION


    system_state = {

        "mode": mode,

        "state": state,

        "fps": fps_value,

        "confidence": None,

        "gesture": stable_gesture if mode == "FAST" else gesture,

        "pending_action": pending_action,

        "swipe_direction": swipe_direction,

        "logs": list(log_buffer),

        "speaking": speaking
    }


    render_ui(frame, system_state)

    cv2.imshow("BOT1 - AI Assistant", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()