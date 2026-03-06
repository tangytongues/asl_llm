"""
Minimal F.R.I.D.A.Y.-style HUD for BOT1.
Pure OpenCV rendering — no external GUI frameworks.
"""

import cv2
import time
import numpy as np

# -----------------------------------------------------------------------------
# Palette (BGR, low intensity)
# -----------------------------------------------------------------------------
BG_OVERLAY_ALPHA = 0.25
CYAN = (200, 220, 120)          # Soft cyan (toned down)
TEAL = (180, 180, 140)          # Muted teal
WHITE = (204, 204, 204)         # White at ~80%
DIM = (160, 160, 160)
LINE_COLOR = (80, 100, 100)     # Thin line accent
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.45
FONT_SCALE_MED = 0.55
FONT_THICK = 1


def _blend_overlay(frame, x1, y1, x2, y2, alpha=0.5, color=(0, 0, 0)):
    """Draw a semi-transparent overlay rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_background(frame):
    """
    Apply soft dark transparent overlay over the whole frame.
    Frame is the camera feed; we darken it slightly for a calm look.
    """
    h, w = frame.shape[:2]
    _blend_overlay(frame, 0, 0, w, h, alpha=BG_OVERLAY_ALPHA, color=(0, 0, 0))
    return frame


def draw_top_bar(frame):
    """Title area: BOT1 + subtitle, thin clean font, soft cyan."""
    h, w = frame.shape[:2]
    cx = w // 2

    # Subtle top bar strip
    bar_h = 72
    _blend_overlay(frame, 0, 0, w, bar_h, alpha=0.35, color=(10, 10, 10))

    # Title
    title = "BOT1"
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.9, 2)
    tx = cx - tw // 2
    cv2.putText(frame, title, (tx, 42), FONT, 0.9, CYAN, 2)

    # Subtitle
    sub = "AI Assistant"
    (sw, _), _ = cv2.getTextSize(sub, FONT, FONT_SCALE_SMALL, FONT_THICK)
    sx = cx - sw // 2
    cv2.putText(frame, sub, (sx, 62), FONT, FONT_SCALE_SMALL, TEAL, FONT_THICK)


def draw_left_panel(frame, mode, state, fps, confidence):
    """Left panel: MODE, STATE, Confidence %, FPS. Minimal text, thin lines only."""
    h, w = frame.shape[:2]
    x, y = 28, 100
    line_h = 22

    # Vertical line separator
    cv2.line(frame, (200, 80), (200, 220), LINE_COLOR, 1)

    cv2.putText(frame, "MODE", (x, y), FONT, FONT_SCALE_SMALL, DIM, FONT_THICK)
    cv2.putText(frame, str(mode), (x, y + line_h), FONT, FONT_SCALE_MED, WHITE, FONT_THICK)
    y += line_h + 14

    cv2.putText(frame, "STATE", (x, y), FONT, FONT_SCALE_SMALL, DIM, FONT_THICK)
    cv2.putText(frame, str(state), (x, y + line_h), FONT, FONT_SCALE_MED, WHITE, FONT_THICK)
    y += line_h + 14

    cv2.putText(frame, "CONFIDENCE", (x, y), FONT, FONT_SCALE_SMALL, DIM, FONT_THICK)
    conf_str = f"{confidence:.0%}" if confidence is not None else "—"
    cv2.putText(frame, conf_str, (x, y + line_h), FONT, FONT_SCALE_MED, WHITE, FONT_THICK)
    y += line_h + 14

    cv2.putText(frame, "FPS", (x, y), FONT, FONT_SCALE_SMALL, DIM, FONT_THICK)
    fps_str = f"{fps:.0f}" if fps is not None else "—"
    cv2.putText(frame, fps_str, (x, y + line_h), FONT, FONT_SCALE_MED, WHITE, FONT_THICK)


def draw_right_panel(frame, gesture, pending_action, swipe_direction=None):
    """Right panel: current gesture, pending action (if CONFIRM), swipe direction if active."""
    h, w = frame.shape[:2]
    x = w - 220
    y = 100
    line_h = 22

    # Vertical line separator
    cv2.line(frame, (w - 240, 80), (w - 240, 260), LINE_COLOR, 1)

    cv2.putText(frame, "GESTURE", (x, y), FONT, FONT_SCALE_SMALL, DIM, FONT_THICK)
    cv2.putText(frame, str(gesture), (x, y + line_h), FONT, FONT_SCALE_MED, CYAN, FONT_THICK)
    y += line_h + 14

    if pending_action:
        cv2.putText(frame, "PENDING", (x, y), FONT, FONT_SCALE_SMALL, DIM, FONT_THICK)
        cv2.putText(frame, pending_action.replace("_", " "), (x, y + line_h), FONT, FONT_SCALE_MED, WHITE, FONT_THICK)
        y += line_h + 14

    if swipe_direction:
        cv2.putText(frame, "SWIPE", (x, y), FONT, FONT_SCALE_SMALL, DIM, FONT_THICK)
        cv2.putText(frame, swipe_direction, (x, y + line_h), FONT, FONT_SCALE_MED, TEAL, FONT_THICK)


def draw_logs(frame, logs):
    """Bottom panel: last 5 system logs with subtle fade for older entries."""
    h, w = frame.shape[:2]
    log_h = 28
    max_logs = 5
    y_start = h - 20 - max_logs * log_h
    x = 24

    # Soft dark strip at bottom
    _blend_overlay(frame, 0, y_start - 10, w, h, alpha=0.5, color=(0, 0, 0))

    for i, line in enumerate(logs[-max_logs:]):
        alpha = 0.5 + 0.5 * (i + 1) / max_logs  # Older = more faded
        color = tuple(int(c * alpha) for c in DIM)
        cv2.putText(frame, line[:60], (x, y_start + i * log_h), FONT, FONT_SCALE_SMALL, color, FONT_THICK)


def draw_center_indicator(frame, state, speaking, gesture_detected=False):
    """
    Minimal circular ring: thin outline, soft glow, slight pulse when gesture
    detected, subtle rotating arc. No aggressive animation.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    r_base = 52
    t = time.time()

    # Pulse radius when gesture detected (subtle)
    if gesture_detected:
        pulse = 1.0 + 0.08 * np.sin(t * 4)
        r = int(r_base * pulse)
    else:
        r = r_base

    # Main ring (thin) with a soft inner highlight instead of a heavy glow copy
    cv2.circle(frame, (cx, cy), r, CYAN, 1)
    cv2.circle(frame, (cx, cy), max(r - 6, 4), (120, 130, 120), 1)

    # Rotating arc (slow, elegant) — 60° arc that rotates
    angle_start = (t * 0.3) % (2 * np.pi)
    angle_end = angle_start + np.pi / 3
    pts = []
    for a in np.linspace(angle_start, angle_end, 24):
        px = cx + (r - 2) * np.cos(a)
        py = cy - (r - 2) * np.sin(a)
        pts.append((int(px), int(py)))
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], TEAL, 1)

    # Speaking state: small label below ring
    if speaking:
        label = "Speaking..."
        (lw, _), _ = cv2.getTextSize(label, FONT, FONT_SCALE_SMALL, FONT_THICK)
        lx = cx - lw // 2
        cv2.putText(frame, label, (lx, cy + r + 28), FONT, FONT_SCALE_SMALL, TEAL, FONT_THICK)


def draw_speaking_waveform(frame, speaking):
    """Subtle waveform bars at bottom center when speaking."""
    if not speaking:
        return
    h, w = frame.shape[:2]
    cx, y = w // 2, h - 48
    t = time.time()
    n_bars = 5
    bar_w = 4
    gap = 6
    total_w = n_bars * bar_w + (n_bars - 1) * gap
    x0 = cx - total_w // 2
    for i in range(n_bars):
        height = int(8 + 6 * np.sin(t * 6 + i * 1.2))
        x = x0 + i * (bar_w + gap)
        cv2.rectangle(frame, (x, y - height), (x + bar_w, y), TEAL, -1)


def render_ui(frame, system_state_dict):
    """
    Single entry point: apply background then all panels and center indicator.
    system_state_dict: mode, state, fps, confidence, gesture, pending_action,
                       swipe_direction, logs, speaking
    """
    draw_background(frame)
    draw_top_bar(frame)

    mode = system_state_dict.get("mode", "FAST")
    state = system_state_dict.get("state", "IDLE")
    fps = system_state_dict.get("fps")
    confidence = system_state_dict.get("confidence")
    draw_left_panel(frame, mode, state, fps, confidence)

    gesture = system_state_dict.get("gesture", "—")
    pending_action = system_state_dict.get("pending_action")
    swipe_direction = system_state_dict.get("swipe_direction")
    draw_right_panel(frame, gesture, pending_action, swipe_direction)

    logs = system_state_dict.get("logs", [])
    draw_logs(frame, logs)

    speaking = system_state_dict.get("speaking", False)
    gesture_detected = (gesture != "UNKNOWN" and gesture != "—")
    draw_center_indicator(frame, state, speaking, gesture_detected)
    draw_speaking_waveform(frame, speaking)

    return frame
