"""
main.py — Phase 2 integration changes
======================================
Apply these changes to the existing main.py generated in Phase 1.
Every change is marked with ── PHASE 2 ADD ── or ── PHASE 2 CHANGE ──
so you can locate the exact insertion point in the file.

No existing lines need to be deleted — only additions and one update
to the gesture-dispatch block inside the main loop.
"""

# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — Imports  (add below: "import cv2")
# ════════════════════════════════════════════════════════════════════════

# ── PHASE 2 ADD ──
import cv2                          # already present
from hand_tracker import HandTracker          # already present
from gesture_controller import GestureController   # NEW
from utils import fingers_up, calculate_distance, midpoint  # NEW


# ════════════════════════════════════════════════════════════════════════
# SECTION 2 — Constants  (add after existing constants block)
# ════════════════════════════════════════════════════════════════════════

# ── PHASE 2 ADD ──
# Landmark IDs referenced directly in main for clarity
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12

# Overlay colours (BGR)
COLOR_MOVE   = (255, 255, 255)   # white  — cursor movement
COLOR_LCLICK = (0,   255, 80)    # green  — left click
COLOR_RCLICK = (80,  120, 255)   # blue   — right click
COLOR_SCROLL = (255, 200, 0)     # amber  — scroll mode


# ════════════════════════════════════════════════════════════════════════
# SECTION 3 — Initialization  (add inside main(), after tracker = HandTracker(...))
# ════════════════════════════════════════════════════════════════════════

# ── PHASE 2 ADD ──
controller = GestureController()
frame_h, frame_w = 0, 0    # will be set on first valid frame


# ════════════════════════════════════════════════════════════════════════
# SECTION 4 — Main loop gesture block
#
# REPLACE the existing block:
#
#     if landmark_list:
#         tip_id, tip_x, tip_y = landmark_list[FINGERTIP_ID]
#         print(f"[Fingertip #{tip_id}]  x={tip_x:>4}  y={tip_y:>4}")
#         highlight_fingertip(frame, tip_x, tip_y)
#         ...
#
# WITH the full block below.
# ════════════════════════════════════════════════════════════════════════

# ── PHASE 2 CHANGE ──
        if landmark_list:

            # ── Capture frame dimensions once ─────────────────────────
            frame_h, frame_w = frame.shape[:2]

            # ── Finger state detection ─────────────────────────────────
            # Returns [thumb, index, middle, ring, pinky] — 1=up, 0=down
            finger_states = fingers_up(landmark_list)

            # ── Landmark shortcuts ─────────────────────────────────────
            thumb_lm  = landmark_list[THUMB_TIP]   # (id, x, y)
            index_lm  = landmark_list[INDEX_TIP]
            middle_lm = landmark_list[MIDDLE_TIP]

            # ── Distance calculations for pinch detection ──────────────
            thumb_index_dist  = calculate_distance(thumb_lm,  index_lm)
            index_middle_dist = calculate_distance(index_lm,  middle_lm)

            # ── Gesture dispatch ───────────────────────────────────────
            gesture_label = ""

            # Gesture 1 ── Index up only → move cursor
            if finger_states == [0, 1, 0, 0, 0]:
                _, ix, iy = index_lm
                controller.move_cursor(ix, iy, frame_w, frame_h)
                highlight_fingertip(frame, ix, iy)
                gesture_label = "MOVE"
                cv2.putText(frame, gesture_label, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MOVE, 2)

            # Gesture 2 ── Thumb + Index pinch → left click
            elif controller.is_left_click_pinch(thumb_index_dist):
                fired = controller.left_click()
                gesture_label = "LEFT CLICK" if fired else "LEFT CLICK (cd)"
                mid = midpoint(thumb_lm, index_lm)
                if mid:
                    cv2.circle(frame, mid, 12, COLOR_LCLICK, cv2.FILLED)
                cv2.putText(frame, gesture_label, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LCLICK, 2)

            # Gesture 3 ── Index + Middle pinch → right click
            elif controller.is_right_click_pinch(index_middle_dist):
                fired = controller.right_click()
                gesture_label = "RIGHT CLICK" if fired else "RIGHT CLICK (cd)"
                mid = midpoint(index_lm, middle_lm)
                if mid:
                    cv2.circle(frame, mid, 12, COLOR_RCLICK, cv2.FILLED)
                cv2.putText(frame, gesture_label, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RCLICK, 2)

            # Gesture 4 ── Index + Middle both up → scroll
            elif finger_states[1] == 1 and finger_states[2] == 1:
                # Index higher than middle → scroll up; else scroll down
                direction = "up" if index_lm[2] < middle_lm[2] else "down"
                controller.scroll(direction)
                gesture_label = f"SCROLL {direction.upper()}"
                cv2.putText(frame, gesture_label, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SCROLL, 2)

            # ── Debug: print finger states to console ──────────────────
            print(
                f"[Fingers] T={finger_states[0]} I={finger_states[1]} "
                f"M={finger_states[2]} R={finger_states[3]} P={finger_states[4]}"
                f"  |  Gesture: {gesture_label or 'none'}"
            )


# ════════════════════════════════════════════════════════════════════════
# COMPLETE updated main.py for reference
# (copy-paste this if you prefer a clean file over a manual patch)
# ════════════════════════════════════════════════════════════════════════
COMPLETE_MAIN = '''
"""
main.py
-------
Phase 2 — Hand Gesture Navigation System.
Adds gesture recognition and system control on top of Phase 1 tracking.
"""

import time
import cv2
from hand_tracker import HandTracker
from gesture_controller import GestureController
from utils import fingers_up, calculate_distance, midpoint

# ── Configuration ─────────────────────────────────────────────────────
CAMERA_INDEX = 0
WINDOW_TITLE  = "Hand Gesture Navigation"
FPS_COLOR     = (0, 255, 255)
TIP_MARKER_COLOR  = (0, 0, 255)
TIP_MARKER_RADIUS = 10

THUMB_TIP  = 4
INDEX_TIP  = 8
MIDDLE_TIP = 12

COLOR_MOVE   = (255, 255, 255)
COLOR_LCLICK = (0,   255, 80)
COLOR_RCLICK = (80,  120, 255)
COLOR_SCROLL = (255, 200, 0)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, FPS_COLOR, 2)


def highlight_fingertip(frame, x, y):
    cv2.circle(frame, (x, y), TIP_MARKER_RADIUS, TIP_MARKER_COLOR, cv2.FILLED)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker    = HandTracker(max_hands=1, detection_confidence=0.75, tracking_confidence=0.75)
    controller = GestureController()
    prev_time  = 0.0

    print("[INFO] Phase 2 running — gesture control active. Press q to quit.")

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("[WARNING] Frame read failed. Retrying...")
            continue

        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame, draw=True)
        landmark_list = tracker.find_position(frame, hand_no=0)

        if landmark_list:
            frame_h, frame_w = frame.shape[:2]
            finger_states     = fingers_up(landmark_list)

            thumb_lm  = landmark_list[THUMB_TIP]
            index_lm  = landmark_list[INDEX_TIP]
            middle_lm = landmark_list[MIDDLE_TIP]

            thumb_index_dist  = calculate_distance(thumb_lm,  index_lm)
            index_middle_dist = calculate_distance(index_lm,  middle_lm)

            gesture_label = ""

            if finger_states == [0, 1, 0, 0, 0]:
                _, ix, iy = index_lm
                controller.move_cursor(ix, iy, frame_w, frame_h)
                highlight_fingertip(frame, ix, iy)
                gesture_label = "MOVE"
                cv2.putText(frame, gesture_label, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MOVE, 2)

            elif controller.is_left_click_pinch(thumb_index_dist):
                fired = controller.left_click()
                gesture_label = "LEFT CLICK" if fired else "LEFT CLICK (cd)"
                mid = midpoint(thumb_lm, index_lm)
                if mid:
                    cv2.circle(frame, mid, 12, COLOR_LCLICK, cv2.FILLED)
                cv2.putText(frame, gesture_label, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LCLICK, 2)

            elif controller.is_right_click_pinch(index_middle_dist):
                fired = controller.right_click()
                gesture_label = "RIGHT CLICK" if fired else "RIGHT CLICK (cd)"
                mid = midpoint(index_lm, middle_lm)
                if mid:
                    cv2.circle(frame, mid, 12, COLOR_RCLICK, cv2.FILLED)
                cv2.putText(frame, gesture_label, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RCLICK, 2)

            elif finger_states[1] == 1 and finger_states[2] == 1:
                direction = "up" if index_lm[2] < middle_lm[2] else "down"
                controller.scroll(direction)
                gesture_label = f"SCROLL {direction.upper()}"
                cv2.putText(frame, gesture_label, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SCROLL, 2)

            print(
                f"[Fingers] T={finger_states[0]} I={finger_states[1]} "
                f"M={finger_states[2]} R={finger_states[3]} P={finger_states[4]}"
                f"  |  Gesture: {gesture_label or \'none\'}"
            )

            label = tracker.get_hand_label(hand_no=0)
            if label:
                cv2.putText(frame, f"Hand: {label}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        draw_fps(frame, fps)

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\\n[INFO] Quit. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''
