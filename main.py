"""
main.py
-------
Entry point for Phase 1 of the Hand Gesture Navigation System.

What this does:
  - Captures live webcam feed via OpenCV.
  - Passes each frame to HandTracker for real-time hand landmark detection.
  - Extracts and prints the index fingertip position (landmark 8).
  - Displays the annotated feed with an FPS counter.
  - Press 'q' to quit.

Phase 2 will extend this with gesture classification and mouse/keyboard control.
"""

import time

import cv2

from hand_tracker import HandTracker

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CAMERA_INDEX = 0          # 0 = default webcam; change if using an external camera
WINDOW_TITLE = "Hand Gesture Navigation"
FINGERTIP_ID = 8          # MediaPipe landmark 8 = index finger tip
FPS_FONT_SCALE = 0.8
FPS_COLOR = (0, 255, 255)     # Yellow
TIP_MARKER_COLOR = (0, 0, 255)  # Red circle on fingertip
TIP_MARKER_RADIUS = 10


def draw_fps(frame, fps: float) -> None:
    """Overlay a real-time FPS counter in the top-left corner of the frame."""
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        FPS_FONT_SCALE,
        FPS_COLOR,
        2,
    )


def highlight_fingertip(frame, x: int, y: int) -> None:
    """Draw a visible red circle on the index fingertip position."""
    cv2.circle(frame, (x, y), TIP_MARKER_RADIUS, TIP_MARKER_COLOR, cv2.FILLED)


def main():
    # ── Webcam setup ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check CAMERA_INDEX or permissions.")
        return

    # Optional: request a higher resolution (device must support it)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── HandTracker setup ─────────────────────────────────────────────
    tracker = HandTracker(
        max_hands=1,               # Track one hand in Phase 1
        detection_confidence=0.75,
        tracking_confidence=0.75,
    )

    # FPS calculation variables
    prev_time = 0.0

    print("[INFO] Hand Gesture Navigation System — Phase 1 running.")
    print("[INFO] Press 'q' to quit.\n")

    # ── Main loop ─────────────────────────────────────────────────────
    while True:
        success, frame = cap.read()

        # Handle dropped or failed frames gracefully
        if not success or frame is None:
            print("[WARNING] Failed to read frame from webcam. Retrying...")
            continue

        # Flip horizontally so the feed acts like a mirror (more intuitive)
        frame = cv2.flip(frame, 1)

        # ── Hand detection ────────────────────────────────────────────
        frame = tracker.find_hands(frame, draw=True)
        landmark_list = tracker.find_position(frame, hand_no=0)

        # ── Index fingertip extraction ────────────────────────────────
        if landmark_list:
            # landmark_list[8] = (id=8, x, y)
            tip_id, tip_x, tip_y = landmark_list[FINGERTIP_ID]

            # Console output — will be replaced by gesture logic in Phase 2
            print(f"[Fingertip #{tip_id}]  x={tip_x:>4}  y={tip_y:>4}")

            # Visually mark the fingertip on the frame
            highlight_fingertip(frame, tip_x, tip_y)

            # Optionally show handedness label
            label = tracker.get_hand_label(hand_no=0)
            if label:
                cv2.putText(
                    frame,
                    f"Hand: {label}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 200, 0),
                    2,
                )

        # ── FPS calculation & display ─────────────────────────────────
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        draw_fps(frame, fps)

        # ── Render frame ──────────────────────────────────────────────
        cv2.imshow(WINDOW_TITLE, frame)

        # 'q' to quit — waitKey(1) keeps the loop at ~max camera FPS
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit signal received. Exiting...")
            break

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye.")


if __name__ == "__main__":
    main()
