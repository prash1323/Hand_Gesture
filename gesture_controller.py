"""
gesture_controller.py
---------------------
Maps recognized hand gestures to system actions via PyAutoGUI.

Gesture → Action table
──────────────────────────────────────────────────────────────
  Index up only            →  Move cursor
  Thumb + Index pinch      →  Left click
  Index + Middle pinch     →  Right click
  Index + Middle both up   →  Scroll  (index=up, middle=up)
──────────────────────────────────────────────────────────────

Design notes:
  • Coordinate mapping uses linear interpolation between a configurable
    webcam ROI and the full screen resolution.
  • Exponential moving average (EMA) smoothing eliminates jitter without
    adding a noticeable lag for deliberate movement.
  • A per-action cooldown timer prevents repeated unintended triggers
    from a single held gesture.
"""

import time
import pyautogui
import numpy as np

# Disable PyAutoGUI's built-in fail-safe pause — we handle timing ourselves.
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True   # Keep True: move mouse to corner to abort


class GestureController:
    """
    Translates hand landmark data into mouse and scroll events.

    Attributes
    ----------
    screen_w, screen_h : int
        Full screen resolution reported by PyAutoGUI.
    smooth_x, smooth_y : float
        EMA-smoothed cursor position (updated each frame).
    """

    # ── Tunable constants ─────────────────────────────────────────────────────

    # Fraction of the webcam frame used as the "active zone" for cursor movement.
    # Shrinking this gives you the full screen range with less physical hand movement.
    ROI_LEFT   = 0.15    # 15% from left edge
    ROI_RIGHT  = 0.85    # 85% from left edge  (so active width = 70% of frame)
    ROI_TOP    = 0.15    # 15% from top edge
    ROI_BOTTOM = 0.85    # 85% from top edge   (active height = 70% of frame)

    # Exponential moving average factor.  Range: (0, 1]
    #   → Higher = more responsive but jittery
    #   → Lower  = smoother but slightly lagged
    EMA_ALPHA = 0.25

    # Pinch distance thresholds (in pixels, before any scaling)
    LEFT_CLICK_THRESHOLD  = 38   # thumb + index tip distance for left click
    RIGHT_CLICK_THRESHOLD = 42   # index + middle tip distance for right click

    # Minimum seconds between consecutive triggers of the same action
    CLICK_COOLDOWN  = 0.45   # seconds
    SCROLL_COOLDOWN = 0.12   # seconds — faster repeat for scroll feels natural

    # Scroll step size in "clicks" passed to pyautogui.scroll()
    SCROLL_STEP = 3

    # ── Initialization ────────────────────────────────────────────────────────

    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

        # Smoothed cursor position — seeded at screen center on startup
        self.smooth_x = self.screen_w / 2
        self.smooth_y = self.screen_h / 2

        # Per-action last-triggered timestamps
        self._last_left_click  = 0.0
        self._last_right_click = 0.0
        self._last_scroll      = 0.0

        print(
            f"[GestureController] Screen: {self.screen_w}×{self.screen_h}  |  "
            f"EMA α={self.EMA_ALPHA}  |  ROI x=[{self.ROI_LEFT}–{self.ROI_RIGHT}] "
            f"y=[{self.ROI_TOP}–{self.ROI_BOTTOM}]"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def move_cursor(
        self,
        index_x: int,
        index_y: int,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        Translate the index fingertip's webcam position to a screen position
        and move the OS cursor there.

        The raw landmark coordinate is:
          1. Clamped to the active ROI region.
          2. Linearly mapped to full screen resolution (with x-axis flipped
             because the webcam feed is mirrored).
          3. Smoothed via exponential moving average to remove jitter.

        Args:
            index_x  (int): Fingertip x in webcam pixels.
            index_y  (int): Fingertip y in webcam pixels.
            frame_w  (int): Full frame width in pixels.
            frame_h  (int): Full frame height in pixels.
        """
        # ── 1. Define ROI bounds in absolute pixel values ─────────────
        roi_x1 = int(frame_w * self.ROI_LEFT)
        roi_x2 = int(frame_w * self.ROI_RIGHT)
        roi_y1 = int(frame_h * self.ROI_TOP)
        roi_y2 = int(frame_h * self.ROI_BOTTOM)

        # ── 2. Clamp fingertip to ROI ─────────────────────────────────
        clamped_x = max(roi_x1, min(index_x, roi_x2))
        clamped_y = max(roi_y1, min(index_y, roi_y2))

        # ── 3. Map ROI → screen (x is flipped for mirrored feed) ──────
        # np.interp(value, [from_range], [to_range])
        screen_x = np.interp(clamped_x, [roi_x1, roi_x2], [self.screen_w, 0])
        screen_y = np.interp(clamped_y, [roi_y1, roi_y2], [0, self.screen_h])

        # ── 4. Exponential moving average smoothing ───────────────────
        self.smooth_x += self.EMA_ALPHA * (screen_x - self.smooth_x)
        self.smooth_y += self.EMA_ALPHA * (screen_y - self.smooth_y)

        # ── 5. Move cursor (duration=0 = instant, no PyAutoGUI tween) ─
        pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y), duration=0)

    def left_click(self) -> bool:
        """
        Fire a left mouse click if the cooldown has elapsed.

        Returns:
            bool: True if the click was executed, False if still in cooldown.
        """
        now = time.time()
        if now - self._last_left_click >= self.CLICK_COOLDOWN:
            pyautogui.click()
            self._last_left_click = now
            return True
        return False

    def right_click(self) -> bool:
        """
        Fire a right mouse click if the cooldown has elapsed.

        Returns:
            bool: True if the click was executed, False if still in cooldown.
        """
        now = time.time()
        if now - self._last_right_click >= self.CLICK_COOLDOWN:
            pyautogui.rightClick()
            self._last_right_click = now
            return True
        return False

    def scroll(self, direction: str) -> bool:
        """
        Scroll the active window up or down.

        Args:
            direction (str): 'up' or 'down'.

        Returns:
            bool: True if scroll was executed, False if in cooldown.
        """
        now = time.time()
        if now - self._last_scroll >= self.SCROLL_COOLDOWN:
            clicks = self.SCROLL_STEP if direction == "up" else -self.SCROLL_STEP
            pyautogui.scroll(clicks)
            self._last_scroll = now
            return True
        return False

    # ── Gesture detection helpers ─────────────────────────────────────────────
    # These live here rather than in utils.py because they depend on the same
    # threshold constants that govern the controller's behaviour.

    def is_left_click_pinch(self, distance: float) -> bool:
        """Return True when thumb–index distance signals a left-click pinch."""
        return distance is not None and distance < self.LEFT_CLICK_THRESHOLD

    def is_right_click_pinch(self, distance: float) -> bool:
        """Return True when index–middle distance signals a right-click pinch."""
        return distance is not None and distance < self.RIGHT_CLICK_THRESHOLD
