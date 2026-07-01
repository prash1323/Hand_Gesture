"""
utils.py
--------
Utility functions for the Hand Gesture Navigation System.

Provides:
  - fingers_up()          : Determines which fingers are extended.
  - calculate_distance()  : Euclidean distance between two landmarks.

These are pure helper functions — no state, no side effects — so they
can be imported and tested independently of the rest of the pipeline.
"""

import math
from typing import Optional


# ── MediaPipe landmark indices ────────────────────────────────────────────────
# Each finger has a TIP and a lower PIP/MCP joint used for up/down comparison.
#
#   Wrist                                      →  0
#   Thumb  : CMC=1  MCP=2  IP=3   TIP=4
#   Index  : MCP=5  PIP=6  DIP=7  TIP=8
#   Middle : MCP=9  PIP=10 DIP=11 TIP=12
#   Ring   : MCP=13 PIP=14 DIP=15 TIP=16
#   Pinky  : MCP=17 PIP=18 DIP=19 TIP=20
# ─────────────────────────────────────────────────────────────────────────────

# Tip landmark IDs for each finger (thumb → pinky)
_TIP_IDS = [4, 8, 12, 16, 20]

# The joint one step below each fingertip — used as the "is it extended?" baseline
# Thumb uses IP (3); the rest use PIP joints.
_PIP_IDS = [3, 6, 10, 14, 18]


def fingers_up(landmarks: list) -> list[int]:
    """
    Determine which fingers are currently extended (up).

    The function compares each fingertip's y-coordinate with the joint
    just below it (PIP). Because the webcam frame has y=0 at the top,
    a smaller y value means the point is *higher* on screen.

    Thumb detection is handled differently: it compares x-coordinates
    to detect lateral extension, since the thumb moves sideways rather
    than vertically. The comparison direction is handedness-agnostic —
    it simply checks whether the tip is further from the wrist than the
    IP joint along the x-axis.

    Args:
        landmarks (list): 21 tuples of (id, x_px, y_px) from HandTracker.

    Returns:
        list[int]: 5 binary values — [thumb, index, middle, ring, pinky].
                   1 = extended (up), 0 = folded (down).
                   Returns [0, 0, 0, 0, 0] if landmarks are missing or malformed.
    """
    # Guard: need all 21 landmarks
    if not landmarks or len(landmarks) < 21:
        return [0, 0, 0, 0, 0]

    fingers = []

    try:
        # ── Thumb (landmark 4 vs 3) ──────────────────────────────────
        # Wrist is landmark 0; compare x-distance of tip vs IP joint.
        wrist_x = landmarks[0][1]
        thumb_tip_x = landmarks[_TIP_IDS[0]][1]
        thumb_ip_x = landmarks[_PIP_IDS[0]][1]

        # Tip further from wrist than IP → thumb is extended
        if abs(thumb_tip_x - wrist_x) > abs(thumb_ip_x - wrist_x):
            fingers.append(1)
        else:
            fingers.append(0)

        # ── Index → Pinky (landmarks 8,12,16,20 vs 6,10,14,18) ───────
        for tip_id, pip_id in zip(_TIP_IDS[1:], _PIP_IDS[1:]):
            tip_y = landmarks[tip_id][2]   # y-coordinate of fingertip
            pip_y = landmarks[pip_id][2]   # y-coordinate of PIP joint

            # Tip above PIP (smaller y) → finger is extended
            fingers.append(1 if tip_y < pip_y else 0)

    except (IndexError, TypeError):
        # Return all-down if landmark data is corrupt or incomplete
        return [0, 0, 0, 0, 0]

    return fingers


def calculate_distance(
    p1: tuple,
    p2: tuple,
) -> Optional[float]:
    """
    Calculate the Euclidean distance between two landmark points.

    Args:
        p1 (tuple): First  point as (id, x, y).
        p2 (tuple): Second point as (id, x, y).

    Returns:
        float : Pixel distance between the two points.
        None  : If either point is None or malformed.
    """
    if p1 is None or p2 is None:
        return None

    try:
        _, x1, y1 = p1
        _, x2, y2 = p2
        return math.hypot(x2 - x1, y2 - y1)
    except (TypeError, ValueError):
        return None


def midpoint(p1: tuple, p2: tuple) -> Optional[tuple[int, int]]:
    """
    Return the pixel midpoint between two landmarks.
    Useful for rendering pinch indicators on screen.

    Args:
        p1 (tuple): (id, x, y)
        p2 (tuple): (id, x, y)

    Returns:
        tuple(int, int) | None: (mid_x, mid_y) in pixel coordinates.
    """
    if p1 is None or p2 is None:
        return None

    try:
        _, x1, y1 = p1
        _, x2, y2 = p2
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    except (TypeError, ValueError):
        return None
