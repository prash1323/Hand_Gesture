"""
hand_tracker.py
---------------
Core hand tracking module for the Hand Gesture Navigation System.
Uses MediaPipe Hands for landmark detection and OpenCV for rendering.
"""

import cv2
import mediapipe as mp


class HandTracker:
    """
    Wraps MediaPipe Hands to provide real-time hand landmark detection
    and coordinate extraction from webcam frames.
    """

    def __init__(
        self,
        mode=False,
        max_hands=2,
        model_complexity=1,
        detection_confidence=0.7,
        tracking_confidence=0.7,
    ):
        """
        Initialize MediaPipe Hands with configurable parameters.

        Args:
            mode               (bool):  True = static image mode, False = video stream mode.
            max_hands          (int):   Maximum number of hands to detect simultaneously.
            model_complexity   (int):   Model complexity: 0 (lite) or 1 (full). 1 is more accurate.
            detection_confidence (float): Minimum confidence for initial hand detection (0–1).
            tracking_confidence  (float): Minimum confidence for subsequent frame tracking (0–1).
        """
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # MediaPipe Hands solution
        self._mp_hands = mp.solutions.hands
        self.hands = self._mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

        # Drawing utilities for rendering landmarks and connections
        self._mp_draw = mp.solutions.drawing_utils
        self._draw_spec_landmark = self._mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=4
        )
        self._draw_spec_connection = self._mp_draw.DrawingSpec(
            color=(255, 255, 255), thickness=2
        )

        # Stores the latest detection results (updated every frame)
        self.results = None

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def find_hands(self, frame, draw=True):
        """
        Detect hands in a BGR frame and optionally draw landmarks.

        MediaPipe requires RGB input, so the frame is converted internally.
        The original BGR frame (with optional drawings) is returned so the
        caller can directly display it with cv2.imshow().

        Args:
            frame (np.ndarray): BGR frame from cv2.VideoCapture.
            draw  (bool):       Whether to draw landmarks and connections on the frame.

        Returns:
            np.ndarray: The input frame, annotated with landmarks if draw=True.
        """
        # Convert BGR → RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe is faster when the frame is marked as non-writeable
        rgb_frame.flags.writeable = False
        self.results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Draw landmarks for every detected hand
        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._draw_spec_landmark,
                    self._draw_spec_connection,
                )

        return frame

    def find_position(self, frame, hand_no=0):
        """
        Extract pixel coordinates for all 21 hand landmarks.

        MediaPipe returns landmark positions as normalized floats (0–1),
        so they are scaled to actual pixel values using the frame dimensions.

        Args:
            frame   (np.ndarray): Current video frame (used to get width/height).
            hand_no (int):        Index of the hand to extract (0 = first detected hand).

        Returns:
            list[tuple]: List of 21 tuples in the format (landmark_id, x_px, y_px).
                         Returns an empty list if no hands are detected or the
                         requested hand index is out of range.
        """
        landmark_list = []

        # Guard: no hands detected in this frame
        if not self.results or not self.results.multi_hand_landmarks:
            return landmark_list

        # Guard: requested hand index doesn't exist
        if hand_no >= len(self.results.multi_hand_landmarks):
            return landmark_list

        frame_h, frame_w, _ = frame.shape
        hand = self.results.multi_hand_landmarks[hand_no]

        for landmark_id, lm in enumerate(hand.landmark):
            # Scale normalized coords to pixel coords
            x_px = int(lm.x * frame_w)
            y_px = int(lm.y * frame_h)
            landmark_list.append((landmark_id, x_px, y_px))

        return landmark_list

    def get_hand_label(self, hand_no=0):
        """
        Return the handedness label ('Left' or 'Right') for a detected hand.

        Args:
            hand_no (int): Index of the hand.

        Returns:
            str | None: 'Left', 'Right', or None if not available.
        """
        if (
            self.results
            and self.results.multi_handedness
            and hand_no < len(self.results.multi_handedness)
        ):
            return self.results.multi_handedness[hand_no].classification[0].label
        return None
