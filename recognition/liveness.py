"""
Anti-spoofing / liveness detection module.

Detects presentation attacks (photos, screens) using:
1. Eye Aspect Ratio (EAR) blink detection
2. LBP texture analysis (screens/prints have different texture)
3. Color space analysis (screens have different color distribution)
"""

import numpy as np
import cv2
from collections import deque

try:
    import face_recognition as _fr
except ImportError:
    _fr = None


class LivenessDetector:
    """Detects whether a face is live or a spoof (photo/screen)."""

    def __init__(self, config):
        liveness_cfg = config.section("liveness")
        self.enabled = liveness_cfg.get("enabled", True)
        self.blink_threshold = liveness_cfg.get("blink_threshold", 0.25)
        self.texture_threshold = liveness_cfg.get("texture_threshold", 80.0)
        self.min_blinks = liveness_cfg.get("min_blinks", 1)
        self.check_interval = liveness_cfg.get("check_interval", 30)

        # Per-face state tracking (keyed by face ID or position)
        self._blink_counters = {}
        self._ear_history = {}
        self._frame_counter = 0

    def _eye_aspect_ratio(self, eye_points):
        """
        Compute Eye Aspect Ratio (EAR).
        EAR drops significantly during a blink.
        """
        eye = np.array(eye_points, dtype=np.float64)
        # Vertical distances
        v1 = np.linalg.norm(eye[1] - eye[5])
        v2 = np.linalg.norm(eye[2] - eye[4])
        # Horizontal distance
        h = np.linalg.norm(eye[0] - eye[3])
        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def _texture_score(self, face_roi):
        """
        LBP-like texture variance.
        Real faces have richer texture than flat photos/screens.
        """
        if face_roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        gray = cv2.resize(gray, (64, 64))
        # Compute local binary pattern approximation via gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.var(magnitude))

    def _color_analysis(self, face_roi):
        """
        Screens emit light differently than real skin.
        Check color distribution in YCrCb space.
        Returns a score (higher = more likely real).
        """
        if face_roi.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        cr_std = np.std(ycrcb[:, :, 1])
        cb_std = np.std(ycrcb[:, :, 2])
        # Real skin has natural Cr/Cb variance; screens are more uniform
        return float(cr_std + cb_std)

    def check_liveness(self, frame, face_location, face_id="default"):
        """
        Run liveness checks on a detected face.

        Args:
            frame: BGR image (full frame)
            face_location: (top, right, bottom, left) in frame coordinates
            face_id: unique identifier for tracking blinks over time

        Returns:
            dict with keys:
                is_live: bool
                confidence: float 0-1
                checks: dict of individual check results
        """
        if not self.enabled:
            return {"is_live": True, "confidence": 1.0, "checks": {}}

        self._frame_counter += 1
        top, right, bottom, left = face_location
        h, w = frame.shape[:2]
        # Clamp to image bounds
        top = max(0, top)
        left = max(0, left)
        bottom = min(h, bottom)
        right = min(w, right)
        face_roi = frame[top:bottom, left:right]

        results = {}
        scores = []

        # 1. Texture analysis
        tex_score = self._texture_score(face_roi)
        tex_pass = tex_score > self.texture_threshold
        results["texture"] = {"score": tex_score, "pass": tex_pass}
        scores.append(1.0 if tex_pass else 0.3)

        # 2. Color analysis
        color_score = self._color_analysis(face_roi)
        color_pass = color_score > 15.0  # empirical threshold
        results["color"] = {"score": color_score, "pass": color_pass}
        scores.append(1.0 if color_pass else 0.4)

        # 3. Blink detection (requires landmarks)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_list = _fr.face_landmarks(rgb, [face_location]) if _fr else []
        blink_detected = False
        if landmarks_list:
            lm = landmarks_list[0]
            left_ear = self._eye_aspect_ratio(lm["left_eye"])
            right_ear = self._eye_aspect_ratio(lm["right_eye"])
            avg_ear = (left_ear + right_ear) / 2.0

            # Track EAR history for blink detection
            if face_id not in self._ear_history:
                self._ear_history[face_id] = deque(maxlen=30)
                self._blink_counters[face_id] = 0

            history = self._ear_history[face_id]
            history.append(avg_ear)

            # Detect blink: EAR drops below threshold then recovers
            if len(history) >= 3:
                if history[-2] < self.blink_threshold and history[-1] > self.blink_threshold:
                    self._blink_counters[face_id] += 1
                    blink_detected = True

            blink_count = self._blink_counters.get(face_id, 0)
            blink_pass = blink_count >= self.min_blinks
            results["blink"] = {
                "ear": avg_ear,
                "blinks": blink_count,
                "pass": blink_pass,
                "detected_now": blink_detected,
            }
            scores.append(1.0 if blink_pass else 0.5)
        else:
            results["blink"] = {"ear": 0, "blinks": 0, "pass": False, "detected_now": False}
            scores.append(0.3)

        confidence = sum(scores) / len(scores)
        is_live = confidence > 0.6

        return {"is_live": is_live, "confidence": confidence, "checks": results}

    def reset(self, face_id=None):
        """Reset tracking state for a face or all faces."""
        if face_id:
            self._blink_counters.pop(face_id, None)
            self._ear_history.pop(face_id, None)
        else:
            self._blink_counters.clear()
            self._ear_history.clear()
