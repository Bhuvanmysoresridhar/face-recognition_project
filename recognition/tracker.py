"""
Face tracker - maintains identity across frames using centroid tracking.
Avoids re-encoding every processed frame for better FPS.
"""

import numpy as np
from collections import OrderedDict


class FaceTracker:
    """
    Centroid-based multi-object tracker for faces.
    Associates detected faces across frames by spatial proximity.
    """

    def __init__(self, config):
        tracker_cfg = config.section("tracker")
        self.enabled = tracker_cfg.get("enabled", True)
        self.max_disappeared = tracker_cfg.get("max_disappeared", 15)
        self.max_distance = tracker_cfg.get("max_distance", 75)

        self._next_id = 0
        self.objects = OrderedDict()       # id -> centroid (cx, cy)
        self.bboxes = OrderedDict()        # id -> (top, right, bottom, left)
        self.names = OrderedDict()         # id -> recognized name
        self.confidences = OrderedDict()   # id -> confidence
        self.disappeared = OrderedDict()   # id -> frames missing

    def _register(self, centroid, bbox, name="Unknown", confidence=0.0):
        obj_id = self._next_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.names[obj_id] = name
        self.confidences[obj_id] = confidence
        self.disappeared[obj_id] = 0
        self._next_id += 1
        return obj_id

    def _deregister(self, obj_id):
        for store in (self.objects, self.bboxes, self.names,
                      self.confidences, self.disappeared):
            store.pop(obj_id, None)

    @staticmethod
    def _centroid(bbox):
        """Compute centroid from (top, right, bottom, left)."""
        top, right, bottom, left = bbox
        return np.array([(left + right) / 2.0, (top + bottom) / 2.0])

    def update(self, detections, names=None, confidences=None):
        """
        Update tracker with new detections.

        Args:
            detections: list of (top, right, bottom, left) bounding boxes
            names: optional list of recognized names (parallel to detections)
            confidences: optional list of confidence scores

        Returns:
            dict of {object_id: (bbox, name, confidence)} for all tracked objects
        """
        if not self.enabled:
            result = {}
            for i, bbox in enumerate(detections):
                n = names[i] if names else "Unknown"
                c = confidences[i] if confidences else 0.0
                result[i] = (bbox, n, c)
            return result

        if names is None:
            names = ["Unknown"] * len(detections)
        if confidences is None:
            confidences = [0.0] * len(detections)

        # No detections: increment disappeared for all
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self._current_state()

        input_centroids = np.array([self._centroid(d) for d in detections])

        # No existing objects: register all
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self._register(input_centroids[i], detections[i], names[i], confidences[i])
            return self._current_state()

        # Match existing objects to new detections
        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))

        # Compute distance matrix
        dist_matrix = np.linalg.norm(
            obj_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        # Greedy assignment by minimum distance
        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist_matrix[row, col] > self.max_distance:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = detections[col]
            # Update name only if new detection is recognized
            if names[col] != "Unknown":
                self.names[obj_id] = names[col]
                self.confidences[obj_id] = confidences[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        for row in range(len(obj_ids)):
            if row not in used_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        # Register new detections
        for col in range(len(detections)):
            if col not in used_cols:
                self._register(input_centroids[col], detections[col],
                               names[col], confidences[col])

        return self._current_state()

    def _current_state(self):
        """Return current tracked objects as {id: (bbox, name, confidence)}."""
        result = {}
        for obj_id in self.objects:
            result[obj_id] = (
                self.bboxes[obj_id],
                self.names.get(obj_id, "Unknown"),
                self.confidences.get(obj_id, 0.0),
            )
        return result

    def reset(self):
        self._next_id = 0
        self.objects.clear()
        self.bboxes.clear()
        self.names.clear()
        self.confidences.clear()
        self.disappeared.clear()
