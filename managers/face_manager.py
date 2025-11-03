# managers/face_manager.py
import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python._framework_bindings as _mp_fb  # force load
from typing import List, Optional


class FaceManager:
    def __init__(self, max_faces: int = 3):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.frame_skip = 2
        self.counter = 0
        self.cached_landmarks: Optional[List[np.ndarray]] = None

    def process(self, frame_bgr: np.ndarray) -> Optional[List[np.ndarray]]:
        """Run face landmark detection on frame (with internal caching and duplicate suppression)."""
        self.counter += 1
        if self.counter < self.frame_skip and self.cached_landmarks is not None:
            return self.cached_landmarks
        self.counter = 0

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.cached_landmarks = None
            return None

        faces = []
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark
            landmarks = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            if not np.any(np.isnan(landmarks)):
                faces.append(landmarks)

        # --- Post-process: remove duplicates ---
        faces = self._filter_duplicates(faces, min_distance=50.0)

        self.cached_landmarks = faces if faces else None
        return self.cached_landmarks

    # -------------------------------------------
    # 🔹 Duplicate suppression helper
    # -------------------------------------------
    def _filter_duplicates(self, faces: List[np.ndarray],
                       min_distance: float = 50.0,
                       iou_thresh: float = 0.6) -> List[np.ndarray]:
        """Remove faces that overlap or are too close (duplicate detections)."""
        if len(faces) <= 1:
            return faces

        # compute bounding boxes & centroids
        boxes, centroids = [], []
        for f in faces:
            x_min, y_min = np.min(f, axis=0)
            x_max, y_max = np.max(f, axis=0)
            boxes.append((x_min, y_min, x_max, y_max))
            centroids.append(np.mean(f, axis=0))

        keep = []
        used = set()

        def iou(box1, box2):
            xA = max(box1[0], box2[0])
            yA = max(box1[1], box2[1])
            xB = min(box1[2], box2[2])
            yB = min(box1[3], box2[3])
            inter_area = max(0, xB - xA) * max(0, yB - yA)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0

        for i, c1 in enumerate(centroids):
            if i in used:
                continue
            keep.append(faces[i])
            for j in range(i + 1, len(centroids)):
                if j in used:
                    continue
                c2 = centroids[j]
                dist = np.linalg.norm(c1 - c2)
                if dist < min_distance or iou(boxes[i], boxes[j]) > iou_thresh:
                    used.add(j)

        return keep
