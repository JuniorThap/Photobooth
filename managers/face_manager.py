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
        """Run face landmark detection on frame (with internal caching)."""
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
            landmarks = np.array([[p.x * w, p.y * h] for p in lm])
            if not np.any(np.isnan(landmarks)):
                faces.append(landmarks)

        self.cached_landmarks = faces if faces else None
        return self.cached_landmarks
