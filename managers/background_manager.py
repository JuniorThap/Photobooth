from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage

THUMBNAIL_SIZE = (150, 100)

class BackgroundManager:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.current_background: Optional[np.ndarray] = None

    def load_thumbnails(self, limit: int = 20):
        thumbnails = {"Images": [], "Videos": []}
        if not self.root_path.exists():
            return thumbnails

        # --- อ่านไฟล์ภาพที่อยู่ใน root_path โดยตรง ---
        for file in self.root_path.iterdir():
            if file.is_file() and file.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                img = cv2.imread(str(file))
                if img is None:
                    continue
                resized = cv2.resize(img, THUMBNAIL_SIZE)
                pixmap = self._numpy_to_pixmap(resized)
                thumbnails['Images'].append((str(file), pixmap))
                if len(thumbnails['Images']) + len(thumbnails['Videos']) >= limit:
                    return thumbnails
            
            if file.is_file() and file.suffix.lower() in {'.mp4', '.mov'}:
                cap = cv2.VideoCapture(str(file))
                if not cap.isOpened():
                    continue
                _, frame = cap.read()
                resized = cv2.resize(frame, THUMBNAIL_SIZE)
                pixmap = self._numpy_to_pixmap(resized)
                thumbnails['Videos'].append((str(file), pixmap))
                cap.release()
                if len(thumbnails['Images']) + len(thumbnails["Videos"]) >= limit:
                    return thumbnails

        return thumbnails

    def set_background(self, path: str) -> bool:

        if isinstance(self.current_background, cv2.VideoCapture):
            self.current_background.release()

        img = cv2.imread(path)
        if img is not None:
            self.current_background = img
            return True
        
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            self.current_background = cap
            return True

        return False

    def get_background(self) -> Optional[np.ndarray]:
        if isinstance(self.current_background, cv2.VideoCapture):
            ret, frame = self.current_background.read()
            if not ret:
                self.current_background.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.current_background.read()
            return frame
        elif self.current_background is not None:
            return self.current_background.copy()
        return None
    
    def restart_background(self):
        if isinstance(self.current_background, cv2.VideoCapture):
            self.current_background.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def get_video_length(self):
        if isinstance(self.current_background, cv2.VideoCapture):
            fps = self.current_background.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.current_background.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0
            return duration_sec
        return None

    @staticmethod
    def _numpy_to_pixmap(img: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)
