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

    def load_thumbnails(self, limit: int = 20) -> List[Tuple[str, QPixmap]]:
        thumbnails = []
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
                thumbnails.append((str(file), pixmap))
                if len(thumbnails) >= limit:
                    return thumbnails

        # --- อ่านไฟล์ภาพในโฟลเดอร์ย่อย ---
        for folder in self.root_path.iterdir():
            if not folder.is_dir():
                continue
            for file in folder.iterdir():
                if file.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                    continue
                img = cv2.imread(str(file))
                if img is None:
                    continue
                resized = cv2.resize(img, THUMBNAIL_SIZE)
                pixmap = self._numpy_to_pixmap(resized)
                thumbnails.append((str(file), pixmap))
                if len(thumbnails) >= limit:
                    return thumbnails
        return thumbnails

    def set_background(self, path: str) -> bool:
        img = cv2.imread(path)
        if img is not None:
            self.current_background = img
            return True
        return False

    def get_background(self) -> Optional[np.ndarray]:
        return self.current_background.copy() if self.current_background is not None else None

    @staticmethod
    def _numpy_to_pixmap(img: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)
