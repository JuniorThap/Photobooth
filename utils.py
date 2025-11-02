import numpy as np
from pathlib import Path
import cv2

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFrame, QColorDialog, QScrollArea
)


PROJECT_ROOT = Path(__file__).resolve().parent

UI_SCALE = 0.75  # tweak 0.65–0.85 to taste for 13" screens
def S(x: int) -> int:
    return int(x * UI_SCALE)

THUMBNAIL_SIZE = QSize(S(150), S(100))   # was 150x100
BUTTON_SIZE = (S(160), S(110))           # was (160, 110)


def create_photobooth_frame(images: list, frame_config_name="fibooth_modern"):
    """Create photobooth strip with 3 images using custom frame template"""
    import json

    config_path = PROJECT_ROOT / "frames" / "frame_config.json"

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
            config = configs.get(frame_config_name, configs.get("default"))
    else:
        config = {
            "frame_path": None,
            "image_positions": [
                {"x": 100, "y": 150, "width": 1720, "height": 968},
                {"x": 100, "y": 1218, "width": 1720, "height": 968},
                {"x": 100, "y": 2286, "width": 1720, "height": 968}
            ],
            "canvas_size": [1920, 3404]
        }

    frame_width, frame_height = config["canvas_size"]
    positions = config["image_positions"]

    frame_path = config.get("frame_path")
    if frame_path:
        frame_full_path = PROJECT_ROOT / frame_path
        if frame_full_path.exists():
            frame_template = cv2.imread(str(frame_full_path), cv2.IMREAD_UNCHANGED)
            if frame_template is not None:
                if frame_template.shape[2] == 4:
                    canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
                    alpha = frame_template[:, :, 3:] / 255.0
                    for c in range(3):
                        canvas[:, :, c] = (alpha[:, :, 0] * frame_template[:, :, c] +
                                          (1 - alpha[:, :, 0]) * canvas[:, :, c])
                else:
                    canvas = frame_template.copy()
            else:
                canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        else:
            canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    else:
        canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

    for img, pos in zip(images[:3], positions):
        if img is not None:
            x = pos["x"]
            y = pos["y"]
            target_width = pos["width"]
            target_height = pos["height"]
            final_image = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            canvas[y:y+target_height, x:x+target_width] = final_image

    return canvas