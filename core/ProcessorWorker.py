import cv2
import numpy as np
from collections import deque

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage

# ใช้แบบแพ็กเกจ managers (ให้แน่ใจว่าโฟลเดอร์และ __init__.py มีอยู่)
from managers.background_manager import BackgroundManager
from managers.filter_manager import FilterManager
from managers.outline_manager import OutlineManager
from managers.face_manager import FaceManager

def load_prop(path):
    """Load prop image with alpha channel"""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        print(f"[WARN] ไม่พบไฟล์ prop: {path}")
    return im

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """Overlay transparent image with alpha channel (optimized)"""
    if overlay is None:
        return background

    bg_h, bg_w = background.shape[:2]

    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_LINEAR)

    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)
    elif overlay.shape[2] == 3:
        overlay = np.concatenate(
            [overlay, np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255], axis=2
        )

    o_h, o_w = overlay.shape[:2]

    if x >= bg_w or y >= bg_h or x + o_w <= 0 or y + o_h <= 0:
        return background

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + o_w, bg_w), min(y + o_h, bg_h)
    ov_x1, ov_y1 = x1 - x, y1 - y
    ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)

    overlay_crop = overlay[int(ov_y1):int(ov_y2), int(ov_x1):int(ov_x2)]
    if overlay_crop.size == 0:
        return background

    bg_crop = background[int(y1):int(y2), int(x1):int(x2)]

    alpha = overlay_crop[:, :, 3:] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(3):
        bg_crop[:, :, c] = (alpha[:, :, 0] * overlay_crop[:, :, c] +
                            alpha_inv[:, :, 0] * bg_crop[:, :, c])

    background[int(y1):int(y2), int(x1):int(x2)] = bg_crop
    return background

class ProcessorWorker(QThread):
    processed_ready = pyqtSignal(QImage)
    fps_ready = pyqtSignal(float)

    def __init__(self, PROJECT_ROOT, segment, bg_manager: BackgroundManager, filter_manager: FilterManager, outline_manager: OutlineManager, face_manager: FaceManager):
        super().__init__()
        self.PROJECT_ROOT = PROJECT_ROOT
        self.segment = segment
        self.bg_manager = bg_manager
        self.filter_manager = filter_manager
        self.outline_manager = outline_manager
        self.face_manager = face_manager

        self.queue: deque = deque(maxlen=1)
        self.running = True
        self._ema_fps = None

        self.enable_sharpen = True

        self.face_frame_skip = 2
        self.face_frame_counter = 0
        self.cached_landmarks = None

        self.props = {}
        props_files = {
            "glasses": "black-eyeglasses-frame-png.webp",
            "mustache": "pngimg.com - moustache_PNG43.png",
            "hat": "Black_Top_Hat_PNG_Clip_Art-1381.png"
        }
        for key, filename in props_files.items():
            prop = load_prop(str(self.PROJECT_ROOT / "resources" / "props" / filename))
            if prop is not None and prop.shape[2] == 3:
                prop = cv2.cvtColor(prop, cv2.COLOR_BGR2BGRA)
            self.props[key] = prop

        logo_path = self.PROJECT_ROOT / "resources" / "icon" / "logo-fac_logo-fibo.png"
        self.logo = load_prop(str(logo_path)) if logo_path.exists() else None
        self.logo_size = (640, 240)

        self.props_enabled = {
            "glasses": False,
            "mustache": False,
            "hat": False,
            "logo": True
        }

    @pyqtSlot(np.ndarray, dict)
    def enqueue(self, frame: np.ndarray, settings: dict):
        self.queue.append((frame, settings))

    def run(self):
        import time
        while self.running:
            if not self.queue:
                self.msleep(1)
                continue

            frame, settings = self.queue.pop()
            t0 = time.time()

            processed = self._process_frame(frame, settings)

            qimg = self._numpy_to_qimage(processed)
            self.processed_ready.emit(qimg)

            dt = time.time() - t0
            inst_fps = 1.0 / max(1e-6, dt)
            self._ema_fps = inst_fps if self._ema_fps is None else self._ema_fps * 0.8 + inst_fps * 0.2
            self.fps_ready.emit(float(self._ema_fps))

    def stop(self):
        self.running = False

    def _process_frame(self, frame: np.ndarray, settings: dict) -> np.ndarray:
        outline_style = settings["outline_style"]
        outline_thickness_val = settings["outline_thickness_val"]
        outline_color = settings["outline_color"]
        outline_opacity = settings["outline_opacity"]

        bg = self.bg_manager.get_background()
        self.segment.outline_thickness = outline_thickness_val

        apply_outline = outline_style != "None"
        if (bg is not None) or apply_outline:
            frame = self.segment.replace_background(
                frame,
                bg,
                apply_outline=apply_outline,
                outline_style=outline_style,
                outline_color=outline_color,
                outline_opacity=outline_opacity
            )

        frame = self._apply_props(frame)

        if self.enable_sharpen:
            blur = cv2.GaussianBlur(frame, (0, 0), 0.8)
            frame = cv2.addWeighted(frame, 1.22, blur, -0.22, 0)

        return frame

    def _apply_props(self, frame: np.ndarray) -> np.ndarray:
        if not any(self.props_enabled.values()):
            return frame

        h, w = frame.shape[:2]

        self.face_frame_counter += 1
        if self.face_frame_counter >= self.face_frame_skip:
            self.face_frame_counter = 0
            faces = self.face_manager.process(frame)
            if faces:
                self.cached_landmarks = faces


        if self.cached_landmarks:
            for landmarks in self.cached_landmarks:

                if self.props_enabled.get("glasses", False):
                    try:
                        left_eye = landmarks[33]
                        right_eye = landmarks[263]
                        eye_dist = np.linalg.norm(right_eye - left_eye)
                        eye_center = ((left_eye + right_eye) / 2).astype(int)

                        if self.props["glasses"] is not None:
                            glasses_w = int(eye_dist * 2.2)
                            glasses_h = int(glasses_w * self.props["glasses"].shape[0] / self.props["glasses"].shape[1])
                            gx = int(eye_center[0] - glasses_w / 2)
                            gy = int(eye_center[1] - glasses_h / 2)
                            frame = overlay_transparent(frame, self.props["glasses"], gx, gy, overlay_size=(glasses_w, glasses_h))
                    except Exception:
                        pass

                if self.props_enabled.get("hat", False):
                    try:
                        left_eye = landmarks[33]
                        right_eye = landmarks[263]
                        eye_dist = np.linalg.norm(right_eye - left_eye)
                        forehead = landmarks[10]
                        chin = landmarks[152]
                        head_h = max(1, int(np.linalg.norm(forehead - chin)))
                        hat_w = int(eye_dist * 2.6)
                        hat_h = int(hat_w * (self.props["hat"].shape[0] / self.props["hat"].shape[1])) if self.props["hat"] is not None else int(head_h * 0.6)
                        hx = int(forehead[0] - hat_w / 2)
                        hy = int(forehead[1] - hat_h * 0.85)
                        if self.props["hat"] is not None:
                            frame = overlay_transparent(frame, self.props["hat"], hx, hy, overlay_size=(hat_w, hat_h))
                    except Exception:
                        pass

                if self.props_enabled.get("mustache", False):
                    try:
                        left_eye = landmarks[33]
                        right_eye = landmarks[263]
                        eye_dist = np.linalg.norm(right_eye - left_eye)
                        mouth_upper = landmarks[13]
                        nose_bottom = landmarks[2]
                        must_w = int(eye_dist * 1.0)
                        must_h = int(must_w * (self.props["mustache"].shape[0] / self.props["mustache"].shape[1])) if self.props["mustache"] is not None else int(must_w * 0.3)
                        mx = int(mouth_upper[0] - must_w / 2)
                        my = int(nose_bottom[1] + (mouth_upper[1] - nose_bottom[1]) * 0.001)
                        if self.props["mustache"] is not None:
                            frame = overlay_transparent(frame, self.props["mustache"], mx, my, overlay_size=(must_w, must_h))
                    except Exception:
                        pass

        if self.props_enabled.get("logo", False) and self.logo is not None:
            frame = overlay_transparent(frame, self.logo, 0, 10, overlay_size=self.logo_size)

        return frame

    @staticmethod
    def _numpy_to_qimage(img: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

