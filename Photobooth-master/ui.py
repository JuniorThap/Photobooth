import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QFont, QColor
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QStackedWidget, QScrollArea,
    QGridLayout, QPushButton, QSlider, QFrame, QColorDialog
)

# MediaPipe for Face Mesh
import mediapipe as mp

# ใช้แบบแพ็กเกจ managers (ให้แน่ใจว่าโฟลเดอร์และ __init__.py มีอยู่)
from managers.background_manager import BackgroundManager
from managers.filter_manager import FilterManager
from managers.outline_manager import OutlineManager
from managers.segment_manager import create_segment

# ---------------- UI/Camera Parameters ----------------
CAMERA_INDEX = 1               # ใช้กล้อง index 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 60
THUMBNAIL_SIZE = QSize(150, 100)
BUTTON_SIZE = (160, 110)

# พาธ background แบบ relative: Photobooth-master/backgrounds/photos
PROJECT_ROOT = Path(__file__).resolve().parent
BG_ROOT = PROJECT_ROOT / "backgrounds" / "photos"

DARK_QSS = """
* { color: #eaeaea; font-family: "Inter", "Segoe UI", system-ui; }
QWidget { background-color: #111; }
QLabel[hero="true"] { font-size: 24px; font-weight: 700; color: #fff; }
QLabel[section="true"] { font-size: 18px; font-weight: 600; color: #ddd; }
QPushButton {
    background-color: #1f1f1f; border: 2px solid #2d2d2d; border-radius: 12px;
    padding: 12px; font-weight: 600;
}
QPushButton:hover { background-color: #2a2a2a; border: 2px solid #3a3a3a; }
QPushButton:pressed { background-color: #161616; }
QSlider::groove:horizontal { height: 6px; background: #2b2b2b; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #3f7ae0; width: 16px; border-radius: 8px; margin: -6px 0;
}
QScrollArea { border: none; }
QFrame#line { background-color: #2a2a2a; max-height: 2px; min-height: 2px; }
"""

# ---------------- Props/MediaPipe Functions ----------------
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """Overlay transparent image with alpha channel (optimized)"""
    if overlay is None:
        return background

    bg_h, bg_w = background.shape[:2]

    # Optimize: resize only if needed
    if overlay_size is not None:
        # Use INTER_LINEAR for speed (good quality for downscaling)
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_LINEAR)

    # Optimize: expect BGRA format (pre-converted)
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)
    elif overlay.shape[2] == 3:
        overlay = np.concatenate([overlay, np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype)*255], axis=2)

    o_h, o_w = overlay.shape[:2]

    # Early exit if completely out of bounds
    if x >= bg_w or y >= bg_h or x + o_w <= 0 or y + o_h <= 0:
        return background

    # Calculate intersection
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + o_w, bg_w), min(y + o_h, bg_h)
    ov_x1, ov_y1 = x1 - x, y1 - y
    ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)

    overlay_crop = overlay[int(ov_y1):int(ov_y2), int(ov_x1):int(ov_x2)]
    if overlay_crop.size == 0:
        return background

    bg_crop = background[int(y1):int(y2), int(x1):int(x2)]

    # Optimize: use in-place operations where possible
    alpha = overlay_crop[:, :, 3:] / 255.0
    alpha_inv = 1.0 - alpha

    # Vectorized blending (faster than explicit loops)
    for c in range(3):
        bg_crop[:, :, c] = (alpha[:, :, 0] * overlay_crop[:, :, c] +
                           alpha_inv[:, :, 0] * bg_crop[:, :, c])

    background[int(y1):int(y2), int(x1):int(x2)] = bg_crop
    return background


def load_prop(path):
    """Load prop image with alpha channel"""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        print(f"[WARN] ไม่พบไฟล์ prop: {path}")
    return im


def create_photobooth_frame(images: list, frame_config_name="default"):
    """Create photobooth strip with 3 images using custom frame template"""
    import json

    # Load frame configuration
    config_path = PROJECT_ROOT / "frames" / "frame_config.json"

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
            config = configs.get(frame_config_name, configs.get("default"))
    else:
        # Default configuration if no config file
        # Use 4:6 aspect ratio (portrait), each image keeps its original aspect ratio
        config = {
            "frame_path": None,
            "image_positions": [
                {"x": 100, "y": 150, "width": 1720, "height": 968},
                {"x": 100, "y": 1218, "width": 1720, "height": 968},
                {"x": 100, "y": 2286, "width": 1720, "height": 968}
            ],
            "canvas_size": [1920, 3404]  # 4:6 ratio with margin
        }

    frame_width, frame_height = config["canvas_size"]
    positions = config["image_positions"]

    # Load custom frame template if exists
    frame_path = config.get("frame_path")
    if frame_path:
        frame_full_path = PROJECT_ROOT / frame_path
        if frame_full_path.exists():
            # Load frame with alpha channel
            frame_template = cv2.imread(str(frame_full_path), cv2.IMREAD_UNCHANGED)
            if frame_template is not None:
                # Create canvas from frame template
                if frame_template.shape[2] == 4:
                    # Has alpha channel
                    canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
                    # Composite frame template onto white background
                    alpha = frame_template[:, :, 3:] / 255.0
                    for c in range(3):
                        canvas[:, :, c] = (alpha[:, :, 0] * frame_template[:, :, c] +
                                          (1 - alpha[:, :, 0]) * canvas[:, :, c])
                else:
                    canvas = frame_template.copy()
            else:
                # Fallback to simple white canvas
                canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        else:
            canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    else:
        # Simple white canvas
        canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

    # Place images in configured positions
    for img, pos in zip(images[:3], positions):
        if img is not None:
            x = pos["x"]
            y = pos["y"]
            target_width = pos["width"]
            target_height = pos["height"]

            # Get original image dimensions
            img_h, img_w = img.shape[:2]
            img_aspect = img_w / img_h
            target_aspect = target_width / target_height

            # Resize keeping aspect ratio (fit inside target area)
            if img_aspect > target_aspect:
                # Image is wider - fit to width
                new_width = target_width
                new_height = int(target_width / img_aspect)
            else:
                # Image is taller - fit to height
                new_height = target_height
                new_width = int(target_height * img_aspect)

            # Resize image
            resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Center the image in the target area
            offset_x = (target_width - new_width) // 2
            offset_y = (target_height - new_height) // 2

            # Place image on canvas
            canvas[y+offset_y:y+offset_y+new_height, x+offset_x:x+offset_x+new_width] = resized

    return canvas

# ---------------- Threads ----------------
class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, cam_index: int = CAMERA_INDEX, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        super().__init__()
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.running = True
        self.cap: Optional[cv2.VideoCapture] = None

    def run(self):
        # Sony Imaging Edge Camera ต้องใช้ DSHOW และ format พิเศษ
        print(f"[Camera] กำลังเปิดกล้อง index {self.cam_index}...")

        # ลอง DSHOW ก่อน (ดีที่สุดสำหรับ Sony Imaging Edge)
        try_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        for backend in try_backends:
            backend_name = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "ANY"}
            print(f"[Camera] ลอง backend: {backend_name.get(backend, backend)}")

            self.cap = cv2.VideoCapture(self.cam_index, backend)
            if self.cap.isOpened():
                print(f"[Camera] ✓ เปิดกล้องสำเร็จด้วย {backend_name.get(backend, backend)}")
                break
            else:
                print(f"[Camera] ✗ ไม่สามารถเปิดด้วย {backend_name.get(backend, backend)}")

        if not self.cap or not self.cap.isOpened():
            print("[Camera] ERROR: ไม่สามารถเปิดกล้องได้!")
            return

        # --- ตั้งค่ากล้อง (สำหรับ Sony Imaging Edge) ---
        # ลอง YUY2 ก่อน (Sony Imaging Edge มักใช้ YUY2 ดีกว่า MJPG)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"YUY2")
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            print("[Camera] ตั้งค่า format: YUY2")
        except Exception:
            pass

        # ความละเอียด - เริ่มจาก 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # ตรวจสอบว่าตั้งได้จริง
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] ความละเอียดที่ได้: {actual_w}x{actual_h}")

        if actual_w < 1920 or actual_h < 1080:
            # ลอง 1280x720
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Camera] Fallback ความละเอียด: {actual_w}x{actual_h}")

        # FPS ตามที่ตั้ง (60 ถ้ากล้องไหว)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        # ลด latency: ใช้เฟรมล่าสุดเสมอ
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ช่วยเรื่องความชัด/สว่าง (ไดรเวอร์รองรับถึงจะได้ผล)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)       # เปิดออโต้โฟกัส
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # 0.75 = auto (บาง backend)
        # ถ้าอยากลอง manual:
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25); self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        # --- จบการตั้งค่ากล้อง ---

        print("[Camera] เริ่มอ่านเฟรม...")
        frame_count = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if frame_count == 0:
                    print("[Camera] ERROR: ไม่สามารถอ่านเฟรมได้!")
                continue

            if frame_count == 0:
                print(f"[Camera] ✓ อ่านเฟรมสำเร็จ! ขนาด: {frame.shape}")

            # mirror สำหรับ photobooth
            frame = cv2.flip(frame, 1)
            self.frame_ready.emit(frame)
            frame_count += 1

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False


class ProcessorWorker(QThread):
    processed_ready = pyqtSignal(QImage)
    fps_ready = pyqtSignal(float)

    def __init__(self, segment, bg_manager: BackgroundManager, filter_manager: FilterManager, outline_manager: OutlineManager):
        super().__init__()
        self.segment = segment
        self.bg_manager = bg_manager
        self.filter_manager = filter_manager
        self.outline_manager = outline_manager

        self.queue: deque = deque(maxlen=1)  # เก็บเฟรมล่าสุดเท่านั้น
        self.running = True
        self._ema_fps = None

        # sharpen อ่อนๆ
        self.enable_sharpen = True

        # MediaPipe Face Mesh (optimized for speed)
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,  # Reduce from 6 to 3 for better performance
            refine_landmarks=False,  # Disable refinement for speed
            min_detection_confidence=0.4,  # Slightly lower for faster detection
            min_tracking_confidence=0.4
        )

        # Frame skip for face detection (process every N frames)
        self.face_frame_skip = 2  # Process face detection every 2 frames
        self.face_frame_counter = 0
        self.cached_landmarks = None  # Cache last detected landmarks

        # Load props and pre-convert to BGRA for faster overlay
        self.props = {}
        props_files = {
            "glasses": "black-eyeglasses-frame-png.webp",
            "mustache": "pngimg.com - moustache_PNG43.png",
            "hat": "Black_Top_Hat_PNG_Clip_Art-1381.png"
        }
        for key, filename in props_files.items():
            prop = load_prop(str(PROJECT_ROOT / "props" / filename))
            if prop is not None:
                # Ensure BGRA format for faster processing
                if prop.shape[2] == 3:
                    prop = cv2.cvtColor(prop, cv2.COLOR_BGR2BGRA)
            self.props[key] = prop

        # Logo
        logo_path = PROJECT_ROOT / "icon" / "logo-fac_logo-fibo.png"
        if not logo_path.exists():
            # Fallback: check for icon subdirectory
            logo_path = PROJECT_ROOT / "icon" / "icon" / "logo-fac_logo-fibo.png"
        self.logo = load_prop(str(logo_path)) if logo_path.exists() else None
        self.logo_size = (640, 240)

        # Props toggles
        self.props_enabled = {
            "glasses": False,
            "mustache": False,
            "hat": False,
            "logo": False
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

            # คำนวณ FPS แบบ EMA
            dt = time.time() - t0
            inst_fps = 1.0 / max(1e-6, dt)
            self._ema_fps = inst_fps if self._ema_fps is None else self._ema_fps * 0.8 + inst_fps * 0.2
            self.fps_ready.emit(float(self._ema_fps))

    def stop(self):
        self.running = False

    # ---------------- Processing ----------------
    def _process_frame(self, frame: np.ndarray, settings: dict) -> np.ndarray:
        outline_style = settings["outline_style"]
        outline_thickness_val = settings["outline_thickness_val"]
        outline_color = settings["outline_color"]
        outline_opacity = settings["outline_opacity"]
        current_filter = settings["current_filter"]

        bg = self.bg_manager.get_background()

        # ตั้งค่าความหนาเส้นในโมเดล
        self.segment.outline_thickness = outline_thickness_val

        # แทนพื้นหลัง + เส้นขอบ
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

        # ฟิลเตอร์
        frame = self.filter_manager.apply_filter(frame, current_filter)

        # Apply virtual props (glasses, mustache, hat)
        frame = self._apply_props(frame)

        # sharpen (unsharp mask อ่อนๆ)
        if self.enable_sharpen:
            blur = cv2.GaussianBlur(frame, (0, 0), 0.8)
            frame = cv2.addWeighted(frame, 1.22, blur, -0.22, 0)

        return frame

    def _apply_props(self, frame: np.ndarray) -> np.ndarray:
        """Apply virtual props using MediaPipe Face Mesh (optimized)"""
        # Check if any props are enabled
        if not any(self.props_enabled.values()):
            return frame

        h, w = frame.shape[:2]

        # Frame skip optimization: only detect faces every N frames
        self.face_frame_counter += 1
        if self.face_frame_counter >= self.face_frame_skip:
            self.face_frame_counter = 0
            # Only convert to RGB when actually processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # Cache landmarks for next frames
                self.cached_landmarks = []
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark
                    landmarks = np.array([[p.x * w, p.y * h] for p in lm])
                    if not np.any(np.isnan(landmarks)):
                        self.cached_landmarks.append(landmarks)
            else:
                self.cached_landmarks = None

        # Use cached landmarks for rendering
        if self.cached_landmarks:
            for landmarks in self.cached_landmarks:

                # Apply glasses
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

                # Apply hat
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

                # Apply mustache
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

        # Apply logo
        if self.props_enabled.get("logo", False) and self.logo is not None:
            frame = overlay_transparent(frame, self.logo, 0, 10, overlay_size=self.logo_size)

        return frame

    @staticmethod
    def _numpy_to_qimage(img: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# ---------------- UI ----------------
class PhotoBoothUI(QWidget):
    """Main photobooth application window (high FPS + better UX)."""

    def __init__(self, camera_index: int = CAMERA_INDEX):
        super().__init__()

        # Core managers / model
        self.segment = create_segment()  # ใช้ U-Net (max_size=512)
        self.bg_manager = BackgroundManager(BG_ROOT)
        self.filter_manager = FilterManager()
        self.outline_manager = OutlineManager()  # มี Glow แล้ว

        # Photobooth mode variables
        self.photobooth_mode = False
        self.captured_images = []
        self.countdown_value = 0
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self._countdown_tick)

        # Store latest processed frame for capture (full resolution)
        self.latest_processed_frame = None

        # Threads
        self.camera = CameraWorker(cam_index=camera_index, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        self.processor = ProcessorWorker(self.segment, self.bg_manager, self.filter_manager, self.outline_manager)

        # UI
        self._init_ui()
        self.setStyleSheet(DARK_QSS)

        # Connect threads (PyQt5: ใช้ Qt.QueuedConnection)
        self.camera.frame_ready.connect(self._on_camera_frame, Qt.QueuedConnection)
        self.processor.processed_ready.connect(self._display_qimage, Qt.QueuedConnection)
        self.processor.fps_ready.connect(self._update_fps, Qt.QueuedConnection)

        self.camera.start()
        self.processor.start()

    # ---------- UI Building ----------
    def _init_ui(self):
        from PyQt5.QtWidgets import QColorDialog
        import os
        self.setWindowTitle("Photobooth")
        self.showMaximized()

        # Video preview
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 360)

        # ปุ่มถ่ายภาพ
        self.capture_btn = QPushButton("📸 ถ่าย 3 รูป")
        self.capture_btn.setFixedHeight(48)
        self.capture_btn.setStyleSheet("font-size: 18px; font-weight: bold; background: #3f7ae0; color: white; border-radius: 12px;")
        self.capture_btn.clicked.connect(self._start_photobooth_sequence)

        # Countdown overlay label
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("""
            background: rgba(0, 0, 0, 180);
            color: white;
            font-size: 120px;
            font-weight: bold;
            border-radius: 20px;
        """)
        self.countdown_label.hide()

        # Floating info (FPS)
        self.info_label = QLabel("FPS: --")
        self.info_label.setStyleSheet("padding:4px; background: rgba(0,0,0,0.7); border-radius: 8px;")
        self.info_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # --- Background Section ---
        bg_section = QVBoxLayout()
        bg_title = QLabel("🖼 Background")
        bg_title.setProperty("section", True)
        bg_title.setAlignment(Qt.AlignCenter)
        bg_section.addWidget(bg_title)
        bg_scroll = QScrollArea()
        bg_scroll.setWidgetResizable(True)
        bg_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        bg_content = QWidget()
        bg_grid = QGridLayout(bg_content)
        bg_grid.setSpacing(10)
        thumbnails = self.bg_manager.load_thumbnails(limit=60)
        for i, (path, pixmap) in enumerate(thumbnails):
            btn = QPushButton()
            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(THUMBNAIL_SIZE)
            btn.setFixedSize(*BUTTON_SIZE)
            btn.setToolTip(path)
            btn.clicked.connect(lambda checked=False, p=path: self.bg_manager.set_background(p))
            bg_grid.addWidget(btn, i // 3, i % 3)
        bg_scroll.setWidget(bg_content)
        bg_section.addWidget(bg_scroll)

        # --- Outline Section ---
        outline_section = QVBoxLayout()
        outline_title = QLabel("✒️ Stroke")
        outline_title.setProperty("section", True)
        outline_title.setAlignment(Qt.AlignCenter)
        outline_section.addWidget(outline_title)
        style_row = QHBoxLayout()
        self.stroke_btn = QPushButton("Stroke: ON")
        self.stroke_btn.setCheckable(True)
        self.stroke_btn.setChecked(True)
        self.stroke_btn.clicked.connect(self._toggle_stroke)
        style_row.addWidget(self.stroke_btn)
        outline_section.addLayout(style_row)
        outline_section.addWidget(self._make_line())
        color_label = QLabel("Color")
        color_label.setAlignment(Qt.AlignCenter)
        outline_section.addWidget(color_label)
        color_row = QHBoxLayout()
        self.color_btn = QPushButton()
        self._update_color_btn()
        self.color_btn.clicked.connect(self._open_color_picker)
        color_row.addWidget(self.color_btn)
        outline_section.addLayout(color_row)

        # --- Filter Section ---
        filter_section = QVBoxLayout()
        filter_title = QLabel("🎨 Filters")
        filter_title.setProperty("section", True)
        filter_title.setAlignment(Qt.AlignCenter)
        filter_section.addWidget(filter_title)
        filter_row = QHBoxLayout()
        for fname in self.filter_manager.FILTER_NAMES:
            btn = QPushButton(fname)
            btn.setFixedSize(120, 60)
            btn.clicked.connect(lambda checked=False, f=fname: self._set_filter(f))
            filter_row.addWidget(btn)
        filter_section.addLayout(filter_row)

        # --- Props Section ---
        props_section = QVBoxLayout()
        props_title = QLabel("🎭 Virtual Props")
        props_title.setProperty("section", True)
        props_title.setAlignment(Qt.AlignCenter)
        props_section.addWidget(props_title)

        props_grid = QGridLayout()
        props_grid.setSpacing(10)

        # Glasses button
        self.glasses_btn = QPushButton("👓 Glasses")
        self.glasses_btn.setCheckable(True)
        self.glasses_btn.setFixedSize(120, 60)
        self.glasses_btn.clicked.connect(self._toggle_glasses)
        props_grid.addWidget(self.glasses_btn, 0, 0)

        # Hat button
        self.hat_btn = QPushButton("🎩 Hat")
        self.hat_btn.setCheckable(True)
        self.hat_btn.setFixedSize(120, 60)
        self.hat_btn.clicked.connect(self._toggle_hat)
        props_grid.addWidget(self.hat_btn, 0, 1)

        # Mustache button
        self.mustache_btn = QPushButton("🥸 Mustache")
        self.mustache_btn.setCheckable(True)
        self.mustache_btn.setFixedSize(120, 60)
        self.mustache_btn.clicked.connect(self._toggle_mustache)
        props_grid.addWidget(self.mustache_btn, 1, 0)

        # Logo button
        self.logo_btn = QPushButton("🏢 Logo")
        self.logo_btn.setCheckable(True)
        self.logo_btn.setFixedSize(120, 60)
        self.logo_btn.clicked.connect(self._toggle_logo)
        props_grid.addWidget(self.logo_btn, 1, 1)

        props_section.addLayout(props_grid)

        # --- Assemble right panel ---
        right_panel = QVBoxLayout()
        title = QLabel("Photobooth")
        title.setProperty("hero", True)
        title.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(title)
        right_panel.addSpacing(10)
        right_panel.addLayout(bg_section)
        right_panel.addSpacing(16)
        right_panel.addLayout(outline_section)
        right_panel.addSpacing(16)
        right_panel.addLayout(filter_section)
        right_panel.addSpacing(16)
        right_panel.addLayout(props_section)
        right_panel.addStretch()

        # --- Main layout ---
        main_layout = QHBoxLayout()

        # Video area with countdown overlay
        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)

        # Stack video and countdown
        from PyQt5.QtWidgets import QStackedLayout
        video_stack = QWidget()
        video_stack_layout = QVBoxLayout(video_stack)
        video_stack_layout.setContentsMargins(0, 0, 0, 0)
        video_stack_layout.addWidget(self.video_label)
        video_stack_layout.addWidget(self.countdown_label)
        self.countdown_label.raise_()  # Bring to front

        video_wrap = QVBoxLayout()
        video_wrap.addWidget(video_stack, stretch=1)
        video_wrap.addWidget(self.capture_btn)
        video_wrap.addWidget(self._make_line())
        video_wrap.addWidget(self.info_label, alignment=Qt.AlignLeft)

        main_layout.addLayout(video_wrap, stretch=3)
        main_layout.addLayout(right_panel, stretch=2)
        self.setLayout(main_layout)

    def _make_line(self):
        line = QFrame()
        line.setObjectName("line")
        return line

    # ลบ _create_background_page, _create_outline_page, _create_filter_page, _create_navigation ไม่ต้องใช้แล้ว

    # ---------- Event handlers ----------
    @pyqtSlot(np.ndarray)
    def _on_camera_frame(self, frame: np.ndarray):
        settings = {
            "outline_style": self.outline_manager.current_style,
            "outline_thickness_val": OutlineManager.THICKNESS["Medium"],  # ใช้ Medium ตลอด
            "outline_color": self.outline_manager.current_color,
            "outline_opacity": 1.0,  # เต็มเสมอ
            "current_filter": self.filter_manager.current_filter,
        }
        self.processor.enqueue(frame, settings)

    @pyqtSlot(QImage)
    def _display_qimage(self, qimg: QImage):
        # Store full resolution image for capture
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        self.latest_processed_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # Display scaled version
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    @pyqtSlot(float)
    def _update_fps(self, fps: float):
        self.info_label.setText(f"FPS: {fps:0.1f}")

    # ---------- UI setters ----------
    def _set_outline_style(self, style: str):
        self.outline_manager.current_style = style
        print(f"✓ Outline style: {style}")

    def _set_outline_thickness(self, thickness: str):
        self.outline_manager.current_thickness = thickness
        print(f"✓ Outline thickness: {thickness}")

    def _set_outline_color(self, color: Tuple[int, int, int], color_name: str):
        self.outline_manager.current_color = color
        self.outline_manager.current_color_name = color_name
        print(f"✓ Outline color: {color_name}")

    def _set_filter(self, filter_name: str):
        self.filter_manager.current_filter = filter_name
        print(f"Filter: {filter_name}")

    def _toggle_stroke(self):
        if self.stroke_btn.isChecked():
            self.stroke_btn.setText("Stroke: ON")
            self._set_outline_style("Solid")
        else:
            self.stroke_btn.setText("Stroke: OFF")
            self._set_outline_style("None")

    def _open_color_picker(self):
        col = QColorDialog.getColor()
        if col.isValid():
            rgb = (col.red(), col.green(), col.blue())
            self._set_outline_color((rgb[2], rgb[1], rgb[0]), f"rgb{rgb}")
            self._update_color_btn()

    def _update_color_btn(self):
        # อัปเดตสีปุ่มให้ตรงกับ current_color
        c = self.outline_manager.current_color
        self.color_btn.setStyleSheet(f"background-color: rgb({c[2]}, {c[1]}, {c[0]}); min-width: 80px; min-height: 40px;")
        self.color_btn.setText("")

    def _start_photobooth_sequence(self):
        """Start 3-shot photobooth sequence"""
        if self.photobooth_mode:
            return  # Already in progress

        self.photobooth_mode = True
        self.captured_images = []
        self.capture_btn.setEnabled(False)
        self.capture_btn.setText("⏳ กำลังถ่ายภาพ...")

        # Start countdown for first shot
        self.countdown_value = 3
        self.countdown_label.setText(str(self.countdown_value))
        self.countdown_label.show()
        self.countdown_timer.start(1000)  # 1 second interval

    def _countdown_tick(self):
        """Handle countdown timer tick"""
        self.countdown_value -= 1

        if self.countdown_value > 0:
            self.countdown_label.setText(str(self.countdown_value))
        elif self.countdown_value == 0:
            # Capture!
            self.countdown_label.setText("📸")
            QTimer.singleShot(200, self._capture_one_image)
        else:
            # Check if we have all 3 images
            if len(self.captured_images) < 3:
                # Start countdown for next shot
                self.countdown_value = 3
                self.countdown_label.setText(str(self.countdown_value))
            else:
                # All done! Create photobooth strip
                self.countdown_timer.stop()
                self.countdown_label.hide()
                self._create_photobooth_strip()

    def _capture_one_image(self):
        """Capture one image from video"""
        if self.latest_processed_frame is not None:
            # Use full resolution processed frame
            self.captured_images.append(self.latest_processed_frame.copy())
            print(f"[Photobooth] Captured image {len(self.captured_images)}/3 (resolution: {self.latest_processed_frame.shape[1]}x{self.latest_processed_frame.shape[0]})")

    def _create_photobooth_strip(self):
        """Create final photobooth strip with 3 images"""
        from datetime import datetime
        from PyQt5.QtWidgets import QMessageBox

        if len(self.captured_images) < 3:
            QMessageBox.warning(self, "ข้อผิดพลาด", "ไม่สามารถถ่ายภาพครบ 3 รูปได้")
            self._reset_photobooth_mode()
            return

        # Create photobooth frame
        strip = create_photobooth_frame(self.captured_images)

        # Save
        out_dir = Path(__file__).resolve().parent / "output_images"
        out_dir.mkdir(exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"photobooth_strip_{now}.png"
        cv2.imwrite(str(out_path), strip)

        print(f"[Photobooth] Saved strip: {out_path}")
        QMessageBox.information(self, "สำเร็จ! 🎉", f"บันทึก Photobooth Strip แล้ว:\n{out_path}")

        self._reset_photobooth_mode()

    def _reset_photobooth_mode(self):
        """Reset photobooth mode"""
        self.photobooth_mode = False
        self.captured_images = []
        self.capture_btn.setEnabled(True)
        self.capture_btn.setText("📸 ถ่าย 3 รูป (Photobooth Strip)")

    def _toggle_glasses(self):
        self.processor.props_enabled["glasses"] = self.glasses_btn.isChecked()
        status = "ON" if self.glasses_btn.isChecked() else "OFF"
        print(f"Glasses: {status}")

    def _toggle_hat(self):
        self.processor.props_enabled["hat"] = self.hat_btn.isChecked()
        status = "ON" if self.hat_btn.isChecked() else "OFF"
        print(f"Hat: {status}")

    def _toggle_mustache(self):
        self.processor.props_enabled["mustache"] = self.mustache_btn.isChecked()
        status = "ON" if self.mustache_btn.isChecked() else "OFF"
        print(f"Mustache: {status}")

    def _toggle_logo(self):
        self.processor.props_enabled["logo"] = self.logo_btn.isChecked()
        status = "ON" if self.logo_btn.isChecked() else "OFF"
        print(f"Logo: {status}")

    # ---------- Cleanup ----------
    def closeEvent(self, event):
        try:
            self.camera.stop()
            self.processor.stop()
            self.camera.wait(1000)
            self.processor.wait(1000)
        finally:
            super().closeEvent(event)
