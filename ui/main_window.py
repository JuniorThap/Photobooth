import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFrame, QColorDialog, QScrollArea, QSizePolicy
)

from managers.background_manager import BackgroundManager
from managers.filter_manager import FilterManager
from managers.outline_manager import OutlineManager
from managers.segment_manager import create_segment
from managers.face_manager import FaceManager

from core.ProcessorWorker import ProcessorWorker
from core.CameraWorker import CameraWorker

# ---------------- UI/Camera Parameters ----------------
CAMERA_INDEX = 2
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 60

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BG_ROOT = PROJECT_ROOT / "resources" / "backgrounds" / "photos"

# FRAB8 Graduation Theme Colors
FRAB8_GOLD = "#D4AF37"
FRAB8_NAVY = "#1A1F3A"
FRAB8_LIGHT_GOLD = "#F4E4C1"
FRAB8_DARK_NAVY = "#0F1423"
FRAB8_WHITE = "#FFFFFF"
FRAB8_ACCENT = "#8B7355"

# Graduation themed QSS
FRAB8_GRADUATION_QSS = f"""
QWidget {{
    background-color: {FRAB8_NAVY};
    color: {FRAB8_WHITE};
    font-family: 'Segoe UI', Arial, sans-serif;
}}

QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {FRAB8_GOLD}, stop:1 {FRAB8_ACCENT});
    color: {FRAB8_DARK_NAVY};
    border: 2px solid {FRAB8_GOLD};
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: bold;
    font-size: 14px;
}}

QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {FRAB8_LIGHT_GOLD}, stop:1 {FRAB8_GOLD});
    border: 2px solid {FRAB8_LIGHT_GOLD};
}}

QPushButton:pressed {{
    background: {FRAB8_ACCENT};
}}

QPushButton:checked {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #4A7C59, stop:1 #2D5A3D);
    border: 2px solid #5A9C6F;
    color: white;
}}

QPushButton:disabled {{
    background: #555555;
    color: #888888;
    border: 2px solid #666666;
}}

QLabel[hero="true"] {{
    font-size: 48px;
    font-weight: bold;
    color: {FRAB8_GOLD};
    font-family: 'Georgia', serif;
    text-transform: uppercase;
    letter-spacing: 3px;
}}

QLabel[section="true"] {{
    font-size: 20px;
    font-weight: bold;
    color: {FRAB8_GOLD};
    padding: 12px;
    background: {FRAB8_DARK_NAVY};
    border: 2px solid {FRAB8_GOLD};
    border-radius: 8px;
    margin: 8px 0;
}}

QLabel[subtitle="true"] {{
    font-size: 16px;
    color: {FRAB8_LIGHT_GOLD};
    font-weight: 600;
}}

QFrame#line {{
    background-color: {FRAB8_GOLD};
    max-height: 2px;
    margin: 10px 0;
}}

QScrollArea {{
    border: none;
    background: {FRAB8_NAVY};
}}

QScrollBar:vertical {{
    border: none;
    background: {FRAB8_DARK_NAVY};
    width: 12px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {FRAB8_GOLD};
    border-radius: 6px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {FRAB8_LIGHT_GOLD};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""

THUMBNAIL_SIZE = QSize(180, 120)
BUTTON_SIZE = (190, 130)


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


# ============================================================
# DISPLAY WINDOW - Shows only the video preview
# ============================================================
class DisplayWindow(QWidget):
    """Window that displays the video preview and countdown"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FRAB8 Graduation PhotoBooth - Display")
        self.setStyleSheet(FRAB8_GRADUATION_QSS)
        
        # Make fullscreen
        self.showFullScreen()
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header with graduation theme
        header = QLabel("🎓 FRAB8 GRADUATION 2025 🎓")
        header.setProperty("hero", True)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Video preview
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"""
            background-color: black;
            border: 6px solid {FRAB8_GOLD};
            border-radius: 20px;
        """)
        self.video_label.setScaledContents(False)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        # Countdown overlay
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet(f"""
            background: rgba(26, 31, 58, 220);
            color: {FRAB8_GOLD};
            font-size: 180px;
            font-weight: bold;
            border: 5px solid {FRAB8_GOLD};
            border-radius: 25px;
        """)
        self.countdown_label.hide()
        
        # QR Code display
        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignCenter)
        self.qr_label.setStyleSheet(f"""
            background: white;
            border: 5px solid {FRAB8_GOLD};
            border-radius: 15px;
            padding: 20px;
        """)
        self.qr_label.hide()
        
        self.qr_text_label = QLabel("Scan to Download Your Photos!")
        self.qr_text_label.setProperty("subtitle", True)
        self.qr_text_label.setAlignment(Qt.AlignCenter)
        self.qr_text_label.setStyleSheet(f"font-size: 24px; color: {FRAB8_GOLD}; margin-top: 10px;")
        self.qr_text_label.hide()
        
        # FPS info
        self.info_label = QLabel("FPS: --")
        self.info_label.setStyleSheet(f"""
            padding: 12px 18px;
            background: rgba(212, 175, 55, 0.9);
            color: {FRAB8_DARK_NAVY};
            border-radius: 10px;
            font-weight: bold;
            font-size: 16px;
        """)
        
        # Stack video and countdown in fixed container
        video_container = QWidget()
        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add video label to container
        video_layout.addWidget(self.video_label, stretch=1, alignment=Qt.AlignCenter)
        
        # Overlay countdown and QR on top of video
        self.countdown_label.setParent(self.video_label)
        self.countdown_label.setGeometry(0, 0, self.video_label.width(), self.video_label.height())
        
        self.qr_container = QWidget(self.video_label)
        qr_layout = QVBoxLayout(self.qr_container)
        qr_layout.addWidget(self.qr_label, alignment=Qt.AlignCenter)
        qr_layout.addWidget(self.qr_text_label, alignment=Qt.AlignCenter)
        self.qr_container.hide()
        
        layout.addWidget(video_container, stretch=1)
        
        # Footer with FPS and instruction
        footer_layout = QHBoxLayout()
        footer_layout.addWidget(self.info_label)
        footer_layout.addStretch()
        
        instruction = QLabel("Press ESC to exit fullscreen")
        instruction.setStyleSheet(f"color: {FRAB8_LIGHT_GOLD}; font-size: 14px;")
        footer_layout.addWidget(instruction)
        
        layout.addLayout(footer_layout)
        
        self.setLayout(layout)
    
    def resizeEvent(self, event):
        """Handle window resize to reposition overlays"""
        super().resizeEvent(event)
        if hasattr(self, 'countdown_label'):
            # Center countdown on video label
            video_rect = self.video_label.geometry()
            countdown_size = min(video_rect.width(), video_rect.height()) * 0.8
            self.countdown_label.setGeometry(
                int(video_rect.width() * 0.1),
                int(video_rect.height() * 0.3),
                int(video_rect.width() * 0.8),
                int(video_rect.height() * 0.4)
            )
            
        if hasattr(self, 'qr_container'):
            # Center QR container
            video_rect = self.video_label.geometry()
            self.qr_container.setGeometry(
                int(video_rect.width() * 0.2),
                int(video_rect.height() * 0.2),
                int(video_rect.width() * 0.6),
                int(video_rect.height() * 0.6)
            )
    
    def keyPressEvent(self, event):
        """Handle ESC key to exit fullscreen"""
        if event.key() == Qt.Key_Escape:
            self.showNormal()
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
    
    def update_frame(self, qimg: QImage):
        """Update the video display"""
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)
    
    def update_fps(self, fps: float):
        """Update FPS display"""
        self.info_label.setText(f"FPS: {fps:0.1f}")
    
    def show_countdown(self, value: int):
        """Show countdown number"""
        self.qr_label.hide()
        self.qr_text_label.hide()
        if value > 0:
            self.countdown_label.setText(str(value))
        else:
            self.countdown_label.setText("📸 SMILE!")
        self.countdown_label.show()
    
    def hide_countdown(self):
        """Hide countdown overlay"""
        self.countdown_label.hide()
    
    def show_qr_code(self, qr_image=None):
        """Display QR code for photo download"""
        self.countdown_label.hide()
        
        if qr_image is not None:
            # If QR code image is provided, display it
            pixmap = QPixmap.fromImage(qr_image)
            scaled = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.qr_label.setPixmap(scaled)
            self.qr_label.setFixedSize(400, 400)
        else:
            # Placeholder text if no QR library available
            self.qr_label.setText("QR Code Here\n\n(QR library not available)")
            self.qr_label.setFixedSize(400, 400)
            self.qr_label.setStyleSheet(f"""
                background: white;
                border: 5px solid {FRAB8_GOLD};
                border-radius: 15px;
                padding: 60px;
                font-size: 20px;
                color: black;
            """)
        
        self.qr_container.show()
        
        # Auto-hide after 10 seconds
        QTimer.singleShot(10000, self.hide_qr_code)
    
    def hide_qr_code(self):
        """Hide QR code display"""
        self.qr_container.hide()


# ============================================================
# CONTROL WINDOW - Shows all controls and settings
# ============================================================
class ControlWindow(QWidget):
    """Window with all control buttons and settings"""
    
    # Signals to communicate with main app
    capture_requested = pyqtSignal()
    background_changed = pyqtSignal(str)
    outline_style_changed = pyqtSignal(str)
    outline_color_changed = pyqtSignal(tuple, str)
    prop_toggled = pyqtSignal(str, bool)
    frame_selected = pyqtSignal(str)
    
    def __init__(self, bg_manager, outline_manager):
        super().__init__()
        self.bg_manager = bg_manager
        self.outline_manager = outline_manager
        
        self.setWindowTitle("FRAB8 Graduation PhotoBooth - Controls")
        self.setStyleSheet(FRAB8_GRADUATION_QSS)
        
        # Make fullscreen
        self.showFullScreen()
        
        self._init_ui()
    
    def keyPressEvent(self, event):
        """Handle ESC key to exit fullscreen"""
        if event.key() == Qt.Key_Escape:
            self.showNormal()
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
    
    def _init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)
        
        # Title
        title = QLabel("🎓 FRAB8")
        title.setProperty("hero", True)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        subtitle = QLabel("GRADUATION PHOTOBOOTH 2025")
        subtitle.setProperty("subtitle", True)
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        
        # Large capture button
        self.capture_btn = QPushButton("📸 CAPTURE PHOTO")
        self.capture_btn.setMinimumHeight(90)
        self.capture_btn.setStyleSheet(f"""
            font-size: 28px; 
            font-weight: bold;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {FRAB8_GOLD}, stop:1 {FRAB8_ACCENT});
            color: {FRAB8_DARK_NAVY};
            border: 4px solid {FRAB8_GOLD};
            border-radius: 20px;
            padding: 20px;
        """)
        self.capture_btn.clicked.connect(self.capture_requested.emit)
        main_layout.addWidget(self.capture_btn)
        main_layout.addWidget(self._make_line())
        
        # Create scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(25)
        
        # Add all control sections
        content_layout.addLayout(self._create_background_section())
        content_layout.addLayout(self._create_outline_section())
        content_layout.addLayout(self._create_props_section())
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll, stretch=1)
        
        # Footer
        footer = QLabel("Press ESC to exit fullscreen | F11 to toggle")
        footer.setStyleSheet(f"color: {FRAB8_LIGHT_GOLD}; font-size: 12px; padding: 10px;")
        footer.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer)
        
        self.setLayout(main_layout)
    
    def _create_background_section(self):
        section = QVBoxLayout()
        section.setSpacing(15)
        
        title = QLabel("🖼️ BACKGROUNDS")
        title.setProperty("section", True)
        title.setAlignment(Qt.AlignCenter)
        section.addWidget(title)
        
        # Video background note
        note = QLabel("💡 Tip: Video backgrounds supported!")
        note.setStyleSheet(f"color: {FRAB8_LIGHT_GOLD}; font-size: 13px; padding: 8px;")
        note.setAlignment(Qt.AlignCenter)
        section.addWidget(note)
        
        grid = QGridLayout()
        grid.setSpacing(15)
        grid.setContentsMargins(10, 10, 10, 10)
        
        thumbnails = self.bg_manager.load_thumbnails(limit=60)
        cols = 3
        for i, (path, pixmap) in enumerate(thumbnails):
            btn = QPushButton()
            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(THUMBNAIL_SIZE)
            btn.setFixedSize(*BUTTON_SIZE)
            btn.setToolTip(path)
            btn.clicked.connect(lambda checked=False, p=path: self._on_bg_select(p))
            grid.addWidget(btn, i // cols, i % cols)
        
        section.addLayout(grid)
        return section
    
    def _create_outline_section(self):
        section = QVBoxLayout()
        section.setSpacing(15)
        
        title = QLabel("✏️ OUTLINE STYLE")
        title.setProperty("section", True)
        title.setAlignment(Qt.AlignCenter)
        section.addWidget(title)
        
        # Stroke toggle
        self.stroke_btn = QPushButton("Outline: ON")
        self.stroke_btn.setCheckable(True)
        self.stroke_btn.setChecked(True)
        self.stroke_btn.setMinimumHeight(50)
        self.stroke_btn.setStyleSheet(f"font-size: 16px;")
        self.stroke_btn.clicked.connect(self._toggle_stroke)
        section.addWidget(self.stroke_btn)
        
        # Color picker
        color_label = QLabel("Outline Color")
        color_label.setAlignment(Qt.AlignCenter)
        color_label.setStyleSheet(f"font-size: 15px; color: {FRAB8_LIGHT_GOLD}; margin-top: 10px;")
        section.addWidget(color_label)
        
        self.color_btn = QPushButton()
        self.color_btn.setMinimumHeight(60)
        self._update_color_btn()
        self.color_btn.clicked.connect(self._open_color_picker)
        section.addWidget(self.color_btn)
        
        return section
    
    def _create_props_section(self):
        section = QVBoxLayout()
        section.setSpacing(15)
        
        title = QLabel("🎭 VIRTUAL PROPS")
        title.setProperty("section", True)
        title.setAlignment(Qt.AlignCenter)
        section.addWidget(title)
        
        grid = QGridLayout()
        grid.setSpacing(15)
        grid.setContentsMargins(10, 10, 10, 10)
        
        # Props buttons with larger size
        btn_width, btn_height = 180, 80
        
        self.glasses_btn = QPushButton("👓 Glasses")
        self.glasses_btn.setCheckable(True)
        self.glasses_btn.setFixedSize(btn_width, btn_height)
        self.glasses_btn.setStyleSheet("font-size: 16px;")
        self.glasses_btn.clicked.connect(lambda: self.prop_toggled.emit("glasses", self.glasses_btn.isChecked()))
        grid.addWidget(self.glasses_btn, 0, 0)
        
        self.hat_btn = QPushButton("🎓 Grad Cap")
        self.hat_btn.setCheckable(True)
        self.hat_btn.setFixedSize(btn_width, btn_height)
        self.hat_btn.setStyleSheet("font-size: 16px;")
        self.hat_btn.clicked.connect(lambda: self.prop_toggled.emit("hat", self.hat_btn.isChecked()))
        grid.addWidget(self.hat_btn, 0, 1)
        
        self.mustache_btn = QPushButton("🥸 Mustache")
        self.mustache_btn.setCheckable(True)
        self.mustache_btn.setFixedSize(btn_width, btn_height)
        self.mustache_btn.setStyleSheet("font-size: 16px;")
        self.mustache_btn.clicked.connect(lambda: self.prop_toggled.emit("mustache", self.mustache_btn.isChecked()))
        grid.addWidget(self.mustache_btn, 1, 0)
        
        self.logo_btn = QPushButton("🏫 School Logo")
        self.logo_btn.setCheckable(True)
        self.logo_btn.setChecked(True)
        self.logo_btn.setFixedSize(btn_width, btn_height)
        self.logo_btn.setStyleSheet("font-size: 16px;")
        self.logo_btn.clicked.connect(lambda: self.prop_toggled.emit("logo", self.logo_btn.isChecked()))
        grid.addWidget(self.logo_btn, 1, 1)
        
        section.addLayout(grid)
        return section
    
    def _make_line(self):
        line = QFrame()
        line.setObjectName("line")
        return line
    
    def _on_bg_select(self, path):
        self.bg_manager.set_background(path)
        self.background_changed.emit(path)
    
    def _toggle_stroke(self):
        if self.stroke_btn.isChecked():
            self.stroke_btn.setText("Outline: ON")
            self.outline_style_changed.emit("Solid")
        else:
            self.stroke_btn.setText("Outline: OFF")
            self.outline_style_changed.emit("None")
    
    def _open_color_picker(self):
        col = QColorDialog.getColor()
        if col.isValid():
            rgb = (col.red(), col.green(), col.blue())
            bgr = (rgb[2], rgb[1], rgb[0])
            self.outline_manager.current_color = bgr
            self.outline_color_changed.emit(bgr, f"rgb{rgb}")
            self._update_color_btn()
    
    def _update_color_btn(self):
        c = self.outline_manager.current_color
        self.color_btn.setStyleSheet(
            f"background-color: rgb({c[2]}, {c[1]}, {c[0]}); "
            f"border: 3px solid {FRAB8_GOLD}; "
            f"border-radius: 12px; "
            f"min-height: 60px;"
        )
        self.color_btn.setText("")
    
    def set_capture_enabled(self, enabled: bool):
        """Enable/disable capture button"""
        self.capture_btn.setEnabled(enabled)
        if enabled:
            self.capture_btn.setText("📸 CAPTURE PHOTO")
        else:
            self.capture_btn.setText("⏳ Processing...")


# ============================================================
# MAIN APPLICATION - Coordinates both windows
# ============================================================
class PhotoBoothApp(QWidget):
    """Main application that manages both windows"""
    
    def __init__(self, camera_index: int = CAMERA_INDEX):
        super().__init__()
        
        # Core managers
        self.segment = create_segment()
        self.bg_manager = BackgroundManager(BG_ROOT)
        self.filter_manager = FilterManager()
        self.outline_manager = OutlineManager()
        self.face_manager = FaceManager()
        
        # Photobooth state
        self.photobooth_mode = False
        self.captured_images = []
        self.countdown_value = 0
        self.latest_processed_frame = None
        self.latest_photo_path = None
        
        # Create windows
        self.display_window = DisplayWindow()
        self.control_window = ControlWindow(self.bg_manager, self.outline_manager)
        
        # Connect control signals
        self.control_window.capture_requested.connect(self._capture_single_photo)
        self.control_window.outline_style_changed.connect(self._set_outline_style)
        self.control_window.outline_color_changed.connect(self._set_outline_color)
        self.control_window.prop_toggled.connect(self._toggle_prop)
        
        # Countdown timer
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self._countdown_tick)
        
        # Create threads
        self.camera = CameraWorker(
            cam_index=camera_index,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            target_fps=TARGET_FPS
        )
        self.processor = ProcessorWorker(
            PROJECT_ROOT,
            self.segment,
            self.bg_manager,
            self.filter_manager,
            self.outline_manager,
            self.face_manager
        )
        
        # Connect threads
        self.camera.frame_ready.connect(self._on_camera_frame, Qt.QueuedConnection)
        self.processor.processed_ready.connect(self._display_qimage, Qt.QueuedConnection)
        self.processor.fps_ready.connect(self._update_fps, Qt.QueuedConnection)
        
        # Position windows on different screens if available
        self._position_windows()
        
        # Start threads
        self.camera.start()
        self.processor.start()
    
    def _position_windows(self):
        """Position windows on different screens - Control on Monitor 1, Display on Monitor 2"""
        desktop = QApplication.desktop()
        screens = QApplication.screens()
        
        print(f"[INFO] Available monitors: {len(screens)}")
        
        if len(screens) >= 2:
            # Monitor 1 (Primary) - Control Window
            screen1_geometry = screens[0].geometry()
            print(f"[INFO] Monitor 1 (Control): {screen1_geometry.width()}x{screen1_geometry.height()} at ({screen1_geometry.x()}, {screen1_geometry.y()})")
            
            # Monitor 2 (Secondary) - Display Window
            screen2_geometry = screens[1].geometry()
            print(f"[INFO] Monitor 2 (Display): {screen2_geometry.width()}x{screen2_geometry.height()} at ({screen2_geometry.x()}, {screen2_geometry.y()})")
            
            # Move control window to monitor 1
            self.control_window.move(screen1_geometry.x(), screen1_geometry.y())
            self.control_window.resize(screen1_geometry.width(), screen1_geometry.height())
            
            # Move display window to monitor 2
            self.display_window.move(screen2_geometry.x(), screen2_geometry.y())
            self.display_window.resize(screen2_geometry.width(), screen2_geometry.height())
            
            print("[INFO] Windows positioned on separate monitors")
        else:
            # Single screen fallback - show side by side
            print("[INFO] Only one monitor detected - using split screen mode")
            screen = screens[0].geometry()
            half_width = screen.width() // 2
            
            # Control on left half
            self.control_window.move(screen.x(), screen.y())
            self.control_window.resize(half_width, screen.height())
            
            # Display on right half
            self.display_window.move(screen.x() + half_width, screen.y())
            self.display_window.resize(half_width, screen.height())
    
    def show(self):
        """Show both windows on separate monitors"""
        # Position first, then show
        self._position_windows()
        
        # Show windows
        self.control_window.showFullScreen()
        self.display_window.showFullScreen()
        
        print("[INFO] Both windows displayed")
    
    @pyqtSlot(np.ndarray)
    def _on_camera_frame(self, frame: np.ndarray):
        settings = {
            "outline_style": self.outline_manager.current_style,
            "outline_thickness_val": OutlineManager.THICKNESS["Medium"],
            "outline_color": self.outline_manager.current_color,
            "outline_opacity": 1.0,
        }
        self.processor.enqueue(frame, settings)
    
    @pyqtSlot(QImage)
    def _display_qimage(self, qimg: QImage):
        # Store processed frame
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        self.latest_processed_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
        # Update display
        self.display_window.update_frame(qimg)
    
    @pyqtSlot(float)
    def _update_fps(self, fps: float):
        self.display_window.update_fps(fps)
    
    def _set_outline_style(self, style: str):
        self.outline_manager.current_style = style
        print(f"✓ Outline style: {style}")
    
    def _set_outline_color(self, color: Tuple[int, int, int], color_name: str):
        self.outline_manager.current_color = color
        self.outline_manager.current_color_name = color_name
        print(f"✓ Outline color: {color_name}")
    
    def _toggle_prop(self, prop_name: str, enabled: bool):
        self.processor.props_enabled[prop_name] = enabled
        print(f"{prop_name.capitalize()}: {'ON' if enabled else 'OFF'}")
    
    def _capture_single_photo(self):
        if self.photobooth_mode:
            return
        
        self.photobooth_mode = True
        self.control_window.set_capture_enabled(False)
        
        self.countdown_value = 3
        self.display_window.show_countdown(self.countdown_value)
        self.countdown_timer.start(1000)
    
    def _countdown_tick(self):
        self.countdown_value -= 1
        if self.countdown_value > 0:
            self.display_window.show_countdown(self.countdown_value)
        elif self.countdown_value == 0:
            self.display_window.show_countdown(0)
            QTimer.singleShot(300, self._save_single_image)
            QTimer.singleShot(800, self._finish_capture)
    
    def _save_single_image(self):
        from datetime import datetime
        from PyQt5.QtWidgets import QMessageBox
        
        if self.latest_processed_frame is None:
            QMessageBox.warning(
                self.control_window,
                "Error",
                "Unable to capture photo"
            )
            self._reset_photobooth_mode()
            return
        
        self.captured_images.append(self.latest_processed_frame.copy())
        print(f"[Photobooth] Captured image {len(self.captured_images)}")
        
        # Save individual photo
        out_dir = PROJECT_ROOT / "resources" / "output_images"
        out_dir.mkdir(exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = out_dir / f"FRAB8_Grad_{now}.png"
        cv2.imwrite(str(out_path), self.latest_processed_frame)
        
        self.latest_photo_path = str(out_path)
        print(f"[Photobooth] Saved photo: {out_path}")
    
    def _finish_capture(self):
        self.countdown_timer.stop()
        self.display_window.hide_countdown()
        
        # Generate and show QR code
        self._generate_qr_code()
        
        self._reset_photobooth_mode()
    
    def _generate_qr_code(self):
        """Generate QR code for photo download"""
        try:
            # Try to import qrcode library
            import qrcode
            from io import BytesIO
            
            if self.latest_photo_path:
                # Create URL or path for photo download
                # You can replace this with your actual server URL
                download_url = f"http://yourserver.com/download/{Path(self.latest_photo_path).name}"
                
                # Generate QR code
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                qr.add_data(download_url)
                qr.make(fit=True)
                
                # Create QR code image
                qr_img = qr.make_image(fill_color="black", back_color="white")
                
                # Convert PIL image to QImage
                buffer = BytesIO()
                qr_img.save(buffer, format='PNG')
                buffer.seek(0)
                
                qimage = QImage()
                qimage.loadFromData(buffer.read())
                
                self.display_window.show_qr_code(qimage)
            else:
                # Show placeholder if no photo path
                self.display_window.show_qr_code(None)
                
        except ImportError:
            print("[INFO] qrcode library not installed. Install with: pip install qrcode[pil]")
            # Show placeholder QR code
            self.display_window.show_qr_code(None)
        except Exception as e:
            print(f"[ERROR] Failed to generate QR code: {e}")
            self.display_window.show_qr_code(None)
    
    def _reset_photobooth_mode(self):
        self.photobooth_mode = False
        self.control_window.set_capture_enabled(True)
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.camera.stop()
            self.processor.stop()
            self.camera.wait(1000)
            self.processor.wait(1000)
        except Exception as e:
            print(f"Cleanup error: {e}")


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    import sys
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    photobooth = PhotoBoothApp(camera_index=CAMERA_INDEX)
    photobooth.show()
    
    # Cleanup on exit
    app.aboutToQuit.connect(photobooth.cleanup)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()