import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFrame, QColorDialog, QScrollArea
)

from managers.background_manager import BackgroundManager
from managers.filter_manager import FilterManager
from managers.outline_manager import OutlineManager
from managers.segment_manager import create_segment
from managers.face_manager import FaceManager

from core.ProcessorWorker import ProcessorWorker
from core.CameraWorker import CameraWorker

from ui.qss_theme import ORANGE_WHITE_QSS

# ---------------- UI/Camera Parameters ----------------
CAMERA_INDEX = 2
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 60

UI_SCALE = 0.75
def S(x: int) -> int:
    return int(x * UI_SCALE)

THUMBNAIL_SIZE = QSize(S(150), S(100))
BUTTON_SIZE = (S(160), S(110))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BG_ROOT = PROJECT_ROOT / "resources" / "backgrounds" / "photos"


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
        self.setWindowTitle("FIBOOTH - Display")
        self.setStyleSheet(ORANGE_WHITE_QSS)
        
        # Set window to be large and centered
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.6), int(screen.height() * 0.8))
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Video preview
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: black;
            border: 5px solid #ff8c00;
            border-radius: 15px;
        """)
        self.video_label.setMinimumSize(800, 600)
        
        # Countdown overlay
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("""
            background: rgba(0, 0, 0, 180);
            color: white;
            font-size: 150px;
            font-weight: bold;
            border-radius: 20px;
        """)
        self.countdown_label.hide()
        
        # FPS info
        self.info_label = QLabel("FPS: --")
        self.info_label.setStyleSheet("""
            padding: 10px 15px;
            background: rgba(255, 140, 0, 0.9);
            color: white;
            border-radius: 8px;
            font-weight: bold;
            font-size: 16px;
        """)
        
        # Stack video and countdown
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.countdown_label)
        self.countdown_label.raise_()
        
        layout.addWidget(video_container, stretch=1)
        layout.addWidget(self.info_label, alignment=Qt.AlignLeft)
        
        self.setLayout(layout)
    
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
        if value > 0:
            self.countdown_label.setText(str(value))
        else:
            self.countdown_label.setText("📸 SMILE!")
        self.countdown_label.show()
    
    def hide_countdown(self):
        """Hide countdown overlay"""
        self.countdown_label.hide()


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
        
        self.setWindowTitle("FIBOOTH - Controls")
        self.setStyleSheet(ORANGE_WHITE_QSS)
        
        # Size and position
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.35), int(screen.height() * 0.8))
        
        self._init_ui()
    
    def _init_ui(self):
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("FIBOOTH")
        title.setProperty("hero", True)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        subtitle = QLabel("KMUTT OPENHOUSE 2025")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #ff8c00; font-weight: 600;")
        main_layout.addWidget(subtitle)
        main_layout.addSpacing(20)
        
        # Capture button
        self.capture_btn = QPushButton("📸 ถ่ายภาพ")
        self.capture_btn.setFixedHeight(70)
        self.capture_btn.setStyleSheet("""
            font-size: 22px; font-weight: bold;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ff8c00, stop:1 #ff6600);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 10px 20px;
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
        
        # Add all control sections
        content_layout.addLayout(self._create_background_section())
        content_layout.addSpacing(20)
        content_layout.addLayout(self._create_outline_section())
        content_layout.addSpacing(20)
        content_layout.addLayout(self._create_props_section())
        content_layout.addSpacing(20)
        content_layout.addLayout(self._create_frame_section())
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        self.setLayout(main_layout)
    
    def _create_background_section(self):
        section = QVBoxLayout()
        
        title = QLabel("🖼 Background")
        title.setProperty("section", True)
        title.setAlignment(Qt.AlignCenter)
        section.addWidget(title)
        
        grid = QGridLayout()
        grid.setSpacing(S(10))
        
        thumbnails = self.bg_manager.load_thumbnails(limit=60)
        for i, (path, pixmap) in enumerate(thumbnails):
            btn = QPushButton()
            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(THUMBNAIL_SIZE)
            btn.setFixedSize(*BUTTON_SIZE)
            btn.setToolTip(path)
            btn.clicked.connect(lambda checked=False, p=path: self._on_bg_select(p))
            grid.addWidget(btn, i // 3, i % 3)
        
        section.addLayout(grid)
        return section
    
    def _create_outline_section(self):
        section = QVBoxLayout()
        
        title = QLabel("✒️ Stroke")
        title.setProperty("section", True)
        title.setAlignment(Qt.AlignCenter)
        section.addWidget(title)
        
        # Stroke toggle
        self.stroke_btn = QPushButton("Stroke: ON")
        self.stroke_btn.setCheckable(True)
        self.stroke_btn.setChecked(True)
        self.stroke_btn.clicked.connect(self._toggle_stroke)
        section.addWidget(self.stroke_btn)
        section.addWidget(self._make_line())
        
        # Color picker
        color_label = QLabel("Color")
        color_label.setAlignment(Qt.AlignCenter)
        section.addWidget(color_label)
        
        self.color_btn = QPushButton()
        self._update_color_btn()
        self.color_btn.clicked.connect(self._open_color_picker)
        section.addWidget(self.color_btn)
        
        return section
    
    def _create_props_section(self):
        section = QVBoxLayout()
        
        title = QLabel("🎭 Virtual Props")
        title.setProperty("section", True)
        title.setAlignment(Qt.AlignCenter)
        section.addWidget(title)
        
        grid = QGridLayout()
        grid.setSpacing(S(10))
        
        # Props buttons
        self.glasses_btn = QPushButton("👓 Glasses")
        self.glasses_btn.setCheckable(True)
        self.glasses_btn.setFixedSize(S(120), S(60))
        self.glasses_btn.clicked.connect(lambda: self.prop_toggled.emit("glasses", self.glasses_btn.isChecked()))
        grid.addWidget(self.glasses_btn, 0, 0)
        
        self.hat_btn = QPushButton("🎩 Hat")
        self.hat_btn.setCheckable(True)
        self.hat_btn.setFixedSize(S(120), S(60))
        self.hat_btn.clicked.connect(lambda: self.prop_toggled.emit("hat", self.hat_btn.isChecked()))
        grid.addWidget(self.hat_btn, 0, 1)
        
        self.mustache_btn = QPushButton("🥸 Mustache")
        self.mustache_btn.setCheckable(True)
        self.mustache_btn.setFixedSize(S(120), S(60))
        self.mustache_btn.clicked.connect(lambda: self.prop_toggled.emit("mustache", self.mustache_btn.isChecked()))
        grid.addWidget(self.mustache_btn, 1, 0)
        
        self.logo_btn = QPushButton("🏢 Logo")
        self.logo_btn.setCheckable(True)
        self.logo_btn.setChecked(True)
        self.logo_btn.setFixedSize(S(120), S(60))
        self.logo_btn.clicked.connect(lambda: self.prop_toggled.emit("logo", self.logo_btn.isChecked()))
        grid.addWidget(self.logo_btn, 1, 1)
        
        section.addLayout(grid)
        return section
    
    def _create_frame_section(self):
        section = QVBoxLayout()
        
        title = QLabel("🖼️ Frame Style")
        title.setProperty("section", True)
        title.setAlignment(Qt.AlignCenter)
        section.addWidget(title)
        
        # Frame preview
        self.frame_preview = QLabel()
        self.frame_preview.setAlignment(Qt.AlignCenter)
        self.frame_preview.setFixedSize(S(250), S(375))
        self.frame_preview.setStyleSheet("""
            border: 3px solid #ff8c00;
            border-radius: 8px;
            background-color: #fff;
        """)
        section.addWidget(self.frame_preview, alignment=Qt.AlignCenter)
        
        # Frame selection buttons
        grid = QGridLayout()
        grid.setSpacing(S(10))
        
        self.frame_gradient_btn = QPushButton("Gradient")
        self.frame_gradient_btn.setCheckable(True)
        self.frame_gradient_btn.setChecked(True)
        self.frame_gradient_btn.setFixedSize(S(120), S(60))
        self.frame_gradient_btn.clicked.connect(lambda: self._select_frame("fibooth_gradient"))
        grid.addWidget(self.frame_gradient_btn, 0, 0)
        
        self.frame_bold_btn = QPushButton("Bold")
        self.frame_bold_btn.setCheckable(True)
        self.frame_bold_btn.setFixedSize(S(120), S(60))
        self.frame_bold_btn.clicked.connect(lambda: self._select_frame("fibooth_bold"))
        grid.addWidget(self.frame_bold_btn, 0, 1)
        
        section.addLayout(grid)
        self._update_frame_preview("fibooth_gradient")
        
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
            self.stroke_btn.setText("Stroke: ON")
            self.outline_style_changed.emit("Solid")
        else:
            self.stroke_btn.setText("Stroke: OFF")
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
            f"min-width: {S(80)}px; min-height: {S(40)}px;"
        )
        self.color_btn.setText("")
    
    def _select_frame(self, frame_name: str):
        self.frame_gradient_btn.setChecked(frame_name == "fibooth_gradient")
        self.frame_bold_btn.setChecked(frame_name == "fibooth_bold")
        self._update_frame_preview(frame_name)
        self.frame_selected.emit(frame_name)
    
    def _update_frame_preview(self, frame_name: str):
        frame_files = {
            "fibooth_gradient": "frame_fibooth_gradient.png",
            "fibooth_bold": "frame_fibooth_bold.png"
        }
        frame_file = frame_files.get(frame_name, "frame_fibooth_gradient.png")
        frame_path = PROJECT_ROOT / "resources" / "frames" / frame_file
        
        if frame_path.exists():
            pixmap = QPixmap(str(frame_path))
            scaled_pixmap = pixmap.scaled(
                self.frame_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.frame_preview.setPixmap(scaled_pixmap)
        else:
            self.frame_preview.setText("No Preview")
    
    def set_capture_enabled(self, enabled: bool):
        """Enable/disable capture button"""
        self.capture_btn.setEnabled(enabled)
        if enabled:
            self.capture_btn.setText("📸 ถ่ายภาพ")
        else:
            self.capture_btn.setText("⏳ กำลังถ่าย...")


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
        self.current_frame_config = "fibooth_gradient"
        self.latest_processed_frame = None
        
        # Create windows
        self.display_window = DisplayWindow()
        self.control_window = ControlWindow(self.bg_manager, self.outline_manager)
        
        # Connect control signals
        self.control_window.capture_requested.connect(self._capture_single_photo)
        self.control_window.outline_style_changed.connect(self._set_outline_style)
        self.control_window.outline_color_changed.connect(self._set_outline_color)
        self.control_window.prop_toggled.connect(self._toggle_prop)
        self.control_window.frame_selected.connect(self._select_frame)
        
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
        
        # Position windows side by side
        self._position_windows()
        
        # Start threads
        self.camera.start()
        self.processor.start()
    
    def _position_windows(self):
        """Position windows side by side"""
        screen = QApplication.primaryScreen().availableGeometry()
        
        # Control window on left
        self.control_window.move(screen.left(), screen.top())
        
        # Display window on right
        control_right = self.control_window.x() + self.control_window.width()
        self.display_window.move(control_right + 10, screen.top())
    
    def show(self):
        """Show both windows"""
        self.control_window.show()
        self.display_window.show()
    
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
    
    def _select_frame(self, frame_name: str):
        self.current_frame_config = frame_name
        print(f"Frame style: {frame_name}")
    
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
                "ข้อผิดพลาด",
                "ไม่สามารถถ่ายภาพได้"
            )
            self._reset_photobooth_mode()
            return
        
        self.captured_images.append(self.latest_processed_frame.copy())
        print(f"[Photobooth] Captured image {len(self.captured_images)}")
        
        if len(self.captured_images) >= 3:
            self._create_photobooth_strip()
        else:
            out_dir = PROJECT_ROOT / "resources" / "output_images"
            out_dir.mkdir(exist_ok=True)
            now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out_path = out_dir / f"photo_{now}.png"
            
            print(f"[Photobooth] Saved photo: {out_path}")
            QMessageBox.information(
                self.control_window,
                "สำเร็จ! 🎉",
                f"บันทึกรูปภาพแล้ว:\n{out_path}\n\n"
                f"รูปที่ {len(self.captured_images)}/3 สำหรับ Photobooth Strip"
            )
    
    def _finish_capture(self):
        self.countdown_timer.stop()
        self.display_window.hide_countdown()
        self._reset_photobooth_mode()
    
    def _create_photobooth_strip(self):
        from datetime import datetime
        from PyQt5.QtWidgets import QMessageBox
        
        if len(self.captured_images) < 3:
            return
        
        strip = create_photobooth_frame(
            self.captured_images[-3:],
            self.current_frame_config
        )
        
        out_dir = PROJECT_ROOT / "output_images"
        out_dir.mkdir(exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"photobooth_strip_{now}.png"
        cv2.imwrite(str(out_path), strip)
        
        print(f"[Photobooth] Saved strip: {out_path}")
        QMessageBox.information(
            self.control_window,
            "สำเร็จ! 🎉 Photobooth Strip",
            f"บันทึก Photobooth Strip แล้ว:\n{out_path}\n\n"
            "ถ่ายรูปใหม่ต่อได้เลย!"
        )
        self.captured_images = []
    
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
    
    photobooth = PhotoBoothApp(camera_index=CAMERA_INDEX)
    photobooth.show()
    
    # Cleanup on exit
    app.aboutToQuit.connect(photobooth.cleanup)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()