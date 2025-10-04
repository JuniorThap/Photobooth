import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QStackedWidget, QScrollArea, QGridLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QTimer, QSize

from managers.background_manager import BackgroundManager
from managers.filter_manager import FilterManager
from managers.outline_manager import OutlineManager
from managers.segment_manager import create_segment

BG_ROOT = Path(r"C:\USERS\THAPA\DOCUMENTS\AIWORKS\PHOTOBOOTH\BACKGROUNDS")
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FRAME_UPDATE_INTERVAL = 30
THUMBNAIL_SIZE = QSize(150, 100)
BUTTON_SIZE = (160, 110)

DARK_STYLE = """ ... your stylesheet here ... """

class PhotoBoothUI(QWidget):
    """Main photobooth application window."""
    
    def __init__(self):
        super().__init__()
        self.segment = create_segment()
        self.bg_manager = BackgroundManager(BG_ROOT)
        self.filter_manager = FilterManager()
        self.outline_manager = OutlineManager()
        
        self._init_ui()
        self._init_camera()
        
        # Apply dark theme
        style_path = Path(__file__).parent / "styles" / "dark.qss"
        self.setStyleSheet(self.load_stylesheet(str(style_path)))
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Photobooth")
        self.showMaximized()
        
        # Video preview
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        
        # Right panel with stacked pages
        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(self._create_background_page())
        self.right_stack.addWidget(self._create_outline_page())
        self.right_stack.addWidget(self._create_filter_page())
        
        # Navigation
        nav_layout = self._create_navigation()
        
        # Assemble right panel
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.right_stack)
        right_panel.addLayout(nav_layout)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label, stretch=3)
        main_layout.addLayout(right_panel, stretch=1)
        self.setLayout(main_layout)
    
    def load_stylesheet(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()


    def _create_background_page(self) -> QWidget:
        """Create the background selection page."""
        page = QWidget()
        
        # Use scroll area for many backgrounds
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QGridLayout()
        layout.setSpacing(10)
        
        thumbnails = self.bg_manager.load_thumbnails()
        for i, (path, pixmap) in enumerate(thumbnails):
            btn = QPushButton()
            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(THUMBNAIL_SIZE)
            btn.setFixedSize(*BUTTON_SIZE)
            btn.setToolTip(path)
            btn.clicked.connect(lambda checked, p=path: self.bg_manager.set_background(p))
            layout.addWidget(btn, i // 2, i % 2)
        
        content.setLayout(layout)
        scroll.setWidget(content)
        
        # 🔹 Add title label
        title = QLabel("<h1>🖼 Background</h1>")
        title.setAlignment(Qt.AlignCenter)

        page_layout = QVBoxLayout()
        page_layout.addWidget(title)
        page_layout.addWidget(scroll)
        page.setLayout(page_layout)
        
        return page
    
    def _create_outline_page(self) -> QWidget:
        """Create the outline settings page - simple and clear for users."""
        page = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # ON/OFF Section - Very obvious
        on_off_label = QLabel("<h1>OUTLINE</h1>")
        on_off_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(on_off_label)
        
        on_off_layout = QHBoxLayout()
        on_off_layout.setSpacing(15)
        
        for style in OutlineManager.STYLES:
            btn = QPushButton(style.upper())
            btn.setFixedSize(150, 100)
            
            if style == "None":
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #8B0000;
                        color: white;
                        font-size: 24px;
                        font-weight: bold;
                        border: 3px solid #A00000;
                        border-radius: 12px;
                    }
                    QPushButton:hover {
                        background-color: #A00000;
                        border: 3px solid #B00000;
                    }
                """)
            else:  # Solid
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #006400;
                        color: white;
                        font-size: 24px;
                        font-weight: bold;
                        border: 3px solid #008000;
                        border-radius: 12px;
                    }
                    QPushButton:hover {
                        background-color: #008000;
                        border: 3px solid #00A000;
                    }
                """)
            
            btn.clicked.connect(lambda checked, s=style: self._set_outline_style(s))
            on_off_layout.addWidget(btn)
        
        main_layout.addLayout(on_off_layout)
        
        # Divider
        main_layout.addSpacing(20)
        
        # Thickness section - Big and clear
        thickness_label = QLabel("<h2>THICKNESS</h2>")
        thickness_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(thickness_label)
        
        thickness_layout = QHBoxLayout()
        thickness_layout.setSpacing(15)
        
        thickness_display = {
            "Thin": "│",
            "Medium": "┃┃",
            "Thick": "███"
        }
        
        for thickness_name in OutlineManager.THICKNESS.keys():
            btn = QPushButton(f"{thickness_display[thickness_name]}\n\n{thickness_name.upper()}")
            btn.setFixedSize(100, 90)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d2d2d;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    border: 3px solid #4d4d4d;
                    border-radius: 12px;
                }
                QPushButton:hover {
                    background-color: #3d3d3d;
                    border: 3px solid #6d6d6d;
                }
            """)
            btn.clicked.connect(lambda checked, t=thickness_name: self._set_outline_thickness(t))
            thickness_layout.addWidget(btn)
        
        main_layout.addLayout(thickness_layout)
        
        # Divider
        main_layout.addSpacing(20)
        
        # Color section - Large color blocks
        color_label = QLabel("<h2>COLOR</h2>")
        color_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(color_label)
        
        color_grid = QGridLayout()
        color_grid.setSpacing(12)
        
        color_items = list(OutlineManager.COLORS.items())
        
        for i, (color_name, bgr) in enumerate(color_items):
            btn = QPushButton(color_name.upper())
            btn.setFixedSize(100, 70)
            
            text_color = 'black' if color_name in ['White', 'Yellow'] else 'white'
            
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({bgr[2]}, {bgr[1]}, {bgr[0]}); 
                    color: {text_color};
                    font-weight: bold;
                    font-size: 14px;
                    border: 4px solid rgba(0, 0, 0, 0.3);
                    border-radius: 12px;
                }}
                QPushButton:hover {{
                    border: 4px solid white;
                }}
            """)
            btn.clicked.connect(lambda checked, c=bgr, n=color_name: self._set_outline_color(c, n))
            
            row = i // 3
            col = i % 3
            color_grid.addWidget(btn, row, col)
        
        main_layout.addLayout(color_grid)
        main_layout.addStretch()
        
        page.setLayout(main_layout)
        return page
    
    def _create_filter_page(self) -> QWidget:
        """Create the filter selection page with preview images."""
        page = QWidget()
        
        # Use scroll area for filters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        content = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("<h1>🎨 Filters</h1>")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Create a sample image for filter previews
        sample_img = self._create_filter_preview_images()
        
        # Filter icons/emojis for better representation
        filter_info = {
            "Normal": {"icon": "📷", "desc": "Original"},
            "Gray": {"icon": "⚫", "desc": "Grayscale"},
            "Sepia": {"icon": "🟤", "desc": "Vintage"},
            "Cartoon": {"icon": "🎨", "desc": "Comic"},
            "Beauty": {"icon": "✨", "desc": "Smooth"}
        }
        
        for filter_name in FilterManager.FILTER_NAMES:
            # Container for each filter option
            container = QWidget()
            container.setStyleSheet("""
                QWidget {
                    background-color: #252525;
                    border-radius: 12px;
                    padding: 8px;
                }
                QWidget:hover {
                    background-color: #2d2d2d;
                }
            """)
            
            h_layout = QHBoxLayout()
            h_layout.setContentsMargins(10, 10, 10, 10)
            h_layout.setSpacing(15)
            
            # Preview image
            preview_label = QLabel()
            if sample_img is not None:
                filtered = self.filter_manager.apply_filter(sample_img.copy(), filter_name)
                icon_pixmap = self._numpy_to_pixmap(filtered)
                preview_label.setPixmap(icon_pixmap.scaled(120, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            preview_label.setFixedSize(120, 80)
            preview_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #3d3d3d;
                    border-radius: 8px;
                    background-color: #1a1a1a;
                }
            """)
            
            # Text info
            text_layout = QVBoxLayout()
            text_layout.setSpacing(5)
            
            info = filter_info.get(filter_name, {"icon": "🔲", "desc": filter_name})
            
            name_label = QLabel(f"{info['icon']} {filter_name}")
            name_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
            
            desc_label = QLabel(info['desc'])
            desc_label.setStyleSheet("font-size: 12px; color: #aaaaaa;")
            
            text_layout.addWidget(name_label)
            text_layout.addWidget(desc_label)
            text_layout.addStretch()
            
            # Apply button
            apply_btn = QPushButton("Apply")
            apply_btn.setFixedSize(80, 60)
            apply_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d5aa6;
                    color: white;
                    font-size: 13px;
                    font-weight: bold;
                    border-radius: 8px;
                    border: 2px solid #3d6ab6;
                }
                QPushButton:hover {
                    background-color: #3d6ab6;
                    border: 2px solid #4d7ac6;
                }
                QPushButton:pressed {
                    background-color: #1d4a96;
                }
            """)
            apply_btn.clicked.connect(lambda checked, f=filter_name: self._set_filter(f))
            
            # Add to horizontal layout
            h_layout.addWidget(preview_label)
            h_layout.addLayout(text_layout, stretch=1)
            h_layout.addWidget(apply_btn)
            
            container.setLayout(h_layout)
            main_layout.addWidget(container)
        
        main_layout.addStretch()
        content.setLayout(main_layout)
        scroll.setWidget(content)
        
        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        page.setLayout(page_layout)
        
        return page
    
    def _create_filter_preview_images(self) -> Optional[np.ndarray]:
        """Create or capture a sample image for filter previews."""
        # Try to get current frame from camera
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize to thumbnail size
                frame = cv2.resize(frame, (THUMBNAIL_SIZE.width(), THUMBNAIL_SIZE.height()))
                return frame
        
        # Fallback: create a gradient image as sample
        img = np.zeros((THUMBNAIL_SIZE.height(), THUMBNAIL_SIZE.width(), 3), dtype=np.uint8)
        for i in range(THUMBNAIL_SIZE.height()):
            color = int(255 * i / THUMBNAIL_SIZE.height())
            img[i, :] = [color, 128, 255 - color]
        
        return img
    
    def _numpy_to_pixmap(self, img: np.ndarray) -> QPixmap:
        """Convert numpy array (BGR) to QPixmap."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)
    
    def _create_navigation(self) -> QHBoxLayout:
        """Create navigation buttons for page switching."""
        self.prev_btn = QPushButton("⬆ Previous")
        self.next_btn = QPushButton("⬇ Next")
        
        # Style navigation buttons
        nav_style = """
            QPushButton {
                background-color: #2d5aa6;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
                border-radius: 8px;
                border: 2px solid #3d6ab6;
            }
            QPushButton:hover {
                background-color: #3d6ab6;
                border: 2px solid #4d7ac6;
            }
            QPushButton:pressed {
                background-color: #1d4a96;
            }
        """
        
        self.prev_btn.setStyleSheet(nav_style)
        self.next_btn.setStyleSheet(nav_style)
        self.prev_btn.setFixedHeight(50)
        self.next_btn.setFixedHeight(50)
        
        self.prev_btn.clicked.connect(self._show_prev_page)
        self.next_btn.clicked.connect(self._show_next_page)
        
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        return layout
    
    def _init_camera(self):
        """Initialize camera and frame update timer."""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(FRAME_UPDATE_INTERVAL)
    
    def _update_frame(self):
        """Capture and display the current camera frame with effects."""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Apply background replacement
        # Check if outline should be applied (not "None")
        apply_outline = self.outline_manager.current_style != "None"
        bg = self.bg_manager.get_background()
        if (bg is not None) or apply_outline:
            # Set outline thickness in segment model
            thickness_value = OutlineManager.THICKNESS[self.outline_manager.current_thickness]
            self.segment.outline_thickness = thickness_value
            
            frame = self.segment.replace_background(
                frame, 
                bg,
                apply_outline=apply_outline,
                outline_style=self.outline_manager.current_style,
                outline_color=self.outline_manager.current_color,
                outline_opacity=0.9
            )

        
        # Apply filter
        frame = self.filter_manager.apply_filter(frame, self.filter_manager.current_filter)
        
        # Display frame
        self._display_frame(frame)
    
    def _display_frame(self, frame: np.ndarray):
        """Convert and display a frame in the video label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)
    
    def _set_outline_style(self, style: str):
        """Set the outline style."""
        self.outline_manager.current_style = style
        print(f"✓ Outline style: {style}")
    
    def _set_outline_thickness(self, thickness: str):
        """Set the outline thickness."""
        self.outline_manager.current_thickness = thickness
        print(f"✓ Outline thickness: {thickness}")
    
    def _set_outline_color(self, color: Tuple[int, int, int], color_name: str):
        """Set the outline color."""
        self.outline_manager.current_color = color
        self.outline_manager.current_color_name = color_name
        print(f"✓ Outline color: {color_name}")
    
    def _set_filter(self, filter_name: str):
        """Set the current filter."""
        self.filter_manager.current_filter = filter_name
        print(f"Filter: {filter_name}")
    
    def _show_prev_page(self):
        """Navigate to the previous page."""
        idx = (self.right_stack.currentIndex() - 1) % self.right_stack.count()
        self.right_stack.setCurrentIndex(idx)
    
    def _show_next_page(self):
        """Navigate to the next page."""
        idx = (self.right_stack.currentIndex() + 1) % self.right_stack.count()
        self.right_stack.setCurrentIndex(idx)
    
    def closeEvent(self, event):
        """Clean up resources on window close."""
        self.cap.release()
        super().closeEvent(event)