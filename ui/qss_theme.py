ORANGE_WHITE_QSS = """
* { color: #333; font-family: "Segoe UI", "Arial", system-ui; }
QWidget { background-color: #ffffff; }
QLabel[hero="true"] {
    font-size: 32px; font-weight: 700; color: #ff8c00;
    padding: 10px; background-color: #fff5e6; border-radius: 8px;
}
QLabel[section="true"] {
    font-size: 18px; font-weight: 600; color: #ff8c00;
    padding: 8px; background-color: #fff; border-bottom: 2px solid #ff8c00;
}
QPushButton {
    background-color: #ffffff; border: 2px solid #ff8c00; border-radius: 12px;
    padding: 12px; font-weight: 600; color: #ff8c00;
}
QPushButton:hover {
    background-color: #fff5e6; border: 2px solid #ff6600;
}
QPushButton:pressed {
    background-color: #ffe6cc; border: 2px solid #ff8c00;
}
QPushButton:checked {
    background-color: #ff8c00; color: white; border: 2px solid #ff8c00;
}
QPushButton:checked:hover {
    background-color: #ff6600; border: 2px solid #ff6600;
}
QSlider::groove:horizontal {
    height: 6px; background: #ffe6cc; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ff8c00; width: 16px; border-radius: 8px; margin: -6px 0;
}
QFrame#line {
    background-color: #ffe6cc; max-height: 2px; min-height: 2px;
}
"""