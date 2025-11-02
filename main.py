import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui import PhotoBoothUI

def main():
    # ทำให้ภาพคมบนจอ HiDPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = PhotoBoothUI(camera_index=0)  # 0 = Laptop Webcam
    window.showFullScreen()  # Full screen mode
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
