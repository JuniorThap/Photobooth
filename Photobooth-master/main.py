import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui import PhotoBoothUI

def main():
    # ทำให้ภาพคมบนจอ HiDPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = PhotoBoothUI(camera_index=0)  # ใช้กล้อง index 1
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
