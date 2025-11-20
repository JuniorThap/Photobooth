from ui.main_window import PhotoBoothApp
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

def main():
    import sys
    app = QApplication(sys.argv)

    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    photobooth = PhotoBoothApp(camera_index=0, server_url="https://82df8cf1ab68.ngrok-free.app")
    photobooth.show()
    
    # Cleanup on exit
    app.aboutToQuit.connect(photobooth.cleanup)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()