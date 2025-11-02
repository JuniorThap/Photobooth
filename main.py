from ui.main_window import PhotoBoothApp
from PyQt5.QtWidgets import QApplication

def main():
    import sys
    app = QApplication(sys.argv)
    
    photobooth = PhotoBoothApp(camera_index=0)
    photobooth.show()
    
    # Cleanup on exit
    app.aboutToQuit.connect(photobooth.cleanup)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()