from ui.main_window import PhotoBoothApp
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
import argparse
import sys

def main():
    # ---- Parse arguments ----
    parser = argparse.ArgumentParser(description="Start the Photobooth App")
    parser.add_argument(
        "--server_url",
        type=str,
        default="https://82df8cf1ab68.ngrok-free.app",
        help="Backend server URL"
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera index to use"
    )

    args = parser.parse_args()

    # ---- PyQt Application ----
    app = QApplication(sys.argv)

    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    photobooth = PhotoBoothApp(
        camera_index=args.camera_index,
        server_url=args.server_url
    )
    photobooth.show()

    # Cleanup on exit
    app.aboutToQuit.connect(photobooth.cleanup)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
