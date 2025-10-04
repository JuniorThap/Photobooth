import sys
from PyQt5.QtWidgets import QApplication
from ui import PhotoBoothUI

def main():
    app = QApplication(sys.argv)
    window = PhotoBoothUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
