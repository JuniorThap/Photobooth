import win32print
import win32ui
from PIL import Image, ImageWin

class PrinterManager:
    def __init__(self):
        self.printer_name = win32print.GetDefaultPrinter()

    def print_image(self, file_path):
        # Open image
        image = Image.open(file_path)
        hprinter = win32print.OpenPrinter(self.printer_name)
        printer_info = win32print.GetPrinter(hprinter, 2)
        hDC = win32ui.CreateDC()
        hDC.CreatePrinterDC(self.printer_name)

        # Set up print area
        printer_size = printer_info["pDevMode"].PaperSize
        hDC.StartDoc(file_path)
        hDC.StartPage()

        dib = ImageWin.Dib(image)
        dib.draw(hDC.GetHandleOutput(), (0, 0, image.width, image.height))

        hDC.EndPage()
        hDC.EndDoc()
        hDC.DeleteDC()
        win32print.ClosePrinter(hprinter)
