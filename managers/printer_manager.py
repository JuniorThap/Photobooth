import win32print
import win32ui
from PIL import Image, ImageWin

class PrinterManager:
    def __init__(self):
        self.printer_name = win32print.GetDefaultPrinter()

    # ----------------------------------------------------------
    #   Build an A4 canvas (portrait) and paste image at top-left
    # ----------------------------------------------------------
    def make_a4_canvas(self, image):
        # Create A4 canvas at 300 DPI
        a4_w = int(8.27 * 300)   # width in px
        a4_h = int(11.69 * 300)  # height in px

        canvas = Image.new("RGB", (a4_w, a4_h), "white")

        # Paste at top-left corner
        canvas.paste(image, (0, 0))

        return canvas

    # ----------------------------------------------------------
    #   Simple preview window BEFORE printing
    # ----------------------------------------------------------
    def preview_image(self, image):
        image.show()

    # ----------------------------------------------------------
    #   Print the A4 canvas
    # ----------------------------------------------------------
    def print_image(self, file_path):

        # Load image
        original = Image.open(file_path).convert("RGB")

        # Build A4 canvas
        a4 = self.make_a4_canvas(original)

        # self.preview_image(a4)

        # ---- PRINTING ----
        hprinter = win32print.OpenPrinter(self.printer_name)
        hDC = win32ui.CreateDC()
        hDC.CreatePrinterDC(self.printer_name)

        pw = hDC.GetDeviceCaps(8)   # HORZRES
        ph = hDC.GetDeviceCaps(10)  # VERTRES

        # Resize A4 canvas to printer's area
        print_img = a4.resize((pw, ph))
        dib = ImageWin.Dib(print_img)

        hDC.StartDoc("A4_Print")
        hDC.StartPage()

        # Print entire A4 on paper
        scale = 3
        dib.draw(hDC.GetHandleOutput(), (0, 0, pw*scale, ph*scale))

        hDC.EndPage()
        hDC.EndDoc()

        hDC.DeleteDC()
        win32print.ClosePrinter(hprinter)


if __name__ == "__main__":
    pm = PrinterManager()
    pm.print_image(
        r"C:\Users\thapa\Documents\AIWorks\Photobooth\resource\output\session_20251110_104319\FIBO_Grad.png"
    )
