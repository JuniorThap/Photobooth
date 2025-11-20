import win32print
import win32ui
from PIL import Image, ImageWin

class PrinterManager:
    def __init__(self):
        self.printer_name = win32print.GetDefaultPrinter()

    # ----------------------------------------------------------
    #   Build A6 PORTRAIT canvas and center image
    # ----------------------------------------------------------
    def make_a6_portrait_canvas(self, image, pw, ph):
        # Get image dimensions
        img_w, img_h = image.size
        
        # Calculate scaling to fit image within canvas while maintaining aspect ratio
        scale_w = pw / img_w
        scale_h = ph / img_h
        
        # Use the smaller scale to ensure image fits completely (no crop)
        scale = min(scale_w, scale_h) * 0.85
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        return image.resize((new_w, new_h), Image.LANCZOS)

    # ----------------------------------------------------------
    #   Simple preview window BEFORE printing
    # ----------------------------------------------------------
    def preview_image(self, image):
        image.show()

    # ----------------------------------------------------------
    #   Print on A6 portrait paper
    # ----------------------------------------------------------
    def print_image(self, file_path):
        # Load image WITHOUT rotation
        original = Image.open(file_path).convert("RGB").rotate(90, expand=True)

        # ---- PRINTING ----
        hprinter = win32print.OpenPrinter(self.printer_name)
        hDC = win32ui.CreateDC()
        hDC.CreatePrinterDC(self.printer_name)

        # Get printer's printable area
        pw = hDC.GetDeviceCaps(8)   # HORZRES - printable width
        ph = hDC.GetDeviceCaps(10)  # VERTRES - printable height
        print("Printable:", pw, ph)

        # Get printer DPI
        printer_dpi_x = hDC.GetDeviceCaps(88)  # LOGPIXELSX
        printer_dpi_y = hDC.GetDeviceCaps(90)  # LOGPIXELSY
        print("Printer dpi:", printer_dpi_x, printer_dpi_y)

        # Build A6 PORTRAIT canvas with centered image
        a6_canvas = self.make_a6_portrait_canvas(original, pw, ph)
        print("A6 Canvas size:", a6_canvas.size)

        # Preview before printing
        self.preview_image(a6_canvas)

        # Resize canvas to actual print size
        dib = ImageWin.Dib(a6_canvas)

        hDC.StartDoc("A6_Portrait_Print")
        hDC.StartPage()

        # Print centered on page
        x_offset = (pw - a6_canvas.size[0]) // 2
        print((pw - a6_canvas.size[0]) // 2, 0, x_offset + a6_canvas.size[0], a6_canvas.size[1])
        dib.draw(hDC.GetHandleOutput(), (x_offset, 0, x_offset + a6_canvas.size[0], a6_canvas.size[1]))

        hDC.EndPage()
        hDC.EndDoc()

        hDC.DeleteDC()
        win32print.ClosePrinter(hprinter)


if __name__ == "__main__":
    pm = PrinterManager()
    pm.print_image(
        r"C:\Users\thapa\Documents\AIWorks\Photobooth\resource\output\session_20251110_104319\FIBO_Grad.png"
    )