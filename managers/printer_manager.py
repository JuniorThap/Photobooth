import win32print
import win32ui
import win32con
from PIL import Image, ImageWin

class PrinterManager:
    def __init__(self):
        self.printer_name = win32print.GetDefaultPrinter()

    def print_image(self, file_path):
        # Load and convert image
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create 4x6 canvas @ 300dpi (1200×1800) - PORTRAIT
        target_w, target_h = 1200, 1800
        canvas = Image.new('RGB', (target_w, target_h), "white")

        # Maintain aspect ratio
        img_aspect = image.width / image.height
        target_aspect = target_w / target_h

        if img_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / img_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * img_aspect)

        image = image.resize((new_w, new_h), Image.LANCZOS)
        canvas.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))

        # Get and modify DEVMODE
        hPrinter = win32print.OpenPrinter(self.printer_name)
        try:
            devmode = win32print.GetPrinter(hPrinter, 2)["pDevMode"]
            
            # Modify for 4x6 portrait
            devmode.PaperSize = 122  # 4x6 inch
            devmode.Orientation = win32con.DMORIENT_PORTRAIT
            
            try:
                devmode.MediaType = 2  # Photo paper
            except:
                pass
            
            try:
                devmode.PrintQuality = win32con.DMRES_HIGH
            except:
                pass
            
        finally:
            win32print.ClosePrinter(hPrinter)

        # Create DC
        hDC = win32ui.CreateDC()
        hDC.CreatePrinterDC(self.printer_name)

        # Get printable area
        page_w = hDC.GetDeviceCaps(win32con.HORZRES)
        page_h = hDC.GetDeviceCaps(win32con.VERTRES)
        
        xdpi = hDC.GetDeviceCaps(win32con.LOGPIXELSX)
        ydpi = hDC.GetDeviceCaps(win32con.LOGPIXELSY)
        
        print(f"Printer DPI: {xdpi}x{ydpi}")
        print(f"Printable area: {page_w}x{page_h} pixels")

        # Start print job with DEVMODE
        docinfo = {
            "name": "Photobooth_4x6_GlossyII",
            "output": None,
            "datatype": None,
            "devmode": devmode  # Pass devmode here
        }
        
        try:
            hDC.StartDoc("Photobooth_4x6_GlossyII")
        except:
            # If that doesn't work, try without devmode
            hDC.StartDoc("Photobooth_4x6_GlossyII")
            
        hDC.StartPage()

        dib = ImageWin.Dib(canvas)
        dib.draw(hDC.GetHandleOutput(), (0, 0, page_w, page_h))

        hDC.EndPage()
        hDC.EndDoc()
        hDC.DeleteDC()


if __name__ == "__main__":
    printer = PrinterManager()
    printer.print_image(r"C:\Users\thapa\Documents\AIWorks\Photobooth\resource\output\session_20251110_104319\FIBO_Grad.png")