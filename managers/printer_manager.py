import win32print
import win32ui
import win32con
from PIL import Image, ImageWin


class PrinterManager:
    def __init__(self):
        self.printer_name = win32print.GetDefaultPrinter()

    # -----------------------------------------------------------
    #   Function #2: Prepare Landscape Photo on Landscape A4
    # -----------------------------------------------------------
    def prepare_4x6_on_a4(self, file_path, xdpi, ydpi):
        """
        - Input: landscape photo path
        - Output: a landscape A4 image containing the 4x6 photo
        """

        # ---- Load the photo ----
        img = Image.open(file_path).convert("RGB")

        # Force landscape orientation
        if img.height < img.width:
            img = img.rotate(90, expand=True)

        # ---- 4x6 landscape at 300 DPI is 1800×1200 px ----
        photo_w, photo_h = 6 * xdpi, 4 * ydpi

        # Resize image to fit 4×6 landscape
        img_aspect = img.width / img.height
        target_aspect = photo_w / photo_h

        if img_aspect > target_aspect:
            new_w = photo_w
            new_h = int(photo_w / img_aspect)
        else:
            new_h = photo_h
            new_w = int(photo_h * img_aspect)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        # ---- Create Landscape A4 Canvas (3508×2480 @ 300 dpi) ----
        width_px  = int(8.27 * xdpi)
        height_px = int(11.69 * ydpi)
        a4_canvas = Image.new("RGB", (width_px, height_px), "white")

        # Margin in inches (you can change this)
        margin_in = 0.25  # 0.25 inch margin from corner

        # Convert to pixels
        margin_x = int(margin_in * xdpi)
        margin_y = int(margin_in * ydpi)

        # Place at TOP LEFT with margin
        x = margin_x
        y = margin_y

        a4_canvas.paste(img, (x, y))


        return a4_canvas  # <-- Return final image for printing


    # -----------------------------------------------------------
    #   Function #1: Print image (already prepared)
    # -----------------------------------------------------------
    def print_image(self, file_path):
        # Printer DC
        hDC = win32ui.CreateDC()
        hDC.CreatePrinterDC(self.printer_name)

        pw = hDC.GetDeviceCaps(win32con.HORZRES)
        ph = hDC.GetDeviceCaps(win32con.VERTRES)

        xdpi = hDC.GetDeviceCaps(win32con.LOGPIXELSX)
        ydpi = hDC.GetDeviceCaps(win32con.LOGPIXELSY)
        image = self.prepare_4x6_on_a4(file_path, xdpi, ydpi)

        # Convert to DIB
        dib = ImageWin.Dib(image)

        print(f"Printable area: {pw} x {ph}")

        # Original prepared A4 image size
        cw, ch = image.size  # 3508 × 2480

        # ---------------------------------------------------------
        # SCALE TO FIT (corner placement)
        # ---------------------------------------------------------
        # Landscape A4 is wider than tall, so match width
        # scale = pw / cw
        # new_w = pw
        # new_h = int(ch * scale)

        # If height overflows, adjust by height instead
        # if new_h > ph:
        #     scale = ph / ch
        #     new_h = ph
        #     new_w = int(cw * scale)

        # Print at corner (0,0)
        x = 0
        y = 0

        hDC.StartDoc("A4_4x6_Print")
        hDC.StartPage()

        # DRAW SCALED TO CORNER
        dib.draw(hDC.GetHandleOutput(), (x, y, x + cw, y + ch))

        hDC.EndPage()
        hDC.EndDoc()
        hDC.DeleteDC()

    def preview_image(self, image):
        image.show()  # simple preview using default image viewer




if __name__ == "__main__": 
    pm = PrinterManager() 
    pm.print_image(r"C:\Users\thapa\Documents\AIWorks\Photobooth\resource\output\session_20251110_104319\FIBO_Grad.png")
