import cv2
import numpy as np
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal



class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, cam_index: int, width, height: int, target_fps: int):
        super().__init__()
        self.target_fps = target_fps
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.running = True
        self.cap: Optional[cv2.VideoCapture] = None

    def run(self):
        print(f"[Camera] กำลังเปิดกล้อง index {self.cam_index}...")
        try_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        for backend in try_backends:
            backend_name = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "ANY"}
            print(f"[Camera] ลอง backend: {backend_name.get(backend, backend)}")
            self.cap = cv2.VideoCapture(self.cam_index, backend)
            if self.cap.isOpened():
                print(f"[Camera] ✓ เปิดกล้องสำเร็จด้วย {backend_name.get(backend, backend)}")
                break
            else:
                print(f"[Camera] ✗ ไม่สามารถเปิดด้วย {backend_name.get(backend, backend)}")

        if not self.cap or not self.cap.isOpened():
            print("[Camera] ERROR: ไม่สามารถเปิดกล้องได้!")
            return

        try:
            fourcc = cv2.VideoWriter_fourcc(*"YUY2")
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            print("[Camera] ตั้งค่า format: YUY2")
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] ความละเอียดที่ได้: {actual_w}x{actual_h}")

        if actual_w < 1920 or actual_h < 1080:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Camera] Fallback ความละเอียด: {actual_w}x{actual_h}")

        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        print("[Camera] เริ่มอ่านเฟรม...")
        frame_count = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if frame_count == 0:
                    print("[Camera] ERROR: ไม่สามารถอ่านเฟรมได้!")
                continue

            if frame_count == 0:
                print(f"[Camera] ✓ อ่านเฟรมสำเร็จ! ขนาด: {frame.shape}")

            frame = cv2.flip(frame, 1)
            self.frame_ready.emit(frame)
            frame_count += 1

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False