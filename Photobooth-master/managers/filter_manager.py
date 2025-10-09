import cv2
import numpy as np

class FilterManager:
    FILTER_NAMES = ["Normal", "Gray", "Sepia", "Cartoon", "Beauty"]

    def __init__(self):
        self.current_filter = "Normal"

    def apply_filter(self, frame: np.ndarray, filter_name: str) -> np.ndarray:
        if filter_name == "Normal":
            return frame
        elif filter_name == "Gray":
            return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif filter_name == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            return cv2.transform(frame, kernel)
        elif filter_name == "Cartoon":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(frame, 9, 300, 300)
            return cv2.bitwise_and(color, color, mask=edges)
        elif filter_name == "Beauty":
            return cv2.bilateralFilter(frame, 9, 75, 75)
        return frame
