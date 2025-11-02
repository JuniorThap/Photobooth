class OutlineManager:
    STYLES = ["None", "Solid"]  # GLOW removed
    THICKNESS = {"Thin": 1, "Medium": 5, "Thick": 10}
    COLORS = {
        "White": (255, 255, 255),
        "Red": (0, 0, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Yellow": (0, 255, 255),
        "Purple": (255, 0, 255),
        "Orange": (0, 165, 255)
    }

    def __init__(self):
        self.current_style = "Solid"  # Default เป็น Solid (มี Stroke)
        self.current_thickness = "Medium"
        self.current_color = (255, 255, 255)  # White
        self.current_color_name = "White"
