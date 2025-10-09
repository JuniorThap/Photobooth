"""
Create 3 beautiful frame designs for FIBOOTH
Each frame includes: FIBOOTH, KMUTT, OPENHOUSE 2025, 10-12 OCTOBER
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Frame dimensions (portrait 4:6 ratio)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 2880

# Image positions (3 photos)
IMAGE_POSITIONS = [
    {"x": 160, "y": 280, "width": 1600, "height": 900},
    {"x": 160, "y": 1230, "width": 1600, "height": 900},
    {"x": 160, "y": 2180, "width": 1600, "height": 900}
]

# Colors (Orange and White theme for KMUTT)
ORANGE = (255, 140, 0)  # RGB
DARK_ORANGE = (230, 100, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (240, 240, 240)
BLACK = (0, 0, 0)


def create_frame_modern():
    """Frame 1: Modern minimalist design with orange accents"""
    # Create white background
    img = Image.new('RGB', (FRAME_WIDTH, FRAME_HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)

    # Draw orange header bar
    draw.rectangle([0, 0, FRAME_WIDTH, 220], fill=ORANGE)

    # Draw orange accent bars between photo slots
    draw.rectangle([0, 1200, FRAME_WIDTH, 1210], fill=ORANGE)
    draw.rectangle([0, 2150, FRAME_WIDTH, 2160], fill=ORANGE)

    # Draw orange footer bar
    draw.rectangle([0, FRAME_HEIGHT - 150, FRAME_WIDTH, FRAME_HEIGHT], fill=ORANGE)

    # Draw photo frames with rounded corners
    for pos in IMAGE_POSITIONS:
        x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
        # Draw shadow
        draw.rounded_rectangle([x+5, y+5, x+w+5, y+h+5], radius=15, fill=(200, 200, 200))
        # Draw white frame
        draw.rounded_rectangle([x, y, x+w, y+h], radius=15, fill=WHITE, outline=ORANGE, width=8)

    # Add text with PIL (better Thai font support)
    try:
        # Try to load a nice font (fallback to default if not available)
        try:
            title_font = ImageFont.truetype("arial.ttf", 100)
            subtitle_font = ImageFont.truetype("arial.ttf", 50)
            info_font = ImageFont.truetype("arial.ttf", 40)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Header text
        draw.text((FRAME_WIDTH//2, 110), "FIBOOTH", fill=WHITE, font=title_font, anchor="mm")

        # Footer text
        draw.text((FRAME_WIDTH//2, FRAME_HEIGHT - 110), "KMUTT", fill=WHITE, font=subtitle_font, anchor="mm")
        draw.text((FRAME_WIDTH//2, FRAME_HEIGHT - 60), "OPENHOUSE 2025 | 10-12 OCTOBER", fill=WHITE, font=info_font, anchor="mm")

    except Exception as e:
        print(f"Warning: Could not add text - {e}")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def create_frame_gradient():
    """Frame 2: Gradient design with modern style"""
    # Create gradient background (white to light orange)
    img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Create vertical gradient
    for y in range(FRAME_HEIGHT):
        ratio = y / FRAME_HEIGHT
        r = int(255 - (255 - 255) * ratio * 0.3)
        g = int(255 - (255 - 200) * ratio * 0.3)
        b = int(255 - (255 - 150) * ratio * 0.3)
        img[y, :] = [b, g, r]  # BGR

    # Convert to PIL for drawing
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Draw decorative orange corner elements
    corner_size = 150
    # Top left
    draw.polygon([(0, 0), (corner_size, 0), (0, corner_size)], fill=ORANGE)
    # Top right
    draw.polygon([(FRAME_WIDTH, 0), (FRAME_WIDTH - corner_size, 0), (FRAME_WIDTH, corner_size)], fill=ORANGE)
    # Bottom left
    draw.polygon([(0, FRAME_HEIGHT), (corner_size, FRAME_HEIGHT), (0, FRAME_HEIGHT - corner_size)], fill=ORANGE)
    # Bottom right
    draw.polygon([(FRAME_WIDTH, FRAME_HEIGHT), (FRAME_WIDTH - corner_size, FRAME_HEIGHT),
                  (FRAME_WIDTH, FRAME_HEIGHT - corner_size)], fill=ORANGE)

    # Draw photo frames
    for pos in IMAGE_POSITIONS:
        x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
        # Draw thick white border
        draw.rounded_rectangle([x-10, y-10, x+w+10, y+h+10], radius=20, fill=WHITE)
        # Draw inner orange border
        draw.rounded_rectangle([x-5, y-5, x+w+5, y+h+5], radius=15, fill=ORANGE)
        # Draw photo area
        draw.rounded_rectangle([x, y, x+w, y+h], radius=12, fill=WHITE)

    # Add text
    try:
        try:
            title_font = ImageFont.truetype("arial.ttf", 120)
            subtitle_font = ImageFont.truetype("arial.ttf", 55)
            info_font = ImageFont.truetype("arial.ttf", 45)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Title with shadow
        draw.text((FRAME_WIDTH//2 + 3, 113), "FIBOOTH", fill=BLACK, font=title_font, anchor="mm")
        draw.text((FRAME_WIDTH//2, 110), "FIBOOTH", fill=ORANGE, font=title_font, anchor="mm")

        # Subtitle
        draw.text((FRAME_WIDTH//2, FRAME_HEIGHT - 100), "KMUTT OPENHOUSE 2025", fill=DARK_ORANGE, font=subtitle_font, anchor="mm")
        draw.text((FRAME_WIDTH//2, FRAME_HEIGHT - 40), "10-12 OCTOBER", fill=ORANGE, font=info_font, anchor="mm")

    except Exception as e:
        print(f"Warning: Could not add text - {e}")

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def create_frame_bold():
    """Frame 3: Bold design with strong orange accents"""
    # Create white background
    img_pil = Image.new('RGB', (FRAME_WIDTH, FRAME_HEIGHT), WHITE)
    draw = ImageDraw.Draw(img_pil)

    # Draw bold orange borders
    border_width = 40
    draw.rectangle([0, 0, FRAME_WIDTH, FRAME_HEIGHT], outline=ORANGE, width=border_width)
    draw.rectangle([border_width, border_width, FRAME_WIDTH - border_width, FRAME_HEIGHT - border_width],
                  outline=DARK_ORANGE, width=15)

    # Draw orange header section with curved design
    header_height = 250
    draw.rectangle([0, 0, FRAME_WIDTH, header_height], fill=ORANGE)
    # Add decorative circles
    circle_y = header_height - 50
    for i in range(0, FRAME_WIDTH, 200):
        draw.ellipse([i, circle_y, i + 100, circle_y + 100], fill=DARK_ORANGE)

    # Draw orange footer section
    footer_y = FRAME_HEIGHT - 200
    draw.rectangle([0, footer_y, FRAME_WIDTH, FRAME_HEIGHT], fill=DARK_ORANGE)
    # Add decorative pattern
    for i in range(0, FRAME_WIDTH, 150):
        y = footer_y + 20
        draw.polygon([(i, y), (i + 50, y + 40), (i + 100, y)], fill=ORANGE)

    # Draw photo frames with double borders
    for pos in IMAGE_POSITIONS:
        x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
        # Outer orange border
        draw.rectangle([x-15, y-15, x+w+15, y+h+15], fill=ORANGE)
        # Inner white border
        draw.rectangle([x-8, y-8, x+w+8, y+h+8], fill=WHITE)
        # Photo area
        draw.rectangle([x, y, x+w, y+h], fill=LIGHT_GRAY)

    # Add text
    try:
        try:
            title_font = ImageFont.truetype("arialbd.ttf", 140)  # Bold
            subtitle_font = ImageFont.truetype("arialbd.ttf", 60)
            info_font = ImageFont.truetype("arial.ttf", 48)
        except:
            try:
                title_font = ImageFont.truetype("arial.ttf", 140)
                subtitle_font = ImageFont.truetype("arial.ttf", 60)
                info_font = ImageFont.truetype("arial.ttf", 48)
            except:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                info_font = ImageFont.load_default()

        # Header text
        draw.text((FRAME_WIDTH//2, 125), "FIBOOTH", fill=WHITE, font=title_font, anchor="mm")

        # Footer text
        draw.text((FRAME_WIDTH//2, footer_y + 60), "KMUTT", fill=WHITE, font=subtitle_font, anchor="mm")
        draw.text((FRAME_WIDTH//2, footer_y + 130), "OPENHOUSE 2025", fill=WHITE, font=info_font, anchor="mm")
        draw.text((FRAME_WIDTH//2, footer_y + 175), "10-12 OCTOBER", fill=ORANGE, font=info_font, anchor="mm")

    except Exception as e:
        print(f"Warning: Could not add text - {e}")

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    """Generate all three frame designs"""
    frames_dir = Path(__file__).parent / "frames"
    frames_dir.mkdir(exist_ok=True)

    print("Creating FIBOOTH frame designs...")

    # Frame 1: Modern
    print("  Creating frame 1: Modern Minimalist...")
    frame1 = create_frame_modern()
    cv2.imwrite(str(frames_dir / "frame_fibooth_modern.png"), frame1)
    print("    ✓ Saved: frame_fibooth_modern.png")

    # Frame 2: Gradient
    print("  Creating frame 2: Gradient Style...")
    frame2 = create_frame_gradient()
    cv2.imwrite(str(frames_dir / "frame_fibooth_gradient.png"), frame2)
    print("    ✓ Saved: frame_fibooth_gradient.png")

    # Frame 3: Bold
    print("  Creating frame 3: Bold Design...")
    frame3 = create_frame_bold()
    cv2.imwrite(str(frames_dir / "frame_fibooth_bold.png"), frame3)
    print("    ✓ Saved: frame_fibooth_bold.png")

    # Update config file
    import json
    config_path = frames_dir / "frame_config.json"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Add new frame configurations
    new_positions = [
        {"x": 160, "y": 280, "width": 1600, "height": 900},
        {"x": 160, "y": 1230, "width": 1600, "height": 900},
        {"x": 160, "y": 2180, "width": 1600, "height": 900}
    ]

    config["fibooth_modern"] = {
        "frame_path": "frames/frame_fibooth_modern.png",
        "image_positions": new_positions,
        "canvas_size": [FRAME_WIDTH, FRAME_HEIGHT]
    }

    config["fibooth_gradient"] = {
        "frame_path": "frames/frame_fibooth_gradient.png",
        "image_positions": new_positions,
        "canvas_size": [FRAME_WIDTH, FRAME_HEIGHT]
    }

    config["fibooth_bold"] = {
        "frame_path": "frames/frame_fibooth_bold.png",
        "image_positions": new_positions,
        "canvas_size": [FRAME_WIDTH, FRAME_HEIGHT]
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n✓ All frames created successfully!")
    print("✓ Updated frame_config.json")
    print("\nYou can now use these frames in the photobooth:")
    print("  - fibooth_modern")
    print("  - fibooth_gradient")
    print("  - fibooth_bold")


if __name__ == "__main__":
    main()
