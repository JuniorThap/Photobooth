"""
Simple Frame Generator for Photobooth
สร้าง Frame แบบง่ายๆ ด้วย Python

Usage:
    python create_simple_frame.py --style classic
    python create_simple_frame.py --style polaroid --title "Birthday Party 2025"
    python create_simple_frame.py --style gradient --color1 "#FF6B9D" --color2 "#4ECDC4"
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def create_classic_frame(width=1920, height=2560, border_color=(50, 50, 50), title="FIBOOTH"):
    """สร้าง Frame แบบ Classic พร้อม Border"""
    # Create transparent canvas
    frame = np.zeros((height, width, 4), dtype=np.uint8)

    # Fill background with semi-transparent white
    frame[:, :, :3] = 255
    frame[:, :, 3] = 230  # Slightly transparent

    # Define photo positions
    img_width, img_height = 1600, 720
    margin_x = (width - img_width) // 2
    positions = [
        (margin_x, 200),
        (margin_x, 970),
        (margin_x, 1740)
    ]

    # Cut out transparent areas for photos
    for x, y in positions:
        frame[y:y+img_height, x:x+img_width, 3] = 0

        # Draw decorative border around each photo
        border_thickness = 15
        cv2.rectangle(frame,
                     (x - border_thickness, y - border_thickness),
                     (x + img_width + border_thickness, y + img_height + border_thickness),
                     (*border_color, 255), border_thickness)

    # Add header bar
    cv2.rectangle(frame, (0, 0), (width, 150), (*border_color, 255), -1)
    cv2.putText(frame, title, (width//2 - 300, 95),
                cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255, 255), 5)

    # Add footer bar
    footer_y = height - 120
    cv2.rectangle(frame, (0, footer_y), (width, height), (*border_color, 255), -1)
    cv2.putText(frame, "FIBO KMUTT", (width//2 - 250, footer_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255, 255), 3)

    return frame


def create_polaroid_frame(width=1920, height=2560, title="MEMORIES"):
    """สร้าง Frame แบบ Polaroid"""
    frame = np.zeros((height, width, 4), dtype=np.uint8)

    # Pastel background
    frame[:, :] = (220, 240, 255, 255)  # Light blue

    # Photo positions (slightly smaller with white border like polaroid)
    img_width, img_height = 1520, 680
    margin_x = (width - img_width) // 2
    positions = [
        (margin_x, 250),
        (margin_x, 1020),
        (margin_x, 1790)
    ]

    for x, y in positions:
        # White polaroid border
        polaroid_padding = 40
        polaroid_bottom = 100
        cv2.rectangle(frame,
                     (x - polaroid_padding, y - polaroid_padding),
                     (x + img_width + polaroid_padding, y + img_height + polaroid_bottom),
                     (255, 255, 255, 255), -1)

        # Shadow effect
        shadow_offset = 8
        cv2.rectangle(frame,
                     (x - polaroid_padding + shadow_offset, y - polaroid_padding + shadow_offset),
                     (x + img_width + polaroid_padding + shadow_offset, y + img_height + polaroid_bottom + shadow_offset),
                     (150, 150, 150, 100), -1)

        # Transparent area for photo
        frame[y:y+img_height, x:x+img_width, 3] = 0

    # Title at top
    cv2.putText(frame, title, (width//2 - 200, 150),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2.5, (80, 80, 80, 255), 4)

    return frame


def create_gradient_frame(width=1920, height=2560, color1=(255, 107, 157), color2=(78, 205, 196)):
    """สร้าง Frame แบบ Gradient สีสวย"""
    frame = np.zeros((height, width, 4), dtype=np.uint8)

    # Create gradient background
    for i in range(height):
        ratio = i / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        frame[i, :] = (b, g, r, 255)

    # Photo positions with rounded corners effect
    img_width, img_height = 1600, 720
    margin_x = (width - img_width) // 2
    positions = [
        (margin_x, 200),
        (margin_x, 970),
        (margin_x, 1740)
    ]

    for x, y in positions:
        # White rounded frame
        padding = 20
        cv2.rectangle(frame,
                     (x - padding, y - padding),
                     (x + img_width + padding, y + img_height + padding),
                     (255, 255, 255, 255), -1)

        # Transparent area for photo
        frame[y:y+img_height, x:x+img_width, 3] = 0

        # Add decorative corners
        corner_size = 60
        corner_color = (255, 255, 255, 255)
        # Top-left
        cv2.circle(frame, (x - padding, y - padding), corner_size, corner_color, -1)
        # Top-right
        cv2.circle(frame, (x + img_width + padding, y - padding), corner_size, corner_color, -1)
        # Bottom-left
        cv2.circle(frame, (x - padding, y + img_height + padding), corner_size, corner_color, -1)
        # Bottom-right
        cv2.circle(frame, (x + img_width + padding, y + img_height + padding), corner_size, corner_color, -1)

    return frame


def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))


def main():
    parser = argparse.ArgumentParser(description='สร้าง Photobooth Frame')
    parser.add_argument('--style', type=str, default='classic',
                       choices=['classic', 'polaroid', 'gradient'],
                       help='รูปแบบ Frame (classic/polaroid/gradient)')
    parser.add_argument('--title', type=str, default='FIBOOTH',
                       help='ข้อความ Title')
    parser.add_argument('--color', type=str, default='#323232',
                       help='สีหลัก (hex format)')
    parser.add_argument('--color1', type=str, default='#FF6B9D',
                       help='สีที่ 1 สำหรับ gradient')
    parser.add_argument('--color2', type=str, default='#4ECDC4',
                       help='สีที่ 2 สำหรับ gradient')
    parser.add_argument('--output', type=str, default=None,
                       help='ชื่อไฟล์ output')

    args = parser.parse_args()

    # Create frame based on style
    if args.style == 'classic':
        border_color = hex_to_bgr(args.color)
        frame = create_classic_frame(title=args.title, border_color=border_color)
    elif args.style == 'polaroid':
        frame = create_polaroid_frame(title=args.title)
    elif args.style == 'gradient':
        color1 = hex_to_bgr(args.color1)
        color2 = hex_to_bgr(args.color2)
        frame = create_gradient_frame(color1=color1, color2=color2)

    # Save frame
    frames_dir = Path(__file__).parent / "frames"
    frames_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = frames_dir / args.output
    else:
        output_path = frames_dir / f"frame_{args.style}.png"

    cv2.imwrite(str(output_path), frame)
    print(f"✓ สร้าง Frame สำเร็จ: {output_path}")
    print(f"  ขนาด: {frame.shape[1]} x {frame.shape[0]} pixels")
    print(f"  รูปแบบ: {args.style}")


if __name__ == "__main__":
    main()
