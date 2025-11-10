"""
Printer Manager for FIBO Graduation PhotoBooth
Handles photo printing with various printer configurations
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import platform
import subprocess
import tempfile


class PrinterManager:
    """Manages photo printing operations"""
    
    # Standard photo print sizes (width, height) in pixels at 300 DPI
    PRINT_SIZES = {
        "4x6": (1800, 1200),      # 4x6 inches at 300 DPI
        "5x7": (2100, 1500),      # 5x7 inches at 300 DPI
        "6x4": (1200, 1800),      # 6x4 inches at 300 DPI (portrait)
        "wallet": (900, 600),      # 2x3 inches wallet size
        "strip": (1200, 3600),     # Photo strip (4x12 inches)
    }
    
    def __init__(self, default_printer: Optional[str] = None, default_size: str = "4x6"):
        """
        Initialize PrinterManager
        
        Args:
            default_printer: Name of default printer (None for system default)
            default_size: Default print size key from PRINT_SIZES
        """
        self.default_printer = default_printer
        self.default_size = default_size
        self.system = platform.system()
        
        print(f"[PrinterManager] Initialized on {self.system}")
        if default_printer:
            print(f"[PrinterManager] Default printer: {default_printer}")
    
    def get_available_printers(self) -> list:
        """Get list of available printers on the system"""
        printers = []
        
        try:
            if self.system == "Windows":
                # Use Windows command to list printers
                result = subprocess.run(
                    ['powershell', '-Command', 
                     'Get-Printer | Select-Object Name | Format-Table -HideTableHeaders'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    printers = [line.strip() for line in result.stdout.split('\n') 
                               if line.strip()]
            
            elif self.system == "Linux":
                # Use lpstat command
                result = subprocess.run(
                    ['lpstat', '-p'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('printer'):
                            printer_name = line.split()[1]
                            printers.append(printer_name)
            
            elif self.system == "Darwin":  # macOS
                # Use lpstat command
                result = subprocess.run(
                    ['lpstat', '-p'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('printer'):
                            printer_name = line.split()[1]
                            printers.append(printer_name)
        
        except Exception as e:
            print(f"[PrinterManager] Error getting printers: {e}")
        
        return printers
    
    def prepare_image_for_print(
        self, 
        image: np.ndarray, 
        size: str = None,
        dpi: int = 300,
        border: int = 0
    ) -> np.ndarray:
        """
        Prepare image for printing by resizing and adding borders
        
        Args:
            image: Input image (BGR format)
            size: Print size key from PRINT_SIZES
            dpi: Dots per inch for printing
            border: Border size in pixels
        
        Returns:
            Prepared image ready for printing
        """
        size = size or self.default_size
        
        if size not in self.PRINT_SIZES:
            print(f"[PrinterManager] Invalid size '{size}', using default")
            size = self.default_size
        
        target_width, target_height = self.PRINT_SIZES[size]
        
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Calculate aspect ratios
        target_ratio = target_width / target_height
        image_ratio = orig_width / orig_height
        
        # Resize to fit while maintaining aspect ratio
        if image_ratio > target_ratio:
            # Image is wider - fit to width
            new_width = target_width
            new_height = int(target_width / image_ratio)
        else:
            # Image is taller - fit to height
            new_height = target_height
            new_width = int(target_height * image_ratio)
        
        resized = cv2.resize(image, (new_width, new_height), 
                            interpolation=cv2.INTER_LANCZOS4)
        
        # Create canvas with white background
        canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
        
        # Center the image on canvas
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        # Add border if specified
        if border > 0:
            canvas = cv2.copyMakeBorder(
                canvas, border, border, border, border,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        
        print(f"[PrinterManager] Prepared {size} print ({target_width}x{target_height}px)")
        return canvas
    
    def print_image(
        self,
        image: np.ndarray,
        printer_name: Optional[str] = None,
        size: str = None,
        copies: int = 1
    ) -> bool:
        """
        Print an image
        
        Args:
            image: Image to print (BGR format)
            printer_name: Specific printer name (None for default)
            size: Print size key
            copies: Number of copies to print
        
        Returns:
            True if print successful, False otherwise
        """
        try:
            # Prepare image for printing
            print_image = self.prepare_image_for_print(image, size)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, print_image, 
                           [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # Print based on OS
            printer = printer_name or self.default_printer
            
            if self.system == "Windows":
                success = self._print_windows(temp_path, printer, copies)
            elif self.system == "Linux":
                success = self._print_linux(temp_path, printer, copies)
            elif self.system == "Darwin":
                success = self._print_macos(temp_path, printer, copies)
            else:
                print(f"[PrinterManager] Unsupported OS: {self.system}")
                success = False
            
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except:
                pass
            
            if success:
                print(f"[PrinterManager] ✅ Print job sent ({copies} copies)")
            else:
                print(f"[PrinterManager] ❌ Print job failed")
            
            return success
            
        except Exception as e:
            print(f"[PrinterManager] Error printing: {e}")
            return False
    
    def _print_windows(self, file_path: str, printer: Optional[str], copies: int) -> bool:
        """Print on Windows"""
        try:
            # Use Windows Photo Viewer or default image viewer
            if printer:
                cmd = [
                    'powershell', '-Command',
                    f'Start-Process -FilePath "{file_path}" -Verb PrintTo -ArgumentList "{printer}"'
                ]
            else:
                # Print to default printer
                cmd = ['powershell', '-Command', f'Start-Process -FilePath "{file_path}" -Verb Print']
            
            for _ in range(copies):
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                if result.returncode != 0:
                    return False
            
            return True
            
        except Exception as e:
            print(f"[PrinterManager] Windows print error: {e}")
            return False
    
    def _print_linux(self, file_path: str, printer: Optional[str], copies: int) -> bool:
        """Print on Linux using lp command"""
        try:
            cmd = ['lp']
            
            if printer:
                cmd.extend(['-d', printer])
            
            cmd.extend(['-n', str(copies)])
            cmd.append(file_path)
            
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
            
        except Exception as e:
            print(f"[PrinterManager] Linux print error: {e}")
            return False
    
    def _print_macos(self, file_path: str, printer: Optional[str], copies: int) -> bool:
        """Print on macOS using lp command"""
        try:
            cmd = ['lp']
            
            if printer:
                cmd.extend(['-d', printer])
            
            cmd.extend(['-n', str(copies)])
            cmd.append(file_path)
            
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
            
        except Exception as e:
            print(f"[PrinterManager] macOS print error: {e}")
            return False
    
    def print_photo_strip(
        self,
        images: list,
        printer_name: Optional[str] = None,
        copies: int = 1
    ) -> bool:
        """
        Print a photo strip with multiple images
        
        Args:
            images: List of 2-4 images to combine into strip
            printer_name: Specific printer name
            copies: Number of copies
        
        Returns:
            True if successful
        """
        if not images or len(images) == 0:
            print("[PrinterManager] No images provided for photo strip")
            return False
        
        try:
            # Create photo strip layout (4x12 inches at 300 DPI)
            strip_width = 1200
            strip_height = 3600
            
            # Calculate individual photo size
            photo_height = strip_height // len(images)
            
            # Create white canvas
            strip = np.ones((strip_height, strip_width, 3), dtype=np.uint8) * 255
            
            # Add each image to the strip
            for i, img in enumerate(images):
                # Resize to fit strip width
                h, w = img.shape[:2]
                aspect = w / h
                new_width = strip_width - 20  # 10px margin on each side
                new_height = int(new_width / aspect)
                
                # Ensure it fits in allocated space
                if new_height > photo_height - 20:
                    new_height = photo_height - 20
                    new_width = int(new_height * aspect)
                
                resized = cv2.resize(img, (new_width, new_height), 
                                    interpolation=cv2.INTER_LANCZOS4)
                
                # Calculate position (centered)
                y_pos = i * photo_height + (photo_height - new_height) // 2
                x_pos = (strip_width - new_width) // 2
                
                # Place image on strip
                strip[y_pos:y_pos+new_height, x_pos:x_pos+new_width] = resized
            
            # Print the strip
            return self.print_image(strip, printer_name, size="strip", copies=copies)
            
        except Exception as e:
            print(f"[PrinterManager] Error creating photo strip: {e}")
            return False
    
    def test_print(self, printer_name: Optional[str] = None) -> bool:
        """
        Print a test page
        
        Args:
            printer_name: Specific printer to test
        
        Returns:
            True if test successful
        """
        # Create test image
        test_img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_img, "FIBO PhotoBooth", (150, 200), 
                   font, 2, (0, 0, 0), 3)
        cv2.putText(test_img, "Test Print", (250, 350), 
                   font, 1.5, (0, 0, 0), 2)
        
        # Add date/time
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(test_img, timestamp, (250, 450), 
                   font, 1, (100, 100, 100), 2)
        
        print("[PrinterManager] Sending test print...")
        return self.print_image(test_img, printer_name, size="4x6", copies=1)


# Example usage and testing
if __name__ == "__main__":
    # Initialize printer manager
    printer = PrinterManager()
    
    # List available printers
    printers = printer.get_available_printers()
    print(f"\nAvailable printers: {printers}")
    
    # Test print
    if printers:
        print("\nSending test print...")
        printer.test_print(printers[0] if printers else None)