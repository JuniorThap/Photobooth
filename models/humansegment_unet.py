"""
Human segmentation module using U-Net for background replacement and effects.
"""

from typing import Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import albumentations as albu
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from people_segmentation.pre_trained_models import create_model


@dataclass
class SegmentConfig:
    """Configuration for the segmentation model."""
    model_name: str = "Unet_2020-07-20"
    device: str = "cuda"
    max_size: int = 512
    outline_thickness: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            self.device = "cpu"


@dataclass
class MaskRefinementParams:
    """Parameters for mask refinement."""
    blur_strength: int = 5
    morph_kernel: int = 5
    feather: int = 5


class Segment:
    """
    Human segmentation class for real-time background replacement.

    Uses a U-Net model to segment people from backgrounds and provides
    utilities for mask refinement, outline generation, and background replacement.
    """

    # Mask thresholds
    MASK_THRESHOLD = 127
    BINARY_MAX = 255

    # Outline thickness scaling factor
    OUTLINE_SCALE_FACTOR = 5

    def __init__(
        self,
        model_name: str = "Unet_2020-07-20",
        device: str = "cuda",
        max_size: int = 512,
        outline_thickness: int = 1
    ):
        """
        Initialize the segmentation model.

        Args:
            model_name: Name of the pre-trained U-Net model
            device: Device to run model on ('cuda' or 'cpu')
            max_size: Maximum size for image preprocessing
            outline_thickness: Base thickness for outline generation
        """
        self.config = SegmentConfig(model_name, device, max_size, outline_thickness)
        self._outline_thickness_internal = outline_thickness

        self.model = self._load_model()
        self.transform = self._create_transform()

    def _load_model(self) -> torch.nn.Module:
        """Load and prepare the segmentation model."""
        print(f"Loading U-Net model: {self.config.model_name}...")
        model = create_model(self.config.model_name).to(self.config.device)
        model.eval()
        print(f"✓ Model loaded on {self.config.device}")
        return model

    def _create_transform(self) -> albu.Compose:
        """Create albumentations preprocessing pipeline (legacy/original)."""
        return albu.Compose([
            albu.LongestMaxSize(max_size=self.config.max_size),
            albu.Normalize(p=1)
        ], p=1)

    @property
    def outline_thickness(self) -> float:
        """Get the effective outline thickness."""
        return self._outline_thickness_internal / self.OUTLINE_SCALE_FACTOR

    @outline_thickness.setter
    def outline_thickness(self, value: float):
        """Set the outline thickness with scaling."""
        self._outline_thickness_internal = int(value * self.OUTLINE_SCALE_FACTOR)

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate binary person segmentation mask from frame.

        Args:
            frame: Input BGR image from camera/video

        Returns:
            Binary mask (0-255) where 255 indicates person pixels
        """
        # Convert to RGB for model input
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_height, original_width = rgb_image.shape[:2]

        # Preprocess image (legacy behavior)
        preprocessed = self.transform(image=rgb_image)["image"]
        padded_image, pads = pad(
            preprocessed,
            factor=self.config.max_size,
            border=cv2.BORDER_CONSTANT
        )

        # Convert to tensor and run inference
        input_tensor = self._prepare_tensor(padded_image)
        prediction = self._run_inference(input_tensor)

        # Post-process mask
        mask = self._postprocess_mask(
            prediction,
            pads,
            (original_width, original_height)
        )

        return mask

    def _prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to model input tensor."""
        tensor = tensor_from_rgb_image(image)
        return torch.unsqueeze(tensor, 0).to(self.config.device)

    def _run_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run model inference with appropriate precision."""
        with torch.no_grad():
            if self.config.device == "cuda":
                with torch.amp.autocast("cuda"):
                    prediction = self.model(input_tensor)[0][0]
            else:
                prediction = self.model(input_tensor)[0][0]

        return prediction

    def _postprocess_mask(
        self,
        prediction: torch.Tensor,
        pads: Tuple[int, ...],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Convert prediction tensor to binary mask at target size."""
        # Threshold and convert to numpy
        mask = (prediction > 0).cpu().numpy().astype(np.uint8)

        # Remove padding
        mask = unpad(mask, pads)

        # Resize to original dimensions
        mask = cv2.resize(
            mask,
            target_size,
            interpolation=cv2.INTER_NEAREST
        )

        return mask * self.BINARY_MAX

    def refine_mask(
        self,
        mask: np.ndarray,
        blur_strength: int = 5,
        morph_kernel: int = 5,
        feather: int = 5
    ) -> np.ndarray:
        """
        Refine segmentation mask for better quality.
        """
        # Ensure binary mask
        mask = self._binarize_mask(mask)

        # Morphological cleanup
        if morph_kernel > 0:
            mask = self._apply_morphology(mask, morph_kernel)

        # Smooth edges
        if blur_strength > 0:
            mask = self._apply_blur(mask, blur_strength)

        # Feather edges
        if feather > 0:
            mask = self._apply_feathering(mask)

        return self._binarize_mask(mask)

    def _binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to strict binary (0 or 255)."""
        return (mask > self.MASK_THRESHOLD).astype(np.uint8) * self.BINARY_MAX

    def _apply_morphology(self, mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply morphological operations to clean up mask."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _apply_blur(self, mask: np.ndarray, blur_strength: int) -> np.ndarray:
        """Apply Gaussian blur to smooth mask edges."""
        # Ensure odd kernel size
        kernel_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    def _apply_feathering(self, mask: np.ndarray) -> np.ndarray:
        """Apply distance transform for soft edge feathering."""
        binary = (mask > self.MASK_THRESHOLD).astype(np.uint8)

        # Calculate distance from edge
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

        # Normalize to 0-255 range
        dist_normalized = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

        return (dist_normalized * self.BINARY_MAX).astype(np.uint8)

    def make_outline(
        self,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Create a colored solid outline from a binary mask with smooth edges."""
        # Ensure binary mask
        mask_binary = self._binarize_mask(mask)

        # Apply additional smoothing to mask before creating outline
        mask_smooth = cv2.GaussianBlur(mask_binary, (5, 5), 0)
        mask_smooth = self._binarize_mask(mask_smooth)

        # Create dilation kernel
        thickness = max(1, self._outline_thickness_internal)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))

        # Dilate mask to create outline
        dilated = cv2.dilate(mask_smooth, kernel, iterations=1)

        # Extract only the outline (difference between dilated and original)
        outline = cv2.subtract(dilated, mask_smooth)

        # Smooth the outline edges
        outline = cv2.GaussianBlur(outline, (3, 3), 0)

        # Create colored outline image with anti-aliasing
        outline_bgr = np.zeros((*mask.shape, 3), dtype=np.uint8)

        # Use the blurred outline as alpha channel for smoother edges
        for c in range(3):
            outline_bgr[:, :, c] = (outline.astype(np.float32) / 255.0 * color[c]).astype(np.uint8)

        return outline_bgr

    def make_glow_outline(
        self,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Create a glowing outline effect (kept for completeness)."""
        mask_binary = self._binarize_mask(mask)

        # Apply smoothing to mask
        mask_smooth = cv2.GaussianBlur(mask_binary, (7, 7), 0)
        mask_smooth = self._binarize_mask(mask_smooth)

        # Create multiple dilations for glow effect
        thickness = max(1, self._outline_thickness_internal)
        glow_layers = 4
        glow_image = np.zeros((*mask.shape, 3), dtype=np.float32)

        for i in range(glow_layers, 0, -1):
            kernel_size = max(3, thickness * i)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(mask_smooth, kernel, iterations=1)
            outline = cv2.subtract(dilated, mask_smooth)

            blur_size = 21 + (i * 4)
            if blur_size % 2 == 0:
                blur_size += 1
            outline_blurred = cv2.GaussianBlur(outline, (blur_size, blur_size), 0)

            intensity = 1.2 / (i * 0.8)
            for c in range(3):
                glow_image[:, :, c] += outline_blurred.astype(np.float32) * color[c] * intensity / 255.0

        glow_image = np.clip(glow_image, 0, 255).astype(np.uint8)
        glow_image = cv2.GaussianBlur(glow_image, (5, 5), 0)

        return glow_image

    def replace_background(
        self,
        frame: np.ndarray,
        background: Optional[np.ndarray],
        apply_outline: bool = False,
        outline_style: str = "Solid",
        outline_color: Tuple[int, int, int] = (0, 255, 0),
        outline_opacity: float = 0.8
    ) -> np.ndarray:
        """
        Replace the background of a frame with a new background.
        """
        if background is not None:
            # Resize frame to match background while maintaining aspect ratio
            frame, scale_info = self._resize_frame_to_background(frame, background)

        # Generate and refine mask
        mask = None
        if (background is not None) or apply_outline:
            mask = self.get_mask(frame)
            # Legacy behavior (เหมือนที่คุณเคยแม่นก่อนแก้): feather ปิดไว้
            mask = self.refine_mask(mask, blur_strength=7, morph_kernel=5, feather=-1)
            mask_inv = cv2.bitwise_not(mask)

        # Optional outline
        outline = None
        if apply_outline:
            if outline_style == "Solid":
                outline = self.make_outline(mask, outline_color)
            elif outline_style == "Glow":
                outline = self.make_glow_outline(mask, outline_color)

        # Replace background
        output = self._replace_same_size(frame, background, mask, mask_inv, outline, outline_opacity)

        return output

    def _replace_same_size(
        self,
        frame: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
        mask_inv: np.ndarray,
        outline: Optional[np.ndarray],
        outline_opacity: float
    ) -> np.ndarray:
        """Replace background when frame and background have same dimensions."""
        if mask is None:
            return frame

        # Extract foreground
        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        # Add outline if requested
        if outline is not None:
            foreground = cv2.addWeighted(foreground, 1.0, outline, outline_opacity, 0)

        # Extract background
        if background is not None:
            background_masked = cv2.bitwise_and(background, background, mask=mask_inv)
        else:
            background_masked = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Combine
        return cv2.add(foreground, background_masked)

    def _replace_different_size(
        self,
        frame: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
        mask_inv: np.ndarray,
        outline: Optional[np.ndarray],
        outline_opacity: float
    ) -> np.ndarray:
        """Replace background when frame and background have different dimensions."""
        h_fg, w_fg = frame.shape[:2]
        h_bg, w_bg = background.shape[:2]

        # Calculate center position
        x = (w_bg - w_fg) // 2
        y = (h_bg - h_fg)  # Align to bottom

        # Ensure we don't go out of bounds
        if x < 0 or y < 0 or x + w_fg > w_bg or y + h_fg > h_bg:
            # If frame is larger than background, resize background
            background = cv2.resize(background, (w_fg, h_fg))
            return self._replace_same_size(frame, background, mask, mask_inv, outline, outline_opacity)

        # Extract foreground
        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        # Add outline if requested
        if outline is not None:
            foreground = cv2.addWeighted(foreground, 1.0, outline, outline_opacity, 0)

        # Extract background ROI
        roi = background[y:y+h_fg, x:x+w_fg]
        background_masked = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Combine and place back
        combined = cv2.add(foreground, background_masked)
        output = background.copy()  # Don't modify original
        output[y:y+h_fg, x:x+w_fg] = combined

        return output

    def _resize_frame_to_background(
        self,
        frame: np.ndarray,
        background: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Resize frame to match background dimensions while maintaining aspect ratio.
        """
        h_fg, w_fg = frame.shape[:2]
        h_bg, w_bg = background.shape[:2]

        # Calculate scale to fit frame to background
        scale_w = w_bg / w_fg
        scale_h = h_bg / h_fg

        # Use the larger scale to ensure frame fills the background
        scale = max(scale_w, scale_h)

        # Calculate new dimensions
        new_w = int(w_fg * scale)
        new_h = int(h_fg * scale)

        # Resize frame
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # If resized frame is larger than background, crop it to fit
        if new_w > w_bg or new_h > h_bg:
            # Center crop
            start_x = (new_w - w_bg) // 2
            start_y = (new_h - h_bg) // 2
            frame_resized = frame_resized[start_y:start_y+h_bg, start_x:start_x+w_bg]

        scale_info = {
            'original_size': (w_fg, h_fg),
            'new_size': (frame_resized.shape[1], frame_resized.shape[0]),
            'scale': scale
        }

        return frame_resized, scale_info

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
