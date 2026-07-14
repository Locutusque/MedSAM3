"""
Image loading utilities shared across the training scripts.

Provides a robust RGB loader that supports the usual JPG/PNG inputs as well as
TIFF images (including the high-bit-depth / single-channel variants that are
common in medical imaging datasets).

Why this exists:
  ``PIL.Image.open(path).convert("RGB")`` works for 8-bit JPG/PNG, but it does
  NOT do the right thing for many TIFFs:
    * 16-bit / 32-bit grayscale ("I;16", "I", "F") images get truncated instead
      of scaled, producing black or clipped images.
    * Multi-page TIFFs silently expose only the first frame.
  This module normalizes any bit depth down to an 8-bit RGB image so the rest of
  the pipeline (which assumes 8-bit RGB) works unchanged.
"""

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

# Image extensions the datasets should pick up when globbing a directory.
SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff",
)

# Modes that indicate a bit depth higher than 8-bit-per-channel (or otherwise
# not directly convertible to RGB without rescaling).
_HIGH_BIT_DEPTH_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")


def _normalize_high_bit_depth(arr: np.ndarray) -> np.ndarray:
    """Min-max scale an array of arbitrary dtype into uint8 [0, 255]."""
    arr = arr.astype(np.float32)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax > amin:
        arr = (arr - amin) / (amax - amin) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def load_image_as_rgb(path) -> PILImage.Image:
    """
    Load an image from ``path`` and return an 8-bit RGB ``PIL.Image``.

    Handles JPG/PNG/BMP as well as TIFF, including 16-bit and single-channel
    TIFFs common in medical imaging. Multi-page TIFFs use the first frame.
    """
    path = Path(path)

    img = PILImage.open(path)

    # For multi-page TIFFs, PIL defaults to the first frame; make it explicit.
    try:
        img.seek(0)
    except (EOFError, AttributeError):
        pass

    mode = img.mode

    if mode in _HIGH_BIT_DEPTH_MODES:
        # High bit depth (e.g. 16-bit medical scans): rescale to 8-bit.
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = _normalize_high_bit_depth(arr)
        img = PILImage.fromarray(arr)
    elif mode not in ("RGB", "RGBA", "L", "P", "LA"):
        # Uncommon multi-channel / high-bit TIFF layouts that PIL exposes as a
        # numpy-convertible array. Fall back to array-based normalization.
        arr = np.array(img)
        if arr.dtype != np.uint8:
            if arr.ndim == 3 and arr.shape[2] >= 3:
                # Per-channel min-max scaling for multi-channel high-bit images.
                arr = np.stack(
                    [_normalize_high_bit_depth(arr[..., c]) for c in range(3)],
                    axis=-1,
                )
            else:
                arr = _normalize_high_bit_depth(arr)
        img = PILImage.fromarray(arr)

    return img.convert("RGB")
