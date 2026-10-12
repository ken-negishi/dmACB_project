"""Linear intensity scaling used for image preprocessing.

This module contains the custom linear rescaling step used to convert source
images to 8-bit PNG images before downstream segmentation and quantification.
"""

from pathlib import Path

import numpy as np
from PIL import Image


# Permit loading of very large whole-slide-derived images.
Image.MAX_IMAGE_PIXELS = None


def linear_scaling(image_path, upper, lower, output_dir):
    """Linearly rescale an image to the 8-bit range and save it as PNG.

    Pixel intensities at or below ``lower`` are mapped to 0, intensities at or
    above ``upper`` are mapped to 255, and intermediate values are linearly
    rescaled between those limits.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the input image.
    upper : int or float
        Upper intensity bound mapped to 255.
    lower : int or float
        Lower intensity bound mapped to 0.
    output_dir : str or pathlib.Path
        Directory in which to save the rescaled PNG image.

    Returns
    -------
    pathlib.Path
        Path to the saved 8-bit PNG image.
    """
    if upper <= lower:
        raise ValueError("upper must be greater than lower")

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        image_array = np.asarray(image, dtype=np.float64)

    scale_factor = round(255.0 / (upper - lower), 10)
    scaled = (image_array - lower) * scale_factor
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)

    output_path = output_dir / f"{image_path.stem}_LS_{lower}_{upper}.png"
    Image.fromarray(scaled).save(output_path, "PNG", dpi=(300, 300))

    return output_path


# -----------------------------------------------------------------------------
# Optional GPU-accelerated implementation (not required for reproduction)
# -----------------------------------------------------------------------------
# The function below performs the same linear rescaling operation using CuPy
# for GPU acceleration. It is retained as commented reference code because GPU
# acceleration can be useful for very large images. To use it, install CuPy,
# uncomment the import and function, and ensure a compatible CUDA environment.
#
# import cupy as cp
#
# def linear_scaling_cuda(image_path, upper, lower, output_dir, invert=False):
#     """GPU-accelerated linear intensity scaling using CuPy."""
#     if upper <= lower:
#         raise ValueError("upper must be greater than lower")
#
#     image_path = Path(image_path)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     with Image.open(image_path) as image:
#         image_array = np.asarray(image, dtype=np.float32)
#
#     gpu_array = cp.asarray(image_array)
#     scale_factor = cp.round(255.0 / (upper - lower), 10).astype(cp.float32)
#     gpu_array -= lower
#     gpu_array *= scale_factor
#
#     if invert:
#         gpu_array = 255 - gpu_array
#
#     cp.clip(gpu_array, 0, 255, out=gpu_array)
#     scaled = cp.asnumpy(gpu_array.astype(cp.uint8))
#
#     del gpu_array
#     cp.get_default_memory_pool().free_all_blocks()
#
#     output_path = output_dir / f"{image_path.stem}_LS_{lower}_{upper}.png"
#     Image.fromarray(scaled).save(output_path, "PNG", dpi=(300, 300))
#
#     return output_path
