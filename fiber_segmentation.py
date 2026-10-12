"""U-Net inference for PHAL-labeled axon segmentation.

This module contains the inference procedure used to segment PHAL-labeled
axons for the associated publication. A trained single-channel U-Net is
loaded from a PyTorch checkpoint and applied to grayscale microscopy images.

The network architecture itself is imported from the U-Net implementation
used to train the model and is not redefined here.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from unet import UNet
from data_loading import BasicDataset

Image.MAX_IMAGE_PIXELS = None


def predict_mask(model, image, device, scale_factor=1.0):
    """Return the continuous U-Net prediction for a grayscale image.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded U-Net model.
    image : PIL.Image.Image
        Grayscale input image.
    device : torch.device
        CPU or CUDA device used for inference.
    scale_factor : float, default=1.0
        Image scaling factor passed to the U-Net preprocessing routine.

    Returns
    -------
    numpy.ndarray
        Continuous prediction values in the range [0, 1].
    """
    model.eval()

    image_tensor = torch.from_numpy(
        BasicDataset.preprocess(image, scale_factor, is_mask=False)
    )
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)[0]

        resize = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image.size[1], image.size[0])),
                transforms.ToTensor(),
            ]
        )
        prediction = resize(prediction.cpu()).squeeze()

    return prediction.numpy()


def load_unet(model_path, device=None):
    """Load the trained single-channel U-Net used for axon segmentation."""
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"U-Net checkpoint not found: {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = UNet(n_channels=1, n_classes=1, bilinear=False)
    model.to(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model, device


def segment_fibers(image, model, device, threshold=0.3):
    """Segment PHAL-labeled axons in an image.

    Parameters
    ----------
    image : str, pathlib.Path, PIL.Image.Image, or numpy.ndarray
        Grayscale microscopy image.
    model : torch.nn.Module
        Loaded U-Net model.
    device : torch.device
        Device on which the model is running.
    threshold : float, default=0.3
        Threshold applied to the continuous U-Net prediction.

    Returns
    -------
    prediction : numpy.ndarray
        Continuous U-Net prediction in the range [0, 1].
    binary_mask : numpy.ndarray
        Binary axon segmentation mask.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            pil_image = opened.convert("L").copy()
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)
        if image.ndim != 2:
            raise ValueError("NumPy input must be a 2-D grayscale image.")
        pil_image = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Image.Image):
        pil_image = image.convert("L")
    else:
        raise TypeError(
            "image must be a path, PIL image, or 2-D NumPy array"
        )

    prediction = predict_mask(model, pil_image, device)
    binary_mask = (prediction > threshold).astype(np.uint8)

    return prediction, binary_mask


def segment_image(image_path, model_path, output_path, threshold=0.3, device=None):
    """Apply the trained U-Net to one image and save the binary axon mask.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Input grayscale microscopy image.
    model_path : str or pathlib.Path
        PyTorch checkpoint for the trained U-Net.
    output_path : str or pathlib.Path
        Destination for the binary PNG segmentation mask.
    threshold : float, default=0.3
        Threshold applied to the continuous model prediction.
    device : str, torch.device, or None
        Inference device. If None, CUDA is used when available.

    Returns
    -------
    pathlib.Path
        Path to the saved binary segmentation mask.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, device = load_unet(model_path, device=device)
    _, binary_mask = segment_fibers(
        image_path,
        model,
        device,
        threshold=threshold,
    )

    Image.fromarray(binary_mask * 255).convert("1").save(output_path)
    return output_path
