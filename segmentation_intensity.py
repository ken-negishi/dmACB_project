"""Intensity analysis for segmented microscopy objects.

This module contains the intensity measurements used to evaluate segmented
Fos-positive objects in the associated publication. It measures mean signal
within SVG segmentation outlines, samples background pixels outside those
objects, and can export signal-versus-background intensity distributions.

SVG coordinates are assumed to use the standard 72-dpi coordinate system.
Microscopy images in the study were exported at 300 dpi.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree
from svgpathtools import svg2paths

Image.MAX_IMAGE_PIXELS = None


def _path_centroids(paths):
    """Calculate length-weighted centroids for SVG paths."""
    centroids = []

    for path in paths:
        centroid = np.array([0.0, 0.0])
        total_length = 0.0

        for subpath in path.continuous_subpaths():
            for segment in subpath:
                length = segment.length()
                midpoint = 0.5 * (segment.start + segment.end)
                centroid += np.array(
                    [midpoint.real, midpoint.imag]
                ) * length
                total_length += length

        if total_length > 0:
            centroids.append(centroid / total_length)
        else:
            centroids.append(np.array([np.nan, np.nan]))

    return np.asarray(centroids)


def _remove_redundant_paths(paths, tolerance=3):
    """Remove segmentation paths with centroids within ``tolerance`` units.

    When two objects are identified as redundant, the later SVG path is
    removed. This preserves the duplicate-removal behavior of the analysis
    pipeline used for the publication.
    """
    if not paths:
        return []

    centroids = _path_centroids(paths)
    valid = np.all(np.isfinite(centroids), axis=1)

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        return paths

    tree = cKDTree(centroids[valid])
    close_pairs = tree.query_pairs(r=tolerance)

    remove_indices = {
        max(valid_indices[i], valid_indices[j])
        for i, j in close_pairs
    }

    return [
        path for index, path in enumerate(paths)
        if index not in remove_indices
    ]


def _path_coordinates(path, scale):
    """Convert an SVG path to image-pixel coordinates."""
    coordinates = []

    for subpath in path.continuous_subpaths():
        for segment in subpath:
            coordinates.append(
                (segment.start.real * scale, segment.start.imag * scale)
            )

        if len(subpath):
            coordinates.append(
                (subpath[-1].end.real * scale, subpath[-1].end.imag * scale)
            )

    return coordinates


def _segmentation_label_image(paths, image_shape, scale):
    """Rasterize SVG segmentation outlines as uniquely labeled objects."""
    height, width = image_shape
    label_image = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(label_image)

    for index, path in enumerate(paths):
        coordinates = _path_coordinates(path, scale)

        if len(coordinates) >= 3:
            draw.polygon(coordinates, fill=index + 1)

    return np.asarray(label_image, dtype=np.uint32)


def average_intensity_per_object(
    image_path,
    segmentation_svg,
    dpi=300,
    remove_duplicates=True,
    duplicate_tolerance=3,
):
    """Measure mean grayscale intensity within each segmented object.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Grayscale microscopy image.
    segmentation_svg : str or pathlib.Path
        SVG containing segmentation outlines.
    dpi : float, default=300
        Resolution of the microscopy image.
    remove_duplicates : bool, default=True
        Remove spatially redundant segmentation paths before measurement.
    duplicate_tolerance : float, default=3
        Maximum centroid distance used to identify redundant objects.

    Returns
    -------
    numpy.ndarray
        Mean grayscale intensity for each segmented object.
    """
    image_path = Path(image_path)
    segmentation_svg = Path(segmentation_svg)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not segmentation_svg.is_file():
        raise FileNotFoundError(
            f"Segmentation SVG not found: {segmentation_svg}"
        )

    with Image.open(image_path) as image:
        image_array = np.asarray(image.convert("L"))

    paths, _ = svg2paths(str(segmentation_svg))

    if remove_duplicates:
        paths = _remove_redundant_paths(
            paths,
            tolerance=duplicate_tolerance,
        )

    scale = float(dpi) / 72.0
    labels = _segmentation_label_image(
        paths,
        image_array.shape,
        scale,
    )

    object_ids = np.arange(1, len(paths) + 1)

    means = ndimage.mean(
        image_array,
        labels=labels,
        index=object_ids,
    )

    return np.nan_to_num(np.asarray(means), nan=0.0)


def signal_and_background_intensity(
    image_path,
    segmentation_svg,
    dpi=300,
    remove_duplicates=True,
    duplicate_tolerance=3,
    exclude_zero=True,
):
    """Measure segmented-object signal and unsegmented background intensity.

    Background is defined as image pixels that do not fall within any
    segmentation outline. Zero-valued pixels can optionally be excluded,
    matching the analysis used for the publication.

    Returns
    -------
    object_intensity : numpy.ndarray
        Mean grayscale intensity for each segmented object.
    background_intensity : numpy.ndarray
        Intensities of background pixels outside segmented objects.
    """
    image_path = Path(image_path)
    segmentation_svg = Path(segmentation_svg)

    with Image.open(image_path) as image:
        image_array = np.asarray(image.convert("L"))

    paths, _ = svg2paths(str(segmentation_svg))

    if remove_duplicates:
        paths = _remove_redundant_paths(
            paths,
            tolerance=duplicate_tolerance,
        )

    scale = float(dpi) / 72.0
    labels = _segmentation_label_image(
        paths,
        image_array.shape,
        scale,
    )

    object_ids = np.arange(1, len(paths) + 1)
    object_intensity = ndimage.mean(
        image_array,
        labels=labels,
        index=object_ids,
    )
    object_intensity = np.nan_to_num(
        np.asarray(object_intensity),
        nan=0.0,
    )

    background_mask = labels == 0
    if exclude_zero:
        background_mask &= image_array != 0

    background_intensity = image_array[background_mask]

    return object_intensity, background_intensity


def intensity_histograms(
    image_path,
    segmentation_svg,
    bins=None,
    dpi=300,
    remove_duplicates=True,
    duplicate_tolerance=3,
):
    """Calculate normalized signal and background intensity histograms.

    The default bin edges reproduce the 0-255 grayscale histogram binning
    used in the analysis (4 intensity units per bin).
    """
    if bins is None:
        bins = np.arange(0, 260, 4)

    signal, background = signal_and_background_intensity(
        image_path=image_path,
        segmentation_svg=segmentation_svg,
        dpi=dpi,
        remove_duplicates=remove_duplicates,
        duplicate_tolerance=duplicate_tolerance,
    )

    signal_hist, edges = np.histogram(
        signal,
        bins=bins,
        density=True,
    )
    background_hist, _ = np.histogram(
        background,
        bins=bins,
        density=True,
    )

    return signal_hist, background_hist, edges


def save_intensity_histograms(
    image_segmentation_pairs,
    output_csv,
    bins=None,
    dpi=300,
):
    """Save signal/background histograms for multiple image-SVG pairs.

    Parameters
    ----------
    image_segmentation_pairs : iterable
        Iterable of ``(name, image_path, segmentation_svg)`` tuples.
    output_csv : str or pathlib.Path
        Destination CSV file.
    bins : array-like or None
        Histogram bin edges. Defaults to 0-256 in 4-unit intervals.
    dpi : float, default=300
        Resolution of the microscopy images.

    Returns
    -------
    pathlib.Path
        Path to the saved CSV file.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tables = []

    for name, image_path, segmentation_svg in image_segmentation_pairs:
        signal_hist, background_hist, edges = intensity_histograms(
            image_path=image_path,
            segmentation_svg=segmentation_svg,
            bins=bins,
            dpi=dpi,
        )

        tables.append(
            pd.DataFrame(
                {
                    "sample": name,
                    "intensity_bin": edges[:-1],
                    "segmented_object_density": signal_hist,
                    "background_density": background_hist,
                }
            )
        )

    if not tables:
        raise ValueError("No image/segmentation pairs were provided.")

    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output_csv, index=False)

    return output_csv
