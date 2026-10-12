"""Cellpose-SAM segmentation used for microscopy image analysis.

Applies a trained Cellpose model to large grayscale microscopy images.
Images are divided into overlapping tiles, segmented independently, and
object outlines are returned to the original image coordinate system and
saved as SVG.

For general instructions on training custom Cellpose models, see the
official Cellpose documentation.
"""

from pathlib import Path
import numpy as np
import svgwrite
from cellpose import models, utils
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def split_image_into_tiles(image_path, tile_size=5000, overlap=50):
    """Split a microscopy image into overlapping square tiles."""
    with Image.open(image_path) as image:
        image_array = np.asarray(image)

    height, width = image_array.shape[:2]
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    x_starts = list(range(0, max(width - tile_size + 1, 1), step))
    y_starts = list(range(0, max(height - tile_size + 1, 1), step))
    final_x = max(0, width - tile_size)
    final_y = max(0, height - tile_size)

    if final_x not in x_starts:
        x_starts.append(final_x)
    if final_y not in y_starts:
        y_starts.append(final_y)

    tiles, starts = [], []
    for y_start in y_starts:
        for x_start in x_starts:
            tiles.append(
                image_array[
                    y_start:min(y_start + tile_size, height),
                    x_start:min(x_start + tile_size, width),
                ]
            )
            starts.append((x_start, y_start))
    return tiles, starts


def _outlines_to_image_coordinates(outlines, x_start, y_start, scale_factor):
    """Translate tile-local outlines into full-image SVG coordinates."""
    translated = []
    for outline in outlines:
        if len(outline) < 2:
            continue
        points = [
            ((float(x) + x_start) * scale_factor,
             (float(y) + y_start) * scale_factor)
            for x, y in outline
        ]
        if points[0] != points[-1]:
            points.append(points[0])
        translated.append(points)
    return translated


def _remove_overlap_duplicates(objects):
    """Resolve duplicate objects generated in overlapping image tiles.

    When outlines from overlapping tiles share a coordinate, the outline
    containing the larger number of vertices is retained. This preserves
    the behavior of the analysis pipeline used for the publication.
    """
    unique_objects = []
    for points in objects:
        duplicate_index = None
        for index, existing in enumerate(unique_objects):
            if any(point in existing for point in points):
                duplicate_index = index
                break

        if duplicate_index is None:
            unique_objects.append(points)
        elif len(points) >= len(unique_objects[duplicate_index]):
            unique_objects[duplicate_index] = points
    return unique_objects


def run_cellpose(
    image_path,
    model_path,
    output_svg,
    tile_size=5000,
    overlap=50,
    dpi=300,
    gpu=True,
):
    """Segment an image with a trained Cellpose model and save SVG outlines.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Input microscopy image.
    model_path : str or pathlib.Path
        Trained Cellpose model.
    output_svg : str or pathlib.Path
        Output SVG segmentation file.
    tile_size : int
        Width and height of image tiles in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    dpi : float
        Source image resolution. Pixel coordinates are converted to
        72-dpi SVG coordinates using 72 / dpi.
    gpu : bool
        Request Cellpose GPU acceleration.

    Returns
    -------
    pathlib.Path
        Saved SVG path.
    """
    image_path = Path(image_path)
    model_path = Path(model_path)
    output_svg = Path(output_svg)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Cellpose model not found: {model_path}")

    output_svg.parent.mkdir(parents=True, exist_ok=True)

    tiles, starts = split_image_into_tiles(
        image_path, tile_size=tile_size, overlap=overlap
    )

    model = models.CellposeModel(
        pretrained_model=str(model_path),
        gpu=gpu,
    )

    scale_factor = 72.0 / float(dpi)
    all_objects = []

    for tile, (x_start, y_start) in zip(tiles, starts):
        masks, _, _ = model.eval(tile)
        outlines = utils.outlines_list(masks)
        all_objects.extend(
            _outlines_to_image_coordinates(
                outlines, x_start, y_start, scale_factor
            )
        )

    unique_objects = _remove_overlap_duplicates(all_objects)

    drawing = svgwrite.Drawing(str(output_svg), profile="tiny")
    group = drawing.add(
        drawing.g(id="segmentation", stroke="black", fill="none")
    )
    for points in unique_objects:
        group.add(drawing.polyline(points, stroke="black", fill="none"))

    drawing.save()
    return output_svg
