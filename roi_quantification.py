"""Quantification of segmented objects within anatomical ROIs.

This module contains the ROI-based quantification used for the associated
publication. Segmented objects and anatomical regions are read from SVG
files, object centroids are assigned to ROIs, ROI areas are calculated in
square millimeters, and object counts and densities are exported to CSV.

The study images were acquired with an Olympus VS200 at 10x and exported at
300 dpi. The default area conversion therefore preserves the calibration
used in the original analysis.
"""

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from svgpathtools import parser, CubicBezier, QuadraticBezier, Line


# Area represented by one SVG coordinate unit squared for the acquisition
# and export settings used in the study.
AREA_PER_SVG_UNIT_SQ_MM = 0.000000422100344536 * (300 / 72) ** 2


def _parse_svg_points(points):
    """Parse SVG polygon/polyline point coordinates."""
    points = points.strip()
    if "," in points:
        values = [
            value
            for pair in points.split()
            for value in pair.split(",")
        ]
    else:
        values = points.split()

    return [
        (float(values[i]), float(values[i + 1]))
        for i in range(0, len(values), 2)
    ]


def _element_centroid(element):
    """Calculate the centroid used by the original analysis pipeline."""
    tag = element.tag.split("}")[-1]

    if tag == "path":
        parsed_path = parser.parse_path(element.attrib["d"])
        centroid = np.array([0.0, 0.0])
        total_length = 0.0

        for subpath in parsed_path.continuous_subpaths():
            for segment in subpath:
                length = segment.length()
                midpoint = 0.5 * (segment.start + segment.end)
                centroid += np.array(
                    [midpoint.real, midpoint.imag]
                ) * length
                total_length += length

        if total_length == 0:
            return None

        return tuple(centroid / total_length)

    if tag in ("polyline", "polygon"):
        coordinates = _parse_svg_points(element.get("points", ""))
        if not coordinates:
            return None

        return (
            float(np.mean([p[0] for p in coordinates])),
            float(np.mean([p[1] for p in coordinates])),
        )

    return None


def extract_segmentations(svg_file, layer_name="seg"):
    """Extract object centroids from segmentation layers in an SVG.

    Parameters
    ----------
    svg_file : str or pathlib.Path
        SVG containing segmented objects.
    layer_name : str, default="seg"
        Parent SVG group containing segmentation subgroups.

    Returns
    -------
    dict
        Mapping ``segmentation_name -> list of (x, y) centroids``.
    """
    svg_file = Path(svg_file)
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    if root.get("id") == layer_name:
        layer = root
    else:
        layer = root.find(f'.//svg:g[@id="{layer_name}"]', ns)

    if layer is None:
        raise ValueError(
            f"Layer '{layer_name}' was not found in {svg_file}"
        )

    subgroups = layer.findall("svg:g", ns)

    # If the parent contains objects directly rather than named subgroups,
    # quantify them as a single segmentation class.
    groups = subgroups if subgroups else [layer]

    segmentations = {}

    for group in groups:
        name = group.get("id") or layer_name
        elements = (
            group.findall(".//svg:path", ns)
            + group.findall(".//svg:polyline", ns)
            + group.findall(".//svg:polygon", ns)
        )

        centroids = []
        for element in elements:
            centroid = _element_centroid(element)
            if centroid is not None:
                centroids.append(centroid)

        segmentations[name] = centroids

    return segmentations


def _path_to_mpl(svg_path):
    """Convert an svgpathtools path to a closed Matplotlib Path."""
    vertices = []
    codes = []

    for subpath in svg_path.continuous_subpaths():
        for index, segment in enumerate(subpath):
            if index == 0:
                vertices.append(
                    (segment.start.real, segment.start.imag)
                )
                codes.append(MplPath.MOVETO)

            if isinstance(segment, CubicBezier):
                vertices.extend(
                    [
                        (segment.control1.real, segment.control1.imag),
                        (segment.control2.real, segment.control2.imag),
                        (segment.end.real, segment.end.imag),
                    ]
                )
                codes.extend(
                    [MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
                )
            elif isinstance(segment, QuadraticBezier):
                vertices.extend(
                    [
                        (segment.control.real, segment.control.imag),
                        (segment.end.real, segment.end.imag),
                    ]
                )
                codes.extend([MplPath.CURVE3, MplPath.CURVE3])
            elif isinstance(segment, Line):
                vertices.append(
                    (segment.end.real, segment.end.imag)
                )
                codes.append(MplPath.LINETO)

    if not vertices:
        raise ValueError("ROI path contains no usable coordinates.")

    vertices.append(vertices[0])
    codes.append(MplPath.CLOSEPOLY)

    return MplPath(vertices, codes)


def _polygon_to_mpl(points):
    """Convert SVG polygon coordinates to a closed Matplotlib Path."""
    coordinates = _parse_svg_points(points)
    if len(coordinates) < 3:
        raise ValueError("ROI polygon contains fewer than three points.")

    vertices = coordinates + [coordinates[0]]
    codes = (
        [MplPath.MOVETO]
        + [MplPath.LINETO] * (len(coordinates) - 1)
        + [MplPath.CLOSEPOLY]
    )
    return MplPath(vertices, codes)


def _roi_area_mm2(path, area_per_unit_sq=AREA_PER_SVG_UNIT_SQ_MM):
    """Calculate ROI area using the rasterized procedure used in the study."""
    x_min, y_min = np.min(path.vertices, axis=0)
    x_max, y_max = np.max(path.vertices, axis=0)

    x_values = np.arange(np.floor(x_min), np.ceil(x_max) + 1, 1)
    y_values = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1)
    xx, yy = np.meshgrid(x_values, y_values)

    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = path.contains_points(points)

    return float(np.count_nonzero(inside) * area_per_unit_sq)


def extract_rois(
    svg_file,
    layer_name="ROI",
    area_per_unit_sq=AREA_PER_SVG_UNIT_SQ_MM,
):
    """Extract named anatomical ROIs and calculate their areas.

    Returns
    -------
    list of dict
        Each dictionary contains ``name``, ``path``, and ``area_mm2``.
    """
    svg_file = Path(svg_file)
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    layer = root.find(f'.//svg:g[@id="{layer_name}"]', ns)
    if layer is None and root.get("id") == layer_name:
        layer = root

    if layer is None:
        raise ValueError(
            f"ROI layer '{layer_name}' was not found in {svg_file}"
        )

    elements = (
        layer.findall(".//svg:path", ns)
        + layer.findall(".//svg:polygon", ns)
    )

    rois = []

    for index, element in enumerate(elements):
        tag = element.tag.split("}")[-1]

        if tag == "path":
            roi_path = _path_to_mpl(
                parser.parse_path(element.attrib["d"])
            )
        else:
            roi_path = _polygon_to_mpl(element.get("points", ""))

        name = element.get("id") or f"ROI_{index + 1}"

        rois.append(
            {
                "name": name,
                "path": roi_path,
                "area_mm2": _roi_area_mm2(
                    roi_path,
                    area_per_unit_sq=area_per_unit_sq,
                ),
            }
        )

    return rois


def quantify_rois(segmentations, rois):
    """Count segmented objects within each ROI and calculate densities.

    Object membership is determined from the segmentation centroid, matching
    the analysis procedure used for the publication.

    Returns
    -------
    pandas.DataFrame
        One row per ROI with area, object counts, and objects/mm2 for each
        segmentation class.
    """
    rows = []

    for roi in rois:
        row = {
            "ROI": roi["name"],
            "Area_mm2": roi["area_mm2"],
        }

        for segmentation_name, centroids in segmentations.items():
            count = sum(
                roi["path"].contains_point(centroid)
                for centroid in centroids
            )

            row[f"{segmentation_name}_count"] = count
            row[f"{segmentation_name}_per_mm2"] = (
                count / roi["area_mm2"]
                if roi["area_mm2"] > 0
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def quantify_svg(
    svg_file,
    output_csv=None,
    segmentation_layer="seg",
    roi_layer="ROI",
    area_per_unit_sq=AREA_PER_SVG_UNIT_SQ_MM,
):
    """Quantify segmented objects within ROIs from one combined SVG.

    Parameters
    ----------
    svg_file : str or pathlib.Path
        SVG containing segmentation and ROI layers.
    output_csv : str, pathlib.Path, or None
        Optional destination for the resulting CSV.
    segmentation_layer : str, default="seg"
        Parent layer containing segmented objects.
    roi_layer : str, default="ROI"
        Layer containing anatomical ROI boundaries.
    area_per_unit_sq : float
        Conversion from one squared SVG coordinate unit to mm2.

    Returns
    -------
    pandas.DataFrame
        ROI counts and densities.
    """
    segmentations = extract_segmentations(
        svg_file,
        layer_name=segmentation_layer,
    )
    rois = extract_rois(
        svg_file,
        layer_name=roi_layer,
        area_per_unit_sq=area_per_unit_sq,
    )

    result = quantify_rois(segmentations, rois)
    result.insert(0, "File", Path(svg_file).name)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)

    return result
