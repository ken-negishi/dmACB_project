"""Kernel-density analysis of anatomical projection maps.

This module contains the spatial kernel-density procedures used for the
associated publication. Segmented anatomical objects are represented by
their centroids, converted to two-dimensional Gaussian kernel-density
estimates (KDEs), normalized as probability-density maps, and combined to
quantify spatial overlap between projection-defined territories.

SVG coordinates are retained throughout so that KDE maps can be compared
directly with anatomical ROI boundaries.
"""

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.path import Path as MplPath
from svgpathtools import svg2paths, parser, CubicBezier, QuadraticBezier, Line


def find_centroids(svg_file):
    """Return length-weighted centroids of objects in an SVG file.

    Parameters
    ----------
    svg_file : str or pathlib.Path
        SVG containing segmented objects.

    Returns
    -------
    list of tuple[float, float]
        Object centroid coordinates.
    """
    objects, _ = svg2paths(str(svg_file))
    centroids = []

    for obj in objects:
        for subpath in obj.continuous_subpaths():
            centroid = np.array([0.0, 0.0])
            total_length = 0.0

            for segment in subpath:
                length = segment.length()
                midpoint = 0.5 * (segment.start + segment.end)
                centroid += np.array(
                    [midpoint.real, midpoint.imag]
                ) * length
                total_length += length

            if total_length > 0:
                centroid /= total_length
                centroids.append(
                    (float(centroid[0]), float(centroid[1]))
                )

    return centroids


def centroid_kde(svg_file, bandwidth=0.005, grid_scale=2):
    """Calculate a 2-D Gaussian KDE from segmented-object centroids.

    Parameters
    ----------
    svg_file : str or pathlib.Path
        SVG containing segmented anatomical objects.
    bandwidth : float, default=0.005
        ``bw_method`` passed to ``scipy.stats.gaussian_kde``.
    grid_scale : int, default=2
        Number of KDE evaluation points per SVG coordinate unit.

    Returns
    -------
    dict
        Dictionary containing X/Y grids, density values, SVG dimensions,
        and the area represented by one KDE grid element.
    """
    svg_file = Path(svg_file)

    tree = ET.parse(svg_file)
    root = tree.getroot()
    viewbox = root.get("viewBox")

    if viewbox is None:
        raise ValueError(f"SVG has no viewBox: {svg_file}")

    _, _, width, height = map(float, viewbox.split())
    width = math.ceil(width)
    height = math.ceil(height)

    centroids = find_centroids(svg_file)

    nx = max(2, int(width * grid_scale))
    ny = max(2, int(height * grid_scale))

    x_values = np.linspace(0, width, num=nx)
    y_values = np.linspace(0, height, num=ny)
    X, Y = np.meshgrid(x_values, y_values)

    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]
    grid_area = abs(dx * dy)

    if not centroids:
        density = np.zeros_like(X, dtype=float)
    else:
        data = np.asarray(centroids).T
        kde = gaussian_kde(data, bw_method=bandwidth)
        evaluation_points = np.vstack([X.ravel(), Y.ravel()])
        density = kde(evaluation_points).reshape(X.shape)

    return {
        "X": X,
        "Y": Y,
        "density": density,
        "width": width,
        "height": height,
        "grid_area": grid_area,
        "source": svg_file.name,
    }


def normalize_density(kde_result):
    """Normalize a KDE so its numerical integral equals 1."""
    density = kde_result["density"]
    integral = np.sum(density) * kde_result["grid_area"]

    result = dict(kde_result)

    if integral > 0:
        result["density"] = density / integral
    else:
        result["density"] = density.copy()

    return result


def average_density(kde_results):
    """Calculate the mean of individually normalized KDE maps.

    Each input map is first normalized to unit integral so that each
    specimen contributes equally to the group-average spatial distribution.
    """
    if not kde_results:
        raise ValueError("At least one KDE result is required.")

    normalized = [normalize_density(kde) for kde in kde_results]

    reference_shape = normalized[0]["density"].shape
    if any(kde["density"].shape != reference_shape for kde in normalized):
        raise ValueError("All KDE maps must have matching dimensions.")

    result = dict(normalized[0])
    result["density"] = np.mean(
        [kde["density"] for kde in normalized],
        axis=0,
    )
    result["source"] = "group_average"

    return result


def probability_mass_threshold(kde_result, mass=0.95):
    """Return the density threshold enclosing a specified probability mass.

    Parameters
    ----------
    kde_result : dict
        Unit-normalized KDE result.
    mass : float, default=0.95
        Fraction of total probability mass to retain.

    Returns
    -------
    float
        Density threshold defining the highest-density region containing
        the requested probability mass.
    """
    if not 0 < mass <= 1:
        raise ValueError("mass must be greater than 0 and no greater than 1")

    normalized = normalize_density(kde_result)
    density = normalized["density"]

    sorted_density = np.sort(density.ravel())[::-1]
    cumulative_mass = (
        np.cumsum(sorted_density) * normalized["grid_area"]
    )

    index = np.searchsorted(cumulative_mass, mass)
    index = min(index, len(sorted_density) - 1)

    return float(sorted_density[index])


def density_product(*kde_results):
    """Calculate the pointwise product of two or more KDE maps.

    This operation was used to identify spatial intersections among
    projection-defined anatomical density maps.
    """
    if len(kde_results) < 2:
        raise ValueError("At least two KDE maps are required.")

    shape = kde_results[0]["density"].shape
    if any(kde["density"].shape != shape for kde in kde_results):
        raise ValueError("All KDE maps must have matching dimensions.")

    result = dict(kde_results[0])
    result["density"] = np.prod(
        [kde["density"] for kde in kde_results],
        axis=0,
    )
    result["source"] = "density_product"

    return result


def _svg_path_to_mpl(element):
    """Convert an SVG path or polygon element to a Matplotlib Path."""
    tag = element.tag.split("}")[-1]

    if tag == "polygon":
        raw = element.get("points", "").replace(",", " ").split()
        values = [float(value) for value in raw]
        coordinates = [
            (values[i], values[i + 1])
            for i in range(0, len(values), 2)
        ]

        vertices = coordinates + [coordinates[0]]
        codes = (
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(coordinates) - 1)
            + [MplPath.CLOSEPOLY]
        )
        return MplPath(vertices, codes)

    parsed_path = parser.parse_path(element.attrib["d"])
    vertices = []
    codes = []

    for subpath in parsed_path.continuous_subpaths():
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

    vertices.append(vertices[0])
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(vertices, codes)


def roi_masks(roi_svg, roi_layer, X, Y):
    """Rasterize named anatomical ROIs onto a KDE evaluation grid.

    Parameters
    ----------
    roi_svg : str or pathlib.Path
        SVG containing anatomical ROI boundaries.
    roi_layer : str
        SVG group ID containing the ROIs.
    X, Y : numpy.ndarray
        KDE coordinate grids.

    Returns
    -------
    dict
        Mapping ROI names to Boolean masks.
    """
    tree = ET.parse(roi_svg)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    layer = root.find(f'.//svg:g[@id="{roi_layer}"]', ns)
    if layer is None and root.get("id") == roi_layer:
        layer = root

    if layer is None:
        raise ValueError(f"ROI layer '{roi_layer}' was not found.")

    elements = (
        layer.findall(".//svg:path", ns)
        + layer.findall(".//svg:polygon", ns)
    )

    points = np.column_stack([X.ravel(), Y.ravel()])
    masks = {}

    for index, element in enumerate(elements):
        name = element.get("id") or f"ROI_{index + 1}"
        path = _svg_path_to_mpl(element)
        masks[name] = path.contains_points(points).reshape(X.shape)

    return masks


def quantify_density_by_roi(kde_result, masks):
    """Quantify KDE probability mass falling within each anatomical ROI.

    Returns
    -------
    dict
        Mapping ROI names to integrated density within each ROI.
    """
    density = kde_result["density"]
    grid_area = kde_result["grid_area"]

    return {
        name: float(np.sum(density[mask]) * grid_area)
        for name, mask in masks.items()
    }
