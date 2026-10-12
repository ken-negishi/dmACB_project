# Analysis code for Negishi et al., *Journal of Comparative Neurology* (2026)

This repository contains custom Python code used for image processing and
quantitative anatomical analyses reported in the associated article (https://doi.org/10.1002/cne.70209).

The repository is a publication-specific code snapshot rather than a
general-purpose software package. It contains the custom procedures needed
to document the analyses performed for the study; standard third-party
software functionality is not reimplemented here.

## Repository contents

### `linear_scaling.py`
Converts microscopy images to 8-bit grayscale by linearly mapping a specified
lower and upper intensity bound to 0 and 255, respectively. The executable
implementation is CPU-based. A commented CuPy implementation is retained as
an optional GPU-accelerated equivalent.

### `run_cellpose.py`
Applies trained Cellpose-SAM models to large microscopy images. Images are
divided into overlapping tiles, segmented independently, and object outlines
are transformed back to full-image coordinates and saved as SVG. Duplicate
objects arising from overlapping tiles are resolved by retaining the larger
outline when two outlines share a coordinate.

This repository documents the inference procedure used in the study. General
instructions for creating and training custom Cellpose models should be
obtained from the official Cellpose documentation.

### `fiber_segmentation.py`
Applies a trained single-channel U-Net to grayscale microscopy images for
segmentation of PHAL-labeled axons. Continuous model predictions are
thresholded to generate binary axon masks.

This script uses the `UNet` architecture and `BasicDataset.preprocess` implementation
from the Pytorch-UNet project by Milesial:
https://github.com/milesial/Pytorch-UNet

For exact reproduction, use the same Pytorch-UNet commit/version used for model
training together with the trained model checkpoint.

### `segmentation_intensity.py`
Measures mean grayscale intensity within segmented objects and samples
background pixels outside segmented objects. It also generates normalized
signal-versus-background intensity histograms.

### `roi_quantification.py`
Quantifies segmented objects within anatomical regions of interest (ROIs).
Object membership is determined from segmentation centroids. ROI areas are
calculated in square millimeters, and object counts and densities
(objects/mm²) are returned for each ROI.

### `kernel_density.py`
Generates two-dimensional Gaussian kernel-density estimates (KDEs) from
segmented-object centroids. The module also supports normalization of
individual density maps, group averaging, highest-density probability-mass
thresholds, pointwise intersection of density maps, ROI masks, and
quantification of density within anatomical ROIs.

## Analysis workflow

The custom analysis workflow used in the study can be summarized as:

1. Convert microscopy images to 8-bit grayscale using linear intensity
   scaling (`linear_scaling.py`).
2. Segment labeled cells with trained Cellpose-SAM models
   (`run_cellpose.py`) or segment PHAL-labeled axons with the trained U-Net
   (`fiber_segmentation.py`).
3. Evaluate signal intensity within segmented objects relative to background
   (`segmentation_intensity.py`).
4. Quantify segmented objects within anatomical ROIs
   (`roi_quantification.py`).
5. Generate and compare spatial density maps of anatomical labeling
   (`kernel_density.py`).

Manual anatomical delineation and registration steps described in the
published Methods are not replaced by automated registration code in this
repository.

## Study-specific parameters

The following parameters are preserved from the analysis implementation used
for the study.

| Analysis | Parameter | Value |
|---|---|---|
| Image export / coordinate conversion | Image resolution | 300 dpi |
| Cellpose inference | Default tile size | 5000 x 5000 pixels |
| Cellpose inference | Tile overlap | 50 pixels |
| Cellpose SVG conversion | Coordinate scale | 72 / 300 |
| PHAL U-Net segmentation | Binary prediction threshold | 0.3 |
| Signal/background analysis | Histogram bin width | 4 grayscale units |
| Signal/background analysis | Histogram range | 0-256 |
| Signal/background analysis | Zero-valued background pixels | Excluded |
| ROI quantification | ROI membership | Segmentation centroid within ROI |
| ROI quantification | Area calibration | Olympus VS200, 10x, 300-dpi export |
| KDE | Evaluation-grid density | 2 points per SVG coordinate unit |
| KDE | Gaussian KDE bandwidth (`bw_method`) | 0.005 |
| KDE | Probability-mass contour | 0.95 where used |

Parameters supplied explicitly to functions can be changed for other
datasets. The defaults above are intended to preserve the analysis settings
used for this publication. See `ANALYSIS_PARAMETERS.md` for additional
implementation details and parameters requiring external documentation.

## Cellpose model training

The study used custom-trained Cellpose-SAM models. The repository provides
the code used to apply those trained models but does not reproduce the
general Cellpose training workflow. For instructions on training custom
models, consult the official Cellpose documentation.

Paper-specific training information should be reported here or in the
published Methods. Do not assume that Cellpose default training parameters
were used unless explicitly documented.

## U-Net model

PHAL-labeled axons were segmented using a custom-trained U-Net based on the
Pytorch-UNet implementation by Milesial:
https://github.com/milesial/Pytorch-UNet

`fiber_segmentation.py` uses the `UNet` architecture and
`BasicDataset.preprocess` implementation from that project. The trained model
checkpoint is loaded by the script, and predictions are thresholded at 0.3 to
generate binary axon masks. For exact reproduction, the Pytorch-UNet
commit/version used for model training should be used together with the trained
checkpoint.

## Input and output conventions

Microscopy images are expected to be grayscale image files. Segmentation and
ROI geometry are stored as SVG. The analysis scripts assume that corresponding
images and SVG files use the same spatial reference after the coordinate
conversion described above.

The publication code intentionally does not contain local filesystem paths,
usernames, laboratory directory structures, or unrelated development code.

## Data availability

The article's underlying behavioral and neuroanatomical data are deposited
separately as described in the published Data Availability Statement. This
repository contains the custom analysis code associated with those data.

## Citation

If using this code, please cite the associated *Journal of Comparative
Neurology* article. The manuscript is available at https://doi.org/10.1002/cne.70209
