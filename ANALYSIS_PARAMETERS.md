# Study-specific analysis parameters

This file records analysis settings represented in the publication code.
It is intended as a compact companion to the Methods and README.

## Image scaling

- Output grayscale range: 0-255
- Image export resolution used for coordinate conversion: 300 dpi
- Linear transformation: user-selected lower and upper image-intensity bounds
  are mapped to 0 and 255; values outside this interval are clipped.

The actual lower/upper scaling bounds were image/channel-specific inputs.
They should not be inferred from the publication code unless explicitly
reported for the relevant dataset.

## Cellpose-SAM segmentation

- Default tile size: 5000 x 5000 pixels
- Tile overlap: 50 pixels
- SVG coordinate conversion: 72 / 300
- GPU execution: requested when available; GPU timing/synchronization is not
  part of the publication implementation.
- Objects duplicated across overlapping tiles: when outlines share a
  coordinate, the outline with the larger number of vertices is retained.

The trained Cellpose-SAM model file is an external analysis input.

## PHAL axon segmentation

- Network: single-channel, single-class U-Net
- Output activation for binary prediction: sigmoid
- Prediction threshold: 0.3
- Device: CUDA when available, otherwise CPU

The U-Net architecture and preprocessing implementation are from the
Pytorch-UNet project by Milesial:
https://github.com/milesial/Pytorch-UNet

The trained U-Net checkpoint and the exact Pytorch-UNet commit/version used
for model training are external analysis inputs.

## Segmentation intensity analysis

- Source images: 8-bit grayscale
- Object statistic: mean pixel intensity within each segmented object
- Background: pixels outside segmentation outlines
- Zero-valued image pixels: excluded from background distribution
- Histogram edges: 0 to 256 in increments of 4 grayscale units
- Histograms: normalized with `numpy.histogram(..., density=True)`

## ROI quantification

- Object location: segmentation centroid
- ROI membership: centroid falls within anatomical ROI boundary
- ROI area: rasterized at one SVG coordinate-unit resolution
- Area conversion:
  `0.000000422100344536 * (300 / 72)^2 mm^2` per squared SVG coordinate unit
- Acquisition/export calibration represented by this conversion:
  Olympus VS200, 10x, 300-dpi export
- Reported density: object count / ROI area (objects/mm²)

## Kernel-density analysis

- KDE implementation: `scipy.stats.gaussian_kde`
- Paper-specific default `bw_method`: 0.005
- Evaluation grid: 2 points per SVG coordinate unit in x and y
- Individual KDE normalization: numerical integral normalized to 1 before
  group averaging
- Group map: arithmetic mean of individually normalized KDE maps
- Probability-mass threshold: 0.95 where used
- Spatial intersection: pointwise product of density maps
- ROI density: numerical integral of the density values falling within the
  ROI mask

## Parameters requiring external documentation

The following are not safely recoverable from the cleaned publication scripts
alone and should be documented separately if needed for exact reproduction:

- Trained Cellpose-SAM model weights
- Exact Cellpose-SAM training dataset and model-training provenance
- Trained U-Net checkpoint
- Pytorch-UNet commit/version used for model training
- U-Net training procedure
- Image/channel-specific linear-scaling bounds
- Manually defined anatomical ROI and registration files
