# Soil Color Organic Carbon Estimation

This repository estimates soil organic carbon from sample images by:

1. extracting the dominant soil color from JFIF/JPG/PNG images,
2. converting the color to CIE Lab,
3. matching it to the closest Munsell color chip,
4. estimating organic carbon from Munsell Value and Chroma,
5. optionally enriching laboratory organic carbon Excel files with image-based estimates.

This is a prototype / calibration tool. Final organic carbon estimates should be calibrated against laboratory measurements.

## Repository structure

```text
soil-color-orgc/
├── data/
│   ├── munsell/rit_munsell.csv
│   ├── samples/
│   └── lab/
├── outputs/
├── debug_masks/
├── scripts/
└── src/soil_color_orgc/
