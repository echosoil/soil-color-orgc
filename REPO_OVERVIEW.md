# Repository Overview

Generated: `2026-06-23 02:21:07`

This file is generated automatically by:

```bash
python3 scripts/create_repo_overview.py
```

## Project tree

Generated files, debug outputs, sample images, lab files and virtual environments are omitted.

```text
soil-color-orgc/
├── data
│   └── munsell
│       └── rit_munsell.csv
├── scripts
│   ├── compare_orgC_lab_vs_CS.py
│   ├── create_repo_overview.py
│   ├── diagnose_color_profile.py
│   ├── enrich_lab_file.py
│   ├── make_color_swatches.py
│   ├── run_all.py
│   └── run_color_estimation.py
├── src
│   └── soil_color_orgc
│       ├── __pycache__
│       ├── __init__.py
│       ├── code_extraction.py
│       ├── code_extraction_light.py
│       ├── gray_calibration.py
│       ├── image_io.py
│       ├── image_processing.py
│       ├── lab_merge.py
│       ├── munsell.py
│       ├── pipeline.py
│       ├── soc_estimation.py
│       └── train_calibration_models.py
├── .gitignore
├── config.example.yml
├── LICENSE
├── README.md
└── requirements.txt
```

## Important commands

Run standard pipeline without grey-scale correction:

```bash
python3 scripts/run_all.py \
  --samples data/samples \
  --results outputs/results_no_gray.csv \
  --enriched outputs/test_stat_orgC_enriched_no_gray.xlsx \
  --no-gray-calibration
```

Run pipeline with grey-scale correction:

```bash
python3 scripts/run_all.py \
  --samples data/samples/with_gray \
  --results outputs/results_with_gray.csv \
  --enriched outputs/test_stat_orgC_enriched_with_gray.xlsx
```

Create colour comparison cards:

```bash
python3 scripts/make_color_swatches.py \
  --results outputs/results_with_gray.csv \
  --output-dir outputs/color_cards_with_gray
```

Train calibration models:

```bash
python3 scripts/train_calibration_models.py \
  --input outputs/test_stat_orgC_enriched_with_gray.xlsx \
  --target orgC_lab
```

## Python modules

### `scripts/compare_orgC_lab_vs_CS.py`

Imports:

```text
matplotlib.pyplot, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics
```

Functions:

- `to_numeric_clean(series)` — Convert values to numeric, accepting either decimal dots or decimal commas.
- `concordance_correlation_coefficient(x, y)` — Lin's concordance correlation coefficient.
- `main()`

### `scripts/create_repo_overview.py`

Create REPO_OVERVIEW.md from the current repository structure.

Imports:

```text
__future__, ast, datetime, os, pathlib
```

Functions:

- `rel(path)`
- `should_ignore(path)`
- `build_tree(base, max_depth)`
- `parse_python_file(path)`
- `first_line(text)`
- `collect_python_files()`
- `read_short_file(path, max_lines)`
- `write_overview()`
- `main()`

### `scripts/diagnose_color_profile.py`

Imports:

```text
PIL, argparse, cv2, io, numpy, os, pandas, pathlib
```

Functions:

- `list_images(path)`
- `get_icc_info(image_path)`
- `read_cv2_bgr(image_path)`
- `read_pillow_no_icc_bgr(image_path)`
- `read_pillow_with_icc_bgr(image_path)`
- `lab_summary_from_bgr(image_bgr)` — Quick OpenCV Lab summary.
- `mean_abs_pixel_diff(a, b)`
- `fit_to_panel(img_bgr, width, height)` — Fit image into a fixed-size white panel while preserving aspect ratio.
- `save_comparison_image(image_path, cv2_bgr, pil_no_icc_bgr, pil_icc_bgr, output_dir)`
- `main()`

### `scripts/enrich_lab_file.py`

Imports:

```text
argparse, pathlib, soil_color_orgc.lab_merge, sys
```

Functions:

- `main()`

### `scripts/make_color_swatches.py`

Imports:

```text
argparse, colour, cv2, numpy, os, pandas, re
```

Functions:

- `lab_to_rgb_u8(L, a, b)` — Convert Lab to uint8 RGB for visualization.
- `make_swatch(rgb_u8, size)` — Create a square swatch image from RGB uint8.
- `format_munsell_notation(h, v, c)` — Create notation like:
- `load_munsell_rgb_lookup(munsell_csv)` — Build mapping:
- `fit_image_to_panel(img_bgr, width, height)` — Fit an image into a white panel while preserving aspect ratio.
- `add_label(img_bgr, text, y, scale)`
- `make_labeled_panel(content_bgr, title, width, height)`
- `load_roi_image(debug_dir, image_name)`
- `build_comparison_card(row, munsell_rgb_lookup, debug_dir, out_dir)`
- `main()`

### `scripts/run_all.py`

Imports:

```text
argparse, pathlib, soil_color_orgc.lab_merge, soil_color_orgc.pipeline, sys
```

Functions:

- `main()`

### `scripts/run_color_estimation.py`

Imports:

```text
argparse, pathlib, soil_color_orgc.pipeline, sys
```

Functions:

- `main()`

### `src/soil_color_orgc/__init__.py`

### `src/soil_color_orgc/code_extraction.py`

Imports:

```text
cv2, image_io, os, re
```

Functions:

- `find_code_in_text(text)` — Find sample code like BMXU-2646 in arbitrary OCR/QR text.
- `base_code_from_sample_code(code)` — BMXU-2646 -> BMXU
- `extract_code_from_filename(image_path)` — Extract code from image filename.
- `crop_code_region(image_bgr)` — Crop likely code region.
- `extract_code_from_qr(image_bgr)` — Try to decode QR code using OpenCV.
- `preprocess_for_ocr(image_bgr)` — Prepare image or crop for OCR.
- `extract_code_from_ocr(image_bgr)` — Use Tesseract OCR to detect printed text.
- `extract_sample_code_with_source(image_path)` — Extract sample code and record how it was found.

### `src/soil_color_orgc/code_extraction_light.py`

Imports:

```text
os, re
```

Functions:

- `extract_code_from_filename(image_path)` — Extract four-letter sample code from filename.
- `extract_sample_code_from_filename_only(image_path)` — Filename-only code extraction.

### `src/soil_color_orgc/gray_calibration.py`

Imports:

```text
cv2, numpy, os, pandas
```

Functions:

- `target_grays_from_black_scale(n_patches, left_label, right_label)`
- `crop_gray_search_region(image_bgr, x_min, x_max, y_min, y_max)` — Broad search region where the grey scale is expected.
- `score_gray_candidate(candidate_bgr)` — Score whether a candidate crop looks like a grey-scale strip.
- `detect_gray_scale_rectangle(image_bgr)` — Detect a long horizontal grey-scale rectangle in the lower part of the image.
- `crop_fixed_gray_scale_roi(image_bgr, x_min, x_max, y_min, y_max)`
- `score_gray_ramp_candidate(candidate_bgr, n_patches)` — Score whether a candidate crop looks like the 10-to-0 grey scale.
- `detect_gray_scale_by_gray_ramp(image_bgr, x_min, x_max, y_min, y_max, n_patches)` — Detect grey scale by finding an 11-patch left-to-right brightness ramp.
- `trim_gray_roi_horizontally_by_ramp(gray_roi_bgr, rect, n_patches, min_width_fraction, max_width_fraction, pad_fraction)` — Trim a grey-scale ROI horizontally by finding the x-range that best behaves
- `trim_gray_roi_vertically(gray_roi_bgr, rect, white_v_threshold, max_saturation, min_nonwhite_fraction, min_horizontal_range, pad_fraction)` — Trim grey-scale ROI vertically using a stricter white threshold.
- `crop_gray_scale_roi(image_bgr)` — Main grey-scale ROI function.
- `split_gray_patches(gray_roi_bgr, n_patches)` — Split grey scale into patches from left to right.
- `measure_patch_rgb(patch_bgr)` — Robust median RGB measurement.
- `neutral_target_from_rgb(rgb)` — Target neutral grey.
- `build_channel_lut(measured_values, target_values)` — Build a 0..255 lookup table mapping measured channel values to target grey values.
- `apply_luts_to_bgr(image_bgr, lut_r, lut_g, lut_b)` — Apply RGB LUTs to OpenCV BGR image.
- `correct_image_using_gray_scale(image_bgr, n_patches, debug_dir, image_name)` — Correct colour balance using the grey scale at the bottom of the image.
- `save_gray_calibration_debug(original_bgr, corrected_bgr, gray_roi_bgr, rect, report_df, debug_dir, image_name)`
- `resize_preview(img_bgr, width)`

### `src/soil_color_orgc/image_io.py`

Imports:

```text
PIL, cv2, io, numpy
```

Functions:

- `read_image_bgr(image_path, use_icc)` — Read image consistently.

### `src/soil_color_orgc/image_processing.py`

Imports:

```text
colour, cv2, gray_calibration, image_io, numpy, os
```

Functions:

- `ensure_portrait_orientation(image_bgr)` — Ensure image is portrait.
- `downscale_max_side(image_bgr, max_side)` — Downscale large images for faster processing.
- `crop_central_soil_roi(image_bgr, x_min, x_max, y_min, y_max)` — Fixed central crop.
- `build_simple_roi_mask(roi_bgr, min_v, max_v, white_v, white_s, white_min_rgb, white_spread)` — Simple mask inside the fixed ROI.
- `bgr_pixels_to_lab(pixels_bgr)` — Convert BGR uint8 pixels to CIE Lab using colour-science.
- `trim_lab_by_lightness(lab_pixels, trim_percent)` — Remove darkest and brightest pixels by L.
- `save_debug_images(image_bgr, roi_bgr, mask_u8, rect, image_path, debug_dir)` — Save visual diagnostics:
- `trim_soil_roi_to_nonwhite_content(roi_bgr, rect, min_v, white_v, white_s, white_min_rgb, white_spread, min_row_fraction, min_col_fraction, pad_fraction)` — Trim a soil ROI to remove white/pale paper strips.
- `extract_dominant_lab(image_path, save_debug_masks, debug_dir, downscale_max, kmeans_clusters, trim_percent, use_gray_calibration)` — Extract representative soil color from a fixed central ROI.

### `src/soil_color_orgc/lab_merge.py`

Imports:

```text
os, pandas, re
```

Functions:

- `base_code_from_lab(id_str)` — Extract the leading code before '-'.
- `base_code_from_image(name)` — Strip extension and take the leading alphanumeric block.
- `_available_columns(df, wanted_columns)`
- `enrich_lab_file(lab_xlsx, predictions_csv, output_xlsx)`

### `src/soil_color_orgc/munsell.py`

Imports:

```text
colour, colour.difference, numpy, pandas
```

Functions:

- `load_munsell(csv_path)` — Load RIT Munsell CSV.
- `find_best_munsell(lab_color, munsell_dict, threshold)` — Find the closest Munsell chip using CIEDE2000.

### `src/soil_color_orgc/pipeline.py`

Imports:

```text
code_extraction_light, glob, image_processing, munsell, os, pandas, soc_estimation
```

Functions:

- `find_image_paths(samples_dir)`
- `safe_extract_sample_code(image_path)` — Fast filename-only sample code extraction.
- `make_error_row(image_name, error)`
- `run_image_pipeline(samples_dir, munsell_csv, output_csv, debug_dir, save_debug_masks, deltae_threshold, downscale_max, kmeans_clusters, trim_percent, use_gray_calibration)`

### `src/soil_color_orgc/soc_estimation.py`

Functions:

- `soc_from_munsell(notation)` — Estimate SOC (%) from Munsell Value and Chroma.
- `soc_from_L(lab)` — Fallback SOC (%) from CIE Lab lightness only.
- `estimate_soc(lab, best_munsell)` — Prefer Munsell-based SOC if match is valid.

### `src/soil_color_orgc/train_calibration_models.py`

Example usage:

Imports:

```text
argparse, numpy, os, pandas, pathlib, re, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys
```

Functions:

- `read_input_table(path)`
- `parse_munsell_value_chroma(best_munsell)` — Examples:
- `prepare_data(df, target_col)`
- `make_stratify_bins(y, max_bins)` — Stratify continuous target approximately by quantile bins.
- `evaluate_predictions(y_true, y_pred)`
- `fit_one_model(model_name, feature_cols, train_df, test_df, target_col)`
- `main()`

## Selected file previews

### `README.md`

```
# Soil Color orgC Estimation

This repository contains a prototype workflow for estimating soil organic carbon (`orgC`) from soil sample photographs using image-derived colour features, Munsell colour matching, and optional grey-scale colour correction.

The pipeline is currently exploratory. It is intended for testing whether image-based soil colour information can provide a useful proxy for laboratory-measured organic carbon. The raw image-derived estimate should not be treated as a validated laboratory substitute without calibration.

## Current workflow

The pipeline performs the following steps:

1. Reads soil sample images.
2. Extracts a sample code from the filename, for example `ABCD.jfif` -> `ABCD`.
3. Optionally corrects image colour using a grey-scale reference card.
4. Extracts a representative soil colour from a central region of interest.
5. Converts the colour to CIE Lab.
6. Finds the closest Munsell colour using DeltaE2000.
7. Produces a heuristic `SOC_est%` value.
8. Merges image-derived results with laboratory `orgC` data.
9. Produces matching reports, calibration statistics, and visual comparison cards.

## Repository structure

```text
soil-color-orgc/
├── data/
│   ├── lab/
│   │   └── test_stat_orgC.xlsx
│   ├── munsell/
│   │   └── rit_munsell.csv
│   └── samples/
│       ├── ABCD.jfif
│       └── with_gray/
│           └── ABCD.jfif
├── debug_gray/
├── debug_masks/
├── outputs/
├── scripts/
│   ├── run_all.py
│   ├── make_color_swatches.py
│   ├── train_calibration_models.py
│   ├── diagnose_color_profile.py
│   └── create_repo_overview.py
└── src/
    └── soil_color_orgc/
        ├── image_io.py
        ├── image_processing.py
        ├── gray_calibration.py
        ├── munsell.py
        ├── soc_estimation.py
        ├── code_extraction_light.py
        ├── lab_merge.py
        └── pipeline.py
```

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Typical requirements include:

```text
opencv-python
pandas
numpy
scikit-learn
colour-science
openpyxl
Pillow
... truncated after 80 lines ...
```

### `requirements.txt`

```
opencv-python
pandas
numpy
scikit-learn
colour-science
openpyxl
matplotlib 
scipy
Pillow
```

### `scripts/run_all.py`

```
#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soil_color_orgc.pipeline import run_image_pipeline
from soil_color_orgc.lab_merge import enrich_lab_file


def main():
    parser = argparse.ArgumentParser(
        description="Run image SOC estimation and enrich lab Excel file."
    )

    parser.add_argument(
        "--samples",
        default="data/samples",
        help="Folder containing sample images.",
    )

    parser.add_argument(
        "--munsell",
        default="data/munsell/rit_munsell.csv",
        help="Path to RIT Munsell CSV.",
    )

    parser.add_argument(
        "--lab",
        default="data/lab/test_stat_orgC.xlsx",
        help="Input laboratory Excel file.",
    )

    parser.add_argument(
        "--results",
        default="outputs/results.csv",
        help="Output predictions CSV.",
    )

    parser.add_argument(
        "--enriched",
        default="outputs/test_stat_orgC_enriched.xlsx",
        help="Output enriched Excel file.",
    )

    parser.add_argument(
        "--debug-dir",
        default="debug_masks",
        help="Folder for debug mask images.",
    )

    parser.add_argument(
        "--no-debug-masks",
        action="store_true",
        help="Disable debug mask generation.",
    )

    parser.add_argument(
        "--deltae-threshold",
        type=float,
        default=8.0,
        help="Maximum accepted DeltaE2000 for Munsell match.",
    )

    parser.add_argument(
        "--no-gray-calibration",
        action="store_true",
        help="Disable grey-scale colour calibration.",
    )

    args = parser.parse_args()

    run_image_pipeline(
        samples_dir=args.samples,
        munsell_csv=args.munsell,
        output_csv=args.results,
        debug_dir=args.debug_dir,
... truncated after 80 lines ...
```

### `scripts/make_color_swatches.py`

```
#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
import pandas as pd
import cv2

from colour import Lab_to_XYZ, XYZ_to_sRGB


def lab_to_rgb_u8(L, a, b):
    """
    Convert Lab to uint8 RGB for visualization.
    """
    lab = np.array([L, a, b], dtype=float)
    xyz = Lab_to_XYZ(lab)
    rgb = XYZ_to_sRGB(xyz)

    rgb = np.clip(rgb, 0, 1)
    rgb_u8 = (rgb * 255).round().astype(np.uint8)

    return rgb_u8


def make_swatch(rgb_u8, size=220):
    """
    Create a square swatch image from RGB uint8.
    Returned image is BGR for OpenCV writing.
    """
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)
    bgr = rgb_u8[::-1]
    swatch = np.full((size, size, 3), bgr, dtype=np.uint8)
    return swatch


def format_munsell_notation(h, v, c):
    """
    Create notation like:
        10YR 3/4

    Avoids accidental 3.0/4.0 formatting.
    """
    def clean_number(x):
        x = float(x)
        if x.is_integer():
            return str(int(x))
        return str(x).rstrip("0").rstrip(".")

    return f"{str(h).strip()} {clean_number(v)}/{clean_number(c)}"


def load_munsell_rgb_lookup(munsell_csv):
    """
    Build mapping:
        notation -> (R, G, B)

    RIT Munsell CSV may contain:
        R,G,B      as normalized 0..1 floats
        dR,dG,dB   as 8-bit display RGB values

    For visualization, prefer dR,dG,dB when available.
    """
    df = pd.read_csv(munsell_csv)

    lookup = {}

    for _, row in df.iterrows():
        notation = format_munsell_notation(row["h"], row["V"], row["C"])

        if {"dR", "dG", "dB"}.issubset(df.columns):
            rgb = (
                int(round(row["dR"])),
                int(round(row["dG"])),
                int(round(row["dB"])),
            )
        else:
            r = float(row["R"])
            g = float(row["G"])
... truncated after 80 lines ...
```

### `scripts/compare_orgC_lab_vs_CS.py`

```
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


INPUT_XLSX = "outputs/test_stat_orgC_enriched_with_gray.xlsx"
OUTPUT_DIR = "outputs/orgC_comparison_outputs"

LAB_COL = "orgC_lab"
# CS_COL = "orgC_CS"
CS_COL = "SOC_est%"
ID_COL = "ID"


def to_numeric_clean(series):
    """
    Convert values to numeric, accepting either decimal dots or decimal commas.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def concordance_correlation_coefficient(x, y):
    """
    Lin's concordance correlation coefficient.
    Measures agreement, not only correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2)
    return ccc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(INPUT_XLSX)

    required = {ID_COL, LAB_COL, CS_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[LAB_COL] = to_numeric_clean(df[LAB_COL])
    df[CS_COL] = to_numeric_clean(df[CS_COL])

    clean = df.dropna(subset=[LAB_COL, CS_COL]).copy()

    if clean.empty:
        raise ValueError("No valid rows after removing missing lab/CS values.")

    y_true = clean[LAB_COL].values
    y_pred = clean[CS_COL].values

    error = y_pred - y_true
    abs_error = np.abs(error)

    clean["error_CS_minus_lab"] = error
    clean["abs_error"] = abs_error
    clean["mean_lab_CS"] = (y_true + y_pred) / 2

... truncated after 80 lines ...
```

### `scripts/diagnose_color_profile.py`

```
#!/usr/bin/env python3

import argparse
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageCms


IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}


def list_images(path):
    path = Path(path)

    if path.is_file():
        return [path]

    images = []

    for p in path.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)

    return sorted(images)


def get_icc_info(image_path):
    img = Image.open(image_path)
    icc = img.info.get("icc_profile")

    if not icc:
        return {
            "has_icc": False,
            "icc_name": None,
            "icc_description": None,
            "icc_bytes": 0,
        }

    name = None
    description = None

    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        try:
            name = ImageCms.getProfileName(profile)
        except Exception:
            name = None

        try:
            description = ImageCms.getProfileDescription(profile)
        except Exception:
            description = None

    except Exception as exc:
        description = f"Could not read ICC profile: {exc}"

    return {
        "has_icc": True,
        "icc_name": name,
        "icc_description": description,
        "icc_bytes": len(icc),
    }


def read_cv2_bgr(image_path):
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def read_pillow_no_icc_bgr(image_path):
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_pillow_with_icc_bgr(image_path):
... truncated after 80 lines ...
```

### `src/soil_color_orgc/pipeline.py`

```
import os
import glob
import pandas as pd

from .munsell import load_munsell, find_best_munsell
from .image_processing import extract_dominant_lab
from .soc_estimation import estimate_soc
from .code_extraction_light import extract_sample_code_from_filename_only


def find_image_paths(samples_dir: str):
    patterns = [
        "*.jpg", "*.jpeg", "*.jfif", "*.png", "*.webp",
        "*.JPG", "*.JPEG", "*.JFIF", "*.PNG", "*.WEBP",
    ]

    image_paths = []

    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(samples_dir, pattern)))

    return sorted(set(image_paths))


def safe_extract_sample_code(image_path: str):
    """
    Fast filename-only sample code extraction.
    No OCR, no QR recognition.
    """
    try:
        return extract_sample_code_from_filename_only(image_path)

    except Exception as exc:
        return {
            "sample_code_full": None,
            "sample_code_base": None,
            "sample_code_source": "code_error",
            "sample_code_match_score": None,
            "sample_code_matched_fragment": None,
            "sample_code_error": str(exc),
            "filename_code_base": None,
            "code_conflict": False,
        }


def make_error_row(image_name: str, error: Exception):
    return {
        "image": image_name,
        "sample_code_full": None,
        "sample_code_base": None,
        "sample_code_source": None,
        "sample_code_match_score": None,
        "sample_code_matched_fragment": None,
        "sample_code_error": None,
        "filename_code_base": None,
        "code_conflict": None,
        "L": None,
        "a": None,
        "b": None,
        "best_munsell": None,
        "deltaE2000": None,
        "SOC_est%": None,
        "SOC_method": None,
        "processing_status": "error",
        "processing_error": str(error),
    }


def run_image_pipeline(
    samples_dir: str,
    munsell_csv: str,
    output_csv: str,
    debug_dir: str = "debug_masks",
    save_debug_masks: bool = True,
    deltae_threshold: float = 8.0,
    downscale_max: int = 800,
    kmeans_clusters: int = 4,
    trim_percent: float = 0.05,
    use_gray_calibration: bool = True,
):
... truncated after 80 lines ...
```

### `src/soil_color_orgc/image_processing.py`

```
import os
import cv2
import numpy as np
from colour import sRGB_to_XYZ, XYZ_to_Lab
from .gray_calibration import correct_image_using_gray_scale
from .image_io import read_image_bgr

def ensure_portrait_orientation(image_bgr):
    """
    Ensure image is portrait.

    In our dataset, valid sample images should have:
        height > width

    If width > height, rotate 90 degrees.

    Note:
        This does not 'swap x and y' mathematically.
        It rotates the image so that x remains width and y remains height.
    """
    height, width = image_bgr.shape[:2]

    if width > height:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)

    return image_bgr


def downscale_max_side(image_bgr, max_side=1200):
    """
    Downscale large images for faster processing.
    Keeps aspect ratio.
    """
    h, w = image_bgr.shape[:2]
    current_max = max(h, w)

    if current_max <= max_side:
        return image_bgr

    scale = max_side / current_max
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_central_soil_roi(
    image_bgr,
    x_min=0.40,
    x_max=0.60,
    y_min=0.50,
    y_max=0.70,
):
    """
    Fixed central crop.

    Default:
        x: 40% to 60%
        y: 50% to 70%

    This intentionally ignores most borders, labels and background.
    """
    h, w = image_bgr.shape[:2]

    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)

    roi = image_bgr[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2)


def build_simple_roi_mask(
    roi_bgr,
    min_v=20,
    max_v=245,
    white_v=190,
    white_s=70,
... truncated after 80 lines ...
```

### `src/soil_color_orgc/gray_calibration.py`

```
import os
import cv2
import numpy as np
import pandas as pd


def target_grays_from_black_scale(n_patches=11, left_label=10, right_label=0):
    labels = np.linspace(left_label, right_label, n_patches)
    black_fraction = labels / 10.0

    target_gray = 255.0 * (1.0 - black_fraction)

    # Avoid forcing extreme black/white too strongly.
    target_gray = np.clip(target_gray, 20, 235)

    return target_gray.astype(np.float32)

def crop_gray_search_region(
    image_bgr,
    x_min=0.08,
    x_max=0.92,
    y_min=0.65,
    y_max=0.92,
):
    """
    Broad search region where the grey scale is expected.

    y is measured from the top of the image.
    """
    h, w = image_bgr.shape[:2]

    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)

    search = image_bgr[y1:y2, x1:x2].copy()

    return search, (x1, y1, x2, y2)


def score_gray_candidate(candidate_bgr):
    """
    Score whether a candidate crop looks like a grey-scale strip.

    Good candidate:
        - low saturation overall
        - wide brightness range, because it goes dark -> light
        - approximately neutral
    """
    if candidate_bgr is None or candidate_bgr.size == 0:
        return -1

    h, w = candidate_bgr.shape[:2]

    if h < 10 or w < 50:
        return -1

    hsv = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    rgb = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    median_saturation = float(np.median(S))
    brightness_range = float(np.percentile(V, 95) - np.percentile(V, 5))

    # Neutrality: greys should have R, G, B close together.
    channel_spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    median_channel_spread = float(np.median(channel_spread))

    aspect = w / max(h, 1)

    # Prefer long horizontal strips.
    aspect_score = min(aspect / 6.0, 1.5)

    # Prefer strong dark-to-light range.
    brightness_score = brightness_range / 255.0

    # Penalize colourful candidates.
    saturation_penalty = median_saturation / 255.0
... truncated after 80 lines ...
```

### `src/soil_color_orgc/image_io.py`

```
import io
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps


def read_image_bgr(image_path: str, use_icc: bool = True):
    """
    Read image consistently.

    Steps:
        1. Open with Pillow.
        2. Apply EXIF orientation.
        3. If ICC profile exists and use_icc=True, convert to sRGB.
        4. Return OpenCV-style BGR uint8 image.
    """
    img = Image.open(image_path)

    # Very important: respect camera/phone orientation metadata.
    img = ImageOps.exif_transpose(img)

    icc_profile = img.info.get("icc_profile")

    if use_icc and icc_profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            target_profile = ImageCms.createProfile("sRGB")

            img = ImageCms.profileToProfile(
                img,
                source_profile,
                target_profile,
                outputMode="RGB",
            )
        except Exception:
            img = img.convert("RGB")
    else:
        img = img.convert("RGB")

    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return bgr
```

### `src/soil_color_orgc/munsell.py`

```
import pandas as pd
import numpy as np
from colour import XYZ_to_Lab
from colour.difference import delta_E


def load_munsell(csv_path: str):
    """
    Load RIT Munsell CSV.

    Expected columns:
    file order,h,V,C,x,y,Y,X_C,Y_C,Z_C,X_D65,Y_D65,Z_D65,R,G,B,dR,dG,dB

    Uses X_D65, Y_D65, Z_D65 directly and converts them to CIE Lab.
    """
    df = pd.read_csv(csv_path)

    required = {"h", "V", "C", "X_D65", "Y_D65", "Z_D65"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Munsell CSV is missing required columns: {sorted(missing)}")

    df["notation"] = (
        df["h"].astype(str)
        + " "
        + df["V"].astype(str)
        + "/"
        + df["C"].astype(str)
    )

    def xyz_to_lab(row):
        return tuple(XYZ_to_Lab([row["X_D65"], row["Y_D65"], row["Z_D65"]]))

    df["lab"] = df.apply(xyz_to_lab, axis=1)

    return {
        row["notation"]: row["lab"]
        for _, row in df.iterrows()
    }


def find_best_munsell(lab_color, munsell_dict, threshold: float = 8.0):
    """
    Find the closest Munsell chip using CIEDE2000.

    Returns:
        (best_notation, deltaE)

    If no match is closer than threshold, best_notation is:
        "No close Munsell match"
    """
    best_match = None
    best_delta = float("inf")

    sample = np.array(lab_color).reshape(1, 3)

    for notation, lab_ref in munsell_dict.items():
        ref = np.array(lab_ref).reshape(1, 3)
        dE = delta_E(sample, ref, method="CIE 2000")[0]

        if dE < best_delta:
            best_delta = float(dE)
            best_match = notation

    if best_delta > threshold:
        return "No close Munsell match", best_delta

    return best_match, best_delta
```

### `src/soil_color_orgc/soc_estimation.py`

```
def soc_from_munsell(notation: str):
    """
    Estimate SOC (%) from Munsell Value and Chroma.

    Example notation:
        "10YR 3/4"

    Heuristic:
        SOC% ≈ 12 - 1.5*Value - 0.3*Chroma

    This should later be calibrated against lab measurements.
    """
    try:
        vc = notation.split()[1]
        value_str, chroma_str = vc.split("/")

        value = int(value_str)
        chroma = int(chroma_str)

        soc = 12 - 1.5 * value - 0.3 * chroma
        return max(0.0, float(soc))

    except Exception:
        return None


def soc_from_L(lab):
    """
    Fallback SOC (%) from CIE Lab lightness only.

    Heuristic:
        SOC% ≈ 10 - 0.15*L
    """
    L = float(lab[0])
    soc = 10 - 0.15 * L

    return max(0.0, float(soc))


def estimate_soc(lab, best_munsell: str):
    """
    Prefer Munsell-based SOC if match is valid.
    Otherwise use L-based fallback.

    Returns:
        (soc_estimate, method)
    """
    if best_munsell != "No close Munsell match":
        soc = soc_from_munsell(best_munsell)

        if soc is not None:
            return soc, "munsell"

    return soc_from_L(lab), "L_fallback"
```

### `src/soil_color_orgc/lab_merge.py`

```
import re
import os
import pandas as pd


def base_code_from_lab(id_str: str):
    """
    Extract the leading code before '-'.

    Examples:
        ABCD-1234 -> ABCD
        XYZ99-7A  -> XYZ99
    """
    if pd.isna(id_str):
        return None

    m = re.match(r"\s*([A-Za-z0-9]+)", str(id_str))
    return m.group(1).upper() if m else None


def base_code_from_image(name: str):
    """
    Strip extension and take the leading alphanumeric block.

    Examples:
        ABCD.jfif       -> ABCD
        ABCD.JPG        -> ABCD
        XYZ99-photo.jpg -> XYZ99
    """
    if pd.isna(name):
        return None

    stem = re.sub(r"\.\w+$", "", str(name))
    m = re.match(r"\s*([A-Za-z0-9]+)", stem)

    return m.group(1).upper() if m else None


def _available_columns(df, wanted_columns):
    return [col for col in wanted_columns if col in df.columns]


def enrich_lab_file(
    lab_xlsx: str,
    predictions_csv: str,
    output_xlsx: str,
):
    lab = pd.read_excel(lab_xlsx)
    predictions = pd.read_csv(predictions_csv)

    if "ID" not in lab.columns:
        raise ValueError("Lab file must contain an 'ID' column.")

    if "image" not in predictions.columns:
        raise ValueError("Predictions CSV must contain an 'image' column.")

    # Build matching keys
    lab["SampleCode"] = lab["ID"].apply(base_code_from_lab)

    if "sample_code_base" in predictions.columns:
        predictions["SampleCode"] = predictions["sample_code_base"]

        missing_prediction_code = predictions["SampleCode"].isna()

        predictions.loc[missing_prediction_code, "SampleCode"] = (
            predictions.loc[missing_prediction_code, "image"].apply(base_code_from_image)
        )
    else:
        predictions["SampleCode"] = predictions["image"].apply(base_code_from_image)

    # Normalize keys
    lab["SampleCode"] = lab["SampleCode"].astype("string").str.upper()
    predictions["SampleCode"] = predictions["SampleCode"].astype("string").str.upper()

    # Columns to bring into enriched lab file
    desired_prediction_columns = [
        "SampleCode",
        "image",
        "sample_code_full",
        "sample_code_base",
... truncated after 80 lines ...
```

### `src/soil_color_orgc/code_extraction_light.py`

```
import os
import re


FILENAME_CODE_PATTERN = re.compile(
    r"^([A-Za-z]{4})(?:[-_].*)?$"
)


def extract_code_from_filename(image_path: str):
    """
    Extract four-letter sample code from filename.

    Examples:
        ABCD.jfif       -> ABCD
        ABCD.jpg        -> ABCD
        ABCD-1234.jfif  -> ABCD
        ABCD_test.jfif  -> ABCD

    Returns:
        ABCD or None
    """
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]

    match = FILENAME_CODE_PATTERN.match(stem)

    if match:
        return match.group(1).upper()

    return None


def extract_sample_code_from_filename_only(image_path: str):
    """
    Filename-only code extraction.

    Returns the same structure as the OCR-based extractor,
    so the rest of the pipeline does not need to change.
    """
    base_code = extract_code_from_filename(image_path)

    if base_code:
        return {
            "sample_code_full": None,
            "sample_code_base": base_code,
            "sample_code_source": "filename",
            "sample_code_match_score": None,
            "sample_code_matched_fragment": base_code,
            "sample_code_error": None,
            "filename_code_base": base_code,
            "code_conflict": False,
        }

    return {
        "sample_code_full": None,
        "sample_code_base": None,
        "sample_code_source": "filename_not_found",
        "sample_code_match_score": None,
        "sample_code_matched_fragment": None,
        "sample_code_error": "Could not extract four-letter code from filename",
        "filename_code_base": None,
        "code_conflict": False,
    }
```

## Ignored/generated paths

The overview intentionally excludes or truncates generated and heavy data paths such as:

- `.git`
- `.idea`
- `.mypy_cache`
- `.pytest_cache`
- `.venv`
- `.vscode`
- `__pycache__`
- `data/lab`
- `data/samples`
- `debug_gray`
- `debug_masks`
- `env`
- `ocr_debug`
- `outputs`
- `venv`
