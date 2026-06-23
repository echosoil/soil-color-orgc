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
matplotlib
scipy
```

## Input data

### Images

Sample images are expected in:

```text
data/samples/
```

For filename-only matching, images should be named with the four-letter sample code:

```text
ABCD.jfif
BMXU.jfif
KRLF.jpg
```

The filename code is matched to laboratory IDs such as:

```text
ABCD-1234
BMXU-2646
KRLF-5417
```

The matching key is the four-letter prefix.

### Images with grey scale

Images containing the grey-scale reference card can be placed in:

```text
data/samples/with_gray/
```

The grey-scale card is expected to be near the lower part of the image. The current card is interpreted as an 11-patch blackness scale:

```text
10 9 8 7 6 5 4 3 2 1 0
```

where:

```text
10 = 100% black
0  = 0% black / white
```

The grey-scale correction is used to improve colour balance before extracting the soil colour.

### Laboratory file

The laboratory Excel file is expected at:

```text
data/lab/test_stat_orgC.xlsx
```

It should contain at least:

```text
ID
orgC_lab
```

Example:

```text
ID          orgC_lab
BMXU-2646   2.31
KRLF-5417   1.26
```

## Running the pipeline

### Mode 1: standard mode, without grey-scale correction

Use this mode for images that do not contain a grey-scale reference card, or when you want to compare uncorrected results.

```bash
python3 scripts/run_all.py \
  --samples data/samples \
  --results outputs/results_no_gray.csv \
  --enriched outputs/test_stat_orgC_enriched_no_gray.xlsx \
  --no-gray-calibration
```

This produces:

```text
outputs/results_no_gray.csv
outputs/test_stat_orgC_enriched_no_gray.xlsx
```

### Mode 2: grey-scale correction mode

Use this mode for images that include the grey-scale reference card.

```bash
python3 scripts/run_all.py \
  --samples data/samples/with_gray \
  --results outputs/results_with_gray.csv \
  --enriched outputs/test_stat_orgC_enriched_with_gray.xlsx
```

This produces:

```text
outputs/results_with_gray.csv
outputs/test_stat_orgC_enriched_with_gray.xlsx
debug_gray/
debug_masks/
```

The grey-scale correction is attempted image by image. If grey-scale detection or correction fails for one image, the pipeline should continue using the uncorrected image and print a warning.

### Mode 3: compare grey and non-grey processing

Run both modes:

```bash
python3 scripts/run_all.py \
  --samples data/samples/with_gray \
  --results outputs/results_with_gray.csv \
  --enriched outputs/test_stat_orgC_enriched_with_gray.xlsx
```

```bash
python3 scripts/run_all.py \
  --samples data/samples/with_gray \
  --results outputs/results_with_gray_nocalib.csv \
  --enriched outputs/test_stat_orgC_enriched_with_gray_nocalib.xlsx \
  --no-gray-calibration
```

Then compare:

```text
outputs/results_with_gray.csv
outputs/results_with_gray_nocalib.csv
```

Useful columns to compare include:

```text
L
a
b
best_munsell
deltaE2000
SOC_est%
```

## Output files

### `outputs/results*.csv`

The main image-processing result table.

Important columns:

```text
image
sample_code_base
sample_code_source
L
a
b
best_munsell
deltaE2000
SOC_est%
SOC_method
processing_status
processing_error
```

Interpretation:

* `L`, `a`, `b`: representative soil colour in CIE Lab.
* `best_munsell`: closest Munsell colour.
* `deltaE2000`: colour distance to the closest Munsell chip.
* `SOC_est%`: heuristic colour-derived estimate.
* `processing_status`: `ok` or `error`.

### `outputs/test_stat_orgC_enriched*.xlsx`

Laboratory table enriched with image-derived colour estimates.

It contains several sheets:

```text
enriched
summary
lab_without_image
image_without_lab
duplicate_lab_codes
duplicate_image_codes
```

Use these sheets to detect matching problems:

* `lab_without_image`: lab entries without a corresponding image.
* `image_without_lab`: images without a corresponding lab entry.
* `duplicate_lab_codes`: repeated sample codes in the lab file.
* `duplicate_image_codes`: repeated image sample codes.

## Debug outputs

### `debug_masks/`

Contains soil-colour ROI diagnostics:

```text
*_roi_rect.jpg
*_roi.jpg
*_roi_mask.png
*_roi_used.jpg
```

Important files:

* `*_roi_rect.jpg`: original image with the selected soil ROI marked.
* `*_roi.jpg`: cropped soil ROI used for colour extraction.
* `*_roi_mask.png`: binary mask used inside the ROI.
* `*_roi_used.jpg`: rejected pixels shown in red.

If a sample gets an unrealistic colour estimate, inspect these files first.

### `debug_gray/`

Contains grey-scale correction diagnostics:

```text
*_gray_roi_rect.jpg
*_gray_roi.jpg
*_gray_before_after.jpg
*_gray_report.csv
```

Important files:

* `*_gray_roi_rect.jpg`: detected grey-scale region marked on the image.
* `*_gray_roi.jpg`: cropped grey-scale reference.
* `*_gray_before_after.jpg`: visual comparison before and after correction.
* `*_gray_report.csv`: measured patch values before and after correction.

Useful columns in `*_gray_report.csv`:

```text
patch_index
scale_label
measured_R
measured_G
measured_B
target_gray
after_R
after_G
after_B
imbalance_before
imbalance_after
```

The grey-scale correction is working better if `imbalance_after` is generally smaller than `imbalance_before`.

## Making colour swatches / comparison cards

After running the pipeline, create visual colour cards:

```bash
python3 scripts/make_color_swatches.py \
  --results outputs/results_with_gray.csv \
  --output-dir outputs/color_cards_with_gray
```

For non-grey runs:

```bash
python3 scripts/make_color_swatches.py \
  --results outputs/results_no_gray.csv \
  --output-dir outputs/color_cards_no_gray
```

Each card contains:

```text
ROI used
Estimated colour
Closest Munsell colour
Lab values
Munsell match
DeltaE2000
SOC_est%
```

These cards are for visual quality control.

Use them to check:

* whether the selected ROI is actually soil,
* whether the estimated colour visually resembles the ROI,
* whether the closest Munsell colour is plausible,
* whether a very bright/dark estimate is caused by bad masking or cropping.

The colour cards are not absolute colour standards. Their appearance also depends on the monitor and image viewer. They are mainly useful for debugging and comparison.

## Pulling comparison statistics

If you have an enriched file containing laboratory `orgC_lab` and algorithm estimates, run the comparison/statistics script.

Typical direct comparison:

```bash
python3 scripts/compare_orgC_lab_vs_CS.py \
  --input outputs/test_stat_orgC_enriched_with_gray.xlsx \
  --lab-col orgC_lab \
  --estimate-col "SOC_est%" \
  --output-prefix outputs/orgC_with_gray
```

Expected outputs:

```text
outputs/orgC_with_gray_statistics.csv
outputs/orgC_with_gray_predictions.csv
outputs/orgC_with_gray_bland_altman.png
outputs/orgC_with_gray_scatter.png
```

Important statistics:

* `pearson_r`: linear association.
* `spearman_r`: rank/monotonic association.
* `CCC_agreement`: agreement, not only correlation.
* `bias_mean_CS_minus_lab`: average over- or under-estimation.
* `MAE`: average absolute error.
* `RMSE`: error with stronger penalty for large mistakes.
* `R2_direct_prediction`: how well the raw algorithm predicts lab values directly.
* `calibration_slope_lab_from_CS`: slope for lab-from-estimate calibration.
* `R2_calibration_regression`: how much lab variation is explained after linear calibration.

A high correlation but poor agreement means the algorithm may be useful after calibration, but should not be used directly as a lab substitute.

## Training calibration models

To train simple calibration models:

```bash
python3 scripts/train_calibration_models.py \
  --input outputs/test_stat_orgC_enriched_with_gray.xlsx \
  --target orgC_lab \
  --test-size 19 \
  --seed 42
```

This creates:

```text
outputs/calibration/calibration_results.xlsx
outputs/calibration/calibration_summary.csv
outputs/calibration/calibration_coefficients.csv
outputs/calibration/calibration_predictions.csv
outputs/calibration/train_test_split.csv
```

Current simple models include:

```text
Model 1: lab_orgC ~ SOC_est%
Model 2: lab_orgC ~ L
Model 3: lab_orgC ~ L + a + b
Model 4: lab_orgC ~ Munsell value + Munsell chroma
```

For small datasets, interpret train/test results cautiously. With very small grey-scale subsets, leave-one-out or repeated cross-validation is usually more informative than one fixed split.

## Generating repository overview

To generate a repository overview file:

```bash
python3 scripts/create_repo_overview.py
```

This creates:

```text
REPO_OVERVIEW.md
```

The overview includes:

* project tree,
* Python modules,
* top-level functions/classes,
* short module docstrings where available,
* command examples,
* ignored/generated folders.

This is useful for project documentation, handover, proposal reporting, and repository review.

## Scientific interpretation

The current method should be considered exploratory.

The image-derived signal can be useful, especially when grey-scale correction improves the correlation with laboratory `orgC`. However, direct agreement may still be poor. In practice, the algorithm should be treated as:

```text
colour-derived proxy + calibration model
```

not as a standalone laboratory measurement.

Important limitations:

* lighting variation,
* soil moisture,
* shadows,
* grain size and surface texture,
* camera processing,
* imperfect grey-scale detection,
* small calibration datasets.

Recommended next step:

```text
Increase the number of lab-matched samples, keep grey-scale correction, and evaluate calibrated models using cross-validation.
```
