#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soil_color_orgc.lab_merge import enrich_lab_file
from soil_color_orgc.pipeline import run_image_pipeline


SUPPORTED_IMAGE_SUFFIXES = {
    ".jfif",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


def list_direct_image_files(
    directory: str | Path,
) -> list[Path]:
    """
    Return image files directly inside a directory.

    Subdirectories are deliberately excluded. Therefore, when directory is
    data/samples, files under data/samples/with_gray are not returned.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Image input is not a directory: {directory}"
        )

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def stage_image_files(
    image_files: list[Path],
    destination: str | Path,
) -> Path:
    """
    Put selected image files into a temporary flat directory.

    Symlinks are preferred so the images do not need to be copied. A regular
    copy is used if symlink creation is unavailable.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for source in image_files:
        source = source.resolve()
        target = destination / source.name

        if target.exists():
            raise ValueError(
                f"Duplicate filename while staging images: {target.name}"
            )

        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)

    return destination


def normalize_sample_code(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text or text.upper() in {
        "NAN",
        "NONE",
        "NULL",
        "<NA>",
    }:
        return None

    return text.upper()


def prediction_sample_code(row: pd.Series) -> str | None:
    """
    Determine the laboratory matching code.

    IMPORTANT:
    The image filename is authoritative for the lab/image join.

    Examples:
        ABYU.jfif       -> ABYU
        ABYU.jpg        -> ABYU
        ABYU-foo.jpg    -> ABYU

    Pipeline/OCR sample-code fields are only fallbacks.
    """

    # 1. Filename first
    image = row.get("image")

    if image is not None and not pd.isna(image):
        filename = os.path.basename(str(image).strip())
        stem = os.path.splitext(filename)[0]

        match = re.match(
            r"\s*([A-Za-z0-9]+)",
            stem,
        )

        if match:
            return match.group(1).upper()

    # 2. Fallback to pipeline-derived codes
    for column in (
        "filename_code_base",
        "sample_code_base",
        "sample_code_full",
    ):
        value = row.get(column)

        if value is None or pd.isna(value):
            continue

        text = str(value).strip()

        if not text:
            continue

        match = re.match(
            r"\s*([A-Za-z0-9]+)",
            text,
        )

        if match:
            return match.group(1).upper()

    return None

def coerce_nullable_boolean(
    series: pd.Series,
) -> pd.Series:
    """
    Convert boolean-like values to pandas nullable BooleanDtype.
    """
    if str(series.dtype) == "boolean":
        return series

    if series.dtype == bool:
        return series.astype("boolean")

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "on": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "off": False,
    }

    return normalized.map(mapping).astype("boolean")


def find_gray_applied_column(
    predictions: pd.DataFrame,
) -> str | None:
    """
    Find an existing pipeline column reporting whether correction was applied.
    """
    candidates = [
        "gray_scale_applied",
        "grey_scale_applied",
        "gray_calibration_applied",
        "grey_calibration_applied",
    ]

    for column in candidates:
        if column in predictions.columns:
            return column

    return None


def annotate_gray_predictions(
    predictions: pd.DataFrame,
    gray_calibration_requested: bool,
) -> pd.DataFrame:
    predictions = predictions.copy()

    predictions["image_source"] = "with_gray"
    predictions["gray_scale_requested"] = (
        gray_calibration_requested
    )

    if not gray_calibration_requested:
        predictions["gray_scale_applied"] = False
        predictions["calibration_mode"] = "uncorrected"

        return predictions

    existing_column = find_gray_applied_column(
        predictions
    )

    if existing_column:
        predictions["gray_scale_applied"] = (
            coerce_nullable_boolean(
                predictions[existing_column]
            )
        )
    else:
        # Do not claim that correction succeeded when the pipeline did not
        # provide an explicit status column.
        predictions["gray_scale_applied"] = pd.Series(
            pd.NA,
            index=predictions.index,
            dtype="boolean",
        )

    def determine_mode(value):
        if pd.isna(value):
            return "gray_requested_status_unknown"

        if bool(value):
            return "gray_corrected"

        return "gray_requested_but_not_applied"

    predictions["calibration_mode"] = predictions[
        "gray_scale_applied"
    ].apply(determine_mode)

    return predictions


def annotate_without_gray_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    predictions = predictions.copy()

    predictions["image_source"] = "without_gray"
    predictions["gray_scale_requested"] = False
    predictions["gray_scale_applied"] = False
    predictions["calibration_mode"] = "uncorrected"

    return predictions


def combine_prediction_tables(
    gray_predictions: pd.DataFrame,
    without_gray_predictions: pd.DataFrame | None,
    gray_calibration_requested: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine corrected and uncorrected prediction rows.

    Rules
    -----
    1. Duplicate sample codes within one source are treated as errors.
    2. When a sample occurs in both sources, the with-gray row wins.
    3. Rows with no extracted SampleCode are all preserved.
    """
    gray_predictions = annotate_gray_predictions(
        gray_predictions,
        gray_calibration_requested=gray_calibration_requested,
    )

    frames = [gray_predictions]

    if without_gray_predictions is not None:
        frames.append(
            annotate_without_gray_predictions(
                without_gray_predictions
            )
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined["SampleCode"] = combined.apply(
        prediction_sample_code,
        axis=1,
    )

    combined["filename_match_code"] = combined["image"].apply(
        lambda value: (
            prediction_sample_code(
                pd.Series({"image": value})
            )
            if pd.notna(value)
            else None
        )
    )
    
    combined["source_priority"] = (
        combined["image_source"]
        .map(
            {
                "with_gray": 0,
                "without_gray": 1,
            }
        )
        .fillna(9)
    )

    rows_with_code = combined[
        combined["SampleCode"].notna()
    ].copy()

    rows_without_code = combined[
        combined["SampleCode"].isna()
    ].copy()

    # Duplicate codes inside one source are ambiguous.
    same_source_duplicates = rows_with_code[
        rows_with_code.duplicated(
            subset=[
                "image_source",
                "SampleCode",
            ],
            keep=False,
        )
    ].sort_values(
        [
            "image_source",
            "SampleCode",
            "image",
        ],
        na_position="last",
    )

    if not same_source_duplicates.empty:
        columns = [
            column
            for column in [
                "image_source",
                "SampleCode",
                "image",
                "processing_status",
                "processing_error",
            ]
            if column in same_source_duplicates.columns
        ]

        raise ValueError(
            "Duplicate sample codes were found within the same "
            "image source:\n\n"
            + same_source_duplicates[
                columns
            ].to_string(index=False)
        )

    rows_with_code = rows_with_code.sort_values(
        [
            "SampleCode",
            "source_priority",
            "image",
        ],
        na_position="last",
    )

    # Record samples existing in both folders.
    cross_source_duplicates = rows_with_code[
        rows_with_code.duplicated(
            subset=["SampleCode"],
            keep=False,
        )
    ].copy()

    if not cross_source_duplicates.empty:
        cross_source_duplicates[
            "selected_for_combined_dataset"
        ] = ~cross_source_duplicates.duplicated(
            subset=["SampleCode"],
            keep="first",
        )

    # Corrected rows have source_priority=0 and therefore win.
    selected_rows_with_code = (
        rows_with_code
        .drop_duplicates(
            subset=["SampleCode"],
            keep="first",
        )
        .copy()
    )

    # Do not use drop_duplicates on the complete table because it would
    # collapse every missing SampleCode into one row.
    combined = pd.concat(
        [
            selected_rows_with_code,
            rows_without_code,
        ],
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined
        .sort_values(
            [
                "source_priority",
                "SampleCode",
                "image",
            ],
            na_position="last",
        )
        .drop(columns=["source_priority"])
        .reset_index(drop=True)
    )

    if "source_priority" in cross_source_duplicates.columns:
        cross_source_duplicates = (
            cross_source_duplicates
            .drop(columns=["source_priority"])
            .reset_index(drop=True)
        )

    return combined, cross_source_duplicates


def run_pipeline_to_dataframe(
    *,
    samples_dir: str | Path,
    munsell_csv: str | Path,
    output_csv: str | Path,
    debug_dir: str | Path,
    save_debug_masks: bool,
    deltae_threshold: float,
    use_gray_calibration: bool,
) -> pd.DataFrame:
    """
    Run the existing pipeline and read its CSV output.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_image_pipeline(
        samples_dir=str(samples_dir),
        munsell_csv=str(munsell_csv),
        output_csv=str(output_csv),
        debug_dir=str(debug_dir),
        save_debug_masks=save_debug_masks,
        deltae_threshold=deltae_threshold,
        use_gray_calibration=use_gray_calibration,
    )

    if not output_csv.exists():
        raise RuntimeError(
            "Image pipeline completed without creating the expected "
            f"CSV file: {output_csv}"
        )

    try:
        predictions = pd.read_csv(output_csv)
    except pd.errors.EmptyDataError as exc:
        raise RuntimeError(
            f"Image pipeline created an empty CSV file: {output_csv}"
        ) from exc

    print(
        f"Loaded {len(predictions)} prediction rows from "
        f"{output_csv}."
    )

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process grey-scale and optional legacy soil images, "
            "combine their predictions, and enrich laboratory results."
        )
    )

    parser.add_argument(
        "--samples",
        default="data/samples/with_gray",
        help=(
            "Folder containing images with a grey scale. "
            "Grey-scale correction is enabled by default."
        ),
    )

    parser.add_argument(
        "--munsell",
        default="data/munsell/rit_munsell.csv",
        help="Path to RIT Munsell CSV.",
    )

    parser.add_argument(
        "--lab",
        nargs="+",
        default=["data/lab/test_stat_orgC.xlsx"],
        help=(
            "One or more laboratory XLSX files or glob patterns. "
            "Every workbook must contain ID and orgC_lab."
        ),
    )

    parser.add_argument(
        "--results",
        default="outputs/results_combined.csv",
        help="Output combined predictions CSV.",
    )

    parser.add_argument(
        "--enriched",
        default="outputs/test_stat_orgC_enriched_combined.xlsx",
        help="Output enriched Excel file.",
    )

    parser.add_argument(
        "--debug-dir",
        default="debug_masks",
        help=(
            "Folder for debug images from the primary grey-scale "
            "pipeline. Legacy debug images are stored in a "
            "without_gray subdirectory."
        ),
    )

    parser.add_argument(
        "--no-debug-masks",
        action="store_true",
        help="Disable debug-mask generation.",
    )

    parser.add_argument(
        "--deltae-threshold",
        type=float,
        default=8.0,
        help="Maximum accepted DeltaE2000 for Munsell matching.",
    )

    parser.add_argument(
        "--no-gray-calibration",
        action="store_true",
        help=(
            "Disable grey-scale correction for the primary --samples "
            "folder. This is mainly intended for testing."
        ),
    )

    parser.add_argument(
        "--include-without-gray",
        action="store_true",
        help=(
            "Also process root-level image files from "
            "--without-gray-samples without colour correction."
        ),
    )

    parser.add_argument(
        "--without-gray-samples",
        default="data/samples",
        help=(
            "Folder containing legacy images without a grey scale. "
            "Only files directly inside this directory are processed. "
            "Subdirectories such as with_gray are excluded."
        ),
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_image_files = list_direct_image_files(
        args.samples
    )

    if not primary_image_files:
        raise ValueError(
            f"No supported images were found directly inside "
            f"{args.samples}."
        )

    print("\n=== Primary grey-scale image source ===")
    print(f"Directory: {args.samples}")
    print(f"Images found: {len(primary_image_files)}")
    print(
        "Grey-scale calibration: "
        f"{'disabled' if args.no_gray_calibration else 'enabled'}"
    )

    with tempfile.TemporaryDirectory(
        prefix="soil_color_orgc_"
    ) as temporary_directory:
        temporary_directory = Path(
            temporary_directory
        )

        gray_csv = (
            temporary_directory
            / "predictions_with_gray.csv"
        )

        gray_predictions = run_pipeline_to_dataframe(
            samples_dir=args.samples,
            munsell_csv=args.munsell,
            output_csv=gray_csv,
            debug_dir=args.debug_dir,
            save_debug_masks=not args.no_debug_masks,
            deltae_threshold=args.deltae_threshold,
            use_gray_calibration=(
                not args.no_gray_calibration
            ),
        )

        without_gray_predictions = None

        if args.include_without_gray:
            legacy_image_files = list_direct_image_files(
                args.without_gray_samples
            )

            print("\n=== Legacy image source ===")
            print(
                f"Directory: {args.without_gray_samples}"
            )
            print(
                f"Root-level images found: "
                f"{len(legacy_image_files)}"
            )
            print("Grey-scale calibration: disabled")

            if legacy_image_files:
                staged_legacy_directory = stage_image_files(
                    legacy_image_files,
                    temporary_directory
                    / "legacy_images",
                )

                legacy_csv = (
                    temporary_directory
                    / "predictions_without_gray.csv"
                )

                legacy_debug_dir = (
                    Path(args.debug_dir)
                    / "without_gray"
                )

                without_gray_predictions = (
                    run_pipeline_to_dataframe(
                        samples_dir=staged_legacy_directory,
                        munsell_csv=args.munsell,
                        output_csv=legacy_csv,
                        debug_dir=legacy_debug_dir,
                        save_debug_masks=(
                            not args.no_debug_masks
                        ),
                        deltae_threshold=(
                            args.deltae_threshold
                        ),
                        use_gray_calibration=False,
                    )
                )
            else:
                print(
                    "No root-level legacy images were found; "
                    "continuing with the grey-scale dataset only."
                )

        combined_predictions, cross_source_duplicates = (
            combine_prediction_tables(
                gray_predictions=gray_predictions,
                without_gray_predictions=(
                    without_gray_predictions
                ),
                gray_calibration_requested=(
                    not args.no_gray_calibration
                ),
            )
        )

    combined_predictions.to_csv(
        results_path,
        index=False,
    )

    duplicate_report_path = results_path.with_name(
        f"{results_path.stem}"
        "_cross_source_duplicates.csv"
    )

    cross_source_duplicates.to_csv(
        duplicate_report_path,
        index=False,
    )

    print("\n=== Combined prediction dataset ===")
    print(
        f"Combined prediction rows: "
        f"{len(combined_predictions)}"
    )

    if "image_source" in combined_predictions.columns:
        source_counts = (
            combined_predictions["image_source"]
            .value_counts(dropna=False)
        )

        for source, count in source_counts.items():
            print(f"  {source}: {count}")

    missing_codes = int(
        combined_predictions["SampleCode"]
        .isna()
        .sum()
    )

    print(
        f"Rows without extracted SampleCode: "
        f"{missing_codes}"
    )

    if cross_source_duplicates.empty:
        print(
            "Samples occurring in both image sources: 0"
        )
    else:
        duplicate_codes = (
            cross_source_duplicates["SampleCode"]
            .nunique(dropna=True)
        )

        print(
            "Samples occurring in both image sources: "
            f"{duplicate_codes}"
        )
        print(
            "The with-gray version was retained for "
            "each overlapping sample."
        )

    print(
        f"Saved combined predictions: {results_path}"
    )
    print(
        "Saved cross-source duplicate report: "
        f"{duplicate_report_path}"
    )

    enrich_lab_file(
        lab_xlsx=args.lab,
        predictions_csv=results_path,
        output_xlsx=args.enriched,
    )


if __name__ == "__main__":
    main()