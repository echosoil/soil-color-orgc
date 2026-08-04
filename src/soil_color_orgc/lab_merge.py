from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .lab_io import read_lab_workbooks


def base_code_from_lab(id_str) -> str | None:
    """
    Extract the leading alphanumeric sample code from a laboratory ID.

    Examples:
        ABCD-1234 -> ABCD
        XYZ99-7A  -> XYZ99
        ABCD_1234 -> ABCD
    """
    if id_str is None or pd.isna(id_str):
        return None

    match = re.match(r"\s*([A-Za-z0-9]+)", str(id_str))

    return match.group(1).upper() if match else None


def base_code_from_image(name) -> str | None:
    """
    Extract the leading alphanumeric code from an image filename.

    Examples:
        ABCD.jfif                    -> ABCD
        ABCD.JPG                     -> ABCD
        XYZ99-photo.jpg              -> XYZ99
        data/samples/ABCD.jfif       -> ABCD

    os.path.basename() is important here. Without it, a path such as
    data/samples/ABCD.jfif could incorrectly produce DATA.
    """
    if name is None or pd.isna(name):
        return None

    filename = os.path.basename(str(name).strip())
    stem = os.path.splitext(filename)[0]

    match = re.match(r"\s*([A-Za-z0-9]+)", stem)

    return match.group(1).upper() if match else None


def normalize_sample_codes(series: pd.Series) -> pd.Series:
    """
    Normalize sample-code values to uppercase nullable strings.
    """
    normalized = (
        series.astype("string")
        .str.strip()
        .str.upper()
    )

    invalid_values = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "<NA>",
    }

    return normalized.mask(normalized.isin(invalid_values))


def _available_columns(
    df: pd.DataFrame,
    wanted_columns: Iterable[str],
) -> list[str]:
    return [
        column
        for column in wanted_columns
        if column in df.columns
    ]


def _duplicate_error_message(
    title: str,
    duplicates: pd.DataFrame,
    preferred_columns: Iterable[str],
) -> str:
    columns = _available_columns(
        duplicates,
        preferred_columns,
    )

    preview = duplicates[columns].head(100)

    return (
        f"{title}\n\n"
        f"{preview.to_string(index=False)}\n\n"
        f"Duplicate rows found: {len(duplicates)}. "
        "The merge was stopped because duplicate sample codes would create "
        "a many-to-many merge and would incorrectly duplicate training rows."
    )


def enrich_lab_file(
    lab_xlsx: str | Path | Iterable[str | Path],
    predictions_csv: str | Path,
    output_xlsx: str | Path,
    required_lab_columns: Iterable[str] = ("ID", "orgC_lab"),
    fail_on_duplicate_lab_codes: bool = True,
    fail_on_duplicate_image_codes: bool = True,
) -> pd.DataFrame:
    """
    Merge one or more laboratory XLSX files with image predictions.

    Parameters
    ----------
    lab_xlsx:
        One XLSX path, several XLSX paths, or glob patterns accepted by
        read_lab_workbooks(), for example:

            "data/lab/results.xlsx"

            [
                "data/lab/results_1.xlsx",
                "data/lab/results_2.xlsx",
            ]

            "data/lab/source/*.xlsx"

    predictions_csv:
        CSV produced by the image-processing pipeline.

    output_xlsx:
        Destination enriched workbook.

    required_lab_columns:
        Columns which every laboratory workbook must contain.

    fail_on_duplicate_lab_codes:
        Stop if the same SampleCode occurs more than once in the combined
        laboratory data.

    fail_on_duplicate_image_codes:
        Stop if the same SampleCode occurs more than once in predictions.
    """
    predictions_csv = Path(predictions_csv)
    output_xlsx = Path(output_xlsx)

    lab = read_lab_workbooks(
        lab_xlsx,
        required_columns=required_lab_columns,
    )

    if not predictions_csv.exists():
        raise FileNotFoundError(
            f"Predictions CSV not found: {predictions_csv}"
        )

    predictions = pd.read_csv(predictions_csv)

    if "ID" not in lab.columns:
        raise ValueError(
            "Combined laboratory data must contain an 'ID' column."
        )

    if "image" not in predictions.columns:
        raise ValueError(
            "Predictions CSV must contain an 'image' column."
        )

    # ------------------------------------------------------------------
    # Build laboratory matching keys
    # ------------------------------------------------------------------
    lab["SampleCode"] = lab["ID"].apply(base_code_from_lab)
    lab["SampleCode"] = normalize_sample_codes(
        lab["SampleCode"]
    )

    # ------------------------------------------------------------------
    # Build prediction matching keys
    # ------------------------------------------------------------------
    if "sample_code_base" in predictions.columns:
        predictions["SampleCode"] = normalize_sample_codes(
            predictions["sample_code_base"]
        )

        missing_prediction_code = predictions["SampleCode"].isna()

        predictions.loc[
            missing_prediction_code,
            "SampleCode",
        ] = predictions.loc[
            missing_prediction_code,
            "image",
        ].apply(base_code_from_image)
    else:
        predictions["SampleCode"] = predictions[
            "image"
        ].apply(base_code_from_image)

    predictions["SampleCode"] = normalize_sample_codes(
        predictions["SampleCode"]
    )

    # ------------------------------------------------------------------
    # Detect duplicates before merging
    #
    # A duplicate in both tables could produce a many-to-many merge and
    # artificially multiply the number of training samples.
    # ------------------------------------------------------------------
    duplicate_lab_codes = (
        lab[
            lab["SampleCode"].notna()
            & lab["SampleCode"].duplicated(keep=False)
        ]
        .sort_values(
            ["SampleCode", "ID"],
            na_position="last",
        )
        .copy()
    )

    duplicate_image_codes = (
        predictions[
            predictions["SampleCode"].notna()
            & predictions["SampleCode"].duplicated(keep=False)
        ]
        .sort_values(
            ["SampleCode", "image"],
            na_position="last",
        )
        .copy()
    )

    if fail_on_duplicate_lab_codes and not duplicate_lab_codes.empty:
        raise ValueError(
            _duplicate_error_message(
                title=(
                    "Duplicate laboratory SampleCodes were found across "
                    "the combined XLSX files."
                ),
                duplicates=duplicate_lab_codes,
                preferred_columns=[
                    "SampleCode",
                    "ID",
                    "orgC_lab",
                    "sd_lab",
                    "orgC_CS",
                    "_lab_source_file",
                    "_lab_source_sheet",
                ],
            )
        )

    if fail_on_duplicate_image_codes and not duplicate_image_codes.empty:
        raise ValueError(
            _duplicate_error_message(
                title=(
                    "Duplicate image-prediction SampleCodes were found."
                ),
                duplicates=duplicate_image_codes,
                preferred_columns=[
                    "SampleCode",
                    "image",
                    "sample_code_base",
                    "sample_code_source",
                    "processing_status",
                    "processing_error",
                ],
            )
        )

    # ------------------------------------------------------------------
    # Prediction columns to append to the combined laboratory data
    # ------------------------------------------------------------------
    desired_prediction_columns = [
        "SampleCode",
        "image",
        "sample_code_full",
        "sample_code_base",
        "sample_code_source",
        "sample_code_match_score",
        "sample_code_matched_fragment",
        "sample_code_error",
        "filename_code_base",
        "code_conflict",
        "L",
        "a",
        "b",
        "best_munsell",
        "deltaE2000",
        "SOC_est%",
        "SOC_method",
        "processing_status",
        "processing_error",

        # Optional grey-scale calibration diagnostics.
        "gray_scale_detected",
        "grey_scale_detected",
        "gray_scale_detection_score",
        "grey_scale_detection_score",
        "gray_calibration_status",
        "grey_calibration_status",
        "gray_calibration_error",
        "grey_calibration_error",
        "mean_imbalance_before",
        "mean_imbalance_after",
        "imbalance_before",
        "imbalance_after",
        "correction_magnitude",
        "gray_patch_fit_error",
        "grey_patch_fit_error",
    ]

    prediction_columns = _available_columns(
        predictions,
        desired_prediction_columns,
    )

    # SampleCode must always be retained for the merge.
    if "SampleCode" not in prediction_columns:
        prediction_columns.insert(0, "SampleCode")

    # ------------------------------------------------------------------
    # Main merge
    # ------------------------------------------------------------------
    enriched = lab.merge(
        predictions[prediction_columns],
        on="SampleCode",
        how="left",
        suffixes=("", "_pred"),
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    # Matching diagnostics
    # ------------------------------------------------------------------
    lab_codes = set(
        lab["SampleCode"].dropna().tolist()
    )

    prediction_codes = set(
        predictions["SampleCode"].dropna().tolist()
    )

    matched_lab_mask = (
        lab["SampleCode"].notna()
        & lab["SampleCode"].isin(prediction_codes)
    )

    matched_prediction_mask = (
        predictions["SampleCode"].notna()
        & predictions["SampleCode"].isin(lab_codes)
    )

    # 1. Lab entries without a corresponding image
    lab_without_image = lab[
        lab["SampleCode"].isna()
        | ~lab["SampleCode"].isin(prediction_codes)
    ].copy()

    lab_without_image["match_problem"] = (
        lab_without_image["SampleCode"].apply(
            lambda value: (
                "no_code_extracted_from_lab_ID"
                if pd.isna(value)
                else "no_image_for_lab_entry"
            )
        )
    )

    # 2. Images without a corresponding lab entry
    image_without_lab = predictions[
        predictions["SampleCode"].isna()
        | ~predictions["SampleCode"].isin(lab_codes)
    ].copy()

    image_without_lab["match_problem"] = (
        image_without_lab["SampleCode"].apply(
            lambda value: (
                "no_code_extracted_from_image_filename"
                if pd.isna(value)
                else "no_lab_entry_for_image"
            )
        )
    )

    lab_source_count = (
        lab["_lab_source_file"].nunique()
        if "_lab_source_file" in lab.columns
        else 1
    )

    summary = pd.DataFrame(
        [
            {
                "lab_source_files": lab_source_count,
                "lab_rows": len(lab),
                "lab_unique_sample_codes": lab[
                    "SampleCode"
                ].nunique(dropna=True),
                "image_prediction_rows": len(predictions),
                "image_unique_sample_codes": predictions[
                    "SampleCode"
                ].nunique(dropna=True),
                "matched_lab_rows": int(
                    matched_lab_mask.sum()
                ),
                "matched_prediction_rows": int(
                    matched_prediction_mask.sum()
                ),
                "lab_without_image_count": len(
                    lab_without_image
                ),
                "image_without_lab_count": len(
                    image_without_lab
                ),
                "duplicate_lab_code_rows": len(
                    duplicate_lab_codes
                ),
                "duplicate_image_code_rows": len(
                    duplicate_image_codes
                ),
            }
        ]
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n=== Matching report ===")
    print(f"Laboratory source files: {lab_source_count}")
    print(f"Combined laboratory rows: {len(lab)}")
    print(
        "Unique laboratory SampleCodes: "
        f"{lab['SampleCode'].nunique(dropna=True)}"
    )
    print(f"Image prediction rows: {len(predictions)}")
    print(
        "Unique image SampleCodes: "
        f"{predictions['SampleCode'].nunique(dropna=True)}"
    )
    print(
        f"Matched laboratory rows: "
        f"{summary.loc[0, 'matched_lab_rows']}"
    )
    print(
        f"Matched prediction rows: "
        f"{summary.loc[0, 'matched_prediction_rows']}"
    )
    print(
        f"Laboratory entries without image: "
        f"{len(lab_without_image)}"
    )
    print(
        f"Images without laboratory entry: "
        f"{len(image_without_lab)}"
    )
    print(
        f"Duplicate laboratory-code rows: "
        f"{len(duplicate_lab_codes)}"
    )
    print(
        f"Duplicate image-code rows: "
        f"{len(duplicate_image_codes)}"
    )

    if "_lab_source_file" in lab.columns:
        source_counts = (
            lab.groupby("_lab_source_file")
            .size()
            .sort_index()
        )

        print("\nLaboratory rows by source file:")

        for source_file, count in source_counts.items():
            print(f"  {source_file}: {count}")

    if not lab_without_image.empty:
        print(
            "\nWARNING: Laboratory entries without "
            "a corresponding image:"
        )

        columns = _available_columns(
            lab_without_image,
            [
                "ID",
                "SampleCode",
                "orgC_lab",
                "_lab_source_file",
                "match_problem",
            ],
        )

        print(
            lab_without_image[columns].to_string(
                index=False
            )
        )

    if not image_without_lab.empty:
        print(
            "\nWARNING: Images without a corresponding "
            "laboratory entry:"
        )

        columns = _available_columns(
            image_without_lab,
            [
                "image",
                "SampleCode",
                "sample_code_base",
                "sample_code_source",
                "match_problem",
            ],
        )

        print(
            image_without_lab[columns].to_string(
                index=False
            )
        )

    if not duplicate_lab_codes.empty:
        print("\nWARNING: Duplicate laboratory SampleCodes:")

        columns = _available_columns(
            duplicate_lab_codes,
            [
                "ID",
                "SampleCode",
                "orgC_lab",
                "_lab_source_file",
            ],
        )

        print(
            duplicate_lab_codes[columns].to_string(
                index=False
            )
        )

    if not duplicate_image_codes.empty:
        print("\nWARNING: Duplicate image SampleCodes:")

        columns = _available_columns(
            duplicate_image_codes,
            [
                "image",
                "SampleCode",
                "sample_code_base",
                "sample_code_source",
            ],
        )

        print(
            duplicate_image_codes[columns].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # Save output files
    # ------------------------------------------------------------------
    output_xlsx.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_xlsx,
        engine="openpyxl",
    ) as writer:
        enriched.to_excel(
            writer,
            sheet_name="enriched",
            index=False,
        )

        summary.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        lab_without_image.to_excel(
            writer,
            sheet_name="lab_without_image",
            index=False,
        )

        image_without_lab.to_excel(
            writer,
            sheet_name="image_without_lab",
            index=False,
        )

        duplicate_lab_codes.to_excel(
            writer,
            sheet_name="duplicate_lab_codes",
            index=False,
        )

        duplicate_image_codes.to_excel(
            writer,
            sheet_name="duplicate_image_codes",
            index=False,
        )

        # Useful when several laboratory files have been combined.
        if "_lab_source_file" in lab.columns:
            laboratory_sources = (
                lab.groupby(
                    [
                        "_lab_source_file",
                        "_lab_source_sheet",
                    ],
                    dropna=False,
                )
                .size()
                .reset_index(name="row_count")
            )

            laboratory_sources.to_excel(
                writer,
                sheet_name="laboratory_sources",
                index=False,
            )

    base = output_xlsx.with_suffix("")

    enriched.to_csv(
        f"{base}_enriched.csv",
        index=False,
    )

    lab_without_image.to_csv(
        f"{base}_lab_without_image.csv",
        index=False,
    )

    image_without_lab.to_csv(
        f"{base}_image_without_lab.csv",
        index=False,
    )

    duplicate_lab_codes.to_csv(
        f"{base}_duplicate_lab_codes.csv",
        index=False,
    )

    duplicate_image_codes.to_csv(
        f"{base}_duplicate_image_codes.csv",
        index=False,
    )

    summary.to_csv(
        f"{base}_summary.csv",
        index=False,
    )

    print(f"\nSaved enriched Excel file: {output_xlsx}")
    print(f"Saved CSV reports with prefix: {base}_*.csv")

    return enriched