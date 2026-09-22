from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path
from numbers import Number

from openpyxl import writer
import pandas as pd

from .lab_io import read_lab_workbooks


def sample_code_from_prediction_row(
    row: pd.Series,
) -> str | None:
    """
    Extract the base sample code used to join an image prediction
    with the laboratory table.

    Image filename has priority.
    """

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

    for column in (
        "filename_code_base",
        "sample_code_base",
        "sample_code_full",
    ):
        value = row.get(column)

        if value is None or pd.isna(value):
            continue

        match = re.match(
            r"\s*([A-Za-z0-9]+)",
            str(value).strip(),
        )

        if match:
            return match.group(1).upper()

    return None

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


def _is_missing_value(value) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in {
            "",
            "nan",
            "none",
            "null",
            "<na>",
        }

    try:
        result = pd.isna(value)
        return bool(result)
    except (TypeError, ValueError):
        return False


def _comparison_key(value):
    """
    Normalize values for conflict detection.

    Treat 1 and 1.0 as the same value.
    """
    if isinstance(value, bool):
        return ("bool", value)

    if isinstance(value, Number):
        return ("number", round(float(value), 12))

    if isinstance(value, pd.Timestamp):
        return ("datetime", value.isoformat())

    return ("text", str(value).strip())


def _ordered_unique_text(values) -> list[str]:
    result = []
    seen = set()

    for value in values:
        if _is_missing_value(value):
            continue

        text = str(value).strip()

        if text not in seen:
            seen.add(text)
            result.append(text)

    return result


def resolve_duplicate_lab_rows(
    lab: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Collapse duplicate laboratory SampleCodes column by column.

    Resolution rules:
      - non-empty values fill empty values;
      - identical values are kept once;
      - conflicting non-empty values use the later --lab input file;
      - conflicts are recorded for inspection.

    Returns:
      resolved_lab
      duplicate_input_rows
      resolution_report
      conflict_report
    """
    lab = lab.copy()

    if "SampleCode" not in lab.columns:
        raise ValueError(
            "Cannot resolve laboratory duplicates without SampleCode."
        )

    if "_lab_source_order" not in lab.columns:
        lab["_lab_source_order"] = 0

    if "_lab_row_order" not in lab.columns:
        lab["_lab_row_order"] = range(len(lab))

    if "_lab_input_order" not in lab.columns:
        lab["_lab_input_order"] = range(len(lab))

    duplicate_mask = (
        lab["SampleCode"].notna()
        & lab["SampleCode"].duplicated(keep=False)
    )

    duplicate_input_rows = (
        lab.loc[duplicate_mask]
        .sort_values(
            [
                "SampleCode",
                "_lab_source_order",
                "_lab_row_order",
            ]
        )
        .copy()
    )

    output_rows = []
    resolution_records = []
    conflict_records = []

    internal_columns = {
        "_lab_source_file",
        "_lab_source_sheet",
        "_lab_source_order",
        "_lab_row_order",
        "_lab_input_order",
        "_lab_source_files",
        "_lab_source_sheets",
        "_lab_duplicate_rows_merged",
        "_lab_conflict_columns",
    }

    rows_with_code = lab[
        lab["SampleCode"].notna()
    ].copy()

    for sample_code, group in rows_with_code.groupby(
        "SampleCode",
        sort=False,
    ):
        group = group.sort_values(
            [
                "_lab_source_order",
                "_lab_row_order",
            ]
        )

        # Start with the last row because later input files have priority.
        merged = group.iloc[-1].copy()
        conflict_columns = []

        for column in lab.columns:
            if column in internal_columns:
                continue

            candidates = []

            for _, row in group.iterrows():
                value = row[column]

                if not _is_missing_value(value):
                    candidates.append((row, value))

            if not candidates:
                merged[column] = pd.NA
                continue

            # Last non-empty value wins.
            chosen_row, chosen_value = candidates[-1]
            merged[column] = chosen_value

            unique_values = {}

            for source_row, value in candidates:
                key = _comparison_key(value)

                unique_values.setdefault(
                    key,
                    [],
                ).append(
                    {
                        "value": value,
                        "source_file": source_row.get(
                            "_lab_source_file",
                            "",
                        ),
                        "source_sheet": source_row.get(
                            "_lab_source_sheet",
                            "",
                        ),
                    }
                )

            if len(unique_values) > 1:
                conflict_columns.append(column)

                displayed_values = []

                for source_row, value in candidates:
                    source_file = source_row.get(
                        "_lab_source_file",
                        "",
                    )
                    source_sheet = source_row.get(
                        "_lab_source_sheet",
                        "",
                    )

                    displayed_values.append(
                        f"{source_file}"
                        f"[{source_sheet}]: {value!r}"
                    )

                conflict_records.append(
                    {
                        "SampleCode": sample_code,
                        "column": column,
                        "chosen_value": chosen_value,
                        "chosen_source_file": chosen_row.get(
                            "_lab_source_file",
                            "",
                        ),
                        "chosen_source_sheet": chosen_row.get(
                            "_lab_source_sheet",
                            "",
                        ),
                        "all_values": " || ".join(
                            displayed_values
                        ),
                    }
                )

        source_files = _ordered_unique_text(
            group["_lab_source_file"]
            if "_lab_source_file" in group.columns
            else []
        )

        source_sheets = _ordered_unique_text(
            group["_lab_source_sheet"]
            if "_lab_source_sheet" in group.columns
            else []
        )

        merged["_lab_source_files"] = " | ".join(
            source_files
        )

        merged["_lab_source_sheets"] = " | ".join(
            source_sheets
        )

        merged["_lab_duplicate_rows_merged"] = len(group)

        merged["_lab_conflict_columns"] = " | ".join(
            conflict_columns
        )

        # Retain the provenance of the highest-priority row.
        merged["_lab_source_file"] = group.iloc[-1].get(
            "_lab_source_file",
            "",
        )

        merged["_lab_source_sheet"] = group.iloc[-1].get(
            "_lab_source_sheet",
            "",
        )

        merged["_lab_input_order"] = group[
            "_lab_input_order"
        ].min()

        output_rows.append(merged.to_dict())

        if len(group) > 1:
            resolution_records.append(
                {
                    "SampleCode": sample_code,
                    "input_rows": len(group),
                    "rows_removed": len(group) - 1,
                    "source_files": " | ".join(
                        source_files
                    ),
                    "selected_source_file": merged[
                        "_lab_source_file"
                    ],
                    "conflict_columns": " | ".join(
                        conflict_columns
                    ),
                    "conflict_count": len(
                        conflict_columns
                    ),
                }
            )

    # Rows whose SampleCode could not be extracted cannot be combined.
    rows_without_code = lab[
        lab["SampleCode"].isna()
    ].copy()

    for _, row in rows_without_code.iterrows():
        row = row.copy()

        source_file = row.get(
            "_lab_source_file",
            "",
        )

        source_sheet = row.get(
            "_lab_source_sheet",
            "",
        )

        row["_lab_source_files"] = source_file
        row["_lab_source_sheets"] = source_sheet
        row["_lab_duplicate_rows_merged"] = 1
        row["_lab_conflict_columns"] = ""

        output_rows.append(row.to_dict())

    resolved = pd.DataFrame(output_rows)

    if not resolved.empty:
        resolved = resolved.sort_values(
            "_lab_input_order",
            na_position="last",
        ).reset_index(drop=True)

    resolved = resolved.drop(
        columns=[
            "_lab_source_order",
            "_lab_row_order",
            "_lab_input_order",
        ],
        errors="ignore",
    )

    resolution_report = pd.DataFrame(
        resolution_records,
        columns=[
            "SampleCode",
            "input_rows",
            "rows_removed",
            "source_files",
            "selected_source_file",
            "conflict_columns",
            "conflict_count",
        ],
    )

    conflict_report = pd.DataFrame(
        conflict_records,
        columns=[
            "SampleCode",
            "column",
            "chosen_value",
            "chosen_source_file",
            "chosen_source_sheet",
            "all_values",
        ],
    )

    remaining_duplicates = (
        resolved["SampleCode"].notna()
        & resolved["SampleCode"].duplicated(
            keep=False
        )
    )

    if remaining_duplicates.any():
        raise RuntimeError(
            "Laboratory duplicate resolution failed: duplicate "
            "SampleCodes remain after collapsing rows."
        )

    return (
        resolved,
        duplicate_input_rows,
        resolution_report,
        conflict_report,
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

    (
        lab,
        duplicate_lab_codes,
        duplicate_lab_resolution,
        duplicate_lab_conflicts,
    ) = resolve_duplicate_lab_rows(lab)

    # ------------------------------------------------------------------
    # Build prediction matching keys
    # ------------------------------------------------------------------
    predictions["SampleCode"] = predictions.apply(
        sample_code_from_prediction_row,
        axis=1,
    )

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


    lab_codes = set(
        lab["SampleCode"].dropna().astype(str)
    )

    prediction_codes = set(
        predictions["SampleCode"].dropna().astype(str)
    )

    common_codes = lab_codes & prediction_codes

    print("\n=== Lab/image matching keys ===")
    print(f"Unique lab codes:        {len(lab_codes)}")
    print(f"Unique prediction codes: {len(prediction_codes)}")
    print(f"Common codes:            {len(common_codes)}")

    print("\nExample lab codes:")
    print(sorted(lab_codes)[:20])

    print("\nExample prediction codes:")
    print(sorted(prediction_codes)[:20])

    print("\nExample common codes:")
    print(sorted(common_codes)[:20])

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

    duplicate_lab_rows_collapsed = (
        int(
            duplicate_lab_resolution[
                "rows_removed"
            ].sum()
        )
        if not duplicate_lab_resolution.empty
        else 0
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
                "duplicate_lab_input_rows": len(
                    duplicate_lab_codes
                ),
                "duplicate_lab_codes_resolved": len(
                    duplicate_lab_resolution
                ),
                "duplicate_lab_rows_collapsed": (
                    duplicate_lab_rows_collapsed
                ),
                "duplicate_lab_conflict_cells": len(
                    duplicate_lab_conflicts
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

        duplicate_lab_codes.to_excel(
            writer,
            sheet_name="duplicate_lab_input_rows",
            index=False,
        )

        duplicate_lab_resolution.to_excel(
            writer,
            sheet_name="duplicate_lab_resolution",
            index=False,
        )

        duplicate_lab_conflicts.to_excel(
            writer,
            sheet_name="duplicate_lab_conflicts",
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

    duplicate_lab_codes.to_csv(
        f"{base}_duplicate_lab_input_rows.csv",
        index=False,
    )

    duplicate_lab_resolution.to_csv(
        f"{base}_duplicate_lab_resolution.csv",
        index=False,
    )

    duplicate_lab_conflicts.to_csv(
        f"{base}_duplicate_lab_conflicts.csv",
        index=False,
    )
    
    summary.to_csv(
        f"{base}_summary.csv",
        index=False,
    )

    print(f"\nSaved enriched Excel file: {output_xlsx}")
    print(f"Saved CSV reports with prefix: {base}_*.csv")

    return enriched