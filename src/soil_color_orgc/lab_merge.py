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
    ]

    prediction_columns = _available_columns(predictions, desired_prediction_columns)

    # Main enriched table: all lab rows, with image estimates if available
    enriched = lab.merge(
        predictions[prediction_columns],
        on="SampleCode",
        how="left",
        suffixes=("", "_pred"),
    )

    # 1. Lab entries without corresponding image
    lab_codes = set(lab["SampleCode"].dropna())
    prediction_codes = set(predictions["SampleCode"].dropna())

    lab_without_image = lab[
        lab["SampleCode"].isna() | ~lab["SampleCode"].isin(prediction_codes)
    ].copy()

    lab_without_image["match_problem"] = lab_without_image["SampleCode"].apply(
        lambda x: "no_code_extracted_from_lab_ID" if pd.isna(x) else "no_image_for_lab_entry"
    )

    # 2. Images without corresponding lab entry
    image_without_lab = predictions[
        predictions["SampleCode"].isna() | ~predictions["SampleCode"].isin(lab_codes)
    ].copy()

    image_without_lab["match_problem"] = image_without_lab["SampleCode"].apply(
        lambda x: "no_code_extracted_from_image_filename" if pd.isna(x) else "no_lab_entry_for_image"
    )

    # 3. Duplicate codes in lab file
    duplicate_lab_codes = (
        lab[lab["SampleCode"].notna()]
        .groupby("SampleCode")
        .filter(lambda g: len(g) > 1)
        .sort_values("SampleCode")
        .copy()
    )

    # 4. Duplicate codes in image predictions
    duplicate_image_codes = (
        predictions[predictions["SampleCode"].notna()]
        .groupby("SampleCode")
        .filter(lambda g: len(g) > 1)
        .sort_values("SampleCode")
        .copy()
    )

    # Summary
    summary = pd.DataFrame([
        {
            "lab_rows": len(lab),
            "image_prediction_rows": len(predictions),
            "matched_lab_rows": enriched["image"].notna().sum() if "image" in enriched.columns else 0,
            "lab_without_image_count": len(lab_without_image),
            "image_without_lab_count": len(image_without_lab),
            "duplicate_lab_code_rows": len(duplicate_lab_codes),
            "duplicate_image_code_rows": len(duplicate_image_codes),
        }
    ])

    # Console report
    print("\n=== Matching report ===")
    print(f"Lab rows: {len(lab)}")
    print(f"Image prediction rows: {len(predictions)}")
    print(f"Matched lab rows: {summary.loc[0, 'matched_lab_rows']}")
    print(f"Lab entries without image: {len(lab_without_image)}")
    print(f"Images without lab entry: {len(image_without_lab)}")
    print(f"Duplicate lab-code rows: {len(duplicate_lab_codes)}")
    print(f"Duplicate image-code rows: {len(duplicate_image_codes)}")

    if not lab_without_image.empty:
        print("\nWARNING: Lab entries without corresponding image:")
        print(lab_without_image[["ID", "SampleCode", "match_problem"]].to_string(index=False))

    if not image_without_lab.empty:
        print("\nWARNING: Images without corresponding lab entry:")
        cols = _available_columns(
            image_without_lab,
            ["image", "SampleCode", "sample_code_base", "sample_code_source", "match_problem"]
        )
        print(image_without_lab[cols].to_string(index=False))

    if not duplicate_lab_codes.empty:
        print("\nWARNING: Duplicate lab SampleCodes:")
        print(duplicate_lab_codes[["ID", "SampleCode"]].to_string(index=False))

    if not duplicate_image_codes.empty:
        print("\nWARNING: Duplicate image SampleCodes:")
        cols = _available_columns(
            duplicate_image_codes,
            ["image", "SampleCode", "sample_code_base", "sample_code_source"]
        )
        print(duplicate_image_codes[cols].to_string(index=False))

    # Save Excel with several sheets
    output_dir = os.path.dirname(output_xlsx)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        enriched.to_excel(writer, sheet_name="enriched", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        lab_without_image.to_excel(writer, sheet_name="lab_without_image", index=False)
        image_without_lab.to_excel(writer, sheet_name="image_without_lab", index=False)
        duplicate_lab_codes.to_excel(writer, sheet_name="duplicate_lab_codes", index=False)
        duplicate_image_codes.to_excel(writer, sheet_name="duplicate_image_codes", index=False)

    # Also save CSV reports next to the Excel
    base, _ = os.path.splitext(output_xlsx)

    enriched.to_csv(f"{base}_enriched.csv", index=False)
    lab_without_image.to_csv(f"{base}_lab_without_image.csv", index=False)
    image_without_lab.to_csv(f"{base}_image_without_lab.csv", index=False)
    summary.to_csv(f"{base}_summary.csv", index=False)

    print(f"\nSaved enriched Excel file: {output_xlsx}")
    print(f"Saved CSV reports with prefix: {base}_*.csv")

    return enriched