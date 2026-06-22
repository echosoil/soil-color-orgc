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

    lab["SampleCode"] = lab["ID"].apply(base_code_from_lab)
    predictions["SampleCode"] = predictions["image"].apply(base_code_from_image)

    prediction_columns = [
        "SampleCode",
        "image",
        "L",
        "a",
        "b",
        "best_munsell",
        "deltaE2000",
        "SOC_est%",
        "SOC_method",
    ]

    missing_prediction_columns = [
        col for col in prediction_columns
        if col not in predictions.columns
    ]

    if missing_prediction_columns:
        raise ValueError(
            f"Predictions CSV is missing columns: {missing_prediction_columns}"
        )

    enriched = lab.merge(
        predictions[prediction_columns],
        on="SampleCode",
        how="left",
        suffixes=("", "_pred"),
    )

    missing = enriched[enriched["image"].isna()][["ID", "SampleCode"]]

    if not missing.empty:
        print("WARNING: No prediction found for these lab rows:")
        print(missing.to_string(index=False))

    dupes = predictions["SampleCode"].value_counts()
    dupes = dupes[dupes > 1]

    if not dupes.empty:
        print("WARNING: Multiple images share the same SampleCode:")
        print(dupes)

    output_dir = os.path.dirname(output_xlsx)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    enriched.to_excel(output_xlsx, index=False)

    print(f"Saved enriched file: {output_xlsx}")

    return enriched