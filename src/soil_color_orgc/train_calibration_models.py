#!/usr/bin/env python3
""" Example usage:
# --------------
 python3 scripts/train_calibration_models.py \
  --input outputs/test_stat_orgC_enriched.xlsx \
  --target orgC_lab \
  --test-size 19 \
  --seed 42
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_input_table(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        # Your enriched workbook should have this sheet.
        try:
            return pd.read_excel(path, sheet_name="enriched")
        except Exception:
            return pd.read_excel(path)

    if ext == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported input file type: {ext}")


def parse_munsell_value_chroma(best_munsell):
    """
    Examples:
        10YR 4/4   -> value=4, chroma=4
        2.5Y 4/2   -> value=4, chroma=2
        7.5YR 3/6  -> value=3, chroma=6
    """
    if pd.isna(best_munsell):
        return None, None

    text = str(best_munsell).strip()

    match = re.search(r"\s([0-9.]+)\s*/\s*([0-9.]+)", text)

    if not match:
        return None, None

    value = float(match.group(1))
    chroma = float(match.group(2))

    return value, chroma


def prepare_data(df: pd.DataFrame, target_col: str):
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    required_possible = [
        target_col,
        "SOC_est%",
        "L",
        "a",
        "b",
        "best_munsell",
        "SampleCode",
        "ID",
        "image",
        "processing_status",
    ]

    print("\nAvailable useful columns:")
    for col in required_possible:
        if col in df.columns:
            print(f"  OK: {col}")
        else:
            print(f"  missing: {col}")

    # Keep only successfully processed image rows if this column exists.
    if "processing_status" in df.columns:
        df = df[df["processing_status"].fillna("").astype(str).str.lower().eq("ok")].copy()

    # Convert numeric columns.
    numeric_cols = [target_col, "SOC_est%", "L", "a", "b"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse Munsell value/chroma.
    if "best_munsell" in df.columns:
        parsed = df["best_munsell"].apply(parse_munsell_value_chroma)
        df["munsell_value"] = parsed.apply(lambda x: x[0])
        df["munsell_chroma"] = parsed.apply(lambda x: x[1])
    else:
        df["munsell_value"] = np.nan
        df["munsell_chroma"] = np.nan

    # Remove rows without lab target.
    df = df[df[target_col].notna()].copy()

    return df


def make_stratify_bins(y, max_bins=5):
    """
    Stratify continuous target approximately by quantile bins.

    This helps avoid a split where all high-orgC samples end up only in train or only in test.
    """
    y = pd.Series(y)

    for bins in range(max_bins, 1, -1):
        try:
            q = pd.qcut(y, q=bins, duplicates="drop")
            counts = q.value_counts()

            if len(counts) >= 2 and counts.min() >= 2:
                return q.astype(str)

        except Exception:
            continue

    return None


def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual = y_pred - y_true

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    bias = float(np.mean(residual))
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = np.nan

    return {
        "n": len(y_true),
        "MAE": mae,
        "RMSE": rmse,
        "bias_pred_minus_lab": bias,
        "R2": r2,
        "pearson_r": pearson_r,
    }


def fit_one_model(model_name, feature_cols, train_df, test_df, target_col):
    missing = [col for col in feature_cols if col not in train_df.columns]

    if missing:
        print(f"Skipping {model_name}: missing columns {missing}")
        return None, None, None

    train_model_df = train_df.dropna(subset=[target_col] + feature_cols).copy()
    test_model_df = test_df.dropna(subset=[target_col] + feature_cols).copy()

    if len(train_model_df) < len(feature_cols) + 2:
        print(f"Skipping {model_name}: not enough train rows after dropping missing values")
        return None, None, None

    if len(test_model_df) < 2:
        print(f"Skipping {model_name}: not enough test rows after dropping missing values")
        return None, None, None

    X_train = train_model_df[feature_cols]
    y_train = train_model_df[target_col]

    X_test = test_model_df[feature_cols]
    y_test = test_model_df[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = evaluate_predictions(y_train, train_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)

    summary_rows = []

    for split_name, metrics in [("train", train_metrics), ("test", test_metrics)]:
        row = {
            "model": model_name,
            "split": split_name,
            "features": " + ".join(feature_cols),
            "intercept": float(model.intercept_),
        }
        row.update(metrics)
        summary_rows.append(row)

    coef_rows = []

    coef_rows.append({
        "model": model_name,
        "term": "intercept",
        "coefficient": float(model.intercept_),
    })

    for feature, coef in zip(feature_cols, model.coef_):
        coef_rows.append({
            "model": model_name,
            "term": feature,
            "coefficient": float(coef),
        })

    prediction_rows = []

    for split_name, part_df, pred in [
        ("train", train_model_df, train_pred),
        ("test", test_model_df, test_pred),
    ]:
        for idx, yhat in zip(part_df.index, pred):
            source_row = part_df.loc[idx]

            prediction_rows.append({
                "model": model_name,
                "split": split_name,
                "row_index": idx,
                "ID": source_row.get("ID"),
                "SampleCode": source_row.get("SampleCode"),
                "image": source_row.get("image"),
                "lab_value": source_row[target_col],
                "predicted_lab_value": float(yhat),
                "residual_pred_minus_lab": float(yhat - source_row[target_col]),
                **{col: source_row.get(col) for col in feature_cols},
            })

    return summary_rows, coef_rows, prediction_rows


def main():
    parser = argparse.ArgumentParser(
        description="Train simple calibration models from image-derived soil color features."
    )

    parser.add_argument(
        "--input",
        default="outputs/test_stat_orgC_enriched.xlsx",
        help="Input enriched Excel/CSV file.",
    )

    parser.add_argument(
        "--target",
        default="orgC_lab",
        help="Target lab column to predict, usually orgC_lab.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/calibration",
        help="Output folder.",
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=19,
        help="Number of samples to hold out for testing.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading: {args.input}")
    df_raw = read_input_table(args.input)
    df = prepare_data(df_raw, args.target)

    print(f"\nRows available after filtering valid lab target and processed images: {len(df)}")

    if len(df) < 30:
        raise ValueError("Too few valid rows for calibration. Check matching and missing values.")

    if len(df) <= args.test_size:
        raise ValueError(
            f"Not enough rows ({len(df)}) for test size {args.test_size}."
        )

    train_size = len(df) - args.test_size

    print(f"Requested split: train={train_size}, test={args.test_size}")

    stratify_bins = make_stratify_bins(df[args.target])

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify_bins,
    )

    print(f"Actual split: train={len(train_df)}, test={len(test_df)}")

    split_table = df.copy()
    split_table["split"] = "unused"
    split_table.loc[train_df.index, "split"] = "train"
    split_table.loc[test_df.index, "split"] = "test"

    models = {
        "model_1_SOC_est": ["SOC_est%"],
        "model_2_L_only": ["L"],
        "model_3_Lab": ["L", "a", "b"],
        "model_4_Munsell_value_chroma": ["munsell_value", "munsell_chroma"],
    }

    all_summary = []
    all_coefficients = []
    all_predictions = []

    for model_name, feature_cols in models.items():
        print(f"\nTraining {model_name}: {feature_cols}")

        summary_rows, coef_rows, prediction_rows = fit_one_model(
            model_name=model_name,
            feature_cols=feature_cols,
            train_df=train_df,
            test_df=test_df,
            target_col=args.target,
        )

        if summary_rows is None:
            continue

        all_summary.extend(summary_rows)
        all_coefficients.extend(coef_rows)
        all_predictions.extend(prediction_rows)

    summary_df = pd.DataFrame(all_summary)
    coefficients_df = pd.DataFrame(all_coefficients)
    predictions_df = pd.DataFrame(all_predictions)

    summary_csv = os.path.join(args.output_dir, "calibration_summary.csv")
    coefficients_csv = os.path.join(args.output_dir, "calibration_coefficients.csv")
    predictions_csv = os.path.join(args.output_dir, "calibration_predictions.csv")
    split_csv = os.path.join(args.output_dir, "train_test_split.csv")
    output_xlsx = os.path.join(args.output_dir, "calibration_results.xlsx")

    summary_df.to_csv(summary_csv, index=False)
    coefficients_df.to_csv(coefficients_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)
    split_table.to_csv(split_csv, index=False)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        coefficients_df.to_excel(writer, sheet_name="coefficients", index=False)
        predictions_df.to_excel(writer, sheet_name="predictions", index=False)
        split_table.to_excel(writer, sheet_name="train_test_split", index=False)

    print("\n=== Test-set results ===")
    if not summary_df.empty:
        test_summary = summary_df[summary_df["split"] == "test"].copy()
        print(
            test_summary[
                ["model", "n", "MAE", "RMSE", "bias_pred_minus_lab", "R2", "pearson_r"]
            ].to_string(index=False)
        )

    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {coefficients_csv}")
    print(f"Saved: {predictions_csv}")
    print(f"Saved: {split_csv}")
    print(f"Saved: {output_xlsx}")


if __name__ == "__main__":
    main()