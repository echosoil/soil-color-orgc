#!/usr/bin/env python3

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colour import XYZ_to_Lab
from colour.difference import delta_E

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    RepeatedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


MUNSELL_HUE_FAMILIES = [
    "R",
    "YR",
    "Y",
    "GY",
    "G",
    "BG",
    "B",
    "PB",
    "P",
    "RP",
]


def read_table(path: str) -> pd.DataFrame:
    extension = Path(path).suffix.lower()

    if extension in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)

        if "enriched" in workbook.sheet_names:
            return pd.read_excel(path, sheet_name="enriched")

        return pd.read_excel(path, sheet_name=workbook.sheet_names[0])

    if extension == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {extension}")


def base_code_from_id(value):
    if pd.isna(value):
        return None

    match = re.match(r"\s*([A-Za-z]{4})", str(value))
    return match.group(1).upper() if match else None


def prepare_dataset(
    path: str,
    target_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    df = read_table(path).copy()

    if target_column not in df.columns:
        raise ValueError(
            f"{dataset_name}: target column {target_column!r} was not found."
        )

    if "SampleCode" not in df.columns:
        if "ID" not in df.columns:
            raise ValueError(
                f"{dataset_name}: neither SampleCode nor ID exists."
            )

        df["SampleCode"] = df["ID"].apply(base_code_from_id)

    df["SampleCode"] = (
        df["SampleCode"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if "processing_status" in df.columns:
        status = (
            df["processing_status"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        df = df[status.eq("ok")].copy()

    numeric_candidates = [
        target_column,
        "L",
        "a",
        "b",
        "SOC_est%",
        "deltaE2000",
    ]

    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    required = [
        "SampleCode",
        target_column,
        "L",
        "a",
        "b",
    ]

    before = len(df)
    df = df.dropna(subset=required).copy()

    print(
        f"{dataset_name}: retained {len(df)} of {before} rows "
        f"after requiring SampleCode, {target_column}, L, a and b."
    )

    duplicates = df[
        df["SampleCode"].duplicated(keep=False)
    ].sort_values("SampleCode")

    if not duplicates.empty:
        duplicate_codes = sorted(
            duplicates["SampleCode"].dropna().unique()
        )

        raise ValueError(
            f"{dataset_name}: duplicate SampleCodes found: "
            f"{duplicate_codes[:20]}. "
            "Each soil sample must occur only once for this experiment."
        )

    return df


def parse_munsell_hue_angle(hue) -> float:
    """
    Convert a hue such as 10YR or 2.5Y into a circular angle.

    There are 40 hue positions:
        2.5R, 5R, 7.5R, 10R,
        2.5YR, ..., 10RP.
    """
    if pd.isna(hue):
        return np.nan

    text = str(hue).strip().upper()

    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)([A-Z]+)",
        text,
    )

    if not match:
        return np.nan

    hue_number = float(match.group(1))
    family = match.group(2)

    if family not in MUNSELL_HUE_FAMILIES:
        return np.nan

    family_index = MUNSELL_HUE_FAMILIES.index(family)

    within_family = (hue_number / 2.5) - 1.0
    circular_position = family_index * 4.0 + within_family

    return 2.0 * np.pi * circular_position / 40.0


def load_munsell_reference(munsell_csv: str):
    df = pd.read_csv(munsell_csv)

    required = {
        "h",
        "V",
        "C",
        "X_D65",
        "Y_D65",
        "Z_D65",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Munsell CSV is missing columns: {sorted(missing)}"
        )

    xyz = df[
        ["X_D65", "Y_D65", "Z_D65"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(xyz).all(axis=1)

    df = df.loc[valid].reset_index(drop=True)
    xyz = xyz[valid]

    # Support either 0..1 or 0..100 XYZ scales.
    if np.nanmax(xyz) > 2.0:
        xyz = xyz / 100.0

    lab = XYZ_to_Lab(xyz)

    value = pd.to_numeric(df["V"], errors="coerce").to_numpy(dtype=float)
    chroma = pd.to_numeric(df["C"], errors="coerce").to_numpy(dtype=float)

    hue_angle = np.array(
        [parse_munsell_hue_angle(x) for x in df["h"]],
        dtype=float,
    )

    notation = (
        df["h"].astype(str)
        + " "
        + df["V"].astype(str)
        + "/"
        + df["C"].astype(str)
    ).to_numpy()

    valid = (
        np.isfinite(lab).all(axis=1)
        & np.isfinite(value)
        & np.isfinite(chroma)
    )

    return {
        "lab": np.asarray(lab[valid], dtype=float),
        "value": value[valid],
        "chroma": chroma[valid],
        "hue_angle": hue_angle[valid],
        "notation": notation[valid],
    }


def interpolate_one_munsell(
    sample_lab,
    reference,
    neighbours=6,
    power=2.0,
):
    """
    Interpolate Munsell properties using inverse-DeltaE weighting.

    This does not invent a new official printed Munsell chip. It creates
    continuous numerical features for model training.
    """
    reference_lab = reference["lab"]

    sample_array = np.broadcast_to(
        np.asarray(sample_lab, dtype=float),
        reference_lab.shape,
    )

    distances = np.asarray(
        delta_E(
            sample_array,
            reference_lab,
            method="CIE 2000",
        ),
        dtype=float,
    ).reshape(-1)

    neighbours = min(neighbours, len(distances))

    indexes = np.argpartition(
        distances,
        neighbours - 1,
    )[:neighbours]

    indexes = indexes[
        np.argsort(distances[indexes])
    ]

    local_distances = distances[indexes]

    weights = 1.0 / np.power(
        local_distances + 1e-6,
        power,
    )

    weights = weights / weights.sum()

    interpolated_value = float(
        np.sum(weights * reference["value"][indexes])
    )

    interpolated_chroma = float(
        np.sum(weights * reference["chroma"][indexes])
    )

    local_angles = reference["hue_angle"][indexes]
    valid_hue = np.isfinite(local_angles)

    if valid_hue.any():
        hue_weights = weights[valid_hue]
        hue_weights = hue_weights / hue_weights.sum()

        hue_sin = float(
            np.sum(
                hue_weights
                * np.sin(local_angles[valid_hue])
            )
        )

        hue_cos = float(
            np.sum(
                hue_weights
                * np.cos(local_angles[valid_hue])
            )
        )
    else:
        hue_sin = np.nan
        hue_cos = np.nan

    return {
        "munsell_value_interp": interpolated_value,
        "munsell_chroma_interp": interpolated_chroma,
        "munsell_hue_sin": hue_sin,
        "munsell_hue_cos": hue_cos,
        "munsell_nearest_deltaE": float(local_distances[0]),
        "munsell_weighted_deltaE": float(
            np.sum(weights * local_distances)
        ),
        "munsell_nearest_notation": str(
            reference["notation"][indexes[0]]
        ),
    }


def add_interpolated_munsell_features(
    df: pd.DataFrame,
    reference,
    neighbours=6,
) -> pd.DataFrame:
    df = df.copy()

    rows = []

    for sample_lab in df[["L", "a", "b"]].to_numpy(dtype=float):
        rows.append(
            interpolate_one_munsell(
                sample_lab,
                reference,
                neighbours=neighbours,
            )
        )

    feature_df = pd.DataFrame(rows, index=df.index)

    for column in feature_df.columns:
        df[column] = feature_df[column]

    return df


def make_stratification_bins(target: pd.Series):
    """
    Create approximate target strata for the final train/test split.
    """
    for number_of_bins in range(5, 1, -1):
        try:
            bins = pd.qcut(
                target,
                q=number_of_bins,
                duplicates="drop",
            )

            counts = bins.value_counts()

            if len(counts) >= 2 and counts.min() >= 2:
                return bins.astype(str)

        except Exception:
            continue

    return None


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_pred - y_true

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        pearson = float(
            np.corrcoef(y_true, y_pred)[0, 1]
        )
    else:
        pearson = np.nan

    return {
        "n": len(y_true),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(
            mean_squared_error(y_true, y_pred) ** 0.5
        ),
        "bias_pred_minus_lab": float(np.mean(residuals)),
        "R2": float(r2_score(y_true, y_pred)),
        "pearson_r": pearson,
    }


def evaluate_repeated_cv(
    estimator,
    X,
    y,
    number_of_repeats,
    seed,
):
    splitter = RepeatedKFold(
        n_splits=5,
        n_repeats=number_of_repeats,
        random_state=seed,
    )

    rows = []

    for fold_number, (train_index, valid_index) in enumerate(
        splitter.split(X),
        start=1,
    ):
        fitted = clone(estimator)

        fitted.fit(
            X.iloc[train_index],
            y.iloc[train_index],
        )

        predictions = fitted.predict(
            X.iloc[valid_index]
        )

        metrics = calculate_metrics(
            y.iloc[valid_index],
            predictions,
        )

        metrics["fold"] = fold_number
        rows.append(metrics)

    return pd.DataFrame(rows)


def available_qc_features(df: pd.DataFrame):
    candidates = [
        "mean_imbalance_before",
        "mean_imbalance_after",
        "imbalance_before",
        "imbalance_after",
        "correction_magnitude",
        "gray_patch_fit_error",
        "grey_patch_fit_error",
        "gray_calibration_error",
        "grey_calibration_error",
        "gray_scale_detection_score",
        "grey_scale_detection_score",
        "deltaE2000",
    ]

    result = []

    for column in candidates:
        if column in df.columns:
            numeric = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if numeric.notna().sum() >= 10:
                df[column] = numeric
                result.append(column)

    return result


def create_models(df: pd.DataFrame):
    ridge_alphas = np.logspace(-3, 3, 25)

    munsell_features = [
        "munsell_value_interp",
        "munsell_chroma_interp",
        "munsell_hue_sin",
        "munsell_hue_cos",
        "munsell_nearest_deltaE",
        "munsell_weighted_deltaE",
    ]

    qc_features = available_qc_features(df)

    gradient_features = [
        "L",
        "a",
        "b",
        *munsell_features,
        *qc_features,
    ]

    print(
        "Gradient boosting QC features:",
        qc_features if qc_features else "none available",
    )

    return {
        "baseline_mean": {
            "features": ["L"],
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="mean")),
            ]),
        },
        "model_1_L": {
            "features": ["L"],
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression()),
            ]),
        },
        "model_2_Lab": {
            "features": ["L", "a", "b"],
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression()),
            ]),
        },
        "model_3_Lab_polynomial_ridge": {
            "features": ["L", "a", "b"],
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "polynomial",
                    PolynomialFeatures(
                        degree=2,
                        include_bias=False,
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "model",
                    RidgeCV(
                        alphas=ridge_alphas,
                        cv=5,
                    ),
                ),
            ]),
        },
        "model_4_Munsell_interpolated": {
            "features": munsell_features,
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    RidgeCV(
                        alphas=ridge_alphas,
                        cv=5,
                    ),
                ),
            ]),
        },
        "model_5_gradient_boosting": {
            "features": gradient_features,
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=8,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]),
        },
    }


def extract_selected_alpha(fitted_estimator):
    if not isinstance(fitted_estimator, Pipeline):
        return None

    model = fitted_estimator.named_steps.get("model")

    if model is not None and hasattr(model, "alpha_"):
        return float(model.alpha_)

    return None


def safe_filename(text):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text),
    )


def save_test_plot(
    y_true,
    y_pred,
    dataset_name,
    model_name,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))

    padding = max((upper - lower) * 0.05, 0.1)
    limits = [lower - padding, upper + padding]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred)
    plt.plot(limits, limits, linestyle="--")
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xlabel("Laboratory orgC")
    plt.ylabel("Predicted orgC")
    plt.title(f"{dataset_name}: {model_name}")
    plt.tight_layout()

    filename = (
        f"{safe_filename(dataset_name)}__"
        f"{safe_filename(model_name)}__test.png"
    )

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=160,
    )

    plt.close()


def run_models_for_dataset(
    dataset_name,
    df,
    train_codes,
    test_codes,
    target_column,
    repeats,
    seed,
    output_dir,
):
    models = create_models(df)

    train_df = df[
        df["SampleCode"].isin(train_codes)
    ].copy()

    test_df = df[
        df["SampleCode"].isin(test_codes)
    ].copy()

    summary_rows = []
    prediction_rows = []

    model_dir = os.path.join(
        output_dir,
        "models",
    )

    plot_dir = os.path.join(
        output_dir,
        "plots",
    )

    os.makedirs(model_dir, exist_ok=True)

    print(
        f"\n{dataset_name}: "
        f"train={len(train_df)}, test={len(test_df)}"
    )

    for model_name, configuration in models.items():
        features = configuration["features"]
        estimator = configuration["estimator"]

        missing_features = [
            column
            for column in features
            if column not in df.columns
        ]

        if missing_features:
            print(
                f"Skipping {dataset_name}/{model_name}: "
                f"missing {missing_features}"
            )
            continue

        print(
            f"\nRunning {dataset_name}/{model_name}: "
            f"{features}"
        )

        X_train = train_df[features]
        y_train = train_df[target_column]

        X_test = test_df[features]
        y_test = test_df[target_column]

        cv_results = evaluate_repeated_cv(
            estimator=estimator,
            X=X_train,
            y=y_train,
            number_of_repeats=repeats,
            seed=seed,
        )

        fitted = clone(estimator)
        fitted.fit(X_train, y_train)

        test_predictions = fitted.predict(X_test)
        test_metrics = calculate_metrics(
            y_test,
            test_predictions,
        )

        summary_rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "features": " + ".join(features),
            "train_n": len(train_df),
            "test_n": len(test_df),
            "cv_repeats": repeats,
            "cv_MAE_mean": cv_results["MAE"].mean(),
            "cv_MAE_sd": cv_results["MAE"].std(ddof=1),
            "cv_RMSE_mean": cv_results["RMSE"].mean(),
            "cv_RMSE_sd": cv_results["RMSE"].std(ddof=1),
            "cv_R2_mean": cv_results["R2"].mean(),
            "cv_R2_sd": cv_results["R2"].std(ddof=1),
            "cv_pearson_mean": cv_results["pearson_r"].mean(),
            "cv_pearson_sd": cv_results["pearson_r"].std(ddof=1),
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_bias_pred_minus_lab": (
                test_metrics["bias_pred_minus_lab"]
            ),
            "test_R2": test_metrics["R2"],
            "test_pearson_r": test_metrics["pearson_r"],
            "selected_ridge_alpha": extract_selected_alpha(fitted),
        })

        for row_index, prediction in zip(
            test_df.index,
            test_predictions,
        ):
            source = test_df.loc[row_index]
            laboratory_value = float(
                source[target_column]
            )

            prediction_rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "SampleCode": source.get("SampleCode"),
                "ID": source.get("ID"),
                "image": source.get("image"),
                "lab_value": laboratory_value,
                "predicted_value": float(prediction),
                "residual_pred_minus_lab": float(
                    prediction - laboratory_value
                ),
            })

        model_path = os.path.join(
            model_dir,
            f"{safe_filename(dataset_name)}__"
            f"{safe_filename(model_name)}.joblib",
        )

        joblib.dump(
            {
                "dataset": dataset_name,
                "model_name": model_name,
                "features": features,
                "target": target_column,
                "estimator": fitted,
            },
            model_path,
        )

        save_test_plot(
            y_true=y_test,
            y_pred=test_predictions,
            dataset_name=dataset_name,
            model_name=model_name,
            output_dir=plot_dir,
        )

    return summary_rows, prediction_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare soil-orgC calibration models using corrected "
            "and uncorrected image features."
        )
    )

    parser.add_argument(
        "--with-gray",
        required=True,
        help="Enriched dataset produced using grey-scale correction.",
    )

    parser.add_argument(
        "--no-gray",
        default=None,
        help=(
            "Optional enriched dataset produced without grey-scale "
            "correction. When supplied, both datasets use the same split."
        ),
    )

    parser.add_argument(
        "--munsell",
        default="data/munsell/rit_munsell.csv",
    )

    parser.add_argument(
        "--target",
        default="orgC_lab",
    )

    parser.add_argument(
        "--test-count",
        type=int,
        default=51,
        help="Number of samples reserved for the final test.",
    )

    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=10,
        help="Number of repetitions of 5-fold CV.",
    )

    parser.add_argument(
        "--munsell-neighbours",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--out",
        default="outputs/model_experiment",
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading Munsell reference...")
    reference = load_munsell_reference(args.munsell)

    datasets = {}

    with_gray = prepare_dataset(
        args.with_gray,
        target_column=args.target,
        dataset_name="with_gray",
    )

    with_gray = add_interpolated_munsell_features(
        with_gray,
        reference,
        neighbours=args.munsell_neighbours,
    )

    datasets["with_gray"] = with_gray

    if args.no_gray:
        no_gray = prepare_dataset(
            args.no_gray,
            target_column=args.target,
            dataset_name="no_gray",
        )

        no_gray = add_interpolated_munsell_features(
            no_gray,
            reference,
            neighbours=args.munsell_neighbours,
        )

        datasets["no_gray"] = no_gray

    common_codes = set(with_gray["SampleCode"])

    for dataset in datasets.values():
        common_codes &= set(dataset["SampleCode"])

    common_codes = sorted(common_codes)

    if len(common_codes) <= args.test_count:
        raise ValueError(
            f"Only {len(common_codes)} common usable samples exist, "
            f"which is not enough for test-count={args.test_count}."
        )

    print(
        f"\nCommon usable independent samples: {len(common_codes)}"
    )

    split_reference = (
        with_gray.set_index("SampleCode")
        .loc[common_codes]
        .reset_index()
    )

    stratification = make_stratification_bins(
        split_reference[args.target]
    )

    train_codes, test_codes = train_test_split(
        common_codes,
        test_size=args.test_count,
        random_state=args.seed,
        shuffle=True,
        stratify=stratification,
    )

    train_codes = sorted(train_codes)
    test_codes = sorted(test_codes)

    split_rows = []

    for code in common_codes:
        source = split_reference[
            split_reference["SampleCode"] == code
        ].iloc[0]

        split_rows.append({
            "SampleCode": code,
            "ID": source.get("ID"),
            "orgC_lab": source[args.target],
            "split": (
                "test"
                if code in set(test_codes)
                else "train"
            ),
        })

    split_df = pd.DataFrame(split_rows)

    print(
        f"Shared split: train={len(train_codes)}, "
        f"test={len(test_codes)}"
    )

    all_summary = []
    all_predictions = []

    for dataset_name, dataset in datasets.items():
        summary_rows, prediction_rows = run_models_for_dataset(
            dataset_name=dataset_name,
            df=dataset,
            train_codes=train_codes,
            test_codes=test_codes,
            target_column=args.target,
            repeats=args.cv_repeats,
            seed=args.seed,
            output_dir=args.out,
        )

        all_summary.extend(summary_rows)
        all_predictions.extend(prediction_rows)

    summary_df = pd.DataFrame(all_summary)
    predictions_df = pd.DataFrame(all_predictions)

    summary_df = summary_df.sort_values(
        ["dataset", "test_RMSE", "test_MAE"]
    )

    summary_path = os.path.join(
        args.out,
        "model_experiment_summary.csv",
    )

    predictions_path = os.path.join(
        args.out,
        "model_experiment_predictions.csv",
    )

    split_path = os.path.join(
        args.out,
        "model_experiment_split.csv",
    )

    features_path = os.path.join(
        args.out,
        "model_experiment_features.csv",
    )

    workbook_path = os.path.join(
        args.out,
        "model_experiment_results.xlsx",
    )

    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    split_df.to_csv(split_path, index=False)

    feature_exports = []

    for dataset_name, dataset in datasets.items():
        exported = dataset.copy()
        exported.insert(0, "dataset", dataset_name)
        feature_exports.append(exported)

    all_features_df = pd.concat(
        feature_exports,
        ignore_index=True,
    )

    all_features_df.to_csv(features_path, index=False)

    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
    ) as writer:
        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        predictions_df.to_excel(
            writer,
            sheet_name="test_predictions",
            index=False,
        )

        split_df.to_excel(
            writer,
            sheet_name="shared_split",
            index=False,
        )

        all_features_df.to_excel(
            writer,
            sheet_name="features",
            index=False,
        )

    print("\n=== Final-test ranking ===")

    columns = [
        "dataset",
        "model",
        "test_n",
        "test_MAE",
        "test_RMSE",
        "test_bias_pred_minus_lab",
        "test_R2",
        "test_pearson_r",
        "cv_RMSE_mean",
        "cv_RMSE_sd",
    ]

    print(summary_df[columns].to_string(index=False))

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {split_path}")
    print(f"Saved: {features_path}")
    print(f"Saved: {workbook_path}")
    print(f"Saved fitted models under: {args.out}/models")
    print(f"Saved plots under: {args.out}/plots")


if __name__ == "__main__":
    main()