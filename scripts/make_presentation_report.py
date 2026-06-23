#!/usr/bin/env python3
"""
Generate an automated static HTML report comparing:

1. Laboratory orgC vs citizen-science orgC estimates.
2. Laboratory orgC vs image algorithm without grey-scale correction, using
   cross-validated training.
3. Laboratory orgC vs image algorithm with grey-scale correction, without
   training, because the current grey-scale subset is still too small.

Example
-------
python3 scripts/make_presentation_report.py \
  --lab data/lab/test_stat_orgC.xlsx \
  --no-gray outputs/test_stat_orgC_enriched_no_gray.xlsx \
  --with-gray outputs/test_stat_orgC_enriched_with_gray.xlsx \
  --out outputs/presentation_report
"""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ComparisonResult:
    key: str
    title: str
    subtitle: str
    df: pd.DataFrame
    y_true_col: str
    y_pred_col: str
    method_note: str
    metrics: dict
    figures: dict
    interpretation: list[str]


def fmt(value, digits: int = 3) -> str:
    """Format a numeric value for display."""
    if value is None:
        return "NA"

    try:
        value = float(value)
    except Exception:
        return "NA"

    if not np.isfinite(value):
        return "NA"

    return f"{value:.{digits}f}"


def fmt_p(value) -> str:
    """Format a p-value for display."""
    if value is None:
        return "NA"

    try:
        value = float(value)
    except Exception:
        return "NA"

    if not np.isfinite(value):
        return "NA"

    if value < 0.001:
        return "<0.001"

    return f"{value:.3f}"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "section"


def read_table(path: str | Path, preferred_sheet: str = "enriched") -> pd.DataFrame:
    """Read CSV/XLS/XLSX, preferring a sheet named 'enriched' when present."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        sheet = preferred_sheet if preferred_sheet in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def filter_processing_ok(df: pd.DataFrame) -> pd.DataFrame:
    """Keep successful rows when a processing_status column exists."""
    df = df.copy()

    if "processing_status" not in df.columns:
        return df

    status = df["processing_status"].astype(str).str.lower().str.strip()
    keep = status.isin(["ok", "nan", "", "none"])

    return df.loc[keep].copy()


def clean_pair_df(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Keep identifier columns and rows with valid numeric true/predicted values."""
    df = df.copy()

    required = [y_true_col, y_pred_col]
    if extra_cols:
        required += extra_cols

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = []

    for candidate in ["ID", "SampleCode", "sample_code_base", "image"]:
        if candidate in df.columns:
            keep_cols.append(candidate)

    keep_cols += required
    keep_cols = list(dict.fromkeys(keep_cols))

    return df[keep_cols].dropna(subset=[y_true_col, y_pred_col]).copy()


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    if len(y_true) < 2:
        return np.nan

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true, ddof=1)
    var_pred = np.var(y_pred, ddof=1)
    covariance = np.cov(y_true, y_pred, ddof=1)[0, 1]

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    if denominator == 0:
        return np.nan

    return float((2 * covariance) / denominator)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = len(y_true)

    out = {
        "n": n,
        "lab_mean": np.nan,
        "lab_sd": np.nan,
        "estimate_mean": np.nan,
        "estimate_sd": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "CCC_agreement": np.nan,
        "bias_mean_estimate_minus_lab": np.nan,
        "median_error_estimate_minus_lab": np.nan,
        "MAE": np.nan,
        "median_absolute_error": np.nan,
        "RMSE": np.nan,
        "R2_direct_prediction": np.nan,
        "bland_altman_lower_95": np.nan,
        "bland_altman_upper_95": np.nan,
        "calibration_intercept_lab_from_estimate": np.nan,
        "calibration_slope_lab_from_estimate": np.nan,
        "R2_calibration_regression": np.nan,
    }

    if n == 0:
        return out

    residual = y_pred - y_true

    out["lab_mean"] = float(np.mean(y_true))
    out["estimate_mean"] = float(np.mean(y_pred))
    out["bias_mean_estimate_minus_lab"] = float(np.mean(residual))
    out["median_error_estimate_minus_lab"] = float(np.median(residual))
    out["MAE"] = float(mean_absolute_error(y_true, y_pred))
    out["median_absolute_error"] = float(np.median(np.abs(residual)))
    out["RMSE"] = float(math.sqrt(mean_squared_error(y_true, y_pred)))

    if n >= 2:
        out["lab_sd"] = float(np.std(y_true, ddof=1))
        out["estimate_sd"] = float(np.std(y_pred, ddof=1))
        out["CCC_agreement"] = compute_ccc(y_true, y_pred)

        try:
            out["R2_direct_prediction"] = float(r2_score(y_true, y_pred))
        except Exception:
            pass

        residual_sd = float(np.std(residual, ddof=1))
        bias = out["bias_mean_estimate_minus_lab"]

        out["bland_altman_lower_95"] = float(bias - 1.96 * residual_sd)
        out["bland_altman_upper_95"] = float(bias + 1.96 * residual_sd)

    if n >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        try:
            pearson = stats.pearsonr(y_true, y_pred)
            out["pearson_r"] = float(pearson.statistic)
            out["pearson_p"] = float(pearson.pvalue)
        except Exception:
            pass

        try:
            spearman = stats.spearmanr(y_true, y_pred)
            out["spearman_r"] = float(spearman.statistic)
            out["spearman_p"] = float(spearman.pvalue)
        except Exception:
            pass

        try:
            model = LinearRegression()
            model.fit(y_pred.reshape(-1, 1), y_true)
            calibrated = model.predict(y_pred.reshape(-1, 1))

            out["calibration_intercept_lab_from_estimate"] = float(model.intercept_)
            out["calibration_slope_lab_from_estimate"] = float(model.coef_[0])
            out["R2_calibration_regression"] = float(r2_score(y_true, calibrated))
        except Exception:
            pass

    return out


def choose_cv(n: int):
    if n < 3:
        raise ValueError("At least 3 samples are required for cross-validation.")

    if n <= 50:
        return LeaveOneOut()

    return KFold(n_splits=5, shuffle=True, random_state=42)


def train_cv_predictions(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    prediction_col: str = "predicted_orgC_cv",
) -> tuple[pd.DataFrame, dict]:
    """Train a simple calibrated model and return cross-validated predictions."""
    df = df.copy()

    for col in [target_col] + feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[target_col] + feature_cols).copy()

    if len(df) < 3:
        raise ValueError(f"Not enough rows for training. Need at least 3, got {len(df)}.")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regression", LinearRegression()),
        ]
    )

    cv = choose_cv(len(df))
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)

    df[prediction_col] = y_pred_cv

    # Fit final model on all data only for reporting coefficients.
    model.fit(X, y)

    regression = model.named_steps["regression"]
    scaler = model.named_steps["scaler"]

    coef_original_units = regression.coef_ / scaler.scale_
    intercept_original_units = regression.intercept_ - np.sum(
        regression.coef_ * scaler.mean_ / scaler.scale_
    )

    model_info = {
        "n_training_rows": len(df),
        "features": feature_cols,
        "cv": "LeaveOneOut" if len(df) <= 50 else "5-fold shuffled CV",
        "intercept_original_units": float(intercept_original_units),
        "coefficients_original_units": {
            col: float(coef) for col, coef in zip(feature_cols, coef_original_units)
        },
    }

    return df, model_info


def make_scatter_plot(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    title: str,
    subtitle: str,
    output_path: Path,
    metrics: dict,
) -> None:
    y_true = df[y_true_col].to_numpy(dtype=float)
    y_pred = df[y_pred_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.8)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)

    if finite.any():
        min_val = float(np.nanmin([np.min(y_true[finite]), np.min(y_pred[finite])]))
        max_val = float(np.nanmax([np.max(y_true[finite]), np.max(y_pred[finite])]))
        padding = (max_val - min_val) * 0.08 if max_val > min_val else 1.0
        min_axis = min_val - padding
        max_axis = max_val + padding

        ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--")
        ax.set_xlim(min_axis, max_axis)
        ax.set_ylim(min_axis, max_axis)

    label = (
        f"n = {metrics.get('n', 0)}\n"
        f"Pearson r = {fmt(metrics.get('pearson_r'))}\n"
        f"Spearman r = {fmt(metrics.get('spearman_r'))}\n"
        f"MAE = {fmt(metrics.get('MAE'))}\n"
        f"RMSE = {fmt(metrics.get('RMSE'))}\n"
        f"R² = {fmt(metrics.get('R2_direct_prediction'))}"
    )

    ax.text(
        0.04,
        0.96,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "alpha": 0.15},
    )

    ax.set_title(title)
    ax.set_xlabel("Laboratory orgC")
    ax.set_ylabel(subtitle)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_bland_altman_plot(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    title: str,
    output_path: Path,
    metrics: dict,
) -> None:
    y_true = df[y_true_col].to_numpy(dtype=float)
    y_pred = df[y_pred_col].to_numpy(dtype=float)

    mean_values = (y_true + y_pred) / 2.0
    residual = y_pred - y_true

    bias = metrics.get("bias_mean_estimate_minus_lab", np.nan)
    lower = metrics.get("bland_altman_lower_95", np.nan)
    upper = metrics.get("bland_altman_upper_95", np.nan)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(mean_values, residual, alpha=0.8)

    if np.isfinite(bias):
        ax.axhline(bias, linestyle="-", label=f"Bias = {fmt(bias)}")

    if np.isfinite(lower):
        ax.axhline(lower, linestyle="--", label=f"Lower 95% = {fmt(lower)}")

    if np.isfinite(upper):
        ax.axhline(upper, linestyle="--", label=f"Upper 95% = {fmt(upper)}")

    ax.axhline(0, linestyle=":", label="Zero error")
    ax.set_title(title)
    ax.set_xlabel("Mean of laboratory and estimate")
    ax.set_ylabel("Estimate minus laboratory")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_residual_plot(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    title: str,
    output_path: Path,
) -> None:
    y_true = df[y_true_col].to_numpy(dtype=float)
    y_pred = df[y_pred_col].to_numpy(dtype=float)
    residual = y_pred - y_true

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(y_true, residual, alpha=0.8)
    ax.axhline(0, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Laboratory orgC")
    ax.set_ylabel("Estimate minus laboratory")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_metrics_bar_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.copy()

    required = ["scenario", "MAE", "RMSE"]
    if any(c not in plot_df.columns for c in required):
        return

    scenarios = plot_df["scenario"].astype(str).tolist()
    x = np.arange(len(scenarios))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    width = 0.35
    mae = plot_df["MAE"].to_numpy(dtype=float)
    rmse = plot_df["RMSE"].to_numpy(dtype=float)

    ax.bar(x - width / 2, mae, width, label="MAE")
    ax.bar(x + width / 2, rmse, width, label="RMSE")

    ax.set_title("Error comparison across methods")
    ax.set_ylabel("Error")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_figures(result: ComparisonResult, figures_dir: Path) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)

    key = slugify(result.key)

    scatter_path = figures_dir / f"{key}_scatter.png"
    bland_path = figures_dir / f"{key}_bland_altman.png"
    residual_path = figures_dir / f"{key}_residual.png"

    make_scatter_plot(
        result.df,
        result.y_true_col,
        result.y_pred_col,
        title=f"{result.title}: laboratory vs estimate",
        subtitle=result.subtitle,
        output_path=scatter_path,
        metrics=result.metrics,
    )

    make_bland_altman_plot(
        result.df,
        result.y_true_col,
        result.y_pred_col,
        title=f"{result.title}: Bland-Altman agreement",
        output_path=bland_path,
        metrics=result.metrics,
    )

    make_residual_plot(
        result.df,
        result.y_true_col,
        result.y_pred_col,
        title=f"{result.title}: residual pattern",
        output_path=residual_path,
    )

    return {
        "scatter": scatter_path,
        "bland_altman": bland_path,
        "residual": residual_path,
    }


def interpret_metrics(metrics: dict, mode: str) -> list[str]:
    n = metrics.get("n", 0)
    pearson = metrics.get("pearson_r", np.nan)
    pearson_p = metrics.get("pearson_p", np.nan)
    spearman = metrics.get("spearman_r", np.nan)
    spearman_p = metrics.get("spearman_p", np.nan)
    bias = metrics.get("bias_mean_estimate_minus_lab", np.nan)
    mae = metrics.get("MAE", np.nan)
    rmse = metrics.get("RMSE", np.nan)
    r2 = metrics.get("R2_direct_prediction", np.nan)
    ccc = metrics.get("CCC_agreement", np.nan)
    lower = metrics.get("bland_altman_lower_95", np.nan)
    upper = metrics.get("bland_altman_upper_95", np.nan)

    comments: list[str] = []

    comments.append(f"This comparison contains n = {n} matched samples.")

    if n < 20:
        comments.append(
            "The sample size is small, so the result should be presented as preliminary "
            "and sensitive to individual outliers."
        )
    elif n < 50:
        comments.append(
            "The sample size is moderate for exploration, but still limited for final validation."
        )
    else:
        comments.append(
            "The sample size is large enough for a more stable exploratory validation, "
            "although independent validation is still recommended."
        )

    if np.isfinite(pearson):
        if abs(pearson) >= 0.7:
            strength = "strong"
        elif abs(pearson) >= 0.5:
            strength = "moderate"
        elif abs(pearson) >= 0.3:
            strength = "weak-to-moderate"
        else:
            strength = "weak"

        comments.append(
            f"The linear association is {strength}: Pearson r = {fmt(pearson)}, "
            f"p = {fmt_p(pearson_p)}."
        )

    if np.isfinite(spearman):
        if np.isfinite(pearson) and abs(spearman) > abs(pearson) + 0.1:
            comments.append(
                f"The rank association is stronger than the linear association: "
                f"Spearman r = {fmt(spearman)}, p = {fmt_p(spearman_p)}. "
                "This suggests that the method may rank samples better than it predicts "
                "exact numeric values."
            )
        else:
            comments.append(
                f"The rank association is Spearman r = {fmt(spearman)}, "
                f"p = {fmt_p(spearman_p)}."
            )

    if np.isfinite(bias):
        if bias > 0:
            comments.append(f"The method overestimates laboratory orgC on average by {fmt(bias)} units.")
        elif bias < 0:
            comments.append(f"The method underestimates laboratory orgC on average by {fmt(abs(bias))} units.")
        else:
            comments.append("The mean bias is approximately zero.")

    if np.isfinite(mae) and np.isfinite(rmse):
        comments.append(f"The absolute error is MAE = {fmt(mae)} and RMSE = {fmt(rmse)}.")

    if np.isfinite(ccc):
        if ccc >= 0.75:
            agreement = "good"
        elif ccc >= 0.5:
            agreement = "moderate"
        elif ccc >= 0.25:
            agreement = "limited"
        else:
            agreement = "poor"

        comments.append(
            f"Agreement is {agreement}: CCC = {fmt(ccc)}. CCC is stricter than "
            "correlation because it also penalizes bias and scale differences."
        )

    if np.isfinite(r2):
        if r2 < 0:
            comments.append(
                f"The direct-prediction R² is negative ({fmt(r2)}), meaning direct "
                "predictions are worse than simply using the mean laboratory value."
            )
        elif r2 < 0.25:
            comments.append(
                f"The direct-prediction R² is low ({fmt(r2)}), so this should not be "
                "presented as a validated direct estimator."
            )
        elif r2 < 0.5:
            comments.append(
                f"The direct-prediction R² is moderate ({fmt(r2)}), suggesting useful "
                "signal but still substantial unexplained variability."
            )
        else:
            comments.append(
                f"The direct-prediction R² is relatively strong ({fmt(r2)}), although "
                "independent validation is still needed."
            )

    if np.isfinite(lower) and np.isfinite(upper):
        comments.append(
            f"The Bland-Altman 95% limits of agreement are {fmt(lower)} to {fmt(upper)}, "
            "showing the likely range of individual errors."
        )

    if mode == "trained":
        comments.append(
            "This section uses cross-validated predictions, so each sample is predicted "
            "by a model that was not trained on that same sample."
        )
    elif mode == "gray_no_training":
        comments.append(
            "No training is applied in this section because the current grey-scale subset "
            "is too small. The purpose is to show whether a useful signal is already visible."
        )
    elif mode == "citizen":
        comments.append(
            "This section is useful as a baseline comparison against the citizen-science estimate."
        )

    return comments


def build_citizen_comparison(
    lab_df: pd.DataFrame,
    lab_col: str,
    citizen_col: str,
) -> ComparisonResult | None:
    if lab_col not in lab_df.columns or citizen_col not in lab_df.columns:
        return None

    df = clean_pair_df(lab_df, lab_col, citizen_col)

    if len(df) == 0:
        return None

    metrics = compute_metrics(
        df[lab_col].to_numpy(dtype=float),
        df[citizen_col].to_numpy(dtype=float),
    )

    return ComparisonResult(
        key="citizen",
        title="Lab vs citizen results",
        subtitle="Citizen estimate",
        df=df,
        y_true_col=lab_col,
        y_pred_col=citizen_col,
        method_note=(
            "Direct comparison between laboratory organic carbon and the citizen-science "
            "organic carbon estimate."
        ),
        metrics=metrics,
        figures={},
        interpretation=interpret_metrics(metrics, mode="citizen"),
    )


def build_no_gray_trained_comparison(
    no_gray_df: pd.DataFrame,
    lab_col: str,
    feature_cols: list[str],
) -> tuple[ComparisonResult | None, dict | None]:
    df = filter_processing_ok(no_gray_df)

    missing = [c for c in [lab_col] + feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"No-gray table is missing columns required for training: {missing}"
        )

    trained_df, model_info = train_cv_predictions(
        df,
        target_col=lab_col,
        feature_cols=feature_cols,
        prediction_col="predicted_orgC_cv_no_gray",
    )

    metrics = compute_metrics(
        trained_df[lab_col].to_numpy(dtype=float),
        trained_df["predicted_orgC_cv_no_gray"].to_numpy(dtype=float),
    )

    result = ComparisonResult(
        key="no_gray_trained",
        title="Lab vs no-grey-scale algorithm",
        subtitle="Cross-validated calibrated prediction",
        df=trained_df,
        y_true_col=lab_col,
        y_pred_col="predicted_orgC_cv_no_gray",
        method_note=(
            "The no-grey-scale image features are calibrated using a linear model with "
            "cross-validation. By default the model is: lab orgC ~ L + a + b."
        ),
        metrics=metrics,
        figures={},
        interpretation=interpret_metrics(metrics, mode="trained"),
    )

    return result, model_info


def build_with_gray_direct_comparison(
    with_gray_df: pd.DataFrame,
    lab_col: str,
    estimate_col: str,
) -> ComparisonResult | None:
    df = filter_processing_ok(with_gray_df)

    if lab_col not in df.columns or estimate_col not in df.columns:
        raise ValueError(
            f"With-gray table must contain columns {lab_col!r} and {estimate_col!r}."
        )

    df = clean_pair_df(df, lab_col, estimate_col)

    if len(df) == 0:
        return None

    metrics = compute_metrics(
        df[lab_col].to_numpy(dtype=float),
        df[estimate_col].to_numpy(dtype=float),
    )

    return ComparisonResult(
        key="with_gray_direct",
        title="Lab vs grey-scale algorithm",
        subtitle="Direct grey-corrected algorithm estimate",
        df=df,
        y_true_col=lab_col,
        y_pred_col=estimate_col,
        method_note=(
            "Direct comparison between laboratory organic carbon and the grey-scale-corrected "
            "algorithm estimate. No training is applied because the current grey-scale dataset "
            "is still too small."
        ),
        metrics=metrics,
        figures={},
        interpretation=interpret_metrics(metrics, mode="gray_no_training"),
    )


def metrics_to_summary_row(result: ComparisonResult) -> dict:
    row = {"scenario": result.title}
    row.update(result.metrics)
    row["method_note"] = result.method_note
    return row


def relative_figure_path(path: Path, out_dir: Path) -> str:
    return path.relative_to(out_dir).as_posix()


def html_metric_table(summary_df: pd.DataFrame) -> str:
    columns = [
        "scenario",
        "n",
        "lab_mean",
        "lab_sd",
        "estimate_mean",
        "estimate_sd",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "CCC_agreement",
        "bias_mean_estimate_minus_lab",
        "MAE",
        "RMSE",
        "R2_direct_prediction",
        "bland_altman_lower_95",
        "bland_altman_upper_95",
        "R2_calibration_regression",
    ]

    existing = [c for c in columns if c in summary_df.columns]
    headers = "".join(f"<th>{html.escape(c)}</th>" for c in existing)

    rows = []
    for _, row in summary_df.iterrows():
        cells = []

        for col in existing:
            value = row[col]

            if col == "scenario":
                cells.append(f"<td>{html.escape(str(value))}</td>")
            elif col == "n":
                cells.append(f"<td>{int(value) if pd.notna(value) else 'NA'}</td>")
            elif col.endswith("_p"):
                cells.append(f"<td>{fmt_p(value)}</td>")
            else:
                cells.append(f"<td>{fmt(value)}</td>")

        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<table>
  <thead>
    <tr>{headers}</tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
"""


def html_model_info(model_info: dict | None) -> str:
    if not model_info:
        return ""

    coef = model_info.get("coefficients_original_units", {})
    rows = []

    rows.append(
        "<tr><td>Training rows</td>"
        f"<td>{html.escape(str(model_info.get('n_training_rows')))}</td></tr>"
    )
    rows.append(
        "<tr><td>Cross-validation</td>"
        f"<td>{html.escape(str(model_info.get('cv')))}</td></tr>"
    )
    rows.append(
        "<tr><td>Features</td>"
        f"<td>{html.escape(', '.join(model_info.get('features', [])))}</td></tr>"
    )
    rows.append(
        "<tr><td>Intercept</td>"
        f"<td>{fmt(model_info.get('intercept_original_units'))}</td></tr>"
    )

    for name, value in coef.items():
        rows.append(
            f"<tr><td>Coefficient: {html.escape(name)}</td><td>{fmt(value)}</td></tr>"
        )

    return f"""
<section class="section-card">
  <h2>Training model used for the no-grey-scale section</h2>
  <table>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</section>
"""


def render_html_report(
    results: list[ComparisonResult],
    summary_df: pd.DataFrame,
    out_dir: Path,
    model_info: dict | None,
    title: str,
) -> None:
    figures_dir = out_dir / "figures"
    summary_chart = figures_dir / "method_error_comparison.png"

    if len(summary_df) > 0:
        make_metrics_bar_plot(summary_df, summary_chart)

    cards = []

    for result in results:
        fig_scatter = relative_figure_path(result.figures["scatter"], out_dir)
        fig_ba = relative_figure_path(result.figures["bland_altman"], out_dir)
        fig_residual = relative_figure_path(result.figures["residual"], out_dir)

        interpretation_items = "\n".join(
            f"<li>{html.escape(item)}</li>" for item in result.interpretation
        )

        cards.append(
            f"""
<section class="section-card" id="{html.escape(slugify(result.key))}">
  <h2>{html.escape(result.title)}</h2>

  <p class="method-note">{html.escape(result.method_note)}</p>

  <div class="metric-strip">
    <div><span>n</span><strong>{fmt(result.metrics.get("n"), 0)}</strong></div>
    <div><span>Pearson r</span><strong>{fmt(result.metrics.get("pearson_r"))}</strong></div>
    <div><span>Spearman r</span><strong>{fmt(result.metrics.get("spearman_r"))}</strong></div>
    <div><span>Bias</span><strong>{fmt(result.metrics.get("bias_mean_estimate_minus_lab"))}</strong></div>
    <div><span>MAE</span><strong>{fmt(result.metrics.get("MAE"))}</strong></div>
    <div><span>RMSE</span><strong>{fmt(result.metrics.get("RMSE"))}</strong></div>
    <div><span>R²</span><strong>{fmt(result.metrics.get("R2_direct_prediction"))}</strong></div>
  </div>

  <h3>Commentary</h3>
  <ul>
    {interpretation_items}
  </ul>

  <div class="figure-grid">
    <figure>
      <img src="{html.escape(fig_scatter)}" alt="Scatter plot">
      <figcaption>
        Laboratory value against the estimate. The dashed line is the 1:1 line.
      </figcaption>
    </figure>

    <figure>
      <img src="{html.escape(fig_ba)}" alt="Bland-Altman plot">
      <figcaption>
        Agreement plot. The y-axis is estimate minus laboratory value.
      </figcaption>
    </figure>

    <figure>
      <img src="{html.escape(fig_residual)}" alt="Residual plot">
      <figcaption>
        Error pattern against laboratory orgC.
      </figcaption>
    </figure>
  </div>
</section>
"""
        )

    summary_chart_html = ""
    if summary_chart.exists():
        summary_chart_html = f"""
<figure>
  <img src="{html.escape(relative_figure_path(summary_chart, out_dir))}" alt="Method error comparison">
  <figcaption>MAE and RMSE comparison across the available methods.</figcaption>
</figure>
"""

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #1f2933;
      background: #f5f7fa;
      line-height: 1.5;
    }}

    header {{
      background: #111827;
      color: white;
      padding: 32px 40px;
    }}

    header h1 {{
      margin: 0 0 8px 0;
      font-size: 30px;
    }}

    header p {{
      margin: 0;
      max-width: 980px;
      color: #d1d5db;
    }}

    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 20px 60px 20px;
    }}

    .section-card {{
      background: white;
      border-radius: 14px;
      padding: 26px;
      margin-bottom: 28px;
      box-shadow: 0 2px 14px rgba(15, 23, 42, 0.08);
    }}

    h2 {{
      margin-top: 0;
      color: #111827;
    }}

    h3 {{
      margin-top: 24px;
      color: #1f2937;
    }}

    .method-note {{
      color: #4b5563;
      margin-bottom: 18px;
    }}

    .metric-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}

    .metric-strip div {{
      background: #f3f4f6;
      border-radius: 10px;
      padding: 12px;
    }}

    .metric-strip span {{
      display: block;
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 4px;
    }}

    .metric-strip strong {{
      font-size: 20px;
      color: #111827;
    }}

    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 18px;
      margin-top: 16px;
    }}

    figure {{
      margin: 0;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 12px;
    }}

    figure img {{
      width: 100%;
      display: block;
      border-radius: 8px;
      background: white;
    }}

    figcaption {{
      font-size: 13px;
      color: #4b5563;
      margin-top: 8px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      overflow-x: auto;
      display: block;
    }}

    th,
    td {{
      border: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: right;
      white-space: nowrap;
    }}

    th:first-child,
    td:first-child {{
      text-align: left;
    }}

    th {{
      background: #f3f4f6;
      color: #374151;
    }}

    ul {{
      padding-left: 22px;
    }}

    code {{
      background: #f3f4f6;
      padding: 2px 5px;
      border-radius: 4px;
    }}

    .warning {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 12px;
      padding: 14px 16px;
      color: #7c2d12;
    }}
  </style>
</head>

<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>
      Automated presentation report comparing laboratory organic carbon results with
      citizen-science estimates and image-based soil colour algorithms.
    </p>
  </header>

  <main>
    <section class="section-card">
      <h2>Executive summary</h2>

      <p>This report separates three questions:</p>

      <ol>
        <li>How do citizen-science organic carbon estimates compare with laboratory results?</li>
        <li>How well can non-grey-scale image results be calibrated against laboratory results?</li>
        <li>Do grey-scale-corrected image results already show a useful signal, even before training?</li>
      </ol>

      <div class="warning">
        The grey-scale subset should be presented as preliminary if the number of matched
        samples is still small. The appropriate conclusion is not final validation, but whether
        the colour signal is promising enough to continue collecting the planned 100-200
        grey-scale samples.
      </div>
    </section>

    <section class="section-card">
      <h2>Metrics overview</h2>
      {html_metric_table(summary_df)}

      <h3>Error comparison</h3>
      {summary_chart_html}
    </section>

    {html_model_info(model_info)}

    {''.join(cards)}

    <section class="section-card">
      <h2>How to present the current status</h2>

      <p>The safest scientific interpretation is:</p>

      <ul>
        <li>
          Citizen estimates provide a useful baseline for comparison, but the level of agreement
          with laboratory results must be assessed directly.
        </li>
        <li>
          The no-grey-scale algorithm should be presented using cross-validated calibrated
          predictions, not only training-set performance.
        </li>
        <li>
          The grey-scale algorithm should currently be presented as a preliminary direct
          comparison. If it shows stronger correlation but still weak agreement, it should be
          described as a promising calibration signal, not as a validated direct estimator.
        </li>
        <li>
          Once 100-200 grey-scale samples are available, the same report can be rerun and the
          grey-scale section can be changed from direct comparison to trained cross-validated
          calibration.
        </li>
      </ul>
    </section>
  </main>
</body>
</html>
"""

    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def write_predictions(result: ComparisonResult, out_dir: Path) -> None:
    output_path = out_dir / f"{slugify(result.key)}_predictions.csv"

    cols = []

    for col in ["ID", "SampleCode", "sample_code_base", "image"]:
        if col in result.df.columns:
            cols.append(col)

    cols += [result.y_true_col, result.y_pred_col]
    cols = list(dict.fromkeys(cols))

    df = result.df[cols].copy()
    df["error_estimate_minus_lab"] = df[result.y_pred_col] - df[result.y_true_col]
    df["absolute_error"] = df["error_estimate_minus_lab"].abs()
    df.to_csv(output_path, index=False)


def run_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    figures_dir = out_dir / "figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    lab_df = normalize_columns(read_table(args.lab))

    no_gray_df = None
    with_gray_df = None

    if args.no_gray:
        no_gray_df = normalize_columns(read_table(args.no_gray))

    if args.with_gray:
        with_gray_df = normalize_columns(read_table(args.with_gray))

    results: list[ComparisonResult] = []
    model_info = None

    citizen_result = build_citizen_comparison(
        lab_df=lab_df,
        lab_col=args.lab_col,
        citizen_col=args.citizen_col,
    )

    if citizen_result:
        results.append(citizen_result)
    else:
        print(
            "WARNING: Citizen comparison skipped. Could not find columns "
            f"{args.lab_col!r} and {args.citizen_col!r} in {args.lab}.",
            file=sys.stderr,
        )

    if no_gray_df is not None:
        no_gray_result, model_info = build_no_gray_trained_comparison(
            no_gray_df=no_gray_df,
            lab_col=args.lab_col,
            feature_cols=args.no_gray_features,
        )

        if no_gray_result:
            results.append(no_gray_result)

    if with_gray_df is not None:
        with_gray_result = build_with_gray_direct_comparison(
            with_gray_df=with_gray_df,
            lab_col=args.lab_col,
            estimate_col=args.gray_estimate_col,
        )

        if with_gray_result:
            results.append(with_gray_result)

    if not results:
        raise RuntimeError("No comparisons could be generated.")

    for result in results:
        result.figures = create_figures(result, figures_dir)
        write_predictions(result, out_dir)

    summary_df = pd.DataFrame([metrics_to_summary_row(r) for r in results])
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    render_html_report(
        results=results,
        summary_df=summary_df,
        out_dir=out_dir,
        model_info=model_info,
        title=args.title,
    )

    print("")
    print(f"Report created: {out_dir / 'index.html'}")
    print(f"Summary metrics: {out_dir / 'summary_metrics.csv'}")
    print("")
    print("Open locally with:")
    print(f"  python3 -m http.server 8088 --directory {out_dir}")
    print("")
    print("Then open:")
    print("  http://localhost:8088")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate automated presentation report for orgC comparisons."
    )

    parser.add_argument(
        "--lab",
        default="data/lab/test_stat_orgC.xlsx",
        help="Laboratory file containing orgC_lab and optionally orgC_CS.",
    )
    parser.add_argument(
        "--no-gray",
        default="outputs/test_stat_orgC_enriched_no_gray.xlsx",
        help="Enriched results file for images processed without grey-scale correction.",
    )
    parser.add_argument(
        "--with-gray",
        default="outputs/test_stat_orgC_enriched_with_gray.xlsx",
        help="Enriched results file for images processed with grey-scale correction.",
    )
    parser.add_argument(
        "--out",
        default="outputs/presentation_report",
        help="Output directory for the static HTML report.",
    )
    parser.add_argument(
        "--lab-col",
        default="orgC_lab",
        help="Column name for laboratory organic carbon.",
    )
    parser.add_argument(
        "--citizen-col",
        default="orgC_CS",
        help="Column name for citizen-science organic carbon estimate.",
    )
    parser.add_argument(
        "--gray-estimate-col",
        default="SOC_est%",
        help="Column name for the direct grey-scale algorithm estimate.",
    )
    parser.add_argument(
        "--no-gray-features",
        nargs="+",
        default=["L", "a", "b"],
        help="Feature columns used to train the no-grey-scale calibration model.",
    )
    parser.add_argument(
        "--title",
        default="Soil orgC comparison report",
        help="Title shown in the HTML report.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_report(args)


if __name__ == "__main__":
    main()
