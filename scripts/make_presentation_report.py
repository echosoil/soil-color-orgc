#!/usr/bin/env python3
"""Generate a grey-scale-only soil orgC model and QC report.

The report uses only:
1. Laboratory orgC vs citizen-science estimates as an external baseline.
2. The direct grey-scale-corrected SOC formula as a non-trained baseline.
3. The grey-scale-only model experiment produced by run_model_experiment.py.
4. Grey-scale detection, correction, soil-ROI, and colour-card QC images.

The report never trains models and never uses another image dataset. Model
selection is based on repeated cross-validation in the experiment output; the
held-out test set is used only for final evaluation.
"""

from __future__ import annotations

import argparse
import html
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    interpretation: list[str]
    figures: dict | None = None


def fmt(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "section"


def read_table(path: str | Path, preferred_sheet: str = "enriched") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        book = pd.ExcelFile(path)
        sheet = preferred_sheet if preferred_sheet in book.sheet_names else book.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def filter_processing_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "processing_status" not in df.columns:
        return df.copy()
    status = df["processing_status"].fillna("").astype(str).str.lower().str.strip()
    return df.loc[status.isin({"ok", "", "none", "nan"})].copy()


def clean_pair_df(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    missing = [c for c in [true_col, pred_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing comparison columns: {missing}")
    df = df.copy()
    df[true_col] = pd.to_numeric(df[true_col], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    keep = [c for c in ["ID", "SampleCode", "sample_code_base", "image", "model"] if c in df.columns]
    keep += [true_col, pred_col]
    return df[list(dict.fromkeys(keep))].dropna(subset=[true_col, pred_col]).copy()


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    var_true = np.var(y_true, ddof=1)
    var_pred = np.var(y_pred, ddof=1)
    covariance = np.cov(y_true, y_pred, ddof=1)[0, 1]
    denominator = var_true + var_pred + (np.mean(y_true) - np.mean(y_pred)) ** 2
    return float(2 * covariance / denominator) if denominator else np.nan


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    n = len(y_true)
    out = {
        "n": n, "lab_mean": np.nan, "lab_sd": np.nan,
        "estimate_mean": np.nan, "estimate_sd": np.nan,
        "pearson_r": np.nan, "pearson_p": np.nan,
        "spearman_r": np.nan, "spearman_p": np.nan,
        "CCC_agreement": np.nan, "bias_mean_estimate_minus_lab": np.nan,
        "MAE": np.nan, "RMSE": np.nan, "R2_direct_prediction": np.nan,
        "bland_altman_lower_95": np.nan, "bland_altman_upper_95": np.nan,
    }
    if n == 0:
        return out
    residual = y_pred - y_true
    out.update({
        "lab_mean": float(np.mean(y_true)),
        "estimate_mean": float(np.mean(y_pred)),
        "bias_mean_estimate_minus_lab": float(np.mean(residual)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    })
    if n >= 2:
        out["lab_sd"] = float(np.std(y_true, ddof=1))
        out["estimate_sd"] = float(np.std(y_pred, ddof=1))
        out["CCC_agreement"] = compute_ccc(y_true, y_pred)
        out["R2_direct_prediction"] = float(r2_score(y_true, y_pred))
        sd = float(np.std(residual, ddof=1))
        bias = out["bias_mean_estimate_minus_lab"]
        out["bland_altman_lower_95"] = bias - 1.96 * sd
        out["bland_altman_upper_95"] = bias + 1.96 * sd
    if n >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = stats.pearsonr(y_true, y_pred)
        spearman = stats.spearmanr(y_true, y_pred)
        out["pearson_r"], out["pearson_p"] = float(pearson.statistic), float(pearson.pvalue)
        out["spearman_r"], out["spearman_p"] = float(spearman.statistic), float(spearman.pvalue)
    return out


def load_model_experiment(experiment_dir: str | Path) -> dict:
    """Load experiment outputs and retain dataset='with_gray' only."""
    root = Path(experiment_dir)
    paths = {
        "summary": root / "model_experiment_summary.csv",
        "predictions": root / "model_experiment_predictions.csv",
        "split": root / "model_experiment_split.csv",
        "features": root / "model_experiment_features.csv",
    }
    missing = [str(paths[k]) for k in ["summary", "predictions", "split"] if not paths[k].exists()]
    if missing:
        raise FileNotFoundError("Missing model-experiment outputs:\n  " + "\n  ".join(missing))

    summary = normalize_columns(pd.read_csv(paths["summary"]))
    predictions = normalize_columns(pd.read_csv(paths["predictions"]))
    split = normalize_columns(pd.read_csv(paths["split"]))
    features = normalize_columns(pd.read_csv(paths["features"])) if paths["features"].exists() else None

    for name, df in [("summary", summary), ("predictions", predictions)]:
        if "dataset" not in df.columns:
            raise ValueError(f"Experiment {name} must contain a 'dataset' column.")

    summary = summary[summary["dataset"].astype(str).str.lower().eq("with_gray")].copy()
    predictions = predictions[predictions["dataset"].astype(str).str.lower().eq("with_gray")].copy()
    if features is not None and "dataset" in features.columns:
        features = features[features["dataset"].astype(str).str.lower().eq("with_gray")].copy()

    if summary.empty or predictions.empty:
        raise ValueError("Experiment outputs contain no dataset='with_gray' rows.")

    required_summary = {"model", "cv_RMSE_mean", "cv_MAE_mean", "test_MAE", "test_RMSE", "test_R2", "test_pearson_r"}
    required_predictions = {"model", "lab_value", "predicted_value"}
    if required_summary - set(summary.columns):
        raise ValueError(f"Experiment summary missing: {sorted(required_summary - set(summary.columns))}")
    if required_predictions - set(predictions.columns):
        raise ValueError(f"Experiment predictions missing: {sorted(required_predictions - set(predictions.columns))}")

    return {"summary": summary, "predictions": predictions, "split": split, "features": features}


def select_best_model_by_cv(summary: pd.DataFrame) -> pd.Series:
    candidates = summary[summary["model"].astype(str).ne("baseline_mean")].copy()
    candidates["cv_RMSE_mean"] = pd.to_numeric(candidates["cv_RMSE_mean"], errors="coerce")
    candidates["cv_MAE_mean"] = pd.to_numeric(candidates["cv_MAE_mean"], errors="coerce")
    candidates = candidates.dropna(subset=["cv_RMSE_mean", "cv_MAE_mean"]).sort_values(
        ["cv_RMSE_mean", "cv_MAE_mean", "model"]
    )
    if candidates.empty:
        raise ValueError("No non-baseline model has valid cross-validation metrics.")
    return candidates.iloc[0]

def interpret_metrics(metrics: dict, mode: str, model_name: str | None = None) -> list[str]:
    n = metrics.get("n", 0)
    r = metrics.get("pearson_r", np.nan)
    bias = metrics.get("bias_mean_estimate_minus_lab", np.nan)
    mae = metrics.get("MAE", np.nan)
    rmse = metrics.get("RMSE", np.nan)
    r2 = metrics.get("R2_direct_prediction", np.nan)
    comments = [f"This comparison contains n = {n} samples."]
    if np.isfinite(r):
        strength = "strong" if abs(r) >= .7 else "moderate" if abs(r) >= .5 else "weak-to-moderate" if abs(r) >= .3 else "weak"
        comments.append(f"The linear association is {strength}: Pearson r = {fmt(r)}.")
    if np.isfinite(bias):
        direction = "overestimates" if bias > 0 else "underestimates" if bias < 0 else "has approximately zero bias relative to"
        amount = f" by {fmt(abs(bias))} units on average" if bias else ""
        comments.append(f"The method {direction} laboratory orgC{amount}.")
    if np.isfinite(mae) and np.isfinite(rmse):
        comments.append(f"Prediction error is MAE = {fmt(mae)} and RMSE = {fmt(rmse)}.")
    if np.isfinite(r2):
        if r2 < 0:
            comments.append(f"R² is negative ({fmt(r2)}), so predictions are worse than predicting the test-set mean.")
        elif r2 < .25:
            comments.append(f"R² is low ({fmt(r2)}); this is not yet a validated quantitative estimator.")
        elif r2 < .5:
            comments.append(f"R² is moderate ({fmt(r2)}), showing useful signal with substantial unexplained variability.")
        else:
            comments.append(f"R² is relatively strong ({fmt(r2)}), although further independent validation remains appropriate.")
    if mode == "final_test":
        comments.append(
            f"Model {model_name!r} was selected by repeated cross-validation on development data. "
            "The values shown here come from the untouched final test set."
        )
    elif mode == "direct_gray":
        comments.append("This is a direct formula-based SOC estimate, not a trained prediction model.")
    elif mode == "citizen":
        comments.append("Citizen estimates are an external baseline and do not participate in image-model selection.")
    return comments


def build_citizen_comparison(lab: pd.DataFrame, lab_col: str, citizen_col: str) -> ComparisonResult | None:
    if lab_col not in lab.columns or citizen_col not in lab.columns:
        return None
    df = clean_pair_df(lab, lab_col, citizen_col)
    if df.empty:
        return None
    metrics = compute_metrics(df[lab_col], df[citizen_col])
    return ComparisonResult(
        "citizen_baseline", "Laboratory vs citizen-science estimate", "Citizen-science orgC estimate",
        df, lab_col, citizen_col,
        "External baseline only; it is not used to train or select image models.",
        metrics, interpret_metrics(metrics, "citizen")
    )


def build_direct_gray_comparison(df: pd.DataFrame, lab_col: str, estimate_col: str) -> ComparisonResult | None:
    df = filter_processing_ok(df)
    if lab_col not in df.columns or estimate_col not in df.columns:
        return None
    df = clean_pair_df(df, lab_col, estimate_col)
    if df.empty:
        return None
    metrics = compute_metrics(df[lab_col], df[estimate_col])
    return ComparisonResult(
        "direct_gray_heuristic", "Laboratory vs direct grey-corrected SOC heuristic",
        "Direct formula-based SOC estimate", df, lab_col, estimate_col,
        "Formula-based estimate after grey-scale correction; no supervised training is involved.",
        metrics, interpret_metrics(metrics, "direct_gray")
    )


def build_selected_model_comparison(summary: pd.DataFrame, predictions: pd.DataFrame) -> tuple[ComparisonResult, pd.Series]:
    selected = select_best_model_by_cv(summary)
    model_name = str(selected["model"])
    df = predictions[predictions["model"].astype(str).eq(model_name)].copy()
    if df.empty:
        raise ValueError(f"No final-test predictions found for selected model {model_name!r}.")
    df["lab_value"] = pd.to_numeric(df["lab_value"], errors="coerce")
    df["predicted_value"] = pd.to_numeric(df["predicted_value"], errors="coerce")
    df = df.dropna(subset=["lab_value", "predicted_value"])
    metrics = compute_metrics(df["lab_value"], df["predicted_value"])
    result = ComparisonResult(
        f"selected_model_{model_name}", f"Held-out final test: {model_name}",
        "Predicted laboratory orgC", df, "lab_value", "predicted_value",
        "Selected solely by the lowest repeated-CV RMSE among non-baseline models; test metrics were not used for selection.",
        metrics, interpret_metrics(metrics, "final_test", model_name)
    )
    return result, selected


def make_scatter(result: ComparisonResult, path: Path) -> None:
    x = result.df[result.y_true_col].to_numpy(float)
    y = result.df[result.y_pred_col].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, alpha=.8)
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (high - low) * .08 if high > low else 1
    ax.plot([low-pad, high+pad], [low-pad, high+pad], "--")
    ax.set(xlabel="Laboratory orgC", ylabel=result.subtitle, title=f"{result.title}: laboratory vs estimate")
    ax.grid(alpha=.25)
    label = f"n={result.metrics['n']}\nr={fmt(result.metrics['pearson_r'])}\nMAE={fmt(result.metrics['MAE'])}\nRMSE={fmt(result.metrics['RMSE'])}\nR²={fmt(result.metrics['R2_direct_prediction'])}"
    ax.text(.04, .96, label, transform=ax.transAxes, va="top", bbox={"boxstyle":"round", "alpha":.15})
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def make_bland_altman(result: ComparisonResult, path: Path) -> None:
    true = result.df[result.y_true_col].to_numpy(float)
    pred = result.df[result.y_pred_col].to_numpy(float)
    means, residual = (true + pred) / 2, pred - true
    fig, ax = plt.subplots(figsize=(7, 5.5)); ax.scatter(means, residual, alpha=.8)
    for value, style, label in [
        (result.metrics["bias_mean_estimate_minus_lab"], "-", "Bias"),
        (result.metrics["bland_altman_lower_95"], "--", "Lower 95%"),
        (result.metrics["bland_altman_upper_95"], "--", "Upper 95%")]:
        if np.isfinite(value): ax.axhline(value, linestyle=style, label=f"{label}={fmt(value)}")
    ax.axhline(0, linestyle=":", label="Zero error")
    ax.set(xlabel="Mean of laboratory and estimate", ylabel="Estimate minus laboratory", title=f"{result.title}: Bland-Altman")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def make_residual(result: ComparisonResult, path: Path) -> None:
    true = result.df[result.y_true_col].to_numpy(float)
    residual = result.df[result.y_pred_col].to_numpy(float) - true
    fig, ax = plt.subplots(figsize=(7, 5.5)); ax.scatter(true, residual, alpha=.8); ax.axhline(0, linestyle=":")
    ax.set(xlabel="Laboratory orgC", ylabel="Estimate minus laboratory", title=f"{result.title}: residuals")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def create_figures(result: ComparisonResult, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    stem = slugify(result.key)
    paths = {k: directory / f"{stem}_{k}.png" for k in ["scatter", "bland_altman", "residual"]}
    make_scatter(result, paths["scatter"]); make_bland_altman(result, paths["bland_altman"]); make_residual(result, paths["residual"])
    return paths


def write_predictions(result: ComparisonResult, out_dir: Path) -> None:
    columns = [c for c in ["ID", "SampleCode", "sample_code_base", "image", "model"] if c in result.df.columns]
    columns += [result.y_true_col, result.y_pred_col]
    output = result.df[list(dict.fromkeys(columns))].copy()
    output["error_estimate_minus_lab"] = output[result.y_pred_col] - output[result.y_true_col]
    output["absolute_error"] = output["error_estimate_minus_lab"].abs()
    output.to_csv(out_dir / f"{slugify(result.key)}_predictions.csv", index=False)


def extract_sample_code(value) -> str | None:
    if value is None or pd.isna(value): return None
    match = re.search(r"[A-Za-z]{4}", Path(str(value).strip()).stem)
    return match.group(0).upper() if match else None


def collect_qc_codes(explicit: list[str] | None, df: pd.DataFrame, maximum: int) -> list[str]:
    codes: list[str] = []
    values = explicit or []
    if not explicit:
        for col in ["sample_code_base", "SampleCode", "ID", "image"]:
            if col in df.columns:
                values = df[col].dropna().tolist(); break
    for value in values:
        code = extract_sample_code(value)
        if code and code not in codes: codes.append(code)
        if len(codes) >= maximum: break
    return codes


def find_ci(directory: Path, filename: str) -> Path | None:
    direct = directory / filename
    if direct.exists(): return direct
    if not directory.exists(): return None
    target = filename.lower()
    return next((p for p in directory.iterdir() if p.is_file() and p.name.lower() == target), None)


def copy_asset(source: Path | None, out_dir: Path, subdir: str, name: str) -> str | None:
    if source is None or not source.exists(): return None
    target_dir = out_dir / subdir; target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name; shutil.copy2(source, target)
    return target.relative_to(out_dir).as_posix()


def image_html(title: str, relative: str | None, caption: str, missing: str) -> str:
    if relative:
        return f'<figure><img src="{html.escape(relative)}" alt="{html.escape(title)}"><figcaption><strong>{html.escape(title)}.</strong> {html.escape(caption)}</figcaption></figure>'
    return f'<figure class="missing-asset"><div class="missing-box">Missing image</div><figcaption><strong>{html.escape(title)}.</strong> {html.escape(missing)}</figcaption></figure>'


def build_qc_html(out_dir: Path, codes: list[str], masks: Path, gray: Path, cards: Path) -> str:
    if not codes:
        return '<section class="section-card"><h2>Image-processing QC</h2><p>No QC codes selected.</p></section>'
    sections = []
    for code in codes:
        assets = {
            "soil": copy_asset(find_ci(masks, f"{code}_roi_rect.jpg"), out_dir, "assets/qc/masks", f"{code}_roi_rect.jpg"),
            "grayrect": copy_asset(find_ci(gray, f"{code}_gray_roi_rect.jpg"), out_dir, "assets/qc/gray", f"{code}_gray_roi_rect.jpg"),
            "grayroi": copy_asset(find_ci(gray, f"{code}_gray_roi.jpg"), out_dir, "assets/qc/gray", f"{code}_gray_roi.jpg"),
            "beforeafter": copy_asset(find_ci(gray, f"{code}_gray_before_after.jpg"), out_dir, "assets/qc/gray", f"{code}_gray_before_after.jpg"),
            "card": copy_asset(find_ci(cards, f"{code}_comparison.jpg"), out_dir, "assets/qc/cards", f"{code}_comparison.jpg"),
        }
        sections.append(f'''<section class="qc-sample-card"><h3>Sample {html.escape(code)}</h3>
        <div class="figure-grid">
        {image_html("Soil ROI", assets["soil"], "The rectangle should contain representative soil only.", str(masks / f"{code}_roi_rect.jpg"))}
        {image_html("Detected grey-scale rectangle", assets["grayrect"], "The rectangle should surround the full 11-patch scale.", str(gray / f"{code}_gray_roi_rect.jpg"))}
        {image_html("Extracted grey-scale ROI", assets["grayroi"], "The crop should contain the darkest-to-lightest patches with little paper or text.", str(gray / f"{code}_gray_roi.jpg"))}
        {image_html("Before/after correction", assets["beforeafter"], "Correction should neutralise colour cast without clipping or distortion.", str(gray / f"{code}_gray_before_after.jpg"))}
        {image_html("Colour/Munsell card", assets["card"], "Visual check of ROI colour, Lab estimate, closest Munsell chip, and direct SOC heuristic.", str(cards / f"{code}_comparison.jpg"))}
        </div></section>''')
    return '<section class="section-card"><h2>Image-processing quality control</h2><p>These examples verify that grey-scale detection and soil colour extraction are visually plausible.</p>' + ''.join(sections) + '</section>'

def metrics_summary_row(result: ComparisonResult) -> dict:
    row = {"scenario": result.title, **result.metrics, "method_note": result.method_note}
    return row


def html_table(df: pd.DataFrame, columns: list[str], selected_model: str | None = None) -> str:
    columns = [c for c in columns if c in df.columns]
    head = ''.join(f'<th>{html.escape(c)}</th>' for c in columns)
    body = []
    for _, row in df.iterrows():
        is_selected = selected_model is not None and str(row.get("model")) == selected_model
        cells = []
        for col in columns:
            value = row[col]
            if col in {"scenario", "model", "features", "SampleCode", "ID", "image", "interpretation"}:
                text = html.escape(str(value))
                if col == "model" and is_selected: text += " <strong>(selected by CV)</strong>"
            elif col in {"n", "train_n", "test_n"}:
                text = str(int(value)) if pd.notna(value) else "NA"
            else:
                text = fmt(value)
            cells.append(f'<td>{text}</td>')
        klass = ' class="selected-row"' if is_selected else ''
        body.append(f'<tr{klass}>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def build_experiment_html(summary: pd.DataFrame, split: pd.DataFrame, selected: pd.Series) -> str:
    model = str(selected["model"])
    counts = split["split"].astype(str).str.lower().value_counts() if "split" in split.columns else pd.Series(dtype=int)
    train_n, test_n = int(counts.get("train", 0)), int(counts.get("test", 0))

    baseline = summary[summary["model"].astype(str).eq("baseline_mean")]
    if not baseline.empty:
        gain = float(baseline.iloc[0]["test_RMSE"]) - float(selected["test_RMSE"])
        baseline_text = (
            f"The selected model improves held-out RMSE over the mean baseline by {fmt(gain)} orgC units."
            if gain > 0 else
            f"The selected model does not improve on the mean baseline; RMSE difference is {fmt(gain)}."
        )
    else:
        baseline_text = "No mean-prediction baseline row was available."

    munsell = summary[summary["model"].astype(str).eq("model_4_Munsell_interpolated")]
    if not munsell.empty:
        difference = float(munsell.iloc[0]["cv_RMSE_mean"]) - float(selected["cv_RMSE_mean"])
        if abs(difference) < 1e-12:
            munsell_text = "The interpolated-Munsell model ties the selected model on mean CV RMSE."
        elif difference > 0:
            munsell_text = f"The interpolated-Munsell model has {fmt(difference)} higher mean CV RMSE than the selected model."
        else:
            munsell_text = f"The interpolated-Munsell model has {fmt(abs(difference))} lower mean CV RMSE; review the selection output."
    else:
        munsell_text = "No interpolated-Munsell model row was found."

    ordered = summary.copy()
    ordered["cv_RMSE_mean"] = pd.to_numeric(ordered["cv_RMSE_mean"], errors="coerce")
    ordered = ordered.sort_values(["cv_RMSE_mean", "model"], na_position="last")
    columns = ["model", "features", "train_n", "test_n", "cv_RMSE_mean", "cv_RMSE_sd", "cv_MAE_mean", "cv_MAE_sd", "cv_R2_mean", "test_RMSE", "test_MAE", "test_bias_pred_minus_lab", "test_R2", "test_pearson_r"]

    return f'''<section class="section-card"><h2>Grey-scale model experiment</h2>
    <p>The shared experiment contains <strong>{train_n + test_n}</strong> usable grey-scale samples: <strong>{train_n}</strong> development samples and <strong>{test_n}</strong> untouched final-test samples.</p>
    <p>Models are selected by repeated five-fold cross-validation on development data. Final-test metrics are not used for selection.</p>
    <div class="metric-strip">
      <div><span>Selected model</span><strong>{html.escape(model)}</strong></div>
      <div><span>CV RMSE</span><strong>{fmt(selected.get("cv_RMSE_mean"))}</strong></div>
      <div><span>CV MAE</span><strong>{fmt(selected.get("cv_MAE_mean"))}</strong></div>
      <div><span>Test RMSE</span><strong>{fmt(selected.get("test_RMSE"))}</strong></div>
      <div><span>Test MAE</span><strong>{fmt(selected.get("test_MAE"))}</strong></div>
      <div><span>Test R²</span><strong>{fmt(selected.get("test_R2"))}</strong></div>
    </div>
    <div class="info-box"><p>{html.escape(baseline_text)}</p><p>{html.escape(munsell_text)}</p></div>
    <h3>All candidate models</h3>{html_table(ordered, columns, model)}</section>'''


def build_worst_errors_html(predictions: pd.DataFrame, model: str, count: int) -> str:
    df = predictions[predictions["model"].astype(str).eq(model)].copy()
    if df.empty: return ""
    df["lab_value"] = pd.to_numeric(df["lab_value"], errors="coerce")
    df["predicted_value"] = pd.to_numeric(df["predicted_value"], errors="coerce")
    df = df.dropna(subset=["lab_value", "predicted_value"])
    df["residual"] = df["predicted_value"] - df["lab_value"]
    df["absolute_error"] = df["residual"].abs()
    df = df.nlargest(count, "absolute_error")
    columns = ["SampleCode", "ID", "image", "lab_value", "predicted_value", "residual", "absolute_error"]
    return f'<section class="section-card"><h2>Largest final-test errors</h2><p>These cases should be prioritised for image-QC review and investigation of non-colour soil factors.</p>{html_table(df, columns)}</section>'


def make_error_bar(summary: pd.DataFrame, path: Path) -> None:
    df = summary.dropna(subset=["MAE", "RMSE"]).copy()
    if df.empty: return
    x = np.arange(len(df)); width = .35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x-width/2, df["MAE"].astype(float), width, label="MAE")
    ax.bar(x+width/2, df["RMSE"].astype(float), width, label="RMSE")
    ax.set_xticks(x); ax.set_xticklabels(df["scenario"], rotation=20, ha="right")
    ax.set_ylabel("orgC error"); ax.set_title("Error comparison"); ax.grid(axis="y", alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def build_mae_rmse_html(summary: pd.DataFrame) -> str:
    rows = []
    for _, row in summary.iterrows():
        mae, rmse = float(row["MAE"]), float(row["RMSE"])
        ratio = rmse / mae if mae > 0 else np.nan
        note = "Large errors are influential." if ratio > 1.5 else "Some larger errors are present." if ratio > 1.2 else "Error sizes are relatively even."
        rows.append({"scenario": row["scenario"], "MAE": mae, "RMSE": rmse, "RMSE_MAE_ratio": ratio, "interpretation": note})
    return f'<section class="section-card"><h2>How to interpret MAE and RMSE</h2><p>MAE is average absolute error; RMSE penalises large errors more strongly. Lower is better. Comparisons are most meaningful on the same sample set.</p>{html_table(pd.DataFrame(rows), ["scenario", "MAE", "RMSE", "RMSE_MAE_ratio", "interpretation"])}</section>'


def result_card(result: ComparisonResult, out_dir: Path) -> str:
    figures = result.figures or {}
    rel = {k: figures[k].relative_to(out_dir).as_posix() for k in figures}
    comments = ''.join(f'<li>{html.escape(c)}</li>' for c in result.interpretation)
    return f'''<section class="section-card" id="{slugify(result.key)}"><h2>{html.escape(result.title)}</h2>
    <p class="method-note">{html.escape(result.method_note)}</p>
    <div class="metric-strip">
      <div><span>n</span><strong>{fmt(result.metrics["n"], 0)}</strong></div>
      <div><span>Pearson r</span><strong>{fmt(result.metrics["pearson_r"])}</strong></div>
      <div><span>CCC</span><strong>{fmt(result.metrics["CCC_agreement"])}</strong></div>
      <div><span>Bias</span><strong>{fmt(result.metrics["bias_mean_estimate_minus_lab"])}</strong></div>
      <div><span>MAE</span><strong>{fmt(result.metrics["MAE"])}</strong></div>
      <div><span>RMSE</span><strong>{fmt(result.metrics["RMSE"])}</strong></div>
      <div><span>R²</span><strong>{fmt(result.metrics["R2_direct_prediction"])}</strong></div>
    </div><h3>Commentary</h3><ul>{comments}</ul>
    <div class="figure-grid">
      <figure><img src="{rel["scatter"]}"><figcaption>Laboratory value against estimate; dashed line is 1:1.</figcaption></figure>
      <figure><img src="{rel["bland_altman"]}"><figcaption>Agreement plot; y-axis is estimate minus laboratory.</figcaption></figure>
      <figure><img src="{rel["residual"]}"><figcaption>Residual pattern against laboratory orgC.</figcaption></figure>
    </div></section>'''

def render_report(
    out_dir: Path,
    title: str,
    results: list[ComparisonResult],
    summary: pd.DataFrame,
    experiment_html: str,
    worst_html: str,
    qc_html: str,
) -> None:
    error_chart = out_dir / "figures" / "method_error_comparison.png"
    make_error_bar(summary, error_chart)
    chart_html = f'<figure><img src="{error_chart.relative_to(out_dir).as_posix()}"><figcaption>MAE and RMSE for the citizen baseline, direct grey-corrected heuristic, and selected held-out model. Consult n because sample sets may differ.</figcaption></figure>' if error_chart.exists() else ""
    cards = ''.join(result_card(r, out_dir) for r in results)
    overview = html_table(summary, ["scenario", "n", "pearson_r", "CCC_agreement", "bias_mean_estimate_minus_lab", "MAE", "RMSE", "R2_direct_prediction", "bland_altman_lower_95", "bland_altman_upper_95"])
    mae_html = build_mae_rmse_html(summary)

    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
body{{margin:0;font-family:Arial,Helvetica,sans-serif;color:#1f2933;background:#f5f7fa;line-height:1.5}}
header{{background:#111827;color:white;padding:32px 40px}} header h1{{margin:0 0 8px}} header p{{margin:0;color:#d1d5db}}
main{{max-width:1240px;margin:auto;padding:28px 20px 60px}} .section-card{{background:white;border-radius:14px;padding:26px;margin-bottom:28px;box-shadow:0 2px 14px rgba(15,23,42,.08)}}
.metric-strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:12px;margin:18px 0}} .metric-strip div{{background:#f3f4f6;border-radius:10px;padding:12px}} .metric-strip span{{display:block;font-size:12px;color:#6b7280;margin-bottom:4px}} .metric-strip strong{{font-size:18px;overflow-wrap:anywhere}}
.figure-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px;margin-top:16px}} figure{{margin:0;background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:12px}} figure img{{width:100%;display:block;border-radius:8px;background:white}} figcaption{{font-size:13px;color:#4b5563;margin-top:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px;overflow-x:auto;display:block}} th,td{{border:1px solid #e5e7eb;padding:8px 10px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{background:#f3f4f6;color:#374151}} .selected-row td{{background:#ecfdf5}}
.info-box{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:14px 16px;color:#1e3a8a}} .qc-sample-card{{border:1px solid #e5e7eb;border-radius:12px;padding:18px;margin-top:22px}} .missing-asset{{background:#fef2f2;border-color:#fecaca}} .missing-box{{height:220px;border-radius:8px;background:repeating-linear-gradient(45deg,#fee2e2,#fee2e2 10px,#fecaca 10px,#fecaca 20px);display:flex;align-items:center;justify-content:center;color:#991b1b;font-weight:bold}} code{{background:#f3f4f6;padding:2px 5px;border-radius:4px}}
</style></head><body><header><h1>{html.escape(title)}</h1><p>Grey-scale-corrected model experiment, held-out validation, external baselines, and image-processing quality control.</p></header><main>
<section class="section-card"><h2>Executive summary</h2><ol>
<li>How do citizen-science orgC estimates compare with laboratory results?</li>
<li>How does the direct grey-corrected SOC formula perform as a non-trained baseline?</li>
<li>Which grey-scale image model is selected by repeated cross-validation, and how does it perform on the untouched final test set?</li>
<li>Are grey-scale detection, correction, soil ROI, and colour outputs visually plausible?</li>
</ol><div class="info-box">Model training and selection are performed only by <code>run_model_experiment.py</code>. This report does not retrain models and does not use final-test performance to select the winner.</div></section>
{experiment_html}
<section class="section-card"><h2>Metrics overview</h2>{overview}<h3>Error comparison</h3>{chart_html}</section>
{cards}{worst_html}{mae_html}{qc_html}
<section class="section-card"><h2>How to present the current status</h2><ul>
<li>The primary statistical result is the selected grey-scale model's performance on the untouched final test set.</li>
<li>Repeated cross-validation is used for model selection; final-test metrics are used only for final evaluation.</li>
<li>The direct SOC formula and citizen-science estimate are baselines, not inputs to model selection.</li>
<li>Interpolated Munsell features add predictive value only if their CV performance improves on continuous Lab features.</li>
<li>Visual QC remains necessary because an incorrect grey-scale crop or soil ROI can invalidate numerical results.</li>
</ul></section></main></body></html>'''
    (out_dir / "index.html").write_text(document, encoding="utf-8")


def run_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    lab = normalize_columns(read_table(args.lab))
    with_gray = normalize_columns(read_table(args.with_gray))
    experiment = load_model_experiment(args.experiment_dir)
    selected_result, selected_row = build_selected_model_comparison(experiment["summary"], experiment["predictions"])

    results: list[ComparisonResult] = []
    citizen = build_citizen_comparison(lab, args.lab_col, args.citizen_col)
    if citizen is not None:
        results.append(citizen)
    else:
        print(f"WARNING: citizen comparison skipped; missing {args.lab_col!r} or {args.citizen_col!r}.", file=sys.stderr)

    direct = build_direct_gray_comparison(with_gray, args.lab_col, args.gray_estimate_col)
    if direct is not None:
        results.append(direct)
    else:
        print(f"WARNING: direct grey-scale baseline skipped; missing {args.lab_col!r} or {args.gray_estimate_col!r}.", file=sys.stderr)

    results.append(selected_result)
    for result in results:
        result.figures = create_figures(result, figures_dir)
        write_predictions(result, out_dir)

    summary = pd.DataFrame([metrics_summary_row(r) for r in results])
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    experiment["summary"].to_csv(out_dir / "model_experiment_summary_with_gray.csv", index=False)
    experiment["predictions"].to_csv(out_dir / "model_experiment_predictions_with_gray.csv", index=False)
    experiment["split"].to_csv(out_dir / "model_experiment_split.csv", index=False)

    experiment_html = build_experiment_html(experiment["summary"], experiment["split"], selected_row)
    worst_html = build_worst_errors_html(experiment["predictions"], str(selected_row["model"]), args.worst_error_count)
    qc_codes = collect_qc_codes(args.qc_sample_codes, with_gray, args.qc_max_samples)
    qc_html = build_qc_html(out_dir, qc_codes, Path(args.debug_masks_dir), Path(args.debug_gray_dir), Path(args.color_cards_with_gray_dir))

    render_report(out_dir, args.title, results, summary, experiment_html, worst_html, qc_html)
    print(f"\nReport created: {out_dir / 'index.html'}")
    print(f"Summary metrics: {out_dir / 'summary_metrics.csv'}")
    print(f"Selected model: {selected_row['model']} (CV RMSE={fmt(selected_row.get('cv_RMSE_mean'))}, test RMSE={fmt(selected_row.get('test_RMSE'))})")
    print(f"\nOpen locally with:\n  python3 -m http.server 8088 --directory {out_dir}\n\nThen open:\n  http://localhost:8088")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a grey-scale-only orgC model and QC report.")
    parser.add_argument("--lab", default="data/lab/test_stat_orgC.xlsx", help="Laboratory file containing orgC_lab and optionally orgC_CS.")
    parser.add_argument("--with-gray", default="outputs/test_stat_orgC_enriched_with_gray.xlsx", help="Enriched grey-scale-corrected image results.")
    parser.add_argument("--experiment-dir", default="outputs/model_experiment", help="Directory containing run_model_experiment.py outputs.")
    parser.add_argument("--out", default="outputs/presentation_report", help="Output directory for static HTML report.")
    parser.add_argument("--lab-col", default="orgC_lab")
    parser.add_argument("--citizen-col", default="orgC_CS")
    parser.add_argument("--gray-estimate-col", default="SOC_est%")
    parser.add_argument("--title", default="Grey-scale soil orgC model report")
    parser.add_argument("--qc-sample-codes", nargs="*", default=None, help="Example: --qc-sample-codes APKC HGCM XGXK")
    parser.add_argument("--qc-max-samples", type=int, default=8)
    parser.add_argument("--debug-masks-dir", default="debug_masks")
    parser.add_argument("--debug-gray-dir", default="debug_gray")
    parser.add_argument("--color-cards-with-gray-dir", default="outputs/color_cards_with_gray")
    parser.add_argument("--worst-error-count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    run_report(parse_args())


if __name__ == "__main__":
    main()
