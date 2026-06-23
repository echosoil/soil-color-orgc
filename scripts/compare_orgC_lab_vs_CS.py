import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


INPUT_XLSX = "outputs/test_stat_orgC_enriched_with_gray.xlsx"
OUTPUT_DIR = "outputs/orgC_comparison_outputs"

LAB_COL = "orgC_lab"
# CS_COL = "orgC_CS"
CS_COL = "SOC_est%"
ID_COL = "ID"


def to_numeric_clean(series):
    """
    Convert values to numeric, accepting either decimal dots or decimal commas.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def concordance_correlation_coefficient(x, y):
    """
    Lin's concordance correlation coefficient.
    Measures agreement, not only correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2)
    return ccc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(INPUT_XLSX)

    required = {ID_COL, LAB_COL, CS_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[LAB_COL] = to_numeric_clean(df[LAB_COL])
    df[CS_COL] = to_numeric_clean(df[CS_COL])

    clean = df.dropna(subset=[LAB_COL, CS_COL]).copy()

    if clean.empty:
        raise ValueError("No valid rows after removing missing lab/CS values.")

    y_true = clean[LAB_COL].values
    y_pred = clean[CS_COL].values

    error = y_pred - y_true
    abs_error = np.abs(error)

    clean["error_CS_minus_lab"] = error
    clean["abs_error"] = abs_error
    clean["mean_lab_CS"] = (y_true + y_pred) / 2

    n = len(clean)

    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_direct = r2_score(y_true, y_pred)

    bias = np.mean(error)
    median_error = np.median(error)
    median_abs_error = np.median(abs_error)

    sd_error = np.std(error, ddof=1)
    loa_lower = bias - 1.96 * sd_error
    loa_upper = bias + 1.96 * sd_error

    ccc = concordance_correlation_coefficient(y_true, y_pred)

    # Calibration regression: lab = intercept + slope * citizen estimate
    X = y_pred.reshape(-1, 1)
    model = LinearRegression().fit(X, y_true)
    calibrated = model.predict(X)
    r2_calibration = r2_score(y_true, calibrated)

    stats = {
        "n": n,
        "lab_mean": np.mean(y_true),
        "lab_sd": np.std(y_true, ddof=1),
        "CS_mean": np.mean(y_pred),
        "CS_sd": np.std(y_pred, ddof=1),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "CCC_agreement": ccc,
        "bias_mean_CS_minus_lab": bias,
        "median_error_CS_minus_lab": median_error,
        "MAE": mae,
        "median_absolute_error": median_abs_error,
        "RMSE": rmse,
        "R2_direct_prediction": r2_direct,
        "bland_altman_lower_95": loa_lower,
        "bland_altman_upper_95": loa_upper,
        "calibration_intercept_lab_from_CS": model.intercept_,
        "calibration_slope_lab_from_CS": model.coef_[0],
        "R2_calibration_regression": r2_calibration,
    }

    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(OUTPUT_DIR, "orgC_lab_vs_CS_statistics.csv")
    clean_path = os.path.join(OUTPUT_DIR, "orgC_lab_vs_CS_clean_data.csv")

    stats_df.to_csv(stats_path, index=False)
    clean.to_csv(clean_path, index=False)

    print("\n=== Organic carbon: lab vs citizen estimate ===")
    print(f"N: {n}")
    print(f"Pearson r: {pearson_r:.3f}  p={pearson_p:.4g}")
    print(f"Spearman r: {spearman_r:.3f}  p={spearman_p:.4g}")
    print(f"CCC agreement: {ccc:.3f}")
    print(f"Bias, CS - lab: {bias:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R² direct prediction: {r2_direct:.3f}")
    print(f"Bland-Altman 95% limits: {loa_lower:.3f} to {loa_upper:.3f}")
    print(
        "Calibration formula: "
        f"orgC_lab ≈ {model.intercept_:.3f} + {model.coef_[0]:.3f} * orgC_CS"
    )
    print(f"Calibration R²: {r2_calibration:.3f}")

    # ---------------- Scatter plot ----------------
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.75)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="1:1 line")

    plt.xlabel("Laboratory orgC (%)")
    plt.ylabel("Citizen estimate orgC (%)")
    plt.title("Laboratory organic carbon vs citizen estimate")
    plt.legend()
    plt.tight_layout()

    scatter_path = os.path.join(OUTPUT_DIR, "scatter_lab_vs_CS.png")
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    # ---------------- Residual plot ----------------
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, error, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.axhline(bias, linestyle="-", label=f"Mean bias = {bias:.2f}")

    plt.xlabel("Laboratory orgC (%)")
    plt.ylabel("Error: citizen - lab (%)")
    plt.title("Citizen estimate error vs laboratory orgC")
    plt.legend()
    plt.tight_layout()

    residual_path = os.path.join(OUTPUT_DIR, "error_vs_lab.png")
    plt.savefig(residual_path, dpi=200)
    plt.close()

    # ---------------- Bland-Altman plot ----------------
    plt.figure(figsize=(7, 6))
    plt.scatter(clean["mean_lab_CS"], error, alpha=0.75)

    plt.axhline(bias, linestyle="-", label=f"Bias = {bias:.2f}")
    plt.axhline(loa_lower, linestyle="--", label=f"Lower 95% = {loa_lower:.2f}")
    plt.axhline(loa_upper, linestyle="--", label=f"Upper 95% = {loa_upper:.2f}")

    plt.xlabel("Mean of lab and citizen orgC (%)")
    plt.ylabel("Difference: citizen - lab (%)")
    plt.title("Bland-Altman plot")
    plt.legend()
    plt.tight_layout()

    ba_path = os.path.join(OUTPUT_DIR, "bland_altman.png")
    plt.savefig(ba_path, dpi=200)
    plt.close()

    print("\nSaved outputs:")
    print(stats_path)
    print(clean_path)
    print(scatter_path)
    print(residual_path)
    print(ba_path)


if __name__ == "__main__":
    main()