#!/usr/bin/env python3
# Must be run from the project root.

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict

in_file = Path("outputs/test_stat_orgC_enriched_no_gray.xlsx")
out_file = Path("outputs/test_stat_orgC_enriched_no_gray_cv_Lab_prediction.xlsx")

df = pd.read_excel(in_file, sheet_name="enriched")

if "processing_status" in df.columns:
    status = df["processing_status"].astype(str).str.lower().str.strip()
    df = df[status.isin(["ok", "nan", "", "none"])].copy()

required = ["orgC_lab", "L", "a", "b"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

for c in required:
    df[c] = pd.to_numeric(df[c], errors="coerce")

clean = df.dropna(subset=required).copy()

if len(clean) < 3:
    raise SystemExit(f"Not enough rows for cross-validation: {len(clean)}")

X = clean[["L", "a", "b"]].to_numpy(float)
y = clean["orgC_lab"].to_numpy(float)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression()),
])

cv = LeaveOneOut() if len(clean) <= 50 else KFold(n_splits=5, shuffle=True, random_state=42)

pred = cross_val_predict(model, X, y, cv=cv)

clean["orgC_pred_cv_Lab"] = pred
clean["error_cv_pred_minus_lab"] = clean["orgC_pred_cv_Lab"] - clean["orgC_lab"]
clean["abs_error_cv_pred"] = clean["error_cv_pred_minus_lab"].abs()

with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    clean.to_excel(writer, sheet_name="enriched", index=False)

print(f"Created {out_file}")
print(f"Rows used: {len(clean)}")