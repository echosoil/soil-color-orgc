"""Paired, offline validation of the deployed feature extraction method.

No model artifact is exported or promoted. Groups must be assigned by the
operator using scientific knowledge, not inferred from row order.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .analysis import analyze_image
from .service_image import decode_image, MAX_BYTES


def parse_region(value):
    if not value or not value.strip():
        return None
    r = tuple(float(x) for x in value.split(","))
    if len(r) != 4 or not all(math.isfinite(x) for x in r) or not (
        0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1
    ):
        raise ValueError("Invalid normalized ROI")
    return r


def read_manifest(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "group_id", "image_path", "orgC_lab"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires sample_id, group_id, image_path, orgC_lab")
        rows = []
        seen = set()
        for line, row in enumerate(reader, 2):
            try:
                sample, group, image = (row[k].strip() for k in ("sample_id", "group_id", "image_path"))
                y = float(row["orgC_lab"])
                if not sample or not group or not image or sample in seen or not math.isfinite(y) or not 0 <= y <= 100:
                    raise ValueError("Blank/duplicate ID, blank group/path or invalid lab percentage")
                seen.add(sample)
                rows.append({"sample_id": sample, "group_id": group, "orgC_lab": y,
                             "image_path": path.parent / image,
                             "soil_roi": parse_region(row.get("soil_roi")),
                             "gray_roi": parse_region(row.get("gray_roi"))})
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid manifest row {line}: {exc}") from exc
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def extract_pairs(rows, reference):
    """One row per physical sample, identical input bytes and requested soil ROI."""
    records = []
    seen_hashes = set()
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".jfif": "image/jpeg",
             ".png": "image/png", ".webp": "image/webp"}
    for row in rows:
        record = {k: row[k] for k in ("sample_id", "group_id", "orgC_lab", "soil_roi", "gray_roi")}
        record["included"] = False
        records.append(record)
        try:
            with row["image_path"].open("rb") as handle:
                payload = handle.read(MAX_BYTES + 1)
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) > MAX_BYTES:
                record["reason"] = "image_too_large"
                continue
            record["image_sha256"] = digest
            if digest in seen_hashes:
                raise DuplicatePhoto("Duplicate image bytes; resolve physical sample identity before analysis")
            seen_hashes.add(digest)
            im, decoding = decode_image(payload, media.get(row["image_path"].suffix.lower(), ""))
            raw = analyze_image(im, reference, roi=row["soil_roi"])
            corrected = analyze_image(im, reference, roi=row["soil_roi"], gray=True, gray_roi=row["gray_roi"])
            record.update({"raw": raw, "corrected": corrected, "decoding": decoding})
            # No Munsell match does not invalidate continuous Lab features.
            acceptable = {"no_close_munsell_match"}
            if raw["lab"] is None or corrected["lab"] is None:
                record["reason"] = "insufficient_soil_pixels"
            elif corrected["gray_calibration"] != "applied":
                record["reason"] = "gray_not_applied"
            elif (set(raw["warnings"]) | set(corrected["warnings"])) - acceptable:
                record["reason"] = "quality_review"
            else:
                record["included"] = True
                record["reason"] = "paired_eligible"
        except DuplicatePhoto:
            raise
        except (OSError, ValueError, TypeError) as exc:
            record["reason"] = "image_or_analysis_error"
            record["error_type"] = type(exc).__name__
    return records


class DuplicatePhoto(ValueError):
    pass


def metrics(y, pred):
    return {"mae_pct_points": float(mean_absolute_error(y, pred)),
            "rmse_pct_points": float(np.sqrt(mean_squared_error(y, pred))),
            "bias_pct_points": float(np.mean(pred-y)),
            "r2": float(r2_score(y, pred)) if np.ptp(y) > 0 else None}


def evaluate_pairs(records):
    paired = [r for r in records if r["included"]]
    groups = np.asarray([r["group_id"] for r in paired])
    summary = {"total_samples": len(records), "paired_samples": len(paired),
               "paired_groups": len(set(groups)), "dispositions": dict(Counter(r["reason"] for r in records)),
               "status": "insufficient_paired_data", "metrics": {}}
    if len(paired) < 20 or len(set(groups)) < 5:
        return summary, []
    y = np.asarray([r["orgC_lab"] for r in paired], dtype=float)
    raw = np.asarray([r["raw"]["lab"] for r in paired], dtype=float)
    gray = np.asarray([r["corrected"]["lab"] for r in paired], dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(raw).all() or not np.isfinite(gray).all():
        raise ValueError("Nonfinite paired features")
    if np.ptp(y) == 0:
        summary["status"] = "constant_lab_target"
        return summary, []
    models = {"mean_baseline": (raw[:, :1], DummyRegressor),
              "raw_L": (raw[:, :1], LinearRegression),
              "gray_L": (gray[:, :1], LinearRegression),
              "raw_Lab": (raw, LinearRegression),
              "gray_Lab": (gray, LinearRegression)}
    folds = list(GroupKFold(n_splits=5).split(raw, y, groups))
    if any(len(train) < 10 or np.ptp(y[train]) == 0 for train, test in folds):
        summary["status"] = "insufficient_training_fold"
        return summary, []
    preds = {name: np.full(len(y), np.nan) for name in models}
    fold_ids = np.full(len(y), -1)
    for i, (train, test) in enumerate(folds):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Group leakage")
        fold_ids[test] = i + 1
        for name, (x, factory) in models.items():
            estimator = factory()
            estimator.fit(x[train], y[train])
            preds[name][test] = estimator.predict(x[test])
    if not all(np.isfinite(p).all() for p in preds.values()) or np.any(fold_ids < 1):
        raise RuntimeError("Incomplete validation predictions")
    summary["status"] = "exploratory_grouped_cv_completed"
    summary["metrics"] = {name: metrics(y, p) for name, p in preds.items()}
    summary["paired_gray_minus_raw_mae_pct_points"] = {
        key: summary["metrics"]["gray_"+key]["mae_pct_points"] -
             summary["metrics"]["raw_"+key]["mae_pct_points"] for key in ("L", "Lab")
    }
    summary["lab_range_pct"] = [float(y.min()), float(y.max())]
    summary["fold_sizes"] = [int(np.sum(fold_ids == i)) for i in range(1,6)]
    predictions = [{"sample_id": row["sample_id"], "group_id": row["group_id"],
                    "fold": int(fold_ids[i]), "orgC_lab": float(y[i]),
                    **{name: float(p[i]) for name, p in preds.items()}}
                   for i, row in enumerate(paired)]
    return summary, predictions


def write_results(out, summary, records, predictions):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    for filename, data in [("validation-summary.json", summary), ("paired-features.local.json", records),
                           ("predictions.local.json", predictions)]:
        with (out/filename).open("w", encoding="utf-8") as h:
            json.dump(data, h, indent=2, allow_nan=False)
            h.write("\n")
