import os
import cv2
import numpy as np
from colour import sRGB_to_XYZ, XYZ_to_Lab
from .gray_calibration import correct_image_using_gray_scale, locate_gray_scale_roi
from .image_io import read_image_bgr

def ensure_portrait_orientation(image_bgr):
    """
    Ensure image is portrait.

    In our dataset, valid sample images should have:
        height > width

    If width > height, rotate 90 degrees.

    Note:
        This does not 'swap x and y' mathematically.
        It rotates the image so that x remains width and y remains height.
    """
    height, width = image_bgr.shape[:2]

    if width > height:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)

    return image_bgr


def downscale_max_side(image_bgr, max_side=1200):
    """
    Downscale large images for faster processing.
    Keeps aspect ratio.
    """
    h, w = image_bgr.shape[:2]
    current_max = max(h, w)

    if current_max <= max_side:
        return image_bgr

    scale = max_side / current_max
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_central_soil_roi(
    image_bgr,
    x_min=0.40,
    x_max=0.60,
    y_min=0.50,
    y_max=0.70,
):
    """
    Fixed central crop.

    Default:
        x: 40% to 60%
        y: 50% to 70%

    This intentionally ignores most borders, labels and background.
    """
    h, w = image_bgr.shape[:2]

    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)

    roi = image_bgr[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2)


def build_simple_roi_mask(
    roi_bgr,
    min_v=20,
    max_v=245,
    white_v=190,
    white_s=70,
    white_min_rgb=180,
    white_spread=45,
):
    """
    Simple mask inside the fixed ROI.

    Goal:
        keep soil-like pixels,
        remove white/pale paper and very dark shadow holes.

    This version is stricter against paper than the previous one.
    """
    smoothed = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)

    _, s, v = cv2.split(hsv)

    rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb_min = np.min(rgb, axis=2)
    rgb_max = np.max(rgb, axis=2)
    rgb_spread = rgb_max - rgb_min

    not_too_dark = v > min_v
    not_too_bright = v < max_v

    white_like = (
        ((v > white_v) & (s < white_s))
        | ((rgb_min > white_min_rgb) & (rgb_spread < white_spread))
    )

    mask = not_too_dark & not_too_bright & (~white_like)

    mask_u8 = mask.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    return mask_u8


def bgr_pixels_to_lab(pixels_bgr):
    """
    Convert BGR uint8 pixels to CIE Lab using colour-science.

    This keeps the Lab calculation consistent with the Munsell conversion.
    """
    pixels_rgb = pixels_bgr[:, ::-1].astype(np.float32) / 255.0

    xyz = sRGB_to_XYZ(pixels_rgb)
    lab = XYZ_to_Lab(xyz)

    return lab


def trim_lab_by_lightness(lab_pixels, trim_percent=0.08):
    """
    Remove darkest and brightest pixels by L.

    This reduces influence of:
        - dark shadow crevices between grains
        - bright highlights / paper leakage
    """
    if lab_pixels.shape[0] < 50:
        return lab_pixels

    L = lab_pixels[:, 0]

    low = np.quantile(L, trim_percent)
    high = np.quantile(L, 1.0 - trim_percent)

    keep = (L >= low) & (L <= high)

    trimmed = lab_pixels[keep]

    if trimmed.shape[0] < 50:
        return lab_pixels

    return trimmed


def save_debug_images(
    image_bgr,
    roi_bgr,
    mask_u8,
    rect,
    image_path,
    debug_dir,
):
    """
    Save visual diagnostics:
        *_roi_rect.jpg   original image with selected ROI
        *_roi.jpg        selected ROI
        *_roi_mask.png   mask used inside ROI
        *_roi_used.jpg   ROI with rejected pixels marked red
    """
    os.makedirs(debug_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(image_path))[0]

    x1, y1, x2, y2 = rect

    marked = image_bgr.copy()
    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 0), 4)

    used = roi_bgr.copy()
    used[mask_u8 == 0] = (0, 0, 255)

    cv2.imwrite(os.path.join(debug_dir, f"{stem}_roi_rect.jpg"), marked)
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_roi.jpg"), roi_bgr)
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_roi_mask.png"), mask_u8)
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_roi_used.jpg"), used)


def trim_soil_roi_to_nonwhite_content(
    roi_bgr,
    rect,
    min_v=20,
    white_v=190,
    white_s=70,
    white_min_rgb=180,
    white_spread=45,
    min_row_fraction=0.35,
    min_col_fraction=0.20,
    pad_fraction=0.04,
):
    """
    Trim a soil ROI to remove white/pale paper strips.

    This is useful when the fixed crop includes part of the filter paper
    or tray edge. It trims rows/columns that are mostly white-like.

    Returns:
        trimmed_roi_bgr, trimmed_rect
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr, rect

    h, w = roi_bgr.shape[:2]

    if h < 20 or w < 20:
        return roi_bgr, rect

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb_min = np.min(rgb, axis=2)
    rgb_max = np.max(rgb, axis=2)
    rgb_spread = rgb_max - rgb_min

    # White / pale paper:
    #   - high brightness and low saturation
    #   - or all RGB channels high and close together
    white_like = (
        ((v > white_v) & (s < white_s))
        | ((rgb_min > white_min_rgb) & (rgb_spread < white_spread))
    )

    # Candidate soil-like pixels are not white-like and not extremely dark.
    soil_like = (~white_like) & (v > min_v)

    row_fraction = soil_like.mean(axis=1)
    col_fraction = soil_like.mean(axis=0)

    good_rows = np.where(row_fraction >= min_row_fraction)[0]
    good_cols = np.where(col_fraction >= min_col_fraction)[0]

    if len(good_rows) == 0 or len(good_cols) == 0:
        return roi_bgr, rect

    y1_local = int(good_rows[0])
    y2_local = int(good_rows[-1]) + 1

    x1_local = int(good_cols[0])
    x2_local = int(good_cols[-1]) + 1

    pad_y = int((y2_local - y1_local) * pad_fraction)
    pad_x = int((x2_local - x1_local) * pad_fraction)

    y1_local = max(0, y1_local - pad_y)
    y2_local = min(h, y2_local + pad_y)

    x1_local = max(0, x1_local - pad_x)
    x2_local = min(w, x2_local + pad_x)

    if y2_local <= y1_local or x2_local <= x1_local:
        return roi_bgr, rect

    trimmed_roi = roi_bgr[y1_local:y2_local, x1_local:x2_local].copy()

    x1, y1, x2, y2 = rect

    trimmed_rect = (
        x1 + x1_local,
        y1 + y1_local,
        x1 + x2_local,
        y1 + y2_local,
    )

    return trimmed_roi, trimmed_rect


def extract_color_features(
    image_bgr,
    *,
    roi=None,
    use_gray_calibration=False,
    gray_roi=None,
    downscale_max=800,
    trim_percent=0.05,
    strict=True,
    debug_dir=None,
    gray_debug_dir=None,
    image_name="image",
):
    """Return Lab and diagnostics. ROI is normalized in portrait-oriented space.

    Strict mode never falls back to unmasked pixels. The legacy batch wrapper
    explicitly opts out to preserve its historical calculation.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 32:
        raise ValueError("Expected a BGR image at least 32 by 32 pixels")
    if not 32 <= downscale_max <= 2000 or not 0 <= trim_percent < 0.5:
        raise ValueError("Invalid analysis configuration")
    if roi is None:
        roi = (0.40, 0.50, 0.60, 0.70)
    if len(roi) != 4 or not all(np.isfinite(v) for v in roi):
        raise ValueError("ROI must contain four finite coordinates")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("ROI must be ordered and inside the image")
    image_bgr = downscale_max_side(ensure_portrait_orientation(image_bgr), downscale_max)
    height, width = image_bgr.shape[:2]
    warnings = []
    calibration = "not_requested"
    calibration_metrics = None
    gray_region = None
    if gray_roi is not None and not use_gray_calibration:
        raise ValueError("Manual grey region requires grey correction")
    if use_gray_calibration:
        try:
            selection = None
            if strict or gray_roi is not None:
                selection = locate_gray_scale_roi(image_bgr, gray_roi)
                gray_region = selection[2]
                if not gray_region["usable"]:
                    warnings.append("gray_region_not_accepted")
                    raise ValueError("Grey region did not pass the ramp check")
            corrected, report = correct_image_using_gray_scale(
                image_bgr, n_patches=11, debug_dir=gray_debug_dir, image_name=image_name,
                selection=selection,
            )
            if len(report) != 11 or not np.isfinite(
                report[["imbalance_before", "imbalance_after"]].to_numpy()
            ).all():
                raise ValueError("Invalid gray reference diagnostics")
            calibration_metrics = {
                "mean_imbalance_before": float(report["imbalance_before"].mean()),
                "mean_imbalance_after": float(report["imbalance_after"].mean()),
            }
            if gray_region is not None:
                gray_region["patch_measurements"] = [
                    {k: float(row[k]) for k in (
                        "measured_R", "measured_G", "measured_B", "after_R",
                        "after_G", "after_B", "target_gray", "imbalance_before", "imbalance_after"
                    )} | {"scale_label": int(row["scale_label"])}
                    for _, row in report.iterrows()
                ]
            image_bgr = corrected
            calibration = "applied"
            if calibration_metrics["mean_imbalance_after"] >= calibration_metrics["mean_imbalance_before"]:
                warnings.append("gray_imbalance_not_improved")
        except Exception as exc:
            calibration = "failed_uncorrected"
            warnings.append("gray_calibration_failed")
            if not strict:
                print(f"  WARNING: grey-scale calibration failed for {image_name}: {exc}. Using uncorrected image.", flush=True)

    roi_bgr, rect = crop_central_soil_roi(
        image_bgr, x_min=x0, x_max=x1, y_min=y0, y_max=y1
    )
    roi_bgr, rect = trim_soil_roi_to_nonwhite_content(roi_bgr, rect)
    if roi_bgr.size == 0:
        raise ValueError("Empty soil ROI")
    mask = build_simple_roi_mask(roi_bgr)
    if debug_dir:
        save_debug_images(image_bgr, roi_bgr, mask, rect, image_name, debug_dir)
    blurred = cv2.GaussianBlur(roi_bgr, (9, 9), 0)
    selected = blurred[mask > 0]
    count = int(selected.shape[0])
    total = int(mask.size)
    diagnostics = {
        "image_width": width, "image_height": height,
        "roi_pixels": list(rect), "selected_pixels": count,
        "roi_pixel_count": total, "selected_fraction": count / total,
        "gray_calibration": calibration, "gray_metrics": calibration_metrics,
        "gray_region": gray_region,
        "warnings": warnings, "used_full_roi_fallback": False,
    }
    if count < 500 or (strict and count / total < 0.10):
        warnings.append("insufficient_soil_pixels")
        if strict:
            return {"status": "insufficient_soil_pixels", "lab": None, **diagnostics}
        selected = blurred.reshape(-1, 3)
        diagnostics["used_full_roi_fallback"] = True
    pixels = trim_lab_by_lightness(bgr_pixels_to_lab(selected), trim_percent)
    median = np.median(pixels, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("Nonfinite colour result")
    if median[0] > 75:
        warnings.append("high_lightness_check_roi")
        if not strict:
            print(f"  WARNING: very bright estimated soil color for {image_name}: L={median[0]:.2f}. Check ROI/mask.", flush=True)
    return {"status": "review" if warnings else "ok",
            "lab": tuple(float(x) for x in median), **diagnostics}


def extract_dominant_lab(
    image_path: str,
    save_debug_masks: bool = True,
    debug_dir: str = "debug_masks",
    downscale_max: int = 1200,
    kmeans_clusters: int = 4,
    trim_percent: float = 0.08,
    use_gray_calibration: bool = True,
):
    """Compatibility wrapper for the historical batch pipeline."""
    result = extract_color_features(
        read_image_bgr(image_path, use_icc=True),
        use_gray_calibration=use_gray_calibration,
        downscale_max=downscale_max,
        trim_percent=trim_percent,
        strict=False,
        debug_dir=debug_dir if save_debug_masks else None,
        gray_debug_dir="debug_gray" if save_debug_masks else None,
        image_name=os.path.basename(image_path),
    )
    return result["lab"]
