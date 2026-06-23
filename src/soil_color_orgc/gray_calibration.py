import os
import cv2
import numpy as np
import pandas as pd


def crop_fixed_gray_scale_roi(
    image_bgr,
    x_min=0.12,
    x_max=0.80,
    y_min=0.73,
    y_max=0.84,
):
    """
    Fallback fixed crop for the grey-scale strip.
    """
    h, w = image_bgr.shape[:2]

    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)

    roi = image_bgr[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2)


def crop_gray_search_region(
    image_bgr,
    x_min=0.08,
    x_max=0.92,
    y_min=0.65,
    y_max=0.92,
):
    """
    Broad search region where the grey scale is expected.

    y is measured from the top of the image.
    """
    h, w = image_bgr.shape[:2]

    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)

    search = image_bgr[y1:y2, x1:x2].copy()

    return search, (x1, y1, x2, y2)


def score_gray_candidate(candidate_bgr):
    """
    Score whether a candidate crop looks like a grey-scale strip.

    Good candidate:
        - low saturation overall
        - wide brightness range, because it goes dark -> light
        - approximately neutral
    """
    if candidate_bgr is None or candidate_bgr.size == 0:
        return -1

    h, w = candidate_bgr.shape[:2]

    if h < 10 or w < 50:
        return -1

    hsv = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    rgb = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    median_saturation = float(np.median(S))
    brightness_range = float(np.percentile(V, 95) - np.percentile(V, 5))

    # Neutrality: greys should have R, G, B close together.
    channel_spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    median_channel_spread = float(np.median(channel_spread))

    aspect = w / max(h, 1)

    # Prefer long horizontal strips.
    aspect_score = min(aspect / 6.0, 1.5)

    # Prefer strong dark-to-light range.
    brightness_score = brightness_range / 255.0

    # Penalize colourful candidates.
    saturation_penalty = median_saturation / 255.0

    # Penalize non-neutral RGB.
    spread_penalty = median_channel_spread / 255.0

    score = (
        2.0 * aspect_score
        + 3.0 * brightness_score
        - 1.5 * saturation_penalty
        - 1.0 * spread_penalty
    )

    return float(score)


def detect_gray_scale_rectangle(image_bgr):
    """
    Detect a long horizontal grey-scale rectangle in the lower part of the image.

    Returns:
        roi_bgr, rect, debug_mask

    where rect is:
        x1, y1, x2, y2
    in full-image coordinates.
    """
    search_bgr, search_rect = crop_gray_search_region(image_bgr)

    sx1, sy1, sx2, sy2 = search_rect

    hsv = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)

    # The grey card and white paper are low-saturation.
    # This removes the green table and most soil.
    neutral_mask = (S < 75).astype(np.uint8) * 255

    # Edge detection catches rectangle edges and patch boundaries.
    edges = cv2.Canny(gray, 40, 120)

    # Keep mainly neutral edges.
    edges = cv2.bitwise_and(edges, neutral_mask)

    # Join patch boundaries and rectangle edges into larger components.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    closed = cv2.dilate(closed, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    search_h, search_w = search_bgr.shape[:2]
    image_h, image_w = image_bgr.shape[:2]

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w <= 0 or h <= 0:
            continue

        aspect = w / h
        area = w * h

        # Basic geometry filters.
        if aspect < 4.0:
            continue

        if w < search_w * 0.35:
            continue

        if h < image_h * 0.025:
            continue

        if h > image_h * 0.18:
            continue

        if area < image_w * image_h * 0.005:
            continue

        # Expand a little, but not too much.
        pad_x = int(w * 0.03)
        pad_y = int(h * 0.08)

        xx1 = max(0, x - pad_x)
        yy1 = max(0, y - pad_y)
        xx2 = min(search_w, x + w + pad_x)
        yy2 = min(search_h, y + h + pad_y)

        candidate = search_bgr[yy1:yy2, xx1:xx2].copy()

        score = score_gray_candidate(candidate)

        candidates.append({
            "score": score,
            "roi": candidate,
            "rect": (
                sx1 + xx1,
                sy1 + yy1,
                sx1 + xx2,
                sy1 + yy2,
            ),
            "local_rect": (xx1, yy1, xx2, yy2),
            "aspect": aspect,
            "area": area,
        })

    if not candidates:
        return None, None, closed

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)

    best = candidates[0]

    # Require at least some confidence.
    if best["score"] < 1.0:
        return None, None, closed

    return best["roi"], best["rect"], closed


def crop_fixed_gray_scale_roi(
    image_bgr,
    x_min=0.10,
    x_max=0.90,
    y_min=0.75,
    y_max=0.95,
):
    """
    Fallback fixed crop for the grey-scale strip.
    """
    h, w = image_bgr.shape[:2]

    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)

    roi = image_bgr[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2)


def detect_gray_scale_by_horizontal_brightness_profile(
    image_bgr,
    x_min=0.08,
    x_max=0.92,
    y_min=0.62,
    y_max=0.96,
    band_height_frac=0.10,
):
    """
    Detect grey scale using the fact that it has strong left-to-right
    brightness variation: dark patches on one side, light patches on the other.

    This avoids confusing it with a long white paper rectangle.
    """
    h, w = image_bgr.shape[:2]

    sx1 = int(w * x_min)
    sx2 = int(w * x_max)
    sy1 = int(h * y_min)
    sy2 = int(h * y_max)

    search = image_bgr[sy1:sy2, sx1:sx2].copy()

    if search.size == 0:
        return None, None

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    search_h, search_w = search.shape[:2]

    band_h = max(20, int(h * band_height_frac))
    band_h = min(band_h, search_h)

    best_score = -1
    best_rect_local = None

    # Slide vertical band down the lower part of the image.
    step = max(4, band_h // 10)

    for y in range(0, search_h - band_h + 1, step):
        band_s = s[y:y + band_h, :]
        band_v = v[y:y + band_h, :]

        # Use central vertical part of the band to avoid labels/ticks.
        cy1 = int(band_h * 0.20)
        cy2 = int(band_h * 0.80)

        core_s = band_s[cy1:cy2, :]
        core_v = band_v[cy1:cy2, :]

        # Median brightness per column.
        col_v = np.median(core_v, axis=0)

        # Median saturation. Grey scale should be low saturation.
        median_s = float(np.median(core_s))

        # Main grey-scale clue: big dark-to-light range across x.
        horizontal_range = float(np.percentile(col_v, 95) - np.percentile(col_v, 5))

        # Also useful: many vertical boundaries between patches.
        col_gradient = np.abs(np.diff(col_v))
        gradient_score = float(np.percentile(col_gradient, 90))

        # Penalize high saturation regions, e.g. green table / soil.
        saturation_penalty = median_s

        score = (
            2.5 * horizontal_range
            + 2.0 * gradient_score
            - 1.2 * saturation_penalty
        )

        if score > best_score:
            best_score = score
            best_rect_local = (0, y, search_w, y + band_h)

    if best_rect_local is None:
        return None, None

    lx1, ly1, lx2, ly2 = best_rect_local

    # Convert local search coordinates to full image coordinates.
    x1 = sx1 + lx1
    x2 = sx1 + lx2
    y1 = sy1 + ly1
    y2 = sy1 + ly2

    # Tighten x range a bit: grey strip does not need full 8-92% width.
    # This removes left/right background.
    strip_w = x2 - x1
    x1 = x1 + int(strip_w * 0.06)
    x2 = x2 - int(strip_w * 0.06)

    # Basic confidence check.
    # If the best score is too low, fall back.
    if best_score < 40:
        return None, None

    roi = image_bgr[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2)


def crop_gray_scale_roi(image_bgr):
    """
    Main grey-scale ROI function.

    First tries brightness-profile detection.
    If detection fails, uses fixed fallback crop.
    """
    detected_roi, detected_rect = detect_gray_scale_by_horizontal_brightness_profile(
        image_bgr,
        x_min=0.08,
        x_max=0.92,
        y_min=0.62,
        y_max=0.96,
        band_height_frac=0.11,
    )

    if detected_roi is not None:
        return detected_roi, detected_rect

    return crop_fixed_gray_scale_roi(
        image_bgr,
        x_min=0.10,
        x_max=0.90,
        y_min=0.75,
        y_max=0.95,
    )


def split_gray_patches(gray_roi_bgr, n_patches=11):
    """
    Split grey scale into patches from left to right.

    For each patch, use only the central area to avoid:
        - borders
        - tick marks
        - printed labels
        - transitions between patches
    """
    h, w = gray_roi_bgr.shape[:2]

    patches = []

    for i in range(n_patches):
        x1 = int(w * i / n_patches)
        x2 = int(w * (i + 1) / n_patches)

        patch = gray_roi_bgr[:, x1:x2].copy()

        ph, pw = patch.shape[:2]

        # central crop inside patch
        cx1 = int(pw * 0.30)
        cx2 = int(pw * 0.70)
        cy1 = int(ph * 0.25)
        cy2 = int(ph * 0.65)

        patch_center = patch[cy1:cy2, cx1:cx2].copy()

        patches.append({
            "patch_index": i,
            "patch_bgr": patch_center,
            "x1": x1,
            "x2": x2,
            "cx1": x1 + cx1,
            "cx2": x1 + cx2,
            "cy1": cy1,
            "cy2": cy2,
        })

    return patches


def measure_patch_rgb(patch_bgr):
    """
    Robust median RGB measurement.
    """
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    # remove extreme outliers inside patch
    low = np.quantile(pixels, 0.05, axis=0)
    high = np.quantile(pixels, 0.95, axis=0)

    keep = np.all((pixels >= low) & (pixels <= high), axis=1)

    if keep.sum() > 20:
        pixels = pixels[keep]

    med = np.median(pixels, axis=0)

    return med  # RGB


def neutral_target_from_rgb(rgb):
    """
    Target neutral grey.

    This preserves approximate luminance while forcing neutrality.
    """
    r, g, b = rgb

    # Standard luminance weights for sRGB-like data.
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return float(y)


def build_channel_lut(measured_values, target_values):
    """
    Build a 0..255 lookup table mapping measured channel values to target grey values.

    Uses piecewise linear interpolation.
    """
    measured_values = np.asarray(measured_values, dtype=np.float32)
    target_values = np.asarray(target_values, dtype=np.float32)

    # Add anchors so values outside measured patch range do not explode.
    x = np.concatenate([[0.0], measured_values, [255.0]])
    y = np.concatenate([[0.0], target_values, [255.0]])

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove duplicate x values for np.interp.
    unique_x = []
    unique_y = []

    for val in np.unique(x):
        idx = np.where(x == val)[0]
        unique_x.append(val)
        unique_y.append(np.mean(y[idx]))

    unique_x = np.asarray(unique_x, dtype=np.float32)
    unique_y = np.asarray(unique_y, dtype=np.float32)

    lut = np.interp(np.arange(256), unique_x, unique_y)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut


def apply_luts_to_bgr(image_bgr, lut_r, lut_g, lut_b):
    """
    Apply RGB LUTs to OpenCV BGR image.
    """
    b, g, r = cv2.split(image_bgr)

    r2 = cv2.LUT(r, lut_r)
    g2 = cv2.LUT(g, lut_g)
    b2 = cv2.LUT(b, lut_b)

    return cv2.merge([b2, g2, r2])


def correct_image_using_gray_scale(
    image_bgr,
    n_patches=11,
    debug_dir=None,
    image_name=None,
):
    """
    Correct colour balance using the grey scale at the bottom of the image.

    The correction is based on neutralizing each grey patch:
        measured R/G/B -> neutral grey target

    Returns:
        corrected_bgr, report_df
    """
    gray_roi_bgr, rect = crop_gray_scale_roi(image_bgr)

    if gray_roi_bgr.size == 0:
        raise ValueError("Empty grey-scale ROI")

    patches = split_gray_patches(gray_roi_bgr, n_patches=n_patches)

    measured_rgbs = []
    targets = []
    rows = []

    for patch in patches:
        rgb = measure_patch_rgb(patch["patch_bgr"])
        target = neutral_target_from_rgb(rgb)

        measured_rgbs.append(rgb)
        targets.append(target)

        rows.append({
            "image": image_name,
            "patch_index": patch["patch_index"],
            "measured_R": float(rgb[0]),
            "measured_G": float(rgb[1]),
            "measured_B": float(rgb[2]),
            "target_gray": float(target),
            "imbalance_before": float(max(rgb) - min(rgb)),
        })

    measured_rgbs = np.asarray(measured_rgbs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    lut_r = build_channel_lut(measured_rgbs[:, 0], targets)
    lut_g = build_channel_lut(measured_rgbs[:, 1], targets)
    lut_b = build_channel_lut(measured_rgbs[:, 2], targets)

    corrected_bgr = apply_luts_to_bgr(image_bgr, lut_r, lut_g, lut_b)

    # Measure grey patches after correction for diagnostics.
    corrected_gray_roi_bgr = corrected_bgr[
        rect[1]:rect[3],
        rect[0]:rect[2],
    ].copy()

    corrected_patches = split_gray_patches(
        corrected_gray_roi_bgr,
        n_patches=n_patches,
    )

    for row, patch in zip(rows, corrected_patches):
        rgb_after = measure_patch_rgb(patch["patch_bgr"])

        row["after_R"] = float(rgb_after[0])
        row["after_G"] = float(rgb_after[1])
        row["after_B"] = float(rgb_after[2])
        row["imbalance_after"] = float(max(rgb_after) - min(rgb_after))

    report_df = pd.DataFrame(rows)

    if debug_dir and image_name:
        save_gray_calibration_debug(
            original_bgr=image_bgr,
            corrected_bgr=corrected_bgr,
            gray_roi_bgr=gray_roi_bgr,
            rect=rect,
            report_df=report_df,
            debug_dir=debug_dir,
            image_name=image_name,
        )

    return corrected_bgr, report_df


def save_gray_calibration_debug(
    original_bgr,
    corrected_bgr,
    gray_roi_bgr,
    rect,
    report_df,
    debug_dir,
    image_name,
):
    os.makedirs(debug_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(image_name))[0]

    x1, y1, x2, y2 = rect

    marked = original_bgr.copy()
    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 0), 4)

    before_small = resize_preview(original_bgr)
    after_small = resize_preview(corrected_bgr)

    before_after = np.hstack([before_small, after_small])

    cv2.putText(
        before_after,
        "before",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        before_after,
        "after grey-scale correction",
        (before_small.shape[1] + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(os.path.join(debug_dir, f"{stem}_gray_roi_rect.jpg"), marked)
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_gray_roi.jpg"), gray_roi_bgr)
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_gray_before_after.jpg"), before_after)

    report_df.to_csv(
        os.path.join(debug_dir, f"{stem}_gray_report.csv"),
        index=False,
    )


def resize_preview(img_bgr, width=500):
    h, w = img_bgr.shape[:2]
    scale = width / w
    new_w = width
    new_h = int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)