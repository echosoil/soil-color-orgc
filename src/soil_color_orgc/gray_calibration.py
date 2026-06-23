import os
import cv2
import numpy as np
import pandas as pd


def target_grays_from_black_scale(n_patches=11, left_label=10, right_label=0):
    labels = np.linspace(left_label, right_label, n_patches)
    black_fraction = labels / 10.0

    target_gray = 255.0 * (1.0 - black_fraction)

    # Avoid forcing extreme black/white too strongly.
    target_gray = np.clip(target_gray, 20, 235)

    return target_gray.astype(np.float32)

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
    x_min=0.08,
    x_max=0.88,
    y_min=0.78,
    y_max=0.91,
):
    h, w = image_bgr.shape[:2]

    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)

    roi = image_bgr[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2)


def score_gray_ramp_candidate(candidate_bgr, n_patches=11):
    """
    Score whether a candidate crop looks like the 10-to-0 grey scale.

    Expected:
        patch 0 is darkest
        patch 10 is lightest
        brightness increases from left to right
        patches are low saturation
    """
    if candidate_bgr is None or candidate_bgr.size == 0:
        return -1, {}

    h, w = candidate_bgr.shape[:2]

    if h < 20 or w < 120:
        return -1, {}

    hsv = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    patch_values = []
    patch_saturations = []
    patch_channel_spreads = []

    for i in range(n_patches):
        x1 = int(w * i / n_patches)
        x2 = int(w * (i + 1) / n_patches)

        patch_s = s[:, x1:x2]
        patch_v = v[:, x1:x2]
        patch_bgr = candidate_bgr[:, x1:x2]

        ph, pw = patch_v.shape[:2]

        # Use central patch region only.
        # Avoid borders, tick marks, labels, transitions.
        cx1 = int(pw * 0.30)
        cx2 = int(pw * 0.70)
        cy1 = int(ph * 0.25)
        cy2 = int(ph * 0.65)

        core_v = patch_v[cy1:cy2, cx1:cx2]
        core_s = patch_s[cy1:cy2, cx1:cx2]
        core_bgr = patch_bgr[cy1:cy2, cx1:cx2]

        if core_v.size == 0:
            continue

        rgb = cv2.cvtColor(core_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        channel_spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)

        patch_values.append(float(np.median(core_v)))
        patch_saturations.append(float(np.median(core_s)))
        patch_channel_spreads.append(float(np.median(channel_spread)))

    if len(patch_values) != n_patches:
        return -1, {}

    patch_values = np.asarray(patch_values, dtype=np.float32)
    patch_saturations = np.asarray(patch_saturations, dtype=np.float32)
    patch_channel_spreads = np.asarray(patch_channel_spreads, dtype=np.float32)

    # Expected left-to-right brightness increase:
    # label 10 is black, label 0 is white.
    expected = np.arange(n_patches, dtype=np.float32)

    if np.std(patch_values) < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(expected, patch_values)[0, 1])

    contrast = float(np.max(patch_values) - np.min(patch_values))

    diffs = np.diff(patch_values)

    # Penalize if brightness often goes backwards.
    negative_steps = int(np.sum(diffs < -5))

    # Penalize if candidate is colourful. Grey card should be low saturation.
    median_saturation = float(np.median(patch_saturations))

    # Penalize if RGB channels differ strongly. Grey patches should be neutral.
    median_channel_spread = float(np.median(patch_channel_spreads))

    # Prefer candidates with strong monotonic ramp and contrast.
    score = (
        120.0 * corr
        + 1.5 * contrast
        - 15.0 * negative_steps
        - 0.7 * median_saturation
        - 0.5 * median_channel_spread
    )

    diagnostics = {
        "corr": corr,
        "contrast": contrast,
        "negative_steps": negative_steps,
        "median_saturation": median_saturation,
        "median_channel_spread": median_channel_spread,
        "patch_values": patch_values.tolist(),
        "score": score,
    }

    return float(score), diagnostics


def detect_gray_scale_by_gray_ramp(
    image_bgr,
    x_min=0.05,
    x_max=0.90,
    y_min=0.70,
    y_max=0.96,
    n_patches=11,
):
    """
    Detect grey scale by finding an 11-patch left-to-right brightness ramp.

    This is designed for your card:
        left = 10 = black
        right = 0 = white
    """
    h, w = image_bgr.shape[:2]

    sx1 = int(w * x_min)
    sx2 = int(w * x_max)
    sy1 = int(h * y_min)
    sy2 = int(h * y_max)

    search = image_bgr[sy1:sy2, sx1:sx2].copy()

    if search.size == 0:
        return None, None, None

    search_h, search_w = search.shape[:2]

    # Try several possible band heights because scale size may vary.
    band_heights = [
        int(h * 0.08),
        int(h * 0.10),
        int(h * 0.12),
        int(h * 0.14),
    ]

    best = None

    for band_h in band_heights:
        band_h = max(25, min(band_h, search_h))

        step = max(4, band_h // 8)

        for y in range(0, search_h - band_h + 1, step):
            candidate = search[y:y + band_h, :].copy()

            score, diag = score_gray_ramp_candidate(
                candidate,
                n_patches=n_patches,
            )

            # Position prior: prefer lower bands.
            # This helps avoid the soil/paper boundary above the grey scale.
            band_center_y_full = sy1 + y + band_h / 2
            relative_center_y = band_center_y_full / h

            if relative_center_y < 0.74:
                score -= 40

            if relative_center_y > 0.82:
                score += 15

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "diag": diag,
                    "local_y1": y,
                    "local_y2": y + band_h,
                    "band_h": band_h,
                }

    if best is None:
        return None, None, None

    # Confidence check.
    # If this fails, use fixed fallback.
    diag = best["diag"]

    if (
        best["score"] < 80
        or diag.get("corr", 0) < 0.65
        or diag.get("contrast", 0) < 45
        or diag.get("negative_steps", 99) > 3
    ):
        return None, None, diag

    y1 = sy1 + best["local_y1"]
    y2 = sy1 + best["local_y2"]

    # Use broad fixed x range, but trim very slightly.
    x1 = sx1
    x2 = sx2

    strip_w = x2 - x1
    x1 = x1 + int(strip_w * 0.02)
    x2 = x2 - int(strip_w * 0.02)

    roi = image_bgr[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2), diag


def trim_gray_roi_horizontally_by_ramp(
    gray_roi_bgr,
    rect,
    n_patches=11,
    min_width_fraction=0.55,
    max_width_fraction=0.98,
    pad_fraction=0.01,
):
    """
    Trim a grey-scale ROI horizontally by finding the x-range that best behaves
    like the 11-patch grey ramp.

    This is better than simply removing white columns, because the rightmost
    patch is intentionally very light and may look similar to the paper.

    Expected scale:
        left  = label 10 = darkest
        right = label 0  = lightest

    Returns:
        trimmed_roi_bgr, trimmed_rect
    """
    if gray_roi_bgr is None or gray_roi_bgr.size == 0:
        return gray_roi_bgr, rect

    h, w = gray_roi_bgr.shape[:2]

    if h < 20 or w < 120:
        return gray_roi_bgr, rect

    best = None

    min_width = int(w * min_width_fraction)
    max_width = int(w * max_width_fraction)

    min_width = max(min_width, 120)
    max_width = min(max_width, w)

    # Try several candidate widths.
    candidate_widths = np.linspace(
        min_width,
        max_width,
        num=12,
        dtype=int,
    )

    for candidate_w in candidate_widths:
        if candidate_w <= 0 or candidate_w > w:
            continue

        step = max(3, candidate_w // 60)

        for x1_local in range(0, w - candidate_w + 1, step):
            x2_local = x1_local + candidate_w

            candidate = gray_roi_bgr[:, x1_local:x2_local].copy()

            score, diag = score_gray_ramp_candidate(
                candidate,
                n_patches=n_patches,
            )

            if not diag:
                continue

            corr = diag.get("corr", 0)
            contrast = diag.get("contrast", 0)
            negative_steps = diag.get("negative_steps", 99)
            patch_values = diag.get("patch_values", [])

            if len(patch_values) != n_patches:
                continue

            first_patch = patch_values[0]
            last_patch = patch_values[-1]

            # Basic sanity:
            # - should increase left to right
            # - should have meaningful dark-to-light contrast
            # - should not have too many backwards steps
            if corr < 0.60:
                continue

            if contrast < 35:
                continue

            if negative_steps > 4:
                continue

            # Important penalty:
            # If x starts too far left in white paper, the first patch will be too bright.
            adjusted_score = score

            if first_patch > 130:
                adjusted_score -= 80

            if last_patch < first_patch + 35:
                adjusted_score -= 60

            # Prefer candidates where the first patch is genuinely dark.
            adjusted_score += max(0, 130 - first_patch) * 0.4

            # Prefer candidates with a strong dark-to-light span.
            adjusted_score += contrast * 0.4

            # Slightly prefer wider candidates, but not too strongly.
            adjusted_score += (candidate_w / w) * 10

            if best is None or adjusted_score > best["score"]:
                best = {
                    "score": adjusted_score,
                    "x1": x1_local,
                    "x2": x2_local,
                    "diag": diag,
                    "candidate_w": candidate_w,
                }

    if best is None:
        return gray_roi_bgr, rect

    x1_local = best["x1"]
    x2_local = best["x2"]

    pad = int((x2_local - x1_local) * pad_fraction)

    x1_local = max(0, x1_local - pad)
    x2_local = min(w, x2_local + pad)

    if x2_local <= x1_local:
        return gray_roi_bgr, rect

    trimmed_roi = gray_roi_bgr[:, x1_local:x2_local].copy()

    x1, y1, x2, y2 = rect

    trimmed_rect = (
        x1 + x1_local,
        y1,
        x1 + x2_local,
        y2,
    )

    return trimmed_roi, trimmed_rect


def trim_gray_roi_vertically(
    gray_roi_bgr,
    rect,
    white_v_threshold=215,
    max_saturation=140,
    min_nonwhite_fraction=0.32,
    min_horizontal_range=30,
    pad_fraction=0.08,
):
    """
    Trim grey-scale ROI vertically using a stricter white threshold.

    Idea:
        - White paper has high V.
        - Grey-scale patch rows contain many non-white pixels across the row.
        - Printed numbers/ticks contain dark pixels, but only in a small row fraction.
        - Soil/paper edges may be dark, but usually fail the grey-row range/fraction rules.

    Returns:
        trimmed_roi_bgr, trimmed_rect
    """
    if gray_roi_bgr is None or gray_roi_bgr.size == 0:
        return gray_roi_bgr, rect

    h, w = gray_roi_bgr.shape[:2]

    if h < 20 or w < 80:
        return gray_roi_bgr, rect

    hsv = cv2.cvtColor(gray_roi_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # Focus away from extreme left/right margins.
    # This avoids paper edges and background.
    x1_core = int(w * 0.05)
    x2_core = int(w * 0.95)

    s_core = s[:, x1_core:x2_core]
    v_core = v[:, x1_core:x2_core]

    # A pixel is considered part of the grey-scale material if:
    #   - it is not white paper
    #   - it is not strongly coloured
    nonwhite_grayish = (v_core < white_v_threshold) & (s_core < max_saturation)

    # Fraction of each row that is non-white/greyish.
    row_fraction = nonwhite_grayish.mean(axis=1)

    # Grey-scale rows have a strong dark-to-light horizontal range.
    row_range = (
        np.percentile(v_core, 90, axis=1)
        - np.percentile(v_core, 10, axis=1)
    )

    good_rows = (
        (row_fraction >= min_nonwhite_fraction)
        & (row_range >= min_horizontal_range)
    )

    # Smooth vertically so the band is continuous.
    kernel_size = max(5, h // 30)

    if kernel_size % 2 == 0:
        kernel_size += 1

    smooth = np.convolve(
        good_rows.astype(np.float32),
        np.ones(kernel_size, dtype=np.float32) / kernel_size,
        mode="same",
    )

    good_rows = smooth > 0.35

    indices = np.where(good_rows)[0]

    if len(indices) == 0:
        return gray_roi_bgr, rect

    # Build contiguous row segments.
    segments = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev))
            start = idx
            prev = idx

    segments.append((start, prev))

    # Choose the best segment:
    # not only the tallest, but the one with strong non-white fraction and horizontal range.
    best_segment = None
    best_score = -1

    for y1, y2 in segments:
        height = y2 - y1 + 1

        if height < h * 0.08:
            continue

        frac_score = float(np.mean(row_fraction[y1:y2 + 1]))
        range_score = float(np.mean(row_range[y1:y2 + 1]))

        score = height * frac_score * max(range_score, 1)

        if score > best_score:
            best_score = score
            best_segment = (y1, y2)

    if best_segment is None:
        return gray_roi_bgr, rect

    y1_local, y2_local = best_segment

    pad = int((y2_local - y1_local + 1) * pad_fraction)

    y1_local = max(0, y1_local - pad)
    y2_local = min(h, y2_local + pad + 1)

    if y2_local <= y1_local:
        return gray_roi_bgr, rect

    trimmed_roi = gray_roi_bgr[y1_local:y2_local, :].copy()

    x1, y1, x2, y2 = rect

    trimmed_rect = (
        x1,
        y1 + y1_local,
        x2,
        y1 + y2_local,
    )

    return trimmed_roi, trimmed_rect


def crop_gray_scale_roi(image_bgr):
    """
    Main grey-scale ROI function.

    1. Detect approximate grey-scale ramp.
    2. Trim vertically to remove white space / numbers.
    3. Trim horizontally to remove left/right margins.
    4. Fall back to fixed crop if detection fails.
    """
    detected_roi, detected_rect, diag = detect_gray_scale_by_gray_ramp(
        image_bgr,
        x_min=0.05,
        x_max=0.90,
        y_min=0.76,
        y_max=0.96,
        n_patches=11,
    )

    if detected_roi is not None:
        detected_roi, detected_rect = trim_gray_roi_vertically(
            detected_roi,
            detected_rect,
        )

        detected_roi, detected_rect = trim_gray_roi_horizontally_by_ramp(
            detected_roi,
            detected_rect,
            n_patches=11,
        )

        return detected_roi, detected_rect

    fallback_roi, fallback_rect = crop_fixed_gray_scale_roi(
        image_bgr,
        x_min=0.08,
        x_max=0.88,
        y_min=0.78,
        y_max=0.91,
    )

    fallback_roi, fallback_rect = trim_gray_roi_vertically(
        fallback_roi,
        fallback_rect,
    )

    fallback_roi, fallback_rect = trim_gray_roi_horizontally_by_ramp(
        fallback_roi,
        fallback_rect,
        n_patches=11,
    )

    return fallback_roi, fallback_rect
    

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
        cy1 = int(ph * 0.20)
        cy2 = int(ph * 0.80)

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
    rows = []

    target_grays = target_grays_from_black_scale(
        n_patches=n_patches,
        left_label=10,
        right_label=0,
    )

    for patch in patches:
        patch_index = patch["patch_index"]

        rgb = measure_patch_rgb(patch["patch_bgr"])
        target = float(target_grays[patch_index])

        measured_rgbs.append(rgb)

        rows.append({
            "image": image_name,
            "patch_index": patch_index,
            "scale_label": 10 - patch_index,
            "measured_R": float(rgb[0]),
            "measured_G": float(rgb[1]),
            "measured_B": float(rgb[2]),
            "target_gray": float(target),
            "imbalance_before": float(max(rgb) - min(rgb)),
        })

    targets = target_grays

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