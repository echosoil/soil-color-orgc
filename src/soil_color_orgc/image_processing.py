import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from colour import sRGB_to_XYZ, XYZ_to_Lab


def downscale_max_side(img_bgr, max_side: int = 800):
    h, w = img_bgr.shape[:2]
    max_current_side = max(h, w)

    if max_current_side <= max_side:
        return img_bgr

    scale = max_side / float(max_current_side)

    return cv2.resize(
        img_bgr,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def bilateral_denoise(img_bgr):
    return cv2.bilateralFilter(
        img_bgr,
        d=7,
        sigmaColor=50,
        sigmaSpace=50,
    )


def build_soil_mask(img_bgr):
    """
    Conservative soil mask.

    It rejects:
    - strong highlights,
    - deep shadows,
    - very desaturated background/card/paper regions.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(img_hsv)

    mask_value = (value > 25) & (value < 230)
    mask_saturation = saturation > 25
    base_mask = (mask_value & mask_saturation).astype(np.uint8)

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    _, a_channel, b_channel = cv2.split(img_lab)

    dist_ab = np.sqrt(
        (a_channel.astype(np.int16) - 128) ** 2
        + (b_channel.astype(np.int16) - 128) ** 2
    )

    mask_ab = dist_ab > 6
    mask = (base_mask & mask_ab).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def save_debug_mask(img_bgr, mask, out_path):
    """
    Save a PNG debug overlay, regardless of original image extension.
    """
    base = os.path.splitext(os.path.basename(out_path))[0]
    out_png = os.path.join(os.path.dirname(out_path), f"{base}_mask.png")

    debug_img = img_bgr.copy()

    green = np.zeros_like(debug_img)
    green[:, :, 1] = 255

    overlay = cv2.addWeighted(debug_img, 0.8, green, 0.4, 0)
    debug_img[mask > 0] = overlay[mask > 0]

    cv2.imwrite(out_png, debug_img)


def extract_dominant_lab(
    image_path: str,
    save_debug_masks: bool = True,
    debug_dir: str = "debug_masks",
    downscale_max: int = 800,
    kmeans_clusters: int = 4,
    trim_percent: float = 0.05,
):
    """
    Extract robust dominant soil color in CIE Lab.

    Returns:
        (L, a, b)
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = downscale_max_side(img, downscale_max)
    img = bilateral_denoise(img)

    mask = build_soil_mask(img)

    if save_debug_masks:
        os.makedirs(debug_dir, exist_ok=True)
        save_debug_mask(
            img,
            mask,
            os.path.join(debug_dir, os.path.basename(image_path)),
        )

    if mask.sum() == 0:
        roi_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    else:
        roi = cv2.bitwise_and(img, img, mask=mask)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_rgb = roi_rgb[mask > 0]

    if len(roi_rgb) < 100:
        rgb_selected = roi_rgb
    else:
        lows = np.quantile(roi_rgb, trim_percent, axis=0)
        highs = np.quantile(roi_rgb, 1 - trim_percent, axis=0)

        keep = np.all((roi_rgb >= lows) & (roi_rgb <= highs), axis=1)
        rgb_selected = roi_rgb[keep]

    if rgb_selected.shape[0] > 50000:
        idx = np.random.choice(rgb_selected.shape[0], 50000, replace=False)
        rgb_selected = rgb_selected[idx]

    rgb_norm = rgb_selected.astype(np.float32) / 255.0

    XYZ = np.apply_along_axis(sRGB_to_XYZ, 1, rgb_norm)
    Lab = np.apply_along_axis(XYZ_to_Lab, 1, XYZ)

    lab_median = np.median(Lab, axis=0)

    try:
        kmeans = KMeans(
            n_clusters=kmeans_clusters,
            random_state=42,
            n_init=5,
        ).fit(rgb_selected)

        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster = labels[np.argmax(counts)]

        dominant_rgb = kmeans.cluster_centers_[dominant_cluster] / 255.0
        dominant_lab = XYZ_to_Lab(sRGB_to_XYZ(dominant_rgb))

        final_lab = 0.5 * lab_median + 0.5 * np.array(dominant_lab)

    except Exception:
        final_lab = lab_median

    return tuple(map(float, final_lab))