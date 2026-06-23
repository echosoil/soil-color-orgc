#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
import pandas as pd
import cv2

from colour import Lab_to_XYZ, XYZ_to_sRGB


def lab_to_rgb_u8(L, a, b):
    """
    Convert Lab to uint8 RGB for visualization.
    """
    lab = np.array([L, a, b], dtype=float)
    xyz = Lab_to_XYZ(lab)
    rgb = XYZ_to_sRGB(xyz)

    rgb = np.clip(rgb, 0, 1)
    rgb_u8 = (rgb * 255).round().astype(np.uint8)

    return rgb_u8


def make_swatch(rgb_u8, size=220):
    """
    Create a square swatch image from RGB uint8.
    Returned image is BGR for OpenCV writing.
    """
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)
    bgr = rgb_u8[::-1]
    swatch = np.full((size, size, 3), bgr, dtype=np.uint8)
    return swatch


def format_munsell_notation(h, v, c):
    """
    Create notation like:
        10YR 3/4

    Avoids accidental 3.0/4.0 formatting.
    """
    def clean_number(x):
        x = float(x)
        if x.is_integer():
            return str(int(x))
        return str(x).rstrip("0").rstrip(".")

    return f"{str(h).strip()} {clean_number(v)}/{clean_number(c)}"


def load_munsell_rgb_lookup(munsell_csv):
    """
    Build mapping:
        notation -> (R, G, B)

    RIT Munsell CSV may contain:
        R,G,B      as normalized 0..1 floats
        dR,dG,dB   as 8-bit display RGB values

    For visualization, prefer dR,dG,dB when available.
    """
    df = pd.read_csv(munsell_csv)

    lookup = {}

    for _, row in df.iterrows():
        notation = format_munsell_notation(row["h"], row["V"], row["C"])

        if {"dR", "dG", "dB"}.issubset(df.columns):
            rgb = (
                int(round(row["dR"])),
                int(round(row["dG"])),
                int(round(row["dB"])),
            )
        else:
            r = float(row["R"])
            g = float(row["G"])
            b = float(row["B"])

            # If RGB is normalized 0..1, scale to 0..255.
            if max(r, g, b) <= 1.0:
                rgb = (
                    int(round(r * 255)),
                    int(round(g * 255)),
                    int(round(b * 255)),
                )
            else:
                rgb = (
                    int(round(r)),
                    int(round(g)),
                    int(round(b)),
                )

        rgb = tuple(max(0, min(255, x)) for x in rgb)

        lookup[notation] = rgb

    return lookup


def fit_image_to_panel(img_bgr, width=320, height=320):
    """
    Fit an image into a white panel while preserving aspect ratio.
    """
    panel = np.full((height, width, 3), 255, dtype=np.uint8)

    if img_bgr is None or img_bgr.size == 0:
        return panel

    h, w = img_bgr.shape[:2]
    scale = min(width / w, height / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2

    panel[y0:y0 + new_h, x0:x0 + new_w] = resized
    return panel


def add_label(img_bgr, text, y=28, scale=0.8):
    cv2.putText(
        img_bgr,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img_bgr


def make_labeled_panel(content_bgr, title, width=320, height=360):
    panel = np.full((height, width, 3), 255, dtype=np.uint8)

    title_h = 40
    body = fit_image_to_panel(content_bgr, width=width, height=height - title_h)

    panel[title_h:, :, :] = body
    add_label(panel, title, y=28, scale=0.75)

    return panel


def load_roi_image(debug_dir, image_name):
    stem = os.path.splitext(os.path.basename(image_name))[0]
    roi_path = os.path.join(debug_dir, f"{stem}_roi.jpg")

    if os.path.exists(roi_path):
        return cv2.imread(roi_path, cv2.IMREAD_COLOR)

    return None


def build_comparison_card(row, munsell_rgb_lookup, debug_dir, out_dir):
    image_name = row["image"]
    stem = os.path.splitext(os.path.basename(image_name))[0]

    L = float(row["L"])
    a = float(row["a"])
    b = float(row["b"])

    best_munsell = row.get("best_munsell", None)
    delta_e = row.get("deltaE2000", None)
    soc_est = row.get("SOC_est%", None)

    # ROI panel
    roi_bgr = load_roi_image(debug_dir, image_name)
    roi_panel = make_labeled_panel(roi_bgr, "ROI used")

    # Estimated color swatch
    est_rgb = lab_to_rgb_u8(L, a, b)
    est_swatch = make_swatch(est_rgb, size=220)
    est_panel = make_labeled_panel(est_swatch, "Estimated color")

    # Munsell swatch
    if pd.notna(best_munsell) and best_munsell in munsell_rgb_lookup:
        munsell_rgb = munsell_rgb_lookup[best_munsell]
        munsell_swatch = make_swatch(munsell_rgb, size=220)
    else:
        munsell_swatch = np.full((220, 220, 3), 240, dtype=np.uint8)

    munsell_panel = make_labeled_panel(munsell_swatch, "Closest Munsell")

    top = np.hstack([roi_panel, est_panel, munsell_panel])

    info_h = 170
    info = np.full((info_h, top.shape[1], 3), 255, dtype=np.uint8)

    text_lines = [
        f"Sample: {image_name}",
        f"Estimated Lab: L={L:.2f}, a={a:.2f}, b={b:.2f}",
        f"Best Munsell: {best_munsell}",
        f"DeltaE2000: {delta_e}",
        f"SOC_est%: {soc_est}",
    ]

    y = 35
    for line in text_lines:
        cv2.putText(
            info,
            str(line),
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    card = np.vstack([top, info])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}_comparison.jpg")
    cv2.imwrite(out_path, card)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="outputs/results.csv",
        help="Results CSV from the pipeline.",
    )
    parser.add_argument(
        "--munsell",
        default="data/munsell/rit_munsell.csv",
        help="RIT Munsell CSV.",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug_masks",
        help="Directory containing *_roi.jpg debug images.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/color_cards",
        help="Directory to write comparison cards.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.results)
    munsell_rgb_lookup = load_munsell_rgb_lookup(args.munsell)

    needed = {"image", "L", "a", "b"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {sorted(missing)}")

    created = []

    for _, row in df.iterrows():
        if pd.isna(row["L"]) or pd.isna(row["a"]) or pd.isna(row["b"]):
            continue

        out_path = build_comparison_card(
            row=row,
            munsell_rgb_lookup=munsell_rgb_lookup,
            debug_dir=args.debug_dir,
            out_dir=args.output_dir,
        )
        created.append(out_path)

    print(f"Created {len(created)} comparison cards in: {args.output_dir}")


if __name__ == "__main__":
    main()