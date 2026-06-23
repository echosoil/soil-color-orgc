#!/usr/bin/env python3

import argparse
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageCms


IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}


def list_images(path):
    path = Path(path)

    if path.is_file():
        return [path]

    images = []

    for p in path.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)

    return sorted(images)


def get_icc_info(image_path):
    img = Image.open(image_path)
    icc = img.info.get("icc_profile")

    if not icc:
        return {
            "has_icc": False,
            "icc_name": None,
            "icc_description": None,
            "icc_bytes": 0,
        }

    name = None
    description = None

    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        try:
            name = ImageCms.getProfileName(profile)
        except Exception:
            name = None

        try:
            description = ImageCms.getProfileDescription(profile)
        except Exception:
            description = None

    except Exception as exc:
        description = f"Could not read ICC profile: {exc}"

    return {
        "has_icc": True,
        "icc_name": name,
        "icc_description": description,
        "icc_bytes": len(icc),
    }


def read_cv2_bgr(image_path):
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def read_pillow_no_icc_bgr(image_path):
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_pillow_with_icc_bgr(image_path):
    img = Image.open(image_path)
    icc = img.info.get("icc_profile")

    if icc:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            target_profile = ImageCms.createProfile("sRGB")

            img = ImageCms.profileToProfile(
                img,
                source_profile,
                target_profile,
                outputMode="RGB",
            )
        except Exception:
            img = img.convert("RGB")
    else:
        img = img.convert("RGB")

    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def lab_summary_from_bgr(image_bgr):
    """
    Quick OpenCV Lab summary.

    OpenCV uint8 Lab:
      L is 0..255, converted here to 0..100.
      a and b are centered around 128.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    L = lab[:, :, 0] * 100.0 / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    return {
        "mean_L": float(np.mean(L)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_L": float(np.median(L)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
    }


def mean_abs_pixel_diff(a, b):
    if a is None or b is None:
        return None

    if a.shape != b.shape:
        return None

    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def fit_to_panel(img_bgr, width=350, height=500):
    """
    Fit image into a fixed-size white panel while preserving aspect ratio.
    This guarantees all panels have the same dimensions.
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


def save_comparison_image(image_path, cv2_bgr, pil_no_icc_bgr, pil_icc_bgr, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    panels = []

    for label, img in [
        ("cv2", cv2_bgr),
        ("pillow_no_icc", pil_no_icc_bgr),
        ("pillow_with_icc", pil_icc_bgr),
    ]:
        preview = fit_to_panel(img, width=350, height=500)

        cv2.putText(
            preview,
            label,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        panels.append(preview)

    comparison = np.hstack(panels)

    out_path = os.path.join(
        output_dir,
        f"{Path(image_path).stem}_reader_comparison.jpg",
    )

    cv2.imwrite(out_path, comparison)

    return out_path
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="Image file or folder, e.g. data/samples",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/color_profile_diagnosis.csv",
    )
    parser.add_argument(
        "--preview-dir",
        default="outputs/color_profile_previews",
    )
    parser.add_argument(
        "--save-previews",
        action="store_true",
    )

    args = parser.parse_args()

    images = list_images(args.path)

    if not images:
        raise ValueError(f"No images found in {args.path}")

    rows = []

    for image_path in images:
        print(f"Checking {image_path}")

        icc_info = get_icc_info(image_path)

        cv2_bgr = read_cv2_bgr(image_path)
        pil_no_icc_bgr = read_pillow_no_icc_bgr(image_path)
        pil_icc_bgr = read_pillow_with_icc_bgr(image_path)

        if cv2_bgr is None:
            print(f"  ERROR: OpenCV could not read {image_path}")
            continue

        stats_cv2 = lab_summary_from_bgr(cv2_bgr)
        stats_pil_no_icc = lab_summary_from_bgr(pil_no_icc_bgr)
        stats_pil_icc = lab_summary_from_bgr(pil_icc_bgr)

        row = {
            "image": image_path.name,
            **icc_info,

            "cv2_mean_L": stats_cv2["mean_L"],
            "cv2_mean_a": stats_cv2["mean_a"],
            "cv2_mean_b": stats_cv2["mean_b"],

            "pillow_no_icc_mean_L": stats_pil_no_icc["mean_L"],
            "pillow_no_icc_mean_a": stats_pil_no_icc["mean_a"],
            "pillow_no_icc_mean_b": stats_pil_no_icc["mean_b"],

            "pillow_icc_mean_L": stats_pil_icc["mean_L"],
            "pillow_icc_mean_a": stats_pil_icc["mean_a"],
            "pillow_icc_mean_b": stats_pil_icc["mean_b"],

            "diff_cv2_vs_pillow_no_icc": mean_abs_pixel_diff(cv2_bgr, pil_no_icc_bgr),
            "diff_cv2_vs_pillow_icc": mean_abs_pixel_diff(cv2_bgr, pil_icc_bgr),
            "diff_pillow_no_icc_vs_pillow_icc": mean_abs_pixel_diff(pil_no_icc_bgr, pil_icc_bgr),
        }

        if args.save_previews:
            row["preview"] = save_comparison_image(
                image_path,
                cv2_bgr,
                pil_no_icc_bgr,
                pil_icc_bgr,
                args.preview_dir,
            )

        rows.append(row)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print(f"\nSaved diagnosis: {args.output_csv}")

    print("\nSummary of pixel differences:")
    print(
        df[
            [
                "image",
                "has_icc",
                "icc_description",
                "diff_cv2_vs_pillow_no_icc",
                "diff_cv2_vs_pillow_icc",
                "diff_pillow_no_icc_vs_pillow_icc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()