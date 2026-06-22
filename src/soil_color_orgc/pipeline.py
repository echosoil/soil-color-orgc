import os
import glob
import pandas as pd

from .munsell import load_munsell, find_best_munsell
from .image_processing import extract_dominant_lab
from .soc_estimation import estimate_soc
from .code_extraction import extract_sample_code_with_source


def find_image_paths(samples_dir: str):
    patterns = [
        "*.jpg", "*.jpeg", "*.jfif", "*.png", "*.webp",
        "*.JPG", "*.JPEG", "*.JFIF", "*.PNG", "*.WEBP",
    ]

    image_paths = []

    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(samples_dir, pattern)))

    return sorted(set(image_paths))


def safe_extract_sample_code(image_path: str):
    """
    Extract sample code, but never let OCR/QR problems stop the pipeline.
    """
    try:
        return extract_sample_code_with_source(image_path)
    except Exception as exc:
        return {
            "sample_code_full": None,
            "sample_code_base": None,
            "sample_code_source": "code_error",
            "sample_code_error": str(exc),
        }


def run_image_pipeline(
    samples_dir: str,
    munsell_csv: str,
    output_csv: str,
    debug_dir: str = "debug_masks",
    save_debug_masks: bool = True,
    deltae_threshold: float = 8.0,
    downscale_max: int = 800,
    kmeans_clusters: int = 4,
    trim_percent: float = 0.05,
):
    munsell_dict = load_munsell(munsell_csv)

    print(f"Loaded {len(munsell_dict)} Munsell colors")

    image_paths = find_image_paths(samples_dir)

    if not image_paths:
        print(f"No images found in: {samples_dir}")
        return []

    rows = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)

        try:
            lab = extract_dominant_lab(
                image_path,
                save_debug_masks=save_debug_masks,
                debug_dir=debug_dir,
                downscale_max=downscale_max,
                kmeans_clusters=kmeans_clusters,
                trim_percent=trim_percent,
            )

            code_info = safe_extract_sample_code(image_path)

            best_munsell, delta_e = find_best_munsell(
                lab,
                munsell_dict,
                threshold=deltae_threshold,
            )

            soc, soc_method = estimate_soc(lab, best_munsell)

            row = {
                "image": image_name,
                "sample_code_full": code_info.get("sample_code_full"),
                "sample_code_base": code_info.get("sample_code_base"),
                "sample_code_source": code_info.get("sample_code_source"),
                "sample_code_error": code_info.get("sample_code_error"),
                "L": round(lab[0], 3),
                "a": round(lab[1], 3),
                "b": round(lab[2], 3),
                "best_munsell": best_munsell,
                "deltaE2000": round(float(delta_e), 2),
                "SOC_est%": round(float(soc), 2),
                "SOC_method": soc_method,
                "processing_status": "ok",
                "processing_error": None,
            }

            rows.append(row)

            print(
                f"{image_path} → {best_munsell} "
                f"(ΔE2000={round(float(delta_e), 2)}) "
                f"LAB={tuple(round(x, 1) for x in lab)} "
                f"SOC_est%={round(float(soc), 2)} [{soc_method}] "
                f"code={code_info.get('sample_code_base')} "
                f"[{code_info.get('sample_code_source')}]"
            )

        except Exception as exc:
            print(f"Error processing {image_path}: {exc}")

            rows.append({
                "image": image_name,
                "sample_code_full": None,
                "sample_code_base": None,
                "sample_code_source": None,
                "sample_code_error": None,
                "L": None,
                "a": None,
                "b": None,
                "best_munsell": None,
                "deltaE2000": None,
                "SOC_est%": None,
                "SOC_method": None,
                "processing_status": "error",
                "processing_error": str(exc),
            })

    output_dir = os.path.dirname(output_csv)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved {output_csv} ({len(rows)} images)")

    return rows