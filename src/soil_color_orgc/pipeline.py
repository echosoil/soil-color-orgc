import os
import glob
import pandas as pd

from .munsell import load_munsell, find_best_munsell
from .image_processing import extract_dominant_lab
from .soc_estimation import estimate_soc


def find_image_paths(samples_dir: str):
    patterns = [
        "*.jpg", "*.jpeg", "*.jfif", "*.png", "*.webp",
        "*.JPG", "*.JPEG", "*.JFIF", "*.PNG", "*.WEBP",
    ]

    image_paths = []

    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(samples_dir, pattern)))

    return sorted(set(image_paths))


def safe_extract_sample_code(image_path: str, use_ocr: bool = False, known_codes=None):
    """
    Extract sample code, but never let OCR/QR problems stop the pipeline.

    Important:
    - If use_ocr=False, this only uses the filename.
    - If use_ocr=True, it imports EasyOCR code lazily.
    """
    try:
        if use_ocr:
            from .code_extraction import extract_sample_code_with_source
            return extract_sample_code_with_source(image_path, known_codes=known_codes)

        # Fast filename-only fallback.
        from .code_extraction_light import extract_code_from_filename

        filename_base = extract_code_from_filename(image_path)

        if filename_base:
            return {
                "sample_code_full": None,
                "sample_code_base": filename_base,
                "sample_code_source": "filename",
                "sample_code_match_score": None,
                "sample_code_matched_fragment": filename_base,
                "sample_code_error": None,
                "filename_code_base": filename_base,
                "code_conflict": False,
            }

        return {
            "sample_code_full": None,
            "sample_code_base": None,
            "sample_code_source": "none",
            "sample_code_match_score": None,
            "sample_code_matched_fragment": None,
            "sample_code_error": None,
            "filename_code_base": None,
            "code_conflict": False,
        }

    except Exception as exc:
        return {
            "sample_code_full": None,
            "sample_code_base": None,
            "sample_code_source": "code_error",
            "sample_code_match_score": None,
            "sample_code_matched_fragment": None,
            "sample_code_error": str(exc),
            "filename_code_base": None,
            "code_conflict": False,
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
    use_ocr: bool = False,
    known_codes=None,
):
    print("Loading Munsell table...", flush=True)
    munsell_dict = load_munsell(munsell_csv)
    print(f"Loaded {len(munsell_dict)} Munsell colors", flush=True)

    print(f"Searching images in: {samples_dir}", flush=True)
    image_paths = find_image_paths(samples_dir)
    print(f"Found {len(image_paths)} images", flush=True)

    if not image_paths:
        print(f"No images found in: {samples_dir}")
        return []

    rows = []

    for i, image_path in enumerate(image_paths, start=1):
        image_name = os.path.basename(image_path)
        print(f"\n[{i}/{len(image_paths)}] Processing {image_name}", flush=True)

        try:
            print("  extracting dominant Lab...", flush=True)
            lab = extract_dominant_lab(
                image_path,
                save_debug_masks=save_debug_masks,
                debug_dir=debug_dir,
                downscale_max=downscale_max,
                kmeans_clusters=kmeans_clusters,
                trim_percent=trim_percent,
            )

            print("  extracting sample code...", flush=True)
            code_info = safe_extract_sample_code(
                image_path,
                use_ocr=use_ocr,
                known_codes=known_codes,
            )

            print("  matching Munsell...", flush=True)
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
                "sample_code_match_score": code_info.get("sample_code_match_score"),
                "sample_code_matched_fragment": code_info.get("sample_code_matched_fragment"),
                "sample_code_error": code_info.get("sample_code_error"),
                "filename_code_base": code_info.get("filename_code_base"),
                "code_conflict": code_info.get("code_conflict"),
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
                f"  OK → {best_munsell} "
                f"(ΔE2000={round(float(delta_e), 2)}) "
                f"LAB={tuple(round(x, 1) for x in lab)} "
                f"SOC_est%={round(float(soc), 2)} [{soc_method}] "
                f"code={code_info.get('sample_code_base')} "
                f"[{code_info.get('sample_code_source')}]",
                flush=True,
            )

        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)

            rows.append({
                "image": image_name,
                "sample_code_full": None,
                "sample_code_base": None,
                "sample_code_source": None,
                "sample_code_match_score": None,
                "sample_code_matched_fragment": None,
                "sample_code_error": None,
                "filename_code_base": None,
                "code_conflict": None,
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
    print(f"\nSaved {output_csv} ({len(rows)} images)", flush=True)

    return rows