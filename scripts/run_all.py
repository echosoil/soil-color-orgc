#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soil_color_orgc.pipeline import run_image_pipeline
from soil_color_orgc.lab_merge import enrich_lab_file


def main():
    parser = argparse.ArgumentParser(
        description="Run image SOC estimation and enrich lab Excel file."
    )

    parser.add_argument(
        "--samples",
        default="data/samples",
        help="Folder containing sample images.",
    )

    parser.add_argument(
        "--munsell",
        default="data/munsell/rit_munsell.csv",
        help="Path to RIT Munsell CSV.",
    )

    parser.add_argument(
        "--lab",
        default="data/lab/test_stat_orgC.xlsx",
        help="Input laboratory Excel file.",
    )

    parser.add_argument(
        "--results",
        default="outputs/results.csv",
        help="Output predictions CSV.",
    )

    parser.add_argument(
        "--enriched",
        default="outputs/test_stat_orgC_enriched.xlsx",
        help="Output enriched Excel file.",
    )

    parser.add_argument(
        "--debug-dir",
        default="debug_masks",
        help="Folder for debug mask images.",
    )

    parser.add_argument(
        "--no-debug-masks",
        action="store_true",
        help="Disable debug mask generation.",
    )

    parser.add_argument(
        "--deltae-threshold",
        type=float,
        default=8.0,
        help="Maximum accepted DeltaE2000 for Munsell match.",
    )

    parser.add_argument(
        "--no-gray-calibration",
        action="store_true",
        help="Disable grey-scale colour calibration.",
    )

    args = parser.parse_args()

    run_image_pipeline(
        samples_dir=args.samples,
        munsell_csv=args.munsell,
        output_csv=args.results,
        debug_dir=args.debug_dir,
        save_debug_masks=not args.no_debug_masks,
        deltae_threshold=args.deltae_threshold,
        use_gray_calibration=not args.no_gray_calibration,
    )

    enrich_lab_file(
        lab_xlsx=args.lab,
        predictions_csv=args.results,
        output_xlsx=args.enriched,
    )


if __name__ == "__main__":
    main()