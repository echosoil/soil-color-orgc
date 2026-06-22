#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soil_color_orgc.lab_merge import enrich_lab_file


def main():
    parser = argparse.ArgumentParser(
        description="Enrich lab Excel file with image-based SOC estimates."
    )

    parser.add_argument(
        "--lab",
        default="data/lab/test_stat_orgC.xlsx",
        help="Input laboratory Excel file.",
    )

    parser.add_argument(
        "--predictions",
        default="outputs/results.csv",
        help="Predictions CSV produced by run_color_estimation.py.",
    )

    parser.add_argument(
        "--output",
        default="outputs/test_stat_orgC_enriched.xlsx",
        help="Output enriched Excel file.",
    )

    args = parser.parse_args()

    enrich_lab_file(
        lab_xlsx=args.lab,
        predictions_csv=args.predictions,
        output_xlsx=args.output,
    )


if __name__ == "__main__":
    main()