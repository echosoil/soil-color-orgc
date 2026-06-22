#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soil_color_orgc.pipeline import run_image_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Estimate soil organic carbon from sample images."
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
        "--output",
        default="outputs/results.csv",
        help="Output CSV path.",
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
        "--downscale-max",
        type=int,
        default=800,
        help="Maximum image side length used for processing.",
    )

    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=4,
        help="Number of KMeans clusters.",
    )

    parser.add_argument(
        "--trim-percent",
        type=float,
        default=0.05,
        help="Trim percent for robust RGB selection.",
    )

    args = parser.parse_args()

    run_image_pipeline(
        samples_dir=args.samples,
        munsell_csv=args.munsell,
        output_csv=args.output,
        debug_dir=args.debug_dir,
        save_debug_masks=not args.no_debug_masks,
        deltae_threshold=args.deltae_threshold,
        downscale_max=args.downscale_max,
        kmeans_clusters=args.kmeans_clusters,
        trim_percent=args.trim_percent,
    )


if __name__ == "__main__":
    main()