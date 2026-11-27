#!/usr/bin/env python3
"""Evaluate the current service feature extraction against matched lab results."""
import argparse
import importlib.metadata
import os
from pathlib import Path

from soil_color_orgc.analysis import Reference
from soil_color_orgc.validation import read_manifest, file_digest, extract_pairs, evaluate_pairs, write_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--reference", default=Path("data/munsell/rit_munsell.csv"), type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--commit", required=True, help="Actual code checkout revision")
    p.add_argument("--ack-lab-percent-and-groups", choices=["yes"], required=True)
    args = p.parse_args()
    os.umask(0o077)
    if args.out.exists():
        p.error("Output directory already exists; choose a fresh run directory")
    before = file_digest(args.manifest)
    rows = read_manifest(args.manifest)
    if before != file_digest(args.manifest):
        raise RuntimeError("Manifest changed during read")
    ref = Reference.from_csv(args.reference)
    records = extract_pairs(rows, ref)
    summary, predictions = evaluate_pairs(records)
    summary.update({
        "method_version": "soil-colour-v2-preview", "code_revision": args.commit,
        "manifest_sha256": before, "reference_sha256": ref.sha256,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "colour-science", "opencv-python-headless", "Pillow"]},
        "limitations": [
            "Operator confirms laboratory organic carbon mass percent and scientific grouping.",
            "One photo per physical sample; common paired QC subset, not all uploaded photos.",
            "Five-fold group-separated cross-validation of fixed linear models; not independent external validation.",
            "No model selection, tuning, uncertainty calibration, model promotion or production prediction."
        ],
    })
    write_results(args.out, summary, records, predictions)
    print(f"{summary['status']}: {summary['paired_samples']}/{summary['total_samples']} paired samples; {summary['paired_groups']} groups")
    print("Dispositions:", summary["dispositions"])
    for name, scores in summary["metrics"].items():
        print(name, scores)
    print("Aggregate summary:", args.out/"validation-summary.json")


if __name__ == "__main__":
    main()
