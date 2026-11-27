#!/usr/bin/env python3
"""Prepare a private review manifest from a lab Excel file and photo directory.

Only unique four-letter-prefix matches are proposed; group IDs must be
assigned manually. This is not scientific confirmation of sample identity.
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd


def code(value):
    match = re.fullmatch(r"([A-Za-z]{4})(?:-\d+)?", str(value).strip())
    return match.group(1).upper() if match else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lab", required=True, type=Path)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--id-column", default="ID")
    p.add_argument("--target", default="orgC_lab")
    p.add_argument("--sheet", default=None, help="Named sheet; first sheet by default")
    args = p.parse_args()
    os.umask(0o077)
    lab = pd.read_excel(args.lab, sheet_name=args.sheet if args.sheet else 0)
    if not {args.id_column,args.target}.issubset(lab.columns):
        p.error("Expected lab ID and target columns not found")
    photos = defaultdict(list)
    for path in sorted(args.images.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg",".jpeg",".jfif",".png",".webp"} and code(path.stem):
            photos[code(path.stem)].append(path)
    ids = lab[args.id_column].map(code)
    counts = Counter(ids.dropna())
    stats = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["sample_id","group_id","image_path","orgC_lab","soil_roi","gray_roi"])
        writer.writeheader()
        for i, row in lab.iterrows():
            key = ids.loc[i]
            if not key or counts[key] != 1 or len(photos[key]) != 1:
                stats["unmatched_or_ambiguous"] += 1
                continue
            try:
                y = float(row[args.target])
                if not 0 <= y <= 100:
                    raise ValueError
            except (TypeError, ValueError):
                stats["invalid_lab_value"] += 1
                continue
            writer.writerow({"sample_id": str(row[args.id_column]).strip(),
                             "group_id": "",
                             "image_path": os.path.relpath(photos[key][0].resolve(), args.out.parent.resolve()),
                             "orgC_lab": y, "soil_roi": "", "gray_roi": ""})
            stats["proposed_matches"] += 1
    print(dict(stats))
    print("Review each match, confirm target units and fill group_id before validation:", args.out)


if __name__ == "__main__":
    main()
