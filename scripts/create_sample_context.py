#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SAMPLE_CODE_CANDIDATES = [
    "SampleCode",
    "sample_code_base",
    "sample_code_full",
    "ID",
    "image",
    "QR Code",
    "QR Codes",
    "qr_code",
    "QR_qrCode",
]

LAT_CANDIDATES = [
    "lat",
    "latitude",
    "Latitude",
    "LAT",
    "GPS_lat",
]

LON_CANDIDATES = [
    "lon",
    "lng",
    "longitude",
    "Longitude",
    "LON",
    "GPS_long",
]

COUNTRY_CANDIDATES = [
    "country",
    "Country",
    "country_code",
    "actual_cc",
    "country_planned",
    "Country Planned",
    "planned_country",
]

REGION_CANDIDATES = [
    "region",
    "Region",
    "admin_region",
    "province",
    "Province",
    "state",
    "State",
]


def extract_sample_code(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    stem = Path(text).stem

    match = re.search(r"[A-Za-z]{4}", stem)
    if not match:
        return None

    return match.group(0).upper()


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {c: c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c in exact:
            return exact[c]

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    return None


def read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(path)
        sheet = "enriched" if "enriched" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet)
    
    if suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python")
    
    raise ValueError(f"Unsupported file type: {path}")


def extract_context_from_table(path: Path) -> pd.DataFrame:
    df = read_any_table(path)
    df.columns = [str(c).strip() for c in df.columns]

    sample_col = find_col(df, SAMPLE_CODE_CANDIDATES)
    lat_col = find_col(df, LAT_CANDIDATES)
    lon_col = find_col(df, LON_CANDIDATES)
    country_col = find_col(df, COUNTRY_CANDIDATES)
    region_col = find_col(df, REGION_CANDIDATES)

    if sample_col is None:
        raise ValueError(
            f"Could not find sample-code column in {path}. "
            f"Tried: {SAMPLE_CODE_CANDIDATES}"
        )

    out = pd.DataFrame()
    out["SampleCode"] = df[sample_col].apply(extract_sample_code)

    if lat_col:
        out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    else:
        out["lat"] = pd.NA

    if lon_col:
        out["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        out["lon"] = pd.NA

    if country_col:
        out["country"] = df[country_col]
    else:
        out["country"] = pd.NA

    if region_col:
        out["region"] = df[region_col]
    else:
        out["region"] = pd.NA

    out["source_file"] = path.as_posix()

    out = out.dropna(subset=["SampleCode"]).copy()

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Create data/context/sample_context.csv from existing project files."
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSV/XLSX files to scan for SampleCode, lat, lon, country, region.",
    )

    parser.add_argument(
        "--output",
        default="data/context/sample_context.csv",
        help="Output context CSV.",
    )

    args = parser.parse_args()

    frames = []

    for input_path in args.inputs:
        path = Path(input_path)

        if not path.exists():
            print(f"WARNING: file not found, skipping: {path}")
            continue

        extracted = extract_context_from_table(path)
        print(f"{path}: extracted {len(extracted)} sample rows")
        frames.append(extracted)

    if not frames:
        raise SystemExit("No input rows extracted.")

    combined = pd.concat(frames, ignore_index=True)

    # Prefer rows with coordinates.
    combined["has_coordinates"] = combined["lat"].notna() & combined["lon"].notna()

    combined = combined.sort_values(
        by=["SampleCode", "has_coordinates"],
        ascending=[True, False],
    )

    # Keep one row per sample code.
    context = combined.drop_duplicates(subset=["SampleCode"], keep="first").copy()

    context = context[
        [
            "SampleCode",
            "lat",
            "lon",
            "country",
            "region",
            "source_file",
        ]
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    context.to_csv(output, index=False)

    print("")
    print(f"Created: {output}")
    print(f"Unique samples: {len(context)}")
    print(f"Samples with lat/lon: {(context['lat'].notna() & context['lon'].notna()).sum()}")

    if context["lat"].isna().all() or context["lon"].isna().all():
        print("")
        print("WARNING: No coordinates were found.")
        print("The file was created as a template, but you need to fill lat/lon from the source that has geolocation.")


if __name__ == "__main__":
    main()