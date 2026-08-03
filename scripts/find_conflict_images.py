#!/usr/bin/env python3

"""
python3 scripts/find_conflict_images.py \
  --conflicts data/lab/test_stat_orgC_with_orgC_CS_ambiguous_base_codes.csv \
  --images-dir data/samples/with_gray \
  --output outputs/conflict_images_to_inspect.csv \
  --copy-dir outputs/conflict_images_to_inspect
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


def extract_base_code(value: Any) -> str | None:
    """
    Extract the four-letter sample prefix.

    Examples:
        WVSU             -> WVSU
        WVSU-1234        -> WVSU
        WVSU.jpg         -> WVSU
        ECHO-WVSU-1234   -> WVSU
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().upper()

    if not text:
        return None

    text = Path(text).stem

    # Prefer a four-letter token at the beginning.
    match = re.match(r"^([A-Z]{4})(?:$|[-_\s])", text)

    if match:
        return match.group(1)

    # Also support strings such as ECHO-WVSU-1234.
    match = re.search(r"(?<![A-Z])([A-Z]{4})(?![A-Z])", text)

    if match:
        return match.group(1)

    return None


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )


def detect_conflict_code_column(df: pd.DataFrame) -> str:
    """
    Find the column containing the conflicting base/sample code.
    """
    candidates = [
        "base_code",
        "sample_code",
        "exact_source_code",
        "normalized_code",
        "QR_qrCode",
        "ID",
    ]

    lower_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        found = lower_lookup.get(candidate.lower())

        if found is not None:
            return found

    raise ValueError(
        "Could not identify the conflict-code column.\n"
        "Available columns:\n  "
        + "\n  ".join(str(column) for column in df.columns)
    )


def load_conflicts(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)

    if df.empty:
        raise ValueError(f"The conflict file is empty: {path}")

    code_column = detect_conflict_code_column(df)

    df = df.copy()
    df["base_code"] = df[code_column].apply(extract_base_code)
    df = df.dropna(subset=["base_code"])
    df["base_code"] = df["base_code"].astype(str).str.upper()

    # Keep one record per four-letter code.
    df = df.drop_duplicates(subset=["base_code"], keep="first")

    return df


def scan_images(images_dir: Path) -> dict[str, list[Path]]:
    images_by_code: dict[str, list[Path]] = {}

    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        code = extract_base_code(path.stem)

        if code is None:
            continue

        images_by_code.setdefault(code, []).append(path)

    return images_by_code


def safe_copy_name(code: str, source: Path, sequence: int) -> str:
    """
    Create a unique filename in case multiple images share the same base code.
    """
    if sequence == 1:
        return f"{code}{source.suffix.lower()}"

    return f"{code}_{sequence}{source.suffix.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find with-gray images whose four-letter filenames occur "
            "in the orgC_CS conflict report."
        )
    )

    parser.add_argument(
        "--conflicts",
        default=(
            "data/lab/"
            "test_stat_orgC_with_orgC_CS_ambiguous_base_codes.csv"
        ),
        help="Conflict-report CSV.",
    )

    parser.add_argument(
        "--images-dir",
        default="data/samples/with_gray",
        help="Directory containing the with-gray images.",
    )

    parser.add_argument(
        "--output",
        default="outputs/conflict_images_to_inspect.csv",
        help="Output CSV listing matching images.",
    )

    parser.add_argument(
        "--copy-dir",
        default="outputs/conflict_images_to_inspect",
        help=(
            "Folder into which matching images will be copied. "
            "Use an empty string to disable copying."
        ),
    )

    args = parser.parse_args()

    conflicts_path = Path(args.conflicts)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    copy_dir = Path(args.copy_dir) if args.copy_dir else None

    if not conflicts_path.exists():
        raise FileNotFoundError(
            f"Conflict file not found: {conflicts_path}"
        )

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {images_dir}"
        )

    conflicts = load_conflicts(conflicts_path)
    images_by_code = scan_images(images_dir)

    output_rows: list[dict[str, Any]] = []

    if copy_dir is not None:
        if copy_dir.exists():
            shutil.rmtree(copy_dir)

        copy_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("Images to inspect")
    print("=================")

    found_codes = 0
    missing_codes = 0
    copied_images = 0

    conflict_columns = [
        column
        for column in conflicts.columns
        if column != "base_code"
    ]

    for _, conflict in conflicts.sort_values("base_code").iterrows():
        code = conflict["base_code"]
        matching_images = images_by_code.get(code, [])

        conflict_details = {
            column: conflict[column]
            for column in conflict_columns
        }

        if not matching_images:
            missing_codes += 1
            print(f"{code}: NO IMAGE FOUND")

            output_rows.append(
                {
                    "base_code": code,
                    "status": "image_not_found",
                    "image_path": "",
                    "copied_path": "",
                    **conflict_details,
                }
            )
            continue

        found_codes += 1

        print(f"{code}:")

        for sequence, image_path in enumerate(matching_images, start=1):
            copied_path = ""

            if copy_dir is not None:
                destination = copy_dir / safe_copy_name(
                    code,
                    image_path,
                    sequence,
                )

                shutil.copy2(image_path, destination)
                copied_path = destination.as_posix()
                copied_images += 1

            print(f"  {image_path}")

            output_rows.append(
                {
                    "base_code": code,
                    "status": "found",
                    "image_path": image_path.as_posix(),
                    "copied_path": copied_path,
                    **conflict_details,
                }
            )

    output = pd.DataFrame(output_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Also create a simple plain-text checklist.
    checklist_path = output_path.with_suffix(".txt")

    with checklist_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "Conflict images to inspect\n"
            "==========================\n\n"
        )

        for code in sorted(conflicts["base_code"].unique()):
            matching_images = images_by_code.get(code, [])

            if matching_images:
                for image_path in matching_images:
                    handle.write(
                        f"[ ] {code}: {image_path.as_posix()}\n"
                    )
            else:
                handle.write(
                    f"[!] {code}: IMAGE NOT FOUND\n"
                )

    print("")
    print("Summary")
    print("=======")
    print(f"Conflict codes: {len(conflicts)}")
    print(f"Codes with images: {found_codes}")
    print(f"Codes without images: {missing_codes}")
    print(f"Images copied: {copied_images}")
    print("")
    print(f"Detailed CSV: {output_path}")
    print(f"Checklist: {checklist_path}")

    if copy_dir is not None:
        print(f"Inspection folder: {copy_dir}")


if __name__ == "__main__":
    main()