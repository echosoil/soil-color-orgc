#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from soil_color_orgc.context_features import load_context, merge_context


def read_excel_prefer_enriched(path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    xls = pd.ExcelFile(path)

    sheets = {
        sheet: pd.read_excel(path, sheet_name=sheet)
        for sheet in xls.sheet_names
    }

    if "enriched" in sheets:
        main = sheets["enriched"]
    else:
        first = xls.sheet_names[0]
        main = sheets[first]

    return main, sheets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--context", default="data/context/sample_context.csv")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    main_df, sheets = read_excel_prefer_enriched(input_path)
    context_df = load_context(args.context)

    merged = merge_context(main_df, context_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        merged.to_excel(writer, sheet_name="enriched", index=False)

        for sheet_name, df in sheets.items():
            if sheet_name == "enriched":
                continue
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Created {output_path}")
    print(f"Rows: {len(merged)}")
    print(f"Rows with lat/lon: {merged[['lat', 'lon']].dropna().shape[0]}")


if __name__ == "__main__":
    main()