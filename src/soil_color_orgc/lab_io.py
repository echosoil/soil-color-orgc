from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import pandas as pd


def resolve_lab_paths(
    lab_inputs: str | Path | Iterable[str | Path],
) -> list[Path]:
    """
    Resolve explicit XLSX paths and glob patterns.

    Examples:
        data/lab/results_1.xlsx
        data/lab/*.xlsx
        ["data/lab/a.xlsx", "data/lab/b.xlsx"]
    """
    if isinstance(lab_inputs, (str, Path)):
        lab_inputs = [lab_inputs]

    resolved: list[Path] = []

    for item in lab_inputs:
        text = str(item)

        # Supports quoted patterns such as "data/lab/*.xlsx".
        matches = glob.glob(text)

        if matches:
            candidates = [Path(match) for match in matches]
        else:
            candidates = [Path(text)]

        for path in candidates:
            path = path.resolve()

            if not path.exists():
                raise FileNotFoundError(f"Laboratory file not found: {path}")

            if not path.is_file():
                raise ValueError(f"Laboratory input is not a file: {path}")

            if path.suffix.lower() not in {".xlsx", ".xls"}:
                raise ValueError(
                    f"Unsupported laboratory file type: {path}. "
                    "Expected XLSX or XLS."
                )

            if path not in resolved:
                resolved.append(path)

    if not resolved:
        raise ValueError("No laboratory XLSX files were found.")

    return resolved


def read_lab_workbooks(
    lab_inputs: str | Path | Iterable[str | Path],
    required_columns: Iterable[str] = ("ID", "orgC_lab"),
    preferred_sheet: str | None = None,
) -> pd.DataFrame:
    """
    Read and concatenate multiple laboratory workbooks.

    Every workbook must contain all required columns. Additional columns are
    preserved. A `_lab_source_file` column records the source workbook.
    """
    paths = resolve_lab_paths(lab_inputs)
    required_columns = list(required_columns)

    frames: list[pd.DataFrame] = []

    for source_order, path in enumerate(paths):
        workbook = pd.ExcelFile(path)

        if preferred_sheet and preferred_sheet in workbook.sheet_names:
            sheet_name = preferred_sheet
        elif "enriched" in workbook.sheet_names:
            sheet_name = "enriched"
        else:
            sheet_name = workbook.sheet_names[0]

        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
        )

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Laboratory file {path} is missing required "
                f"columns: {missing}. Available columns: "
                f"{list(df.columns)}"
            )

        df = df.dropna(how="all").copy()

        df["_lab_source_file"] = path.name
        df["_lab_source_sheet"] = sheet_name
        df["_lab_source_order"] = source_order
        df["_lab_row_order"] = range(len(df))

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined["_lab_input_order"] = range(len(combined))

    print(
        f"Combined {len(paths)} laboratory files into "
        f"{len(combined)} rows."
    )

    return combined