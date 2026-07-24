from pathlib import Path
import re
import pandas as pd


def extract_sample_code(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    stem = Path(text).stem
    match = re.search(r"[A-Za-z]{4}", stem)

    if not match:
        return None

    return match.group(0).upper()


def add_sample_code(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "SampleCode" in df.columns:
        df["SampleCode"] = df["SampleCode"].apply(extract_sample_code)
        return df

    for col in ["sample_code_base", "ID", "image"]:
        if col in df.columns:
            df["SampleCode"] = df[col].apply(extract_sample_code)
            return df

    raise ValueError("Could not create SampleCode: no suitable column found.")


def load_context(context_csv: str | Path) -> pd.DataFrame:
    context = pd.read_csv(context_csv)
    context = add_sample_code(context)

    required = ["SampleCode", "lat", "lon"]

    missing = [c for c in required if c not in context.columns]
    if missing:
        raise ValueError(f"Context file is missing required columns: {missing}")

    context["lat"] = pd.to_numeric(context["lat"], errors="coerce")
    context["lon"] = pd.to_numeric(context["lon"], errors="coerce")

    return context


def merge_context(
    enriched_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched_df = add_sample_code(enriched_df)

    context_cols = [
        c for c in context_df.columns
        if c not in enriched_df.columns or c == "SampleCode"
    ]

    merged = enriched_df.merge(
        context_df[context_cols],
        on="SampleCode",
        how="left",
        validate="many_to_one",
    )

    return merged