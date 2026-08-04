#!/usr/bin/env python3
# Must be run from the project root.

import importlib.util
from pathlib import Path


script = Path("scripts/compare_orgC_lab_vs_CS.py")

spec = importlib.util.spec_from_file_location(
    "compare_orgC",
    script,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not load comparison script: {script}"
    )

compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

compare.INPUT_XLSX = (
    "outputs/test_stat_orgC_enriched_combined.xlsx"
)

compare.OUTPUT_DIR = (
    "outputs/orgC_comparison_outputs/"
    "02_combined_lab_vs_SOC_est"
)

compare.LAB_COL = "orgC_lab"
compare.CS_COL = "SOC_est%"
compare.ID_COL = "ID"

compare.main()