#!/bin/bash
# Must be run from the project root.
python3 scripts/run_all.py \
  --samples data/samples \
  --results outputs/results_no_gray.csv \
  --enriched outputs/test_stat_orgC_enriched_no_gray.xlsx \
  --no-gray-calibration