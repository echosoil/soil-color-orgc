#!/usr/bin/env bash
# Must be run from the project root.

set -euo pipefail

INCLUDE_WITHOUT_GRAY="${INCLUDE_WITHOUT_GRAY:-1}"

ARGS=(
  --samples data/samples/with_gray
  --lab
    data/lab/test_stat_orgC.xlsx
    data/lab/test_stat_orgC_v1_enriched.xlsx
  --results outputs/results_combined.csv
  --enriched outputs/test_stat_orgC_enriched_combined.xlsx
)

if [[ "$INCLUDE_WITHOUT_GRAY" == "1" ]]; then
  ARGS+=(
    --include-without-gray
    --without-gray-samples data/samples
  )
fi

python3 scripts/run_all.py "${ARGS[@]}"