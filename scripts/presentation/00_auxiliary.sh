#!/usr/bin/env bash
# Must be run from the project root.

set -euo pipefail

python3 scripts/make_color_swatches.py \
  --results outputs/results_combined.csv \
  --output-dir outputs/color_cards_combined