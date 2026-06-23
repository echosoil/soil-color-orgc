#!/bin/bash
# Must be run from the root of the repository

python3 scripts/make_color_swatches.py \
        --results outputs/results_no_gray.csv \
        --output-dir outputs/color_cards_no_gray

python3 scripts/make_color_swatches.py \
        --results outputs/results_with_gray.csv \
        --output-dir outputs/color_cards_with_gray