#!/usr/bin/env bash
# Must be run from the root of the repository.

set -euo pipefail

INCLUDE_WITHOUT_GRAY="${INCLUDE_WITHOUT_GRAY:-1}"

LAB_FILES=(
  data/lab/test_stat_orgC.xlsx
  data/lab/test_stat_orgC_v1_enriched.xlsx
)

RESULTS_FILE="outputs/results_combined.csv"
ENRICHED_FILE="outputs/test_stat_orgC_enriched_combined.xlsx"
EXPERIMENT_DIR="outputs/model_experiment"
REPORT_DIR="outputs/presentation_report"

mkdir -p outputs

echo
echo "=== 1. Process images and merge laboratory files ==="

RUN_ALL_ARGS=(
  --samples data/samples/with_gray
  --lab "${LAB_FILES[@]}"
  --results "$RESULTS_FILE"
  --enriched "$ENRICHED_FILE"
)

if [[ "$INCLUDE_WITHOUT_GRAY" == "1" ]]; then
  RUN_ALL_ARGS+=(
    --include-without-gray
    --without-gray-samples data/samples
  )

  echo "Including legacy images without grey-scale correction."
else
  echo "Using only grey-scale images."
fi

python3 scripts/run_all.py "${RUN_ALL_ARGS[@]}"

echo
echo "=== 2. Generate colour-card QC images ==="

python3 scripts/make_color_swatches.py \
  --results "$RESULTS_FILE" \
  --output-dir outputs/color_cards_combined

echo
echo "=== 3. Compare citizen estimates with laboratory results ==="

python3 scripts/presentation/01_orgc_lab_vs_cs.py

echo
echo "=== 4. Compare direct SOC formula with laboratory results ==="

python3 scripts/presentation/06_combined_lab_vs_SOC_est.py

echo
echo "=== 5. Run trained model experiment ==="

python3 scripts/run_model_experiment.py \
  --with-gray "$ENRICHED_FILE" \
  --munsell data/munsell/rit_munsell.csv \
  --target orgC_lab \
  --test-count 51 \
  --cv-repeats 10 \
  --munsell-neighbours 6 \
  --seed 42 \
  --out "$EXPERIMENT_DIR"

echo
echo "=== 6. Generate presentation report ==="

python3 scripts/make_presentation_report.py \
  --lab "${LAB_FILES[@]}" \
  --with-gray "$ENRICHED_FILE" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --color-cards-with-gray-dir outputs/color_cards_combined \
  --out "$REPORT_DIR" \
  --qc-max-samples 4

echo
echo "Report created:"
echo "  $REPORT_DIR/index.html"
echo
echo "Open with:"
echo "  python3 -s -m http.server 18080 --directory $REPORT_DIR"