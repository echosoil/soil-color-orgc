#!/bin/bash
# Must be run from the root of the repository
# Remember first to chmod +x scripts/presentation/*  

./scripts/presentation/00_*
./scripts/presentation/01_*
./scripts/presentation/02_*
./scripts/presentation/03_*
./scripts/presentation/04_*
./scripts/presentation/05_*
./scripts/presentation/06_*

python3 scripts/run_model_experiment.py \
  --with-gray outputs/test_stat_orgC_enriched_with_gray.xlsx \
  --munsell data/munsell/rit_munsell.csv \
  --target orgC_lab \
  --test-count 51 \
  --cv-repeats 10 \
  --munsell-neighbours 6 \
  --seed 42 \
  --out outputs/model_experiment

python3 scripts/make_presentation_report.py \
  --lab data/lab/test_stat_orgC.xlsx \
  --with-gray outputs/test_stat_orgC_enriched_with_gray.xlsx \
  --experiment-dir outputs/model_experiment \
  --out outputs/presentation_report \
  --qc-max-samples 4
#  --qc-sample-codes APKC HGCM XGXK

# optional: start a local web server to view the report
# python3 -m http.server 8088 --directory outputs/presentation_report