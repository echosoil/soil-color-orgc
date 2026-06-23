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

python3 scripts/make_presentation_report.py \
  --lab data/lab/test_stat_orgC.xlsx \
  --no-gray outputs/test_stat_orgC_enriched_no_gray_cv_Lab_prediction.xlsx \
  --with-gray outputs/test_stat_orgC_enriched_with_gray.xlsx \
  --out outputs/presentation_report

# optional: start a local web server to view the report
# python3 -m http.server 8088 --directory outputs/presentation_report