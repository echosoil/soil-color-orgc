# Automated Soil orgC Presentation Report

This package contains a properly formatted version of the report-generation instructions and a ready-to-save Python script.

The script generates a **static HTML report** with:

1. Laboratory orgC vs citizen-science orgC estimates.
2. Laboratory orgC vs image algorithm without grey-scale correction, using cross-validated training.
3. Laboratory orgC vs image algorithm with grey-scale correction, without training, because the current grey-scale subset is still too small.

## 1. Save the script

Place the script here:

```text
scripts/make_presentation_report.py
```

In this package, the formatted script is provided as:

```text
make_presentation_report.py
```

Copy it into your repository:

```bash
mkdir -p scripts
cp make_presentation_report.py scripts/make_presentation_report.py
```

## 2. Install dependencies

```bash
python3 -m pip install pandas numpy scipy scikit-learn matplotlib openpyxl
```

## 3. Generate the report

```bash
python3 scripts/make_presentation_report.py   --lab data/lab/test_stat_orgC.xlsx   --no-gray outputs/test_stat_orgC_enriched_no_gray.xlsx   --with-gray outputs/test_stat_orgC_enriched_with_gray.xlsx   --out outputs/presentation_report
```

The script will generate:

```text
outputs/presentation_report/index.html
outputs/presentation_report/summary_metrics.csv
outputs/presentation_report/citizen_predictions.csv
outputs/presentation_report/no_gray_trained_predictions.csv
outputs/presentation_report/with_gray_direct_predictions.csv
outputs/presentation_report/figures/
```

## 4. Open the report locally

```bash
python3 -m http.server 8088 --directory outputs/presentation_report
```

Open this URL in your browser:

```text
http://localhost:8088
```

## 5. Suggested full workflow

Run both processing modes first, then build the presentation report:

```bash
python3 scripts/run_all.py   --samples data/samples   --results outputs/results_no_gray.csv   --enriched outputs/test_stat_orgC_enriched_no_gray.xlsx   --no-gray-calibration

python3 scripts/run_all.py   --samples data/samples/with_gray   --results outputs/results_with_gray.csv   --enriched outputs/test_stat_orgC_enriched_with_gray.xlsx

python3 scripts/make_presentation_report.py   --lab data/lab/test_stat_orgC.xlsx   --no-gray outputs/test_stat_orgC_enriched_no_gray.xlsx   --with-gray outputs/test_stat_orgC_enriched_with_gray.xlsx   --out outputs/presentation_report
```

## 6. Optional Docker hosting

Because the report is static HTML, you do not need a real web app. Nginx is enough.

Create this file:

```text
Dockerfile.report
```

```dockerfile
FROM nginx:alpine
COPY outputs/presentation_report /usr/share/nginx/html
EXPOSE 80
```

Build and run:

```bash
docker build -f Dockerfile.report -t soil-orgc-report .
docker run --rm -p 8088:80 soil-orgc-report
```

Then open:

```text
http://localhost:8088
```

For server deployment, copy the repository or only this folder:

```text
outputs/presentation_report/
```

Then run the same Nginx container on the server.

## 7. Interpretation to use in the presentation

The safest scientific interpretation is:

- Citizen estimates provide a useful baseline, but the level of agreement with laboratory results must be assessed directly.
- The no-grey-scale algorithm should be presented using cross-validated calibrated predictions, not only training-set performance.
- The grey-scale algorithm should currently be presented as a preliminary direct comparison. If it shows stronger correlation but still weak agreement, describe it as a promising calibration signal, not as a validated direct estimator.
- Once 100-200 grey-scale samples are available, rerun the same report and change the grey-scale section from direct comparison to trained cross-validated calibration.
