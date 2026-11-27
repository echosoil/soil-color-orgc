# Phase 4A — paired scientific validation of the actual service engine

The existing run_model_experiment.py is valuable historical exploratory work. Phase 4A deliberately re-extracts features using the current strict service implementation, rather than treating legacy enriched tables as equivalent. This is an offline local experiment, not a production prediction service. No lab/photos are in the Git repository. Do not send the raw manifest, private per-sample outputs or photographs without authorization.

## Prepare the review manifest

Use the local authorized photographs with grey cards and the matching lab spreadsheet. The existing project uses ID (e.g. ABCD-1234) and orgC_lab. A unique four-letter-prefix match is only a candidate; review actual sample identity and units. Ambiguous/unmatched IDs are excluded, not silently resolved.

    source .venv-service/bin/activate
    python -m pip install -r requirements-validation.txt pytest httpx
    mkdir -p outputs/validation
    python scripts/prepare_validation_manifest.py \
      --lab data/lab/test_stat_orgC.xlsx \
      --images data/samples/with_gray \
      --out outputs/validation/manifest.csv

Adapt file/sheet/target names if needed. The first sheet is default. The manifest contains sample_id, group_id, image_path, orgC_lab, soil_roi, gray_roi. Open and review it privately; fill group_id with a meaningful independent grouping (site, campaign, or higher-level grouping that keeps related photos/soils together). Do not make up independent groups simply to pass the check. If site/device/campaign grouping is unavailable, report that limitation; do not claim transferability across sites. Use one image per physical sample. Confirm lab units are organic carbon mass percent, not organic matter, g/kg or an unrelated target. Optional normalized ROI columns use quoted comma-separated coordinates from the web UI. For grey override use the precise confirmed strip; blank means automatic. Avoid selecting ROIs based on known lab values. Do not tune exclusions or thresholds after viewing results without recording an exploratory change.

## Execute

    OUT="outputs/validation/run-$(date -u +%Y%m%dT%H%M%SZ)"
    PYTHONPATH=src python scripts/validate_soilvision.py \
      --manifest outputs/validation/manifest.csv \
      --reference data/munsell/rit_munsell.csv \
      --ack-lab-percent-and-groups yes \
      --commit "$(git rev-parse HEAD)" --out "$OUT"

The script holds only one image pair at a time. It fingerprints input bytes and the manifest/reference; it refuses duplicate image bytes or duplicate sample IDs. The same photo and requested soil ROI are analyzed with correction off/on; the effective masks may differ. Corrected analysis must actually apply correction. Both Lab features must exist and warnings other than no_close_munsell_match exclude a pair. This predeclared conservative selection can bias results toward easy cases; report all disposition counts. Record exact input version and grouping separately. Code revision is operator-supplied; run on a clean tracked checkout. Output directory must be new.

At least 20 paired samples and five distinct groups are required for the initial experiment (not a scientifically sufficient sample-size claim). Identical five-fold GroupKFold splits are used for mean baseline, L-only and Lab linear models in each correction mode. No hyperparameter tuning or model persistence. Metrics are out-of-fold sample-weighted MAE/RMSE/bias in percentage points and R2; negative gray-minus-raw MAE favors gray correction on the common subset. No independent external test, uncertainty estimates or statistical significance claim. Grouping quality depends on operator data; not independently verified. Large site imbalance can affect aggregate metrics. If insufficient data, report the reason and improve the evidence rather than relaxing requirements silently.

Send only validation-summary.json for review. paired-features.local.json and predictions.local.json contain sample/group IDs and laboratory values; retain privately. The summary is aggregate but review for small-group disclosure. Next: inspect coverage and errors, compare with historical results, design independent site/camera holdout and uncertainty evaluation; only promote a model after scientific approval. The API still exposes no SOC prediction.
