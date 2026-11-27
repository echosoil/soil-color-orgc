# Phase 4B — exploratory model comparison

Operator-reported 27 Nov 2026 Phase 4A sample-level run: 221/256 paired, 35 gray_not_applied, all 35 fixed_fallback with gray_region_not_accepted and gray_calibration_failed. Lab mean 3.420%, median 2.153%, max 45.769%; five >15%. Raw Lab MAE 2.223 pp, RMSE 4.296 pp, R2 0.2522; corrected Lab MAE 2.240 pp, RMSE 4.265 pp, R2 0.2629. Negative predictions 15 raw and 20 corrected Lab. These are supplied aggregate results, not independently reproduced. Check original lab units and IDs for high-carbon samples; do not discard based on errors. No geographic grouping or independent validation.

Run from the repository root in the existing validation environment, using cached paired features from the prior run (no new image processing):

    ROOT=outputs/validation/run-20261127T151736Z
    OUT="outputs/validation/model-comparison-$(date -u +%Y%m%dT%H%M%SZ)"
    PYTHONPATH=src python scripts/compare_soilvision_models.py \
      --features "$ROOT/paired-features.local.json" \
      --commit "$(git rev-parse HEAD)" --out "$OUT"

Uses same eligible paired cohort for raw/corrected, and a separately reported all-usable-raw cohort, including grey failures when raw QC permits. No pooling of metrics between cohorts. Mean and median baselines, historical unconstrained L and Lab models, clipped Lab, log-linear, polynomial ridge, log-polynomial ridge and gradient boosting (Huber). Prediction clipping to physical range [0,100] except historical linear comparators. Log inversion is not bias-corrected; evaluates original-scale errors, may underestimate high values. Degree/regularization and boosting hyperparameters selected by inner three-fold GroupKFold with clipped MAE; all families share identical five outer group folds within each cohort. Per-range errors, negative predictions, MAE, RMSE, bias, R2, median/p90/max errors. Output summary records chosen parameters and versions; private predictions contain sample IDs and lab values. No exported model or web changes. No significance tests, uncertainty estimates or independent holdout; the same previously examined cohort makes this exploratory even with nested CV. Do not choose a winner and claim its CV score as independent validation. Next establish independent/site-grouped holdout and prediction intervals before deployment. No retuning grey detector in this experiment. Inspect the 35 fallback images visually and only change ROI independent of lab outcomes, document separately.
