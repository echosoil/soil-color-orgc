# Soil Colour Analysis — Phase 1

The service is a proposed independent ECHO research offering, not an EOSC-integrated service. The initial engine reuses the existing portrait orientation, grayscale correction, ROI trimming, masking, Lab median and Munsell logic. It makes calibration failures and insufficient pixels visible. Strict mode refuses an unmasked fallback; the existing batch wrapper retains its historical fallback. Munsell lookup is vectorized with matching threshold and insertion-order ties. No network, database, model training or shared debug directories are used by the standalone entrypoint. The gray correction status "applied" means the algorithm ran, not that a physical reference card was verified. The proposed 500-pixel and 10% coverage thresholds are heuristics, not experimentally established acceptance criteria.

## Run offline tests

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-service.txt pytest
    PYTHONPATH=src python -m pytest -q tests

Use a new environment, not the historical CLI environment (opencv-python and headless OpenCV should not coexist). Tests use synthetic images and synthetic reference values; they do not validate real soils, calibration accuracy or reference rights. Dependency ranges are initial compatibility bounds, not a reproducible deployment lock.

## Python interface

    from soil_color_orgc.analysis import Reference, analyze_image
    from soil_color_orgc.image_io import read_image_bgr
    ref = Reference.from_csv("data/munsell/rit_munsell.csv")
    result = analyze_image(read_image_bgr("sample.jpg"), ref)
    print(result)

Optional roi=(x0,y0,x1,y1) uses normalized coordinates after EXIF and automatic portrait orientation; gray=True requests the existing eleven-patch correction. Review warnings; absence of warnings is not proof of a valid soil measurement. Carbon proxy is disabled by default, opt-in only for exploratory status-ok results, never a validated SOC estimate. No calibrated model is supplied. Reference hash identifies bytes, not provenance or legal rights.

## Next phases

Phase 2: bounded image upload and decoding (byte/pixel/format/ICC limits before decode), isolated temporary data, request limits, FastAPI/OpenAPI, web preview with correct orientation and interactive ROI, no uploaded-image retention, safe error codes, container with locked dependencies, health checks. Verify reference CSV provenance and redistribution terms before packaging/deploying it; software licence does not automatically cover reference data. Phase 3: scientific evaluation with approved lab-matched data, grouping by physical sample/site/device, independent holdout, model card and uncertainty; review card detection/quality thresholds, diverse soils and capture protocol. Phase 4: controlled deployment, capacity/security/privacy/support and EOSC onboarding. No production deployment or public service yet.
