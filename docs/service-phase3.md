# SoilVision — Phase 3 browser interface (restricted preview)

Phase 2 was merged into main. Operator reported successful real-photo analysis of ABVA.jpg with gray=false: Lab [34.5447,13.2750,27.5474], 97.97% selected ROI pixels, 7.5YR 3/4, DeltaE2000 4.0352, status ok and no fallback. This is an execution observation, not scientific validation of colour or carbon.

Phase 3 adds the browser interface at /, with no separate frontend process. The server returns an oriented/downscaled PNG preview using the same decoder and geometry as the analysis; it contains no original EXIF. Preview is uncorrected even when analysis requests grey correction. The selected rectangle is normalized to preview coordinates, and the used (trimmed) rectangle is overlaid separately after analysis. Numeric inputs provide keyboard alternatives to dragging. Swatches are approximate sRGB conversions of Lab and reference Lab; clipped gamut is disclosed. A rejected colour result is not presented as a valid match. JSON and CSV export the result, requested coordinates, grey request and diagnostics. No validated carbon output. No image persistence in the application; the original bytes are sent twice (preview and analysis). Both endpoints share existing upload and concurrency limits. The UI has no third-party frontend dependencies or external analytics. Existing API fields remain unchanged; swatches is additive. No changes to Docker port, legacy CLI, or reference licence posture.

## Test and start

Use the same separate service environment from Phase 2. Stop the old foreground Uvicorn instance before reusing port 8086.

    python -m pip install -r requirements-api.txt pytest httpx
    PYTHONPATH=src python -m pytest -q tests
    export SOIL_COLOUR_REFERENCE="$PWD/data/munsell/rit_munsell.csv"
    PYTHONPATH=src python -m uvicorn soil_color_orgc.service_api:app --host 127.0.0.1 --port 8086 --workers 1 --no-access-log

If using Docker instead, rebuild the independent compose project per service-phase2.md. Do not run both simultaneously.

Open http://localhost:8086/ in a browser. On a remote host, forward local port 8086 through SSH or VS Code Ports to remote port 8086. Keep loopback binding. Check upload, ROI via dragging and numeric inputs, grey on/off, insufficient-soil and malformed images, JSON/CSV downloads, mobile width, and rotation with landscape/EXIF images. Verify displayed result against direct API requests with the same options. Check that changing inputs clears stale results. Record aggregate observations; do not commit photos or sensitive data.

No public Nginx exposure yet: authentication, proxy limits, rate limits, dependency locks, scientific evaluation, reference rights, and operational checks remain. Planned hostname soilvision.quanta-labs.com; port 8086 on host, 8080 inside container. Formal EOSC acceptance remains pending.
