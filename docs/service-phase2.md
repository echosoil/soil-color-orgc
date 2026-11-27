# Soil Colour Analysis — Phase 2: local API and container preview

This phase builds on Phase 1. No ECHOREPO database, credentials, uploaded-image storage, model training, public deployment or EOSC integration. The legacy CLI and requirements.txt are unchanged. The API intentionally exposes **colour characterization only**, not the experimental carbon proxy. The reference CSV must be supplied locally by an authorized operator; do not assume its redistribution rights from the Apache software licence. It is not copied into the container build context. Tests and CI use synthetic images and a synthetic reference.

## Local Python (Python 3.12 recommended)

Use the separate Phase 1 virtual environment. On Ubuntu:

    source .venv-service/bin/activate
    python -m pip install -r requirements-api.txt pytest httpx
    PYTHONPATH=src python -m pytest -q tests
    export SOIL_COLOUR_REFERENCE="$PWD/data/munsell/rit_munsell.csv"
    PYTHONPATH=src python -m uvicorn soil_color_orgc.service_api:app --host 127.0.0.1 --port 8086 --workers 1 --no-access-log

On Windows Git Bash activate with source .venv-service/Scripts/activate. Use a locally authorized copy of the reference. Do not expose this unauthenticated preview beyond loopback.

## Docker Compose preview

    export SOIL_COLOUR_REFERENCE="$PWD/data/munsell/rit_munsell.csv"
    docker compose -p soil-colour-preview -f docker-compose.service.yml up -d --build
    docker compose -p soil-colour-preview -f docker-compose.service.yml ps
    docker compose -p soil-colour-preview -f docker-compose.service.yml logs --tail 50

The reference must exist and be readable by container UID 10001; mount read-only. Port 8086 on localhost only (override SOIL_COLOUR_PORT if needed). The service runs non-root, with a read-only root filesystem, a bounded tmpfs, 1 GiB memory, two CPUs and no added capabilities. These are initial preview limits, not validated service capacity. Dependency ranges are not a reproducible deployment lock; capture tested versions and image digest, pin dependencies, review vulnerabilities before public release.

For remote server testing from the local computer, use an authorized SSH tunnel (forward local port 8086 to 127.0.0.1:8086 on the server). Do not publish the container port on all interfaces. Stop only this independent project:

    docker compose -p soil-colour-preview -f docker-compose.service.yml down

## HTTP contract

- GET /health: ready after loading local reference, reference SHA-256.
- GET /api/v1/models: method identity, limits and disabled carbon prediction.
- GET /docs: interactive OpenAPI; /openapi.json for clients.
- POST /api/v1/analyze: raw JPEG, PNG or WebP bytes (not multipart), Content-Type must match decoded format; optional query gray=true and roi=x0,y0,x1,y1. ROI uses normalized coordinates after EXIF orientation and automatic portrait rotation, before downscaling. Defaults to central ROI. Returned pixel rectangle is in the processed image coordinates. Never silently treat the original landscape preview coordinates as processed coordinates.

Example (substitute an authorized soil photo):

    curl -fsS http://127.0.0.1:8086/health
    curl -fsS http://127.0.0.1:8086/api/v1/models
    curl -fsS -H 'Content-Type: image/jpeg' --data-binary @sample.jpg \
      'http://127.0.0.1:8086/api/v1/analyze?gray=false&roi=0.4,0.5,0.6,0.7'

For PNG set Content-Type: image/png. Response includes Lab, Munsell, DeltaE2000, quality warnings, selected pixel coverage, gray calibration status, reference hash and decode diagnostics. HTTP 200 with status review or insufficient_soil_pixels is a scientific quality result, not a transport failure; Lab is null if insufficient pixels. The ICC status is converted_to_srgb or assumed_srgb. Color appearance depends on capture conditions. A successful gray-calibration calculation is not proof of a real reference card or measurement accuracy. No validated SOC prediction is returned.

Limits: 8 MiB raw body (declared and actual stream), 15 seconds body read deadline, 20 million decoded pixels, maximum dimension 10,000, minimum 32x32; ICC <=1 MiB after extraction, no multi-frame, transparent or unsupported modes. Two in-flight uploads/analyses per process, excess 503 with Retry-After. CPU work is off the event loop. Limits are bounded heuristics; no hard execution deadline or comprehensive malicious-image security guarantee. Use one worker to maintain process-local limits. No application persistence of uploads or debug images; raw bytes are held transiently in memory, not logged by the application. Reverse proxy or host logging/dumps are separate concerns.

## Next phases

Phase 3: browser upload, preview in the same EXIF/portrait coordinate space, ROI selection and visual results. Phase 4: real image and lab-matched evaluation, reference provenance, model card and uncertainty. Phase 5: production-grade authentication, quotas, rate limiting, request isolation, dependency locks, reverse proxy, monitoring, capacity and privacy review. Phase 6: EOSC onboarding under agreed requirements. This preview is not intended for untrusted public traffic.
