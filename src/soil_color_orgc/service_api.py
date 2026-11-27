"""Local-preview HTTP interface for soil-colour analysis."""
import asyncio
from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
from threading import BoundedSemaphore

import cv2
from fastapi import FastAPI, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse

from .analysis import Reference, analyze_image
from .service_image import (
    MEDIA_FORMATS, MAX_BYTES, MAX_DIMENSION, MAX_PIXELS, ImageRejected, decode_image,
)

MAX_ACTIVE_REQUESTS = 2
BODY_TIMEOUT_SECONDS = 15


def parse_roi(raw):
    if raw is None:
        return None
    try:
        points = tuple(float(v) for v in raw.split(","))
        if len(points) != 4 or not all(math.isfinite(v) for v in points):
            raise ValueError
        x0, y0, x1, y1 = points
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError
        return points
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "invalid_roi") from exc


async def read_bounded_body(request):
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError as exc:
            raise HTTPException(400, "invalid_content_length") from exc
        if length < 0:
            raise HTTPException(400, "invalid_content_length")
        if length > MAX_BYTES:
            raise HTTPException(413, "image_too_large")
    body = bytearray()
    try:
        async with asyncio.timeout(BODY_TIMEOUT_SECONDS):
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_BYTES:
                    raise HTTPException(413, "image_too_large")
                body.extend(chunk)
    except TimeoutError as exc:
        raise HTTPException(408, "image_upload_timeout") from exc
    except ClientDisconnect as exc:
        raise HTTPException(400, "image_upload_disconnected") from exc
    if not body:
        raise HTTPException(400, "empty_image")
    return bytes(body)


def analyze_payload(body, media, reference, roi, gray):
    image, decoding = decode_image(body, media)
    result = analyze_image(image, reference, roi=roi, gray=gray)
    result["image_decoding"] = decoding
    result["scientific_status"] = "exploratory_colour_characterization_not_lab_measurement"
    return result


def create_app(reference_path=None):
    @asynccontextmanager
    async def lifespan(app):
        configured = reference_path or os.environ.get("SOIL_COLOUR_REFERENCE")
        if not configured:
            raise RuntimeError("Set SOIL_COLOUR_REFERENCE to an approved local reference CSV")
        app.state.reference = Reference.from_csv(Path(configured))
        app.state.slots = BoundedSemaphore(MAX_ACTIVE_REQUESTS)
        cv2.setNumThreads(1)
        yield

    app = FastAPI(
        title="ECHO Soil Colour Analysis",
        version="0.2.0",
        description=(
            "Local-preview API for exploratory soil colour characterization. "
            "No validated carbon prediction, authentication or public deployment approval."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "reference_sha256": request.app.state.reference.sha256}

    @app.get("/api/v1/models")
    def models(request: Request):
        return {
            "method": "soil-colour-v1-preview",
            "reference_sha256": request.app.state.reference.sha256,
            "carbon_prediction": "disabled",
            "limits": {
                "max_upload_bytes": MAX_BYTES, "max_pixels": MAX_PIXELS,
                "max_dimension": MAX_DIMENSION, "media_types": list(MEDIA_FORMATS),
                "max_active_requests": MAX_ACTIVE_REQUESTS,
            },
        }

    @app.post(
        "/api/v1/analyze",
        responses={
            400: {"description": "Empty or malformed upload"},
            408: {"description": "Upload timed out"},
            413: {"description": "Upload or image dimensions exceed limits"},
            415: {"description": "Unsupported media type or encoding"},
            422: {"description": "Invalid image or ROI"},
            503: {"description": "Analysis capacity exhausted"},
        },
        openapi_extra={"requestBody": {"required": True, "content": {
            m: {"schema": {"type": "string", "format": "binary"}} for m in MEDIA_FORMATS
        }}},
    )
    async def analyze(
        request: Request,
        gray: bool = Query(False, description="Request eleven-patch grey correction"),
        roi: str | None = Query(None, max_length=128, description=(
            "x0,y0,x1,y1 normalized coordinates after EXIF and portrait orientation"
        )),
    ):
        region = parse_roi(roi)
        media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media not in MEDIA_FORMATS:
            raise HTTPException(415, "unsupported_media_type")
        if request.headers.get("content-encoding", "identity").strip().lower() != "identity":
            raise HTTPException(415, "unsupported_content_encoding")
        slots = request.app.state.slots
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "analysis_busy", headers={"Retry-After": "2"})
        try:
            body = await read_bounded_body(request)
            try:
                result = await run_in_threadpool(
                    analyze_payload, body, media, request.app.state.reference, region, gray
                )
            except ImageRejected as exc:
                status = 413 if str(exc) in {"image_dimensions_exceeded", "icc_profile_too_large"} else 422
                raise HTTPException(status, str(exc)) from exc
            except (ValueError, cv2.error) as exc:
                raise HTTPException(422, "image_analysis_failed") from exc
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
        finally:
            slots.release()

    return app


app = create_app()
