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
from starlette.responses import FileResponse, JSONResponse, Response

from .service_preview import preview_payload, colour_swatches, gray_preview_payload

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


def analyze_payload(body, media, reference, roi, gray, gray_roi=None):
    image, decoding = decode_image(body, media)
    result = analyze_image(image, reference, roi=roi, gray=gray, gray_roi=gray_roi)
    result["swatches"] = colour_swatches(result, reference)
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
        version="0.3.1",
        description=(
            "Local-preview API for exploratory soil colour characterization. "
            "No validated carbon prediction, authentication or public deployment approval."
        ),
        lifespan=lifespan,
    )

    web_dir = Path(__file__).with_name("web")
    page_headers = {
        "Content-Security-Policy": (
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob:; connect-src 'self'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'none'"
        ),
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web_dir / "index.html", headers=page_headers)

    @app.get("/assets/soilvision.css", include_in_schema=False)
    def stylesheet():
        return FileResponse(web_dir / "soilvision.css", media_type="text/css",
                            headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/assets/soilvision.js", include_in_schema=False)
    def javascript():
        return FileResponse(web_dir / "soilvision.js", media_type="text/javascript",
                            headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "reference_sha256": request.app.state.reference.sha256}

    @app.get("/api/v1/models")
    def models(request: Request):
        return {
            "method": "soil-colour-v2-preview",
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
        gray_roi: str | None = Query(None, max_length=128, description="Exact manual grey strip, normalized x0,y0,x1,y1"),
        roi: str | None = Query(None, max_length=128, description=(
            "x0,y0,x1,y1 normalized coordinates after EXIF and portrait orientation"
        )),
    ):
        region = parse_roi(roi)
        manual = parse_roi(gray_roi)
        if manual is not None and not gray:
            raise HTTPException(422, "gray_roi_requires_gray_true")
        return await process_upload(
            request, analyze_payload, request.app.state.reference, region, gray, manual
        )

    @app.post(
        "/api/v1/preview",
        response_class=Response,
        responses={200: {"content": {"image/png": {}}, "description": "Oriented uncorrected preview"}},
        openapi_extra={"requestBody": {"required": True, "content": {
            m: {"schema": {"type": "string", "format": "binary"}} for m in MEDIA_FORMATS
        }}},
    )
    async def preview(request: Request):
        return await process_upload(request, preview_payload)

    @app.post("/api/v1/gray-preview", openapi_extra={"requestBody": {"required": True, "content": {
        m: {"schema": {"type": "string", "format": "binary"}} for m in MEDIA_FORMATS
    }}})
    async def gray_preview(request: Request, gray_roi: str | None = Query(None, max_length=128)):
        return await process_upload(request, gray_preview_payload, parse_roi(gray_roi))

    return app


async def process_upload(request, processor, *args):
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
            result = await run_in_threadpool(processor, body, media, *args)
        except ImageRejected as exc:
            status = 413 if str(exc) in {"image_dimensions_exceeded", "icc_profile_too_large"} else 422
            raise HTTPException(status, str(exc)) from exc
        except (ValueError, cv2.error) as exc:
            raise HTTPException(422, "image_analysis_failed") from exc
        if isinstance(result, bytes):
            return Response(result, media_type="image/png", headers={
                "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            })
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    finally:
        slots.release()


app = create_app()
