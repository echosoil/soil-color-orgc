from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageCms

from soil_color_orgc import service_api, service_image


@pytest.fixture
def client(tmp_path):
    reference = tmp_path / "reference.csv"
    reference.write_text("h,V,C,X_D65,Y_D65,Z_D65\n10YR,3,4,0.15,0.14,0.07\n")
    with TestClient(service_api.create_app(reference)) as c:
        yield c


def image_bytes(color=(125, 90, 60), size=(300, 400), fmt="PNG", **kwargs):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def test_health_models_and_docs(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    models = client.get("/api/v1/models").json()
    assert models["reference_sha256"] == health["reference_sha256"]
    assert models["carbon_prediction"] == "disabled"
    spec = client.get("/openapi.json").json()
    assert "image/png" in spec["paths"]["/api/v1/analyze"]["post"]["requestBody"]["content"]


def test_analysis_and_roi(client):
    r = client.post("/api/v1/analyze?roi=0.2,0.2,0.8,0.8",
                    content=image_bytes(), headers={"Content-Type": "image/png"})
    assert r.status_code == 200, r.text
    result = r.json()
    assert len(result["lab"]) == 3
    assert result["carbon_proxy"] is None
    assert result["roi_pixels"] == [60, 80, 240, 320]
    assert result["image_decoding"]["icc_status"] == "assumed_srgb"
    assert r.headers["Cache-Control"] == "no-store"


def test_white_rejected_as_scientific_result(client):
    r = client.post("/api/v1/analyze", content=image_bytes((255, 255, 255)),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 200
    assert r.json()["status"] == "insufficient_soil_pixels"
    assert r.json()["lab"] is None


@pytest.mark.parametrize("media,body,code", [
    ("text/plain", b"bad", 415),
    ("image/png", b"", 400),
    ("image/png", b"bad", 422),
    ("image/jpeg", image_bytes(), 422),
])
def test_rejections(client, media, body, code):
    r = client.post("/api/v1/analyze", content=body, headers={"Content-Type": media})
    assert r.status_code == code


def test_invalid_roi(client):
    r = client.post("/api/v1/analyze?roi=0,0,2,1", content=image_bytes(),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 422 and r.json()["detail"] == "invalid_roi"


def test_declared_and_streamed_body_limits(client, monkeypatch):
    monkeypatch.setattr(service_api, "MAX_BYTES", 100)
    r = client.post("/api/v1/analyze", content=b"x" * 101,
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 413
    r = client.post("/api/v1/analyze", content=iter([b"x" * 60, b"x" * 60]),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 413
    # Failure must release the capacity slot.
    assert client.app.state.slots.acquire(blocking=False)
    assert client.app.state.slots.acquire(blocking=False)
    client.app.state.slots.release()
    client.app.state.slots.release()


def test_pixel_limit(client, monkeypatch):
    monkeypatch.setattr(service_image, "MAX_PIXELS", 10000)
    r = client.post("/api/v1/analyze", content=image_bytes(),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 413
    assert r.json()["detail"] == "image_dimensions_exceeded"


def test_busy(client):
    slots = client.app.state.slots
    for _ in range(service_api.MAX_ACTIVE_REQUESTS):
        assert slots.acquire(blocking=False)
    try:
        r = client.post("/api/v1/analyze", content=image_bytes(),
                        headers={"Content-Type": "image/png"})
        assert r.status_code == 503 and r.headers["Retry-After"] == "2"
    finally:
        for _ in range(service_api.MAX_ACTIVE_REQUESTS):
            slots.release()


def test_icc_and_rotation(client):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    r = client.post("/api/v1/analyze", content=image_bytes(size=(400, 300), icc_profile=profile),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 200
    result = r.json()
    assert result["image_decoding"]["icc_status"] == "converted_to_srgb"
    assert result["image_decoding"]["portrait_rotation_clockwise"] == 90
    assert result["image_width"] == 300 and result["image_height"] == 400


def test_exif_orientation(client):
    exif = Image.Exif()
    exif[274] = 6
    r = client.post("/api/v1/analyze", content=image_bytes(size=(400, 300), fmt="JPEG", exif=exif),
                    headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200
    assert r.json()["image_decoding"]["portrait_rotation_clockwise"] == 0


def test_transparent_rejected(client):
    output = BytesIO()
    Image.new("RGBA", (300, 400), (125, 90, 60, 0)).save(output, "PNG")
    r = client.post("/api/v1/analyze", content=output.getvalue(),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 422
    assert r.json()["detail"] == "transparent_image_not_supported"


def test_no_reference_fails_startup(monkeypatch):
    monkeypatch.delenv("SOIL_COLOUR_REFERENCE", raising=False)
    with pytest.raises(RuntimeError, match="SOIL_COLOUR_REFERENCE"):
        with TestClient(service_api.create_app()):
            pass
