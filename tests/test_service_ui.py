from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from soil_color_orgc.service_api import create_app


@pytest.fixture
def client(tmp_path):
    ref = tmp_path / "reference.csv"
    ref.write_text("h,V,C,X_D65,Y_D65,Z_D65\n10YR,3,4,0.15,0.14,0.07\n")
    with TestClient(create_app(ref)) as c:
        yield c


def photo(size=(1200, 800), orientation=None):
    b = BytesIO()
    im = Image.new("RGB", size, (125, 90, 60))
    if orientation:
        exif = Image.Exif()
        exif[274] = orientation
        im.save(b, "JPEG", exif=exif)
    else:
        im.save(b, "JPEG")
    return b.getvalue()


def test_page_and_assets(client):
    page = client.get("/")
    assert page.status_code == 200 and "SoilVision" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    for path, media in [("/assets/soilvision.css","text/css"),("/assets/soilvision.js","text/javascript")]:
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith(media)


@pytest.mark.parametrize("orientation", [None, 6, 3])
def test_preview_matches_analysis_geometry(client, orientation):
    data = photo(orientation=orientation)
    headers = {"Content-Type":"image/jpeg"}
    p = client.post("/api/v1/preview", content=data, headers=headers)
    assert p.status_code == 200
    assert p.headers["cache-control"] == "no-store"
    with Image.open(BytesIO(p.content)) as preview:
        assert preview.format == "PNG"
        assert not preview.getexif()
        size = preview.size
    r = client.post("/api/v1/analyze", content=data, headers=headers)
    assert r.status_code == 200
    result = r.json()
    assert size == (result["image_width"], result["image_height"])
    assert result["swatches"]["extracted"]["hex"].startswith("#")
    assert result["carbon_proxy"] is None


def test_preview_rejects_bad_image(client):
    r = client.post("/api/v1/preview", content=b"bad", headers={"Content-Type":"image/jpeg"})
    assert r.status_code == 422


def test_preview_busy(client):
    slots = client.app.state.slots
    assert slots.acquire(blocking=False)
    assert slots.acquire(blocking=False)
    try:
        assert client.post("/api/v1/preview",content=photo(),headers={"Content-Type":"image/jpeg"}).status_code == 503
    finally:
        slots.release()
        slots.release()
