from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from soil_color_orgc.gray_calibration import locate_gray_scale_roi, crop_gray_scale_roi
from soil_color_orgc.image_processing import extract_color_features
from soil_color_orgc.service_api import create_app


def image():
    arr = np.full((800, 550, 3), (60, 90, 125), dtype=np.uint8)
    for i in range(11):
        arr[640:704, 44+40*i:44+40*(i+1)] = 20+20*i
    return arr


REGION = (44/550, .8, 484/550, .88)


def test_manual_geometry_and_patches():
    _, rect, info = locate_gray_scale_roi(image(), REGION)
    assert rect == (44, 640, 484, 704)
    assert info["source"] == "manual" and info["usable"]
    assert len(info["patch_rects"]) == 11
    r = extract_color_features(image(), gray_roi=REGION, use_gray_calibration=True)
    assert r["gray_calibration"] == "applied"
    assert r["gray_region"]["rect"] == list(rect)
    assert len(r["gray_region"]["patch_measurements"]) == 11


def test_reject_wrong_region_and_legacy_fallback():
    arr = np.full((800,550,3),(60,90,125),dtype=np.uint8)
    r = extract_color_features(arr, use_gray_calibration=True)
    assert r["gray_region"]["source"] == "fixed_fallback"
    assert not r["gray_region"]["usable"]
    assert r["gray_calibration"] == "failed_uncorrected"
    assert "gray_region_not_accepted" in r["warnings"]
    assert len(crop_gray_scale_roi(arr)) == 2


def test_api_gray_preview_matches_used_region(tmp_path):
    ref = tmp_path/"ref.csv"
    ref.write_text("h,V,C,X_D65,Y_D65,Z_D65\n10YR,3,4,0.15,0.14,0.07\n")
    buf = BytesIO()
    Image.fromarray(image()[...,::-1]).save(buf,"PNG")
    data=buf.getvalue()
    params={"gray_roi":",".join(map(str,REGION))}
    with TestClient(create_app(ref)) as c:
        headers={"Content-Type":"image/png"}
        p=c.post("/api/v1/gray-preview",params=params,content=data,headers=headers)
        assert p.status_code==200, p.text
        r=c.post("/api/v1/analyze",params={**params,"gray":"true"},content=data,headers=headers)
        assert r.status_code==200, r.text
        assert r.json()["gray_region"]["rect"]==p.json()["gray_region"]["rect"]
        assert r.json()["gray_calibration"]=="applied"
        assert c.post("/api/v1/analyze",params=params,content=data,headers=headers).status_code==422
