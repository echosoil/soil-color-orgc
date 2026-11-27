import numpy as np
import pytest

from soil_color_orgc import image_processing as ip
from soil_color_orgc.analysis import Reference, analyze_image
from soil_color_orgc.munsell import find_best_munsell


def reference():
    values = {"10YR 3/4": (40., 12., 24.), "10YR 5/4": (60., 10., 20.)}
    return Reference(tuple(values), np.array(list(values.values())), "synthetic")


def soil():
    return np.full((400, 300, 3), (60, 90, 125), dtype=np.uint8)


def test_vectorized_matches_original():
    ref = reference()
    for lab in [(40., 12., 24.), (60., 10., 20.), (85., 0., 0.)]:
        expected, d = find_best_munsell(lab, dict(zip(ref.names, ref.lab)))
        name, distance = ref.match(lab)
        assert name == (None if expected == "No close Munsell match" else expected)
        assert distance == pytest.approx(d)


def test_white_background_rejected():
    r = analyze_image(np.full((400, 300, 3), 255, dtype=np.uint8), reference())
    assert r["status"] == "insufficient_soil_pixels"
    assert r["lab"] is None and r["carbon_proxy"] is None
    assert not r["used_full_roi_fallback"]


def test_colour_and_roi_no_carbon_default():
    lab = ip.extract_color_features(soil())["lab"]
    ref = Reference(("10YR 3/4",), np.array([lab]), "synthetic")
    r = analyze_image(soil(), ref, roi=(.2, .2, .8, .8))
    assert r["status"] == "ok"
    assert r["munsell"] == "10YR 3/4"
    assert r["delta_e_2000"] == pytest.approx(0)
    assert r["carbon_proxy"] is None
    assert r["roi_pixels"] == [60, 80, 240, 320]
    assert r["selected_fraction"] == 1
    assert analyze_image(soil(), ref, exploratory_carbon=True)["carbon_proxy"]["scientific_status"].startswith("unvalidated")


def test_gray_failure_visible(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("no card")
    monkeypatch.setattr(ip, "correct_image_using_gray_scale", fail)
    r = analyze_image(soil(), reference(), gray=True, exploratory_carbon=True)
    assert r["gray_calibration"] == "failed_uncorrected"
    assert "gray_calibration_failed" in r["warnings"]
    assert r["status"] == "review" and r["carbon_proxy"] is None


@pytest.mark.parametrize("roi", [(0, 0, 2, 1), (.5, 0, .1, 1), (0, 0, float("nan"), 1)])
def test_invalid_roi(roi):
    with pytest.raises(ValueError):
        analyze_image(soil(), reference(), roi=roi)


def test_legacy_fallback():
    r = ip.extract_color_features(np.full((400, 300, 3), 255, dtype=np.uint8), strict=False)
    assert r["lab"] is not None and r["used_full_roi_fallback"]
