"""Standalone, bounded soil-colour analysis; no database or batch files."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from colour.difference import delta_E

from .image_processing import extract_color_features
from .munsell import load_munsell
from .soc_estimation import estimate_soc


@dataclass(frozen=True)
class Reference:
    names: tuple
    lab: np.ndarray
    sha256: str

    @classmethod
    def from_csv(cls, path):
        path = Path(path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        values = load_munsell(str(path))
        if before != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Reference changed during load")
        lab = np.asarray(list(values.values()), dtype=float)
        if not len(values) or lab.shape != (len(values), 3) or not np.isfinite(lab).all():
            raise ValueError("Invalid Munsell reference")
        lab.setflags(write=False)
        return cls(tuple(values), lab, before)

    def match(self, lab, threshold=8.0):
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid colour distance threshold")
        distances = delta_E(np.asarray(lab), self.lab, method="CIE 2000")
        idx = int(np.argmin(distances))
        distance = float(distances[idx])
        return (self.names[idx] if distance <= threshold else None), distance


def analyze_image(image_bgr, reference, *, roi=None, gray=False, gray_roi=None, exploratory_carbon=False):
    result = extract_color_features(image_bgr, roi=roi, use_gray_calibration=gray, gray_roi=gray_roi)
    result.update({
        "method_version": "soil-colour-v2-preview",
        "reference_sha256": reference.sha256,
        "munsell": None, "delta_e_2000": None, "carbon_proxy": None,
    })
    if result["lab"] is None:
        return result
    notation, distance = reference.match(result["lab"])
    result["munsell"] = notation
    result["delta_e_2000"] = distance
    if notation is None:
        result["warnings"].append("no_close_munsell_match")
        result["status"] = "review"
    if exploratory_carbon and result["status"] == "ok":
        value, method = estimate_soc(result["lab"], notation)
        result["carbon_proxy"] = {
            "value_pct": value,
            "method": method,
            "scientific_status": "unvalidated_heuristic_not_lab_measurement",
        }
    return result
