"""Preview geometry and display-only colour conversions."""
import cv2
import numpy as np
from colour import Lab_to_XYZ, XYZ_to_sRGB

from .image_processing import downscale_max_side, ensure_portrait_orientation
from .service_image import decode_image


def preview_payload(body, media):
    image, _ = decode_image(body, media)
    image = downscale_max_side(ensure_portrait_orientation(image), 800)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Preview encoding failed")
    return encoded.tobytes()


def display_colour(lab):
    rgb = np.asarray(XYZ_to_sRGB(Lab_to_XYZ(lab)), dtype=float)
    if not np.isfinite(rgb).all():
        return None
    clipped = bool(np.any((rgb < 0) | (rgb > 1)))
    channels = np.rint(np.clip(rgb, 0, 1) * 255).astype(int)
    return {"hex": "#" + "".join(f"{v:02x}" for v in channels),
            "gamut_clipped": clipped}


def colour_swatches(result, reference):
    measured = display_colour(result["lab"]) if result["lab"] is not None else None
    match = result["munsell"]
    matched = display_colour(reference.lab[reference.names.index(match)]) if match else None
    return {"extracted": measured, "munsell": matched, "space": "sRGB display approximation"}
