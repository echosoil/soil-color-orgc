"""Bounded, in-memory image decoder for the service (not the legacy CLI)."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageCms, ImageOps

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_DIMENSION = 10000
MAX_ICC_BYTES = 1024 * 1024
MEDIA_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


class ImageRejected(ValueError):
    """A safe, machine-readable image rejection code."""


def decode_image(payload: bytes, media_type: str):
    expected = MEDIA_FORMATS.get(media_type)
    if expected is None:
        raise ImageRejected("unsupported_media_type")
    if not payload or len(payload) > MAX_BYTES:
        raise ImageRejected("invalid_image_size")
    try:
        with Image.open(BytesIO(payload), formats=list(MEDIA_FORMATS.values())) as original:
            if original.format != expected:
                raise ImageRejected("image_format_mismatch")
            width, height = original.size
            if min(width, height) < 32:
                raise ImageRejected("image_too_small")
            if max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
                raise ImageRejected("image_dimensions_exceeded")
            if getattr(original, "n_frames", 1) != 1:
                raise ImageRejected("animated_image_not_supported")
            if original.mode not in {"RGB", "RGBA", "L", "LA", "P", "CMYK"}:
                raise ImageRejected("unsupported_colour_mode")
            if original.mode in {"RGBA", "LA"} or "transparency" in original.info:
                raise ImageRejected("transparent_image_not_supported")
            profile = original.info.get("icc_profile")
            if profile and len(profile) > MAX_ICC_BYTES:
                raise ImageRejected("icc_profile_too_large")
            oriented = ImageOps.exif_transpose(original)
            if profile:
                try:
                    rgb = ImageCms.profileToProfile(
                        oriented,
                        ImageCms.ImageCmsProfile(BytesIO(profile)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGB",
                    )
                except Exception as exc:
                    raise ImageRejected("invalid_icc_profile") from exc
                icc_status = "converted_to_srgb"
            else:
                if oriented.mode == "CMYK":
                    raise ImageRejected("cmyk_requires_icc_profile")
                rgb = oriented.convert("RGB")
                icc_status = "assumed_srgb"
            pixels = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[..., ::-1])
            return pixels, {
                "decoded_format": expected,
                "icc_status": icc_status,
                "exif_orientation_applied": True,
                "portrait_rotation_clockwise": 90 if pixels.shape[1] > pixels.shape[0] else 0,
            }
    except ImageRejected:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageRejected("image_dimensions_exceeded") from exc
    except (OSError, ValueError, SyntaxError, EOFError, TypeError, OverflowError) as exc:
        raise ImageRejected("invalid_image") from exc
