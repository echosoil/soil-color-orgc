import io
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps


def read_image_bgr(image_path: str, use_icc: bool = True):
    """
    Read image consistently.

    Steps:
        1. Open with Pillow.
        2. Apply EXIF orientation.
        3. If ICC profile exists and use_icc=True, convert to sRGB.
        4. Return OpenCV-style BGR uint8 image.
    """
    img = Image.open(image_path)

    # Very important: respect camera/phone orientation metadata.
    img = ImageOps.exif_transpose(img)

    icc_profile = img.info.get("icc_profile")

    if use_icc and icc_profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            target_profile = ImageCms.createProfile("sRGB")

            img = ImageCms.profileToProfile(
                img,
                source_profile,
                target_profile,
                outputMode="RGB",
            )
        except Exception:
            img = img.convert("RGB")
    else:
        img = img.convert("RGB")

    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return bgr