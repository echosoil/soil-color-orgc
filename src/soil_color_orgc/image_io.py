import io
import numpy as np
import cv2
from PIL import Image, ImageCms


def read_image_bgr_color_managed(image_path: str):
    img = Image.open(image_path)

    icc_profile = img.info.get("icc_profile")

    if icc_profile:
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
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)