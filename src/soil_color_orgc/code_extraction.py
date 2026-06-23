import re
import os
import cv2
from .image_io import read_image_bgr

try:
    import pytesseract
except ImportError:
    pytesseract = None


# Accepts:
# BMXU-2646
# BMXU 2646
# BMXU_2646
# BMXU2646
# BMXU–2646
CODE_PATTERN = re.compile(r"\b([A-Z]{4})\s*[-_–— ]?\s*(\d{4})\b", re.IGNORECASE)

# For filenames such as APKC.jfif or APKC-2646.jfif
FILENAME_CODE_PATTERN = re.compile(r"\b([A-Z]{4})(?:-\d{4})?\b", re.IGNORECASE)


def find_code_in_text(text: str):
    """
    Find sample code like BMXU-2646 in arbitrary OCR/QR text.
    Returns normalized code: BMXU-2646
    """
    if not text:
        return None

    text = text.upper()
    text = (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("_", "-")
    )

    match = CODE_PATTERN.search(text)

    if not match:
        return None

    letters = match.group(1).upper()
    digits = match.group(2)

    return f"{letters}-{digits}"


def base_code_from_sample_code(code: str):
    """
    BMXU-2646 -> BMXU
    """
    if not code:
        return None

    return code.split("-")[0].upper()


def extract_code_from_filename(image_path: str):
    """
    Extract code from image filename.

    Examples:
        APKC.jfif      -> APKC
        APKC-1234.jpg  -> APKC
    """
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]

    match = FILENAME_CODE_PATTERN.search(stem)

    if match:
        return match.group(1).upper()

    return None


def crop_code_region(image_bgr):
    """
    Crop likely code region.

    Assumption:
        code is in upper 30% of the image,
        between 20% and 80% of the width.
    """
    height, width = image_bgr.shape[:2]

    y1 = 0
    y2 = int(height * 0.30)

    x1 = int(width * 0.20)
    x2 = int(width * 0.80)

    crop = image_bgr[y1:y2, x1:x2]

    return crop


def extract_code_from_qr(image_bgr):
    """
    Try to decode QR code using OpenCV.
    Returns full code like BMXU-2646 or None.
    """
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image_bgr)

    if data:
        return find_code_in_text(data)

    return None


def preprocess_for_ocr(image_bgr):
    """
    Prepare image or crop for OCR.
    Since the code is expected to be horizontal, we use OCR modes
    suitable for a single line / sparse text.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale strongly because sample codes are often small.
    gray = cv2.resize(
        gray,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC,
    )

    # Mild denoising.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Increase contrast.
    gray = cv2.equalizeHist(gray)

    # Threshold.
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return thresh


def extract_code_from_ocr(image_bgr):
    """
    Use Tesseract OCR to detect printed text.
    Returns full code like BMXU-2646 or None.
    """
    if pytesseract is None:
        return None

    processed = preprocess_for_ocr(image_bgr)

    configs = [
        # Single horizontal text line.
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
        # Single uniform block.
        "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
        # Sparse text.
        "--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    ]

    for config in configs:
        text = pytesseract.image_to_string(processed, config=config)
        code = find_code_in_text(text)

        if code:
            return code

    return None


def extract_sample_code_with_source(image_path: str):
    """
    Extract sample code and record how it was found.

    Priority:
        1. QR code in likely upper-center region
        2. OCR in likely upper-center region
        3. QR code in full image
        4. OCR in full image
        5. Filename fallback

    Returns:
        {
            "sample_code_full": "BMXU-2646" or None,
            "sample_code_base": "BMXU" or None,
            "sample_code_source": "qr_roi" / "ocr_roi" / "qr_full" /
                                  "ocr_full" / "filename" / "none"
        }
    """

    image_bgr = read_image_bgr(image_path)

    if image_bgr is not None:
        roi = crop_code_region(image_bgr)

        # 1. Try QR in expected region.
        qr_roi_code = extract_code_from_qr(roi)

        if qr_roi_code:
            return {
                "sample_code_full": qr_roi_code,
                "sample_code_base": base_code_from_sample_code(qr_roi_code),
                "sample_code_source": "qr_roi",
            }

        # 2. Try OCR in expected region.
        ocr_roi_code = extract_code_from_ocr(roi)

        if ocr_roi_code:
            return {
                "sample_code_full": ocr_roi_code,
                "sample_code_base": base_code_from_sample_code(ocr_roi_code),
                "sample_code_source": "ocr_roi",
            }

        # 3. Fallback: QR in full image.
        qr_full_code = extract_code_from_qr(image_bgr)

        if qr_full_code:
            return {
                "sample_code_full": qr_full_code,
                "sample_code_base": base_code_from_sample_code(qr_full_code),
                "sample_code_source": "qr_full",
            }

        # 4. Fallback: OCR in full image.
        ocr_full_code = extract_code_from_ocr(image_bgr)

        if ocr_full_code:
            return {
                "sample_code_full": ocr_full_code,
                "sample_code_base": base_code_from_sample_code(ocr_full_code),
                "sample_code_source": "ocr_full",
            }

    # 5. Last fallback: filename.
    filename_code = extract_code_from_filename(image_path)

    if filename_code:
        return {
            "sample_code_full": None,
            "sample_code_base": filename_code,
            "sample_code_source": "filename",
        }

    return {
        "sample_code_full": None,
        "sample_code_base": None,
        "sample_code_source": "none",
    }