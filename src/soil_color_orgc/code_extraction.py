import re
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None


CODE_PATTERN = re.compile(r"\b[A-Z]{4}-\d{4}\b")


def find_code_in_text(text: str):
    """
    Find sample code like BMXU-2646 in arbitrary text.
    """
    if not text:
        return None

    text = text.upper()
    text = text.replace("—", "-").replace("–", "-").replace("_", "-")

    match = CODE_PATTERN.search(text)

    if match:
        return match.group(0)

    return None


def extract_code_from_qr(image_bgr):
    """
    Try to decode QR code using OpenCV.
    Returns code like BMXU-2646 or None.
    """
    detector = cv2.QRCodeDetector()

    data, points, _ = detector.detectAndDecode(image_bgr)

    if data:
        return find_code_in_text(data)

    return None


def preprocess_for_ocr(image_bgr):
    """
    Prepare image for OCR.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale: OCR works better on larger text
    gray = cv2.resize(
        gray,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC,
    )

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return thresh


def extract_code_from_ocr(image_bgr):
    """
    Use Tesseract OCR to detect text and extract ABCD-1234-style code.
    """
    if pytesseract is None:
        return None

    processed = preprocess_for_ocr(image_bgr)

    configs = [
        "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        "--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        "--psm 12 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
    ]

    for config in configs:
        text = pytesseract.image_to_string(processed, config=config)
        code = find_code_in_text(text)

        if code:
            return code

    return None


def extract_sample_code(image_path: str):
    """
    Extract sample code from an image.

    Strategy:
    1. Try QR code.
    2. Try OCR text recognition.

    Returns:
        "BMXU-2646" or None
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    code = extract_code_from_qr(image_bgr)

    if code:
        return code

    code = extract_code_from_ocr(image_bgr)

    if code:
        return code

    return None


def base_code_from_sample_code(code: str):
    """
    BMXU-2646 -> BMXU
    """
    if not code:
        return None

    return code.split("-")[0].upper()