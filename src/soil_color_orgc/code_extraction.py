import re
import os
import cv2

try:
    import pytesseract
except ImportError:
    pytesseract = None


CODE_PATTERN = re.compile(r"\b[A-Z]{4}-\d{4}\b")
FILENAME_CODE_PATTERN = re.compile(r"\b([A-Z]{4})(?:-\d{4})?\b", re.IGNORECASE)


def find_code_in_text(text: str):
    """
    Find full sample code like BMXU-2646 in arbitrary text.
    """
    if not text:
        return None

    text = text.upper()
    text = (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("_", "-")
        .replace(" ", "")
    )

    match = CODE_PATTERN.search(text)

    if match:
        return match.group(0)

    return None


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
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

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


def extract_sample_code_with_source(image_path: str):
    """
    Extract sample code and record how it was found.

    Priority:
        1. QR code in image
        2. OCR text in image
        3. Filename

    Returns:
        {
            "sample_code_full": "BMXU-2646" or None,
            "sample_code_base": "BMXU" or None,
            "sample_code_source": "qr" / "ocr" / "filename" / "none"
        }
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is not None:
        qr_code = extract_code_from_qr(image_bgr)

        if qr_code:
            return {
                "sample_code_full": qr_code,
                "sample_code_base": base_code_from_sample_code(qr_code),
                "sample_code_source": "qr",
            }

        ocr_code = extract_code_from_ocr(image_bgr)

        if ocr_code:
            return {
                "sample_code_full": ocr_code,
                "sample_code_base": base_code_from_sample_code(ocr_code),
                "sample_code_source": "ocr",
            }

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