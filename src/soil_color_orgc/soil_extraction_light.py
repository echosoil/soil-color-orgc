import re
import os


FILENAME_CODE_PATTERN = re.compile(
    r"\b([A-Z]{4})(?:-\d{4})?\b",
    re.IGNORECASE,
)


def extract_code_from_filename(image_path: str):
    """
    Extract four-letter code from filename.

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