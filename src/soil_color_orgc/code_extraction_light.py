import os
import re


FILENAME_CODE_PATTERN = re.compile(
    r"^([A-Za-z]{4})(?:[-_].*)?$"
)


def extract_code_from_filename(image_path: str):
    """
    Extract four-letter sample code from filename.

    Examples:
        ABCD.jfif       -> ABCD
        ABCD.jpg        -> ABCD
        ABCD-1234.jfif  -> ABCD
        ABCD_test.jfif  -> ABCD

    Returns:
        ABCD or None
    """
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]

    match = FILENAME_CODE_PATTERN.match(stem)

    if match:
        return match.group(1).upper()

    return None


def extract_sample_code_from_filename_only(image_path: str):
    """
    Filename-only code extraction.

    Returns the same structure as the OCR-based extractor,
    so the rest of the pipeline does not need to change.
    """
    base_code = extract_code_from_filename(image_path)

    if base_code:
        return {
            "sample_code_full": None,
            "sample_code_base": base_code,
            "sample_code_source": "filename",
            "sample_code_match_score": None,
            "sample_code_matched_fragment": base_code,
            "sample_code_error": None,
            "filename_code_base": base_code,
            "code_conflict": False,
        }

    return {
        "sample_code_full": None,
        "sample_code_base": None,
        "sample_code_source": "filename_not_found",
        "sample_code_match_score": None,
        "sample_code_matched_fragment": None,
        "sample_code_error": "Could not extract four-letter code from filename",
        "filename_code_base": None,
        "code_conflict": False,
    }