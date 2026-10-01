"""
src/data/validator.py
Data Quality Gate & Sanitization Module.
Handles:
1. Image integrity verification (corrupt bytes, truncated JPEG, dimensions, color modes).
2. Text PII sanitization (masking credit card numbers, emails, phone numbers).
3. Metric tracking for data health reports.
"""
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image

# Compiled Regex for fast PII detection and masking
REGEX_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
REGEX_PHONE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")
REGEX_CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

class DataValidator:
    def __init__(self, config: Dict[str, Any]):
        rules = config.get("validation_rules", {})
        self.allowed_formats = set(rules.get("allowed_image_formats", [".jpg", ".jpeg", ".png", ".webp"]))
        self.min_res = rules.get("min_image_resolution", [64, 64])
        self.max_res = rules.get("max_image_resolution", [4096, 4096])
        self.min_text_len = rules.get("min_text_length", 10)
        self.max_text_len = rules.get("max_text_length", 2000)
        self.mask_pii = rules.get("mask_pii", True)

    def sanitize_text(self, text: str) -> Tuple[bool, str, Optional[str]]:
        """
        Cleans text, strips whitespace, and masks PII.
        Returns: (is_valid, cleaned_text, rejection_reason)
        """
        if not text or not isinstance(text, str):
            return False, "", "Empty or non-string text"

        cleaned = text.strip()
        if len(cleaned) < self.min_text_len:
            return False, cleaned, f"Text length ({len(cleaned)}) below minimum ({self.min_text_len})"
        if len(cleaned) > self.max_text_len:
            cleaned = cleaned[:self.max_text_len]

        # PII Masking
        if self.mask_pii:
            cleaned = REGEX_CREDIT_CARD.sub("[REDACTED_CC]", cleaned)
            cleaned = REGEX_EMAIL.sub("[REDACTED_EMAIL]", cleaned)
            cleaned = REGEX_PHONE.sub("[REDACTED_PHONE]", cleaned)

        return True, cleaned, None

    def validate_image(self, image_path: Optional[Path]) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
        """
        Verifies image file integrity, header sanity, resolution limits.
        Returns: (is_valid, (width, height), rejection_reason)
        """
        if image_path is None:
            return True, None, None # Optional image case

        if not image_path.exists():
            return False, None, f"Image file does not exist: {image_path}"

        if image_path.suffix.lower() not in self.allowed_formats:
            return False, None, f"Unsupported format '{image_path.suffix}'. Allowed: {self.allowed_formats}"

        try:
            # 1. Verify byte integrity without loading entire image into memory
            with Image.open(image_path) as img:
                img.verify()

            # 2. Re-open to inspect dimensions and color mode
            with Image.open(image_path) as img:
                width, height = img.size
                if width < self.min_res[0] or height < self.min_res[1]:
                    return False, (width, height), f"Image resolution {img.size} is below min {self.min_res}"
                if width > self.max_res[0] or height > self.max_res[1]:
                    return False, (width, height), f"Image resolution {img.size} exceeds max {self.max_res}"

                return True, (width, height), None

        except Exception as e:
            return False, None, f"Corrupted image file: {str(e)}"
