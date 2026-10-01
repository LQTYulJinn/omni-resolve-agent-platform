"""
tests/test_data_pipeline.py
Production Quality Gate Tests for OmniResolve Data Pipeline.
Verifies:
1. Pydantic schema validation.
2. PII masking in customer text.
3. Corrupted image detection.
4. Zero Data Leakage in Group Splitting.
"""
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data.schema import ClaimTicketRaw, ClaimType
from src.data.validator import DataValidator
from src.data.processor import DatasetProcessor

@pytest.fixture
def sample_config():
    return {
        "raw_data": {
            "tickets_file": "data/raw/tickets.jsonl",
            "images_dir": "data/raw/images"
        },
        "processed_data": {
            "output_dir": "data/processed",
            "manifest_train": "data/processed/train_manifest.parquet",
            "manifest_val": "data/processed/val_manifest.parquet",
            "validation_report": "data/processed/validation_report.yaml"
        },
        "validation_rules": {
            "allowed_image_formats": [".jpg", ".jpeg", ".png", ".webp"],
            "min_image_resolution": [64, 64],
            "max_image_resolution": [4096, 4096],
            "min_text_length": 10,
            "max_text_length": 2000,
            "mask_pii": True
        },
        "split": {
            "val_size": 0.2,
            "group_column": "user_id",
            "random_seed": 42
        }
    }

def test_schema_rejection_on_invalid_type():
    with pytest.raises(ValidationError):
        ClaimTicketRaw(
            ticket_id="T-01",
            user_id="U-01",
            merchant_id="M-01",
            claim_type="completely_invalid_type",
            customer_text="Valid length complaint text here.",
            transaction_amount=50.0
        )

def test_pii_sanitization(sample_config):
    validator = DataValidator(sample_config)
    raw_text = "Call me at 0912345678 or email john.doe@example.com, card 1234-5678-9012-3456."
    ok, cleaned, reason = validator.sanitize_text(raw_text)
    assert ok is True
    assert "[REDACTED_PHONE]" in cleaned
    assert "[REDACTED_EMAIL]" in cleaned
    assert "[REDACTED_CC]" in cleaned
    assert "0912345678" not in cleaned
    assert "john.doe@example.com" not in cleaned
    assert "1234-5678-9012-3456" not in cleaned

def test_corrupt_image_detection(sample_config, tmp_path):
    validator = DataValidator(sample_config)
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"\x00\x01\x02") # Bogus bytes

    ok, res, reason = validator.validate_image(corrupt_file)
    assert ok is False
    assert "Corrupted image file" in reason

def test_pipeline_execution_and_zero_data_leakage(sample_config):
    import pandas as pd
    processor = DatasetProcessor(sample_config, BASE_DIR)
    report = processor.process()

    assert report["rejection_summary"]["total_records"] >= 30
    assert report["rejection_summary"]["valid_records"] == 30
    assert report["rejection_summary"]["corrupted_images"] >= 1
    assert report["rejection_summary"]["invalid_text"] >= 1
    assert report["rejection_summary"]["schema_errors"] >= 1

    # Check zero data leakage
    train_df = pd.read_parquet(BASE_DIR / sample_config["processed_data"]["manifest_train"])
    val_df = pd.read_parquet(BASE_DIR / sample_config["processed_data"]["manifest_val"])

    train_users = set(train_df["user_id"].unique())
    val_users = set(val_df["user_id"].unique())

    # THE GOLDEN TEST: Intersect of user sets must be empty
    overlap = train_users.intersection(val_users)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping users: {overlap}"
