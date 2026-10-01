"""
src/data/schema.py
Data Contracts & Pydantic Schemas for OmniResolve Multimodal Data.
Guarantees schema enforcement, type validation, and data sanity across all pipeline stages.
"""
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class ClaimType(str, Enum):
    DAMAGED_ITEM = "damaged_item"
    WRONG_ITEM = "wrong_item"
    FAKE_ITEM = "fake_item"
    TRANSACTION_DISPUTE = "transaction_dispute"
    DELIVERY_DELAYED = "delivery_delayed"

class ResolutionStatus(str, Enum):
    REFUND_APPROVED = "refund_approved"
    REFUND_REJECTED = "refund_rejected"
    ESCALATE_TO_HUMAN = "escalate_to_human"

class ClaimTicketRaw(BaseModel):
    """Raw ticket received from customer support ingestion stream."""
    ticket_id: str = Field(..., description="Unique ticket identifier")
    user_id: str = Field(..., description="Customer ID")
    merchant_id: str = Field(..., description="Merchant/Store ID")
    claim_type: ClaimType
    customer_text: str = Field(..., description="Customer complaint description")
    image_rel_path: Optional[str] = Field(None, description="Relative path to uploaded evidence image")
    transaction_amount: float = Field(..., ge=0.0, description="Value of transaction in USD")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    ground_truth_resolution: Optional[ResolutionStatus] = None

class ValidatedTicket(BaseModel):
    """Cleaned, validated, and normalized ticket ready for embedding, retrieval, and modeling."""
    ticket_id: str
    user_id: str
    merchant_id: str
    claim_type: ClaimType
    cleaned_text: str
    has_valid_image: bool
    image_abs_path: Optional[str] = None
    image_resolution: Optional[List[int]] = None # [width, height]
    transaction_amount: float
    split: Optional[str] = None # 'train' or 'val'
    ground_truth_resolution: Optional[ResolutionStatus] = None
