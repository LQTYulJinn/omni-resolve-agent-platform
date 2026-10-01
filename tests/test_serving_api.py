"""
tests/test_serving_api.py
Unit tests verifying Production FastAPI Serving Microservice.
"""
import sys
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.serving.service import app

def test_api_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

def test_guardrail_high_value_transaction():
    with TestClient(app) as client:
        payload = {
            "ticket_id": "TEST-HIGH-VAL",
            "user_id": "usr_999",
            "claim_type": "damaged_item",
            "customer_text": "Broken display on high-end device.",
            "transaction_amount": 350.00 # Exceeds $200 threshold
        }
        resp = client.post("/api/v1/resolve", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # Verify Guardrail triggered
        assert data["action"] == "ESCALATE_TO_HUMAN"
        assert data["decision"] == "escalate_to_human"
        assert "High Transaction Value" in data["guardrail_reason"]
        assert "X-Process-Time-Ms" in resp.headers

def test_api_validation_error():
    with TestClient(app) as client:
        payload = {
            "ticket_id": "TEST-INVALID",
            "user_id": "usr_999",
            "claim_type": "invalid_type_here",
            "customer_text": "short", # Too short (< 5)
            "transaction_amount": -10.0 # Negative amount
        }
        resp = client.post("/api/v1/resolve", json=payload)
        assert resp.status_code == 422 # Pydantic Unprocessable Entity
