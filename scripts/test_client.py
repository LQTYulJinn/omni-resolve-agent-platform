"""
scripts/test_client.py
Interactive Client to test the live running FastAPI service.
"""
import requests
import json
import time

URL = "http://localhost:8000/api/v1/resolve"

def send_claim(ticket_id: str, text: str, amount: float, claim_type: str = "damaged_item"):
    payload = {
        "ticket_id": ticket_id,
        "user_id": "usr_test_client",
        "claim_type": claim_type,
        "customer_text": text,
        "transaction_amount": amount
    }
    t0 = time.perf_counter()
    resp = requests.post(URL, json=payload)
    elapsed = (time.perf_counter() - t0) * 1000
    
    print(f"\n--- Ticket: {ticket_id} (${amount:.2f}) ---")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Status Code        : {resp.status_code}")
        print(f"Server Process Time: {resp.headers.get('X-Process-Time-Ms')} ms")
        print(f"Round-trip Latency : {elapsed:.2f} ms")
        print(f"Action             : {data['action']}")
        print(f"Decision           : {data['decision']}")
        print(f"Confidence         : {data['confidence']:.4f}")
        if data["guardrail_reason"]:
            print(f"Guardrail Alert    : {data['guardrail_reason']}")
    else:
        print(f"Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    print("[CLIENT] Sending test claims to OmniResolve Service...")
    # Case 1: Normal claim ($45.00)
    send_claim("TCK-001", "The phone arrived with broken glass.", 45.00)
    # Case 2: High value claim ($450.00) -> Must trigger Guardrail
    send_claim("TCK-002", "Luxury watch package empty.", 450.00)
