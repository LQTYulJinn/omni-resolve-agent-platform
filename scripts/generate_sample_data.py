"""
scripts/generate_sample_data.py
Generates a realistic multimodal customer claim dataset with both valid records
and deliberate "dirty/corrupted" anomalies to rigorously test the Data Quality Gate.
"""
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"
IMAGES_DIR = DATA_RAW / "images"

def create_sample_image(path: Path, text: str, color=(200, 220, 240)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (256, 256), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 236, 236], outline=(100, 100, 100), width=3)
    draw.text((40, 120), text, fill=(20, 20, 20))
    img.save(path, format="JPEG")

def generate():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    tickets_file = DATA_RAW / "tickets.jsonl"

    users = [f"usr_{100 + i}" for i in range(15)]
    merchants = ["shop_apple_store", "shop_nike_flagship", "shop_samsung_official"]

    claim_templates = [
        ("damaged_item", "The package arrived yesterday and the phone screen is completely cracked. Please refund me. My phone number is 0912345678 and email is customer@gmail.com."),
        ("wrong_item", "I ordered size 42 shoes but received size 38 instead. Unopened box. Card charged 4111-2222-3333-4444."),
        ("fake_item", "The watch feels very light and the serial number does not exist on manufacturer site. Counterfeit."),
        ("transaction_dispute", "I was charged twice on my invoice for transaction #9981. Please check transaction logs."),
        ("delivery_delayed", "Tracking says delivered 3 days ago but front porch is empty. Carrier lost it.")
    ]

    records = []

    # 1. Generate 30 Valid Tickets across users
    for i in range(30):
        user = random.choice(users)
        claim_type, text_tmpl = random.choice(claim_templates)
        img_name = f"claim_{i:03d}.jpg"
        create_sample_image(IMAGES_DIR / img_name, f"Ticket #{i}\n{claim_type}", color=(random.randint(180, 240), random.randint(180, 240), random.randint(180, 240)))

        record = {
            "ticket_id": f"TCK-{1000 + i}",
            "user_id": user,
            "merchant_id": random.choice(merchants),
            "claim_type": claim_type,
            "customer_text": f"Ticket #{i}: {text_tmpl}",
            "image_rel_path": img_name,
            "transaction_amount": round(random.uniform(25.0, 450.0), 2),
            "ground_truth_resolution": random.choice(["refund_approved", "refund_rejected", "escalate_to_human"])
        }
        records.append(record)

    # 2. Insert ANOMALY 1: Corrupted Image file (0 bytes)
    corrupted_img = IMAGES_DIR / "claim_corrupt.jpg"
    corrupted_img.write_bytes(b"") # 0 bytes
    records.append({
        "ticket_id": "TCK-CORRUPT-IMG",
        "user_id": "usr_999",
        "merchant_id": "shop_nike_flagship",
        "claim_type": "damaged_item",
        "customer_text": "Item is broken, please check the attached corrupt photo.",
        "image_rel_path": "claim_corrupt.jpg",
        "transaction_amount": 120.0
    })

    # 3. Insert ANOMALY 2: Text too short (< 5 chars)
    records.append({
        "ticket_id": "TCK-SHORT-TXT",
        "user_id": "usr_998",
        "merchant_id": "shop_apple_store",
        "claim_type": "wrong_item",
        "customer_text": "bad", # Rejection: text too short
        "image_rel_path": None,
        "transaction_amount": 50.0
    })

    # 4. Insert ANOMALY 3: Invalid Claim Type (Schema rejection)
    records.append({
        "ticket_id": "TCK-INVALID-SCHEMA",
        "user_id": "usr_997",
        "merchant_id": "shop_apple_store",
        "claim_type": "some_random_unsupported_claim_type",
        "customer_text": "This ticket has an unsupported claim type in schema.",
        "image_rel_path": None,
        "transaction_amount": 80.0
    })

    with open(tickets_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[DATA GENERATOR] Created {len(records)} sample tickets (including 3 anomalies) in {tickets_file}")

if __name__ == "__main__":
    generate()
