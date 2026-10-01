"""
src/data/processor.py
Dataset Preprocessor, Smart Splitting, and Parquet Manifest Generator.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.schema import ClaimTicketRaw, ValidatedTicket
from src.data.validator import DataValidator

class DatasetProcessor:
    def __init__(self, config: Dict[str, Any], base_dir: Path):
        self.config = config
        self.base_dir = base_dir
        self.validator = DataValidator(config)

    def process(self):
        raw_cfg = self.config["raw_data"]
        processed_cfg = self.config["processed_data"]
        split_cfg = self.config["split"]

        tickets_path = self.base_dir / raw_cfg["tickets_file"]
        images_dir = self.base_dir / raw_cfg["images_dir"]
        output_dir = self.base_dir / processed_cfg["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        if not tickets_path.exists():
            raise FileNotFoundError(f"Tickets file not found: {tickets_path}")

        print(f"[PROCESSOR] Reading raw tickets from: {tickets_path}")
        validated_tickets: List[ValidatedTicket] = []
        rejection_stats = {
            "total_records": 0,
            "valid_records": 0,
            "corrupted_images": 0,
            "invalid_text": 0,
            "schema_errors": 0,
            "reasons": []
        }

        # 1. Ingestion & Validation
        with open(tickets_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rejection_stats["total_records"] += 1
                try:
                    raw_dict = json.loads(line)
                    raw_ticket = ClaimTicketRaw(**raw_dict)
                except Exception as e:
                    rejection_stats["schema_errors"] += 1
                    rejection_stats["reasons"].append(f"Line {line_no} schema error: {str(e)}")
                    continue

                # Validate Text & Mask PII
                text_ok, cleaned_text, text_reason = self.validator.sanitize_text(raw_ticket.customer_text)
                if not text_ok:
                    rejection_stats["invalid_text"] += 1
                    rejection_stats["reasons"].append(f"Ticket {raw_ticket.ticket_id}: {text_reason}")
                    continue

                # Validate Image
                has_image = False
                img_abs = None
                img_res = None
                if raw_ticket.image_rel_path:
                    img_abs_candidate = images_dir / raw_ticket.image_rel_path
                    img_ok, res, img_reason = self.validator.validate_image(img_abs_candidate)
                    if not img_ok:
                        rejection_stats["corrupted_images"] += 1
                        rejection_stats["reasons"].append(f"Ticket {raw_ticket.ticket_id}: {img_reason}")
                        continue
                    has_image = True
                    img_abs = str(img_abs_candidate.resolve())
                    img_res = list(res) if res else None

                validated_obj = ValidatedTicket(
                    ticket_id=raw_ticket.ticket_id,
                    user_id=raw_ticket.user_id,
                    merchant_id=raw_ticket.merchant_id,
                    claim_type=raw_ticket.claim_type,
                    cleaned_text=cleaned_text,
                    has_valid_image=has_image,
                    image_abs_path=img_abs,
                    image_resolution=img_res,
                    transaction_amount=raw_ticket.transaction_amount,
                    ground_truth_resolution=raw_ticket.ground_truth_resolution
                )
                validated_tickets.append(validated_obj)
                rejection_stats["valid_records"] += 1

        print(f"[PROCESSOR] Scanned {rejection_stats['total_records']} tickets. "
              f"Valid: {rejection_stats['valid_records']}, "
              f"Rejected: {rejection_stats['total_records'] - rejection_stats['valid_records']}")

        if not validated_tickets:
            raise ValueError("No valid records found after data cleansing!")

        # 2. Smart Group Splitting (Prevent Data Leakage by user_id)
        df = pd.DataFrame([t.model_dump() for t in validated_tickets])
        group_col = split_cfg.get("group_column", "user_id")
        val_size = split_cfg.get("val_size", 0.2)
        seed = split_cfg.get("random_seed", 42)

        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"

        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()

        # 3. Export Manifests as Parquet (High performance columnar format)
        train_manifest_path = self.base_dir / processed_cfg["manifest_train"]
        val_manifest_path = self.base_dir / processed_cfg["manifest_val"]

        train_df.to_parquet(train_manifest_path, index=False)
        val_df.to_parquet(val_manifest_path, index=False)
        print(f"[PROCESSOR] Saved Train Manifest ({len(train_df)} rows) to: {train_manifest_path}")
        print(f"[PROCESSOR] Saved Val Manifest ({len(val_df)} rows) to: {val_manifest_path}")

        # 4. Save Health & Validation Report
        report_path = self.base_dir / processed_cfg["validation_report"]
        report_data = {
            "rejection_summary": rejection_stats,
            "splits": {
                "train_count": len(train_df),
                "val_count": len(val_df),
                "unique_users_train": int(train_df["user_id"].nunique()),
                "unique_users_val": int(val_df["user_id"].nunique()),
                "multimodal_ratio_train": float((train_df["has_valid_image"]).mean()),
                "multimodal_ratio_val": float((val_df["has_valid_image"]).mean()),
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            yaml.dump(report_data, f, default_flow_style=False)
        print(f"[PROCESSOR] Validation Report saved to: {report_path}")

        return report_data
