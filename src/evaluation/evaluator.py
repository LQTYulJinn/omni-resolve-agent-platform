"""
src/evaluation/evaluator.py
Production Slice-Based & Cost-Sensitive Model Evaluator.
Performs:
1. Per-Class Precision, Recall, F1 breakdown.
2. Slice-based analysis:
   - Multimodal (With Image) vs Unimodal (Text-only fallback)
   - High-Value Transactions (> $100) vs Low-Value Transactions
   - Claim Type Breakdown
3. Policy Guardrail Threshold Analysis:
   - Automation Rate (% tickets resolved autonomously)
   - Human Escalation Rate (% tickets handed off to human ops)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

RESOLUTION_CLASSES = ["refund_approved", "refund_rejected", "escalate_to_human"]

class ProductionModelEvaluator:
    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        val_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Comprehensive production evaluation suite.
        probs: shape (N, 3) softmax probabilities
        """
        preds = np.argmax(probs, axis=1)
        max_confs = np.max(probs, axis=1)

        # 1. Global Metrics
        acc = float(accuracy_score(y_true, preds))
        p, r, f1, support = precision_recall_fscore_support(y_true, preds, labels=[0, 1, 2], zero_division=0)
        cm = confusion_matrix(y_true, preds, labels=[0, 1, 2]).tolist()

        per_class = {}
        for idx, cls_name in enumerate(RESOLUTION_CLASSES):
            per_class[cls_name] = {
                "precision": round(float(p[idx]), 4),
                "recall": round(float(r[idx]), 4),
                "f1_score": round(float(f1[idx]), 4),
                "support": int(support[idx])
            }

        # 2. Policy Guardrail & Human-in-the-loop Automation Rate
        # If confidence < threshold, policy escalates to human review
        final_decisions = []
        escalated_count = 0
        for pred, conf in zip(preds, max_confs):
            if conf < self.confidence_threshold:
                final_decisions.append(2) # escalate_to_human
                escalated_count += 1
            else:
                final_decisions.append(pred)

        automation_rate = float(1.0 - (escalated_count / len(y_true)))

        # 3. Slice-Based Analysis
        val_df = val_df.copy()
        val_df["pred"] = preds
        val_df["true"] = y_true
        val_df["is_correct"] = (val_df["pred"] == val_df["true"]).astype(int)

        # Slice A: Multimodal vs Text-Only
        has_img_mask = val_df["has_valid_image"] == True
        img_acc = float(val_df[has_img_mask]["is_correct"].mean()) if has_img_mask.sum() > 0 else 0.0
        no_img_acc = float(val_df[~has_img_mask]["is_correct"].mean()) if (~has_img_mask).sum() > 0 else 0.0

        # Slice B: High-Value (> $100) vs Low-Value
        high_val_mask = val_df["transaction_amount"] >= 100.0
        high_val_acc = float(val_df[high_val_mask]["is_correct"].mean()) if high_val_mask.sum() > 0 else 0.0
        low_val_acc = float(val_df[~high_val_mask]["is_correct"].mean()) if (~high_val_mask).sum() > 0 else 0.0

        # Slice C: Claim Type Accuracy
        claim_accs = {}
        for c_type, grp in val_df.groupby("claim_type"):
            claim_accs[str(c_type)] = {
                "count": len(grp),
                "accuracy": round(float(grp["is_correct"].mean()), 4)
            }

        report = {
            "summary": {
                "total_val_samples": len(y_true),
                "overall_accuracy": round(acc, 4),
                "automation_rate": round(automation_rate, 4),
                "escalation_to_human_rate": round(1.0 - automation_rate, 4),
                "confidence_threshold": self.confidence_threshold
            },
            "per_class_performance": per_class,
            "confusion_matrix": cm,
            "slice_based_evaluation": {
                "multimodal_with_image_accuracy": round(img_acc, 4),
                "unimodal_text_only_accuracy": round(no_img_acc, 4),
                "high_value_transaction_accuracy": round(high_val_acc, 4),
                "low_value_transaction_accuracy": round(low_val_acc, 4),
                "performance_by_claim_type": claim_accs
            }
        }
        return report
