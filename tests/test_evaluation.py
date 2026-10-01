"""
tests/test_evaluation.py
Unit tests verifying Production Evaluation logic and slice metrics.
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.evaluation.evaluator import ProductionModelEvaluator

def test_evaluator_metrics_and_slices():
    evaluator = ProductionModelEvaluator(confidence_threshold=0.60)
    
    y_true = np.array([0, 1, 2, 0])
    probs = np.array([
        [0.8, 0.1, 0.1],  # confident -> class 0
        [0.1, 0.7, 0.2],  # confident -> class 1
        [0.4, 0.3, 0.3],  # low confidence (< 0.6) -> will escalate
        [0.9, 0.05, 0.05] # confident -> class 0
    ])

    df = pd.DataFrame({
        "has_valid_image": [True, True, False, True],
        "transaction_amount": [150.0, 50.0, 20.0, 300.0],
        "claim_type": ["damaged_item", "wrong_item", "fake_item", "damaged_item"]
    })

    report = evaluator.evaluate(y_true, probs, df)

    # 1. Check summary bounds
    summary = report["summary"]
    assert 0.0 <= summary["overall_accuracy"] <= 1.0
    assert abs(summary["automation_rate"] + summary["escalation_to_human_rate"] - 1.0) < 1e-5
    assert summary["escalation_to_human_rate"] == 0.25 # exactly 1 out of 4 escalated

    # 2. Check slices exist
    slices = report["slice_based_evaluation"]
    assert "multimodal_with_image_accuracy" in slices
    assert "high_value_transaction_accuracy" in slices
    assert "performance_by_claim_type" in slices
