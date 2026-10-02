"""
tests/test_monitoring.py
Unit tests verifying Statistical Drift Detection logic (KS-test & Chi-Square).
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.monitoring.drift_detector import StatisticalDriftDetector

@pytest.fixture
def baseline_sample():
    np.random.seed(42)
    return pd.DataFrame({
        "transaction_amount": np.random.uniform(20.0, 100.0, 100),
        "claim_type": np.random.choice(
            ["damaged_item", "wrong_item", "fake_item", "transaction_dispute", "delivery_delayed"],
            100
        )
    })

def test_numerical_drift_detection_accuracy(baseline_sample):
    detector = StatisticalDriftDetector(baseline_df=baseline_sample, alpha=0.05)
    
    # 1. Test In-Distribution (Same range) -> Must NOT drift
    np.random.seed(123)
    same_dist = np.random.uniform(20.0, 100.0, 50)
    res_stable = detector.detect_numerical_drift(same_dist)
    assert res_stable["drift_detected"] is False
    assert res_stable["alert"] == "STABLE"

    # 2. Test Extreme Drift ($500 - $1000) -> Must trigger DRIFT
    drifted_dist = np.random.uniform(500.0, 1000.0, 50)
    res_drift = detector.detect_numerical_drift(drifted_dist)
    assert res_drift["drift_detected"] is True
    assert res_drift["alert"] == "CRITICAL_DRIFT"
    assert res_drift["p_value"] < 0.01

def test_categorical_drift_detection_accuracy(baseline_sample):
    detector = StatisticalDriftDetector(baseline_df=baseline_sample, alpha=0.05)
    
    # 1. 100% single category -> Extreme Category Drift
    skewed_claims = ["delivery_delayed"] * 50
    res_cat = detector.detect_categorical_drift(skewed_claims)
    assert res_cat["drift_detected"] is True
    assert res_cat["alert"] == "CATEGORY_DRIFT"
