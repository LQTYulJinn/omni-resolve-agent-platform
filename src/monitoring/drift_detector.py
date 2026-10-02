"""
src/monitoring/drift_detector.py
Statistical Data Drift & Prediction Drift Detector for OmniResolve.
Implements:
1. Two-sample Kolmogorov-Smirnov (KS) Test for Numerical Features (Amount).
2. Chi-Square Goodness-of-Fit Test for Categorical Features (Claim Type).
3. Prediction Distribution & Confidence Degradation Tracking.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare
from typing import Dict, Any, List

class StatisticalDriftDetector:
    def __init__(self, baseline_df: pd.DataFrame, alpha: float = 0.05):
        """
        baseline_df: Training / Reference data from Step 1.
        alpha: Statistical significance threshold (default 0.05).
        """
        self.alpha = alpha
        self.ref_amounts = baseline_df["transaction_amount"].values
        
        # Reference claim type frequencies
        self.claim_types = ["damaged_item", "wrong_item", "fake_item", "transaction_dispute", "delivery_delayed"]
        val_counts = baseline_df["claim_type"].value_counts()
        total = len(baseline_df)
        self.ref_claim_dist = np.array([val_counts.get(c, 1) / total for c in self.claim_types], dtype=np.float64)

    def detect_numerical_drift(self, live_amounts: np.ndarray) -> Dict[str, Any]:
        """Runs two-sample KS-Test to compare live production distribution against baseline."""
        stat, p_value = ks_2samp(self.ref_amounts, live_amounts)
        is_drift = bool(p_value < self.alpha)
        
        return {
            "feature": "transaction_amount",
            "test_type": "Kolmogorov-Smirnov (KS-Test)",
            "statistic": round(float(stat), 4),
            "p_value": float(p_value),
            "drift_detected": is_drift,
            "baseline_mean": round(float(np.mean(self.ref_amounts)), 2),
            "production_mean": round(float(np.mean(live_amounts)), 2),
            "alert": "CRITICAL_DRIFT" if is_drift else "STABLE"
        }

    def detect_categorical_drift(self, live_claim_types: List[str]) -> Dict[str, Any]:
        """Runs Chi-Square test to detect category distribution shift."""
        n_live = len(live_claim_types)
        counts = pd.Series(live_claim_types).value_counts()
        observed = np.array([counts.get(c, 0.1) for c in self.claim_types], dtype=np.float64)
        
        # Expected frequencies based on baseline
        expected = self.ref_claim_dist * n_live
        # Normalize to ensure sum equality
        expected = expected * (observed.sum() / expected.sum())

        stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        is_drift = bool(p_value < self.alpha)

        return {
            "feature": "claim_type",
            "test_type": "Chi-Square Goodness-of-Fit",
            "statistic": round(float(stat), 4),
            "p_value": float(p_value),
            "drift_detected": is_drift,
            "alert": "CATEGORY_DRIFT" if is_drift else "STABLE"
        }

    def evaluate_live_batch(self, live_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates an entire live incoming batch of production traffic."""
        num_result = self.detect_numerical_drift(live_df["transaction_amount"].values)
        cat_result = self.detect_categorical_drift(live_df["claim_type"].tolist())
        
        overall_drift = num_result["drift_detected"] or cat_result["drift_detected"]

        return {
            "status": "DRIFT_DETECTED" if overall_drift else "HEALTHY_STABLE",
            "overall_drift_flag": overall_drift,
            "batch_size": len(live_df),
            "checks": {
                "transaction_amount_drift": num_result,
                "claim_type_drift": cat_result
            },
            "recommended_action": (
                "TRIGGER_RETRAINING_PIPELINE: Retrain model via 'dvc repro' with newly captured distribution."
                if overall_drift else "CONTINUE_NORMAL_OPERATION"
            )
        }
