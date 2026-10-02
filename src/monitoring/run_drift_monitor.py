"""
src/monitoring/run_drift_monitor.py
Simulates live production traffic batches (Normal vs Shifted/Black Friday),
runs statistical drift tests, and exports the Production Drift Report.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.monitoring.drift_detector import StatisticalDriftDetector

def run_monitor():
    processed_dir = BASE_DIR / "data" / "processed"
    train_manifest_path = processed_dir / "train_manifest.parquet"

    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Baseline train manifest not found at {train_manifest_path}.")

    print(f"[MONITOR] Loading reference baseline dataset from {train_manifest_path.name}...")
    baseline_df = pd.read_parquet(train_manifest_path)
    detector = StatisticalDriftDetector(baseline_df=baseline_df, alpha=0.05)

    # =========================================================================
    # SIMULATION 1: Normal In-Distribution Production Traffic (50 tickets)
    # =========================================================================
    np.random.seed(42)
    normal_amounts = np.random.uniform(25.0, 180.0, 50)
    normal_claims = np.random.choice(
        ["damaged_item", "wrong_item", "fake_item", "transaction_dispute", "delivery_delayed"],
        50
    ).tolist()
    normal_batch = pd.DataFrame({"transaction_amount": normal_amounts, "claim_type": normal_claims})

    normal_report = detector.evaluate_live_batch(normal_batch)

    # =========================================================================
    # SIMULATION 2: Drifting Production Traffic (Black Friday / Carrier Crisis)
    # =========================================================================
    # High-value surge ($400 - $950) + Carrier shipping delays surge (70%)
    drifted_amounts = np.random.uniform(400.0, 950.0, 50)
    drifted_claims = np.random.choice(
        ["delivery_delayed", "damaged_item"],
        50,
        p=[0.75, 0.25]
    ).tolist()
    drifted_batch = pd.DataFrame({"transaction_amount": drifted_amounts, "claim_type": drifted_claims})

    drifted_report = detector.evaluate_live_batch(drifted_batch)

    # =========================================================================
    # EXECUTIVE LOGGING
    # =========================================================================
    print("\n" + "="*65)
    print("           PRODUCTION DATA DRIFT MONITORING SUMMARY           ")
    print("="*65)
    print(f"BATCH 1 (Normal Operations):")
    print(f" - Status               : {normal_report['status']}")
    print(f" - Baseline Mean Amount : ${normal_report['checks']['transaction_amount_drift']['baseline_mean']:.2f}")
    print(f" - Batch 1 Mean Amount  : ${normal_report['checks']['transaction_amount_drift']['production_mean']:.2f}")
    print(f" - KS-Test p-value      : {normal_report['checks']['transaction_amount_drift']['p_value']:.4f}")
    print(f" - Recommended Action   : {normal_report['recommended_action']}")
    print("-"*65)
    print(f"BATCH 2 (Anomaly / Black Friday Shift):")
    print(f" - Status               : [!] {drifted_report['status']}")
    print(f" - Baseline Mean Amount : ${drifted_report['checks']['transaction_amount_drift']['baseline_mean']:.2f}")
    print(f" - Batch 2 Mean Amount  : ${drifted_report['checks']['transaction_amount_drift']['production_mean']:.2f}")
    print(f" - KS-Test p-value      : {drifted_report['checks']['transaction_amount_drift']['p_value']:.4e} (< 0.05)")
    print(f" - Category Drift       : {drifted_report['checks']['claim_type_drift']['alert']}")
    print(f" - Recommended Action   : [!] {drifted_report['recommended_action']}")
    print("="*65)

    final_report = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "baseline_summary": {
            "samples": len(baseline_df),
            "mean_amount": float(np.mean(baseline_df["transaction_amount"]))
        },
        "simulation_normal_batch": normal_report,
        "simulation_drifted_batch": drifted_report
    }

    out_path = processed_dir / "drift_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"[MONITOR] Drift report successfully exported to: {out_path}\n")
    return final_report

if __name__ == "__main__":
    run_monitor()
