"""
src/evaluation/run_evaluation.py
Executes Production Evaluation against the Optimized ONNX Inference Engine.
Generates Slice-Based & Human-In-The-Loop Report.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.evaluation.evaluator import ProductionModelEvaluator

RESOLUTION_MAP = {
    "refund_approved": 0,
    "refund_rejected": 1,
    "escalate_to_human": 2
}

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def run():
    models_dir = BASE_DIR / "models"
    onnx_path = models_dir / "optimized" / "multimodal_fusion.onnx"
    processed_dir = BASE_DIR / "data" / "processed"
    features_dir = processed_dir / "features"

    val_manifest_path = processed_dir / "val_manifest.parquet"
    val_npz_path = features_dir / "val_features.npz"

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX Model not found at {onnx_path}. Run export_optimized.py first.")

    print(f"[EVAL] Loading Validation set and ONNX Engine: {onnx_path.name}...")
    val_df = pd.read_parquet(val_manifest_path)
    val_npz = np.load(val_npz_path)

    y_true = np.array([RESOLUTION_MAP.get(r, 0) for r in val_df["ground_truth_resolution"]], dtype=np.int64)

    # 1. Run Inference using ONNX Runtime
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    inputs = {
        "text_emb": val_npz["text_embeddings"],
        "tab_feats": val_npz["tabular_features"],
        "img_tensor": val_npz["image_tensors"]
    }
    logits = session.run(None, inputs)[0]
    probs = softmax(logits)

    # 2. Comprehensive Evaluation
    evaluator = ProductionModelEvaluator(confidence_threshold=0.60)
    report = evaluator.evaluate(y_true, probs, val_df)

    # 3. Print Executive Summary
    print("\n" + "="*60)
    print("         PRODUCTION MODEL EVALUATION EXECUTIVE REPORT         ")
    print("="*60)
    print(f"Total Validation Samples      : {report['summary']['total_val_samples']}")
    print(f"Overall Accuracy              : {report['summary']['overall_accuracy'] * 100:.1f}%")
    print(f"Autonomous Automation Rate    : {report['summary']['automation_rate'] * 100:.1f}%")
    print(f"Escalation to Human Ops Rate  : {report['summary']['escalation_to_human_rate'] * 100:.1f}%")
    print("-"*60)
    print("Slice Analysis:")
    slice_data = report["slice_based_evaluation"]
    print(f" - With Image Accuracy        : {slice_data['multimodal_with_image_accuracy'] * 100:.1f}%")
    print(f" - High Value (> $100) Acc    : {slice_data['high_value_transaction_accuracy'] * 100:.1f}%")
    print(f" - Low Value (< $100) Acc     : {slice_data['low_value_transaction_accuracy'] * 100:.1f}%")
    print("="*60)

    # 4. Save Artifact
    out_path = models_dir / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[EVAL] Detailed report successfully saved to: {out_path}\n")
    return report

if __name__ == "__main__":
    run()
