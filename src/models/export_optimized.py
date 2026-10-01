"""
src/models/export_optimized.py
Model Optimization & Graph Fusion Stage.
Exports PyTorch Checkpoint -> ONNX Runtime Engine with Dynamic Batching.
Verifies Numerical Equivalence & Benchmarks Speedup.
"""
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.models.network import MultimodalFusionClassifier
import onnxruntime as ort

def export_and_optimize():
    models_dir = BASE_DIR / "models"
    checkpoint_path = models_dir / "checkpoints" / "best_fusion_model.pt"
    optimized_dir = models_dir / "optimized"
    optimized_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = optimized_dir / "multimodal_fusion.onnx"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train_and_benchmark.py first.")

    print(f"[OPTIMIZE] Loading PyTorch model from {checkpoint_path}...")
    model = MultimodalFusionClassifier(text_dim=64, tab_dim=6, vision_dim=16, num_classes=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Create dummy inputs with batch_size=1
    dummy_text = torch.randn(1, 64, dtype=torch.float32)
    dummy_tab = torch.randn(1, 6, dtype=torch.float32)
    dummy_img = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"[OPTIMIZE] Exporting to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_text, dummy_tab, dummy_img),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["text_emb", "tab_feats", "img_tensor"],
        output_names=["logits"],
        dynamic_axes={
            "text_emb": {0: "batch_size"},
            "tab_feats": {0: "batch_size"},
            "img_tensor": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        dynamo=False
    )
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[OPTIMIZE] ONNX model successfully saved! Size: {onnx_size_mb:.2f} MB")

    # 2. Verification with ONNX Runtime
    print("[OPTIMIZE] Initializing ONNX Runtime Session...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    # Parity check
    with torch.no_grad():
        pt_out = model(dummy_text, dummy_tab, dummy_img).numpy()

    ort_inputs = {
        "text_emb": dummy_text.numpy(),
        "tab_feats": dummy_tab.numpy(),
        "img_tensor": dummy_img.numpy()
    }
    ort_out = session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"[VERIFY] Parity check Max Absolute Error: {max_diff:.8f}")
    assert max_diff < 1e-4, f"Parity check failed! Diff: {max_diff}"
    print("[VERIFY] Numerical parity check PASSED!")

    # 3. Benchmark PyTorch vs ONNX Runtime across batch sizes
    batch_sizes = [1, 4, 8]
    runs = 50
    warmup = 10
    benchmark_report = {
        "onnx_size_mb": round(onnx_size_mb, 3),
        "parity_max_diff": max_diff,
        "results": {}
    }

    print("\n" + "="*65)
    print(f"{'Batch':<8}{'Engine':<12}{'Latency (p50)':<16}{'Latency (p95)':<16}{'Throughput (RPS)':<16}")
    print("="*65)

    for bs in batch_sizes:
        b_text = torch.randn(bs, 64)
        b_tab = torch.randn(bs, 6)
        b_img = torch.randn(bs, 3, 224, 224)

        # Benchmark PyTorch
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(b_text, b_tab, b_img)
            pt_times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = model(b_text, b_tab, b_img)
                pt_times.append((time.perf_counter() - t0) * 1000)

        # Benchmark ONNX Runtime
        ort_in = {
            "text_emb": b_text.numpy(),
            "tab_feats": b_tab.numpy(),
            "img_tensor": b_img.numpy()
        }
        for _ in range(warmup):
            _ = session.run(None, ort_in)
        ort_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = session.run(None, ort_in)
            ort_times.append((time.perf_counter() - t0) * 1000)

        pt_p50 = float(np.percentile(pt_times, 50))
        pt_p95 = float(np.percentile(pt_times, 95))
        pt_rps = (bs * 1000.0) / pt_p50

        ort_p50 = float(np.percentile(ort_times, 50))
        ort_p95 = float(np.percentile(ort_times, 95))
        ort_rps = (bs * 1000.0) / ort_p50

        speedup = pt_p50 / ort_p50

        print(f"{bs:<8}{'PyTorch':<12}{pt_p50:>8.2f} ms     {pt_p95:>8.2f} ms     {pt_rps:>10.1f}")
        print(f"{bs:<8}{'ONNX-ORT':<12}{ort_p50:>8.2f} ms     {ort_p95:>8.2f} ms     {ort_rps:>10.1f}  (Speedup: {speedup:.2f}x)")
        print("-" * 65)

        benchmark_report["results"][f"batch_{bs}"] = {
            "pytorch_latency_p50_ms": round(pt_p50, 2),
            "pytorch_rps": round(pt_rps, 1),
            "onnx_latency_p50_ms": round(ort_p50, 2),
            "onnx_rps": round(ort_rps, 1),
            "speedup": round(speedup, 2)
        }

    report_path = optimized_dir / "optimization_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print(f"\n[OPTIMIZE] Optimization Report saved to: {report_path}")
    return benchmark_report

if __name__ == "__main__":
    export_and_optimize()
