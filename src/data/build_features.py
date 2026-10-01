"""
src/data/build_features.py
Feature Engineering & Transformation Pipeline Stage.
Loads Parquet manifests -> Extracts Multimodal Tensors -> Saves Feature Artifacts.
"""
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.data.feature_extractor import MultimodalFeatureExtractor

def process_features():
    processed_dir = BASE_DIR / "data" / "processed"
    features_dir = processed_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train_manifest.parquet"
    val_path = processed_dir / "val_manifest.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Manifest files not found! Ensure Step 1 (pipeline.py) has run.")

    print(f"[FEATURES] Loading manifests from {processed_dir}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    extractor = MultimodalFeatureExtractor(text_dim=64, image_size=(224, 224))

    # 1. Fit STRICTLY on Train
    print("[FEATURES] Fitting vocabulary and numerical scalers on Train split...")
    extractor.fit(
        train_texts=train_df["cleaned_text"].tolist(),
        train_amounts=train_df["transaction_amount"].tolist()
    )

    # 2. Transform Train Features
    print(f"[FEATURES] Transforming Train ({len(train_df)} samples)...")
    train_text_emb = extractor.transform_text_embeddings(train_df["cleaned_text"].tolist())
    train_tab = extractor.transform_tabular(
        amounts=train_df["transaction_amount"].tolist(),
        claim_types=train_df["claim_type"].tolist()
    )
    # Vision tensors
    train_img_tensors = torch.stack([
        extractor.transform_image(p) for p in train_df["image_abs_path"]
    ]).numpy()

    # 3. Transform Val Features
    print(f"[FEATURES] Transforming Val ({len(val_df)} samples)...")
    val_text_emb = extractor.transform_text_embeddings(val_df["cleaned_text"].tolist())
    val_tab = extractor.transform_tabular(
        amounts=val_df["transaction_amount"].tolist(),
        claim_types=val_df["claim_type"].tolist()
    )
    val_img_tensors = torch.stack([
        extractor.transform_image(p) for p in val_df["image_abs_path"]
    ]).numpy()

    # 4. Save to Compressed NPZ
    train_npz = features_dir / "train_features.npz"
    val_npz = features_dir / "val_features.npz"

    np.savez_compressed(
        train_npz,
        ticket_ids=train_df["ticket_id"].values,
        text_embeddings=train_text_emb,
        tabular_features=train_tab,
        image_tensors=train_img_tensors
    )
    np.savez_compressed(
        val_npz,
        ticket_ids=val_df["ticket_id"].values,
        text_embeddings=val_text_emb,
        tabular_features=val_tab,
        image_tensors=val_img_tensors
    )

    metadata = {
        "num_train": len(train_df),
        "num_val": len(val_df),
        "text_embedding_dim": int(train_text_emb.shape[1]),
        "tabular_feature_dim": int(train_tab.shape[1]),
        "image_tensor_shape": list(train_img_tensors.shape[1:]),
        "vocab_size": len(extractor.vocab),
        "amount_scale_range": [extractor.amount_min, extractor.amount_max]
    }
    with open(features_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[FEATURES] Success! Saved feature artifacts to {features_dir}")
    print(f"           - Train NPZ: {train_npz.stat().st_size / (1024*1024):.2f} MB")
    print(f"           - Val NPZ  : {val_npz.stat().st_size / (1024*1024):.2f} MB")
    return metadata

if __name__ == "__main__":
    process_features()
