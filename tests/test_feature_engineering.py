"""
tests/test_feature_engineering.py
Unit tests verifying Multimodal Feature Processing logic.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data.feature_extractor import MultimodalFeatureExtractor

def test_unfitted_extractor_raises_error():
    extractor = MultimodalFeatureExtractor()
    with pytest.raises(RuntimeError):
        extractor.transform_tabular([100.0], ["damaged_item"])

def test_multimodal_feature_shapes_and_values():
    extractor = MultimodalFeatureExtractor(text_dim=64, image_size=(224, 224))
    
    train_texts = ["The package arrived broken and damaged.", "Wrong size shoes delivered."]
    train_amounts = [50.0, 150.0]
    train_claims = ["damaged_item", "wrong_item"]

    extractor.fit(train_texts, train_amounts)
    assert extractor.fitted is True
    assert len(extractor.vocab) > 0

    # 1. Text Embeddings
    text_embs = extractor.transform_text_embeddings(train_texts)
    assert text_embs.shape == (2, 64)
    assert not np.isnan(text_embs).any()
    assert not np.isinf(text_embs).any()

    # 2. Tabular Features
    tab_feats = extractor.transform_tabular(train_amounts, train_claims)
    assert tab_feats.shape == (2, 6) # 1 scaled amount + 5 one-hot classes
    assert (tab_feats >= 0.0).all() and (tab_feats <= 1.0).all()
    assert not np.isnan(tab_feats).any()

    # 3. Vision Tensor (Synthetic / Missing fallback)
    img_tensor = extractor.transform_image(None)
    assert img_tensor.shape == (3, 224, 224)
