"""
tests/test_model_pipeline.py
Unit tests verifying Multimodal Fusion Network and Model Selection logic.
"""
import sys
import torch
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.models.network import MultimodalFusionClassifier

def test_multimodal_network_forward_shape():
    model = MultimodalFusionClassifier(text_dim=64, tab_dim=6, vision_dim=16, num_classes=3)
    batch_size = 4
    
    text_emb = torch.randn(batch_size, 64)
    tab_feats = torch.randn(batch_size, 6)
    img_tensor = torch.randn(batch_size, 3, 224, 224)

    logits = model(text_emb, tab_feats, img_tensor)
    assert logits.shape == (batch_size, 3)
    assert not torch.isnan(logits).any()

def test_dynamic_batching_capability():
    model = MultimodalFusionClassifier()
    model.eval()
    # Test batch size 1 and batch size 7
    for bs in [1, 7]:
        t = torch.randn(bs, 64)
        b = torch.randn(bs, 6)
        i = torch.randn(bs, 3, 224, 224)
        out = model(t, b, i)
        assert out.shape == (bs, 3)
