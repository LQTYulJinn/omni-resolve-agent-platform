"""
src/models/network.py
Production-ready Multimodal Late-Fusion Decision Network.
Fuses:
- Text Semantic Embedding (64-dim)
- Tabular / Numerical Features (6-dim: amount + one-hot claim types)
- Vision Projection Vector (16-dim pooled from image tensor)
Total Fusion Dimension = 64 + 6 + 16 = 86-dim.
Predicts Resolution Status (3 classes: refund_approved, refund_rejected, escalate_to_human).
"""
import torch
import torch.nn as nn

class LightweightVisionProjector(nn.Module):
    """Extremely fast Conv projection to extract compact vision features without heavy ViT overhead."""
    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3), # 224x224 -> 56x56
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling -> (16, 1, 1)
            nn.Flatten(),
            nn.Linear(16, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class MultimodalFusionClassifier(nn.Module):
    def __init__(self, text_dim: int = 64, tab_dim: int = 6, vision_dim: int = 16, num_classes: int = 3):
        super().__init__()
        self.vision_encoder = LightweightVisionProjector(out_dim=vision_dim)
        
        fusion_dim = text_dim + tab_dim + vision_dim # 86
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, text_emb, tab_feats, img_tensor):
        # 1. Project vision
        vis_emb = self.vision_encoder(img_tensor) # (B, 16)
        
        # 2. Late Fusion Concatenation
        fused = torch.cat([text_emb, tab_feats, vis_emb], dim=1) # (B, 86)
        
        # 3. Final prediction
        logits = self.classifier(fused)
        return logits
