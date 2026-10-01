"""
src/data/feature_extractor.py
Production-grade Multimodal Feature Extractor.
Extracts:
1. Lexical BM25 Tokens & Vocabulary for Exact Match Search.
2. Dense Semantic Embeddings for Text.
3. Computer Vision Tensors & Feature Vectors for Evidence Images.
4. Scaled Tabular Numerical & Categorical Features.
"""
import re
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class MultimodalFeatureExtractor:
    def __init__(self, text_dim: int = 64, image_size: Tuple[int, int] = (224, 224)):
        self.text_dim = text_dim
        self.image_size = image_size
        
        # Single Source of Truth for Vision Transform
        self.vision_transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # State to prevent Data Leakage: Scalers and Vocab fitted ONLY on Train
        self.fitted = False
        self.vocab: Dict[str, int] = {}
        self.amount_min: float = 0.0
        self.amount_max: float = 1.0
        self.claim_types = ["damaged_item", "wrong_item", "fake_item", "transaction_dispute", "delivery_delayed"]

    def _tokenize(self, text: str) -> List[str]:
        """Simple clean alphanumeric tokenization for lexical search / BM25."""
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9_]+\b", text)
        return tokens

    def fit(self, train_texts: List[str], train_amounts: List[float]):
        """Fit vocabulary and numerical scalers strictly on training data."""
        # 1. Build Lexical Vocabulary
        word_counts = {}
        for t in train_texts:
            for tok in self._tokenize(t):
                word_counts[tok] = word_counts.get(tok, 0) + 1
        
        # Keep top terms
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:1000]
        self.vocab = {w: idx + 1 for idx, (w, _) in enumerate(sorted_words)} # 0 reserved for unknown/pad

        # 2. Fit Log-Scaler for Amounts
        log_amounts = np.log1p(np.clip(train_amounts, 0, None))
        self.amount_min = float(np.min(log_amounts))
        self.amount_max = float(np.max(log_amounts)) if np.max(log_amounts) > self.amount_min else (self.amount_min + 1.0)

        self.fitted = True

    def transform_text_lexical(self, texts: List[str]) -> List[List[str]]:
        """Tokenize texts for BM25 search index."""
        return [self._tokenize(t) for t in texts]

    def transform_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate deterministic semantic embeddings for text.
        In production, replace with sentence-transformers or text-embedding-3-small.
        """
        embeddings = np.zeros((len(texts), self.text_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Deterministic hash-based projection for reproducible pipeline verification
            seed = sum(ord(c) for c in text) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.text_dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            embeddings[i] = vec / (norm + 1e-8)
        return embeddings

    def transform_image(self, image_path: Optional[str]) -> torch.Tensor:
        """Loads and applies standardized vision transformations."""
        if image_path and Path(image_path).exists():
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    return self.vision_transform(img)
            except Exception:
                pass
        # Return neutral zero-tensor if image is missing or failed
        return torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)

    def transform_tabular(self, amounts: List[float], claim_types: List[str]) -> np.ndarray:
        """Scales amount and applies one-hot encoding for claim types."""
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform_tabular()!")

        n = len(amounts)
        num_claim_classes = len(self.claim_types)
        features = np.zeros((n, 1 + num_claim_classes), dtype=np.float32)

        # 1. Log-scale and MinMax normalize amount
        log_amounts = np.log1p(np.clip(amounts, 0, None))
        scaled_amounts = (log_amounts - self.amount_min) / (self.amount_max - self.amount_min + 1e-8)
        features[:, 0] = np.clip(scaled_amounts, 0.0, 1.0)

        # 2. One-hot encode claim type
        for i, c_type in enumerate(claim_types):
            if c_type in self.claim_types:
                col_idx = 1 + self.claim_types.index(c_type)
                features[i, col_idx] = 1.0

        return features
