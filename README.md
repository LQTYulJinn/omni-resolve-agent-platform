# 🚀 OmniResolve: Autonomous Multimodal Operations & Claim Resolution Agent Platform

> Production-grade AI Agent Platform for automating high-scale multimodal dispute resolution, e-commerce warranty claims, and fintech transaction operations with strict guardrails, zero data leakage, and hybrid search.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Data%20Validation-Pydantic%20v2-green.svg)](https://docs.pydantic.dev/)
[![DVC](https://img.shields.io/badge/MLOps-DVC%20Pipeline-purple.svg)](https://dvc.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](https://pytest.org/)

---

## 📌 Architecture Overview

```text
OmniResolve Platform Architecture
├── 1. Data Pipeline (Ingestion, PII Masking, Image Gate, GroupSplit, DVC)  <-- [COMPLETED]
├── 2. Hybrid Retrieval Engine (Dense Semantic + BM25 Lexical Search)        <-- [NEXT]
├── 3. Multi-Step Decision Agent (Vision Tool, Policy Tool, Fraud Scorer)
├── 4. Serving & Production API (FastAPI, Lifespan Async Serving)
└── 5. Continuous Reliability & Guardrails (Data Drift, Human-in-the-loop)
```

---

## 🛠️ Phase 1: Multimodal Data Engineering & Quality Gate

### 1. Key Engineering Highlights
* **Strict Data Contracts**: Enforced via Pydantic schemas (`ClaimTicketRaw`, `ValidatedTicket`) preventing downstream runtime crashes.
* **Multimodal Validation Gate**:
  * PIL Byte Verification: Detects truncated or corrupted image headers without loading whole bitmaps into memory.
  * PII Sanitization: High-speed Regex masking for Credit Card numbers, Phone numbers, and Emails (`[REDACTED_CC]`, `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`).
* **Zero Data Leakage Group Splitting**:
  * Utilizes `GroupShuffleSplit` on `user_id` ensuring a customer's claims never leak across both Train and Validation sets.
* **Storage Optimization**: Serializes manifests to Apache Parquet for high-throughput columnar querying.
* **DVC DAG Orchestration**: Tracked via `dvc.yaml` for deterministic pipeline reproduction.

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone <your-repo-url>
cd omni_resolve_platform
pip install -r requirements.txt
```

### 2. Run Data Quality Pipeline
```bash
# Execute via DVC (reproducible DAG)
dvc repro

# Or run standalone CLI
python src/data/pipeline.py --config configs/data_pipeline.yaml
```

### 3. Run Quality Gate Unit Tests
```bash
python -m pytest tests/test_data_pipeline.py -v
```
