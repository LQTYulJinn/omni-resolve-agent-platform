"""
src/data/pipeline.py
Entry point CLI for running the OmniResolve Data Pipeline.
Can be executed standalone, via CI/CD, or as a DVC stage.
"""
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.data.processor import DatasetProcessor

def main():
    parser = argparse.ArgumentParser(description="Run OmniResolve Data Cleansing and Manifest Generation Pipeline")
    parser.add_argument("--config", type=str, default="configs/data_pipeline.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config_path = BASE_DIR / args.config
    if not config_path.exists():
        print(f"[ERROR] Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processor = DatasetProcessor(config=config, base_dir=BASE_DIR)
    report = processor.process()
    print("[PIPELINE COMPLETE] Data Ingestion & Cleansing successfully executed!")

if __name__ == "__main__":
    main()
