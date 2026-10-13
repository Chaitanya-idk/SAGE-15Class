import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("src"))

from sage.config import load_classes_config, load_dataset_config, load_training_config
from sage.training.train import run_training_pipeline
from sage.utils.logging import get_logger

logger = get_logger("sage.scripts.train")


def main():
    parser = argparse.ArgumentParser(description="Run 2-epoch QLoRA fine-tuning on Qwen2.5-VL-3B-Instruct.")
    parser.add_argument("--training-config", type=str, default="configs/training.yaml", help="Path to training config YAML")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset.yaml", help="Path to dataset config YAML")
    parser.add_argument("--classes-config", type=str, default="configs/classes.yaml", help="Path to classes config YAML")
    parser.add_argument("--manifest-path", type=str, default=None, help="Override path to sample_manifest.parquet")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing image files or Kaggle input mount")
    args = parser.parse_args()

    train_cfg = load_training_config(args.training_config)
    dataset_cfg = load_dataset_config(args.dataset_config)
    classes_cfg = load_classes_config(args.classes_config)

    manifest_path = args.manifest_path or dataset_cfg.manifest_path
    if not os.path.isfile(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}. Please run scripts/build_manifest.py first.")
        sys.exit(1)

    logger.info(f"Loading manifest from {manifest_path}...")
    manifest_df = pd.read_parquet(manifest_path)

    logger.info("Initiating 2-epoch training pipeline...")
    results = run_training_pipeline(
        manifest_df=manifest_df,
        classes_config=classes_cfg,
        dataset_config=dataset_cfg,
        training_config=train_cfg,
        image_dir=args.image_dir
    )

    logger.info("Training pipeline completed!")
    logger.info(f"Final adapter directory: {results.get('final_adapter_dir')}")


if __name__ == "__main__":
    main()
