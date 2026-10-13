import argparse
import os
import sys
import pandas as pd
from transformers import AutoProcessor
from peft import PeftModel

sys.path.insert(0, os.path.abspath("src"))

from sage.config import load_classes_config, load_dataset_config, load_evaluation_config, load_training_config
from sage.evaluation.evaluate import run_evaluation_pipeline
from sage.models.load_model import load_qwen_vl_model
from sage.utils.logging import get_logger

logger = get_logger("sage.scripts.evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen2.5-VL model on validation or test set.")
    parser.add_argument("--eval-config", type=str, default="configs/evaluation.yaml", help="Path to evaluation config YAML")
    parser.add_argument("--training-config", type=str, default="configs/training.yaml", help="Path to training config YAML")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset.yaml", help="Path to dataset config YAML")
    parser.add_argument("--classes-config", type=str, default="configs/classes.yaml", help="Path to classes config YAML")
    parser.add_argument("--adapter-dir", type=str, default="artifacts/checkpoints/final_adapter", help="Path to trained LoRA adapter")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--manifest-path", type=str, default=None, help="Override path to sample_manifest.parquet")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing images")
    args = parser.parse_args()

    eval_cfg = load_evaluation_config(args.eval_config)
    train_cfg = load_training_config(args.training_config)
    dataset_cfg = load_dataset_config(args.dataset_config)
    classes_cfg = load_classes_config(args.classes_config)

    manifest_path = args.manifest_path or dataset_cfg.manifest_path
    if not os.path.isfile(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest_df = pd.read_parquet(manifest_path)

    logger.info("Loading base model and processor...")
    base_model, processor = load_qwen_vl_model(train_cfg, is_training=False)

    if os.path.isdir(args.adapter_dir):
        logger.info(f"Loading LoRA adapter from {args.adapter_dir}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    else:
        logger.warning(f"Adapter directory {args.adapter_dir} not found. Running zero-shot evaluation on base model.")
        model = base_model

    logger.info(f"Running evaluation on '{args.split}' split...")
    metrics = run_evaluation_pipeline(
        model=model,
        processor=processor,
        manifest_df=manifest_df,
        classes_config=classes_cfg,
        eval_config=eval_cfg,
        split=args.split,
        image_dir=args.image_dir
    )

    logger.info("Evaluation finished successfully!")
    logger.info(f"Exact Match Accuracy: {metrics.get('exact_match_accuracy', 0.0):.4f}")
    logger.info(f"Macro F1: {metrics.get('macro_f1', 0.0):.4f}")
    logger.info(f"Reports saved to: {eval_cfg.reports_dir}")


if __name__ == "__main__":
    main()
