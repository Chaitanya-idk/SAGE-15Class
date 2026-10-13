import argparse
import os
import sys
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

from sage.config import load_classes_config, load_dataset_config
from sage.data.manifest import build_sample_manifest
from sage.utils.logging import get_logger

logger = get_logger("sage.scripts.manifest")


def main():
    parser = argparse.ArgumentParser(description="Build reproducible SAGE sample manifest.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset config YAML")
    parser.add_argument("--classes-config", type=str, default="configs/classes.yaml", help="Path to classes config YAML")
    parser.add_argument("--metadata-file", type=str, default=None, help="Override path to sage_metadata.parquet")
    args = parser.parse_args()

    dataset_cfg = load_dataset_config(args.config)
    classes_cfg = load_classes_config(args.classes_config)

    meta_file = args.metadata_file or dataset_cfg.metadata_file
    if not os.path.isfile(meta_file):
        logger.error(f"Metadata file not found: {meta_file}")
        sys.exit(1)

    logger.info(f"Loading metadata from {meta_file}...")
    df = pd.read_parquet(meta_file)

    logger.info("Executing balanced sampling and building manifest...")
    manifest_df, split_summary = build_sample_manifest(
        metadata_df=df,
        classes_config=classes_cfg,
        dataset_config=dataset_cfg
    )

    logger.info("Manifest creation completed successfully!")
    logger.info(f"Manifest path: {dataset_cfg.manifest_path}")
    logger.info(f"Total selected records: {len(manifest_df):,}")
    logger.info(f"Split distribution: {split_summary['split_counts']}")


if __name__ == "__main__":
    main()
