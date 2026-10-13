import argparse
import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from sage.config import load_classes_config, load_dataset_config
from sage.data.inventory import audit_dataset_inventory
from sage.utils.logging import get_logger

logger = get_logger("sage.scripts.audit")


def main():
    parser = argparse.ArgumentParser(description="Audit SAGE dataset schema and generate inventory artifacts.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset config YAML")
    parser.add_argument("--classes-config", type=str, default="configs/classes.yaml", help="Path to classes config YAML")
    parser.add_argument("--metadata-file", type=str, default=None, help="Override path to sage_metadata.parquet")
    args = parser.parse_args()

    dataset_cfg = load_dataset_config(args.config)
    classes_cfg = load_classes_config(args.classes_config)

    meta_file = args.metadata_file or dataset_cfg.metadata_file
    if not os.path.isfile(meta_file):
        logger.error(f"Metadata file not found: {meta_file}. Please download sage_metadata.parquet first.")
        sys.exit(1)

    logger.info(f"Loading metadata from {meta_file}...")
    df = pd.read_parquet(meta_file)

    logger.info(f"Loaded {len(df):,} records. Running schema and inventory audit...")
    schema_art, inv_art = audit_dataset_inventory(
        metadata_df=df,
        classes_config=classes_cfg,
        dataset_config=dataset_cfg
    )

    logger.info("Audit successfully completed!")
    logger.info(f"Schema artifact: {os.path.join(dataset_cfg.output_dir, 'dataset_schema.json')}")
    logger.info(f"Inventory artifact: {os.path.join(dataset_cfg.output_dir, 'dataset_inventory.json')}")
    logger.info(f"Total enumerated records: {inv_art['total_records_enumerated']:,}")
    logger.info(f"Records with valid crop & disease: {inv_art['records_with_valid_crop_and_disease']:,}")


if __name__ == "__main__":
    main()
