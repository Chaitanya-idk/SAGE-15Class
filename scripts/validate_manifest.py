import argparse
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

from sage.config import load_dataset_config
from sage.data.manifest import validate_sample_manifest
from sage.utils.logging import get_logger

logger = get_logger("sage.scripts.validate")


def main():
    parser = argparse.ArgumentParser(description="Validate SAGE sample manifest for completeness and split leakage.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset config YAML")
    parser.add_argument("--manifest-path", type=str, default=None, help="Override path to sample_manifest.parquet")
    args = parser.parse_args()

    dataset_cfg = load_dataset_config(args.config)
    manifest_path = args.manifest_path or dataset_cfg.manifest_path

    if not os.path.isfile(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    logger.info(f"Validating manifest: {manifest_path}...")
    res = validate_sample_manifest(manifest_path)

    if res["valid"]:
        logger.info("Manifest validation PASSED!")
        logger.info(f"Total samples: {res['total_samples']:,}")
        logger.info(f"Classes verified: {res['num_classes']}")
        logger.info(f"Splits verified: {res['splits']}")
        logger.info("Zero split leakage detected across duplicate groups.")
        sys.exit(0)
    else:
        logger.error("Manifest validation FAILED!")
        for issue in res["issues"]:
            logger.error(f"  - {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
