import argparse
import os
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.abspath("src"))

from sage.utils.logging import get_logger

logger = get_logger("sage.scripts.export")


def main():
    parser = argparse.ArgumentParser(description="Export and package SAGE training artifacts, checkpoints, and reports.")
    parser.add_argument("--output-tar", type=str, default="artifacts/sage_run_artifacts.tar.gz", help="Output archive path")
    parser.add_argument("--include-adapter", action="store_true", default=True, help="Include trained LoRA adapter")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_tar)), exist_ok=True)
    logger.info(f"Packaging artifacts into {args.output_tar}...")

    with tarfile.open(args.output_tar, "w:gz") as tar:
        # Add metadata
        if os.path.exists("artifacts/metadata"):
            tar.add("artifacts/metadata", arcname="metadata")
            logger.info("Added artifacts/metadata")

        # Add reports
        if os.path.exists("artifacts/reports"):
            tar.add("artifacts/reports", arcname="reports")
            logger.info("Added artifacts/reports")

        # Add predictions
        if os.path.exists("artifacts/predictions"):
            tar.add("artifacts/predictions", arcname="predictions")
            logger.info("Added artifacts/predictions")

        # Add final adapter
        if args.include_adapter and os.path.exists("artifacts/checkpoints/final_adapter"):
            tar.add("artifacts/checkpoints/final_adapter", arcname="final_adapter")
            logger.info("Added artifacts/checkpoints/final_adapter")

    logger.info(f"Archive successfully created: {args.output_tar} ({os.path.getsize(args.output_tar) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
