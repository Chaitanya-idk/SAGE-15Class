import glob
import json
import os
from typing import Any, Dict, Optional, Tuple
import torch
from sage.config import TrainingConfig
from sage.utils.hashing import compute_file_hash
from sage.utils.logging import get_logger

logger = get_logger("sage.resume")


def find_resumable_checkpoint(
    checkpoint_dir: str,
    expected_manifest_path: str,
    expected_class_registry_path: str
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Scans checkpoint directory, validates manifest and class hashes, and returns the latest valid checkpoint."""
    if not os.path.isdir(checkpoint_dir):
        logger.info(f"No checkpoint directory at {checkpoint_dir}. Starting fresh run.")
        return None

    expected_manifest_hash = compute_file_hash(expected_manifest_path) if os.path.exists(expected_manifest_path) else None
    expected_class_hash = compute_file_hash(expected_class_registry_path) if os.path.exists(expected_class_registry_path) else None

    # Check progress file first
    progress_file = os.path.join(checkpoint_dir, "training_progress.json")
    candidate_dirs = []

    if os.path.isfile(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            latest_path = p_data.get("latest_checkpoint")
            if latest_path and os.path.isdir(latest_path):
                candidate_dirs.append(latest_path)
        except Exception as e:
            logger.warning(f"Failed to read progress file: {e}")

    # Also search subdirectories
    for d in sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*_step_*")), reverse=True):
        if d not in candidate_dirs:
            candidate_dirs.append(d)

    for c_dir in candidate_dirs:
        meta_file = os.path.join(c_dir, "checkpoint_meta.json")
        state_file = os.path.join(c_dir, "training_state.pt")

        if not (os.path.isfile(meta_file) and os.path.isfile(state_file)):
            logger.warning(f"Incomplete checkpoint found at {c_dir}, skipping.")
            continue

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Validate manifest hash compatibility
            c_manifest_hash = meta.get("manifest_hash")
            if expected_manifest_hash and c_manifest_hash != expected_manifest_hash:
                logger.warning(f"Checkpoint at {c_dir} was built with a different manifest hash! Skipping to avoid corruption.")
                continue

            c_class_hash = meta.get("class_registry_hash")
            if expected_class_hash and c_class_hash != expected_class_hash:
                logger.warning(f"Checkpoint at {c_dir} has incompatible class registry hash! Skipping.")
                continue

            logger.info(f"Valid resumable checkpoint found: {c_dir} (Epoch {meta.get('epoch')}, Step {meta.get('global_step')})")
            return c_dir, meta

        except Exception as e:
            logger.warning(f"Error inspecting checkpoint {c_dir}: {e}")

    logger.info("No valid compatible checkpoint found. Starting fresh training run.")
    return None
