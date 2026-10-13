from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
from typing import Any, Dict, Optional
import torch
from sage.config import TrainingConfig
from sage.utils.hashing import compute_file_hash
from sage.utils.logging import get_logger

logger = get_logger("sage.checkpointing")


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash if in a Git repository."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


class CheckpointManager:
    """Manages atomic checkpoint saving, rotation, and metadata tracking."""

    def __init__(self, training_config: TrainingConfig, manifest_path: str, class_registry_path: str):
        self.config = training_config
        self.checkpoint_dir = training_config.checkpoint_dir
        self.manifest_path = manifest_path
        self.class_registry_path = class_registry_path
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.manifest_hash = compute_file_hash(manifest_path) if os.path.exists(manifest_path) else "unknown"
        self.class_registry_hash = compute_file_hash(class_registry_path) if os.path.exists(class_registry_path) else "unknown"
        self.git_commit = get_git_commit_hash()

    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        epoch: int,
        global_step: int,
        batch_idx: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        tag: Optional[str] = None
    ) -> str:
        """Atomically saves checkpoint directory containing adapter weights, training state, and metadata."""
        tag_str = tag if tag else f"epoch_{epoch}_step_{global_step}"
        final_dir = os.path.join(self.checkpoint_dir, tag_str)
        temp_dir = os.path.join(self.checkpoint_dir, f".temp_{tag_str}")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"Saving checkpoint to temporary path {temp_dir}...")

        # 1. Save adapter weights
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(temp_dir)
        else:
            torch.save(model.state_dict(), os.path.join(temp_dir, "model_state.pt"))

        # 2. Save training optimizer, scheduler, and RNG state
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "batch_idx": batch_idx,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "manifest_hash": self.manifest_hash,
            "class_registry_hash": self.class_registry_hash,
            "git_commit": self.git_commit,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(training_state, os.path.join(temp_dir, "training_state.pt"))

        # 3. Save metadata JSON
        meta_json = {
            "epoch": epoch,
            "global_step": global_step,
            "batch_idx": batch_idx,
            "metrics": metrics,
            "manifest_hash": self.manifest_hash,
            "class_registry_hash": self.class_registry_hash,
            "git_commit": self.git_commit,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "is_best": is_best
        }
        with open(os.path.join(temp_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_json, f, indent=2)

        # Atomic rename
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(temp_dir, final_dir)

        # Update global progress file
        progress_data = {
            "latest_checkpoint": final_dir,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
            "manifest_hash": self.manifest_hash,
            "last_saved_utc": datetime.now(timezone.utc).isoformat()
        }
        with open(self.config.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2)

        # If best, copy or save to 'best'
        if is_best:
            best_dir = os.path.join(self.checkpoint_dir, "best")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(final_dir, best_dir)
            logger.info(f"Saved new BEST checkpoint to {best_dir}!")

        logger.info(f"Checkpoint successfully finalized: {final_dir}")
        return final_dir
