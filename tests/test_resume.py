import json
import os
import torch
try:
    import pytest
except ImportError:
    pytest = None
from sage.config import TrainingConfig
from sage.training.checkpointing import CheckpointManager
from sage.training.resume import find_resumable_checkpoint


def test_checkpoint_save_and_resume(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    manifest_file = str(tmp_path / "manifest.parquet")
    registry_file = str(tmp_path / "registry.json")

    # Create dummy files
    with open(manifest_file, "w") as f:
        f.write("dummy manifest content")
    with open(registry_file, "w") as f:
        f.write("{}")

    cfg = TrainingConfig(checkpoint_dir=ckpt_dir, progress_file=os.path.join(ckpt_dir, "progress.json"))
    manager = CheckpointManager(cfg, manifest_file, registry_file)

    # Dummy model and optimizer
    dummy_model = torch.nn.Linear(4, 2)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

    saved_path = manager.save_checkpoint(
        model=dummy_model,
        optimizer=dummy_optimizer,
        scheduler=None,
        epoch=1,
        global_step=50,
        batch_idx=50,
        metrics={"loss": 0.42},
        is_best=True
    )

    assert os.path.exists(saved_path)
    assert os.path.exists(os.path.join(saved_path, "checkpoint_meta.json"))
    assert os.path.exists(os.path.join(saved_path, "training_state.pt"))
    assert os.path.exists(os.path.join(ckpt_dir, "best"))

    # Test resume discovery
    res = find_resumable_checkpoint(ckpt_dir, manifest_file, registry_file)
    assert res is not None
    res_path, meta = res
    assert res_path == saved_path
    assert meta["epoch"] == 1
    assert meta["global_step"] == 50
