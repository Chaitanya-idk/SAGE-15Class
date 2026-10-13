import time
from typing import Optional
from tqdm import tqdm


class TrainingProgressTracker:
    """Tracks training progress with live per-batch loss, throughput, and dynamic epoch ETA."""

    def __init__(self, total_epochs: int, batches_per_epoch: int):
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.current_epoch = 0
        self.pbar: Optional[tqdm] = None
        self.start_time = time.time()
        self.batch_start_time = time.time()
        self.running_loss = 0.0
        self.batch_count = 0

    def start_epoch(self, epoch: int, initial_batch: int = 0):
        self.current_epoch = epoch
        self.batch_start_time = time.time()
        self.running_loss = 0.0
        self.batch_count = initial_batch
        self.pbar = tqdm(
            total=self.batches_per_epoch,
            initial=initial_batch,
            desc=f"Epoch {epoch}/{self.total_epochs}",
            dynamic_ncols=True,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

    def update_batch(self, batch_loss: float, lr: float, batch_size: int = 2, opt_step: int = 0):
        if self.pbar is None:
            return

        now = time.time()
        batch_duration = max(1e-5, now - self.batch_start_time)
        samples_per_sec = batch_size / batch_duration
        self.batch_start_time = now

        self.batch_count += 1
        self.running_loss += (batch_loss - self.running_loss) / max(1, self.batch_count)

        self.pbar.set_postfix({
            "batch_loss": f"{batch_loss:.4f}",
            "avg_loss": f"{self.running_loss:.4f}",
            "lr": f"{lr:.2e}",
            "step": opt_step,
            "samples/s": f"{samples_per_sec:.1f}",
        })
        self.pbar.update(1)

    def close_epoch(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None
