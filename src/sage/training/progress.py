import time
from typing import Optional
from tqdm import tqdm


class TrainingProgressTracker:
    """Tracks training progress with visible metrics, running loss, throughput, and ETA."""

    def __init__(self, total_epochs: int, steps_per_epoch: int):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.current_epoch = 0
        self.pbar: Optional[tqdm] = None
        self.start_time = time.time()
        self.step_start_time = time.time()

    def start_epoch(self, epoch: int, initial_step: int = 0):
        self.current_epoch = epoch
        self.step_start_time = time.time()
        self.pbar = tqdm(
            total=self.steps_per_epoch,
            initial=initial_step,
            desc=f"Epoch {epoch}/{self.total_epochs}",
            dynamic_ncols=True,
            unit="step"
        )

    def update_step(self, loss: float, lr: float, batch_size: int = 2):
        if self.pbar is None:
            return

        now = time.time()
        step_duration = max(1e-5, now - self.step_start_time)
        samples_per_sec = batch_size / step_duration
        self.step_start_time = now

        self.pbar.set_postfix({
            "loss": f"{loss:.4f}",
            "lr": f"{lr:.2e}",
            "samples/s": f"{samples_per_sec:.1f}",
        })
        self.pbar.update(1)

    def close_epoch(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None
