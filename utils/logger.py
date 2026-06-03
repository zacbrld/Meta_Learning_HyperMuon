import csv
import os
import math
from pathlib import Path


_COLUMNS = [
    "step", "epoch", "train_loss",
    "val_loss", "val_accuracy", "test_accuracy",
    "lr", "mu", "a", "b", "c",
    "hypgrad_lr", "hypgrad_mu", "hypgrad_abc",
    "update_rms",
]


class CSVLogger:
    """Writes one CSV row per step or per epoch according to spec."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=_COLUMNS)
        self._writer.writeheader()

    def log_step(self, step: int, epoch: int, train_loss: float, metrics: dict):
        """Called every training step — logs train_loss, NaN for epoch-only columns."""
        row = {col: float("nan") for col in _COLUMNS}
        row["step"] = step
        row["epoch"] = epoch
        row["train_loss"] = train_loss
        # Hyperparameter values (available every step)
        for key in ("lr", "mu", "a", "b", "c",
                    "hypgrad_lr", "hypgrad_mu", "hypgrad_abc", "update_rms"):
            if key in metrics:
                row[key] = metrics[key]
        self._writer.writerow(row)

    def log_epoch(self, step: int, epoch: int, val_loss: float,
                  val_accuracy: float, metrics: dict,
                  test_accuracy: float = float("nan")):
        """Called once per epoch — logs val metrics and current hyperparams."""
        row = {col: float("nan") for col in _COLUMNS}
        row["step"] = step
        row["epoch"] = epoch
        row["val_loss"] = val_loss
        row["val_accuracy"] = val_accuracy
        row["test_accuracy"] = test_accuracy
        for key in ("lr", "mu", "a", "b", "c",
                    "hypgrad_lr", "hypgrad_mu", "hypgrad_abc", "update_rms"):
            if key in metrics:
                row[key] = metrics[key]
        self._writer.writerow(row)

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass
