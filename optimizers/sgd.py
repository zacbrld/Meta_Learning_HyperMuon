import math
import torch
import torch.nn as nn


class SGDOptimizer:
    """SGD with momentum. Thin wrapper around torch.optim.SGD."""

    def __init__(self, model: nn.Module, lr: float = 0.1, momentum: float = 0.9):
        self._lr = lr
        self._mu = momentum
        self._optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        self._update_rms = float("nan")

    def zero_grad(self):
        self._optim.zero_grad()

    def step(self, model: nn.Module):
        # Track RMS of update before stepping
        rms_vals = []
        for p in model.parameters():
            if p.grad is not None:
                rms_vals.append(p.grad.detach().pow(2).mean().item())
        if rms_vals:
            self._update_rms = math.sqrt(sum(rms_vals) / len(rms_vals))
        self._optim.step()

    def get_metrics(self) -> dict:
        return {
            "lr": self._lr,
            "mu": self._mu,
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "hypgrad_lr": float("nan"),
            "hypgrad_mu": float("nan"),
            "hypgrad_abc": float("nan"),
            "update_rms": self._update_rms,
        }

    def lr_scheduler_step(self, scheduler):
        scheduler.step()
        self._lr = self._optim.param_groups[0]["lr"]
