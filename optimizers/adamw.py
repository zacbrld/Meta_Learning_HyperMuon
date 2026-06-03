import math
import torch
import torch.nn as nn


class AdamWOptimizer:
    """AdamW. Thin wrapper around torch.optim.AdamW."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
    ):
        self._lr = lr
        self._optim = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self._update_rms = float("nan")

    def zero_grad(self):
        self._optim.zero_grad()

    def step(self, model: nn.Module):
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
            "mu": float("nan"),
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
