"""
HyperAdam — reproduces GD-UO (Chandra et al., NeurIPS 2022) for AdamW.

Learns learning rate η via automatic differentiation.
η is log-parameterized: lr = exp(lr_raw), lr_raw is an nn.Parameter.

Hypergradient via proxy approximation (first-order Taylor):
  proxy = -Σ_W ⟨g_W, update_W(θ)⟩
  proxy.backward() → lr_raw.grad
  lr_raw.data -= kappa * lr_raw.grad
"""

import math
import torch
import torch.nn as nn


class HyperAdamOptimizer:
    """
    AdamW with learned learning rate.

    Parameters
    ----------
    model       : nn.Module
    lr_init     : initial learning rate (default 1e-3)
    betas       : (β1, β2) for AdamW  (fixed)
    weight_decay: λ  (fixed)
    kappa       : meta-learning rate for lr_raw  (default 1e-5)
    eps         : AdamW epsilon  (default 1e-8)
    """

    def __init__(
        self,
        model: nn.Module,
        lr_init: float = 1e-3,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        kappa: float = 1e-5,
        eps: float = 1e-8,
    ):
        self._beta1, self._beta2 = betas
        self._wd = weight_decay
        self._kappa = kappa
        self._eps = eps

        # Learnable log-lr
        self.lr_raw = nn.Parameter(
            torch.tensor(math.log(lr_init)), requires_grad=True
        )

        self._params = list(model.parameters())

        # Adam state: first/second moments, step counter
        self._m = [torch.zeros_like(p) for p in self._params]
        # Initialize v to ε to avoid √0 division (per GD-UO paper note)
        self._v = [torch.full_like(p, fill_value=eps) for p in self._params]
        self._t = 0

        self._update_rms = float("nan")

    # ------------------------------------------------------------------

    def zero_grad(self):
        for p in self._params:
            p.grad = None
        if self.lr_raw.grad is not None:
            self.lr_raw.grad = None

    def step(self, model: nn.Module = None):
        """
        Call after loss.backward().
        Computes proxy loss → hypergradient → updates lr_raw and all params.
        """
        self._t += 1
        t = self._t
        b1, b2, eps = self._beta1, self._beta2, self._eps
        wd = self._wd

        lr = torch.exp(self.lr_raw)  # lr in computation graph

        proxy_terms = []
        rms_vals = []

        for i, param in enumerate(self._params):
            if param.grad is None:
                continue
            g = param.grad.detach()

            # Update Adam moments (detached — only lr is in graph)
            m_new = b1 * self._m[i] + (1 - b1) * g
            v_new = b2 * self._v[i] + (1 - b2) * g * g

            m_hat = m_new / (1 - b1 ** t)
            v_hat = v_new / (1 - b2 ** t)

            # adam_step depends on lr (in graph), m_hat/v_hat are detached
            m_hat_d = m_hat.detach()
            v_hat_d = v_hat.detach()
            adam_step = m_hat_d / (v_hat_d.sqrt() + eps)

            update = lr * (adam_step + wd * param.detach())

            # Proxy: approximate ΔL ≈ -⟨g, update⟩
            proxy_terms.append((g, update))

            rms_vals.append(update.detach().pow(2).mean().item())

            # Save updated moments
            self._m[i] = m_new.detach()
            self._v[i] = v_new.detach()

        # Hypergradient via proxy
        if proxy_terms:
            proxy = -sum((g * upd).sum() for g, upd in proxy_terms)
            proxy.backward()

            torch.nn.utils.clip_grad_norm_([self.lr_raw], max_norm=1.0)
            self.lr_raw.data -= self._kappa * self.lr_raw.grad
            self.lr_raw.grad = None

        # Apply parameter updates (with now-updated lr)
        lr_val = math.exp(self.lr_raw.item())
        for i, param in enumerate(self._params):
            if param.grad is None:
                continue
            g = param.grad.detach()

            t_cur = self._t
            # Re-derive m_hat / v_hat from stored (already-updated) moments
            m_hat = self._m[i] / (1 - b1 ** t_cur)
            v_hat = self._v[i] / (1 - b2 ** t_cur)
            adam_step = m_hat / (v_hat.sqrt() + eps)

            param.data -= lr_val * (adam_step + wd * param.data)

        if rms_vals:
            self._update_rms = math.sqrt(sum(rms_vals) / len(rms_vals))

    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        return {
            "lr": math.exp(self.lr_raw.item()),
            "mu": float("nan"),
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "hypgrad_lr": abs(self.lr_raw.grad.item()) if self.lr_raw.grad is not None else float("nan"),
            "hypgrad_mu": float("nan"),
            "hypgrad_abc": float("nan"),
            "update_rms": self._update_rms,
        }
