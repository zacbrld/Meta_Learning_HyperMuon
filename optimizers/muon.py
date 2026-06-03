"""
Muon optimizer with fixed hyperparameters.

Algorithm per step for each matrix parameter W:
  1. Nesterov momentum: M = mu*M_prev + g
  2. Normalize:         X = M / (||M||_F + eps)
  3. Newton-Schulz:     X ← 5 iterations → UVᵀ polar factor (float32)
  4. RMS scaling:       O = 0.2 * X * sqrt(max(m, n))
  5. Weight decay + update: W ← W - lr*(O + λ*W)

Non-matrix params (bias, BN) are handled by AdamW.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Newton-Schulz helper
# ---------------------------------------------------------------------------

def _ns_iter(X: torch.Tensor, a: float, b: float, c: float, n_iters: int = 5):
    """
    Run n_iters of Newton-Schulz on X [m, n] with m <= n.
    Operates in float32; returns float32.
    """
    X = X.float()
    for _ in range(n_iters):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * ((A @ A) @ X)
    return X


def newton_schulz(W_2d: torch.Tensor, a: float, b: float, c: float, n_iters: int = 5):
    """
    Apply Newton-Schulz orthogonalization to a 2D matrix.
    Transposes tall matrices (m > n) for efficiency and numerical stability.
    Returns tensor in the original dtype.
    """
    dtype_orig = W_2d.dtype
    transposed = W_2d.shape[0] > W_2d.shape[1]
    X = W_2d.T if transposed else W_2d
    X = _ns_iter(X, a, b, c, n_iters)
    if transposed:
        X = X.T
    return X.to(dtype=dtype_orig)


# ---------------------------------------------------------------------------
# MuonOptimizer
# ---------------------------------------------------------------------------

class MuonOptimizer:
    """
    Muon with fixed hyperparameters.

    Parameters
    ----------
    model       : the nn.Module whose parameters will be optimised
    lr          : learning rate η  (default 1e-3)
    momentum    : Nesterov momentum µ  (default 0.95)
    weight_decay: λ applied to Muon params  (default 0.1)
    ns_a/b/c    : Newton-Schulz polynomial coefficients
    ns_iters    : number of N-S iterations  (default 5)
    adamw_lr    : learning rate for non-matrix params
    adamw_wd    : weight decay for non-matrix params
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_a: float = 3.4445,
        ns_b: float = -4.7750,
        ns_c: float = 2.0315,
        ns_iters: int = 5,
        adamw_lr: float = 1e-3,
        adamw_wd: float = 0.01,
    ):
        self._lr = lr
        self._mu = momentum
        self._wd = weight_decay
        self._a = ns_a
        self._b = ns_b
        self._c = ns_c
        self._ns_iters = ns_iters

        # Split parameters into Muon (dim >= 2) and AdamW (dim < 2)
        muon_params, adamw_params = [], []
        for p in model.parameters():
            if p.dim() >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        self._muon_params = muon_params
        self._adamw_optim = torch.optim.AdamW(
            adamw_params, lr=adamw_lr, weight_decay=adamw_wd
        ) if adamw_params else None

        # Momentum buffers
        self._M = [torch.zeros_like(p, memory_format=torch.preserve_format)
                   for p in muon_params]

        self._update_rms = float("nan")

    # ------------------------------------------------------------------

    def zero_grad(self):
        for p in self._muon_params:
            p.grad = None
        if self._adamw_optim is not None:
            self._adamw_optim.zero_grad()

    def step(self, model: nn.Module = None):
        """Apply one Muon step (call after loss.backward())."""
        rms_vals = []

        for i, param in enumerate(self._muon_params):
            if param.grad is None:
                continue

            g = param.grad.detach()

            # 1. Nesterov momentum
            M = self._mu * self._M[i] + g

            # 2–4. Newton-Schulz on 2D view
            shape_orig = param.shape
            if param.dim() > 2:
                M_2d = M.reshape(shape_orig[0], -1)
            else:
                M_2d = M

            norm = M_2d.norm(p="fro") + 1e-8
            X = M_2d / norm

            X_orth = newton_schulz(X, self._a, self._b, self._c, self._ns_iters)

            A_dim = M_2d.shape[0]
            B_dim = M_2d.shape[1]
            O_2d = 0.2 * X_orth * math.sqrt(max(A_dim, B_dim))

            if param.dim() > 2:
                O = O_2d.to(param.dtype).reshape(shape_orig)
            else:
                O = O_2d.to(param.dtype)

            # 5. Update (track RMS of O — the orthogonalized update before lr scaling)
            rms_vals.append(O.float().pow(2).mean().item())

            update = self._lr * (O + self._wd * param.detach())
            param.data.sub_(update)

            # Save momentum (detached)
            self._M[i] = M.detach()

        if rms_vals:
            self._update_rms = math.sqrt(sum(rms_vals) / len(rms_vals))  # RMS of O ≈ 0.2

        if self._adamw_optim is not None:
            self._adamw_optim.step()

    def get_metrics(self) -> dict:
        return {
            "lr": self._lr,
            "mu": self._mu,
            "a": self._a,
            "b": self._b,
            "c": self._c,
            "hypgrad_lr": float("nan"),
            "hypgrad_mu": float("nan"),
            "hypgrad_abc": float("nan"),
            "update_rms": self._update_rms,
        }
