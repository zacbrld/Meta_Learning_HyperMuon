"""
Newton-Muon optimizer.

Newton-Muon applies Muon's matrix-sign update after right-preconditioning each
selected matrix gradient by an inverse input second moment:

    G <- G (ZZ^T / N + gamma I)^-1

The preconditioned gradient then goes through the standard Muon pipeline:
momentum, Frobenius normalization, Newton-Schulz orthogonalization, RMS scaling,
and weight decay. Non-selected parameters are handled by AdamW.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .muon import newton_schulz


class NewtonMuonOptimizer:
    def __init__(
        self,
        model: nn.Module,
        matrix_param_names: Optional[set[str]] = None,
        lr: float = 0.16,
        momentum: float = 0.75,
        weight_decay: float = 3e-4,
        ns_a: float = 3.4445,
        ns_b: float = -4.7750,
        ns_c: float = 2.0315,
        ns_iters: int = 5,
        adamw_lr: float = 8e-4,
        adamw_wd: float = 1e-2,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        ewma_beta: float = 0.95,
        ridge: float = 0.05,
        refresh_interval: int = 16,
        second_moment_init: float = 1e-3,
    ):
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be >= 1")

        self._base_lr = lr
        self._base_adamw_lr = adamw_lr
        self._lr = lr
        self._mu = momentum
        self._wd = weight_decay
        self._a = ns_a
        self._b = ns_b
        self._c = ns_c
        self._ns_iters = ns_iters
        self._ewma_beta = ewma_beta
        self._ridge = ridge
        self._refresh_interval = refresh_interval
        self._second_moment_init = second_moment_init

        named_params = dict(model.named_parameters())
        if matrix_param_names is not None:
            missing = set(matrix_param_names) - set(named_params)
            if missing:
                raise ValueError(f"Unknown matrix parameters: {sorted(missing)}")

        matrix_params, matrix_names, adamw_params = [], [], []
        for name, param in model.named_parameters():
            use_newton_muon = param.dim() >= 2
            if matrix_param_names is not None:
                use_newton_muon = name in matrix_param_names

            if use_newton_muon:
                matrix_params.append(param)
                matrix_names.append(name)
            else:
                adamw_params.append(param)

        self._matrix_params = matrix_params
        self._matrix_names = matrix_names
        self._adamw_optim = (
            torch.optim.AdamW(
                adamw_params,
                lr=adamw_lr,
                weight_decay=adamw_wd,
                betas=adamw_betas,
            )
            if adamw_params
            else None
        )

        self._M = [
            torch.zeros_like(param, memory_format=torch.preserve_format)
            for param in matrix_params
        ]
        self._K: list[Optional[torch.Tensor]] = [None for _ in matrix_params]
        self._K_inv: list[Optional[torch.Tensor]] = [None for _ in matrix_params]
        self._input_cache: dict[int, torch.Tensor] = {}
        self._param_id_to_index = {id(param): i for i, param in enumerate(matrix_params)}
        self._hooks = []

        self._register_activation_hooks(model, set(matrix_names))

        self._step_count = 0
        self._update_rms = float("nan")
        self._refresh_count = 0

    def _register_activation_hooks(self, model: nn.Module, matrix_names: set[str]):
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            weight_name = f"{module_name}.weight" if module_name else "weight"
            if weight_name not in matrix_names:
                continue

            def hook(linear_module, inputs, param_id=id(module.weight)):
                if not inputs:
                    return
                x = inputs[0].detach()
                if x.numel() == 0:
                    return
                x = x.reshape(-1, x.shape[-1])
                self._input_cache[param_id] = x

            self._hooks.append(module.register_forward_pre_hook(hook))

    def zero_grad(self):
        for param in self._matrix_params:
            param.grad = None
        if self._adamw_optim is not None:
            self._adamw_optim.zero_grad()

    def _identity(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.eye(n, device=device, dtype=torch.float32)

    def _safe_inverse(self, matrix: torch.Tensor) -> torch.Tensor:
        n = matrix.shape[0]
        eye = self._identity(n, matrix.device)
        mean_diag = torch.diagonal(matrix).mean().clamp_min(1e-12)

        for attempt in range(6):
            jitter = 0.0 if attempt == 0 else (10.0 ** (attempt - 7)) * mean_diag
            try:
                chol = torch.linalg.cholesky(matrix + jitter * eye)
                return torch.cholesky_inverse(chol)
            except RuntimeError:
                continue

        return torch.linalg.pinv(matrix)

    def _maybe_refresh_preconditioner(self, index: int, param: torch.Tensor):
        input_activations = self._input_cache.get(id(param))
        input_dim = param.reshape(param.shape[0], -1).shape[1]
        device = param.device

        if self._K_inv[index] is None:
            self._K_inv[index] = self._identity(input_dim, device)

        should_refresh = (self._step_count + 1) % self._refresh_interval == 0
        if not should_refresh or input_activations is None:
            return

        x = input_activations.float()
        if x.shape[1] != input_dim:
            return

        batch_second_moment = (x.T @ x) / max(1, x.shape[0])
        if self._K[index] is None:
            self._K[index] = self._second_moment_init * self._identity(input_dim, device)

        self._K[index].mul_(self._ewma_beta).add_(
            batch_second_moment, alpha=1.0 - self._ewma_beta
        )

        mean_trace = torch.trace(self._K[index]) / input_dim
        damping = (self._ridge * mean_trace).clamp_min(1e-12)
        damped = self._K[index] + damping * self._identity(input_dim, device)
        self._K_inv[index] = self._safe_inverse(damped)
        self._refresh_count += 1

    def _precondition_gradient(
        self,
        index: int,
        param: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        self._maybe_refresh_preconditioner(index, param)

        shape_orig = grad.shape
        grad_2d = grad.reshape(shape_orig[0], -1) if grad.dim() > 2 else grad
        K_inv = self._K_inv[index]
        if K_inv is None or K_inv.shape[0] != grad_2d.shape[1]:
            return grad

        preconditioned = grad_2d.float() @ K_inv.to(device=grad_2d.device)
        preconditioned = preconditioned.to(dtype=grad.dtype)
        return preconditioned.reshape(shape_orig) if grad.dim() > 2 else preconditioned

    def step(self, model: nn.Module = None):
        rms_vals = []

        for index, param in enumerate(self._matrix_params):
            if param.grad is None:
                continue

            grad = self._precondition_gradient(index, param, param.grad.detach())

            # Newton-Muon applies right-preconditioning before Muon momentum.
            momentum = self._mu * self._M[index] + grad

            shape_orig = param.shape
            if param.dim() > 2:
                momentum_2d = momentum.reshape(shape_orig[0], -1)
            else:
                momentum_2d = momentum

            norm = momentum_2d.norm(p="fro") + 1e-8
            X = momentum_2d / norm
            X_orth = newton_schulz(X, self._a, self._b, self._c, self._ns_iters)

            rows, cols = momentum_2d.shape
            update_2d = 0.2 * X_orth * math.sqrt(max(rows, cols))
            if param.dim() > 2:
                update = update_2d.to(param.dtype).reshape(shape_orig)
            else:
                update = update_2d.to(param.dtype)

            rms_vals.append(update.float().pow(2).mean().item())
            param.data.sub_(self._lr * (update + self._wd * param.detach()))
            self._M[index] = momentum.detach()

        if rms_vals:
            self._update_rms = math.sqrt(sum(rms_vals) / len(rms_vals))

        if self._adamw_optim is not None:
            self._adamw_optim.step()

        self._step_count += 1

    def set_lr_scale(self, scale: float):
        self._lr = self._base_lr * scale
        if self._adamw_optim is not None:
            for group in self._adamw_optim.param_groups:
                group["lr"] = self._base_adamw_lr * scale

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
            "ewma_beta": self._ewma_beta,
            "ridge": self._ridge,
            "refresh_interval": self._refresh_interval,
            "precond_refreshes": self._refresh_count,
        }
