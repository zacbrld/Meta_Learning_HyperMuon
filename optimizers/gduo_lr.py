"""
LR-only GD-UO optimizers.

These optimizers learn only the scalar learning rate with the one-step
hypergradient from Chandra et al., "Gradient Descent: The Ultimate Optimizer".
For a previous update

    W_t = W_{t-1} - lr_{t-1} * U_{t-1},

the current gradient gives

    d loss_t / d lr_{t-1} = - <grad_t, U_{t-1}>.

This is different from the local proxy used in HyperMuon: the signal comes from
the update that was actually applied on the previous step.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .muon import newton_schulz
from .newton_muon import NewtonMuonOptimizer


def _clip_scalar(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _logit(x: float) -> float:
    x = min(max(x, 1e-12), 1.0 - 1e-12)
    return math.log(x / (1.0 - x))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _finite_difference_direction(
    direction_fn,
    momentum: torch.Tensor,
    dmomentum_draw: Optional[torch.Tensor],
    eps: float = 1e-3,
) -> Optional[torch.Tensor]:
    if dmomentum_draw is None:
        return None
    if not torch.isfinite(dmomentum_draw.float()).all():
        return None
    if dmomentum_draw.float().abs().max().item() == 0.0:
        return None

    plus = direction_fn(momentum + eps * dmomentum_draw)
    minus = direction_fn(momentum - eps * dmomentum_draw)
    return ((plus - minus) / (2.0 * eps)).detach()


class _GDUOLRState:
    def _init_gduo_lr(
        self,
        lr_init: float,
        learn_lr: bool = True,
        hyper_lr: float = 1e-3,
        hypergrad_clip: float = 1.0,
        lr_min_ratio: float = 0.05,
        lr_max_ratio: float = 5.0,
    ):
        self.lr_raw = torch.tensor(math.log(lr_init), dtype=torch.float64)
        self._gduo_learn_lr = learn_lr
        self._lr_scale = 1.0
        self._prev_actual_lr: Optional[float] = None
        self._gduo_hyper_lr = hyper_lr
        self._gduo_hypergrad_clip = hypergrad_clip
        self._lr_min = lr_init * lr_min_ratio
        self._lr_max = lr_init * lr_max_ratio
        self._hypgrad_lr = float("nan")
        self._hypgrad_lr_unclipped = float("nan")
        self._gduo_alignment = float("nan")

    def _learned_base_lr(self) -> float:
        return math.exp(self.lr_raw.item())

    def _actual_lr(self) -> float:
        return self._learned_base_lr() * self._lr_scale

    def _clamp_lr_raw(self):
        min_raw = math.log(self._lr_min)
        max_raw = math.log(self._lr_max)
        raw = self.lr_raw.item()
        raw = min(max(raw, min_raw), max_raw)
        self.lr_raw.fill_(raw)

    def _update_lr_from_previous(self, params, prev_directions):
        if not self._gduo_learn_lr:
            self._hypgrad_lr = float("nan")
            self._hypgrad_lr_unclipped = float("nan")
            self._gduo_alignment = float("nan")
            return

        dot = None
        for param, direction in zip(params, prev_directions):
            if param.grad is None or direction is None:
                continue
            term = (param.grad.detach().float() * direction.float()).sum()
            dot = term if dot is None else dot + term

        if dot is None or self._prev_actual_lr is None:
            self._hypgrad_lr = float("nan")
            self._hypgrad_lr_unclipped = float("nan")
            self._gduo_alignment = float("nan")
            return

        alignment = dot.item()
        # d loss / d raw_lr = d loss / d actual_lr * d actual_lr / d raw_lr.
        # d loss / d actual_lr = -alignment.
        hypgrad_raw = -alignment * self._prev_actual_lr
        clipped = max(
            -self._gduo_hypergrad_clip,
            min(self._gduo_hypergrad_clip, hypgrad_raw),
        )

        self.lr_raw.sub_(self._gduo_hyper_lr * clipped)
        self._clamp_lr_raw()
        self._hypgrad_lr = clipped
        self._hypgrad_lr_unclipped = hypgrad_raw
        self._gduo_alignment = alignment


class _GDUOMomentumState:
    def _init_gduo_momentum(
        self,
        momentum_init: float,
        learn_momentum: bool = False,
        hyper_lr: float = 1e-3,
        hypergrad_clip: float = 1.0,
        mu_min: float = 0.0,
        mu_max: float = 0.99,
    ):
        if not mu_min <= momentum_init <= mu_max:
            raise ValueError("momentum_init must be inside [mu_min, mu_max]")

        self._gduo_learn_momentum = learn_momentum
        self._mu_min = mu_min
        self._mu_max = mu_max
        scaled = (momentum_init - mu_min) / max(1e-12, mu_max - mu_min)
        self.mu_raw = torch.tensor(_logit(scaled), dtype=torch.float64)
        self._gduo_mu_hyper_lr = hyper_lr
        self._gduo_mu_hypergrad_clip = hypergrad_clip
        self._hypgrad_mu = float("nan")
        self._hypgrad_mu_unclipped = float("nan")

    def _learned_mu(self) -> float:
        scaled = _sigmoid(self.mu_raw.item())
        return self._mu_min + (self._mu_max - self._mu_min) * scaled

    def _dmu_draw(self) -> float:
        scaled = _sigmoid(self.mu_raw.item())
        return (self._mu_max - self._mu_min) * scaled * (1.0 - scaled)

    def _update_momentum_from_previous(self, params, prev_direction_derivs):
        if not self._gduo_learn_momentum:
            self._hypgrad_mu = float("nan")
            self._hypgrad_mu_unclipped = float("nan")
            return

        dot = None
        for param, deriv in zip(params, prev_direction_derivs):
            if param.grad is None or deriv is None:
                continue
            term = (param.grad.detach().float() * deriv.float()).sum()
            dot = term if dot is None else dot + term

        if dot is None or self._prev_actual_lr is None:
            self._hypgrad_mu = float("nan")
            self._hypgrad_mu_unclipped = float("nan")
            return

        hypgrad_raw = -self._prev_actual_lr * dot.item()
        clipped = _clip_scalar(hypgrad_raw, self._gduo_mu_hypergrad_clip)
        self.mu_raw.sub_(self._gduo_mu_hyper_lr * clipped)
        self._hypgrad_mu = clipped
        self._hypgrad_mu_unclipped = hypgrad_raw


class _GDUORidgeState:
    def _init_gduo_ridge(
        self,
        ridge_init: float,
        learn_ridge: bool = False,
        hyper_lr: float = 1e-3,
        hypergrad_clip: float = 1.0,
        ridge_min_ratio: float = 0.05,
        ridge_max_ratio: float = 20.0,
    ):
        if ridge_init <= 0:
            raise ValueError("ridge_init must be positive")

        self._gduo_learn_ridge = learn_ridge
        self.ridge_raw = torch.tensor(math.log(ridge_init), dtype=torch.float64)
        self._ridge_min = ridge_init * ridge_min_ratio
        self._ridge_max = ridge_init * ridge_max_ratio
        self._gduo_ridge_hyper_lr = hyper_lr
        self._gduo_ridge_hypergrad_clip = hypergrad_clip
        self._hypgrad_ridge = float("nan")
        self._hypgrad_ridge_unclipped = float("nan")

    def _learned_ridge(self) -> float:
        return math.exp(self.ridge_raw.item())

    def _clamp_ridge_raw(self):
        raw = self.ridge_raw.item()
        raw = min(max(raw, math.log(self._ridge_min)), math.log(self._ridge_max))
        self.ridge_raw.fill_(raw)

    def _update_ridge_from_previous(self, params, prev_direction_derivs):
        if not self._gduo_learn_ridge:
            self._hypgrad_ridge = float("nan")
            self._hypgrad_ridge_unclipped = float("nan")
            return

        dot = None
        for param, deriv in zip(params, prev_direction_derivs):
            if param.grad is None or deriv is None:
                continue
            term = (param.grad.detach().float() * deriv.float()).sum()
            dot = term if dot is None else dot + term

        if dot is None or self._prev_actual_lr is None:
            self._hypgrad_ridge = float("nan")
            self._hypgrad_ridge_unclipped = float("nan")
            return

        hypgrad_raw = -self._prev_actual_lr * dot.item()
        clipped = _clip_scalar(hypgrad_raw, self._gduo_ridge_hypergrad_clip)
        self.ridge_raw.sub_(self._gduo_ridge_hyper_lr * clipped)
        self._clamp_ridge_raw()
        self._hypgrad_ridge = clipped
        self._hypgrad_ridge_unclipped = hypgrad_raw


class GDUOAdamWOptimizer(_GDUOLRState):
    def __init__(
        self,
        model: nn.Module,
        lr_init: float = 8e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
        eps: float = 1e-8,
        hyper_lr: float = 1e-3,
        hypergrad_clip: float = 1.0,
        lr_min_ratio: float = 0.05,
        lr_max_ratio: float = 5.0,
    ):
        self._params = list(model.parameters())
        self._beta1, self._beta2 = betas
        self._wd = weight_decay
        self._eps = eps
        self._t = 0
        self._m = [torch.zeros_like(param) for param in self._params]
        self._v = [torch.zeros_like(param) for param in self._params]
        self._prev_directions = [None for _ in self._params]
        self._update_rms = float("nan")
        self._init_gduo_lr(
            lr_init,
            hyper_lr=hyper_lr,
            hypergrad_clip=hypergrad_clip,
            lr_min_ratio=lr_min_ratio,
            lr_max_ratio=lr_max_ratio,
        )

    def zero_grad(self):
        for param in self._params:
            param.grad = None

    def set_lr_scale(self, scale: float):
        self._lr_scale = scale

    def step(self, model: nn.Module = None):
        self._update_lr_from_previous(self._params, self._prev_directions)
        lr = self._actual_lr()
        self._t += 1
        rms_vals = []

        beta1, beta2 = self._beta1, self._beta2
        for index, param in enumerate(self._params):
            if param.grad is None:
                continue

            grad = param.grad.detach()
            m_new = beta1 * self._m[index] + (1.0 - beta1) * grad
            v_new = beta2 * self._v[index] + (1.0 - beta2) * grad * grad
            m_hat = m_new / (1.0 - beta1 ** self._t)
            v_hat = v_new / (1.0 - beta2 ** self._t)
            direction = m_hat / (v_hat.sqrt() + self._eps)
            direction = direction + self._wd * param.detach()

            param.data.sub_(lr * direction)
            self._m[index] = m_new.detach()
            self._v[index] = v_new.detach()
            self._prev_directions[index] = direction.detach()
            rms_vals.append(direction.detach().float().pow(2).mean().item())

        if rms_vals:
            self._update_rms = math.sqrt(sum(rms_vals) / len(rms_vals))
        self._prev_actual_lr = lr

    def get_metrics(self) -> dict:
        return {
            "lr": self._actual_lr(),
            "base_lr": self._learned_base_lr(),
            "lr_scale": self._lr_scale,
            "mu": float("nan"),
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "hypgrad_lr": self._hypgrad_lr,
            "hypgrad_lr_unclipped": self._hypgrad_lr_unclipped,
            "gduo_alignment": self._gduo_alignment,
            "update_rms": self._update_rms,
        }


class GDUOMuonOptimizer(_GDUOLRState, _GDUOMomentumState):
    def __init__(
        self,
        model: nn.Module,
        matrix_param_names: Optional[set[str]] = None,
        lr_init: float = 0.16,
        momentum: float = 0.8,
        learn_lr: bool = True,
        learn_momentum: bool = False,
        weight_decay: float = 1e-3,
        ns_a: float = 3.4445,
        ns_b: float = -4.7750,
        ns_c: float = 2.0315,
        ns_iters: int = 5,
        adamw_lr: float = 1.6e-3,
        adamw_wd: float = 1e-2,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        hyper_lr: float = 1e-3,
        hypergrad_clip: float = 1.0,
        momentum_hyper_lr: Optional[float] = None,
        momentum_hypergrad_clip: Optional[float] = None,
        lr_min_ratio: float = 0.05,
        lr_max_ratio: float = 5.0,
        mu_min: float = 0.0,
        mu_max: float = 0.99,
    ):
        self._mu = momentum
        self._wd = weight_decay
        self._a = ns_a
        self._b = ns_b
        self._c = ns_c
        self._ns_iters = ns_iters
        self._base_adamw_lr = adamw_lr

        muon_params, adamw_params = [], []
        for name, param in model.named_parameters():
            use_muon = param.dim() >= 2
            if matrix_param_names is not None:
                use_muon = name in matrix_param_names
            if use_muon:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        self._muon_params = muon_params
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
            for param in muon_params
        ]
        self._prev_directions = [None for _ in muon_params]
        self._prev_direction_mu_derivs = [None for _ in muon_params]
        self._update_rms = float("nan")
        self._init_gduo_lr(
            lr_init,
            learn_lr=learn_lr,
            hyper_lr=hyper_lr,
            hypergrad_clip=hypergrad_clip,
            lr_min_ratio=lr_min_ratio,
            lr_max_ratio=lr_max_ratio,
        )
        self._init_gduo_momentum(
            momentum,
            learn_momentum=learn_momentum,
            hyper_lr=momentum_hyper_lr if momentum_hyper_lr is not None else hyper_lr,
            hypergrad_clip=(
                momentum_hypergrad_clip
                if momentum_hypergrad_clip is not None
                else hypergrad_clip
            ),
            mu_min=mu_min,
            mu_max=mu_max,
        )

    def zero_grad(self):
        for param in self._muon_params:
            param.grad = None
        if self._adamw_optim is not None:
            self._adamw_optim.zero_grad()

    def set_lr_scale(self, scale: float):
        self._lr_scale = scale
        if self._adamw_optim is not None:
            for group in self._adamw_optim.param_groups:
                group["lr"] = self._base_adamw_lr * scale

    def _direction_from_momentum(
        self,
        momentum: torch.Tensor,
        param: torch.Tensor,
    ) -> torch.Tensor:
        shape_orig = param.shape
        momentum_2d = (
            momentum.reshape(shape_orig[0], -1) if param.dim() > 2 else momentum
        )

        norm = momentum_2d.norm(p="fro") + 1e-8
        X = momentum_2d / norm
        X_orth = newton_schulz(X, self._a, self._b, self._c, self._ns_iters)
        rows, cols = momentum_2d.shape
        update_2d = 0.2 * X_orth * math.sqrt(max(rows, cols))
        if param.dim() > 2:
            return update_2d.to(param.dtype).reshape(shape_orig)
        return update_2d.to(param.dtype)

    def step(self, model: nn.Module = None):
        self._update_lr_from_previous(self._muon_params, self._prev_directions)
        self._update_momentum_from_previous(
            self._muon_params,
            self._prev_direction_mu_derivs,
        )

        self._mu = self._learned_mu()
        lr = self._actual_lr()
        rms_vals = []
        dmu_draw = self._dmu_draw()

        for index, param in enumerate(self._muon_params):
            if param.grad is None:
                continue

            grad = param.grad.detach()
            prev_momentum = self._M[index]
            momentum = self._mu * self._M[index] + grad
            update = self._direction_from_momentum(momentum, param)
            direction = update + self._wd * param.detach()

            param.data.sub_(lr * direction)
            self._M[index] = momentum.detach()
            self._prev_directions[index] = direction.detach()
            if self._gduo_learn_momentum:
                dmomentum_mu = dmu_draw * prev_momentum
                self._prev_direction_mu_derivs[index] = _finite_difference_direction(
                    lambda mom, p=param: self._direction_from_momentum(mom, p),
                    momentum.detach(),
                    dmomentum_mu.detach(),
                )
            else:
                self._prev_direction_mu_derivs[index] = None
            rms_vals.append(update.float().pow(2).mean().item())

        if rms_vals:
            self._update_rms = math.sqrt(sum(rms_vals) / len(rms_vals))

        if self._adamw_optim is not None:
            self._adamw_optim.step()
        self._prev_actual_lr = lr

    def get_metrics(self) -> dict:
        return {
            "lr": self._actual_lr(),
            "base_lr": self._learned_base_lr(),
            "lr_scale": self._lr_scale,
            "mu": self._mu,
            "a": self._a,
            "b": self._b,
            "c": self._c,
            "hypgrad_lr": self._hypgrad_lr,
            "hypgrad_lr_unclipped": self._hypgrad_lr_unclipped,
            "hypgrad_mu": self._hypgrad_mu,
            "hypgrad_mu_unclipped": self._hypgrad_mu_unclipped,
            "gduo_alignment": self._gduo_alignment,
            "update_rms": self._update_rms,
        }


class GDUONewtonMuonOptimizer(
    NewtonMuonOptimizer,
    _GDUOLRState,
    _GDUOMomentumState,
    _GDUORidgeState,
):
    def __init__(
        self,
        model: nn.Module,
        matrix_param_names: Optional[set[str]] = None,
        lr_init: float = 0.16,
        momentum: float = 0.75,
        learn_lr: bool = True,
        learn_momentum: bool = False,
        learn_ridge: bool = False,
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
        hyper_lr: float = 1e-3,
        hypergrad_clip: float = 1.0,
        momentum_hyper_lr: Optional[float] = None,
        momentum_hypergrad_clip: Optional[float] = None,
        ridge_hyper_lr: Optional[float] = None,
        ridge_hypergrad_clip: Optional[float] = None,
        lr_min_ratio: float = 0.05,
        lr_max_ratio: float = 5.0,
        mu_min: float = 0.0,
        mu_max: float = 0.99,
        ridge_min_ratio: float = 0.05,
        ridge_max_ratio: float = 20.0,
    ):
        super().__init__(
            model,
            matrix_param_names=matrix_param_names,
            lr=lr_init,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_a=ns_a,
            ns_b=ns_b,
            ns_c=ns_c,
            ns_iters=ns_iters,
            adamw_lr=adamw_lr,
            adamw_wd=adamw_wd,
            adamw_betas=adamw_betas,
            ewma_beta=ewma_beta,
            ridge=ridge,
            refresh_interval=refresh_interval,
            second_moment_init=second_moment_init,
        )
        self._prev_directions = [None for _ in self._matrix_params]
        self._prev_direction_mu_derivs = [None for _ in self._matrix_params]
        self._prev_direction_ridge_derivs = [None for _ in self._matrix_params]
        self._K_inv_ridge_deriv: list[Optional[torch.Tensor]] = [
            None for _ in self._matrix_params
        ]
        self._init_gduo_lr(
            lr_init,
            learn_lr=learn_lr,
            hyper_lr=hyper_lr,
            hypergrad_clip=hypergrad_clip,
            lr_min_ratio=lr_min_ratio,
            lr_max_ratio=lr_max_ratio,
        )
        self._init_gduo_momentum(
            momentum,
            learn_momentum=learn_momentum,
            hyper_lr=momentum_hyper_lr if momentum_hyper_lr is not None else hyper_lr,
            hypergrad_clip=(
                momentum_hypergrad_clip
                if momentum_hypergrad_clip is not None
                else hypergrad_clip
            ),
            mu_min=mu_min,
            mu_max=mu_max,
        )
        self._init_gduo_ridge(
            ridge,
            learn_ridge=learn_ridge,
            hyper_lr=ridge_hyper_lr if ridge_hyper_lr is not None else hyper_lr,
            hypergrad_clip=(
                ridge_hypergrad_clip
                if ridge_hypergrad_clip is not None
                else hypergrad_clip
            ),
            ridge_min_ratio=ridge_min_ratio,
            ridge_max_ratio=ridge_max_ratio,
        )

    def set_lr_scale(self, scale: float):
        self._lr_scale = scale
        if self._adamw_optim is not None:
            for group in self._adamw_optim.param_groups:
                group["lr"] = self._base_adamw_lr * scale

    def _maybe_refresh_preconditioner(self, index: int, param: torch.Tensor):
        input_activations = self._input_cache.get(id(param))
        input_dim = param.reshape(param.shape[0], -1).shape[1]
        device = param.device

        if self._K_inv[index] is None:
            self._K_inv[index] = self._identity(input_dim, device)
            self._K_inv_ridge_deriv[index] = None

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

        self._ridge = self._learned_ridge()
        mean_trace = torch.trace(self._K[index]) / input_dim
        damping = (self._ridge * mean_trace).clamp_min(1e-12)
        damped = self._K[index] + damping * self._identity(input_dim, device)
        self._K_inv[index] = self._safe_inverse(damped)

        if self._gduo_learn_ridge:
            K_inv = self._K_inv[index].float()
            dK_inv_draw = -mean_trace * self._ridge * (K_inv @ K_inv)
            self._K_inv_ridge_deriv[index] = dK_inv_draw.detach()
        else:
            self._K_inv_ridge_deriv[index] = None

        self._refresh_count += 1

    def _direction_from_momentum(
        self,
        momentum: torch.Tensor,
        param: torch.Tensor,
    ) -> torch.Tensor:
        shape_orig = param.shape
        momentum_2d = (
            momentum.reshape(shape_orig[0], -1) if param.dim() > 2 else momentum
        )
        norm = momentum_2d.norm(p="fro") + 1e-8
        X = momentum_2d / norm
        X_orth = newton_schulz(X, self._a, self._b, self._c, self._ns_iters)

        rows, cols = momentum_2d.shape
        update_2d = 0.2 * X_orth * math.sqrt(max(rows, cols))
        if param.dim() > 2:
            return update_2d.to(param.dtype).reshape(shape_orig)
        return update_2d.to(param.dtype)

    def _ridge_momentum_derivative(
        self,
        index: int,
        param: torch.Tensor,
        grad: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        dK_inv = self._K_inv_ridge_deriv[index]
        if dK_inv is None:
            return None

        shape_orig = grad.shape
        grad_2d = grad.reshape(shape_orig[0], -1) if grad.dim() > 2 else grad
        if dK_inv.shape[0] != grad_2d.shape[1]:
            return None

        dpreconditioned = grad_2d.float() @ dK_inv.to(device=grad_2d.device)
        dpreconditioned = dpreconditioned.to(dtype=grad.dtype)
        if grad.dim() > 2:
            return dpreconditioned.reshape(shape_orig)
        return dpreconditioned

    def step(self, model: nn.Module = None):
        self._update_lr_from_previous(self._matrix_params, self._prev_directions)
        self._update_momentum_from_previous(
            self._matrix_params,
            self._prev_direction_mu_derivs,
        )
        self._update_ridge_from_previous(
            self._matrix_params,
            self._prev_direction_ridge_derivs,
        )

        self._mu = self._learned_mu()
        self._ridge = self._learned_ridge()
        lr = self._actual_lr()
        rms_vals = []
        dmu_draw = self._dmu_draw()

        for index, param in enumerate(self._matrix_params):
            if param.grad is None:
                continue

            grad = self._precondition_gradient(index, param, param.grad.detach())
            prev_momentum = self._M[index]
            momentum = self._mu * self._M[index] + grad

            update = self._direction_from_momentum(momentum, param)
            direction = update + self._wd * param.detach()

            param.data.sub_(lr * direction)
            self._M[index] = momentum.detach()
            self._prev_directions[index] = direction.detach()
            if self._gduo_learn_momentum:
                dmomentum_mu = dmu_draw * prev_momentum
                self._prev_direction_mu_derivs[index] = _finite_difference_direction(
                    lambda mom, p=param: self._direction_from_momentum(mom, p),
                    momentum.detach(),
                    dmomentum_mu.detach(),
                )
            else:
                self._prev_direction_mu_derivs[index] = None

            if self._gduo_learn_ridge:
                dmomentum_ridge = self._ridge_momentum_derivative(
                    index,
                    param,
                    param.grad.detach(),
                )
                self._prev_direction_ridge_derivs[index] = _finite_difference_direction(
                    lambda mom, p=param: self._direction_from_momentum(mom, p),
                    momentum.detach(),
                    dmomentum_ridge.detach() if dmomentum_ridge is not None else None,
                )
            else:
                self._prev_direction_ridge_derivs[index] = None
            rms_vals.append(update.float().pow(2).mean().item())

        if rms_vals:
            self._update_rms = math.sqrt(sum(rms_vals) / len(rms_vals))

        if self._adamw_optim is not None:
            self._adamw_optim.step()

        self._step_count += 1
        self._prev_actual_lr = lr

    def get_metrics(self) -> dict:
        return {
            "lr": self._actual_lr(),
            "base_lr": self._learned_base_lr(),
            "lr_scale": self._lr_scale,
            "mu": self._mu,
            "a": self._a,
            "b": self._b,
            "c": self._c,
            "hypgrad_lr": self._hypgrad_lr,
            "hypgrad_lr_unclipped": self._hypgrad_lr_unclipped,
            "hypgrad_mu": self._hypgrad_mu,
            "hypgrad_mu_unclipped": self._hypgrad_mu_unclipped,
            "hypgrad_ridge": self._hypgrad_ridge,
            "hypgrad_ridge_unclipped": self._hypgrad_ridge_unclipped,
            "gduo_alignment": self._gduo_alignment,
            "update_rms": self._update_rms,
            "ewma_beta": self._ewma_beta,
            "ridge": self._ridge,
            "refresh_interval": self._refresh_interval,
            "precond_refreshes": self._refresh_count,
        }
