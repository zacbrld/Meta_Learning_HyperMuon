"""
HyperMuon — Muon with hyperparameters tuned via automatic differentiation.

Three levels:
  L1: learns η  (lr)
  L2: learns η, µ
  L3: learns η, µ, a, b, c  (Newton-Schulz coefficients)

Hypergradient via first-order proxy approximation:
  proxy = -Σ_W ⟨g_W, update_W(θ)⟩
  proxy.backward() → gradients for all learnable hyperparameters
  θ.data -= κ_θ * θ.grad

Log-parameterization for η > 0: lr = exp(lr_raw)
Sigmoid parameterization for µ ∈ (0,1): mu = sigmoid(mu_raw)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Newton-Schulz (differentiable version used for L3)
# ---------------------------------------------------------------------------

def _ns_differentiable(X: torch.Tensor, a, b, c, n_iters: int = 5):
    """
    Newton-Schulz iterations keeping a, b, c in the computation graph.
    X must already be float32 and is modified in-place conceptually.
    """
    for _ in range(n_iters):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * ((A @ A) @ X)
    return X


def _ns_fixed(X: torch.Tensor, a: float, b: float, c: float, n_iters: int = 5):
    """Newton-Schulz with scalar coefficients (L1/L2 — a,b,c not in graph)."""
    X = X.float()
    for _ in range(n_iters):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * ((A @ A) @ X)
    return X


# ---------------------------------------------------------------------------
# Core Muon update (used in proxy computation)
# ---------------------------------------------------------------------------

def _muon_update_2d(
    M_2d: torch.Tensor,
    a, b, c,
    n_iters: int,
    differentiable: bool,
) -> torch.Tensor:
    """
    Given the 2D momentum M_2d [m, n]:
      1. Normalise by Frobenius norm
      2. Newton-Schulz → orthogonal update
      3. Scale by 0.2 * sqrt(max(m, n))
    Returns O with same shape as M_2d.
    """
    norm = M_2d.norm(p="fro") + 1e-8
    X = M_2d / norm

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T  # make it wide (m < n) for A = X@X.T to be smaller

    if differentiable:
        X_f = X.float()  # cast to float32; keeps graph intact
        X_f = _ns_differentiable(X_f, a, b, c, n_iters)
        # Cast back preserving grad
        X_out = X_f.to(dtype=M_2d.dtype if not M_2d.is_floating_point() else M_2d.dtype)
        # Stay float32 for the scaling — dtype_orig will be handled by caller
        X_out = X_f  # float32
    else:
        X_out = _ns_fixed(X, float(a), float(b), float(c), n_iters)  # float32

    if transposed:
        X_out = X_out.T

    m_dim, n_dim = M_2d.shape
    O = 0.2 * X_out * math.sqrt(max(m_dim, n_dim))
    return O  # float32


# ---------------------------------------------------------------------------
# HyperMuon
# ---------------------------------------------------------------------------

class HyperMuonOptimizer:
    """
    Muon with AD-tuned hyperparameters.

    Parameters
    ----------
    model        : nn.Module
    level        : 1, 2, or 3 (number of hyperparameters learned)
    lr_init      : initial η  (default 1e-3)
    mu_init      : initial µ  (default 0.95)
    ns_a/b/c     : initial N-S coefficients
    ns_iters     : number of N-S iterations  (default 5)
    lambda_muon  : weight decay for Muon params  (default 0.1)
    kappa_lr     : meta-lr for lr_raw  (default 1e-5)
    kappa_mu     : meta-lr for mu_raw  (default 1e-6)
    kappa_abc    : meta-lr for a, b, c  (default 1e-7)
    adamw_lr     : lr for non-matrix AdamW params
    adamw_wd     : weight decay for non-matrix AdamW params
    """

    def __init__(
        self,
        model: nn.Module,
        level: int = 1,
        lr_init: float = 1e-3,
        mu_init: float = 0.95,
        ns_a: float = 3.4445,
        ns_b: float = -4.7750,
        ns_c: float = 2.0315,
        ns_iters: int = 5,
        lambda_muon: float = 0.1,
        kappa_lr: float = 1e-5,
        kappa_mu: float = 1e-6,
        kappa_abc: float = 1e-7,
        adamw_lr: float = 1e-3,
        adamw_wd: float = 0.01,
    ):
        assert level in (1, 2, 3), "level must be 1, 2 or 3"
        self._level = level
        self._ns_iters = ns_iters
        self._lambda = lambda_muon

        # ── Learnable hyperparameters ──────────────────────────────────────
        self.lr_raw = nn.Parameter(torch.tensor(math.log(lr_init)))
        self._kappa_lr = kappa_lr

        if level >= 2:
            # sigmoid(mu_raw) ≈ mu_init
            mu_raw_init = math.log(mu_init / (1 - mu_init))
            self.mu_raw = nn.Parameter(torch.tensor(mu_raw_init))
            self._kappa_mu = kappa_mu
        else:
            self._mu_fixed = mu_init

        if level >= 3:
            self.a = nn.Parameter(torch.tensor(ns_a))
            self.b = nn.Parameter(torch.tensor(ns_b))
            self.c = nn.Parameter(torch.tensor(ns_c))
            self._kappa_abc = kappa_abc
        else:
            self._a_fixed = ns_a
            self._b_fixed = ns_b
            self._c_fixed = ns_c

        # ── Model parameter split ──────────────────────────────────────────
        muon_params, adamw_params = [], []
        self._muon_names = []
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                muon_params.append(p)
                self._muon_names.append(name)
            else:
                adamw_params.append(p)

        self._muon_params = muon_params
        self._adamw_optim = torch.optim.AdamW(
            adamw_params, lr=adamw_lr, weight_decay=adamw_wd
        ) if adamw_params else None

        # Momentum buffers for Muon params (detached)
        self._M = [torch.zeros_like(p) for p in muon_params]

        self._update_rms = float("nan")
        self._hypgrad_lr = float("nan")
        self._hypgrad_mu = float("nan")
        self._hypgrad_abc = float("nan")

    # ------------------------------------------------------------------

    def _get_hyperparams(self):
        """Return (lr, mu, a, b, c) — some may be Tensors in graph, some floats."""
        lr = torch.exp(self.lr_raw)

        if self._level >= 2:
            mu = torch.sigmoid(self.mu_raw)
        else:
            mu = self._mu_fixed  # plain float

        if self._level >= 3:
            a, b, c = self.a, self.b, self.c
        else:
            a = self._a_fixed
            b = self._b_fixed
            c = self._c_fixed

        return lr, mu, a, b, c

    # ------------------------------------------------------------------

    def zero_grad(self):
        for p in self._muon_params:
            p.grad = None
        if self._adamw_optim is not None:
            self._adamw_optim.zero_grad()
        # Clear hyperparameter gradients
        for hp in self._hyperparams_list():
            if hasattr(hp, "grad"):
                hp.grad = None

    def _hyperparams_list(self):
        hps = [self.lr_raw]
        if self._level >= 2:
            hps.append(self.mu_raw)
        if self._level >= 3:
            hps += [self.a, self.b, self.c]
        return hps

    # ------------------------------------------------------------------

    def step(self, model: nn.Module = None):
        """
        Call after loss.backward().
        1. Compute proxy loss from current gradients and hyperparams.
        2. Backprop proxy to get hypergradients.
        3. Clip + update hyperparams.
        4. Apply Muon update to Muon params.
        5. Step AdamW for non-matrix params.
        """
        lr, mu, a, b, c = self._get_hyperparams()
        differentiable = (self._level == 3)

        proxy_terms = []
        rms_vals = []

        for i, param in enumerate(self._muon_params):
            if param.grad is None:
                continue

            g = param.grad.detach()  # gradient always detached from model graph
            M_prev = self._M[i]      # detached from previous step

            # Step 1 — Nesterov momentum (mu may be in graph for L2/L3)
            if isinstance(mu, torch.Tensor):
                M = mu * M_prev + g
            else:
                M = mu * M_prev + g  # scalar mul, no graph issue

            # Step 2–4 — Newton-Schulz on 2D view
            shape_orig = param.shape
            if param.dim() > 2:
                M_2d = M.reshape(shape_orig[0], -1)
            else:
                M_2d = M

            O_2d = _muon_update_2d(M_2d, a, b, c, self._ns_iters, differentiable)

            if param.dim() > 2:
                O = O_2d.reshape(shape_orig)
            else:
                O = O_2d

            # Step 5 — update tensor (lr in graph)
            O_cast = O.to(dtype=param.dtype if torch.is_floating_point(param) else torch.float32)
            update = lr * (O_cast + self._lambda * param.detach().to(O_cast.dtype))

            # Proxy term: -⟨g, update⟩
            proxy_terms.append((g.to(update.dtype), update))
            # Track RMS of O (before lr scaling) — target ≈ 0.2
            rms_vals.append(O_cast.detach().float().pow(2).mean().item())

            # Save momentum (detached, using mu.detach() for buffer)
            mu_val = mu.detach().item() if isinstance(mu, torch.Tensor) else mu
            self._M[i] = (mu_val * M_prev + g).detach()

        # ── Proxy backward → hypergradients ───────────────────────────────
        if proxy_terms:
            proxy = -sum((g * upd).sum() for g, upd in proxy_terms)

            # Ensure hyperparams have require_grad so backward works
            proxy.backward()

            hps = self._hyperparams_list()
            torch.nn.utils.clip_grad_norm_(hps, max_norm=1.0)

            # Save norms before zeroing
            if self.lr_raw.grad is not None:
                self._hypgrad_lr = self.lr_raw.grad.abs().item()
            if self._level >= 2 and self.mu_raw.grad is not None:
                self._hypgrad_mu = self.mu_raw.grad.abs().item()
            if self._level >= 3:
                abc_norms = []
                for p in [self.a, self.b, self.c]:
                    if p.grad is not None:
                        abc_norms.append(p.grad.abs().item())
                if abc_norms:
                    self._hypgrad_abc = sum(abc_norms) / len(abc_norms)

            # Update hyperparams
            self.lr_raw.data -= self._kappa_lr * self.lr_raw.grad
            self.lr_raw.grad = None

            if self._level >= 2:
                self.mu_raw.data -= self._kappa_mu * self.mu_raw.grad
                self.mu_raw.grad = None

            if self._level >= 3:
                self.a.data -= self._kappa_abc * self.a.grad
                self.b.data -= self._kappa_abc * self.b.grad
                self.c.data -= self._kappa_abc * self.c.grad
                self.a.grad = None
                self.b.grad = None
                self.c.grad = None

        # ── Apply Muon update to model params ─────────────────────────────
        lr_val = math.exp(self.lr_raw.item())
        mu_val = torch.sigmoid(self.mu_raw).item() if self._level >= 2 else self._mu_fixed
        a_val = self.a.item() if self._level >= 3 else self._a_fixed
        b_val = self.b.item() if self._level >= 3 else self._b_fixed
        c_val = self.c.item() if self._level >= 3 else self._c_fixed

        for i, param in enumerate(self._muon_params):
            if param.grad is None:
                continue

            M = self._M[i]  # already updated above (detached)
            shape_orig = param.shape
            if param.dim() > 2:
                M_2d = M.reshape(shape_orig[0], -1)
            else:
                M_2d = M

            norm = M_2d.float().norm(p="fro") + 1e-8
            X = M_2d.float() / norm

            transposed = X.shape[0] > X.shape[1]
            if transposed:
                X = X.T
            X = _ns_fixed(X, a_val, b_val, c_val, self._ns_iters)
            if transposed:
                X = X.T

            m_dim, n_dim = M_2d.shape
            O_2d = 0.2 * X * math.sqrt(max(m_dim, n_dim))
            if param.dim() > 2:
                O = O_2d.reshape(shape_orig)
            else:
                O = O_2d

            O = O.to(param.dtype)
            param.data -= lr_val * (O + self._lambda * param.data)

        if rms_vals:
            self._update_rms = math.sqrt(sum(rms_vals) / len(rms_vals))

        if self._adamw_optim is not None:
            self._adamw_optim.step()

    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        lr_val = math.exp(self.lr_raw.item())
        mu_val = torch.sigmoid(self.mu_raw).item() if self._level >= 2 else self._mu_fixed
        a_val = self.a.item() if self._level >= 3 else self._a_fixed
        b_val = self.b.item() if self._level >= 3 else self._b_fixed
        c_val = self.c.item() if self._level >= 3 else self._c_fixed
        return {
            "lr": lr_val,
            "mu": mu_val,
            "a": a_val,
            "b": b_val,
            "c": c_val,
            "hypgrad_lr": self._hypgrad_lr,
            "hypgrad_mu": self._hypgrad_mu,
            "hypgrad_abc": self._hypgrad_abc,
            "update_rms": self._update_rms,
        }
