import os

import torch
import torch.distributed as dist
import torch.nn as nn

from .muon import zeropower_via_newtonschulz5


class NewtonMuon(torch.optim.Optimizer):
    """
    Newton-Muon for GPT-style linear layers.

    This follows the local Muon implementation used by this repo, with an
    additional right-preconditioner based on an EWMA of linear-layer input
    second moments. Non-matrix parameters use the same AdamW fallback as Muon.
    """

    def __init__(
        self,
        muon_params,
        model=None,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
        ewma_beta=0.95,
        ridge=0.2,
        refresh_interval=32,
        second_moment_init=1.0,
        max_precond_dim=1024,
        block_size=0,
        precond_clip=0.0,
        precond_log_interval=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        self.ewma_beta = ewma_beta
        self.ridge = ridge
        self.refresh_interval = max(1, refresh_interval)
        self.second_moment_init = second_moment_init
        self.max_precond_dim = max_precond_dim
        self.block_size = block_size
        self.precond_clip = precond_clip
        self.precond_log_interval = precond_log_interval
        self._precond_stats = dict(count=0, clipped=0, max_ratio=0.0)
        self._step = 0
        self._hook_handles = []

        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
            self.state[p]["use_newton_muon"] = False
        for p in adamw_params:
            self.state[p]["use_muon"] = False
            self.state[p]["use_newton_muon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

        if model is not None:
            self.register_model_hooks(model)

    def register_model_hooks(self, model):
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            p = module.weight
            if p not in self.state or not self.state[p].get("use_muon", False):
                continue
            self.state[p]["use_newton_muon"] = True
            self._hook_handles.append(module.register_forward_pre_hook(self._make_hook(p)))

    def _make_hook(self, param):
        def hook(module, inputs):
            if not module.training:
                return
            if not inputs or not torch.is_tensor(inputs[0]):
                return

            x = inputs[0].detach()
            if x.shape[-1] != param.shape[1]:
                return

            flat = x.reshape(-1, x.shape[-1]).float()
            if flat.numel() == 0:
                return

            mode, block = self._preconditioner_mode(flat.shape[1])
            if mode == "skip":
                return

            state = self.state[param]
            if mode == "full":
                cov_sum = flat.T @ flat
            else:
                n_blocks = flat.shape[1] // block
                flat_blocks = flat.reshape(flat.shape[0], n_blocks, block)
                cov_sum = torch.einsum("nbi,nbj->bij", flat_blocks, flat_blocks)

            if "pending_cov_sum" in state:
                state["pending_cov_sum"].add_(cov_sum)
                state["pending_cov_count"] += flat.shape[0]
            else:
                state["pending_cov_sum"] = cov_sum
                state["pending_cov_count"] = flat.shape[0]
                state["preconditioner_mode"] = mode
                state["preconditioner_block"] = block

        return hook

    def _preconditioner_mode(self, input_dim):
        if input_dim <= self.max_precond_dim:
            return "full", input_dim
        if (
            self.block_size > 0
            and self.block_size <= self.max_precond_dim
            and input_dim % self.block_size == 0
        ):
            return "block", self.block_size
        return "skip", input_dim

    @torch.no_grad()
    def step(self):
        self._step += 1

        for group in self.param_groups:
            ############################
            #       Newton-Muon        #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            momentum = group["momentum"]

            total_params = sum(p.numel() for p in params)
            if total_params > 0:
                device = params[0].device
                updates_flat = torch.zeros(
                    total_params, device=device, dtype=torch.bfloat16
                )
            else:
                updates_flat = None

            curr_idx = 0
            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    g = p.grad
                    assert g is not None
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    g = self._maybe_precondition(g, state)

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)

                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            if self.world_size > 1 and updates_flat is not None:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in params:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            self._maybe_log_preconditioner_stats()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["adamw_lr_ratio"] * group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

    def _maybe_precondition(self, grad, state):
        if not state.get("use_newton_muon", False):
            return grad

        cov = self._consume_covariance(state)
        if cov is not None:
            if "input_cov" not in state:
                eye = self._identity_like_cov(cov)
                state["input_cov"] = cov.add(eye, alpha=self.second_moment_init)
            else:
                state["input_cov"].lerp_(cov, 1 - self.ewma_beta)

            if (
                "input_cov_inv" not in state
                or self._step % self.refresh_interval == 0
            ):
                state["input_cov_inv"] = self._invert_covariance(state["input_cov"])

        inv = state.get("input_cov_inv", None)
        if inv is None:
            return grad

        mode = state.get("preconditioner_mode", "full")
        raw_grad = grad.float()
        if mode == "full":
            precond_grad = raw_grad @ inv.to(device=grad.device)
            return self._clip_preconditioned_grad(raw_grad, precond_grad)

        block = state["preconditioner_block"]
        if grad.shape[1] % block != 0:
            return grad
        n_blocks = grad.shape[1] // block
        grad_blocks = raw_grad.reshape(grad.shape[0], n_blocks, block)
        precond_grad = torch.einsum(
            "obi,bij->obj", grad_blocks, inv.to(device=grad.device)
        ).reshape_as(grad)
        return self._clip_preconditioned_grad(raw_grad, precond_grad)

    def _clip_preconditioned_grad(self, raw_grad, precond_grad):
        if not torch.isfinite(precond_grad).all():
            return raw_grad

        raw_norm = raw_grad.norm().clamp_min(1e-12)
        precond_norm = precond_grad.norm()
        ratio = (precond_norm / raw_norm).detach()
        ratio_float = float(ratio.clamp_max(1e6).cpu())
        self._precond_stats["count"] += 1
        self._precond_stats["max_ratio"] = max(
            self._precond_stats["max_ratio"], ratio_float
        )

        if self.precond_clip > 0 and ratio > self.precond_clip:
            self._precond_stats["clipped"] += 1
            precond_grad = precond_grad * (self.precond_clip / ratio)
        return precond_grad

    def _maybe_log_preconditioner_stats(self):
        if self.precond_log_interval <= 0:
            return
        if self.rank != 0 or self._step % self.precond_log_interval != 0:
            return
        stats = self._precond_stats
        if stats["count"] == 0:
            return
        print(
            "[NewtonMuon] "
            f"step={self._step} precond_count={stats['count']} "
            f"precond_clipped={stats['clipped']} "
            f"precond_max_ratio={stats['max_ratio']:.3e}"
        )
        self._precond_stats = dict(count=0, clipped=0, max_ratio=0.0)

    def _consume_covariance(self, state):
        cov_sum = state.pop("pending_cov_sum", None)
        cov_count = state.pop("pending_cov_count", None)
        if cov_sum is None or cov_count is None or cov_count == 0:
            return None
        return cov_sum / cov_count

    def _identity_like_cov(self, cov):
        if cov.ndim == 2:
            return torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        return eye.unsqueeze(0).expand(cov.shape[0], -1, -1)

    def _invert_covariance(self, cov):
        cov = torch.nan_to_num(cov.float(), nan=0.0, posinf=0.0, neginf=0.0)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        eye = self._identity_like_cov(cov)

        if cov.ndim == 2:
            ridge = self.ridge * cov.diag().mean().clamp_min(1e-8)
        else:
            ridge = self.ridge * cov.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)
            ridge = ridge.view(-1, 1, 1)

        for multiplier in (1.0, 10.0, 100.0, 1000.0):
            matrix = cov + (ridge * multiplier) * eye
            chol, info = torch.linalg.cholesky_ex(matrix)
            if not torch.any(info != 0):
                return torch.cholesky_inverse(chol)

        denom = ridge.clamp_min(1e-6)
        return eye / denom
