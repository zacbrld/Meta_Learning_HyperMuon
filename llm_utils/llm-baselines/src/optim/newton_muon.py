import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from .gduo_meta import GDUOMetaMixin
from .muon import zeropower_via_newtonschulz5


def _matrix_bucket(p):
    if p.ndim < 2:
        return "non_matrix"
    out_dim = p.shape[0]
    in_dim = int(p.numel() // max(1, out_dim))
    if out_dim >= 10000 or in_dim >= 10000:
        return "embed_head"
    ratio = out_dim / max(1, in_dim)
    inv_ratio = in_dim / max(1, out_dim)
    if abs(ratio - 3.0) < 0.25:
        return "attn_qkv"
    if abs(ratio - 4.0) < 0.25:
        return "mlp_fc"
    if abs(inv_ratio - 4.0) < 0.25:
        return "mlp_proj"
    if abs(ratio - 1.0) < 0.25:
        return "attn_proj"
    return "other_matrix"


def _new_precond_stats():
    return dict(count=0, clipped=0, ratio_sum=0.0, max_ratio=0.0, by_bucket={})


def _clip_scalar(value, limit):
    if limit is None or limit <= 0:
        return value
    return max(-limit, min(limit, value))


def _logit(x):
    x = min(max(x, 1e-12), 1.0 - 1e-12)
    return math.log(x / (1.0 - x))


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class NewtonMuon(GDUOMetaMixin, torch.optim.Optimizer):
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
        learn_precond_strength=False,
        precond_strength_init=1.0,
        precond_strength_min=0.0,
        precond_strength_max=1.0,
        precond_strength_hyper_lr=1e-3,
        precond_strength_hypergrad_clip=1.0,
        precond_strength_fd_eps=1e-3,
        gduo_learn_lr=False,
        gduo_learn_momentum=False,
        gduo_ema_beta=0.9,
        gduo_lr_hyper_lr=1e-3,
        gduo_momentum_hyper_lr=1e-3,
        gduo_hypergrad_clip=1.0,
        gduo_lr_min_ratio=0.25,
        gduo_lr_max_ratio=4.0,
        gduo_log_interval=0,
        gduo_scope="tensor",
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
        self.learn_precond_strength = learn_precond_strength
        self.precond_strength_min = precond_strength_min
        self.precond_strength_max = precond_strength_max
        self.precond_strength_hyper_lr = precond_strength_hyper_lr
        self.precond_strength_hypergrad_clip = precond_strength_hypergrad_clip
        self.precond_strength_fd_eps = precond_strength_fd_eps
        self._precond_stats = _new_precond_stats()
        self._step = 0
        self._hook_handles = []

        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
            self.state[p]["use_newton_muon"] = False
            self.state[p]["bucket"] = _matrix_bucket(p)
            self._init_precond_strength_state(p, precond_strength_init)
        for p in adamw_params:
            self.state[p]["use_muon"] = False
            self.state[p]["use_newton_muon"] = False
            self.state[p]["bucket"] = _matrix_bucket(p)
            self._init_precond_strength_state(p, precond_strength_init)

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

        if model is not None:
            self.register_model_hooks(model)

        self._init_gduo_meta(
            learn_lr=gduo_learn_lr,
            learn_momentum=gduo_learn_momentum,
            lr_hyper_lr=gduo_lr_hyper_lr,
            momentum_hyper_lr=gduo_momentum_hyper_lr,
            hypergrad_clip=gduo_hypergrad_clip,
            lr_min_ratio=gduo_lr_min_ratio,
            lr_max_ratio=gduo_lr_max_ratio,
            ema_beta=gduo_ema_beta,
            log_interval=gduo_log_interval,
            scope=gduo_scope,
        )

    def get_metrics(self) -> dict:
        group = self.param_groups[0]
        params_with_lr = [p for p in group["params"] if self._gduo_has_meta(p)]
        muon_params = [p for p in group["params"] if self.state[p].get("use_muon", False)]
        if not params_with_lr and not muon_params:
            return {}

        metrics = {}
        if params_with_lr:
            avg_lr_scale = sum(self._gduo_lr_scale(p) for p in params_with_lr) / len(params_with_lr)
            avg_actual_lr = sum(self._gduo_actual_lr(p, group) for p in params_with_lr) / len(params_with_lr)
            metrics.update(
                {
                    "gduo_lr_scale_avg": avg_lr_scale,
                    "gduo_actual_lr_avg": avg_actual_lr,
                }
            )
        if self.gduo_learn_momentum:
            avg_momentum = sum(self._gduo_momentum(p) for p in params_with_lr) / len(params_with_lr)
            metrics["gduo_momentum_avg"] = avg_momentum
        if self.learn_precond_strength and muon_params:
            strengths = [self._precond_strength(p) for p in muon_params]
            metrics["precond_strength_avg"] = sum(strengths) / len(strengths)
            metrics["precond_strength_min"] = min(strengths)
            metrics["precond_strength_max"] = max(strengths)
        return metrics

    def _init_precond_strength_state(self, p, init):
        if self.precond_strength_max <= self.precond_strength_min:
            scaled = 1.0
        else:
            scaled = (init - self.precond_strength_min) / (
                self.precond_strength_max - self.precond_strength_min
            )
        self.state[p]["precond_strength_raw"] = torch.tensor(
            _logit(scaled), dtype=torch.float64
        )
        self.state[p]["precond_strength_hypgrad"] = float("nan")
        self.state[p]["precond_strength_hypgrad_unclipped"] = float("nan")
        self.state[p]["precond_strength_alignment"] = float("nan")
        self.state[p]["precond_strength_prev_deriv"] = None
        self.state[p]["precond_strength_prev_lr"] = None

    def _precond_strength(self, p):
        if not self.learn_precond_strength:
            return 1.0
        scaled = _sigmoid(float(self.state[p]["precond_strength_raw"].item()))
        return self.precond_strength_min + (
            self.precond_strength_max - self.precond_strength_min
        ) * scaled

    def _precond_strength_draw(self, p):
        scaled = _sigmoid(float(self.state[p]["precond_strength_raw"].item()))
        return (
            self.precond_strength_max - self.precond_strength_min
        ) * scaled * (1.0 - scaled)

    def _update_precond_strength_from_previous(self, params):
        if not self.learn_precond_strength:
            return
        for p in params:
            if p.grad is None:
                continue
            state = self.state[p]
            prev_deriv = state.get("precond_strength_prev_deriv")
            prev_lr = state.get("precond_strength_prev_lr")
            if prev_deriv is None or prev_lr is None:
                state["precond_strength_hypgrad"] = float("nan")
                state["precond_strength_hypgrad_unclipped"] = float("nan")
                state["precond_strength_alignment"] = float("nan")
                continue
            grad = p.grad.detach().float()
            if grad.ndim > 2:
                grad = grad.view(grad.size(0), -1)
            alignment = float((grad * prev_deriv.float()).sum().detach().cpu()) / math.sqrt(
                max(1, grad.numel())
            )
            ema = 0.9 * state.get("precond_strength_ema_alignment", 0.0) + 0.1 * alignment
            state["precond_strength_ema_alignment"] = ema
            hypgrad = -prev_lr * ema
            clipped = _clip_scalar(hypgrad, self.precond_strength_hypergrad_clip)
            draw = self._precond_strength_draw(p)
            state["precond_strength_raw"].sub_(
                self.precond_strength_hyper_lr * clipped * draw
            )
            state["precond_strength_hypgrad"] = clipped
            state["precond_strength_hypgrad_unclipped"] = hypgrad
            state["precond_strength_alignment"] = alignment

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
        self.gduo_step += 1

        for group in self.param_groups:
            ############################
            #       Newton-Muon        #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            self._gduo_update_from_previous(params, group)
            self._update_precond_strength_from_previous(params)

            total_params = sum(p.numel() for p in params)
            if total_params > 0:
                device = params[0].device
                updates_flat = torch.zeros(
                    total_params, device=device, dtype=torch.float32
                )
                derivs_flat = (
                    torch.zeros(total_params, device=device, dtype=torch.float32)
                    if self.gduo_learn_momentum
                    else None
                )
            else:
                updates_flat = None
                derivs_flat = None

            curr_idx = 0
            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    momentum = self._gduo_momentum(p)
                    
                    g = p.grad
                    assert g is not None
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    g = self._maybe_precondition(g, state)
                    precond_g = g

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    old_buf = buf.clone() if self.gduo_learn_momentum else None
                    old_buf_for_precond = (
                        buf.clone() if self.learn_precond_strength else None
                    )
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)

                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                    if self.gduo_learn_momentum and old_buf is not None:
                        eps = 1e-3
                        dmu_draw = self._gduo_dmu_draw(p)
                        deriv = self._finite_diff_muon_direction(
                            precond_g,
                            old_buf,
                            momentum,
                            group["nesterov"],
                            group["ns_steps"],
                            eps,
                        )
                        deriv.mul_(dmu_draw)
                        derivs_flat[curr_idx : curr_idx + p.numel()] = deriv.flatten()
                    if self.learn_precond_strength and old_buf_for_precond is not None:
                        raw_grad = state.get("precond_raw_grad")
                        precond_grad = state.get("precond_precond_grad")
                        if raw_grad is not None and precond_grad is not None:
                            deriv = self._finite_diff_precond_strength_direction(
                                raw_grad,
                                precond_grad,
                                old_buf_for_precond,
                                self._precond_strength(p),
                                momentum,
                                group["nesterov"],
                                group["ns_steps"],
                                self.precond_strength_fd_eps,
                            )
                            state["precond_strength_prev_deriv"] = deriv.detach().float().clone()
                        else:
                            state["precond_strength_prev_deriv"] = None
                curr_idx += p.numel()

            if self.world_size > 1 and updates_flat is not None:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
                if derivs_flat is not None:
                    dist.all_reduce(derivs_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in params:
                g_meta = updates_flat[curr_idx : curr_idx + p.numel()].view_as(p.data)
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                
                p_lr = self._gduo_actual_lr(p, group)
                p.data.add_(g, alpha=-p_lr)
                
                mu_deriv = None
                if derivs_flat is not None:
                    mu_deriv = derivs_flat[curr_idx : curr_idx + p.numel()].view_as(p.data)
                self._gduo_store_previous(p, g_meta, mu_deriv)
                self.state[p]["gduo_prev_actual_lr"] = p_lr
                self.state[p]["precond_strength_prev_lr"] = p_lr
                curr_idx += p.numel()

            self._maybe_log_preconditioner_stats()
            self._gduo_log(group, prefix="GDUO-NewtonMuon")

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

    def _finite_diff_muon_direction(self, grad, old_buf, momentum, nesterov, ns_steps, eps):
        def direction(mu):
            buf = old_buf * mu + grad
            d = grad + buf * mu if nesterov else buf
            d = zeropower_via_newtonschulz5(d, steps=ns_steps)
            return d * max(1, d.size(0) / d.size(1)) ** 0.5

        plus = direction(momentum + eps)
        minus = direction(momentum - eps)
        return (plus - minus) / (2.0 * eps)

    def _finite_diff_precond_strength_direction(
        self,
        raw_grad,
        precond_grad,
        old_buf,
        strength,
        momentum,
        nesterov,
        ns_steps,
        eps,
    ):
        delta = precond_grad - raw_grad

        def direction(strength_value):
            strength_value = min(
                max(strength_value, self.precond_strength_min),
                self.precond_strength_max,
            )
            grad = raw_grad + strength_value * delta
            buf = old_buf * momentum + grad
            d = grad + buf * momentum if nesterov else buf
            d = zeropower_via_newtonschulz5(d, steps=ns_steps)
            return d * max(1, d.size(0) / d.size(1)) ** 0.5

        plus_s = min(strength + eps, self.precond_strength_max)
        minus_s = max(strength - eps, self.precond_strength_min)
        if plus_s == minus_s:
            return torch.zeros_like(raw_grad)
        return (direction(plus_s) - direction(minus_s)) / (plus_s - minus_s)

    def _maybe_precondition(self, grad, state):
        if not state.get("use_newton_muon", False):
            state["precond_raw_grad"] = None
            state["precond_precond_grad"] = None
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
            state["precond_raw_grad"] = None
            state["precond_precond_grad"] = None
            return grad

        mode = state.get("preconditioner_mode", "full")
        raw_grad = grad.float()
        if mode == "full":
            precond_grad = raw_grad @ inv.to(device=grad.device)
            return self._blend_preconditioned_grad(raw_grad, precond_grad, state)

        block = state["preconditioner_block"]
        if grad.shape[1] % block != 0:
            return grad
        n_blocks = grad.shape[1] // block
        grad_blocks = raw_grad.reshape(grad.shape[0], n_blocks, block)
        precond_grad = torch.einsum(
            "obi,bij->obj", grad_blocks, inv.to(device=grad.device)
        ).reshape_as(grad)
        return self._blend_preconditioned_grad(raw_grad, precond_grad, state)

    def _blend_preconditioned_grad(self, raw_grad, precond_grad, state):
        precond_grad = self._clip_preconditioned_grad(raw_grad, precond_grad, state)
        state["precond_raw_grad"] = raw_grad.detach().float().clone()
        state["precond_precond_grad"] = precond_grad.detach().float().clone()
        if self.learn_precond_strength:
            strength = self._precond_strength_from_state(state)
            return raw_grad + strength * (precond_grad - raw_grad)
        return precond_grad

    def _precond_strength_from_state(self, state):
        if not self.learn_precond_strength:
            return 1.0
        scaled = _sigmoid(float(state["precond_strength_raw"].item()))
        return self.precond_strength_min + (
            self.precond_strength_max - self.precond_strength_min
        ) * scaled

    def _record_precond_stats(self, state, ratio_float, clipped):
        bucket = state.get("bucket", "unknown")
        stats = self._precond_stats
        stats["count"] += 1
        stats["ratio_sum"] += ratio_float
        stats["max_ratio"] = max(stats["max_ratio"], ratio_float)
        if clipped:
            stats["clipped"] += 1
        bucket_stats = stats["by_bucket"].setdefault(
            bucket, dict(count=0, clipped=0, ratio_sum=0.0, max_ratio=0.0)
        )
        bucket_stats["count"] += 1
        bucket_stats["ratio_sum"] += ratio_float
        bucket_stats["max_ratio"] = max(bucket_stats["max_ratio"], ratio_float)
        if clipped:
            bucket_stats["clipped"] += 1

    def _format_precond_buckets(self):
        parts = []
        for bucket in sorted(self._precond_stats["by_bucket"]):
            stats = self._precond_stats["by_bucket"][bucket]
            count = stats["count"]
            avg = stats["ratio_sum"] / count if count else float("nan")
            parts.append(
                f"{bucket}:n={count},clip={stats['clipped']},avg={avg:.3g},max={stats['max_ratio']:.3g}"
            )
        return ";".join(parts) if parts else "none"

    def _format_precond_strength_buckets(self):
        if not self.learn_precond_strength:
            return "disabled"
        buckets = {}
        for group in self.param_groups:
            for p in group["params"]:
                if not self.state[p].get("use_muon", False):
                    continue
                bucket = self.state[p].get("bucket", "unknown")
                buckets.setdefault(bucket, []).append(self._precond_strength(p))
        parts = []
        for bucket in sorted(buckets):
            vals = buckets[bucket]
            parts.append(
                f"{bucket}:avg={sum(vals) / len(vals):.4g},min={min(vals):.4g},max={max(vals):.4g}"
            )
        return ";".join(parts) if parts else "none"

    def _precond_strength_summary(self):
        if not self.learn_precond_strength:
            return "precond_strength=disabled"
        vals = []
        hypgrads = []
        alignments = []
        for group in self.param_groups:
            for p in group["params"]:
                if not self.state[p].get("use_muon", False):
                    continue
                vals.append(self._precond_strength(p))
                hypgrad = self.state[p].get("precond_strength_hypgrad", float("nan"))
                if math.isfinite(hypgrad):
                    hypgrads.append(hypgrad)
                alignment = self.state[p].get("precond_strength_alignment", float("nan"))
                if math.isfinite(alignment):
                    alignments.append(alignment)
        if not vals:
            return "precond_strength=none"
        return (
            f"precond_strength_avg={sum(vals) / len(vals):.6g} "
            f"(min={min(vals):.6g}, max={max(vals):.6g}) "
            f"precond_strength_hypgrad_avg={sum(hypgrads) / len(hypgrads) if hypgrads else float('nan'):.6g} "
            f"precond_strength_alignment_avg={sum(alignments) / len(alignments) if alignments else float('nan'):.6g} "
            f"precond_strength_by_bucket={self._format_precond_strength_buckets()}"
        )

    def _clip_preconditioned_grad(self, raw_grad, precond_grad, state):
        if not torch.isfinite(precond_grad).all():
            return raw_grad

        raw_norm = raw_grad.norm().clamp_min(1e-12)
        precond_norm = precond_grad.norm()
        ratio = (precond_norm / raw_norm).detach()
        ratio_float = float(ratio.clamp_max(1e6).cpu())
        clipped = self.precond_clip > 0 and ratio > self.precond_clip
        self._record_precond_stats(state, ratio_float, clipped)

        if clipped:
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
        avg_ratio = stats["ratio_sum"] / stats["count"]
        print(
            "[NewtonMuon] "
            f"step={self._step} precond_count={stats['count']} "
            f"precond_clipped={stats['clipped']} "
            f"precond_avg_ratio={avg_ratio:.3e} "
            f"precond_max_ratio={stats['max_ratio']:.3e} "
            f"precond_by_bucket={self._format_precond_buckets()} "
            f"{self._precond_strength_summary()}"
        )
        self._precond_stats = _new_precond_stats()

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
