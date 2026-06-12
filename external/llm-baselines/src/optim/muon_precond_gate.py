import math

import torch
from torch.optim.optimizer import Optimizer

from optim.muon import zeropower_via_newtonschulz5


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


def _format_bucket_means(bucket_values):
    parts = []
    for bucket in sorted(bucket_values):
        vals = bucket_values[bucket]
        if vals:
            parts.append(f"{bucket}:{sum(vals) / len(vals):.4g}")
    return ",".join(parts) if parts else "none"


class MuonPrecondGate(Optimizer):
    """Muon with a learned bucket-wise preconditioner strength before Muon."""

    def __init__(
        self,
        muon_params,
        adamw_params=None,
        lr=0.001,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        ns_a=3.4445,
        ns_b=-4.775,
        ns_c=2.0315,
        adamw_lr=None,
        adamw_wd=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        precond_kind="adagrad_ema",
        precond_beta=0.95,
        precond_eps=1e-8,
        precond_strength_init=0.1,
        precond_strength_min=0.0,
        precond_strength_max=1.0,
        precond_strength_hyper_lr=1e-3,
        precond_strength_hypergrad_clip=1.0,
        max_precond_dim=1024,
        log_interval=200,
    ):
        if precond_kind not in {"adagrad_ema", "soap_lite"}:
            raise ValueError(f"Unsupported precond_kind: {precond_kind}")
        adamw_lr = lr if adamw_lr is None else adamw_lr
        adamw_wd = weight_decay if adamw_wd is None else adamw_wd
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_a=ns_a,
            ns_b=ns_b,
            ns_c=ns_c,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_wd=adamw_wd,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        muon_params = list(muon_params)
        params = list(muon_params)
        params.extend(list(adamw_params) if adamw_params is not None else [])
        Optimizer.__init__(self, params, defaults)

        self.precond_kind = precond_kind
        self.precond_beta = precond_beta
        self.precond_eps = precond_eps
        self.precond_strength_min = precond_strength_min
        self.precond_strength_max = precond_strength_max
        self.precond_strength_hyper_lr = precond_strength_hyper_lr
        self.precond_strength_hypergrad_clip = precond_strength_hypergrad_clip
        self.max_precond_dim = max_precond_dim
        self.log_interval = log_interval
        self.step_count = 0

        strength_scaled = (precond_strength_init - precond_strength_min) / max(
            1e-12, precond_strength_max - precond_strength_min
        )
        strength_raw_init = _logit(strength_scaled)
        self.bucket_precond_strength_raw = {}

        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
            self.state[p]["bucket"] = _matrix_bucket(p)
        if adamw_params is not None:
            for p in adamw_params:
                self.state[p]["use_muon"] = False
                self.state[p]["bucket"] = _matrix_bucket(p)

        for group in self.param_groups:
            for p in group["params"]:
                bucket = self.state[p].get("bucket", _matrix_bucket(p))
                if bucket not in self.bucket_precond_strength_raw:
                    self.bucket_precond_strength_raw[bucket] = torch.tensor(
                        strength_raw_init, dtype=torch.float64
                    )
                self.state[p]["precond_strength_raw"] = self.bucket_precond_strength_raw[bucket]
                self.state[p]["precond_strength_hypgrad"] = float("nan")
                self.state[p]["precond_strength_alignment"] = float("nan")
                self.state[p]["muon_precond_cos"] = float("nan")
                self.state[p]["precond_delta_ratio"] = float("nan")
                self.state[p]["precond_ratio"] = float("nan")

    def _precond_strength(self, p):
        scaled = _sigmoid(float(self.state[p]["precond_strength_raw"].item()))
        return self.precond_strength_min + (
            self.precond_strength_max - self.precond_strength_min
        ) * scaled

    def _precond_strength_draw(self, p):
        scaled = _sigmoid(float(self.state[p]["precond_strength_raw"].item()))
        return (
            (self.precond_strength_max - self.precond_strength_min)
            * scaled
            * (1.0 - scaled)
        )

    def _muon_direction(self, direction, p, group):
        shape_orig = p.shape
        direction_2d = direction.reshape(shape_orig[0], -1) if p.dim() > 2 else direction
        direction_2d = zeropower_via_newtonschulz5(direction_2d, steps=group["ns_steps"])
        direction_2d *= max(1, direction_2d.size(0) / direction_2d.size(1)) ** 0.5
        return direction_2d.reshape(shape_orig) if p.dim() > 2 else direction_2d

    def _adagrad_ema_precond(self, grad, state):
        if "precond_v" not in state:
            state["precond_v"] = torch.zeros_like(grad.float())
        v = state["precond_v"]
        v.mul_(self.precond_beta).addcmul_(grad.float(), grad.float(), value=1 - self.precond_beta)
        precond = grad.float() / v.sqrt().add(self.precond_eps)
        return self._norm_match(precond, grad)

    def _soap_lite_precond(self, grad, state):
        grad_f = grad.float()
        if grad_f.ndim > 2:
            grad_f = grad_f.reshape(grad_f.shape[0], -1)
        out_dim, in_dim = grad_f.shape
        if max(out_dim, in_dim) > self.max_precond_dim:
            return self._adagrad_ema_precond(grad, state)
        if "soap_l" not in state:
            state["soap_l"] = torch.zeros(out_dim, device=grad.device, dtype=torch.float32)
            state["soap_r"] = torch.zeros(in_dim, device=grad.device, dtype=torch.float32)
        state["soap_l"].mul_(self.precond_beta).add_(grad_f.square().mean(dim=1), alpha=1 - self.precond_beta)
        state["soap_r"].mul_(self.precond_beta).add_(grad_f.square().mean(dim=0), alpha=1 - self.precond_beta)
        left = state["soap_l"].sqrt().add(self.precond_eps).rsqrt().view(-1, 1)
        right = state["soap_r"].sqrt().add(self.precond_eps).rsqrt().view(1, -1)
        precond = grad_f * left * right
        precond = self._norm_match(precond, grad_f)
        return precond.reshape_as(grad)

    def _norm_match(self, source, target, eps=1e-12):
        source_f = source.float()
        target_norm = target.float().norm().clamp_min(eps)
        source_norm = source_f.norm().clamp_min(eps)
        return source_f * (target_norm / source_norm)

    def _precondition(self, grad, state):
        if self.precond_kind == "soap_lite":
            return self._soap_lite_precond(grad, state)
        return self._adagrad_ema_precond(grad, state)

    def _update_strength_from_previous(self, params):
        for p in params:
            if p.grad is None:
                continue
            state = self.state[p]
            prev_direction = state.get("prev_precond_strength_direction")
            prev_lr = state.get("prev_lr")
            if prev_direction is None or prev_lr is None:
                continue
            grad = p.grad.detach().float()
            alignment = float((grad * prev_direction.float()).sum().detach().cpu()) / math.sqrt(max(1, grad.numel()))
            draw = state.get("prev_precond_strength_draw", self._precond_strength_draw(p))
            hypgrad_raw = -prev_lr * draw * alignment
            hypgrad = _clip_scalar(hypgrad_raw, self.precond_strength_hypergrad_clip)
            state["precond_strength_raw"].sub_(
                self.precond_strength_hyper_lr * hypgrad / math.sqrt(max(1, len(params)))
            )
            state["precond_strength_hypgrad"] = hypgrad
            state["precond_strength_alignment"] = alignment

    def _maybe_log(self, group):
        if self.log_interval <= 0 or self.step_count % self.log_interval != 0:
            return
        strength_by_bucket = {}
        hypgrad_by_bucket = {}
        alignment_by_bucket = {}
        cos_by_bucket = {}
        precond_ratio_by_bucket = {}
        delta_ratio_by_bucket = {}
        for p in group["params"]:
            if not self.state[p].get("use_muon", False):
                continue
            state = self.state[p]
            bucket = state.get("bucket", "unknown")
            strength_by_bucket.setdefault(bucket, []).append(self._precond_strength(p))
            hypgrad = state.get("precond_strength_hypgrad", float("nan"))
            if math.isfinite(hypgrad):
                hypgrad_by_bucket.setdefault(bucket, []).append(hypgrad)
            alignment = state.get("precond_strength_alignment", float("nan"))
            if math.isfinite(alignment):
                alignment_by_bucket.setdefault(bucket, []).append(alignment)
            cos = state.get("muon_precond_cos", float("nan"))
            if math.isfinite(cos):
                cos_by_bucket.setdefault(bucket, []).append(cos)
            ratio = state.get("precond_ratio", float("nan"))
            if math.isfinite(ratio):
                precond_ratio_by_bucket.setdefault(bucket, []).append(ratio)
            delta_ratio = state.get("precond_delta_ratio", float("nan"))
            if math.isfinite(delta_ratio):
                delta_ratio_by_bucket.setdefault(bucket, []).append(delta_ratio)
        print(
            f"[MuonPrecondGate:{self.precond_kind}] step={self.step_count} "
            f"precond_strength_by_bucket={_format_bucket_means(strength_by_bucket)} "
            f"precond_strength_hypgrad_by_bucket={_format_bucket_means(hypgrad_by_bucket)} "
            f"precond_strength_alignment_by_bucket={_format_bucket_means(alignment_by_bucket)} "
            f"muon_precond_cos_by_bucket={_format_bucket_means(cos_by_bucket)} "
            f"precond_ratio_by_bucket={_format_bucket_means(precond_ratio_by_bucket)} "
            f"precond_delta_ratio_by_bucket={_format_bucket_means(delta_ratio_by_bucket)}"
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            matrix_params = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            self._update_strength_from_previous(matrix_params)

            momentum = group["momentum"]
            for p in matrix_params:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                state = self.state[p]
                if "muon_momentum" not in state:
                    state["muon_momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                precond_grad = self._precondition(grad, state)
                strength = self._precond_strength(p)
                geo_grad = grad.float() + strength * (precond_grad.float() - grad.float())

                muon_mom = state["muon_momentum"]
                muon_mom.mul_(momentum).add_(geo_grad)
                direction = geo_grad.add(muon_mom, alpha=momentum) if group["nesterov"] else muon_mom
                u_final = self._muon_direction(direction, p, group)

                if group["weight_decay"] > 0:
                    u_final = u_final + p.detach() * group["weight_decay"]
                p.sub_(u_final, alpha=group["lr"])

                u_raw = self._muon_direction(grad, p, group).float()
                u_geo = self._muon_direction(geo_grad, p, group).float()
                raw_norm = u_raw.norm().clamp_min(1e-12)
                geo_norm = u_geo.norm().clamp_min(1e-12)
                state["muon_precond_cos"] = float(((u_raw * u_geo).sum() / (raw_norm * geo_norm)).detach().clamp(-1, 1).cpu())
                state["precond_ratio"] = float((precond_grad.float().norm() / grad.float().norm().clamp_min(1e-12)).detach().clamp_max(1e6).cpu())
                state["precond_delta_ratio"] = float(((precond_grad.float() - grad.float()).norm() / grad.float().norm().clamp_min(1e-12)).detach().clamp_max(1e6).cpu())
                state["prev_precond_strength_direction"] = (u_geo - u_raw).detach().float().clone() / max(strength, 1e-6)
                state["prev_precond_strength_draw"] = self._precond_strength_draw(p)
                state["prev_lr"] = group["lr"]

            lr = group["adamw_lr_ratio"] * group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]
            for p in group["params"]:
                if p.grad is None or self.state[p].get("use_muon", False):
                    continue
                grad = p.grad.detach()
                state = self.state[p]
                if "adamw_step" not in state:
                    state["adamw_step"] = 0
                    state["moment1"] = torch.zeros_like(grad)
                    state["moment2"] = torch.zeros_like(grad)
                state["adamw_step"] += 1
                step = state["adamw_step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(grad, 1 - beta1)
                buf2.lerp_(grad.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())
                scale = (1 - beta1**step) / (1 - beta2**step) ** 0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

            self._maybe_log(group)

        self.step_count += 1
        return loss

    def get_metrics(self) -> dict:
        group = self.param_groups[0]
        strength_by_bucket = {}
        for p in group["params"]:
            if not self.state[p].get("use_muon", False):
                continue
            bucket = self.state[p].get("bucket", "unknown")
            strength_by_bucket.setdefault(bucket, []).append(self._precond_strength(p))
        metrics = {}
        for bucket, vals in strength_by_bucket.items():
            if vals:
                metrics[f"precond_strength_avg_{bucket}"] = sum(vals) / len(vals)
        return metrics
