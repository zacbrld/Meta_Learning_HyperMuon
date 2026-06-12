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


class AdamMuonGate(Optimizer):
    """
    Learns a tensor-wise residual Adam correction on top of Muon.

    U = U_Muon + alpha * normalize(U_Adam - U_Muon, ||U_Muon||)

    This keeps Muon as the stable matrix-geometry backbone and asks whether
    Adam's coordinate-wise geometry adds useful information for a layer.
    """

    def __init__(
        self,
        muon_params,
        adamw_params=None,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        ns_a=3.4445,
        ns_b=-4.775,
        ns_c=2.0315,
        adamw_lr=None,
        adamw_wd=None,
        adamw_betas=None,
        adamw_eps=None,
        gate_hyper_lr=1e-3,
        gate_hypergrad_clip=1.0,
        gate_init=0.5,
        log_interval=200,
    ):
        adamw_lr = lr if adamw_lr is None else adamw_lr
        adamw_wd = weight_decay if adamw_wd is None else adamw_wd
        adamw_betas = betas if adamw_betas is None else adamw_betas
        adamw_eps = eps if adamw_eps is None else adamw_eps

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
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

        self.gate_hyper_lr = gate_hyper_lr
        self.gate_hypergrad_clip = gate_hypergrad_clip
        self.log_interval = log_interval
        self.step_count = 0

        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
            self.state[p]["bucket"] = _matrix_bucket(p)
        if adamw_params is not None:
            for p in adamw_params:
                self.state[p]["use_muon"] = False
                self.state[p]["bucket"] = _matrix_bucket(p)

        raw_init = _logit(gate_init)
        self.bucket_gate_raw = {}
        for group in self.param_groups:
            for p in group["params"]:
                bucket = self.state[p].get("bucket", _matrix_bucket(p))
                if bucket not in self.bucket_gate_raw:
                    self.bucket_gate_raw[bucket] = torch.tensor(raw_init, dtype=torch.float64)
                self.state[p]["gate_raw"] = self.bucket_gate_raw[bucket]
                self.state[p]["hypgrad_gate"] = float("nan")
                self.state[p]["hypgrad_gate_unclipped"] = float("nan")
                self.state[p]["alignment_gate"] = float("nan")
                self.state[p]["adam_muon_cos"] = float("nan")
                self.state[p]["geometry_correction_ratio"] = float("nan")

    def _gate_val(self, p):
        return _sigmoid(float(self.state[p]["gate_raw"].item()))

    def _geometry_residual(self, u_muon, u_adam, state):
        u_muon_f = u_muon.float()
        u_adam_f = u_adam.float()
        correction = u_adam_f - u_muon_f
        correction_norm = correction.norm()
        muon_norm = u_muon_f.norm().clamp_min(1e-12)
        adam_norm = u_adam_f.norm().clamp_min(1e-12)
        if not torch.isfinite(correction_norm) or correction_norm <= 0:
            state["adam_muon_cos"] = float("nan")
            state["geometry_correction_ratio"] = 0.0
            return torch.zeros_like(u_muon_f)
        cos = (u_adam_f * u_muon_f).sum() / (adam_norm * muon_norm)
        state["adam_muon_cos"] = float(cos.detach().clamp(-1, 1).cpu())
        state["geometry_correction_ratio"] = float((correction_norm / muon_norm).detach().clamp_max(1e6).cpu())
        return correction * (muon_norm / correction_norm.clamp_min(1e-12))

    def _update_gates_from_previous(self, params, group):
        for p in params:
            if p.grad is None or not self.state[p].get("use_muon", False):
                continue
            state = self.state[p]
            prev_correction = state.get("prev_geometry_correction")
            prev_lr = state.get("prev_lr")
            if prev_correction is None or prev_lr is None:
                continue

            grad = p.grad.detach().float()
            alignment = float((grad * prev_correction.float()).sum().detach().cpu()) / math.sqrt(max(1, grad.numel()))
            gate = self._gate_val(p)
            hypgrad_raw = -prev_lr * gate * (1.0 - gate) * alignment
            hypgrad = _clip_scalar(hypgrad_raw, self.gate_hypergrad_clip)
            state["gate_raw"].sub_(self.gate_hyper_lr * hypgrad / math.sqrt(max(1, len(params))))
            state["hypgrad_gate"] = hypgrad
            state["hypgrad_gate_unclipped"] = hypgrad_raw
            state["alignment_gate"] = alignment

    def _maybe_log(self, group):
        if self.log_interval <= 0 or self.step_count % self.log_interval != 0:
            return
        gates = [
            self._gate_val(p)
            for p in group["params"]
            if self.state[p].get("use_muon", False)
        ]
        if not gates:
            return
        hypgrads = [
            self.state[p].get("hypgrad_gate", float("nan"))
            for p in group["params"]
            if self.state[p].get("use_muon", False)
        ]
        alignments = [
            self.state[p].get("alignment_gate", float("nan"))
            for p in group["params"]
            if self.state[p].get("use_muon", False)
        ]
        finite_hypgrads = [x for x in hypgrads if math.isfinite(x)]
        finite_alignments = [x for x in alignments if math.isfinite(x)]
        gate_by_bucket = {}
        alignment_by_bucket = {}
        hypgrad_by_bucket = {}
        cos_by_bucket = {}
        correction_ratio_by_bucket = {}
        for p in group["params"]:
            if not self.state[p].get("use_muon", False):
                continue
            bucket = self.state[p].get("bucket", "unknown")
            gate_by_bucket.setdefault(bucket, []).append(self._gate_val(p))
            alignment = self.state[p].get("alignment_gate", float("nan"))
            if math.isfinite(alignment):
                alignment_by_bucket.setdefault(bucket, []).append(alignment)
            hypgrad = self.state[p].get("hypgrad_gate", float("nan"))
            if math.isfinite(hypgrad):
                hypgrad_by_bucket.setdefault(bucket, []).append(hypgrad)
            cos = self.state[p].get("adam_muon_cos", float("nan"))
            if math.isfinite(cos):
                cos_by_bucket.setdefault(bucket, []).append(cos)
            ratio = self.state[p].get("geometry_correction_ratio", float("nan"))
            if math.isfinite(ratio):
                correction_ratio_by_bucket.setdefault(bucket, []).append(ratio)
        print(
            f"[AdamMuonGate] step={self.step_count} "
            f"geometry_alpha_avg={sum(gates) / len(gates):.6g} "
            f"(min={min(gates):.6g}, max={max(gates):.6g}) "
            f"geometry_alpha_bound_frac={sum(g <= 1e-4 or g >= 1 - 1e-4 for g in gates) / len(gates):.3g} "
            f"hypgrad_gate_avg={sum(finite_hypgrads) / len(finite_hypgrads) if finite_hypgrads else float('nan'):.6g} "
            f"alignment_avg={sum(finite_alignments) / len(finite_alignments) if finite_alignments else float('nan'):.6g} "
            f"geometry_alpha_by_bucket={_format_bucket_means(gate_by_bucket)} "
            f"hypgrad_by_bucket={_format_bucket_means(hypgrad_by_bucket)} "
            f"alignment_by_bucket={_format_bucket_means(alignment_by_bucket)} "
            f"adam_muon_cos_by_bucket={_format_bucket_means(cos_by_bucket)} "
            f"geometry_correction_ratio_by_bucket={_format_bucket_means(correction_ratio_by_bucket)}"
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            matrix_params = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            self._update_gates_from_previous(matrix_params, group)

            beta1, beta2 = group["betas"]
            eps = group["eps"]
            momentum = group["momentum"]
            for p in matrix_params:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["adam_exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["adam_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["muon_momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                exp_avg = state["adam_exp_avg"]
                exp_avg_sq = state["adam_exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                u_adam = (exp_avg / denom) * (math.sqrt(bias_correction2) / bias_correction1)
                u_adam = u_adam * group["adamw_lr_ratio"]

                muon_mom = state["muon_momentum"]
                muon_mom.mul_(momentum).add_(grad)
                u_muon = grad.add(muon_mom, alpha=momentum) if group["nesterov"] else muon_mom
                shape_orig = p.shape
                u_muon_2d = u_muon.reshape(shape_orig[0], -1) if p.dim() > 2 else u_muon
                u_muon_2d = zeropower_via_newtonschulz5(u_muon_2d, steps=group["ns_steps"])
                u_muon_2d *= max(1, u_muon_2d.size(0) / u_muon_2d.size(1)) ** 0.5
                u_muon = u_muon_2d.reshape(shape_orig) if p.dim() > 2 else u_muon_2d

                gate = self._gate_val(p)
                correction = self._geometry_residual(u_muon, u_adam, state)
                u_final = u_muon.float() + gate * correction
                if group["weight_decay"] > 0:
                    u_final = u_final + p.detach() * group["weight_decay"]
                p.sub_(u_final, alpha=group["lr"])

                state["prev_U_Adam"] = u_adam.detach().float().clone()
                state["prev_U_Muon"] = u_muon.detach().float().clone()
                state["prev_geometry_correction"] = correction.detach().float().clone()
                state["prev_lr"] = group["lr"]

            # AdamW fallback for non-matrix params.
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
        gates = [
            self._gate_val(p)
            for p in group["params"]
            if self.state[p].get("use_muon", False)
        ]
        if not gates:
            return {}
        metrics = {
            "geometry_alpha_avg": sum(gates) / len(gates),
            "geometry_alpha_min": min(gates),
            "geometry_alpha_max": max(gates),
        }
        gate_by_bucket = {}
        cos_by_bucket = {}
        correction_ratio_by_bucket = {}
        for p in group["params"]:
            if not self.state[p].get("use_muon", False):
                continue
            bucket = self.state[p].get("bucket", "unknown")
            gate_by_bucket.setdefault(bucket, []).append(self._gate_val(p))
            cos = self.state[p].get("adam_muon_cos", float("nan"))
            if math.isfinite(cos):
                cos_by_bucket.setdefault(bucket, []).append(cos)
            ratio = self.state[p].get("geometry_correction_ratio", float("nan"))
            if math.isfinite(ratio):
                correction_ratio_by_bucket.setdefault(bucket, []).append(ratio)
        for bucket, vals in gate_by_bucket.items():
            if vals:
                metrics[f"geometry_alpha_avg_{bucket}"] = sum(vals) / len(vals)
        for bucket, vals in cos_by_bucket.items():
            if vals:
                metrics[f"adam_muon_cos_avg_{bucket}"] = sum(vals) / len(vals)
        for bucket, vals in correction_ratio_by_bucket.items():
            if vals:
                metrics[f"geometry_correction_ratio_avg_{bucket}"] = sum(vals) / len(vals)
        return metrics
