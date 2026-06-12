import math

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from optim.gduo_meta import GDUOMetaMixin
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


def _new_precond_stats():
    return dict(count=0, clipped=0, ratio_sum=0.0, max_ratio=0.0, by_bucket={})


class MuonNewtonGate(GDUOMetaMixin, Optimizer):
    """
    Learns a tensor-wise residual Newton correction on top of Muon.

    U = U_Muon + alpha * normalize(U_NewtonMuon - U_Muon, ||U_Muon||)

    This keeps Muon as the stable backbone and asks the meta-optimizer whether
    the Newton preconditioner adds useful curvature information.
    """

    def __init__(
        self,
        muon_params,
        model,
        adamw_params=None,
        lr=0.001,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        ns_a=3.4445,
        ns_b=-4.775,
        ns_c=2.0315,
        weight_decay=1e-3,
        adamw_lr=None,
        adamw_wd=None,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
        ewma_beta=0.95,
        ridge=0.2,
        refresh_interval=32,
        second_moment_init=1e-3,
        max_precond_dim=1024,
        block_size=512,
        precond_clip=0.0,
        precond_strength_init=0.1,
        precond_strength_min=0.0,
        precond_strength_max=1.0,
        precond_strength_hyper_lr=1e-3,
        precond_strength_hypergrad_clip=1.0,
        gate_hyper_lr=1e-3,
        gate_hypergrad_clip=1.0,
        gate_init=0.5,
        gduo_learn_lr=False,
        gduo_learn_momentum=False,
        gduo_ema_beta=0.9,
        gduo_lr_hyper_lr=1e-3,
        gduo_momentum_hyper_lr=1e-3,
        gduo_hypergrad_clip=1.0,
        gduo_lr_min_ratio=0.25,
        gduo_lr_max_ratio=4.0,
        gduo_scope="tensor",
        log_interval=200,
    ):
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

        self.model = model
        self.ewma_beta = ewma_beta
        self.ridge = ridge
        self.refresh_interval = max(1, refresh_interval)
        self.second_moment_init = second_moment_init
        self.max_precond_dim = max_precond_dim
        self.block_size = block_size
        self.precond_clip = precond_clip
        self.precond_strength_min = precond_strength_min
        self.precond_strength_max = precond_strength_max
        self.precond_strength_hyper_lr = precond_strength_hyper_lr
        self.precond_strength_hypergrad_clip = precond_strength_hypergrad_clip
        self.gate_hyper_lr = gate_hyper_lr
        self.gate_hypergrad_clip = gate_hypergrad_clip
        self.log_interval = log_interval
        self.step_count = 0
        self.refresh_count = 0
        self._precond_stats = _new_precond_stats()
        self._hook_handles = []

        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
            self.state[p]["use_newton"] = False
            self.state[p]["bucket"] = _matrix_bucket(p)
        if adamw_params is not None:
            for p in adamw_params:
                self.state[p]["use_muon"] = False
                self.state[p]["use_newton"] = False
                self.state[p]["bucket"] = _matrix_bucket(p)

        raw_init = _logit(gate_init)
        strength_scaled = (precond_strength_init - precond_strength_min) / max(
            1e-12, precond_strength_max - precond_strength_min
        )
        strength_raw_init = _logit(strength_scaled)
        self.bucket_gate_raw = {}
        self.bucket_precond_strength_raw = {}
        for group in self.param_groups:
            for p in group["params"]:
                bucket = self.state[p].get("bucket", _matrix_bucket(p))
                if bucket not in self.bucket_gate_raw:
                    self.bucket_gate_raw[bucket] = torch.tensor(raw_init, dtype=torch.float64)
                if bucket not in self.bucket_precond_strength_raw:
                    self.bucket_precond_strength_raw[bucket] = torch.tensor(
                        strength_raw_init, dtype=torch.float64
                    )
                self.state[p]["gate_raw"] = self.bucket_gate_raw[bucket]
                self.state[p]["precond_strength_raw"] = self.bucket_precond_strength_raw[bucket]
                self.state[p]["hypgrad_gate"] = float("nan")
                self.state[p]["hypgrad_gate_unclipped"] = float("nan")
                self.state[p]["alignment_gate"] = float("nan")
                self.state[p]["precond_strength_hypgrad"] = float("nan")
                self.state[p]["precond_strength_alignment"] = float("nan")
                self.state[p]["muon_newton_cos"] = float("nan")
                self.state[p]["curvature_correction_ratio"] = float("nan")

        self._init_gduo_meta(
            learn_lr=gduo_learn_lr,
            learn_momentum=gduo_learn_momentum,
            lr_hyper_lr=gduo_lr_hyper_lr,
            momentum_hyper_lr=gduo_momentum_hyper_lr,
            hypergrad_clip=gduo_hypergrad_clip,
            lr_min_ratio=gduo_lr_min_ratio,
            lr_max_ratio=gduo_lr_max_ratio,
            ema_beta=gduo_ema_beta,
            log_interval=log_interval,
            scope=gduo_scope,
        )
        self.register_model_hooks(model)

    def register_model_hooks(self, model):
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            p = module.weight
            if p not in self.state or not self.state[p].get("use_muon", False):
                continue
            self.state[p]["use_newton"] = True
            self._hook_handles.append(module.register_forward_pre_hook(self._make_hook(p)))

    def _make_hook(self, param):
        def hook(module, inputs):
            if not module.training or not inputs or not torch.is_tensor(inputs[0]):
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

    def _identity_like_cov(self, cov):
        if cov.ndim == 2:
            return torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        return eye.unsqueeze(0).expand(cov.shape[0], -1, -1)

    def _consume_covariance(self, state):
        cov_sum = state.pop("pending_cov_sum", None)
        cov_count = state.pop("pending_cov_count", None)
        if cov_sum is None or cov_count is None or cov_count == 0:
            return None
        return cov_sum / cov_count

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
        return eye / ridge.clamp_min(1e-6)

    def _maybe_precondition(self, grad, state):
        if not state.get("use_newton", False):
            return grad
        cov = self._consume_covariance(state)
        if cov is not None:
            if "input_cov" not in state:
                eye = self._identity_like_cov(cov)
                state["input_cov"] = cov.add(eye, alpha=self.second_moment_init)
            else:
                state["input_cov"].lerp_(cov, 1 - self.ewma_beta)
            if "input_cov_inv" not in state or self.step_count % self.refresh_interval == 0:
                state["input_cov_inv"] = self._invert_covariance(state["input_cov"])
                self.refresh_count += 1

        inv = state.get("input_cov_inv")
        if inv is None:
            return grad
        raw_grad = grad.float()
        mode = state.get("preconditioner_mode", "full")
        if mode == "full":
            precond = raw_grad @ inv.to(device=grad.device)
        else:
            block = state["preconditioner_block"]
            if grad.shape[1] % block != 0:
                return grad
            n_blocks = grad.shape[1] // block
            grad_blocks = raw_grad.reshape(grad.shape[0], n_blocks, block)
            precond = torch.einsum(
                "obi,bij->obj", grad_blocks, inv.to(device=grad.device)
            ).reshape_as(grad)
        return self._clip_preconditioned_grad(raw_grad, precond, state)

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

    def _clip_preconditioned_grad(self, raw_grad, precond_grad, state):
        if not torch.isfinite(precond_grad).all():
            return raw_grad
        raw_norm = raw_grad.norm().clamp_min(1e-12)
        ratio = (precond_grad.norm() / raw_norm).detach()
        ratio_float = float(ratio.clamp_max(1e6).cpu())
        clipped = self.precond_clip > 0 and ratio > self.precond_clip
        self._record_precond_stats(state, ratio_float, clipped)
        if clipped:
            precond_grad = precond_grad * (self.precond_clip / ratio)
        precond_norm = precond_grad.norm().clamp_min(1e-12)
        precond_grad = precond_grad * (raw_norm / precond_norm)
        return precond_grad

    def _gate_val(self, p):
        return _sigmoid(float(self.state[p]["gate_raw"].item()))

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

    def _update_precond_strength_from_previous(self, params, group):
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

    def _norm_match(self, source, target, eps=1e-12):
        source_f = source.float()
        target_norm = target.float().norm().clamp_min(eps)
        source_norm = source_f.norm().clamp_min(eps)
        return source_f * (target_norm / source_norm)

    def _curvature_residual(self, u_muon, u_newton, state, record=True):
        u_muon_f = u_muon.float()
        u_newton_f = u_newton.float()
        correction = u_newton_f - u_muon_f
        correction_norm = correction.norm()
        muon_norm = u_muon_f.norm().clamp_min(1e-12)
        newton_norm = u_newton_f.norm().clamp_min(1e-12)
        if not torch.isfinite(correction_norm) or correction_norm <= 0:
            if record:
                state["muon_newton_cos"] = float("nan")
                state["curvature_correction_ratio"] = 0.0
            return torch.zeros_like(u_muon_f)
        cos = (u_muon_f * u_newton_f).sum() / (muon_norm * newton_norm)
        if record:
            state["muon_newton_cos"] = float(cos.detach().clamp(-1, 1).cpu())
            state["curvature_correction_ratio"] = float((correction_norm / muon_norm).detach().clamp_max(1e6).cpu())
        return correction * (muon_norm / correction_norm.clamp_min(1e-12))

    def _update_gates_from_previous(self, params, group):
        for p in params:
            if p.grad is None:
                continue
            state = self.state[p]
            prev_correction = state.get("prev_curvature_correction")
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

    def _muon_direction(self, direction, p, group):
        shape_orig = p.shape
        direction_2d = direction.reshape(shape_orig[0], -1) if p.dim() > 2 else direction
        direction_2d = zeropower_via_newtonschulz5(direction_2d, steps=group["ns_steps"])
        direction_2d *= max(1, direction_2d.size(0) / direction_2d.size(1)) ** 0.5
        return direction_2d.reshape(shape_orig) if p.dim() > 2 else direction_2d

    def _finite_diff_final_direction(
        self,
        grad,
        grad_precond,
        old_pure_mom,
        old_newton_mom,
        momentum,
        gate,
        p,
        group,
        eps=1e-3,
    ):
        def direction(mu):
            pure_buf = old_pure_mom * mu + grad
            pure_dir = grad.add(pure_buf, alpha=mu) if group["nesterov"] else pure_buf
            u_muon = self._muon_direction(pure_dir, p, group)

            newton_buf = old_newton_mom * mu + grad_precond
            newton_dir = (
                grad_precond.add(newton_buf, alpha=mu)
                if group["nesterov"]
                else newton_buf
            )
            u_newton = self._muon_direction(newton_dir, p, group)
            correction = self._curvature_residual(u_muon, u_newton, self.state[p], record=False)
            return u_muon + gate * correction

        plus = direction(momentum + eps)
        minus = direction(momentum - eps)
        return (plus - minus) / (2.0 * eps)

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
        strength_by_bucket = {}
        strength_hypgrad_by_bucket = {}
        strength_alignment_by_bucket = {}
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
            strength_by_bucket.setdefault(bucket, []).append(self._precond_strength(p))
            strength_hypgrad = self.state[p].get("precond_strength_hypgrad", float("nan"))
            if math.isfinite(strength_hypgrad):
                strength_hypgrad_by_bucket.setdefault(bucket, []).append(strength_hypgrad)
            strength_alignment = self.state[p].get("precond_strength_alignment", float("nan"))
            if math.isfinite(strength_alignment):
                strength_alignment_by_bucket.setdefault(bucket, []).append(strength_alignment)
            cos = self.state[p].get("muon_newton_cos", float("nan"))
            if math.isfinite(cos):
                cos_by_bucket.setdefault(bucket, []).append(cos)
            ratio = self.state[p].get("curvature_correction_ratio", float("nan"))
            if math.isfinite(ratio):
                correction_ratio_by_bucket.setdefault(bucket, []).append(ratio)
        stats = self._precond_stats
        precond_avg_ratio = stats["ratio_sum"] / stats["count"] if stats["count"] else float("nan")
        print(
            f"[MuonNewtonGate] step={self.step_count} "
            f"curvature_alpha_avg={sum(gates) / len(gates):.6g} "
            f"(min={min(gates):.6g}, max={max(gates):.6g}) "
            f"curvature_alpha_bound_frac={sum(g <= 1e-4 or g >= 1 - 1e-4 for g in gates) / len(gates):.3g} "
            f"hypgrad_gate_avg={sum(finite_hypgrads) / len(finite_hypgrads) if finite_hypgrads else float('nan'):.6g} "
            f"alignment_avg={sum(finite_alignments) / len(finite_alignments) if finite_alignments else float('nan'):.6g} "
            f"precond_count={stats['count']} precond_clipped={stats['clipped']} "
            f"precond_avg_ratio={precond_avg_ratio:.3e} precond_max_ratio={stats['max_ratio']:.3e} "
            f"curvature_alpha_by_bucket={_format_bucket_means(gate_by_bucket)} "
            f"hypgrad_by_bucket={_format_bucket_means(hypgrad_by_bucket)} "
            f"alignment_by_bucket={_format_bucket_means(alignment_by_bucket)} "
            f"precond_strength_by_bucket={_format_bucket_means(strength_by_bucket)} "
            f"precond_strength_hypgrad_by_bucket={_format_bucket_means(strength_hypgrad_by_bucket)} "
            f"precond_strength_alignment_by_bucket={_format_bucket_means(strength_alignment_by_bucket)} "
            f"muon_newton_cos_by_bucket={_format_bucket_means(cos_by_bucket)} "
            f"curvature_correction_ratio_by_bucket={_format_bucket_means(correction_ratio_by_bucket)} "
            f"precond_by_bucket={self._format_precond_buckets()}"
        )
        self._precond_stats = _new_precond_stats()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.gduo_step += 1

        for group in self.param_groups:
            matrix_params = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            self._update_gates_from_previous(matrix_params, group)
            self._update_precond_strength_from_previous(matrix_params, group)
            self._gduo_update_from_previous(matrix_params, group)

            for p in matrix_params:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                state = self.state[p]
                if "pure_muon_momentum" not in state:
                    state["pure_muon_momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["newton_muon_momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                momentum = self._gduo_momentum(p)
                pure_mom = state["pure_muon_momentum"]
                old_pure_mom = pure_mom.clone() if self.gduo_learn_momentum else None
                pure_mom.mul_(momentum).add_(grad)
                pure_dir = grad.add(pure_mom, alpha=momentum) if group["nesterov"] else pure_mom
                u_muon = self._muon_direction(pure_dir, p, group)

                grad_2d = grad.view(grad.size(0), -1) if grad.ndim > 2 else grad
                grad_precond = self._maybe_precondition(grad_2d, state)
                if grad.ndim > 2:
                    grad_precond = grad_precond.view_as(grad)
                raw_strength_delta = grad_precond.float() - grad.float()
                strength = self._precond_strength(p)
                grad_precond = grad.float() + strength * raw_strength_delta
                newton_mom = state["newton_muon_momentum"]
                old_newton_mom = newton_mom.clone() if self.gduo_learn_momentum else None
                newton_mom.mul_(momentum).add_(grad_precond)
                newton_dir = grad_precond.add(newton_mom, alpha=momentum) if group["nesterov"] else newton_mom
                u_newton = self._muon_direction(newton_dir, p, group)

                gate = self._gate_val(p)
                correction = self._curvature_residual(u_muon, u_newton, state)
                u_final = u_muon + gate * correction
                if group["weight_decay"] > 0:
                    u_final = u_final + p.detach() * group["weight_decay"]
                p_lr = self._gduo_actual_lr(p, group)
                p.sub_(u_final, alpha=p_lr)

                mu_deriv = None
                if (
                    self.gduo_learn_momentum
                    and old_pure_mom is not None
                    and old_newton_mom is not None
                ):
                    mu_deriv = self._finite_diff_final_direction(
                        grad,
                        grad_precond,
                        old_pure_mom,
                        old_newton_mom,
                        momentum,
                        gate,
                        p,
                        group,
                    )
                    mu_deriv.mul_(self._gduo_dmu_draw(p))

                state["prev_U_Muon"] = u_muon.detach().float().clone()
                state["prev_U_Newton"] = u_newton.detach().float().clone()
                state["prev_curvature_correction"] = correction.detach().float().clone()
                state["prev_precond_strength_direction"] = (
                    correction.detach().float().clone() / max(strength, 1e-6)
                )
                state["prev_precond_strength_draw"] = self._precond_strength_draw(p)
                state["prev_lr"] = p_lr
                self._gduo_store_previous(p, u_final, mu_deriv)
                state["gduo_prev_actual_lr"] = p_lr

            # AdamW fallback for non-matrix params, once.
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
            self._gduo_log(group, prefix="GDUO-MuonNewtonGate")

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
            "curvature_alpha_avg": sum(gates) / len(gates),
            "curvature_alpha_min": min(gates),
            "curvature_alpha_max": max(gates),
            "precond_refreshes": self.refresh_count,
        }
        params_with_lr = [p for p in group["params"] if self._gduo_has_meta(p)]
        if params_with_lr:
            metrics["gduo_lr_scale_avg"] = sum(
                self._gduo_lr_scale(p) for p in params_with_lr
            ) / len(params_with_lr)
            if self.gduo_learn_momentum:
                metrics["gduo_momentum_avg"] = sum(
                    self._gduo_momentum(p) for p in params_with_lr
                ) / len(params_with_lr)
        gate_by_bucket = {}
        strength_by_bucket = {}
        cos_by_bucket = {}
        correction_ratio_by_bucket = {}
        for p in group["params"]:
            if not self.state[p].get("use_muon", False):
                continue
            bucket = self.state[p].get("bucket", "unknown")
            gate_by_bucket.setdefault(bucket, []).append(self._gate_val(p))
            strength_by_bucket.setdefault(bucket, []).append(self._precond_strength(p))
            cos = self.state[p].get("muon_newton_cos", float("nan"))
            if math.isfinite(cos):
                cos_by_bucket.setdefault(bucket, []).append(cos)
            ratio = self.state[p].get("curvature_correction_ratio", float("nan"))
            if math.isfinite(ratio):
                correction_ratio_by_bucket.setdefault(bucket, []).append(ratio)
        for bucket, vals in gate_by_bucket.items():
            if vals:
                metrics[f"curvature_alpha_avg_{bucket}"] = sum(vals) / len(vals)
        for bucket, vals in strength_by_bucket.items():
            if vals:
                metrics[f"precond_strength_avg_{bucket}"] = sum(vals) / len(vals)
        for bucket, vals in cos_by_bucket.items():
            if vals:
                metrics[f"muon_newton_cos_avg_{bucket}"] = sum(vals) / len(vals)
        for bucket, vals in correction_ratio_by_bucket.items():
            if vals:
                metrics[f"curvature_correction_ratio_avg_{bucket}"] = sum(vals) / len(vals)
        return metrics
