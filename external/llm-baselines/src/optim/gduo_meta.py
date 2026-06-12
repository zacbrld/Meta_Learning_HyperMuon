import math
import torch

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

class GDUOMetaMixin:
    def _gduo_group_momentum_init(self, group):
        if "momentum" in group:
            return float(group["momentum"])
        if "betas" in group:
            return float(group["betas"][0])
        if "adamw_betas" in group:
            return float(group["adamw_betas"][0])
        return 0.0

    def _init_gduo_meta(
        self,
        learn_lr=False,
        learn_momentum=False,
        lr_hyper_lr=1e-3,
        momentum_hyper_lr=1e-3,
        hypergrad_clip=1.0,
        lr_min_ratio=0.25,
        lr_max_ratio=4.0,
        mu_min=0.0,
        mu_max=0.99,
        ema_beta=0.9,
        log_interval=0,
        scope="tensor",
    ):
        if scope not in {"tensor", "global"}:
            raise ValueError(f"Unsupported GD-UO scope: {scope}")
        self.gduo_learn_lr = learn_lr
        self.gduo_learn_momentum = learn_momentum
        self.gduo_lr_hyper_lr = lr_hyper_lr
        self.gduo_momentum_hyper_lr = momentum_hyper_lr
        self.gduo_hypergrad_clip = hypergrad_clip
        self.gduo_lr_min_ratio = lr_min_ratio
        self.gduo_lr_max_ratio = lr_max_ratio
        self.gduo_mu_min = mu_min
        self.gduo_mu_max = mu_max
        self.gduo_ema_beta = ema_beta
        self.gduo_log_interval = log_interval
        self.gduo_scope = scope
        self.gduo_step = 0
        self._gduo_param_to_group = {}

        for group in self.param_groups:
            momentum = self._gduo_group_momentum_init(group)
            scaled = (momentum - mu_min) / max(1e-12, mu_max - mu_min)
            scaled = min(max(scaled, 1e-12), 1.0 - 1e-12)
            if self.gduo_scope == "global":
                group["gduo_lr_raw"] = torch.tensor(0.0, dtype=torch.float64)
                group["gduo_mu_raw"] = torch.tensor(_logit(scaled), dtype=torch.float64)
                group["gduo_ema_alignment"] = 0.0
                group["gduo_ema_mu_alignment"] = 0.0
            for p in group["params"]:
                self._gduo_param_to_group[id(p)] = group
                if self.gduo_scope == "tensor":
                    self.state[p]["gduo_lr_raw"] = torch.tensor(0.0, dtype=torch.float64)
                    self.state[p]["gduo_mu_raw"] = torch.tensor(_logit(scaled), dtype=torch.float64)
                self.state[p]["gduo_prev_actual_lr"] = None
                self.state[p]["gduo_hypgrad_lr"] = float("nan")
                self.state[p]["gduo_hypgrad_lr_unclipped"] = float("nan")
                self.state[p]["gduo_hypgrad_mu"] = float("nan")
                self.state[p]["gduo_hypgrad_mu_unclipped"] = float("nan")
                self.state[p]["gduo_alignment"] = float("nan")
                self.state[p]["gduo_alignment_raw"] = float("nan")
                self.state[p]["gduo_alignment_cos"] = float("nan")
                self.state[p]["gduo_ema_alignment"] = 0.0
                self.state[p]["gduo_mu_alignment"] = float("nan")
                self.state[p]["gduo_mu_alignment_raw"] = float("nan")
                self.state[p]["gduo_ema_mu_alignment"] = 0.0

    def _gduo_holder(self, p):
        if self.gduo_scope == "global":
            return self._gduo_param_to_group[id(p)]
        return self.state[p]

    def _gduo_has_meta(self, p):
        return id(p) in self._gduo_param_to_group

    def _gduo_lr_scale(self, p):
        return math.exp(float(self._gduo_holder(p)["gduo_lr_raw"].item()))

    def _gduo_actual_lr(self, p, group):
        return float(group["lr"]) * self._gduo_lr_scale(p)

    def _gduo_momentum(self, p):
        scaled = _sigmoid(float(self._gduo_holder(p)["gduo_mu_raw"].item()))
        return self.gduo_mu_min + (self.gduo_mu_max - self.gduo_mu_min) * scaled

    def _gduo_dmu_draw(self, p):
        scaled = _sigmoid(float(self._gduo_holder(p)["gduo_mu_raw"].item()))
        return (self.gduo_mu_max - self.gduo_mu_min) * scaled * (1.0 - scaled)

    def _gduo_clamp_lr(self, p):
        holder = self._gduo_holder(p)
        raw = float(holder["gduo_lr_raw"].item())
        raw = min(max(raw, math.log(self.gduo_lr_min_ratio)), math.log(self.gduo_lr_max_ratio))
        holder["gduo_lr_raw"].fill_(raw)

    def _gduo_update_from_previous(self, params, group):
        if not (self.gduo_learn_lr or self.gduo_learn_momentum):
            return
        if self.gduo_scope == "global":
            self._gduo_update_global_from_previous(params, group)
            return

        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.detach().float()
            
            # LR Hypergradient
            prev_direction = self.state[p].get("gduo_prev_direction")
            prev_lr = self.state[p].get("gduo_prev_actual_lr")
            
            if self.gduo_learn_lr and prev_direction is not None and prev_lr is not None:
                prev_direction = prev_direction.float()
                dot = float((grad * prev_direction).sum().detach().cpu())
                grad_norm = grad.norm()
                direction_norm = prev_direction.norm()
                cos_sim = float((dot / (grad_norm * direction_norm + 1e-8)).detach().cpu())
                alignment = dot / math.sqrt(max(1, grad.numel()))
                ema = self.gduo_ema_beta * self.state[p].get("gduo_ema_alignment", 0.0) + (1.0 - self.gduo_ema_beta) * alignment
                self.state[p]["gduo_ema_alignment"] = ema
                
                hypgrad = -ema * prev_lr
                clipped = _clip_scalar(hypgrad, self.gduo_hypergrad_clip)
                self.state[p]["gduo_lr_raw"].sub_(self.gduo_lr_hyper_lr * clipped)
                self._gduo_clamp_lr(p)
                self.state[p]["gduo_hypgrad_lr"] = clipped
                self.state[p]["gduo_hypgrad_lr_unclipped"] = hypgrad
                self.state[p]["gduo_alignment"] = alignment
                self.state[p]["gduo_alignment_raw"] = dot
                self.state[p]["gduo_alignment_cos"] = cos_sim
            else:
                self.state[p]["gduo_hypgrad_lr"] = float("nan")
                self.state[p]["gduo_hypgrad_lr_unclipped"] = float("nan")
                self.state[p]["gduo_alignment"] = float("nan")
                self.state[p]["gduo_alignment_raw"] = float("nan")
                self.state[p]["gduo_alignment_cos"] = float("nan")

            # Momentum Hypergradient
            prev_mu_deriv = self.state[p].get("gduo_prev_mu_deriv")
            if self.gduo_learn_momentum and prev_mu_deriv is not None and prev_lr is not None:
                dot_mu = float((grad * prev_mu_deriv.float()).sum().detach().cpu())
                mu_alignment = dot_mu / math.sqrt(max(1, grad.numel()))
                ema_mu = self.gduo_ema_beta * self.state[p].get("gduo_ema_mu_alignment", 0.0) + (1.0 - self.gduo_ema_beta) * mu_alignment
                self.state[p]["gduo_ema_mu_alignment"] = ema_mu
                hypgrad = -ema_mu * prev_lr
                clipped = _clip_scalar(hypgrad, self.gduo_hypergrad_clip)
                self.state[p]["gduo_mu_raw"].sub_(self.gduo_momentum_hyper_lr * clipped)
                self.state[p]["gduo_hypgrad_mu"] = clipped
                self.state[p]["gduo_hypgrad_mu_unclipped"] = hypgrad
                self.state[p]["gduo_mu_alignment"] = mu_alignment
                self.state[p]["gduo_mu_alignment_raw"] = dot_mu
            else:
                self.state[p]["gduo_hypgrad_mu"] = float("nan")
                self.state[p]["gduo_hypgrad_mu_unclipped"] = float("nan")
                self.state[p]["gduo_mu_alignment"] = float("nan")
                self.state[p]["gduo_mu_alignment_raw"] = float("nan")

    def _gduo_update_global_from_previous(self, params, group):
        valid_lr = []
        valid_mu = []
        for p in params:
            if p.grad is None:
                continue
            prev_lr = self.state[p].get("gduo_prev_actual_lr")
            grad = p.grad.detach().float()
            prev_direction = self.state[p].get("gduo_prev_direction")
            if self.gduo_learn_lr and prev_direction is not None and prev_lr is not None:
                valid_lr.append((p, grad, prev_direction.float(), prev_lr))
            prev_mu_deriv = self.state[p].get("gduo_prev_mu_deriv")
            if self.gduo_learn_momentum and prev_mu_deriv is not None and prev_lr is not None:
                valid_mu.append((p, grad, prev_mu_deriv.float(), prev_lr))

        if self.gduo_learn_lr and valid_lr:
            dot = sum(float((grad * direction).sum().detach().cpu()) for _, grad, direction, _ in valid_lr)
            total_numel = sum(grad.numel() for _, grad, _, _ in valid_lr)
            grad_norm_sq = sum(float((grad * grad).sum().detach().cpu()) for _, grad, _, _ in valid_lr)
            direction_norm_sq = sum(float((direction * direction).sum().detach().cpu()) for _, _, direction, _ in valid_lr)
            alignment = dot / math.sqrt(max(1, total_numel))
            cos_sim = dot / (math.sqrt(grad_norm_sq) * math.sqrt(direction_norm_sq) + 1e-8)
            ema = self.gduo_ema_beta * group.get("gduo_ema_alignment", 0.0) + (1.0 - self.gduo_ema_beta) * alignment
            group["gduo_ema_alignment"] = ema
            prev_lr = sum(x[3] for x in valid_lr) / len(valid_lr)
            hypgrad = -ema * prev_lr
            clipped = _clip_scalar(hypgrad, self.gduo_hypergrad_clip)
            group["gduo_lr_raw"].sub_(self.gduo_lr_hyper_lr * clipped)
            raw = float(group["gduo_lr_raw"].item())
            raw = min(max(raw, math.log(self.gduo_lr_min_ratio)), math.log(self.gduo_lr_max_ratio))
            group["gduo_lr_raw"].fill_(raw)
            for p, _, _, _ in valid_lr:
                self.state[p]["gduo_hypgrad_lr"] = clipped
                self.state[p]["gduo_hypgrad_lr_unclipped"] = hypgrad
                self.state[p]["gduo_alignment"] = alignment
                self.state[p]["gduo_alignment_raw"] = dot
                self.state[p]["gduo_alignment_cos"] = cos_sim

        if self.gduo_learn_momentum and valid_mu:
            dot_mu = sum(float((grad * deriv).sum().detach().cpu()) for _, grad, deriv, _ in valid_mu)
            total_numel = sum(grad.numel() for _, grad, _, _ in valid_mu)
            mu_alignment = dot_mu / math.sqrt(max(1, total_numel))
            ema_mu = self.gduo_ema_beta * group.get("gduo_ema_mu_alignment", 0.0) + (1.0 - self.gduo_ema_beta) * mu_alignment
            group["gduo_ema_mu_alignment"] = ema_mu
            prev_lr = sum(x[3] for x in valid_mu) / len(valid_mu)
            hypgrad = -ema_mu * prev_lr
            clipped = _clip_scalar(hypgrad, self.gduo_hypergrad_clip)
            group["gduo_mu_raw"].sub_(self.gduo_momentum_hyper_lr * clipped)
            for p, _, _, _ in valid_mu:
                self.state[p]["gduo_hypgrad_mu"] = clipped
                self.state[p]["gduo_hypgrad_mu_unclipped"] = hypgrad
                self.state[p]["gduo_mu_alignment"] = mu_alignment
                self.state[p]["gduo_mu_alignment_raw"] = dot_mu

    def _gduo_store_previous(self, p, direction, mu_deriv=None):
        self.state[p]["gduo_prev_direction"] = direction.detach().float().clone()
        if mu_deriv is not None:
            self.state[p]["gduo_prev_mu_deriv"] = mu_deriv.detach().float().clone()
        else:
            self.state[p]["gduo_prev_mu_deriv"] = None

    def _gduo_log(self, group, prefix="GDUO"):
        if self.gduo_log_interval <= 0 or self.gduo_step % self.gduo_log_interval != 0:
            return
        if getattr(self, "rank", 0) != 0:
            return
            
        lr_scales = [self._gduo_lr_scale(p) for p in group["params"] if self._gduo_has_meta(p)]
        momentums = [self._gduo_momentum(p) for p in group["params"] if self._gduo_has_meta(p)]
        
        if not lr_scales:
            return
            
        avg_lr_scale = sum(lr_scales) / len(lr_scales)
        min_lr_scale, max_lr_scale = min(lr_scales), max(lr_scales)
        lr_bound_frac = sum(
            scale <= self.gduo_lr_min_ratio * (1.0 + 1e-6)
            or scale >= self.gduo_lr_max_ratio * (1.0 - 1e-6)
            for scale in lr_scales
        ) / len(lr_scales)
        
        avg_momentum = sum(momentums) / len(momentums) if momentums else 0.0
        min_momentum = min(momentums) if momentums else float("nan")
        max_momentum = max(momentums) if momentums else float("nan")
        lr_hypgrads = [
            self.state[p].get("gduo_hypgrad_lr", float("nan"))
            for p in group["params"]
            if self._gduo_has_meta(p)
        ]
        lr_hypgrads_unclipped = [
            self.state[p].get("gduo_hypgrad_lr_unclipped", float("nan"))
            for p in group["params"]
            if self._gduo_has_meta(p)
        ]
        alignments = [
            self.state[p].get("gduo_alignment", float("nan"))
            for p in group["params"]
            if self._gduo_has_meta(p)
        ]
        mu_hypgrads = [
            self.state[p].get("gduo_hypgrad_mu", float("nan"))
            for p in group["params"]
            if self._gduo_has_meta(p)
        ]
        mu_hypgrads_unclipped = [
            self.state[p].get("gduo_hypgrad_mu_unclipped", float("nan"))
            for p in group["params"]
            if self._gduo_has_meta(p)
        ]
        finite_hypgrads = [x for x in lr_hypgrads if math.isfinite(x)]
        finite_unclipped = [x for x in lr_hypgrads_unclipped if math.isfinite(x)]
        finite_alignments = [x for x in alignments if math.isfinite(x)]
        finite_mu_hypgrads = [x for x in mu_hypgrads if math.isfinite(x)]
        finite_mu_unclipped = [x for x in mu_hypgrads_unclipped if math.isfinite(x)]
        avg_hypgrad = sum(finite_hypgrads) / len(finite_hypgrads) if finite_hypgrads else float("nan")
        avg_alignment = sum(finite_alignments) / len(finite_alignments) if finite_alignments else float("nan")
        avg_mu_hypgrad = sum(finite_mu_hypgrads) / len(finite_mu_hypgrads) if finite_mu_hypgrads else float("nan")
        clip_frac = (
            sum(abs(x) >= self.gduo_hypergrad_clip for x in finite_unclipped) / len(finite_unclipped)
            if finite_unclipped and self.gduo_hypergrad_clip > 0
            else 0.0
        )
        mu_clip_frac = (
            sum(abs(x) >= self.gduo_hypergrad_clip for x in finite_mu_unclipped) / len(finite_mu_unclipped)
            if finite_mu_unclipped and self.gduo_hypergrad_clip > 0
            else 0.0
        )
        
        print(
            f"[{prefix}] step={self.gduo_step} "
            f"lr_scale_avg={avg_lr_scale:.6g} (min={min_lr_scale:.6g}, max={max_lr_scale:.6g}) "
            f"lr_bound_frac={lr_bound_frac:.3g} "
            f"momentum_avg={avg_momentum:.6g} (min={min_momentum:.6g}, max={max_momentum:.6g}) "
            f"alignment_avg={avg_alignment:.6g} "
            f"hypgrad_lr_avg={avg_hypgrad:.6g} "
            f"clip_frac={clip_frac:.3g} "
            f"hypgrad_mu_avg={avg_mu_hypgrad:.6g} "
            f"mu_clip_frac={mu_clip_frac:.3g}"
        )
