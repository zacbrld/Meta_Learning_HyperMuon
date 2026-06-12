"""
Here is an original implementation of Muon and Distributed Muon. 
_Note_ that 'class Muon'  is an old version which does not include weight decay for matrix params
the newer version with weight decay is in the 'class DistributedMuon'.
Source: https://github.com/KellerJordan/modded-nanogpt
Source: https://github.com/toothacher17/Megatron-LM/tree/moonshot/distributedmuon-impl
"""

import math
import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from .gduo_meta import GDUOMetaMixin
from .schedule import cos_inf_schedule, wsd_schedule


# copy from https://github.com/KellerJordan/Muon/tree/master
# @torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


# @torch.compile
# def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
#     """
#     Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
#     quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
#     of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
#     zero even beyond the point where the iteration no longer converges all the way to one everywhere
#     on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
#     where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
#     performance at all relative to UV^T, where USV^T = G is the SVD.
#     """
#     assert len(G.shape) == 2
#     a, b, c = (3.4445, -4.7750, 2.0315)
#     X = G.bfloat16()
#     X /= X.norm() + eps  # ensure top singular value <= 1
#     if G.size(0) > G.size(1):
#         X = X.T
#     for _ in range(steps):
#         A = X @ X.T
#         B = b * A + c * A @ A
#         X = a * X + B @ X
#     if G.size(0) > G.size(1):
#         X = X.T
#     return X


def normalize_range(range: Tuple[int, int], start):
    return (range[0] - start, range[1] - start)


class MuonDistMeta:

    # which buffer and bucket param belongs to
    buffer_idx: int = 0
    bucket_idx: int = 0
    # param shape after tp
    shape: torch.Size = None
    # param location in global buffer
    global_range: Tuple[int, int] = None
    tp_split_dim: int = -1
    # param location in global buffer (current dp slice)
    local_range: Tuple[int, int] = None

    def __init__(
        self,
        buffer_idx: int,
        bucket_idx: int,
        shape: torch.Size,
        global_range: Tuple[int, int],
        tp_split_dim: int,
    ):
        self.buffer_idx = buffer_idx
        self.bucket_idx = bucket_idx
        self.shape = shape
        self.global_range = global_range
        self.tp_split_dim = tp_split_dim

    def set_local_buffer_range(self, local_buffer_range: Tuple[int, int]):
        start = max(self.global_range[0], local_buffer_range[0])
        end = min(self.global_range[1], local_buffer_range[1])
        self.local_range = (
            (start, end)
            if start < end
            else (local_buffer_range[0], local_buffer_range[0])
        )


# adjust LR based on: https://github.com/MoonshotAI/Moonlight
def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape[:2]
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


# copy from https://github.com/KellerJordan/Muon/tree/master and support distributed solution
class DistributedMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        param_groups,
        lr=2e-2,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        super().__init__(param_groups, defaults)
        self.distributed_mode = False
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for group in self.param_groups:
            for p in group["params"]:
                # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
                if p.ndim >= 2 and p.size(0) < 10000:
                    self.state[p]["use_muon"] = True
                else:
                    self.state[p]["use_muon"] = False

    def enable_distributed_mode(
        self,
        global_buffer_sizes,
        dist_group,
        tp_group,
        dist_metas: Dict[torch.nn.Parameter, MuonDistMeta],
    ):
        """
        enable distributed mode
        Args:
            global_buffer_size: global buffer size
            dist group: optimizer sharding group
            tp group: param tp group
            dist metas: dist metas for all param
        """

        self.global_buffer_sizes = global_buffer_sizes
        self.dist_group = dist_group
        self.tp_group = tp_group
        self.dist_metas = dist_metas

        world_size = dist.get_world_size(dist_group)
        rank = dist.get_rank(dist_group)

        # calc local buffer range
        self.local_buffer_sizes = []
        self.local_buffer_ranges = []
        for bucket_sizes in global_buffer_sizes:
            local_bucket_sizes = []
            local_bucket_ranges = []
            for global_bucket_size, bucket_offset in bucket_sizes:
                assert global_bucket_size % world_size == 0
                local_buffer_size = global_bucket_size // world_size
                local_buffer_start = local_buffer_size * rank + bucket_offset
                local_buffer_range = (
                    local_buffer_start,
                    local_buffer_start + local_buffer_size,
                )
                local_bucket_sizes.append(local_buffer_size)
                local_bucket_ranges.append(local_buffer_range)

            self.local_buffer_sizes.append(local_bucket_sizes)
            self.local_buffer_ranges.append(local_bucket_ranges)

        # calc local range for params
        for dist_meta in dist_metas.values():
            local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                dist_meta.bucket_idx
            ]
            dist_meta.set_local_buffer_range(local_buffer_range)

        self.distributed_mode = True

    def step(self):

        dtype = torch.bfloat16
        device = torch.cuda.current_device()

        ns_inputs = {}

        # update muon momentum first
        for group in self.param_groups:

            momentum = group["momentum"]
            params = group["params"]

            for p in params:

                if not self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                # 1-dim grad for distributed mode
                assert self.distributed_mode or g.dim() == 2

                # prepare muon buffer in state
                state = self.state[p]
                if not "muon_buffer" in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                buf = state["muon_buffer"]
                buf.mul_(momentum).add_(g)

                # save to ns input
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                ns_inputs[p] = g.bfloat16()

        # rewrite ns_inputs if distributed
        if self.distributed_mode:

            # initialize buffers
            ns_input_local_buffers = [
                [
                    torch.empty((local_buffer_size), device=device, dtype=dtype)
                    for local_buffer_size in local_bucket_sizes
                ]
                for local_bucket_sizes in self.local_buffer_sizes
            ]
            ns_input_global_buffers = [
                [
                    torch.empty((global_buffer_size), device=device, dtype=dtype)
                    for (global_buffer_size, bucket_offset) in global_bucket_sizes
                ]
                for global_bucket_sizes in self.global_buffer_sizes
            ]

            # fill ns input data to local buffer
            for param, ns_input in ns_inputs.items():
                dist_meta = self.dist_metas[param]
                ns_input_local_buffer = ns_input_local_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_range = normalize_range(
                    dist_meta.local_range, local_buffer_range[0]
                )
                ns_input_local_buffer[local_range[0] : local_range[1]].copy_(
                    ns_input.view(-1)
                )

            # all gather buffers
            for ns_input_global_buffer, ns_input_local_buffer in zip(
                ns_input_global_buffers, ns_input_local_buffers
            ):
                for ns_input_global_bucket, ns_input_local_bucket in zip(
                    ns_input_global_buffer, ns_input_local_buffer
                ):
                    dist.all_gather_into_tensor(
                        ns_input_global_bucket,
                        ns_input_local_bucket,
                        group=self.dist_group,
                    )

            # overwrite ns input
            for p in ns_inputs.keys():
                dist_meta = self.dist_metas[p]
                ns_input_global_buffer = ns_input_global_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                global_range = dist_meta.global_range
                offset = self.global_buffer_sizes[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ][1]
                ns_inputs[p] = ns_input_global_buffer[
                    global_range[0] - offset : global_range[1] - offset
                ].view(dist_meta.shape)

            # set tp info
            tp_world_size = dist.get_world_size(self.tp_group)
            tp_rank = dist.get_rank(self.tp_group)

        # update muon momentum first
        for group in self.param_groups:

            # if not group.get('use_muon', False):
            #     continue

            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]
            params = group["params"]

            for p in params:

                if not self.state[p].get("use_muon", False):
                    continue

                ns_input = ns_inputs[p]
                tp_split_dim = -1

                if self.distributed_mode:
                    dist_meta = self.dist_metas[p]
                    tp_split_dim = dist_meta.tp_split_dim

                # gather tensor parallel ( if tp )
                if tp_split_dim != -1:
                    ns_input_shards = [
                        torch.empty_like(ns_input) for _ in range(tp_world_size)
                    ]
                    dist.all_gather(ns_input_shards, ns_input, self.tp_group)
                    ns_input = torch.cat(ns_input_shards, dim=tp_split_dim)

                # calc update
                update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)

                # only local tp part
                if tp_split_dim != -1:
                    update = update.chunk(tp_world_size, dim=tp_split_dim)[tp_rank]

                # only local buffer part
                if self.distributed_mode:
                    local_range_in_global_range = normalize_range(
                        dist_meta.local_range, dist_meta.global_range[0]
                    )
                    update = update.reshape(-1)[
                        local_range_in_global_range[0] : local_range_in_global_range[1]
                    ]

                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)

                #  adjust lr and apply update
                adjusted_lr = adjust_lr_wd_for_muon(
                    lr, matched_adamw_rms, ns_input.shape
                )
                p.data.add_(update, alpha=-adjusted_lr)

        # use adam for other params
        for group in self.param_groups:

            # if group.get('use_muon', False):
            #     continue

            # init step
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            step = group["step"]
            params = group["params"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in params:

                if self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                state = self.state[p]

                if "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                buf1 = state["adamw_exp_avg"]
                buf2 = state["adamw_exp_avg_sq"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)


class Muon(GDUOMetaMixin, torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
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

        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]["use_muon"] = True
            # self.state[p]["use_muon"] = True
            else:
                self.state[p]["use_muon"] = False
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

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
        if not params_with_lr:
            return {}
            
        avg_lr_scale = sum(self._gduo_lr_scale(p) for p in params_with_lr) / len(params_with_lr)
        avg_actual_lr = sum(self._gduo_actual_lr(p, group) for p in params_with_lr) / len(params_with_lr)
        
        metrics = {
            "gduo_lr_scale_avg": avg_lr_scale,
            "gduo_actual_lr_avg": avg_actual_lr,
        }
        if self.gduo_learn_momentum:
            avg_momentum = sum(self._gduo_momentum(p) for p in params_with_lr) / len(params_with_lr)
            metrics["gduo_momentum_avg"] = avg_momentum
        return metrics


    def step(self):
        self.gduo_step += 1
        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            self._gduo_update_from_previous(params, group)

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in params)
            device = params[0].device if params else group["params"][0].device
            updates_flat = torch.zeros(
                total_params, device=device, dtype=torch.float32
            )
            derivs_flat = (
                torch.zeros(total_params, device=device, dtype=torch.float32)
                if self.gduo_learn_momentum
                else None
            )
            curr_idx = 0
            for i, p in enumerate(params):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    momentum = self._gduo_momentum(p)
                    
                    g = p.grad
                    assert g is not None
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    old_buf = buf.clone() if self.gduo_learn_momentum else None
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
                            p.grad.view_as(g) if p.grad.ndim == 2 else p.grad.view(g.size(0), -1),
                            old_buf,
                            momentum,
                            group["nesterov"],
                            group["ns_steps"],
                            eps,
                        )
                        deriv.mul_(dmu_draw)
                        derivs_flat[curr_idx : curr_idx + p.numel()] = deriv.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
                if derivs_flat is not None:
                    dist.all_reduce(derivs_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
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
                
                curr_idx += p.numel()
                
            self._gduo_log(group, prefix="GDUO-Muon")

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = (
                group["adamw_lr_ratio"] * group["lr"]
            )  # in order for lr schedule to work
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


def separate_params(param_groups):
    param_groups_2d = []
    param_groups_non2d = []
    total_param_2d_count = 0
    total_param_non2d_count = 0

    # Check if param_groups is a list of dicts or list of params
    if (
        isinstance(param_groups, list) and isinstance(param_groups[0], dict)
    ) or isinstance(param_groups, dict):
        if isinstance(param_groups, dict):
            param_groups = [param_groups]
        # param_groups is a list of dicts
        for group in param_groups:
            (
                params_2d,
                params_non2d,
                param_2d_count,
                param_non2d_count,
            ) = separate_params(group["params"])
            param_group_2d = {"params": params_2d}
            param_group_non2d = {"params": params_non2d}
            # Copy the group dict and replace the 'params' key with the separated params
            for k in group.keys():
                if k != "params":
                    param_group_2d[k] = group[k]
                    param_group_non2d[k] = group[k]

            param_groups_2d.append(param_group_2d)
            param_groups_non2d.append(param_group_non2d)
            total_param_2d_count += param_2d_count
            total_param_non2d_count += param_non2d_count

        return (
            param_groups_2d,
            param_groups_non2d,
            total_param_2d_count,
            total_param_non2d_count,
        )

    elif isinstance(param_groups, list) and isinstance(param_groups[0], torch.Tensor):
        params_2d = []
        params_non2d = []
        param_group = param_groups
        # param_group is a list of param tensors
        for param in param_group:
            if param.ndim == 2:
                params_2d.append(param)
            else:
                params_non2d.append(param)
        return params_2d, params_non2d, len(params_2d), len(params_non2d)
    else:
        breakpoint()


class CombinedScheduler:
    """
    CombinedScheduler implements a scheduler for the Muon optimizer: it leverages both Muon and AdamW learning rates, and applies the same sort of scheduler for both of them.

    Arguments:
        optimizer: Muon optimizer.
        cfg: arguments used for schedulers.
        muon_lr_key: defaults["lr"] is responsible for the Muon learning rate.
        adamw_lr_key: defaults["adamw_r"] is responsible for the AdamW learning rate.
    """

    def __init__(self, optimizer, cfg, muon_lr_key="lr", adamw_lr_key="adamw_lr"):
        self.schedulers = []
        scheduler_map = {
            "cos": torch.optim.lr_scheduler.OneCycleLR,
            "linear": torch.optim.lr_scheduler.OneCycleLR,
            "cos_inf": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cos_inf_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    n_inf=cfg.cos_inf_steps,
                    div_factor=1e2,
                    final_div_factor=0.1,
                ),
            ),
            "wsd": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                wsd_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    fract_decay=cfg.wsd_fract_decay,
                    init_div_factor=1e2,
                    final_lr_factor=cfg.wsd_final_lr_scale,
                    decay_type=cfg.decay_type,
                ),
            ),
        }

        if not optimizer.param_groups:
            return

        # Build a single scheduler covering all param_groups
        scheduler_cls = scheduler_map.get(cfg.scheduler, None)
        if scheduler_cls:
            if cfg.scheduler in ["cos", "linear"]:
                max_lrs = []
                for group in optimizer.param_groups:
                    lr_key = muon_lr_key if muon_lr_key in group else adamw_lr_key
                    max_lrs.append(group.get(lr_key, cfg.lr))
                scheduler = scheduler_cls(
                    optimizer,
                    max_lr=max_lrs,
                    total_steps=cfg.iterations,
                    pct_start=cfg.warmup_steps / cfg.iterations,
                    anneal_strategy=cfg.scheduler,
                    cycle_momentum=False,
                    div_factor=1e2,
                    final_div_factor=1,
                )
                self.schedulers.append(scheduler)
            else:
                # For wsd / cos_inf: one LambdaLR per param_group
                for group in optimizer.param_groups:
                    lr_key = muon_lr_key if muon_lr_key in group else adamw_lr_key
                    if lr_key in group:
                        scheduler = scheduler_cls(
                            optimizer, group.get(lr_key, cfg.lr)
                        )
                        self.schedulers.append(scheduler)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        state_dict = {}
        for i, scheduler in enumerate(self.schedulers):
            state_dict[f"scheduler_{i}"] = scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for i, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(state_dict[f"scheduler_{i}"])
