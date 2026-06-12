import math
import torch
from torch.optim.optimizer import Optimizer
from optim.gduo_meta import GDUOMetaMixin

class AdamWGDUO(Optimizer, GDUOMetaMixin):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, 
                 gduo_learn_lr=True, gduo_learn_momentum=False,
                 gduo_ema_beta=0.9,
                 gduo_lr_hyper_lr=1e-3, gduo_momentum_hyper_lr=1e-3,
                 gduo_hypergrad_clip=1.0, gduo_lr_min_ratio=0.25, gduo_lr_max_ratio=4.0,
                 gduo_mu_min=0.0, gduo_mu_max=0.99, gduo_log_interval=0,
                 gduo_scope="tensor"):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        Optimizer.__init__(self, params, defaults)
        self._init_gduo_meta(
            learn_lr=gduo_learn_lr,
            learn_momentum=gduo_learn_momentum,
            lr_hyper_lr=gduo_lr_hyper_lr,
            momentum_hyper_lr=gduo_momentum_hyper_lr,
            hypergrad_clip=gduo_hypergrad_clip,
            lr_min_ratio=gduo_lr_min_ratio,
            lr_max_ratio=gduo_lr_max_ratio,
            mu_min=gduo_mu_min,
            mu_max=gduo_mu_max,
            ema_beta=gduo_ema_beta,
            log_interval=gduo_log_interval,
            scope=gduo_scope,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            self._gduo_update_from_previous(group['params'], group)
            
            beta2 = group['betas'][1]
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                actual_lr = self._gduo_actual_lr(p, group)
                beta1 = group['betas'][0]
                if self.gduo_learn_momentum:
                    beta1 = self._gduo_momentum(p)
                    dmu_draw = self._gduo_dmu_draw(p)
                else:
                    dmu_draw = 0.0
                grad = p.grad.detach()
                state = self.state[p]
                param_for_decay = p.detach().clone() if group['weight_decay'] > 0 else None
                
                if 'exp_avg' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # We need the previous exp_avg to compute finite difference derivative for momentum
                prev_exp_avg = exp_avg.clone() if self.gduo_learn_momentum else None

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size_factor = math.sqrt(bias_correction2) / bias_correction1
                adaptive_dir = (exp_avg / denom) * step_size_factor
                
                # Compute momentum derivative
                mu_deriv = None
                if self.gduo_learn_momentum:
                    # FD derivative of adaptive_dir w.r.t beta1. Decoupled weight decay
                    # does not depend on beta1, so it has zero momentum derivative.
                    # beta1 changes by + eps * dmu_draw
                    eps_fd = 1e-3
                    beta1_plus = beta1 + eps_fd * dmu_draw
                    beta1_minus = beta1 - eps_fd * dmu_draw
                    
                    exp_avg_plus = prev_exp_avg * beta1_plus + grad * (1 - beta1_plus)
                    exp_avg_minus = prev_exp_avg * beta1_minus + grad * (1 - beta1_minus)
                    
                    bc1_plus = 1 - beta1_plus ** state['step']
                    bc1_minus = 1 - beta1_minus ** state['step']
                    
                    step_size_plus = math.sqrt(bias_correction2) / bc1_plus
                    step_size_minus = math.sqrt(bias_correction2) / bc1_minus
                    
                    update_plus = (exp_avg_plus / denom) * step_size_plus
                    update_minus = (exp_avg_minus / denom) * step_size_minus
                        
                    mu_deriv = (update_plus - update_minus) / (2.0 * eps_fd)

                update_dir = adaptive_dir
                if group['weight_decay'] > 0:
                    update_dir = adaptive_dir + param_for_decay * group['weight_decay']
                    p.mul_(1.0 - actual_lr * group['weight_decay'])
                p.sub_(adaptive_dir, alpha=actual_lr)
                
                self._gduo_store_previous(p, update_dir, mu_deriv)
                self.state[p]['gduo_prev_actual_lr'] = actual_lr
                
            self._gduo_log(group, prefix="AdamW-GDUO")
            
        self.gduo_step += 1
        return loss

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
