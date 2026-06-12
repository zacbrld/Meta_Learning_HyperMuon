import re

def fix_optimizer(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Change __init__ signature
    content = content.replace("def __init__(self, muon_params, adamw_params,", "def __init__(self, muon_params, adamw_params=None,")
    content = content.replace("def __init__(self, muon_params, model, adamw_params,", "def __init__(self, muon_params, model, adamw_params=None,")

    # Remove self.adamw_opt
    content = re.sub(r"        if adamw_lr is None.*?self\.adamw_opt.*?None\n", "", content, flags=re.DOTALL)
    
    # Merge params in init
    if "def __init__(self, muon_params, adamw_params=None," in content:
        init_merge = """        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        Optimizer.__init__(self, params, defaults)
        
        for p in muon_params:
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]["use_muon"] = True
            else:
                self.state[p]["use_muon"] = False
        for p in adamw_params:
            self.state[p]["use_muon"] = False"""
        content = re.sub(r"        Optimizer\.__init__\(self, muon_params, defaults\)", init_merge, content)
        
    elif "def __init__(self, muon_params, model, adamw_params=None," in content:
        init_merge = """        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        Optimizer.__init__(self, params, defaults)
        
        for p in muon_params:
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]["use_muon"] = True
            else:
                self.state[p]["use_muon"] = False
        for p in adamw_params:
            self.state[p]["use_muon"] = False"""
        content = re.sub(r"        Optimizer\.__init__\(self, muon_params, defaults\)", init_merge, content)

    # In step(), for gate update, only use params with use_muon=True
    content = content.replace("            for p in group[\"params\"]:\n                if p.grad is None:", "            for p in group[\"params\"]:\n                if p.grad is None or not self.state[p].get('use_muon', False):")
    
    # In step(), for main update, add condition
    content = content.replace("            for p in group[\"params\"]:\n                if p.grad is None:\n                    continue\n                grad = p.grad.detach()", 
"""            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if not self.state[p].get("use_muon", False):
                    # AdamW fallback
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['adam_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['adam_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['use_muon'] = False
                    
                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['adam_exp_avg'], state['adam_exp_avg_sq']
                    beta1, beta2 = group.get('adamw_betas', group['betas'])
                    eps = group.get('adamw_eps', group['eps'])
                    wd = group.get('adamw_wd', group['weight_decay'])
                    lr = group.get('adamw_lr', group['lr'])
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    
                    p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    continue
""")

    content = content.replace("            for index, p in enumerate(group[\"params\"]):\n                if p.grad is None:\n                    continue\n                grad = p.grad.detach()", 
"""            for index, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if not self.state[p].get("use_muon", False):
                    # AdamW fallback
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['adam_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['adam_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['use_muon'] = False
                    
                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['adam_exp_avg'], state['adam_exp_avg_sq']
                    beta1, beta2 = group.get('adamw_betas', group['betas'])
                    eps = group.get('adamw_eps', group['eps'])
                    wd = group.get('adamw_wd', group['weight_decay'])
                    lr = group.get('adamw_lr', group['lr'])
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    
                    p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    continue
""")

    # Remove self.adamw_opt.step()
    content = re.sub(r"        if self\.adamw_opt is not None:\n            self\.adamw_opt\.step\(\)\n", "", content)

    with open(file_path, 'w') as f:
        f.write(content)

fix_optimizer('external/llm-baselines/src/optim/gating.py')
fix_optimizer('external/llm-baselines/src/optim/muon_newton_gate.py')

