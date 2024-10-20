import torch
import copy
import math
from typing import Any, Callable, overload



class CustomAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params: torch.optim.optimizer.ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': False,
        }
        super(CustomAdam, self).__init__(params, defaults)

        self.state_idx = {}
        self.max_p_len = max(
            len(group['params']) for group in self.param_groups
        )

        for gi, group in enumerate(self.param_groups):
            for pi in range(len(group['params'])):
                self.state_idx[gi * self.max_p_len + pi] = {}

    def __setstate__(self, state: dict[str, Any]):
        super(CustomAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @overload
    def step(self, closure: None = ...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state_idx[gi * self.max_p_len + pi]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                step = state['step']
                lr = group['lr']

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                bias_correction2_sqrt = self._dispatch_sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

    def _dispatch_sqrt(self, x: float | torch.Tensor) -> float | torch.Tensor:
        if not torch.jit.is_scripting() and isinstance(x, torch.Tensor): # type: ignore
            return x.sqrt()
        else:
            return math.sqrt(x)

    def state_dict(self) -> dict[str, Any]:
        hard_copy = {
            'state': {},
            'param_groups': []
        }
        with torch.no_grad():
            for s in self.state_idx:
                hard_copy['state'][s] = {}
                if len(self.state_idx[s].keys()) > 0:
                    hard_copy['state'][s]['step'] = copy.deepcopy(self.state_idx[s]['step'])
                    hard_copy['state'][s]['exp_avg'] = copy.deepcopy(self.state_idx[s]['exp_avg'].data.detach())
                    hard_copy['state'][s]['exp_avg_sq'] = copy.deepcopy(self.state_idx[s]['exp_avg_sq'].data.detach())
            for group in self.param_groups:
                hard_copy['param_groups'].append({})
                hard_copy['param_groups'][-1]['lr'] = copy.deepcopy(group['lr'])
                hard_copy['param_groups'][-1]['betas'] = copy.deepcopy(group['betas'])
                hard_copy['param_groups'][-1]['eps'] = copy.deepcopy(group['eps'])
                hard_copy['param_groups'][-1]['weight_decay'] = copy.deepcopy(group['weight_decay'])
                hard_copy['param_groups'][-1]['amsgrad'] = copy.deepcopy(group['amsgrad'])
        return hard_copy

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        def inposition_fill(ref, data):
            if len(ref.data.size()) == 2:
                idx = torch.arange(ref.data.size()[1])
                idx = torch.tile(idx, (ref.data.size()[0], 1))
                ref.data.scatter_(1, idx, data)
            elif len(ref.data.size()) == 1:
                idx = torch.arange(ref.data.size()[0])
                ref.data.scatter_(0, idx, data)
            else:
                ref.data.fill_(data)

        with torch.no_grad():
            for g in range(len(state_dict['param_groups'])):
                self.param_groups[g]['lr'] = state_dict['param_groups'][g]['lr']
                self.param_groups[g]['betas'] = state_dict['param_groups'][g]['betas']
                self.param_groups[g]['eps'] = state_dict['param_groups'][g]['eps']
                self.param_groups[g]['weight_decay'] = state_dict['param_groups'][g]['weight_decay']
                self.param_groups[g]['amsgrad'] = state_dict['param_groups'][g]['amsgrad']

            s_keys = self.state_idx.keys()
            for s in s_keys:
                if len(state_dict['state'][s].keys()) > 0:
                    self.state_idx[s]['step'] = state_dict['state'][s]['step']
                    inposition_fill(self.state_idx[s]['exp_avg'], state_dict['state'][s]['exp_avg'])
                    inposition_fill(self.state_idx[s]['exp_avg_sq'], state_dict['state'][s]['exp_avg_sq'])
                else:
                    ssk = list(self.state_idx[s].keys())
                    for k in ssk:
                        del self.state_idx[s][k]
