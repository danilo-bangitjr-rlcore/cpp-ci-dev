import math
from typing import Any, Callable, overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT


class ArmijoAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        c: float = 0.1,
        tau: float = 0.5,
        beta: float = 0.1,
        max_backtracks: int = 10,
        min_lr: float = 1e-4,
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': False,
            'c': c,
            'tau': tau,
            'beta': beta,
            'max_backtracks': max_backtracks,
            'min_lr': min_lr,
        }
        super().__init__(params, defaults)

        self.state_idx = {}
        self.max_p_len = max(
            len(group['params']) for group in self.param_groups
        )

        for gi, group in enumerate(self.param_groups):
            for pi in range(len(group['params'])):
                self.state_idx[gi * self.max_p_len + pi] = {}

    def __setstate__(self, state: dict[str, Any]):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @overload
    def step(self, closure: None = ...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is None:
            raise ValueError("Armijo line search requires closure function")

        initial_loss = None
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
                    state['prev_params'] = p.data.clone()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                step = state['step']

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                bias_correction2_sqrt = self._dispatch_sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                step_direction = exp_avg / denom

                alpha = group['lr'] / bias_correction1
                old_params = p.data.clone()

                initial_loss = float(closure())

                for _j in range(group['max_backtracks']):
                    p.data.copy_(old_params)
                    p.data.add_(step_direction, alpha=-alpha)

                    current_loss = float(closure())
                    improvement = initial_loss - current_loss
                    grad_norm = torch.norm(grad * step_direction).item()
                    expected_improvement = group['c'] * alpha * grad_norm
                    armijo_condition = improvement >= group['beta'] * expected_improvement

                    if armijo_condition or alpha < group['min_lr']:
                        alpha = max(alpha, group['min_lr'])
                        break

                    alpha *= group['tau']

                state['prev_params'].copy_(p.data)

        return initial_loss

    def _dispatch_sqrt(self, x: float | torch.Tensor) -> float | torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.sqrt()
        else:
            return math.sqrt(x)
