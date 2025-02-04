from typing import Any, Callable

import torch
from torch.optim.optimizer import ParamsT


class EnsembleOptimizer:
    def __init__(
        self,
        individual_optim: Callable,
        param: ParamsT,
        kwargs: dict[str, Any],
    ):
        self.optim = []
        if isinstance(param, dict):  # handle vmap case
            param_groups = []
            for _name, p in param.items():
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    new_param = p.clone().detach().requires_grad_()
                    param_groups.append(new_param)

            if param_groups:
                self.optim.append(individual_optim(param_groups, **kwargs))
                # store original parameters for gradient synchronization
                self.original_params = {
                    name: p
                    for name, p in param.items()
                    if isinstance(p, torch.Tensor) and p.requires_grad
                }
        else:  # handle non-vmap case
            for p in param:
                param_list = []
                for orig_p in list(p):
                    if orig_p.requires_grad:
                        param_list.append(orig_p.clone().detach().requires_grad_())
                if param_list:
                    self.optim.append(individual_optim(param_list, **kwargs))

    def zero_grad(self) -> None:
        for opt in self.optim:
            opt.zero_grad()
        return

    def step(self, *args: Any, **kwargs: Any) -> None:
        for opt in self.optim:
            opt.step(*args, **kwargs)

        # sync parameters after optimization step if using vmap
        if hasattr(self, 'original_params'):
            for opt in self.optim:
                for group in opt.param_groups:
                    for i, p in enumerate(group['params']):
                        name = list(self.original_params.keys())[i]
                        self.original_params[name].data.copy_(p.data)
        return

    def state_dict(self) -> list[dict]:
        return [opt.state_dict() for opt in self.optim]

    def load_state_dict(self, state_dict_lst: list[dict]) -> None:
        for opt, sd in zip(self.optim, state_dict_lst, strict=True):
            opt.load_state_dict(sd)
        return

    @property
    def param_groups(self) -> list[dict]:
        pg = []
        for opt in self.optim:
            pg += opt.param_groups
        return pg
