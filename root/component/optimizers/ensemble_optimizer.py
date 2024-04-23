from typing import Optional, Callable


class EnsembleOptimizer:
    def __init__(self, individual_optim: Callable, param: list[dict], lr: float, kwargs: Optional):
        self.optim = [
            individual_optim(list(p), lr, **kwargs) for p in param
        ]

    def zero_grad(self) -> None:
        for opt in self.optim:
            opt.zero_grad()
        return

    def step(self) -> None:
        for opt in self.optim:
            opt.step()
        return

    def state_dict(self) -> list[dict]:
        return [opt.state_dict() for opt in self.optim]

    def load_state_dict(self, state_dict_lst: list[dict]) -> None:
        for opt, sd in zip(self.optim, state_dict_lst):
            opt.load_state_dict(sd)
        return

    @property
    def param_groups(self) -> list[dict]:
        pg = []
        for opt in self.optim:
            pg += opt.param_groups
        return pg
