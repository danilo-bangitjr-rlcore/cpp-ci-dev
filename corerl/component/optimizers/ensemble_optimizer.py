from typing import Any, Callable
from torch.optim.optimizer import ParamsT

class EnsembleOptimizer:
    def __init__(
        self,
        individual_optim: Callable,
        param: ParamsT,
        kwargs: dict[str, Any],
    ):
        self.optim = [individual_optim(list(p), **kwargs) for p in param]

    def zero_grad(self) -> None:
        for opt in self.optim:
            opt.zero_grad()
        return

    def step(self, *args: Any, **kwargs: Any) -> None:
        for opt in self.optim:
            opt.step(*args, **kwargs)
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
