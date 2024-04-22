import torch

class EnsembleOptimizer:
    def __init__(self, individual_optim: torch.optim.Optimizer, param: list[torch.Tensor], lr: float, kwargs):
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

    def load_state_dict(self, state_dict_list: list[dict]) -> None:
        for opt, sd in zip(self.optim, state_dict_list):
            opt.load_state_dict(sd)
        return
