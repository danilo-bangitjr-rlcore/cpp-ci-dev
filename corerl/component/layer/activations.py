import torch.nn as nn
import torch


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class Clamp(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self._min = min
        self._max = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self._min, self._max)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TanhShift(nn.Module):
    def __init__(self, shift, denom, high, low):
        super().__init__()
        self._shift = shift
        self._denom = denom

        self._high = high
        self._low = low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tanh_out = torch.tanh((x + self._shift * self._denom) / self._denom)
        normalized = (tanh_out + 1) / 2
        return normalized * (self._high - self._low) + self._low


def init_activation(cfg) -> nn.Module:
    name = cfg["name"]
    args = cfg.get("args", tuple())
    kwargs = cfg.get("kwargs", {})

    activations = {
        "exp": Exp,
        "clamp": Clamp,
        "tanh_shift": TanhShift,
        "none": Identity,
        "identity": Identity,
        None: Identity,
        "tanh_shrink": torch.nn.Tanhshrink,
        "soft_shrink": torch.nn.Softshrink,
        "soft_sign": torch.nn.Softsign,
        "threshold": torch.nn.Threshold,
        "relu": torch.nn.ReLU,
        "relu6": torch.nn.ReLU6,
        "prelu": torch.nn.PReLU,
        "rrelu": torch.nn.RReLU,
        "celu": torch.nn.CELU,
        "selu": torch.nn.SELU,
        "gelu": torch.nn.GELU,
        "softplus": torch.nn.Softplus,
        "softmax": torch.nn.Softmax,
        "softmin": torch.nn.Softmin,
        "sigmoid": torch.nn.Sigmoid,
        "log_sigmoid": torch.nn.LogSigmoid,
        "hard_sigmoid": torch.nn.Hardsigmoid,
        "elu": torch.nn.ELU,
        "hard_tanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
    }

    if name.lower() not in activations.keys():
        raise NotImplementedError(
            f"unknown activation function {name}, known activations include " +
            f"{list(activations.keys())}",
        )

    return activations[name.lower()](*args, **kwargs)


