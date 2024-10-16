import torch.nn as nn
import torch
from fractions import Fraction
from decimal import Decimal


def _fuzzy_indicator(
    x: torch.Tensor,
    eta: torch.Tensor | float,
) -> torch.Tensor:
    return (x <= eta) * x + (x > eta)


class FTA(nn.Module):
    def __init__(
        self,
        eta: float,
        lower: float,
        upper:float,
        delta: float,
    ):
        super().__init__()
        self._eta = eta
        self._lower = lower
        self._upper = upper
        self._delta = delta

        # Trick to ensure upper - lower is divisible by delta, which is
        # required by FTA
        l = Fraction(Decimal(f"{self._lower}"))
        u = Fraction(Decimal(f"{self._upper}"))
        δ = Fraction(Decimal(f"{self._delta}"))
        assert (u - l) % δ == Fraction(0, 1)

        n = (self._upper - self._lower) // self._delta
        self._c = nn.Parameter(
            torch.arange(self._lower, self._upper, self._delta),
            requires_grad=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(*z.shape, 1)

        z = 1 - _fuzzy_indicator(
            torch.clip(self._c - z, min=0, max=None) +
            torch.clip(z - self._delta - self._c, min=0, max=None),
            self._eta,
        )

        return z.view(*z.shape[:-2], -1)


class Multiply(nn.Module):
    """
    Multiply is an arbitrary scalar multiplication activation function
    """
    def __init__(self, by):
        super().__init__()
        self._by = by

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._by


class Functional(nn.Module):
    """
    Functional converts a function from torch.nn.functional to a
    torch.nn.Module
    """
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        if isinstance(f, str):
            self._f = getattr(torch.nn.functional, f)
        else:
            self._f = f

        self._args = args
        self._kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._f(x, *self._args, **self._kwargs)


class Bias(nn.Module):
    """
    Bias is an arbitrary scalar addition activation function
    """
    def __init__(self, value=1):
        super().__init__()
        self._value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._value


class Identity(nn.Module):
    """
    Identity is the identity activation function
    """
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
        "bias": Bias,
        "multiply": Multiply,
        "functional": Functional,
        "exp": lambda *args, **kwargs: Functional(torch.exp, *args, **kwargs),
        "clamp": lambda *args, **kwargs: Functional(
            torch.clamp, *args, **kwargs,
        ),
        "tanh_shift": TanhShift,
        "none": Identity,
        "identity": Identity,
        None: Identity,
        "glu": torch.nn.GLU,
        "tanh_shrink": torch.nn.Tanhshrink,
        "soft_shrink": torch.nn.Softshrink,
        "hard_shrink": torch.nn.Hardshrink,
        "soft_sign": torch.nn.Softsign,
        "threshold": torch.nn.Threshold,
        "relu": torch.nn.ReLU,
        "relu6": torch.nn.ReLU6,
        "prelu": torch.nn.PReLU,
        "rrelu": torch.nn.RReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "celu": torch.nn.CELU,
        "selu": torch.nn.SELU,
        "gelu": torch.nn.GELU,
        "softplus": torch.nn.Softplus,
        "log_softmax": torch.nn.LogSoftmax,
        "softmax": torch.nn.Softmax,
        "softmin": torch.nn.Softmin,
        "silu": torch.nn.SiLU,
        "mish": torch.nn.Mish,
        "sigmoid": torch.nn.Sigmoid,
        "log_sigmoid": torch.nn.LogSigmoid,
        "hard_sigmoid": torch.nn.Hardsigmoid,
        "elu": torch.nn.ELU,
        "hard_tanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "hard_swish": torch.nn.Hardswish,
    }

    if name.lower() not in activations.keys():
        raise NotImplementedError(
            f"unknown activation function '{name}', known activations include " +
            f"{list(activations.keys())}",
        )

    return activations[name.lower()](*args, **kwargs)


