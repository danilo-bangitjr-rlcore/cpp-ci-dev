import numpy
import torch
import torch.nn as nn
from corerl.utils.device import Device, device as global_device
from collections.abc import Callable


class Float(torch.nn.Module):
    def __init__(self, device: str | torch.device, init_value: float):
        super().__init__()
        d = torch.device(device)
        self.constant = torch.nn.Parameter(torch.tensor(init_value, dtype=torch.float32).to(d))

    def forward(self) -> torch.Tensor:
        return self.constant


def expectile_loss(diff: torch.Tensor, expectile: float = 0.9) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, (1 - expectile)).to(global_device.device)
    return (weight * (diff ** 2)).to(global_device.device)


def layer_init_normal(layer: nn.Module, bias: bool = True) -> nn.Module:
    nn.init.normal_(layer.weight)
    if int(bias):
        nn.init.constant_(layer.bias.data, 0)
    return layer.to(global_device.device)


def layer_init_zero(layer: nn.Module, bias: bool = True) -> nn.Module:
    nn.init.constant_(layer.weight, 0)
    if int(bias):
        nn.init.constant_(layer.bias.data, 0)
    return layer.to(global_device.device)


def layer_init_constant(layer: nn.Module, const: float, bias: bool = True) -> nn.Module:
    nn.init.constant_(layer.weight, float(const))
    if int(bias):
        nn.init.constant_(layer.bias.data, float(const))
    return layer.to(global_device.device)


def layer_init_xavier(layer: nn.Module, bias: bool = True) -> nn.Module:
    nn.init.xavier_uniform_(layer.weight)
    if int(bias):
        nn.init.constant_(layer.bias.data, 0)
    return layer.to(global_device.device)


def layer_init_uniform(layer: nn.Module, low: float = -0.003, high: float = 0.003, bias: float = 0) -> nn.Module:
    nn.init.uniform_(layer.weight, low, high)
    if float(bias):
        nn.init.constant_(layer.bias.data, bias)
    return layer.to(global_device.device)


def tensor(
    x: float | numpy.ndarray | torch.Tensor,
    device: str | torch.device | Device | None = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x

    if device is None:
        device = global_device.device

    elif isinstance(device, Device):
        device = device.device

    return torch.tensor(x, dtype=torch.float32).to(device)


def state_to_tensor(state: numpy.ndarray,  device: str | torch.device | None = None) -> torch.Tensor:
    return tensor(state.reshape((1, -1)), device)


def to_np(t: numpy.ndarray | torch.Tensor) -> numpy.ndarray:
    if isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy()
    else:
        return t


def init_layer(init: str) -> Callable[[torch.nn.modules.Module], torch.nn.modules.Module]:
    if init.lower() == 'xavier':
        return layer_init_xavier
    elif init.lower() == 'zero':
        return layer_init_zero
    elif init.lower() == 'normal':
        return layer_init_normal

    raise NotImplementedError(f"unknown weight initialization {init}")
