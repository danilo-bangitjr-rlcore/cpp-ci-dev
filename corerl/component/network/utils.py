import numpy
import torch
import torch.nn as nn
from corerl.utils.device import device as global_device
from collections.abc import Callable
from corerl.component.layer import Identity
import warnings
import corerl.component.layer.activations as activations


class Float(torch.nn.Module):
    def __init__(self, device: str, init_value: float):
        super().__init__()
        d = torch.device(device)
        self.constant = torch.nn.Parameter(torch.tensor(init_value, dtype=torch.float32).to(d))

    def forward(self) -> torch.Tensor:
        return self.constant


def expectile_loss(diff: torch.Tensor, expectile: float = 0.9) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, (1 - expectile)).to(global_device.device)
    return (weight * (diff ** 2)).to(global_device.device)


def ensemble_expectile_loss(q: torch.Tensor, vs: list[torch.Tensor], expectile: float = 0.9) -> list[torch.Tensor]:
    losses = []
    for v in vs:
        diff = (q - v).to(global_device.device)
        weight = torch.where(diff > 0, expectile, (1 - expectile)).to(global_device.device)
        loss_v = (weight * (diff ** 2)).mean()
        losses.append(loss_v.to(global_device.device))
    return losses


def ensemble_mse(target, q_ens) -> list[torch.Tensor]:
    """
    Calculate the MSE of an ensemble of q values

    Parameters
    ----------
    q_ens : torch.Tensor
        An ensemble of predicted q values, with batch dimension 0
    target : torch.Tensor
        The targets of prediction for each q value in `q_ens`. If each q value
        in `q_ens` should have a different target for prediction, then `target`
        should have batch dimension 0 with `target.shape == q_ens.shape`.
    """
    assert q_ens.ndim == 3
    ensemble_target = target.ndim == 3
    if ensemble_target:
        mses = [nn.functional.mse_loss(t, q).to(global_device.device) for (t, q) in zip(target, q_ens)]
    else:
        mses = [nn.functional.mse_loss(target, q).to(global_device.device) for q in q_ens]
    return mses


def reset_weight_random(old_net: nn.Module, new_net: nn.Module, param: list[torch.Tensor]) -> nn.Module:
    return new_net.to(global_device.device)


def reset_weight_shift(old_net: nn.Module, new_net: nn.Module, param: list[torch.Tensor]) -> nn.Module:
    with torch.no_grad():
        for p, p_new in zip(old_net.parameters(), new_net.parameters()):
            p_new.data.mul_(0)
            p_new.data.add_(p.data + param)
    return new_net


def reset_weight_shrink(old_net: nn.Module, new_net: nn.Module, param: list[torch.Tensor]) -> nn.Module:
    with torch.no_grad():
        for p, p_new in zip(old_net.parameters(), new_net.parameters()):
            p_new.data.mul_(0)
            p_new.data.add_(p.data * param)
    return new_net


def reset_weight_shrink_rnd(old_net: nn.Module, new_net: nn.Module, param: list[torch.Tensor]) -> nn.Module:
    with torch.no_grad():
        for p, p_new in zip(old_net.parameters(), new_net.parameters()):
            p_new.data.mul_(0.5)
            p_new.data.add_(p.data * param * 0.5)
    return new_net


def reset_weight_pass(old_net: nn.Module, new_net: nn.Module, param: list[torch.Tensor]) -> nn.Module:
    return old_net


def clone_model_0to1(net0: nn.Module, net1: nn.Module) -> nn.Module:
    with torch.no_grad():
        net1.load_state_dict(net0.state_dict())
    return net1


def clone_gradient(model: nn.Module) -> dict:
    grad_rec = {}
    for idx, param in enumerate(model.parameters()):
        grad_rec[idx] = param.grad
    return grad_rec


def move_gradient_to_network(model: nn.Module, grad_rec: dict, weight: float) -> nn.Module:
    for idx, param in enumerate(model.parameters()):
        if grad_rec[idx] is not None:
            param.grad = grad_rec[idx] * weight
    return model


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
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if device is not None:
        x = torch.tensor(x, dtype=torch.float32).to(device)
    else:
        x = torch.tensor(x, dtype=torch.float32).to(global_device.device)
    return x


def state_to_tensor(state: numpy.ndarray,  device: str | torch.device | None = None) -> torch.Tensor:
    return tensor(state.reshape((1, -1)), device)


def to_np(t: numpy.ndarray | torch.Tensor) -> numpy.ndarray:
    if isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy()
    elif isinstance(t, numpy.ndarray):
        return t
    else:
        raise AssertionError("")


def init_activation(name: str) -> type[nn.Module]:
    warnings.warn(
        "init_activation in module utils is deprecated and will be removed, " +
        "use activations.init_activation instead"
    )

    return type(activations.init_activation({"name": name}))


def init_activation_function(name: str) -> nn.Module:
    warnings.warn(
        "init_activation in module utils is deprecated and will be removed, " +
        "use activations.init_activation instead"
    )

    return activations.init_activation({"name": name})()


def init_layer(init: str) -> Callable[[torch.nn.modules.Module], torch.nn.modules.Module]:
    if init.lower() == 'xavier':
        return layer_init_xavier
    elif init.lower() == 'const':
        return layer_init_constant
    elif init.lower() == 'zero':
        return layer_init_zero
    elif init.lower() == 'normal':
        return layer_init_normal

    raise NotImplementedError(f"unknown weight initialization {init}")
