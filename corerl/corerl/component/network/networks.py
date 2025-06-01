from typing import Literal

import torch.nn as nn

from corerl.component import layer
from corerl.component.layer.activations import ActivationConfig
from corerl.component.network import utils
from corerl.configs.config import config, list_
from corerl.utils.device import device

EPSILON = 1e-6


@config()
class NNTorsoConfig:
    name: Literal['fc'] = 'fc'

    bias: bool = True
    layer_init: str = 'orthogonal'
    hidden: list[int] = list_([256, 256])
    activation: list[ActivationConfig] = list_([
        {'name': 'relu'},
        {'name': 'relu'},
    ])

def create_mlp(
    cfg: NNTorsoConfig, input_dim: int, output_dim: int | None,
) -> nn.Sequential:
    assert cfg.name.lower() in ("mlp", "fc")

    hidden = cfg.hidden
    act = cfg.activation
    bias = cfg.bias
    assert len(hidden) == len(act)
    layer_init = utils.init_layer(cfg.layer_init)

    net = []

    # Add the first layer to the network
    layer_ = nn.Linear(input_dim, hidden[0], bias=bias)
    layer_ = layer_init(layer_)
    net.append(layer_)
    net.append(layer.init_activation(act[0]))

    # Create the base layers of the network
    for j in range(1, len(hidden)):
        layer_ = nn.Linear(hidden[j-1], hidden[j], bias, device=device.device)
        layer_ = layer_init(layer_)
        net.append(layer_)
        net.append(layer.init_activation(act[j]))

    if output_dim is not None:
        layer_ = nn.Linear(hidden[-1], output_dim, bias, device=device.device)
        layer_ = layer_init(layer_)
        net.append(layer_)

    return nn.Sequential(*net).to(device.device)
