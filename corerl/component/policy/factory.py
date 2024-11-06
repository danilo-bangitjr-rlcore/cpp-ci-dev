from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING
import corerl.component.network.utils as utils
from corerl.component.policy.softmax import Softmax, Policy
from corerl.component.policy.policy import ContinuousIIDPolicy, UnBounded, _get_type_from_dist
from corerl.component.distribution import get_dist_type
from corerl.component.network.networks import _create_layer, create_base, NNTorsoConfig
from corerl.component.layer import init_activation, Parallel
import torch.nn as nn
import torch
from corerl.utils.device import device


def get_type_from_str(type_: str) -> type[Policy]:
    if type_.lower() == "softmax":
        return Softmax

    try:
        return _get_type_from_dist(get_dist_type(type_))

    except NotImplementedError as e:
        raise NotImplementedError(f"unknown policy type {type_}") from e


@dataclass
class BaseNNConfig:
    base: NNTorsoConfig = field(default_factory=NNTorsoConfig)

    dist: str = MISSING
    head_layer_init: str = MISSING
    head_activation: list[list[dict[str, Any]]] = MISSING
    head_bias: bool = MISSING


def _create_nn(
    cfg: BaseNNConfig,
    policy_type: type[Policy],
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    name = cfg.base.name
    if name.lower() in ("mlp", "fc"):
        return _create_mlp(cfg, policy_type, input_dim, output_dim, action_min, action_max)
    raise NotImplementedError(f"unknown neural network type {name}")


def _create_mlp(
    cfg: BaseNNConfig,
    policy_type: type[Policy],
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    continuous = policy_type.continuous
    if not continuous:
        return _create_discrete_mlp(cfg, input_dim, output_dim)
    return _create_continuous_mlp(cfg, input_dim, output_dim, action_min, action_max)


def _create_discrete_mlp(cfg: BaseNNConfig, input_dim: int, output_dim: int):
    assert cfg.base.name.lower() in ("mlp", "fc")

    hidden = cfg.base.hidden
    act = cfg.base.activation
    bias = cfg.base.bias

    head_act = cfg.head_activation
    head_bias = cfg.head_bias

    head_layer_init = utils.init_layer(cfg.head_layer_init)
    layer_init = utils.init_layer(cfg.base.layer_init)

    assert len(hidden) == len(act)

    net = []
    layer = nn.Linear(input_dim, hidden[0], bias=bias)
    layer = layer_init(layer)
    net.append(layer)
    net.append(init_activation(act[0]))

    placeholder_input = torch.empty((input_dim,))

    # Create the base layers
    for j in range(1, len(hidden)):
        layer = _create_layer(
            nn.Linear, layer_init, net, hidden[j], bias, placeholder_input,
        )

        net.append(layer)
        net.append(init_activation(act[j]))

    # Create the head layer(s)
    head_layer = _create_layer(
        nn.Linear, head_layer_init, net, output_dim, head_bias,
        placeholder_input,
    )
    net.append(head_layer)
    net.append(init_activation(head_act))

    return nn.Sequential(*net).to(device.device)


def _create_continuous_mlp(
    cfg: BaseNNConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    assert cfg.base.name.lower() in ("mlp", "fc")

    dist = get_dist_type(cfg.dist)
    policy_type = get_type_from_str(cfg.dist)
    assert issubclass(policy_type, ContinuousIIDPolicy)

    paths = policy_type.from_(None, dist, action_min, action_max).n_params

    head_act = cfg.head_activation
    head_bias = cfg.head_bias
    head_layer_init = utils.init_layer(cfg.head_layer_init)

    placeholder_input = torch.empty((input_dim,))
    net = [create_base(cfg.base, input_dim, None)]

    # Create head layer(s) to the network
    head_layers = [[] for _ in range(paths)]
    for i in range(len(head_layers)):
        head_layer = _create_layer(
            nn.Linear, head_layer_init, net, output_dim, head_bias,
            placeholder_input,
        )

        head_layers[i].append(head_layer)

        for k in range(len(head_act[i])):
            # Head layers may have multiple activation functions
            head_layers[i].append(init_activation(head_act[i][k]))

    head = Parallel(*(nn.Sequential(*path) for path in head_layers))

    return nn.Sequential(nn.Sequential(*net), head).to(device.device)


def create(
    cfg: BaseNNConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None = None,
    action_max: torch.Tensor | float | None = None,
):
    policy_type = get_type_from_str(cfg.dist)
    net = _create_nn(cfg, policy_type, input_dim, output_dim, action_min, action_max)

    if policy_type is Softmax:
        return Softmax(net, input_dim, output_dim)

    if not policy_type.continuous():
        assert policy_type is Softmax
        return policy_type(net, input_dim, output_dim)

    dist_type = get_dist_type(cfg.dist)
    if policy_type is UnBounded:
        return policy_type(net, dist_type)
    return ContinuousIIDPolicy.from_(
        net,
        dist_type,
        action_min=action_min,
        action_max=action_max,
    )
