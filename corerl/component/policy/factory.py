import corerl.component.network.utils as utils
from . import *
from .policy import _get_type_from_dist
from corerl.component.distribution import get_dist_type
from corerl.component.layer import init_activation, Parallel
import torch.nn as nn
import torch
from corerl.utils.device import device


def get_type_from_str(type_: str):
    if type_.lower() == "softmax":
        return Softmax

    try:
        return _get_type_from_dist(get_dist_type(type_))
    except NotImplementedError:
        raise NotImplementedError(f"unknown policy type {type_}")


def _create_nn(cfg, policy_type, input_dim, output_dim):
    name = cfg["base"]["name"]
    if name.lower() in ("mlp", "fc"):
        return _create_mlp(cfg, policy_type, input_dim, output_dim)
    raise NotImplementedError(f"unknown neural network type {name}")


def _create_mlp(cfg, policy_type, input_dim, output_dim):
    continuous = policy_type.continuous
    if not continuous:
        return _create_discrete_mlp(cfg, input_dim, output_dim)
    return _create_continuous_mlp(cfg, input_dim, output_dim)


def _create_discrete_mlp(cfg, input_dim, output_dim):
    assert cfg["base"]["name"].lower() in ("mlp", "fc")

    hidden = cfg["base"]["hidden"]
    act = cfg["base"]["activation"]
    bias = cfg["base"]["bias"]

    head_act = cfg["head_activation"]
    head_bias = cfg["head_bias"]

    head_layer_init = utils.init_layer(cfg["head_layer_init"])
    layer_init = utils.init_layer(cfg["base"]["layer_init"])

    assert len(hidden) == len(act)

    net = []
    layer = nn.Linear(input_dim, hidden[0], bias=bias)
    layer = layer_init(layer)
    net.append(layer)
    net.append(init_activation(act[0]))

    for j in range(1, len(hidden)):
        layer = nn.Linear(hidden[j-1], hidden[j], bias=bias)
        layer = layer_init(layer)
        net.append(layer)
        net.append(init_activation(act[j]))

    head_layer = nn.Linear(hidden[-1], output_dim, bias=head_bias)
    head_layer = head_layer_init(head_layer)
    net.append(head_layer)
    net.append(init_activation(head_act))
    return nn.Sequential(*net).to(device.device)


def _create_continuous_mlp(cfg, input_dim, output_dim):
    assert cfg["base"]["name"].lower() in ("mlp", "fc")

    dist = get_dist_type(cfg["dist"])
    policy_type = get_type_from_str(cfg["dist"])

    hidden = cfg["base"]["hidden"]
    act = cfg["base"]["activation"]
    bias = cfg["base"]["bias"]

    head_act = cfg["head_activation"]
    paths = policy_type(None, dist).n_params
    head_bias = cfg["head_bias"]

    head_layer_init = utils.init_layer(cfg["head_layer_init"])
    layer_init = utils.init_layer(cfg["base"]["layer_init"])

    assert len(hidden) == len(act)
    assert len(head_act) == paths

    net = []
    layer = nn.Linear(input_dim, hidden[0], bias=bias)
    layer = layer_init(layer)
    net.append(layer)
    net.append(init_activation(act[0]))

    placeholder_input = torch.empty((input_dim,))

    n_inputs: int
    for j in range(1, len(hidden)):

        output_shape = nn.Sequential(*net)(placeholder_input).shape
        assert len(output_shape) == 1
        n_inputs = output_shape[0]
        layer = nn.Linear(n_inputs, hidden[j], bias=bias)

        layer = layer_init(layer)
        net.append(layer)
        net.append(init_activation(act[j]))

    head_layers = [[] for _ in range(paths)]
    for i in range(len(head_layers)):
        output_shape = nn.Sequential(*net)(placeholder_input).shape
        assert len(output_shape) == 1
        n_inputs = output_shape[0]

        head_layer = nn.Linear(n_inputs, output_dim, bias=head_bias)
        head_layer = head_layer_init(head_layer)
        head_layers[i].append(head_layer)

        for k in range(len(head_act[i])):
            head_layers[i].append(init_activation(head_act[i][k]))

    head = Parallel(*(nn.Sequential(*path) for path in head_layers))

    return nn.Sequential(nn.Sequential(*net), head).to(device.device)


def create(cfg, input_dim, output_dim, action_min=None, action_max=None):
    policy_type = get_type_from_str(cfg["dist"])

    net = _create_nn(cfg, policy_type, input_dim, output_dim)

    if not policy_type.continuous():
        return policy_type(net, input_dim, output_dim)

    dist_type = get_dist_type(cfg["dist"])
    if policy_type is UnBounded:
        return policy_type(net, dist_type)
    return ContinuousIIDPolicy.from_(
        net, dist_type, action_min=action_min, action_max=action_max,
    )
