import corerl.component.network.utils as utils
from . import Bounded, HalfBounded, UnBounded
from corerl.component.distribution import get_dist_type
from corerl.component.layer import init_activation, Parallel
import torch.nn as nn



def get_type(type_):
    if type_.lower() == "bounded":
        return Bounded
    elif type_.lower() == "unbounded":
        return UnBounded
    elif type_.lower() == "halfbounded":
        return HalfBounded
    else:
        raise NotImplementedError(f"unknown policy type {type_}")


def create_nn(cfg, input_dim, output_dim):
    name = cfg["base"]["name"]
    if name.lower() in ("mlp", "fc"):
        return create_mlp(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError(f"unknown neural network type {name}")


def create_mlp(cfg, input_dim, output_dim):
    assert cfg["base"]["name"].lower() in ("mlp", "fc")

    policy_type = get_type(cfg["type"])
    dist = get_dist_type(cfg["dist"])

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

    for j in range(1, len(hidden)):
        layer = nn.Linear(hidden[j-1], hidden[j], bias=bias)
        layer = layer_init(layer)
        net.append(layer)
        net.append(init_activation(act[j]))

    head_layers = [[] for _ in range(paths)]
    for i in range(len(head_layers)):
        head_layer = nn.Linear(hidden[-1], output_dim, bias=head_bias)
        head_layer = head_layer_init(head_layer)
        head_layers[i].append(head_layer)

        for k in range(len(head_act[i])):
            head_layers[i].append(init_activation(head_act[i][k]))

    head = Parallel(*(nn.Sequential(*path) for path in head_layers))
    return nn.Sequential(nn.Sequential(*net), head)


def create(cfg, input_dim, output_dim, action_min, action_max):
    net = create_nn(cfg, input_dim, output_dim)

    policy_type = get_type(cfg["type"])
    dist_type = get_dist_type(cfg["dist"])

    if policy_type is UnBounded:
        return policy_type(net, dist_type)
    else:
        return policy_type(
            net, dist_type, action_min=action_min, action_max=action_max,
        )
