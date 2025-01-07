from dataclasses import field
from typing import Any, Literal

from corerl.configs.config import config, MISSING, list_
from corerl.configs.group import Group
from corerl.component.layer.activations import ActivationConfig
import corerl.component.network.utils as utils
from corerl.component.policy.softmax import Softmax, Policy
from corerl.component.policy.policy import ContinuousIIDPolicy
from corerl.component.distribution import get_dist_type
from corerl.component.network.networks import _create_layer, create_base, NNTorsoConfig
from corerl.component.layer import init_activation, Parallel
import torch.nn as nn
import torch
from corerl.utils.device import device


HeadActivation = list[list[ActivationConfig]]


@config(frozen=True)
class BaseNNConfig:
    name: str = MISSING

    base: NNTorsoConfig = field(default_factory=NNTorsoConfig)
    head_layer_init: str = 'Xavier'
    head_activation: HeadActivation = MISSING
    head_bias: bool = True


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

    for k in range(len(head_act[0])):
        net.append(init_activation(head_act[0][k]))

    return nn.Sequential(*net).to(device.device)


def _create_continuous_mlp(
    cfg: BaseNNConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    assert cfg.base.name.lower() in ("mlp", "fc")

    dist = get_dist_type(cfg.name)
    model: Any = None
    paths = ContinuousIIDPolicy.from_(model, dist, action_min, action_max).n_params

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



policy_group = Group[
    [int, int, torch.Tensor | float | None, torch.Tensor | float | None],
    Policy,
]()


@config(frozen=True)
class BetaPolicyConfig(BaseNNConfig):
    name: Literal['beta'] = 'beta'

    head_activation: HeadActivation = list_([
        [{'name': 'softplus'}, {'name': 'bias', 'args': [1]}],
        [{
            'name': 'tanh_shift',
            'kwargs': { 'shift': -4, 'denom': 1, 'high': 10_000, 'low': 1 },
        }],
    ])


@policy_group.dispatcher
def _(
    cfg: BetaPolicyConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    return ContinuousIIDPolicy.from_(
        _create_continuous_mlp(cfg, input_dim, output_dim, action_min, action_max),
        get_dist_type('beta'),
        action_min=action_min,
        action_max=action_max,
    )


@config(frozen=True)
class GammaPolicyConfig(BaseNNConfig):
    name: Literal['gamma'] = 'gamma'

    # Since policies should always return actions in [0, 1], we can force the
    # mean to stay within this range
    head_activation: HeadActivation = list_([
        [{'name': 'softplus'}, {'name': 'bias', 'args': [1]}],
        [{'name': 'softplus'}, {'name': 'bias', 'args': [1]}],
    ])

@policy_group.dispatcher
def _(
    cfg: GammaPolicyConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    return ContinuousIIDPolicy.from_(
        _create_continuous_mlp(cfg, input_dim, output_dim, action_min, action_max),
        get_dist_type('gamma'),
        action_min=action_min,
    )



@config(frozen=True)
class LaplacePolicyConfig(BaseNNConfig):
    name: Literal['laplace'] = 'laplace'

    head_activation: HeadActivation = list_([
        [{"name": "functional", "args": ["sigmoid"]}],
        [{"name": "clamp", "args": [-20, 2]}, {"name": "exp"}],
    ])

@policy_group.dispatcher
def _(
    cfg: LaplacePolicyConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    return ContinuousIIDPolicy.from_(
        _create_continuous_mlp(cfg, input_dim, output_dim, action_min, action_max),
        get_dist_type('laplace'),
    )


@config(frozen=True)
class NormalPolicyConfig(BaseNNConfig):
    name: Literal['normal'] = 'normal'

    head_activation: HeadActivation = list_([
        [{"name": "functional", "args": ["sigmoid"]}],
        [{"name": "clamp", "args": [-20, 2]}, {"name": "exp"}],
    ])

@policy_group.dispatcher
def _(
    cfg: NormalPolicyConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    return ContinuousIIDPolicy.from_(
        _create_continuous_mlp(cfg, input_dim, output_dim, action_min, action_max),
        get_dist_type('normal'),
    )


@config(frozen=True)
class SquashedGaussianPolicyConfig(BaseNNConfig):
    name: Literal['squashed_gaussian'] = 'squashed_gaussian'

    head_activation: HeadActivation = list_([
        [{"name": "identity"}],
        [{"name": "clamp", "args": [-20, 2]}, {"name": "exp"}],
    ])


@policy_group.dispatcher
def _(
    cfg: SquashedGaussianPolicyConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    return ContinuousIIDPolicy.from_(
        _create_continuous_mlp(cfg, input_dim, output_dim, action_min, action_max),
        get_dist_type('squashed_gaussian'),
        action_min=action_min,
        action_max=action_max,
    )


@config(frozen=True)
class SoftmaxPolicyConfig(BaseNNConfig):
    name: Literal['softmax'] = 'softmax'

    head_activation: HeadActivation = list_([
        [{"name": "identity"}],
    ])

@policy_group.dispatcher
def _(
    cfg: SoftmaxPolicyConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None,
    action_max: torch.Tensor | float | None,
):
    net = _create_discrete_mlp(cfg, input_dim, output_dim)
    return Softmax(net, input_dim, output_dim)



def create(
    cfg: BaseNNConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None = None,
    action_max: torch.Tensor | float | None = None,
):
    return policy_group.dispatch(cfg, input_dim, output_dim, action_min, action_max)
