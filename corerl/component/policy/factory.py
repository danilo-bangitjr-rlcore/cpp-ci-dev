from typing import Annotated, Any, Literal

import torch
import torch.nn as nn
from pydantic import Field

import corerl.component.network.utils as utils
from corerl.component.distribution import get_dist_type
from corerl.component.layer import Parallel, init_activation
from corerl.component.layer.activations import ActivationConfig
from corerl.component.network.networks import NNTorsoConfig, create_mlp
from corerl.component.policy.policy import ContinuousIIDPolicy, Policy
from corerl.configs.config import MISSING, config, list_
from corerl.configs.group import Group
from corerl.utils.device import device

HeadActivation = list[list[ActivationConfig]]


@config(frozen=True)
class BaseNNConfig:
    name: str = MISSING

    base: NNTorsoConfig = Field(default_factory=NNTorsoConfig)
    head_layer_init: str = 'Xavier'
    head_activation: HeadActivation = MISSING
    head_bias: bool = True

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

    base_net = create_mlp(cfg.base, input_dim, None)

    head_act = cfg.head_activation
    head_bias = cfg.head_bias
    head_layer_init = utils.init_layer(cfg.head_layer_init)

    # Create head layer(s) to the network
    head_layers = [[] for _ in range(paths)]
    for i in range(len(head_layers)):
        head_layer = nn.Linear(cfg.base.hidden[-1], output_dim, head_bias, device=device.device)
        head_layer = head_layer_init(head_layer)
        head_layers[i].append(head_layer)

        for k in range(len(head_act[i])):
            # Head layers may have multiple activation functions
            head_layers[i].append(init_activation(head_act[i][k]))

    head = Parallel(*(nn.Sequential(*path) for path in head_layers))

    return nn.Sequential(nn.Sequential(*base_net), head).to(device.device)



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


def create(
    cfg: BaseNNConfig,
    input_dim: int,
    output_dim: int,
    action_min: torch.Tensor | float | None = None,
    action_max: torch.Tensor | float | None = None,
):
    return policy_group.dispatch(cfg, input_dim, output_dim, action_min, action_max)

PolicyConfig = Annotated[
    BetaPolicyConfig
    | GammaPolicyConfig
    | LaplacePolicyConfig
    | NormalPolicyConfig
    | SquashedGaussianPolicyConfig
, Field(discriminator='name')]
