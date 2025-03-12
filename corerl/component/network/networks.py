import copy
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
from pydantic import Field

import corerl.component.layer as layer
import corerl.component.network.utils as utils
from corerl.component.layer.activations import ActivationConfig
from corerl.component.network.ensemble.reductions import (
    MeanReduct,
    ReductConfig,
    bootstrap_reduct_group,
)
from corerl.configs.config import config, list_
from corerl.utils.device import device

EPSILON = 1e-6


@config()
class NNTorsoConfig:
    name: Literal['fc'] = 'fc'

    bias: bool = True
    layer_init: str = 'Xavier'
    hidden: list[int] = list_([64, 64])
    activation: list[ActivationConfig] = list_([
        {'name': 'relu'},
        {'name': 'relu'},
    ])


@config()
class EnsembleNetworkConfig:
    name: Literal['ensemble'] = 'ensemble'
    ensemble: int = 1
    bootstrap_reduct: ReductConfig = Field(default_factory=MeanReduct)
    base: NNTorsoConfig = Field(default_factory=NNTorsoConfig)

def create_mlp(
    cfg: NNTorsoConfig, input_dim: int, output_dim: int | None,
) -> nn.Module:
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

class EnsembleNetworkReturn(NamedTuple):
    # some reduction over ensemble members, producing a single
    # value function
    reduced_value: torch.Tensor

    # the value function for every member of the ensemble
    ensemble_values: torch.Tensor

    # the variance of the ensemble values
    ensemble_variance: torch.Tensor


class EnsembleNetwork(nn.Module):
    def __init__(self, cfg: EnsembleNetworkConfig, input_dim: int, output_dim: int):
        super().__init__()
        self.ensemble = cfg.ensemble
        self.subnetworks = [
            create_mlp(cfg.base, input_dim, output_dim)
            for _ in range(self.ensemble)
        ]

        self.base_model = copy.deepcopy(self.subnetworks[0])
        self.base_model = self.base_model.to(device.device)

        self.bootstrap_reduct = bootstrap_reduct_group.dispatch(cfg.bootstrap_reduct)
        self.to(device.device)


    def forward(
        self,
        input_tensor: torch.Tensor,
    ):
        # Input shape is either (ensemble_size, batch_size, feature dim) if a different batch is used for
        #   each member of the ensemble
        # Oherwise, the shape of the input_tensor is (batch_size, feature dim)
        #   if the same batch is expected to be used for each ensemble member
        if len(input_tensor.shape) == 3 and input_tensor.shape[0] == self.ensemble:
            # Each element of the 'input_tensor' is evaluated by the corresponding member of the ensemble
            qs = [self.subnetworks[i](input_tensor[i]) for i in range(self.ensemble)]

        elif len(input_tensor.shape) == 2:
            # Each member of the ensemble evaluates the same batch of state-action pairs
            qs = [net(input_tensor) for net in self.subnetworks]

        else:
            raise Exception()

        qs = torch.cat([
            torch.unsqueeze(q, 0) for q in qs
        ], dim=0)

        q = self.bootstrap_reduct(qs, dim=0)
        variance = self.get_ensemble_variance(qs)
        return EnsembleNetworkReturn(q, qs, variance)

    def state_dict(self) -> list: # type: ignore
        return [net.state_dict() for net in self.subnetworks]

    def load_state_dict(self, state_dict_list: list) -> None: # type: ignore
        for i in range(self.ensemble):
            self.subnetworks[i].load_state_dict(state_dict_list[i])
        return

    def parameters(self, independent: bool = False) -> list: # type: ignore
        param_list = []
        if independent:
            for i in range(self.ensemble):
                param_list.append(self.subnetworks[i].parameters())
        else:
            for i in range(self.ensemble):
                param_list += list(self.subnetworks[i].parameters())
        return param_list

    def get_ensemble_variance(self, qs: torch.Tensor) -> torch.Tensor:
        if self.ensemble > 1:
            return torch.var(qs, dim=0)
        else:
            return torch.zeros(1)
