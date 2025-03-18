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

# ---------------------------------------------------------------------------- #
#                             Late Fusion Networks                             #
# ---------------------------------------------------------------------------- #

@config()
class LateFusionConfig:
    name: Literal['late_fusion'] = 'late_fusion'
    input_cfg : NNTorsoConfig =  Field(default_factory=NNTorsoConfig)
    skip_input : bool = True # choose to not use the input networks
    combined_cfg : NNTorsoConfig =  Field(default_factory=NNTorsoConfig)

class LateFusionNetwork(nn.Module):
    def __init__(
            self,
            cfg: LateFusionConfig,
            input_dims: list[int],
            output_dim: int | None,
        ):
        """
        Neural network that processes multiple separate inputs through subnet architectures
        and then combines their outputs.
        """
        super().__init__()

        self.skip_input = cfg.skip_input

        if not self.skip_input:
            # Create multiple subnets - one for each input
            self.input_nets = nn.ModuleList()
            for input_dim in input_dims:
                input_net = create_mlp(cfg.input_cfg, input_dim, output_dim=None)
                self.input_nets.append(input_net)

            input_net_out_dim = cfg.input_cfg.hidden[-1]
            num_input_nets = len(input_dims)
            combined_input_dim = input_net_out_dim*num_input_nets
        else:
            combined_input_dim = sum(input_dims)

        self.combined_net = create_mlp(
            cfg.combined_cfg,
            input_dim=combined_input_dim,
            output_dim=output_dim,
        )
        if output_dim is None:
            self.output_dim = cfg.combined_cfg.hidden[-1]
        else:
            self.output_dim = output_dim

    def forward(self, inputs: list[torch.Tensor]):
        if self.skip_input: # don't use the input networks
            combined = torch.cat(inputs, dim=1)
            return self.combined_net(combined)

        assert len(inputs) == len(self.input_nets), f"Expected {len(self.input_nets)} inputs, got {len(inputs)}"
        # Process each input through its respective subnet
        subnet_outputs = []
        for i, input_tensor in enumerate(inputs):
            output = self.input_nets[i](input_tensor)
            subnet_outputs.append(output)

        # Concatenate the outputs from all subnets
        combined = torch.cat(subnet_outputs, dim=1)

        # Process through the combined layers
        output = self.combined_net(combined)
        return output

# ---------------------------------------------------------------------------- #
#                               Ensemble Networks                              #
# ---------------------------------------------------------------------------- #
class EnsembleNetworkReturn(NamedTuple):
    # some reduction over ensemble members, producing a single
    # value function
    reduced_value: torch.Tensor

    # the value function for every member of the ensemble
    ensemble_values: torch.Tensor

    # the variance of the ensemble values
    ensemble_variance: torch.Tensor


@config()
class EnsembleNetworkConfig:
    name: Literal['ensemble'] = 'ensemble'
    ensemble: int = 1
    bootstrap_reduct: ReductConfig = Field(default_factory=MeanReduct)
    base: LateFusionConfig = Field(default_factory=LateFusionConfig)


class EnsembleNetwork(nn.Module):
    def __init__(self, cfg: EnsembleNetworkConfig, input_dims: list[int], output_dim: int):
        super().__init__()
        self.ensemble = cfg.ensemble
        self.subnetworks = [
            LateFusionNetwork(cfg.base, input_dims, output_dim)
            for _ in range(self.ensemble)
        ]

        self.base_model = copy.deepcopy(self.subnetworks[0])
        self.base_model = self.base_model.to(device.device)

        self.bootstrap_reduct = bootstrap_reduct_group.dispatch(cfg.bootstrap_reduct)
        self.to(device.device)

    def forward(
        self,
        input_tensors: list[list[torch.Tensor]],
    ):
        """
        Passes inputs to forward() of subnets.
        We assume that subnets are LateFusionNetworks, so input_tensors is a list of lists of tensors,
        where the first list is over the inputs, the second is over the ensemble.
        The tensors themselves are of shape (batch_size x input_dim).

        If there is only one element along the ensemble dimension, we pass that same input to all ensembles.
        """
        transposed_inputs = [[row[i] for row in input_tensors] for i in range(len(input_tensors[0]))]
        assert len(transposed_inputs) == 1 or len(transposed_inputs) == self.ensemble

        if len(transposed_inputs) == 1:
            # pass the same batch to all subnets
            qs = [self.subnetworks[i](transposed_inputs[0]) for i in range(self.ensemble)]
        else:
            # pass each subnet their own batch
            qs = [self.subnetworks[i](transposed_inputs[i]) for i in range(self.ensemble)]

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
