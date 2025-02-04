import copy
from collections.abc import Iterable
from typing import Any, Callable, Literal, Optional

import torch
import torch.nn as nn
from pydantic import Field
from torch.func import functional_call, stack_module_state  # type: ignore

import corerl.component.layer as layer
import corerl.component.network.utils as utils
from corerl.component.layer.activations import ActivationConfig
from corerl.component.network.base import BaseNetworkConfig
from corerl.component.network.ensemble.reductions import MeanReduct, Reduct, bootstrap_reduct_group
from corerl.configs.config import config, list_
from corerl.utils.device import device

EPSILON = 1e-6


@config(frozen=True)
class NNTorsoConfig(BaseNetworkConfig):
    name: Literal['fc'] = 'fc'

    bias: bool = True
    layer_init: str = 'Xavier'
    hidden: list[int] = list_([64, 64])
    activation: list[ActivationConfig] = list_([
        {'name': 'relu'},
        {'name': 'relu'},
    ])


@config(frozen=True)
class EnsembleCriticNetworkConfig(BaseNetworkConfig):
    name: Literal['ensemble'] = 'ensemble'
    ensemble: int = 1
    bootstrap_reduct: Reduct = Field(default_factory=MeanReduct)
    policy_reduct: Reduct = Field(default_factory=MeanReduct)
    vmap: bool = False

    base: NNTorsoConfig = Field(default_factory=NNTorsoConfig)



def _init_ensemble_reducts(cfg: EnsembleCriticNetworkConfig):
    def bs_reduct(x: torch.Tensor, dim: int):
        return bootstrap_reduct_group.dispatch(cfg.bootstrap_reduct, x, dim)

    def p_reduct(x: torch.Tensor, dim: int):
        return bootstrap_reduct_group.dispatch(cfg.policy_reduct, x, dim)

    return bs_reduct, p_reduct


def create_base(
    cfg: NNTorsoConfig, input_dim: int, output_dim: Optional[int],
) -> nn.Module:
    if cfg.name.lower() in ("mlp", "fc"):
        return _create_base_mlp(cfg, input_dim, output_dim)
    else:
        raise ValueError(f"unknown network type {cfg.name}")


def _create_base_mlp(
    cfg: NNTorsoConfig, input_dim: int, output_dim: Optional[int],
) -> nn.Module:
    assert cfg.name.lower() in ("mlp", "fc")

    hidden = cfg.hidden
    act = cfg.activation
    bias = cfg.bias
    assert len(hidden) == len(act)
    layer_init = utils.init_layer(cfg.layer_init)

    net = []

    # Add the first layer to the network
    layer_ = nn.Linear(input_dim, hidden[0], bias=bias, device=device.device)
    layer_ = layer_init(layer_)
    net.append(layer_)
    net.append(layer.init_activation(act[0]))

    placeholder_input = torch.empty((input_dim,), device=device.device)

    # Create the base layers of the network
    for j in range(1, len(hidden)):
        layer_ = _create_layer(
            nn.Linear, layer_init, net, hidden[j], bias, placeholder_input,
        )

        net.append(layer_)
        net.append(layer.init_activation(act[j]))


    if output_dim is not None:
        layer_ = _create_layer(
            nn.Linear, layer_init, net, output_dim, bias, placeholder_input,
        )
        net.append(layer_)

    return nn.Sequential(*net).to(device.device)


def _create_layer(
    layer_type: type[nn.Module],
    layer_init: Callable[[nn.Module], nn.Module],
    base_net:
    Iterable[nn.Module],
    hidden: int,
    bias: bool,
    placeholder_input: torch.Tensor,
) -> nn.Module:
    """
    Create a single layer of type `layer_type` initialized with `layer_init`.

    The argument `base_net` is the base net that the layer will be accepting
    input from, and is used to determine the input size for the layer under
    construction, together with `placeholder_input`.
    """
    if layer_type is nn.Linear:
        n_inputs = _get_output_shape(
            base_net, placeholder_input, dim=0,
        )
        layer = layer_type(n_inputs, hidden, bias=bias)
        return layer_init(layer)

    raise NotImplementedError(f"unknown layer type {layer_type}")


def _get_output_shape(
    net: Iterable[nn.Module],
    placeholder_input: torch.Tensor,
    *,
    dim: Optional[int]=None,
) -> int:
    output_shape = nn.Sequential(*net)(placeholder_input).shape
    assert len(output_shape) == 1
    return output_shape[dim]


class EnsembleFC(nn.Module):
    def __init__(self, cfg: EnsembleCriticNetworkConfig, input_dim: int, output_dim: int):
        super(EnsembleFC, self).__init__()
        self.ensemble = cfg.ensemble
        self.subnetworks = [
            create_base(cfg.base, input_dim, output_dim)
            for _ in range(self.ensemble)
        ]
        self.to(device.device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        outs = [net(input_tensor) for net in self.subnetworks]
        for i in range(self.ensemble):
            outs[i] = torch.unsqueeze(outs[i], 0)
        outs = torch.cat(outs, dim=0)
        return outs

    def state_dict(self) -> list[dict[str, Any]]: # type: ignore
        sd = [net.state_dict() for net in self.subnetworks]
        return sd

    def load_state_dict(self, state_dict_list: list) -> None: # type: ignore
        for i in range(self.ensemble):
            self.subnetworks[i].load_state_dict(state_dict_list[i])
        return

    def parameters(self, independent: bool = False) -> list[torch.nn.Parameter]: # type: ignore
        param_list = []
        if independent:
            for i in range(self.ensemble):
                param_list.append(self.subnetworks[i].parameters())
        else:
            for i in range(self.ensemble):
                param_list += list(self.subnetworks[i].parameters())
        return param_list


class EnsembleCritic(nn.Module):
    def __init__(self, cfg: EnsembleCriticNetworkConfig, input_dim: int, output_dim: int):
        super(EnsembleCritic, self).__init__()
        self.ensemble = cfg.ensemble
        self.vmap = cfg.vmap
        self.subnetworks = [
            create_base(cfg.base, input_dim, output_dim)
            for _ in range(self.ensemble)
        ]

        # Vectorizing the ensemble to use torch.vmap to avoid sequentially querrying the ensemble
        self.params, self.buffers = stack_module_state(self.subnetworks) # type: ignore

        self.base_model = copy.deepcopy(self.subnetworks[0])
        self.base_model = self.base_model.to(device.device)

        self.bootstrap_reduct, self.policy_reduct = _init_ensemble_reducts(cfg)
        self.to(device.device)

    def fmodel(self, params: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor], x: torch.Tensor):
        return functional_call(self.base_model, (params, buffers), (x,))

    def forward(
            self, input_tensor: torch.Tensor, bootstrap_reduct: Optional[bool] = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # For ensemble critic updates, expecting a different batch for each member of the ensemble
        # Therefore, we expect the shape of the input_tensor to be (ensemble_size, batch_size, state-action dim)
        if len(input_tensor.shape) == 3 and input_tensor.shape[0] == self.ensemble:
            # Each element of the 'input_tensor' is evaluated by the corresponding member of the ensemble
            # Used in critic updates
            if self.vmap:
                qs = torch.vmap(self.fmodel)(self.params, self.buffers, input_tensor)
            else:
                qs = [self.subnetworks[i](input_tensor[i]) for i in range(self.ensemble)]
                for i in range(self.ensemble):
                    qs[i] = torch.unsqueeze(qs[i], 0)
                qs = torch.cat(qs, dim=0)
        elif len(input_tensor.shape) == 2:
            # Each member of the ensemble evaluates the same batch of state-action pairs
            # Used in policy updates and when evaluating alerts
            if self.vmap:
                qs = torch.vmap(self.fmodel, in_dims=(0, 0, None))(self.params, self.buffers, input_tensor)
            else:
                qs = [net(input_tensor) for net in self.subnetworks]
                for i in range(self.ensemble):
                    qs[i] = torch.unsqueeze(qs[i], 0)
                qs = torch.cat(qs, dim=0)
        else:
            raise NotImplementedError

        if bootstrap_reduct:
            q = self.bootstrap_reduct(qs, dim=0)
        else:
            q = self.policy_reduct(qs, dim=0)

        return q, qs

    def state_dict(self) -> list: # type: ignore
        sd = [net.state_dict() for net in self.subnetworks]
        return sd

    def load_state_dict(self, state_dict_list: list) -> None: # type: ignore
        for i in range(self.ensemble):
            self.subnetworks[i].load_state_dict(state_dict_list[i])
        return

    def parameters(self, independent: bool = False) -> list: # type: ignore
        if self.vmap:
            # https://github.com/pytorch/pytorch/issues/120581
            return self.params.values() # type: ignore
        else:
            param_list = []
            if independent:
                for i in range(self.ensemble):
                    param_list.append(self.subnetworks[i].parameters())
            else:
                for i in range(self.ensemble):
                    param_list += list(self.subnetworks[i].parameters())
            return param_list



@config(frozen=True)
class GRUConfig(BaseNetworkConfig):
    name: Literal['gru'] = 'gru'

    gru_hidden_dim: int = -1
    num_gru_layers: int = 1

    base: NNTorsoConfig = Field(default_factory=NNTorsoConfig)


class GRU(nn.Module):
    def __init__(self, cfg: GRUConfig, input_dim: int, output_dim: int):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.gru_hidden_dim = cfg.gru_hidden_dim
        self.num_gru_layers = cfg.num_gru_layers
        self.output_net = create_base(cfg.base, self.gru_hidden_dim, output_dim)
        # self.gru = nn.GRU(input_dim, self.gru_hidden_dim, self.num_gru_layers, batch_first=True)
        self.gru = nn.RNN(input_dim, self.gru_hidden_dim, self.num_gru_layers, batch_first=True)
        self.to(device.device)

    def forward(self, x: torch.Tensor, prediction_start: int | None = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        h = torch.zeros(self.num_gru_layers, batch_size, self.gru_hidden_dim).to(device.device)
        if prediction_start is None:
            out, _ = self.gru(x, h)
            out = self.output_net(out)
        else:
            out = []
            last: Any = None
            for t in range(seq_length):
                x_t = x[:, t, :].unsqueeze(1)

                if t <= prediction_start:
                    out_t, h = self.gru(x_t, h)
                    out_t = self.output_net(out_t)
                    last = out_t

                else:  # feed the networks predictions back in.
                    assert last is not None
                    out_t_len = last.size(-1)
                    # replace the first out_t_len elements of x_t with out_t
                    # the network only predicts endogenous variables, so we grab the exogenous from that time step
                    # Note: out_t is the prediction of endogenous variables from the prev. time step
                    x_t = torch.cat((last, x_t[:, :, out_t_len:]), dim=-1)
                    out_t, h = self.gru(x_t, h)
                    out_t = self.output_net(out_t)
                    last = out_t

                out.append(out_t)
            out = torch.cat(out, dim=1)

        return out
