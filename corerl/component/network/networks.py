import copy
import torch
import torch.nn as nn
from torch.func import stack_module_state, functional_call
from collections.abc import Mapping
import numpy as np
import corerl.component.network.utils as utils
from corerl.utils.device import device
import corerl.component.layer as layer

from warnings import warn

from omegaconf import DictConfig
from typing import Optional

# Differences of this size are representable up to ~ 15
FLOAT32_EPS = 10 * np.finfo(np.float32).eps
EPSILON = 1e-6


def _percentile_bootstrap(
        x, dim, batch_size, n_samples, percentile, statistic=torch.mean
):
    size = (*x.shape[:dim], batch_size * n_samples, *x.shape[dim + 1:])

    # Randomly sampling integers from numpy is faster than from torch
    ind = np.random.randint(0, x.shape[dim], size)
    samples = torch.gather(x, dim, torch.from_numpy(ind))

    size = (
        *x.shape[:dim],
        batch_size,
        n_samples,
        *x.shape[dim + 1:],
    )
    samples = samples.reshape(size)
    bootstr_stat = statistic(samples, dim=(len(x.shape[:dim])))

    return torch.quantile(bootstr_stat, percentile, dim=dim)


def _init_ensemble_reduct(cfg: DictConfig, reduct: str):
    if reduct.startswith("torch.nn."):
        return getattr(torch, reduct[9:])
    elif reduct.startswith("torch."):
        return getattr(torch, reduct[6:])
    elif reduct.lower() == "min":
        def _f(x, dim):
            return torch.min(x, dim=dim)[0]
    elif reduct.lower() == "max":
        def _f(x, dim):
            return torch.max(x, dim=dim)[0]
    elif reduct.lower() == "mean":
        def _f(x, dim):
            return torch.mean(x, dim=dim)
    elif reduct.lower() == "median":
        def _f(x, dim):
            return torch.quantile(x, q=0.5, dim=dim)
    elif reduct.lower() == "percentile":
        def _f(x, dim):
            return _percentile_bootstrap(
                x, dim, cfg.bootstrap_batch_size, cfg.bootstrap_samples,
                cfg.percentile,
            )
    else:
        raise ValueError(f"unknown reduct type {reduct}")

    return _f

def _init_ensemble_reducts(cfg: DictConfig):
    bootstrap_reduct = cfg.bootstrap_reduct
    bootstrap_reduct_fn = _init_ensemble_reduct(cfg, bootstrap_reduct)

    policy_reduct = cfg.policy_reduct
    policy_reduct_fn = _init_ensemble_reduct(cfg, policy_reduct)

    return bootstrap_reduct_fn, policy_reduct_fn


def create_base(cfg: Mapping, input_dim: int, output_dim: Optional[int]):
    if cfg["name"].lower() in ("mlp", "fc"):
        return _create_base_mlp(cfg, input_dim, output_dim)
    else:
        raise ValueError(f"unknown network type {cfg['name']}")


def _create_base_mlp(cfg: Mapping, input_dim: int, output_dim: Optional[int]):
    assert cfg["name"].lower() in ("mlp", "fc")

    hidden = cfg["hidden"]
    act = cfg["activation"]
    bias = cfg["bias"]
    assert len(hidden) == len(act)
    layer_init = utils.init_layer(cfg["layer_init"])

    # In the previous iteration of the codebase, the create_base function
    # allowed for the creation of activations on network heads/output layers.
    # Here, we explicitly disallow that, since the feature was never even used.
    #
    # That being said, this was a central part in how networks were
    # constructed. Hence for now, we are going to explicitly raise an error
    # when such keys exist in the configuration, to ensure everyone is aware of
    # this change.
    ks = cfg.keys()
    filt = list(filter(lambda x: x.startswith("head_"), ks))
    if len(filt) > 0:
        warn(
            f"create_base: unexpected config key(s) {filt}"
        )

    net = []

    # Add the first layer to the network
    layer_ = nn.Linear(input_dim, hidden[0], bias=bias)
    layer_ = layer_init(layer_)
    net.append(layer_)
    net.append(layer.init_activation(act[0]))

    placeholder_input = torch.empty((input_dim,))

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
    layer_init,
    base_net,
    hidden,
    bias,
    placeholder_input,
):
    if layer_type is nn.Linear:
        n_inputs = _get_output_shape(
            base_net, placeholder_input, dim=0,
        )
        layer = layer_type(n_inputs, hidden, bias=bias)
        return layer_init(layer)

    raise NotImplementedError(f"unknown layer type {layer_type}")


def _get_output_shape(net, placeholder_input, *, dim=None):
    output_shape = nn.Sequential(*net)(placeholder_input).shape
    assert len(output_shape) == 1
    return output_shape[dim]


# TODO: here is an example of initializing a network.
class FC(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        warn(
            "FC is deprecated and will be removed in a future version" +
            "to create an MLP, use `create_base` instead"
        )
        super(FC, self).__init__()
        layer_norm = cfg.layer_norm
        arch = cfg.arch
        activation = cfg.activation
        head_activation = cfg.head_activation
        layer_init = utils.init_layer(cfg.layer_init)
        activation_cls = utils.init_activation(activation)

        d = input_dim
        modules = []
        for hidden_size in arch:
            fc = layer_init(nn.Linear(d, hidden_size, bias=cfg.bias))
            modules.append(fc)
            if layer_norm:
                modules.append(nn.LayerNorm(hidden_size))
            modules.append(activation_cls())
            d = hidden_size
        last_fc = layer_init(nn.Linear(d, output_dim, bias=cfg.bias))
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)
        self.head_act = utils.init_activation(head_activation)()
        self.to(device.device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out = self.network(input_tensor)
        out = self.head_act(out)
        return out


class EnsembleFC(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(EnsembleFC, self).__init__()
        self.ensemble = cfg.ensemble
        self.subnetworks = [
            create_base(cfg.base, input_dim, output_dim)
            for _ in range(self.ensemble)
        ]
        self.to(device.device)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outs = [net(input_tensor) for net in self.subnetworks]
        for i in range(self.ensemble):
            outs[i] = torch.unsqueeze(outs[i], 0)
        outs = torch.cat(outs, dim=0)
        return outs

    def state_dict(self) -> list:
        sd = [net.state_dict() for net in self.subnetworks]
        return sd

    def load_state_dict(self, state_dict_list: list) -> None:
        for i in range(self.ensemble):
            self.subnetworks[i].load_state_dict(state_dict_list[i])
        return

    def parameters(self, independent: bool = False) -> list:
        param_list = []
        if independent:
            for i in range(self.ensemble):
                param_list.append(self.subnetworks[i].parameters())
        else:
            for i in range(self.ensemble):
                param_list += list(self.subnetworks[i].parameters())
        return param_list


class EnsembleCritic(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(EnsembleCritic, self).__init__()
        self.ensemble = cfg.ensemble
        self.vmap = cfg.vmap
        self.subnetworks = [
            create_base(cfg.base, input_dim, output_dim)
            for _ in range(self.ensemble)
        ]

        # Vectorizing the ensemble to use torch.vmap to avoid sequentially querrying the ensemble
        self.params, self.buffers = stack_module_state(self.subnetworks)

        self.base_model = copy.deepcopy(self.subnetworks[0])
        self.base_model = self.base_model.to(device.device)

        self.bootstrap_reduct, self.policy_reduct = _init_ensemble_reducts(cfg)
        self.to(device.device)

    def fmodel(self, params, buffers, x: torch.Tensor):
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

    def state_dict(self) -> list:
        sd = [net.state_dict() for net in self.subnetworks]
        return sd

    def load_state_dict(self, state_dict_list: list) -> None:
        for i in range(self.ensemble):
            self.subnetworks[i].load_state_dict(state_dict_list[i])
        return

    def parameters(self, independent: bool = False) -> list:
        if self.vmap:
            # https://github.com/pytorch/pytorch/issues/120581
            return self.params.values()
        else:
            param_list = []
            if independent:
                for i in range(self.ensemble):
                    param_list.append(self.subnetworks[i].parameters())
            else:
                for i in range(self.ensemble):
                    param_list += list(self.subnetworks[i].parameters())
            return param_list


class RndLinearUncertainty(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(RndLinearUncertainty, self).__init__()
        self.output_dim = output_dim
        arch = cfg.arch
        self.random_network = FC(cfg, input_dim, arch[-1])
        layer_init = utils.init_layer(cfg.layer_init)
        self.linear_head = layer_init(
            nn.Linear(arch[-1], output_dim, bias=cfg.bias),
        )
        self.to(device.device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.random_network(input_tensor)
        out = self.linear_head(base)
        info = {
            "rep": base.squeeze().detach().numpy(),
            "out": out.squeeze().detach().numpy(),
        }
        return out, info


class GRU(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.gru_hidden_dim = cfg.gru_hidden_dim
        self.num_gru_layers = cfg.num_gru_layers
        self.output_net = create_base(cfg.base, self.gru_hidden_dim, output_dim)
        # self.gru = nn.GRU(input_dim, self.gru_hidden_dim, self.num_gru_layers, batch_first=True)
        self.gru = nn.RNN(input_dim, self.gru_hidden_dim, self.num_gru_layers, batch_first=True)
        self.to(device.device)

    def forward(self, x: torch.Tensor, prediction_start=None) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        h = torch.zeros(self.num_gru_layers, batch_size, self.gru_hidden_dim).to(device.device)
        if prediction_start is None:
            out, _ = self.gru(x, h)
            out = self.output_net(out)
        else:
            out = []
            for t in range(seq_length):
                x_t = x[:, t, :].unsqueeze(1)

                if t <= prediction_start:
                    out_t, h = self.gru(x_t, h)
                    out_t = self.output_net(out_t)

                else:  # feed the networks predictions back in.
                    out_t_len = out_t.size(-1)
                    # replace the first out_t_len elements of x_t with out_t
                    # the network only predicts endogenous variables, so we grab the exogenous from that time step
                    # Note: out_t is the prediction of endogenous variables from the prev. time step
                    x_t = torch.cat((out_t, x_t[:, :, out_t_len:]), dim=-1)
                    out_t, h = self.gru(x_t, h)
                    out_t = self.output_net(out_t)

                out.append(out_t)
            out = torch.cat(out, dim=1)

        return out
