import time
import torch
import torch.nn as nn
import torch.distributions as distrib
import numpy as np
import corerl.component.network.utils as utils
from corerl.utils.device import device

from omegaconf import DictConfig

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


def _init_ensemble_reduct(cfg: DictConfig):
    reduct = cfg.reduct
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
    elif reduct.lower() == "percentile":
        def _f(x, dim):
            return _percentile_bootstrap(
                x, dim, cfg.bootstrap_batch_size, cfg.bootstrap_samples,
                cfg.percentile,
            )
    else:
        raise ValueError(f"unknown reduct type {reduct}")

    return _f


def create_base(
        cfg: DictConfig, input_dim: int, output_dim: int,
) -> nn.Module:
    if cfg.name == "fc":
        return FC(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError


# TODO: here is an example of initializing a network.
class FC(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super().__init__()
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
        self.to(device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out = self.network(input_tensor)
        out = self.head_act(out)
        return out


class EnsembleCritic(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(EnsembleCritic, self).__init__()
        self.ensemble = cfg.ensemble
        self.subnetworks = [
            create_base(cfg.base, input_dim, output_dim)
            for _ in range(self.ensemble)
        ]
        self._reduct = _init_ensemble_reduct(cfg)
        self.to(device)

    def forward(
            self, input_tensor: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        qs = [net(input_tensor) for net in self.subnetworks]
        for i in range(self.ensemble):
            qs[i] = torch.unsqueeze(qs[i], 0)
        qs = torch.cat(qs, dim=0)
        q = self._reduct(qs, dim=0)

        return q, qs

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


class SquashedGaussian(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(SquashedGaussian, self).__init__()
        layer_init = utils.init_layer(cfg.layer_init)
        arch = cfg.base.arch

        if len(arch) > 0:
            self.base_network = create_base(cfg.base, input_dim, arch[-1])
            self.mean_head = layer_init(
                nn.Linear(arch[-1], output_dim, bias=cfg.bias),
            )
            self.logstd_head = layer_init(
                nn.Linear(arch[-1], output_dim, bias=cfg.bias),
            )
        else:
            raise NotImplementedError

        self.to(device)

    def distribution_bounds(self):
        return -1, 1

    # TODO: include n samples
    def forward(self, state: torch.Tensor) -> (torch.Tensor, dict):
        base = self.base_network(state)
        mean = self.mean_head(base)
        log_std = torch.clamp(self.logstd_head(base), min=-20, max=2)
        std = log_std.exp()

        normal = distrib.Normal(mean, std)
        normal = distrib.Independent(normal, 1)
        out = normal.rsample()
        tanhout = torch.tanh(out)
        action = tanhout
        action = (action + 1) / 2

        info = {
            "mean": mean.squeeze().detach().numpy(),
            "variance": std.squeeze().detach().numpy() ** 2,
            "param1": mean.squeeze().detach().numpy(),
            "param2": std.squeeze().detach().numpy(),
        }

        return action, info

    def log_prob(
            self, state: torch.Tensor, action: torch.Tensor,
    ) -> (torch.Tensor, dict):
        base = self.base_network(state)
        mean = self.mean_head(base)
        log_std = torch.clamp(self.logstd_head(base), min=-20, max=2)
        std = log_std.exp()

        normal = distrib.Normal(mean, std)
        normal = distrib.Independent(normal, 1)

        tanhout = action * 2 - 1
        out = torch.atanh(torch.clamp(tanhout, -1.0 + EPSILON, 1.0 - EPSILON))
        logp = normal.log_prob(out)
        logp -= torch.log(
            (1 - tanhout.pow(2)) + EPSILON
        ).sum(axis=-1).reshape(logp.shape)
        logp += torch.log(2)

        info = None

        logp = logp.view(-1, 1)
        return logp, info


class BetaPolicy(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(BetaPolicy, self).__init__()
        layer_init = utils.init_layer(cfg.layer_init)
        arch = cfg.base.arch
        head_activation = cfg.head_activation
        bias = cfg.bias
        beta_param_bias = cfg.beta_param_bias
        beta_param_bound = cfg.beta_param_bound

        if len(arch) > 0:
            self.base_network = create_base(cfg.base, input_dim, arch[-1])
            self.alpha_head = layer_init(
                nn.Linear(arch[-1], output_dim, bias=bias),
            )
            self.beta_head = layer_init(
                nn.Linear(arch[-1], output_dim, bias=bias),
            )
        else:
            """
            A special case of learning alpha and beta directly.
            Initialize the weight using constant
            """
            self.base_network = lambda x: x
            self.alpha_head = layer_init(
                nn.Linear(input_dim, output_dim, bias=False),
            )
            self.beta_head = layer_init(
                nn.Linear(input_dim, output_dim, bias=False),
            )
        self.head_activation_fn = utils.init_activation_function(
            head_activation,
        )
        self.beta_param_bias = torch.tensor(beta_param_bias)
        self.beta_param_bound = torch.tensor(beta_param_bound)
        self.tanh_shift = cfg.tanh_shift
        self.to(device)

    def distribution_bounds(self):
        return 0, 1

    def squash_dist_param(
            self,
            dist_param: torch.Tensor,
            low: float | torch.Tensor,
            high: float | torch.Tensor,
    ) -> torch.Tensor:
        tanh_out = torch.tanh(dist_param + self.tanh_shift)
        normalized_param = ((tanh_out + 1) / 2)
        scaled_param = normalized_param * (high - low) + low  # ∈ [low, high]
        return scaled_param

    def get_dist_params(
            self, state: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        if self.beta_param_bound == 0:
            """ Not using the squash function"""
            base = self.base_network(state)
            alpha = self.head_activation_fn(self.alpha_head(base)) + EPSILON
            beta = self.head_activation_fn(self.beta_head(base)) + EPSILON
            alpha += self.beta_param_bias
            beta += self.beta_param_bias
        else:
            base = self.base_network(state)
            alpha_head_out = self.alpha_head(base)
            beta_head_out = self.beta_head(base)
            low = self.beta_param_bias
            high = self.beta_param_bound
            alpha = self.squash_dist_param(alpha_head_out, low, high)
            beta = self.squash_dist_param(beta_head_out, low, high)
        return alpha, beta

    def forward(self, state: torch.Tensor) -> (torch.Tensor, dict):
        alpha, beta = self.get_dist_params(state)
        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)
        out = dist.rsample()  # samples of alpha and beta

        logp = dist.log_prob(
            torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS),
        )
        logp = logp.view((logp.shape[0], 1))
        action = out

        # see https://en.wikipedia.org/wiki/Beta_distribution
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        info = {
            'logp': logp,
            'mean': mean,
            'variance': variance,
            "param1": alpha,
            "param2": beta,
        }

        return action, info

    def log_prob(
            self, state: torch.Tensor, action: torch.Tensor,
    ) -> (torch.Tensor, dict):
        out = action
        out = torch.clamp(out, 0, 1)

        alpha, beta = self.get_dist_params(state)
        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)

        logp = dist.log_prob(
            torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS),
        )
        logp = logp.view(-1, 1)
        return logp, {}


class Softmax(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(Softmax, self).__init__()
        self.output_dim = output_dim
        self.base_network = create_base(cfg.base, input_dim, output_dim)
        self.to(device)

    def get_probs(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.base_network(state)
        probs = nn.functional.softmax(x, dim=1)
        return probs, x

    def forward(
            self, state: torch.Tensor, debug: bool = False,
    ) -> (torch.Tensor, dict):
        probs, x = self.get_probs(state)
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()

        logp = dist.log_prob(actions)
        logp = logp.view((logp.shape[0], 1))

        actions = actions.reshape((-1, 1))
        a_onehot = torch.FloatTensor(actions.size()[0], self.output_dim)
        a_onehot.zero_()
        actions = a_onehot.scatter_(1, actions, 1)
        return actions, {'logp': logp}

    def log_prob(
            self, states: torch.Tensor, actions: torch.Tensor, debug: bool = False,
    ) -> (torch.Tensor, dict):
        actions = (actions == 1).nonzero(as_tuple=False)
        actions = actions[:, 1:]
        probs, _ = self.get_probs(states)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(actions.squeeze(-1))
        logp = logp.view(-1, 1)
        return logp, {}


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
        self.to(cfg.device)
        self.device = cfg.device

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.random_network(input_tensor)
        out = self.linear_head(base)
        info = {
            "rep": base.squeeze().detach().numpy(),
            "out": out.squeeze().detach().numpy(),
        }
        return out, info


class UniformRandomCont(BetaPolicy):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(UniformRandomCont, self).__init__(cfg, input_dim, output_dim)
        self.output_dim = output_dim

    def get_dist_params(self, state):
        alpha = torch.ones(state.size()[0], self.output_dim)
        beta = torch.ones(state.size()[0], self.output_dim)
        return alpha, beta


class UniformRandomDisc(Softmax):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(UniformRandomDisc, self).__init__(cfg, input_dim, output_dim)
        self.output_dim = output_dim

    def get_probs(self, state):
        x = torch.ones(state.size()[0], self.output_dim)
        probs = nn.functional.softmax(x, dim=1)
        return probs, x


class GRU(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.gru_hidden_dim = cfg.gru_hidden_dim
        self.num_gru_layers = cfg.num_gru_layers
        self.output_net = create_base(cfg.base, self.gru_hidden_dim, output_dim)
        # self.gru = nn.GRU(input_dim, self.gru_hidden_dim, self.num_gru_layers, batch_first=True)
        self.gru = nn.RNN(input_dim, self.gru_hidden_dim, self.num_gru_layers, batch_first=True)
        self.to(device)

    def forward(self, x: torch.Tensor, prediction_start=None) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        h = torch.zeros(self.num_gru_layers, batch_size, self.gru_hidden_dim).to(device)
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
                    # Note: out_t is the prediction of endogenous variables from the prev. time ste[
                    # print(x_t[0, 0, :])
                    x_t = torch.cat((out_t, x_t[:, :, out_t_len:]), dim=-1)
                    #
                    # print(x_t[0, 0, :])
                    # print("\n")
                    out_t, h = self.gru(x_t, h)
                    out_t = self.output_net(out_t)

                out.append(out_t)
            out = torch.cat(out, dim=1)

        return out