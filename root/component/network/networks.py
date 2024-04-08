import torch
import torch.nn as nn
import torch.distributions as distrib
import numpy as np
import root.component.network.utils as utils

from omegaconf import DictConfig

FLOAT32_EPS = 10 * \
              np.finfo(np.float32).eps  # differences of this size are
# representable up to ~ 15
EPSILON = 1e-6


# TODO: here is an example of initializing a actor_network.
class FC(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super().__init__()

        self.device = cfg.device
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
        self.to(self.device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out = self.network(input_tensor)
        out = self.head_act(out)
        return out


class EnsembleCritic(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(EnsembleCritic, self).__init__()
        self.device = cfg.device
        self.ensemble = cfg.ensemble
        self.subnetworks = [
            FC(cfg, input_dim, output_dim) for _ in range(self.ensemble)]
        self.to(self.device)

    def forward(self, input_tensor: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        qs = [net(input_tensor) for net in self.subnetworks]
        for i in range(self.ensemble):
            qs[i] = torch.unsqueeze(qs[i], 0)
        qs = torch.cat(qs, dim=0)
        q, _ = torch.min(qs, dim=0)

        return q, qs

    def state_dict(self) -> list:
        sd = [net.state_dict() for net in self.subnetworks]
        return sd

    def load_state_dict(self, state_dict_lst: list) -> None:
        for i in range(self.ensemble):
            self.subnetworks[i].load_state_dict(state_dict_lst[i])
        return

    def parameters(self, independent: bool = False) -> list:
        param_lst = []
        if independent:
            for i in range(self.ensemble):
                param_lst.append(self.subnetworks[i].parameters())
        else:
            for i in range(self.ensemble):
                param_lst += list(self.subnetworks[i].parameters())
        return param_lst


class SquashedGaussian(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(SquashedGaussian, self).__init__()

        self.device = cfg.device
        layer_init = utils.init_layer(cfg.layer_init)
        arch = cfg.arch

        if len(arch) > 0:
            self.base_network = FC(cfg, input_dim, arch[-1])
            self.mean_head = layer_init(nn.Linear(arch[-1], output_dim, bias=cfg.bias))
            self.logstd_head = layer_init(nn.Linear(arch[-1], output_dim, bias=cfg.bias))
        else:
            raise NotImplementedError

        self.to(self.device)

    # TODO: include n samples
    # TODO: rename observations to state
    def forward(self, observation: torch.Tensor, debug: bool = False) -> (torch.Tensor, dict):
        base = self.base_network(observation)
        mean = self.mean_head(base)
        log_std = torch.clamp(self.logstd_head(base), min=-20, max=2)
        std = log_std.exp()

        normal = distrib.Normal(mean, std)
        normal = distrib.Independent(normal, 1)
        out = normal.rsample()
        tanhout = torch.tanh(out)
        action = tanhout

        logp = normal.log_prob(out)
        logp -= torch.log((1 - tanhout.pow(2)) + EPSILON).sum(axis=-1)
        logp = logp.view((logp.shape[0], 1))

        info = {
            # "distribution": normal,
            "param1": mean.squeeze().detach().numpy(),
            "param2": std.squeeze().detach().numpy(),
        }

        if debug:
            info = {
                # "distribution": normal,
                "param1": mean.squeeze().detach().numpy(),
                "param2": std.squeeze().detach().numpy(),
            }
        else:
            info = None
        return action, info

    def log_prob(self, observation: torch.Tensor, action: torch.Tensor, debug: bool = False) -> (torch.Tensor, dict):
        base = self.base_network(observation)
        mean = self.mean_head(base)
        log_std = torch.clamp(self.logstd_head(base), min=-20, max=2)
        std = log_std.exp()

        normal = distrib.Normal(mean, std)
        normal = distrib.Independent(normal, 1)

        tanhout = action
        out = torch.atanh(torch.clamp(tanhout, -1.0 + EPSILON, 1.0 - EPSILON))
        logp = normal.log_prob(out)
        logp -= torch.log((1 - tanhout.pow(2)) + EPSILON).sum(axis=-1).reshape(logp.shape)

        if debug:
            info = {
                "param1": mean.squeeze().detach().numpy(),
                "param2": std.squeeze().detach().numpy()
            }
        else:
            info = None
        logp = logp.view(-1, 1)
        return logp, info


class BetaPolicy(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(BetaPolicy, self).__init__()

        layer_init = utils.init_layer(cfg.layer_init)
        arch = cfg.arch
        self.device = cfg.device
        head_activation = cfg.head_activation
        bias = cfg.bias
        beta_param_bias = cfg.beta_param_bias
        beta_param_bound = cfg.beta_param_bound

        if len(arch) > 0:
            self.base_network = FC(cfg, input_dim, arch[-1])
            self.alpha_head = layer_init(nn.Linear(arch[-1], output_dim, bias=bias))
            self.beta_head = layer_init(nn.Linear(arch[-1], output_dim, bias=bias))
        else:
            """ 
            A special case of learning alpha and beta directly. 
            Initialize the weight using constant  
            """
            self.base_network = lambda x: x
            self.alpha_head = layer_init(nn.Linear(input_dim, output_dim, bias=False))
            self.beta_head = layer_init(nn.Linear(input_dim, output_dim, bias=False))
        self.head_activation_fn = utils.init_activation_function(head_activation)
        self.beta_param_bias = torch.tensor(beta_param_bias)
        self.beta_param_bound = torch.tensor(beta_param_bound)
        self.to(self.device)

    def squash_dist_param(self, dist_param: torch.Tensor, low: float, high: float) -> torch.Tensor:
        tanh_out = torch.tanh(dist_param)
        normalized_param = ((tanh_out + 1) / 2)
        scaled_param = normalized_param * (high - low) + low  # ∈ [low, high]

        return scaled_param

    def get_dist_params(self, observation: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if self.beta_param_bound == 0:
            """ Not using the squash function"""
            base = self.base_network(observation)
            alpha = self.head_activation_fn(self.alpha_head(base)) + EPSILON
            beta = self.head_activation_fn(self.beta_head(base)) + EPSILON
            alpha += self.beta_param_bias
            beta += self.beta_param_bias
        else:
            base = self.base_network(observation)
            alpha_head_out = self.alpha_head(base)
            beta_head_out = self.beta_head(base)
            low = self.beta_param_bias
            high = self.beta_param_bound
            alpha = self.squash_dist_param(alpha_head_out, low, high)
            beta = self.squash_dist_param(beta_head_out, low, high)
        return alpha, beta

    def forward(self, observation: torch.Tensor, debug: bool = False) -> (torch.Tensor, dict):
        alpha, beta = self.get_dist_params(observation)
        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)
        out = dist.rsample()  # samples of alpha and beta

        logp = dist.log_prob(torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS))
        logp = logp.view((logp.shape[0], 1))
        action = out

        return action, {'logp': logp}

    def log_prob(self, observation: torch.Tensor, action: torch.Tensor, debug: bool = False) -> (torch.Tensor, dict):
        out = action
        out = torch.clamp(out, 0, 1)

        alpha, beta = self.get_dist_params(observation)
        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)

        logp = dist.log_prob(torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS))
        logp = logp.view(-1, 1)
        return logp, {}


class Softmax(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super(Softmax, self).__init__()
        self.output_dim = output_dim
        self.base_network = FC(cfg, input_dim, output_dim)
        self.to(cfg.device)
        self.device = cfg.device

    def get_probs(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.base_network(state)
        probs = nn.functional.softmax(x, dim=1)
        return probs, x

    def forward(self, state: torch.Tensor, debug: bool = False) -> (torch.Tensor, dict):
        probs, x = self.get_probs(state)
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()

        logp = dist.log_prob(actions)
        logp = logp.view((logp.shape[0], 1))

        actions = actions.reshape((-1, 1))
        a_onehot = torch.FloatTensor(actions.size()[0], self.output_dim)
        a_onehot.zero_()
        actions = a_onehot.scatter_(1, actions, 1)
        return actions,  {'logp': logp}

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor, debug: bool = False) -> (torch.Tensor, dict):
        actions = (actions == 1).nonzero(as_tuple=False)
        actions = actions[:, 1:]
        probs, _ = self.get_probs(states)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(actions.squeeze(-1))
        logp = logp.view(-1, 1)
        return logp, {}
