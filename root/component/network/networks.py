import torch
import torch.nn as nn
import torch.distributions as distrib
import numpy as np
import root.component.network.utils as utils

FLOAT32_EPS = 10 * \
              np.finfo(np.float32).eps  # differences of this size are
# representable up to ~ 15
EPSILON = 1e-6


# TODO: here is an example of initializing a network.
class FC(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__()

        device = cfg.device
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

    def forward(self, input_tensor):
        out = self.network(input_tensor)
        out = self.head_act(out)
        return out


class EnsembleCritic(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(EnsembleCritic, self).__init__()

        device = cfg.device
        self.ensemble = cfg.ensemble
        self.subnetworks = [
            FC(cfg, input_dim, output_dim) for _ in range(self.ensemble)]
        self.to(device)

    def forward(self, input_tensor):
        qs = [net(input_tensor) for net in self.subnetworks]
        for i in range(self.ensemble):
            qs[i] = torch.unsqueeze(qs[i], 0)
        qs = torch.cat(qs, dim=0)
        q, _ = torch.min(qs, dim=0)

        return q, qs

    def state_dict(self):
        sd = [net.state_dict() for net in self.subnetworks]
        return sd

    def load_state_dict(self, state_dict_lst):
        for i in range(self.ensemble):
            self.subnetworks[i].load_state_dict(state_dict_lst[i])
        return

    def parameters(self, independent=False):
        param_lst = []
        if independent:
            for i in range(self.ensemble):
                param_lst.append(self.subnetworks[i].parameters())
        else:
            for i in range(self.ensemble):
                param_lst += list(self.subnetworks[i].parameters())
        return param_lst


class SquashedGaussian(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(SquashedGaussian, self).__init__()

        device = cfg.device
        layer_init = utils.init_layer(cfg.layer_init)
        arch = cfg.arch

        if len(arch) > 0:
            self.base_network = FC(cfg, input_dim, arch[-1])
            self.mean_head = layer_init(nn.Linear(arch[-1], output_dim, bias=cfg.bias))
            self.logstd_head = layer_init(nn.Linear(arch[-1], output_dim, bias=cfg.bias))
        else:
            raise NotImplementedError

        self.to(device)

    # TODO: include n samples
    def forward(self, observation, debug=False):
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

    def log_prob(self, observation, action, debug=False):
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
    def __init__(self, cfg, input_dim, output_dim):
        super(BetaPolicy, self).__init__()

        layer_init = utils.init_layer(cfg.layer_init)
        arch = cfg.arch
        device = cfg.device
        head_activation = cfg.head_activation
        bias = cfg.bias
        beta_param_bias = cfg.beta_param_bias
        beta_param_bound = cfg.beta_param_bound

        if len(arch) > 0:
            self.base_network = FC(cfg, input_dim, arch[-1])
            self.alpha_head = layer_init(nn.Linear(arch[-1], output_dim, bias=bias))
            self.beta_head = layer_init(nn.Linear(arch[-1],  output_dim, bias=bias))
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
        self.to(device)
        self.device = device

    def squash_dist_param(self, dist_param, low, high):
        tanh_out = torch.tanh(dist_param)
        normalized_param = ((tanh_out + 1) / 2)
        scaled_param = normalized_param * (high - low) + low  # ∈ [low, high]

        return scaled_param

    def get_dist_params(self, observation):
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

    def forward(self, observation, debug=False):
        alpha, beta = self.get_dist_params(observation)
        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)
        out = dist.rsample()  # samples of alpha and beta

        logp = dist.log_prob(torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS))
        logp = logp.view((logp.shape[0], 1))
        action = out
        if debug:
            info = {
                # "distribution": dist,
                "param1": alpha.squeeze().detach().numpy(),
                "param2": beta.squeeze().detach().numpy()
            }
        else:
            info = None

        return action, info

    def log_prob(self, observation, action, debug=False):
        out = action
        out = torch.clamp(out, 0, 1)

        # base = self.base_network(observation)
        # alpha = self.head_activation_fn(self.alpha_head(base)) + EPSILON
        # beta = self.head_activation_fn(self.beta_head(base)) + EPSILON
        # alpha += self.beta_param_bias
        # beta += self.beta_param_bias

        alpha, beta = self.get_dist_params(observation)

        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)

        logp = dist.log_prob(torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS))

        if debug:
            info = {
                # "distribution": dist,
                "param1": alpha.squeeze().detach().numpy(),
                "param2": beta.squeeze().detach().numpy()
            }
        else:
            info = None
        logp = logp.view(-1, 1)
        return logp, info

