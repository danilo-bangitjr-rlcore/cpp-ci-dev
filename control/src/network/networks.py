import torch
import src.network.torch_utils as internal_factory
import torch.nn as nn
import torch.distributions as distrib
import numpy as np

from src.network import torch_utils

FLOAT32_EPS = 10 * \
              np.finfo(np.float32).eps # differences of this size are
                                       # representable up to ~ 15
EPSILON = 1e-6



class FC(nn.Module):
    def __init__(self, device, input_dim, arch, output_dim, activation="ReLU", head_activation="None", init='Xavier',
                 layer_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        init_args = init.split("/")
        layer_init = internal_factory.init_layer(init_args[0])
        activation_cls = internal_factory.init_activation(activation)
        d = input_dim
        modules = []
        for hidden_size in arch:
            fc = layer_init(nn.Linear(d, hidden_size, bias=bool(init_args[-1])), *init_args[1:])
            modules.append(fc)
            if layer_norm:
                modules.append(nn.LayerNorm(hidden_size))
            modules.append(activation_cls())
            d = hidden_size
        last_fc = layer_init(nn.Linear(d, output_dim, bias=bool(init_args[-1])), *init_args[1:])
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)
        self.head_act = internal_factory.init_activation(head_activation)()
        self.to(device)

    def forward(self, input_tensor):
        out = self.network(input_tensor)
        out = self.head_act(out)
        return out


class SquashedGaussianPolicy(nn.Module):
    def __init__(self, device, observation_dim, arch, action_dim, init='Xavier', activation="ReLU", layer_norm=False):
        super(SquashedGaussianPolicy, self).__init__()

        init_args = init.split("/")
        layer_init = internal_factory.init_layer(init_args[0])
        if len(arch) > 0:
            self.base_network = FC(device, observation_dim, arch[:-1], arch[-1], activation=activation, head_activation=activation,
                                   init=init, layer_norm=layer_norm)
            self.mean_head = layer_init(nn.Linear(arch[-1], action_dim, bias=bool(init_args[-1])), *init_args[1:])
            self.logstd_head = layer_init(nn.Linear(arch[-1], action_dim, bias=bool(init_args[-1])), *init_args[1:])
        else:
            raise NotImplementedError


        # self.action_clip = [c * action_scale + action_bias for c in action_clip]
        self.to(device)

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
     
        # action = torch.clamp(action, min=self.action_clip[0], max=self.action_clip[1])
        logp = normal.log_prob(out)
        logp -= torch.log((1 - tanhout.pow(2)) + EPSILON).sum(axis=-1).reshape(logp.shape)

        if debug:
            info = {
                # "distribution": normal,
                "param1": mean.squeeze().detach().numpy(),
                "param2": std.squeeze().detach().numpy(),
            }
        else:
            info = None
        return action, logp, info

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
                # "distribution": normal,
                "param1": mean.squeeze().detach().numpy(),
                "param2": std.squeeze().detach().numpy()
            }
        else:
            info = None
        return logp, info


class BetaPolicy(nn.Module):
    def __init__(self, device, observation_dim, arch, action_dim,
                 beta_param_bias=0, beta_param_bound=1e8,
                 init='Xavier', activation="ReLU", head_activation="Softplus", layer_norm=False):
        super(BetaPolicy, self).__init__()

        self.action_dim = action_dim

        init_args = init.split("/")
        layer_init = internal_factory.init_layer(init_args[0])
        if len(arch) > 0:
            self.base_network = FC(device, observation_dim, arch[:-1], arch[-1], activation=activation,
                                   head_activation=activation,
                                   init=init, layer_norm=layer_norm)
            self.alpha_head = layer_init(nn.Linear(arch[-1], action_dim, bias=bool(init_args[-1])), *init_args[1:])
            self.beta_head = layer_init(nn.Linear(arch[-1], action_dim, bias=bool(init_args[-1])), *init_args[1:])
        else:
            """ 
            A special case of learning alpha and beta directly. 
            Initialize the weight using constant  
            """
            layer_init = internal_factory.init_layer(init_args[0])
            self.base_network = lambda x: x
            self.alpha_head = layer_init(nn.Linear(observation_dim, action_dim, bias=False), *init_args[1:])
            self.beta_head = layer_init(nn.Linear(observation_dim, action_dim, bias=False), *init_args[1:])
        self.head_activation_fn = internal_factory.init_activation_function(head_activation)
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
        # base = self.base_network(observation)
        # alpha = self.head_activation_fn(self.alpha_head(base)) + EPSILON
        # beta = self.head_activation_fn(self.beta_head(base)) + EPSILON
        # alpha += self.beta_param_bias
        # beta += self.beta_param_bias

        alpha, beta = self.get_dist_params(observation)
        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)
        out = dist.rsample()  # samples of alpha and beta

        logp = dist.log_prob(torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS))
        action = out
        # action = torch.clamp(action, min=self.action_clip[0], max=self.action_clip[1])
        if debug:
            info = {
                # "distribution": dist,
                "param1": alpha.squeeze().detach().numpy(),
                "param2": beta.squeeze().detach().numpy()
            }
        else:
            info = None

        return action, logp, info

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
        return logp, info


class BetaInvParam(nn.Module):
    def __init__(self, device, observation_dim, arch, action_dim,
                 init='Xavier', activation="ReLU", head_activation="Softplus", layer_norm=False):
        super(BetaInvParam, self).__init__()

        self.action_dim = action_dim

        init_args = init.split("/")
        layer_init = internal_factory.init_layer(init_args[0])
        if len(arch) > 0:
            self.base_network = FC(device, observation_dim, arch[:-1], arch[-1], activation=activation, head_activation=activation,
                                   init=init, layer_norm=layer_norm)
            self.inv_alpha_head = layer_init(nn.Linear(arch[-1], action_dim, bias=bool(init_args[-1])), *init_args[1:])
            self.inv_beta_head = layer_init(nn.Linear(arch[-1], action_dim, bias=bool(init_args[-1])), *init_args[1:])
        else:
            """ 
            A special case of learning alpha and beta directly. 
            Initialize the weight using constant 
            """
            layer_init = internal_factory.init_layer(init_args[0])
            self.base_network = lambda x:x
            self.inv_alpha_head = layer_init(nn.Linear(observation_dim, action_dim, bias=False), *init_args[1:])
            self.inv_beta_head = layer_init(nn.Linear(observation_dim, action_dim, bias=False), *init_args[1:])
        self.head_activation_fn = internal_factory.init_activation_function(head_activation)
        # self.beta_param_bias = torch.tensor(beta_param_bias) # shouldn't need this
        self.to(device)

    def forward(self, observation, debug=False):
        base = self.base_network(observation)
        inv_alpha = self.head_activation_fn(self.inv_alpha_head(base)) + EPSILON
        inv_beta = self.head_activation_fn(self.inv_beta_head(base)) + EPSILON
        alpha = inv_alpha.inverse()
        beta = inv_beta.inverse()

        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)
        out = dist.rsample() # samples of alpha and beta

        logp = dist.log_prob(torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS))
        action = out
        if debug:
            info = {
                # "distribution": dist,
                "param1": alpha.squeeze().detach().numpy(),
                "param2": beta.squeeze().detach().numpy()
            }
        else:
            info = None

        return action, logp, info

    def log_prob(self, observation, action, debug=False):
        out = action
        out = torch.clamp(out, 0, 1)

        base = self.base_network(observation)
        inv_alpha = self.head_activation_fn(self.inv_alpha_head(base)) + EPSILON
        inv_beta = self.head_activation_fn(self.inv_beta_head(base)) + EPSILON
        alpha = inv_alpha.inverse()
        beta = inv_beta.inverse()

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
        return logp, info


class Softmax(nn.Module):
    def __init__(self, device, observation_dim, arch, num_actions, init='Xavier', activation="ReLU", layer_norm=False):
        super(Softmax, self).__init__()
        self.num_actions = num_actions
        self.base_network = FC(device, observation_dim, arch, num_actions, init=init, activation=activation, layer_norm=layer_norm)
        self.to(device)
        self.device = device

    def get_probs(self, state):
        x = self.base_network(state)
        probs = nn.functional.softmax(x, dim=1)
        return probs, x

    def forward(self, state, debug=False):
        probs, x = self.get_probs(state)
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()

        
        # log_prob = nn.functional.log_softmax(x, dim=1)
        # log_prob = torch.gather(log_prob, dim=1, index=actions)

        log_prob = dist.log_prob(actions)

        if debug:
            info = {
                # "distribution": dist,
                "param1": x.squeeze().detach().numpy(),
            }
        else:
            info = None
        actions = actions.reshape((-1, 1))
        a_onehot = torch.FloatTensor(actions.size()[0], self.num_actions)
        a_onehot.zero_()
        actions = a_onehot.scatter_(1, actions, 1)
        return actions, log_prob, info

    def log_prob(self, states, actions, debug=False):
        actions = (actions == 1).nonzero(as_tuple=False)
        actions = actions[:, 1:]

        probs, _ = self.get_probs(states)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions.squeeze(-1))

        if debug:
            info = {
                # "distribution": dist,
                "param1": probs.squeeze().detach().numpy(),
            }
        else:
            info = None
        return log_prob, info


class RndLinearUncertainty(nn.Module):
    def __init__(self, device, input_dim, hidden_units, output_dim, activation, init, layer_norm):
        super(RndLinearUncertainty, self).__init__()
        init_args = init.split("/")
        layer_init = internal_factory.init_layer(init_args[0])
        self.random_network = FC(device, input_dim, hidden_units[:-1], hidden_units[-1], activation=activation,
                                 head_activation="None", init=init, layer_norm=layer_norm)
        self.linear_head = layer_init(nn.Linear(hidden_units[-1], output_dim, bias=bool(init_args[-1])), *init_args[1:])
        self.to(device)

    def forward(self, in_, debug=False):
        with torch.no_grad():
            base = self.random_network(in_)
        out = self.linear_head(base)
        if debug:
            info = {
                "rep": base.squeeze().detach().numpy(),
                "out": out.squeeze().detach().numpy(),
            }
        else:
            info = None
        return out, info


class UniformRandomCont(BetaPolicy):
    def __init__(self, device, observation_dim, arch, action_dim):
        super(UniformRandomCont, self).__init__(device, observation_dim, arch, action_dim, beta_param_bias=0, beta_param_bound=1e8,
                                                init='Const/1/0', activation="ReLU", head_activation="ReLU", layer_norm=False)
    def get_dist_params(self, observation):
        alpha = torch_utils.tensor([[1.0]*self.action_dim], device=self.device)
        beta = torch_utils.tensor([[1.0]*self.action_dim], device=self.device)
        return alpha, beta

class UniformRandomDisc(Softmax):
    def __init__(self, device, observation_dim, arch, num_actions):
        super(UniformRandomDisc, self).__init__(device, observation_dim, arch, num_actions,
                                                init='Const/1/0', activation="ReLU", layer_norm=False)

    def get_probs(self, state):
        x = torch.ones(state.size()[0], self.num_actions)
        probs = nn.functional.softmax(x, dim=1)
        return probs, x
