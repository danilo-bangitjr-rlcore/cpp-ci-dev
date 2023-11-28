import torch
import src.network.torch_utils as internal_factory
import torch.nn as nn
import torch.distributions as distrib
import numpy as np
FLOAT32_EPS = 10 * \
              np.finfo(np.float32).eps # differences of this size are
                                       # representable up to ~ 15
EPSILON = 1e-6



class FC(nn.Module):
    def __init__(self, device, input_dim, arch, output_dim, activation="ReLU", head_activation="None", init='Xavier'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layer_init = internal_factory.init_layer(init)
        activation_cls = internal_factory.init_activation(activation)
        d = input_dim
        modules = []
        for hidden_size in arch:
            fc = layer_init(nn.Linear(d, hidden_size))
            modules.append(fc)
            modules.append(activation_cls())
            d = hidden_size
        last_fc = layer_init(nn.Linear(d, output_dim))
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)
        self.head_act = internal_factory.init_activation(head_activation)()
        self.to(device)

    def forward(self, input_tensor):
        out = self.network(input_tensor)
        out = self.head_act(out)
        return out


class SquashedGaussianPolicy(nn.Module):
    def __init__(self, device, observation_dim, arch, action_dim,
                 action_scale=1, action_bias=0, init='Xavier', activation="ReLU"):#, action_clip=[-0.999, 0.999]):
        super(SquashedGaussianPolicy, self).__init__()
        layer_init = internal_factory.init_layer(init)
        self.base_network = FC(device, observation_dim, arch[:-1], arch[-1], activation=activation, head_activation=activation, init=init)
        self.mean_head = layer_init(nn.Linear(arch[-1], action_dim))
        self.logstd_head = layer_init(nn.Linear(arch[-1], action_dim))

        self.action_scale = torch.tensor(action_scale)
        self.action_bias = torch.tensor(action_bias)
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
        action = tanhout * self.action_scale + self.action_bias
        # action = torch.clamp(action, min=self.action_clip[0], max=self.action_clip[1])
        logp = normal.log_prob(out)
        logp -= torch.log(self.action_scale * (1 - tanhout.pow(2)) + EPSILON).sum(axis=-1).reshape(logp.shape)

        if debug:
            info = {
                # "distribution": normal,
                "param1": mean.squeeze().detach().numpy(),
                "param2": std.squeeze().detach().numpy(),
            }
        else:
            info = normal
        return action, logp, info

    def log_prob(self, observation, action, debug=False):
        base = self.base_network(observation)
        mean = self.mean_head(base)
        log_std = torch.clamp(self.logstd_head(base), min=-20, max=2)
        std = log_std.exp()
    
        normal = distrib.Normal(mean, std)
        normal = distrib.Independent(normal, 1)
        
        tanhout = (action - self.action_bias) / self.action_scale
        out = torch.atanh(torch.clamp(tanhout, -1.0 + EPSILON, 1.0 - EPSILON))
        logp = normal.log_prob(out)
        logp -= torch.log(self.action_scale * (1 - tanhout.pow(2)) + EPSILON).sum(axis=-1).reshape(logp.shape)

        if debug:
            info = {
                # "distribution": normal,
                "param1": mean.squeeze().detach().numpy(),
                "param2": std.squeeze().detach().numpy()
            }
        else:
            info = normal
        return logp, info


class BetaPolicy(nn.Module):
    def __init__(self, device, observation_dim, arch, action_dim,
                 action_scale=1, action_bias=0, init='Xavier', activation="ReLU"):
        super(BetaPolicy, self).__init__()
        layer_init = internal_factory.init_layer(init)
        self.base_network = FC(device, observation_dim, arch[:-1], arch[-1], activation=activation, head_activation=activation, init=init)
        self.alpha_head = layer_init(nn.Linear(arch[-1], action_dim))
        self.beta_head = layer_init(nn.Linear(arch[-1], action_dim))

        self.action_scale = torch.tensor(action_scale)
        self.action_bias = torch.tensor(action_bias)
        self.to(device)

    def forward(self, observation, debug=False):
        base = self.base_network(observation)
        alpha = nn.functional.softplus(self.alpha_head(base)) + EPSILON
        beta = nn.functional.softplus(self.beta_head(base)) + EPSILON
        
        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)
        out = dist.rsample()
        logp = dist.log_prob(torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS)) - np.log(self.action_scale)

        action = out * self.action_scale + self.action_bias
        # action = torch.clamp(action, min=self.action_clip[0], max=self.action_clip[1])
        if debug:
            info = {
                # "distribution": dist,
                "param1": alpha.squeeze().detach().numpy(),
                "param2": beta.squeeze().detach().numpy()
            }
        else:
            info = dist
        return action, logp, info
        
    def log_prob(self, observation, action, debug=False):
        out = (action - self.action_bias) / self.action_scale
        out = torch.clamp(out, 0, 1)
    
        base = self.base_network(observation)
        alpha = nn.functional.softplus(self.alpha_head(base)) + EPSILON
        beta = nn.functional.softplus(self.beta_head(base)) + EPSILON
    
        dist = distrib.Beta(alpha, beta)
        dist = distrib.Independent(dist, 1)

        logp = dist.log_prob(torch.clamp(out, 0 + FLOAT32_EPS, 1 - FLOAT32_EPS)) - np.log(self.action_scale)

        if debug:
            info = {
                # "distribution": dist,
                "param1": alpha.squeeze().detach().numpy(),
                "param2": beta.squeeze().detach().numpy()
            }
        else:
            info = dist
        return logp, info


class Softmax(nn.Module):
    def __init__(self, device, observation_dim, arch, num_actions, init='Xavier', activation="ReLU"):
        super(Softmax, self).__init__()
        self.num_actions = num_actions
        self.base_network = FC(device, observation_dim, arch, num_actions, init=init, activation=activation)
        self.to(device)
        
    def forward(self, state, debug=False):
        x = self.base_network(state)
        probs = nn.functional.softmax(x, dim=1)
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
            info = dist
        return actions.reshape((-1, 1)), log_prob, info

    def log_prob(self, states, actions, debug=False):
        x = self.base_network(states)

        # log_prob = nn.functional.log_softmax(x, dim=1)
        # log_prob = torch.gather(log_prob, dim=1, index=actions.long())

        probs = nn.functional.softmax(x, dim=1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions.squeeze(-1))

        if debug:
            info = {
                # "distribution": dist,
                "param1": probs.squeeze().detach().numpy(),
            }
        else:
            info = dist
        return log_prob, info
