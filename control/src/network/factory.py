import torch
from src.network.torch_utils import *
from src.network.networks import BetaPolicy, SquashedGaussianPolicy, Softmax
from src.network.networks import FC


def init_policy_network(name, device, state_dim, hidden_units, action_dim, action_scale, action_bias, activation):
    if name == "Beta":
        return BetaPolicy(device, state_dim, hidden_units, action_dim, action_scale=action_scale, action_bias=action_bias, activation=activation)
    elif name == "SGaussian":
        return SquashedGaussianPolicy(device, state_dim, hidden_units, action_dim, action_scale=action_scale, action_bias=action_bias, activation=activation)
    elif name == "Softmax":
        return Softmax(device, state_dim, hidden_units, action_dim, activation=activation)
    else:
        raise NotImplementedError
    
def init_critic_network(name, device, input_dim, hidden_units, output_dim, activation):
    if name == "FC":
        return FC(device, input_dim, hidden_units, output_dim, activation=activation, head_activation="None")
    else:
        raise NotImplementedError

def init_optimizer(name, param, lr):
    if name == "RMSprop":
        return torch.optim.RMSprop(param, lr)
    else:
        raise NotImplementedError

