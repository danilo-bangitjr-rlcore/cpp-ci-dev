import torch
from src.network.torch_utils import *
from src.network.networks import BetaPolicy, BetaInvParam, SquashedGaussianPolicy, Softmax, RndLinearUncertainty
from src.network.networks import FC


def init_policy_network(name, device, state_dim, hidden_units, action_dim, beta_param_bias, beta_param_bound,
                        activation, head_activation, layer_init, layer_norm):
    hidden_units = [i for i in hidden_units if i > 0]
    if name == "Beta":
        return BetaPolicy(device, state_dim, hidden_units, action_dim, beta_param_bias=beta_param_bias, beta_param_bound=beta_param_bound,
                          activation=activation, head_activation=head_activation, init=layer_init, layer_norm=layer_norm)
    elif name == "BetaInv":
        return BetaInvParam(device, state_dim, hidden_units, action_dim,
                            activation=activation, head_activation=head_activation, init=layer_init, layer_norm=layer_norm)
    elif name == "SGaussian":
        return SquashedGaussianPolicy(device, state_dim, hidden_units, action_dim, activation=activation, init=layer_init, layer_norm=layer_norm)
    elif name == "Softmax":
        return Softmax(device, state_dim, hidden_units, action_dim,
                       activation=activation, init=layer_init, layer_norm=layer_norm)
    else:
        raise NotImplementedError
    
def init_critic_network(name, device, input_dim, hidden_units, output_dim, activation, layer_init, layer_norm):
    hidden_units = [i for i in hidden_units if i > 0]
    if name == "FC":
        return FC(device, input_dim, hidden_units, output_dim,
                  activation=activation, head_activation="None", init=layer_init, layer_norm=layer_norm)
    else:
        raise NotImplementedError

def init_custom_network(name, device, input_dim, hidden_units, output_dim, activation, head_activation, layer_init, layer_norm):
    hidden_units = [i for i in hidden_units if i > 0]
    if name == "FC":
        return FC(device, input_dim, hidden_units, output_dim,
                  activation=activation, head_activation=head_activation, init=layer_init, layer_norm=layer_norm)
    elif name == "Softmax":
        return Softmax(device, input_dim, hidden_units, output_dim, activation=activation, init=layer_init, layer_norm=layer_norm)
    elif name == "RndLinearUncertainty":
        return RndLinearUncertainty(device, input_dim, hidden_units, output_dim, activation=activation, init=layer_init, layer_norm=layer_norm)
    else:
        raise NotImplementedError

def init_optimizer(name, param, lr):
    if name == "RMSprop":
        return torch.optim.RMSprop(param, lr)
    elif name == 'Adam':
        return torch.optim.Adam(param, lr)
    elif name == "SGD":
        return torch.optim.SGD(param, lr)
    else:
        raise NotImplementedError

