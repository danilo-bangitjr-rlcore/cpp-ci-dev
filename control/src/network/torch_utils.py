import torch
import torch.nn as nn


class Float(torch.nn.Module):
    def __init__(self, device, init_value):
        super().__init__()
        self.constant = torch.nn.Parameter(torch.tensor(init_value, dtype=torch.float32).to(device))

    def forward(self):
        return self.constant

class NoneActivation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


def layer_init_normal(layer, bias=True):
    nn.init.normal_(layer.weight)
    if bias:
        nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_zero(layer, bias=True):
    nn.init.constant_(layer.weight, 0)
    if bias:
        nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_constant(layer, const, bias=True):
    nn.init.constant_(layer.weight, const)
    if bias:
        nn.init.constant_(layer.bias.data, const)
    return layer


def layer_init_xavier(layer, bias=True):
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_uniform(layer, low=-0.003, high=0.003, bias=0):
    nn.init.uniform_(layer.weight, low, high)
    if not (type(bias)==bool and bias==False):
        nn.init.constant_(layer.bias.data, bias)
    return layer

def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x

def to_np(t):
    return t.cpu().detach().numpy()

def init_activation(name):
    if name == "ReLU":
        return torch.nn.ReLU
    elif name == "Softplus":
        return torch.nn.Softplus
    elif name == "ReLU6":
        return torch.nn.ReLU6
    elif name == "None":
        return NoneActivation
    else:
        raise NotImplementedError

def init_layer(init):
    if init == 'Xavier':
        layer_init = layer_init_xavier
    elif init == 'Const':
        layer_init = layer_init_constant
    elif init == 'Zero':
        layer_init = layer_init_zero
    elif init == 'Normal':
        layer_init = layer_init_normal
    else:
        raise NotImplementedError
    return layer_init

