import torch
import torch.nn as nn


class Float(torch.nn.Module):
    def __init__(self, device, init_value):
        super().__init__()
        self.constant = torch.nn.Parameter(torch.tensor(init_value, dtype=torch.float32).to(device))

    def forward(self):
        return self.constant


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

def init_fn_factory(init):
    if init == 'xavier':
        layer_init = layer_init_xavier
    elif init == 'const':
        layer_init = layer_init_constant
    elif init == 'zero':
        layer_init = layer_init_zero
    else:
        raise NotImplementedError
    return layer_init

def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x

def to_np(t):
    return t.cpu().detach().numpy()
