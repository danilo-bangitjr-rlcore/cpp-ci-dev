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


def clone_model_0to1(net0, net1):
    with torch.no_grad():
        net1.load_state_dict(net0.state_dict())
    return net1

def clone_gradient(model):
    grad_rec = {}
    for idx, param in enumerate(model.parameters()):
        grad_rec[idx] = param.grad
    return grad_rec

def move_gradient_to_network(model, grad_rec, weight):
    for idx, param in enumerate(model.parameters()):
        param.grad = grad_rec[idx] * weight
    return model

def layer_init_normal(layer, bias=True):
    nn.init.normal_(layer.weight)
    if int(bias):
        nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_zero(layer, bias=True):
    nn.init.constant_(layer.weight, 0)
    if int(bias):
        nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_constant(layer, const, bias=True):
    nn.init.constant_(layer.weight, float(const))
    if int(bias):
        nn.init.constant_(layer.bias.data, float(const))
    return layer


def layer_init_xavier(layer, bias=True):
    nn.init.xavier_uniform_(layer.weight)
    if int(bias):
        nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_uniform(layer, low=-0.003, high=0.003, bias=0):
    nn.init.uniform_(layer.weight, low, high)
    if int(bias):
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

def init_activation_function(name):
    if name == "ReLU":
        return torch.nn.functional.relu
    elif name == "Softplus":
        return torch.nn.functional.softplus
    elif name == "ReLU6":
        return torch.nn.functional.relu6
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

