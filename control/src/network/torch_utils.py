import torch
import torch.nn as nn
import math


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

def reset_weight_random(old_net, new_net, param):
    return new_net

def reset_weight_shift(old_net, new_net, param):
    with torch.no_grad():
        for p, p_new in zip(old_net.parameters(), new_net.parameters()):
            p_new.data.mul_(0)
            p_new.data.add_(p.data + param)
    return new_net

def reset_weight_shrink(old_net, new_net, param):
    with torch.no_grad():
        for p, p_new in zip(old_net.parameters(), new_net.parameters()):
            p_new.data.mul_(0)
            p_new.data.add_(p.data * param)
    return new_net

def reset_weight_shrink_rnd(old_net, new_net, param):
    with torch.no_grad():
        for p, p_new in zip(old_net.parameters(), new_net.parameters()):
            p_new.data.mul_(0.5)
            p_new.data.add_(p.data * param * 0.5)
    return new_net

def reset_weight_pass(old_net, new_net, param):
    return old_net

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
        if grad_rec[idx] is not None:
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


class EnsembleOptimizer:
    def __init__(self, individual_optim, param, lr, kwargs):
        self.optim = [
            individual_optim(list(p), lr, **kwargs) for p in param
        ]

    def zero_grad(self):
        for opt in self.optim:
            opt.zero_grad()
        return

    def step(self):
        for opt in self.optim:
            opt.step()
        return

    def state_dict(self):
        return [opt.state_dict() for opt in self.optim]

    def load_state_dict(self, state_dict_lst):
        for opt, sd in zip(self.optim, state_dict_lst):
            opt.load_state_dict(sd)
        return


class CustomADAM(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=False)
        super(CustomADAM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomADAM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                exp_avg_sq.mul_(beta2).add_((1 - beta2) * torch.square(grad))
                m_hat = exp_avg / (1 - beta1 ** state['step'])
                v_hat = exp_avg_sq / (1 - beta2 ** state['step'])
                lr = group['lr'] / (v_hat.sqrt().add_(group['eps']))
                p.data.add_(-lr.mul_(m_hat))
        return loss