import copy

import torch
import torch.nn as nn
import math
import collections

from torch.optim.optimizer import StateDict


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

        self.state_idx = {}
        max_p_len = 0
        for gi, group in enumerate(self.param_groups):
            if len(group['params']) > max_p_len:
                max_p_len = len(group['params'])
        self.max_p_len = max_p_len
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group['params']):
                self.state_idx[gi * self.max_p_len + pi] = {}

    def __setstate__(self, state):
        super(CustomADAM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, network_param_groups, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # for gi, group in enumerate(network_param_groups):
        for gi, group in enumerate(self.param_groups):
            # for pi, p in enumerate(group['params']):
            for pi, p in enumerate(network_param_groups[gi]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                # state = self.state[p]
                state = self.state_idx[gi * self.max_p_len + pi]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                step = state['step']
                lr = group['lr']
                # if pi == 5 and len(self.state_idx[5].keys()) > 0:
                #     print("step", self.state_idx[5]['exp_avg'], self.state_idx[5]['exp_avg_sq'])

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                bias_correction2_sqrt = self._dispatch_sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # if pi == 5 and len(self.state_idx[5].keys()) > 0:
                #     print("updated", self.state_idx[5]['exp_avg'], self.state_idx[5]['exp_avg_sq'])

        return loss

    def _dispatch_sqrt(self, x: float):  # float annotation is needed because of torchscript type inference
        if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
            return x.sqrt()
        else:
            return math.sqrt(x)

    def state_dict(self):
        hard_copy = {
            'state': {},
            'param_groups': []
        }
        with torch.no_grad():
            for s in self.state_idx:
                # if s == 5 and len(self.state_idx[5].keys()) > 0:
                #     print("saving state", self.state_idx[5]['exp_avg'], self.state_idx[5]['exp_avg_sq'])

                hard_copy['state'][s] = {}
                if len(self.state_idx[s].keys()) > 0:
                    hard_copy['state'][s]['step'] = copy.deepcopy(self.state_idx[s]['step'])
                    hard_copy['state'][s]['exp_avg'] = copy.deepcopy(self.state_idx[s]['exp_avg'].data.detach())
                    hard_copy['state'][s]['exp_avg_sq'] = copy.deepcopy(self.state_idx[s]['exp_avg_sq'].data.detach())
            #     if len(self.state[st].keys()) > 0:
            #         hard_copy['state'][s]['step'] = copy.deepcopy(self.state[st]['step'])
            #         print("state_dict", self.state[st].keys(), hard_copy['state'][s]['step'])
            #
            #         hard_copy['state'][s]['exp_avg'] = copy.deepcopy(self.state[st]['exp_avg'].data.detach())
            #         hard_copy['state'][s]['exp_avg_sq'] = copy.deepcopy(self.state[st]['exp_avg_sq'].data.detach())
            for gi, group in enumerate(self.param_groups):
            # for group in self.param_groups:
                hard_copy['param_groups'].append({})
                
                # hard_copy['param_groups'][-1]['params'] = []
                # for ele in group['params']:
                #     hard_copy['param_groups'][-1]['params'].append(copy.deepcopy(ele.data.detach()))

                hard_copy['param_groups'][-1]['lr'] = copy.deepcopy(group['lr'])
                hard_copy['param_groups'][-1]['betas'] = copy.deepcopy(group['betas'])
                hard_copy['param_groups'][-1]['eps'] = copy.deepcopy(group['eps'])
                hard_copy['param_groups'][-1]['weight_decay'] = copy.deepcopy(group['weight_decay'])
                hard_copy['param_groups'][-1]['amsgrad'] = copy.deepcopy(group['amsgrad'])
        return hard_copy

    def load_state_dict(self, state_dict):
        def inposition_fill(ref, data):
            if len(ref.data.size()) == 2:
                idx = torch.arange(ref.data.size()[1])
                idx = torch.tile(idx, (ref.data.size()[0], 1))
                ref.data.scatter_(1, idx, data)
            elif len(ref.data.size()) == 1:
                idx = torch.arange(ref.data.size()[0])
                ref.data.scatter_(0, idx, data)
            else:
                ref.data.fill_(data)

        with torch.no_grad():
            for g in range(len(state_dict['param_groups'])):
                self.param_groups[g]['lr'] = state_dict['param_groups'][g]['lr']
                self.param_groups[g]['betas'] = state_dict['param_groups'][g]['betas']
                self.param_groups[g]['eps'] = state_dict['param_groups'][g]['eps']
                self.param_groups[g]['weight_decay'] = state_dict['param_groups'][g]['weight_decay']
                self.param_groups[g]['amsgrad'] = state_dict['param_groups'][g]['amsgrad']

                # for i in range(len(state_dict['param_groups'][g]['params'])):
                #     inposition_fill(self.param_groups[g]['params'][i], state_dict['param_groups'][g]['params'][i])

            s_keys = self.state_idx.keys()
            # s_keys = list(self.state.keys())
            for s in s_keys:
                if len(state_dict['state'][s].keys()) > 0:
                    self.state_idx[s]['step'] = state_dict['state'][s]['step']
                    inposition_fill(self.state_idx[s]['exp_avg'], state_dict['state'][s]['exp_avg'])
                    inposition_fill(self.state_idx[s]['exp_avg_sq'], state_dict['state'][s]['exp_avg_sq'])

                    # if s==5 and len(state_dict['state'][5].keys()) > 0:
                    #     print("loaded in loop", self.state_idx[5]['step'], state_dict['state'][5]['step'])
                    #     print("loaded in loop", self.state_idx[5]['exp_avg'], state_dict['state'][5]['exp_avg'])
                    #     print("loaded in loop", self.state_idx[5]['exp_avg_sq'], state_dict['state'][5]['exp_avg_sq'])

                # if state_dict['state'].get(s) is None:
                #     del self.state[s]
                # elif state_dict['state'].get(s) is not None and (len(state_dict['state'][s].keys()) > 0):
                #     print("state_dict['state'][s].keys()", state_dict['state'][s].keys())
                #     print("self.state[s]['step']", self.state[s]['step'])
                #     self.state[s]['step'] = state_dict['state'][s]['step']
                #     inposition_fill(self.state[s]['exp_avg'], state_dict['state'][s]['exp_avg'])
                #     inposition_fill(self.state[s]['exp_avg_sq'], state_dict['state'][s]['exp_avg_sq'])
                else:
                    ssk = list(self.state_idx[s].keys())
                    for k in ssk:
                        del self.state_idx[s][k]
            # if len(state_dict['state'][5].keys()) > 0:
            #     print("loaded", self.state_idx[5]['exp_avg'], self.state_idx[5]['exp_avg_sq'])
