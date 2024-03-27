import torch
import torch.nn as nn
import numpy as np
import utils

FLOAT32_EPS = 10 * \
              np.finfo(np.float32).eps  # differences of this size are
# representable up to ~ 15
EPSILON = 1e-6


# TODO: here is an example of initializing a network.
class FC(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        device = cfg.device
        input_dim = cfg.input_dim
        output_dim = cfg.output_dim
        layer_norm = cfg.layer_norm
        arch = cfg.arch
        init_args = cfg.init_args
        activation = cfg.activation
        head_activation = cfg.head_activation
        layer_init = utils.init_layer(init_args[0])
        activation_cls = utils.init_activation(activation)

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
        self.head_act = utils.init_activation(head_activation)()
        self.to(device)

    def forward(self, input_tensor):
        out = self.network(input_tensor)
        out = self.head_act(out)
        return out


class EnsembleCritic(nn.Module):
    def __init__(self, cfg):
        super(EnsembleCritic, self).__init__()

        device = cfg.device
        self.ensemble = cfg.ensemble
        self.subnetworks = [
            FC(cfg) for _ in range(self.ensemblee)]
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


