import numpy as np
import torch


class BaseNormalizer:
    def __init__(self):
        return

    def __call__(self, x):
        return x

    def denormalize(self, x):
        return x

class Identity(BaseNormalizer):
    def __init__(self):
        super(Identity, self).__init__()
        

class OneHot(BaseNormalizer):
    def __init__(self, total_count, start_from):
        super(OneHot, self).__init__()
        self.total_count = total_count
        self.start = start_from

    def __call__(self, x):
        assert len(x.shape) == 2 and x.shape[1]==1 # shape = batch_size * 1
        oneh = torch.zeros((x.shape[0], self.total_count))
        oneh[np.arange(x.shape[0]), (x - self.start).astype(int).squeeze()] = 1
        return oneh


class Scale(BaseNormalizer):
    def __init__(self, scaler, bias):
        super(Scale, self).__init__()
        self.scaler = scaler
        self.bias = bias

    def __call__(self, x):
        return (x - self.bias) / self.scaler

    def denormalize(self, x):
        return x * self.scaler + self.bias

def init_normalizer(name, info):
    if name == "Identity":
        return Identity()
    elif name == "OneHot":
        return OneHot(total_count=info.n, start_from=info.start)
    elif name == "Scale":
        return Scale(scaler=info.scaler, bias=info.bias)
    else:
        raise NotImplementedError