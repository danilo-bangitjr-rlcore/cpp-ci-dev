import numpy as np
import torch


class BaseNormalizer:
    def __init__(self):
        return

    def __call__(self, x):
        return x

    def denormalize(self, x):
        return x

    def get_new_dim(self, d):
        return d


class Identity(BaseNormalizer):
    def __init__(self):
        super(Identity, self).__init__()


class Scale(BaseNormalizer):
    def __init__(self, args):
        super(Scale, self).__init__()
        scaler, bias = args
        self.scaler = scaler
        self.bias = bias

    def __call__(self, x):
        return (x - self.bias) / self.scaler

    def denormalize(self, x):
        return x * self.scaler + self.bias


class Clip(BaseNormalizer):
    def __init__(self, args):
        super(Clip, self).__init__()
        min_, max_ = args
        self.min_ = min_
        self.max_ = max_

    def __call__(self, x):
        return np.clip(x, self.min_, self.max_)

    def denormalize(self, x):
        raise NotImplementedError



def init_normalizer(name, *args):
    if name == "Identity":
        return Identity()
    elif name == "OneHot":
        return OneHot(*args)
    elif name == "Scale":
        return Scale(*args)
    elif name == "Clip":
        return Clip(*args)
    else:
        raise NotImplementedError