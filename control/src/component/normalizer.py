import numpy as np
import torch


class BaseNormalizer:
    def __init__(self):
        return

    def __call__(self, x):
        return x


class Identity(BaseNormalizer):
    def __init__(self):
        super(Identity, self).__init__()
        

class OneHot(BaseNormalizer):
    def __init__(self, total_count):
        super(OneHot, self).__init__()
        self.total_count = total_count

    def __call__(self, x):
        assert len(x.shape) == 2

        oneh = torch.zeros((x.shape[0], self.total_count))
        oneh[np.arange(x.shape[0]), x.int()] = 1
        return oneh


def init_normalizer(name, info):
    if name == "Identity":
        return Identity()
    elif name == "OneHot":
        return OneHot(info)
    else:
        raise NotImplementedError