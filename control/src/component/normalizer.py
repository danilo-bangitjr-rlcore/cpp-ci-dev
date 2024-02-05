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
        

class OneHot(BaseNormalizer):
    def __init__(self, args):
        super(OneHot, self).__init__()
        total_count, start_from = args
        self.total_count = total_count
        self.start = start_from

    def __call__(self, x):
        assert len(x.shape) == 2 and x.shape[1]==1 # shape = batch_size * 1
        oneh = torch.zeros((x.shape[0], self.total_count))
        if type(x) == np.ndarray:
            oneh[np.arange(x.shape[0]), (x - self.start).astype(int).squeeze()] = 1
        elif type(x) == torch.Tensor:
            oneh[np.arange(x.shape[0]), (x - self.start).to(torch.int).squeeze()] = 1
        return oneh

    def get_new_dim(self, d):
        return self.total_count

    def denormalize(self, x):
        if type(x) == np.ndarray:
            idx = np.where(x==1)[1]
            idx = np.expand_dims(idx, axis=1)
        elif type(x) == torch.Tensor:
            idx = (x == 1).nonzero(as_tuple=False)
            idx = idx[:, 1:]
        return idx


class Scale(BaseNormalizer):
    def __init__(self, args):
        super(Scale, self).__init__()
        # # if arguments passed as float, use a constant action_scale and action_bias for all action dimensions.
        # if type(action_scale) == float:
        #     action_scale = np.ones(action_dim)*action_scale
        # else:
        #     raise NotImplementedError
        #
        # if type(action_bias) == float:
        #     action_bias = np.ones(action_dim)*action_bias
        # else:
        #     raise NotImplementedError
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
        # # if arguments passed as float, use a constant action_scale and action_bias for all action dimensions.
        # if type(action_scale) == float:
        #     action_scale = np.ones(action_dim)*action_scale
        # else:
        #     raise NotImplementedError
        #
        # if type(action_bias) == float:
        #     action_bias = np.ones(action_dim)*action_bias
        # else:
        #     raise NotImplementedError
        min_, max_ = args
        self.min_ = min_
        self.max_ = max_

    def __call__(self, x):
        return np.clip(x, self.min_, self.max_)

    def denormalize(self, x):
        raise NotImplementedError


class ThreeTanksReward(BaseNormalizer):
    def __init__(self):
        super(ThreeTanksReward, self).__init__()
        self.min_ = -800
        self.clip = -1

    def __call__(self, x):
        if x < self.clip:
            # x = self.clip - (x - self.min_) / (self.clip - self.min_)
            x = self.clip
        return x

class TTChangeActionState(Scale):
    def __init__(self):
        super(TTChangeActionState, self).__init__([10., 0.])

def init_normalizer(name, *args):
    if name == "Identity":
        return Identity()
    elif name == "OneHot":
        return OneHot(*args)
    elif name == "Scale":
        return Scale(*args)
    elif name == "Clip":
        return Clip(*args)
    elif name == "ThreeTanksReward":
        return ThreeTanksReward()
    elif name == "TTChangeActionState":
        return TTChangeActionState()
    else:
        raise NotImplementedError