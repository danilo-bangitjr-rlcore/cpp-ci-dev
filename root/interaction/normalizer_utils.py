import numpy as np
import gymnasium

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


def init_action_normalizer(cfg, env: gymnasium.Env) -> BaseNormalizer:
    if cfg.discrete_control:
        return Identity()

    else:  # continuous control
        name = cfg.name
        action_low = env.action_space.low
        action_high = env.action_space.high
        if name == "identity":
            return Identity()
        elif name == "scale":
            if cfg.action_low:  # use high and low specified in the config
                assert cfg.action_high, "Please specify cfg.action_normalizer.action_high in config file"
                action_low, action_high = float(cfg.action_low), float(cfg.action_high)
                action_low = np.array(action_low)
                action_high = np.array(action_high)
            else:  # use information from the environment
                action_high = env_high
                action_low = env_low
            scaler = (action_high - action_low)
            bias = action_high - action_low
            return Scale([scaler, bias])
        elif name == "clip":
            return Clip([cfg.clip_min, cfg.clip_max])
        else:
            raise NotImplementedError


def init_reward_normalizer(cfg) -> BaseNormalizer:
    print(cfg)
    name = cfg.name
    if name == "identity":
        return Identity()
    elif name == "scale":
        assert cfg.reward_high, "Please specify cfg.reward_normalizer.reward_high in config file"
        reward_high = float(cfg.reward_high)
        reward_low = float(cfg.reward_low)
        scaler = reward_high - reward_low
        bias = reward_low
        return Scale([scaler, bias])
    elif name == "clip":
        return Clip([cfg.clip_min, cfg.clip_max])
    else:
        raise NotImplementedError
