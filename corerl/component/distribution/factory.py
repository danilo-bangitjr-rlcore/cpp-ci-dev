import torch.distributions as d
from . import ArctanhNormal


def get_dist_type(type_):
    if type_.lower() in ("arctanhnormal", "squashed_gaussian"):
        return ArctanhNormal
    elif type_.lower() == "beta":
        return d.Beta
    elif type_.lower() == "logitnormal":
        return d.LogitNormal
    elif type_.lower() == "gamma":
        return d.Gamma
    elif type_.lower() == "laplace":
        return d.Laplace
    elif type_.lower() == "normal":
        return d.Normal
    elif type_.lower() == "kumaraswamy":
        return d.Kumaraswamy
    elif type_.lower() == "lognormal":
        return d.LogNormal
    else:
        try:
            getattr(d, type_)
        except AttributeError:
            raise NotImplementedError(f"unknown policy type {type_}")
