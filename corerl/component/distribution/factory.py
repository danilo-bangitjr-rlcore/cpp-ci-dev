import torch.distributions as d

from . import ArctanhNormal

_dist_types: dict[str, type[d.Distribution]] = {
    "arctanh_normal": ArctanhNormal,
    "squashed_gaussian": ArctanhNormal,
    "beta": d.Beta,
    "kumaraswamy": d.Kumaraswamy,
    "gamma": d.Gamma,
    "laplace": d.Laplace,
    "normal": d.Normal,
    "log_normal": d.LogNormal,
}


def get_dist_type(type_: str) -> type[d.Distribution]:
    if type_.lower() in _dist_types.keys():
        return _dist_types[type_.lower()]

    # Try to get the distribution from torch.distributions
    elif hasattr(d, type_):
        return getattr(d, type_)

    raise NotImplementedError(
        f"unknown policy type '{type_}', known policy types include " +
        f"{list(_dist_types.keys())}",
    )
