import torch

from corerl.component.layer.activations import FTA as FTA
from corerl.component.layer.activations import ActivationConfig
from corerl.component.layer.activations import init_activation as init_activation


# ----------
# -- FTA --
# ----------
def test_fta1():
    """Test scalar case
    """
    eta = 0.1
    lower = 0
    upper = 1
    bins = 4
    fta = FTA(eta, lower, upper, bins)

    # Test with a scalar torch.Tensor type where ndim == 0
    input_ = torch.tensor(0.52)

    got = fta(input_)
    expected = torch.Tensor([0.0, 0.98, 1.0, 0.0])

    assert torch.all(got == expected)
    assert got.shape == expected.shape

    # Test with a "scalar" torch.Tensor type where ndim == 1
    input_ = torch.Tensor([0.52])

    got = fta(input_)
    expected = torch.Tensor([0.0, 0.98, 1.0, 0.0])

    assert torch.all(got == expected)
    assert got.shape == expected.shape


def test_fta2():
    """Test batch case
    """
    eta = 0.1
    lower = 0
    upper = 1
    bins = 4
    fta = FTA(eta, lower, upper, bins)

    input_ = torch.Tensor([0.0, 0.1, 0.52, 0.72, 1.0])

    got = fta(input_)
    expected = torch.Tensor([
        1.0, 0.0,  0.0, 0.0,
        1.0, 0.0,  0.0, 0.0,
        0.0, 0.98, 1.0, 0.0,
        0.0, 0.0,  1.0, 0.97,
        0.0, 0.0,  0.0, 1.0,
    ])

    assert torch.all(got == expected)
    assert got.shape == expected.shape


def test_fta3():
    """η = 0 induces a one-hot encoding
    """
    eta = 0.0
    lower = 0
    upper = 1
    bins = 4
    fta = FTA(eta, lower, upper, bins)

    input_ = torch.Tensor([0.52])

    got = fta(input_)
    expected = torch.Tensor([0.0, 0.0, 1.0, 0.0])

    assert torch.all(got == expected)
    assert got.shape == expected.shape


def test_fta_factory():
    """FTA can be constructed by factory
    """

    # kwargs only
    cfg: ActivationConfig = {
        "name": "FTA",
        "kwargs": {
            "eta": 0.1, "lower": 0, "upper": 1, "n_bins": 4,
        }
    }
    act = init_activation(cfg)
    assert isinstance(act, FTA)

    # args only
    cfg = {"name": "FTA", "args": (0.1, 0, 1, 5)}
    act = init_activation(cfg)
    assert isinstance(act, FTA)

    # Mix of kwargs and args
    cfg = {"name": "FTA", "args": (0.1, 0, 1), "kwargs": {"n_bins": 3}}
    act = init_activation(cfg)
    assert isinstance(act, FTA)
