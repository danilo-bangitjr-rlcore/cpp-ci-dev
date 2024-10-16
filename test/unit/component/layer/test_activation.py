import torch
from corerl.component.layer.activations import FTA as FTA


# ----------
# -- FTA --
# ----------
def test_fta1():
    """
    Test scalar case
    """
    eta = 0.1
    lower = 0
    upper = 1
    delta = 0.25
    fta = FTA(eta, lower, upper, delta)

    # Test with a scalar torch.Tensor type where ndim == 0
    input_ = torch.tensor(0.52)

    got = fta(input_)
    expected = torch.Tensor([0.0, 0.98, 1.0, 0.0])

    assert torch.all(got == expected)
    assert got.shape == expected.shape

    # Test with a "scalar" torch.Tensor type where ndim == 1
    input_ = torch.Tensor([0.52])

    got = fta(input_)
    expected = torch.Tensor([[0.0, 0.98, 1.0, 0.0]])

    assert torch.all(got == expected)
    assert got.shape == expected.shape


def test_fta2():
    """
    Test batch case
    """
    eta = 0.1
    lower = 0
    upper = 1
    delta = 0.25
    fta = FTA(eta, lower, upper, delta)

    input_ = torch.Tensor([0.0, 0.1, 0.52, 0.72, 1.0])

    got = fta(input_)
    expected = torch.Tensor([
        [1.0, 0.0,  0.0, 0.0 ],
        [1.0, 0.0,  0.0, 0.0 ],
        [0.0, 0.98, 1.0, 0.0 ],
        [0.0, 0.0,  1.0, 0.97],
        [0.0, 0.0,  0.0, 1.0 ],
    ])

    assert torch.all(got == expected)
    assert got.shape == expected.shape


def test_fta3():
    """
    η = 0 induces a one-hot encoding
    """
    eta = 0.0
    lower = 0
    upper = 1
    delta = 0.25
    fta = FTA(eta, lower, upper, delta)

    input_ = torch.Tensor([0.52])

    got = fta(input_)
    expected = torch.Tensor([0.0, 0.0, 1.0, 0.0])

    assert torch.all(got == expected)
