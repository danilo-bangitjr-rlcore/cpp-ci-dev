import pytest
import torch
import torch.distributions as d
from corerl.component.distribution.arctanhnormal import ArctanhNormal


# ----------
# -- Ctor --
# ----------
def test_ctor():
    """Test that ctor matches behaviour of other Torch distribution ctors
    """
    loc = torch.Tensor([0, 0])
    scale = torch.Tensor([1, 1])
    dist = ArctanhNormal(loc, scale)

    assert torch.all(dist.loc == loc)
    assert torch.all(dist.scale == scale)

    # Ensure batch/event shapes match default torch behaviour
    assert dist.batch_shape == d.Normal(loc, scale).batch_shape
    assert dist.event_shape == d.Normal(loc, scale).event_shape

    # Ensure batch/event shapes match default torch behaviour
    loc = torch.Tensor([[0, 0]])
    scale = torch.Tensor([[1, 1]])
    dist = ArctanhNormal(loc, scale)
    assert dist.batch_shape == d.Normal(loc, scale).batch_shape
    assert dist.event_shape == d.Normal(loc, scale).event_shape


# -------------
# -- rsample --
# -------------
def test_grad_sample():
    """Test that gradients are propagated properly when sampling
    """
    loc = torch.Tensor([0, 0])
    scale = torch.Tensor([1, 1])

    # Turn gradient recording off
    loc.requires_grad_(False)
    scale.requires_grad_(False)

    # Ensure gradients are not propagated when required_grad is False
    dist = ArctanhNormal(loc, scale)
    assert not dist.rsample(sample_shape=(3, 2)).requires_grad
    assert not dist.rsample(sample_shape=(0, 2)).requires_grad

    # Turn gradient recording on
    loc.requires_grad_(True)
    scale.requires_grad_(True)

    # Ensure gradients not propagated when required_grad is True
    dist = ArctanhNormal(loc, scale)
    assert dist.rsample(sample_shape=(3, 2)).requires_grad
    assert dist.rsample(sample_shape=(0, 2)).requires_grad

    assert not dist.sample(sample_shape=(3, 2)).requires_grad
    assert not dist.sample(sample_shape=(0, 2)).requires_grad


def test_sample_shape():
    """Test that samples are shaped consistently with other torch distributions
    """
    params = torch.Tensor([1, 1])

    dist = ArctanhNormal(params, params)

    for sh in ((3, 2), (0, 2), (2, 0)):
        got = dist.sample(sample_shape=sh).shape

        expected = d.Normal(params, params).sample(sample_shape=sh).shape
        assert got == expected

        expected = d.Beta(params, params).sample(sample_shape=sh).shape
        assert got == expected


# -------------
# -- LogProb --
# -------------
def test_logprob():
    """Test that the log density is calculated properly
    """
    loc = torch.Tensor([0, 0])
    scale = torch.Tensor([1, 1])

    dist = ArctanhNormal(loc, scale)
    norm = d.Normal(loc, scale)

    y = torch.Tensor([0, 0])
    lp_got = dist.log_prob(y)

    x = torch.atanh(y)
    lp_expected = norm.log_prob(x)
    offset = torch.log1p(
        -y.pow(2) + ArctanhNormal._EPSILON
    )
    lp_expected -= offset

    assert torch.all(lp_got == lp_expected)

# -------------
# ---- CDF ----
# -------------
def test_cdf():
    """Test that the cdf is calculated properly
    """
    loc = torch.Tensor([0, 0])
    scale = torch.Tensor([1, 1])

    dist = ArctanhNormal(loc, scale)
    norm = d.Normal(loc, scale)

    y = torch.Tensor([0, 0])
    cdf_got = dist.cdf(y)

    x = torch.atanh(y)
    cdf_expected = norm.cdf(x)

    assert torch.all(cdf_got == cdf_expected)

# -------------
# --- ICDF ----
# -------------
def test_icdf():
    """Test that the cdf is calculated properly
    """
    loc = torch.Tensor([0, 0])
    scale = torch.Tensor([1, 1])

    dist = ArctanhNormal(loc, scale)
    norm = d.Normal(loc, scale)

    y = torch.Tensor([0, 0])
    icdf_got = dist.icdf(y)

    x = torch.atanh(y)
    icdf_expected = norm.icdf(x)

    assert torch.all(icdf_got == icdf_expected)
