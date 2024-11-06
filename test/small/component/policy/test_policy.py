import pytest
import torch
import torch.nn as nn
import torch.distributions as d
from corerl.component.policy.policy import Bounded, HalfBounded, UnBounded
from corerl.component.layer.parallel import Parallel


# ------------
# --- Ctor ---
# ------------
def test_bounded_ctor():
    model = nn.Linear(1, 1)

    # Test that Bounded policies cannot be constructed with unbounded-support
    # distributions
    with pytest.raises(ValueError):
        dist = d.Laplace
        Bounded(model, dist)

    # Test that Bounded policies cannot be constructed with half
    # bounded-support distributions
    with pytest.raises(ValueError):
        dist = d.Gamma
        Bounded(model, dist)

    dist = d.Beta
    Bounded(model, dist)


def test_unbounded_ctor():
    model = nn.Linear(1, 1)

    # Test that UnBounded policies cannot be constructed with bounded-support
    # distributions
    with pytest.raises(ValueError):
        dist = d.Beta
        UnBounded(model, dist)

    # Test that UnBounded policies cannot be constructed with half
    # bounded-support distributions
    with pytest.raises(ValueError):
        dist = d.Gamma
        UnBounded(model, dist)

    dist = d.Laplace
    UnBounded(model, dist)


def test_halfbounded_ctor():
    model = nn.Linear(1, 1)

    # Test that HalfBounded policies cannot be constructed with
    # unbounded-support distributions
    with pytest.raises(ValueError):
        dist = d.Laplace
        HalfBounded(model, dist)

    # Test that HalfBounded policies cannot be constructed with bounded-support
    # distributions
    with pytest.raises(ValueError):
        dist = d.Beta
        HalfBounded(model, dist)

    dist = d.Gamma
    HalfBounded(model, dist)
    HalfBounded(model, dist, action_min=torch.Tensor([1, 1]))


# ---------------
# --- LogProb ---
# ---------------
def test_bounded_logprob_beta():
    # Construct model
    model = nn.Linear(1, 1)
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    model = Parallel(model, model)

    # Construct policy
    dist = d.Beta
    policy = Bounded(model, dist)

    # Construct input to the policy's log-density function and get the
    # log-density
    input_ = torch.zeros((1,))
    lp_got = policy.log_prob(input_, input_)[0]

    # Construct distribution with same parameters as the policy has and get the
    # log density
    dist = dist(1, 1)
    lp_expected = dist.log_prob(input_)

    assert torch.isclose(lp_got, lp_expected)


def test_unbounded_logprob_normal():
    # Construct model
    model = nn.Linear(1, 1)
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    model = Parallel(model, model)

    # Construct policy
    dist = d.Normal
    policy = UnBounded(model, dist)

    # Construct input to the policy's log-density function and get the
    # log-density
    input_ = torch.zeros((1,))
    lp_got = policy.log_prob(input_, input_)[0]

    # Construct distribution with same parameters as the policy has and get the
    # log density
    dist = dist(1, 1)
    lp_expected = dist.log_prob(input_)

    assert torch.isclose(lp_got, lp_expected)


def test_unbounded_logprob_laplace():
    # Construct model
    model = nn.Linear(1, 1)
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    model = Parallel(model, model)

    # Construct policy
    dist = d.Laplace
    policy = UnBounded(model, dist)

    # Construct input to the policy's log-density function and get the
    # log-density
    input_ = torch.zeros((1,))
    lp_got = policy.log_prob(input_, input_)[0]

    # Construct distribution with same parameters as the policy has and get the
    # log density
    dist = dist(1, 1)
    lp_expected = dist.log_prob(input_)

    assert torch.isclose(lp_got, lp_expected)


def test_halfbounded_logprob_gamma():
    # Construct model
    model = nn.Linear(1, 1)
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    model = Parallel(model, model)

    # Construct policy
    dist = d.Gamma
    policy = HalfBounded(model, dist)

    # Construct input to the policy's log-density function and get the
    # log-density
    input_ = torch.zeros((1,))
    lp_got = policy.log_prob(input_, input_)[0]

    # Construct distribution with same parameters as the policy has and get the
    # log density
    dist = dist(1, 1)
    lp_expected = dist.log_prob(input_)

    assert torch.isclose(lp_got, lp_expected)


# ----------------
# --- Sampling ---
# ----------------
def test_bounded_sample_beta():
    # Construct model
    model = nn.Linear(1, 1)
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    model = Parallel(model, model)

    # Construct policy
    dist = d.Beta
    policy = Bounded(model, dist)

    # Sample from the policy
    input_ = torch.zeros((1,))
    torch.manual_seed(1)
    samples_got = policy.forward(input_)[0]

    # Samples should have been reparameterized
    assert samples_got.requires_grad

    # Sample from a distribution which has the same parameters as the policy in
    # state `input_`
    torch.manual_seed(1)
    dist = dist(1, 1)
    samples_expected = dist.sample()

    # Ensure samples are close
    assert torch.isclose(samples_got, samples_expected)

    # Samples should **not** have been reparameterized
    samples_got = policy.forward(input_, rsample=False)[0]
    assert not samples_got.requires_grad


def test_unbounded_sample_normal():
    # Construct model
    model = nn.Linear(1, 1)
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    model = Parallel(model, model)

    # Construct policy
    dist = d.Normal
    policy = UnBounded(model, dist)

    # Sample from the policy
    input_ = torch.zeros((1,))
    torch.manual_seed(1)
    samples_got = policy.forward(input_)[0]

    # Samples should have been reparameterized
    assert samples_got.requires_grad

    # Sample from a distribution which has the same parameters as the policy in
    # state `input_`
    torch.manual_seed(1)
    dist = dist(1, 1)
    samples_expected = dist.sample()

    # Ensure samples are close
    assert torch.isclose(samples_got, samples_expected)

    # Samples should **not** have been reparameterized
    samples_got = policy.forward(input_, rsample=False)[0]
    assert not samples_got.requires_grad


def test_unbounded_sample_laplace():
    # Construct model
    model = nn.Linear(1, 1)
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    model = Parallel(model, model)

    # Construct policy
    dist = d.Laplace
    policy = UnBounded(model, dist)

    # Sample from the policy
    input_ = torch.zeros((1,))
    torch.manual_seed(1)
    samples_got = policy.forward(input_)[0]

    # Samples should have been reparameterized
    assert samples_got.requires_grad

    # Sample from a distribution which has the same parameters as the policy in
    # state `input_`
    torch.manual_seed(1)
    dist = dist(1, 1)
    samples_expected = dist.sample()

    # Ensure samples are close
    assert torch.isclose(samples_got, samples_expected)

    # Samples should **not** have been reparameterized
    samples_got = policy.forward(input_, rsample=False)[0]
    assert not samples_got.requires_grad


def test_halfbounded_sample_gamma():
    # Construct model
    model = nn.Linear(1, 1)
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    model = Parallel(model, model)

    # Construct policy
    dist = d.Gamma
    policy = HalfBounded(model, dist)

    # Sample from the policy
    input_ = torch.zeros((1,))
    torch.manual_seed(1)
    samples_got = policy.forward(input_)[0]

    # Samples should have been reparameterized
    assert samples_got.requires_grad

    # Sample from a distribution which has the same parameters as the policy in
    # state `input_`
    torch.manual_seed(1)
    dist = dist(1, 1)
    samples_expected = dist.sample()

    # Ensure samples are close
    assert torch.isclose(samples_got, samples_expected)

    # Samples should **not** have been reparameterized
    samples_got = policy.forward(input_, rsample=False)[0]
    assert not samples_got.requires_grad
