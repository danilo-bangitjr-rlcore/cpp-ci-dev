import torch

from corerl.utils.random import rejection_sample


def test_rejection_sample_easy():
    """
    With this configuration, it should take ~2-4 attempts
    to get a complete sample using the oversampling method.

    In the last attempt, with very high probability we will
    have more than the requested number of samples and will
    need to truncate.
    """

    n_samples = 100
    action_dim = 5

    def sampler(n: int):
        dist = torch.distributions.Normal(loc=0.2, scale=0.2)
        return dist.sample((n, action_dim))

    def fallback(n: int):
        dist = torch.distributions.Uniform(0, 1)
        return dist.sample((n, action_dim))

    def predicate(arr: torch.Tensor):
        return ((0 < arr) & (arr < 1)).all(dim=1)

    sample = rejection_sample(
        sampler,
        predicate,
        fallback=fallback,
        n_samples=n_samples,
    )

    assert sample.shape == (n_samples, action_dim)
    assert predicate(sample).all()


def test_rejection_sample_out_of_sample():
    """
    With this configuration, it is nearly impossible for the sampler
    to meet the predicate bounds. As a result, we will need to rely
    on the fallback.
    """

    n_samples = 100
    action_dim = 5

    def sampler(n: int):
        dist = torch.distributions.Normal(loc=-5, scale=1)
        return dist.sample((n, action_dim))

    def fallback(n: int):
        dist = torch.distributions.Uniform(0, 1)
        return dist.sample((n, action_dim))

    def predicate(arr: torch.Tensor):
        return ((0 < arr) & (arr < 1)).all(dim=1)

    sample = rejection_sample(
        sampler,
        predicate,
        max_iter=10,
        fallback=fallback,
        n_samples=n_samples,
    )

    assert sample.shape == (n_samples, action_dim)
    assert predicate(sample).all()
