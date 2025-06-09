import distrax
import jax

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

    rng = jax.random.PRNGKey(0)

    def sampler(n: int):
        dist = distrax.Normal(loc=0.5, scale=0.2)
        return dist.sample(seed=rng, sample_shape=(n, action_dim))

    def fallback(n: int):
        dist = distrax.Uniform(low=0, high=1)
        return dist.sample(seed=rng, sample_shape=(n, action_dim))

    def predicate(arr: jax.Array):
        return ((0 < arr) & (arr < 1)).all(axis=1)

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

    rng = jax.random.PRNGKey(0)

    def sampler(n: int):
        dist = distrax.Normal(loc=-5, scale=1)
        return dist.sample(seed=rng, sample_shape=(n, action_dim))

    def fallback(n: int):
        dist = distrax.Uniform(low=0, high=1)
        return dist.sample(seed=rng, sample_shape=(n, action_dim))

    def predicate(arr: jax.Array):
        return ((0 < arr) & (arr < 1)).all(axis=1)

    sample = rejection_sample(
        sampler,
        predicate,
        max_iter=10,
        fallback=fallback,
        n_samples=n_samples,
    )

    assert sample.shape == (n_samples, action_dim)
    assert predicate(sample).all()
