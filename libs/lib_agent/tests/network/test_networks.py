from collections.abc import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import pytest

from lib_agent.network.networks import NoisyLinearConfig, noisy_linear


class TestNoisyLinear:
    @pytest.fixture
    def noisy_network(self):
        """Create a transformed noisy linear network."""
        def _make_network(cfg: NoisyLinearConfig):
            noisy_fn = noisy_linear(cfg)

            def network(x: jax.Array) -> jax.Array:
                return noisy_fn(x)

            return hk.transform(network)
        return _make_network

    def test_noise_changes_output(self, noisy_network: Callable[[NoisyLinearConfig], hk.Transformed]):
        """Test that noise actually affects the output."""
        cfg = NoisyLinearConfig(size=32)
        transformed = noisy_network(cfg)
        x = jnp.ones((5, 16))

        rng1 = jax.random.PRNGKey(42)
        params = transformed.init(rng1, x)

        rng2 = jax.random.PRNGKey(123)
        rng3 = jax.random.PRNGKey(456)

        output1 = transformed.apply(params, rng2, x)
        output2 = transformed.apply(params, rng3, x)

        assert not jnp.allclose(output1, output2)

    def test_deterministic_with_same_key(self, noisy_network: Callable[[NoisyLinearConfig], hk.Transformed]):
        """Test that output is deterministic with the same random key."""
        cfg = NoisyLinearConfig(size=32)
        transformed = noisy_network(cfg)
        rng = jax.random.PRNGKey(42)
        x = jnp.ones((5, 16))

        params = transformed.init(rng, x)

        # Apply with same key multiple times
        apply_rng = jax.random.PRNGKey(123)
        output1 = transformed.apply(params, apply_rng, x)
        output2 = transformed.apply(params, apply_rng, x)

        assert jnp.allclose(output1, output2)

    def test_batch_dimension_handling(self, noisy_network: Callable[[NoisyLinearConfig], hk.Transformed]):
        """Test with different batch sizes."""
        cfg = NoisyLinearConfig(size=32, activation='relu')
        transformed = noisy_network(cfg)
        rng = jax.random.PRNGKey(42)

        batch_sizes = [1, 8, 16, 64]
        input_size = 10

        for batch_size in batch_sizes:
            x = jnp.ones((batch_size, input_size))
            params = transformed.init(rng, x)
            output = transformed.apply(params, rng, x)

            assert output.shape == (batch_size, 32)
