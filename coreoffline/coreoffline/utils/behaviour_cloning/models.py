import logging
from abc import ABC, abstractmethod
from typing import Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
import optax
from corerl.state import AppState
from lib_agent.network.networks import LinearConfig, TorsoConfig, torso_builder
from lib_config.config import config
from lib_progress.tracker import ProgressTracker
from pydantic import Field
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)


class BaseRegressor(ABC):
    """Abstract base class for regression models."""

    def __init__(self, app_state: AppState):
        self._app_state = app_state

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class LinearRegressor(BaseRegressor):
    """Wrapper for sklearn LinearRegression with common interface."""

    def __init__(self, app_state: AppState):
        self.model = LinearRegression()
        self._app_state = app_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        self.model.fit(X, y)
        train_loss = np.mean((self.model.predict(X) - y) ** 2)
        test_loss = np.mean((self.model.predict(X_test) - y_test) ** 2)
        log.info(f"Linear Regression - Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    def predict(self, X: np.ndarray):
        return np.clip(self.model.predict(X), 0, 1)


# ---------------------------------------------------------------------------- #
#                                      MLP                                     #
# ---------------------------------------------------------------------------- #


class BatchGenerator:
    """Generates batches of training data with shuffling."""

    def __init__(self, X: jax.Array, y: jax.Array, batch_size: int):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = X.shape[0]
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size

    def generate_batches(self, rng_key: chex.PRNGKey):
        """Generate shuffled batches for one epoch."""
        perm = jax.random.permutation(rng_key, self.n_samples)
        X_shuffled = self.X[perm]
        y_shuffled = self.y[perm]

        for batch_idx in range(self.n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]


def _create_network(output_dim: int, hidden_layers: list[int]):
    def mlp_fn(x: jax.Array) -> jax.Array:
        hidden_layer_configs = [LinearConfig(size=hidden_size, activation="crelu") for hidden_size in hidden_layers]
        hidden_layer_configs.append(LinearConfig(size=output_dim, activation="identity"))
        config = TorsoConfig(layers=hidden_layer_configs)
        torso = torso_builder(config)
        output = torso(x)
        return jnp.clip(output, 0, 1)

    return hk.transform(mlp_fn)


@config()
class MLPConfig:
    hidden_layers: list[int] = Field(default_factory=lambda: [1000, 1000])
    learning_rate: float = 0.0001
    epochs: int = 300
    batch_size: int = 256


class MLPRegressor(BaseRegressor):
    def __init__(
        self,
        cfg: MLPConfig,
        app_state: AppState,
    ):
        self._app_state = app_state
        self._cfg = cfg
        self._net: hk.Transformed | None = None
        self._params: chex.ArrayTree | None = None
        self._optimizer = optax.adam(self._cfg.learning_rate)
        self._opt_state: chex.ArrayTree | None = None
        self._rng = jax.random.PRNGKey(app_state.cfg.seed)

    def _init_net(self, X: jax.Array, y: jax.Array):
        self._net = _create_network(y.shape[1], self._cfg.hidden_layers)
        assert self._net is not None
        init_key, self._rng = jax.random.split(self._rng, 2)
        sample_input = jnp.ones((1, X.shape[1]))
        self._params = self._net.init(init_key, sample_input)
        self._opt_state = self._optimizer.init(self._params)

    @jax_u.method_jit
    def _forward(self, params: chex.ArrayTree, rng: chex.PRNGKey, x: jax.Array):
        assert self._net is not None
        f = self._net.apply
        if x.ndim == 1:
            return f(params, rng, x)

        # batch mode - vmap over batch dim
        chex.assert_rank(x, 2)
        f = jax_u.vmap(f, (None, 0, 0))
        return f(params, rng, x)

    @jax_u.method_jit
    def _batch_mae_loss(self, params: Any, rngs: chex.PRNGKey, x_batch: jax.Array, y_batch: jax.Array):
        pred = self._forward(params, rngs, x_batch)
        return jnp.mean(jnp.abs(pred - y_batch))

    @jax_u.method_jit
    def _update_params(
        self,
        params: chex.ArrayTree,
        opt_state: Any,
        x_batch: jax.Array,
        y_batch: jax.Array,
        rngs: chex.PRNGKey,
        ):

        loss, grads = jax.value_and_grad(self._batch_mae_loss)(params, rngs, x_batch, y_batch)
        updates, opt_state = self._optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def _train_epoch(self, batch_generator: BatchGenerator, rng_key: chex.PRNGKey):
        assert self._params is not None
        epoch_loss = 0.0

        for X_batch, y_batch in batch_generator.generate_batches(rng_key):
            batch_key, rng_key = jax.random.split(rng_key, 2)
            batch_keys = jax.random.split(batch_key, X_batch.shape[0])

            self._params, self._opt_state, batch_loss = self._update_params(
                self._params,
                self._opt_state,
                X_batch,
                y_batch,
                batch_keys,
            )
            epoch_loss += batch_loss

        return epoch_loss / batch_generator.n_batches, rng_key

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):

        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        X_test_jax = jnp.array(X_test)
        y_test_jax = jnp.array(y_test)
        self._init_net(X_jax, y_jax)

        batch_generator = BatchGenerator(X_jax, y_jax, self._cfg.batch_size)

        with ProgressTracker(total=self._cfg.epochs, desc="Training MLP", update_interval=50) as tracker:
            for epoch in range(self._cfg.epochs):
                epoch_key, self._rng = jax.random.split(self._rng, 2)
                train_loss, self._rng = self._train_epoch(batch_generator, epoch_key)

                test_key, self._rng = jax.random.split(self._rng, 2)
                batch_keys = jax.random.split(test_key, X_test_jax.shape[0])
                test_loss = self._batch_mae_loss(self._params, batch_keys, X_test_jax, y_test_jax)

                self._app_state.metrics.write(epoch, "dl_train_loss", train_loss)
                self._app_state.metrics.write(epoch, "dl_test_loss", test_loss)

                tracker.update(
                    metrics={
                        "train_loss": float(train_loss),
                        "test_loss": float(test_loss),
                    },
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "Must initialize network first. Please call fit()"
        assert self._params is not None
        X_jax = jnp.array(X)
        batch_keys = jax.random.split(self._rng, X.shape[0])
        predictions = self._forward(self._params, batch_keys, X_jax)
        return np.array(predictions)
