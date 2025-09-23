import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from corerl.state import AppState
from lib_config.config import config
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
        return self.model.predict(X)


# ---------------------------------------------------------------------------- #
#                                      MLP                                     #
# ---------------------------------------------------------------------------- #


def init_mlp_params(layer_sizes: Sequence[int], key: jax.Array):
    """Initialize MLP parameters."""
    keys = jax.random.split(key, len(layer_sizes))
    params = []
    for i in range(len(layer_sizes) - 1):
        key = keys[i]
        w_key, _ = jax.random.split(key)

        # Xavier initialization
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        w_std = jnp.sqrt(2.0 / (fan_in + fan_out))

        w = jax.random.normal(w_key, (layer_sizes[i], layer_sizes[i + 1])) * w_std
        b = jnp.zeros(layer_sizes[i + 1])
        params.append((w, b))
    return params


def mlp_forward(params: list, x: jax.Array) -> jax.Array:
    """Forward pass through MLP."""
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)

    # Linear output layer
    w, b = params[-1]
    return x @ w + b


def mae_loss(params: list, x: jax.Array, y: jax.Array) -> jax.Array:
    pred = mlp_forward(params, x)
    return jnp.mean(jnp.abs(pred - y))


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
        self.params: Any = None
        self.optimizer: Any = None
        self.opt_state: Any = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Train the MLP."""
        input_dim = X.shape[1]
        output_dim = y.shape[1] if y.ndim > 1 else 1

        layer_sizes = [input_dim, *self._cfg.hidden_layers, output_dim]

        key = jax.random.PRNGKey(42)
        self.params = init_mlp_params(layer_sizes, key)

        self.optimizer = optax.adam(self._cfg.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        @jax.jit
        def update_step(params: Any, opt_state: Any, x_batch: Any, y_batch: Any) -> tuple[Any, Any, Any]:
            loss, grads = jax.value_and_grad(mae_loss)(params, x_batch, y_batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)  # type: ignore
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        X_jax = jnp.array(X)
        y_jax = jnp.array(y)

        # Prepare test data
        X_test_jax = jnp.array(X_test)
        y_test_jax = jnp.array(y_test)

        n_samples = X_jax.shape[0]
        n_batches = (n_samples + self._cfg.batch_size - 1) // self._cfg.batch_size

        for epoch in range(self._cfg.epochs):
            # Shuffle data for each epoch
            key = jax.random.PRNGKey(epoch)
            perm = jax.random.permutation(key, n_samples)
            X_shuffled = X_jax[perm]
            y_shuffled = y_jax[perm]

            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self._cfg.batch_size
                end_idx = min(start_idx + self._cfg.batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                self.params, self.opt_state, batch_loss = update_step(
                    self.params,
                    self.opt_state,
                    X_batch,
                    y_batch,
                )
                epoch_loss += batch_loss

            train_loss = epoch_loss / n_batches

            # Calculate test loss
            test_loss = mae_loss(self.params, X_test_jax, y_test_jax)
            self._app_state.metrics.write(epoch, "dl_train_loss", train_loss)
            self._app_state.metrics.write(epoch, "dl_test_loss", test_loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        X_jax = jnp.array(X)
        predictions = mlp_forward(self.params, X_jax)
        return np.array(predictions)
