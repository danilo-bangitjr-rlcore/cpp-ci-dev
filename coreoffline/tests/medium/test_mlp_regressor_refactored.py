"""Tests for the refactored MLPRegressor with BatchGenerator."""
import jax
import jax.numpy as jnp
import numpy as np
from corerl.state import AppState

from coreoffline.utils.behaviour_cloning.models import BatchGenerator, MLPConfig, MLPRegressor
from coreoffline.utils.config import OfflineMainConfig


class TestBatchGenerator:
    def test_batch_generator_initialization(self):
        """Test BatchGenerator initialization and properties."""
        X = jnp.array(np.random.rand(100, 10).astype(np.float32))
        y = jnp.array(np.random.rand(100, 1).astype(np.float32))
        batch_size = 32

        generator = BatchGenerator(X, y, batch_size)

        assert generator.n_samples == 100
        assert generator.batch_size == 32
        assert generator.n_batches == 4  # ceil(100/32) = 4

    def test_batch_generator_generates_correct_batches(self):
        """Test that BatchGenerator produces correct number and size of batches."""
        X = jnp.array(np.random.rand(50, 5).astype(np.float32))
        y = jnp.array(np.random.rand(50, 1).astype(np.float32))
        batch_size = 20

        generator = BatchGenerator(X, y, batch_size)
        rng_key = jax.random.PRNGKey(42)

        batches = list(generator.generate_batches(rng_key))

        assert len(batches) == 3  # ceil(50/20) = 3
        assert batches[0][0].shape[0] == 20  # First batch size
        assert batches[1][0].shape[0] == 20  # Second batch size
        assert batches[2][0].shape[0] == 10  # Last batch size (remainder)

        # Check all data is covered
        total_samples = sum(batch[0].shape[0] for batch in batches)
        assert total_samples == 50

    def test_batch_generator_shuffles_data(self):
        """Test that BatchGenerator shuffles data between epochs."""
        X = jnp.arange(20).reshape(20, 1).astype(jnp.float32)
        y = jnp.arange(20).reshape(20, 1).astype(jnp.float32)
        batch_size = 20

        generator = BatchGenerator(X, y, batch_size)

        # Generate batches with different random keys
        batches1 = list(generator.generate_batches(jax.random.PRNGKey(1)))
        batches2 = list(generator.generate_batches(jax.random.PRNGKey(2)))

        # With different random seeds, shuffling should produce different orders
        first_batch_1 = batches1[0][0].flatten()
        first_batch_2 = batches2[0][0].flatten()

        # The batches should be different due to shuffling
        assert not jnp.array_equal(first_batch_1, first_batch_2)


def generate_synthetic_data(n_samples: int, input_dim: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for mean threshold classification."""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(n_samples, input_dim))
    y_labels = (np.mean(X, axis=1) >= 0.5).astype(np.float32)
    y = y_labels.reshape(-1, 1)
    return X.astype(np.float32), y


class TestMLPRegressorRefactored:
    """Test the MLPRegressor"""

    def test_mlp_regressor_synthetic_classification(
        self,
        dummy_app_state: AppState,
        offline_cfg: OfflineMainConfig,
    ):
        """Test MLPRegressor on synthetic mean threshold classification task."""
        dummy_app_state.cfg = offline_cfg

        input_dim = 8
        n_train_samples = 1000
        n_test_samples = 50

        # Generate synthetic data
        X_train, y_train = generate_synthetic_data(n_train_samples, input_dim, seed=42)
        X_test, y_test = generate_synthetic_data(n_test_samples, input_dim, seed=123)

        # Create model configuration - small network for quick testing
        mlp_config = MLPConfig(
            hidden_layers=[32, 16],
            learning_rate=0.005,
            epochs=5,
            batch_size=32,
        )
        model = MLPRegressor(mlp_config, dummy_app_state)
        model.fit(X_train, y_train, X_test=X_test, y_test=y_test)

        # Test predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Verify output shapes
        assert train_predictions.shape == y_train.shape
        assert test_predictions.shape == y_test.shape

        # Verify predictions are in valid range [0, 1] due to clipping
        assert np.all(train_predictions >= 0)
        assert np.all(train_predictions <= 1)
        assert np.all(test_predictions >= 0)
        assert np.all(test_predictions <= 1)

        # Check that model learns something (accuracy should be better than random)
        train_accuracy = np.mean((train_predictions > 0.5) == (y_train > 0.5))
        test_accuracy = np.mean((test_predictions > 0.5) == (y_test > 0.5))

        # For this simple task, should achieve reasonable accuracy
        assert train_accuracy > 0.9
        assert test_accuracy > 0.9
