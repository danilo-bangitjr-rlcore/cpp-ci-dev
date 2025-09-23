import logging
from abc import ABC, abstractmethod

import numpy as np
from corerl.state import AppState
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
