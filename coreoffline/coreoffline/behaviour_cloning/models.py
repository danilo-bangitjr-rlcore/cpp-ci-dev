import logging
from abc import ABC, abstractmethod

import numpy as np
from corerl.state import AppState

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
