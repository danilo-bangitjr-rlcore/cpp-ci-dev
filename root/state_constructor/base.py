from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

import gymnasium


class BaseStateConstructor(ABC):
    """
    A base class for state constructor. You may extend this class and implement your own project-specific
    state constructors. Alternatively, use CompositeStateConstructor and the module of state constructor components in
    components.py to build a state_constructor.

    """

    @abstractmethod
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_state_dim(self, *args) -> int:
        raise NotImplementedError


class CompositeStateConstructor(BaseStateConstructor):
    @abstractmethod
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        self.sc = None  # a placeholder for the final component in the graph
        raise NotImplementedError

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        state = self._call_graph(obs)
        self._reset_graph_call()
        return state

    def get_state_dim(self, obs: np.ndarray) -> int:
        state = self(obs)
        assert len(state.shape) == 1
        state_dim = state.shape[0]
        self._reset_graph_state()
        return state_dim

    def _call_graph(self, obs: np.ndarray) -> np.ndarray:
        return self.sc(obs)

    def _reset_graph_call(self) -> None:
        """
        Resets the self.called variables in the graph
        """
        self.sc.reset_called()

    def _reset_graph_state(self) -> None:
        """
        Resets the state of all components in the graph
        """
        self.sc.clear_state()
