import numpy as np
import torch
import gymnasium

from omegaconf import DictConfig
from abc import ABC, abstractmethod
from typing import Optional


class BaseStateConstructor(ABC):
    """
    A base class for state constructor. You may extend this class and implement your own project-specific
    state constructors. Alternatively, use CompositeStateConstructor and the module of state constructor components in
    components.py to build a state_constructor.

    """

    @abstractmethod
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        raise NotImplementedError

    def __call__(self, obs: np.ndarray, action: np.ndarray, initial_state=False, **kwargs) -> np.ndarray:
        """
        We pass in the new observation, the last action, and whether or not this state is an initial state
        """

        # TODO: have this include a dummy action
        init_array = np.zeros(1, dtype=bool)  # whether or not this is an initial state
        init_array[0] = initial_state
        state = np.concatenate([init_array, action, obs])  # convention: action goes first in the state array
        raise state

    @abstractmethod
    def get_state_dim(self, *args) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def get_current_state(self) -> np.ndarray:
        return self.state


class CompositeStateConstructor(BaseStateConstructor):
    @abstractmethod
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        self.sc = None  # a placeholder for the final component in the graph
        raise NotImplementedError

    def __call__(self, obs: np.ndarray | torch.Tensor, action: np.ndarray | torch.Tensor, initial_state=False,
                 **kwargs) -> np.ndarray | torch.Tensor:

        if isinstance(obs, np.ndarray):
            assert isinstance(action, np.ndarray), 'obs and action must have the same type'
            concat_fn = np.concatenate
            zero_fn =  np.zeros
        elif isinstance(obs, torch.Tensor):
            assert isinstance(action, torch.Tensor), 'obs and action must have the same type'
            concat_fn = torch.concat
            zero_fn = torch.zeros

        a_obs = concat_fn([action, obs])  # convention: action goes first in the state array
        state = self._call_graph(a_obs, **kwargs)
        self._reset_graph_call()
        init_array = zero_fn(1, dtype=bool)  # whether or not this is an initial state
        init_array[0] = initial_state
        state = concat_fn([init_array, state])
        return state

    def get_state_dim(self, obs: np.ndarray, action: np.ndarray) -> int:
        state = self(obs, action)
        assert len(state.shape)  # not sure if this will always be necessary or desired. But we are assuming that
        state_dim = state.shape[0]
        self._reset_graph_state()
        return state_dim

    def _call_graph(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return self.sc(obs, **kwargs)

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

    def reset(self) -> None:
        self._reset_graph_state()
