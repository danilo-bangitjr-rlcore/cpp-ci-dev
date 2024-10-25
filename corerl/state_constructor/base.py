import numpy as np
from abc import ABC, abstractmethod
from corerl.utils.hydra import Group

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from corerl.state_constructor.components import BaseStateConstructorComponent


class BaseStateConstructor(ABC):
    """
    A base class for state constructor. You may extend this class and implement your own project-specific
    state constructors. Alternatively, use CompositeStateConstructor and the module of state constructor components in
    components.py to build a state_constructor.

    """

    def __init__(self):
        self.state = None

    @abstractmethod
    def __call__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        initial_state: bool = False,
        get_state_dim: bool = False,
        **kwargs,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def get_state_dim(self, obs: np.ndarray, action: np.ndarray) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def get_current_state(self) -> np.ndarray:
        assert self.state is not None
        return self.state

    def reset_called(self): # noqa: B027
        ...

    def clear_state(self): # noqa: B027
        ...

class CompositeStateConstructor(BaseStateConstructor):
    def __init__(self):
        self.sc: 'BaseStateConstructorComponent | None' = None

    def __call__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        initial_state: bool = False,
        get_state_dim: bool = False,
        **kwargs,
    ) -> np.ndarray:
        a_obs = np.concatenate([action, obs])  # convention: action goes first in the state array
        # get_state_dim is a flag whether we are calling just for the purpose of getting the state dim
        state = self._call_graph(a_obs, get_state_dim=get_state_dim, **kwargs)
        self._reset_graph_call()
        init_array = np.zeros(1, dtype=bool)  # whether this is an initial state
        init_array[0] = initial_state
        state = np.concatenate([init_array, state])
        self.state = state
        return state

    def get_state_dim(self, obs: np.ndarray, action: np.ndarray) -> int:
        state = self(obs, action, get_state_dim=True)
        assert len(state.shape)  # not sure if this will always be necessary or desired. But we are assuming that
        state_dim = state.shape[0]
        self.reset()
        return state_dim

    def _call_graph(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        assert self.sc is not None
        return self.sc(obs, **kwargs)

    def _reset_graph_call(self) -> None:
        """
        Resets the self.called variables in the graph
        """
        assert self.sc is not None
        self.sc.reset_called()

    def _reset_graph_state(self) -> None:
        """
        Resets the state of all components in the graph
        """
        assert self.sc is not None
        self.sc.clear_state()

    def reset(self) -> None:
        self.state = None
        self._reset_graph_state()



# set up config groups
sc_group = Group(
    'state_constructor',
    return_type=CompositeStateConstructor,
)
