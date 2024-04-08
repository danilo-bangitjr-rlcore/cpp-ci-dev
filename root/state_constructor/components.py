"""
A module of simple state constructors that can composed to produce more complex state constructors
"""
import numpy as np
import gymnasium

from abc import ABC, abstractmethod
from copy import deepcopy
from collections import deque


class BaseStateConstructorComponent(ABC):
    def __init__(self, parents: list | None = None):
        if parents is None:
            self.parents = []
        else:
            self.parents = parents
        self.children = None
        self.called = False  # this is necessary for ensuring parents are not called multiple times
        self.obs_next = None
        return

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if not self.called:
            if len(self.parents) == 0:  # base case
                obs_parents = [obs]
            else:
                obs_parents = [p(obs) for p in self.parents]
            self.obs_next = self.process_observation(obs_parents)
            self.called = True
        return self.obs_next

    def set_parents(self, parents: list) -> None:
        self.parents = parents

    def reset_called(self) -> None:
        self.called = False
        for p in self.parents:
            p.reset_called()

    @abstractmethod
    def process_observation(self, obs_parents: list) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _clear_state(self) -> None:
        raise NotImplementedError

    def clear_state(self) -> None:
        self._clear_state()
        for p in self.parents:
            p.clear_state()


class Identity(BaseStateConstructorComponent):
    def process_observation(self, obs_parents: list) -> np.ndarray:
        assert len(obs_parents) == 1
        return obs_parents[0]

    def _clear_state(self) -> None:
        return


class KOrderHistory(BaseStateConstructorComponent):
    def __init__(self, k: int = 1, parents: list | None = None):
        super().__init__(parents=parents)
        self.k = k
        self.obs_history = []
        self.num_elements = 0

    def process_observation(self, obs_parents: np.ndarray) -> np.ndarray:
        assert len(obs_parents) == 1
        obs = obs_parents[0]
        self.obs_history.append(obs)

        if len(self.obs_history) > self.k:
            self.obs_history.pop(0)

        return_list = deepcopy(self.obs_history)
        # ensure returned list has self.k elements
        if len(return_list) < self.k:
            last_element = return_list[-1]
            for _ in range(len(return_list), self.k):
                return_list.append(last_element)
        return np.array(return_list)

    def _clear_state(self) -> None:
        self.obs_history = []
        self.num_elements = 0


class MemoryTrace(BaseStateConstructorComponent):
    def __init__(self, trace_decay: float, parents: list | None = None):
        super().__init__(parents=parents)
        assert 0 <= trace_decay <= 1
        self.trace_decay = trace_decay
        self.trace = None

    def process_observation(self, obs_parents: list) -> np.ndarray:
        assert len(obs_parents) == 1
        obs = obs_parents[0]
        if self.trace is None:  # first observation received
            self.trace = (1 - self.trace_decay) * obs + self.trace_decay * np.zeros_like(obs)
        else:
            self.trace = (1 - self.trace_decay) * obs + self.trace_decay * self.trace
        return self.trace

    def _clear_state(self) -> None:
        self.trace = None


class Concatenate(BaseStateConstructorComponent):
    def process_observation(self, obs_parents: list) -> np.ndarray:
        return np.concatenate(obs_parents, axis=0)

    def _clear_state(self) -> None:
        return


class MaxminNormalize(BaseStateConstructorComponent):
    def __init__(self, env: gymnasium.Env, parents: list | None = None):
        super().__init__(parents=parents)
        # NOTE: this should only be used for environments with continuous observation spaces
        self.low = env.observation_space.low
        self.high = env.observation_space.high

    def process_observation(self, obs_parents: list) -> np.ndarray:
        assert len(obs_parents) == 1
        obs = obs_parents[0]
        obs = (obs - self.low) / (self.high - self.low)
        return obs

    def _clear_state(self) -> None:
        return


class Difference(BaseStateConstructorComponent):
    """
    Difference between the first and last element in a queue
    """
    def __init__(self, memory: int, parents: list | None = None):
        super().__init__(parents=parents)
        self.memory = memory
        self.queue = deque([], self.memory)

    def process_observation(self, obs_parents: list) -> np.ndarray:
        assert (len(obs_parents)) == 1
        obs = obs_parents[0]
        self.queue.appendleft(obs)
        return self.queue[0] - self.queue[-1]

    def _clear_state(self) -> None:
        self.queue = deque([], self.memory)


class Average(BaseStateConstructorComponent):
    def process_observation(self, obs_parents: list) -> np.ndarray:
        assert len(obs_parents) == 1
        obs = obs_parents[0].copy()
        assert len(obs.shape) == 2
        return np.mean(obs, axis=0).reshape(1, -1)

    def _clear_state(self) -> None:
        return
