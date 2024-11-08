"""
A module of simple state constructors that can be composed
to produce more complex state constructors
"""
import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy
from collections import deque

import corerl.utils.nullable as nullable


class BaseStateConstructorComponent(ABC):
    def __init__(self, parents: list | None = None):
        self.parents = nullable.default(parents, list)
        self.children = None
        self.called = False
        self.obs_next: np.ndarray | None = None

    def __call__(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        if not self.called:
            if len(self.parents) == 0:  # base case
                obs_parents = [obs]
            else:
                obs_parents = [p(obs, **kwargs) for p in self.parents]
            self.obs_next = self.process_observation(obs_parents, **kwargs)
            self.called = True

        assert self.obs_next is not None
        return self.obs_next

    def set_parents(self, parents: list) -> None:
        self.parents = parents

    def reset_called(self) -> None:
        self.called = False
        for p in self.parents:
            p.reset_called()

    @abstractmethod
    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _clear_state(self) -> None:
        raise NotImplementedError

    def clear_state(self) -> None:
        self._clear_state()
        for p in self.parents:
            p.clear_state()


class Identity(BaseStateConstructorComponent):
    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
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

    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
        assert len(obs_parents) == 1
        obs = obs_parents[0]
        self.obs_history.append(obs)
        self.num_elements += 1

        if self.num_elements > self.k:
            self.obs_history.pop(0)
            self.num_elements -= 1

        return_list = deepcopy(self.obs_history)
        # ensure returned list has self.k elements
        if self.num_elements < self.k:
            last_element = return_list[-1]
            for _ in range(self.num_elements, self.k):
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

    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
        assert len(obs_parents) == 1
        obs = obs_parents[0]

        zeros = np.zeros_like(obs)
        if self.trace is None:  # first observation received
            self.trace = (1 - self.trace_decay) * obs + self.trace_decay * zeros
        else:
            self.trace = (1 - self.trace_decay) * obs + self.trace_decay * self.trace
        return self.trace

    def _clear_state(self) -> None:
        self.trace = None


class Concatenate(BaseStateConstructorComponent):
    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
        return np.concatenate(obs_parents)

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

    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
        assert (len(obs_parents)) == 1
        obs = obs_parents[0]
        self.queue.appendleft(obs)
        return self.queue[0] - self.queue[-1]

    def _clear_state(self) -> None:
        self.queue = deque([], self.memory)


class LongAverage(BaseStateConstructorComponent):
    def __init__(self, memory: int, parents: list | None = None):
        super().__init__(parents=parents)
        self.memory = memory
        self.queue = deque([], self.memory)

    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
        assert len(obs_parents) == 1
        o = obs_parents[0]
        self.queue.appendleft(o)
        sum_ = 0
        for i in self.queue:
            sum_ += i

        return np.array(sum_ / self.memory)

    def _clear_state(self) -> None:
        self.queue = deque([], self.memory)


class Flatten(BaseStateConstructorComponent):
    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
        assert len(obs_parents) == 1
        return obs_parents[0].flatten()

    def _clear_state(self) -> None:
        return


class KeepCols(BaseStateConstructorComponent):
    """
    Get an individual column in an array of sensor readings
    """

    def __init__(self, keep_cols: int, parents: list | None = None):
        super().__init__(parents=parents)
        self.keep_cols = keep_cols

    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
        assert len(obs_parents) == 1
        o = obs_parents[0].copy()
        return o[self.keep_cols:self.keep_cols + 1]  # slicing since we want to return an array

    def _clear_state(self) -> None:
        return


class ErrorIntegral(BaseStateConstructorComponent):
    def __init__(self, column: int, setpoint: float, memory: int, parents: list | None = None):
        super().__init__(parents=parents)
        self.column = column
        self.setpoint = setpoint
        self.memory = memory
        self.queue = deque([], self.memory)

    def process_observation(self, obs_parents: list, **kwargs) -> np.ndarray:
        assert len(obs_parents) == 1
        o = obs_parents[0]
        error = o[self.column] - self.setpoint
        self.queue.appendleft(error)

        sum_ = 0
        for i in self.queue:
            sum_ += i

        return np.array([sum_]) / self.memory

    def _clear_state(self) -> None:
        self.queue = deque([], self.memory)


class AnytimeCountDown(BaseStateConstructorComponent):
    def __init__(self, steps_per_decision: int, parents: list | None = None, use_indicator: bool = True):
        super().__init__(parents=parents)
        self.steps_per_decision = steps_per_decision
        self.steps_until_decision = self.steps_per_decision
        self.use_indicator = use_indicator

    def process_observation(self, obs_parents: list, decision_point=False, steps_until_decision=-1,
                            **kwargs) -> np.ndarray:
        if steps_until_decision >= 0:
            self.steps_until_decision = steps_until_decision
        elif decision_point:
            self.steps_until_decision = self.steps_per_decision
        else:
            self.steps_until_decision -= 1

        indicator = 1 if decision_point else 0
        countdown = self.steps_until_decision / self.steps_per_decision
        assert 1 >= countdown >= 0, 'countdown must be between 0 and 1'
        if self.use_indicator:
            return np.array([indicator, countdown])
        else:
            return np.array([countdown])

    def _clear_state(self) -> None:
        self.steps_until_decision = self.steps_per_decision


class AnytimeOneHot(BaseStateConstructorComponent):
    def __init__(self, steps_per_decision: int, parents: list | None = None, use_indicator: bool = True):
        super().__init__(parents=parents)
        self.steps_per_decision = steps_per_decision
        self.steps_until_decision = self.steps_per_decision
        self.use_indicator = use_indicator

    def process_observation(self, obs_parents: list, decision_point=False, steps_until_decision=-1, get_state_dim=False,
                            **kwargs) -> np.ndarray:
        if steps_until_decision >= 0:
            self.steps_until_decision = steps_until_decision
        elif decision_point:
            self.steps_until_decision = self.steps_per_decision
        else:
            self.steps_until_decision -= 1

        one_hot = [0 for i in range(self.steps_per_decision)]
        one_hot[self.steps_until_decision - 1] = 1
        decision_point = self.steps_until_decision == self.steps_per_decision
        indicator = 1 if decision_point else 0
        if self.use_indicator:
            return np.array([indicator] + one_hot)
        else:
            return np.array(one_hot)

    def _clear_state(self) -> None:
        self.steps_until_decision = self.steps_per_decision


class AnytimeThermometer(BaseStateConstructorComponent):
    def __init__(self, steps_per_decision: int, parents: list | None = None, use_indicator: bool = True):
        super().__init__(parents=parents)
        self.steps_per_decision = steps_per_decision
        self.steps_until_decision = self.steps_per_decision
        self.use_indicator = use_indicator

    def process_observation(self, obs_parents: list, decision_point=False, steps_until_decision=-1,
                            get_state_dim=False, **kwargs) -> np.ndarray:
        if steps_until_decision >= 0:
            self.steps_until_decision = steps_until_decision
        elif decision_point:
            self.steps_until_decision = self.steps_per_decision
        else:
            self.steps_until_decision -= 1

        one_hot = [0 for _ in range(self.steps_per_decision)]
        for i in range(self.steps_until_decision):
            one_hot[i] = 1

        decision_point = self.steps_until_decision == self.steps_per_decision
        indicator = 1 if decision_point else 0
        if self.use_indicator:
            return np.array([indicator] + one_hot)
        else:
            return np.array(one_hot)

    def _clear_state(self) -> None:
        self.steps_until_decision = self.steps_per_decision
