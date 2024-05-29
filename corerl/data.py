import numpy as np
from torch import Tensor

from dataclasses import dataclass, fields
from corerl.state_constructor.base import BaseStateConstructor


@dataclass
class Transition:
    state: np.array
    action: np.array
    reward: float
    next_state: np.array
    terminated: bool
    truncate: bool
    decision_point: bool  # whether state is a decision point
    next_decision_point: bool  # whether next_state is a decision point
    gamma_exponent: int  # the exponent of gamma used for bootstrapping
    observation: np.array  # the raw observation of state
    next_observation: np.array  # the raw observation of next_state

    def __iter__(self):
        for field in fields(self):
            yield getattr(self, field.name)

    @property
    def field_names(self):
        return [field.name for field in fields(self)]


@dataclass
class TransitionBatch:
    """
    Like transition, but is a batch of the above attributions
    """
    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor
    terminated: Tensor
    truncate: Tensor
    decision_point: Tensor
    next_decision_point: Tensor
    gamma_exponent: Tensor
    observation: Tensor
    next_observation: Tensor

    def __post_init__(self):
        # ensure all the attributes have the same dimension
        state_batch_size = self.state.size(0)
        for field in fields(self):
            assert getattr(self, field.name).size(0) == state_batch_size, \
                f"Element {field.name} does not have the same batch size as the state"

    @property
    def batch_size(self) -> int:
        return self.state.size(0)


@dataclass
class Trajectory:
    is_test: bool

    def __post_init__(self):
        self.transitions = []
        self.start_sc = None

    def add_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)

    @property
    def num_transitions(self):
        return len(self.transitions)

    def add_start_sc(self, sc: BaseStateConstructor) -> None:
        self.start_sc = sc
