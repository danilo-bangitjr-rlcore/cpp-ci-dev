import numpy as np
from torch import Tensor

from dataclasses import dataclass, fields
from corerl.state_constructor.base import BaseStateConstructor


@dataclass
class ObsTransition:
    prev_action: np.array  # the action taken over the duration of 'obs'. 'prev_action' and 'obs' are passed to the state constructor
    obs: np.array  # the raw observation of state
    obs_steps_until_decision: int
    obs_dp: bool  # Whether 'obs' is at a decision point
    action: np.array  # the action taken after 'obs' that occurs concurrently with 'next_obs'
    reward: float
    next_obs: np.array  # the immediate next observation
    next_obs_steps_until_decision: int
    next_obs_dp: bool  # Whether 'next_obs' is at a decision point
    terminated: bool
    truncate: bool
    gap: bool  # whether there is a gap in the dataset following next_obs

    def __iter__(self):
        for field in fields(self):
            yield getattr(self, field.name)

    @property
    def field_names(self):
        return [field.name for field in fields(self)]

    def __str__(self):
        string = ''
        for field in fields(self):
            string += f"{field.name}: {getattr(self, field.name)}\n"
        return string


@dataclass
class Transition:
    obs: np.array  # the raw observation of state
    state: np.array
    action: np.array
    next_obs: np.array  # the immediate next observation
    next_state: np.array  # the next state in the
    # NOTE: we distinguish between the next state and the next state which we bootstrap off of. All following
    # attributes are defined w.r.t. the boot strap state.
    reward: float # one-step reward
    n_step_reward: float
    n_step_cumulants: np.array
    # the state which we bootstrap off of, which is not necesssarily the next state
    # in the MDP
    boot_obs: np.array  # the raw observation of next_state
    boot_state: np.array
    terminated: bool
    truncate: bool
    state_dp: bool  # whether state is a decision point
    next_state_dp: bool  # Whether 'next_obs' is at a decision point
    boot_state_dp: bool  # whether next_state is a decision point
    gamma_exponent: int  # the exponent of gamma used for bootstrapping
    gap: bool  # whether there is a gap in the dataset following next_obs, always false online

    def __iter__(self):
        for field in fields(self):
            yield getattr(self, field.name)

    @property
    def field_names(self):
        return [field.name for field in fields(self)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):  # Not the same class, so not equal
            return False

        for field in fields(self):
            attr_self = getattr(self, field.name)
            attr_other = getattr(other, field.name)

            if not isinstance(attr_self, type(attr_other)):  # attributes are not the same class
                return False

            if isinstance(attr_self, np.ndarray):
                if not np.array_equal(attr_self, attr_other):
                    return False
            elif attr_self != attr_other:
                return False

        return True

    def __str__(self):
        string = ''
        for field in fields(self):
            string += f"{field.name}: {getattr(self, field.name)}\n"
        return string


@dataclass
class TransitionBatch:
    """
    Like transition, but is a batch of the above attributions
    """
    obs: Tensor
    state: Tensor
    action: Tensor
    next_obs: Tensor
    next_state: Tensor
    reward: Tensor
    n_step_reward: Tensor
    n_step_cumulants: Tensor
    boot_obs: Tensor
    boot_state: Tensor
    terminated: Tensor
    truncate: Tensor
    state_dp: Tensor
    next_state_dp: Tensor
    boot_state_dp: Tensor
    gamma_exponent: Tensor
    gap: Tensor

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
    def __post_init__(self):
        self.transitions = []
        self.start_sc = None
        self.scs = []

    def add_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def add_sc(self, sc: BaseStateConstructor) -> None:
        self.scs.append(sc)

    @property
    def num_transitions(self):
        return len(self.transitions)

    def add_start_sc(self, sc: BaseStateConstructor) -> None:
        self.start_sc = sc

    def get_transitions_attr(self, attr):
        """
        Returns a numpy array, which is the concatenation of all the transitions attribute for attr
        """
        if len(self.transitions) > 0:
            if hasattr(self.transitions[0], attr):
                # return array of that attribute for all transitions
                attribute_list = [getattr(transition, attr).reshape(1, -1) for transition in self.transitions]
                return np.concatenate(attribute_list, axis=0)
            else:
                raise AttributeError("Invalid attribute for Trajectory")
        else:
            raise AssertionError("Please ensure that transitions have been added")

    def split_at(self, idx):
        child_1 = Trajectory()
        child_2 = Trajectory()

        child_1.transitions = self.transitions[:idx]
        child_2.transitions = self.transitions[idx:]

        child_1.scs = self.scs[:idx]
        child_2.scs = self.scs[idx:]

        child_1.start_sc = child_1.scs[0]
        child_2.start_sc = child_2.scs[0]

        return child_1, child_2
