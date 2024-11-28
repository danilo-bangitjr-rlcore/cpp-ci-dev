from typing import Dict
import numpy as np
from torch import Tensor
from copy import deepcopy
from math import isclose
import pandas as pd
import torch
import datetime

from dataclasses import dataclass, fields, field

from corerl.state_constructor.base import BaseStateConstructor

from enum import IntFlag, auto, Enum


class MissingType(IntFlag):
    NULL = auto()
    MISSING = auto()  # indicates data did not exist in db
    BOUNDS = auto()
    OUTLIER = auto()


# for use to create sparse pandas dataframes
# for example: sparse_df = pd.DataFrame(..., dtype=SparseMissingType)
SparseMissingType = pd.SparseDtype(dtype=int, fill_value=MissingType.NULL)


@dataclass
class OldObsTransition:
    # the action taken over the duration of 'obs'. 'prev_action' and 'obs' are passed to the state constructor
    prev_action: np.ndarray | None
    obs: np.ndarray | None  # the raw observation of state
    obs_steps_until_decision: int
    obs_dp: bool  # Whether 'obs' is at a decision point
    action: np.ndarray  # the action taken after 'obs' that occurs concurrently with 'next_obs'
    reward: float
    next_obs: np.ndarray | None  # the immediate next observation
    next_obs_steps_until_decision: int
    next_obs_dp: bool  # Whether 'next_obs' is at a decision point
    terminated: bool
    truncate: bool
    gap: bool  # whether there is a gap in the dataset following next_obs

    def __iter__(self):
        for f in fields(self):
            yield getattr(self, f.name)

    @property
    def field_names(self):
        return [f.name for f in fields(self)]

    def __str__(self):
        string = ''
        for f in fields(self):
            string += f"{f.name}: {getattr(self, f.name)}\n"
        return string


@dataclass
class ObsTransition:
    obs: np.ndarray  # the raw observation of state
    action: np.ndarray  # the action taken after 'obs' that occurs concurrently with 'next_obs'
    reward: float
    next_obs: np.ndarray  # the immediate next observation
    terminated: bool = False
    truncate: bool = False
    gap: bool = False  # whether there is a gap in the dataset following next_obs

    def __iter__(self):
        for f in fields(self):
            yield getattr(self, f.name)

    @property
    def field_names(self):
        return [f.name for f in fields(self)]

    def __str__(self):
        string = ''
        for f in fields(self):
            string += f"{f.name}: {getattr(self, f.name)}\n"
        return string


@dataclass
class GORAS:
    gamma: float
    obs: np.ndarray
    reward: float
    action: np.ndarray
    state: np.ndarray


@dataclass
class NewTransition2:
    pre: GORAS
    post: GORAS
    n_steps: int


@dataclass
class NewTransition:
    state: Tensor
    action: Tensor
    n_steps: int
    n_step_reward: float
    next_state: Tensor
    # next_state_dp: bool # not sure if we need this yet.
    terminated: bool
    truncate: bool

    def __str__(self):
        string = ''
        for f in fields(self):
            string += f"{f.name}: {getattr(self, f.name)}\n"
        return string

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):  # Not the same class, so not equal
            return False

        for f in fields(self):
            attr_self = getattr(self, f.name)
            attr_other = getattr(other, f.name)

            if not isinstance(attr_self, type(attr_other)):  # attributes are not the same class
                return False

            if isinstance(attr_self, Tensor):
                if not torch.allclose(attr_self, attr_other):
                    return False

            elif isinstance(attr_self, float):
                if not isclose(attr_self, attr_other):
                    return False

            elif attr_self != attr_other:
                return False

        return True


@dataclass
class Transition:
    obs: np.ndarray | None  # the raw observation of state
    state: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray | None  # the immediate next observation
    next_state: np.ndarray  # the next state in the
    # NOTE: we distinguish between the next state and the next state which we bootstrap off of. All following
    # attributes are defined w.r.t. the boot strap state.
    reward: float  # one-step reward
    n_step_reward: float
    n_step_cumulants: np.ndarray | None = None
    # the state which we bootstrap off of, which is not necesssarily the next state
    # in the MDP
    boot_obs: np.ndarray | None = None  # the raw observation of next_state
    boot_state: np.ndarray | None = None
    terminated: bool = False
    truncate: bool = False
    state_dp: bool = True  # whether state is a decision point
    next_state_dp: bool = True  # Whether 'next_obs' is at a decision point
    boot_state_dp: bool = True  # whether next_state is a decision point
    gamma_exponent: int = 1  # the exponent of gamma used for bootstrapping
    gap: bool = False  # whether there is a gap in the dataset following next_obs, always false online
    steps_until_decision: int = 1
    next_steps_until_decision: int = 1
    boot_steps_until_decision: int = 1

    def __iter__(self):
        for f in fields(self):
            yield getattr(self, f.name)

    @property
    def field_names(self):
        return [f.name for f in fields(self)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):  # Not the same class, so not equal
            return False

        for f in fields(self):
            attr_self = getattr(self, f.name)
            attr_other = getattr(other, f.name)

            if not isinstance(attr_self, type(attr_other)):  # attributes are not the same class
                return False

            if isinstance(attr_self, np.ndarray):
                if not np.allclose(attr_self, attr_other):
                    return False

            elif isinstance(attr_self, float):
                if not isclose(attr_self, attr_other):
                    return False

            elif attr_self != attr_other:
                return False

        return True

    def __str__(self):
        string = ''
        for f in fields(self):
            string += f"{f.name}: {getattr(self, f.name)}\n"
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
    steps_until_decision: Tensor
    next_steps_until_decision: Tensor
    boot_steps_until_decision: Tensor

    def __post_init__(self):
        # ensure all the attributes have the same dimension
        state_batch_size = self.state.size(0)
        for f in fields(self):
            assert getattr(self, f.name).size(0) == state_batch_size, \
                f"Element {f.name} does not have the same batch size as the state"

    @property
    def batch_size(self) -> int:
        return self.state.size(0)


@dataclass
class Trajectory:
    def __post_init__(self):
        self.transitions: list[Transition] = []
        self.start_sc: BaseStateConstructor | None = None
        self.scs: list[BaseStateConstructor] | None = None

    def add_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)

    @property
    def num_transitions(self):
        return len(self.transitions)

    def add_start_sc(self, sc: BaseStateConstructor) -> None:
        self.start_sc = sc

    def cache_scs(self) -> list[BaseStateConstructor]:
        assert self.start_sc is not None
        self.scs = []
        sc = deepcopy(self.start_sc)
        self.scs.append(deepcopy(sc))
        for transition in self.transitions[:-1]:
            assert transition.next_obs is not None
            sc(transition.next_obs,
                transition.action,
                initial_state=False,  # assume the next state will never be an initial state.
                decision_point=transition.next_state_dp,
                steps_until_decision=transition.next_steps_until_decision)
            self.scs.append(deepcopy(sc))

        return self.scs

    def get_sc_at_idx(self, idx: int) -> BaseStateConstructor:
        """
        rolls the initial state constructor forward to
        """
        assert self.start_sc is not None
        if self.scs is not None:
            return deepcopy(self.scs[idx])

        sc = deepcopy(self.start_sc)
        for transition in self.transitions[:idx]:
            assert transition.next_obs is not None
            sc(transition.next_obs,
                transition.action,
                initial_state=False,  # assume the next state will never be an initial state.
                decision_point=transition.next_state_dp,
                steps_until_decision=transition.next_steps_until_decision)
        return sc

    def get_transitions_attr(self, attr: str):
        """
        Returns a numpy array, which is the concatenation of all the transitions attribute for attr
        """
        if len(self.transitions) == 0:
            raise AssertionError("Please ensure that transitions have been added")

        if not hasattr(self.transitions[0], attr):
            raise AttributeError("Invalid attribute for Trajectory")

        # return array of that attribute for all transitions
        attribute_list = [getattr(transition, attr).reshape(1, -1) for transition in self.transitions]
        return np.concatenate(attribute_list, axis=0)

    def split_at(self, idx: int):
        child_1 = Trajectory()
        child_2 = Trajectory()

        child_2.start_sc = self.get_sc_at_idx(idx)

        if self.scs is not None:
            child_1.scs = self.scs[:idx]
            child_2.scs = self.scs[idx:]

        child_1.transitions = self.transitions[:idx]
        child_2.transitions = self.transitions[idx:]

        return child_1, child_2


class CallerCode(Enum):
    OFFLINE = auto()
    ONLINE = auto()
    REFRESH = auto()


class StageCode(Enum):
    IMPUTER = auto()
    TC = auto()
    SC = auto()


type TemporalState = Dict[StageCode, object | None]


@dataclass
class PipelineFrame:
    data: pd.DataFrame
    missing_info: pd.DataFrame = field(init=False)
    caller_code: CallerCode
    action_tags: list[str] = field(default_factory=list)
    obs_tags: list[str] = field(default_factory=list)
    reward_tags: list[str] = field(default_factory=list)
    data_gap: bool = False  # Revan: set by data
    terminate: bool = False  # Revan: IDK where these will come from yet
    truncate: bool = False  # Revan: IDK where these will come from yet
    temporal_state: TemporalState = field(default_factory=dict)
    transitions: list[NewTransition] | None = None

    def __post_init__(self):
        missing_info = pd.DataFrame(index=self.data.index, dtype=SparseMissingType)
        N = len(self.data)
        # initialize filled with NULL (no memory cost)
        null_cols = {col: [MissingType.NULL] * N for col in self.data.columns}
        self.missing_info = missing_info.assign(**null_cols)

    def get_last_timestamp(self) -> None | datetime.datetime:
        if not len(self.data.index):
            return None

        last_index = self.data.index[-1]
        match last_index:  # matches on type
            case datetime.datetime():
                return last_index
            case None:
                return None
            case _:
                raise ValueError("Indices should datetime.datetime or None")

    def get_first_timestamp(self) -> None | datetime.datetime:
        if not len(self.data.index):
            return None

        first_index = self.data.index[0]
        match first_index:  # matches on type
            case datetime.datetime():
                return first_index
            case None:
                return None
            case _:
                raise ValueError("Indices should datetime.datetime or None")
