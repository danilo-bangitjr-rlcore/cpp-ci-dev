import datetime
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import Enum, IntFlag, auto
from math import isclose

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from corerl.utils.torch import tensor_allclose

type TagName = str  # alias to clarify semantics of PipelineStage and stage dict
type PipelineStage[T] = Callable[[T, TagName], T]

class MissingType(IntFlag):
    NULL = auto()     # data is not missing
    MISSING = auto()  # indicates data did not exist in db
    FILTER = auto()   # filtered by conditional filter stage
    BOUNDS = auto()
    OUTLIER = auto()


# for use to create sparse pandas dataframes
# for example: sparse_df = pd.DataFrame(..., dtype=SparseMissingType)
SparseMissingType = pd.SparseDtype(dtype=int, fill_value=MissingType.NULL)

@dataclass
class Step:
    """
    Dataclass for storing the information of a single step.
    Two of these make up a transition.
    """
    reward: float
    action: Tensor
    gamma: float
    state: Tensor
    dp: bool
    timestamp: datetime.datetime | None = None

    def __eq__(self, other: object):
        if not isinstance(other, Step):
            return False

        return (
                isclose(self.gamma, other.gamma)
                and isclose(self.reward, other.reward)
                and tensor_allclose(self.action, other.action)
                and tensor_allclose(self.state, other.state)
                and self.dp == other.dp
        )

    def __str__(self):
        return '\n'.join(
            f'{f.name}: {getattr(self, f.name)}'
            for f in fields(self)
        )

    def __iter__(self):
        """
        This iterator is used in the buffer with magic ordering
        """
        for f in fields(self):
            attr = getattr(self, f.name)
            # skip timestamp in buffer
            if f.name == "timestamp":
                continue
            yield attr


@dataclass
class Transition:
    steps: list[Step]
    n_step_reward: float
    n_step_gamma: float

    @property
    def prior(self):
        return self.steps[0]

    @property
    def post(self):
        return self.steps[-1]

    @property
    def n_steps(self):
        return len(self.steps) - 1

    def __eq__(self, other: object):
        if not isinstance(other, Transition):
            return False

        if len(self.steps) != len(other.steps):
            return False

        if self.n_steps != other.n_steps:
            return False

        if self.n_step_gamma != other.n_step_gamma:
            return False

        for i in range(len(self.steps)):
            if self.steps[i] != other.steps[i]:
                return False

        return True

    def __iter__(self):
        """
        This iterator is used in the buffer with magic ordering
        """
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, list):  # if attr = steps
                prior = attr[0]
                post = attr[-1]
                yield from iter(prior)
                yield from iter(post)
            else:
                yield attr

    def __len__(self) -> int:
        return len(self.steps)-1


@dataclass
class StepBatch:
    """
    Step attributes aggregated into Tensors for the replay buffer
    """
    reward: Tensor
    action: Tensor
    gamma: Tensor
    state: Tensor
    dp: Tensor

    def __iter__(self):
        """
        This iterator is used in the buffer with magic ordering
        """
        for f in fields(self):
            yield getattr(self, f.name)

    def __eq__(self, other: object):
        if not isinstance(other, StepBatch):
            return False

        return (
                torch.equal(self.reward, other.reward)
                and torch.equal(self.action, other.action)
                and torch.equal(self.gamma, other.gamma)
                and torch.equal(self.state, other.state)
                and torch.equal(self.dp, other.dp)
        )

    def __getitem__(self, idx: int|slice) -> "StepBatch":
        if isinstance(idx, int):
            idx = slice(idx, idx+1)

        return StepBatch(
            reward=self.reward[idx],
            action=self.action[idx],
            gamma=self.gamma[idx],
            state=self.state[idx],
            dp=self.dp[idx]
        )

@dataclass
class TransitionBatch:
    idxs: np.ndarray
    prior: StepBatch
    post: StepBatch
    n_step_reward: Tensor
    n_step_gamma: Tensor

    def __eq__(self, other: object):
        if not isinstance(other, TransitionBatch):
            return False

        return (
                self.prior == other.prior
                and self.post == other.post
                and torch.equal(self.n_step_reward, other.n_step_reward)
                and torch.equal(self.n_step_gamma, other.n_step_gamma)
        )

    def __getitem__(self, idx: int|slice) -> "TransitionBatch":
        if isinstance(idx, (int, np.integer)):
            idx = slice(idx, idx+1)

        return TransitionBatch(
            idxs=self.idxs[idx],
            prior=self.prior[idx],
            post=self.post[idx],
            n_step_reward=self.n_step_reward[idx],
            n_step_gamma=self.n_step_gamma[idx]
        )

class DataMode(Enum):
    OFFLINE = auto()
    ONLINE = auto()
    REFRESH = auto()


class StageCode(Enum):
    INIT = auto()
    FILTER = auto()
    BOUNDS = auto()
    PREPROCESS = auto()
    IMPUTER = auto()
    ODDITY = auto()
    TC = auto()
    AC = auto()
    SC = auto()
    RC = auto()
    TF = auto()


type TemporalState = dict[StageCode, object | None]


@dataclass
class PipelineFrame:
    data: pd.DataFrame
    data_mode: DataMode
    states: pd.DataFrame = field(init=False)
    actions: pd.DataFrame = field(init=False)
    rewards: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_info: pd.DataFrame = field(init=False)
    decision_points: np.ndarray = field(init=False)
    action_change: np.ndarray = field(init=False)
    temporal_state: TemporalState = field(default_factory=dict)
    transitions: list[Transition] | None = None

    def __post_init__(self):
        missing_info = pd.DataFrame(index=self.data.index, dtype=SparseMissingType)
        N = len(self.data)
        # initialize filled with NULL (no memory cost)
        null_cols = {col: [MissingType.NULL] * N for col in self.data.columns}
        self.missing_info = missing_info.assign(**null_cols)

        # initialize dp flags
        self.decision_points = np.zeros(N, dtype=np.bool_)
        # initialize action change flags
        self.action_change = np.zeros(N, dtype=np.bool_)

        # initialize rl containers
        self.actions = self.data.copy(deep=False)

    def get_last_timestamp(self) -> datetime.datetime:

        last_index = self.data.index[-1]
        assert isinstance(last_index, datetime.datetime)
        return last_index

    def get_first_timestamp(self) -> datetime.datetime:

        first_index = self.data.index[0]
        assert isinstance(first_index, datetime.datetime)
        return first_index
