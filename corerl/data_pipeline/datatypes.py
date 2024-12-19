import torch
import datetime
import numpy as np
import pandas as pd

from collections.abc import Callable
from dataclasses import dataclass, fields, field
from enum import IntFlag, auto, Enum
from torch import Tensor
from math import isclose

from corerl.utils.torch import tensor_allclose

type TagName = str  # alias to clarify semantics of PipelineStage and stage dict
type PipelineStage[T] = Callable[[T, TagName], T]
type WarmupPruner = Callable[[PipelineFrame, int], PipelineFrame]

class MissingType(IntFlag):
    NULL = auto()
    MISSING = auto()  # indicates data did not exist in db
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
        string = ''
        for f in fields(self):
            string += f"{f.name}: {getattr(self, f.name)}\n"
        return string

    def __iter__(self):
        for f in fields(self):
            yield getattr(self, f.name)


@dataclass
class NewTransition:
    prior: Step
    post: Step
    n_steps: int

    def __eq__(self, other: object):
        if not isinstance(other, NewTransition):
            return False

        return (
                self.prior == other.prior
                and self.post == other.post
                and self.n_steps == other.n_steps
        )

    def __iter__(self):
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Step):
                yield from iter(attr)
            else:
                yield attr


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

    def num_attrs(self) -> int:
        return len(fields(self))


@dataclass
class NewTransitionBatch:
    prior: StepBatch
    post: StepBatch
    n_steps: Tensor

    def __post_init__(self):
        # ensure all the attributes have the same dimension
        state_batch_size = self.prior.state.size(0)
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, StepBatch):
                for sub_field in iter(attr):
                    assert sub_field.size(0) == state_batch_size, \
                        f"Element {sub_field.name} does not have the same batch size as the state"
            else:
                assert attr.size(0) == state_batch_size, \
                    f"Element {f.name} does not have the same batch size as the state"

    def __eq__(self, other: object):
        if not isinstance(other, NewTransitionBatch):
            return False

        return (
                self.prior == other.prior
                and self.post == other.post
                and torch.equal(self.n_steps, other.n_steps)
        )

    @property
    def batch_size(self) -> int:
        return self.n_steps.size(0)


class CallerCode(Enum):
    OFFLINE = auto()
    ONLINE = auto()
    REFRESH = auto()


class StageCode(Enum):
    BOUNDS = auto()
    IMPUTER = auto()
    ODDITY = auto()
    TC = auto()
    SC = auto()
    RC = auto()


type TemporalState = dict[StageCode, object | None]


@dataclass
class PipelineFrame:
    data: pd.DataFrame
    caller_code: CallerCode
    missing_info: pd.DataFrame = field(init=False)
    decision_points: np.ndarray = field(init=False)
    temporal_state: TemporalState = field(default_factory=dict)
    transitions: list[NewTransition] | None = None

    def __post_init__(self):
        missing_info = pd.DataFrame(index=self.data.index, dtype=SparseMissingType)
        N = len(self.data)
        # initialize filled with NULL (no memory cost)
        null_cols = {col: [MissingType.NULL] * N for col in self.data.columns}
        self.missing_info = missing_info.assign(**null_cols)

        # initialize dp flags
        self.decision_points = np.zeros(N, dtype=np.bool_)

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
