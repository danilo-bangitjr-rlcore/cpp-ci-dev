import datetime
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd
from lib_agent.buffer.datatypes import DataMode, Trajectory

type TagName = str  # alias to clarify semantics of PipelineStage and stage dict
type PipelineStage[T] = Callable[[T, TagName], T]


class StageCode(Enum):
    INIT = auto()
    VIRTUAL = auto()
    TRIGGER = auto()
    BOUNDS = auto()
    PREPROCESS = auto()
    IMPUTER = auto()
    ODDITY = auto()
    TC = auto()
    AC = auto()
    SC = auto()
    RC = auto()
    TF = auto()
    ZONES = auto()


type TemporalState = dict[StageCode, object | None]


@dataclass
class PipelineFrame:
    data: pd.DataFrame
    data_mode: DataMode
    last_stage : StageCode | None = None
    states: pd.DataFrame = field(init=False)
    actions: pd.DataFrame = field(init=False)
    action_lo: pd.DataFrame = field(init=False)
    action_hi: pd.DataFrame = field(init=False)
    rewards: pd.DataFrame = field(default_factory=pd.DataFrame)
    decision_points: np.ndarray = field(init=False)
    action_change: np.ndarray = field(init=False)
    temporal_state: TemporalState = field(default_factory=dict)
    trajectories: list[Trajectory] | None = None

    def __post_init__(self):
        N = len(self.data)

        # initialize dp flags
        self.decision_points = np.zeros(N, dtype=np.bool_)
        # initialize action change flags
        self.action_change = np.zeros(N, dtype=np.bool_)

        # initialize rl containers
        self.states = self.data.copy(deep=False)
        self.actions = self.data.copy(deep=False)

    def get_last_timestamp(self) -> datetime.datetime:

        last_index = self.data.index[-1]
        assert isinstance(last_index, datetime.datetime)
        return last_index

    def get_first_timestamp(self) -> datetime.datetime:

        first_index = self.data.index[0]
        assert isinstance(first_index, datetime.datetime)
        return first_index
