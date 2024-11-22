import warnings
import datetime

from collections.abc import Mapping
from typing import Callable, Dict, Self
from pandas import DataFrame
from enum import Enum, auto
from dataclasses import dataclass, field

from corerl.config import MainConfig
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.bound_checker import bound_checker_builder
from corerl.data_pipeline.outlier_detectors.factory import init_outlier_detector
from corerl.data_pipeline.imputers.factory import init_imputer
from corerl.data_pipeline.transition_creators.factory import init_transition_creator
from corerl.data_pipeline.state_constructors.factory import init_state_constructor

from corerl.data_pipeline.datatypes import Transition, MissingType, SparseMissingType
from corerl.data_pipeline.pipeline_utils import warmup_pruning, handle_data_gaps

WARMUP = 0

type PipelineStage[T] = Callable[[T, str], T]
type WarmupPruner = Callable[[PipelineFrame, int], PipelineFrame]


class CallerCode(Enum):
    OFFLINE = auto()
    ONLINE = auto
    REFRESH = auto()


class StageCode(Enum):
    TC = auto()
    SC = auto()


@dataclass
class StageTemporalState:
    pass


type TagDict = Dict[str, StageTemporalState]
type StageDict = Dict[StageCode, StageTemporalState | TagDict]


def _get_tag_ts(tag_dict: TagDict, tag: str | None) -> StageTemporalState | None:
    if tag is None:
        raise AssertionError("You are accessing the temporal state for a stage that has "
                             "different temporal states per tag without specifying a tag.")
    elif tag not in tag_dict:
        return None
    else:
        return tag_dict[tag]


class TemporalState:
    def __init__(self):
        self._stage_dict: StageDict = {}
        self.time_stamp: datetime.datetime | None = None

    def get_ts(self, stage_code: StageCode, tag: str | None = None) -> StageTemporalState | None:
        if stage_code not in self._stage_dict:
            return None
        else:
            stage_val: TagDict | StageTemporalState = self._stage_dict[stage_code]
            if type(stage_val) is TagDict:
                return _get_tag_ts(stage_val, tag)
            elif type(stage_val) is StageTemporalState:
                return stage_val
            else:
                raise AssertionError("Return type is invalid.")

    def update(self, ts: StageTemporalState, stage_code: StageCode, tag: str | None = None) -> Self:
        if tag is None:
            self._stage_dict[stage_code] = ts
        else:
            self._stage_dict[stage_code][tag] = ts
        return self


@dataclass
class PipelineFrame:
    data: DataFrame
    missing_info: DataFrame = field(init=False)
    data_gap: bool = False  # Revan: set by data
    terminate: bool = False  # Revan: IDK where these will come from yet
    truncate: bool = False  # Revan: IDK where these will come from yet
    temporal_state: TemporalState | None = None
    transitions: list[Transition] | None = None

    def __post_init__(self):
        missing_info = DataFrame(index=self.data.index, dtype=SparseMissingType)
        N = len(self.data)
        # initialize filled with NULL (no memory cost)
        null_cols = {col: [MissingType.NULL] * N for col in self.data.columns}
        self.missing_info = missing_info.assign(**null_cols)

    def get_last_timestamp(self) -> None | datetime.datetime:
        if not len(self.data.index):
            return None
        else:
            return self.data.index[-1]

    def get_first_timestamp(self) -> None | datetime.datetime:
        if not len(self.data.index):
            return None
        else:
            return self.data.index[0]


def invoke_stage_per_tag[T](carry: T, stage: Mapping[str, PipelineStage[T]]):
    for tag, f in stage.items():
        carry = f(carry, tag)

    return carry


class Pipeline:
    def __init__(self, cfg: MainConfig):
        self.tags = cfg.tags
        self.missing_data = {
            cfg.name: missing_data_checker for cfg in self.tags
        }

        self.bound_checker = {
            cfg.name: bound_checker_builder(cfg) for cfg in cfg.tags
        }

        self.transition_creator = init_transition_creator(cfg.agent_transition_creator)

        self.outlier_detector = {
            cfg.name: init_outlier_detector(cfg.outlier) for cfg in self.tags
        }

        self.imputer = {
            cfg.name: init_imputer(cfg.imputer) for cfg in self.tags
        }

        self.state_constructor = {
            cfg.name: init_state_constructor(cfg.state_constructor) for cfg in self.tags
        }

        self.warmup_pruning: WarmupPruner = warmup_pruning
        self.ts_dict: dict = {caller_code: None for caller_code in CallerCode}

        self.valid_thresh: datetime.timedelta = datetime.timedelta(10)  # TODO: this comes from somewhere

    def _init_temporal_state(self, pf: PipelineFrame, caller_code: CallerCode, reset_ts: bool = False):
        ts = self.ts_dict[caller_code]
        if ts is None or reset_ts:
            ts = TemporalState()
        else:
            pf_first_time_stamp = pf.get_first_timestamp()
            if pf_first_time_stamp - ts.timestamp > self.valid_thresh:
                warnings.warn("The temporal state is invalid. "
                              f"The temporal state has timestamp {ts.timestamp} "
                              f"while the current pipeframe has initial timestamp {pf_first_time_stamp}")

        return ts

    def _save_ts(self, ts: TemporalState, caller_code: CallerCode):
        self.ts_dict[caller_code] = ts

    def __call__(self, data: DataFrame,
                 caller_code: CallerCode,
                 reset_temporal_state: bool = False) -> list[Transition]:
        pf = PipelineFrame(data)
        ts = self._init_temporal_state(pf, caller_code, reset_temporal_state)
        pf.temporal_state = ts
        pf = invoke_stage_per_tag(pf, self.missing_data)
        pf = invoke_stage_per_tag(pf, self.bound_checker)
        pf = invoke_stage_per_tag(pf, self.outlier_detector)
        pf = invoke_stage_per_tag(pf, self.imputer)
        pfs = handle_data_gaps(pf)
        transitions = []
        for pf in pfs:
            pf_with_transitions = self.transition_creator(pf)
            pf_with_transitions = invoke_stage_per_tag(pf_with_transitions, self.state_constructor)
            pf_with_transitions = self.warmup_pruning(pf_with_transitions,
                WARMUP)
            transitions += pf_with_transitions.transitions

        pf.temporal_state.time_stamp = pf.get_last_timestamp()
        self._save_ts(pf.ts, caller_code)
        return transitions
