from dataclasses import dataclass, field
import warnings
import datetime

from collections.abc import Mapping, Sequence
from typing import Any, Callable
from omegaconf import MISSING
from pandas import DataFrame
import logging

from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.bound_checker import bound_checker_builder
from corerl.data_pipeline.oddity_filters.factory import init_oddity_filter
from corerl.data_pipeline.imputers.factory import init_imputer
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transition_creators.dummy import DummyTransitionCreatorConfig
from corerl.data_pipeline.transition_creators.factory import init_transition_creator
from corerl.data_pipeline.state_constructors.sc import StateConstructor

from corerl.data_pipeline.datatypes import NewTransition, PipelineFrame, CallerCode, StageCode, TemporalState

logger = logging.getLogger(__name__)

WARMUP = 0

type TagName = str  # alias to clarify semantics of PipelineStage and stage dict
type PipelineStage[T] = Callable[[T, TagName], T]


@dataclass
class PipelineConfig:
    tags: list[TagConfig] = MISSING

    obs_interval_minutes: float = MISSING
    agent_transition_creator: Any = field(default_factory=DummyTransitionCreatorConfig)


def invoke_stage_per_tag[T](carry: T, stage: Mapping[TagName, PipelineStage[T]]) -> T:
    for tag, f in stage.items():
        carry = f(carry, tag)

    return carry


@dataclass
class PipelineReturn:
    df: DataFrame
    transitions: list[NewTransition] | None


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.tags = cfg.tags
        self.missing_data_checkers = {
            tag.name: missing_data_checker for tag in self.tags
        }

        self.bound_checkers = {
            tag.name: bound_checker_builder(tag) for tag in self.tags
        }

        self.transition_creator = init_transition_creator(
            cfg.agent_transition_creator,
            self.tags,
        )

        self.outlier_detectors = {
            tag.name: init_oddity_filter(tag.outlier) for tag in self.tags
        }

        self.imputers = {
            tag.name: init_imputer(tag.imputer, tag) for tag in self.tags
        }

        self.state_constructors = {
            tag.name: StateConstructor(tag.state_constructor)
            for tag in self.tags
            if not tag.is_action
        }

        self.ts_dict: dict = {caller_code: None for caller_code in CallerCode}
        self.dt_dict: dict = {caller_code: None for caller_code in CallerCode}

        self.valid_thresh: datetime.timedelta = datetime.timedelta(minutes=cfg.obs_interval_minutes)

        self._stage_invokers: dict[StageCode, Callable[[PipelineFrame], PipelineFrame]] = {
            StageCode.BOUNDS:  lambda pf: invoke_stage_per_tag(pf, self.bound_checkers),
            StageCode.ODDITY:  lambda pf: invoke_stage_per_tag(pf, self.outlier_detectors),
            StageCode.IMPUTER: lambda pf: invoke_stage_per_tag(pf, self.imputers),
            StageCode.SC:      lambda pf: invoke_stage_per_tag(pf, self.state_constructors),
            StageCode.TC:      self.transition_creator,
        }

    def _init_temporal_state(self, pf: PipelineFrame, reset_ts: bool = False):
        ts = self.ts_dict[pf.caller_code]
        if ts is None or reset_ts:
            return {}

        pf_first_time_stamp = pf.get_first_timestamp()
        if pf_first_time_stamp - self.dt_dict[pf.caller_code] > self.valid_thresh:
            warnings.warn(
                "The temporal state is invalid. "
                f"The temporal state has timestamp {self.dt_dict[pf.caller_code]} "
                f"while the current pipeframe has initial timestamp {pf_first_time_stamp}",
                stacklevel=2,
            )

        return ts

    def _save_ts(self, ts: TemporalState, caller_code: CallerCode):
        self.ts_dict[caller_code] = ts

    def __call__(
            self, data: DataFrame,
            caller_code: CallerCode,
            reset_temporal_state: bool = False,
            stages: Sequence[StageCode] | None = None,
    ) -> PipelineReturn:
        if stages is None:
            stages = (StageCode.BOUNDS, StageCode.ODDITY, StageCode.IMPUTER, StageCode.SC, StageCode.TC)

        pf = PipelineFrame(data, caller_code)
        ts = self._init_temporal_state(pf, reset_temporal_state)
        pf.temporal_state = ts

        for stage in stages:
            pf = self._stage_invokers[stage](pf)

        self.dt_dict[caller_code] = pf.get_last_timestamp()
        self.ts_dict[caller_code] = pf.temporal_state

        return PipelineReturn(
            df=pf.data,
            transitions=pf.transitions,
        )

    def get_state_action_dims(self):
        num_actions = sum(
            tag.is_action for tag in self.tags
        )

        state_dim = sum(
            self.state_constructors[tag.name].state_dim(tag.name)
            for tag in self.tags
            if not tag.is_action
        )

        return state_dim, num_actions
