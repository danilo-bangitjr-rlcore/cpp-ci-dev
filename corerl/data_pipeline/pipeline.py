from dataclasses import dataclass
import warnings
import datetime

from collections.abc import Mapping
from typing import Any, Callable
from omegaconf import MISSING
from pandas import DataFrame

from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.bound_checker import bound_checker_builder
from corerl.data_pipeline.outlier_detectors.factory import init_outlier_detector
from corerl.data_pipeline.imputers.factory import init_imputer
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transition_creators.factory import init_transition_creator
from corerl.data_pipeline.state_constructors.sc import StateConstructor

from corerl.data_pipeline.datatypes import Transition, PipelineFrame, CallerCode, TemporalState
from corerl.data_pipeline.pipeline_utils import warmup_pruning, handle_data_gaps
from corerl.utils.hydra import interpolate

WARMUP = 0

type TagName = str # alias to clarify semantics of PipelineStage and stage dict
type PipelineStage[T] = Callable[[T, TagName], T]
type WarmupPruner = Callable[[PipelineFrame, int], PipelineFrame]


@dataclass
class PipelineConfig:
    tags: list[TagConfig] = MISSING

    state_constructor: Any = MISSING
    agent_transition_creator: Any = interpolate('{agent_transition_creator}')


def invoke_stage_per_tag[T](carry: T, stage: Mapping[TagName, PipelineStage[T]]) -> T:
    for tag, f in stage.items():
        carry = f(carry, tag)

    return carry


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.tags = cfg.tags
        self.missing_data_checkers = {
            cfg.name: missing_data_checker for cfg in self.tags
        }

        self.bound_checkers = {
            cfg.name: bound_checker_builder(cfg) for cfg in self.tags
        }

        self.transition_creator = init_transition_creator(cfg.agent_transition_creator)

        self.outlier_detectors = {
            cfg.name: init_outlier_detector(cfg.outlier) for cfg in self.tags
        }

        self.imputers = {
            cfg.name: init_imputer(cfg.imputer, cfg) for cfg in self.tags
        }

        self.state_constructors = {
            cfg.name: StateConstructor(cfg.state_constructor) for cfg in self.tags
        }

        self.warmup_pruning: WarmupPruner = warmup_pruning
        self.ts_dict: dict = {caller_code: None for caller_code in CallerCode}
        self.dt_dict: dict = {caller_code: None for caller_code in CallerCode}

        self.valid_thresh: datetime.timedelta = datetime.timedelta(10)  # TODO: this comes from somewhere

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

    def __call__(self, data: DataFrame,
                 caller_code: CallerCode,
                 reset_temporal_state: bool = False) -> list[Transition]:

        pf = PipelineFrame(data, caller_code)
        ts = self._init_temporal_state(pf, reset_temporal_state)
        pf.temporal_state = ts
        pf = invoke_stage_per_tag(pf, self.missing_data_checkers)
        pf = invoke_stage_per_tag(pf, self.bound_checkers)
        pf = invoke_stage_per_tag(pf, self.outlier_detectors)
        pf = invoke_stage_per_tag(pf, self.imputers)
        pfs = handle_data_gaps(pf)
        transitions: list[Transition] = []
        for gapless_pf in pfs:
            gapless_pf = self.transition_creator(gapless_pf)
            gapless_pf = invoke_stage_per_tag(gapless_pf, self.state_constructors)
            gapless_pf = self.warmup_pruning(gapless_pf, WARMUP)
            assert gapless_pf.transitions is not None
            transitions += gapless_pf.transitions

        self.dt_dict[caller_code] = pf.get_last_timestamp()
        self.ts_dict[caller_code] = pf.temporal_state

        return transitions
