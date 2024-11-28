import warnings
import datetime

from collections.abc import Mapping
from typing import Callable
from pandas import DataFrame

from corerl.config import MainConfig
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.bound_checker import bound_checker_builder
from corerl.data_pipeline.outlier_detectors.factory import init_outlier_detector
from corerl.data_pipeline.imputers.factory import init_imputer
from corerl.data_pipeline.transition_creators.factory import init_transition_creator
from corerl.data_pipeline.state_constructors.factory import init_state_constructor

from corerl.data_pipeline.datatypes import Transition, PipelineFrame, CallerCode, TemporalState
from corerl.data_pipeline.pipeline_utils import warmup_pruning, handle_data_gaps

WARMUP = 0

type PipelineStage[T] = Callable[[T, str], T]
type WarmupPruner = Callable[[PipelineFrame, int], PipelineFrame]


def invoke_stage_per_tag[T](carry: T, stage: Mapping[str, PipelineStage[T]]):
    for tag, f in stage.items():
        carry = f(carry, tag)

    return carry


class Pipeline:
    def __init__(self, main_cfg: MainConfig):
        self.tags = main_cfg.tags
        self.missing_data = {
            cfg.name: missing_data_checker for cfg in self.tags
        }

        self.bound_checker = {
            cfg.name: bound_checker_builder(cfg) for cfg in self.tags
        }

        self.transition_creator = init_transition_creator(main_cfg.agent_transition_creator)

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
        self.dt_dict: dict = {caller_code: None for caller_code in CallerCode}

        self.valid_thresh: datetime.timedelta = datetime.timedelta(10)  # TODO: this comes from somewhere

    def _init_temporal_state(self, pf: PipelineFrame, caller_code: CallerCode, reset_ts: bool = False):
        ts = self.ts_dict[caller_code]
        if ts is None or reset_ts:
            return {}

        pf_first_time_stamp = pf.get_first_timestamp()
        if pf_first_time_stamp - self.dt_dict[caller_code] > self.valid_thresh:
            warnings.warn(
                "The temporal state is invalid. "
                f"The temporal state has timestamp {self.dt_dict[caller_code]} "
                f"while the current pipeframe has initial timestamp {pf_first_time_stamp}",
                stacklevel=2,
            )

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
        transitions: list[Transition] = []
        for gapless_pf in pfs:
            gapless_pf = self.transition_creator(gapless_pf)
            gapless_pf = invoke_stage_per_tag(gapless_pf, self.state_constructor)
            gapless_pf = self.warmup_pruning(gapless_pf, WARMUP)
            assert gapless_pf.transitions is not None
            transitions += gapless_pf.transitions

        self.dt_dict[caller_code] = pf.get_last_timestamp()
        self.ts_dict[caller_code] = pf.temporal_state

        return transitions
