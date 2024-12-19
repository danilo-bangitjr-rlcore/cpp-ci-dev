from collections import defaultdict
from dataclasses import dataclass, field
import warnings
import datetime

from collections.abc import Sequence
from typing import Any, Callable
from pandas import DataFrame
import logging

from corerl.configs.config import config, list_
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.bound_checker import bound_checker_builder
from corerl.data_pipeline.oddity_filters.factory import init_oddity_filter
from corerl.data_pipeline.imputers.factory import init_imputer
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.all_the_time import AllTheTimeTC, AllTheTimeTCConfig
from corerl.data_pipeline.state_constructors.sc import SCConfig, StateConstructor
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.data_pipeline.reward.rc import RewardComponentConstructor, RewardConstructor
from corerl.data_pipeline.transition_filter import TransitionFilter, TransitionFilterConfig
from corerl.data_pipeline.utils import invoke_stage_per_tag
from corerl.data_pipeline.datatypes import NewTransition, PipelineFrame, CallerCode, StageCode


logger = logging.getLogger(__name__)


@config()
class PipelineConfig:
    tags: list[TagConfig] = list_()
    db: TagDBConfig = field(default_factory=TagDBConfig)
    obs_interval_minutes: float = 0
    state_constructor: SCConfig = field(default_factory=SCConfig)
    transition_creator: AllTheTimeTCConfig = field(default_factory=AllTheTimeTCConfig)
    transition_filter: TransitionFilterConfig = field(default_factory=TransitionFilterConfig)


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
            tag.name: bound_checker_builder(tag.bounds) for tag in self.tags
        }

        self.transition_creator = AllTheTimeTC(
            cfg.transition_creator,
            self.tags,
        )

        self.transition_filter = TransitionFilter(cfg.transition_filter)

        self.outlier_detectors = {
            tag.name: init_oddity_filter(tag.outlier) for tag in self.tags
        }

        self.imputers = {
            tag.name: init_imputer(tag.imputer) for tag in self.tags
        }

        reward_components = {cfg.name: RewardComponentConstructor(cfg.reward_constructor) for cfg in self.tags}
        self.reward_constructor = RewardConstructor(reward_components)

        self.state_constructor = StateConstructor(self.tags, cfg.state_constructor)

        self.ts_dict: dict = {caller_code: None for caller_code in CallerCode}
        self.dt_dict: dict = {caller_code: None for caller_code in CallerCode}

        self.valid_thresh: datetime.timedelta = datetime.timedelta(minutes=cfg.obs_interval_minutes)

        self._hooks: dict[StageCode, list[Callable[[PipelineFrame], Any]]] = defaultdict(list)
        self._stage_invokers: dict[StageCode, Callable[[PipelineFrame], PipelineFrame]] = {
            StageCode.BOUNDS:  lambda pf: invoke_stage_per_tag(pf, self.bound_checkers),
            StageCode.ODDITY:  lambda pf: invoke_stage_per_tag(pf, self.outlier_detectors),
            StageCode.IMPUTER: lambda pf: invoke_stage_per_tag(pf, self.imputers),
            StageCode.RC:      self.reward_constructor,
            StageCode.SC:      self.state_constructor,
            StageCode.TC:      self.transition_creator,
            StageCode.TF:      self.transition_filter,
        }

        self._default_stages = (
            StageCode.BOUNDS,
            StageCode.ODDITY,
            StageCode.IMPUTER,
            StageCode.RC,
            StageCode.SC,
            StageCode.TC,
            StageCode.TF
        )

    def _init_temporal_state(self, pf: PipelineFrame, reset_ts: bool = False):
        ts = self.ts_dict[pf.caller_code]
        if ts is None or reset_ts:
            return {}

        first_time = pf.get_first_timestamp()
        if first_time - self.dt_dict[pf.caller_code] > self.valid_thresh:
            warnings.warn(
                "The temporal state is invalid. "
                f"The temporal state has timestamp {self.dt_dict[pf.caller_code]} "
                f"while the current pipeframe has initial timestamp {first_time}",
                stacklevel=2,
            )

        return ts

    def __call__(
            self, data: DataFrame,
            caller_code: CallerCode = CallerCode.OFFLINE,
            reset_temporal_state: bool = False,
            stages: Sequence[StageCode] | None = None,
    ) -> PipelineReturn:
        if stages is None:
            stages = self._default_stages

        # handle the no data case with an empty return
        if data.empty:
            return PipelineReturn(
                df=data,
                transitions=[],
            )

        # construct the internal carry object that is mutated
        # by each stage of the pipeline
        pf = PipelineFrame(data, caller_code)
        pf.temporal_state = self._init_temporal_state(pf, reset_temporal_state)

        pf = invoke_stage_per_tag(pf, self.missing_data_checkers)
        for stage in stages:
            pf = self._stage_invokers[stage](pf)

            for hook in self._hooks[stage]:
                hook(pf)

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

        state_dim = self.state_constructor.state_dim()
        return state_dim, num_actions

    def register_hook(self, stage: StageCode, f: Callable[[PipelineFrame], Any]):
        self._hooks[stage].append(f)
