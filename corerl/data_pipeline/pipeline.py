from __future__ import annotations

import datetime
import logging
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Self, Tuple

import pandas as pd
from pandas import DataFrame
from pydantic import Field

from corerl.configs.config import MISSING, computed, config, list_, sanitizer
from corerl.data_pipeline.all_the_time import AllTheTimeTC, AllTheTimeTCConfig
from corerl.data_pipeline.bound_checker import bound_checker_builder
from corerl.data_pipeline.constructors.ac import ActionConstructor
from corerl.data_pipeline.constructors.conditional_filter import ConditionalFilter
from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.constructors.rc import RewardConstructor
from corerl.data_pipeline.constructors.sc import SCConfig, StateConstructor, construct_default_sc_configs
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame, StageCode, TemporalState, Transition
from corerl.data_pipeline.imputers.factory import ImputerStageConfig, init_imputer
from corerl.data_pipeline.imputers.imputer_stage import PerTagImputerConfig
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.oddity_filters.config import GlobalOddityFilterConfig
from corerl.data_pipeline.oddity_filters.factory import init_oddity_filter
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import register_dispatchers
from corerl.data_pipeline.transition_filter import TransitionFilter, TransitionFilterConfig
from corerl.data_pipeline.utils import invoke_stage_per_tag
from corerl.data_pipeline.zones import default_configs_from_zones

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)
register_dispatchers()


@config()
class PipelineConfig:
    tags: list[TagConfig] = list_()
    max_data_gap: timedelta = MISSING

    # stage-wide configs
    imputer: ImputerStageConfig = Field(default_factory=PerTagImputerConfig)
    oddity_filter: GlobalOddityFilterConfig = Field(default_factory=GlobalOddityFilterConfig)
    state_constructor: SCConfig = Field(default_factory=SCConfig)
    transition_creator: AllTheTimeTCConfig = Field(default_factory=AllTheTimeTCConfig)
    transition_filter: TransitionFilterConfig = Field(default_factory=TransitionFilterConfig)


    @sanitizer
    def _default_imputers(self, cfg: MainConfig):
        if not isinstance(self.imputer, PerTagImputerConfig):
            return

        for tag in self.tags:
            if tag.imputer is not None:
                continue

            tag.imputer = self.imputer.default

    @sanitizer
    def _default_oddity_filter(self, cfg: MainConfig):
        for tag in self.tags:
            if tag.outlier is not None:
                continue

            tag.outlier = self.oddity_filter.default


    @computed('max_data_gap')
    @classmethod
    def _max_data_gap(cls, cfg: MainConfig):
        return 2 * cfg.interaction.obs_period

@dataclass
class PipelineReturn:
    data_mode: DataMode
    df: DataFrame
    states: DataFrame
    actions: DataFrame
    rewards: DataFrame
    transitions: list[Transition] | None

    def _add(self, other: Self) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, list[Transition] | None]:
        assert self.data_mode == other.data_mode, "PipelineReturn objects must have the same DataMode to be added"

        df = pd.concat([self.df, other.df])
        states = pd.concat([self.states, other.states])
        actions = pd.concat([self.actions, other.actions])
        rewards = pd.concat([self.rewards, other.rewards])

        transitions = []
        if self.transitions is None:
            transitions = other.transitions
        elif other.transitions is not None:
            transitions = self.transitions + other.transitions

        return df, states, actions, rewards, transitions

    def __iadd__(self, other: Self):
        df, states, actions, rewards, transitions = self._add(other)

        self.df = df
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.transitions = transitions

        return self

    def __add__(self, other: Self):
        df, states, actions, rewards, transitions = self._add(other)

        return PipelineReturn(self.data_mode, df, states, actions, rewards, transitions)


@dataclass
class ColumnDescriptions:
    state_cols: list[str]
    action_cols: list[str]

    @property
    def state_dim(self):
        return len(self.state_cols)

    @property
    def action_dim(self):
        return len(self.action_cols)


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        # sanity checking
        cfg = self._construct_config(cfg)
        self.cfg = cfg
        self.tags = cfg.tags

        # initialization all stateful stages
        self.missing_data_checkers = {tag.name: missing_data_checker for tag in self.tags}
        self.bound_checkers = {
            tag.name: bound_checker_builder(tag.operating_range)
            for tag in self.tags
            if tag.operating_range is not None
        }
        self.conditional_filter = ConditionalFilter(self.tags)
        self.preprocessor = Preprocessor(self.tags)
        self.transition_creator = AllTheTimeTC(cfg.transition_creator, self.tags)
        self.transition_filter = TransitionFilter(cfg.transition_filter)
        self.outlier_detectors = {tag.name: init_oddity_filter(tag.outlier) for tag in self.tags}
        self.imputers = init_imputer(cfg.imputer, self.tags)
        self.action_constructor = ActionConstructor(self.tags)
        self.state_constructor = StateConstructor(self.tags, cfg.state_constructor)
        self.reward_constructor = RewardConstructor(self.tags, self.preprocessor)

        # build pipeline state
        self.ts_dict: dict[DataMode, TemporalState | None] = {data_mode: None for data_mode in DataMode}
        self.dt_dict: dict[DataMode, datetime.datetime | None] = {data_mode: None for data_mode in DataMode}

        self._pre_invoke_hooks: dict[DataMode, dict[StageCode, list[Callable[[PipelineFrame], Any]]]] = {
            data_mode: defaultdict(list) for data_mode in DataMode}

        self._post_invoke_hooks: dict[DataMode, dict[StageCode, list[Callable[[PipelineFrame], Any]]]] = {
            data_mode: defaultdict(list) for data_mode in DataMode}

        self._stage_invokers: dict[StageCode, Callable[[PipelineFrame], PipelineFrame]] = {
            StageCode.INIT:       lambda pf: pf,
            StageCode.FILTER:     self.conditional_filter,
            StageCode.BOUNDS:     lambda pf: invoke_stage_per_tag(pf, self.bound_checkers),
            StageCode.PREPROCESS: self.preprocessor,
            StageCode.ODDITY:     lambda pf: invoke_stage_per_tag(pf, self.outlier_detectors),
            StageCode.IMPUTER:    self.imputers,
            StageCode.AC:         self.action_constructor,
            StageCode.RC:         self.reward_constructor,
            StageCode.SC:         self.state_constructor,
            StageCode.TC:         self.transition_creator,
            StageCode.TF:         self.transition_filter,
        }

        self._default_stages = (
            StageCode.INIT,
            StageCode.FILTER,
            StageCode.BOUNDS,
            StageCode.PREPROCESS,
            StageCode.ODDITY,
            StageCode.IMPUTER,
            StageCode.AC,
            StageCode.RC,
            StageCode.SC,
            StageCode.TC,
            StageCode.TF
        )

    def _construct_config(self, cfg: PipelineConfig) -> PipelineConfig:
        construct_default_sc_configs(cfg.state_constructor, cfg.tags)
        default_configs_from_zones(cfg.tags)
        return cfg


    def _init_temporal_state(self, pf: PipelineFrame, reset_ts: bool = False):
        ts = self.ts_dict[pf.data_mode]
        if ts is None or reset_ts:
            return {}

        last_seen_time = self.dt_dict[pf.data_mode]
        if last_seen_time is None:
            return {}

        first_time = pf.get_first_timestamp()
        if first_time - last_seen_time > self.cfg.max_data_gap:
            warnings.warn(
                "The temporal state is invalid. "
                f"The temporal state has timestamp {last_seen_time} "
                f"while the current pipeframe has initial timestamp {first_time}",
                stacklevel=2,
            )

        return ts

    def __call__(
            self, data: DataFrame,
            data_mode: DataMode = DataMode.OFFLINE,
            reset_temporal_state: bool = False,
            stages: Sequence[StageCode] | None = None,
    ) -> PipelineReturn:
        if stages is None:
            stages = self._default_stages

        # handle the no data case with an empty return
        if data.empty:
            return PipelineReturn(
                data_mode=data_mode,
                df=data,
                states=data,
                actions=data,
                rewards=data,
                transitions=[],
            )

        # construct the internal carry object that is mutated
        # by each stage of the pipeline
        pf = PipelineFrame(data, data_mode)
        pf.temporal_state = self._init_temporal_state(pf, reset_temporal_state)

        pf = invoke_stage_per_tag(pf, self.missing_data_checkers)
        for stage in stages:
            for hook in self._pre_invoke_hooks[data_mode][stage]:
                hook(pf)

            pf = self._stage_invokers[stage](pf)

            for hook in self._post_invoke_hooks[data_mode][stage]:
                hook(pf)

        self.dt_dict[data_mode] = pf.get_last_timestamp()
        self.ts_dict[data_mode] = pf.temporal_state

        return PipelineReturn(
            data_mode=data_mode,
            df=pf.data,
            states=pf.states,
            actions=pf.actions,
            rewards=pf.rewards,
            transitions=pf.transitions,
        )


    @cached_property
    def column_descriptions(self):
        return ColumnDescriptions(
            state_cols=self.state_constructor.columns,
            action_cols=self.action_constructor.columns,
        )

    def register_hook(
            self,
            data_modes: DataMode | list[DataMode],
            stages: StageCode | list[StageCode],
            f: Callable[[PipelineFrame], Any],
            order: str = 'post',
        ):
        if isinstance(data_modes, DataMode):
            data_modes = [data_modes]
        if isinstance(stages, StageCode):
            stages = [stages]

        assert order == 'post' or order == 'pre'
        if order == 'post':
            hook_dict = self._post_invoke_hooks
        else:
            hook_dict = self._pre_invoke_hooks

        for data_mode in data_modes:
            for stage in stages:
               hook_dict[data_mode][stage].append(f)

    def reset(self):
        if hasattr(self.state_constructor, 'reset'):
            self.state_constructor.reset()
