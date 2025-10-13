from collections.abc import Iterable, Sequence
from functools import cached_property

import lib_utils.list as list_u
import numpy as np
import pandas as pd
from lib_defs.config_defs.tag_config import TagType
from lib_utils.maybe import Maybe

from corerl.configs.data_pipeline.constructors.sc import SCConfig
from corerl.configs.tags.tag_config import TagConfig
from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode, TemporalState
from corerl.data_pipeline.state_constructors.countdown import DecisionPointDetector
from corerl.data_pipeline.state_constructors.seasonal import SeasonalTagFeatures
from corerl.data_pipeline.transforms.base import Transform
from corerl.data_pipeline.transforms.trace import TraceConstructor, TraceTemporalState, log_trace_quality
from corerl.state import AppState


class StateConstructor(Constructor):
    def __init__(self, app_state: AppState, tag_cfgs: Sequence[TagConfig], cfg: SCConfig):
        self._cfg = cfg
        self._app_state = app_state
        super().__init__(tag_cfgs)

        self._cd_adder = DecisionPointDetector(cfg.countdown) if cfg.countdown is not None else None
        self._seasonal_features = SeasonalTagFeatures(tag_cfgs)


    def _get_relevant_configs(self, tag_cfgs: Sequence[TagConfig]):
        return {
            tag.name: tag.state_constructor if tag.state_constructor is not None else self._cfg.defaults
            for tag in StateConstructor.state_configs(tag_cfgs)
        }


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        if self._cd_adder is not None:
            pf = self._cd_adder(pf)

        pf = self._seasonal_features(pf)

        transformed_parts, tag_names = self._transform_tags(pf, StageCode.SC)

        # put resultant data on PipeFrame
        df = pf.data.drop(tag_names, axis=1, inplace=False).astype(np.float32)
        pf.data = pd.concat([df, pf.action_lo, pf.action_hi, *transformed_parts], axis=1, copy=False)

        # guarantee an ordering over columns
        meta_cols = {
            name for name, cfg in self._tag_cfgs.items()
            if cfg.type == TagType.meta
        }
        cols = [col for col in pf.data.columns if col not in meta_cols]
        sorted_cols = self.sort_cols(cols)
        pf.data = pf.data.loc[:, sorted_cols]
        pf.states = pf.data[sorted_cols]

        maybe_log_trace_quality(self._app_state, self._components, pf.temporal_state)

        return pf


    @cached_property
    def columns(self):
        pf = self._probe_fake_data()
        return self.sort_cols(pf.data.columns)


    def sort_cols(self, cols: Iterable[str]):
        return list_u.multi_level_sort(
            list(cols),
            categories=[
                # endo observations
                lambda tag: (
                    tag in self._tag_cfgs
                    and self._tag_cfgs[tag].is_endogenous
                ),
                # exo observations
                lambda tag: (
                    tag in self._tag_cfgs
                ),
            ],
        )

    @staticmethod
    def state_configs(tag_cfgs: Sequence[TagConfig]):
        return [
            tag for tag in tag_cfgs
            if tag.type != TagType.meta
        ]


def construct_default_sc_configs(sc_cfg: SCConfig, tag_cfgs: Sequence[TagConfig]) -> None:
    for tag_cfg in tag_cfgs:
        if tag_cfg.state_constructor is None:
            tag_cfg.state_constructor = sc_cfg.defaults

def maybe_log_trace_quality(app_state: AppState, components: dict[str, list[Transform]], ts: TemporalState):
    """
    Assumes there is at most one TraceConstructor in each transform list
    """
    def _log_trace_qual(trace_con: TraceConstructor, trace_ts: TraceTemporalState):
        decays = trace_con._cfg.trace_values
        log_trace_quality(app_state, prefix='STATE', decays=decays, trace_ts=trace_ts)

    sc_ts = ts.get(StageCode.SC, {})
    assert isinstance(sc_ts, dict)

    for tag, xforms in components.items():
        tag_ts = sc_ts[tag]

        trace_con, trace_ts = Maybe.find(
            lambda x: isinstance(x[0], TraceConstructor), zip(xforms, tag_ts, strict=True),
        ).split(TraceConstructor, TraceTemporalState)

        Maybe.tap_all(_log_trace_qual, trace_con, trace_ts)
