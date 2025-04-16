from collections.abc import Iterable
from functools import cached_property

import pandas as pd
from pydantic import Field

import corerl.utils.list as list_u
from corerl.configs.config import config, list_
from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig, DecisionPointDetector
from corerl.data_pipeline.state_constructors.seasonal import SeasonalConfig, add_seasonal_features
from corerl.data_pipeline.tag_config import TagConfig, TagType
from corerl.data_pipeline.transforms import TraceConfig, TransformConfig


@config()
class SCConfig:
    defaults: list[TransformConfig] = list_([TraceConfig()])
    countdown: CountdownConfig = Field(default_factory=CountdownConfig)
    seasonal: SeasonalConfig = Field(default_factory=SeasonalConfig)


class StateConstructor(Constructor):
    def __init__(self, tag_cfgs: list[TagConfig], cfg: SCConfig):
        self._cfg = cfg
        super().__init__(tag_cfgs)

        self._cd_adder = DecisionPointDetector(cfg.countdown)


    def _get_relevant_configs(self, tag_cfgs: list[TagConfig]):
        return {
            tag.name: tag.state_constructor if tag.state_constructor is not None else self._cfg.defaults
            for tag in tag_cfgs
            if tag.type != TagType.meta
        }


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        if self._cd_adder is not None:
            pf = self._cd_adder(pf)

        pf = add_seasonal_features(self._cfg.seasonal, pf)

        transformed_parts, tag_names = self._transform_tags(pf, StageCode.SC)

        # put resultant data on PipeFrame
        df = pf.data.drop(tag_names, axis=1, inplace=False)
        pf.data = pd.concat([df] + transformed_parts, axis=1, copy=False)

        # guarantee an ordering over columns
        meta_cols = set(name for name, cfg in self._tag_cfgs.items() if cfg.type == TagType.meta)
        cols = [col for col in pf.data.columns if col not in meta_cols]
        sorted_cols = self.sort_cols(cols)
        pf.data = pf.data.loc[:, sorted_cols]
        pf.states = pf.data[sorted_cols]

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


def construct_default_sc_configs(sc_cfg: SCConfig, tag_cfgs: list[TagConfig]) -> None:
    for tag_cfg in tag_cfgs:
        if tag_cfg.state_constructor is None:
            tag_cfg.state_constructor = sc_cfg.defaults
